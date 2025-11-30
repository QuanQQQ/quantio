"""
Dynamic Backtest Engine with Stop-Loss and Take-Profit Logic

This module implements a sophisticated backtesting engine that supports:
- Dynamic position tracking across multiple days
- Stop-loss and take-profit rules
- Partial position closes
- Portfolio-level position management
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Add local libs directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
libs_path = os.path.join(project_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import get_stock_daily
from src.fetcher import pro


class Position:
    """Represents a single position in a stock."""
    
    def __init__(self, symbol: str, entry_date: str, entry_price: float, 
                 quantity: float, predicted_return: float, horizon: int):
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.initial_quantity = quantity
        self.quantity = quantity  # Current quantity (can be reduced by partial sells)
        self.predicted_return = predicted_return
        self.horizon = horizon
        self.hold_days = 0
        self.trades = []  # List of trade records
        self.last_price = entry_price  # Fallback price when daily price missing
        self.take_profit_done = False  # Only allow one partial take-profit
        self.short_below_hits = 0      # Count of days price < short_trend since entry
        self.short_above_seen = False  # Has price ever been >= short_trend since entry
        self.pending_exit_next_open = False  # If true, will exit at next day's open
        self.pending_exit_reason: Optional[str] = None
        
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        return (current_price - self.entry_price) / self.entry_price * 100
    
    def get_position_value(self, current_price: float) -> float:
        """Get current position value."""
        return self.quantity * current_price
    
    def close_partial(self, sell_ratio: float, exit_price: float, exit_date: str, reason: str) -> Dict:
        """
        Close a portion of the position.
        
        Args:
            sell_ratio: Ratio to sell (0.5 for half)
            exit_price: Exit price
            exit_date: Exit date
            reason: Reason for closing
            
        Returns:
            Trade record dict
        """
        sell_quantity = self.quantity * sell_ratio
        actual_return = (exit_price - self.entry_price) / self.entry_price * 100
        
        trade = {
            'symbol': self.symbol,
            'entry_date': self.entry_date,
            'exit_date': exit_date,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'quantity': sell_quantity,
            'actual_return': actual_return,
            'predicted_return': self.predicted_return,
            'hold_days': self.hold_days,
            'close_reason': reason,
            'position_closed': 'partial'
        }
        
        self.quantity -= sell_quantity
        if reason == 'take_profit':
            self.take_profit_done = True
        self.trades.append(trade)
        return trade
    
    def close_full(self, exit_price: float, exit_date: str, reason: str) -> Dict:
        """
        Close the entire position.
        
        Args:
            exit_price: Exit price
            exit_date: Exit date
            reason: Reason for closing
            
        Returns:
            Trade record dict
        """
        actual_return = (exit_price - self.entry_price) / self.entry_price * 100
        
        trade = {
            'symbol': self.symbol,
            'entry_date': self.entry_date,
            'exit_date': exit_date,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'quantity': self.quantity,
            'actual_return': actual_return,
            'predicted_return': self.predicted_return,
            'hold_days': self.hold_days,
            'close_reason': reason,
            'position_closed': 'full'
        }
        
        self.trades.append(trade)
        self.quantity = 0
        return trade
    
    def is_closed(self) -> bool:
        """Check if position is fully closed."""
        return self.quantity <= 0


class PortfolioManager:
    """Manages the portfolio of positions."""
    
    def __init__(self, initial_capital: float, max_positions: int = 5):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.daily_equity = []
        self.operations = []  # Daily buy/sell operations log
        # Equity index (base=1.0), updated only on realized trades using percentage returns
        self.equity_index = 1.0
        
    def get_active_positions(self) -> List[Position]:
        """Get all active (non-closed) positions."""
        return [p for p in self.positions if not p.is_closed()]
    
    def get_position_count(self) -> int:
        """Get number of active positions."""
        return len(self.get_active_positions())
    
    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total equity (cash + position values).
        
        Args:
            current_prices: Dict mapping symbol to current price
            
        Returns:
            Total equity value
        """
        position_value = 0
        for pos in self.get_active_positions():
            price = current_prices.get(pos.symbol, pos.last_price)
            position_value += pos.get_position_value(price)
        
        return self.cash + position_value
    
    def open_position(self, symbol: str, entry_date: str, entry_price: float,
                     predicted_return: float, horizon: int, capital_to_use: float) -> Optional[Position]:
        """
        Open a new position.
        
        Args:
            symbol: Stock symbol
            entry_date: Entry date
            entry_price: Entry price
            predicted_return: Predicted return
            horizon: Holding horizon
            capital_to_use: Amount of capital to allocate
            
        Returns:
            Position object if successful, None otherwise
        """
        if capital_to_use > self.cash:
            return None
        
        quantity = capital_to_use / entry_price
        position = Position(symbol, entry_date, entry_price, quantity, predicted_return, horizon)
        
        self.positions.append(position)
        self.cash -= capital_to_use
        # Record buy operation
        self.operations.append({
            'date': entry_date,
            'action': 'buy',
            'symbol': symbol,
            'price': entry_price,
            'quantity': quantity,
            'reason': 'open',
            'predicted_return': predicted_return
        })

        return position
    
    def close_position(self, position: Position, exit_price: float, 
                      exit_date: str, reason: str, partial_ratio: Optional[float] = None):
        """
        Close a position (fully or partially).
        
        Args:
            position: Position to close
            exit_price: Exit price
            exit_date: Exit date
            reason: Reason for closing
            partial_ratio: If provided, close this ratio (0.5 for half)
        """
        # Realize trade and update cash
        if partial_ratio is not None and partial_ratio < 1.0:
            trade = position.close_partial(partial_ratio, exit_price, exit_date, reason)
            self.cash += trade['quantity'] * exit_price
        else:
            trade = position.close_full(exit_price, exit_date, reason)
            self.cash += trade['quantity'] * exit_price
        
        self.closed_trades.append(trade)
        # Update equity index based on realized percentage return
        try:
            slot_weight = 1.0 / max(1, self.max_positions)
            realized_ratio = (partial_ratio if (partial_ratio is not None and partial_ratio < 1.0) else 1.0)
            ret_dec = float(trade['actual_return']) / 100.0
            self.equity_index = self.equity_index * (1.0 + slot_weight * realized_ratio * ret_dec)
        except Exception:
            # Non-blocking safeguard: keep index unchanged on error
            pass
        # Record sell operation
        self.operations.append({
            'date': exit_date,
            'action': 'sell',
            'symbol': position.symbol,
            'price': exit_price,
            'quantity': trade['quantity'],
            'reason': reason,
            'partial_ratio': (partial_ratio if partial_ratio is not None else 1.0),
            'hold_days': position.hold_days,
        })
    
    def increment_hold_days(self):
        """Increment hold days for all active positions."""
        for pos in self.get_active_positions():
            pos.hold_days += 1


class BacktestEngine:
    """
    Dynamic backtest engine with stop-loss and take-profit logic.
    """
    
    def __init__(self, initial_capital: float = 100000, max_positions: int = 5,
                 stop_loss_pct: float = -5.0, take_profit_buffer: float = 5.0,
                 variant: str = 'v2'):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            max_positions: Maximum number of concurrent positions
            stop_loss_pct: Stop loss percentage (e.g., -5 for -5%)
            take_profit_buffer: Additional profit buffer for take-profit (e.g., 5 for pred+5%)
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_buffer = take_profit_buffer
        self.portfolio = PortfolioManager(initial_capital, max_positions)
        self.variant = variant
        # Cache for delisting detection via interface
        self._delisted_cache: Dict[str, bool] = {}

    def is_delisted(self, symbol: str, current_date: str, end_date: str) -> bool:
        """
        Determine if a symbol is delisted using interface (Tushare stock_basic).
        Falls back to heuristic (no future prices in DB) when API fails.
        Results are cached per symbol.
        """
        if symbol in self._delisted_cache:
            return self._delisted_cache[symbol]
        delisted = False
        try:
            df = pro.stock_basic(ts_code=symbol, fields='ts_code,list_status')
            if df is not None and not df.empty:
                status = str(df.iloc[0]['list_status']).upper()
                # Tushare list_status: L(上市), D(退市), P(暂停上市)
                delisted = (status == 'D')
        except Exception:
            # Fallback heuristic if API error
            try:
                df_future = self.get_daily_prices(symbol, current_date, end_date)
                delisted = df_future.empty
            except Exception:
                delisted = False
        self._delisted_cache[symbol] = delisted
        return delisted
        
    def get_daily_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get daily price data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with daily prices
        """
        try:
            df = get_stock_daily(symbol, start_date, end_date)
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def check_stop_loss_take_profit(self, position: Position, current_price: float,
                                    current_date: str,
                                    short_trend: Optional[float] = None,
                                    long_trend: Optional[float] = None) -> Optional[Tuple[str, Optional[float]]]:
        """
        Check if stop-loss or take-profit conditions are met.
        
        Args:
            position: Position to check
            current_price: Current price
            current_date: Current date
            
        Returns:
            Tuple of (action, partial_ratio) where action is 'stop_loss', 'take_profit', or None
        """
        unrealized_pnl = position.get_unrealized_pnl(current_price)

        if self.variant == 'v1':
            # Stop-loss first
            if unrealized_pnl <= self.stop_loss_pct:
                return ('stop_loss', None)
            # Take-profit only when there is floating profit
            take_profit_threshold = position.predicted_return + self.take_profit_buffer
            if (not position.take_profit_done) and (unrealized_pnl > 0) and unrealized_pnl >= take_profit_threshold:
                return ('take_profit', 0.5)
            return None
        else:
            # v2 enhanced rules
            # Detect short-trend breach for counting
            below_short = False
            if short_trend is not None:
                try:
                    below_short = (not np.isnan(short_trend)) and (current_price < short_trend)
                except Exception:
                    below_short = False
            # Track if price has ever been above short trend since entry
            above_short = False
            if short_trend is not None:
                try:
                    above_short = (not np.isnan(short_trend)) and (current_price >= short_trend)
                except Exception:
                    above_short = False
            if above_short:
                position.short_above_seen = True
            if below_short:
                # Only count short-trend breaches after price has stood above short trend at least once
                if position.short_above_seen:
                    position.short_below_hits += 1
                    # Second time breach -> full close immediately
                    if position.short_below_hits >= 2:
                        return ('short_trend_break_2x', None)

            # Stop-loss first (priority over take-profit)
            below_long = False
            if long_trend is not None:
                try:
                    below_long = (not np.isnan(long_trend)) and (current_price < long_trend)
                except Exception:
                    below_long = False
            if unrealized_pnl <= self.stop_loss_pct or below_long:
                return ('stop_loss', None)

            # Short-trend first breach partial take-profit (only when floating profit AND has stood above short trend before)
            if (
                below_short and position.short_below_hits == 1 and
                (not position.take_profit_done) and (unrealized_pnl > 0) and position.short_above_seen
            ):
                return ('take_profit', 0.5)

            # Predicted-based take profit (only once, only when floating profit)
            take_profit_threshold = position.predicted_return + self.take_profit_buffer
            if (not position.take_profit_done) and (unrealized_pnl > 0) and unrealized_pnl >= take_profit_threshold:
                return ('take_profit', 0.5)

            return None
    
    def run_backtest(self, predictions: pd.DataFrame, start_date: str, end_date: str,
                    horizon: int) -> pd.DataFrame:
        """
        Run the backtest.
        
        Args:
            predictions: DataFrame with columns ['symbol', 'date', 'predicted_return', 'close']
            start_date: Backtest start date
            end_date: Backtest end date
            horizon: Holding horizon in days
            
        Returns:
            DataFrame with trade results
        """
        # Convert dates
        predictions['date'] = pd.to_datetime(predictions['date'], format='%Y%m%d')
        start_dt = pd.to_datetime(start_date, format='%Y%m%d')
        end_dt = pd.to_datetime(end_date, format='%Y%m%d')
        
        # Sort by date
        predictions = predictions.sort_values('date')
        
        # Get unique dates
        trading_dates = sorted(predictions['date'].unique())
        # Debug: earliest date in generated backtest predictions
        if len(trading_dates) > 0:
            earliest_dt = trading_dates[0]
            try:
                print(f"[DEBUG] Earliest generated trading date: {earliest_dt.strftime('%Y-%m-%d')} (raw: {earliest_dt.strftime('%Y%m%d')})")
            except Exception:
                print(f"[DEBUG] Earliest generated trading date: {earliest_dt}")
        else:
            print("[DEBUG] No trading dates found in predictions.")
        print(f"\nRunning dynamic backtest from {start_date} to {end_date}...")
        print(f"Strategy: Top {self.max_positions} stocks, Stop-Loss: {self.stop_loss_pct}%, Take-Profit: Pred+{self.take_profit_buffer}%")
        print(f"Initial Capital: ${self.initial_capital:,.2f}\n")
        
        # Track daily metrics
        end_date_str = end_dt.strftime('%Y%m%d')
        for i, current_date in enumerate(trading_dates):
            current_date_str = current_date.strftime('%Y%m%d')
            next_date_str = None
            if i + 1 < len(trading_dates):
                try:
                    next_date_str = trading_dates[i+1].strftime('%Y%m%d')
                except Exception:
                    next_date_str = None
            
            # Update hold days
            self.portfolio.increment_hold_days()
            
            # Check each active position for stop-loss, take-profit, or horizon expiry
            for position in list(self.portfolio.get_active_positions()):
                # Get current price
                price_df = self.get_daily_prices(position.symbol, current_date_str, current_date_str)
                if price_df.empty:
                    # If no price today: use interface to check delisting.
                    if self.is_delisted(position.symbol, current_date_str, end_date_str):
                        self.portfolio.close_position(
                            position, position.last_price, current_date_str,
                            'delisted', None
                        )
                        continue
                    else:
                        current_price = position.last_price
                else:
                    current_price = price_df.iloc[0]['close']
                    position.last_price = current_price

                # Execute any pending next-open exit BEFORE other checks
                if position.pending_exit_next_open:
                    try:
                        exit_price = float(price_df.iloc[0]['open']) if 'open' in price_df.columns else current_price
                    except Exception:
                        exit_price = current_price
                    self.portfolio.close_position(
                        position, exit_price, current_date_str,
                        position.pending_exit_reason or 'stop_loss_next_open', None
                    )
                    position.pending_exit_next_open = False
                    position.pending_exit_reason = None
                    continue
                
                # Check for horizon expiry (force close) - only for v1
                if self.variant == 'v1' and position.hold_days >= horizon:
                    self.portfolio.close_position(
                        position, current_price, current_date_str, 
                        f'horizon_expiry_{horizon}d', None
                    )
                    continue
                
                # Check stop-loss and take-profit
                short_trend_val = None
                long_trend_val = None
                try:
                    if 'short_trend' in price_df.columns:
                        short_trend_val = float(price_df.iloc[0]['short_trend'])
                    if 'long_trend' in price_df.columns:
                        long_trend_val = float(price_df.iloc[0]['long_trend'])
                except Exception:
                    short_trend_val = None
                    long_trend_val = None

                action_result = self.check_stop_loss_take_profit(
                    position, current_price, current_date_str, short_trend_val, long_trend_val
                )
                
                if action_result is not None:
                    action, partial_ratio = action_result
                    self.portfolio.close_position(
                        position, current_price, current_date_str,
                        action, partial_ratio
                    )
            
            # Get today's predictions
            daily_predictions = predictions[predictions['date'] == current_date]
            
            if daily_predictions.empty:
                continue
            
            # Sort by predicted return and select top candidates
            daily_predictions = daily_predictions.sort_values('predicted_return', ascending=False)
            
            # Calculate how many positions we need to open
            current_position_count = self.portfolio.get_position_count()
            positions_to_open = self.max_positions - current_position_count
            
            if positions_to_open > 0:
                # Get top candidates
                candidates = daily_predictions.head(positions_to_open * 5)  # More buffer for skip/advance
                
                # Filter out stocks we already hold
                active_symbols = {p.symbol for p in self.portfolio.get_active_positions()}
                candidates = candidates[~candidates['symbol'].isin(active_symbols)]
                
                # Filter predicted_return > 0
                candidates = candidates[candidates['predicted_return'] > 0]
                # Select top N
                candidates = candidates.head(positions_to_open * 5)
                
                # Both v1 and v2 use compounding based on realized equity index
                per_slot_capital = (self.portfolio.equity_index * self.initial_capital) / self.max_positions
                # Determine openable slots by cash availability
                openable_by_cash = int(self.portfolio.cash // per_slot_capital)
                slots_to_open = min(positions_to_open, openable_by_cash)
                
                # If no slots can be opened due to insufficient cash, skip
                if slots_to_open <= 0:
                    pass
                else:
                    opened = 0
                    # If there is no next trading date (e.g., last day), skip opening
                    if next_date_str is None:
                        pass
                    else:
                        for _, row in candidates.iterrows():
                            if self.portfolio.get_position_count() >= self.max_positions:
                                break
                            if opened >= slots_to_open:
                                break
                            # Fetch next-day open and long_trend for buy-time checks
                            try:
                                price_df_open = self.get_daily_prices(row['symbol'], next_date_str, next_date_str)
                                if price_df_open.empty:
                                    continue
                                open_val = float(price_df_open.iloc[0]['open']) if 'open' in price_df_open.columns else float(price_df_open.iloc[0]['close'])
                                if self.variant == 'v2':
                                    long_trend_val = float(price_df_open.iloc[0]['long_trend']) if 'long_trend' in price_df_open.columns else np.nan
                                    # Require long_trend available and price above it at buy time
                                    if np.isnan(long_trend_val) or open_val < long_trend_val:
                                        continue
                            except Exception:
                                continue

                            pos_obj = self.portfolio.open_position(
                                symbol=row['symbol'],
                                entry_date=next_date_str,
                                entry_price=open_val,
                                predicted_return=row['predicted_return'],
                                horizon=horizon,
                                capital_to_use=per_slot_capital
                            )
                            opened += 1

                            # If opened successfully, evaluate BUY-DAY stop-loss conditions and schedule next-open exit if triggered
                            if pos_obj is not None:
                                try:
                                    # Use buy-day close to detect immediate stop-loss triggers
                                    buy_day_close = float(price_df_open.iloc[0]['close']) if 'close' in price_df_open.columns else open_val
                                    unrealized_pnl_buy_day = (buy_day_close - open_val) / open_val * 100.0
                                    long_trend_buy_day = float(price_df_open.iloc[0]['long_trend']) if 'long_trend' in price_df_open.columns else np.nan
                                    trigger_stop = False
                                    if self.variant == 'v2':
                                        trigger_stop = (unrealized_pnl_buy_day <= self.stop_loss_pct) or (not np.isnan(long_trend_buy_day) and buy_day_close < long_trend_buy_day)
                                    else:
                                        trigger_stop = (unrealized_pnl_buy_day <= self.stop_loss_pct)
                                    if trigger_stop:
                                        pos_obj.pending_exit_next_open = True
                                        pos_obj.pending_exit_reason = 'stop_loss_next_open'
                                except Exception:
                                    # Non-blocking
                                    pass
            
            # Record daily equity
            # Use equity index (base=1.0) instead of cash+positions valuation
            equity = self.portfolio.equity_index
            self.portfolio.daily_equity.append({
                'date': current_date_str,
                'equity': equity,
                'cash': self.portfolio.cash,
                'positions': self.portfolio.get_position_count()
            })
        
        # Force close all remaining positions at end date
        final_date_str = trading_dates[-1].strftime('%Y%m%d')
        for position in list(self.portfolio.get_active_positions()):
            price_df = self.get_daily_prices(position.symbol, final_date_str, final_date_str)
            if not price_df.empty:
                final_price = price_df.iloc[0]['close']
            else:
                final_price = position.last_price
            self.portfolio.close_position(
                position, final_price, final_date_str,
                'backtest_end', None
            )
        
        # Return results
        trades_df = pd.DataFrame(self.portfolio.closed_trades)
        return trades_df
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get the equity curve over time."""
        return pd.DataFrame(self.portfolio.daily_equity)

    def get_operations_log(self) -> pd.DataFrame:
        """Get daily buy/sell operations log."""
        return pd.DataFrame(self.portfolio.operations)
    
    def print_summary(self):
        """Print backtest summary statistics."""
        if not self.portfolio.closed_trades:
            print("No trades executed.")
            return
        
        trades_df = pd.DataFrame(self.portfolio.closed_trades)
        
        total_trades = len(trades_df)
        win_trades = len(trades_df[trades_df['actual_return'] > 0])
        win_rate = win_trades / total_trades * 100
        
        avg_return = trades_df['actual_return'].mean()
        # Compute total return from equity index
        total_return = ((self.portfolio.daily_equity[-1]['equity'] if self.portfolio.daily_equity else 1.0) - 1.0) * 100
        
        # Get final equity (index scaled by initial capital for reporting)
        if self.portfolio.daily_equity:
            final_equity = (self.portfolio.daily_equity[-1]['equity']) * self.initial_capital
        else:
            final_equity = self.initial_capital
        
        print("\n" + "="*70)
        print("BACKTEST SUMMARY")
        print("="*70)
        print(f"Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"Final Equity:           ${final_equity:,.2f}")
        print(f"Total Return:           {total_return:.2f}%")
        print(f"Total Trades:           {total_trades}")
        print(f"Win Rate:               {win_rate:.2f}%")
        print(f"Average Return/Trade:   {avg_return:.4f}%")
        print(f"\nStop-Loss Triggers:     {len(trades_df[trades_df['close_reason'] == 'stop_loss'])}")
        print(f"Take-Profit Triggers:   {len(trades_df[trades_df['close_reason'] == 'take_profit'])}")
        print(f"Horizon Expiry:         {len(trades_df[trades_df['close_reason'].str.startswith('horizon')])}")
        print("="*70)
