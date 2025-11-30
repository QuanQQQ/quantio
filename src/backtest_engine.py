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
        self.take_profit_done = False  # Only allow one partial take-profit
        
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
            if pos.symbol in current_prices:
                position_value += pos.get_position_value(current_prices[pos.symbol])
        
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
        if partial_ratio is not None and partial_ratio < 1.0:
            trade = position.close_partial(partial_ratio, exit_price, exit_date, reason)
            self.cash += trade['quantity'] * exit_price
        else:
            trade = position.close_full(exit_price, exit_date, reason)
            self.cash += trade['quantity'] * exit_price
        
        self.closed_trades.append(trade)
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
                 stop_loss_pct: float = -5.0, take_profit_buffer: float = 5.0):
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
                                    current_date: str) -> Optional[Tuple[str, Optional[float]]]:
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
        
        # Check stop loss
        if unrealized_pnl <= self.stop_loss_pct:
            return ('stop_loss', None)  # Sell all
        
        # Check take profit (only once)
        take_profit_threshold = position.predicted_return + self.take_profit_buffer
        if (not position.take_profit_done) and unrealized_pnl >= take_profit_threshold:
            return ('take_profit', 0.5)  # Sell half
        
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
        
        print(f"\nRunning dynamic backtest from {start_date} to {end_date}...")
        print(f"Strategy: Top {self.max_positions} stocks, Stop-Loss: {self.stop_loss_pct}%, Take-Profit: Pred+{self.take_profit_buffer}%")
        print(f"Initial Capital: ${self.initial_capital:,.2f}\n")
        
        # Track daily metrics
        for current_date in trading_dates:
            current_date_str = current_date.strftime('%Y%m%d')
            
            # Update hold days
            self.portfolio.increment_hold_days()
            
            # Check each active position for stop-loss, take-profit, or horizon expiry
            for position in list(self.portfolio.get_active_positions()):
                # Get current price
                price_df = self.get_daily_prices(position.symbol, current_date_str, current_date_str)
                
                if price_df.empty:
                    continue
                
                current_price = price_df.iloc[0]['close']
                
                # Check for horizon expiry (force close)
                if position.hold_days >= horizon:
                    self.portfolio.close_position(
                        position, current_price, current_date_str, 
                        f'horizon_expiry_{horizon}d', None
                    )
                    continue
                
                # Check stop-loss and take-profit
                action_result = self.check_stop_loss_take_profit(position, current_price, current_date_str)
                
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
                candidates = daily_predictions.head(positions_to_open * 3)  # Get extra candidates
                
                # Filter out stocks we already hold
                active_symbols = {p.symbol for p in self.portfolio.get_active_positions()}
                candidates = candidates[~candidates['symbol'].isin(active_symbols)]
                
                # Select top N
                candidates = candidates.head(positions_to_open)
                
                # Calculate capital per position
                available_capital = self.portfolio.cash
                capital_per_position = available_capital / positions_to_open if positions_to_open > 0 else 0
                
                # Open positions
                for _, row in candidates.iterrows():
                    if self.portfolio.get_position_count() >= self.max_positions:
                        break
                    
                    self.portfolio.open_position(
                        symbol=row['symbol'],
                        entry_date=current_date_str,
                        entry_price=row['close'],
                        predicted_return=row['predicted_return'],
                        horizon=horizon,
                        capital_to_use=capital_per_position
                    )
            
            # Record daily equity
            current_prices = {row['symbol']: row['close'] 
                            for _, row in daily_predictions.iterrows()}
            
            # Add prices for active positions not in today's predictions
            for pos in self.portfolio.get_active_positions():
                if pos.symbol not in current_prices:
                    price_df = self.get_daily_prices(pos.symbol, current_date_str, current_date_str)
                    if not price_df.empty:
                        current_prices[pos.symbol] = price_df.iloc[0]['close']
            
            equity = self.portfolio.get_total_equity(current_prices)
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
        total_return = (self.portfolio.cash - self.initial_capital) / self.initial_capital * 100
        
        # Get final equity
        if self.portfolio.daily_equity:
            final_equity = self.portfolio.daily_equity[-1]['equity']
        else:
            final_equity = self.portfolio.cash
        
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
