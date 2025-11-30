import sys
import os
import argparse

# Add local libs directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
libs_path = os.path.join(project_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
from src.model import StockLSTM
from src.data_processor import generate_backtest_data
from src.config import DataConfig, get_preset
from src.backtest_engine import BacktestEngine

def run_backtest(config=None, start_date=None, end_date=None, strategy='dynamic',
                initial_capital=100000, stop_loss=5.0, take_profit_buffer=5.0):
    """
    Run backtest with the given configuration.
    
    Args:
        config: DataConfig instance. If None, uses default configuration.
        start_date: Override start date for backtest period
        end_date: Override end date for backtest period
        strategy: Strategy type - 'simple' or 'dynamic'
        initial_capital: Initial capital for trading (used in dynamic strategy)
        stop_loss: Stop loss percentage (positive number, used in dynamic strategy)
        take_profit_buffer: Take profit buffer above predicted return (used in dynamic strategy)
    """
    # Use provided config or default
    if config is None:
        config = DataConfig()
    
    # Configuration
    START_DATE = start_date or "20230101"
    END_DATE = end_date or "20231231"
    MODEL_PATH = "models/stock_lstm.pth"
    LOOKBACK = config.lookback
    HORIZON = config.horizon
    INPUT_SIZE = config.input_size
    HIDDEN_SIZE = config.hidden_size
    NUM_LAYERS = config.num_layers
    OUTPUT_SIZE = config.output_size
    
    # Print configuration
    print("=" * 70)
    print("BACKTEST CONFIGURATION")
    print("=" * 70)
    print(config)
    print(f"  Backtest Period: {START_DATE} - {END_DATE}")
    print("=" * 70)
    print()
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
        
    # Force CPU to avoid OOM
    device = torch.device('cpu')
    model = StockLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.")
    
    # Generate Data
    print(f"Generating backtest data for {START_DATE}-{END_DATE}...")
    # Note: limit_stocks=None for full backtest, or set a limit for speed
    X, y, metadata = generate_backtest_data(
        lookback=LOOKBACK, 
        horizon=HORIZON, 
        start_date=START_DATE, 
        end_date=END_DATE,
        limit_stocks=None,
        use_multiprocessing=True
    )
    
    if len(X) == 0:
        print("No data generated for backtest.")
        return
        
    print(f"Generated {len(X)} samples.")
    
    # Predict
    X_tensor = torch.from_numpy(X).float().to(device)
    
    batch_size = 1024
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy().flatten())
            
    # Combine with metadata
    results = []
    for i, pred in enumerate(predictions):
        meta = metadata[i]
        # Model output is ratio, convert to percentage
        predicted_return = pred * 100 
        
        results.append({
            'symbol': meta['symbol'],
            'date': meta['date'],
            'buy_price': meta['close'],
            'sell_price': meta['future_close'],
            'actual_return': meta['actual_return'],
            'predicted_return': predicted_return
        })
        
    df_results = pd.DataFrame(results)
    
    # Choose strategy
    if strategy == 'dynamic':
        # Dynamic Strategy with Stop-Loss and Take-Profit
        print(f"\n--- Running Dynamic Backtest Strategy ---")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Stop-Loss: {stop_loss}%")
        print(f"Take-Profit Buffer: {take_profit_buffer}%")
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=initial_capital,
            max_positions=5,
            stop_loss_pct=-stop_loss,  # Convert to negative
            take_profit_buffer=take_profit_buffer
        )
        
        # Prepare predictions dataframe
        predictions_df = df_results[['symbol', 'date', 'predicted_return', 'buy_price']].copy()
        predictions_df.rename(columns={'buy_price': 'close'}, inplace=True)
        
        # Run backtest
        df_trades = engine.run_backtest(
            predictions=predictions_df,
            start_date=START_DATE,
            end_date=END_DATE,
            horizon=HORIZON
        )
        
        # Print summary
        engine.print_summary()
        
        # Save detailed results
        if not df_trades.empty:
            df_trades.to_csv("backtest_trades_dynamic.csv", index=False)
            print("\nDetailed trade log saved to backtest_trades_dynamic.csv")
            
            # Save equity curve
            equity_df = engine.get_equity_curve()
            equity_df.to_csv("backtest_equity_curve.csv", index=False)
            print("Equity curve saved to backtest_equity_curve.csv")

            # Save daily operations (buy/sell)
            ops_df = engine.get_operations_log()
            if not ops_df.empty:
                ops_df.to_csv("backtest_operations.csv", index=False)
                print("Operations log saved to backtest_operations.csv")
        
    else:
        # Simple Strategy: Buy Top 5 Daily and Hold for Horizon Days
        # Strategy: Buy Top N stocks with Predicted Return > Threshold
        # For simplicity: Buy Top 5 stocks daily if Predicted Return > 0
        
        print("\n--- Backtest Results (Strategy: Buy Top 5 Daily, Hold for Horizon) ---")
        
        df_results['date'] = pd.to_datetime(df_results['date'])
        daily_groups = df_results.groupby('date')
        
        trades = []
        
        for date, group in daily_groups:
            # Sort by predicted return descending
            group = group.sort_values('predicted_return', ascending=False)
            
            # Filter positive predictions
            candidates = group[group['predicted_return'] > 0]
            
            # Take top 5
            picks = candidates.head(5)
            
            for _, row in picks.iterrows():
                trades.append(row)
                
        df_trades = pd.DataFrame(trades)
        
        if df_trades.empty:
            print("No trades executed.")
            return
            
        # Calculate Metrics
        total_trades = len(df_trades)
        win_trades = len(df_trades[df_trades['actual_return'] > 0])
        win_rate = win_trades / total_trades * 100
        
        avg_return = df_trades['actual_return'].mean()
        cumulative_return = df_trades['actual_return'].sum() # Simple sum
        
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Return per Trade: {avg_return:.4f}%")
        print(f"Cumulative Return (Simple Sum): {cumulative_return:.4f}%")
        
        # Save detailed results
        df_trades.to_csv("backtest_trades.csv", index=False)
        print("Detailed trade log saved to backtest_trades.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest on stock prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default backtest with simple strategy (2023 data)
  python src/backtest.py
  
  # Dynamic strategy with default parameters
  python src/backtest.py --strategy dynamic
  
  # Dynamic strategy with custom stop-loss and take-profit
  python src/backtest.py --strategy dynamic --stop-loss 3 --take-profit-buffer 8
  
  # Dynamic strategy with custom initial capital
  python src/backtest.py --strategy dynamic --initial-capital 200000
  
  # Backtest with specific parameters
  python src/backtest.py --lookback 20 --horizon 5
  
  # Use preset configuration
  python src/backtest.py --preset medium
  
  # Custom backtest period
  python src/backtest.py --start-date 20230101 --end-date 20231231
  
  # Load from config file
  python src/backtest.py --config data/config_lb20_h3_2010_2022.json
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config", type=str, help="Path to config JSON file")
    config_group.add_argument("--preset", type=str, help="Use a preset configuration (short, medium, long, default)")
    
    # Custom parameters (override config)
    parser.add_argument("--lookback", type=int, help="Lookback period (days)")
    parser.add_argument("--horizon", type=int, help="Prediction horizon (days)")
    
    # Backtest period
    parser.add_argument("--start-date", type=str, help="Backtest start date (YYYYMMDD)")
    parser.add_argument("--end-date", type=str, help="Backtest end date (YYYYMMDD)")
    
    # Strategy selection
    parser.add_argument("--strategy", type=str, default="dynamic", choices=["simple", "dynamic"],
                       help="Backtest strategy: 'simple' (buy and hold) or 'dynamic' (with stop-loss/take-profit)")
    
    # Dynamic strategy parameters
    parser.add_argument("--initial-capital", type=float, default=100000,
                       help="Initial capital for dynamic strategy (default: 100000)")
    parser.add_argument("--stop-loss", type=float, default=5.0,
                       help="Stop-loss percentage for dynamic strategy (default: 5.0)")
    parser.add_argument("--take-profit-buffer", type=float, default=5.0,
                       help="Take-profit buffer above predicted return for dynamic strategy (default: 5.0)")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        # Load from config file
        config = DataConfig.load(args.config)
        print(f"Loaded configuration from {args.config}")
    elif args.preset:
        # Use preset
        config = get_preset(args.preset)
        print(f"Using preset configuration: {args.preset}")
    else:
        # Default configuration (backward compatible)
        config = DataConfig()
    
    # Override with command line arguments if provided
    if args.lookback is not None:
        config.lookback = args.lookback
    if args.horizon is not None:
        config.horizon = args.horizon
    
    # Run backtest
    run_backtest(
        config,
        start_date=args.start_date,
        end_date=args.end_date,
        strategy=args.strategy,
        initial_capital=args.initial_capital,
        stop_loss=args.stop_loss,
        take_profit_buffer=args.take_profit_buffer
    )
