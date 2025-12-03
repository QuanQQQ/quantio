import sys
import os

# Add local libs directory to path (for custom installation location)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
libs_path = os.path.join(project_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

import torch
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

# Add local libs directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model import StockLSTM
from config import DataConfig
from fetcher import update_all
from data_processor import normalize_data
from database import get_all_stocks, get_stock_daily, get_stock_daily_last_n

def load_model(model_path, config, device):
    """Load the trained model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
        
    model = StockLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size
    ).to(device)
    
    try:
        # Try loading state dict directly first (saved via torch.save(model.state_dict(), ...))
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception:
        try:
            # Try loading checkpoint dict (saved via torch.save(checkpoint, ...))
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Error: Could not understand model file format.")
                return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
    model.eval()
    return model

def predict_daily(args):
    # 1. Update Data
    if not args.no_update:
        print("Updating data to ensure we have the latest daily records...")
        # Update only recent data to save time, e.g., last 10 days if just catching up
        # But update_all checks everything. Let's trust update_all's efficiency or user's choice.
        update_all(lookback_years=1) # Reduce lookback for daily update to be faster? 
                                     # Actually update_all checks missing dates, so it should be fast if mostly up to date.

    # 2. Load Config & Model
    config_path = args.config
    if not config_path:
        # Try to find a config file in data/
        import glob
        configs = glob.glob("data/config_*.json")
        if configs:
            config_path = configs[0] # Pick the first one
            print(f"Auto-detected config: {config_path}")
        else:
            print("No config file found. Using default.")
            config_path = None

    if config_path:
        config = DataConfig.load(config_path)
    else:
        config = DataConfig()

    # Override config with args if needed (not implemented for simplicity, relying on saved config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(args.model, config, device)
    if model is None:
        return

    # 3. Scan Stocks
    print("Scanning stocks for candidates (J < 13)...")
    all_stocks = get_all_stocks()
    candidates = []
    
    # We need enough history for the lookback window
    lookback = config.lookback
    
    # Get today's date or latest available date in DB
    # We will fetch the last 'lookback + 20' days for each stock to ensure we have enough data
    # Optimization: Fetch all stocks' latest data? 
    # For simplicity and reliability, let's iterate.
    
    # To speed up, we can filter stocks that are likely to be candidates? 
    # No, we need to check the indicator.
    
    symbols = all_stocks['symbol'].tolist()
    if args.limit:
        symbols = symbols[:args.limit]
        
    print(f"Processing {len(symbols)} stocks...")
    
    predictions = []
    
    for symbol in tqdm(symbols):
        try:
            # Fetch slightly more than lookback to calculate indicators if needed, 
            # but we assume indicators are in DB. 
            # We just need the last 'lookback' rows.
            # But we need to find if the *latest* day triggers the condition.
            
            # Get last 60 days to be safe
            # Use today as end_date for query
            today_str = datetime.now().strftime('%Y%m%d')
            df = get_stock_daily_last_n(symbol, end_date=today_str, n=60)
            
            if df.empty or len(df) < lookback:
                continue
                
            # Check latest row for trigger
            latest_row = df.iloc[-1]
            
            # Check if data is recent (e.g., today or yesterday)
            # If the data is too old, maybe the stock is suspended or data not updated.
            # We can print a warning or just skip.
            # For now, we predict based on whatever latest data is there.
            
            if latest_row['j'] >= 13:
                continue
                
            # Candidate found!
            # Prepare input
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
            
            # Ensure we have enough data for the window
            if len(df) < lookback:
                continue
                
            window_df = df.iloc[-lookback:][feature_cols].copy()
            
            # Normalize
            normalized_window = normalize_data(window_df)
            
            # Ensure numeric types (float32)
            # This fixes "can't convert np.ndarray of type numpy.object_"
            try:
                normalized_window = normalized_window.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            except Exception:
                # Fallback if apply fails
                normalized_window = normalized_window.astype(float).fillna(0.0)
            
            # Convert to tensor
            x_np = normalized_window.values
            x_tensor = torch.from_numpy(x_np).float().unsqueeze(0).to(device) # (1, lookback, input_size)
            
            # Inference
            with torch.no_grad():
                output = model(x_tensor)
                score = output.item()
                
                
            predictions.append({
                'symbol': symbol,
                'name': all_stocks[all_stocks['symbol'] == symbol]['name'].values[0],
                'date': latest_row['date'],
                'close': latest_row['close'],
                'j_value': latest_row['j'],
                'score': score
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    # 4. Output
    if not predictions:
        print("No candidates found.")
        return

    # Sort by score (descending)
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Create DataFrame
    pred_df = pd.DataFrame(predictions)
    
    print("\n" + "="*50)
    print(f"Top Candidates for Next Trading Day")
    print("="*50)
    print(pred_df[['symbol', 'name', 'date', 'close', 'j_value', 'score']].head(20).to_string(index=False))
    print("="*50)
    
    # Save to CSV
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
        
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"predictions/prediction_{date_str}.csv"
    pred_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\nFull results saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stock targets for the next trading day.")
    parser.add_argument("--model", type=str, default="models/stock_lstm.pth", help="Path to the trained model")
    parser.add_argument("--config", type=str, help="Path to the config file (optional, auto-detects if not provided)")
    parser.add_argument("--no-update", action="store_true", help="Skip data update")
    parser.add_argument("--limit", type=int, help="Limit number of stocks to scan (for testing)")
    
    args = parser.parse_args()
    
    predict_daily(args)
