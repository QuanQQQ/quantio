import pandas as pd
import numpy as np
from database import get_all_stocks, get_stock_daily
from datetime import datetime, timedelta

def calculate_kdj(df, period=9):
    """
    Calculate KDJ indicator.
    """
    if df.empty:
        return df
        
    low_list = df['low'].rolling(window=period, min_periods=1).min()
    high_list = df['high'].rolling(window=period, min_periods=1).max()
    
    range_high_low = high_list - low_list
    range_high_low.replace(0, np.nan, inplace=True)
    
    rsv = (df['close'] - low_list) / range_high_low * 100
    rsv.fillna(0, inplace=True)
    
    # 东财公式 SMA(RSV, M1, 1) 中 M1=3 对应 α=1/3；EWMA 的 com=2 时 α=1/(1+com)=1/3，两者等价
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator.
    """
    if df.empty:
        return df
        
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = 2 * (df['macd'] - df['macd_signal'])
    
    return df

def calculate_trend_lines(df):
    """
    Calculate trend lines.
    1. Short-term Trend: EMA(EMA(C, 10), 10)
    2. Long-term Bull/Bear: (MA(14) + MA(28) + MA(57) + MA(114)) / 4
    """
    if df.empty:
        return df
        
    # Short-term Trend
    ema10 = df['close'].ewm(span=10, adjust=False).mean()
    df['short_trend'] = ema10.ewm(span=10, adjust=False).mean()
    
    # Long-term Bull/Bear
    ma14 = df['close'].rolling(window=14).mean()
    ma28 = df['close'].rolling(window=28).mean()
    ma57 = df['close'].rolling(window=57).mean()
    ma114 = df['close'].rolling(window=114).mean()
    
    df['long_trend'] = (ma14 + ma28 + ma57 + ma114) / 4
    
    return df

def normalize_data(df):
    """
    Normalize data using Relative Scaling strategy.
    
    1. Prices (Open, High, Low, Close): Divided by the first Close of the window - 1.
       Pt' = Pt / P_close_0 - 1
    2. Volume: Divided by average volume of the window - 1.
       Vt' = Vt / V_avg - 1
    3. KDJ: Divided by 100.
    4. MACD (macd, signal, hist): Divided by the first Close of the window.
    """
    df_norm = df.copy()
    
    # Base price for scaling (First day's Close)
    base_price = df['close'].iloc[0]
    
    # 1. Normalize Prices (including trend lines)
    price_cols = ['open', 'high', 'low', 'close', 'short_trend', 'long_trend']
    for col in price_cols:
        if col in df.columns:
            df_norm[col] = df[col] / base_price - 1
        
    # 2. Normalize Volume
    avg_volume = df['volume'].mean() + 1e-8
    df_norm['volume'] = df['volume'] / avg_volume - 1
    
    # 3. Normalize KDJ (0-100 -> 0-1)
    kdj_cols = ['k', 'd', 'j']
    for col in kdj_cols:
        df_norm[col] = df[col] / 100.0
        
    # 4. Normalize MACD (Relative to price)
    macd_cols = ['macd', 'macd_signal', 'macd_hist']
    for col in macd_cols:
        df_norm[col] = df[col] / base_price
        
    return df_norm

def process_stock(args):
    """
    Worker function to process a single stock.
    Args:
        args (tuple): (symbol, lookback, horizon, start_date, end_date)
    """
    symbol, lookback, horizon, start_date, end_date = args
    
    try:
        # Fetch data (with precomputed indicators)
        df = get_stock_daily(symbol, start_date=start_date, end_date=end_date)

        if len(df) < lookback + horizon + 30:
            return [], []

        # Require precomputed indicator columns from DB
        required_cols = ['k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
        if any(col not in df.columns for col in required_cols):
            # Indicators not present; skip to avoid recompute here
            return [], []

        # Drop rows with missing essentials
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        df.dropna(subset=base_cols + required_cols, inplace=True)

        if df.empty:
            return [], []

        # Reset index
        df = df.reset_index(drop=True)

        # Find triggers: J < 13
        potential_indices = df[df['j'] < 13].index
        
        X_local = []
        y_local = []
        
        for idx in potential_indices:
            if idx < lookback or idx >= len(df) - horizon:
                continue
                
            # Extract Features (from DB precomputed indicators)
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
            window_df = df.iloc[idx-lookback+1 : idx+1][feature_cols].copy()
            
            # Normalize window
            normalized_window = normalize_data(window_df)
            
            X_local.append(normalized_window.values)
            
            # Calculate Target Score
            # User request: "Consider rise and fall magnitude"
            # Target = Percentage Return over horizon
            # Score = (Future_Close - Current_Close) / Current_Close * 100
            
            current_close = df.iloc[idx]['close']
            future_close = df.iloc[idx+horizon]['close']
            
            score = (future_close - current_close) / current_close * 100
            
            y_local.append(score)
            
        return X_local, y_local
    except Exception as e:
        # print(f"Error processing {symbol}: {e}")
        return [], []

def generate_training_data(lookback=10, horizon=3, limit_stocks=None, start_date=None, end_date=None, use_multiprocessing=True):
    """
    Generate training data for the neural network.
    
    Args:
        use_multiprocessing (bool): Whether to use multiprocessing. Set to False if memory issues occur.
    """
    all_stocks = get_all_stocks()
    if limit_stocks:
        all_stocks = all_stocks.head(limit_stocks)
        
    symbols = all_stocks['symbol'].tolist()
    
    X_all = []
    y_all = []
    
    if use_multiprocessing:
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        
        # Prepare arguments for each worker
        tasks = [(symbol, lookback, horizon, start_date, end_date) for symbol in symbols]
        
        # Use fewer workers to reduce memory pressure
        num_workers = min(cpu_count(), 6)  # Use half of cores or max 4
        print(f"Generating data from {len(symbols)} stocks using {num_workers} cores...")
        
        with Pool(processes=num_workers) as pool:
            # Use imap with chunksize to process in batches
            results = list(tqdm(pool.imap(process_stock, tasks, chunksize=10), total=len(tasks), unit="stock"))
            
        for X_local, y_local in results:
            if X_local:
                X_all.extend(X_local)
                y_all.extend(y_local)
    else:
        # Single-threaded fallback
        from tqdm import tqdm
        print(f"Generating data from {len(symbols)} stocks (single-threaded)...")
        for symbol in tqdm(symbols, unit="stock"):
            X_local, y_local = process_stock((symbol, lookback, horizon, start_date, end_date))
            if X_local:
                X_all.extend(X_local)
                y_all.extend(y_local)
            
    if not X_all:
        return np.array([]), np.array([])
        
    return np.array(X_all), np.array(y_all)

def process_stock_backtest(args):
    """
    Worker function to process a single stock for backtesting.
    Returns metadata alongside features.
    Args:
        args (tuple): (symbol, lookback, horizon, start_date, end_date)
    """
    symbol, lookback, horizon, start_date, end_date = args
    
    try:
        # Fetch data (with precomputed indicators)
        df = get_stock_daily(symbol, start_date=start_date, end_date=end_date)

        if len(df) < lookback + horizon + 30:
            return [], [], []

        # Require precomputed indicator columns from DB
        required_cols = ['k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
        if any(col not in df.columns for col in required_cols):
            return [], [], []

        # Drop rows with missing essentials
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        df.dropna(subset=base_cols + required_cols, inplace=True)

        if df.empty:
            return [], [], []

        # Reset index
        df = df.reset_index(drop=True)

        # Find triggers: J < 13
        potential_indices = df[df['j'] < 13].index
        
        X_local = []
        y_local = []
        metadata_local = []
        
        for idx in potential_indices:
            if idx < lookback or idx >= len(df) - horizon:
                continue
                
            # Extract Features (from DB precomputed indicators)
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
            window_df = df.iloc[idx-lookback+1 : idx+1][feature_cols].copy()
            
            # Normalize window
            normalized_window = normalize_data(window_df)
            
            X_local.append(normalized_window.values)
            
            # Calculate Target Score (Actual Return for verification)
            current_close = df.iloc[idx]['close']
            future_close = df.iloc[idx+horizon]['close']
            score = (future_close - current_close) / current_close * 100
            y_local.append(score)
            
            # Metadata
            metadata_local.append({
                'symbol': symbol,
                'date': df.iloc[idx]['date'],
                'close': current_close,
                'future_close': future_close,
                'actual_return': score
            })
            
        return X_local, y_local, metadata_local
    except Exception as e:
        return [], [], []

def generate_backtest_data(lookback=10, horizon=3, limit_stocks=None, start_date=None, end_date=None, use_multiprocessing=True):
    """
    Generate data for backtesting with metadata.
    """
    all_stocks = get_all_stocks()
    if limit_stocks:
        all_stocks = all_stocks.head(limit_stocks)
        
    symbols = all_stocks['symbol'].tolist()
    
    X_all = []
    y_all = []
    metadata_all = []
    
    if use_multiprocessing:
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        
        tasks = [(symbol, lookback, horizon, start_date, end_date) for symbol in symbols]
        num_workers = min(cpu_count(), 8)
        print(f"Generating backtest data from {len(symbols)} stocks using {num_workers} cores...")
        
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_stock_backtest, tasks, chunksize=10), total=len(tasks), unit="stock"))
            
        for X_local, y_local, meta_local in results:
            if X_local:
                X_all.extend(X_local)
                y_all.extend(y_local)
                metadata_all.extend(meta_local)
    else:
        from tqdm import tqdm
        print(f"Generating backtest data from {len(symbols)} stocks (single-threaded)...")
        for symbol in tqdm(symbols, unit="stock"):
            X_local, y_local, meta_local = process_stock_backtest((symbol, lookback, horizon, start_date, end_date))
            if X_local:
                X_all.extend(X_local)
                y_all.extend(y_local)
                metadata_all.extend(meta_local)
            
    if not X_all:
        return np.array([]), np.array([]), []
        
    # Print daily matches summary (dates and symbols with J<13)
    try:
        meta_df = pd.DataFrame(metadata_all)
        if not meta_df.empty and 'date' in meta_df.columns and 'symbol' in meta_df.columns:
            # Ensure date as str and sorted
            meta_df['date'] = meta_df['date'].astype(str)
            daily_groups = meta_df.groupby('date')['symbol'].apply(list)
            print("\n[DAILY MATCHES] J<13 triggers per day:")
            for d in sorted(daily_groups.index):
                syms = daily_groups.loc[d]
                preview = ", ".join(syms[:10]) + (" ..." if len(syms) > 10 else "")
                print(f"  {d}: {len(syms)} symbols: {preview}")
        else:
            print("[DAILY MATCHES] No matches found in generated metadata.")
    except Exception as _:
        # Non-blocking logging
        pass

    return np.array(X_all), np.array(y_all), metadata_all

if __name__ == "__main__":
    # Test generation
    X, y = generate_training_data(limit_stocks=50)
    print(f"Generated {len(X)} samples.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
