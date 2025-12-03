import pandas as pd
import numpy as np
from database import get_all_stocks, get_stock_daily, get_stock_daily_multi
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
    # args may include optional flag: use_db_fallback
    symbol, lookback, horizon, start_date, end_date = args[:5]
    use_db_fallback = False
    if len(args) >= 6:
        use_db_fallback = bool(args[5])
    
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
            # Calculate Target Score (Scheme B: Max Gain - Max Drawdown before Peak)
            current_close = df.iloc[idx]['close']
            future_window = df.iloc[idx+1 : idx+horizon+1]
            
            if future_window.empty:
                score = 0
            else:
                # Find Peak (Max High)
                future_highs = future_window['high'].values
                peak_idx = np.argmax(future_highs)
                peak_price = future_highs[peak_idx]
                
                # Max Gain Pct
                max_gain_pct = (peak_price - current_close) / current_close
                
                # Max Loss Pct (Drawdown from entry BEFORE peak)
                # Slice lows up to peak_idx (inclusive)
                future_lows = future_window['low'].values
                pre_peak_lows = future_lows[:peak_idx+1]
                min_low_pre_peak = np.min(pre_peak_lows)
                
                max_loss_pct = (current_close - min_low_pre_peak) / current_close
                max_loss_pct = max(0, max_loss_pct)
                
                score = (max_gain_pct - max_loss_pct) * 100
            
            y_local.append(score)
            
        return X_local, y_local
    except Exception as e:
        # print(f"Error processing {symbol}: {e}")
        return [], []

def process_batch_training(args):
    batch_syms, lookback, horizon, start_date, end_date, use_db_fallback = args
    X_res = []
    y_res = []
    df_all = get_stock_daily_multi(batch_syms, start_date=start_date, end_date=end_date)
    if df_all.empty:
        return X_res, y_res
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    for symbol in batch_syms:
        df = df_all[df_all['symbol'] == symbol].copy()
        if len(df) < lookback + horizon + 30:
            continue
        if any(col not in df.columns for col in feature_cols):
            continue
        df.dropna(subset=base_cols + feature_cols, inplace=True)
        if df.empty:
            continue
        df = df.reset_index(drop=True)
        potential_indices = df[df['j'] < 13].index
        for idx in potential_indices:
            if idx < lookback or idx >= len(df) - horizon:
                continue
            window_df = df.iloc[idx-lookback+1 : idx+1][feature_cols].copy()

            normalized_window = normalize_data(window_df)
            X_res.append(normalized_window.values)
            current_close = df.iloc[idx]['close']
            future_window = df.iloc[idx+1 : idx+horizon+1]
            
            if future_window.empty:
                score = 0
            else:
                future_highs = future_window['high'].values
                peak_idx = np.argmax(future_highs)
                peak_price = future_highs[peak_idx]
                
                max_gain_pct = (peak_price - current_close) / current_close
                
                future_lows = future_window['low'].values
                pre_peak_lows = future_lows[:peak_idx+1]
                min_low_pre_peak = np.min(pre_peak_lows)
                
                max_loss_pct = max(0, (current_close - min_low_pre_peak) / current_close)
                score = (max_gain_pct - max_loss_pct) * 100
                
            y_res.append(score)
    return X_res, y_res


def generate_training_data(lookback=10, horizon=3, limit_stocks=None, start_date=None, end_date=None, use_multiprocessing=True, use_db_fallback=False, batch_fetch=False, batch_size=50):
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
    
    if use_multiprocessing and not batch_fetch:
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        
        # Prepare arguments for each worker
        tasks = [(symbol, lookback, horizon, start_date, end_date, use_db_fallback) for symbol in symbols]
        
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
    elif use_multiprocessing and batch_fetch:
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        
        tasks = []
        for i in range(0, len(symbols), batch_size):
            tasks.append((symbols[i:i+batch_size], lookback, horizon, start_date, end_date, use_db_fallback))
        num_workers = min(cpu_count(), 6)
        print(f"Generating data with multiprocessing batch fetch: {len(tasks)} batches using {num_workers} cores...")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_batch_training, tasks, chunksize=1), total=len(tasks), unit="batch"))
        for X_local, y_local in results:
            if X_local:
                X_all.extend(X_local)
                y_all.extend(y_local)
    else:
        # Single-threaded fallback
        from tqdm import tqdm
        if not batch_fetch:
            print(f"Generating data from {len(symbols)} stocks (single-threaded)...")
            for symbol in tqdm(symbols, unit="stock"):
                X_local, y_local = process_stock((symbol, lookback, horizon, start_date, end_date, use_db_fallback))
                if X_local:
                    X_all.extend(X_local)
                    y_all.extend(y_local)
        else:
            print(f"Generating data (batch fetch) from {len(symbols)} stocks, batch_size={batch_size}...")
            # Reduce DB I/O by fetching multiple symbols at once
            for i in tqdm(range(0, len(symbols), batch_size), unit="batch"):
                batch_syms = symbols[i:i+batch_size]
                df_all = get_stock_daily_multi(batch_syms, start_date=start_date, end_date=end_date)
                if df_all.empty:
                    continue
                feature_cols = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
                base_cols = ['open', 'high', 'low', 'close', 'volume']
                for symbol in batch_syms:
                    df = df_all[df_all['symbol'] == symbol].copy()
                    if len(df) < lookback + horizon + 30:
                        continue
                    if any(col not in df.columns for col in feature_cols):
                        continue
                    df.dropna(subset=base_cols + feature_cols, inplace=True)
                    if df.empty:
                        continue
                    df = df.reset_index(drop=True)
                    potential_indices = df[df['j'] < 13].index
                    for idx in potential_indices:
                        if idx < lookback or idx >= len(df) - horizon:
                            continue
                        window_df = df.iloc[idx-lookback+1 : idx+1][feature_cols].copy()
                      
                        normalized_window = normalize_data(window_df)
                        X_all.append(normalized_window.values)
                        current_close = df.iloc[idx]['close']
                        future_window = df.iloc[idx+1 : idx+horizon+1]
                        future_high = future_window['high'].max()
                        future_low = future_window['low'].min()
                        
                        max_gain_pct = (future_high - current_close) / current_close
                        max_loss_pct = max(0, (current_close - future_low) / current_close)
                        
                        score = (max_gain_pct - max_loss_pct) * 100
                        y_all.append(score)
            
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
    symbol, lookback, horizon, start_date, end_date = args[:5]
    use_db_fallback = False
    if len(args) >= 6:
        use_db_fallback = bool(args[5])
    
    try:
        # Calculate fetch start date with buffer
        # Buffer: lookback * 2 + 20 days to be safe (for weekends and holidays)
        if start_date:
            start_dt = datetime.strptime(str(start_date), "%Y%m%d")
            buffer_days = lookback * 2 + 20
            fetch_start_dt = start_dt - timedelta(days=buffer_days)
            fetch_start_date = fetch_start_dt.strftime("%Y%m%d")
        else:
            fetch_start_date = None
            
        # Fetch data (with precomputed indicators)
        df = get_stock_daily(symbol, start_date=fetch_start_date, end_date=end_date)

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
        
        # Parse start_date for filtering
        start_date_int = int(start_date) if start_date else 0
        
        for idx in potential_indices:
            if idx < lookback or idx >= len(df) - horizon:
                continue
            
            # Check if the date is within the requested backtest period
            current_date_str = df.iloc[idx]['date']
            # Ensure date format compatibility (assuming YYYYMMDD in DB)
            try:
                current_date_int = int(current_date_str.replace('-', '').replace('/', ''))
            except:
                continue
                
            if start_date and current_date_int < start_date_int:
                continue
                
            # Extract Features (from DB precomputed indicators)
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
            window_df = df.iloc[idx-lookback+1 : idx+1][feature_cols].copy()

            
            # Normalize window
            normalized_window = normalize_data(window_df)
            
            X_local.append(normalized_window.values)
            
            # Calculate Target Score (Scheme B: Time Dependent)
            current_close = df.iloc[idx]['close']
            future_window = df.iloc[idx+1 : idx+horizon+1]
            
            if future_window.empty:
                score = 0
                future_high = current_close
                future_low = current_close
            else:
                future_highs = future_window['high'].values
                peak_idx = np.argmax(future_highs)
                peak_price = future_highs[peak_idx]
                future_high = peak_price # For metadata
                
                max_gain_pct = (peak_price - current_close) / current_close
                
                future_lows = future_window['low'].values
                pre_peak_lows = future_lows[:peak_idx+1]
                min_low_pre_peak = np.min(pre_peak_lows)
                future_low = min_low_pre_peak # For metadata (showing the low that mattered)
                
                max_loss_pct = max(0, (current_close - min_low_pre_peak) / current_close)
                score = (max_gain_pct - max_loss_pct) * 100
            
            y_local.append(score)
            
            # Metadata
            metadata_local.append({
                'symbol': symbol,
                'date': df.iloc[idx]['date'],
                'close': current_close,
                'future_high': future_high,
                'future_low': future_low, # This is now the pre-peak low
                'score': score
            })
            
        return X_local, y_local, metadata_local
    except Exception as e:
        return [], [], []

def generate_backtest_data(lookback=10, horizon=3, limit_stocks=None, start_date=None, end_date=None, use_multiprocessing=True, use_db_fallback=False, batch_fetch=False, batch_size=50):
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
    
    if use_multiprocessing and not batch_fetch:
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        
        tasks = [(symbol, lookback, horizon, start_date, end_date, use_db_fallback) for symbol in symbols]
        num_workers = min(cpu_count(), 8)
        print(f"Generating backtest data from {len(symbols)} stocks using {num_workers} cores...")
        
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_stock_backtest, tasks, chunksize=10), total=len(tasks), unit="stock"))
            
        for X_local, y_local, meta_local in results:
            if X_local:
                X_all.extend(X_local)
                y_all.extend(y_local)
                metadata_all.extend(meta_local)
    elif use_multiprocessing and batch_fetch:
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        def process_batch_backtest(args):
            batch_syms, lookback, horizon, start_date, end_date, use_db_fallback = args
            X_res = []
            y_res = []
            meta_res = []
            df_all = get_stock_daily_multi(batch_syms, start_date=start_date, end_date=end_date)
            if df_all.empty:
                return X_res, y_res, meta_res
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
            base_cols = ['open', 'high', 'low', 'close', 'volume']
            start_date_int = int(start_date) if start_date else 0
            for symbol in batch_syms:
                df = df_all[df_all['symbol'] == symbol].copy()
                if len(df) < lookback + horizon + 30:
                    continue
                if any(col not in df.columns for col in feature_cols):
                    continue
                df.dropna(subset=base_cols + feature_cols, inplace=True)
                if df.empty:
                    continue
                df = df.reset_index(drop=True)
                potential_indices = df[df['j'] < 13].index
                for idx in potential_indices:
                    if idx < lookback or idx >= len(df) - horizon:
                        continue
                    current_date_str = df.iloc[idx]['date']
                    try:
                        current_date_int = int(str(current_date_str).replace('-', '').replace('/', ''))
                    except Exception:
                        continue
                    if start_date and current_date_int < start_date_int:
                        continue
                    window_df = df.iloc[idx-lookback+1 : idx+1][feature_cols].copy()
                 
                    normalized_window = normalize_data(window_df)
                    X_res.append(normalized_window.values)
                    current_close = df.iloc[idx]['close']
                    future_window = df.iloc[idx+1 : idx+horizon+1]
                    
                    if future_window.empty:
                        score = 0
                        future_high = current_close
                        future_low = current_close
                    else:
                        future_highs = future_window['high'].values
                        peak_idx = np.argmax(future_highs)
                        peak_price = future_highs[peak_idx]
                        future_high = peak_price
                        
                        max_gain_pct = (peak_price - current_close) / current_close
                        
                        future_lows = future_window['low'].values
                        pre_peak_lows = future_lows[:peak_idx+1]
                        min_low_pre_peak = np.min(pre_peak_lows)
                        future_low = min_low_pre_peak
                        
                        max_loss_pct = max(0, (current_close - min_low_pre_peak) / current_close)
                        score = (max_gain_pct - max_loss_pct) * 100
                    
                    y_res.append(score)
                    meta_res.append({
                        'symbol': symbol,
                        'date': df.iloc[idx]['date'],
                        'close': current_close,
                        'future_high': future_high,
                        'future_low': future_low,
                        'score': score
                    })
            return X_res, y_res, meta_res

        tasks = []
        for i in range(0, len(symbols), batch_size):
            tasks.append((symbols[i:i+batch_size], lookback, horizon, start_date, end_date, use_db_fallback))
        num_workers = min(cpu_count(), 8)
        print(f"Generating backtest data with multiprocessing batch fetch: {len(tasks)} batches using {num_workers} cores...")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_batch_backtest, tasks, chunksize=1), total=len(tasks), unit="batch"))
        for X_local, y_local, meta_local in results:
            if X_local:
                X_all.extend(X_local)
                y_all.extend(y_local)
                metadata_all.extend(meta_local)
    else:
        from tqdm import tqdm
        if not batch_fetch:
            print(f"Generating backtest data from {len(symbols)} stocks (single-threaded)...")
            for symbol in tqdm(symbols, unit="stock"):
                X_local, y_local, meta_local = process_stock_backtest((symbol, lookback, horizon, start_date, end_date, use_db_fallback))
                if X_local:
                    X_all.extend(X_local)
                    y_all.extend(y_local)
                    metadata_all.extend(meta_local)
        else:
            print(f"Generating backtest data (batch fetch) from {len(symbols)} stocks, batch_size={batch_size}...")
            for i in tqdm(range(0, len(symbols), batch_size), unit="batch"):
                batch_syms = symbols[i:i+batch_size]
                df_all = get_stock_daily_multi(batch_syms, start_date=start_date, end_date=end_date)
                if df_all.empty:
                    continue
                feature_cols = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
                base_cols = ['open', 'high', 'low', 'close', 'volume']
                start_date_int = int(start_date) if start_date else 0
                for symbol in batch_syms:
                    df = df_all[df_all['symbol'] == symbol].copy()
                    if len(df) < lookback + horizon + 30:
                        continue
                    if any(col not in df.columns for col in feature_cols):
                        continue
                    df.dropna(subset=base_cols + feature_cols, inplace=True)
                    if df.empty:
                        continue
                    df = df.reset_index(drop=True)
                    potential_indices = df[df['j'] < 13].index
                    for idx in potential_indices:
                        if idx < lookback or idx >= len(df) - horizon:
                            continue
                        current_date_str = df.iloc[idx]['date']
                        try:
                            current_date_int = int(str(current_date_str).replace('-', '').replace('/', ''))
                        except Exception:
                            continue
                        if start_date and current_date_int < start_date_int:
                            continue
                        window_df = df.iloc[idx-lookback+1 : idx+1][feature_cols].copy()
                     
                        normalized_window = normalize_data(window_df)
                        X_all.append(normalized_window.values)
                        current_close = df.iloc[idx]['close']
                        future_window = df.iloc[idx+1 : idx+horizon+1]
                        
                        if future_window.empty:
                            score = 0
                            future_high = current_close
                            future_low = current_close
                        else:
                            future_highs = future_window['high'].values
                            peak_idx = np.argmax(future_highs)
                            peak_price = future_highs[peak_idx]
                            future_high = peak_price
                            
                            max_gain_pct = (peak_price - current_close) / current_close
                            
                            future_lows = future_window['low'].values
                            pre_peak_lows = future_lows[:peak_idx+1]
                            min_low_pre_peak = np.min(pre_peak_lows)
                            future_low = min_low_pre_peak
                            
                            max_loss_pct = max(0, (current_close - min_low_pre_peak) / current_close)
                            score = (max_gain_pct - max_loss_pct) * 100
                            
                        y_all.append(score)
                        metadata_all.append({
                            'symbol': symbol,
                            'date': df.iloc[idx]['date'],
                            'close': current_close,
                            'future_high': future_high,
                            'future_low': future_low,
                            'score': score
                        })
            
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
