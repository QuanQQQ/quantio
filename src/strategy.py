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
    
    # Avoid division by zero
    range_high_low = high_list - low_list
    range_high_low.replace(0, np.nan, inplace=True)
    
    rsv = (df['close'] - low_list) / range_high_low * 100
    rsv.fillna(0, inplace=True) # Handle NaN
    
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    return df

def analyze_volume_power(df, window=20):
    """
    Analyze volume power: Avg Red Volume / Avg Green Volume.
    """
    if df.empty:
        return 0.0
        
    subset = df.tail(window).copy()
    
    # Red: Close >= Open
    red_mask = subset['close'] >= subset['open']
    green_mask = ~red_mask
    
    avg_red_vol = subset.loc[red_mask, 'volume'].mean()
    avg_green_vol = subset.loc[green_mask, 'volume'].mean()
    
    if pd.isna(avg_red_vol): avg_red_vol = 0
    if pd.isna(avg_green_vol) or avg_green_vol == 0: 
        return 999.0 if avg_red_vol > 0 else 0.0
        
    return avg_red_vol / avg_green_vol

def scan_stocks(progress_callback=None):
    """
    Scan all stocks for Z-ge Zhanfa criteria.
    """
    all_stocks = get_all_stocks()
    results = []
    
    total = len(all_stocks)
    
    # We need recent data for calculation. 
    # Fetching all data for all stocks is slow.
    # Optimization: We can query DB for last N days for each stock.
    # But we don't have a batch query for "last N rows of each group" in simple SQLite without window functions (SQLite has them but complex to construct efficient query).
    # Or we iterate.
    
    # Let's iterate. It might take a minute.
    
    today = datetime.now()
    start_date_str = (today - timedelta(days=60)).strftime("%Y%m%d") # Get enough data for KDJ
    
    for i, row in all_stocks.iterrows():
        if progress_callback and i % 10 == 0:
            progress_callback(i / total, f"Scanning {row['symbol']}...")
            
        symbol = row['symbol']
        
        # Get data
        df = get_stock_daily(symbol, start_date=start_date_str)
        
        if len(df) < 10:
            continue
            
        # Calculate KDJ
        df = calculate_kdj(df)
        
        # Get latest values
        latest = df.iloc[-1]
        j_value = latest['j']
        
        # Criteria 1: J < 13
        if j_value < 13:
            # Criteria 2: Volume Power
            vol_power = analyze_volume_power(df)
            
            results.append({
                'symbol': symbol,
                'name': row['name'],
                'date': latest['date'],
                'close': latest['close'],
                'j_value': round(j_value, 2),
                'vol_power': round(vol_power, 2)
            })
            
    return pd.DataFrame(results)
