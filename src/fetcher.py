import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import time
from database import init_db, save_stocks, save_daily_data, get_last_date, get_all_stocks

# Initialize Tushare
TOKEN = '72e098f1a916bb0ecc08ba3165108f3116bf00c3b493a405d00f6940'
ts.set_token(TOKEN)
pro = ts.pro_api()

def fetch_stock_list():
    """
    Fetch all A-share stocks list using Tushare.
    Returns DataFrame with columns: symbol, name, sector, listing_date
    """
    print("Fetching stock list from Tushare...")
    try:
        # stock_basic: list_status='L' (Listed)
        # We need ts_code (as unique ID), name, industry, list_date. 
        # We don't need the 'symbol' field from Tushare if we are going to use 'ts_code' as our 'symbol'.
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,list_date')
        
        df = df.rename(columns={
            'ts_code': 'symbol',
            'industry': 'sector',
            'list_date': 'listing_date'
        })
        
        # Ensure columns exist
        if 'sector' not in df.columns:
            df['sector'] = 'Unknown'
        
        return df[['symbol', 'name', 'sector', 'listing_date']]
    except Exception as e:
        print(f"Error fetching stock list: {e}")
        return pd.DataFrame()

def fetch_daily_data_batch(symbols, start_date, end_date):
    """
    Fetch daily data for multiple stocks using Tushare.
    """
    try:
        # Join symbols with comma
        ts_code_str = ",".join(symbols)
        
        # daily: ts_code, start_date, end_date
        df = pro.daily(ts_code=ts_code_str, start_date=start_date, end_date=end_date)
        
        if df.empty:
            return pd.DataFrame()
            
        # Rename columns
        df = df.rename(columns={
            'ts_code': 'symbol',
            'trade_date': 'date',
            'vol': 'volume'
        })
        
        df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        
        return df
    except Exception as e:
        print(f"Error fetching batch data: {e}")
        return pd.DataFrame()

def get_existing_dates():
    """
    Get all dates that already exist in the daily_prices table.
    Returns a set of dates (str YYYYMMDD).
    """
    import sqlite3
    conn = sqlite3.connect('data/stock_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT date FROM daily_prices")
        dates = {row[0] for row in cursor.fetchall()}
        return dates
    except Exception:
        return set()
    finally:
        conn.close()

def get_date_row_count(date: str) -> int:
    """
    Get row count in daily_prices for a specific date (YYYYMMDD).
    """
    import sqlite3
    conn = sqlite3.connect('data/stock_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE date = ?", (date,))
        cnt = cursor.fetchone()[0]
        return int(cnt or 0)
    except Exception:
        return 0
    finally:
        conn.close()

def update_all(lookback_years=2, limit=None, progress_callback=None, should_stop_func=None, min_rows_per_day=3000):
    """
    Main function to update all data (Time-based iteration).
    """
    init_db()
    
    # 1. Update stock list
    if progress_callback:
        progress_callback(0, "Fetching stock list...")
        
    stocks_df = fetch_stock_list()
    if not stocks_df.empty:
        save_stocks(stocks_df)
        print(f"Saved {len(stocks_df)} stocks.")
    else:
        print("Failed to fetch stock list.")
        return

    # 2. Determine Date Range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*lookback_years)
    
    # Generate all dates in range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    all_dates = [d.strftime("%Y%m%d") for d in date_range]
    
    # Filter out existing dates (date-level presence) and detect incomplete dates (row count below threshold)
    print("Checking existing data...")
    existing_dates = get_existing_dates()
    missing_dates = [d for d in all_dates if d not in existing_dates]
    # Even if a date exists, it may be partially saved (e.g., API hiccup). Re-fetch incomplete dates.
    incomplete_dates = []
    for d in all_dates:
        # Skip future dates
        if d > datetime.now().strftime('%Y%m%d'):
            continue
        cnt = get_date_row_count(d)
        if cnt > 0 and cnt < min_rows_per_day:
            incomplete_dates.append(d)
    # Union sets
    dates_to_fetch = sorted(set(missing_dates + incomplete_dates))
    
    # Filter out future dates (if any) and weekends (Tushare might handle, but good to skip if we know)
    # Actually Tushare returns empty for non-trading days, so it's fine to try fetching.
    # But to save API calls, we could use a trading calendar. 
    # For now, let's just try fetching all missing dates.
    
    total_days = len(dates_to_fetch)
    print(f"Found {len(missing_dates)} missing days and {len(incomplete_dates)} incomplete days.")
    print(f"Total days scheduled to fetch: {total_days} (out of {len(all_dates)} days in range).")
    
    if total_days == 0:
        print("All data up to date.")
        if progress_callback:
            progress_callback(1.0, "All data up to date.")
        return

    # Process in batches of 20 days
    batch_size = 20
    # Sort missing dates to fetch in order
    dates_to_fetch.sort()
    
    for i in range(0, total_days, batch_size):
        # Check stop condition
        if should_stop_func and should_stop_func():
            print("Update stopped by user.")
            return

        batch_dates = dates_to_fetch[i:i+batch_size]
        
        if progress_callback:
            progress = min(i / total_days, 1.0)
            progress_callback(progress, f"Updating batch {i}/{total_days} (Date: {batch_dates[0]})")
        
        print(f"Fetching batch of {len(batch_dates)} days starting {batch_dates[0]}...")
        
        batch_df_list = []
        
        for date in batch_dates:
            try:
                # Fetch all stocks for this date
                # rate limit
                time.sleep(0.2)
                
                df = pro.daily(trade_date=date)
                
                if not df.empty:
                    # Rename columns
                    df = df.rename(columns={
                        'ts_code': 'symbol',
                        'trade_date': 'date',
                        'vol': 'volume'
                    })
                    df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                    batch_df_list.append(df)
                    print(f"  Fetched {date}: {len(df)} records.")
                else:
                    # Likely a non-trading day
                    # print(f"  No data for {date} (Non-trading day?)")
                    pass
            except Exception as e:
                print(f"  Error fetching {date}: {e}")
                time.sleep(1)
        
        # Save batch
        if batch_df_list:
            full_batch_df = pd.concat(batch_df_list, ignore_index=True)
            save_daily_data(full_batch_df)
            print(f"  Saved {len(full_batch_df)} records for batch.")
            
    if progress_callback:
        progress_callback(1.0, "Update complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch and update A-share stock data.")
    parser.add_argument("--years", "-y", type=int, default=2, help="Number of years of historical data to fetch (default: 2)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of stocks to update (for testing)")
    
    args = parser.parse_args()
    
    update_all(lookback_years=args.years, limit=args.limit)
