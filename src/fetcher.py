import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import time
from database import init_db, save_stocks, save_daily_data, get_last_date, get_all_stocks, delete_daily_data_range
from tqdm import tqdm
from indicators import compute_and_update_indicators, ensure_indicator_columns
import multiprocessing as mp

# Initialize Tushare
TOKEN = '72e098f1a916bb0ecc08ba3165108f3116bf00c3b493a405d00f6940'
ts.set_token(TOKEN)
pro = ts.pro_api()

# Global variable for worker processes
_rate_limiter_state = None

def init_worker(shared_state):
    """Initialize global shared state for worker processes."""
    global _rate_limiter_state
    _rate_limiter_state = shared_state

def wait_for_token():
    """
    Wait for a rate limit token.
    Shared state: (lock, req_count, window_start)
    """
    global _rate_limiter_state
    if _rate_limiter_state is None:
        return # Should not happen in worker if initialized correctly
        
    lock, req_count, window_start = _rate_limiter_state
    
    while True:
        with lock:
            now = time.time()
            # Check if window needs reset
            if now - window_start.value > 60:
                req_count.value = 0
                window_start.value = now
            
            # Check limit
            if req_count.value < 490:
                req_count.value += 1
                return # Acquired
            
            # If limit reached, calculate wait time
            wait_time = 60 - (now - window_start.value)
            if wait_time < 0:
                wait_time = 0
                
        # Sleep outside lock
        if wait_time > 0:
            time.sleep(wait_time + 0.1) # Sleep a bit extra to be safe

def fetch_daily_safe(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch with Rate Limiting and Infinite Retry.
    """
    while True:
        try:
            # 1. Rate Limit
            wait_for_token()
            
            # 2. Fetch
            df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
            
            if df is None:
                return pd.DataFrame()
                
            if df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                'ts_code': 'symbol',
                'trade_date': 'date',
                'vol': 'volume'
            })
            df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}. Retrying in 10s...")
            time.sleep(10)

def get_trading_dates(start_date: str, end_date: str) -> list:
    """Fetch trading dates from Tushare trade calendar. Returns list of 'YYYYMMDD'."""
    try:
        cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open=1)
        if cal is None or cal.empty:
            print(f"Warning: trade_cal returned empty for {start_date}-{end_date}, fallback to natural dates.")
            return []
        # Ensure ascending order and uniqueness
        dates = cal['cal_date'].astype(str).tolist()
        dates = sorted(set(dates))
        return dates
    except Exception as e:
        print(f"Error fetching trade calendar: {e}. Fallback to natural dates.")
        return []

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

def fetch_daily_for_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV for a single symbol between [start_date, end_date]."""
    try:
        df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            'ts_code': 'symbol',
            'trade_date': 'date',
            'vol': 'volume'
        })
        df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        return df
    except Exception as e:
        print(f"Error fetching daily data for {symbol}: {e}")
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

def get_counts_by_date_range(start_date: str, end_date: str):
    """Return dict: {date: rows_count} for dates within [start_date, end_date]."""
    import sqlite3
    conn = sqlite3.connect('data/stock_data.db')
    try:
        df = pd.read_sql(
            'SELECT date, COUNT(*) AS rows_count FROM daily_prices WHERE date BETWEEN ? AND ? GROUP BY date',
            conn,
            params=(start_date, end_date)
        )
        # Debug: print result length
        print(f"[DEBUG] get_counts_by_date_range rows={len(df)} for {start_date}~{end_date}")
        return {row['date']: int(row['rows_count']) for _, row in df.iterrows()}
    except Exception:
        return {}
    finally:
        conn.close()

def update_all(lookback_years=2, limit=None, progress_callback=None, should_stop_func=None, force_reload_days=0):
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

    # Force reload logic
    if force_reload_days > 0:
        reload_start = end_date - timedelta(days=force_reload_days)
        reload_start_str = reload_start.strftime('%Y%m%d')
        reload_end_str = end_date.strftime('%Y%m%d')
        print(f"Force reloading data from {reload_start_str} to {reload_end_str}...")
        delete_daily_data_range(reload_start_str, reload_end_str)
    
    # Generate trading dates in range (prefer trade calendar)
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    trading_dates = get_trading_dates(start_str, end_str)
    # Debug: print trading calendar fetch summary
    try:
        print(f"[DEBUG] Trading calendar {start_str}~{end_str}: count={len(trading_dates)}")
        if trading_dates:
            print(f"[DEBUG] First={trading_dates[0]} Last={trading_dates[-1]}")
    except Exception:
        pass
    using_trade_calendar = True
    if trading_dates:
        all_dates = trading_dates
    else:
        using_trade_calendar = False
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        all_dates = [d.strftime("%Y%m%d") for d in date_range]
    
    # Filter out existing dates (date-level presence) and detect incomplete dates (row count below threshold)
    print("Checking existing data...")
    now_str = datetime.now().strftime('%Y%m%d')
    counts_map = get_counts_by_date_range(all_dates[0], all_dates[-1])
    missing_dates = [d for d in all_dates if d not in counts_map]
    # Union sets
    dates_to_fetch = sorted(set(missing_dates))
    
    # Filter out future dates (if any) and weekends (Tushare might handle, but good to skip if we know)
    # Actually Tushare returns empty for non-trading days, so it's fine to try fetching.
    # But to save API calls, we could use a trading calendar. 
    # For now, let's just try fetching all missing dates.
    
    total_days = len(dates_to_fetch)
    print(f"Found {len(missing_dates)} missing days")
    print(f"Total days scheduled to fetch: {total_days} (out of {len(all_dates)} days in range).")
    
    if total_days == 0:
        print("All data up to date.")
        if progress_callback:
            progress_callback(1.0, "All data up to date.")
        return

    # Process in batches of 20 days
    batch_size = 1
    # Sort missing dates to fetch in order
    dates_to_fetch.sort()
    
    processed_days = 0
    total_records_saved = 0
    pb = tqdm(total=total_days, desc="Fetching days", unit="day")
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
        # Split batch_dates into contiguous sub-ranges with max natural span <= 30 days
        def _dt(s: str):
            return datetime.strptime(s, '%Y%m%d')
        sub_ranges = []
        j = 0
        while j < len(batch_dates):
            start = batch_dates[j]
            k = j
            while k + 1 < len(batch_dates) and (_dt(batch_dates[k + 1]) - _dt(start)).days <= 2:
                k += 1
            sub_ranges.append(batch_dates[j:k + 1])
            j = k + 1

        for sub in sub_ranges:
            try:
                range_start = sub[0]
                range_end = sub[-1]
                # rate limit
                time.sleep(0.2)
                df = pro.daily(start_date=range_start, end_date=range_end)
                if not df.empty:
                    df = df.rename(columns={
                        'ts_code': 'symbol',
                        'trade_date': 'date',
                        'vol': 'volume'
                    })
                    df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                    dates_set = set(sub)
                    df = df[df['date'].isin(dates_set)]
                    present_dates = set(df['date'].unique().tolist())
                    missing_in_sub = [d for d in sub if d not in present_dates]
                    print(f"  Fetched sub-range {range_start}~{range_end}: {len(df)} records, days covered={len(present_dates)}")
                    if missing_in_sub:
                        print(f"  WARNING: No data returned for days: {', '.join(missing_in_sub[:10])}{' ...' if len(missing_in_sub)>10 else ''}")
                    batch_df_list.append(df)
                    total_records_saved += len(df)
                else:
                    print(f"  WARNING: Empty result for sub-range {range_start}~{range_end}")
            except Exception as e:
                print(f"  Error fetching sub-range {sub[0]}~{sub[-1]}: {e}")
                time.sleep(1)
            processed_days += len(sub)
            pb.update(len(sub))

        # Save batch
        if batch_df_list:
            full_batch_df = pd.concat(batch_df_list, ignore_index=True)
            inserted = save_daily_data(full_batch_df)
            print(f"  Saved {len(full_batch_df)} records for batch. Inserted={inserted}")
            # Compute indicators for symbols present in this batch (recent window)
            try:
                from indicators import compute_and_update_indicators_batch
                sym_list = full_batch_df['symbol'].unique().tolist()
                # Limit compute window to sub-range +/- 150 days
                range_start = min(full_batch_df['date'])
                range_end = max(full_batch_df['date'])
                # Expand window by 150 days on both ends
                def _dt(s: str):
                    return datetime.strptime(s, '%Y%m%d')
                exp_start = (_dt(range_start) - timedelta(days=150)).strftime('%Y%m%d')
                exp_end = (_dt(range_end) + timedelta(days=5)).strftime('%Y%m%d')
                
                print(f"  Computing indicators for {len(sym_list)} symbols (batch)...")
                updated_count = compute_and_update_indicators_batch(sym_list, start_date=exp_start, end_date=exp_end)
                print(f"  Indicators updated: {updated_count} records.")
            except Exception as e:
                print(f"  WARNING: Failed to compute indicators for batch: {e}")
        else:
            print("  No records in this batch (likely non-trading days).")
    pb.close()
    print(f"Completed: {processed_days}/{total_days} days processed. Total records saved: {total_records_saved}")
            
    if progress_callback:
        progress_callback(1.0, "Update complete!")

def task(args):
    # Uses global _rate_limiter_state set by initializer
    sym, start_date, end_date = args
    df = fetch_daily_safe(sym, start_date, end_date)
    return sym, df
    
def update_by_symbols(start_date: str = None, end_date: str = None, years: int = 2,
                      limit: int = None, workers: int = 4, write_chunk_rows: int = 300000,
                      filter_tradable: bool = True):
    """
    Re-fetch daily data per symbol (parallel), then batch write to DB via a single writer.
    Uses Rate Limiting and Retry.
    """
    init_db()

    # Resolve date range
    if not (start_date and end_date):
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365 * years)
        start_date = start_dt.strftime('%Y%m%d')
        end_date = end_dt.strftime('%Y%m%d')

    # Prepare symbols
    stocks_df = get_all_stocks(filter_tradable=filter_tradable)
    symbols = stocks_df['symbol'].tolist()
    if limit:
        symbols = symbols[:limit]
    if not symbols:
        print('No symbols found in DB. Fetching stock list from Tushare...')
        stocks_df = fetch_stock_list()
        if not stocks_df.empty:
            save_stocks(stocks_df)
            print(f"Saved {len(stocks_df)} stocks.")
            symbols = stocks_df['symbol'].tolist()
            if limit:
                symbols = symbols[:limit]
        else:
            print('Failed to fetch stock list. Aborting.')
            return

    print(f'Fetching by symbols: {len(symbols)} symbols, {start_date} ~ {end_date}, workers={workers}')

    # Parallel fetch; single writer aggregation
    workers = max(1, min(workers, mp.cpu_count()))

    # Shared State for Rate Limiting
    manager = mp.Manager()
    lock = manager.Lock()
    req_count = manager.Value('i', 0)
    window_start = manager.Value('d', time.time())
    shared_state = (lock, req_count, window_start)



    total_inserted = 0
    buffer_rows = []
    buffer_count = 0

    def flush_buffer():
        nonlocal buffer_rows, buffer_count, total_inserted
        if not buffer_rows:
            return
        merged = pd.concat(buffer_rows, ignore_index=True)
        inserted = save_daily_data(merged)
        total_inserted += (inserted or 0)
        buffer_rows = []
        buffer_count = 0

    try:
        with mp.Pool(processes=workers, initializer=init_worker, initargs=(shared_state,)) as pool:
            for sym, df in tqdm(pool.imap_unordered(task, [(s, start_date, end_date) for s in symbols], chunksize=1), total=len(symbols), unit='sym', desc='Fetch symbols'):
                if df is None or df.empty:
                    continue
                buffer_rows.append(df)
                buffer_count += len(df)
                if buffer_count >= write_chunk_rows:
                    flush_buffer()
    except Exception as e:
        print(f'Parallel fetch failed: {e}.')
        return

    flush_buffer()
    print(f'Total inserted rows: {total_inserted}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch and update A-share stock data.a")
    parser.add_argument("--years", "-y", type=int, default=2, help="Number of years of historical data to fetch (default: 2)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of stocks to update (for testing)")
    parser.add_argument("--mode", type=str, default="by_dates", choices=["by_dates","by_symbols"], help="Fetch mode: by_dates (existing) or by_symbols (parallel per-symbol)")
    parser.add_argument("--start", type=str, help="Start date YYYYMMDD (for by_symbols mode)")
    parser.add_argument("--end", type=str, help="End date YYYYMMDD (for by_symbols mode)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for by_symbols mode")
    parser.add_argument("--write-chunk-rows", type=int, default=300000, help="Flush writes when buffer reaches this many rows")
    parser.add_argument("--force-reload-days", type=int, default=0, help="Force delete and refetch data for the last N days")
    args = parser.parse_args()

    if args.mode == "by_symbols":
        update_by_symbols(start_date=args.start, end_date=args.end, years=args.years, limit=args.limit,
                          workers=args.workers, write_chunk_rows=args.write_chunk_rows, filter_tradable=False)
    else:
        update_all(lookback_years=args.years, limit=args.limit, force_reload_days=args.force_reload_days)
