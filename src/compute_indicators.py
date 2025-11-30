"""
Standalone script to precompute and store indicators (KDJ, MACD, trend lines, vol_ma5)
into the SQLite database for faster subsequent data access.

Usage examples:
  # Compute for last 2 years for all tradable stocks
  python src/compute_indicators.py --years 2

  # Compute for explicit date range
  python src/compute_indicators.py --start 20220101 --end 20241231

  # Compute for specific symbols (comma-separated)
  python src/compute_indicators.py --symbols 000001.SZ,600000.SH --years 3

  # Limit number of stocks for testing
  python src/compute_indicators.py --years 2 --limit 100
"""

import sys
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional

# Ensure project root in sys.path to allow `from src...` imports when running as script
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.indicators import ensure_indicator_columns, compute_indicator_updates, compute_updates_from_df
from src.database import DB_PATH
import sqlite3
from src.database import get_all_stocks

def fmt_dt(dt: datetime) -> str:
    return dt.strftime('%Y%m%d')

def resolve_date_range(start: Optional[str], end: Optional[str], years: Optional[int]) -> (Optional[str], Optional[str]):
    if start and end:
        return start, end
    if years and years > 0:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365 * years)
        return fmt_dt(start_dt), fmt_dt(end_dt)
    # Full history (None means no bound in DB query)
    return None, None

def parse_symbols(symbols_arg: Optional[str], filter_tradable: bool, limit: Optional[int]) -> List[str]:
    if symbols_arg:
        syms = [s.strip() for s in symbols_arg.split(',') if s.strip()]
        return syms[:limit] if limit else syms
    df = get_all_stocks(filter_tradable=filter_tradable)
    if df.empty:
        return []
    syms = df['symbol'].tolist()
    return syms[:limit] if limit else syms


def worker(args):
    sym, s, e = args
    try:
        return (sym, compute_indicator_updates(sym, start_date=s, end_date=e))
    except Exception as exc:
        print(f'  WARNING: {sym} failed: {exc}')
        return (sym, [])

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Precompute indicators into stock_data.db')
    parser.add_argument('--start', type=str, help='Start date YYYYMMDD')
    parser.add_argument('--end', type=str, help='End date YYYYMMDD')
    parser.add_argument('--years', type=int, help='Compute for last N years (ignored if start/end provided)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g., 000001.SZ,600000.SH)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of symbols (for testing)')
    parser.add_argument('--no-filter', action='store_true', help='Do not filter ChiNext/STAR/BSE')
    parser.add_argument('--bulk', action='store_true', help='Bulk read daily_prices and parallel compute per symbol (memory intensive)')
    parser.add_argument('--chunksize', type=int, default=300000, help='Rows per chunk when bulk reading (default: 300k)')
    args = parser.parse_args()

    start, end = resolve_date_range(args.start, args.end, args.years)
    symbols = parse_symbols(args.symbols, filter_tradable=not args.no_filter, limit=args.limit)

    if not symbols:
        print('No symbols to process.')
        return

    print(f'Preparing to compute indicators for {len(symbols)} symbols.')
    print(f'Date range: {start or "FULL"} ~ {end or "FULL"}')

    ensure_indicator_columns()

    total_changes = 0
    start_time = time.time()

    # Parallel execution options
    import multiprocessing as mp
    workers = min(mp.cpu_count(), 6)


    tasks = [(sym, start, end) for sym in symbols]

    # Single writer connection
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        try:
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
        except Exception:
            pass
        def write_updates(updates):
            if not updates:
                return
            CHUNK = 5000
            for i in range(0, len(updates), CHUNK):
                chunk = updates[i:i+CHUNK]
                before = conn.total_changes
                cur.executemany('''
                    UPDATE daily_prices
                    SET k = ?, d = ?, j = ?, macd = ?, macd_signal = ?, macd_hist = ?, short_trend = ?, long_trend = ?, vol_ma5 = ?
                    WHERE symbol = ? AND date = ?
                ''', chunk)
                conn.commit()
                return conn.total_changes - before

        if not args.bulk:
            try:
                from tqdm import tqdm
                with mp.Pool(processes=workers) as pool:
                    for sym, updates in tqdm(pool.imap_unordered(worker, tasks, chunksize=32), total=len(tasks), unit='sym'):
                        total_changes += (write_updates(updates) or 0)
            except Exception as e:
                print(f'Parallel execution failed ({e}), falling back to single process.')
                for sym in symbols:
                    _, updates = worker((sym, start, end))
                    total_changes += (write_updates(updates) or 0)
        else:
            # Bulk mode: read all rows in chunks and parallel compute per symbol from in-memory DataFrame
            params = [start or '00000000', end or '99999999']
            syms_filter_sql = ''
            if symbols and (not args.symbols):
                # symbols from DB; we won't add filter in SQL to reduce complexity
                pass
            query = (
                'SELECT symbol, date, open, high, low, close, volume '
                'FROM daily_prices WHERE date BETWEEN ? AND ? '
                'ORDER BY symbol, date'
            )
            try:
                from tqdm import tqdm
                chunks = pd.read_sql_query(query, conn, params=params, chunksize=args.chunksize)
                for chunk in tqdm(chunks, desc='Bulk chunks', unit='rows'):
                    # Optional filter by explicit symbols
                    if args.symbols:
                        chunk = chunk[chunk['symbol'].isin(symbols)]
                    groups = chunk.groupby('symbol')
                    task_df = [(sym, grp.copy()) for sym, grp in groups]
                    # Parallel compute updates from DataFrame
                    with mp.Pool(processes=workers) as pool:
                        for sym, updates in pool.imap_unordered(
                                lambda tup: (tup[0], compute_updates_from_df(tup[0], tup[1])), task_df, chunksize=16):
                            total_changes += (write_updates(updates) or 0)
            except Exception as e:
                print(f'Bulk mode failed ({e}), falling back to per-symbol mode.')
                try:
                    from tqdm import tqdm
                    with mp.Pool(processes=workers) as pool:
                        for sym, updates in tqdm(pool.imap_unordered(worker, tasks, chunksize=32), total=len(tasks), unit='sym'):
                            total_changes += (write_updates(updates) or 0)
                except Exception:
                    for sym in symbols:
                        _, updates = worker((sym, start, end))
                        total_changes += (write_updates(updates) or 0)
    finally:
        conn.close()

    dur = time.time() - start_time
    print(f'Completed indicators for {len(symbols)} symbols. Total DB changes: {total_changes}. Elapsed: {dur:.1f}s (workers={workers})')

if __name__ == '__main__':
    main()
