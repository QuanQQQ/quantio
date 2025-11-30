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

from src.indicators import ensure_indicator_columns, compute_and_update_indicators
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Precompute indicators into stock_data.db')
    parser.add_argument('--start', type=str, help='Start date YYYYMMDD')
    parser.add_argument('--end', type=str, help='End date YYYYMMDD')
    parser.add_argument('--years', type=int, help='Compute for last N years (ignored if start/end provided)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g., 000001.SZ,600000.SH)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of symbols (for testing)')
    parser.add_argument('--no-filter', action='store_true', help='Do not filter ChiNext/STAR/BSE')
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

    def worker(args):
        sym, s, e = args
        try:
            return compute_and_update_indicators(sym, start_date=s, end_date=e) or 0
        except Exception as exc:
            print(f'  WARNING: {sym} failed: {exc}')
            return 0

    tasks = [(sym, start, end) for sym in symbols]

    try:
        from tqdm import tqdm
        with mp.Pool(processes=workers) as pool:
            for changed in tqdm(pool.imap_unordered(worker, tasks, chunksize=32), total=len(tasks), unit='sym'):
                total_changes += changed
    except Exception as e:
        print(f'Parallel execution failed ({e}), falling back to single process.')
        for sym in symbols:
            total_changes += worker((sym, start, end))

    dur = time.time() - start_time
    print(f'Completed indicators for {len(symbols)} symbols. Total DB changes: {total_changes}. Elapsed: {dur:.1f}s (workers={workers})')

if __name__ == '__main__':
    main()
