"""
Detect and repair missing daily price rows in SQLite DB.

Repairs options:
- refetch: fetch missing days from Tushare (preferred)

Additionally, you can purge erroneous rows on non-trading days using the Tushare
trading calendar (delete all daily_prices rows where date is not a valid trading day).

Usage examples:
  # Refetch missing days and purge non-trading rows for last 2 years
  python src/repair_data.py --years 2 --method refetch --purge

  # Refetch for explicit date range and purge
  python src/repair_data.py --start 20230101 --end 20231231 --method refetch --purge

  # Specific symbols only
  python src/repair_data.py --years 2 --symbols 000001.SZ,600000.SH --method refetch --purge
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import DB_PATH, get_all_stocks
from src.fetcher import get_trading_dates, pro  # reuse initialized Tushare client
from src.database import save_daily_data

def fmt(dt: datetime) -> str:
    return dt.strftime('%Y%m%d')

def resolve_date_range(start: Optional[str], end: Optional[str], years: Optional[int]):
    if start and end:
        return start, end
    if years and years > 0:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365 * years)
        return fmt(start_dt), fmt(end_dt)
    return None, None

def get_symbol_dates(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        params = []
        query = 'SELECT date, close FROM daily_prices WHERE symbol = ?'
        params.append(symbol)
        if start:
            query += ' AND date >= ?'
            params.append(start)
        if end:
            query += ' AND date <= ?'
            params.append(end)
        query += ' ORDER BY date'
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

def get_prev_close(symbol: str, date: str) -> Optional[float]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute('SELECT close FROM daily_prices WHERE symbol = ? AND date < ? ORDER BY date DESC LIMIT 1', (symbol, date))
        row = cur.fetchone()
        return float(row[0]) if row else None
    finally:
        conn.close()

def refetch_missing(symbol: str, start: str, end: str, missing_days: List[str]) -> int:
    if not missing_days:
        return 0
    try:
        df = pro.daily(ts_code=symbol, start_date=start, end_date=end)
        if df is None or df.empty:
            return 0
        df = df.rename(columns={
            'ts_code': 'symbol',
            'trade_date': 'date',
            'vol': 'volume'
        })
        df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        # keep only missing dates
        s_missing = set(missing_days)
        df = df[df['date'].isin(s_missing)]
        if df.empty:
            return 0
        return save_daily_data(df) or 0
    except Exception:
        return 0

def purge_non_trading_rows(start: Optional[str], end: Optional[str], use_calendar: bool = True) -> int:
    """Delete rows in daily_prices where date is not a valid trading day in [start, end]."""
    # Determine calendar dates
    if use_calendar:
        s = start or '00000000'
        e = end or '99999999'
        tdates = set(get_trading_dates(s, e))
    else:
        # Fallback to natural dates
        if start and end:
            start_dt = datetime.strptime(start, '%Y%m%d')
            end_dt = datetime.strptime(end, '%Y%m%d')
        else:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365)
        tdates = {fmt(d) for d in pd.date_range(start=start_dt, end=end_dt, freq='D')}

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # Get distinct dates present in DB within range
        q = 'SELECT DISTINCT date FROM daily_prices'
        params = []
        if start:
            q += ' WHERE date >= ?'
            params.append(start)
        if end:
            q += ' AND date <= ?' if params else ' WHERE date <= ?'
            params.append(end)
        present_df = pd.read_sql_query(q, conn, params=params)
        present_dates = present_df['date'].tolist() if not present_df.empty else []
        to_delete = [d for d in present_dates if d not in tdates]
        deleted = 0
        if to_delete:
            from tqdm import tqdm
            pb = tqdm(to_delete, desc='Purge non-trading dates', unit='day')
            for d in pb:
                cur.execute('DELETE FROM daily_prices WHERE date = ?', (d,))
                conn.commit()
                deleted += cur.rowcount if hasattr(cur, 'rowcount') else 0
            pb.close()
        return deleted
    finally:
        conn.close()

def detect_and_repair(symbols: List[str], start: Optional[str], end: Optional[str], method: str = 'refetch', use_calendar: bool = True):
    # Prepare trading dates
    if use_calendar:
        s = start or '00000000'
        e = end or '99999999'
        tdates = get_trading_dates(s, e)
        if not tdates:
            # fallback natural
            if start and end:
                start_dt = datetime.strptime(start, '%Y%m%d')
                end_dt = datetime.strptime(end, '%Y%m%d')
            else:
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=365)
            tdates = [fmt(d) for d in pd.date_range(start=start_dt, end=end_dt, freq='D')]
    else:
        if start and end:
            start_dt = datetime.strptime(start, '%Y%m%d')
            end_dt = datetime.strptime(end, '%Y%m%d')
        else:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365)
        tdates = [fmt(d) for d in pd.date_range(start=start_dt, end=end_dt, freq='D')]

    total_inserted = 0
    pb = tqdm(symbols, desc='Repair symbols', unit='sym')
    for sym in pb:
        present_df = get_symbol_dates(sym, start, end)
        present_dates = set(present_df['date'].tolist()) if not present_df.empty else set()
        missing = [d for d in tdates if d not in present_dates]
        if not missing:
            pb.write(f'{sym}: OK (no missing)')
            continue
        pb.write(f'{sym}: missing {len(missing)} days')
        inserted = 0
        if method == 'refetch':
            inserted_refetch = refetch_missing(sym, start or tdates[0], end or tdates[-1], missing)
            inserted += inserted_refetch
            pb.write(f'  refetch inserted {inserted_refetch}')
            # recompute missing after refetch
            present_df = get_symbol_dates(sym, start, end)
            present_dates = set(present_df['date'].tolist()) if not present_df.empty else set()
            missing = [d for d in tdates if d not in present_dates]
        pb.write(f'  inserted total {inserted} rows for {sym}')
        total_inserted += inserted
    pb.close()
    print(f'Total inserted: {total_inserted}')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Detect and repair missing daily prices in DB')
    parser.add_argument('--start', type=str, help='Start date YYYYMMDD')
    parser.add_argument('--end', type=str, help='End date YYYYMMDD')
    parser.add_argument('--years', type=int, help='Range by last N years if start/end not provided')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols to process')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of symbols')
    parser.add_argument('--method', type=str, default='refetch', choices=['refetch'], help='Repair method')
    parser.add_argument('--purge', action='store_true', help='Purge non-trading day rows after refetch')
    parser.add_argument('--no-calendar', action='store_true', help='Do not use trading calendar (natural dates)')
    parser.add_argument('--no-filter', action='store_true', help='Do not filter ChiNext/STAR/BSE')
    args = parser.parse_args()

    start, end = resolve_date_range(args.start, args.end, args.years)
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
    else:
        # 默认过滤创业板/科创板/北交所等需权限标的
        df = get_all_stocks(filter_tradable=not args.no_filter)
        syms = df['symbol'].tolist()
    if args.limit:
        syms = syms[:args.limit]
    print(f'Processing {len(syms)} symbols, date range {start or "MIN"}~{end or "MAX"}, method={args.method}')
    detect_and_repair(syms, start, end, method=args.method, use_calendar=not args.no_calendar)
    if args.purge:
        print('Purging non-trading day rows ...')
        deleted = purge_non_trading_rows(start, end, use_calendar=not args.no_calendar)
        print(f'Purged rows: {deleted}')

if __name__ == '__main__':
    main()
