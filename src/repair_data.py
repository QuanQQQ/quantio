"""
Detect and repair missing daily price rows in SQLite DB.

Repairs options:
- refetch: fetch missing days from Tushare (preferred)
- ffill: forward-fill last known close for missing days (volume=0, amount=0)
- mixed: try refetch, fallback to ffill for remaining missing days

Usage examples:
  python src/repair_data.py --years 2 --method mixed --limit 200
  python src/repair_data.py --start 20230101 --end 20231231 --method refetch --symbols 000001.SZ,600000.SH
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

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

def insert_ffill_rows(symbol: str, dates: List[str]) -> int:
    # Build DataFrame for save_daily_data
    rows = []
    for d in dates:
        prev = get_prev_close(symbol, d)
        if prev is None:
            continue
        rows.append({
            'symbol': symbol,
            'date': d,
            'open': prev,
            'high': prev,
            'low': prev,
            'close': prev,
            'volume': 0.0,
            'amount': 0.0,
        })
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    return save_daily_data(df) or 0

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

def detect_and_repair(symbols: List[str], start: Optional[str], end: Optional[str], method: str = 'mixed', use_calendar: bool = True):
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
    for sym in symbols:
        present_df = get_symbol_dates(sym, start, end)
        present_dates = set(present_df['date'].tolist()) if not present_df.empty else set()
        missing = [d for d in tdates if d not in present_dates]
        if not missing:
            print(f'{sym}: OK (no missing)')
            continue
        print(f'{sym}: missing {len(missing)} days')
        inserted = 0
        if method in ('refetch', 'mixed'):
            inserted += refetch_missing(sym, start or tdates[0], end or tdates[-1], missing)
            # recompute missing after refetch
            present_df = get_symbol_dates(sym, start, end)
            present_dates = set(present_df['date'].tolist()) if not present_df.empty else set()
            missing = [d for d in tdates if d not in present_dates]
        if missing and method in ('ffill', 'mixed'):
            inserted += insert_ffill_rows(sym, missing)
        print(f'  inserted {inserted} rows')
        total_inserted += inserted
    print(f'Total inserted: {total_inserted}')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Detect and repair missing daily prices in DB')
    parser.add_argument('--start', type=str, help='Start date YYYYMMDD')
    parser.add_argument('--end', type=str, help='End date YYYYMMDD')
    parser.add_argument('--years', type=int, help='Range by last N years if start/end not provided')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols to process')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of symbols')
    parser.add_argument('--method', type=str, default='mixed', choices=['refetch','ffill','mixed'], help='Repair method')
    parser.add_argument('--no-calendar', action='store_true', help='Do not use trading calendar (natural dates)')
    parser.add_argument('--no-filter', action='store_true', help='Do not filter ChiNext/STAR/BSE')
    args = parser.parse_args()

    start, end = resolve_date_range(args.start, args.end, args.years)
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
    else:
        df = get_all_stocks(filter_tradable=not args.no_filter)
        syms = df['symbol'].tolist()
    if args.limit:
        syms = syms[:args.limit]
    print(f'Processing {len(syms)} symbols, date range {start or "MIN"}~{end or "MAX"}, method={args.method}')
    detect_and_repair(syms, start, end, method=args.method, use_calendar=not args.no_calendar)

if __name__ == '__main__':
    main()

