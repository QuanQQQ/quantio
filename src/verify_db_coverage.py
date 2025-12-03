"""
Verify DB Coverage: check whether the SQLite daily_prices table contains
near X years of data for all stocks.

Approach:
- Determine target period [start_date, end_date]: end_date = MAX(date) in DB,
  start_date = end_date minus X years (365 * X days).
- Get distinct trading dates in this period from daily_prices.
- For each stock, compute expected trading days count within the period since its listing_date.
- Compare with actual row count in daily_prices for that stock in the period; compute coverage ratio.
- Print summary and top missing stocks.

Notes:
- This check assumes all trading dates appearing in the DB are the canonical trading calendar.
- Delisting is not considered because we do not track a delisting date; coverage may be lower for delisted stocks.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

DB_PATH = 'data/stock_data.db'

def _get_min_max_date(conn) -> Tuple[str, str]:
    cur = conn.cursor()
    cur.execute('SELECT MIN(date), MAX(date) FROM daily_prices')
    mn, mx = cur.fetchone()
    return mn, mx

def _to_dt(s: str) -> datetime:
    return datetime.strptime(s, '%Y%m%d')

def _to_str(dt: datetime) -> str:
    return dt.strftime('%Y%m%d')

def verify_db_coverage(years: int = 20, min_coverage: float = 0.95, filter_tradable: bool = False, top_n: int = 20):
    conn = sqlite3.connect(DB_PATH)
    try:
        mn, mx = _get_min_max_date(conn)
        if not mx:
            print('No data found in daily_prices.')
            return

        end_dt = _to_dt(mx)
        start_dt = end_dt - timedelta(days=365 * years)
        start_str = _to_str(start_dt)
        end_str = _to_str(end_dt)

        print(f'Checking coverage for period: {start_str} - {end_str} (≈{years} years)')

        # Distinct trading dates in period
        dates_df = pd.read_sql(
            'SELECT DISTINCT date FROM daily_prices WHERE date BETWEEN ? AND ? ORDER BY date',
            conn,
            params=(start_str, end_str)
        )
        if dates_df.empty:
            print('No trading dates found in the selected period.')
            return
        trading_dates = dates_df['date'].tolist()

        # Stocks list with listing_date
        stocks_df = pd.read_sql('SELECT symbol, listing_date FROM stocks', conn)
        if filter_tradable:
            def is_tradable(symbol: str) -> bool:
                code = symbol.split('.')[0]
                if code.startswith('300') or code.startswith('688'):
                    return False
                if code.endswith('BJ'):
                    return False
                return True
            stocks_df = stocks_df[stocks_df['symbol'].apply(is_tradable)]

        # Actual counts in period per stock
        counts_df = pd.read_sql(
            'SELECT symbol, COUNT(*) AS rows_count FROM daily_prices WHERE date BETWEEN ? AND ? GROUP BY symbol',
            conn,
            params=(start_str, end_str)
        )

        # Compute expected counts per stock using trading_dates and listing_date
        listing_map = {row['symbol']: (row['listing_date'] or '19000101') for _, row in stocks_df.iterrows()}
        # Precompute index map for fast count
        # Expected count = number of dates >= listing_date within trading_dates
        from bisect import bisect_left
        expected_counts = {}
        for symbol, listing_date in listing_map.items():
            try:
                idx = bisect_left(trading_dates, listing_date)
            except Exception:
                idx = 0
            expected_counts[symbol] = max(0, len(trading_dates) - idx)

        # Merge actual and expected
        merged = pd.merge(stocks_df[['symbol']], counts_df, on='symbol', how='left').fillna({'rows_count': 0})
        merged['expected_count'] = merged['symbol'].map(expected_counts).fillna(0).astype(int)
        merged['coverage'] = merged.apply(lambda r: (r['rows_count'] / r['expected_count']) if r['expected_count'] > 0 else 1.0, axis=1)
        merged['missing_days'] = merged.apply(lambda r: (r['expected_count'] - r['rows_count']), axis=1)

        total_stocks = len(merged)
        ok_stocks = int((merged['coverage'] >= min_coverage).sum())
        print(f'Total stocks checked: {total_stocks}')
        print(f'Stocks with coverage >= {min_coverage*100:.1f}%: {ok_stocks} ({ok_stocks/total_stocks*100:.2f}%)')

        # Top N stocks with missing days
        worst = merged.sort_values(['coverage', 'missing_days'], ascending=[True, False]).head(top_n)
        if not worst.empty:
            print('\nTop stocks with missing days:')
            for _, row in worst.iterrows():
                print(f"  {row['symbol']}: coverage={row['coverage']*100:.2f}% missing={int(row['missing_days'])}")

        # Sanity check on trading date count
        print(f"\nTrading dates in period: {len(trading_dates)}")
        print(f"Date range in DB: min={mn}, max={mx}")

        return merged
    finally:
        conn.close()


def get_targets_with_no_data(conn) -> list:
    """
    Identify stocks that are in the 'stocks' table but have no records in 'daily_prices'.
    """
    # Get all stocks
    stocks_df = pd.read_sql('SELECT symbol, name FROM stocks', conn)
    all_symbols = set(stocks_df['symbol'])
    
    # Get stocks with data
    daily_df = pd.read_sql('SELECT DISTINCT symbol FROM daily_prices', conn)
    existing_symbols = set(daily_df['symbol'])
    
    missing_symbols = list(all_symbols - existing_symbols)
    missing_symbols.sort()
    
    return missing_symbols

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify DB coverage for near X years')
    parser.add_argument('--years', type=int, default=20, help='Years to check, counting back from latest DB date')
    parser.add_argument('--min-coverage', type=float, default=0.95, help='Minimum acceptable coverage ratio')
    parser.add_argument('--top-n', type=int, default=20, help='Show top N worst stocks')
    parser.add_argument('--no-filter', action='store_true', help='Do not filter out ChiNext/STAR/BSE')
    parser.add_argument('--check-empty', action='store_true', help='Check for targets with absolutely no data')
    args = parser.parse_args()

    if args.check_empty:
        conn = sqlite3.connect(DB_PATH)
        try:
            missing = get_targets_with_no_data(conn)
            if missing:
                print(f"Found {len(missing)} targets with NO data:")
                for sym in missing:
                    print(f"  {sym}")
            else:
                print("All targets have at least some data.")
        finally:
            conn.close()
    else:
        verify_db_coverage(years=args.years, min_coverage=args.min_coverage, filter_tradable=not args.no_filter, top_n=args.top_n)

