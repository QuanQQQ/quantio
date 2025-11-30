"""
Validate precomputed indicators in DB by recomputing from raw OHLCV
and comparing per-symbol, per-date values.

Usage examples:
  # Validate last 1 year for 50 symbols
  python src/validate_indicators.py --years 1 --limit 50

  # Validate explicit date range for specific symbols
  python src/validate_indicators.py --start 20230101 --end 20231231 --symbols 000001.SZ,600000.SH
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import get_all_stocks, get_stock_daily
from src.data_processor import calculate_kdj, calculate_macd, calculate_trend_lines

TARGET_COLS = ['k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend', 'vol_ma5']

def fmt(dt: datetime) -> str:
    return dt.strftime('%Y%m%d')

def resolve_date_range(start: str | None, end: str | None, years: int | None):
    if start and end:
        return start, end
    if years and years > 0:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365 * years)
        return fmt(start_dt), fmt(end_dt)
    return None, None

def compute_ref(df: pd.DataFrame) -> pd.DataFrame:
    # Recompute indicators from raw OHLCV
    if df.empty:
        return df
    df = df.copy()
    df = calculate_kdj(df)
    df = calculate_macd(df)
    df = calculate_trend_lines(df)
    df['vol_ma5'] = df['volume'].rolling(window=5).mean()
    return df

def compare_cols(db_df: pd.DataFrame, ref_df: pd.DataFrame, cols: list[str], atol: float = 1e-6, rtol: float = 1e-6):
    stats = {}
    for c in cols:
        if c not in db_df.columns or c not in ref_df.columns:
            stats[c] = {'rows': 0, 'matches': 0, 'mae': np.nan, 'max': np.nan}
            continue
        a = db_df[c].values.astype(float)
        b = ref_df[c].values.astype(float)
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            stats[c] = {'rows': 0, 'matches': 0, 'mae': np.nan, 'max': np.nan}
            continue
        a = a[mask]; b = b[mask]
        diff = np.abs(a - b)
        tol = atol + rtol * np.maximum(np.abs(b), 1.0)
        matches = int(np.sum(diff <= tol))
        rows = int(len(diff))
        mae = float(np.mean(diff))
        mx = float(np.max(diff))
        stats[c] = {'rows': rows, 'matches': matches, 'mae': mae, 'max': mx}
    return stats

def main():
    parser = argparse.ArgumentParser(description='Validate indicators in DB')
    parser.add_argument('--start', type=str, help='Start date YYYYMMDD')
    parser.add_argument('--end', type=str, help='End date YYYYMMDD')
    parser.add_argument('--years', type=int, help='Validate last N years (ignored if start/end provided)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols to check')
    parser.add_argument('--limit', type=int, default=50, help='Limit number of symbols (default: 50)')
    parser.add_argument('--rtol', type=float, default=1e-6, help='Relative tolerance for comparison')
    parser.add_argument('--atol', type=float, default=1e-6, help='Absolute tolerance for comparison')
    args = parser.parse_args()

    start, end = resolve_date_range(args.start, args.end, args.years)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    else:
        df_stocks = get_all_stocks()
        symbols = df_stocks['symbol'].tolist()[:args.limit]

    print(f'Validating {len(symbols)} symbols, date range: {start or "FULL"} ~ {end or "FULL"}')

    total_rows = 0
    total_matches = {c: 0 for c in TARGET_COLS}
    total_counts = {c: 0 for c in TARGET_COLS}

    for sym in symbols:
        db_df = get_stock_daily(sym, start, end)
        if db_df.empty:
            print(f'  {sym}: no data')
            continue
        # Ensure necessary columns exist
        missing_cols = [c for c in TARGET_COLS if c not in db_df.columns]
        if missing_cols:
            print(f'  {sym}: missing indicator columns: {", ".join(missing_cols)}')
            continue
        # Recompute reference on same OHLCV
        ref_df = compute_ref(db_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy())
        # Align index
        db_df = db_df.reset_index(drop=True)
        ref_df = ref_df.reset_index(drop=True)
        stats = compare_cols(db_df, ref_df, TARGET_COLS, atol=args.atol, rtol=args.rtol)
        rows_sym = int(len(db_df))
        total_rows += rows_sym
        # Per-symbol summary
        match_summary = ', '.join([f"{c}:{stats[c]['matches']}/{stats[c]['rows']} mae={stats[c]['mae']:.2e} max={stats[c]['max']:.2e}" for c in TARGET_COLS])
        print(f'  {sym}: rows={rows_sym} | {match_summary}')
        # Aggregate
        for c in TARGET_COLS:
            total_matches[c] += stats[c]['matches']
            total_counts[c] += stats[c]['rows']

    print('\nOverall:')
    for c in TARGET_COLS:
        rows = total_counts[c]
        matches = total_matches[c]
        ratio = (matches / rows * 100) if rows > 0 else 0.0
        print(f'  {c}: {matches}/{rows} ({ratio:.2f}%) within tolerance')

if __name__ == '__main__':
    main()

