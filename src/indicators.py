import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from .database import DB_PATH, get_stock_daily
from .data_processor import calculate_kdj, calculate_macd, calculate_trend_lines

INDICATOR_COLUMNS = [
    ('k', 'REAL'),
    ('d', 'REAL'),
    ('j', 'REAL'),
    ('macd', 'REAL'),
    ('macd_signal', 'REAL'),
    ('macd_hist', 'REAL'),
    ('short_trend', 'REAL'),
    ('long_trend', 'REAL'),
    ('vol_ma5', 'REAL'),
]

def ensure_indicator_columns():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        # Improve concurrency for parallel writes
        try:
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
        except Exception:
            pass
        # Get existing columns
        cur.execute('PRAGMA table_info(daily_prices)')
        cols = {row[1] for row in cur.fetchall()}  # name at index 1
        to_add = [name for (name, typ) in INDICATOR_COLUMNS if name not in cols]
        for name in to_add:
            typ = next(t for (n, t) in INDICATOR_COLUMNS if n == name)
            cur.execute(f'ALTER TABLE daily_prices ADD COLUMN {name} {typ}')
        conn.commit()
    finally:
        conn.close()

def _fmt(dt: datetime) -> str:
    return dt.strftime('%Y%m%d')

def compute_and_update_indicators(symbol: str, start_date: str | None = None, end_date: str | None = None):
    """
    Compute KDJ/MACD/trend lines/vol_ma5 for a symbol within [start_date, end_date] and update DB.
    If start_date is None, computes for full available history of the symbol.
    """
    ensure_indicator_columns()

    df = get_stock_daily(symbol, start_date=start_date, end_date=end_date)
    if df.empty:
        return 0

    # Compute indicators
    df = calculate_kdj(df)
    df = calculate_macd(df)
    df = calculate_trend_lines(df)
    df['vol_ma5'] = df['volume'].rolling(window=5).mean()

    # Prepare update
    updates = []
    for _, row in df.iterrows():
        updates.append((
            float(row.get('k', 0) or 0),
            float(row.get('d', 0) or 0),
            float(row.get('j', 0) or 0),
            float(row.get('macd', 0) or 0),
            float(row.get('macd_signal', 0) or 0),
            float(row.get('macd_hist', 0) or 0),
            float(row.get('short_trend', 0) or 0),
            float(row.get('long_trend', 0) or 0),
            float(row.get('vol_ma5', 0) or 0),
            symbol,
            row['date'],
        ))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        try:
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
        except Exception:
            pass
        cur.executemany('''
            UPDATE daily_prices
            SET k = ?, d = ?, j = ?, macd = ?, macd_signal = ?, macd_hist = ?, short_trend = ?, long_trend = ?, vol_ma5 = ?
            WHERE symbol = ? AND date = ?
        ''', updates)
        conn.commit()
        return conn.total_changes
    finally:
        conn.close()

def compute_indicators_recent(symbol: str, days: int = 400):
    """Compute indicators for recent N days window for a symbol."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    return compute_and_update_indicators(symbol, start_date=_fmt(start_dt), end_date=_fmt(end_dt))
