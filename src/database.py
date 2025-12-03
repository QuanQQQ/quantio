import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import List, Optional

DB_PATH = os.path.join("data", "stock_data.db")

def init_db():
    """Initialize the SQLite database with necessary tables."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for stock basic info
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            listing_date TEXT
        )
    ''')
    
    # Table for daily price data
    # Using composite primary key (symbol, date) to prevent duplicates
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            PRIMARY KEY (symbol, date)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_basic (
            symbol TEXT,
            date TEXT,
            total_mv REAL,
            circ_mv REAL,
            turnover_rate REAL,
            turnover_rate_f REAL,
            PRIMARY KEY (symbol, date)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_stocks(df):
    """
    Save stock list to database.
    df columns expected: symbol, name, ...
    """
    conn = sqlite3.connect(DB_PATH)
    # We only keep essential columns for now, or whatever is passed
    # Assuming df has 'symbol', 'name' at least.
    # We'll use 'replace' or 'append'. For stock list, 'replace' might be safer to handle name changes, 
    # but we need to be careful not to lose other info if we had it. 
    # For simplicity, let's upsert or just replace the list if it's a full refresh.
    # Here we assume a full refresh of the stock list.
    
    # Ensure columns match what we want to store if possible, or just store what we get if it fits.
    # For this simple version, let's just dump the dataframe to a temp table and merge or just replace if structure matches.
    # Actually, pandas to_sql 'replace' drops the table. We defined a schema.
    # Let's use 'append' but we need to handle duplicates.
    # Or just use SQLite upsert syntax manually if we want to be robust.
    # For simplicity in this MVP:
    
    # Clean up existing stocks table and replace (assuming we fetch full list)
    # But we defined a specific schema. Let's stick to it.
    
    # Let's assume df has columns mapping to our schema.
    # If df has more columns, we might filter them.
    
    # A safer approach for "save_stocks" which might be called with new stocks:
    # Use to_sql with if_exists='replace' is easiest but drops the table schema.
    # Let's iterate and execute INSERT OR REPLACE.
    
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute('''
            INSERT OR REPLACE INTO stocks (symbol, name, sector, listing_date)
            VALUES (?, ?, ?, ?)
        ''', (
            row.get('symbol', ''),
            row.get('name', ''),
            row.get('sector', ''),
            row.get('listing_date', '')
        ))
    
    conn.commit()
    conn.close()

def save_daily_data(df):
    """
    Save daily price data.
    df columns expected: symbol, date, open, high, low, close, volume, amount
    """
    if df.empty:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Batch insert is faster
    data_to_insert = []
    for _, row in df.iterrows():
        data_to_insert.append((
            row['symbol'],
            row['date'],
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume'],
            row['amount']
        ))
        
    cursor.executemany('''
        INSERT OR IGNORE INTO daily_prices (symbol, date, open, high, low, close, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', data_to_insert)
    
    # Report how many rows were inserted in this call
    inserted_count = conn.total_changes  # changes since connection opened
    conn.commit()
    conn.close()
    return inserted_count

def save_daily_basic(df):
    if df.empty:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    data_to_insert = []
    for _, row in df.iterrows():
        data_to_insert.append((
            row['symbol'],
            row['date'],
            row.get('total_mv'),
            row.get('circ_mv'),
            row.get('turnover_rate'),
            row.get('turnover_rate_f')
        ))
    cursor.executemany('''
        INSERT OR IGNORE INTO daily_basic (symbol, date, total_mv, circ_mv, turnover_rate, turnover_rate_f)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', data_to_insert)
    inserted_count = conn.total_changes
    conn.commit()
    conn.close()
    return inserted_count

def get_last_date(symbol):
    """Get the last recorded date for a stock."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(date) FROM daily_prices WHERE symbol = ?', (symbol,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_all_stocks(filter_tradable=True):
    """
    Get all stocks info.
    
    Args:
        filter_tradable (bool): If True, filter out stocks that require special trading permissions
                                (ChiNext 300*, STAR Market 688*, and Beijing Stock Exchange 8*/4*)
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql('SELECT * FROM stocks', conn)
    conn.close()
    
    if filter_tradable and not df.empty:
        # Filter out stocks that require special trading permissions:
        # - ChiNext (创业板): starts with 300
        # - STAR Market (科创板): starts with 688
        # - Beijing Stock Exchange (北交所): starts with 8 or 4
        def is_tradable(symbol):
            if pd.isna(symbol):
                return False
            symbol = str(symbol)
            # Extract numeric part (remove exchange suffix like .SZ, .SH)
            parts = symbol.split('.')
            code = parts[0]
            suffix = parts[1].upper() if len(parts) > 1 else ''
            # Filter out: 300xxx, 688xxx, 8xxxxx, 4xxxxx
            # ChiNext: 300xxx 和 301xxx
            if code.startswith('300') or code.startswith('301'):
                return False
            # STAR Market: 688xxx
            if code.startswith('688'):
                return False
            # Beijing Stock Exchange: 后缀 .BJ 更稳妥
            if suffix == 'BJ':
                return False
            if code.startswith('8') or code.startswith('4'):
                # Beijing Stock Exchange codes are typically 6 digits starting with 8 or 4
                # But we need to be careful not to filter out 60xxxx (Shanghai A-shares)
                # BSE codes: 82xxxx, 83xxxx, 87xxxx, 43xxxx
                if len(code) == 6 and code[0] in ['8', '4']:
                    return False
            return True
        
        original_count = len(df)
        df = df[df['symbol'].apply(is_tradable)]
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} stocks (ChiNext/STAR/BSE), {len(df)} stocks remaining.")
    
    return df

def get_stock_daily(symbol, start_date=None, end_date=None):
    """Get daily data for a stock, joined with basic fields (market cap, turnover)."""
    conn = sqlite3.connect(DB_PATH)
    query = (
        'SELECT p.*, b.total_mv, b.circ_mv, b.turnover_rate, b.turnover_rate_f '
        'FROM daily_prices AS p '
        'LEFT JOIN daily_basic AS b ON p.symbol = b.symbol AND p.date = b.date '
        'WHERE p.symbol = ?'
    )
    params = [symbol]
    if start_date:
        query += ' AND p.date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND p.date <= ?'
        params.append(end_date)
    query += ' ORDER BY p.date'
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

def get_stock_daily_last_n(symbol: str, end_date: str, n: int) -> pd.DataFrame:
    """Get last N trading rows up to (and including) end_date for a stock, joined with basic fields.
    Returns rows sorted by date ascending.
    """
    conn = sqlite3.connect(DB_PATH)
    query = (
        'SELECT p.*, b.total_mv, b.circ_mv, b.turnover_rate, b.turnover_rate_f '
        'FROM daily_prices AS p '
        'LEFT JOIN daily_basic AS b ON p.symbol = b.symbol AND p.date = b.date '
        'WHERE p.symbol = ? AND p.date <= ? '
        'ORDER BY p.date DESC LIMIT ?'
    )
    df_desc = pd.read_sql(query, conn, params=[symbol, end_date, int(n)])
    conn.close()
    if df_desc.empty:
        return df_desc
    # Return ascending order
    return df_desc.sort_values('date').reset_index(drop=True)

def get_stock_daily_multi(symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Batch fetch daily prices for multiple symbols, joined with basic fields."""
    if not symbols:
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    placeholders = ','.join(['?'] * len(symbols))
    query = (
        f'SELECT p.*, b.total_mv, b.circ_mv, b.turnover_rate, b.turnover_rate_f '
        f'FROM daily_prices AS p '
        f'LEFT JOIN daily_basic AS b ON p.symbol = b.symbol AND p.date = b.date '
        f'WHERE p.symbol IN ({placeholders})'
    )
    params = list(symbols)
    if start_date:
        query += ' AND p.date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND p.date <= ?'
        params.append(end_date)
    query += ' ORDER BY p.symbol, p.date'
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

def delete_daily_data_range(start_date: str, end_date: str):
    """
    Delete daily price data within the specified date range [start_date, end_date].
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM daily_prices WHERE date >= ? AND date <= ?", (start_date, end_date))
        deleted_count = cursor.rowcount
        conn.commit()
        print(f"Deleted {deleted_count} rows from daily_prices between {start_date} and {end_date}.")
        return deleted_count
    except Exception as e:
        print(f"Error deleting data: {e}")
        return 0
    finally:
        conn.close()
