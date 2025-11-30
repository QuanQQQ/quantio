import sys
import os

# Add local libs directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
libs_path = os.path.join(project_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import get_stock_daily
import pandas as pd

# Check data range for a sample stock
symbol = '000001.SZ'
df = get_stock_daily(symbol, '20230101', '20231231')

if not df.empty:
    print(f"Stock: {symbol}")
    print(f"Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total trading days: {len(df)}")
    print(f"\nFirst 5 records:")
    print(df.head())
    print(f"\nLast 5 records:")
    print(df.tail())
    
    # Check J < 13 condition
    from src.data_processor import calculate_kdj
    df = calculate_kdj(df)
    trigger_days = df[df['j'] < 13]
    print(f"\nDays with J < 13: {len(trigger_days)}")
    if not trigger_days.empty:
        print(f"First trigger date: {trigger_days['date'].min()}")
        print(f"\nSample trigger dates:")
        print(trigger_days[['date', 'j']].head(10))
else:
    print(f"No data found for {symbol} in 2023")
