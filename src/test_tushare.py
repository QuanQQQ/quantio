import tushare as ts
import pandas as pd

TOKEN = '72e098f1a916bb0ecc08ba3165108f3116bf00c3b493a405d00f6940'
ts.set_token(TOKEN)
pro = ts.pro_api()

print("Testing pro.daily(start_date='20231101', end_date='20231102') without ts_code...")
try:
    df = pro.daily(start_date='20231101', end_date='20231102')
    print(f"Result shape: {df.shape}")
    print(df.head())
    if not df.empty:
        print("Unique dates:", df['trade_date'].unique())
        print("Unique stocks count:", df['ts_code'].nunique())
except Exception as e:
    print(f"Error: {e}")
