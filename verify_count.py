import sqlite3
import pandas as pd
from datetime import datetime, timedelta

conn = sqlite3.connect('data/stock_data.db')
cursor = conn.cursor()

end_date = datetime.now()
start_date = end_date - timedelta(days=5)
start_str = start_date.strftime('%Y%m%d')
end_str = end_date.strftime('%Y%m%d')

print(f"Checking data from {start_str} to {end_str}")

df = pd.read_sql(
    'SELECT date, COUNT(*) as count FROM daily_prices WHERE date BETWEEN ? AND ? GROUP BY date ORDER BY date',
    conn,
    params=(start_str, end_str)
)
print(df)
conn.close()
