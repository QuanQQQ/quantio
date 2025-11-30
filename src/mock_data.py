import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .database import init_db, save_stocks, save_daily_data

def generate_mock_data():
    print("Generating mock data...")
    init_db()
    
    # 1. Mock Stocks
    stocks_data = {
        'symbol': ['600000', '600519', '000001', '000858', '601318'],
        'name': ['浦发银行', '贵州茅台', '平安银行', '五粮液', '中国平安'],
        'sector': ['Banking', 'Beverage', 'Banking', 'Beverage', 'Insurance'],
        'listing_date': ['1999-11-10', '2001-08-27', '1991-04-03', '1998-04-27', '2007-03-01']
    }
    stocks_df = pd.DataFrame(stocks_data)
    save_stocks(stocks_df)
    
    # 2. Mock Daily Data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    dates = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
    
    for symbol in stocks_df['symbol']:
        # Generate random price walk
        base_price = np.random.uniform(10, 1000)
        prices = [base_price]
        for _ in range(len(dates)-1):
            change = np.random.uniform(-0.05, 0.05)
            prices.append(prices[-1] * (1 + change))
            
        data = {
            'symbol': symbol,
            'date': dates.strftime('%Y%m%d'),
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'close': prices,
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'volume': np.random.randint(10000, 1000000, size=len(dates)),
            'amount': np.random.uniform(1000000, 100000000, size=len(dates))
        }
        df = pd.DataFrame(data)
        save_daily_data(df)
        
    print("Mock data generated.")

if __name__ == "__main__":
    generate_mock_data()
