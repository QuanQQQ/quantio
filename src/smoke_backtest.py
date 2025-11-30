import sys
import os
import pandas as pd
from datetime import datetime
from typing import Dict

# Ensure project root in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
libs_path = os.path.join(project_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.backtest_engine import BacktestEngine


class MockBacktestEngine(BacktestEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build a small synthetic daily price table per symbol
        self._daily: Dict[str, pd.DataFrame] = {}

    def set_daily(self, symbol: str, rows: list):
        df = pd.DataFrame(rows)
        self._daily[symbol] = df

    def get_daily_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._daily.get(symbol, pd.DataFrame())
        if df.empty:
            return df
        # Filter by date string YYYYMMDD
        m = df[df['date'] == start_date]
        return m.reset_index(drop=True)

    def is_delisted(self, symbol: str, current_date: str, end_date: str) -> bool:
        return False


def main():
    # Build synthetic predictions for one symbol across 5 days
    predictions = pd.DataFrame([
        {'symbol': 'AAA', 'date': '20230102', 'predicted_return': 10.0, 'close': 100.0},
        {'symbol': 'BBB', 'date': '20230102', 'predicted_return': -1.0, 'close': 50.0},   # should be filtered (<=0)
        {'symbol': 'CCC', 'date': '20230102', 'predicted_return': 8.0,  'close': 20.0},   # fallback candidate
        {'symbol': 'AAA', 'date': '20230103', 'predicted_return': 10.0, 'close': 105.0},
        {'symbol': 'AAA', 'date': '20230104', 'predicted_return': 10.0, 'close': 112.0},
        {'symbol': 'AAA', 'date': '20230105', 'predicted_return': 10.0, 'close': 108.0},
        {'symbol': 'AAA', 'date': '20230106', 'predicted_return': 10.0, 'close': 102.0},
    ])

    # Trend lines: short_trend slightly below price until 20230105 where price dips under short_trend;
    # long_trend placed above final day to trigger stop-loss on 20230106
    daily_rows_AAA = [
        {'date': '20230102', 'close': 100.0, 'short_trend': 99.0,  'long_trend': 95.0},
        {'date': '20230103', 'close': 105.0, 'short_trend': 100.0, 'long_trend': 96.0},
        {'date': '20230104', 'close': 112.0, 'short_trend': 105.0, 'long_trend': 97.0},
        {'date': '20230105', 'close': 108.0, 'short_trend': 109.0, 'long_trend': 98.0},  # below short_trend → partial take-profit
        {'date': '20230106', 'close': 102.0, 'short_trend': 104.0, 'long_trend': 103.0},  # below long_trend → stop-loss
    ]
    daily_rows_CCC = [
        {'date': '20230102', 'close': 20.0, 'open': 20.0, 'short_trend': 19.0, 'long_trend': 19.5},  # signal day
        {'date': '20230103', 'close': 20.0, 'open': 20.0, 'short_trend': 19.0, 'long_trend': 19.5},  # buy day
    ]

    engine_v2 = MockBacktestEngine(initial_capital=100000, max_positions=5, stop_loss_pct=-5.0, take_profit_buffer=5.0, variant='v2')
    engine_v2.set_daily('AAA', daily_rows_AAA)
    engine_v2.set_daily('CCC', daily_rows_CCC)
    trades_v2 = engine_v2.run_backtest(predictions, start_date='20230102', end_date='20230106', horizon=10)
    print("\n[SMOKE TEST v2] Trades:\n", trades_v2)
    print("\n[SMOKE TEST v2] Operations:\n", engine_v2.get_operations_log())
    print("\n[SMOKE TEST v2] Equity Curve:\n", engine_v2.get_equity_curve())

    engine_v1 = MockBacktestEngine(initial_capital=100000, max_positions=5, stop_loss_pct=-5.0, take_profit_buffer=5.0, variant='v1')
    engine_v1.set_daily('AAA', daily_rows_AAA)
    trades_v1 = engine_v1.run_backtest(predictions, start_date='20230102', end_date='20230106', horizon=10)
    print("\n[SMOKE TEST v1] Trades:\n", trades_v1)
    print("\n[SMOKE TEST v1] Operations:\n", engine_v1.get_operations_log())
    print("\n[SMOKE TEST v1] Equity Curve:\n", engine_v1.get_equity_curve())


if __name__ == '__main__':
    main()
