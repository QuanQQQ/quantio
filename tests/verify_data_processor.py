import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processor import generate_backtest_data
import pandas as pd

def verify_processor():
    print("Generating backtest data for verification...")
    # Limit stocks to 5 to be quick
    X, y, metadata = generate_backtest_data(limit_stocks=5, lookback=10, horizon=5, use_multiprocessing=False)
    
    if not metadata:
        print("No data generated. Check if database has data or if criteria (J<13) is met.")
        return

    print(f"Generated {len(metadata)} samples.")
    
    # Verify first 5 samples
    for i, item in enumerate(metadata[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Symbol: {item['symbol']}")
        print(f"  Date: {item['date']}")
        print(f"  Close: {item['close']}")
        print(f"  Future High: {item.get('future_high', 'N/A')}")
        print(f"  Future Low: {item.get('future_low', 'N/A')}")
        print(f"  Score: {item['score']:.4f}")
        
        # Manual Calculation
        entry = item['close']
        high = item['future_high']
        low = item['future_low']
        
        max_gain = (high - entry) / entry
        max_loss = max(0, (entry - low) / entry)
        expected_score = (max_gain - max_loss) * 100
        
        print(f"  Manual Calc: {expected_score:.4f}")
        
        if abs(expected_score - item['score']) < 1e-6:
            print("  [PASS] Calculation matches.")
        else:
            print("  [FAIL] Calculation mismatch!")

if __name__ == "__main__":
    verify_processor()
