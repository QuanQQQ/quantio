import pandas as pd
import numpy as np

def calculate_score(current_close, future_window_high, future_window_low):
    """
    Calculate the risk-adjusted score.
    Score = (Max_Gain_Pct - Max_Drawdown_Pct) * 100
    """
    # Max Gain: Percentage increase from entry to highest point
    max_gain_pct = (future_window_high - current_close) / current_close
    
    # Max Drawdown: Percentage decrease from entry to lowest point
    # Note: We treat this as a positive magnitude of loss for the penalty
    max_loss_pct = (current_close - future_window_low) / current_close
    
    # If max_loss_pct is negative (meaning the low is higher than entry), 
    # it means there was no drawdown from entry, effectively 0 risk relative to entry.
    # However, mathematically (entry - low) would be negative if low > entry.
    # We should clamp max_loss_pct to 0 if low > entry (gap up and never looked back).
    max_loss_pct = max(0, max_loss_pct)
    
    # Similarly for gain, if high < entry (gap down and never recovered), gain is 0?
    # Or should we allow negative gain?
    # If high < entry, max_gain_pct is negative.
    # Let's keep it raw: if the best you could do is -2%, that's your "gain".
    
    score = (max_gain_pct - max_loss_pct) * 100
    return score

def test_logic():
    tests = [
        # Case 1: Rise 10%, Drop 5% (from entry)
        {'entry': 100, 'high': 110, 'low': 95, 'expected': 10 - 5},
        
        # Case 2: Rise 2%, Drop 10%
        {'entry': 100, 'high': 102, 'low': 90, 'expected': 2 - 10},
        
        # Case 3: Gap up, never looked back (Low > Entry)
        # Entry 100, Low 101, High 110. Gain 10%, Loss 0% (clamped)
        {'entry': 100, 'high': 110, 'low': 101, 'expected': 10 - 0},
        
        # Case 4: Gap down, never recovered (High < Entry)
        # Entry 100, High 99, Low 90. Gain -1%, Loss 10%. Score -11.
        {'entry': 100, 'high': 99, 'low': 90, 'expected': -1 - 10},
        
        # Case 5: Flat
        {'entry': 100, 'high': 100, 'low': 100, 'expected': 0},
    ]
    
    print("Running Tests...")
    for i, t in enumerate(tests):
        score = calculate_score(t['entry'], t['high'], t['low'])
        print(f"Test {i+1}: Entry={t['entry']}, High={t['high']}, Low={t['low']}")
        print(f"  Expected: {t['expected']}, Got: {score:.2f}")
        assert abs(score - t['expected']) < 1e-6, f"Test {i+1} Failed"
        
    print("All tests passed!")

if __name__ == "__main__":
    test_logic()
