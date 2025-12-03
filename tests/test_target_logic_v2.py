import pandas as pd
import numpy as np

def calculate_score_time_dependent(entry, highs, lows):
    """
    Calculate score: Max Gain - Max Drawdown (only before peak).
    highs: list/array of high prices in the horizon window
    lows: list/array of low prices in the horizon window
    """
    if len(highs) == 0:
        return 0.0
        
    # Find the index of the absolute maximum high
    # Note: If there are multiple peaks with same value, taking the first one is safer (conservative),
    # or last one (optimistic). Let's take the first one as "Target Reached".
    peak_idx = np.argmax(highs)
    peak_price = highs[peak_idx]
    
    # Max Gain
    max_gain_pct = (peak_price - entry) / entry
    
    # Drawdown is looked for in the window [0, peak_idx] (inclusive)
    # We want the lowest low in this period
    pre_peak_lows = lows[:peak_idx+1]
    min_low_pre_peak = np.min(pre_peak_lows)
    
    # Max Loss Pct (Drawdown from entry)
    max_loss_pct = (entry - min_low_pre_peak) / entry
    max_loss_pct = max(0, max_loss_pct)
    
    score = (max_gain_pct - max_loss_pct) * 100
    return score

def test_logic():
    tests = [
        # Case 1: Rise then Fall (Drawdown after peak ignored)
        # Entry 100. Day 1: H 110, L 105. Day 2: H 100, L 90.
        # Peak is 110 at Day 1 (idx 0). Pre-peak low is 105.
        # Gain: (110-100)/100 = 10%. Loss: (100-105)/100 = -5% (clamped to 0).
        # Score: 10 - 0 = 10.
        {'entry': 100, 'highs': [110, 100], 'lows': [105, 90], 'expected': 10.0},
        
        # Case 2: Fall then Rise (Drawdown before peak counts)
        # Entry 100. Day 1: H 95, L 90. Day 2: H 110, L 100.
        # Peak is 110 at Day 2 (idx 1). Pre-peak lows: [90, 100]. Min is 90.
        # Gain: 10%. Loss: (100-90)/100 = 10%.
        # Score: 10 - 10 = 0.
        {'entry': 100, 'highs': [95, 110], 'lows': [90, 100], 'expected': 0.0},
        
        # Case 3: Gap Up and Go (No drawdown)
        # Entry 100. Day 1: H 105, L 101.
        # Peak 105. Min Low 101. Loss 0. Score 5.
        {'entry': 100, 'highs': [105], 'lows': [101], 'expected': 5.0},
        
        # Case 4: Gap Down and Die (Peak is entry or lower)
        # Entry 100. Day 1: H 95, L 90.
        # Peak 95. Min Low 90.
        # Gain: -5%. Loss: 10%. Score -15.
        {'entry': 100, 'highs': [95], 'lows': [90], 'expected': -5 - 10},
    ]
    
    print("Running Time-Dependent Tests...")
    for i, t in enumerate(tests):
        score = calculate_score_time_dependent(t['entry'], np.array(t['highs']), np.array(t['lows']))
        print(f"Test {i+1}: Entry={t['entry']}, Highs={t['highs']}, Lows={t['lows']}")
        print(f"  Expected: {t['expected']}, Got: {score:.2f}")
        assert abs(score - t['expected']) < 1e-6, f"Test {i+1} Failed"
        
    print("All tests passed!")

if __name__ == "__main__":
    test_logic()
