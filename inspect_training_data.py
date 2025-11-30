import numpy as np
import os

DATA_FILE = "data/training_data_2018_2022_relative_scaling.npz"

def inspect_data():
    if not os.path.exists(DATA_FILE):
        print(f"File {DATA_FILE} not found.")
        return

    print(f"Loading {DATA_FILE}...")
    try:
        data = np.load(DATA_FILE)
        print(f"Keys in file: {list(data.keys())}")
        
        X = data['X']
        y = data['y']
        
        print(f"\nData Shapes:")
        print(f"X (Features): {X.shape}")
        print(f"y (Labels):   {y.shape}")
        
        # Feature names from src/data_processor.py
        feature_names = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
        
        print(f"\n--- Sample Data (First 2 samples) ---")
        for i in range(2):
            print(f"\nSample {i+1}:")
            print(f"Target (Return %): {y[i]:.4f}")
            print("Features (Normalized):")
            print(f"{'Index':<5} {'Feature':<15} {'Value':<10}")
            print("-" * 30)
            # X shape is likely (N, Lookback, Features) or (N, Features). 
            # Based on train.py: INPUT_SIZE = 11, so likely (N, Lookback, 11) or flattened?
            # Let's check dimensions.
            if X.ndim == 3:
                # (Batch, Time, Feat)
                # Print the last time step of the sample
                last_step = X[i, -1, :]
                for f_idx, val in enumerate(last_step):
                    print(f"{f_idx:<5} {feature_names[f_idx]:<15} {val:.4f}")
            elif X.ndim == 2:
                # (Batch, Feat)
                for f_idx, val in enumerate(X[i]):
                    print(f"{f_idx:<5} {feature_names[f_idx]:<15} {val:.4f}")
            else:
                print(f"Unexpected X shape: {X.shape}")

        print(f"\n--- Statistics ---")
        # Calculate stats on a subset to save memory/time if huge
        subset_size = min(len(X), 10000)
        X_sub = X[:subset_size]
        y_sub = y[:subset_size]
        
        print(f"Stats based on first {subset_size} samples:")
        print(f"Label Mean: {np.mean(y_sub):.4f}, Std: {np.std(y_sub):.4f}, Min: {np.min(y_sub):.4f}, Max: {np.max(y_sub):.4f}")
        
        if X.ndim == 3:
            # Flatten for feature stats
            X_flat = X_sub.reshape(-1, X_sub.shape[-1])
            print(f"\nFeature Statistics (Mean / Std):")
            for f_idx, name in enumerate(feature_names):
                mean_val = np.mean(X_flat[:, f_idx])
                std_val = np.std(X_flat[:, f_idx])
                print(f"{name:<15}: {mean_val:>8.4f} / {std_val:>8.4f}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_data()
