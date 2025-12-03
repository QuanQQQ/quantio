import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys

# Ensure libs path is added if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
libs_path = os.path.join(project_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

def visualize_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Loading data from {file_path} with mmap_mode='r'...")
    try:
        # Use mmap_mode to avoid loading everything into RAM
        data = np.load(file_path, mmap_mode='r')
        X = data['X']
        y = data['y']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data Shape: X={X.shape}, y={y.shape}")
    
    # Sample data for statistics and plotting to avoid MemoryError
    # Sample size: 100,000 or 10% whichever is smaller, but at least 10000
    total_samples = len(X)
    sample_size = min(100000, total_samples)
    if total_samples > sample_size:
        print(f"Dataset too large for full in-memory analysis. Sampling {sample_size} random samples...")
        indices = np.random.choice(total_samples, sample_size, replace=False)
        # Sort indices for better access pattern if possible, though random access on HDD might be slow
        indices.sort()
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y

    # Basic Statistics (on sample)
    print("\nTarget (y) Statistics (Sampled):")
    print(f"  Mean: {np.mean(y_sample):.4f}")
    print(f"  Std:  {np.std(y_sample):.4f}")
    print(f"  Min:  {np.min(y_sample):.4f}")
    print(f"  Max:  {np.max(y_sample):.4f}")
    
    print("\nFeature (X) Statistics (Global - Sampled):")
    print(f"  Mean: {np.mean(X_sample):.4f}")
    print(f"  Std:  {np.std(X_sample):.4f}")
    print(f"  Min:  {np.min(X_sample):.4f}")
    print(f"  Max:  {np.max(X_sample):.4f}")

    # Check for NaNs/Infs (on sample)
    nan_count = np.isnan(X_sample).sum()
    inf_count = np.isinf(X_sample).sum()
    print(f"\nNaN count in X (Sampled): {nan_count}")
    print(f"Inf count in X (Sampled): {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("WARNING: Data contains NaNs or Infs!")

    # Create plots directory
    plots_dir = os.path.join(project_root, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # 1. Target Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y_sample, kde=True, bins=50)
    plt.title('Target Distribution (y) - Sampled')
    plt.xlabel('Return (%)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, 'target_distribution.png'))
    plt.close()
    print(f"Saved target_distribution.png to {plots_dir}")

    # 2. Feature Distributions (Boxplot)
    # X shape: (N, Lookback, Features)
    num_features = X.shape[2]
    feature_names = ['open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j', 'macd', 'macd_signal', 'macd_hist', 'short_trend', 'long_trend']
    
    # Flatten samples and time steps
    X_flat = X_sample.reshape(-1, num_features)
    
    plt.figure(figsize=(15, 8))
    plt.boxplot(X_flat, labels=feature_names[:num_features])
    plt.title('Feature Distributions (Normalized) - Sampled')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_boxplots.png'))
    plt.close()
    print(f"Saved feature_boxplots.png to {plots_dir}")

    # 3. Feature Histograms (Individual)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot first 9 features
    for i in range(min(9, num_features)):
        sns.histplot(X_flat[:, i], kde=True, ax=axes[i], bins=30)
        axes[i].set_title(f'{feature_names[i]}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_histograms.png'))
    plt.close()
    print(f"Saved feature_histograms.png to {plots_dir}")
    
    # 4. Sample Time Series
    # Plot a random sample from the original data (no need to be in the sample set)
    sample_idx = np.random.randint(0, len(X))
    sample = X[sample_idx] # (Lookback, Features)
    
    plt.figure(figsize=(12, 6))
    # Plot Close price (normalized)
    plt.plot(sample[:, 3], label='Close', marker='o')
    # Plot Short Trend
    if num_features > 11:
        plt.plot(sample[:, 11], label='Short Trend', linestyle='--')
    # Plot Long Trend
    if num_features > 12:
        plt.plot(sample[:, 12], label='Long Trend', linestyle=':')
        
    plt.title(f'Sample {sample_idx} Time Series (Normalized)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'sample_timeseries.png'))
    plt.close()
    print(f"Saved sample_timeseries.png to {plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training data")
    parser.add_argument("--file", type=str, required=True, help="Path to .npz data file")
    args = parser.parse_args()
    
    visualize_data(args.file)
