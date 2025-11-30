"""
Dataset management utility.

List and inspect generated datasets.

Usage:
    # List all datasets
    python src/list_datasets.py
    
    # Show detailed information for a specific dataset
    python src/list_datasets.py --detail training_data_lb20_h3_2010_2022.npz
    
    # Clean up datasets
    python src/list_datasets.py --clean
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import DataConfig


def list_datasets(data_dir="data", show_details=False):
    """
    List all datasets in the data directory.
    
    Args:
        data_dir: Directory containing datasets
        show_details: Whether to show detailed statistics
    """
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return []
    
    # Find all .npz files
    npz_files = sorted(Path(data_dir).glob("training_data_*.npz"))
    
    if not npz_files:
        print("No datasets found.")
        return []
    
    print("=" * 90)
    print(f"AVAILABLE DATASETS ({len(npz_files)} found)")
    print("=" * 90)
    
    datasets = []
    for npz_file in npz_files:
        # Load data
        data = np.load(npz_file)
        X, y = data['X'], data['y']
        
        # Try to load config
        config = DataConfig.from_data_file(npz_file, directory=data_dir)
        
        # File size
        file_size = npz_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        datasets.append({
            'filename': npz_file.name,
            'filepath': str(npz_file),
            'config': config,
            'samples': len(X),
            'shape': X.shape,
            'size_mb': size_mb
        })
    
    # Print table header
    print(f"{'Filename':<50} {'Samples':>10} {'Size':>10}")
    print("-" * 90)
    
    for ds in datasets:
        print(f"{ds['filename']:<50} {ds['samples']:>10} {ds['size_mb']:>9.1f}M")
        
        if show_details and ds['config']:
            print(f"  Config: lb={ds['config'].lookback}, h={ds['config'].horizon}, "
                  f"dates={ds['config'].start_date}-{ds['config'].end_date}")
    
    print("=" * 90)
    
    return datasets


def show_dataset_details(filename, data_dir="data"):
    """
    Show detailed statistics for a specific dataset.
    
    Args:
        filename: Dataset filename
        data_dir: Directory containing datasets
    """
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Dataset not found: {filepath}")
        return
    
    # Load data
    data = np.load(filepath)
    X, y = data['X'], data['y']
    
    # Load config
    config = DataConfig.from_data_file(filepath, directory=data_dir)
    
    print("=" * 70)
    print("DATASET DETAILS")
    print("=" * 70)
    print(f"File: {filename}")
    print(f"Path: {filepath}")
    
    file_size = Path(filepath).stat().st_size
    print(f"Size: {file_size / (1024 * 1024):.2f} MB")
    
    print("\n" + "-" * 70)
    print("CONFIGURATION")
    print("-" * 70)
    if config:
        print(config)
    else:
        print("No configuration file found.")
    
    print("\n" + "-" * 70)
    print("DATA STATISTICS")
    print("-" * 70)
    print(f"Samples: {len(X)}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    print(f"\nTarget (y) statistics:")
    print(f"  Mean: {np.mean(y):.4f}%")
    print(f"  Std: {np.std(y):.4f}%")
    print(f"  Min: {np.min(y):.4f}%")
    print(f"  Max: {np.max(y):.4f}%")
    print(f"  Median: {np.median(y):.4f}%")
    
    # Quartiles
    q25, q75 = np.percentile(y, [25, 75])
    print(f"  25th percentile: {q25:.4f}%")
    print(f"  75th percentile: {q75:.4f}%")
    
    # Distribution
    positive = np.sum(y > 0)
    negative = np.sum(y < 0)
    neutral = len(y) - positive - negative
    
    print(f"\nReturn distribution:")
    print(f"  Positive (y > 0): {positive:>6} ({positive/len(y)*100:>5.1f}%)")
    print(f"  Negative (y < 0): {negative:>6} ({negative/len(y)*100:>5.1f}%)")
    print(f"  Neutral  (y = 0): {neutral:>6} ({neutral/len(y)*100:>5.1f}%)")
    
    # More detailed distribution
    print(f"\nDetailed distribution:")
    ranges = [
        (-float('inf'), -10, "< -10%"),
        (-10, -5, "-10% to -5%"),
        (-5, 0, "-5% to 0%"),
        (0, 5, "0% to 5%"),
        (5, 10, "5% to 10%"),
        (10, float('inf'), "> 10%")
    ]
    
    for low, high, label in ranges:
        count = np.sum((y >= low) & (y < high))
        print(f"  {label:>15}: {count:>6} ({count/len(y)*100:>5.1f}%)")
    
    print("=" * 70)


def clean_datasets(data_dir="data", dry_run=True):
    """
    Interactive cleanup of datasets.
    
    Args:
        data_dir: Directory containing datasets
        dry_run: If True, only show what would be deleted
    """
    datasets = list_datasets(data_dir)
    
    if not datasets:
        return
    
    print("\nCleanup options:")
    print("1. Delete all datasets")
    print("2. Keep only the largest dataset")
    print("3. Keep only the most recent dataset")
    print("4. Cancel")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    to_delete = []
    
    if choice == "1":
        to_delete = [ds['filepath'] for ds in datasets]
    elif choice == "2":
        # Keep largest
        largest = max(datasets, key=lambda x: x['samples'])
        to_delete = [ds['filepath'] for ds in datasets if ds['filepath'] != largest['filepath']]
    elif choice == "3":
        # Keep most recent (by filename, which includes date)
        most_recent = max(datasets, key=lambda x: x['filename'])
        to_delete = [ds['filepath'] for ds in datasets if ds['filepath'] != most_recent['filepath']]
    else:
        print("Cancelled.")
        return
    
    if not to_delete:
        print("No files to delete.")
        return
    
    print(f"\nFiles to delete ({len(to_delete)}):")
    for filepath in to_delete:
        print(f"  - {os.path.basename(filepath)}")
    
    if dry_run:
        print("\n(Dry run - no files actually deleted)")
        print("Run with --no-dry-run to actually delete files.")
    else:
        confirm = input("\nAre you sure? (yes/no): ").strip().lower()
        if confirm == "yes":
            for filepath in to_delete:
                os.remove(filepath)
                print(f"Deleted: {os.path.basename(filepath)}")
                
                # Also delete config if exists
                config_file = filepath.replace("training_data_", "config_").replace(".npz", ".json")
                if os.path.exists(config_file):
                    os.remove(config_file)
                    print(f"Deleted: {os.path.basename(config_file)}")
            
            print("\nCleanup completed.")
        else:
            print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset management utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all datasets
  python src/list_datasets.py
  
  # Show detailed information
  python src/list_datasets.py --detail training_data_lb20_h3_2010_2022.npz
  
  # Clean up datasets (dry run)
  python src/list_datasets.py --clean
  
  # Actually delete files
  python src/list_datasets.py --clean --no-dry-run
        """
    )
    
    parser.add_argument("--detail", type=str, help="Show detailed statistics for a specific dataset")
    parser.add_argument("--clean", action="store_true", help="Clean up old datasets")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually delete files (use with --clean)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--show-config", action="store_true", help="Show configuration for each dataset")
    
    args = parser.parse_args()
    
    if args.detail:
        show_dataset_details(args.detail, data_dir=args.data_dir)
    elif args.clean:
        clean_datasets(data_dir=args.data_dir, dry_run=not args.no_dry_run)
    else:
        list_datasets(data_dir=args.data_dir, show_details=args.show_config)


if __name__ == "__main__":
    main()
