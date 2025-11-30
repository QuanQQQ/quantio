"""
Standalone script for generating training datasets with flexible parameters.

This script allows you to generate datasets with different lookback and horizon
configurations without modifying the training code.

Usage:
    # Generate with specific parameters
    python src/generate_dataset.py --lookback 20 --horizon 3
    
    # Use a preset configuration
    python src/generate_dataset.py --preset medium
    
    # Generate multiple configurations for comparison
    python src/generate_dataset.py --batch
    
    # Limit stocks for testing
    python src/generate_dataset.py --lookback 10 --horizon 3 --limit 50
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import DataConfig, get_preset, PRESETS, list_presets
from data_processor import generate_training_data


def generate_dataset(config: DataConfig, limit_stocks=None, force=False):
    """
    Generate a dataset with the given configuration.
    
    Args:
        config: DataConfig instance
        limit_stocks: Limit number of stocks for testing
        force: Force regeneration even if file exists
    
    Returns:
        Path to the generated data file
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    data_file = os.path.join(data_dir, config.get_data_filename())
    
    # Check if file already exists
    if os.path.exists(data_file) and not force:
        print(f"✓ Data file already exists: {data_file}")
        print("  Use --force to regenerate.")
        
        # Load and show stats
        data = np.load(data_file)
        X, y = data['X'], data['y']
        print(f"\n  Samples: {len(X)}")
        print(f"  Shape: X={X.shape}, y={y.shape}")
        
        return data_file
    
    # Print configuration
    print("=" * 70)
    print("GENERATING DATASET")
    print("=" * 70)
    print(config)
    if limit_stocks:
        print(f"  Limit: {limit_stocks} stocks (for testing)")
    print("=" * 70)
    print()
    
    # Generate data
    X, y = generate_training_data(
        lookback=config.lookback,
        horizon=config.horizon,
        limit_stocks=limit_stocks,
        start_date=config.start_date,
        end_date=config.end_date,
        use_multiprocessing=True
    )
    
    if len(X) == 0:
        print("\n✗ No training data generated. Check your data or criteria.")
        return None
    
    # Save data
    print(f"\nSaving data to {data_file}...")
    np.savez(data_file, X=X, y=y)
    print("Data saved.")
    
    # Save configuration
    config_file = config.save(directory=data_dir)
    print(f"Config saved to {config_file}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Samples: {len(X)}")
    print(f"Shape: X={X.shape}, y={y.shape}")
    print(f"Target (y) statistics:")
    print(f"  Mean: {np.mean(y):.4f}%")
    print(f"  Std: {np.std(y):.4f}%")
    print(f"  Min: {np.min(y):.4f}%")
    print(f"  Max: {np.max(y):.4f}%")
    print(f"  Median: {np.median(y):.4f}%")
    
    # Distribution
    positive = np.sum(y > 0)
    negative = np.sum(y < 0)
    print(f"\nReturn distribution:")
    print(f"  Positive: {positive} ({positive/len(y)*100:.1f}%)")
    print(f"  Negative: {negative} ({negative/len(y)*100:.1f}%)")
    print(f"  Neutral: {len(y) - positive - negative} ({(len(y)-positive-negative)/len(y)*100:.1f}%)")
    print("=" * 70)
    
    return data_file


def batch_generate(configs, limit_stocks=None, force=False):
    """
    Generate multiple datasets in batch.
    
    Args:
        configs: List of DataConfig instances
        limit_stocks: Limit number of stocks for testing
        force: Force regeneration even if file exists
    """
    print("\n" + "=" * 70)
    print(f"BATCH GENERATION: {len(configs)} configurations")
    print("=" * 70)
    print()
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Processing configuration:")
        print(f"  Lookback={config.lookback}, Horizon={config.horizon}")
        
        data_file = generate_dataset(config, limit_stocks=limit_stocks, force=force)
        results.append((config, data_file))
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH GENERATION SUMMARY")
    print("=" * 70)
    for config, data_file in results:
        status = "✓" if data_file else "✗"
        print(f"{status} lb={config.lookback:2d}, h={config.horizon:2d} -> {os.path.basename(data_file) if data_file else 'FAILED'}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training datasets with flexible parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with specific parameters
  python src/generate_dataset.py --lookback 20 --horizon 3
  
  # Use a preset configuration
  python src/generate_dataset.py --preset medium
  
  # Generate all presets for comparison
  python src/generate_dataset.py --batch
  
  # Test with limited stocks
  python src/generate_dataset.py --lookback 10 --horizon 3 --limit 50
  
  # List available presets
  python src/generate_dataset.py --list-presets
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--preset", type=str, help="Use a preset configuration (short, medium, long, default)")
    config_group.add_argument("--batch", action="store_true", help="Generate all preset configurations")
    config_group.add_argument("--list-presets", action="store_true", help="List available preset configurations")
    
    # Custom parameters
    parser.add_argument("--lookback", type=int, help="Lookback period (days)")
    parser.add_argument("--horizon", type=int, help="Prediction horizon (days)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYYMMDD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYYMMDD)")
    parser.add_argument("--description", type=str, default="", help="Configuration description")
    
    # Generation options
    parser.add_argument("--limit", type=int, help="Limit number of stocks (for testing)")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if file exists")
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        list_presets()
        return
    
    # Batch generation
    if args.batch:
        configs = list(PRESETS.values())
        batch_generate(configs, limit_stocks=args.limit, force=args.force)
        return
    
    # Single configuration
    if args.preset:
        config = get_preset(args.preset)
    else:
        # Create custom configuration
        config = DataConfig(
            lookback=args.lookback or 20,
            horizon=args.horizon or 3,
            start_date=args.start_date or "20100101",
            end_date=args.end_date or "20221231",
            description=args.description
        )
    
    # Override config with command line arguments if provided
    if args.lookback is not None:
        config.lookback = args.lookback
    if args.horizon is not None:
        config.horizon = args.horizon
    if args.start_date is not None:
        config.start_date = args.start_date
    if args.end_date is not None:
        config.end_date = args.end_date
    
    generate_dataset(config, limit_stocks=args.limit, force=args.force)


if __name__ == "__main__":
    main()
