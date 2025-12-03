"""
Configuration management for training data generation.

This module provides a flexible way to manage parameters for data generation,
training, and backtesting.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime


@dataclass
class DataConfig:
    """Configuration for data generation and model training."""
    
    # Data generation parameters
    lookback: int = 20
    horizon: int = 3
    start_date: str = "20100101"
    end_date: str = "20221231"
    
    # Model parameters
    input_size: int = 13  # Will be overwritten by actual data feature size
    hidden_size: int = 128
    num_layers: int = 4
    output_size: int = 1
    model_type: str = "transformer"  # transformer or lstm
    nhead: int = 4
    dim_feedforward: int = 256
    
    # Training parameters
    learning_rate: float = 0.001
    num_epochs: int = 50
    batch_size: int = 64
    
    # Optional metadata
    description: str = ""
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_data_filename(self, prefix="training_data", suffix="npz"):
        """
        Generate a unique filename based on configuration parameters.
        
        Example: training_data_lb20_h3_2010_2022.npz
        """
        return f"{prefix}_lb{self.lookback}_h{self.horizon}_{self.start_date}_{self.end_date}.{suffix}"
    
    def get_config_filename(self):
        """
        Generate a config filename matching the data filename.
        
        Example: config_lb20_h3_2010_2022.json
        """
        return self.get_data_filename(prefix="config", suffix="json")
    
    def save(self, directory="data"):
        """
        Save configuration to a JSON file.
        
        Args:
            directory: Directory to save the config file
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, self.get_config_filename())
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to the config file
        
        Returns:
            DataConfig instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def from_data_file(cls, data_filepath, directory="data"):
        """
        Load configuration based on a data filename.
        
        Args:
            data_filepath: Path to the data file (e.g., training_data_lb20_h3_2010_2022.npz)
            directory: Directory containing config files
        
        Returns:
            DataConfig instance or None if config not found
        """
        # Extract config filename from data filename
        basename = os.path.basename(data_filepath)
        config_name = basename.replace("training_data_", "config_").replace(".npz", ".json")
        config_path = os.path.join(directory, config_name)
        
        if os.path.exists(config_path):
            return cls.load(config_path)
        
        return None
    
    def __str__(self):
        """String representation of the configuration."""
        lines = [
            "Data Configuration:",
            f"  Lookback: {self.lookback}",
            f"  Horizon: {self.horizon}",
            f"  Date Range: {self.start_date} - {self.end_date}",
            (
                f"  Model: Transformer(d_model={self.hidden_size}, layers={self.num_layers}, nhead={self.nhead})"
                if self.model_type == "transformer"
                else f"  Model: LSTM(input={self.input_size}, hidden={self.hidden_size}, layers={self.num_layers})"
            ),
            f"  Training: LR={self.learning_rate}, Epochs={self.num_epochs}, Batch={self.batch_size}",
        ]
        if self.description:
            lines.insert(1, f"  Description: {self.description}")
        
        return "\n".join(lines)


# Preset configurations for common use cases
PRESETS = {
    "short": DataConfig(
        lookback=10,
        horizon=3,
        description="Short-term prediction (10 days lookback, 3 days horizon)"
    ),
    "medium": DataConfig(
        lookback=20,
        horizon=5,
        description="Medium-term prediction (20 days lookback, 5 days horizon)"
    ),
    "long": DataConfig(
        lookback=30,
        horizon=7,
        description="Long-term prediction (30 days lookback, 7 days horizon)"
    ),
    "default": DataConfig(
        lookback=20,
        horizon=3,
        description="Default configuration (backward compatible)"
    ),
}


def get_preset(name: str) -> DataConfig:
    """
    Get a preset configuration by name.
    
    Args:
        name: Preset name (short, medium, long, default)
    
    Returns:
        DataConfig instance
    
    Raises:
        KeyError: If preset name not found
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Preset '{name}' not found. Available presets: {available}")
    
    return PRESETS[name]


def list_presets():
    """Print all available preset configurations."""
    print("Available Presets:")
    print("-" * 60)
    for name, config in PRESETS.items():
        print(f"\n{name.upper()}:")
        print(config)


if __name__ == "__main__":
    # Demo usage
    print("=== Configuration Management Demo ===\n")
    
    # Show presets
    list_presets()
    
    print("\n" + "=" * 60)
    print("\n=== Creating and Saving Custom Config ===\n")
    
    # Create custom config
    custom_config = DataConfig(
        lookback=15,
        horizon=4,
        start_date="20150101",
        end_date="20231231",
        description="Custom configuration for experimentation"
    )
    
    print(custom_config)
    print(f"\nData filename: {custom_config.get_data_filename()}")
    print(f"Config filename: {custom_config.get_config_filename()}")
    
    # Save config
    saved_path = custom_config.save()
    print(f"\nConfig saved to: {saved_path}")
    
    # Load config
    loaded_config = DataConfig.load(saved_path)
    print("\n=== Loaded Config ===\n")
    print(loaded_config)
