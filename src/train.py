import sys
import os
import argparse
import glob

# Add local libs directory to path (for custom installation location)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
libs_path = os.path.join(project_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from data_processor import generate_training_data
from model import StockLSTM
from config import DataConfig, get_preset

def train_model(config=None, resume_path=None):
    """
    Train the stock prediction model.
    
    Args:
        config: DataConfig instance. If None, uses default configuration.
        resume_path: Path to checkpoint file to resume from.
    """
    # Use provided config or default
    if config is None:
        config = DataConfig()
    
    # Extract parameters from config
    INPUT_SIZE = config.input_size
    HIDDEN_SIZE = config.hidden_size
    NUM_LAYERS = config.num_layers
    OUTPUT_SIZE = config.output_size
    LEARNING_RATE = config.learning_rate
    NUM_EPOCHS = config.num_epochs
    BATCH_SIZE = config.batch_size
    LOOKBACK = config.lookback
    HORIZON = config.horizon
    START_DATE = config.start_date
    END_DATE = config.end_date
    
    # Generate data filename from config
    DATA_FILE = os.path.join("data", config.get_data_filename())
    
    # Print configuration
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(config)
    print("=" * 70)
    print()
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU. CUDA not available.")
        print(f"PyTorch Version: {torch.__version__}")
    
    # 1. Generate or Load Data
    if os.path.exists(DATA_FILE):
        print(f"Loading training data from {DATA_FILE}...")
        data = np.load(DATA_FILE)
        X = data['X']
        y = data['y']
        print("Data loaded.")
    else:
        print(f"Generating training data from {START_DATE} to {END_DATE}...")
        # Limit stocks for testing if needed, or remove limit for full training
        X, y = generate_training_data(lookback=LOOKBACK, horizon=HORIZON, limit_stocks=None, start_date=START_DATE, end_date=END_DATE) 
        
        if len(X) > 0:
            print(f"Saving data to {DATA_FILE}...")
            np.savez(DATA_FILE, X=X, y=y)
            print("Data saved.")
    
    if len(X) == 0:
        print("No training data generated. Check your data or criteria.")
        return
        
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Scale Target (Percentage -> Ratio)
    # This helps convergence as features are small (~0.1) and target was large (~5.0)
    print("Scaling target by 0.01 (Percentage -> Ratio)...")
    y = y * 0.01
    
    # Convert to Tensor
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().view(-1, 1) # Shape (N, 1)
    
    # Split Train/Test
    dataset_size = len(X)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # DataLoader optimization
    # num_workers=0 is safest for Windows. Increase if you want to try multiprocessing loading.
    # pin_memory=True helps transfer to GPU.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type=='cuda'))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))
    
    # 2. Initialize Model
    model = StockLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    
    # Use HuberLoss for robustness against outliers
    criterion = nn.HuberLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Early Stopping parameters
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint
    if resume_path:
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint '{resume_path}'...")
            checkpoint = torch.load(resume_path, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_loss = checkpoint['best_loss']
            print(f"Resumed from epoch {start_epoch}. Best loss so far: {best_loss:.4f}")
        else:
            print(f"Checkpoint '{resume_path}' not found. Starting from scratch.")

    # 3. Training Loop
    print("Start training...")
    from tqdm import tqdm
    
    # Create checkpoints directory
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        # Progress bar for batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar description with current loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        # Update Scheduler
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'config': config.__dict__ # Save config for reference
        }
        # Save latest checkpoint
        torch.save(checkpoint, 'checkpoints/latest_checkpoint.pth')
        
        # Checkpointing and Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/stock_lstm.pth')
            # Save best checkpoint 
            torch.save(checkpoint, 'checkpoints/best_checkpoint.pth')
            # print(f"  Saved best model (Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
    print(f"Training completed. Best Validation Loss: {best_loss:.4f}")
    print("Model saved to models/stock_lstm.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train stock prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python src/train.py
  
  # Train with specific parameters
  python src/train.py --lookback 20 --horizon 5
  
  # Use a preset configuration
  python src/train.py --preset medium
  
  # Load from config file
  python src/train.py --config data/config_lb20_h3_2010_2022.json

  # Resume training
  python src/train.py --resume checkpoints/latest_checkpoint.pth
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config", type=str, help="Path to config JSON file")
    config_group.add_argument("--preset", type=str, help="Use a preset configuration (short, medium, long, default)")
    
    # Custom parameters (override config)
    parser.add_argument("--lookback", type=int, help="Lookback period (days)")
    parser.add_argument("--horizon", type=int, help="Prediction horizon (days)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYYMMDD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYYMMDD)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--resume", type=str, help="Path to checkpoint file to resume from")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        # Load from config file
        config = DataConfig.load(args.config)
        print(f"Loaded configuration from {args.config}")
    elif args.preset:
        # Use preset
        config = get_preset(args.preset)
        print(f"Using preset configuration: {args.preset}")
    else:
        # Default configuration (backward compatible)
        config = DataConfig()
    
    # Override with command line arguments if provided
    if args.lookback is not None:
        config.lookback = args.lookback
    if args.horizon is not None:
        config.horizon = args.horizon
    if args.start_date is not None:
        config.start_date = args.start_date
    if args.end_date is not None:
        config.end_date = args.end_date
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    
    # Train model
    train_model(config, resume_path=args.resume)
