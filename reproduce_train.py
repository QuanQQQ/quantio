import sys
import os

# Add local libs directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(project_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.model import StockLSTM
import os

def test_training(scale_target=False):
    print(f"\n--- Testing with scale_target={scale_target} ---")
    
    # Load a subset of data
    DATA_FILE = "data/training_data_2018_2022_relative_scaling.npz"
    if not os.path.exists(DATA_FILE):
        print("Data file not found.")
        return

    data = np.load(DATA_FILE)
    X = data['X'][:1000] # Use only 1000 samples
    y = data['y'][:1000]
    
    if scale_target:
        print("Scaling target by 0.01 (Percentage -> Ratio)")
        y = y * 0.01
        
    # Convert to Tensor
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().view(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Model
    INPUT_SIZE = 11
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2 # Reduced for speed
    model = StockLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, output_size=1)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Lower LR
    
    print("Start training...")
    for epoch in range(5):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    test_training(scale_target=False)
    test_training(scale_target=True)
