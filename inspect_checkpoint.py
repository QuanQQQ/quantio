import torch
import os

checkpoint_path = 'checkpoints/best_checkpoint.pth'
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            print("Config found in checkpoint:")
            print(checkpoint['config'])
        else:
            print("No config found in checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print("Checkpoint not found.")
