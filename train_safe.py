"""
Safe training script with multiprocessing guard.
Use this if you encounter memory overflow errors.
"""
if __name__ == "__main__":
    from src.train import train_model
    train_model()
