import os
import pickle

def check_and_train():
    """Train model if pkl files don't exist"""
    if not os.path.exists('models/resume_model.pkl'):
        print("No model found. Training now...")
        from app.model import train_model
        train_model()
    else:
        print("Model already exists. Skipping training.")

if __name__ == "__main__":
    check_and_train()