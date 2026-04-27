# config.py
import os
import torch

class Config:
    def __init__(self):
        self.WORKSPACE_DIR = r"D:\program_project\MUSE\instrument_recognition"
        self.MODEL_VERSION = "VER1.5_AugmentedCNN"
        self.DATASET_DIR = r"D:\program_project\MUSE\data\IRMAS-TrainingData"
        self.CACHE_DIR = r"D:\program_project\MUSE\data\preprocessed_cache"
        self.MODEL_DIR = os.path.join(self.WORKSPACE_DIR, "model")
        self.LOG_DIR = os.path.join(self.MODEL_DIR, "log")
        
        # Audio params
        self.SR = 22050
        self.DURATION = 3 # seconds
        self.N_MELS = 128
        self.N_MFCC = 13
        
        # Train params
        self.BATCH_SIZE = 32
        self.EPOCHS = 200
        self.LR = 1e-3
        
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

config = Config()
