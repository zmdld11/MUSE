# config.py — 音轨分离全局配置
import os
import torch

class Config:
    def __init__(self):
        self.WORKSPACE_DIR = r"D:\program_project\MUSE\source_separation"
        self.MODEL_VERSION = "VER3.0_DemucsLM"
        self.TARGET_INSTRUMENT = "guitar"  # acoustic + electric 合并

        # Audio params
        self.SR = 22050
        self.DURATION = 3  # seconds

        # Model params
        self.DEMUCS_CHANNELS = (16, 32, 64, 128, 256)

        # Train params
        self.BATCH_SIZE = 16
        self.EPOCHS = 100
        self.LR = 3e-4
        self.WEIGHT_DECAY = 1e-5
        self.EARLY_STOPPING_PATIENCE = 10
        self.SCHEDULER_PATIENCE = 3
        self.SCHEDULER_FACTOR = 0.5

        # Data paths
        self.DATASET_DIR = os.path.join(self.WORKSPACE_DIR, "data")
        self.MODEL_DIR = os.path.join(self.WORKSPACE_DIR, "model")
        self.LOG_DIR = os.path.join(self.MODEL_DIR, "log")
        self.OUTPUT_DIR = os.path.join(self.WORKSPACE_DIR, "output")

        # Instrument recognition paths (for integration)
        self.INST_MODEL_DIR = r"D:\program_project\MUSE\instrument_recognition\model\binary"
        self.INST_SRC_DIR = r"D:\program_project\MUSE\instrument_recognition"

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

config = Config()
