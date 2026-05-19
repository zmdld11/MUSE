# config.py — 音轨分离全局配置
import os
import torch

class Config:
    def __init__(self):
        self.WORKSPACE_DIR = r"D:\program_project\MUSE\source_separation"
        self.MODEL_VERSION = "VER1.0_LightweightUMX"
        self.TARGET_INSTRUMENT = "guitar"  # acoustic + electric 合并

        # Audio params
        self.SR = 22050
        self.DURATION = 3  # seconds
        self.N_FFT = 1024
        self.HOP_LENGTH = 256
        self.N_BINS = self.N_FFT // 2 + 1  # 513

        # Model params
        self.BLSTM_HIDDEN = 128
        self.BLSTM_LAYERS = 1

        # Train params
        self.BATCH_SIZE = 64
        self.EPOCHS = 100
        self.LR = 3e-4
        self.WEIGHT_DECAY = 1e-5
        self.EARLY_STOPPING_PATIENCE = 15
        self.SCHEDULER_PATIENCE = 5
        self.SCHEDULER_FACTOR = 0.5

        # Data paths
        self.DATASET_DIR = r"D:\program_project\MUSE\data\separation_dataset"
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
