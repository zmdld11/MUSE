# config.py — 音轨分离全局配置
import os
import torch

class Config:
    def __init__(self):
        self.WORKSPACE_DIR = r"D:\program_project\MUSE\source_separation"
        self.MODEL_VERSION = "VER6.0_DemucsFormat"
        self.TARGET_INSTRUMENT = "guitar"

        # Audio params
        self.SR = 22050
        self.SEGMENT = 6.0   # 片段长度(秒), 可改为 11.0 对齐 Demucs
        self.SHIFT = 3.0     # 验证集滑动步长(秒), 训练随机裁剪

        # Model params (4 层, 拼接式跳跃连接, GroupNorm)
        self.DEMUCS_CHANNELS = (48, 96, 192, 384)
        self.RESCALE = 0.1

        # MRSTFT loss params
        self.MRSTFT_FFT_SIZES = [2048, 1024, 512]
        self.MRSTFT_WEIGHT = 0.5

        # Train params
        self.BATCH_SIZE = 8
        self.EPOCHS = 100
        self.LR = 3e-4
        self.WEIGHT_DECAY = 1e-5
        self.GRAD_CLIP = 1.0
        self.EARLY_STOPPING_PATIENCE = 20

        # Data paths
        self.DATASET_DIR = os.path.join(self.WORKSPACE_DIR, "data")
        self.DEMUCS_FORMAT_DIR = os.path.join(self.DATASET_DIR, "demucs_format")
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
