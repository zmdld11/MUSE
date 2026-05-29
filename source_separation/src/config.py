# config.py — 音轨分离全局配置
import os
import torch

class Config:
    def __init__(self):
        self.WORKSPACE_DIR = r"D:\program_project\MUSE\source_separation"
        self.MODEL_VERSION = "VER5.0_RemixAug"
        self.TARGET_INSTRUMENT = "guitar"  # acoustic + electric 合并

        # Audio params
        self.SR = 22050
        self.NUM_SAMPLES = 65536  # 2.97s, 整除 4^4=256, 解码器完美对齐

        # Model params (4 层, 拼接式跳跃连接)
        self.DEMUCS_CHANNELS = (48, 96, 192, 384)

        # MRSTFT loss params
        self.MRSTFT_FFT_SIZES = [2048, 1024, 512]
        self.MRSTFT_WEIGHT = 0.5

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

        # 原始分轨数据源 (用于在线随机混音)
        self.MEDLEYDB_DIR = r"D:\program_project\MUSE\data\MedleyDB\MedleyDB"
        self.MEDLEYDB_META_DIR = r"D:\program_project\MUSE\data\MedleyDB\Metadata"
        self.MOISESDB_DIR = r"D:\program_project\MUSE\data\moisesdb_v0.1"
        self.STEM_INDEX_CACHE = os.path.join(self.DATASET_DIR, "stem_index.json")
        self.REMIX_TOTAL = 25000  # 每 epoch 的随机混音样本数

        # Instrument recognition paths (for integration)
        self.INST_MODEL_DIR = r"D:\program_project\MUSE\instrument_recognition\model\binary"
        self.INST_SRC_DIR = r"D:\program_project\MUSE\instrument_recognition"

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

config = Config()
