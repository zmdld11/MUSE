# config.py
import os
import torch

class Config:
    def __init__(self):
        self.WORKSPACE_DIR = r"D:\program_project\MUSE\instrument_recognition"
        self.MODEL_VERSION = "VER4.0_BinaryEnsemble"
        self.DATASET_VERSION = "V3_SyntheticStems"
        self.DATASET_DIR = r"D:\program_project\MUSE\data\MedleyDB"  # 丢弃劣质 IRMAS，锁定高质分离音轨
        self.CACHE_DIR = r"D:\program_project\MUSE\data\preprocessed_cache"
        self.MODEL_DIR = os.path.join(self.WORKSPACE_DIR, "model")
        self.LOG_DIR = os.path.join(self.MODEL_DIR, "log")

        self.CLASSES = ['acoustic guitar', 'cello', 'drum set', 'electric bass', 'electric guitar', 'flute', 'piano', 'singer', 'synthesizer', 'violin']

        # Audio params
        self.SR = 22050
        self.DURATION = 3 # seconds
        self.N_MELS = 128
        self.N_MFCC = 13
        self.N_MODGD = 128  # Modified Group Delay gram features (phase information)

        # MedleyDB Control (Prevent RAM explosion on 64GB PC)
        self.MAX_MEDLEY_STEMS_PER_CLASS = 10  # Only grab up to 10 pure audio stems per class
        self.MAX_CHUNKS_PER_STEM = 8          # Up to 8 onset-based 3s chunks per stem
        self.MEDLEY_SAMPLE_WEIGHT = 3.0       # Weight for Medley samples vs 1.0 for IRMAS

        # Train params
        self.BATCH_SIZE = 32
        self.EPOCHS = 200
        self.LR = 5e-4  # 降低学习率适配更大的Transformer模型

        # Transformer params
        self.TRANSFORMER_HEADS = 4
        self.TRANSFORMER_LAYERS = 2
        self.TRANSFORMER_DIM_FF = 512
        self.FOCAL_LOSS_GAMMA = 2.0

        # Inference params
        self.INFER_SMOOTH_WINDOW = 3  # 移动平均平滑窗口（从VER3.2的5降为3，减少短音稀释）
        self.INFER_LOG_OUTPUT = True  # 是否输出每窗口详细预测结果CSV日志
        self.ANTI_HALLUCINATION_WEIGHT = 0.15  # [VER3.5] 抗幻觉辅助loss权重
        self.FEATURE_STATS_PATH = os.path.join(self.MODEL_DIR, "feature_stats.pth")  # [VER3.5] 特征归一化统计量
        
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

config = Config()
