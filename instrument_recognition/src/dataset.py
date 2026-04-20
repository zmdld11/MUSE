import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
# IRMAS 训练集通常是 3 秒的音频切片
DURATION = 3.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

class IRMASDataset(Dataset):
    def __init__(self, data_path, is_train=True, split_ratio=0.8):
        self.data_path = data_path
        self.is_train = is_train
        self.file_paths = []
        self.labels = []
        
        # 遍历数据集目录
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(data_path, class_name)
            if not os.path.exists(class_dir):
                continue
            
            # IRMAS 文件通常是 .wav
            wav_files = glob.glob(os.path.join(class_dir, "*.wav"))
            
            # 简单的训练/验证分割
            split_idx = int(len(wav_files) * split_ratio)
            if is_train:
                files_to_use = wav_files[:split_idx]
            else:
                files_to_use = wav_files[split_idx:]
                
            self.file_paths.extend(files_to_use)
            self.labels.extend([CLASS_TO_IDX[class_name]] * len(files_to_use))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 加载音频，统一截断或填充到 SAMPLES_PER_TRACK
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            if len(y) > SAMPLES_PER_TRACK:
                y = y[:SAMPLES_PER_TRACK]
            else:
                y = np.pad(y, (0, max(0, SAMPLES_PER_TRACK - len(y))))
                
            # 提取 Mel 频谱图
            S = librosa.feature.melspectrogram(
                y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            # 归一化到 [0, 1]
            S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
            
            # 增加 channel 维度 (1, n_mels, time_steps)
            mel_tensor = torch.tensor(S_dB_norm, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            # 文件读取失败时返回一个全零的张量
            mel_tensor = torch.zeros((1, N_MELS, int(SAMPLES_PER_TRACK/HOP_LENGTH) + 1), dtype=torch.float32)
            
        return mel_tensor, torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_path, batch_size=32, num_workers=4):
    train_dataset = IRMASDataset(data_path, is_train=True)
    val_dataset = IRMASDataset(data_path, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
