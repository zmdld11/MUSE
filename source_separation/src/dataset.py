# dataset.py — 吉他分离数据集加载
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import soundfile as sf

SR = 22050
DURATION = 3.0
N_FFT = 1024
HOP_LENGTH = 256
N_BINS = N_FFT // 2 + 1  # 513


def get_stft():
    """返回配置好的 STFT"""
    window = torch.hann_window(N_FFT)
    return window


def compute_mag(audio, window):
    """计算幅度谱 [F, T]"""
    X = torch.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=N_FFT, window=window,
        return_complex=True
    )
    return torch.abs(X)


def target_mask(mix_mag, guitar_mag, eps=1e-8):
    """计算目标比值掩码: guitar_mag / mix_mag, clip到[0,1]"""
    mask = guitar_mag / torch.clamp(mix_mag, min=eps)
    return torch.clamp(mask, 0.0, 1.0)


class GuitarSeparationDataset(Dataset):
    """吉他音轨分离数据集

    每个样本返回 (mix_mag, guitar_mag, target_mask)
    - mix_mag:    [F, T] 混合音频幅度谱
    - guitar_mag: [F, T] 吉他目标幅度谱
    - target_mask:[F, T] 理想比值掩码
    """

    def __init__(self, metadata_path, audio_dir, augment=False):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        self.samples = meta["train_samples"]
        self.audio_dir = audio_dir
        self.augment = augment
        self.window = get_stft()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mix_path = os.path.join(self.audio_dir, sample["mix"])
        gtr_path = os.path.join(self.audio_dir, sample["guitar"])

        mix_audio, _ = sf.read(mix_path)
        gtr_audio, _ = sf.read(gtr_path)

        mix_audio = torch.from_numpy(mix_audio.astype(np.float32))
        gtr_audio = torch.from_numpy(gtr_audio.astype(np.float32))

        # 数据增强：同步音量缩放
        if self.augment:
            gain_db = random.uniform(-3.0, 3.0)
            gain_linear = 10 ** (gain_db / 20.0)
            mix_audio = mix_audio * gain_linear
            gtr_audio = gtr_audio * gain_linear

        # STFT → magnitude
        mix_mag = compute_mag(mix_audio, self.window)   # [F, T]
        gtr_mag = compute_mag(gtr_audio, self.window)   # [F, T]

        # 如果帧数不对齐，裁剪到较短者
        min_T = min(mix_mag.shape[1], gtr_mag.shape[1])
        mix_mag = mix_mag[:, :min_T]
        gtr_mag = gtr_mag[:, :min_T]

        # 目标掩码
        mask = target_mask(mix_mag, gtr_mag)

        return mix_mag, gtr_mag, mask
