# infer.py — LightweightUMX 推理引擎
import os
import torch
import torch.nn as nn
import numpy as np
from src.config import config
from src.model import LightweightUMX


# STFT 参数
N_FFT = 1024
HOP_LENGTH = 256
N_BINS = N_FFT // 2 + 1


def load_model(model_path=None, device=None):
    """加载训练好的分离模型"""
    if device is None:
        device = config.DEVICE
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, "guitar.pth")

    model = LightweightUMX(
        n_bins=N_BINS,
        hidden=config.BLSTM_HIDDEN,
        num_layers=config.BLSTM_LAYERS,
    ).to(device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"加载模型: {model_path}")
    else:
        print(f"警告: 模型不存在 {model_path}, 使用随机权重")

    return model


@torch.no_grad()
def separate(mix_audio, model, device=None):
    """
    对一段混合音频执行吉他分离

    Args:
        mix_audio: [T] torch tensor, mono 22050Hz 混合音频
        model: LightweightUMX 模型
        device: torch device

    Returns:
        guitar_audio: [T] numpy array, 分离后的吉他音频
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    orig_len = len(mix_audio)
    mix = mix_audio.to(device)

    # STFT
    window = torch.hann_window(N_FFT, device=device)
    X = torch.stft(
        mix, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=N_FFT, window=window,
        return_complex=True
    )  # [F, T_frames]
    # 注: 在 DeepSeek CUDA 下 torch.abs(complex) 会编译失败，手动计算
    X_real = X.real
    X_imag = X.imag
    mag = torch.sqrt(X_real ** 2 + X_imag ** 2)  # [F, T_frames]
    # 将 phase 计算延后到重建时直接使用 real/imag

    # 模型推理 → 掩码
    mag_input = mag.unsqueeze(0)  # [1, F, T]
    mask = model(mag_input).squeeze(0)  # [F, T]

    # 应用掩码 + 重建复数谱（直接用 real/imag，避免 angle）
    mag_hat = mask * mag
    cos_theta = X_real / torch.clamp(mag, min=1e-8)
    sin_theta = X_imag / torch.clamp(mag, min=1e-8)
    real = mag_hat * cos_theta
    imag = mag_hat * sin_theta
    complex_spec = torch.complex(real, imag)

    # iSTFT
    guitar = torch.istft(
        complex_spec, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=N_FFT, window=window,
        length=orig_len
    )

    return guitar.cpu().numpy()
