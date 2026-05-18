"""集成推理：加载全部 10 个二分类器，对音频做多标签预测"""
import os
import json
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
from src.binary_model import BinaryInstrumentClassifier

SR = 22050
DURATION = 3
N_MELS = 128
N_MFCC = 13
N_MODGD = 128
HOP = 0.5

MODEL_DIR = r"D:\program_project\MUSE\instrument_recognition\model\binary"
THRESHOLD_PATH = r"D:\program_project\MUSE\instrument_recognition\model\class_thresholds.json"

CLASSES = ['acoustic guitar', 'cello', 'drum set', 'electric bass',
           'electric guitar', 'flute', 'piano', 'singer', 'synthesizer', 'violin']
# 文件名→显示名映射
NAME_MAP = {
    'acoustic_guitar': 'acoustic guitar',
    'cello': 'cello', 'drum_set': 'drum set', 'electric_bass': 'electric bass',
    'electric_guitar': 'electric guitar', 'flute': 'flute', 'piano': 'piano',
    'singer': 'singer', 'synthesizer': 'synthesizer', 'violin': 'violin',
}

# 特征提取器（与训练一致）
MEL = T.MelSpectrogram(sample_rate=SR, n_mels=N_MELS, n_fft=2048, hop_length=512)
DB = T.AmplitudeToDB(stype="power", top_db=80)
MFCC = T.MFCC(sample_rate=SR, n_mfcc=N_MFCC, melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": N_MELS})
MODGD_MEL = T.MelScale(n_mels=N_MODGD, sample_rate=SR, n_stft=2048 // 2 + 1)

def compute_modgd(audio, gamma=0.3):
    n_fft, hop_len = 2048, 512
    window = torch.hann_window(n_fft, device=audio.device)
    X = torch.stft(audio, n_fft=n_fft, hop_length=hop_len, win_length=n_fft, window=window, return_complex=True)
    n = torch.arange(audio.shape[-1], device=audio.device).float() / audio.shape[-1]
    Y = torch.stft(audio * n, n_fft=n_fft, hop_length=hop_len, win_length=n_fft, window=window, return_complex=True)
    tau = X.real * Y.real + X.imag * Y.imag
    S = torch.abs(X)
    S_s = (S[:, :, :-2] + S[:, :, 1:-1] + S[:, :, 2:]) / 3.0
    S_s = nn.functional.pad(S_s, (1, 1), mode='replicate')
    tau = tau / torch.clamp(S_s ** (2 * gamma), min=1e-6)
    mn, mx = tau.min(dim=1, keepdim=True).values, tau.max(dim=1, keepdim=True).values
    tau = (tau - mn) / (mx - mn + 1e-8)
    return MODGD_MEL(tau)

def extract_features(waveform):
    """waveform: [1, T] tensor on any device, returns [1, 269, T'] tensor on same device"""
    # 特征提取器在 CPU 上，先把音频拉回 CPU
    cpu_wav = waveform.cpu()
    mel = DB(MEL(cpu_wav))
    mfcc = MFCC(cpu_wav)
    modgd = compute_modgd(cpu_wav)
    feats = torch.cat([mel, mfcc, modgd], dim=1)
    return feats.to(waveform.device)

def load_ensemble(device):
    """加载全部 10 个二分类器"""
    models = {}
    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".pth"):
            continue
        inst = fname.replace(".pth", "")
        m = BinaryInstrumentClassifier().to(device)
        ckpt = torch.load(os.path.join(MODEL_DIR, fname), map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models[inst] = m
    print(f"  Loaded {len(models)} binary models")
    return models

def predict_file(audio_path, models, device):
    """对完整音频做逐窗口预测，返回 [N_windows, 10] 概率"""
    audio, sr = torchaudio.load(audio_path)
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    y = audio[0]
    win_len = SR * DURATION
    hop_len = int(SR * HOP)
    if y.shape[0] < win_len:
        y = nn.functional.pad(y, (0, win_len - y.shape[0]))
    n_windows = max(1, (y.shape[0] - win_len) // hop_len + 1)

    inst_names = sorted(NAME_MAP.keys())
    all_probs = np.zeros((n_windows, len(inst_names)))

    with torch.no_grad():
        for t in range(n_windows):
            start = t * hop_len
            end = start + win_len
            segment = y[start:end].unsqueeze(0).to(device)  # [1, T]
            feats = extract_features(segment)  # [1, 269, T']
            inp = feats.unsqueeze(0)  # [1, 1, 269, T']

            for i, inst in enumerate(inst_names):
                logit = models[inst](inp)
                all_probs[t, i] = torch.sigmoid(logit).item()

    # 平滑
    window = np.ones(3) / 3
    for i in range(all_probs.shape[1]):
        all_probs[:, i] = np.convolve(all_probs[:, i], window, mode='same')

    return all_probs, inst_names

def predict_file_parallel(audio_path, models, device):
    """并行推理版 — 所有模型同时处理同一段音频"""
    audio, sr = torchaudio.load(audio_path)
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    y = audio[0]
    win_len = SR * DURATION
    hop_len = int(SR * HOP)
    if y.shape[0] < win_len:
        y = nn.functional.pad(y, (0, win_len - y.shape[0]))
    n_windows = max(1, (y.shape[0] - win_len) // hop_len + 1)

    inst_names = sorted(NAME_MAP.keys())
    all_probs = np.zeros((n_windows, len(inst_names)))

    # 预提取所有窗口特征
    segments = []
    for t in range(n_windows):
        start = t * hop_len
        end = start + win_len
        segments.append(y[start:end].clone())

    with torch.no_grad():
        for t in range(n_windows):
            seg = segments[t].unsqueeze(0).to(device)
            feats = extract_features(seg)
            inp = feats.unsqueeze(0)
            for i, inst in enumerate(inst_names):
                logit = models[inst](inp)
                all_probs[t, i] = torch.sigmoid(logit).item()

    # 平滑
    window = np.ones(3) / 3
    for i in range(all_probs.shape[1]):
        all_probs[:, i] = np.convolve(all_probs[:, i], window, mode='same')

    return all_probs, inst_names

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models = load_ensemble(device)
    print("Ensemble ready!")
    print(f"Total params: {sum(sum(p.numel() for p in m.parameters()) for m in models.values()):,}")
