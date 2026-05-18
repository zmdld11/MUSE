"""集成推理：加载全部 10 个二分类器，对音频做多标签预测"""
import os
import json
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm
from src.bmodel import BinaryInstrumentClassifier

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
    print("  加载 10 个二分类模型...", flush=True)
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
    print(f"  共 {n_windows} 个窗口 ({n_windows * len(inst_names)} 次推理)...")

    with torch.no_grad():
        for t in tqdm(range(n_windows), desc="推理进度", leave=False):
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


def _load_cooccurrence():
    """从训练数据加载乐器共现矩阵，返回 [10,10] 条件概率 P(col | row)"""
    matrix_path = os.path.join(os.path.dirname(MODEL_DIR), "cooccurrence.npy")
    if os.path.exists(matrix_path):
        return np.load(matrix_path)
    # 若不存在，从训练标签计算
    labels_path = r"D:\program_project\MUSE\data\muse_real_mixed_dataset\labels.json"
    if not os.path.exists(labels_path):
        return None
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_classes = len(data["classes"])
    cooc = np.zeros((n_classes, n_classes))
    counts = np.zeros(n_classes)
    for s in data.get("train_samples", []):
        lbl = np.array(s["labels"])
        active = np.where(lbl > 0)[0]
        for a in active:
            counts[a] += 1
            for b in active:
                cooc[a, b] += 1
    # 归一化为条件概率 P(col | row)
    cond = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        if counts[i] > 0:
            cond[i] = cooc[i] / counts[i]
    np.save(matrix_path, cond)
    return cond


# 加载共现矩阵（模块级别，只加载一次）
COOCCURRENCE = _load_cooccurrence()


def post_process(probs, inst_names, thresholds=None, min_active_frames=2,
                 cooccurrence_threshold=0.03):
    """后处理：平滑 + 共现门控 + 频段门控 + 最短激活帧数

    Args:
        probs: [N_windows, 10] 原始概率
        thresholds: 每类阈值 or None(默认0.5)
        min_active_frames: 最少连续激活帧数
        cooccurrence_threshold: 共现概率低于此值的乐器对互斥
    """
    n_classes = len(inst_names)
    if thresholds is None:
        thresholds = [0.5] * n_classes
    elif isinstance(thresholds, (int, float)):
        thresholds = [thresholds] * n_classes

    # 1. 按阈值二值化
    binary = np.zeros_like(probs)
    for i in range(n_classes):
        binary[:, i] = (probs[:, i] >= thresholds[i]).astype(int)

    # 2a. 共现门控：对每个窗口，若两个互斥乐器都被检出，压制置信度低的
    if COOCCURRENCE is not None:
        for t in range(len(binary)):
            active = np.where(binary[t] == 1)[0]
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    a, b = active[i], active[j]
                    # 若 P(a|b) 或 P(b|a) 极低，则它们互斥
                    if COOCCURRENCE[a, b] < cooccurrence_threshold and COOCCURRENCE[b, a] < cooccurrence_threshold:
                        # 压制置信度低的那个
                        if probs[t, a] >= probs[t, b]:
                            binary[t, b] = 0
                        else:
                            binary[t, a] = 0

    # 2b. 频段门控 (旧 infer.py 规则)
    def idx_for(name):
        try:
            return inst_names.index(NAME_MAP_TO_INST[name])
        except (ValueError, KeyError):
            return -1

    GATES = [
        ('piano', 'electric bass'),
        ('piano', 'electric guitar'),
        ('singer', 'violin'),
        ('electric guitar', 'violin'),
        ('drum set', 'synthesizer'),
        ('piano', 'synthesizer'),
    ]
    for dominant, suppressed in GATES:
        d = idx_for(dominant)
        s = idx_for(suppressed)
        if d < 0 or s < 0:
            continue
        for t in range(len(binary)):
            if binary[t, d] == 1 and probs[t, d] > probs[t, s] + 0.15:
                binary[t, s] = 0

    # 3. 最少连续激活帧数
    for i in range(n_classes):
        active = binary[:, i].copy()
        start = None
        for t in range(len(active)):
            if active[t] == 1 and start is None:
                start = t
            elif active[t] == 0 and start is not None:
                if t - start < min_active_frames:
                    binary[start:t, i] = 0
                start = None
        if start is not None and len(active) - start < min_active_frames:
            binary[start:, i] = 0

    # 4. 重算去噪后的概率
    cleaned = probs.copy()
    for i in range(n_classes):
        cleaned[:, i] = np.where(binary[:, i] == 1, probs[:, i], probs[:, i] * 0.3)

    return cleaned, binary


# 显示名 → inst名字 反向映射
NAME_MAP_TO_INST = {v: k for k, v in NAME_MAP.items()}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models = load_ensemble(device)
    print("Ensemble ready!")
    print(f"Total params: {sum(sum(p.numel() for p in m.parameters()) for m in models.values()):,}")
