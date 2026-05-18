# binary_train.py — 训练单个乐器的二分类器
# 用法: python -m src.binary_train --instrument drum_set
import os
import sys
import glob
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from datetime import datetime
from src.binary_model import BinaryInstrumentClassifier

SR = 22050
DURATION = 3
N_MELS = 128
N_MFCC = 13
N_MODGD = 128
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

STEMS_DIR = r"D:\program_project\MUSE\data\clean_stems"
MODEL_DIR = r"D:\program_project\MUSE\instrument_recognition\model\binary"
CACHE_DIR = r"D:\program_project\MUSE\data\preprocessed_cache\binary_feats"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 特征提取器（同原模型保持一致）
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

def extract_features(wav_path):
    audio, sr = torchaudio.load(wav_path)
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    target_len = SR * DURATION
    if audio.shape[1] < target_len:
        audio = nn.functional.pad(audio, (0, target_len - audio.shape[1]))
    else:
        audio = audio[:, :target_len]

    mel = DB(MEL(audio))
    mfcc = MFCC(audio)
    modgd = compute_modgd(audio)
    return torch.cat([mel, mfcc, modgd], dim=1)  # [1, 269, T]

def load_class_samples(class_name):
    """加载某类所有样本的特征和标签（带缓存）"""
    cache_path = os.path.join(CACHE_DIR, f"{class_name}.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=False)
        print(f"  缓存命中: {class_name} ({len(data['feats'])} 样本)")
        return data["feats"], data["labels"]

    path = os.path.join(STEMS_DIR, class_name)
    if not os.path.isdir(path):
        return [], []
    files = sorted(glob.glob(os.path.join(path, "*.wav")))
    feats, labels = [], []
    for f in tqdm(files, desc=f"加载 {class_name}"):
        feats.append(extract_features(f))
        labels.append(1.0)
    if len(feats) == 0:
        print(f"警告: {class_name} 没有样本!")
    # 保存缓存
    torch.save({"feats": feats, "labels": labels}, cache_path)
    return feats, labels

def train_one_instrument(instrument, epochs=EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练乐器: {instrument} ({device})")

    # 加载正样本
    pos_feats, pos_labels = load_class_samples(instrument)
    n_pos = len(pos_feats)
    print(f"  正样本: {n_pos}")

    # 加载负样本（所有其他类）
    neg_feats = []
    for cls in sorted(os.listdir(STEMS_DIR)):
        if cls == instrument or not os.path.isdir(os.path.join(STEMS_DIR, cls)):
            continue
        feats, _ = load_class_samples(cls)
        neg_feats.extend(feats)
    n_neg = len(neg_feats)
    print(f"  负样本: {n_neg}")

    if n_pos == 0 or n_neg == 0:
        print("  错误: 正或负样本为空!")
        return

    # 组装数据集 + 平衡采样（正:负 = 1:1）
    all_feats = []
    all_labels = []
    # 全部正样本 + 等量负样本
    indices = np.random.permutation(n_neg)
    neg_selected = [neg_feats[i] for i in indices[:n_pos]]
    all_feats.extend(pos_feats)
    all_feats.extend(neg_selected)
    all_labels.extend([1.0] * n_pos)
    all_labels.extend([0.0] * n_neg if n_neg <= n_pos else [0.0] * n_pos)

    n_total = len(all_feats)
    print(f"  总训练样本: {n_total} ({n_pos} 正, {min(n_pos, n_neg)} 负)")

    # 转为 tensor
    X = torch.cat(all_feats, dim=0).unsqueeze(1)  # [N, 1, 269, T]
    y = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)  # [N, 1]

    # 随机打乱
    perm = torch.randperm(n_total)
    X, y = X[perm], y[perm]

    # 80/20 切分
    split = int(n_total * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = BinaryInstrumentClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # 日志
    log_dir = r"D:\program_project\MUSE\instrument_recognition\model\log"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"binary_{instrument}_{datetime.now():%Y%m%d-%H%M%S}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Instrument: {instrument}\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"Train samples: {n_total} ({n_pos} pos, {min(n_pos, n_neg)} neg)\n")
        f.write(f"Val samples: {n_total - split} ({len([l for l in all_labels[split:] if l==1])} pos)\n")
        f.write("Epoch\tTrain_F1\tVal_F1\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_f1 = 0.0
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        list(zip(X_val, y_val)), batch_size=BATCH_SIZE
    )

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_tp, train_fp, train_fn = 0, 0, 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_tp += ((preds == 1) & (targets == 1)).sum().item()
            train_fp += ((preds == 1) & (targets == 0)).sum().item()
            train_fn += ((preds == 0) & (targets == 1)).sum().item()

        train_f1 = 2 * train_tp / (2 * train_tp + train_fp + train_fn + 1e-8)

        # 验证
        model.eval()
        val_tp, val_fp, val_fn = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_tp += ((preds == 1) & (targets == 1)).sum().item()
                val_fp += ((preds == 1) & (targets == 0)).sum().item()
                val_fn += ((preds == 0) & (targets == 1)).sum().item()

        val_f1 = 2 * val_tp / (2 * val_tp + val_fp + val_fn + 1e-8)
        print(f"  Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1}\t{train_f1:.4f}\t{val_f1:.4f}\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(MODEL_DIR, f"{instrument}.pth")
            torch.save({
                "instrument": instrument,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
            }, save_path)
            print(f"    → Saved {save_path} (Val F1={val_f1:.4f})")

    print(f"  [{instrument}] 完成! 最佳 Val F1 = {best_val_f1:.4f}")
    return best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train_one_instrument(args.instrument, args.epochs)
