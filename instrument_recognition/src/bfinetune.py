"""
Stage 2: 真实混音微调
加载 Stage 1 纯净音轨预训练模型 → 在真实混音上微调

正样本: 该类在混音中活跃的窗口
负样本: 该类不在、但其他乐器在活跃的窗口 (教会模型"有音乐≠有我")
"""
import os, sys, json, glob, time
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from datetime import datetime
from src.bmodel import BinaryInstrumentClassifier

SR = 22050
DURATION = 3
N_MELS = 128
N_MFCC = 13
N_MODGD = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
STEM_MIX_RATIO = 0.2  # 混入 20% 纯净音轨防止遗忘

MIX_DIR = r"D:\program_project\MUSE\data\muse_real_mixed_dataset"
STEM_DIR = r"D:\program_project\MUSE\data\clean_stems"
CACHE_DIR = r"D:\program_project\MUSE\data\preprocessed_cache\binary_feats"
MODEL_DIR = r"D:\program_project\MUSE\instrument_recognition\model\binary"
LOG_DIR = r"D:\program_project\MUSE\instrument_recognition\model\log"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 二分类器名称 → labels.json 类名映射
INST_TO_CLASS = {
    'acoustic_guitar': 'acoustic guitar', 'cello': 'cello',
    'drum_set': 'drum set', 'electric_bass': 'electric bass',
    'electric_guitar': 'electric guitar', 'flute': 'flute',
    'piano': 'piano', 'singer': 'singer',
    'synthesizer': 'synthesizer', 'violin': 'violin',
}
CLASS_TO_INST = {v: k for k, v in INST_TO_CLASS.items()}

# 特征提取器
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
    """提取 [1, 269, T] 特征"""
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
    cpu_wav = audio.cpu()
    mel = DB(MEL(cpu_wav))
    mfcc = MFCC(cpu_wav)
    modgd = compute_modgd(cpu_wav)
    return torch.cat([mel, mfcc, modgd], dim=1)

def load_features_with_cache(wav_path, cache_key):
    """带缓存的特征加载"""
    cache_file = os.path.join(CACHE_DIR, f"mix_{cache_key}.pt")
    if os.path.exists(cache_file):
        return torch.load(cache_file, weights_only=False)
    feat = extract_features(wav_path)
    torch.save(feat, cache_file)
    return feat


def finetune_instrument(instrument, epochs=EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_name = INST_TO_CLASS[instrument]
    print(f"\n{'='*50}")
    print(f"Stage 2 微调: {instrument} ({class_name})")
    print(f"{'='*50}")

    # 加载 Stage 1 预训练模型
    stage1_path = os.path.join(MODEL_DIR, f"{instrument}.pth")
    if not os.path.exists(stage1_path):
        print(f"  错误: 找不到 Stage 1 模型 {stage1_path}，请先运行 btrain.py")
        return
    model = BinaryInstrumentClassifier().to(device)
    ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  已加载 Stage 1 模型 (之前最佳 Val F1={ckpt.get('val_f1', 'N/A')})")

    # 加载真实混音标注
    with open(os.path.join(MIX_DIR, "labels.json"), "r", encoding="utf-8") as f:
        md = json.load(f)
    classes = md["classes"]
    all_samples = md["train_samples"]  # 只用训练集
    print(f"  真实混音训练样本: {len(all_samples)}")

    # 按乐器分类正负样本
    class_idx = classes.index(class_name)
    pos_indices = [i for i, s in enumerate(all_samples) if s["labels"][class_idx] == 1]
    # 负样本: 该类不活跃，但至少有一个其他乐器活跃
    neg_indices = [i for i, s in enumerate(all_samples)
                   if s["labels"][class_idx] == 0 and sum(s["labels"]) > 0]
    print(f"  正样本(混音): {len(pos_indices)}, 负样本(混音): {len(neg_indices)}")

    if len(pos_indices) == 0:
        print(f"  错误: 没有正样本!")
        return

    # 1:1 平衡采样
    n_per_class = min(len(pos_indices), len(neg_indices))
    np.random.seed(42)
    chosen_pos = np.random.choice(pos_indices, n_per_class, replace=False)
    chosen_neg = np.random.choice(neg_indices, n_per_class, replace=False)

    # 加载并缓存特征
    audio_dir = os.path.join(MIX_DIR, "audio")
    mix_feats, mix_labels = [], []
    print("  加载混音特征...")
    for idx in tqdm(chosen_pos, desc="正样本", leave=False):
        s = all_samples[idx]
        feat = load_features_with_cache(os.path.join(audio_dir, s["file"]), f"{instrument}_pos_{idx}")
        mix_feats.append(feat)
        mix_labels.append(1.0)
    for idx in tqdm(chosen_neg, desc="负样本", leave=False):
        s = all_samples[idx]
        feat = load_features_with_cache(os.path.join(audio_dir, s["file"]), f"{instrument}_neg_{idx}")
        mix_feats.append(feat)
        mix_labels.append(0.0)
    print(f"  混音数据: {len(mix_feats)} 样本 ({n_per_class} 正, {n_per_class} 负)")

    # 混入纯净音轨（防遗忘）
    stem_feats, stem_labels, n_stem_pos = [], [], 0
    stem_path = os.path.join(STEM_DIR, instrument)
    if os.path.isdir(stem_path):
        n_stem = int(len(mix_feats) * STEM_MIX_RATIO / (1 + STEM_MIX_RATIO))
        stem_files = sorted(glob.glob(os.path.join(stem_path, "*.wav")))
        # 从其他类取等量负样本
        other_stems = []
        for cls in sorted(os.listdir(STEM_DIR)):
            if cls == instrument or not os.path.isdir(os.path.join(STEM_DIR, cls)):
                continue
            other_stems.extend(glob.glob(os.path.join(STEM_DIR, cls, "*.wav")))
        n_stem_half = max(1, n_stem // 2)
        chosen_stem_pos = np.random.choice(stem_files, min(n_stem_half, len(stem_files)), replace=False)
        chosen_stem_neg = np.random.choice(other_stems, min(n_stem_half, len(other_stems)), replace=False)
        for f in tqdm(chosen_stem_pos, desc="纯净正", leave=False):
            feat = load_features_with_cache(f, f"stem_{instrument}_{os.path.basename(f)}")
            stem_feats.append(feat)
            stem_labels.append(1.0)
            n_stem_pos += 1
        for f in tqdm(chosen_stem_neg, desc="纯净负", leave=False):
            feat = load_features_with_cache(f, f"stem_other_{os.path.basename(f)}")
            stem_feats.append(feat)
            stem_labels.append(0.0)
        print(f"  纯净音轨: {len(stem_feats)} 样本 ({n_stem_pos} 正)")

    # 合并
    all_feats = mix_feats + stem_feats
    all_labels = mix_labels + stem_labels
    X = torch.cat(all_feats, dim=0).unsqueeze(1)
    y = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)
    perm = torch.randperm(len(all_feats))
    X, y = X[perm], y[perm]
    split = int(len(all_feats) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    print(f"  总训练: {len(X_train)}, 验证: {len(X_val)}")

    # 训练
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 日志
    log_path = os.path.join(LOG_DIR, f"bfinetune_{datetime.now():%Y%m%d-%H%M%S}.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Instrument: {instrument} (Stage 2 fine-tune)\n")
        f.write(f"Mix samples: {len(mix_feats)} ({n_per_class} pos, {n_per_class} neg)\n")
        f.write(f"Stem samples: {len(stem_feats)} ({n_stem_pos} pos)\n")
        f.write("Epoch\tTrain_F1\tVal_F1\n")

    best_val_f1 = 0.0
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        list(zip(X_val, y_val)), batch_size=BATCH_SIZE)

    for epoch in range(epochs):
        model.train()
        train_tp, train_fp, train_fn = 0, 0, 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_tp += ((preds == 1) & (targets == 1)).sum().item()
            train_fp += ((preds == 1) & (targets == 0)).sum().item()
            train_fn += ((preds == 0) & (targets == 1)).sum().item()
        train_f1 = 2 * train_tp / (2 * train_tp + train_fp + train_fn + 1e-8)

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
        scheduler.step()

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
                "stage": "2_finetune",
            }, save_path)
            print(f"    → 保存 (Val F1={val_f1:.4f})")

    # ========== 困难负样本挖掘 (Hard Negative Mining) ==========
    print(f"\n  开始困难负样本挖掘...")
    model.eval()
    hard_negatives = []
    batch_size = 64
    for i in range(0, len(all_samples), batch_size):
        batch = all_samples[i:i+batch_size]
        batch_feats = []
        for s in batch:
            f = load_features_with_cache(os.path.join(audio_dir, s["file"]),
                                          f"hnm_{instrument}_{all_samples.index(s)}")
            batch_feats.append(f)
        X_batch = torch.cat(batch_feats, dim=0).unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
        for j, s in enumerate(batch):
            label = s["labels"][class_idx]
            prob = probs[j]
            # FP: 概率高(>0.5)但标签为0，且其他乐器在活跃
            if label == 0 and prob > 0.5 and sum(s["labels"]) > 0:
                hard_negatives.append((all_samples.index(s), prob))

    print(f"  找到 {len(hard_negatives)} 个困难负样本")
    if len(hard_negatives) > 50:
        # 取 top-K 最难负样本 (概率最高的)
        hard_negatives.sort(key=lambda x: -x[1])
        top_k = min(len(hard_negatives), n_per_class // 4)  # 最多加 25% 负样本
        hard_indices = [h[0] for h in hard_negatives[:top_k]]

        # 加载这些困难负样本
        hnm_feats = []
        for idx in tqdm(hard_indices, desc="加载困难负样本", leave=False):
            s = all_samples[idx]
            f = load_features_with_cache(os.path.join(audio_dir, s["file"]),
                                          f"hnm_{instrument}_{idx}")
            hnm_feats.append(f)

        # 合并到训练集 (替换掉等量原始负样本)
        n_replace = min(len(hnm_feats), len(mix_feats) // 4)
        new_feats = mix_feats[:-n_replace] + hnm_feats[:n_replace] + stem_feats
        new_labels = mix_labels[:-n_replace] + [0.0] * len(hnm_feats[:n_replace]) + stem_labels
        X2 = torch.cat(new_feats, dim=0).unsqueeze(1)
        y2 = torch.tensor(new_labels, dtype=torch.float32).unsqueeze(1)

        split2 = int(len(new_feats) * 0.8)
        perm = torch.randperm(len(new_feats))
        X2, y2 = X2[perm], y2[perm]

        # 继续训练 5 轮
        print(f"  继续训练 5 轮 (困难负样本 {len(hnm_feats[:n_replace])} 个)...")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.3)
        train_loader2 = torch.utils.data.DataLoader(
            list(zip(X2[:split2], y2[:split2])), batch_size=BATCH_SIZE, shuffle=True)
        val_loader2 = torch.utils.data.DataLoader(
            list(zip(X2[split2:], y2[split2:])), batch_size=BATCH_SIZE)

        for epoch in range(5):
            model.train()
            tp, fp, fn = 0, 0, 0
            for inp, tgt in tqdm(train_loader2, desc=f"HNM Epoch {epoch+1}/5", leave=False):
                inp, tgt = inp.to(device), tgt.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inp), tgt)
                loss.backward()
                optimizer.step()
                preds = (torch.sigmoid(model(inp)) > 0.5).float()
                tp += ((preds == 1) & (tgt == 1)).sum().item()
                fp += ((preds == 1) & (tgt == 0)).sum().item()
                fn += ((preds == 0) & (tgt == 1)).sum().item()
            train_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

            model.eval()
            tp, fp, fn = 0, 0, 0
            with torch.no_grad():
                for inp, tgt in val_loader2:
                    inp, tgt = inp.to(device), tgt.to(device)
                    preds = (torch.sigmoid(model(inp)) > 0.5).float()
                    tp += ((preds == 1) & (tgt == 1)).sum().item()
                    fp += ((preds == 1) & (tgt == 0)).sum().item()
                    fn += ((preds == 0) & (tgt == 1)).sum().item()
            val_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            print(f"  HNM Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")
            with open(log_path, "a") as f:
                f.write(f"HNM{epoch+1}\t{train_f1:.4f}\t{val_f1:.4f}\n")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "instrument": instrument, "model_state_dict": model.state_dict(),
                    "val_f1": val_f1, "stage": "2_finetune_hnm",
                }, os.path.join(MODEL_DIR, f"{instrument}.pth"))
                print(f"    → 保存 (Val F1={val_f1:.4f})")

    print(f"  [{instrument}] Stage 2 完成! 最佳 Val F1={best_val_f1:.4f}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Best Val F1: {best_val_f1:.4f}\n")
    return best_val_f1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    finetune_instrument(args.instrument, args.epochs)
