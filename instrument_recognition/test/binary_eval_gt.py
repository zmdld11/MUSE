"""用 10 个二分类器集成评估标准对照集"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, r"D:\program_project\MUSE\instrument_recognition")
sys.path.insert(0, r"D:\program_project\MUSE\instrument_recognition\test")
from binary_infer import *

GT_DIR = r"D:\program_project\MUSE\data\ground_truth"
CLASS_NAMES = ['acoustic guitar', 'cello', 'drum set', 'electric bass',
               'electric guitar', 'flute', 'piano', 'singer', 'synthesizer', 'violin']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_ensemble(device)

all_results = []

for song_id in sorted(os.listdir(GT_DIR)):
    song_dir = os.path.join(GT_DIR, song_id)
    gt_path = os.path.join(song_dir, "ground_truth.json")
    mix_path = os.path.join(song_dir, "mix.wav")
    if not os.path.exists(gt_path) or not os.path.exists(mix_path):
        continue

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    print(f"\n=== {song_id} ===")
    probs, inst_names = predict_file(mix_path, models, device)

    gt_windows = gt["windows"]
    num_windows = min(len(probs), len(gt_windows))
    probs = probs[:num_windows]
    gt_labels = np.array([w["multi_label"] for w in gt_windows[:num_windows]])

    # 阈值扫描（全局最优 F1）
    best_thresholds = {}
    for c_idx, cls_name in enumerate(CLASS_NAMES):
        best_f1 = 0
        best_th = 0.3
        for th in np.arange(0.05, 0.95, 0.05):
            preds = (probs[:, c_idx] >= th).astype(int)
            targets = gt_labels[:, c_idx].astype(int)
            tp = ((preds == 1) & (targets == 1)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()
            fn = ((preds == 0) & (targets == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        best_thresholds[cls_name] = {"threshold": best_th, "f1": round(float(best_f1), 4)}

    # 使用最优阈值计算最终结果
    print(f"  Per-class results (best threshold):")
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for c_idx, cls_name in enumerate(CLASS_NAMES):
        th = best_thresholds[cls_name]["threshold"]
        preds = (probs[:, c_idx] >= th).astype(int)
        targets = gt_labels[:, c_idx].astype(int)
        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()
        tn = ((preds == 0) & (targets == 0)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn
        flag = ""
        if f1 < 0.5: flag = " [BAD]"
        elif f1 < 0.8: flag = " [WARN]"
        print(f"    {cls_name:20s}: F1={f1:.4f} th={th:.2f} tp={tp:4d} fp={fp:4d} fn={fn:4d}{flag}")

    micro_f1 = 2 * totals["tp"] / (2 * totals["tp"] + totals["fp"] + totals["fn"] + 1e-8)
    print(f"  {'GLOBAL':20s}: Micro F1={micro_f1:.4f}")
    all_results.append({"song": song_id, "micro_f1": round(float(micro_f1), 4),
                         "per_class": {c: best_thresholds[c] for c in CLASS_NAMES}})

# 汇总
print(f"\n{'='*60}")
print(f"  CROSS-SONG SUMMARY")
print(f"{'='*60}")
class_aggr = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
for r in all_results:
    song_dir = os.path.join(GT_DIR, r["song"])
    gt_path = os.path.join(song_dir, "ground_truth.json")
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    probs, _ = predict_file(os.path.join(song_dir, "mix.wav"), models, device)
    gt_windows = gt["windows"]
    num_windows = min(len(probs), len(gt_windows))
    gt_labels = np.array([w["multi_label"] for w in gt_windows[:num_windows]])
    probs = probs[:num_windows]
    for c_idx, cls_name in enumerate(CLASS_NAMES):
        th = r["per_class"][cls_name]["threshold"]
        preds = (probs[:, c_idx] >= th).astype(int)
        targets = gt_labels[:, c_idx].astype(int)
        class_aggr[cls_name]["tp"] += ((preds == 1) & (targets == 1)).sum().item()
        class_aggr[cls_name]["fp"] += ((preds == 1) & (targets == 0)).sum().item()
        class_aggr[cls_name]["fn"] += ((preds == 0) & (targets == 1)).sum().item()

g_tp, g_fp, g_fn = 0, 0, 0
for cls_name in CLASS_NAMES:
    ca = class_aggr[cls_name]
    f1 = 2 * ca["tp"] / (2 * ca["tp"] + ca["fp"] + ca["fn"] + 1e-8)
    g_tp += ca["tp"]; g_fp += ca["fp"]; g_fn += ca["fn"]
    print(f"  {cls_name:20s}: F1={f1:.4f} tp={ca['tp']:5d} fp={ca['fp']:5d} fn={ca['fn']:5d}")

micro = 2 * g_tp / (2 * g_tp + g_fp + g_fn + 1e-8)
print(f"  {'-'*50}")
print(f"  {'GLOBAL':20s}: Micro F1={micro:.4f}")
print(f"  VER3.5 (multi-label) baseline: Global F1=0.527")
