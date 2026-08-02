"""eval/error_analysis.py — 错误归因: 漏检 / 多检 / 错检(音高错) / 命中.

对每首曲目:
  GT 音符 ↔ 预测音符 按 onset±50ms 匹配, 分成四类:
    TP  : onset 对 + 音高对 (±50 cents)
    错检: onset 对 + 音高错   ← 模型听到有音符但音高分错
    漏检: GT 音符处无任何预测  ← 没听到 / 被后处理删
    多检: 预测音符处无任何 GT  ← 噪声 / 泛音 / 幻听

按 复调度(单音/和弦) 和 音区 拆分, 定位瓶颈.
用法: python -m eval.error_analysis [--n 40] [--model ours]
"""
import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.ERROR)

SR = 22050
ONSET_TOL = 0.05      # ±50ms
PITCH_TOL_CENTS = 50  # ±50 cents = 半音的一半


def midi_to_hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi, dtype=np.float64) - 69) / 12.0)


def analyze_song(gt, est_notes):
    """单首四分类 (标准最大匹配), 返回 (stats, per_gt 列表).

    用 scipy Hungarian 算法做 GT↔est 最大匹配 (mir_eval 同款),
    保证一个 est 至多匹配一个 GT, 计数不重复.
    per_gt: 每个 GT 音符的 {poly, region, kind} (tp/wrong/miss).
    """
    from scipy.optimize import linear_sum_assignment

    stats = defaultdict(int)
    per_gt = []

    est_sorted = sorted(est_notes, key=lambda n: n["onset"])
    n_gt = len(gt["intervals"])
    n_est = len(est_sorted)

    # 匹配矩阵: cost = 0 可匹配, inf 不可
    cost = np.full((n_gt, n_est), np.inf)
    for i, (onset, pitch) in enumerate(zip(gt["intervals"][:, 0], gt["pitches"])):
        for j, n in enumerate(est_sorted):
            if abs(n["onset"] - onset) <= ONSET_TOL and \
               abs(midi_to_hz(n["pitch"]) - midi_to_hz(pitch)) <= PITCH_TOL_CENTS:
                cost[i, j] = 0.0

    # 最大匹配: cost=-1 偏好匹配, 0 为不可匹配 (不惩罚)
    matched_map = {}  # gt_idx -> est_idx
    if n_gt > 0 and n_est > 0 and np.isfinite(cost).any():
        mcost = np.where(np.isfinite(cost), -1.0, 0.0)
        rows, cols = linear_sum_assignment(mcost)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if np.isfinite(cost[r, c]):
                matched_map[r] = c

    est_used = [False] * n_est
    # GT 帧网格 (hop=512)
    gt_frame_sec = 512 / SR
    frame_labels = gt["frame_labels"]

    for i, (onset, pitch) in enumerate(zip(gt["intervals"][:, 0], gt["pitches"])):
        f0 = int(onset / gt_frame_sec)
        poly = 1
        if 0 <= f0 < len(frame_labels):
            poly = max(1, int(frame_labels[f0].sum()))
        region = "low" if pitch < 55 else ("mid" if pitch < 72 else "high")

        if i in matched_map:
            j = matched_map[i]
            est_used[j] = True
            kind = "tp"
            stats["tp"] += 1
        else:
            # 错检: onset±50ms 内有未用 est (音高错); 否则漏检
            has_near = any(not est_used[j] and abs(n["onset"] - onset) <= ONSET_TOL
                           for j, n in enumerate(est_sorted))
            if has_near:
                kind = "wrong"
                stats["wrong"] += 1
            else:
                kind = "miss"
                stats["miss"] += 1

        per_gt.append({"poly": poly, "region": region, "kind": kind})

    n_fp = sum(1 for u in est_used if not u)
    stats["fp"] += n_fp

    return stats, per_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", choices=["ours", "basic"], default="ours")
    args = ap.parse_args()

    from eval.dataset import load, sample_midis
    from src.transcriber import _load_model, _ours_inference, _basic_pitch_inference
    from src.frame_post import process_frames
    from src.note_post import refine_notes
    import scipy.io.wavfile as wavfile

    total = defaultdict(int)
    by_poly = defaultdict(lambda: {"gt": 0, "tp": 0, "wrong": 0, "miss": 0})
    by_region = defaultdict(lambda: {"gt": 0, "tp": 0, "wrong": 0, "miss": 0})
    all_detail = []

    tmp = os.path.join(os.path.dirname(__file__), "reports", "_err.wav")
    midis = sample_midis(args.n, seed=42)

    for i, mid in enumerate(midis, 1):
        gt = load(mid)
        audio = gt["audio"]
        wavfile.write(tmp, SR, (audio * 32767).clip(-32768, 32767).astype(np.int16))
        if args.model == "ours":
            model = _load_model()
            res = _ours_inference(model, tmp)
        else:
            res = _basic_pitch_inference(tmp)
        hop = res.get("hop_length", 512)
        sr = res.get("sr", SR)
        notes = refine_notes(process_frames(res["onset_probs"], res["frame_probs"],
                                            hop_length=hop, sr=sr),
                             audio, sr=sr, hop_length=hop)

        stats, per_gt = analyze_song(gt, notes)
        for k, v in stats.items():
            total[k] += v
        for pg in per_gt:
            by_poly[pg["poly"]]["gt"] += 1
            by_region[pg["region"]]["gt"] += 1
            by_poly[pg["poly"]][pg["kind"]] += 1
            by_region[pg["region"]][pg["kind"]] += 1

    # --- 输出 ---
    tp = total["tp"]; wrong = total["wrong"]; miss = total["miss"]; fp = total["fp"]
    gt_total = tp + wrong + miss
    print(f"\n=== 错误归因 ({args.model}, {args.n} 首) ===")
    print(f"GT 音符总数: {gt_total}")
    print(f"  命中 TP   : {tp:6d} ({tp/gt_total*100:.1f}%)  ← 音高对+onset对")
    print(f"  错检      : {wrong:6d} ({wrong/gt_total*100:.1f}%)  ← 听到音但音高分错 (可修!)")
    print(f"  漏检      : {miss:6d} ({miss/gt_total*100:.1f}%)  ← 没听到/被后处理删")
    print(f"  多检 FP   : {fp:6d}  ← 预测了不存在的音 (噪声/泛音)")
    recall = tp / gt_total if gt_total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"  → recall={recall:.3f} precision={precision:.3f} note_f1={f1:.3f}")

    print(f"\n=== 按复调度拆分 (GT 侧) ===")
    print(f"{'复调度':>5s} {'GT':>7s} {'命中':>6s} {'错检':>6s} {'漏检':>6s} {'命中率':>7s} {'错检率':>7s} {'漏检率':>7s}")
    for poly in sorted(by_poly):
        b = by_poly[poly]
        if b["gt"] == 0:
            continue
        print(f"{poly:5d} {b['gt']:7d} {b['tp']:6d} {b['wrong']:6d} {b['miss']:6d} "
              f"{b['tp']/b['gt']*100:6.1f}% {b['wrong']/b['gt']*100:6.1f}% {b['miss']/b['gt']*100:6.1f}%")

    print(f"\n=== 按音区拆分 (GT 侧) ===")
    print(f"{'音区':>6s} {'GT':>7s} {'命中':>6s} {'错检':>6s} {'漏检':>6s} {'命中率':>7s} {'错检率':>7s} {'漏检率':>7s}")
    for region in ["low", "mid", "high"]:
        b = by_region[region]
        if b["gt"] == 0:
            continue
        print(f"{region:6s} {b['gt']:7d} {b['tp']:6d} {b['wrong']:6d} {b['miss']:6d} "
              f"{b['tp']/b['gt']*100:6.1f}% {b['wrong']/b['gt']*100:6.1f}% {b['miss']/b['gt']*100:6.1f}%")


if __name__ == "__main__":
    main()
