"""eval/compare_real.py — 真实录音对比: 管线输出 vs 手工参考 MIDI (夜の向日葵).

对比维度:
  1. 总体: note_f1 / offset_f1 (mir_eval)
  2. 拾音正确率: GT 音符被检出的比例 (recall) + 多检数 (FP)
  3. 节奏正确率: 匹配上的音符中 onset 偏差分布 (≤50ms/≤100ms/≤200ms)
  4. 按音区拆分: 低/中/高音区的表现

用法: python -m eval.compare_real
"""
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pretty_midi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.ERROR)

ONSET_TOL = 0.05
PITCH_TOL_CENTS = 50


def midi_to_hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi, dtype=np.float64) - 69) / 12.0)


def load_notes(path, merge_instruments=True):
    """加载 MIDI 音符. merge_instruments=True 时合并所有轨道 (忽略声部拆分)."""
    pm = pretty_midi.PrettyMIDI(path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append({"onset": float(n.start), "offset": float(n.end),
                          "pitch": int(n.pitch)})
    notes.sort(key=lambda n: n["onset"])
    return notes


def analyze(est_notes, gt_notes):
    """四分类 + 节奏偏差统计."""
    from scipy.optimize import linear_sum_assignment

    n_gt = len(gt_notes)
    n_est = len(est_notes)

    # 匹配矩阵
    cost = np.full((n_gt, n_est), np.inf)
    for i, g in enumerate(gt_notes):
        for j, e in enumerate(est_notes):
            if abs(e["onset"] - g["onset"]) <= ONSET_TOL and \
               abs(midi_to_hz(e["pitch"]) - midi_to_hz(g["pitch"])) <= PITCH_TOL_CENTS:
                cost[i, j] = 0.0

    matched = {}
    if n_gt > 0 and n_est > 0 and np.isfinite(cost).any():
        mcost = np.where(np.isfinite(cost), -1.0, 0.0)
        rows, cols = linear_sum_assignment(mcost)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if np.isfinite(cost[r, c]):
                matched[r] = c

    tp = len(matched)
    miss = n_gt - tp
    fp = n_est - tp

    # 节奏偏差: 匹配对 (经最大匹配)
    onset_diffs = []
    for gi, ej in matched.items():
        onset_diffs.append(abs(est_notes[ej]["onset"] - gt_notes[gi]["onset"]))
    onset_diffs = np.array(onset_diffs) if onset_diffs else np.array([0.0])

    # 按音区拆分 GT 命中
    by_region = defaultdict(lambda: {"gt": 0, "hit": 0})
    for gi, g in enumerate(gt_notes):
        region = "low" if g["pitch"] < 55 else ("mid" if g["pitch"] < 72 else "high")
        by_region[region]["gt"] += 1
        if gi in matched:
            by_region[region]["hit"] += 1

    return {
        "tp": tp, "miss": miss, "fp": fp,
        "n_gt": n_gt, "n_est": n_est,
        "onset_diffs": onset_diffs,
        "by_region": dict(by_region),
    }


def main():
    est_path = os.path.join(os.path.dirname(__file__), "..", "output",
                            "夜の向日葵 - 松本文紀", "piano.mid")
    gt_path = os.path.join(os.path.dirname(__file__), "..", "output",
                           "himawari_reference", "himawari_reference.mid")

    est = load_notes(est_path)
    gt = load_notes(gt_path)
    print(f"我们的: {len(est)} 音符   对照组: {len(gt)} 音符")

    r = analyze(est, gt)

    print(f"\n=== 真实录音对比 (夜の向日葵) ===")
    print(f"拾音正确率:")
    print(f"  GT 被检出 (recall) : {r['tp']}/{r['n_gt']} ({r['tp']/r['n_gt']*100:.1f}%)")
    print(f"  漏检               : {r['miss']} ({r['miss']/r['n_gt']*100:.1f}%)")
    print(f"  多检 (杂音)        : {r['fp']} ({r['fp']/r['n_est']*100:.1f}% 的预测)")
    precision = r['tp'] / r['n_est'] if r['n_est'] else 0
    recall = r['tp'] / r['n_gt'] if r['n_gt'] else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"  → precision={precision:.3f} recall={recall:.3f} note_f1={f1:.3f}")

    print(f"\n节奏正确率 (匹配对 onset 偏差):")
    d = r["onset_diffs"]
    print(f"  ≤50ms : {(d<=0.05).mean()*100:.1f}%")
    print(f"  ≤100ms: {(d<=0.10).mean()*100:.1f}%")
    print(f"  ≤200ms: {(d<=0.20).mean()*100:.1f}%")
    print(f"  中位数 : {np.median(d)*1000:.0f}ms")

    print(f"\n按音区 (GT 命中率):")
    for region in ["low", "mid", "high"]:
        b = r["by_region"].get(region, {"gt": 0, "hit": 0})
        if b["gt"]:
            print(f"  {region:5s}: {b['hit']}/{b['gt']} ({b['hit']/b['gt']*100:.1f}%)")


if __name__ == "__main__":
    main()
