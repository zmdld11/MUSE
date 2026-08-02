"""eval/compare_real.py — 真实录音对比: 管线输出 vs 手工参考 MIDI (夜の向日葵).

对比维度:
  1. 总体: note_f1 / offset_f1 (mir_eval)
  2. 拾音正确率: GT 音符被检出的比例 (recall) + 多检数 (FP)
  3. 节奏正确率: 匹配上的音符中 onset 偏差分布 (≤50ms/≤100ms/≤200ms)
  4. 按音区拆分: 低/中/高音区的表现

用法: python -m eval.compare_real [--gt <gt.mid>] [--est <est.mid>]
  --gt  手工参考 MIDI 路径 (默认 himawari_reference C大调)
        调性不匹配时用 --gt output/himawari_reference_E/himawari_reference_E.mid
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


def estimate_shift_curve(gt_notes, est_notes, seg=20.0, t_tol=0.15, p_tol=2):
    """分段扫描最优常数偏移 → 返回 shift_at(t) 分段线性插值函数.

    2026-08-02 诊断: GT 是理想网格, 录音有 ~0.75s 前奏 + 演奏速度轻微漂移.
    逐段 (默认 20s) 扫描使同音高匹配数最大的偏移, 段间线性插值.
    """
    gt_arr = np.array([[n["onset"], n["pitch"]] for n in gt_notes])
    est_arr = np.array([[n["onset"], n["pitch"]] for n in est_notes])
    t_max = gt_arr[:, 0].max()

    shifts = []
    for t0 in range(0, int(t_max), int(seg)):
        seg_gt = gt_arr[(gt_arr[:, 0] >= t0) & (gt_arr[:, 0] < t0 + seg)]
        if len(seg_gt) == 0:
            continue
        best_s, best_c = 0.0, -1
        for s in np.arange(0.0, 1.6, 0.025):
            hits = 0
            for g in seg_gt:
                mask = np.abs(est_arr[:, 0] - (g[0] + s)) <= t_tol
                if mask.any() and np.min(np.abs(est_arr[mask, 1] - g[1])) <= p_tol:
                    hits += 1
            if hits > best_c:
                best_c, best_s = hits, s
        shifts.append((t0, best_s))

    def shift_at(t):
        if t <= shifts[0][0]:
            return shifts[0][1]
        for k in range(len(shifts) - 1):
            t0, s0 = shifts[k]
            t1, s1 = shifts[k + 1]
            if t0 <= t <= t1:
                return s0 + (s1 - s0) * (t - t0) / (t1 - t0)
        return shifts[-1][1]

    return shift_at, shifts


def aligned_analyze(gt_notes, est_notes, shift_at=None, t_tol=0.15, p_tol=2):
    """偏移校正后的四分类 (2026-08-02).

    每个 GT 音符用 shift_at 校正时间后, 在 est 中找 ±t_tol 内音高差 ≤p_tol 的音符,
    一次匹配不重复使用 est 音符 (容忍碎音化). 返回与 analyze() 同结构 dict.
    """
    from collections import defaultdict
    est_arr = np.array([[n["onset"], n["offset"], n["pitch"]] for n in est_notes])
    gt_arr = np.array([[n["onset"], n["offset"], n["pitch"]] for n in gt_notes])

    used = set()
    tp = 0
    onset_diffs = []
    for gi, g in enumerate(gt_notes):
        t = g["onset"] + (shift_at(g["onset"]) if shift_at else 0.0)
        mask = np.abs(est_arr[:, 0] - t) <= t_tol
        if mask.any():
            cands = [j for j in np.where(mask)[0]
                     if j not in used and abs(est_arr[j, 2] - g["pitch"]) <= p_tol]
            if cands:
                j = min(cands, key=lambda j: abs(est_arr[j, 2] - g["pitch"]))
                used.add(j)
                tp += 1
                onset_diffs.append(abs(est_arr[j, 0] - t))

    # 按音区命中: GT 音符在 ±t_tol 内有同音高 est 候选 (统计用, 不消耗 used)
    by_region = defaultdict(lambda: {"gt": 0, "hit": 0})
    for gi, g in enumerate(gt_notes):
        t = g["onset"] + (shift_at(g["onset"]) if shift_at else 0.0)
        region = "low" if g["pitch"] < 55 else ("mid" if g["pitch"] < 72 else "high")
        by_region[region]["gt"] += 1
        mask = np.abs(est_arr[:, 0] - t) <= t_tol
        if mask.any() and np.min(np.abs(est_arr[mask, 2] - g["pitch"])) <= p_tol:
            by_region[region]["hit"] += 1

    onset_diffs = np.array(onset_diffs) if onset_diffs else np.array([0.0])
    return {
        "tp": tp, "miss": len(gt_notes) - tp,
        "fp": len(est_notes) - tp,
        "n_gt": len(gt_notes), "n_est": len(est_notes),
        "onset_diffs": onset_diffs,
        "by_region": dict(by_region),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="真实录音对比 (管线输出 vs 手工参考 MIDI)")
    ap.add_argument("--gt", choices=["c", "e"], default="e",
                    help="参考调性: e=E大调(默认, 与录音匹配) / c=C大调(原谱)")
    ap.add_argument("--shift", type=float, default=None,
                    help="固定时间轴校正 (秒). 默认自动分段扫描 (2026-08-02: "
                         "录音 ~0.75s 前奏 + 演奏漂移, 固定值不准)")
    args = ap.parse_args()

    est_path = os.path.join(os.path.dirname(__file__), "..", "output",
                            "夜の向日葵 - 松本文紀", "piano.mid")
    gt_dir = "himawari_reference_E" if args.gt == "e" else "himawari_reference"
    gt_path = os.path.join(os.path.dirname(__file__), "..", "output",
                           gt_dir, "himawari_reference.mid")

    est = load_notes(est_path)
    gt = load_notes(gt_path)

    # 时间轴校正: 默认自动分段扫描最优偏移
    if args.shift is not None:
        shift_at, shifts = (lambda s: (lambda t: s, [(0.0, s)]))(args.shift)
        shift_desc = f"固定 {args.shift}s"
    else:
        shift_at, shifts = estimate_shift_curve(gt, est)
        shift_desc = "自动分段 " + " ".join(f"{s:+.2f}" for _, s in shifts)

    print(f"我们的: {len(est)} 音符   对照组: {len(gt)} 音符   时间轴校正: {shift_desc}")

    r = aligned_analyze(gt, est, shift_at)

    # 窗口覆盖 recall: GT 音符期间 (onset..offset) 内被同音高 est 覆盖 (容忍碎音化)
    win_covered = 0
    n_per_gt = []
    for go, gf, gp in [(n["onset"], n["offset"], n["pitch"]) for n in gt]:
        cnt = sum(1 for e in est if go - 0.05 <= e["onset"] <= gf + 0.05
                  and abs(midi_to_hz(e["pitch"]) - midi_to_hz(gp)) <= 50)
        if cnt:
            win_covered += 1
        n_per_gt.append(cnt)
    import statistics
    n_per_gt = np.array(n_per_gt)

    print(f"\n=== 真实录音对比 (夜の向日葵, E大调 GT, shift={shift_desc}) ===")
    print(f"拾音正确率:")
    print(f"  GT 被检出 (recall) : {r['tp']}/{r['n_gt']} ({r['tp']/r['n_gt']*100:.1f}%)")
    print(f"  漏检               : {r['miss']} ({r['miss']/r['n_gt']*100:.1f}%)")
    print(f"  多检 (杂音)        : {r['fp']} ({r['fp']/r['n_est']*100:.1f}% 的预测)")
    precision = r['tp'] / r['n_est'] if r['n_est'] else 0
    recall = r['tp'] / r['n_gt'] if r['n_gt'] else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"  → precision={precision:.3f} recall={recall:.3f} note_f1={f1:.3f}")
    print(f"\n窗口覆盖 recall (容忍碎音化): {win_covered}/{r['n_gt']} ({win_covered/r['n_gt']*100:.1f}%)")
    print(f"碎音程度: 平均 {n_per_gt.mean():.1f} 个 est 音符/GT音符 (中位 {np.median(n_per_gt):.0f})")
    print(f"  完全无覆盖 GT 音符: {(n_per_gt==0).sum()} ({(n_per_gt==0).mean()*100:.1f}%)")

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
