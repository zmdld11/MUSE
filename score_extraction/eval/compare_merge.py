"""eval/compare_merge.py — A1 碎音合并效果评估 (真实录音).

对夜の向日葵真实录音:
  转录 → BP 后处理候选 → (可选) 碎音合并 → 与 E 大调 GT 匹配
对比不同 gap_tol 的 窗口覆盖 recall / 50ms recall-precision-F1 / 杂音数.

用法: python -m eval.compare_merge [--model ours|basic] [--no-cache]
"""
import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.ERROR)

AUDIO = r"d:\program_project\MUSE\music\夜の向日葵 - 松本文紀.flac"
GT = os.path.join(os.path.dirname(__file__), "..", "output",
                  "himawari_reference_E", "himawari_reference.mid")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SHIFT = 0.15  # 模型 onset 相对 GT 的固定延迟 (2026-08-02 诊断)


def midi_to_hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi, dtype=np.float64) - 69) / 12.0)


def load_notes(path):
    import pretty_midi
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


def transcribe(model_name, use_cache=True):
    """转录 → (onset_probs, frame_probs, hop_length). 缓存到 npz."""
    cache_path = os.path.join(CACHE_DIR, f"himawari_{model_name}.npz")
    if use_cache and os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"[cache] {model_name}: {d['frame_probs'].shape}")
        return d["onset_probs"], d["frame_probs"], int(d["hop_length"])

    from eval.eval import _write_tmp_wav
    import librosa
    audio, sr = librosa.load(AUDIO, sr=22050, mono=True)
    tmp = os.path.join(CACHE_DIR, "_tmp_himawari.wav")
    _write_tmp_wav(audio, tmp)

    if model_name == "ours":
        from src.transcriber import _load_model, _ours_inference
        model = _load_model()
        result = _ours_inference(model, tmp)
    else:
        from src.transcriber import _basic_pitch_inference
        result = _basic_pitch_inference(tmp)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, onset_probs=result["onset_probs"],
             frame_probs=result["frame_probs"], hop_length=result["hop_length"])
    print(f"[transcribe] {model_name}: {result['frame_probs'].shape}, "
          f"hop={result['hop_length']}")
    return (result["onset_probs"], result["frame_probs"], result["hop_length"])


def frames_to_times(n_frames_total, hop_length, sr=22050):
    if hop_length == 256:
        from basic_pitch.note_creation import model_frames_to_time
        return model_frames_to_time(n_frames_total)
    return np.arange(n_frames_total) * hop_length / sr


def candidates_to_notes(cands, hop_length):
    times = frames_to_times(max(c["offset_frame"] for c in cands) + 1, hop_length)
    notes = []
    for c in cands:
        notes.append({
            "onset": float(times[c["onset_frame"]]) - SHIFT,
            "offset": float(times[min(c["offset_frame"], len(times) - 1)]) - SHIFT,
            "pitch": c["pitch"],
            "confidence": c["confidence"],
        })
    return [n for n in notes if n["onset"] >= 0]


def evaluate(est, gt):
    """50ms 匹配 (Hungarian) + 窗口覆盖 recall + 杂音率."""
    from scipy.optimize import linear_sum_assignment

    n_gt, n_est = len(gt), len(est)
    cost = np.full((n_gt, n_est), np.inf)
    for i, g in enumerate(gt):
        for j, e in enumerate(est):
            if abs(e["onset"] - g["onset"]) <= 0.05 and \
               abs(midi_to_hz(e["pitch"]) - midi_to_hz(g["pitch"])) <= 50:
                cost[i, j] = 0.0
    matched = {}
    if n_gt > 0 and n_est > 0 and np.isfinite(cost).any():
        rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), -1.0, 0.0))
        for r, c in zip(rows.tolist(), cols.tolist()):
            if np.isfinite(cost[r, c]):
                matched[r] = c
    tp = len(matched)
    precision = tp / n_est if n_est else 0
    recall = tp / n_gt if n_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    # 窗口覆盖: GT 音符 (onset..offset) 期间内被同音高 est 覆盖
    win_covered = 0
    for g in gt:
        if any(g["onset"] - 0.05 <= e["onset"] <= g["offset"] + 0.05 and
               abs(midi_to_hz(e["pitch"]) - midi_to_hz(g["pitch"])) <= 50
               for e in est):
            win_covered += 1
    return {
        "n_est": n_est, "tp": tp, "precision": precision,
        "recall": recall, "f1": f1, "win_covered": win_covered,
        "fp": n_est - tp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ours", "basic"], default="ours")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--gap-list", default="0,2,4,6,8,12",
                    help="gap_tol 帧数网格 (0=不合并)")
    args = ap.parse_args()

    onset_probs, frame_probs, hop = transcribe(args.model, not args.no_cache)
    gt = load_notes(GT)
    print(f"GT: {len(gt)} 音符 (E大调)")

    from src.frame_post import process_frames_bp
    from src.merge_notes import merge_similar_notes
    base = process_frames_bp(onset_probs, frame_probs, hop_length=hop)
    print(f"BP 候选: {len(base)}")

    print(f"\n{'gap_tol':>7} | {'音符数':>6} | {'50ms-P':>7} {'50ms-R':>7} "
          f"{'50ms-F1':>7} | {'窗口R':>6} | {'杂音FP':>6}")
    print("-" * 70)
    results = {}
    for gap_s in args.gap_list.split(","):
        gap = int(gap_s)
        if gap == 0:
            cands = base
        else:
            cands = merge_similar_notes(list(base), gap_tol_frames=gap)
        est = candidates_to_notes(cands, hop)
        r = evaluate(est, gt)
        results[gap] = r
        print(f"{gap:>7} | {r['n_est']:>6} | {r['precision']*100:>6.1f}% "
              f"{r['recall']*100:>6.1f}% {r['f1']*100:>6.1f}% | "
              f"{r['win_covered']/len(gt)*100:>5.1f}% | {r['fp']:>6}")

    best = max(results, key=lambda g: results[g]["f1"])
    print(f"\n最优 gap_tol={best} (50ms-F1={results[best]['f1']*100:.1f}%)")


if __name__ == "__main__":
    main()
