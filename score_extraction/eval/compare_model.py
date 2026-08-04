"""eval/compare_model.py — 真实录音: 跨模型对比 (ours/basic-pitch/Kong2021).

统一评估: 转录 → 音符 → 与 E 大调 GT 匹配 (网格搜索最优 shift).
指标: 窗口覆盖 recall / 50ms P-R-F1 / 杂音数.

用法:
  python -m eval.compare_model --model kong     # Kong 2021 (piano_transcription_inference)
  python -m eval.compare_model --model basic    # basic-pitch (官方)
  python -m eval.compare_model --model ours     # 自训模型
  python -m eval.compare_model --model all      # 全部对比
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


def transcribe_kong(audio_path):
    """Kong 2021: 返回音符列表 [{onset, offset, pitch, confidence}].

    权重默认路径 ~/piano_transcription_inference_data/note_F1=...pth
    (2026-08-04 已手动代理下载).
    """
    from piano_transcription_inference import PianoTranscription, sample_rate
    import librosa
    transcriber = PianoTranscription(device="cuda")
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    print(f"[kong] 音频 {len(audio)/sample_rate:.1f}s, 转写中...")
    midi_out = os.path.join(CACHE_DIR, "kong_output.mid")
    result = transcriber.transcribe(audio, midi_out)
    events = result["est_note_events"]
    notes = [{"onset": float(e["onset_time"]), "offset": float(e["offset_time"]),
              "pitch": int(e["midi_note"]), "confidence": float(e["velocity"])}
             for e in events]
    print(f"[kong] {len(notes)} 音符")
    return notes


def transcribe_basic(audio_path):
    from eval.eval import _transcribe_basic
    import tempfile, librosa
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    tmp = os.path.join(CACHE_DIR, "_tmp_model_cmp.wav")
    import scipy.io.wavfile as wf
    wf.write(tmp, sr, (audio * 32767).clip(-32768, 32767).astype(np.int16))
    return _transcribe_basic(tmp)


def transcribe_ours(audio_path):
    from eval.eval import _transcribe_ours
    import tempfile, librosa
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    tmp = os.path.join(CACHE_DIR, "_tmp_model_cmp.wav")
    import scipy.io.wavfile as wf
    wf.write(tmp, sr, (audio * 32767).clip(-32768, 32767).astype(np.int16))
    from src.transcriber import _load_model
    return _transcribe_ours(_load_model(), tmp)


def notes_from_frames(result, shift):
    """BP 风格: frame_probs → 候选 → 转秒 (带 shift)."""
    from src.frame_post import process_frames_bp
    from eval.compare_merge import frames_to_times
    cands = process_frames_bp(result["onset_probs"], result["frame_probs"],
                              hop_length=result["hop_length"])
    times = frames_to_times(max(c["offset_frame"] for c in cands) + 1,
                            result["hop_length"])
    notes = []
    for c in cands:
        on = times[c["onset_frame"]] - shift
        if on < 0:
            continue
        notes.append({"onset": on,
                      "offset": times[min(c["offset_frame"], len(times) - 1)] - shift,
                      "pitch": c["pitch"], "confidence": c["confidence"]})
    return notes


def evaluate(est, gt):
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
    p = tp / n_est if n_est else 0
    r = tp / n_gt if n_gt else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    win = sum(1 for g in gt
              if any(g["onset"] - 0.05 <= e["onset"] <= g["offset"] + 0.05 and
                     abs(midi_to_hz(e["pitch"]) - midi_to_hz(g["pitch"])) <= 50
                     for e in est))
    return {"n_est": n_est, "tp": tp, "p": p, "r": r, "f1": f1,
            "win": win / n_gt if n_gt else 0, "fp": n_est - tp}


def best_shift(est_fn, gt, shifts):
    """网格搜索最优 shift (以 50ms-F1 为目标)."""
    best = None
    for s in shifts:
        est = est_fn(s)
        r = evaluate(est, gt)
        if best is None or r["f1"] > best[1]["f1"]:
            best = (s, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["kong", "basic", "ours", "all"],
                    default="all")
    args = ap.parse_args()

    gt = load_notes(GT)
    print(f"GT: {len(gt)} 音符 (E大调)")

    models = ["kong", "basic", "ours"] if args.model == "all" else [args.model]
    results = {}
    for m in models:
        print(f"\n=== 模型: {m} ===")
        if m == "kong":
            raw = transcribe_kong(AUDIO)
            # Kong 输出是绝对秒, 直接测 shift 网格 (其 onset 延迟可能不同于 BP)
            def est_fn(s, raw=raw):
                return [{"onset": n["onset"] - s, "offset": n["offset"] - s,
                         "pitch": n["pitch"], "confidence": n["confidence"]}
                        for n in raw if n["onset"] - s >= 0]
        else:
            result = transcribe_basic(AUDIO) if m == "basic" else transcribe_ours(AUDIO)
            def est_fn(s, result=result):
                return notes_from_frames(result, s)
        s, r = best_shift(est_fn, gt, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        results[m] = (s, r)
        print(f"  最优 shift={s*1000:.0f}ms → n={r['n_est']} P={r['p']*100:.1f}% "
              f"R={r['r']*100:.1f}% F1={r['f1']*100:.1f}% 窗口R={r['win']*100:.1f}% "
              f"FP={r['fp']}")

    print(f"\n{'='*75}")
    print(f"{'模型':>8} | {'shift':>6} | {'音符数':>6} | {'50ms-P':>6} {'50ms-R':>6} "
          f"{'F1':>6} | {'窗口R':>6} | {'FP':>5}")
    print("-" * 75)
    for m, (s, r) in results.items():
        print(f"{m:>8} | {s*1000:>5.0f}ms | {r['n_est']:>6} | {r['p']*100:>5.1f}% "
              f"{r['r']*100:>5.1f}% {r['f1']*100:>5.1f}% | {r['win']*100:>5.1f}% "
              f"| {r['fp']:>5}")


if __name__ == "__main__":
    main()
