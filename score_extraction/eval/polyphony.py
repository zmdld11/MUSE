"""eval/polyphony.py — 按复调度分析音符检测能力.

回答: 模型是只能测单音, 还是能测和弦?
方法: 对每个 GT 音符按其 onset 时刻的复调度 (同时活跃音符数) 分组,
      统计各组的检测 recall (最终音符列表是否匹配到 onset±50ms + 同音高).
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
HOP = {"ours": 512, "basic": 256}


def analyze(midis, model_name, out_file=None):
    from eval.dataset import load
    from src.transcriber import _load_model, _ours_inference, _basic_pitch_inference
    from src.frame_post import process_frames
    from src.note_post import refine_notes
    import scipy.io.wavfile as wavfile

    # 累计: 每个复调度桶的 (GT 音符数, 命中数)
    buckets = defaultdict(lambda: {"gt": 0, "hit": 0})
    tmp = os.path.join(os.path.dirname(__file__), "reports", "_poly.wav")

    for i, mid in enumerate(midis, 1):
        gt = load(mid)
        audio = gt["audio"]
        wavfile.write(tmp, SR, (audio * 32767).clip(-32768, 32767).astype(np.int16))

        if model_name == "ours":
            model = _load_model()
            res = _ours_inference(model, tmp)
        else:
            res = _basic_pitch_inference(tmp)
        hop = res.get("hop_length", HOP[model_name])
        sr = res.get("sr", SR)

        notes = refine_notes(process_frames(res["onset_probs"], res["frame_probs"],
                                            hop_length=hop, sr=sr),
                             audio, sr=sr, hop_length=hop)

        # 按音高索引最终音符 onset
        est_by_pitch = defaultdict(list)
        for n in notes:
            est_by_pitch[int(n["pitch"])].append(n["onset"])

        # GT 每帧复调度 (活跃音符数)
        frame_labels = gt["frame_labels"]
        polyphony_per_frame = frame_labels.sum(axis=1).astype(int)  # (T,)

        # GT 的 frame_labels 在 hop=512 网格上 (dataset 固定)
        gt_frame_sec = 512 / SR
        # 对每个 GT 音符, 找 onset 帧, 统计该帧 polyphony (用 GT 网格)
        for onset, offset, pitch in zip(gt["intervals"][:, 0], gt["intervals"][:, 1], gt["pitches"]):
            f0 = int(onset / gt_frame_sec)
            poly = int(polyphony_per_frame[f0])
            if f0 > 0:
                poly = max(poly, int(polyphony_per_frame[f0 - 1]))
            if f0 < len(polyphony_per_frame) - 1:
                poly = max(poly, int(polyphony_per_frame[f0 + 1]))
            poly = max(1, poly)

            # 该音符是否被检出: 有同音高 onset±50ms 的 est
            hit = any(abs(e - onset) <= 0.05 for e in est_by_pitch.get(int(pitch), []))
            buckets[poly]["gt"] += 1
            buckets[poly]["hit"] += hit

    print(f"\n=== {model_name}: 按复调度 (polyphony) 的音符检测 recall ===")
    print(f"{'复调度':>6s} {'GT音符':>8s} {'命中':>6s} {'Recall':>7s}")
    for poly in sorted(buckets):
        b = buckets[poly]
        r = b["hit"] / b["gt"] * 100 if b["gt"] else 0
        bar = "#" * int(r / 4)
        print(f"{poly:6d} {b['gt']:8d} {b['hit']:6d} {r:6.1f}%  {bar}")

    # 汇总: 单音(poly=1) vs 复调(poly>=2)
    mono = buckets[1]
    mono_gt = mono["gt"]; mono_hit = mono["hit"]
    chord_gt = sum(b["gt"] for p, b in buckets.items() if p >= 2)
    chord_hit = sum(b["hit"] for p, b in buckets.items() if p >= 2)
    print(f"\n  单音 (poly=1):   recall = {mono_hit/mono_gt*100:.1f}%  ({mono_hit}/{mono_gt})")
    print(f"  和弦 (poly>=2):  recall = {chord_hit/chord_gt*100:.1f}%  ({chord_hit}/{chord_gt})")
    print(f"  和弦占比: {chord_gt/(mono_gt+chord_gt)*100:.0f}% 的 GT 音符在和弦里")

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"{model_name} polyphony analysis\n")
            for poly in sorted(buckets):
                b = buckets[poly]
                r = b["hit"] / b["gt"] * 100 if b["gt"] else 0
                f.write(f"poly={poly} gt={b['gt']} hit={b['hit']} recall={r:.1f}%\n")
        print(f"已写入: {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ours", "basic"], default="ours")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from eval.dataset import sample_midis
    analyze(sample_midis(args.n, seed=42), args.model, args.out)


if __name__ == "__main__":
    main()
