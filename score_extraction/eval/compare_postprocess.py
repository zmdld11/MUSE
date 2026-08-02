"""eval/compare_postprocess.py — 方案2: basic-pitch 官方后处理 vs 我们的后处理.

同一模型 (basic-pitch), 同一评测集 (40首), 两种后处理:
  A. 官方: basic_pitch.inference.predict 的 note_events (训练配套后处理)
  B. 我们: raw probs → frame_post.process_frames → note_post.refine_notes

回答: 我们的后处理是不是根本性错误?
  A >> B → 后处理设计有根本问题, 应抄 basic-pitch 逻辑
  A ≈ B → 后处理不是瓶颈, 问题在评测口径/模型
"""
import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.ERROR)

SR = 22050


def run_official(audio, tmp_path):
    """basic-pitch 官方后处理 → notes."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    import scipy.io.wavfile as wavfile
    wavfile.write(tmp_path, SR, (audio * 32767).clip(-32768, 32767).astype(np.int16))
    _, _, note_events = predict(tmp_path, model_or_model_path=ICASSP_2022_MODEL_PATH,
                                onset_threshold=0.4, frame_threshold=0.2)
    # note_events: (onset, offset, pitch, amplitude, velocities)
    return [{"onset": float(e[0]), "offset": float(e[1]), "pitch": int(e[2])}
            for e in note_events if float(e[1]) - float(e[0]) >= 1e-4]


def run_ours(audio, tmp_path):
    """我们的后处理 → notes."""
    from src.transcriber import _basic_pitch_inference
    from src.frame_post import process_frames
    from src.note_post import refine_notes
    import scipy.io.wavfile as wavfile
    wavfile.write(tmp_path, SR, (audio * 32767).clip(-32768, 32767).astype(np.int16))
    res = _basic_pitch_inference(tmp_path)
    hop = res.get("hop_length", 256)
    sr = res.get("sr", SR)
    return refine_notes(process_frames(res["onset_probs"], res["frame_probs"],
                                       hop_length=hop, sr=sr),
                        audio, sr=sr, hop_length=hop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    from eval.dataset import load, sample_midis
    from eval.metrics import note_f1, note_offset_f1

    tmp = os.path.join(os.path.dirname(__file__), "reports", "_cmp2.wav")
    midis = sample_midis(args.n, seed=42)

    agg = {"official": {"f1": [], "offset": [], "est": []},
           "ours": {"f1": [], "offset": [], "est": []}}

    for i, mid in enumerate(midis, 1):
        gt = load(mid)
        audio = gt["audio"]

        off_notes = run_official(audio, tmp)
        ours_notes = run_ours(audio, tmp)

        for tag, notes in [("official", off_notes), ("ours", ours_notes)]:
            p, r, f = note_f1(gt["intervals"], gt["pitches"], notes)
            op_, or_, of = note_offset_f1(gt["intervals"], gt["pitches"], notes)
            agg[tag]["f1"].append(f)
            agg[tag]["offset"].append(of)
            agg[tag]["est"].append(len(notes))

        if i <= 3:
            name = os.path.basename(mid)[:30]
            print(f"[{i}] {name}: official_f1={agg['official']['f1'][-1]:.3f} "
                  f"ours_f1={agg['ours']['f1'][-1]:.3f} "
                  f"(est: {len(off_notes)}/{len(ours_notes)})")

    print(f"\n=== 方案2: 后处理对比 ({args.n} 首) ===")
    for tag in ["official", "ours"]:
        f1s = agg[tag]["f1"]
        ofs = agg[tag]["offset"]
        ests = agg[tag]["est"]
        print(f"  {tag:10s}: note_f1={np.mean(f1s):.4f}  offset_f1={np.mean(ofs):.4f}  "
              f"avg_est={np.mean(ests):.0f}")

    o = np.mean(agg["official"]["f1"])
    u = np.mean(agg["ours"]["f1"])
    print(f"\n  官方 vs 我们: {o:.4f} vs {u:.4f} "
          f"({(o-u)/u*100:+.0f}% 相对差距)")
    if o > u * 1.15:
        print("  → 官方明显更好: 我们的后处理是根本性错误, 应抄 basic-pitch 逻辑")
    elif abs(o - u) < u * 0.15:
        print("  → 差距不大: 后处理不是瓶颈, 问题在模型/评测口径")
    else:
        print("  → 我们更好: 我们的后处理没问题, 模型才是瓶颈")


if __name__ == "__main__":
    main()
