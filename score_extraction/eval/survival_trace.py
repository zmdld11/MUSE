"""eval/survival_trace.py — GT 音符逐层存活追踪.

对每个 GT 音符, 检查它在后处理每一层的存活情况:
  层0 模型输出      : onset/frame 概率是否激活 (≥0.3)
  层1 HMM平滑       : 平滑后概率是否 ≥0.3
  层2 二值化        : 该音高在该帧附近是否被点亮
  层3 连通域        : 是否有候选音符覆盖 (同音高, onset/offset 邻近)
  层4 长度过滤      : 候选是否 ≥min_frames
  层5 onset验证     : 是否通过
  层6 note_post     : 最终音符列表是否匹配

统计每层丢失的 GT 音符数 → 定位"CE 缺的 G"在哪一层丢.
用法: python -m eval.survival_trace [--n 10] [--model ours]
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
ONSET_TOL_FRAMES = 3   # 候选 onset 需在 GT onset ±3 帧内
PITCH_TOL_CENTS = 50


def midi_to_hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi, dtype=np.float64) - 69) / 12.0)


def trace_song(gt, res, audio):
    """对单首曲目逐层追踪每个 GT 音符的存活."""
    from src.frame_post import (_hmm_smooth, _adaptive_threshold_per_register,
                                _label_connected_components, _max_polyphony_filter,
                                _filter_by_length, _verify_onsets)
    from src.note_post import _prune_harmonics, _merge_duplicates

    hop = res.get("hop_length", 512)
    sr = res.get("sr", SR)
    frame_sec = hop / sr
    fp, op = res["frame_probs"], res["onset_probs"]

    # 层0/1: 模型原始 + HMM 平滑后的 GT 音符概率
    smoothed = _hmm_smooth(fp)

    # 层2: 二值化
    binary = _adaptive_threshold_per_register(smoothed)

    # 层3: 连通域候选
    cands = _label_connected_components(binary)
    cands = _max_polyphony_filter(cands, max_per_frame=8)

    # 层4: 长度过滤
    cands_len = _filter_by_length([dict(c) for c in cands], hop, sr)

    # 层5: onset 验证
    cands_ver = _verify_onsets([dict(c) for c in cands_len], op)

    # 层6: note_post 全链
    from src.note_post import refine_notes
    notes_final = refine_notes([dict(c) for c in cands_ver], audio, sr=sr, hop_length=hop)

    # 每层按 (onset_frame, pitch) 索引候选
    def idx(cands_list):
        d = defaultdict(list)
        for c in cands_list:
            d[c["pitch"]].append(c)
        return d

    cands3 = idx(cands)
    cands4 = idx(cands_len)
    cands5 = idx(cands_ver)

    # 最终音符按音高索引
    final_by_pitch = defaultdict(list)
    for n in notes_final:
        final_by_pitch[int(n["pitch"])].append(n)

    layer_hits = defaultdict(int)   # 每层存活的 GT 数
    layer_deaths = defaultdict(int)  # 每层首次死亡的 GT 数
    layer_total = defaultdict(int)

    n_gt = len(gt["intervals"])
    death_at = defaultdict(int)

    for onset, offset, pitch in zip(gt["intervals"][:, 0], gt["intervals"][:, 1], gt["pitches"]):
        bin_idx = pitch - 21
        if bin_idx < 0 or bin_idx >= fp.shape[1]:
            death_at["out_of_range"] += 1
            continue
        onset_frame = int(onset / frame_sec)
        if onset_frame >= fp.shape[0]:
            death_at["out_of_range"] += 1
            continue

        # 层0: 模型原始概率
        raw_peak = fp[max(0, onset_frame - 2):onset_frame + 3, bin_idx].max() \
            if onset_frame < fp.shape[0] else 0.0
        # 层1: HMM 平滑后
        sm_peak = smoothed[max(0, onset_frame - 2):onset_frame + 3, bin_idx].max() \
            if onset_frame < smoothed.shape[0] else 0.0
        # 层2: 二值化
        bin_hit = binary[max(0, onset_frame - 2):onset_frame + 3, bin_idx].any() \
            if onset_frame < binary.shape[0] else False

        # 层3: 连通域候选 (同音高, onset 邻近)
        c3 = [c for c in cands3.get(pitch, [])
              if abs(c["onset_frame"] - onset_frame) <= ONSET_TOL_FRAMES]
        # 层4: 长度过滤后
        c4 = [c for c in cands4.get(pitch, [])
              if abs(c["onset_frame"] - onset_frame) <= ONSET_TOL_FRAMES]
        # 层5: onset 验证后
        c5 = [c for c in cands5.get(pitch, [])
              if abs(c["onset_frame"] - onset_frame) <= ONSET_TOL_FRAMES]
        # 层6: 最终音符
        f6 = [n for n in final_by_pitch.get(int(pitch), [])
              if abs(n["onset"] - onset) <= 0.05]

        # 确定首死层
        if raw_peak < 0.3:
            death_at["L0_模型没激活"] += 1
        elif sm_peak < 0.3:
            death_at["L1_HMM平滑杀"] += 1
        elif not bin_hit:
            death_at["L2_二值化杀"] += 1
        elif not c3:
            death_at["L3_连通域丢"] += 1
        elif not c4:
            death_at["L4_长度过滤杀"] += 1
        elif not c5:
            death_at["L5_onset验证杀"] += 1
        elif not f6:
            death_at["L6_note_post杀"] += 1
        else:
            death_at["存活"] += 1

    return death_at, n_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", choices=["ours", "basic"], default="ours")
    args = ap.parse_args()

    from eval.dataset import load, sample_midis
    from src.transcriber import _load_model, _ours_inference, _basic_pitch_inference
    import scipy.io.wavfile as wavfile

    total = defaultdict(int)
    total_gt = 0
    tmp = os.path.join(os.path.dirname(__file__), "reports", "_trace.wav")
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

        deaths, n_gt = trace_song(gt, res, audio)
        total_gt += n_gt
        for k, v in deaths.items():
            total[k] += v
        if i % 10 == 0:
            print(f"  [{i}/{args.n}] 完成")

    print(f"\n=== GT 音符逐层存活追踪 ({args.model}, {args.n} 首, GT 总数 {total_gt}) ===")
    order = ["存活", "L6_note_post杀", "L5_onset验证杀", "L4_长度过滤杀",
             "L3_连通域丢", "L2_二值化杀", "L1_HMM平滑杀", "L0_模型没激活",
             "out_of_range"]
    for k in order:
        if k in total:
            v = total[k]
            print(f"  {k:20s}: {v:6d} ({v/total_gt*100:5.1f}%)")

    # 累计存活率
    alive = total["存活"]
    print(f"\n  最终存活: {alive}/{total_gt} ({alive/total_gt*100:.1f}%)")
    print(f"  累计死亡: {total_gt - alive}/{total_gt} ({(total_gt-alive)/total_gt*100:.1f}%)")


if __name__ == "__main__":
    main()
