"""eval/diagnose.py — 音符级漏检瓶颈诊断.

对单首曲目逐层追踪音符丢失:
  1. 模型原始输出: GT 音符在 frame_probs / onset_probs 上的激活强度
  2. 每个后处理阶段的候选音符数 (frame_post / note_post 各步)
  3. 检出的音符与 GT 的匹配质量 (onset 偏差 / 音高正确 / 漏检)

用法:
  python -m eval.diagnose --model ours --idx 0
"""
import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

SR = 22050
HOP = {"ours": 512, "basic": 256}


def _gt_midi_bins(frame_labels: np.ndarray) -> list[int]:
    """GT 活跃的 MIDI 音高集合."""
    active = set()
    for t in range(frame_labels.shape[0]):
        bins = np.nonzero(frame_labels[t])[0]
        active.update((b + 21 for b in bins))
    return sorted(active)


def analyze_song(mid_path: str, model_name: str, idx: int) -> None:
    from eval.dataset import load
    from src.transcriber import _load_model, _ours_inference, _basic_pitch_inference
    import scipy.io.wavfile as wavfile

    gt = load(mid_path)
    audio = gt["audio"]
    tmp = os.path.join(os.path.dirname(__file__), "reports", "_diag.wav")
    wavfile.write(tmp, SR, (audio * 32767).clip(-32768, 32767).astype(np.int16))

    name = os.path.basename(mid_path)[:45]
    print(f"\n{'='*70}\n[{idx}] {name}")
    print(f"  GT 音符: {len(gt['intervals'])} 首, 时长 {gt['intervals'][:,1].max():.1f}s")

    # --- 转录 ---
    if model_name == "ours":
        model = _load_model()
        res = _ours_inference(model, tmp)
    else:
        res = _basic_pitch_inference(tmp)
    onset_p, frame_p = res["onset_probs"], res["frame_probs"]
    hop = res.get("hop_length", HOP[model_name])
    frame_sec = hop / SR
    print(f"  [{model_name}] onset={onset_p.shape} frame={frame_p.shape} hop={hop}")

    # --- 分析 1: GT 音符在模型输出上的激活强度 ---
    print("\n  【第 1 层】GT 音符在模型输出的激活情况:")
    fp_gt, op_gt = _resample(onset_p, frame_p, hop, gt)
    _report_gt_activation(gt, fp_gt, op_gt, hop, frame_sec)

    # --- 分析 2: 逐层候选存活 ---
    print("\n  【第 2 层】后处理各阶段音符数:")
    _report_stage_counts(res, audio)

    # --- 分析 3: 最终音符 vs GT 匹配质量 ---
    print("\n  【第 3 层】最终音符 vs GT 匹配:")
    _report_match_quality(res, audio, gt, hop)

    # --- 分析 4: GT 音符逐个追踪各阶段存活 ---
    print("\n  【第 4 层】GT 音符在各阶段的存活率:")
    _trace_gt_notes(gt, res, audio, hop, frame_sec)


def _resample(onset_p, frame_p, hop, gt):
    """把 est frame/onset probs 重采样到 GT 网格 (与 metrics 一致)."""
    from eval.metrics import _resample_probs_to_grid
    n_ref = gt["frame_labels"].shape[0]
    fp = _resample_probs_to_grid(frame_p, est_hop=hop, n_ref_frames=n_ref)
    op = _resample_probs_to_grid(onset_p, est_hop=hop, n_ref_frames=n_ref)
    return fp, op


def _report_gt_activation(gt, fp, op, hop, frame_sec):
    """GT 音符的 onset 帧处, frame/onset 概率有多高?"""
    n_total = len(gt["intervals"])
    weak_frame = weak_onset = 0
    examples = []
    for onset, offset, pitch in zip(gt["intervals"][:, 0], gt["intervals"][:, 1], gt["pitches"]):
        bin_idx = pitch - 21
        if bin_idx < 0 or bin_idx >= fp.shape[1]:
            continue
        onset_frame = int(onset / frame_sec)
        if onset_frame >= fp.shape[0]:
            continue
        # frame 激活: onset~offset 窗口内 frame prob 的峰值
        off_frame = min(int(offset / frame_sec) + 1, fp.shape[0])
        seg = fp[onset_frame:off_frame, bin_idx]
        f_peak = seg.max() if len(seg) else 0.0
        # onset 激活: onset 帧附近 ±2 帧的最大值
        o_win = op[max(0, onset_frame - 2):onset_frame + 3, bin_idx]
        o_peak = o_win.max() if len(o_win) else 0.0
        if f_peak < 0.3:
            weak_frame += 1
        if o_peak < 0.3:
            weak_onset += 1
        if len(examples) < 5 and (f_peak < 0.3 or o_peak < 0.3):
            examples.append((pitch, round(onset, 2), round(f_peak, 3), round(o_peak, 3)))

    print(f"  GT 音符总数: {n_total}")
    print(f"  frame 激活弱 (<0.3): {weak_frame} ({weak_frame/n_total*100:.0f}%)")
    print(f"  onset 激活弱 (<0.3): {weak_onset} ({weak_onset/n_total*100:.0f}%)")
    for ex in examples:
        print(f"    例: pitch={ex[0]} onset={ex[1]}s frame_peak={ex[2]} onset_peak={ex[3]}")


def _report_stage_counts(res, audio):
    """报告 frame_post / note_post 各阶段候选数."""
    from src.frame_post import process_frames, _hmm_smooth, _adaptive_threshold_per_register, \
        _label_connected_components, _max_polyphony_filter, _filter_by_length, _verify_onsets

    hop = res.get("hop_length", 512)
    sr = res.get("sr", SR)
    fp, op = res["frame_probs"], res["onset_probs"]

    smoothed = _hmm_smooth(fp)
    binary = _adaptive_threshold_per_register(smoothed)
    print(f"  HMM+阈值二值化: active bins = {binary.sum()} ({binary.mean()*100:.2f}%)")

    candidates = _label_connected_components(binary)
    print(f"  连通域: {len(candidates)}")
    candidates = _max_polyphony_filter(candidates, max_per_frame=8)
    candidates = _filter_by_length(candidates, hop, sr)
    print(f"  max-polyphony + 长度过滤: {len(candidates)}")
    candidates = _verify_onsets(candidates, op)
    print(f"  onset 验证后: {len(candidates)}")

    # note_post 各步
    from src.note_post import refine_notes, _prune_harmonics, _merge_duplicates
    from eval.metrics import _binarize_frame_probs  # noqa
    notes = []
    for c in candidates:
        notes.append({"onset": c["onset_frame"] * hop / sr, "offset": c["offset_frame"] * hop / sr,
                      "pitch": c["pitch"], "confidence": c["confidence"], "amplitude": 0.1})
    print(f"  → note_post 输入: {len(notes)}")
    pruned = _prune_harmonics([dict(n) for n in notes])
    print(f"  谐波过滤后: {len(pruned)} (删 {len(notes)-len(pruned)})")
    merged = _merge_duplicates([dict(n) for n in pruned])
    print(f"  合并后: {len(merged)} (删 {len(pruned)-len(merged)})")


def _report_match_quality(res, audio, gt, hop):
    """最终音符 vs GT: onset 偏差分布, 音高匹配, 漏检/误检."""
    from src.frame_post import process_frames
    from src.note_post import refine_notes

    hop = res.get("hop_length", 512)
    sr = res.get("sr", SR)
    notes = refine_notes(process_frames(res["onset_probs"], res["frame_probs"],
                                        hop_length=hop, sr=sr),
                         audio, sr=sr, hop_length=hop)
    print(f"  最终音符数: {len(notes)} vs GT {len(gt['intervals'])}")

    # 用 mir_eval 匹配看 onset 偏差
    valid = [n for n in notes if n["offset"] - n["onset"] >= 1e-4]
    est_intervals = np.array([[n["onset"], n["offset"]] for n in valid])
    est_pitches = np.array([int(n["pitch"]) for n in valid])
    if len(est_intervals) == 0:
        print("  无音符! 全漏")
        return

    from mir_eval import transcription as tr
    p, r, f, _ = tr.precision_recall_f1_overlap(
        gt["intervals"], gt["pitches"], est_intervals, est_pitches,
        onset_tolerance=0.05, offset_ratio=None)
    print(f"  Note F1 (onset-only): P={p:.3f} R={r:.3f} F={f:.3f}")

    # onset 偏差分布: 对每个 GT 音符找同音高最近预测
    from collections import defaultdict
    est_by_pitch = defaultdict(list)
    for n in notes:
        est_by_pitch[int(n["pitch"])].append(n["onset"])
    all_diffs = []
    for onset, _, pitch in zip(gt["intervals"][:, 0], gt["intervals"][:, 1], gt["pitches"]):
        if pitch in est_by_pitch:
            diffs = [abs(o - onset) for o in est_by_pitch[pitch]]
            all_diffs.append(min(diffs))
    if all_diffs:
        all_diffs = np.array(all_diffs)
        print(f"  GT→最近同音高预测的 onset 偏差 (秒):")
        print(f"    ≤50ms: {(all_diffs<=0.05).mean()*100:.0f}%  "
              f"≤100ms: {(all_diffs<=0.1).mean()*100:.0f}%  "
              f"≤200ms: {(all_diffs<=0.2).mean()*100:.0f}%  "
              f"中位数: {np.median(all_diffs)*1000:.0f}ms")
        print(f"    完全没有同音高预测: {(len(gt['intervals'])-len(all_diffs))} 个 GT 音符")


def _trace_gt_notes(gt, res, audio, hop, frame_sec):
    """逐个 GT 音符追踪: 在 HMM 平滑二值化中存活? 在 onset 验证中存活?

    把 GT 音符的 onset 帧位置, 对照后处理各阶段的候选, 统计存活率.
    """
    from src.frame_post import _hmm_smooth, _adaptive_threshold_per_register, \
        _verify_onsets, _label_connected_components, _max_polyphony_filter, _filter_by_length

    sr = res.get("sr", SR)
    fp, op = res["frame_probs"], res["onset_probs"]

    # 阶段 A: HMM 平滑 + 二值化 → 哪些 GT 音符的帧被点亮?
    smoothed = _hmm_smooth(fp)
    binary = _adaptive_threshold_per_register(smoothed)

    n_total = len(gt["intervals"])
    alive_bin = 0       # GT onset 帧附近在二值图中存活
    alive_cand = 0      # 存活到候选音符列表
    alive_verify = 0    # 通过 onset 验证

    # 候选集: pitch_bin -> [(onset_frame, offset_frame)]
    from collections import defaultdict
    cand_by_pitch = defaultdict(list)
    candidates = _label_connected_components(binary)
    candidates = _max_polyphony_filter(candidates, max_per_frame=8)
    candidates = _filter_by_length(candidates, hop, sr)
    for c in candidates:
        cand_by_pitch[c["pitch"]].append(c)
    verified = _verify_onsets([dict(c) for c in candidates], op)
    verified_by_pitch = defaultdict(list)
    for c in verified:
        verified_by_pitch[c["pitch"]].append(c)

    missing_at_bin = []
    missing_at_verify = []
    for onset, _, pitch in zip(gt["intervals"][:, 0], gt["intervals"][:, 1], gt["pitches"]):
        bin_idx = pitch - 21
        if bin_idx < 0 or bin_idx >= fp.shape[1]:
            continue
        onset_frame = int(onset / frame_sec)
        if onset_frame >= fp.shape[0]:
            continue
        # 阶段 A: onset 帧 ±2 内在二值图中激活?
        win = binary[max(0, onset_frame - 2):onset_frame + 3, bin_idx]
        if win.any():
            alive_bin += 1
            # 阶段 B: 该帧附近有候选音符?
            has_cand = any(c["onset_frame"] <= onset_frame + 3 and c["offset_frame"] >= onset_frame - 3
                           for c in cand_by_pitch.get(pitch, []))
            if has_cand:
                alive_cand += 1
                # 阶段 C: 候选通过 onset 验证?
                has_ver = any(c["onset_frame"] <= onset_frame + 3 and c["offset_frame"] >= onset_frame - 3
                              for c in verified_by_pitch.get(pitch, []))
                if has_ver:
                    alive_verify += 1
                else:
                    if len(missing_at_verify) < 5:
                        missing_at_verify.append((pitch, round(onset, 2)))
            else:
                if len(missing_at_bin) < 5:
                    missing_at_bin.append((pitch, round(onset, 2)))

    print(f"  GT 音符总数: {n_total}")
    print(f"  ① 二值化后 (onset帧附近激活): {alive_bin}/{n_total} ({alive_bin/n_total*100:.0f}%)")
    print(f"  ② 有候选音符覆盖:            {alive_cand}/{n_total} ({alive_cand/n_total*100:.0f}%)")
    print(f"  ③ 通过 onset 验证:           {alive_verify}/{n_total} ({alive_verify/n_total*100:.0f}%)")
    print(f"  在②处丢失 (二值化亮但无候选) 例: {missing_at_bin}")
    print(f"  在③处丢失 (有候选但验证不过) 例: {missing_at_verify}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ours", "basic"], default="ours")
    ap.add_argument("--idx", type=int, default=0, help="采样曲目序号 (0-39)")
    ap.add_argument("--n", type=int, default=3, help="分析几首")
    args = ap.parse_args()

    from eval.dataset import sample_midis
    midis = sample_midis(40, seed=42)
    for i in range(args.idx, min(args.idx + args.n, 40)):
        analyze_song(midis[i], args.model, i)


if __name__ == "__main__":
    main()
