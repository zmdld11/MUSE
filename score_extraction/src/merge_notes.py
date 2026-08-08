"""A1 碎音合并 (2026-08-02).

真实钢琴 frame probs 抖动 → BP 后处理把一个真实音符切成多个同音高碎音
(真实录音诊断: 平均 4.7 个 est 音符/GT 音符, 杂音率 57.5%).
merge_similar_notes 把同音高、相邻 (gap ≤ tol) 的碎音合并成一个完整音符,
直接减少碎音数, recall/precision 双升.

合并规则:
  1. 按 (pitch, onset_frame) 排序
  2. 同音高且 gap = 后.onset - 前.offset ≤ gap_tol_frames → 合并
     (gap 可为负 = 重叠, 也合并)
  3. 合并后: onset = 前.onset, offset = max(前.offset, 后.offset),
     confidence 累加 (面积代理)
"""
import logging

logger = logging.getLogger(__name__)


def merge_similar_notes(candidates: list[dict], gap_tol_frames: int = 4) -> list[dict]:
    """同音高相邻碎音合并.

    Args:
        candidates: BP 后处理候选 (keys: onset_frame, offset_frame, pitch, ...)
        gap_tol_frames: 允许的最大间隔帧数. hop=512/sr=22050 时 4 帧 ≈ 93ms.

    Returns:
        合并后的候选列表 (保持按 pitch/onset 有序).
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: (c["pitch"], c["onset_frame"]))
    merged: list[dict] = []

    for c in ordered:
        last = merged[-1] if merged else None
        if (last is not None and last["pitch"] == c["pitch"]
                and c["onset_frame"] - last["offset_frame"] <= gap_tol_frames):
            last["offset_frame"] = max(last["offset_frame"], c["offset_frame"])
            last["confidence"] = last.get("confidence", 0) + c.get("confidence", 0)
        else:
            merged.append(dict(c))

    n_before = len(candidates)
    if n_before != len(merged):
        logger.info(f"  Merge: {n_before} → {len(merged)} candidates "
                    f"(-{(1 - len(merged) / n_before) * 100:.0f}%)")
    return merged


def merge_broken_notes(candidates: list[dict],
                       gap_max_sec: float = 0.15,
                       min_total_sec: float = 0.4,
                       onset_prob_thresh: float = 0.3,
                       min_gap_frames: int = 2) -> list[dict]:
    """只合并"长音碎裂"的同音高相邻音符 (2026-08-06 诊断).

    真实录音诊断 (rawflac, VER3.2):
      - 长音被切开的 gap 处 frame prob 掉到 0.14-0.26 (远低于 0.5), 触发 BP 断开;
      - 71% 的碎裂 gap 处 onset prob < 0.2 (无音头, 纯能量抖动);
      - 26% 的碎裂 gap 处 onset prob > 0.5 但 gap 仅 1-2 帧 (23-46ms),
        物理上不可能是真实击键, 属于假 onset.

    合并条件 (全部满足):
      1. 同音高, 0 < gap_sec ≤ gap_max_sec
      2. 合并后总时长 ≥ min_total_sec (避免吞掉快速重复音)
      3. 无音头证据: 后段 onset_prob < onset_prob_thresh
         或 gap_frames ≤ min_gap_frames (1-2 帧 gap 必为假 onset)

    合并后: onset 取前段 (保留真实音头), offset 取后段, 中间 gap 填满.
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: (c["pitch"], c["onset_frame"]))
    merged: list[dict] = []

    for c in ordered:
        last = merged[-1] if merged else None
        if last is not None and last["pitch"] == c["pitch"]:
            gap_sec = c["onset_time"] - last["offset_time"]
            gap_frames = c["onset_frame"] - last["offset_frame"]
            total_sec = c["offset_time"] - last["onset_time"]
            onset_prob = c.get("onset_prob", 0.0)
            no_onset = onset_prob < onset_prob_thresh or gap_frames <= min_gap_frames
            if (0.0 < gap_sec <= gap_max_sec
                    and total_sec >= min_total_sec
                    and no_onset):
                last["offset_frame"] = max(last["offset_frame"], c["offset_frame"])
                last["offset_time"] = max(last["offset_time"], c["offset_time"])
                last["confidence"] = max(last.get("confidence", 0.0),
                                         c.get("confidence", 0.0))
                continue
        merged.append(dict(c))

    n_before = len(candidates)
    if n_before != len(merged):
        logger.info(
            f"  Merge-broken: {n_before} → {len(merged)} candidates "
            f"(-{(1 - len(merged) / n_before) * 100:.0f}%), 仅长音碎裂"
        )
    return merged