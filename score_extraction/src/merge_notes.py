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
