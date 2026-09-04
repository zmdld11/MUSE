"""帧概率 → 音符（M1 解码，2026-08-31）。

规则（立项 md §5）：
- onset 局部峰值 > 0.5，最小间隔 50ms（贪心按概率保留强者）
- 终点 = min(offset 概率首个 >0.5 的帧, 连续 80ms unvoiced 起点, 下一 onset)
- 音高 = 音符窗内 pitch 类（去 class 0）逐帧 argmax 的中位
- 输出 [{onset, offset, pitch}]（秒；onset/offset 键名与冻结口径 evaluate 一致）
"""
from __future__ import annotations

import numpy as np

HOP_SEC = 220 / 22050
PITCH_LO, PITCH_HI = 40, 84


def pick_peaks(prob: np.ndarray, th: float = 0.5,
               min_gap_frames: int = 5) -> list[int]:
    """局部峰值 + 贪心间隔抑制（gap 内保留概率更高者）。"""
    T = len(prob)
    cand = [i for i in range(T)
            if prob[i] >= th
            and (i == 0 or prob[i] >= prob[i - 1])
            and (i == T - 1 or prob[i] > prob[i + 1])]
    cand.sort(key=lambda i: -prob[i])
    kept: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_gap_frames for j in kept):
            kept.append(i)
    return sorted(kept)


def decode_notes(onset_prob: np.ndarray, offset_prob: np.ndarray,
                 pitch_logits: np.ndarray, hop_sec: float = HOP_SEC,
                 onset_th: float = 0.5, offset_th: float = 0.5,
                 min_dur_sec: float = 0.03,
                 unvoiced_run_sec: float = 0.08) -> list[dict]:
    """三路输出 → 音符表。pitch_logits (T, 46)。

    unvoiced_run_sec：连续 unvoiced 多久截尾（S2v2-10 起参数化，默认 0.08
    = 历史 M1 行为不变）。
    """
    T = len(onset_prob)
    if T == 0:
        return []
    voiced_cls = np.argmax(pitch_logits[:, 1:], axis=1) + 1   # 1..45
    voiced_flag = pitch_logits[:, 1:].max(axis=1) > pitch_logits[:, 0]
    peaks = pick_peaks(onset_prob, th=onset_th,
                       min_gap_frames=max(1, round(0.05 / hop_sec)))
    min_frames = max(1, round(min_dur_sec / hop_sec))
    uv_frames = max(1, round(unvoiced_run_sec / hop_sec))
    notes = []
    for k, i0 in enumerate(peaks):
        nxt = peaks[k + 1] if k + 1 < len(peaks) else T
        end = min(nxt, T)
        lo = i0 + min_frames   # 最短音长内的 offset/unvoiced 不触发截止
        # （否则连音处前一音的 offset 帧会误截新音）
        # 1) offset 概率命中
        if lo < end:
            hits = np.nonzero(offset_prob[lo:end] > offset_th)[0]
            if len(hits):
                end = min(end, lo + int(hits[0]))
            # 2) 连续 unvoiced 超过阈值 → 截止到 run 起点
            run = 0
            for j in range(lo, end):
                run = run + 1 if not voiced_flag[j] else 0
                if run >= uv_frames:
                    end = min(end, j - run + 1)
                    break
        if end - i0 < min_frames:
            continue
        seg = voiced_cls[i0:end][voiced_flag[i0:end]]
        if not len(seg):
            seg = voiced_cls[i0:min(end, i0 + 10)]
        pitch = int(np.clip(round(float(np.median(seg))) + PITCH_LO - 1,
                            PITCH_LO, PITCH_HI))
        notes.append({"onset": round(i0 * hop_sec, 4),
                      "offset": round(end * hop_sec, 4),
                      "pitch": pitch})
    return notes


def gate(notes: list[dict], lo: int = PITCH_LO, hi: int = PITCH_HI) -> list[dict]:
    return [n for n in notes if lo <= int(n["pitch"]) <= hi]


# ---------------- M3 元音感知解码（2026-09-01） ----------------

def _seg_pitch(pitch_logits: np.ndarray, i0: int, i1: int) -> int:
    """段内音高中位（voiced 类，复用 decode_notes 的口径）。"""
    voiced_cls = np.argmax(pitch_logits[:, 1:], axis=1) + 1
    voiced_flag = pitch_logits[:, 1:].max(axis=1) > pitch_logits[:, 0]
    i0, i1 = max(0, i0), min(len(voiced_cls), max(i1, i0 + 1))
    seg = voiced_cls[i0:i1][voiced_flag[i0:i1]]
    if not len(seg):
        seg = voiced_cls[i0:min(i1, i0 + 10)]
    return int(np.clip(round(float(np.median(seg))) + PITCH_LO - 1, PITCH_LO,
                       PITCH_HI))


def decode_notes_vowel(onset_prob: np.ndarray, offset_prob: np.ndarray,
                       pitch_logits: np.ndarray, vowel_prob: np.ndarray,
                       hop_sec: float = HOP_SEC, onset_th: float = 0.5,
                       offset_th: float = 0.5, vowel_th: float = 0.5,
                       min_dur_sec: float = 0.03, min_split_sec: float = 0.12,
                       lead_sec: float = 0.06,
                       unvoiced_run_sec: float = 0.08) -> list[dict]:
    """M3：基础解码 + 元音起始强制切分跨字长音。

    规则：一个音内若含元音峰（与音起点/前切点隔 ≥min_split_sec、与音终点隔
    ≥min_dur），在该元音处切分；切点 = 元音前 120ms 内 onset 头局部峰（≥0.3）
    优先，否则元音峰回退 lead_sec。切分不发明新音高，每段独立中位重估。
    """
    notes = decode_notes(onset_prob, offset_prob, pitch_logits, hop_sec,
                         onset_th, offset_th, min_dur_sec, unvoiced_run_sec)
    vpeaks = pick_peaks(vowel_prob, th=vowel_th,
                        min_gap_frames=max(1, round(0.08 / hop_sec)))
    min_split_f = max(1, round(min_split_sec / hop_sec))
    min_dur_f = max(1, round(min_dur_sec / hop_sec))
    lead_f = max(1, round(lead_sec / hop_sec))
    first_guard_f = max(1, round(0.25 / hop_sec))   # 首切点：辅音 lead 窗内
    out: list[dict] = []
    for n in notes:
        b0 = int(round(n["onset"] / hop_sec))
        b1 = int(round(n["offset"] / hop_sec))
        cuts = [b0]
        for vp in vpeaks:
            need = first_guard_f if len(cuts) == 1 else min_split_f
            if cuts[-1] + need <= vp <= b1 - min_dur_f:
                # onset 头在元音前 120ms 内有局部峰则贴合，否则元音峰回退 lead
                lo = max(cuts[-1] + 1, vp - int(0.12 / hop_sec))
                win = onset_prob[lo:vp + 1]
                cut = vp - lead_f
                if len(win):
                    j = int(np.argmax(win))
                    if win[j] >= 0.3:
                        cut = lo + j
                if cut > cuts[-1]:
                    cuts.append(cut)
        for a, b in zip(cuts, cuts[1:] + [b1]):
            if b - a >= min_dur_f:
                out.append({"onset": round(a * hop_sec, 4),
                            "offset": round(b * hop_sec, 4),
                            "pitch": _seg_pitch(pitch_logits, a, b)})
    return out
