"""T1 和声先验层（乐理路线第一阶，2026-08-23）：和弦跟踪 + 每音一致性。

半小节窗模板匹配（maj/min/7/maj7/min7/sus4 × 12 根音 + N.C.）→ Viterbi
平滑（同和弦保持代价低、同根音换性质次之）→ 相邻同和弦合并段。音高类
按窗内持续时长加权（长持续音每窗最多计满一窗），低音区音高对根音加权。

每音一致性三级：chord_tone / diatonic / chromatic（= outlier，只标记不
修正，用户拍板 2026-08-23）。谱面和弦记号由 notation_layer 写入
MusicXML <harmony>。

文献坐标（findings 2026-08-23）：Ojima & Perraudin 2018 和弦 LM 后处理
AMT（架构参照）；McGill Billboard / CoCoPops 进程 n-gram 与王道進行统计
（Ramage 2023）列为后续增强， v1 不接语料。
"""
from __future__ import annotations

import logging
from collections import defaultdict
from fractions import Fraction

from src.notation_layer import _ql

logger = logging.getLogger(__name__)

CHORD_TYPES: dict[str, tuple[int, ...]] = {
    "":      (0, 4, 7),
    "m":     (0, 3, 7),
    "7":     (0, 4, 7, 10),
    "maj7":  (0, 4, 7, 11),
    "m7":    (0, 3, 7, 10),
    "sus4":  (0, 5, 7),
}
_QUALITIES_BY_LEN = sorted(CHORD_TYPES, key=len, reverse=True)  # maj7>m7>m>"" 匹配
STEP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
BASS_PITCH = 48          # 低于 C3 视为低音区（根音证据加权）
W_HALF = Fraction(2)     # 半小节窗（4/4）
W_CHORD_TONE = 3.0
W_DIATONIC = 1.0
W_FOREIGN = -1.0
W_ROOT_BASS = 1.5
TRANS_CHANGE = 1.2       # 换和弦代价
TRANS_QUALITY = 0.6      # 同根音换性质
CORPUS_WEIGHT = 3.0      # 语料 bigram 概率折算系数（T1.1）
TOP_K = 8                # 每窗保留候选数（Viterbi 状态剪枝）

_BIGRAM_CACHE: dict | None = None


def _load_corpus() -> dict | None:
    """CoCoPops-Billboard bigram（data/cocopops_chord_bigrams.json，T1.1）。"""
    global _BIGRAM_CACHE
    if _BIGRAM_CACHE is not None:
        return _BIGRAM_CACHE or None
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "cocopops_chord_bigrams.json")
    try:
        with open(path, encoding="utf-8") as f:
            _BIGRAM_CACHE = json.load(f)
    except OSError:
        _BIGRAM_CACHE = {}
        logger.info("[harmony] corpus bigrams missing — falling back to heuristic transitions")
    return _BIGRAM_CACHE or None


def _tonic_pc(key_str: str) -> int:
    """主音 pc（小调折到关系大调空间，与语料统计口径一致）。"""
    parts = key_str.strip().split()
    tonic = parts[0] if parts else "C"
    mode = parts[1] if len(parts) > 1 else "major"
    pc = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
          "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
          "Bb": 10, "B": 11}.get(tonic, 0)
    return pc if mode.startswith("maj") else (pc + 3) % 12


def parse_label(label: str) -> tuple[int | None, set[int] | None]:
    """'C#m7' → (根音 pc, 和弦音集)；'N.C.' → (None, None)。"""
    if label == "N.C.":
        return None, None
    for q in _QUALITIES_BY_LEN:
        if q and label.endswith(q) and len(label) > len(q):
            root = label[: -len(q)]
            if root in STEP_NAMES:
                root_pc = STEP_NAMES.index(root)
                return root_pc, {(root_pc + i) % 12 for i in CHORD_TYPES[q]}
        if not q and label in STEP_NAMES:  # 大三和弦（无后缀）
            root_pc = STEP_NAMES.index(label)
            return root_pc, {(root_pc + i) % 12 for i in CHORD_TYPES[""]}
    return None, None


def _quality_of(label: str) -> str:
    """标签的性质族（'C#m7'→'m7'；'E'→''）。"""
    for q in _QUALITIES_BY_LEN:
        if q and label.endswith(q) and len(label) > len(q):
            return q
    return ""


def _scale_of_key(key_str: str) -> set[int]:
    parts = key_str.strip().split()
    tonic = parts[0] if parts else "C"
    mode = parts[1] if len(parts) > 1 else "major"
    pc = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
          "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
          "Bb": 10, "B": 11}.get(tonic, 0)
    return MAJOR_SCALE if mode.startswith("maj") else MINOR_SCALE


def _window_weights(notes: list[dict], bpm: float, w_start: Fraction,
                    w_end: Fraction) -> tuple[dict[int, Fraction], dict[int, Fraction]]:
    """窗内音高类 → 加权时长（普通/低音两本账）。"""
    pc_dur: dict[int, Fraction] = defaultdict(Fraction)
    bass_dur: dict[int, Fraction] = defaultdict(Fraction)
    for n in notes:
        on, off = _ql(n["onset"], bpm), _ql(n["offset"], bpm)
        ov = min(off, w_end) - max(on, w_start)
        if ov <= 0:
            continue
        ov = min(ov, W_HALF)  # 长持续音每窗最多计满一窗
        pc = int(n["pitch"]) % 12
        pc_dur[pc] += ov
        if int(n["pitch"]) < BASS_PITCH:
            bass_dur[pc] += ov
    return pc_dur, bass_dur


def _emission(pc_dur, bass_dur, scale: set[int]) -> list[tuple[str, float]]:
    """所有 (根音, 性质) 候选打分，降序。窗内总时长过少 → 只有 N.C.。"""
    if sum(pc_dur.values()) < W_HALF * Fraction(1, 5):
        return [("N.C.", 1.0)]
    out = []
    for root in range(12):
        for q, ivs in CHORD_TYPES.items():
            tones = {(root + i) % 12 for i in ivs}
            score = 0.0
            for pc, dur in pc_dur.items():
                w = (W_CHORD_TONE if pc in tones
                     else W_DIATONIC if pc in scale else W_FOREIGN)
                score += w * float(dur)
            # 缺音惩罚：和弦音在窗内完全没响，每个 -1（防"虚七和弦"）
            score -= sum(1.0 for t in tones if pc_dur.get(t, 0) <= 0)
            if bass_dur.get(root, 0) > 0:
                score += W_ROOT_BASS * float(min(bass_dur[root], W_HALF))
            out.append((f"{STEP_NAMES[root]}{q}", score))
    out.append(("N.C.", -5.0))
    out.sort(key=lambda x: -x[1])
    return out


def track_chords(notes: list[dict], bpm: float, key_str: str,
                 end_ql: Fraction) -> list[dict]:
    """pooled 音符 → 和弦段列表（半小节窗 + Viterbi + 相邻合并）。

    段字段：label / start_ql / end_ql（float 四分音符位）/ bar / confidence。
    """
    scale = _scale_of_key(key_str)
    n_win = (int(end_ql / W_HALF) + 1) if end_ql > 0 else 0
    if n_win == 0:
        return []

    windows: list[list[tuple[str, float]]] = []
    for i in range(n_win):
        ws, we = Fraction(i) * W_HALF, Fraction(i + 1) * W_HALF
        pc_dur, bass_dur = _window_weights(notes, bpm, ws, we)
        windows.append(_emission(pc_dur, bass_dur, scale)[:TOP_K])

    def trans(a: str, b: str) -> float:
        if a == b:
            return 0.0
        ra, _ = parse_label(a)
        rb, _ = parse_label(b)
        if ra is not None and ra == rb:
            return TRANS_QUALITY
        # T1.1：CoCoPops-Billboard bigram 概率调制转移代价
        #（调相对度数空间，与语料统计口径一致）
        corpus = _load_corpus()
        if corpus and ra is not None and rb is not None:
            tonic = _tonic_pc(key_str)
            key_a = f"{(ra - tonic) % 12}:{_quality_of(a)}"
            key_b = f"{(rb - tonic) % 12}:{_quality_of(b)}"
            c_ab = corpus["bigram"].get(f"{key_a}>{key_b}", 0)
            c_a = corpus["unigram"].get(key_a, 0)
            if c_a > 0:
                p = c_ab / c_a
                return max(TRANS_CHANGE * (1.0 - CORPUS_WEIGHT * p), -1.0)
        return TRANS_CHANGE

    # Viterbi（代价 = -emission 累计 + 转移）
    prev: dict[str, tuple[float, str | None]] = {
        label: (-score, None) for label, score in windows[0]
    }
    backptrs: list[dict[str, tuple[float, str | None]]] = [prev]
    for cands in windows[1:]:
        cur: dict[str, tuple[float, str | None]] = {}
        for label, score in cands:
            best_prev, best_cost = None, float("inf")
            for pl, (pcost, _) in prev.items():
                cost = pcost + trans(pl, label)
                if cost < best_cost:
                    best_cost, best_prev = cost, pl
            cur[label] = (best_cost - score, best_prev)
        backptrs.append(cur)
        prev = cur

    # 回溯
    last = min(prev.items(), key=lambda kv: kv[1][0])[0]
    path = [last]
    for i in range(len(backptrs) - 1, 0, -1):
        last = backptrs[i][last][1]
        path.append(last)
    path.reverse()

    # 相邻同和弦合并 + 置信度（段起始窗的 top1/(top1+top2)）
    segments: list[dict] = []
    for i, label in enumerate(path):
        start = Fraction(i) * W_HALF
        if segments and segments[-1]["label"] == label:
            segments[-1]["end_ql"] = float(start + W_HALF)
        else:
            segments.append({
                "label": label,
                "start_ql": float(start),
                "end_ql": float(start + W_HALF),
                "_widx": i,
                "confidence": 0.0,
            })
    for seg in segments:
        cands = windows[seg.pop("_widx")]
        top, second = cands[0][1], (cands[1][1] if len(cands) > 1 else 0.0)
        seg["confidence"] = round(top / (top + second + 1e-9), 3) if top > 0 else 0.0
        seg["bar"] = int(seg["start_ql"] // 4)
    logger.info("  [harmony] %d segments: %s", len(segments),
                " ".join(s["label"] for s in segments[:16]))
    return segments


def note_role(pitch: int, ql: Fraction, segments: list[dict],
              key_str: str) -> str:
    """音高在和声语境中的角色：chord_tone / diatonic / chromatic。"""
    scale = _scale_of_key(key_str)
    pc = pitch % 12
    seg = next((s for s in segments
                if s["start_ql"] <= float(ql) < s["end_ql"]), None)
    _, tones = parse_label(seg["label"]) if seg else (None, None)
    if tones and pc in tones:
        return "chord_tone"
    return "diatonic" if pc in scale else "chromatic"


# ---------------------------------------------------------------------------
# T4 v0：AI 乐曲打分（用户创意点 2026-08-23；吃和弦轨+离群率，idea_backlog K）
# ---------------------------------------------------------------------------

ROYAL_ROAD = [(5, "M"), (7, "M"), (4, "m"), (9, "m")]      # 王道進行 IV-V-iii-vi
CANON = [(0, "M"), (7, "M"), (9, "m"), (4, "m"), (5, "M")]  # 卡农 I-V-vi-iii-IV
JUST_TWO = [(0, "M"), (9, "m"), (4, "m"), (5, "M")]  # I-vi-iii-IV（645 系逆循环）
_MAJORISH = {"", "7", "maj7", "sus4"}  # 大性质族（m/m7 为小性质族）
DIATONIC_DEGREES = {0, 2, 4, 5, 7, 9, 11}
SEVENTH_FAMILIES = {"7", "maj7", "m7"}


def _degree_seq(segments: list[dict], key_str: str) -> list[tuple[int, str]]:
    """和弦段 → 调相对度数序列（去 N.C.；按度数去重——性质抖动=同一和弦）。"""
    tonic = _tonic_pc(key_str)
    seq: list[tuple[int, str]] = []
    for seg in segments:
        root, _ = parse_label(seg["label"])
        if root is None:
            continue
        deg = (root - tonic) % 12
        if not seq or seq[-1][0] != deg:
            seq.append((deg, _quality_of(seg["label"])))
    return seq


def _count_progression(seq: list[tuple[int, str]], prog: list[tuple[int, str]]) -> int:
    """滑动窗计数（度数精确 + 大小性质容差；sus4 两可=通配）。"""
    def parity(q: str) -> str | None:
        if q == "sus4":
            return None
        return "M" if q in _MAJORISH else "m"

    n = 0
    for i in range(len(seq) - len(prog) + 1):
        window = seq[i:i + len(prog)]
        if all(d == pd and (parity(q) is None or parity(q) == pq)
               for (d, q), (pd, pq) in zip(window, prog)):
            n += 1
    return n


def score_analysis(segments: list[dict], key_str: str,
                   outlier_rate: float | None = None) -> dict:
    """乐曲分析 v0：王道/卡农检出、语料常见度、独创度、和声复杂度。"""
    corpus = _load_corpus()
    seq = _degree_seq(segments, key_str)
    n = len(seq)
    common = 0
    if corpus and n >= 2:
        for a, b in zip(seq, seq[1:]):
            ka = f"{a[0]}:{a[1]}"
            kb = f"{b[0]}:{b[1]}"
            c_a = corpus["unigram"].get(ka, 0)
            if c_a > 0 and corpus["bigram"].get(f"{ka}>{kb}", 0) / c_a >= 0.02:
                common += 1
    commonality = round(common / (n - 1), 3) if n >= 2 else 0.0
    sevenths = sum(1 for _, q in seq if q in SEVENTH_FAMILIES)
    borrowed = sum(1 for d, _ in seq if d not in DIATONIC_DEGREES)
    rr = _count_progression(seq, ROYAL_ROAD)
    ca = _count_progression(seq, CANON)
    jt = _count_progression(seq, JUST_TWO)
    return {
        "chords": n,
        "royal_road_hits": rr,
        "canon_hits": ca,
        "just_two_hits": jt,
        "commonality": commonality,            # 语料常见进行覆盖率
        "originality": round(1.0 - commonality, 3),
        "seventh_fraction": round(sevenths / n, 3) if n else 0.0,
        "borrowed_fraction": round(borrowed / n, 3) if n else 0.0,
        "outlier_note_rate": round(outlier_rate, 4) if outlier_rate is not None else None,
        "summary": (f"王道×{rr} · 卡农×{ca} · 645系×{jt} · "
                    f"语料常见 {commonality:.0%} · 独创 {1.0 - commonality:.0%}"
                    + (f" · {n} 和弦/七和弦 {sevenths / n:.0%}" if n else "")),
    }
