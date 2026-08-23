"""Layer 5 v2（阶段19）：记谱层 —— NotationScore 组装 + 单乐器谱导出。

输入 = multi_instrument 的 notes.json 数据层产物（已过 clean_notes 清洗与
准入判定）；产出 = notation/notation.json（NotationScore，忠实+量化双模式
字段）+ notation/solo/{class}.musicxml。规则依据 markdown/记谱规则v1.md：

- 量化：全局池化 onset 对齐（quantize_timing 分段偏移搜索，16 分+三连音
  候选），记谱位置 1/48 四分音符精度网格（12/48=16 分，32/48=8 分三连音）。
- 节奏简化（记谱规则v1 §3.4，治"谱脏"）：量化后三步——①同时性聚类
  （扫弦 stagger 归一为和弦）；②时值再分配（onset 可靠、offset 不可靠，
  时值由 gap/ratio 语境决定：连奏填满、断奏缩短、中间向下取）；③小节内
  单值分解（只跨小节才 tie，链 ≤2 片；持续音封顶一小节）。
- 拼谱后守护：量化域同音高同声部冲突截断/去重（数据层截断在量化后可能
  重新碰撞）。
- TAB：music21 无 TabStaff，TAB 谱用 MusicXML <technical><string>/<fret>
  手写生成（string/fret 来自 guitar_tab 的 DP），与 staff 部分合并导出。
  吉他/贝斯按移调乐器处理（written = sounding + 12，<transpose> -12）。
"""
from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction

from music21 import (chord as m21chord, clef as m21clef, instrument as m21inst,
                     interval as m21interval, key as m21key, layout, meter as m21meter,
                     note as m21note, stream, tempo as m21tempo, tie as m21tie)

from src import guitar_tab, voice_assign
from src.key_estimate import estimate_key
from src.quantize_timing import quantize_onsets

logger = logging.getLogger(__name__)

# ---- 记谱常量 ----
QL_DENOM = 12          # 量化模式记谱网格（1/12 四分 = 16 分 ∪ 8 分三连音并集；
                       # 更细的 1/48 会让 onset 落到 64 分位置→碎休止符，2026-08-23 粗化）
FAITHFUL_DENOM = 48    # 忠实模式记谱网格（1/48；更细的有理数 music21 无法导出）
DIVISIONS = 48         # 手写 TAB XML 的 divisions（每四分音符）
SNAP_TOL_QL = Fraction(1, 16)   # 量化吸附容差（= interval_16th/4，恒 1/16 四分音符）

# music21 可导出的合法时值全集（单层 tuplet 以内）；1/48 网格的任意
# k/48 都能被贪心分解成这些值之和（_legalize）
LEGAL_DURS = sorted({
    Fraction(1, 48), Fraction(1, 32), Fraction(1, 24), Fraction(1, 16),
    Fraction(1, 12), Fraction(1, 8), Fraction(1, 6), Fraction(1, 4),
    Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
    Fraction(1, 1), Fraction(3, 2), Fraction(2, 1), Fraction(3, 1), Fraction(4, 1),
})

GUITAR_CLASSES = {"acoustic_guitar", "distorted_guitar", "electric_guitar_clean",
                  "electric_guitar_muted", "guitar_harmonics"}
KEYBOARD_CLASSES = {"piano", "electric_piano"}
BASS_CLASSES = {"electric_bass", "acoustic_bass", "slap_bass", "synth_bass"}

# 标准时值全集（四分音符单位）；时值选择一律"向下取"（拖尾衰减，宁短勿拖）
ALL_DURS = sorted({
    Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3),
    Fraction(3, 4), Fraction(1, 1), Fraction(3, 2), Fraction(2, 1),
    Fraction(3, 1), Fraction(4, 1),
})
MIN_DUR = Fraction(1, 4)

# 节奏简化常量（记谱规则v1 §3.4）
SUSTAIN_CAP = Fraction(4)   # 持续音封顶一小节（4/4；用户拍板）
LEGATO_RATIO = Fraction(85, 100)  # raw 时值 ≥ gap×0.85 → 连奏填满
DETACHED_RATIO = Fraction(1, 2)   # raw 时值 < gap×0.5 → 断奏缩短
CLUSTER_SEC = 0.040         # 同时性聚类窗口（扫弦 stagger <40ms）
SEAM_SEC = 0.040            # 量化后接缝判定窗口（§3.2 遗留）

# 时值 → (type, 附点数, time-modification) —— 手写 TAB XML 用
DUR_TO_TYPE = {
    Fraction(1, 4): ("16th", 0, None),
    Fraction(1, 3): ("eighth", 0, (3, 2)),
    Fraction(1, 2): ("eighth", 0, None),
    Fraction(2, 3): ("quarter", 0, (3, 2)),
    Fraction(3, 4): ("eighth", 1, None),
    Fraction(1, 1): ("quarter", 0, None),
    Fraction(3, 2): ("quarter", 1, None),
    Fraction(2, 1): ("half", 0, None),
    Fraction(3, 1): ("half", 1, None),
    Fraction(4, 1): ("whole", 0, None),
}

STEP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SHARPED = {1, 3, 6, 8, 10}


def _ql(sec: float, bpm: float) -> Fraction:
    """秒 → 四分音符数（Fraction）。bpm 经 str() 转换避免浮点分母爆炸。"""
    return Fraction(sec).limit_denominator(20000) * Fraction(str(bpm)) / 60


def _snap(fr: Fraction, denom: int) -> Fraction:
    """吸附到 1/denom 四分音符网格。"""
    return Fraction(round(fr * denom), denom)


def _nearest_duration(dur: Fraction) -> Fraction:
    """最近标准时值（忠实路径遗留口径，量化路径已改用向下取）。"""
    return min(ALL_DURS, key=lambda d: (abs(d - dur), d))


# 休止词表（量化模式 _fill_rests 用；比 _legalize 干净——最小 1/12 兜底，
# 正常路径最小 16 分/三连 8 分，杜绝 32/64 分碎休止）
REST_VOCAB = sorted(set(ALL_DURS) | {Fraction(1, 6), Fraction(1, 12)},
                    reverse=True)
TRIPLET_DURS = {Fraction(1, 3), Fraction(2, 3)}  # 仅三连音语境允许
BINARY_DURS = [d for d in ALL_DURS if d not in TRIPLET_DURS]


def _context_vocab(onset_ql: Fraction,
                   next_onset_ql: Fraction | None) -> list[Fraction]:
    """语境词表：当前与下一同 pitch 音头都在 16 分二分网格 → 禁三连音值
    （杜绝二分位置+三连音时值产生 1/6、1/12 残留间隙）。"""
    def binary(x: Fraction) -> bool:
        return (x * 4).denominator == 1

    if binary(onset_ql) and (next_onset_ql is None or binary(next_onset_ql)):
        return BINARY_DURS
    return ALL_DURS


def _floor_duration(bound: Fraction,
                    allowed: list[Fraction] | None = None) -> Fraction:
    """最大标准单值 ≤ bound（拖尾衰减，宁短勿拖）。allowed=语境过滤后的词表。"""
    vocab = allowed if allowed is not None else ALL_DURS
    cands = [d for d in vocab if d <= bound]
    return max(cands) if cands else MIN_DUR


def _fit_fragments(onset_ql: Fraction, bound: Fraction,
                   ql_per_measure: Fraction = Fraction(4),
                   allowed: list[Fraction] | None = None) -> list[Fraction]:
    """在 onset 位置放总时值 ≤ bound 的最干净片段组合（decompose v2）。

    小节内永远单值（允许附点跨拍）；只跨小节才切，且链 ≤2 片
    （cap ≤ 一小节 ⇒ 至多跨一条小节线）。返回片段时值列表。
    """
    vocab = allowed if allowed is not None else ALL_DURS
    m_idx = onset_ql // ql_per_measure
    room_bar = (m_idx + 1) * ql_per_measure - onset_ql
    if bound <= room_bar:
        return [max(_floor_duration(bound, vocab), MIN_DUR)]
    # 跨小节：仅当 room 是标准值（能整段填到小节线，保持 tie 连续性）；
    # 三连音位置 room 非标准 → 就地单值（跨线留给下一小节的新音头）
    if room_bar in vocab:
        rem = bound - room_bar
        if rem >= MIN_DUR:
            return [room_bar, _floor_duration(rem, vocab)]
        return [room_bar]
    return [max(_floor_duration(min(bound, room_bar), vocab), MIN_DUR)]


def _decompose(onset: Fraction, frags: list[Fraction],
               ql_per_measure: Fraction = Fraction(4)) -> list[tuple[int, Fraction, Fraction]]:
    """片段时值列表 → [(小节号, 小节内偏移, 片段时值)]。"""
    out: list[tuple[int, Fraction, Fraction]] = []
    cur = onset
    for d in frags:
        m_idx = cur // ql_per_measure
        m_start = Fraction(m_idx * ql_per_measure)
        out.append((m_idx, cur - m_start, d))
        cur += d
    return out


def _tie_roles(n: int) -> list[str | None]:
    if n <= 1:
        return [None]
    return ["start"] + ["continue"] * (n - 2) + ["stop"]


def _legalize(dur: Fraction) -> list[Fraction]:
    """贪心分解为合法时值链（和恰等于 dur；dur 需为 1/48 的整数倍）。"""
    out: list[Fraction] = []
    rem = dur
    while rem > 0:
        if rem < LEGAL_DURS[0]:
            out.append(LEGAL_DURS[0])  # 舍入残渣兜底
            break
        pick = max(v for v in LEGAL_DURS if v <= rem)
        out.append(pick)
        rem -= pick
    return out


# ---------------------------------------------------------------------------
# NotationScore 组装
# ---------------------------------------------------------------------------

def _assign_track_voices(cls: str, notes: list[dict]) -> None:
    if cls in KEYBOARD_CLASSES:
        voice_assign.assign_voices(notes, "piano")
    elif cls in GUITAR_CLASSES:
        # assign_guitar_fingering 返回新 dict 列表（不原地改）；写回原对象
        # 以保持调用方按 id() 索引的 raw_onsets 有效
        for old, new in zip(notes, guitar_tab.assign_guitar_fingering(notes)):
            old.update(new)
        for n in notes:
            n.setdefault("voice", 1)
    else:
        for n in notes:
            n["voice"] = 1


def _post_snap_guard(events: list[dict]) -> dict:
    """量化域守护：同音高同声部同 onset 去重保前（聚类吸附可能压同格）。

    时值重叠在新管线下由 _reassign_durations 的 gap 约束构造性消除，
    无需在此截断。events 需已按 onset 排序。
    """
    dropped = 0
    seen: set[tuple[int, int, Fraction]] = set()
    kept: list[dict] = []
    for ev in events:
        key = (ev["pitch"], ev["voice"], ev["_onset_ql"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(ev)
    events[:] = kept
    return {"post_snap_dropped": dropped}


def _cluster_simultaneity(events: list[dict]) -> int:
    """节奏简化 Step1 同时性聚类：原始 onset 距簇首 <40ms 的不同音高归到
    最早音头（扫弦 stagger 不再摊成分解琶音）。同音高绝不并入（保护网格
    上的真实再触发，A1 教训）。返回归位计数。
    """
    order = sorted(range(len(events)),
                   key=lambda i: (events[i]["onset_sec"], events[i]["pitch"]))
    merged = 0
    i = 0
    while i < len(order):
        j = i + 1
        cluster = [order[i]]
        while (j < len(order)
               and events[order[j]]["onset_sec"]
               - events[cluster[0]]["onset_sec"] < CLUSTER_SEC):
            cluster.append(order[j])
            j += 1
        if len(cluster) >= 2:
            seen = {events[cluster[0]]["pitch"]}
            anchor = events[cluster[0]]
            for k in cluster[1:]:
                e = events[k]
                if e["pitch"] in seen:
                    continue  # 同音高保持原位（真实再触发）
                e["_onset_ql"] = anchor["_onset_ql"]
                seen.add(e["pitch"])
                merged += 1
        i = j
    if merged:
        events.sort(key=lambda e: (e["_onset_ql"], e["pitch"]))
    return merged


def _merge_quantized_seams(events: list[dict]) -> int:
    """节奏简化接缝合并（规则v1 §3.2 遗留的"量化后判定"）：同音高相邻对，
    后音未吸附网格（rubato）且距前音原始 offset <40ms → 分段接缝，并入
    前音（前音 raw 尾延至后音尾）。kyomu 实测仅 4 例，理论完备性。
    """
    by_pitch: dict[int, list[dict]] = defaultdict(list)
    for ev in events:
        by_pitch[ev["pitch"]].append(ev)
    merged = 0
    for group in by_pitch.values():
        group.sort(key=lambda e: e["_onset_ql"])
        drop = set()
        for prev, nxt in zip(group, group[1:]):
            if id(nxt) in drop or not nxt["rubato"]:
                continue
            gap = nxt["onset_sec"] - prev["offset_sec"]
            if 0 <= gap < SEAM_SEC:
                prev["offset_sec"] = max(prev["offset_sec"], nxt["offset_sec"])
                drop.add(id(nxt))
                merged += 1
        if drop:
            group[:] = [e for e in group if id(e) not in drop]
    if merged:
        events[:] = [e for g in by_pitch.values() for e in g]
        events.sort(key=lambda e: (e["_onset_ql"], e["pitch"]))
    return merged


def _ensure_bar_room(events: list[dict],
                     ql_per_measure: Fraction = Fraction(4)) -> int:
    """小节末尾余量不足一个最小值（16 分）的 onset 前吸附到小节线。

    否则 MIN_DUR 下限会令音符跨线溢出小节，music21 makeNotation 的
    append 定位漂移会炸（Measure offset 分母 6 症状）。仅三连音位置
    触发，位移 ≤ 11/48 四分音符。
    """
    shifted = 0
    for ev in events:
        onset = ev["_onset_ql"]
        m_idx = onset // ql_per_measure
        room = (m_idx + 1) * ql_per_measure - onset
        if room < MIN_DUR:
            ev["_onset_ql"] = (m_idx + 1) * ql_per_measure
            shifted += 1
    if shifted:
        events.sort(key=lambda e: (e["_onset_ql"], e["pitch"]))
    return shifted


def _reassign_durations(events: list[dict], bpm: float,
                        ql_per_measure: Fraction = Fraction(4)) -> dict:
    """节奏简化 Step2 时值再分配：onset 可靠、offset 不可靠 → 时值由
    语境决定（记谱规则v1 §3.4）：

    - 连奏（raw ≥ gap×0.85，实测 43%）：填满——≤ min(gap, 一小节) 的
      最干净片段组合（跨小节才 tie）；
    - 断奏（raw < gap×0.5，实测 46%）：缩短——≤ min(raw, 小节余量) 的
      最大标准单值，剩余交休止符（拖尾消失）；
    - 中间态：≤ min(raw, 一小节) 向下取。
    """
    stats = {"legato": 0, "detached": 0, "middle": 0}
    shifted = _ensure_bar_room(events, ql_per_measure)
    if shifted:
        logger.info("  [notation] bar-room shift: %d onsets", shifted)
    streams: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ev in events:
        streams[(ev["pitch"], ev["voice"])].append(ev)
    for group in streams.values():
        group.sort(key=lambda e: e["_onset_ql"])
        for ev, nxt in zip(group, group[1:] + [None]):
            raw_ql = _ql(ev["offset_sec"] - ev["onset_sec"], bpm)
            next_ql = nxt["_onset_ql"] if nxt is not None else None
            vocab = _context_vocab(ev["_onset_ql"], next_ql)
            if nxt is not None:
                gap_ql = nxt["_onset_ql"] - ev["_onset_ql"]
                if gap_ql <= 0:
                    gap_ql = MIN_DUR
                ratio = raw_ql / gap_ql
            else:
                gap_ql, ratio = None, LEGATO_RATIO  # 末音：按可填满处理但受 cap
            if ratio >= LEGATO_RATIO:
                stats["legato"] += 1
                bound = min(gap_ql, SUSTAIN_CAP) if gap_ql else SUSTAIN_CAP
                ev["_frags_ql"] = _fit_fragments(ev["_onset_ql"], bound,
                                                 ql_per_measure, vocab)
            elif ratio < DETACHED_RATIO:
                stats["detached"] += 1
                m_idx = ev["_onset_ql"] // ql_per_measure
                room_bar = (m_idx + 1) * ql_per_measure - ev["_onset_ql"]
                ev["_frags_ql"] = [_floor_duration(min(raw_ql, room_bar), vocab)]
            else:
                stats["middle"] += 1
                ev["_frags_ql"] = _fit_fragments(
                    ev["_onset_ql"], min(raw_ql, SUSTAIN_CAP),
                    ql_per_measure, vocab)
    return stats


def build_track_events(cls: str, notes: list[dict], raw_onsets: dict[int, float],
                       bpm: float) -> list[dict]:
    """量化后的音笔记谱事件（含 tie 片段与弦品）。

    管线：量化吸附 → 同时性聚类 → 量化域守护 → 接缝合并 → 时值再分配
    → 小节内单值分解（记谱规则v1 §3.4 节奏简化）。
    """
    _assign_track_voices(cls, notes)
    events = []
    for n in notes:
        onset_ql = _snap(_ql(n["onset"], bpm), QL_DENOM)
        residual = abs(_ql(raw_onsets[id(n)], bpm) - onset_ql)
        events.append({
            "pitch": int(n["pitch"]),
            "voice": int(n.get("voice", 1)),
            "velocity": n.get("velocity", 100),
            "onset_sec": round(raw_onsets[id(n)], 4),
            "offset_sec": round(n["offset"], 4),
            "_onset_ql": onset_ql,
            "rubato": bool(residual > SNAP_TOL_QL),
            "quant_confidence": round(max(0.0, 1.0 - float(residual / SNAP_TOL_QL)), 3),
            "string": n.get("string"),
            "fret": n.get("fret"),
        })
    events.sort(key=lambda e: (e["_onset_ql"], e["pitch"]))

    clustered = _cluster_simultaneity(events)
    guard = _post_snap_guard(events)
    seams = _merge_quantized_seams(events)
    from src.rhythm_prior import apply_rhythm_prior  # T2 节奏模板对撞（默认关）
    prior_bars = apply_rhythm_prior(events)
    durations = _reassign_durations(events, bpm)
    logger.info("  [notation] %s simplify: cluster=%d guard=%s seam=%d prior=%d dur=%s",
                cls, clustered, guard, seams, prior_bars, durations)

    for ev in events:
        frags = _decompose(ev["_onset_ql"], ev.pop("_frags_ql"))
        roles = _tie_roles(len(frags))
        ev["bar"] = int(ev["_onset_ql"] // 4)
        ev["beat_in_bar"] = float(ev["_onset_ql"] - Fraction(ev["bar"]) * 4)
        ev["frags"] = [
            {"bar": mi, "offset": float(off), "dur": str(d), "tie": role}
            for (mi, off, d), role in zip(frags, roles)
        ]
        ev["tie"] = roles[0]
        del ev["_onset_ql"]
    # chord 分组：同轨同声部同 onset
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for ev in events:
        groups[(ev["voice"], ev["bar"], ev["beat_in_bar"])].append(ev)
    cid = 0
    for members in groups.values():
        if len(members) >= 2:
            cid += 1
            for ev in members:
                ev["chord_id"] = cid
    for ev in events:
        ev.setdefault("chord_id", None)
    return events


# ---------------------------------------------------------------------------
# music21 单乐器谱组装
# ---------------------------------------------------------------------------

def _m21_instrument(cls: str):
    if cls == "acoustic_guitar":
        return m21inst.AcousticGuitar()
    if cls in GUITAR_CLASSES:
        return m21inst.ElectricGuitar()
    if cls in BASS_CLASSES:
        return m21inst.ElectricBass()
    if cls == "piano":
        return m21inst.Piano()
    if cls == "electric_piano":
        return m21inst.ElectricPiano()
    if cls == "melody":
        return m21inst.Vocalist()
    inst = m21inst.Instrument()
    inst.instrumentName = cls
    return inst


def _transpose_semitones(cls: str) -> int:
    """吉他/贝斯 = 移调乐器（记谱比实音高八度）。"""
    return 12 if (cls in GUITAR_CLASSES or cls in BASS_CLASSES) else 0


def _rest_pieces(gap: Fraction, vocab: list[Fraction] | None) -> list[Fraction]:
    """间隙 → 休止时值链。vocab=None（忠实）→ _legalize 细词表。

    量化模式：词表去掉 1/12 后做最少片数 DP（5/12 应拆 1/4+1/6 而非
    1/3+1/12——贪心会选后者）；仅 gap 本身就是 1/12 时兜底单片。
    """
    if vocab is None:
        return _legalize(gap)
    unit = Fraction(1, 12)
    k = gap / unit
    if k.denominator != 1 or k <= 0:  # 非 1/12 整数倍（理论不发生）
        out = []
        rem = gap
        while rem > 0:
            pick = max((v for v in vocab if v <= rem), default=None)
            if pick is None:
                out.append(rem)
                break
            out.append(pick)
            rem -= pick
        return out
    k = int(k)
    pieces = sorted((v for v in vocab if v != unit and v <= gap), reverse=True)
    dp: list[list[int] | None] = [None] * (k + 1)
    dp[0] = []
    for i in range(1, k + 1):
        for p in pieces:
            u = int(p / unit)
            if u <= i and dp[i - u] is not None:
                cand = [p] + dp[i - u]
                if dp[i] is None or len(cand) < len(dp[i]):
                    dp[i] = cand
    if dp[k] is not None:
        return dp[k]
    return [gap]  # 兜底（gap < 最小片）


def _absorb_tiny_gaps(placed: list[dict],
                      ql_per_measure: Fraction = Fraction(4)) -> None:
    """声部内孤立 1/12 间隙并进前音时值（~31ms@80bpm，视觉几乎不可见）。

    只延长前音/末音、绝不移动后续 onset——避免在连奏串中级联推移。
    代价：前音时值可能变 5/12 类复合值（music21 拆成 tie 小尾巴，实测
    全曲 ~16 个），比碎休止符温和得多。
    """
    unit = Fraction(1, 12)
    placed.sort(key=lambda x: x["offset"])
    if placed and placed[0]["offset"] == unit:
        placed[0]["offset"] = Fraction(0)
    for prev, nxt in zip(placed, placed[1:]):
        gap = nxt["offset"] - (prev["offset"] + prev["dur"])
        if gap == unit:
            prev["dur"] += unit
            prev["obj"].duration.quarterLength = prev["dur"]
    if placed:
        last = placed[-1]
        if ql_per_measure - (last["offset"] + last["dur"]) == unit:
            last["dur"] += unit
            last["obj"].duration.quarterLength = last["dur"]


def _fill_rests(items: list[dict], ql_per_measure: Fraction,
                vocab: list[Fraction] | None = None) -> list[tuple[Fraction, object]]:
    """items = [{offset, dur, obj(NotRest)}] → 补休止符的 (offset, obj) 列表。

    量化模式 vocab=REST_VOCAB（干净词表）；忠实模式 None → _legalize 链。
    """
    out = []
    cursor = Fraction(0)
    for it in sorted(items, key=lambda x: x["offset"]):
        if it["offset"] > cursor:
            off = cursor
            for piece in _rest_pieces(it["offset"] - cursor, vocab):
                r = m21note.Rest()
                r.duration.quarterLength = piece
                out.append((off, r))
                off += piece
        out.append((it["offset"], it["obj"]))
        cursor = it["offset"] + it["dur"]
    if cursor < ql_per_measure:
        off = cursor
        for piece in _rest_pieces(ql_per_measure - cursor, vocab):
            r = m21note.Rest()
            r.duration.quarterLength = piece
            out.append((off, r))
            off += piece
    return out


def _group_chords(items: list[dict]):
    """同小节内 (offset, dur) 相同的片段组 → 和弦。"""
    keyed = defaultdict(list)
    for it in items:
        keyed[(it["offset"], it["dur"])].append(it)
    return sorted(keyed.items(), key=lambda kv: kv[0][0])


def assemble_solo_score(cls: str, events: list[dict], bpm: float,
                        key_signature: str, transposition: int,
                        ql_per_measure: Fraction = Fraction(4),
                        time_signature: str = "4/4",
                        harmony: list[dict] | None = None,
                        clean_rests: bool = True):
    """单乐器谱。键盘族 = 大谱表（双 PartStaff），其余单谱表。

    events 需带 frags（量化产物）。harmony = T1 和弦段 → 首谱表插
    ChordSymbol（MusicXML <harmony>）。返回 stream.Score。
    """
    inst_obj = _m21_instrument(cls)
    is_keyboard = cls in KEYBOARD_CLASSES
    written = transposition  # written pitch = sounding + transposition

    frag_items: dict[int, list[dict]] = defaultdict(list)  # voice -> 片段
    for ev in events:
        for fr in ev["frags"]:
            frag_items[ev["voice"]].append({
                "m": fr["bar"], "offset": Fraction(fr["offset"]).limit_denominator(96),
                "dur": Fraction(fr["dur"]), "tie": fr["tie"],
                "pitch": ev["pitch"] + written, "ev": ev,
            })

    all_frags = [it for its in frag_items.values() for it in its]
    m_lo = min(it["m"] for it in all_frags)
    m_hi = max(it["m"] for it in all_frags)
    voices_used = sorted(frag_items.keys())

    if is_keyboard:
        top, bottom = stream.PartStaff(), stream.PartStaff()
        top.insert(0, m21clef.TrebleClef())
        bottom.insert(0, m21clef.BassClef())
        top.insert(0, inst_obj)
        bottom.insert(0, _m21_instrument(cls))
        part_of_voice = {1: top, 2: bottom}
        # 键盘双谱表：缺声部时给另一个谱表
        if 1 not in part_of_voice or 1 not in voices_used:
            voices_used = sorted(set(voices_used) | {1})
        if 2 not in voices_used and len(voices_used) > 1:
            voices_used = sorted(set(voices_used) | {2})
        parts = [top, bottom]
    else:
        part = stream.Part()
        part.insert(0, inst_obj)
        part_of_voice = {v: part for v in voices_used}
        parts = [part]

    for v in voices_used:
        part = part_of_voice[v]
        by_measure: dict[int, list[dict]] = defaultdict(list)
        for it in frag_items.get(v, []):
            by_measure[it["m"]].append(it)
        for mi in range(m_lo, m_hi + 1):
            m = stream.Measure()
            m.number = mi - m_lo + 1
            items = by_measure.get(mi)
            if not items:
                r = m21note.Rest()
                r.duration.quarterLength = ql_per_measure
                m.insert(0, r)
                part.append(m)
                continue
            # 溢出声部分配：同 offset+同时值叠入同 lane（成和弦）；同声部
            # 不同音高时间重叠开新 lane —— 避免小节 overfull
            lanes: list[dict] = []
            for it in sorted(items, key=lambda x: (x["offset"], -x["pitch"])):
                for lane in lanes:
                    if (it["offset"] == lane["group_off"]
                            and it["dur"] == lane["group_dur"]):
                        lane["items"].append(it)
                        break
                    if it["offset"] >= lane["end"]:
                        lane["items"].append(it)
                        lane["end"] = it["offset"] + it["dur"]
                        lane["group_off"], lane["group_dur"] = it["offset"], it["dur"]
                        break
                else:
                    lanes.append({"items": [it], "end": it["offset"] + it["dur"],
                                  "group_off": it["offset"], "group_dur": it["dur"]})
            for li, lane in enumerate(lanes):
                placed = []
                for (grp_off, _grp_dur), grp in _group_chords(lane["items"]):
                    if len(grp) >= 2:
                        ns = []
                        for it in grp:
                            n = m21note.Note(it["pitch"])
                            n.duration.quarterLength = it["dur"]
                            if it["tie"]:
                                n.tie = m21tie.Tie(it["tie"])
                            ns.append(n)
                        placed.append({"offset": grp_off, "dur": grp[0]["dur"],
                                       "obj": m21chord.Chord(ns)})
                    else:
                        it = grp[0]
                        n = m21note.Note(it["pitch"])
                        n.duration.quarterLength = it["dur"]
                        if it["tie"]:
                            n.tie = m21tie.Tie(it["tie"])
                        if v == 2:
                            n.stemDirection = "down"
                        placed.append({"offset": grp_off, "dur": it["dur"], "obj": n})
                voice = stream.Voice()
                voice.id = li + 1
                _absorb_tiny_gaps(placed, ql_per_measure)
                for off, obj in _fill_rests(placed, ql_per_measure,
                                            REST_VOCAB if clean_rests else None):
                    voice.insert(off, obj)
                m.insert(0, voice)
            part.append(m)

    # 首小节元数据
    m0 = parts[0].getElementsByClass(stream.Measure)[0]
    ks_parts = key_signature.strip().split()
    tonic, mode = ks_parts[0], ks_parts[1] if len(ks_parts) > 1 else "major"
    m0.insert(0, m21tempo.MetronomeMark(number=int(round(bpm))))
    m0.insert(0, m21key.Key(tonic, mode))
    m0.insert(0, m21meter.TimeSignature(time_signature))

    if transposition:
        try:
            inst_obj.transposition = m21interval.Interval(transposition)
        except Exception:
            logger.debug("transposition interval set failed", exc_info=True)

    # T1 和弦记号：首谱表每小节起点（N.C. 不写）
    if harmony:
        from music21 import harmony as m21harmony
        for m in parts[0].getElementsByClass(stream.Measure):
            mi_abs = m.number + m_lo - 1
            bar_start = Fraction(mi_abs) * ql_per_measure
            for seg in harmony:
                s = seg.get("start_ql")
                if (seg.get("label") == "N.C." or s is None
                        or not (bar_start <= s < bar_start + ql_per_measure)):
                    continue
                try:
                    cs = m21harmony.ChordSymbol(figure=seg["label"])
                    cs.writeAsChord = False
                    m.insert(float(Fraction(s).limit_denominator(96) - bar_start), cs)
                except Exception:
                    logger.debug("chord symbol skipped: %s", seg.get("label"),
                                 exc_info=True)

    score = stream.Score()
    if is_keyboard:
        score.insert(0, layout.StaffGroup(parts, symbol="brace"))
    for p in parts:
        score.insert(0, p)
    return score


# ---------------------------------------------------------------------------
# TAB 谱（手写 MusicXML，与 music21 staff 部分合并）
# ---------------------------------------------------------------------------

def _pitch_xml(pitch: int) -> ET.Element:
    el = ET.Element("pitch")
    ET.SubElement(el, "step").text = STEP_NAMES[pitch % 12].replace("#", "")
    ET.SubElement(el, "alter").text = "1" if pitch % 12 in SHARPED else "0"
    ET.SubElement(el, "octave").text = str(pitch // 12 - 1)
    return el


def _tab_note_xml(m_el: ET.Element, dur: Fraction, pitch: int | None = None,
                  tie: str | None = None, chord: bool = False,
                  string: int | None = None, fret: int | None = None,
                  rest: bool = False) -> None:
    note = ET.SubElement(m_el, "note")
    if chord:
        ET.SubElement(note, "chord")
    if rest or pitch is None:
        ET.SubElement(note, "rest")
    else:
        note.append(_pitch_xml(pitch))
    ET.SubElement(note, "duration").text = str(int(dur * DIVISIONS))
    ET.SubElement(note, "voice").text = "1"
    typ, dots, mod = DUR_TO_TYPE.get(dur, ("quarter", 0, None))
    ET.SubElement(note, "type").text = typ
    for _ in range(dots):
        ET.SubElement(note, "dot")
    if mod:
        tm = ET.SubElement(note, "time-modification")
        ET.SubElement(tm, "actual-notes").text = str(mod[0])
        ET.SubElement(tm, "normal-notes").text = str(mod[1])
    if tie in ("start", "continue"):
        ET.SubElement(note, "tie", {"type": "start"})
    if tie in ("stop", "continue"):
        ET.SubElement(note, "tie", {"type": "stop"})
    has_technical = (string is not None and fret is not None and fret >= 0
                     and not rest and pitch is not None)
    if tie or has_technical:
        notations = ET.SubElement(note, "notations")
        if tie:
            if tie in ("start", "continue"):
                ET.SubElement(notations, "tied", {"type": "start"})
            if tie in ("stop", "continue"):
                ET.SubElement(notations, "tied", {"type": "stop"})
        if has_technical:
            tech = ET.SubElement(notations, "technical")
            ET.SubElement(tech, "string").text = str(string)
            ET.SubElement(tech, "fret").text = str(fret)


def _build_tab_part_xml(events: list[dict], transposition: int,
                        ql_per_measure: Fraction = Fraction(4),
                        time_signature: str = "4/4",
                        part_id: str = "Ptab") -> ET.Element:
    """单轨 TAB part：单声部，string/fret technical，divisions=48。"""
    beat_num, beat_den = time_signature.split("/")
    frag_items = []
    for ev in events:
        for fr in ev["frags"]:
            frag_items.append({
                "m": fr["bar"], "offset": Fraction(fr["offset"]).limit_denominator(96),
                "dur": Fraction(fr["dur"]), "tie": fr["tie"],
                "pitch": ev["pitch"] + transposition,
                "string": ev.get("string"), "fret": ev.get("fret"),
            })
    frag_items.sort(key=lambda x: (x["m"], x["offset"], x["pitch"]))
    m_lo = min(it["m"] for it in frag_items)
    m_hi = max(it["m"] for it in frag_items)

    part = ET.Element("part", {"id": part_id})
    for mi in range(m_lo, m_hi + 1):
        m_el = ET.SubElement(part, "measure", {"number": str(mi - m_lo + 1)})
        if mi == m_lo:
            attrs = ET.SubElement(m_el, "attributes")
            ET.SubElement(attrs, "divisions").text = str(DIVISIONS)
            key_el = ET.SubElement(attrs, "key")
            ET.SubElement(key_el, "fifths").text = "0"
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = beat_num
            ET.SubElement(time_el, "beat-type").text = beat_den
            sd = ET.SubElement(attrs, "staff-details")
            ET.SubElement(sd, "staff-lines").text = "6"
            clef_el = ET.SubElement(attrs, "clef")
            ET.SubElement(clef_el, "sign").text = "TAB"
            ET.SubElement(clef_el, "line").text = "6"

        items = [it for it in frag_items if it["m"] == mi]
        if not items:
            _tab_note_xml(m_el, dur=ql_per_measure, rest=True)
            continue

        cursor = Fraction(0)
        groups: dict[Fraction, list[dict]] = defaultdict(list)
        for it in items:
            groups[it["offset"]].append(it)
        for off in sorted(groups):
            if off > cursor:
                _tab_note_xml(m_el, dur=off - cursor, rest=True)
            for k, it in enumerate(sorted(groups[off], key=lambda x: -x["pitch"])):
                _tab_note_xml(m_el, dur=it["dur"], pitch=it["pitch"],
                              tie=it["tie"], chord=k > 0,
                              string=it["string"], fret=it["fret"])
            cursor = off + groups[off][0]["dur"]
        if cursor < ql_per_measure:
            _tab_note_xml(m_el, dur=ql_per_measure - cursor, rest=True)
    return part


def _merge_tab_into_musicxml(xml_path: str, tab_part: ET.Element) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    part_list = root.find("part-list")
    sp = ET.SubElement(part_list, "score-part", {"id": tab_part.get("id")})
    ET.SubElement(sp, "part-name").text = "TAB"
    root.append(tab_part)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------

def _admitted(t: dict, duration: float) -> bool:
    """准入判定：优先读数据层 score_worthy，旧文件按准入线现算（规则v1 §2）。"""
    if "score_worthy" in t:
        return bool(t["score_worthy"])
    from src.multi_instrument import (SCORE_MIN_COVERAGE, SCORE_MIN_NOTES,
                                      union_active_seconds)
    notes = t["notes"]
    cov = union_active_seconds(notes) / duration if duration > 0 else 0.0
    return len(notes) >= SCORE_MIN_NOTES and cov >= SCORE_MIN_COVERAGE


def build_notation(notes_json: dict, output_dir: str, mode: str = "both") -> dict | None:
    """notes_json（multi_instrument 产物）→ notation/ 目录。

    mode: "quantized" | "faithful" | "both"（faithful = 不做标准时值量化，
    位置吸附 1/48 网格且不做标准时值量化；TAB 仅量化模式生成）。
    """
    duration = float(notes_json.get("duration") or 0.0)
    tracks = [t for t in notes_json["tracks"] if _admitted(t, duration)]
    if not tracks:
        logger.warning("  [notation] no score-worthy tracks, skip")
        return None
    bpm = float(notes_json["bpm"])
    tsig = notes_json.get("time_signature", "4/4")
    beat_num, beat_den = int(tsig.split("/")[0]), int(tsig.split("/")[1])
    # 四分音符单位的小节长（v1 仅 4/4 全面支持，bar 计算按 4 拍硬编码）
    ql_per_measure = Fraction(beat_num * 4, beat_den)
    if tsig != "4/4":
        logger.warning("  [notation] time signature %s 非 4/4，小节划分仍按 4 拍处理", tsig)

    pooled = [n for t in tracks for n in t["notes"]]
    raw_onsets = {id(n): float(n["onset"]) for n in pooled}

    quantize_onsets(pooled, bpm, tsig)  # 全局池化，in-place
    key_sig = estimate_key(pooled)

    # T1 和声先验层：和弦跟踪（失败不影响谱面导出）
    harmony_segments: list[dict] | None = None
    try:
        from src.harmony_prior import track_chords
        end_ql = _ql(max(n["offset"] for n in pooled), bpm)
        harmony_segments = track_chords(pooled, bpm, key_sig, end_ql)
    except Exception:
        logger.exception("  [harmony] chord tracking failed")

    # 量化健康度：网格命中率（bpm 反向校验的廉价代理，规则v1 §4）
    hits = sum(1 for n in pooled
               if abs(_ql(raw_onsets[id(n)], bpm)
                      - _snap(_ql(raw_onsets[id(n)], bpm), QL_DENOM)) <= SNAP_TOL_QL)
    snap_rate = hits / len(pooled)
    logger.info("  [notation] grid snap rate %.1f%% (bpm=%.1f)", snap_rate * 100, bpm)
    if snap_rate < 0.4:
        logger.warning("  [notation] snap rate < 40%% — bpm=%s 可能失准", bpm)

    out_tracks = []
    os.makedirs(os.path.join(output_dir, "notation", "solo"), exist_ok=True)

    for ti, t in enumerate(tracks):
        cls = t["instrument_class"]
        if cls == "drums":
            logger.info("  [notation] drums 谱 v1 不含，跳过")
            continue
        events = build_track_events(cls, t["notes"], raw_onsets, bpm)
        if harmony_segments:
            try:
                from src.harmony_prior import note_role
                for ev in events:
                    ev["harmony_role"] = note_role(
                        ev["pitch"], _ql(ev["onset_sec"], bpm),
                        harmony_segments, key_sig)
            except Exception:
                logger.exception("  [harmony] role tagging failed: %s", cls)
        out_tracks.append({
            "track_ref": ti,
            "instrument_class": cls,
            "display_name": t.get("display_name", cls),
            "staff_layout": "grand" if cls in KEYBOARD_CLASSES else "single",
            "events": events,
        })

    # T4 v0 乐曲分析（王道/常见度/独创度/复杂度/离群率）
    analysis: dict | None = None
    if harmony_segments:
        try:
            from src.harmony_prior import score_analysis
            roles = [ev.get("harmony_role") for t in out_tracks for ev in t["events"]]
            n_roles = len(roles)
            outlier_rate = (sum(1 for r in roles if r == "chromatic") / n_roles
                            if n_roles else None)
            analysis = score_analysis(harmony_segments, key_sig, outlier_rate)
            logger.info("  [harmony] analysis: %s", analysis)
        except Exception:
            logger.exception("  [harmony] analysis failed")

    notation = {
        "schema_version": 1,
        "mode": mode,
        "bpm": bpm,
        "time_signature": tsig,
        "key": key_sig,
        "harmony_track": harmony_segments or [],
        "analysis": analysis,
        "meta": {"snap_rate": round(snap_rate, 4),
                 "quantizer": "segmented-16th+triplet"},
        "tracks": out_tracks,
    }
    with open(os.path.join(output_dir, "notation", "notation.json"), "w",
              encoding="utf-8") as f:
        json.dump(notation, f, ensure_ascii=False, indent=1)
    logger.info("  [notation] notation.json written (%d tracks)", len(out_tracks))

    # 单乐器谱导出（量化模式）
    for t in out_tracks:
        cls = t["instrument_class"]
        transposition = _transpose_semitones(cls)
        stem = os.path.join(output_dir, "notation", "solo", cls)
        try:
            score = assemble_solo_score(cls, t["events"], bpm, notation["key"],
                                        transposition, ql_per_measure, tsig,
                                        harmony_segments)
            score.write("musicxml", fp=stem + ".musicxml")
            if cls in GUITAR_CLASSES:
                tab = _build_tab_part_xml(t["events"], transposition,
                                          ql_per_measure, tsig,
                                          part_id=f"Ptab-{cls}")
                _merge_tab_into_musicxml(stem + ".musicxml", tab)
            n_meas = len(list(score.recurse().getElementsByClass(stream.Measure)))
            logger.info("  [notation] solo/%s.musicxml (%d events, %d measures)",
                        cls, len(t["events"]), n_meas)
        except Exception:
            logger.exception("  [notation] solo export failed: %s", cls)

    if mode in ("faithful", "both"):
        _export_faithful(tracks, raw_onsets, bpm, ql_per_measure, notation,
                         output_dir, tsig)
    return notation


def _export_faithful(tracks: list[dict], raw_onsets: dict[int, float], bpm: float,
                     ql_per_measure: Fraction, notation: dict, output_dir: str,
                     time_signature: str = "4/4") -> None:
    """忠实模式：不做分段网格量化，位置/时值取原始值（1/48 网格表达），
    时值经 _legalize 分解为合法 tie 链——谱面会出现非标准时值与碎休止，
    这正是忠实模式的语义。单声部（不做人声部分Split）。"""
    for t in tracks:
        cls = t["instrument_class"]
        if cls == "drums":
            continue
        try:
            evs = []
            for n in t["notes"]:
                onset = _snap(_ql(raw_onsets[id(n)], bpm), FAITHFUL_DENOM)
                dur = max(_snap(_ql(n["offset"] - n["onset"], bpm), FAITHFUL_DENOM),
                          Fraction(1, 48))
                evs.append({"pitch": int(n["pitch"]), "voice": 1,
                            "_onset_ql": onset, "_dur_ql": dur})
            evs.sort(key=lambda e: (e["_onset_ql"], e["pitch"]))
            _post_snap_guard(evs)
            events = []
            for ev in evs:
                cur, rem = ev["_onset_ql"], ev["_dur_ql"]
                frags = []
                while rem > 0:
                    mi = cur // ql_per_measure
                    room = (mi + 1) * ql_per_measure - cur
                    for piece in _legalize(min(rem, room)):
                        take = min(piece, rem)
                        frags.append({"bar": int(mi),
                                      "offset": float(cur - mi * ql_per_measure),
                                      "dur": str(take), "tie": None})
                        cur += take
                        rem -= take
                for fr, role in zip(frags, _tie_roles(len(frags))):
                    fr["tie"] = role
                events.append({"pitch": ev["pitch"], "voice": 1, "frags": frags})
            score = assemble_solo_score(cls, events, bpm, notation["key"],
                                        _transpose_semitones(cls), ql_per_measure,
                                        time_signature, clean_rests=False)
            score.write("musicxml", fp=os.path.join(
                output_dir, "notation", "solo", f"{cls}.faithful.musicxml"))
            logger.info("  [notation] solo/%s.faithful.musicxml", cls)
        except Exception:
            logger.exception("  [notation] faithful export failed: %s", cls)
