"""Layer 5 v2（阶段19）：记谱层 —— NotationScore 组装 + 单乐器谱导出。

输入 = multi_instrument 的 notes.json 数据层产物（已过 clean_notes 清洗与
准入判定）；产出 = notation/notation.json（NotationScore，忠实+量化双模式
字段）+ notation/solo/{class}.musicxml。规则依据 markdown/记谱规则v1.md：

- 量化（时值 v3，音头保持）：全体池化音头联合拟合 1/12 格的**全局相位
  φ**（Cemgil 2003 量化=推断口径：格点 MAP = 最小位移吸附），φ 平移后
  单次吸附到 1/12 网格。替换 v2 的「quantize_onsets 秒域分段网格 + 记谱层
  绝对 1/12 网格」双重吸附——两重网格相位不对齐时每个音头被系统性平移
  （kyomu 实测 p90 ~50ms）；单网格后位移上限恒 1/24 四分（半格）。
- 节奏简化（记谱规则v1 §3.4，治"谱脏"）：量化后三步——①同时性聚类
  （扫弦 stagger 归一为和弦）；②时值再分配（onset 可靠、offset 不可靠，
  时值由 gap/ratio 语境决定：连奏填满、断奏/中间态在间隔 ≤1 拍时同样
  填满到声部内下一音头——演奏短奏属于表情层，谱面写节拍值，休止符只
  留给短语级沉默；间隔 >1 拍才写缩短值+休止）；③小节内单值分解（只跨
  小节才 tie，链 ≤2 片；持续音封顶一小节）。
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
from pathlib import Path

from music21 import (chord as m21chord, clef as m21clef, instrument as m21inst,
                     interval as m21interval, key as m21key, layout, meter as m21meter,
                     note as m21note, stream, tempo as m21tempo, tie as m21tie)

from src import guitar_tab, voice_assign
from src.key_estimate import estimate_key

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

# 时值代价（时值 v2，2026-08-27 Phase-1 拍板"分子尽量是 1"的可计算化）：
# 纯二分单位分数 0 < 单附点 1 < 三连音值 2；连音链每多一片 +1。
# 片段选择一律最小总代价（同代价时取片数少、再取首片大）——
# legato 精确填满不留碎休止，detached/middle 单值优先单位分数。
UNIT_DURS = frozenset({Fraction(1, 4), Fraction(1, 2), Fraction(1),
                       Fraction(2), Fraction(4)})
DOTTED_DURS = frozenset({Fraction(3, 4), Fraction(3, 2), Fraction(3)})
_TRIPLET_SINGLES = frozenset({Fraction(1, 3), Fraction(2, 3),
                              Fraction(1, 12), Fraction(1, 6)})


def _dur_cost(d: Fraction) -> int:
    if d in UNIT_DURS:
        return 0
    if d in DOTTED_DURS:
        return 1
    if d in _TRIPLET_SINGLES:
        return 2
    return 3  # 其余合法单值（7/4 双附点等）：最高代价


def _largest_unit(bound: Fraction, vocab: list[Fraction]) -> Fraction:
    """断奏/中间态单值：词表内 ≤ bound 的最小代价值（同代价取最大）。
    无单位分数可用时退让到附点/三连音——bound < 1/4 时兜底 MIN_DUR。"""
    cands = [d for d in vocab if d <= bound]
    if not cands:
        return MIN_DUR
    best = min(cands, key=lambda d: (_dur_cost(d), -d))
    return max(MIN_DUR, best)


def _note_frag_cost(d: Fraction) -> int:
    """连奏片段代价（第二阶段 #2，2026-08-29 用户拍板"同小节合并"）：
    附点不罚——附点四分单片 优于 8分+16分 tie（现代流行谱惯例；古典
    跨拍附点拆分规范让位可读性，全库实测 739 处 1/2+1/4 / 83 处 1+1/2
    tie 链由此消除）。三连音/复合值罚分保持（1/4+1/12 等位置性拆分
    不受影响）。仅用于音符连奏填充；断奏(_largest_unit 宁短勿拖)与
    休止符(_rest_pieces 古典跨拍拆分)沿用 _dur_cost。"""
    if d in UNIT_DURS or d in DOTTED_DURS:
        return 0
    return _dur_cost(d)


def _choose_fragments(onset_ql: Fraction, bound: Fraction,
                      ql_per_measure: Fraction = Fraction(4),
                      vocab: list[Fraction] | None = None,
                      max_pieces: int = 3) -> list[Fraction]:
    """连奏精确填满 bound 的最小代价片段链（时值 v2 核心）。

    约束：片段不穿小节线（小节内可多片，跨小节必在小节线处切）；
    词表 = 语境词表 ∪ {1/12, 1/6}（奇数 twelfth 的精确填充需要，
    两者均为合法单值）。链超 max_pieces 片时退化为最少片数解。
    """
    base = list(vocab if vocab is not None else ALL_DURS)
    fill_vocab = sorted(set(base) | {Fraction(1, 12), Fraction(1, 6)},
                        reverse=True)
    unit = Fraction(1, 12)
    if bound <= 0:
        return [MIN_DUR]

    # 按小节线切 bound，逐段精确填充（片段永不跨小节线）
    segments: list[tuple[Fraction, Fraction]] = []  # (start, end)
    cur = onset_ql
    rem = bound
    while rem > 0:
        m_end = (cur // ql_per_measure + 1) * ql_per_measure
        span = min(rem, m_end - cur)
        segments.append((cur, span))
        cur += span
        rem -= span

    def dp_exact(span: Fraction, by_pieces: bool = False) -> list[Fraction] | None:
        """精确填满 span 的 DP。默认最小(代价,片数)；by_pieces=True 时
        最小(片数,代价)——超长链的降级目标，供 max_pieces 回退用。"""
        k = int(span / unit)
        if k <= 0:
            return []
        costs: list[tuple[int, int, list[Fraction]] | None] = [None] * (k + 1)
        costs[0] = (0, 0, [])
        for i in range(1, k + 1):
            for p in fill_vocab:
                u = int(p / unit)
                if u <= i and costs[i - u] is not None:
                    c, n, lst = costs[i - u]
                    c2, n2 = (c + _note_frag_cost(p), n + 1)
                    if by_pieces:
                        c2, n2 = n2, c2
                    cand = (c2, n2, [p] + lst)
                    if costs[i] is None or cand[:2] < costs[i][:2]:
                        costs[i] = cand
        return costs[k][2] if costs[k] is not None else None

    pieces: list[Fraction] = []
    for _start, span in segments:
        seg = dp_exact(span)
        if seg is None:  # 理论不发生（1/12 兜底片在词表内）
            return _fit_fragments(onset_ql, bound, ql_per_measure, vocab)
        pieces.extend(seg)
    if len(pieces) > max_pieces:
        # 代价换可读性：改按"分段内最少片数"重解——绝不能退回不分段的
        # _rest_pieces（无小节线约束，曾在 15/4 拍处产出 4 拍全音符越线，
        # 回归门 43 处跨度溢出的根因，2026-08-27）。
        alt: list[Fraction] = []
        for _start, span in segments:
            seg = dp_exact(span, by_pieces=True)
            if seg is None:
                alt = []
                break
            alt.extend(seg)
        if alt and len(alt) < len(pieces):
            return alt
    return pieces or [MIN_DUR]

# 节奏简化常量（记谱规则v1 §3.4）
SUSTAIN_CAP = Fraction(4)   # 持续音封顶一小节（4/4；用户拍板）
LEGATO_RATIO = Fraction(85, 100)  # raw 时值 ≥ gap×0.85 → 连奏填满
DETACHED_RATIO = Fraction(1, 2)   # raw 时值 < gap×0.5 → 断奏缩短
# 休止符削减（时值 v3，2026-08-27）：断奏/中间态在声部内间隔 ≤1 拍时同样
# 填满到下一音头（演奏短奏=表情层，谱面写节拍值；休止只留短语级沉默）。
# 间隔 >1 拍（换气/乐句空档）保留缩短值+休止——乐理上乐句结构需要可见。
FILL_GAP_MAX = Fraction(1)
# score_mid 同音高连排起音保护：填满写法使相邻同音高 MIDI 连成长音（听感
# =漏掉重复音头），起音前留 30ms release 让攻击重新触发（30ms 对节拍
# 观感不可闻）。跨音高不需要——音高变化本身就是新起音。
RELEASE_SEC = 0.03
CLUSTER_SEC = 0.040         # 同时性聚类窗口（扫弦 stagger <40ms）
SEAM_SEC = 0.040            # 量化后接缝判定窗口（§3.2 遗留）

# 时值 → (type, 附点数, time-modification(actual,normal)) —— 手写 TAB XML 用。
# 程序化生成全集（base×附点×3:2 三连音），保证任何合法单值都有正确记谱类型；
# 旧表只覆盖 10 个值，1/6、4/3 等会静默落成错误的 type（2026-08-25 TAB 修复）。
DUR_TO_TYPE: dict[Fraction, tuple[str, int, tuple[int, int] | None]] = {}
for _val, _name in ((Fraction(4), "whole"), (Fraction(2), "half"),
                    (Fraction(1), "quarter"), (Fraction(1, 2), "eighth"),
                    (Fraction(1, 4), "16th"), (Fraction(1, 8), "32nd")):
    for _dots in (0, 1, 2):
        _mult = Fraction(2) - Fraction(1, 2 ** _dots)
        DUR_TO_TYPE[_val * _mult] = (_name, _dots, None)
        DUR_TO_TYPE[_val * _mult * Fraction(2, 3)] = (_name, _dots, (3, 2))

STEP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SHARPED = {1, 3, 6, 8, 10}


def _ql(sec: float, bpm: float) -> Fraction:
    """秒 → 四分音符数（Fraction）。bpm 经 str() 转换避免浮点分母爆炸。"""
    return Fraction(sec).limit_denominator(20000) * Fraction(str(bpm)) / 60


def _snap(fr: Fraction, denom: int) -> Fraction:
    """吸附到 1/denom 四分音符网格。"""
    return Fraction(round(fr * denom), denom)


def _prior_snap_units(x: float) -> int:
    """格点计数 → 整数（节拍先验吸附；Cemgil 2003 位置先验 p(τ mod 1)
    的离散化：整拍 0.80 > 半拍 0.15 > 三连 0.008）。

    演奏前置/滞后 ≤ 亚记谱精度（~0.85 格 ≈ 50ms）时吸收到更粗的格点
    （整拍/半拍/八分）——人类刻谱同样把 50ms 的 anticipation 记在拍上
    （canon 实测第二拍和弦稳定 1.917 而非 2.0）；其余位置取最近格。
    """
    import math
    lo, hi = math.floor(x), math.ceil(x)
    if lo == hi:
        return int(lo)
    dlo, dhi = x - lo, hi - x

    def _grade(k: int) -> int:
        if k % 12 == 0:
            return 4          # 整拍
        if k % 6 == 0:
            return 3          # 半拍
        if k % 3 == 0:
            return 2          # 八分（含三连 4/12）
        return 1 if k % 2 == 0 else 0

    def _cost(k: int, d: float) -> float:
        bonus = {4: 0.85, 3: 0.7, 2: 0.5}.get(_grade(k), 0.0)
        return (d - bonus) if d <= 0.9 else d

    return int(lo if _cost(lo, dlo) <= _cost(hi, dhi) else hi)


def _fit_grid_map(notes: list[dict], bpm: float):
    """时值 v3.1 音头映射：全局 (bpm,φ) 联合精修 + 分段线性速度漂移跟踪。

    模型 = Cemgil 2003（量化=推断，速度偏差平滑游走）的分段化：谱面格点
    计数 k(t) ≈ a·t+b；①先联合精修全局斜率 a 与相位 φ（目标 = 全体 onset
    到最近 1/12 格点的平均距离最小，稀疏曲目改善 <15% 时拒绝改速防混叠
    ——1/12 稠密格在无相位优化时存在 ±3% 量级的假峰，kyomu 实测踩过）；
    ②再以 8 小节窗/2 小节 hop 的 Theil-Sen 稳健回归跟踪局部斜率（钳制
    ±4%，中位数截距），吸收 rubato 速度游走；音符级偏差（真实切分/转写
    噪声）不被吸收。背景：单一全局速度对演奏型录音（canon 实测和弦每
    小节右滑 ~1/12 拍、累计 ~7 拍）结构必然崩坏——小节线切进和弦。

    返回 (bpm_refined, ql_map, resid_map, meta)；ql_map(t) → 吸附后 QL
    （Fraction，1/12 网格、非负），resid_map(t) → 吸附位移（QL）。
    """
    import numpy as np
    ts = np.array(sorted({float(n["onset"]) for n in notes}))
    a0 = bpm / 60.0 * 12  # 全局斜率：格点计数/秒

    def cost_with_phase(a: float) -> tuple[float, float]:
        d = (ts * a) % 1.0
        best = None
        for p in np.unique(d):
            dd = np.abs(d - p)
            c = float(np.minimum(dd, 1 - dd).mean())
            if best is None or c < best[0]:
                best = (c, float(p))
        return best  # (cost, phase)

    base_c, _phi = cost_with_phase(a0)
    best_c, a_ref = base_c, a0
    for f in np.arange(0.960, 1.0401, 0.0004):
        c, _ = cost_with_phase(a0 * f)
        if c < best_c - 1e-6:
            best_c, a_ref = c, a0 * f
    if base_c - best_c < 0.15 * max(base_c, 1e-9):  # 稀疏防混叠
        a_ref, best_c = a0, base_c
    bpm_ref = float(a_ref / 12 * 60)
    _, phi = cost_with_phase(a_ref)

    # 分段线性 F(t)：滑动窗直接做 (斜率, 相位) 格点拟合——窗内 onset 应
    # 密集落在 a·t+b 的整点上；相邻窗整数解缠（相位圆环歧义 ±1 格），
    # 单调保护。第一版教训：用 round(全局直线) 的阶梯整数做锚会丢相位
    # （直线骑格缝，canon 实测中位残差恰好半格 154ms）且斜率被污染。
    bar_sec = 48.0 / a_ref
    win, hop = 8 * bar_sec, 2 * bar_sec
    centers: list[float] = []
    lines: list[tuple[float, float]] = []  # (a, b)
    c_pos = float(ts[0]) + win / 2
    while c_pos <= float(ts[-1]) + win / 2 + 1e-9:
        m = (ts >= c_pos - win / 2) & (ts <= c_pos + win / 2)
        if int(m.sum()) >= 8:
            tt = ts[m]
            best = None
            for a in np.linspace(0.96 * a_ref, 1.04 * a_ref, 41):
                v = (tt * a) % 1.0
                hist, _ = np.histogram(v, bins=48, range=(0.0, 1.0))
                p = (int(np.argmax(hist)) + 0.5) / 48.0
                dd = float(np.minimum((v - p) % 1.0,
                                      1 - (v - p) % 1.0).mean())
                if best is None or dd < best[0]:
                    best = (dd, float(a), p)
            centers.append(c_pos)
            lines.append((best[1], -best[2]))
        c_pos += hop

    if not centers:  # 曲目过短 → 纯全局直线
        def _line(t: float) -> float:
            return a_ref * t - phi
        F = _line
        n_win = 0
    else:
        # 相邻窗解缠：当前窗直线在共享中心处对齐前窗（±0.5 格内取整）
        for i in range(1, len(lines)):
            a_p, b_p = lines[i - 1]
            a_c, b_c = lines[i]
            shift = round((a_p * centers[i] + b_p) - (a_c * centers[i] + b_c))
            lines[i] = (a_c, b_c + shift)
        xs = [float(ts[0])] + centers + [float(ts[-1])]
        ys = [lines[0][0] * xs[0] + lines[0][1]] \
            + [a * c + b for c, (a, b) in zip(centers, lines)] \
            + [lines[-1][0] * xs[-1] + lines[-1][1]]
        for i in range(1, len(xs)):  # 单调保护（窗间断层时保序）
            if ys[i] <= ys[i - 1]:
                ys[i] = ys[i - 1] + 1e-6 * (xs[i] - xs[i - 1] + 1e-6)
        xs_a, ys_a = np.array(xs), np.array(ys)

        def _F(t: float) -> float:
            return float(np.interp(t, xs_a, ys_a))
        F = _F
        n_win = len(centers)

    def ql_map(t: float) -> Fraction:
        return Fraction(max(_prior_snap_units(F(t)), 0), 12)

    def resid_map(t: float) -> Fraction:
        v = F(t)
        return Fraction(round(abs(v - _prior_snap_units(v)) * 2**20),
                        2**20)

    meta = {"bpm_refined": round(bpm_ref, 2),
            "tempo_delta_pct": round((bpm_ref / bpm - 1) * 100, 2),
            "n_windows": n_win}
    return {"ql_map": ql_map, "resid_map": resid_map,
            "cont": lambda t: F(t) / 12, "meta": meta}


def _beat_sync_map(notes: list[dict], beat_times, pulse_sec: float,
                   bpm: float):
    """拍同步格点映射（时值 v3.2）：beat k → QL k·pulse_ql。

    素材 = 音频节拍跟踪（src/beat_track.py）。librosa 拍是脉冲级（四分/
    八分/半分），pulse_ql 吸附到 {1/4, 1/3, 1/2, 1, 2}；小节相位（弱起
    偏移）由 onset 速度权重在小节内位置上投票（2 拍和声变化的曲子 p 与
    p+bar/2 同分属正常，取先验最近）。返回结构与 _fit_grid_map 一致。
    """
    import numpy as np
    bt = np.asarray(beat_times, dtype=float)
    pulse_ql = min([0.25, 1 / 3, 0.5, 1.0, 2.0],
                   key=lambda x: abs(x - pulse_sec / (60.0 / bpm)))
    bar_pulses = int(round(4.0 / pulse_ql))
    ks_idx = np.arange(len(bt), dtype=float)
    ons = np.array([float(n["onset"]) for n in notes])
    vel = np.array([float(n.get("velocity") or 64) for n in notes])
    ks = np.interp(ons, bt, ks_idx)
    best_p, best_s, best_mask = 0, -1.0, None
    for p in range(bar_pulses):
        d = np.abs((ks - p) % bar_pulses)
        d = np.minimum(d, bar_pulses - d)
        m = d < 0.35
        s = float((vel * m).sum())
        if s > best_s:
            best_s, best_p, best_mask = s, p, m

    # 亚脉冲对中已移除（2026-08-27 canon 事故第三弹）：δ 把簇钉在最近格点
    # 上、反而离拍整整 1 格，节拍先验窗口够不着——_prior_snap_units 对
    # 拍附近的簇（≤0.9 格）直接吸收，不再需要格内微调。
    def cont(t: float) -> float:
        return (float(np.interp(t, bt, ks_idx)) - best_p) * pulse_ql

    def ql_map(t: float) -> Fraction:
        return Fraction(max(_prior_snap_units(cont(t) * 12), 0), 12)

    def resid_map(t: float) -> Fraction:
        v = cont(t) * 12
        return Fraction(round(abs(v - _prior_snap_units(v)) * 2**20),
                        2**20)

    return {"ql_map": ql_map, "resid_map": resid_map, "cont": cont,
            "meta": {"pulse_ql": pulse_ql, "bar_phase_pulses": best_p,
                     "n_beats": int(len(bt))}}


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
    """节奏简化 Step2 时值再分配 v3（时值代价制）：onset 可靠、offset
    不可靠 → 时值由语境决定（记谱规则v1 §3.4 + Phase-1 v2/v3）：

    - 连奏（raw ≥ gap×0.85，实测 43%）：**精确填满**——最小代价片段链
      （单位分数优先、不穿小节线、≤3 片），不再留碎休止；
    - 断奏/中间态：**间隔 ≤1 拍同样填满**到声部内下一音头（休止符削减，
      v3）——演奏短奏属于表情层（Cemgil 2003：score 位置与表情偏差
      解耦），谱面写节拍值，听感重复音头由 score_mid 的 release 间隙
      保护；间隔 >1 拍（短语级沉默）保留单值缩短 + 休止——乐句结构
      在谱面上应可见。
    """
    stats = {"legato": 0, "detached": 0, "middle": 0}
    shifted = _ensure_bar_room(events, ql_per_measure)
    if shifted:
        logger.info("  [notation] bar-room shift: %d onsets", shifted)
    streams: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ev in events:
        streams[(ev["pitch"], ev["voice"])].append(ev)
    # 同声部"下一个不同音头"（跨音高）：填充上界不能压过声部内其他音——
    # 精确填满若只看同音高间隔，跨小节填满会与别的音高同声部重叠
    # （回归门 143 处跨度溢出的根因，2026-08-27）。
    voice_nexts: dict[int, list[Fraction]] = {}
    for ev in events:
        voice_nexts.setdefault(ev["voice"], []).append(ev["_onset_ql"])
    for v in voice_nexts:
        voice_nexts[v] = sorted(set(voice_nexts[v]))
    import bisect
    for group in streams.values():
        group.sort(key=lambda e: e["_onset_ql"])
        for ev, nxt in zip(group, group[1:] + [None]):
            raw_ql = _ql(ev["offset_sec"] - ev["onset_sec"], bpm)
            next_ql = nxt["_onset_ql"] if nxt is not None else None
            vocab = _context_vocab(ev["_onset_ql"], next_ql)
            m_idx = ev["_onset_ql"] // ql_per_measure
            room_bar = (m_idx + 1) * ql_per_measure - ev["_onset_ql"]
            # 声部内下一个不同音头（同时性和弦成员互相不算"下一个"）
            onsets = voice_nexts[ev["voice"]]
            i = bisect.bisect_right(onsets, ev["_onset_ql"])
            voice_gap = (onsets[i] - ev["_onset_ql"]) if i < len(onsets) else None
            if nxt is not None:
                gap_ql = nxt["_onset_ql"] - ev["_onset_ql"]
                if gap_ql <= 0:
                    gap_ql = MIN_DUR
                ratio = raw_ql / gap_ql
            else:
                gap_ql, ratio = None, LEGATO_RATIO  # 末音：按可填满处理但受 cap
            if ratio >= LEGATO_RATIO:
                stats["legato"] += 1
                # 填充上界：同音高间隔 ∧ 声部内不同音头 ∧ 一小节；无约束侧兜底
                bound = min(x for x in (gap_ql, voice_gap, SUSTAIN_CAP)
                            if x is not None) if (gap_ql or voice_gap) else SUSTAIN_CAP
                if nxt is None and voice_gap is None:  # 声部末音：不越小节线
                    bound = min(SUSTAIN_CAP, room_bar)
                ev["_frags_ql"] = _choose_fragments(ev["_onset_ql"], bound,
                                                    ql_per_measure, vocab)
            else:
                stats["detached" if ratio < DETACHED_RATIO else "middle"] += 1
                # 休止符削减（v3→v4，2026-08-29 第二阶段 #1）：middle 态
                # （0.5≤ratio<0.85）放宽到 ≤2 拍填满——offset 偏短的转写值
                # 走单值截断会在谱面留 1/12~1/2 拍无规律碎休止（全库 582
                # 处 ≤1 拍真休止的主来源，canon 案例 raw 1.2→写 1→尾巴
                # 1/3 变休止）；断奏（ratio<0.5）保持 ≤1 拍不吞（真顿挫）。
                fill_max = FILL_GAP_MAX * 2 if ratio >= DETACHED_RATIO else FILL_GAP_MAX
                if voice_gap is not None and voice_gap <= fill_max:
                    stats["gap_filled"] = stats.get("gap_filled", 0) + 1
                    bound = min(voice_gap, SUSTAIN_CAP)
                    ev["_frags_ql"] = _choose_fragments(
                        ev["_onset_ql"], bound, ql_per_measure, vocab)
                else:
                    ev["_frags_ql"] = [_largest_unit(min(raw_ql, room_bar),
                                                     vocab)]
    return stats


def build_track_events(cls: str, notes: list[dict], raw_onsets: dict[int, float],
                       bpm: float,
                       ql_map=None, resid_map=None) -> list[dict]:
    """量化后的音笔记谱事件（含 tie 片段与弦品）。

    管线：格点映射吸附（v3.1 = 全局精修+rubato 分段跟踪）→ 同时性聚类 →
    量化域守护 → 接缝合并 → 时值再分配 → 小节内单值分解（规则v1 §3.4）。
    ql_map/resid_map 缺省时退回绝对 1/12 网格（调试用）。
    """
    _assign_track_voices(cls, notes)

    def _default_map(t: float) -> Fraction:
        return max(_snap(_ql(t, bpm), QL_DENOM), Fraction(0))

    if ql_map is None:
        ql_map = _default_map
    if resid_map is None:
        resid_map = lambda t: abs(_ql(t, bpm) - _default_map(t))  # noqa: E731

    events = []
    for n in notes:
        onset_ql = ql_map(n["onset"])
        residual = resid_map(n["onset"])
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
    seams = 0
    if os.environ.get("MUSE_SEAM_MERGE"):  # 默认停用（2026-08-27 用户拍板
        # "忠于音头"）：本合并原意是融合"同一音被转写切段"的接缝，但吉他
        # 同音重复拨弦（音头间隔 0-40ms 的真再攻击）与接缝在纯音符边界
        # 下不可分——kyomu acoustic_guitar 实测被吞 408/940 音（43%），听感
        # 直接缺音头。保留代码仅作调试出口。
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
    """吉他 = 移调乐器（记谱比实音高八度，配 Treble8vb 谱表=标准吉他记谱）。
    贝斯不移调、配低音谱表（2026-08-27 视觉缺陷清单：+12 移调 + 普通高音
    谱表让贝斯音符挂 4-6 条下加线）。"""
    return 12 if cls in GUITAR_CLASSES else 0


def _clef_for(cls: str):
    """单谱表轨的谱表选择：吉他族 Treble8vb（配 +12 记谱）；贝斯族低音
    谱表；其余默认高音谱表。"""
    if cls in GUITAR_CLASSES:
        try:
            return m21clef.Treble8vbClef()
        except AttributeError:  # 老版本 music21 无别名
            c = m21clef.TrebleClef()
            return c
    if cls in BASS_CLASSES:
        return m21clef.BassClef()
    return m21clef.TrebleClef()


def _rest_pieces(gap: Fraction, vocab: list[Fraction] | None) -> list[Fraction]:
    """间隙 → 休止时值链。vocab=None（忠实）→ _legalize 细词表。

    量化模式（时值 v2）：词表去掉 1/12 后做**最小代价** DP（单位分数 0
    < 附点 1 < 三连音 2，同代价取片数少——如 9/12 休止优先 1/2+1/4 而非
    附点 3/4 单片，"分子尽量是 1"）；仅 gap 本身就是 1/12 时兜底单片。
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
    dp: list[tuple[int, int, list[Fraction]] | None] = [None] * (k + 1)
    dp[0] = (0, 0, [])
    for i in range(1, k + 1):
        for p in pieces:
            u = int(p / unit)
            if u <= i and dp[i - u] is not None:
                c, n, lst = dp[i - u]
                cand = (c + _dur_cost(p), n + 1, [p] + lst)
                if dp[i] is None or cand[:2] < dp[i][:2]:
                    dp[i] = cand
    if dp[k] is not None:
        return dp[k][2]
    return [gap]  # 兜底（gap < 最小片）


def _is_legal_single(d: Fraction) -> bool:
    """d 能否作为单个音符/休止符记谱（type+附点+可选 3:2 三连音）。

    MusicXML 里非合法单值（如 13/12、7/6 拍）music21 会导出成离奇连音
    比例（13:12），MuseScore 无法解析 → 小节算术崩坏（2026-08-25 卡农
    事件第二层根因）。这类值必须拆成 tie 链或改由休止符表达。

    注意 mult 不含 7/4（双附点）：双附点×三连音（7/12、7/24…）虽然
    理论可记（DUR_TO_TYPE 有正确映射），但 music21 会把它们导出成
    time-modification 12:7 → OSMD/MuseScore 渲染成荒谬连音比例
    （2026-08-28 用户"14/23 连音"事件；demo 里 14 处 12:7 全来自
    _absorb_tiny_gaps 把 1/2+1/12 并成 7/12）。纯双附点值词表不产、
    拆链兜底即可。
    """
    for base in (Fraction(1, 16), Fraction(1, 8), Fraction(1, 4),
                 Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4)):
        for mult in (Fraction(1), Fraction(3, 2)):
            for tm in (Fraction(1), Fraction(2, 3)):
                if base * mult * tm == d:
                    return True
    return False


_LEGAL_SINGLES = sorted({base * mult * tm
                         for base in (Fraction(1, 16), Fraction(1, 8),
                                      Fraction(1, 4), Fraction(1, 2),
                                      Fraction(1), Fraction(2), Fraction(4))
                         for mult in (Fraction(1), Fraction(3, 2))
                         for tm in (Fraction(1), Fraction(2, 3))})


def _split_legal(d: Fraction) -> list[Fraction]:
    """DP 精确拆成合法单值链（和恰为 d，片数最少）。

    以 1/48 四分音符为单位做最少片数 DP；合法单值中所有 /48 整数值都
    可作片段（含 1/12 兜底片），因此 /48 整数倍的 d 必有精确解。忠实
    模式 /48 网格与量化模式 /12 网格都被覆盖。
    """
    unit = Fraction(1, 48)
    k = d / unit
    if k.denominator != 1 or k <= 0:
        return [d]  # 非 /48 整数倍（理论不发生），交由调用方告警
    pieces_u = sorted({int(p / unit) for p in _LEGAL_SINGLES
                       if (p / unit).denominator == 1}, reverse=True)
    n = int(k)
    dp: list[list[int] | None] = [None] * (n + 1)
    dp[0] = []
    for i in range(1, n + 1):
        for pu in pieces_u:
            if pu <= i and dp[i - pu] is not None:
                cand = [pu] + dp[i - pu]
                if dp[i] is None or len(cand) < len(dp[i]):
                    dp[i] = cand
    if dp[n] is None:
        return [d]
    return [Fraction(pu, 48) for pu in dp[n]]


def _legalize_placed(placed: list[dict]) -> list[dict]:
    """兜底守护：placed 中出现非合法单值时值 → 拆 tie 链并告警。

    正常路径不应触发（词表值全合法 + 吸收器已门控）；防的是未来改动
    再引入复合值直接进 music21。Note/Chord/Rest 均支持 tie+duration，
    统一按同音 tie 链拆分。
    """
    import copy

    out: list[dict] = []
    for it in placed:
        d = it["dur"]
        if _is_legal_single(d):
            out.append(it)
            continue
        pieces = _split_legal(d)
        if len(pieces) <= 1:
            logger.warning("[notation] 非法时值 %s 无法精确拆分，原样保留", d)
            out.append(it)
            continue
        t = it["obj"].tie.type if it["obj"].tie is not None else None
        t_in = t in ("stop", "continue")
        t_out = t in ("start", "continue")
        off = it["offset"]
        for i, p in enumerate(pieces):
            obj = copy.deepcopy(it["obj"])
            obj.duration.quarterLength = p
            if i == 0:
                role = "continue" if t_in else "start"
            elif i == len(pieces) - 1:
                role = "start" if t_out else "stop"
            else:
                role = "continue"
            obj.tie = m21tie.Tie(role) if role else None
            out.append({"offset": off, "dur": p, "obj": obj})
            off += p
        logger.warning("[notation] 非法时值 %s 拆为 %d 片 tie 链", d, len(pieces))
    return out


def _absorb_tiny_gaps(placed: list[dict],
                      ql_per_measure: Fraction = Fraction(4)) -> None:
    """声部内孤立 1/12 间隙并进前音时值（~31ms@80bpm，视觉几乎不可见）。

    只延长前音/末音、绝不移动后续 onset——避免在连奏串中级联推移。
    仅当延长后的时值仍是合法单音值才吸收（否则留作 1/12 三连休止）：
    盲目 +1/12 会造出 13/12 类复合值，music21 导出 13:12 连音比例，
    MuseScore 直接小节算术崩坏（卡农事件教训）。
    """
    unit = Fraction(1, 12)
    placed.sort(key=lambda x: x["offset"])
    if placed and placed[0]["offset"] == unit:
        placed[0]["offset"] = Fraction(0)
    for prev, nxt in zip(placed, placed[1:]):
        gap = nxt["offset"] - (prev["offset"] + prev["dur"])
        if gap == unit and _is_legal_single(prev["dur"] + unit):
            prev["dur"] += unit
            prev["obj"].duration.quarterLength = prev["dur"]
    if placed:
        last = placed[-1]
        if (ql_per_measure - (last["offset"] + last["dur"]) == unit
                and _is_legal_single(last["dur"] + unit)):
            last["dur"] += unit
            last["obj"].duration.quarterLength = last["dur"]


def _legal_rest_chain(gap: Fraction, vocab: list[Fraction] | None) -> list[Fraction]:
    """休止片段链，保证每片都是合法单值（非法片经 _split_legal 展开）。"""
    out: list[Fraction] = []
    for piece in _rest_pieces(gap, vocab):
        if _is_legal_single(piece):
            out.append(piece)
        else:
            out.extend(_split_legal(piece))
    return out


def _fill_rests(items: list[dict], ql_per_measure: Fraction,
                vocab: list[Fraction] | None = None) -> list[tuple[Fraction, object]]:
    """items = [{offset, dur, obj(NotRest)}] → 补休止符的 (offset, obj) 列表。

    量化模式 vocab=REST_VOCAB（干净词表）；忠实模式 None → _legalize 链。
    所有休止片段保证为合法单值（非法值会让 MuseScore 小节算术崩坏）。
    """
    out = []
    cursor = Fraction(0)
    for it in sorted(items, key=lambda x: x["offset"]):
        if it["offset"] > cursor:
            off = cursor
            for piece in _legal_rest_chain(it["offset"] - cursor, vocab):
                r = m21note.Rest()
                r.duration.quarterLength = piece
                out.append((off, r))
                off += piece
        out.append((it["offset"], it["obj"]))
        cursor = it["offset"] + it["dur"]
    if cursor < ql_per_measure:
        off = cursor
        for piece in _legal_rest_chain(ql_per_measure - cursor, vocab):
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
        part.insert(0, _clef_for(cls))  # 贝斯=低音谱表、吉他=Treble8vb（2026-08-27）
        part_of_voice = {v: part for v in voices_used}
        parts = [part]

    def _split_lanes(items: list[dict]) -> list[dict]:
        """溢出声部分配：同 offset+同时值叠入同 lane（成和弦）；同声部不同
        音高时间重叠开新 lane——避免小节 overfull。"""
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
        return lanes

    # 声部号基址：大谱表两个 PartStaff 合并导出为单个 MusicXML part 时，
    # 声部号必须在 part 内全局唯一（MusicXML 语义）。两个谱表各自的溢出
    # lane 若都从 1 编号，MuseScore 会把同号声部跨谱表合并 → 区间重叠被
    # 推挤出小节 →「声部过长/不完整小节」刷屏（2026-08-25 卡农事件，
    # eval/validate_musicxml.py 为回归门）。上谱表 1..n_top，下谱表接续。
    voice_base: dict[int, int] = {}
    if is_keyboard:
        n_top = 1
        for mi in range(m_lo, m_hi + 1):
            items = [it for it in frag_items.get(1, []) if it["m"] == mi]
            if items:
                n_top = max(n_top, len(_split_lanes(items)))
        voice_base[2] = n_top

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
            # 溢出声部分配（lane 内同 offset+同时值成和弦）
            lanes = _split_lanes(items)
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
                voice.id = voice_base.get(v, 0) + li + 1
                placed = _legalize_placed(placed)
                _absorb_tiny_gaps(placed, ql_per_measure)
                for off, obj in _fill_rests(placed, ql_per_measure,
                                            REST_VOCAB if clean_rests else None):
                    voice.insert(off, obj)
                m.insert(0, voice)
            part.append(m)

    # 键盘族：只用单声部时（忠实模式恒 voice=1），另一谱表零小节会让
    # makeNotation=False 导出直接拒绝——补全休止小节（MuseScore 默认还会
    # 隐藏空谱表）。量化模式 voice_assign 双声部天然两表都有内容。
    if is_keyboard:
        for part in parts:
            have = {m.number for m in part.getElementsByClass(stream.Measure)}
            if have:
                continue
            for mi in range(m_lo, m_hi + 1):
                m = stream.Measure()
                m.number = mi - m_lo + 1
                r = m21note.Rest()
                r.duration.quarterLength = ql_per_measure
                m.insert(0, r)
                part.append(m)

    # 谱号落首小节：Part 层 clef 在 makeNotation=False 导出中不物化
    # （2026-08-27 canon 事故：钢琴双谱表全渲染成高音谱号，低音挂满加线）
    if is_keyboard:
        top.getElementsByClass(stream.Measure)[0].insert(0, m21clef.TrebleClef())
        bottom.getElementsByClass(stream.Measure)[0].insert(0, m21clef.BassClef())
    else:
        parts[0].getElementsByClass(stream.Measure)[0].insert(
            0, _clef_for(cls))

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
    # 元数据：占位 "Music21 Fragment" 标题会被导出进 PDF/PNG（2026-08-27
    # 视觉缺陷清单）——写乐曲名 + 轨道名 + 系统署名。
    try:
        from music21 import metadata as m21meta
        score.insert(0, m21meta.Metadata(
            title=f"{_SONG_TITLE or 'MUSE'} — {t_display_name(cls)}",
            composer="MUSE 自动记谱"))
    except Exception:
        logger.debug("score metadata skipped", exc_info=True)
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
        # TAB 谱行的休止符按制谱惯例隐藏（print-object 属性在 note 上；
        # MuseScore 里漂浮在 TAB 行下方成排的休止符就是它，2026-08-27）
        note.set("print-object", "no")
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
        items.sort(key=lambda x: (x["offset"], -x["pitch"]))
        # TAB 时值必须：合法单值 ∧ /48 整数（DIVISIONS=48）
        tab_legal = [v for v in _LEGAL_SINGLES if (v * DIVISIONS).denominator == 1]

        # 列聚类：所有声部的事件按时间摊平成"列"（≤1/12 内聚为一列，列内
        # 为和弦——TAB 不分声部，重叠的持续音由 staff part 的 tie 表达）。
        # 列宽 = max(成员时值)，且 clamp 到下一列 onset 与小节线——保证游程
        # 永不越界（声部串行化曾把游程推爆，2026-08-25 TAB 修复）。
        cols: list[list[dict]] = []
        for it in items:
            if cols and (it["offset"] - cols[-1][0]["offset"]) <= Fraction(1, 12):
                cols[-1].append(it)
            else:
                cols.append([it])

        cursor = Fraction(0)
        for ci, col in enumerate(cols):
            off = col[0]["offset"]
            if off > cursor:
                for piece in _legal_rest_chain(off - cursor, REST_VOCAB):
                    _tab_note_xml(m_el, dur=piece, rest=True)
                cursor = off
            room = ql_per_measure - off
            if ci + 1 < len(cols):
                room = min(room, cols[ci + 1][0]["offset"] - off)
            dur = min(max(g["dur"] for g in col), room)
            if dur not in tab_legal:
                cands = [v for v in tab_legal if v <= dur]
                dur = max(cands) if cands else Fraction(1, 12)
            longest = max(col, key=lambda g: g["dur"])
            for k, g in enumerate(sorted(col, key=lambda x: -x["pitch"])):
                _tab_note_xml(m_el, dur=dur, pitch=g["pitch"],
                              tie=longest["tie"] if k == 0 else g["tie"],
                              chord=k > 0,
                              string=g["string"], fret=g["fret"])
            cursor = off + dur
        if cursor < ql_per_measure:
            for piece in _legal_rest_chain(ql_per_measure - cursor, REST_VOCAB):
                _tab_note_xml(m_el, dur=piece, rest=True)
    return part


def _write_musicxml_exact(score, path: str) -> None:
    """makeNotation=False 精确导出。

    小节/声部/休止/时值已由本层自建完备；music21 默认导出会跑 makeNotation
    重排（自动补休止、拆音符），曾把合法 tie 链改造成 13/12 类复合时值单音
    → music21 写出离奇连音比例 → MuseScore 小节算术崩坏（2026-08-25 卡农
    事件第三层根因）。内容均为 written pitch，故 atSoundingPitch=False。
    """
    from music21.musicxml import m21ToXml

    score.atSoundingPitch = False
    sx = m21ToXml.ScoreExporter(score, makeNotation=False)
    root = sx.parse()
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


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


def _duration_stats(out_tracks: list[dict]) -> dict:
    """时值 v2/v3 验收统计：片段代价值分布 + 连音链率 + 休止符普查。

    休止计数 = 声部内相邻事件"写值终点 → 下一音头"的正间隙个数（v3
    削减前后对比口径；MusicXML 里的实体休止符与之同源）。
    """
    n_frag = {"unit": 0, "dotted": 0, "triplet": 0, "other": 0}
    n_events = chained = n_rests = 0
    for t in out_tracks:
        by_voice: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for ev in t["events"]:
            n_events += 1
            if len(ev["frags"]) > 1:
                chained += 1
            for f in ev["frags"]:
                c = _dur_cost(Fraction(f["dur"]))
                n_frag["unit" if c == 0 else
                        "dotted" if c == 1 else
                        "triplet" if c == 2 else "other"] += 1
            f0 = ev["frags"][0]
            on = float(f0["bar"]) * 4 + float(f0["offset"])
            end = on + sum(float(Fraction(f["dur"])) for f in ev["frags"])
            by_voice[int(ev.get("voice", 1))].append((on, end))
        for spans in by_voice.values():
            spans.sort()
            n_rests += sum(1 for (_, e0), (on1, _) in zip(spans, spans[1:])
                           if on1 - e0 > 1e-9)
    tot = sum(n_frag.values()) or 1
    return {**{k: v for k, v in n_frag.items()},
            "unit_pct": round(n_frag["unit"] / tot, 4),
            "dotted_pct": round(n_frag["dotted"] / tot, 4),
            "triplet_pct": round(n_frag["triplet"] / tot, 4),
            "tie_chained_event_pct": round(chained / max(1, n_events), 4),
            "n_events": n_events,
            "n_phrase_rests": n_rests,
            "phrase_rest_per_event": round(n_rests / max(1, n_events), 4)}


def _export_time_map(rec: dict, pooled: list[dict]) -> list[list[float]]:
    """格点映射 cont(t) → 分段线性采样表 [[t 秒, QL], …]（0.5s 步长）。

    前端谱面光标用它把播放时间换算成谱面位置（QL 单调递增，可逆插值）。
    """
    import numpy as np
    cont = rec["cont"]
    ts = [float(n["onset"]) for n in pooled]
    t0, t1 = min(ts), max(ts)
    grid = np.arange(t0, t1 + 0.5, 0.5)
    if len(grid) < 2:
        grid = np.array([t0, t0 + 1.0])
    return [[round(float(t), 3), round(float(cont(float(t))), 4)]
            for t in grid]


def _export_score_mids(out_tracks: list[dict], bpm: float, output_dir: str,
                       ql_per_measure: Fraction = Fraction(4)) -> list[str]:
    """准入轨道 → notation/score_mid/{cls}.mid（Phase-1：播放与谱面同源）。

    每事件一个持续 MIDI 音（连音链/小节切分只是记谱概念，听感上合并）；
    onset/dur 取谱面量化值（bar·小节长+小节内偏移 → QL → 秒）。
    稀疏轨道已在 build_notation 准入层统一剔除——mid/谱/播放三处同源。
    """
    try:
        import pretty_midi
        import src.ia_amt_frontend as _ia  # 其模块级 sys.path 注入使 taxonomy 可导入
        from instrument_agnostic_amt.taxonomy.instrument_classes import (
            INSTRUMENT_CLASSES, get_program_number_from_class_id)
        from src.multi_instrument import CLASS_PROGRAM_OVERRIDES
    except Exception:
        logger.warning("[notation] pretty_midi/程序号映射不可用，score_mid 跳过",
                       exc_info=True)
        return []
    out_dir = os.path.join(output_dir, "notation", "score_mid")
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for t in out_tracks:
        cls = t["instrument_class"]
        if cls == "drums":
            continue
        class_id = (INSTRUMENT_CLASSES.index(cls)
                    if cls in INSTRUMENT_CLASSES else None)
        program = (0 if class_id is None else
                   CLASS_PROGRAM_OVERRIDES.get(
                       cls, int(get_program_number_from_class_id(class_id))))
        pm = pretty_midi.PrettyMIDI(resolution=1920, initial_tempo=bpm)
        pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
        inst = pretty_midi.Instrument(program=program, name=cls)
        spql = 60.0 / bpm  # 秒/四分音符
        # 同音高连排起音保护：填满写法使相邻同音高事件在 MIDI 里首尾相接
        # 连成长音（听感=漏掉重复音头），起音前留 RELEASE_SEC 重新触发攻击
        next_same_on: dict[int, float] = {}
        by_vp: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
        for i, ev in enumerate(t["events"]):
            by_vp[(int(ev.get("voice", 1)), int(ev["pitch"]))].append(
                (float(ev.get("onset_sec", 0.0)), i))
        for lst in by_vp.values():
            lst.sort()
            for (_, i), (on_next, _) in zip(lst, lst[1:]):
                next_same_on[i] = on_next
        for i, ev in enumerate(t["events"]):
            frags = ev["frags"]
            if not frags:
                continue
            # 音头 = 原始检测秒（rubato 曲目谱面是名义网格、播放对齐原曲
            # 音频——两者解耦；时值 = 谱面名义片段和）
            start = float(ev.get("onset_sec", 0.0))
            dur_ql = sum(float(Fraction(f["dur"])) for f in frags)
            end = start + dur_ql * spql
            if i in next_same_on:
                end = min(end, max(next_same_on[i] - RELEASE_SEC,
                                   start + 0.023))
            inst.notes.append(pretty_midi.Note(
                velocity=max(1, min(127, int(ev.get("velocity", 100)))),
                pitch=int(ev["pitch"]), start=start,
                end=max(end, start + 0.023)))
        pm.instruments.append(inst)
        pm.write(os.path.join(out_dir, f"{cls}.mid"))
        written.append(f"score_mid/{cls}.mid")
        logger.info("  [notation] score_mid/%s.mid (%d events)", cls,
                    len(t["events"]))
    return written


def _post_export_gate(xml_paths: list[str], loud: bool = True) -> dict:
    """导出即校验（2026-08-25 建立）：所有 MusicXML 落盘后立刻跑
    eval/validate_musicxml.py 的四项检查（跨度/重叠/跨谱表/合法时值）。

    历史：两轮 MuseScore 报错（86→66→0）的教训是 music21 回读是假阴性
    QA、人工抽查不可依赖——校验必须自动发生在每次导出后。有问题时
    logger.error + 记入 notation.json，不静默出货。
    """
    summary = {"files": len(xml_paths), "problems": 0, "detail": {}}
    try:
        # 注意：不能 `from eval.validate_musicxml import ...`——eval/ 目录下的
        # eval.py 同名模块会在 sys.path[0]=脚本目录时遮蔽 eval 包（实测踩坑），
        # 按文件路径直接加载校验器，免疫任何打包/路径环境。
        import importlib.util as _ilu
        _vp = Path(__file__).resolve().parents[1] / "eval" / "validate_musicxml.py"
        _spec = _ilu.spec_from_file_location("_muse_musicxml_validator", _vp)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        count_problems = _mod.count_problems
    except Exception:
        logger.warning("[notation] 校验器不可用，跳过导出后校验", exc_info=True)
        return summary
    for p in xml_paths:
        if not os.path.exists(p):
            continue
        try:
            n = count_problems(p)
            summary["problems"] += n
            summary["detail"][os.path.basename(p)] = n
            if n and loud:
                logger.error("[notation] 导出校验失败 %s: %d 处问题，"
                             "跑 eval/validate_musicxml.py %s 看明细",
                             os.path.basename(p), n, p)
        except Exception:
            logger.warning("[notation] 校验 %s 异常", p, exc_info=True)
    if summary["problems"] == 0:
        logger.info("[notation] 导出校验通过：%d 个文件 0 问题", len(xml_paths))
    return summary


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

    # 音头映射（时值 v3.2）：双候选择优——①音频拍同步映射（rubato 解，
    # beat_times 来自 src/beat_track.py，multi_instrument 注入 notes_json）
    # ②onset 格点拟合映射（全局精修+分段跟踪）。按"半拍网格浓度"选：
    # 连续 QL 到最近半拍的平均距离（真实音乐的 onset 集中在拍/半拍上，
    # 映射错旋则趋于均匀 0.125；canon 实测纯 onset 拟合在 rubato 抹花下
    # 无浓度可言，音频提拍 std 16ms 扛得住）。
    bpm_detected = bpm
    lattice = _fit_grid_map(pooled, bpm)
    cands = []
    beat_times = notes_json.get("beat_times") or []
    if len(beat_times) >= 16 and notes_json.get("pulse_sec"):
        bt = _beat_sync_map(pooled, beat_times,
                            float(notes_json["pulse_sec"]), bpm)
        if bt:
            cands.append(("beat-sync", bt))
    cands.append(("piecewise-lattice", lattice))

    def _halfbeat_score(rec) -> float:
        ds = []
        for n in pooled:
            x = abs(float(rec["cont"](raw_onsets[id(n)])) % 0.5)
            ds.append(min(x, 0.5 - x))
        ds.sort()
        return ds[len(ds) // 2]  # 中位（抗离群）

    for name, rec in cands:
        rec["score"] = _halfbeat_score(rec)
        rec["kind"] = name
    map_kind, rec = min(cands, key=lambda r: r[1]["score"])
    ql_map, resid_map, qmeta = rec["ql_map"], rec["resid_map"], rec["meta"]
    logger.info("  [notation] grid map: %s (半拍中位距 %.3f QL；候选 %s)",
                map_kind, rec["score"],
                ", ".join(f"{n}={r['score']:.3f}" for n, r in cands))

    # bpm 精修仅在格点拟合获胜时采纳（拍同步映射的速度语义由 pulse_ql
    # 承载；rubato 曲目的格点精修易踩混叠假峰）
    bpm_ref = lattice["meta"].get("bpm_refined", bpm)
    if map_kind == "piecewise-lattice" and abs(bpm_ref - bpm) > 0.05:
        logger.info("  [notation] tempo refine: %.2f → %.2f (%+.2f%%)",
                    bpm, bpm_ref, lattice["meta"]["tempo_delta_pct"])
        bpm = bpm_ref
        notes_json["bpm"] = round(bpm_ref, 2)  # notes.json 已落盘 → 回写
        try:
            _np = os.path.join(output_dir, "notes.json")
            with open(_np, "r", encoding="utf-8") as f:
                _disk = json.load(f)
            if _disk.get("bpm") != notes_json["bpm"]:
                _disk["bpm"] = notes_json["bpm"]
                with open(_np, "w", encoding="utf-8") as f:
                    json.dump(_disk, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("  [notation] notes.json bpm 回写失败", exc_info=True)
    key_sig = estimate_key(pooled)

    # T1 和声先验层：和弦跟踪（失败不影响谱面导出）
    harmony_segments: list[dict] | None = None
    try:
        from src.harmony_prior import track_chords
        end_ql = _ql(max(n["offset"] for n in pooled), bpm)
        harmony_segments = track_chords(pooled, bpm, key_sig, end_ql)
    except Exception:
        logger.exception("  [harmony] chord tracking failed")

    # 量化健康度：音头漂移统计（吸附位移）+ 半拍浓度（结构代理）。
    # 旧 snap_rate 对 1/12 稠密网格恒为 1（吸附距离 ≤ 半格 34ms 是格点
    # 密度性质、与对不对齐无关——已废弃为空指标，2026-08-27 canon 事故）。
    drift_ms: list[float] = []
    for n in pooled:
        d = resid_map(raw_onsets[id(n)])
        # resid 单位 = 1/12 格（twelfth），换算 QL 需 /12（曾漏除 12 导致
        # v3.1 起漂移虚报 12 倍：canon "185ms" 实为 15.4ms）
        drift_ms.append(float(d) / 12 * 60.0 / bpm * 1000)
    drift_ms.sort()
    onset_drift = {
        "median_ms": round(drift_ms[len(drift_ms) // 2], 1),
        "p90_ms": round(drift_ms[min(len(drift_ms) - 1,
                                     int(len(drift_ms) * 0.9))], 1),
        "max_ms": round(drift_ms[-1], 1),
    }
    halfbeat_med = round(rec["score"], 4)
    logger.info("  [notation] onset_drift=%s halfbeat_med=%.3f QL (bpm=%.2f)",
                onset_drift, rec["score"], bpm)
    if halfbeat_med > 0.15:
        logger.warning("  [notation] 半拍浓度弱（%.3f）——拍/速度可能失准",
                       halfbeat_med)

    out_tracks = []
    os.makedirs(os.path.join(output_dir, "notation", "solo"), exist_ok=True)

    for ti, t in enumerate(tracks):
        cls = t["instrument_class"]
        if cls == "drums":
            logger.info("  [notation] drums 谱 v1 不含，跳过")
            continue
        events = build_track_events(cls, t["notes"], raw_onsets, bpm,
                                     ql_map, resid_map)
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
        # 真实时间→谱面 QL 映射（分段线性采样，rubato 曲目谱面光标/滚动
        # 用它，不再按名义 bpm 匀速跑——"歌没放一半谱滚完了"的根因）
        "time_map": _export_time_map(rec, pooled),
        "meta": {"quantizer": map_kind,
                 "halfbeat_med_ql": halfbeat_med,
                 "bpm_detected": round(bpm_detected, 2),
                 "bpm_refined": lattice["meta"].get("bpm_refined",
                                                    round(bpm, 2)),
                 "tempo_delta_pct": lattice["meta"].get("tempo_delta_pct",
                                                       0.0),
                 **{k: v for k, v in qmeta.items()
                    if k not in ("bpm_refined", "tempo_delta_pct")},
                 "onset_drift_ms": onset_drift},
        "duration_stats": _duration_stats(out_tracks),
        "tracks": out_tracks,
    }
    score_mids = _export_score_mids(out_tracks, bpm, output_dir, ql_per_measure)
    if score_mids:
        notation["score_mids"] = score_mids
    with open(os.path.join(output_dir, "notation", "notation.json"), "w",
              encoding="utf-8") as f:
        json.dump(notation, f, ensure_ascii=False, indent=1)
    logger.info("  [notation] notation.json written (%d tracks, dur_stats=%s)",
                len(out_tracks), notation["duration_stats"])

    # 单乐器谱导出（量化模式）
    exported: list[str] = []
    for t in out_tracks:
        cls = t["instrument_class"]
        transposition = _transpose_semitones(cls)
        stem = os.path.join(output_dir, "notation", "solo", cls)
        try:
            score = assemble_solo_score(cls, t["events"], bpm, notation["key"],
                                        transposition, ql_per_measure, tsig,
                                        harmony_segments)
            _write_musicxml_exact(score, stem + ".musicxml")
            if cls in GUITAR_CLASSES:
                tab = _build_tab_part_xml(t["events"], transposition,
                                          ql_per_measure, tsig,
                                          part_id=f"Ptab-{cls}")
                _merge_tab_into_musicxml(stem + ".musicxml", tab)
            n_meas = len(list(score.recurse().getElementsByClass(stream.Measure)))
            logger.info("  [notation] solo/%s.musicxml (%d events, %d measures)",
                        cls, len(t["events"]), n_meas)
            exported.append(stem + ".musicxml")
        except Exception:
            logger.exception("  [notation] solo export failed: %s", cls)

    if mode in ("faithful", "both"):
        exported += _export_faithful(tracks, raw_onsets, bpm, ql_per_measure,
                                     notation, output_dir, tsig) or []

    # 导出即校验（量化谱必须 0 问题；忠实模式 1/48 精度按设计会告警，单独口径）
    if exported:
        quantized_files = [p for p in exported if "faithful" not in p]
        faithful_files = [p for p in exported if "faithful" in p]
        notation["validation"] = {
            "quantized": _post_export_gate(quantized_files),
            **({"faithful_by_design": _post_export_gate(faithful_files, loud=False)}
               if faithful_files else {}),
        }
        with open(os.path.join(output_dir, "notation", "notation.json"), "w",
                  encoding="utf-8") as f:
            json.dump(notation, f, ensure_ascii=False, indent=1)
    return notation


def _export_faithful(tracks: list[dict], raw_onsets: dict[int, float], bpm: float,
                     ql_per_measure: Fraction, notation: dict, output_dir: str,
                     time_signature: str = "4/4") -> list[str]:
    """忠实模式：不做分段网格量化，位置/时值取原始值（1/48 网格表达），
    时值经 _legalize 分解为合法 tie 链——谱面会出现非标准时值与碎休止，
    这正是忠实模式的语义。单声部（不做人声部分Split）。返回导出路径。"""
    exported: list[str] = []
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
            _write_musicxml_exact(score, os.path.join(
                output_dir, "notation", "solo", f"{cls}.faithful.musicxml"))
            logger.info("  [notation] solo/%s.faithful.musicxml", cls)
            exported.append(os.path.join(
                output_dir, "notation", "solo", f"{cls}.faithful.musicxml"))
        except Exception:
            logger.exception("  [notation] faithful export failed: %s", cls)
    return exported
