"""Voice / stream assignment for notes（v2，2026-08-29 编排算法重设计）。

设计依据：汇报/2026-08-29-编排算法设计（钢琴双手+TAB排指）.md
- 论文：Nakamura et al. Merged-Output HMM (ISMIR 2014)、Takamori et al.
  (SMC 2017)、Sayegh 最优路径传统（arXiv 2408.05024 综述）、Chen (ISMIR 2020)

Piano（v2）：列化 + 分层 DP（beam search）。
  每只手携带"当前音高中心"状态，跨过对方手的音符持续存在（merged-output
  HMM 的结构），手的舒适区随音乐游走——取代 v1 全曲中位数一刀切。
  代价 = 手跨度罚（>14 半音，Takamori 10 度约束软化）+ 中心游走罚 +
  手指数罚（单手 >5 音）+ 切分点跳变罚。和弦按列整组归手，不再被劈开。
Guitar：delegate to guitar_tab.assign_guitar_fingering()（列级 DP v2）。

回退：环境变量 MUSE_ARRANGE_V1=1 切回 v1 实现（对比/回归用）。
"""
import logging
import os

import numpy as np

from src import guitar_tab

logger = logging.getLogger(__name__)

# Instruments that need voice splitting
VOICED_INSTRUMENTS = {"piano"}

# v2 双手分配权重（设计文档 §2.3）
COL_TOL_SEC = 0.03      # 同列 onset 容差（与 multi_instrument 和弦窗对齐取保守）
HAND_SPAN_MAX = 14      # 单手同列音域软阈值（半音，≈10 度，Takamori SMC2017）
HAND_SPAN_CAP = 19      # 硬上限（超过连琶音化都难，候选不生成）
W_SPAN = 6.0            # 每超 1 半音的跨度罚（软；3.0 时先验压不住，宽和弦泛滥）
W_WANDER = 0.3          # 手中心游走罚（每半音）
W_OVERLOAD = 5.0        # 单手 >5 音的每音罚（一只手 5 根手指）
W_SWITCH = 2.0          # 切分点（RH 最低音）跳变 >12 半音的罚
SWITCH_LEAP = 12
EMA_ALPHA = 0.5         # 手中心指数平滑
BEAM_WIDTH = 128        # 每列保留状态数上限
# v2.1（canon 前 4 小节倒挂反馈）：谱表域先验罚——上谱表收 <C4 音、
# 下谱表收 >C5 音都要付代价，低音段整段归左手（上谱表空置）成为最优解
# 而非"音符平分"。注意不钳中心：钳中心会在中音区制造虚构游走代价，
# 把本属左手的音推去右手（实测 canon 全曲 1406/687 倒向）。
STAFF_C4, STAFF_C5 = 60, 72
W_STAFF_RANGE = 0.4     # 越出谱表舒适域的每半音罚（0.15 压不过跨度罚，E2 回流）


def assign_voices(notes: list[dict], instrument: str = "piano") -> list[dict]:
    """
    Assign voice IDs to notes.

    Piano → voice 1 (upper/right-hand staff) / voice 2 (lower/left-hand staff)
    via column DP with per-hand pitch-center states (v2).
    Guitar → call guitar_tab.assign_guitar_fingering() for string/fret.

    Args:
        notes: list of note dicts ("pitch"; piano 还用 "onset")

    Returns:
        notes with "voice" field added (int, 1-based)
    """
    if instrument == "guitar":
        return guitar_tab.assign_guitar_fingering(notes)

    if instrument not in VOICED_INSTRUMENTS:
        for n in notes:
            n["voice"] = 1
        return notes

    if len(notes) < 2:
        for n in notes:
            n["voice"] = 1
        return notes

    if os.environ.get("MUSE_ARRANGE_V1"):
        return _assign_v1(notes)
    return _assign_piano_v2(notes)


# ---------------------------------------------------------------------------
# v1：全曲音高中位数一刀切（保留作回退/对比）
# ---------------------------------------------------------------------------

def _assign_v1(notes: list[dict]) -> list[dict]:
    pitches = [n["pitch"] for n in notes]
    median_pitch = np.median(pitches)
    for n in notes:
        n["voice"] = 1 if n["pitch"] >= median_pitch else 2
    upper = sum(1 for n in notes if n["voice"] == 1)
    logger.info(f"Voice assignment v1 (piano): upper={upper}, "
                f"lower={len(notes) - upper}")
    return notes


# ---------------------------------------------------------------------------
# v2：列化 + 双手中心分层 DP（beam search）
# ---------------------------------------------------------------------------

State = tuple  # (cost, cand, centerR, centerL, parent_idx)


def _columnize(notes: list[dict], tol: float) -> list[list[dict]]:
    """按 onset 聚列：列内为同时发声（和弦），列间按时间、列内降音高。

    列内必须再按音高降序排一次：分组排序键 (onset,-pitch) 在真实转写
    的毫秒级 onset 抖动下列内顺序是乱的——"前 k 个给右手"的候选语义
    曾因此整体失效（canon 声部倒挂/低音进上谱表的顽固残留根因，
    2026-08-29 用户"下加好几间"反馈追查）。"""
    ordered = sorted(notes, key=lambda n: (n.get("onset", 0.0), -n["pitch"]))
    cols: list[list[dict]] = []
    for n in ordered:
        if cols and n.get("onset", 0.0) - cols[-1][0].get("onset", 0.0) <= tol:
            cols[-1].append(n)
        else:
            cols.append([n])
    for col in cols:
        col.sort(key=lambda n: -n["pitch"])
    return cols


def _candidates(m: int) -> list[tuple[int, ...]]:
    """列切分候选：位置 i 的值 1=右手 2=左手（列内已降音高）。

    基本候选 = 连续二分（高 k 音给右手）；交叉候选 = 单音翻转（仅
    最低音给右手 / 仅最高音给左手——真正的交叉手场景；首版曾把翻转
    下标写反变成基本候选的重复，未提供任何交叉选项）。
    单手跨度 >HAND_SPAN_CAP（≈10 度+）或单手 >5 音（手指数）的候选
    **不生成**（物理不可弹，硬约束）；列太大（>10 音）无合法切分时
    退回全量候选（软罚兜底）。
    """
    cands = [tuple(1 if i < k else 2 for i in range(m))
             for k in range(m + 1)]
    if m >= 3:
        cross_low_to_rh = [2] * m
        cross_low_to_rh[m - 1] = 1   # 最低音单独给右手（右手下探）
        cands.append(tuple(cross_low_to_rh))
        cross_high_to_lh = [1] * m
        cross_high_to_lh[0] = 2      # 最高音单独给左手（左手上探）
        cands.append(tuple(cross_high_to_lh))
    return cands


def _playable(cand: tuple[int, ...], col: list[dict]) -> bool:
    """硬约束：单手同列跨度 ≤ HAND_SPAN_CAP 且 ≤5 音（手指数）。

    14~19 半音的宽和弦（canon 开头左手 10 度+配置）真实存在——只付软罚
    不禁候选；>19 连琶音化都难，直接不生成。硬禁 14 曾把 E2 逼进上谱表
    （canon 前 4 小节倒挂的用户反馈根因）。
    """
    for hand in (1, 2):
        ps = [col[i]["pitch"] for i, v in enumerate(cand) if v == hand]
        if not ps:
            continue
        if max(ps) - min(ps) > HAND_SPAN_CAP or len(ps) > 5:
            return False
    return True


def _cands_for(col: list[dict]) -> list[tuple[int, ...]]:
    """列的合法候选（硬约束过滤）；无合法解时退回：C4 阈值切分优先
    （转写噪声超宽和弦的人工制谱直觉：低音归下谱表），再退全量软罚。"""
    all_c = _candidates(len(col))
    ok = [c for c in all_c if _playable(c, col)]
    if ok:
        return ok
    by_pitch = tuple(1 if n["pitch"] >= 60 else 2 for n in col)
    if by_pitch in all_c:
        return [by_pitch]
    return all_c


def _boundary_pitch(cand: tuple[int, ...], col: list[dict]) -> float | None:
    """切分点音高 = 右手组最低音（右手无音则 None）。"""
    rh = [col[i]["pitch"] for i, v in enumerate(cand) if v == 1]
    return min(rh) if rh else None


def _step_cost(cand: tuple[int, ...], col: list[dict],
               pcR: int, pcL: int,
               prev_boundary: float | None) -> tuple[float, int, int]:
    """单列局部代价 + EMA 更新后的新中心（v2.1：中心钳制+谱表域先验）。"""
    rh = [col[i]["pitch"] for i, v in enumerate(cand) if v == 1]
    lh = [col[i]["pitch"] for i, v in enumerate(cand) if v == 2]
    cost = 0.0
    newR, newL = pcR, pcL
    for grp, center, is_rh in ((rh, pcR, True), (lh, pcL, False)):
        if not grp:
            continue  # 该手本列无音：中心冻结（merged-output 的状态保持）
        span = max(grp) - min(grp)
        if span > HAND_SPAN_MAX:
            cost += W_SPAN * (span - HAND_SPAN_MAX)
        if len(grp) > 5:
            cost += W_OVERLOAD * (len(grp) - 5)
        mean = sum(grp) / len(grp)
        cost += W_WANDER * abs(mean - center)
        # 谱表域先验逐音计（组均值版会让和弦 {C6,E2} 均值正常把 E2 藏进
        # 上谱表——canon 仍有 191 个 <C4 音挂 4~7 条下加线的洞，2026-08-29
        # 用户"下加好几间/上加好几间"反馈）
        if is_rh:
            cost += sum(W_STAFF_RANGE * (STAFF_C4 - p)
                        for p in grp if p < STAFF_C4)
        else:
            cost += sum(W_STAFF_RANGE * (p - STAFF_C5)
                        for p in grp if p > STAFF_C5)
        new = int(round(EMA_ALPHA * mean + (1 - EMA_ALPHA) * center))
        if is_rh:
            newR = new
        else:
            newL = new
    boundary = min(rh) if rh else None
    if (prev_boundary is not None and boundary is not None
            and abs(boundary - prev_boundary) > SWITCH_LEAP):
        cost += W_SWITCH
    return cost, newR, newL


def _assign_piano_v2(notes: list[dict]) -> list[dict]:
    cols = _columnize(notes, COL_TOL_SEC)
    pitches = sorted(n["pitch"] for n in notes)
    cR0 = int(np.percentile(pitches, 75))
    cL0 = int(np.percentile(pitches, 25))

    # 分层 DP：layers[ci] = 该列保留的 state 列表（parent_idx 指向上一层）
    layers: list[list[State]] = []
    first: list[State] = []
    for cand in _cands_for(cols[0]):
        cost, cR, cL = _step_cost(cand, cols[0], cR0, cL0, None)
        first.append((cost, cand, cR, cL, -1))
    layers.append(_trim(first))

    for ci in range(1, len(cols)):
        col, prev_col = cols[ci], cols[ci - 1]
        nxt: list[State] = []
        for pi, (pcost, pcand, pcR, pcL, _) in enumerate(layers[-1]):
            pb = _boundary_pitch(pcand, prev_col)
            for cand in _cands_for(col):
                c2, cR, cL = _step_cost(cand, col, pcR, pcL, pb)
                nxt.append((pcost + c2, cand, cR, cL, pi))
        layers.append(_trim(nxt))

    # 回溯最优路径 → 写 voice
    si = min(range(len(layers[-1])), key=lambda i: layers[-1][i][0])
    path: list[tuple[int, ...]] = []
    for ci in range(len(layers) - 1, -1, -1):
        st = layers[ci][si]
        path.append(st[1])
        si = st[4]
    path.reverse()
    for col, cand in zip(cols, path):
        for note, v in zip(col, cand):
            note["voice"] = v

    upper = sum(1 for n in notes if n["voice"] == 1)
    switches = sum(1 for a, b in zip(path, path[1:]) if a != b)
    logger.info(f"Voice assignment v2 (piano): cols={len(cols)}, "
                f"upper={upper}, lower={len(notes) - upper}, "
                f"split-switches={switches}")
    return notes


def _trim(states: list[State]) -> list[State]:
    """按 (cand, cR, cL) 去重保最小代价，再按代价截断 beam。"""
    best: dict[tuple, State] = {}
    for st in states:
        key = (st[1], st[2], st[3])
        if key not in best or st[0] < best[key][0]:
            best[key] = st
    return sorted(best.values(), key=lambda s: s[0])[:BEAM_WIDTH]
