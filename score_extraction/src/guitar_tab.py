"""吉他 TAB 排指（v2，2026-08-29 编排算法重设计）。

设计依据：汇报/2026-08-29-编排算法设计（钢琴双手+TAB排指）.md
- 论文：Sayegh 最优路径传统（arXiv 2408.05024 综述的标准基线）、
  Chen et al. (ISMIR 2020)、Sakai (SMC 2024)

v1 问题：逐音线性 DP → 和弦可能两音同弦（物理不可弹）、无把位概念
（3 品↔15 品乱跳）。v2 = 列级 DP：
  - 列 = 同时发声（≤30ms 聚列），列内候选组合**硬约束**过滤：
    同弦互斥 + 非空弦手跨度 ≤4 品（横按+自然伸展）
  - 代价 = 把位中心移动（主项，Sayegh）+ 低把位先验 + 空弦奖励
    （config.OPEN_STRING_BIAS）+ 同弦延续奖励（旋律线保持在同弦）
  - DP 全局最优，回溯写 string/fret（接口与 v1 一致：返回列表与输入
    notes 顺序一一对应，调用方按位置 zip 写回）

回退：环境变量 MUSE_ARRANGE_V1=1 切回 v1。
"""
import logging
import os

from src.config import config

logger = logging.getLogger(__name__)

OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]  # low E -> high E

# v2 权重（设计文档 §3.2；FRET/STRING/OPEN 沿用 config）
COL_TOL_SEC = 0.03
SPAN_MAX = 4            # 列内非空弦手跨度上限（品）
W_POS = 1.0             # 把位中心移动罚（每品）——Sayegh 主项
W_LOW = 0.5             # 低把位先验（把位中心每品）
W_STRING_KEEP = 0.3     # 同音高跨列同弦延续奖励
MAX_COL_CANDS = 64      # 列候选组合上限（防爆炸）
PER_NOTE_CANDS = 4      # 每音保留 (弦,品) 候选数上限


def get_positions(midi_pitch: int) -> list[tuple[int, int]]:
    positions = []
    for s, open_midi in enumerate(OPEN_STRING_MIDI):
        fret = midi_pitch - open_midi
        if 0 <= fret <= config.MAX_FRET:
            positions.append((s, fret))
    return positions


def _assign_v2(notes: list[dict]) -> list[dict]:
    if not notes:
        return []

    # 列化（索引制，保持与输入 notes 的位置对应）
    order = sorted(range(len(notes)),
                   key=lambda i: (notes[i].get("onset", 0.0),
                                  -notes[i]["pitch"]))
    cols: list[list[int]] = []
    for i in order:
        t = notes[i].get("onset", 0.0)
        if cols and t - notes[cols[-1][0]].get("onset", 0.0) <= COL_TOL_SEC:
            cols[-1].append(i)
        else:
            cols.append([i])

    cand_lists = [_col_candidates([notes[i] for i in col]) for col in cols]

    # DP：dp[i][j] = 第 i 列选第 j 候选的最小累计代价
    INF = float("inf")
    dp = [[INF] * len(cl) for cl in cand_lists]
    bp = [[-1] * len(cl) for cl in cand_lists]
    for j, (c, _) in enumerate(cand_lists[0]):
        dp[0][j] = c

    for i in range(1, len(cols)):
        col_pitches = [notes[idx]["pitch"] for idx in cols[i]]
        pl = cand_lists[i - 1]
        # 每个前驱候选的 pitch→string 表（同弦延续奖励按实际前驱算）
        maps = [{p: s for p, (s, _) in zip(col_pitches, combo)}
                for _, combo in pl]
        for j, (cj, combo) in enumerate(cand_lists[i]):
            pos_j = _pos_center(combo)
            for k in range(len(pl)):
                if dp[i - 1][k] == INF:
                    continue
                prev_combo = pl[k][1]
                pos_k = _pos_center(prev_combo)
                keep = sum(1 for p, (s, _) in zip(col_pitches, combo)
                           if s >= 0 and maps[k].get(p) == s)
                trans = (W_POS * abs(pos_j - pos_k)
                         - W_STRING_KEEP * keep)
                tot = dp[i - 1][k] + trans + cj
                if tot < dp[i][j]:
                    dp[i][j] = tot
                    bp[i][j] = k

    # 回溯
    best_j = min(range(len(cand_lists[-1])), key=lambda j: dp[-1][j])
    chain = [best_j]
    for i in range(len(cols) - 1, 0, -1):
        chain.append(bp[i][chain[-1]])
    chain.reverse()

    result = [dict(n) for n in notes]  # 保持输入顺序（调用方按位置 zip）
    conflicts = 0
    for col, cands, j in zip(cols, cand_lists, chain):
        combo = cands[j][1]
        strings = [s for s, _ in combo if s >= 0]
        if len(set(strings)) < len(strings):
            conflicts += 1
        for idx, (s, f) in zip(col, combo):
            result[idx]["string"] = 6 - s if s >= 0 else 0  # 6-0=6(低E)..1(高E)
            result[idx]["fret"] = f
    logger.info(f"Guitar fingering v2: cols={len(cols)}, "
                f"same-string conflicts={conflicts}")
    return result


def _pos_center(combo: list[tuple[int, int]]) -> float:
    """把位中心 = 非空弦品均值（空弦不锚定把手）。"""
    frets = [f for s, f in combo if s >= 0 and f > 0]
    return sum(frets) / len(frets) if frets else 0.0


def _col_candidates(col: list[dict]) -> list[tuple[float, list[tuple[int, int]]]]:
    """列候选 = 每音 (弦,品) 的合法组合，返回 [(局部代价, combo)]。

    combo 与 col 位置一一对应。音域外音符（无候选）固定 sentinel
    (s=-1)：不参与约束、代价恒 5.0（沿用 v1 口径）。列 >6 音物理不可弹，
    仅前 6 个（列内已降音高=最高 6 音）入排指，其余 sentinel。
    硬约束：同弦互斥（含空弦）；非空弦品跨度 ≤ SPAN_MAX。
    """
    active = col[:6]
    n_unplayable = 0
    per_note: list[list[tuple[int, int]]] = []
    for note in active:
        pos = get_positions(note["pitch"])
        if not pos:
            pos = [(-1, -1)]
            n_unplayable += 1
        # 低品优先截断，控制组合数（把位移动由 DP 代价管，不靠这里）
        per_note.append(sorted(pos, key=lambda sf: sf[1])[:PER_NOTE_CANDS])

    combos: list[list[tuple[int, int]]] = []

    def _enum(i: int, used: set[int], acc: list[tuple[int, int]]) -> None:
        if i == len(per_note):
            combos.append(list(acc))
            return
        for s, f in per_note[i]:
            if s >= 0:
                if s in used:
                    continue  # 同弦互斥
                frets = [fr for (st, fr) in acc if st >= 0 and fr > 0]
                if f > 0:
                    frets.append(f)
                if frets and max(frets) - min(frets) > SPAN_MAX:
                    continue  # 手跨度
                used.add(s)
                acc.append((s, f))
                _enum(i + 1, used, acc)
                acc.pop()
                used.discard(s)
            else:
                acc.append((s, f))
                _enum(i + 1, used, acc)
                acc.pop()

    if active:
        _enum(0, set(), [])
    if not combos:
        combos = [[(-1, -1)] * len(active)]

    out: list[tuple[float, list[tuple[int, int]]]] = []
    seen: set[tuple] = set()
    for combo in combos:
        key = tuple(combo)
        if key in seen:
            continue
        seen.add(key)
        out.append((_local_cost(combo), combo))
    out.sort(key=lambda x: x[0])
    out = out[:MAX_COL_CANDS]
    if n_unplayable:
        out = [(c + 5.0 * n_unplayable, combo) for c, combo in out]
    if len(col) > 6:
        for _, combo in out:
            combo.extend([(-1, -1)] * (len(col) - 6))
    return out


def _local_cost(combo: list[tuple[int, int]]) -> float:
    cost = W_LOW * _pos_center(combo)
    for s, f in combo:
        if s >= 0 and f == 0:
            cost += config.OPEN_STRING_BIAS  # 负值 = 奖励
    return cost


# ---------------------------------------------------------------------------
# v1：逐音线性 DP（保留作回退/对比）
# ---------------------------------------------------------------------------

def _assign_v1(notes: list[dict]) -> list[dict]:
    if len(notes) == 0:
        return []

    candidates = []
    for note in notes:
        pos = get_positions(note["pitch"])
        if not pos:
            pos = [(0, -1)]  # sentinel for unplayable
        candidates.append(pos)

    n = len(notes)
    dp = []
    for i in range(n):
        dp_i = []
        for j, (s, f) in enumerate(candidates[i]):
            if i == 0:
                cost = config.OPEN_STRING_BIAS if f == 0 else 0.0
                dp_i.append((cost, -1))
            else:
                best_cost, best_prev = float("inf"), -1
                for k, (ps, pf) in enumerate(candidates[i - 1]):
                    if pf == -1 or f == -1:
                        step_cost = 5.0
                    else:
                        step_cost = (
                            config.FRET_WEIGHT * abs(f - pf)
                            + config.STRING_WEIGHT * abs(s - ps)
                        )
                        if f == 0:
                            step_cost += config.OPEN_STRING_BIAS
                    total = dp[i - 1][k][0] + step_cost
                    if total < best_cost:
                        best_cost, best_prev = total, k
                dp_i.append((best_cost, best_prev))
        dp.append(dp_i)

    # Backtrack
    result = []
    best_last = min(range(len(dp[-1])), key=lambda j: dp[-1][j][0])
    for i in range(n - 1, -1, -1):
        s, f = candidates[i][best_last]
        note_copy = dict(notes[i])
        note_copy["string"] = 6 - s  # 6-0=6 (低E), 6-5=1 (高E)
        note_copy["fret"] = f
        result.append(note_copy)
        _, best_last = dp[i][best_last] if i > 0 else (0, -1)

    result.reverse()
    return result


def assign_guitar_fingering(notes: list[dict]) -> list[dict]:
    if os.environ.get("MUSE_ARRANGE_V1"):
        return _assign_v1(notes)
    return _assign_v2(notes)
