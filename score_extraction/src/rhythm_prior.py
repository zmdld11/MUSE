"""T2 节奏模板对撞（乐理路线第二阶，2026-08-23）：GT 语料建每小节节奏
模式库，仅对低置信小节做最近邻对撞修正（只挪位置，不增删音符）。

模式 = 小节内 onset 位置集合（1/12 四分网格单位 0..47，去重升序元组）。
语料：BabySlakh 20 首（Lakh 流行，多乐器）+ URMP Sco 44 首（古典）+
GuitarSet GT（吉他）。文献：Nakamura 2016/2021 score-LM、Cemgil 1999。

开关：MUSE_RHYTHM_PRIOR=1（默认关，评测对照后定默认值）。
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from fractions import Fraction

logger = logging.getLogger(__name__)

GRID = 12                    # 每四分 12 格（与记谱层量化网格一致）
BAR_UNITS = 4 * GRID         # 4/4 小节
LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "rhythm_pattern_library.json")
MAX_MATCH_DIST = 2           # 最近邻模式允许的对称差距离上限


def _enabled() -> bool:
    return os.environ.get("MUSE_RHYTHM_PRIOR", "0") == "1"

_LIB_CACHE: dict | None = None


def bar_signature(positions_ql: list[Fraction]) -> tuple[int, ...]:
    """小节内 onset 位置（四分音符 Fraction）→ 1/12 网格签名元组。"""
    units = sorted({int(round(p * GRID)) for p in positions_ql})
    return tuple(u for u in units if 0 <= u < BAR_UNITS)


def midi_bar_signatures(midi_path: str) -> Counter:
    """GT MIDI → 每小节 onset 签名计数（跨音高合并；tempo map 经
    time_to_tick 折算）。"""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    bars: dict[int, set[int]] = {}
    for inst in pm.instruments:
        for n in inst.notes:
            qpos = pm.time_to_tick(n.start) / pm.resolution  # 四分位置
            bar = int(qpos // 4)
            unit = int(round((qpos - bar * 4) * GRID))
            if 0 <= unit < BAR_UNITS:
                bars.setdefault(bar, set()).add(unit)
    out: Counter = Counter()
    for _bar, units in bars.items():
        if len(units) >= 2:  # 单音小节信息量低，不入库
            out[tuple(sorted(units))] += 1
    return out


def build_library(song_counters: dict[str, Counter], min_count: int = 3) -> dict:
    """{song_id: 签名计数} → 库 JSON（含 per-song 计数，供留一法去泄漏）。"""
    total: Counter = Counter()
    for c in song_counters.values():
        total.update(c)
    lib = {
        "sig": {",".join(map(str, sig)): {"count": int(cnt),
                                          "songs": {s: int(c[sig])
                                                    for s, c in song_counters.items()
                                                    if sig in c}}
                for sig, cnt in total.items() if cnt >= min_count},
        "min_count": min_count,
    }
    n_patterns = len(lib["sig"])
    n_total = sum(v["count"] for v in lib["sig"].values())
    logger.info("[rhythm] library: %d patterns, %d bar instances", n_patterns, n_total)
    return lib


def load_library(leave_out_song: str | None = None) -> dict[tuple[int, ...], int]:
    """库 JSON → {签名: 计数}（留一法：扣除当前歌曲贡献）。"""
    global _LIB_CACHE
    if _LIB_CACHE is None:
        with open(LIB_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        _LIB_CACHE = {
            tuple(int(x) for x in k.split(",")): v["count"]
            for k, v in raw["sig"].items()
        }
    if leave_out_song is None:
        return _LIB_CACHE
    with open(LIB_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw["sig"].items():
        sig = tuple(int(x) for x in k.split(","))
        out[sig] = v["count"] - v["songs"].get(leave_out_song, 0)
    return {s: c for s, c in out.items() if c > 0}


def nearest_pattern(sig: tuple[int, ...],
                    lib: dict[tuple[int, ...], int]) -> tuple[tuple[int, ...], int] | None:
    """最近库模式（对称差距离）；超 MAX_MATCH_DIST 或无候选返回 None。"""
    if not lib:
        return None
    best: tuple[tuple[int, ...], int] | None = None
    best_d = MAX_MATCH_DIST + 1
    for cand, cnt in lib.items():
        d = len(set(cand) ^ set(sig))
        if d < best_d or (d == best_d and best is not None and cnt > lib[best]):
            best_d, best = d, cand
    if best is None or best_d > MAX_MATCH_DIST:
        return None
    return best, best_d


def apply_rhythm_prior(events: list[dict]) -> int:
    """稀有节奏小节对撞库模式，微移 onset 到库位置。

    触发（2026-08-23 修正）：小节签名不在库中（网格粗化后 rubato 标志
    恒 False，原触发失效；"非主流节奏"本身即抖动嫌疑）。事件需带
    _onset_ql（在 clean/guard/seam 之后、时值分配之前调用）。
    返回修正小节数。只挪已有 onset（≤1 格），不增删音符。
    """
    if not _enabled():
        return 0
    try:
        lib = load_library()
    except OSError:
        logger.warning("[rhythm] library missing — skip")
        return 0
    bars: dict[int, list[dict]] = {}
    for ev in events:
        bars.setdefault(int(ev["_onset_ql"] // 4), []).append(ev)
    fixed = 0
    for bar, evs in bars.items():
        if len(evs) < 2:
            continue  # 单音小节信息量低，不动
        sig = tuple(sorted({int((ev["_onset_ql"] - bar * 4) * GRID)
                            for ev in evs}))
        if sig in lib:
            continue  # 库内主流节奏，高置信
        hit = nearest_pattern(sig, lib)
        if hit is None:
            continue
        cand, dist = hit
        if dist == 0:
            continue
        # 微移：库位置里有最近格可去的 onset 才动（每音 ≤1 格）
        used: set[int] = set()
        for ev in evs:
            cur = int((ev["_onset_ql"] - bar * 4) * GRID)
            if cur in cand:
                used.add(cur)
                continue
            near = [u for u in cand if abs(u - cur) <= 1 and u not in used]
            if near:
                u = min(near, key=lambda x: abs(x - cur))
                ev["_onset_ql"] = Fraction(bar * 4) + Fraction(u, GRID)
                ev["rhythm_prior"] = True
                used.add(u)
        fixed += 1
    if fixed:
        events.sort(key=lambda e: (e["_onset_ql"], e["pitch"]))
        logger.info("[rhythm] prior fixed %d rare-pattern bars", fixed)
    return fixed
