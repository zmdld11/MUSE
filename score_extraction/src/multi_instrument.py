"""多乐器模式（2026-08-22）：ia-amt default 单模型吃全 mix → 按 instrument_class 分轨。

背景（findings 2026-08-22）：default checkpoint 一次推理即可输出全乐队
（吉他三兄弟/贝斯/钢琴/人声 melody 各归各类；纯钢琴交叉验证 185/185 piano，
乐器头工作正常）。guitar_v1_5 则是吉他专属（全 mix 只出吉他类）。

本模块只做数据落盘：每类一个 .mid（GM program 取自 ia-amt taxonomy，前端
按 GM 音色名/族自动着色）+ notes.json（阶段二契约，schema 见
frontend/markdown/前端计划书.md §6.2）。记谱规则定案见 markdown/记谱规则v1.md，
本模块执行其中的数据层部分（时值清洗 + 轨道准入统计）；Layer 5 组谱留待阶段 19。

启用：MUSE_MULTI_INSTRUMENT=1（开启时替代吉他单线，吉他类已含在输出里）。
"""
import json
import logging
import os

import pretty_midi

from src.config import config

logger = logging.getLogger(__name__)

# 少于该音符数的乐器类视为噪声丢弃（数据层粗筛）
MIN_CLASS_NOTES = 3

# 记谱层准入线（记谱规则v1.md §2）：数据层永不删轨，此线只决定默认入谱集合
SCORE_MIN_NOTES = 16
SCORE_MIN_COVERAGE = 0.05

# 截断后短于该值的残段视为重复音头丢弃（与 .mid 落盘的最小时长一致）
MIN_NOTE_SEC = 0.023

# taxonomy 代表音色修正：strings 类取首个成员 program 40（中提琴），
# 语义应为弦乐合奏（48）
CLASS_PROGRAM_OVERRIDES = {"strings": 48}


def clean_notes(notes: list[dict]) -> tuple[list[dict], dict]:
    """记谱规则v1.md §3 数据层清洗：同音高重叠截断（唯一硬规则）。

    同音高正间隙（ia-amt 系统性 ~23ms release 约定）是真实再触发，不做
    合并——虚無の先贝斯/吉他轨邻接对的 onset 间距精确落在 8 分/4 分/2 分
    网格（0.371/0.720/1.467s @80.7bpm），按间隙合并会摧毁节奏记谱。接缝
    vs 再触发的判定延后到量化后（记谱层，阶段 19）。
    """
    cleaned = [dict(n) for n in notes]
    by_pitch: dict[int, list[dict]] = {}
    for n in cleaned:
        by_pitch.setdefault(int(n["pitch"]), []).append(n)

    truncated = dropped = 0
    out: list[dict] = []
    for _pitch, group in sorted(by_pitch.items()):
        group.sort(key=lambda n: float(n["onset"]))
        kept: list[dict] = []
        for n in group:
            while kept:
                prev = kept[-1]
                if float(n["onset"]) >= float(prev["offset"]):
                    break
                if float(n["onset"]) - float(prev["onset"]) >= MIN_NOTE_SEC:
                    prev["offset"] = float(n["onset"])  # 截断前音，保住后音音头
                    truncated += 1
                    break
                kept.pop()  # 截断后只剩残段（重复音头）→ 丢弃前音，继续回溯
                dropped += 1
            kept.append(n)
        out.extend(kept)
    out.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return out, {"overlap_truncated": truncated, "stub_dropped": dropped}


def union_active_seconds(notes: list[dict]) -> float:
    """音符区间的并集总长（秒），用于活跃覆盖。"""
    total = 0.0
    cur_s = cur_e = None
    for s, e in sorted((float(n["onset"]), float(n["offset"])) for n in notes):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def run_multi_instrument(audio_path: str, output_dir: str, bpm: float) -> bool:
    from src.ia_amt_frontend import transcribe_ia_amt
    from instrument_agnostic_amt.taxonomy.instrument_classes import (
        INSTRUMENT_CLASSES,
        get_program_number_from_class_id,
    )

    logger.info("  [multi] Layer 2: ia-amt default frontend (全乐队)...")
    frontend = transcribe_ia_amt(audio_path, model_type="default")
    notes = frontend.get("notes", [])
    if not notes:
        logger.warning("  [multi] no notes from default model")
        return False

    groups: dict[str, list[dict]] = {}
    for n in notes:
        groups.setdefault(n["instrument_class"], []).append(n)
    groups = {k: v for k, v in groups.items() if len(v) >= MIN_CLASS_NOTES}

    # 记谱规则v1 §3 时值清洗 + §2 准入统计
    track_stats: dict[str, dict] = {}
    for cls, cls_notes in groups.items():
        cls_notes, cstats = clean_notes(cls_notes)
        groups[cls] = cls_notes
        track_stats[cls] = cstats
    duration = max(n["offset"] for n in notes)

    logger.info(f"  [multi] {len(notes)} notes -> {len(groups)} tracks: "
                + ", ".join(f"{k}:{len(v)}" for k, v in
                            sorted(groups.items(), key=lambda x: -len(x[1]))))

    # 每类一个 .mid
    for cls, cls_notes in groups.items():
        class_id = INSTRUMENT_CLASSES.index(cls) if cls in INSTRUMENT_CLASSES else None
        is_drum = cls == "drums"
        if is_drum or class_id is None:
            program = 0
        else:
            program = CLASS_PROGRAM_OVERRIDES.get(
                cls, int(get_program_number_from_class_id(class_id)))
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=cls)
        for n in cls_notes:
            inst.notes.append(pretty_midi.Note(
                velocity=max(1, min(127, int(n["velocity"]))),
                pitch=int(n["pitch"]),
                start=float(n["onset"]),
                end=max(float(n["offset"]), float(n["onset"]) + 0.023),
            ))
        pm.instruments.append(inst)
        pm.write(os.path.join(output_dir, f"{cls}.mid"))
        logger.info(f"  [multi] {cls}.mid ({len(cls_notes)} notes, "
                    f"program={program}, drum={is_drum})")

    # notes.json（阶段二数据契约；track 统计字段为记谱规则v1 §2 准入线输出）
    tracks_json = []
    for cls, cls_notes in sorted(groups.items(), key=lambda x: -len(x[1])):
        coverage = union_active_seconds(cls_notes) / duration if duration > 0 else 0.0
        cstats = track_stats.get(cls, {})
        tracks_json.append({
            "instrument_class": cls,
            "display_name": cls,
            "note_count": len(cls_notes),
            "active_coverage": round(coverage, 4),
            "score_worthy": (len(cls_notes) >= SCORE_MIN_NOTES
                             and coverage >= SCORE_MIN_COVERAGE),
            "cleanup": cstats,
            "notes": cls_notes,
        })
        logger.info(f"  [multi] {cls}: {len(cls_notes)} notes, cov={coverage:.1%}, "
                    f"score_worthy={tracks_json[-1]['score_worthy']}, "
                    f"cleanup={cstats}")

    notes_json = {
        "schema_version": 1,
        "song": os.path.splitext(os.path.basename(audio_path))[0],
        "bpm": bpm,
        "time_signature": config.DEFAULT_TIME_SIG,
        "duration": round(float(duration), 3),
        "source": {"audio": audio_path, "frontend": "ia_amt:default",
                   "separated": False},
        "tracks": tracks_json,
    }
    with open(os.path.join(output_dir, "notes.json"), "w", encoding="utf-8") as f:
        json.dump(notes_json, f, ensure_ascii=False, indent=2)
    logger.info("  [multi] notes.json written")

    # 记谱层（阶段19）：NotationScore + 单乐器谱（量化+忠实双模式）
    try:
        from src.notation_layer import build_notation
        build_notation(notes_json, output_dir, mode=config.NOTATION_MODE)
    except Exception:
        logger.exception("  [multi] notation layer failed (数据落盘不受影响)")
    return True
