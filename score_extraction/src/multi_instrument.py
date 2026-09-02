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

# per-stem 乐器门控（大横评 2026-08-24 结论#7）：stemwise 合并必须按 stem 类型
# 过滤输出音色——每个 stem 都被全乐器前端转写一遍，同类音符会在多 stem 重复
# 出现，合并后假阳性翻倍（ymt3 demucs -32pt 教训）。类名取自 ia-amt taxonomy
# （36 类）。drums 不设门：ia default 不输出鼓类事件，前端亦无鼓谱面。
GUITAR_CLASSES = {"acoustic_guitar", "distorted_guitar", "electric_guitar_clean",
                  "electric_guitar_muted", "guitar_harmonics"}
BASS_CLASSES = {"electric_bass", "acoustic_bass", "slap_bass", "synth_bass"}
KEYS_CLASSES = {"piano", "electric_piano", "organ", "plucked_keyboard"}
VOICE_CLASSES = {"melody", "vocal_harmony", "choir"}
# other stem 的长尾类 = 全集 - 吉他/贝斯/人声 - 鼓/效果类（交由 MIN_CLASS_NOTES 粗筛）。
# 注意 KEYS 类对 other stem 也放行：合成器键琴常落在 demucs "other" 而 piano stem
# 近空（BS 键盘密集曲实测键盘轨整轨消失）；piano/other 两 stem 的 KEYS 事件按类
# 合并后由 clean_notes 截同音高重叠。
_OTHER_EXCLUDE = GUITAR_CLASSES | BASS_CLASSES | VOICE_CLASSES | {
    "drums", "chromatic_percussion", "percussive_fx", "sound_fx", "synth_fx"}
_ALL_CLASSES = {
    "accordion_family", "acoustic_bass", "acoustic_guitar", "brass", "choir",
    "chromatic_percussion", "distorted_guitar", "drums", "electric_bass",
    "electric_guitar_clean", "electric_guitar_muted", "electric_piano", "ethnic",
    "flute_pipe", "guitar_harmonics", "harmonica", "orchestra_hit", "orchestral_harp",
    "orchestral_woodwind", "organ", "percussive_fx", "piano", "pizzicato_strings",
    "plucked_keyboard", "sax", "slap_bass", "sound_fx", "strings", "synth_bass",
    "synth_fx", "synth_lead", "synth_pad", "timpani", "melody", "vocal_harmony",
    "wind_chimes",
}
OTHER_CLASSES = _ALL_CLASSES - _OTHER_EXCLUDE


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


def _stem_mean_amp(path: str) -> float:
    """stem 平均幅度（静音 stem 免推理）。"""
    import soundfile as sf
    x, _ = sf.read(path, dtype="float32", always_2d=True)
    return float(abs(x).mean())


def _separated_stem_specs(audio_path: str, output_dir: str) -> list[tuple[str, str, set]]:
    """分离并给出 (标签, stem 路径, 乐器类门控) 列表。

    VER-SEP 2.0 出吉他（验收：F2 versep 下游 F1 0.531→0.597）；htdemucs_6s
    出其余。drums stem 不喂（无鼓谱面）。产物缓存在 output_dir/sep/，重跑免分离。
    """
    from src.source_separate import separate_tracks
    from src.versep_sep import separate_guitar

    sep_dir = os.path.join(output_dir, "sep")
    guitar_wav = separate_guitar(audio_path, os.path.join(sep_dir, "versep"))
    logger.info("  [multi] demucs htdemucs_6s 分离其余 stem...")
    demucs_stems = separate_tracks(audio_path, os.path.join(sep_dir, "demucs"))

    specs: list[tuple[str, str, set]] = []
    if guitar_wav is not None:
        specs.append(("guitar:versep", guitar_wav, GUITAR_CLASSES))
    elif "guitar" in demucs_stems:
        specs.append(("guitar:demucs", demucs_stems["guitar"], GUITAR_CLASSES))
    specs += [
        ("bass:demucs", demucs_stems.get("bass", ""), BASS_CLASSES),
        ("piano:demucs", demucs_stems.get("piano", ""), KEYS_CLASSES),
        ("vocals:demucs", demucs_stems.get("vocals", ""), VOICE_CLASSES),
        ("other:demucs", demucs_stems.get("other", ""), OTHER_CLASSES),
    ]
    kept = []
    for label, wav, gate in specs:
        if not wav or not os.path.exists(wav):
            logger.warning(f"  [multi] stem 缺失，跳过: {label}")
            continue
        amp = _stem_mean_amp(wav)
        if amp < 1e-4:
            logger.info(f"  [multi] stem 近无声（amp={amp:.2e}），跳过: {label}")
            continue
        kept.append((label, wav, gate))
    return kept


def run_multi_instrument(audio_path: str, output_dir: str, bpm: float,
                         on_stage=None) -> bool:
    """on_stage(stage, label)：管线桥进度钩子（stage ∈ separate/transcribe/
    notation），CLI 直跑时不传、行为不变。"""
    from src.ia_amt_frontend import transcribe_ia_amt
    from instrument_agnostic_amt.taxonomy.instrument_classes import (
        INSTRUMENT_CLASSES,
        get_program_number_from_class_id,
    )

    # 入口 absolutize：下游分离子进程会切 cwd（MSST），相对路径在子进程侧
    # 解析错位 → "Total files found: 0"（2026-08-27 批量 10 首全挂根因；
    # kyomu 曾因分离缓存命中掩盖此 bug）
    audio_path = os.path.abspath(audio_path)
    output_dir = os.path.abspath(output_dir)

    # 直调本函数时调用方可能未建目录（pipeline.py 由上游建；A/B 驱动曾因此全灭）
    os.makedirs(output_dir, exist_ok=True)

    # 分离模式（config.MULTI_SEPARATION）：
    # off = 混音直通单次推理（旧行为）；
    # versep_demucs = VER-SEP 吉他 + htdemucs_6s 其余 + per-stem 门控（全 stemwise）；
    # versep_guitar = 混合（默认）：吉他走 VER-SEP stem（F2 证据 AUPRC 0.250→
    #   0.525 + 2.0 后下游 0.597），其余类混音直通（BS A/B：demucs stemwise 掉精度）
    separated = config.MULTI_SEPARATION != "off"
    piano_stem_wav = None        # versep_guitar 分支填充；其余模式恒 None
    piano_stem_model_name = ""
    if on_stage:
        on_stage("separate", "乐器分离" if separated else "混音直通")
    if config.MULTI_SEPARATION == "versep_guitar":
        from src.versep_sep import separate_guitar
        sep_dir = os.path.join(output_dir, "sep")
        if on_stage:
            on_stage("separate", "吉他分离")
        guitar_wav = separate_guitar(audio_path, os.path.join(sep_dir, "versep"))
        # 钢琴混音路径（2026-08-31 接线，08-30 夜间增测：ia@bs_roformer-piano-stem
        # F2p 0.729→0.882 / Slakh 0.552→0.820）；KEYS 类改从 stem 路来，
        # raw 路门控同步剔除 KEYS（照抄吉他的互斥防双计）。失败回退 raw 直推。
        from src.piano_stem_sep import piano_stem_model, separate_piano_bsroformer
        piano_stem_model_name = piano_stem_model()
        piano_stem_wav = None
        if piano_stem_model_name != "off":
            if on_stage:
                on_stage("separate", "钢琴分离")
            piano_stem_wav = separate_piano_bsroformer(
                audio_path, os.path.join(sep_dir, "bspiano"))
        if guitar_wav is not None:
            runs = [("guitar:versep", guitar_wav, GUITAR_CLASSES)]
            if piano_stem_wav:
                runs.append(("piano:bspiano", piano_stem_wav, KEYS_CLASSES))
                rest_gate = _ALL_CLASSES - GUITAR_CLASSES - KEYS_CLASSES
            else:
                rest_gate = _ALL_CLASSES - GUITAR_CLASSES
            runs.append(("<raw>", audio_path, rest_gate))
        elif piano_stem_wav is not None:  # VER-SEP 不可用但钢琴 stem 在
            runs = [("piano:bspiano", piano_stem_wav, KEYS_CLASSES),
                    ("<raw>", audio_path, _ALL_CLASSES - KEYS_CLASSES)]
        else:  # 双分离均不可用 → 纯直通
            runs, separated = [("<raw>", audio_path, set())], False
    elif separated:
        runs = _separated_stem_specs(audio_path, output_dir)
        if not runs:
            logger.warning("  [multi] 分离未产出可用 stem，回退混音直通")
            runs, separated = [("<raw>", audio_path, set())], False
    else:
        runs = [("<raw>", audio_path, set())]

    groups: dict[str, list[dict]] = {}
    for ri, (label, wav, gate) in enumerate(runs):
        if on_stage:
            on_stage("transcribe", f"多乐器转写 {ri + 1}/{len(runs)}")
        logger.info(f"  [multi] Layer 2: ia-amt default frontend @ {label}...")
        frontend = transcribe_ia_amt(wav, model_type="default",
                                     note_bias=config.IA_NOTE_BIAS)
        stem_notes = frontend.get("notes", [])
        if gate:
            stem_notes = [n for n in stem_notes if n["instrument_class"] in gate]
        for n in stem_notes:
            groups.setdefault(n["instrument_class"], []).append(n)
        logger.info(f"  [multi] {label}: {len(frontend.get('notes', []))} notes "
                    f"-> {len(stem_notes)} 门控后保留")
    if not groups:
        logger.warning("  [multi] no notes from default model")
        return False
    # 三路 ia 转写已毕：卸载模型归还显存，给 MelBand 等 MSST 大显存子进程
    # 让路（WDDM 超订实测让子进程慢 10-20×，见 ia_amt_frontend.unload_ia_amt）
    from src.ia_amt_frontend import unload_ia_amt
    unload_ia_amt()
    # 人声支线骨架（pre 版，2026-08-28 X）：MelBand 分离 + SOME 音符化替代
    # raw 直推的 melody/choir/harmony 类。必须在 ia-amt run 之后执行——两仓
    # 库都有顶层 infer.py，SOME 先加载会劫持 ia-amt 的 infer 解析（08-28 VII
    # 实测）。失败回退旧路径（raw 混音直推的人声类保留）。颤音/碎音算法
    # 修正 = 下版本已知项（本层不动 SOME 原始输出）。
    vocal_notes: list[dict] | None = None
    vocal_backbone = ""
    vocal_lyrics: dict | None = None  # 歌词增强层（chars+lines，无 LRC 恒 None）
    vocal_audit: dict = {}
    if config.MULTI_SEPARATION == "versep_guitar":
        from src.source_separate import separate_vocals_melband
        if on_stage:
            on_stage("separate", "人声分离")
        vocals_wav = separate_vocals_melband(
            audio_path, os.path.join(output_dir, "sep", "melband"))
        if vocals_wav:
            # 器乐曲护栏：MelBand 对无人声曲会有伴奏漏音（gymnopedie 钢琴
            # 漏音 mean_amp 0.007 vs 人声曲 0.09+，实测 2026-08-28 X）——
            # stem 幅度过低视为无人声，回退 raw 直推（由 ia-amt 自然不出
            # melody 类）。SOME 在近空 stem 上也会吐零星碎音，不能只靠它。
            amp = _stem_mean_amp(vocals_wav)
            if amp < 0.02:
                logger.info("  [multi] 人声 stem 幅度 %.4f < 0.02（器乐/近空），"
                            "人声回退 raw 直推", amp)
                vocals_wav = None
        if vocals_wav:
            from src.some_frontend import transcribe_some
            if on_stage:
                on_stage("transcribe", "人声转写")
            try:
                # 歌词增强层（人声专项 v2，2026-08-30）：同名 .lrc 自动发现
                # （music/vocal 约定=音频旁同名 .lrc）；无 LRC = 现状直通
                lrc = None
                lrc_path = os.path.splitext(audio_path)[0] + ".lrc"
                if os.path.exists(lrc_path):
                    from src.lyric_align import parse_lrc as _parse_lrc
                    try:
                        import librosa as _librosa
                        dur_audio = float(
                            _librosa.get_duration(path=audio_path))
                    except Exception:
                        dur_audio = None
                    lrc = _parse_lrc(lrc_path, dur_audio)
                    if lrc:
                        logger.info("  [multi] LRC: %s（%d 行歌词）",
                                    os.path.basename(lrc_path),
                                    len(lrc["lyric_lines"]))
                line_starts = ([l["t0"] for l in lrc["lyric_lines"]]
                               if lrc else None)
                # 人声骨干可换：MUSE_VOCAL_ENGINE=m3 → 自研 VocalCRNN(m3st500，
                # 元音头切分)；默认 some。下游增强层两骨干共用。
                _vocal_engine = os.environ.get("MUSE_VOCAL_ENGINE", "some")
                if _vocal_engine == "m3":
                    from src.vocal_crnn_frontend import transcribe_m3
                    r = transcribe_m3(vocals_wav, line_boundaries=line_starts)
                else:
                    r = transcribe_some(vocals_wav, line_boundaries=line_starts)
                if r["note_count"] >= MIN_CLASS_NOTES:
                    vocal_notes = r["notes"]
                    vocal_backbone = f"melband+{_vocal_engine}"
                    # offset 能量截断（无歌词依赖，SOME 长音拖尾 p90 +455ms）
                    from src.lyric_align import (detect_ornaments,
                                                 trim_vocal_offsets)
                    vocal_notes, n_trim = trim_vocal_offsets(
                        vocal_notes, vocals_wav)
                    vocal_audit["offsets_trimmed"] = n_trim
                    if lrc:
                        from src.lyric_align import (align_chars,
                                                     fill_missing_syllables,
                                                     filter_breath_notes,
                                                     snap_line_anchors,
                                                     split_melisma)
                        # 行锚自校正（VII，2026-08-30）：官方 LRC 也会整张
                        # 专辑系统性偏移（夏日实测 +331ms/55% 行>300ms）。
                        # SOME 音符 onset 做参照（音高检测过的真唱，能量
                        # 法会被漏音/换气骗出假偏移）+ 曲首标题行剔除；
                        # 必须在 filter_breath_notes/align 之前
                        import re as _re
                        _title = _re.sub(
                            r"^\d+\s+", "",
                            os.path.splitext(os.path.basename(audio_path))[0]
                        ).split(" - ")[0].strip()
                        _off = snap_line_anchors(
                            lrc, vocals_wav, song_title=_title,
                            notes=vocal_notes,
                            offset_enabled=os.environ.get("MUSE_LRC_SNAP") == "1")
                        if abs(_off) > 1e-3:
                            vocal_audit["lrc_offset_applied"] = round(_off, 3)
                            logger.info("  [multi] LRC 行锚自校正 %+.3fs", _off)
                        lines = lrc["lyric_lines"]
                        vocal_notes, breath_removed = filter_breath_notes(
                            vocal_notes, lines, vocals_wav)
                        chars = align_chars(lrc, vocals_wav, vocal_notes)
                        # CTC 字界精修（P3，2026-08-30）：逐行 wav2vec2
                        # Viterbi（行间 LRC 锚死防副歌滑移；权重缺失/
                        # OOV 回退 energy）——一致性中位 0.13→0.06s
                        from src.lyric_align import refine_chars_ctc
                        chars, n_ref = refine_chars_ctc(
                            chars, lines, vocals_wav)
                        if n_ref:
                            vocal_audit["chars_ctc_refined"] = n_ref
                        # 一字一音先验（P1）：拖腔拆音 → 补漏（帧 F0 合成）
                        vocal_notes, n_split = split_melisma(
                            vocal_notes, chars, vocals_wav)
                        vocal_notes, n_filled = fill_missing_syllables(
                            vocal_notes, chars, vocals_wav)
                        vocal_lyrics = {
                            "chars": chars,
                            "lines": [{"t0": l["t0"], "t1": l["t1"],
                                       "text": l["text"]} for l in lines],
                            "source_file": lrc_path,
                        }
                        vocal_audit.update({
                            "lyric_source": os.path.basename(lrc_path),
                            "breath_removed": len(breath_removed),
                            "melisma_split": n_split,
                            "syllables_filled": n_filled,
                        })
                        logger.info(
                            "  [multi] 歌词增强层：offset截断 %d，呼吸删 %d，"
                            "拖腔拆 %d，补漏插 %d 字位",
                            n_trim, len(breath_removed), n_split, n_filled)
                    # 技巧标注（颤音/滑音；帧 F0 形态，与歌词无关）
                    vocal_notes = detect_ornaments(vocal_notes, vocals_wav)
                    n_vib = sum(1 for n in vocal_notes
                                if n.get("ornament") == "vibrato")
                    vocal_audit["vibrato_marked"] = n_vib
                    if n_vib:
                        logger.info("  [multi] 技巧标注：vibrato %d", n_vib)
            except Exception:
                logger.warning("  [multi] SOME 人声转写失败，回退 raw 直推",
                               exc_info=True)
        else:
            logger.warning("  [multi] MelBand 分离未产出，人声回退 raw 直推")
    if vocal_notes is not None:
        # 人声类由 SOME 支线承担，raw 直推的人声类丢弃（防双计）
        dropped = {c: len(groups.pop(c)) for c in list(groups)
                   if c in VOICE_CLASSES}
        if dropped:
            logger.info("  [multi] 人声类改走 SOME 支线，raw 直推人声丢弃: %s",
                        dropped)
        groups["melody"] = vocal_notes
    groups = {k: v for k, v in groups.items() if len(v) >= MIN_CLASS_NOTES}

    # 钢琴专用模型旁路（2026-08-30 XI）：钢琴主导曲（无人声 ∧ 唯一实质
    # 非鼓类 = piano）的钢琴轨换 ByteDance 专用模型。canon 两份网友 GT
    # A/B：F1 0.344/0.378 vs ia-amt 0.270/0.286、低音区 recall 14%→25%；
    # 混音侧反向崩溃（花海 2961 音符 pitch 25-104 幻音洪水 vs ia-amt
    # 936）——独奏钢琴训练的模型只在独奏域启用。MUSE_PIANO_ENGINE=
    # bytedance|ia_amt 强制单侧。
    _piano_engine = os.environ.get("MUSE_PIANO_ENGINE", "auto")
    if "piano" in groups and _piano_engine != "ia_amt":
        others_n = sum(len(v) for c, v in groups.items()
                       if c not in ("piano", "drums"))
        # 主导判据用音符占比（canon 的 VER-SEP 吉他漏音类 ~99 音符不该
        # 挡住 2093 音符的钢琴主导曲；kyomu 吉他+弦乐俱在则不触发）
        _solo_piano = vocal_notes is None and len(groups["piano"]) >= 3 * others_n
        if _piano_engine == "bytedance" or (_piano_engine == "auto" and _solo_piano):
            from src.bytedance_frontend import transcribe_bytedance
            try:
                bd = transcribe_bytedance(audio_path)
                bd_notes = [{**n, "instrument_class": "piano",
                             "confidence": 0.95}
                            for n in bd.get("notes", [])]
                if len(bd_notes) >= MIN_CLASS_NOTES:
                    logger.info("  [multi] 钢琴旁路 → ByteDance 专用模型: "
                                "%d 音符（ia-amt %d）",
                                len(bd_notes), len(groups["piano"]))
                    groups["piano"] = bd_notes
            except Exception:
                logger.warning("  [multi] ByteDance 钢琴旁路失败，保留 ia-amt",
                               exc_info=True)

    # 记谱规则v1 §3 时值清洗 + §2 准入统计
    track_stats: dict[str, dict] = {}
    for cls, cls_notes in groups.items():
        cls_notes, cstats = clean_notes(cls_notes)
        groups[cls] = cls_notes
        track_stats[cls] = cstats
    notes = [n for v in groups.values() for n in v]
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

    # 节拍跟踪（时值 v3.2）：rubato 曲目记谱格点映射的素材（失败不致命，
    # 记谱层回退 onset 格点拟合）
    beat_info: dict = {}
    try:
        from src.beat_track import extract_beats
        beat_info = extract_beats(audio_path)
        if beat_info:
            logger.info("  [multi] beat track: %d 拍, 脉冲 %.3fs (%.1f/min)",
                        len(beat_info["beat_times"]), beat_info["pulse_sec"],
                        beat_info["tempo_pulse"])
    except Exception:
        logger.warning("  [multi] beat tracking failed", exc_info=True)

    notes_json = {
        "schema_version": 1,
        "song": os.path.splitext(os.path.basename(audio_path))[0],
        "bpm": bpm,
        "time_signature": config.DEFAULT_TIME_SIG,
        "duration": round(float(duration), 3),
        **beat_info,
        **({"lyrics": vocal_lyrics} if vocal_lyrics else {}),
        "source": {"audio": audio_path, "frontend": "ia_amt:default",
                   "separated": separated,
                   **({"vocal_lyric_audit": vocal_audit} if vocal_audit else {}),
                   **({"separators": {
                          "guitar": "VERSEP2.0+bs4",
                          "vocals": vocal_backbone or "raw-direct",
                          "piano": ("bsroformer-" + piano_stem_model_name)
                          if piano_stem_wav else "raw-direct",
                          "others": "htdemucs_6s" if config.MULTI_SEPARATION == "versep_demucs" else "raw-direct"},
                       "mode": config.MULTI_SEPARATION,
                       "stem_gating": True} if separated else {})},
        "tracks": tracks_json,
    }
    with open(os.path.join(output_dir, "notes.json"), "w", encoding="utf-8") as f:
        json.dump(notes_json, f, ensure_ascii=False, indent=2)
    logger.info("  [multi] notes.json written")

    # 记谱层（阶段19）：NotationScore + 单乐器谱（量化+忠实双模式）
    try:
        from src.notation_layer import build_notation
        if on_stage:
            on_stage("notation", "记谱与谱面导出")
        build_notation(notes_json, output_dir, mode=config.NOTATION_MODE)
    except Exception:
        logger.exception("  [multi] notation layer failed (数据落盘不受影响)")
    return True
