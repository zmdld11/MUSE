"""SOME（openvpi/some，ISMIR 2022）人声前端：进程内单例 → ia-amt 同构 note 格式。

人声支线骨架（2026-08-28 X，用户拍板 pre 版）：MelBand 分离 stem + SOME
音符化替代 raw 直推的 melody 类。颤音/碎音修正是下版本已知项（本层不动
SOME 原始输出）。

约束：
- Windows/torch-cu118 复用 ia_amt_frontend 的 nvrtc 补丁；
- 与 ia-amt 仓库都有顶层 infer.py → 本前端必须在 ia-amt 首次加载之后使用
  （multi_instrument 的调用序天然满足：ia-amt 跑完吉他/混音 run 才轮到人声）；
- 模型构造会扰动 sys.modules（utils 句柄被清）→ Slicer 句柄存模块全局。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1] / "external" / "some"
_CKPT = (Path(__file__).resolve().parents[1] / "external" / "some_pretrained"
         / "0119_continuous256_5spk" / "model_ckpt_steps_100000_simplified.ckpt")

# 人声音域门（G2#–C6）：MIR-1K 实测业余男声低至 ~G2#（GP 基准 48 会误伤）
VOCAL_PITCH_LO, VOCAL_PITCH_HI = 40, 84

# 颤音/碎音修正层参数（第二阶段 #5 v1，2026-08-29）：SOME 输出 note 级
# 连续音高，颤音/滑音被切成 ±1 半音内摆动的碎块（夏日实测 71→69.65→71
# →69.2→69.17）；round 前合并碎块、保留连续值，消除"音抖得离谱"。
VIB_SAME_DP = 0.5    # Δ≤此值=同音被切（offset 抖动），任意时长合并
VIB_NEAR_DP = 1.5    # Δ≤此值且含碎块=颤音/滑音摆动，吸收合并
VIB_SHORT_SEC = 0.30  # 碎块判据（人声稳定音典型 ≥0.3s）

_INS = None
_Slicer = None


def _merge_vibrato(notes: list[dict]) -> list[dict]:
    """无休止间隔的连续 run 内做碎块合并（音头保护：rest 边界=真实
    换气/起音，绝不跨段；Δ>VIB_NEAR_DP 的音高跳变=真音符，不吞）。

    合并块的音高 = 时长加权均值（对称颤音落中心，滑音偏向停留侧）。
    """
    out: list[dict] = []
    run: list[dict] = []

    def flush() -> None:
        nonlocal run
        if run:
            total = sum(n["offset"] - n["onset"] for n in run)
            pitch = sum(n["_midi"] * (n["offset"] - n["onset"])
                        for n in run) / total
            out.append({"onset": run[0]["onset"], "offset": run[-1]["offset"],
                        "_midi": pitch})
            run = []

    prev_end = None
    for n in notes:
        # rest 间隔（时间缝）或跨 run 断裂 → 结束当前 run
        if prev_end is not None and n["onset"] - prev_end > 1e-6:
            flush()
        if run:
            gap_dp = abs(n["_midi"] - run[-1]["_midi"])
            short = min(n["offset"] - n["onset"],
                        run[-1]["offset"] - run[-1]["onset"]) < VIB_SHORT_SEC
            if gap_dp <= VIB_SAME_DP or (gap_dp <= VIB_NEAR_DP and short):
                run.append(n)
                prev_end = n["offset"]
                continue
            flush()
        run.append(n)
        prev_end = n["offset"]
    flush()
    return out


def _load():
    global _INS, _Slicer
    if _INS is not None:
        return
    if not _CKPT.exists():
        raise FileNotFoundError(f"SOME ckpt missing: {_CKPT}")
    from src.ia_amt_frontend import _apply_windows_torch_patches
    _apply_windows_torch_patches()
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    import importlib

    import yaml
    with open(_CKPT.with_name("config.yaml"), encoding="utf8") as f:
        config = yaml.safe_load(f)
    import inference as some_inference
    from utils import slicer2
    infer_cls = some_inference.task_inference_mapping[config["task_cls"]]
    pkg = ".".join(infer_cls.split(".")[:-1])
    cls = getattr(importlib.import_module(pkg), infer_cls.split(".")[-1])
    _INS = cls(config=config, model_path=_CKPT)
    _Slicer = slicer2.Slicer
    logger.info("[some] loaded ckpt=%s", _CKPT.name)


def transcribe_some(audio_path: str) -> dict:
    """wav → {"notes": [...], "note_count": int}（instrument_class 恒 "melody"）。

    note 字段与 ia-amt 前端对齐。音高：先按 note 级连续值做颤音/碎块合并
    （_merge_vibrato），合并块取时长加权均值，最后才取整——round 抖动的
    根源是先取整后合并会把 ±1 摆动钉死成两个半音（2026-08-29 #5）。
    """
    import librosa
    import numpy as np

    _load()
    waveform, _sr = librosa.load(audio_path,
                                 sr=_INS.config["audio_sample_rate"], mono=True)
    slicer = _Slicer(sr=_INS.config["audio_sample_rate"], max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    midis = _INS.infer([c["waveform"] for c in chunks])
    raw: list[dict] = []
    for chunk, seg in zip(chunks, midis):
        pitches = np.asarray(seg["note_midi"], dtype=float)
        durs = np.asarray(seg["note_dur"], dtype=float)
        rests = np.asarray(seg["note_rest"], dtype=bool)
        t = float(chunk["offset"])
        for pitch, dur, rest in zip(pitches, durs, rests):
            end = t + float(dur)
            if not rest and end - t > 0.01:
                raw.append({"onset": round(t, 4), "offset": round(end, 4),
                            "_midi": float(pitch), "velocity": 100})
            t = end
    raw.sort(key=lambda n: n["onset"])
    notes = []
    for n in _merge_vibrato(raw):
        pitch = int(round(n["_midi"]))
        if VOCAL_PITCH_LO <= pitch <= VOCAL_PITCH_HI:
            notes.append({
                "onset": n["onset"], "offset": n["offset"],
                "pitch": pitch, "velocity": 100,
                "confidence": 1.0, "instrument_class": "melody",
            })
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    logger.info("[some] %d notes (raw %d, vibrato-merged, gate [%d,%d]) from %s",
                len(notes), len(raw), VOCAL_PITCH_LO, VOCAL_PITCH_HI, audio_path)
    return {"notes": notes, "note_count": len(notes)}
