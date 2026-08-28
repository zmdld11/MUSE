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

_INS = None
_Slicer = None


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

    note 字段与 ia-amt 前端对齐；pitch 取整（SOME 连续值留待颤音修正层）。
    """
    import librosa
    import numpy as np

    _load()
    waveform, _sr = librosa.load(audio_path,
                                 sr=_INS.config["audio_sample_rate"], mono=True)
    slicer = _Slicer(sr=_INS.config["audio_sample_rate"], max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    midis = _INS.infer([c["waveform"] for c in chunks])
    out = []
    for chunk, seg in zip(chunks, midis):
        pitches = np.round(seg["note_midi"]).astype(np.int64).tolist()
        durs = np.asarray(seg["note_dur"], dtype=float).tolist()
        rests = np.asarray(seg["note_rest"], dtype=bool).tolist()
        t = float(chunk["offset"])
        for pitch, dur, rest in zip(pitches, durs, rests):
            end = t + float(dur)
            if (not rest and end - t > 0.01
                    and VOCAL_PITCH_LO <= int(pitch) <= VOCAL_PITCH_HI):
                out.append({
                    "onset": round(t, 4), "offset": round(end, 4),
                    "pitch": int(pitch), "velocity": 100,
                    "confidence": 1.0, "instrument_class": "melody",
                })
            t = end
    out.sort(key=lambda n: (n["onset"], n["pitch"]))
    logger.info("[some] %d notes (vocal gate [%d,%d]) from %s",
                len(out), VOCAL_PITCH_LO, VOCAL_PITCH_HI, audio_path)
    return {"notes": out, "note_count": len(out)}
