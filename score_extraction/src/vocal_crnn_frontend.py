"""自研 VocalCRNN（M3 系）人声前端：MUSE_VOCAL_ENGINE=m3 时替代 SOME 骨干。

与 some_frontend.transcribe_some 等位（同返回格式 {notes, note_count}，
note={onset, offset, pitch}），下游 LRC 增强层（trim_vocal_offsets /
filter_breath_notes / align_chars / split_melisma / fill）全部复用。

推理：mel(22050/hop220/128bins) → onset/offset/pitch/vowel 头 →
decode_notes_vowel（元音起始切分跨字长音）→ 音域门 40-84。
长曲滑窗（60s 窗 + 2s 重叠平均，GRU 窗界上下文重置被重叠区掩盖）。

ckpt 默认 m3st500（MIR-ST500 域内微调版，官方基准 82 曲宏平均 COn
0.720 / COnP 0.654 / COnPOff 0.401）；MUSE_M3_CKPT 可换任意
vocal_crnn 系 ckpt（模型结构从 ck["args"] 自描述恢复）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent           # src/
_SE = _HERE.parent                                # score_extraction/
_CKPT_DEFAULT = _SE / "model" / "vocal_crnn" / "m3st500.best.ckpt"

# 滑窗推理参数（帧率 ≈100.23fps：hop 220 / SR 22050）
_WIN_FRAMES = 6000        # ≈60s
_OVERLAP_FRAMES = 200     # ≈2s

_INS: dict = {}          # {"model", "device", "ds", "dec"}


def _load():
    if _INS:
        return
    import importlib.util as ilu
    import os

    mods = {}
    for name, fname in (("ds", "dataset"), ("mdl", "model"), ("dec", "decode")):
        spec = ilu.spec_from_file_location(
            f"_vcfe_{fname}", _SE / "train" / "vocal_crnn" / f"{fname}.py")
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[name] = mod

    ckpt_path = Path(os.environ.get("MUSE_M3_CKPT", str(_CKPT_DEFAULT)))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[m3] ckpt missing: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    a = ck.get("args", {})
    conv_ch = a.get("conv_ch", "48,96,192,384")
    if isinstance(conv_ch, str):                   # ckpt 里存的是 "48,96,192,384"
        conv_ch = tuple(int(v) for v in conv_ch.split(","))
    model = mods["mdl"].VocalCRNN(
        with_vowel=a.get("with_vowel", True),
        conv_ch=conv_ch,
        gru_hidden=a.get("gru_hidden", 256),
        gru_layers=a.get("gru_layers", 2))
    model.load_state_dict(ck["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    _INS.update(model=model, device=device, ds=mods["ds"], dec=mods["dec"],
                ckpt=str(ckpt_path))
    logger.info("[m3] loaded %s (%.1fM params) on %s (with_vowel=%s)",
                ckpt_path.name,
                sum(p.numel() for p in model.parameters()) / 1e6,
                device, a.get("with_vowel", True))


@torch.no_grad()
def _heads_windowed(mel2d: np.ndarray) -> dict:
    """滑窗跑四/五头，重叠区帧级取均值拼接成全曲帧级 dict（numpy）。"""
    model, device = _INS["model"], _INS["device"]
    T = mel2d.shape[0]
    keys = ("onset", "offset", "pitch", "vowel")
    wbuf = np.zeros(T, dtype=np.float32)
    acc = {k: np.zeros((T,), dtype=np.float32) for k in ("onset", "offset")}
    acc["pitch"] = np.zeros((T, 46), dtype=np.float32)
    acc["vowel"] = np.zeros((T,), dtype=np.float32)

    start = 0
    while start < T:
        end = min(start + _WIN_FRAMES, T)
        x = (torch.from_numpy(mel2d[start:end].T.astype("float32"))
             .unsqueeze(0).unsqueeze(0).to(device))          # (1,1,128,t)
        out = model(x, torch.tensor([end - start], device=device))
        n = end - start
        for k in keys:
            if k == "pitch":
                v = out[k][0].float().cpu().numpy()
            elif k == "vowel" and "vowel" not in out:
                continue
            else:
                v = torch.sigmoid(out[k][0]).float().cpu().numpy()
            acc[k][start:end] += v
        wbuf[start:end] += 1.0
        start = end if end == T else end - _OVERLAP_FRAMES

    w = np.maximum(wbuf, 1.0)
    wcol = w[:, None] if acc["pitch"].ndim > 1 else w
    return {k: (v / wcol if v.ndim > 1 else v / w) for k, v in acc.items()}


def transcribe_m3(audio_path: str,
                  line_boundaries: list[float] | None = None) -> dict:
    """wav → {"notes": [...], "note_count": int}（instrument_class 恒 melody）。

    line_boundaries 当前未用（元音头已承担跨字切分），签名与 SOME 前端对齐。
    """
    import librosa

    _load()
    ds, dec = _INS["ds"], _INS["dec"]
    wav, _ = librosa.load(audio_path, sr=ds.SR, mono=True)
    mel = ds.wav_to_mel(wav).cpu().numpy()        # (T,128) float32
    buf = _heads_windowed(mel)
    notes = dec.decode_notes_vowel(
        buf["onset"], buf["offset"], buf["pitch"], buf["vowel"])
    notes = dec.gate(notes)
    notes = [{"onset": n["onset"], "offset": n["offset"],
              "pitch": int(n["pitch"]), "velocity": 100,
              "confidence": 0.9,      # 与 some_frontend 字段集对齐
              "instrument_class": "melody"} for n in notes]
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    logger.info("[m3] %d notes from %s (vowel decode, gate [%d,%d])",
                len(notes), Path(audio_path).name, 40, 84)
    return {"notes": notes, "note_count": len(notes)}


if __name__ == "__main__":
    import json
    import sys
    import time

    logging.basicConfig(level=logging.INFO)
    ap_path = sys.argv[1] if len(sys.argv) > 1 else str(
        _SE / "output" / "vocal_swallow_diag" / "2 夏日已所剩无几" / "vocals.wav")
    t0 = time.time()
    r = transcribe_m3(ap_path)
    print(f"{r['note_count']} notes in {time.time() - t0:.1f}s; "
          f"first 6: {json.dumps(r['notes'][:6], ensure_ascii=False)}")
