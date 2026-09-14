"""VV-SVT 塔式人声前端：MUSE_VOCAL_ENGINE=vt 时替代 SOME 骨干。

论文部署配置（vt_demo_local.py 同配方，2026-09-14 接线）：
  音符塔 model/vt_demo/vt_e4ssl_ext.best.ckpt
  （ST500 test82 官方口径 COn .744 / COnP .699 / COnPOff .501，
   解码 0.6/0.7/act0.3/uv0.08 + offset shift +30ms）

与 some_frontend.transcribe_some 等位（同返回格式 {notes, note_count}，
note={onset, offset, pitch}），下游 LRC 增强层（trim_vocal_offsets /
filter_breath_notes / align_chars / split_melisma / fill）全部复用。

推理：vocals stem → 16k（w2v2 前端输入，帧级标准化）→ VocalTowerSSL
三塔 onset/offset/pitch → decode_notes_tower → 音域门 40-84 → +30ms。
超 90s 长曲自动滑窗（90s 窗 + 2s 重叠均值；w2v2 显存 ~6.5G 档，空闲不足
自动落 CPU）。
MUSE_VT_CKPT 可换任意塔式 ckpt（结构从 ck["args"] 自描述恢复；
runtime/vt/ 是推理代码部署副本，训练侧在姊妹仓 VV-SVT）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent           # src/
_SE = _HERE.parent                                # score_extraction/
_RUNTIME = _SE / "runtime" / "vt"
_CKPT_DEFAULT = _SE / "model" / "vt_demo" / "vt_e4ssl_ext.best.ckpt"

# 解码参数 = 论文部署配方（vt_demo_local.py，勿改：改了与主表数字脱钩）
_PO_TH, _OFF_TH, _UV, _ACT_TH = 0.6, 0.7, 0.08, 0.3
_SHIFT_S = 0.030          # 塔式 offset 系统性偏早 −19.6ms 的校准
_CUDA_FREE_GB = 6.5       # w2v2-large 全曲前向的实测门槛

_INS: dict = {}          # {"model", "device", "ds", "dec"}


def _load_mod(fname: str):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(f"_vtfe_{fname}", _RUNTIME / f"{fname}.py")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_model(ckpt_path: Path, dev: str):
    """从 ck["args"] 自描述重建塔式模型（w2v2/mel 塔/CRNN 三型）。"""
    k = torch.load(ckpt_path, map_location=dev, weights_only=False)
    ta = k.get("args", {})
    ds = _INS["ds"]
    if ta.get("frontend") == "w2v2":
        mt = _load_mod("model_tower")
        wf = _load_mod("w2v2_frontend")
        if ta.get("ssl_head") == "lin":
            model = mt.VocalTowerLin(n_vowel_cls=ds.n_vowel_cls()).to(dev)
        else:
            model = mt.VocalTowerSSL(
                gru_hidden=int(ta.get("tower_gru", 256)),
                rope_layers=int(ta.get("rope_layers", 0)),
                n_vowel_cls=ds.n_vowel_cls()).to(dev)
        model.set_encoder(wf.build_w2v2(
            str(_SE / ta.get("ssl_ckpt", "external/w2v2_large"))).to(dev))
    elif ta.get("arch") == "tower":
        mt = _load_mod("model_tower")
        model = mt.VocalTowerCRNN(
            channels=tuple(int(c) for c in
                           str(ta.get("tower_ch", "48,64,96,128")).split(",")),
            gru_hidden=int(ta.get("tower_gru", 256)),
            rope_layers=int(ta.get("rope_layers", 0)),
            n_vowel_cls=ds.n_vowel_cls()).to(dev)
    else:
        md = _load_mod("model")
        model = md.VocalCRNN(
            with_vowel=bool(ta.get("with_vowel", False)),
            conv_ch=tuple(int(c) for c in
                          str(ta.get("conv_ch", "48,96,192,384")).split(",")),
            gru_hidden=int(ta.get("gru_hidden", 256)),
            gru_layers=int(ta.get("gru_layers", 2))).to(dev)
    missing, unexpected = model.load_state_dict(k["model"], strict=False)
    if unexpected or [m for m in missing if m != "head_vib_lw"]:
        raise RuntimeError(
            f"[vt] ckpt mismatch {ckpt_path.name}: missing={list(missing)} "
            f"unexpected={list(unexpected)}")
    model.eval()
    return model


def _load():
    if _INS:
        return
    import os

    ckpt_path = Path(os.environ.get("MUSE_VT_CKPT", str(_CKPT_DEFAULT)))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[vt] ckpt missing: {ckpt_path}")

    device = "cpu"
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 1e9 >= _CUDA_FREE_GB:
            device = "cuda"

    _INS["ds"] = _load_mod("dataset")
    model = _build_model(ckpt_path, device)
    _INS["dec"] = _load_mod("decode_tower")
    _INS.update(model=model, device=device, ckpt=str(ckpt_path))
    logger.info("[vt] loaded %s (%.1fM params) on %s",
                ckpt_path.name,
                sum(p.numel() for p in model.parameters()) / 1e6,
                device)


# 滑窗推理（2026-09-14 Everlasting 段错误教训：w2v2 全曲前向在 ~5min+ 曲
# 上爆显存=原生段错误不抛异常；90s 窗 + 2s 重叠均值，与 m3 前端同款）
_WIN_S = 90.0
_OVL_S = 2.0


@torch.no_grad()
def _forward(wav22: np.ndarray) -> dict:
    """wav22 = 22.05k mono float32 → 前向（超窗自动滑窗拼接），返回 numpy dict。"""
    from scipy.signal import resample_poly
    model, device = _INS["model"], _INS["device"]
    hop = 220                      # ds.SR 22050 的帧 hop
    win = int(_WIN_S * _INS["ds"].SR)
    ovl = int(_OVL_S * _INS["ds"].SR)

    def _one(seg22: np.ndarray) -> dict:
        w16 = resample_poly(seg22, 320, 441).astype(np.float32)   # → 16k
        w16 = (w16 - w16.mean()) / (w16.std() + 1e-7)
        o = model(torch.as_tensor(w16, device=device).unsqueeze(0),
                  torch.tensor([len(w16)], device=device),
                  out_frames=[1 + len(seg22) // 220])
        sig = lambda x: torch.sigmoid(x).float().cpu().numpy()
        return {k: (sig(v[0]) if k in ("onset", "offset", "onset_r", "offset_r",
                                       "pitch", "vowel")
                    else v[0].float().cpu().numpy())
                for k, v in o.items()}

    if len(wav22) <= win:
        return _one(wav22)

    # 滑窗：帧级累加取均值，重叠区掩盖 GRU 窗界上下文重置
    T = 1 + len(wav22) // hop
    acc: dict = {}
    cnt = np.zeros(T, dtype=np.float32)
    start = 0
    while start < len(wav22):
        end = min(start + win, len(wav22))
        out = _one(wav22[start:end])
        f0 = int(round(start / hop))
        n = min(len(out["onset"]), T - f0)
        for k, v in out.items():
            if len(v) > n:                 # 窗尾帧数对齐截断
                v = v[:n]
            if k not in acc:
                acc[k] = np.zeros((T,) + np.asarray(v).shape[1:],
                                  dtype=np.float64)
            acc[k][f0:f0 + n] += v
        cnt[f0:f0 + n] += 1.0
        start = end if end == len(wav22) else end - ovl

    w = np.maximum(cnt, 1.0)
    return {k: (v / w[:, None] if v.ndim > 1 else v / w).astype(np.float32)
            for k, v in acc.items()}


def transcribe_vt(audio_path: str,
                  line_boundaries: list[float] | None = None) -> dict:
    """wav → {"notes": [...], "note_count": int}（instrument_class 恒 melody）。

    line_boundaries 当前未用（塔式 offset 头承担精确切分），签名与 SOME
    前端对齐。
    """
    import librosa

    _load()
    ds, dec = _INS["ds"], _INS["dec"]
    wav, _ = librosa.load(audio_path, sr=ds.SR, mono=True)
    o = _forward(wav)
    notes = dec.gate(dec.decode_notes_tower(
        o["onset_r"], o["offset_r"], o["pitch"],
        po_th=_PO_TH, off_th=_OFF_TH, act_th=_ACT_TH,
        unvoiced_run_sec=_UV))
    for n in notes:
        n["offset"] = n["offset"] + _SHIFT_S
    notes = [n for n in notes if n["offset"] > n["onset"]]
    notes = [{"onset": n["onset"], "offset": n["offset"],
              "pitch": int(n["pitch"]), "velocity": 100,
              "confidence": 0.9,      # 与 some_frontend 字段集对齐
              "instrument_class": "melody"} for n in notes]
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    logger.info("[vt] %d notes from %s (tower decode .6/.7/uv.08 + shift "
                "%.0fms, gate [%d,%d])", len(notes), Path(audio_path).name,
                _SHIFT_S * 1000, 40, 84)
    return {"notes": notes, "note_count": len(notes)}


if __name__ == "__main__":
    import json
    import sys
    import time

    logging.basicConfig(level=logging.INFO)
    ap_path = sys.argv[1] if len(sys.argv) > 1 else str(
        _SE / "output" / "vocal_swallow_diag" / "2 夏日已所剩无几" / "vocals.wav")
    t0 = time.time()
    r = transcribe_vt(ap_path)
    print(f"{r['note_count']} notes in {time.time() - t0:.1f}s; "
          f"first 6: {json.dumps(r['notes'][:6], ensure_ascii=False)}")
