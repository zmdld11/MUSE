"""ia-amt (instrument-agnostic-amt) 前端：进程内推理 → 项目统一 note 事件格式。

模型 anime-song/instrument-agnostic-amt（MIT），Transkun Neural Semi-CRF 区间
解码。2026-08-21 采纳为吉他 AMT 前端（四口径实测见 progress.md；同一口径下
東の空 onset@50 0.3647 vs Riley fl 0.2232，虚無の先输出数量校准、时值中位
0.351s vs 0.163s，原始混音≈stem 输入）。

Windows/torch-cu118 兼容：复数 torch.abs 与 TorchScript GPU 融合都依赖
nvrtc-builtins DLL（本机轮子未携带），导入模型包前打等价补丁（无数值影响，
同 eval/ia_amt_run.py）。

模型单例缓存：进程内首次调用加载（HF 权重自动下载到 ~/.cache），后续复用。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1] / "external" / "ia-amt"

_MODEL_CACHE: dict = {}


def _apply_windows_torch_patches() -> None:
    import torch

    if getattr(torch, "_muse_ia_amt_patched", False):
        return

    _orig_abs = torch.abs

    def _safe_abs(x):
        if torch.is_tensor(x) and x.is_complex() and x.is_cuda:
            return torch.sqrt(x.real * x.real + x.imag * x.imag)
        return _orig_abs(x)

    _orig_tabs = torch.Tensor.abs

    def _safe_tabs(self):
        if self.is_complex() and self.is_cuda:
            return torch.sqrt(self.real * self.real + self.imag * self.imag)
        return _orig_tabs(self)

    torch.abs = _safe_abs
    torch.Tensor.abs = _safe_tabs
    try:
        torch._C._jit_override_can_fuse_on_gpu(False)
    except Exception:
        pass
    torch._muse_ia_amt_patched = True


if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(model_type: str, device: str | None):
    key = (model_type, device or "")
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    _apply_windows_torch_patches()
    import torch

    from infer import _load_model_and_settings  # 仓库根 shim（ia-amt/infer.py）

    ckpt_dir = _REPO
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, mcfg, settings = _load_model_and_settings(
        _resolve_checkpoint(model_type, ckpt_dir),
        device=torch.device(dev),
        window_ms_override=None,
        stride_ms_override=None,
        track_batch_size_override=None,
    )
    _MODEL_CACHE[key] = (model, mcfg, settings, torch.device(dev))
    logger.info(f"[ia-amt] loaded type={model_type} device={dev}")
    return _MODEL_CACHE[key]


def _resolve_checkpoint(model_type: str, repo: Path) -> Path:
    from instrument_agnostic_amt.cli.infer import _ensure_checkpoint

    return _ensure_checkpoint(None, model_type=model_type)


def transcribe_ia_amt(audio_path: str, model_type: str = "guitar",
                      device: str | None = None,
                      velocity: int = 100,
                      note_bias: float = 0.0,
                      merge_onset_ms: float | None = None) -> dict:
    """转写一个音频文件 → {"notes": [...], "instrument_counts": {...}}。

    note 字段与 ByteDance/Riley 前端对齐：onset/offset（秒）、pitch、velocity、
    confidence（恒 1.0，模型不输出逐音置信度）、instrument_class（36 类分类）。
    offset 即 Semi-CRF 区间右端点 + boundary head 亚帧修正，无需后处理。
    merge_onset_ms：窗口缝合时同音高 onset 合并窗（默认 50ms，None=仓库
    默认；快速段落黏音实验用，见 eval/canon_stickiness_ab.py）。
    """
    import torch

    from infer import _load_audio, run_inference
    from instrument_agnostic_amt.cli.infer import DEFAULT_INSTRUMENT_VOLUMES
    from instrument_agnostic_amt.taxonomy.instrument_classes import INSTRUMENT_CLASSES

    model, mcfg, settings, dev = _load(model_type, device)
    if float(note_bias) != 0.0:
        # frozen dataclass，replace 拷贝覆盖，不污染进程级模型缓存
        import dataclasses
        settings = dataclasses.replace(settings, note_bias=float(note_bias))
    waveform, _sr_in, _ch = _load_audio(Path(audio_path),
                                        target_sample_rate=int(mcfg.sample_rate))
    with torch.no_grad():
        notes, stats, _ = run_inference(
            model=model, waveform=waveform, model_config=mcfg, settings=settings,
            device=dev, amp_enabled=False, amp_dtype=None, velocity=velocity,
            disable_tqdm=True,
            **({} if merge_onset_ms is None
               else {"merge_onset_ms": float(merge_onset_ms)}),
        )
    sr = int(mcfg.sample_rate)
    out = []
    inst_counts: dict[str, int] = {}
    for n in notes:
        cls = (INSTRUMENT_CLASSES[int(n.instrument_id)]
               if 0 <= int(n.instrument_id) < len(INSTRUMENT_CLASSES) else "unknown")
        inst_counts[cls] = inst_counts.get(cls, 0) + 1
        out.append({
            "onset": round(int(n.start_sample) / sr, 4),
            "offset": round(int(n.end_sample) / sr, 4),
            "pitch": int(n.pitch),
            "velocity": max(1, min(127, int(n.velocity))),
            "confidence": 1.0,
            "instrument_class": cls,
        })
    logger.info(f"[ia-amt] {len(out)} notes ({model_type}) from {audio_path}; "
                f"instruments={inst_counts}")
    return {"notes": out, "instrument_counts": inst_counts}
