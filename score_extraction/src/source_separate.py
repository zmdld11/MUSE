import logging
import os

import librosa
import numpy as np
import soundfile as sf
import torch

from src.config import config

logger = logging.getLogger(__name__)

# htdemucs_6s stem name -> internal instrument name
TRACK_MAP = {
    "bass": "bass",
    "drums": "drums",
    "vocals": "vocals",
    "guitar": "guitar",
    "piano": "piano",
    "other": "other",
}


def separate_vocals_melband(audio_path: str, output_dir: str) -> str | None:
    """MelBand-RoFormer 人声分离（人声支线骨架 2026-08-28 X）。

    MSST 子进程推理（顶层模块与 SOME/ia-amt 同名互踩 → 必须隔离子进程）；
    产物缓存 output_dir/<basename>/vocals.wav。MIR-1K 实测 SI-SDR 中位
    +16.8dB vs htdemucs +11.8。失败返回 None（调用方回退 raw 直推人声类）。
    """
    import subprocess
    import sys

    msst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "external", "Music-Source-Separation-Training")
    cfg = os.path.join(msst, "configs", "KimberleyJensen",
                       "config_vocals_mel_band_roformer_kj.yaml")
    ckpt = os.path.join(os.path.dirname(msst), "roformer", "melband_vocals.ckpt")
    if not (os.path.exists(cfg) and os.path.exists(ckpt)):
        logger.warning("  [melband] 配置或权重缺失，跳过：%s / %s", cfg, ckpt)
        return None
    out_wav = os.path.join(os.path.abspath(output_dir),
                           os.path.splitext(os.path.basename(audio_path))[0],
                           "vocals.wav")
    if os.path.exists(out_wav):
        logger.info(f"  [melband] cached {out_wav}")
        return out_wav
    # MSST 按目录批处理：必须把音频拷进独立工作目录，避免殃及同目录其它歌
    work_dir = os.path.join(os.path.abspath(output_dir), "_melband_input")
    os.makedirs(work_dir, exist_ok=True)
    import shutil
    local_audio = os.path.join(work_dir, os.path.basename(audio_path))
    if not os.path.exists(local_audio):
        shutil.copy2(audio_path, local_audio)
    cmd = [sys.executable, os.path.join(msst, "inference.py"),
           "--model_type", "mel_band_roformer", "--config_path", cfg,
           "--start_check_point", ckpt,
           "--input_folder", work_dir,
           "--store_dir", os.path.abspath(output_dir)]
    logger.info("  [melband] MelBand-RoFormer 人声分离（子进程）...")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=msst)
    if r.returncode != 0 or not os.path.exists(out_wav):
        logger.warning("  [melband] 分离失败：%s", (r.stderr or "")[-500:])
        return None
    return out_wav


def separate_tracks(audio_path: str, output_dir: str) -> dict[str, str]:
    """
    使用 demucs Python API 分离音轨。
    绕过 torchcodec 依赖：用 librosa 预加载音频。
    结果缓存：output_dir 下 6 个 stem wav 齐全时直接复用（重跑免分离）。
    """
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    cached = {}
    for stem in TRACK_MAP.values():
        wav_path = os.path.join(output_dir, f"{basename}_{stem}.wav")
        if os.path.exists(wav_path):
            cached[stem] = wav_path
    if len(cached) == len(TRACK_MAP):
        logger.info(f"  [demucs] cached {len(cached)} stems in {output_dir}")
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading demucs model on {device}...")

    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model(config.DEMUCS_MODEL)
    model.to(device)
    model.eval()

    # 用 librosa 加载音频（避免 torchcodec）
    logger.info(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=config.SR, mono=False)

    # 确保是 stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2 and audio.shape[0] > 2:
        audio = audio[:2, :]  # 只取前两声道

    # (channels, samples) → torch tensor
    if audio.ndim == 2:
        # shape: (channels, samples)
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
    else:
        # shape: (samples, channels) — librosa default
        audio_tensor = torch.from_numpy(audio.T.astype(np.float32))

    # demucs expects: (batch=1, channels, samples)
    if audio_tensor.shape[0] != 2:
        # mono → stereo
        audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
    audio_tensor = audio_tensor.unsqueeze(0).to(device)

    logger.info(f"Audio shape: {audio_tensor.shape}, running separation...")

    with torch.no_grad():
        sources = apply_model(
            model,
            audio_tensor,
            device=device,
            shifts=1,         # 无 overlap-add 位移（更快）
            split=True,        # 分段处理大文件
            overlap=0.25,
            progress=True,
        )

    # sources shape: (1, num_sources, channels, samples)
    sources = sources.squeeze(0).cpu().numpy()  # (num_sources, channels, samples)
    # sources are ordered by model.sources
    source_names = model.sources

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    tracks = {}

    os.makedirs(output_dir, exist_ok=True)

    for idx, stem_name in enumerate(source_names):
        internal_name = TRACK_MAP.get(stem_name)
        if internal_name is None:
            continue

        # (channels, samples) → (samples, channels) for soundfile
        stem_audio = sources[idx]  # (channels, samples)
        stem_audio = stem_audio.T  # (samples, channels)

        wav_path = os.path.join(output_dir, f"{basename}_{internal_name}.wav")
        sf.write(wav_path, stem_audio, config.SR)
        tracks[internal_name] = wav_path
        logger.info(f"  {internal_name}: {wav_path}")

    if len(tracks) < 3:
        raise RuntimeError(f"demucs output incomplete: {list(tracks.keys())}")

    return tracks


