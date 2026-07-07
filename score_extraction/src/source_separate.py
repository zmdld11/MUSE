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
}


def separate_tracks(audio_path: str, output_dir: str) -> dict[str, str]:
    """
    使用 demucs Python API 分离音轨。
    绕过 torchcodec 依赖：用 librosa 预加载音频。
    """
    os.makedirs(output_dir, exist_ok=True)

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
