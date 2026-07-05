import logging
import os
import subprocess
import sys

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
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", config.DEMUCS_MODEL,
        "-o", output_dir,
        audio_path,
    ]

    logger.info(f"Running demucs: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = f"demucs failed (rc={result.returncode}): {result.stderr[-500:]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    demucs_out = os.path.join(output_dir, config.DEMUCS_MODEL, basename)

    tracks = {}
    for stem_name, internal_name in TRACK_MAP.items():
        wav_path = os.path.join(demucs_out, f"{stem_name}.wav")
        if os.path.exists(wav_path):
            tracks[internal_name] = wav_path
            logger.info(f"  {internal_name}: {wav_path}")
        else:
            logger.warning(f"  {internal_name}: MISSING")

    if len(tracks) < 3:
        raise RuntimeError(f"demucs output incomplete: {list(tracks.keys())}")

    return tracks
