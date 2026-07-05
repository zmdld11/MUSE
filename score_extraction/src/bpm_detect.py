import logging
import librosa
from src.config import config

logger = logging.getLogger(__name__)


def detect_bpm(audio_path: str) -> float | None:
    try:
        y, sr = librosa.load(audio_path, sr=config.SR)

        if len(y) < config.SR * 2:  # at least 2 seconds
            logger.warning(f"Audio too short ({len(y)/sr:.1f}s), cannot detect BPM")
            return None

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])

        if bpm < 40 or bpm > 250:
            logger.warning(f"BPM {bpm} out of range, returning None")
            return None

        logger.info(f"BPM detected: {bpm:.1f}")
        return round(bpm, 1)

    except Exception as e:
        logger.warning(f"BPM detection failed: {e}")
        return None
