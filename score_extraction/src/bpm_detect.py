import logging
import librosa

logger = logging.getLogger(__name__)


def detect_bpm(audio_path: str, sr: int = 44100) -> float | None:
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])

        if bpm < 40 or bpm > 250:
            logger.warning(f"BPM {bpm} out of range, using fallback")
            return None

        logger.info(f"BPM detected: {bpm:.1f}")
        return round(bpm, 1)

    except Exception as e:
        logger.warning(f"BPM detection failed: {e}")
        return None
