import logging
import warnings

import numpy as np
import torch
import torchcrepe
import librosa

from src.config import config

logger = logging.getLogger(__name__)


def detect_pitch_mono(wav_path: str) -> list[dict]:
    try:
        audio, sr = librosa.load(wav_path, sr=config.SR, mono=True)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pitch, periodicity = torchcrepe.predict(
                audio_tensor, sr,
                hop_length=config.HOP_LENGTH,
                fmin=50, fmax=2000,
                model="full", batch_size=1024,
                device=device,
                return_periodicity=True,
            )

        pitch = pitch.squeeze().cpu().numpy()
        confidence = periodicity.squeeze().cpu().numpy()
        frame_time = config.HOP_LENGTH / sr

        notes = _frames_to_notes(pitch, confidence, frame_time, audio)

        if len(notes) == 0:
            logger.warning(f"No notes detected: {wav_path}")
        return notes

    except Exception as e:
        logger.warning(f"Mono pitch detection failed ({wav_path}): {e}")
        return []


def detect_pitch_piano(wav_path: str) -> list[dict]:
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        _, _, note_events = predict(wav_path, model_or_model_path=ICASSP_2022_MODEL_PATH)

        notes = []
        for e in note_events:
            # basic-pitch returns 5-element tuples:
            # (start_time, end_time, pitch, amplitude, note_on_velocities)
            notes.append({
                "onset": float(e[0]),
                "offset": float(e[1]),
                "pitch": int(e[2]),
                "velocity": int(round(float(e[3]))),
                "confidence": float(e[3]),
            })

        logger.info(f"basic-pitch: {len(notes)} notes (piano)")
        return notes

    except ImportError:
        logger.warning("basic-pitch not installed, piano -> crepe mono")
        return detect_pitch_mono(wav_path)
    except Exception as e:
        logger.warning(f"basic-pitch failed ({e}), piano -> crepe mono")
        return detect_pitch_mono(wav_path)


def hz_to_midi(freq_hz: float) -> float:
    """Convert frequency in Hz to MIDI note number."""
    return 69.0 + 12.0 * np.log2(freq_hz / 440.0)


def _frames_to_notes(
    pitch_hz: np.ndarray, confidence: np.ndarray,
    frame_time: float, audio: np.ndarray
) -> list[dict]:
    MIN_CONF = 0.3
    notes = []
    i = 0
    n = len(pitch_hz)

    # convert Hz to MIDI once, suppress log(<=0) warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        pitch_midi = np.where(
            pitch_hz > 0,
            69.0 + 12.0 * np.log2(pitch_hz / 440.0),
            np.nan,
        )

    while i < n:
        if confidence[i] < MIN_CONF or np.isnan(pitch_midi[i]):
            i += 1
            continue

        start_frame = i
        start_pitch_midi = pitch_midi[i]

        while (
            i < n
            and confidence[i] >= MIN_CONF
            and not np.isnan(pitch_midi[i])
            and abs(pitch_midi[i] - start_pitch_midi) < config.SLIDE_PITCH_THRESHOLD
        ):
            i += 1

        end_frame = i

        start_sample = int(start_frame * frame_time * config.SR)
        end_sample = int(end_frame * frame_time * config.SR)
        if end_sample > start_sample and start_sample < len(audio):
            amp = float(np.sqrt(np.mean(audio[start_sample:end_sample] ** 2)))
        else:
            amp = 0.1

        notes.append({
            "onset": round(start_frame * frame_time, 3),
            "offset": round(end_frame * frame_time, 3),
            "pitch": int(round(start_pitch_midi)),
            "confidence": float(np.mean(confidence[start_frame:end_frame])),
            "amplitude": amp,
        })

    return notes
