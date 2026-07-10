"""Layer 4: Note-level post-processing.

Steps:
  1. Onset refinement via spectral flux
  2. Harmonic pruning (remove overtone false positives)
  3. Merge duplicate notes (same pitch, overlapping time)
  4. Convert frame indices → seconds
  5. Estimate amplitude per note
"""
import logging
import numpy as np
import librosa

from src.config import config

logger = logging.getLogger(__name__)

# Harmonic intervals (in semitones) that often appear as overtones
HARMONIC_INTERVALS = [
    12,   # octave (1st overtone)
    19,   # octave + fifth (2nd overtone)
    24,   # 2 octaves (3rd overtone)
    28,   # 2 octaves + major third (4th overtone)
]


def _spectral_flux(audio: np.ndarray, sr: int, hop_length: int,
                   center_frame: int, window: int = 5) -> int:
    """
    Compute spectral flux in a window around center_frame.
    Returns the frame with maximum flux (best onset candidate).
    """
    n_fft = 2048
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    flux = np.sum(np.maximum(spec[:, 1:] - spec[:, :-1], 0), axis=0)  # (T_spec - 1,)

    start = max(0, center_frame - window)
    end = min(len(flux), center_frame + window + 1)
    if end <= start:
        return center_frame

    return int(start + np.argmax(flux[start:end]))


def _harmonic_penalty(pitch_a: int, pitch_b: int) -> int:
    """Return harmonic order if pitch_b is a harmonic of pitch_a, else 0."""
    diff = pitch_b - pitch_a
    for order, interval in enumerate(HARMONIC_INTERVALS, start=1):
        if diff == interval:
            return order
    return 0


def _prune_harmonics(notes: list[dict]) -> list[dict]:
    """
    Remove notes that are likely overtones of stronger, lower notes.
    A note is pruned if:
      - It is exactly a harmonic interval above a lower note
      - Both notes overlap in time
      - The lower note has higher confidence/amplitude
    """
    if len(notes) < 2:
        return notes

    # Sort by pitch ascending
    sorted_notes = sorted(notes, key=lambda n: n["pitch"])
    to_prune = set()

    for i, n_low in enumerate(sorted_notes):
        for j in range(i + 1, len(sorted_notes)):
            n_high = sorted_notes[j]
            pitch_diff = n_high["pitch"] - n_low["pitch"]

            # Only check harmonic intervals
            if pitch_diff not in HARMONIC_INTERVALS:
                if pitch_diff > HARMONIC_INTERVALS[-1]:
                    break  # too far apart
                continue

            # Check temporal overlap
            if n_low["onset"] < n_high["offset"] and n_high["onset"] < n_low["offset"]:
                # Lower note must have higher confidence/amplitude
                low_conf = n_low.get("confidence", 0.5)
                high_conf = n_high.get("confidence", 0.5)
                if low_conf > high_conf * 1.2:
                    to_prune.add(j)
                    logger.debug(
                        f"  Pruned harmonic: pitch {n_high['pitch']} "
                        f"(overtone of {n_low['pitch']})"
                    )

    pruned = [n for i, n in enumerate(sorted_notes) if i not in to_prune]
    pruned.sort(key=lambda n: n["onset"])
    return pruned


def _merge_duplicates(notes: list[dict], max_gap_sec: float = 0.05) -> list[dict]:
    """Merge notes with same pitch that are very close in time."""
    if len(notes) < 2:
        return notes

    notes = sorted(notes, key=lambda n: (n["pitch"], n["onset"]))
    merged = []
    i = 0
    while i < len(notes):
        current = dict(notes[i])
        j = i + 1
        while j < len(notes) and notes[j]["pitch"] == current["pitch"]:
            gap = notes[j]["onset"] - current["offset"]
            if gap <= max_gap_sec:
                current["offset"] = max(current["offset"], notes[j]["offset"])
                current["confidence"] = max(current["confidence"], notes[j]["confidence"])
                j += 1
            else:
                break
        merged.append(current)
        i = j

    merged.sort(key=lambda n: n["onset"])
    return merged


def refine_notes(candidates: list[dict], audio: np.ndarray = None,
                 sr: int = None, hop_length: int = None) -> list[dict]:
    """
    Full note-level post-processing pipeline.

    Args:
        candidates: list of dicts from frame_post.process_frames()
        audio: raw audio array (optional, for onset refinement and amplitude)
        sr: sample rate
        hop_length: hop length used in transcription

    Returns:
        list of cleaned note dicts with keys:
        onset, offset (in seconds), pitch, confidence, amplitude
    """
    sr = sr or config.SR
    hop_length = hop_length or config.HOP_LENGTH
    frame_sec = hop_length / sr

    # Step 1: Convert frames → seconds
    notes = []
    for c in candidates:
        onset_sec = c["onset_frame"] * frame_sec
        offset_sec = c["offset_frame"] * frame_sec
        notes.append({
            "onset": round(onset_sec, 4),
            "offset": round(offset_sec, 4),
            "pitch": c["pitch"],
            "confidence": c["confidence"],
            "amplitude": 0.1,
        })

    logger.info(f"Note post-processing: {len(notes)} input notes")

    # Step 2: Onset refinement with spectral flux
    if audio is not None and len(notes) > 0:
        for n in notes:
            center_frame = int(n["onset"] / frame_sec)
            refined_frame = _spectral_flux(audio, sr, hop_length, center_frame)
            n["onset"] = round(refined_frame * frame_sec, 4)

    # Step 3: Harmonic pruning
    notes = _prune_harmonics(notes)
    logger.info(f"  After harmonic pruning: {len(notes)} notes")

    # Step 4: Merge duplicates
    notes = _merge_duplicates(notes)
    logger.info(f"  After merge: {len(notes)} notes")

    # Step 5: Estimate amplitude per note
    if audio is not None:
        for n in notes:
            start_samp = int(n["onset"] * sr)
            end_samp = int(n["offset"] * sr)
            if end_samp > start_samp and start_samp < len(audio):
                segment = audio[start_samp:min(end_samp, len(audio))]
                n["amplitude"] = float(np.sqrt(np.mean(segment ** 2))) if len(segment) > 0 else 0.1

    # Sort by onset
    notes.sort(key=lambda n: n["onset"])

    return notes
