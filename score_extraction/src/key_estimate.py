import logging
import numpy as np

logger = logging.getLogger(__name__)

# K-S key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _notes_to_pitch_class_vector(notes):
    vec = np.zeros(12)
    for n in notes:
        pc = n["pitch"] % 12
        duration = n["offset"] - n["onset"]
        vec[pc] += duration
    total = vec.sum()
    return vec / total if total > 0 else vec


def estimate_key(all_notes):
    if len(all_notes) == 0:
        return "C major"

    vec = _notes_to_pitch_class_vector(all_notes)
    best_corr, best_key = -999, "C major"

    for tonic in range(12):
        corr = np.corrcoef(vec, np.roll(MAJOR_PROFILE, tonic))[0, 1]
        if corr > best_corr:
            best_corr, best_key = corr, f"{PITCH_NAMES[tonic]} major"

    for tonic in range(12):
        corr = np.corrcoef(vec, np.roll(MINOR_PROFILE, tonic))[0, 1]
        if corr > best_corr:
            best_corr, best_key = corr, f"{PITCH_NAMES[tonic]} minor"

    logger.info(f"Estimated key: {best_key} (corr={best_corr:.3f})")
    return best_key
