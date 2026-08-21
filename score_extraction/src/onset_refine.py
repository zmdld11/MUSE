"""Refine decoded note onsets with VER5 sub-frame predictions."""
from __future__ import annotations

import numpy as np


def refine_onset_times(notes, onset_shifts, hop_length=512, sr=22050):
    """Apply ``frame + shift`` without changing the candidate set."""
    if onset_shifts is None:
        return notes
    frame_sec = hop_length / sr
    refined = []
    for note in notes:
        frame = int(note.get(
            "onset_frame", round(note["onset_time"] / frame_sec)))
        pitch_bin = int(note["pitch_bin"])
        if not (0 <= frame < onset_shifts.shape[0] and
                0 <= pitch_bin < onset_shifts.shape[1]):
            refined.append(note)
            continue
        shift = float(np.clip(onset_shifts[frame, pitch_bin], 0.0, 1.0))
        raw_onset = frame * frame_sec
        onset = raw_onset + shift * frame_sec
        onset = min(onset, float(note["offset_time"]) - 0.010)
        onset = max(onset, raw_onset)
        refined.append({**note, "onset_time": float(onset)})
    return refined
