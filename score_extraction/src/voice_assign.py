"""Voice / stream assignment for notes.

For piano: cluster notes into upper (right-hand) and lower (left-hand) voices
           based on pitch distribution.
For guitar: delegate to guitar_tab.assign_guitar_fingering().
"""
import logging
import numpy as np

from src import guitar_tab

logger = logging.getLogger(__name__)

# Instruments that need voice splitting
VOICED_INSTRUMENTS = {"piano"}


def assign_voices(notes: list[dict], instrument: str = "piano") -> list[dict]:
    """
    Assign voice IDs to notes.

    Piano → split by pitch median into voice 1 (upper/right-hand) and voice 2 (lower/left-hand).
    Guitar → call guitar_tab.assign_guitar_fingering() for string/fret assignment.

    Args:
        notes: list of note dicts (must have "pitch" key)
        instrument: instrument name

    Returns:
        notes with "voice" field added (int, 1-based)
    """
    if instrument == "guitar":
        return guitar_tab.assign_guitar_fingering(notes)

    if instrument not in VOICED_INSTRUMENTS:
        for n in notes:
            n["voice"] = 1
        return notes

    if len(notes) < 2:
        for n in notes:
            n["voice"] = 1
        return notes

    # Piano: split by median pitch
    pitches = [n["pitch"] for n in notes]
    median_pitch = np.median(pitches)

    for n in notes:
        n["voice"] = 1 if n["pitch"] >= median_pitch else 2

    upper = sum(1 for n in notes if n["voice"] == 1)
    lower = sum(1 for n in notes if n["voice"] == 2)
    logger.info(f"Voice assignment (piano): upper={upper}, lower={lower}")

    return notes
