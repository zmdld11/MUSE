"""MIDI → audio + label matrices for supervised training."""
import logging
import numpy as np
import pretty_midi

from train.config import train_config as cfg

logger = logging.getLogger(__name__)


def _synth_note(pitch_midi, dur_sec, sr=22050):
    """Generate one note with additive synthesis (fundamental + 4 harmonics + decay)."""
    freq = 440.0 * (2 ** ((pitch_midi - 69) / 12.0))
    t = np.arange(int(dur_sec * sr)) / sr
    env = np.exp(-2.0 * t / max(dur_sec, 0.01))
    wave = (
        0.6 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.10 * np.sin(2 * np.pi * 3 * freq * t)
        + 0.05 * np.sin(2 * np.pi * 4 * freq * t)
    ) * env
    return wave.astype(np.float32)


def _note_to_midi_bin(pitch):
    """Map MIDI pitch (0-127) to bin index (0-87)."""
    return pitch - cfg.MIDI_OFFSET


def render_midi(midi_path, sr=None, hop_length=None, max_dur_sec=None):
    """
    Render a MIDI file to audio + training labels.

    Args:
        midi_path: path to .mid file
        sr: sample rate (default from config)
        hop_length: hop length for label matrices
        max_dur_sec: truncate audio to this duration

    Returns:
        dict with:
            audio: np.ndarray (samples,) float32
            onset_labels: np.ndarray (T, 88) float32
            frame_labels: np.ndarray (T, 88) float32
            sr: int
    """
    sr = sr or cfg.SR
    hop_length = hop_length or cfg.HOP_LENGTH
    max_dur_sec = max_dur_sec or cfg.MAX_DUR_SEC

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        raise ValueError(f"Failed to parse MIDI: {midi_path}") from e

    # Collect all non-drum notes
    all_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pitch_bin = _note_to_midi_bin(n.pitch)
            if 0 <= pitch_bin < cfg.N_MIDI:
                all_notes.append({
                    "onset": n.start,
                    "offset": n.end,
                    "pitch_bin": pitch_bin,
                    "velocity": n.velocity / 127.0,
                })

    if not all_notes:
        raise ValueError(f"No valid notes in MIDI: {midi_path}")

    total_dur = min(max(n["offset"] for n in all_notes) + 1.0, max_dur_sec)
    n_samples = int(total_dur * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    for note in all_notes:
        onset_samp = int(note["onset"] * sr)
        offset_samp = int(min(note["offset"], total_dur) * sr)
        if offset_samp <= onset_samp:
            continue
        dur = (offset_samp - onset_samp) / sr
        wave = _synth_note(note["pitch_bin"] + cfg.MIDI_OFFSET, dur, sr)
        wave *= note["velocity"]
        end = min(onset_samp + len(wave), n_samples)
        audio[onset_samp:end] += wave[:end - onset_samp]

    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak * 1.1

    # Build label matrices
    n_frames = 1 + (n_samples - 1) // hop_length
    onset_labels = np.zeros((n_frames, cfg.N_MIDI), dtype=np.float32)
    frame_labels = np.zeros((n_frames, cfg.N_MIDI), dtype=np.float32)

    for note in all_notes:
        onset_frame = int(note["onset"] * sr / hop_length)
        offset_frame = int(note["offset"] * sr / hop_length)
        pb = note["pitch_bin"]

        if onset_frame < n_frames:
            onset_labels[onset_frame, pb] = 1.0
        for f in range(onset_frame, min(offset_frame, n_frames)):
            frame_labels[f, pb] = 1.0

    return {
        "audio": audio,
        "onset_labels": onset_labels,
        "frame_labels": frame_labels,
        "sr": sr,
    }
