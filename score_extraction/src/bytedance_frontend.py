"""ByteDance/Kong high-resolution event frontend.

The official model already performs note onset/offset regression, velocity
regression, and sustain-pedal event detection.  This module exposes that as a
stable event frontend for the notation pipeline and diagnostic scripts.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import librosa

logger = logging.getLogger(__name__)

_transcriber = None


def default_checkpoint() -> str:
    configured = os.environ.get("MUSE_BYTEDANCE_CKPT")
    if configured:
        return configured
    return str(Path.home() / "piano_transcription_inference_data" /
               "note_F1=0.9677_pedal_F1=0.9186.pth")


def _get_transcriber(device: str | None = None):
    global _transcriber
    if _transcriber is not None:
        return _transcriber
    import torch
    from piano_transcription_inference import PianoTranscription

    checkpoint = default_checkpoint()
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"ByteDance checkpoint not found: {checkpoint}; set MUSE_BYTEDANCE_CKPT")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading ByteDance HR checkpoint on %s: %s", device, checkpoint)
    _transcriber = PianoTranscription(device=device, checkpoint_path=checkpoint)
    return _transcriber


def _parse_note_events(events) -> list[dict]:
    output = []
    if not events:
        return output
    if isinstance(events[0], dict):
        for event in events:
            output.append({
                "onset": float(event["onset_time"]),
                "offset": float(event["offset_time"]),
                "pitch": int(event["midi_note"]),
                "velocity": int(event.get("velocity", 64)),
            })
        return output
    for onset, offset, pitch, velocity in events:
        output.append({
            "onset": float(onset),
            "offset": float(offset),
            "pitch": int(pitch),
            "velocity": int(velocity),
        })
    return output


def _parse_pedal_events(events) -> list[dict]:
    output = []
    if not events:
        return output
    if isinstance(events[0], dict):
        for event in events:
            output.append({
                "onset": float(event["onset_time"]),
                "offset": float(event["offset_time"]),
            })
        return output
    for onset, offset in events:
        output.append({"onset": float(onset), "offset": float(offset)})
    return output


def transcribe_bytedance(audio_path: str, device: str | None = None) -> dict:
    """Return high-resolution note and sustain-pedal events."""
    from piano_transcription_inference import sample_rate

    transcriber = _get_transcriber(device=device)
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    logger.info("ByteDance inference: %s (%.1fs)", audio_path, len(audio) / sample_rate)
    result = transcriber.transcribe(audio, None)
    notes = _parse_note_events(result.get("est_note_events", []))
    pedals = _parse_pedal_events(result.get("est_pedal_events", []))
    logger.info("ByteDance events: notes=%d pedals=%d", len(notes), len(pedals))
    return {
        "notes": notes,
        "pedal_events": pedals,
        "sample_rate": int(sample_rate),
        "checkpoint": default_checkpoint(),
    }


def write_bytedance_midi(events: dict, midi_path: str | Path,
                         bpm: float = 120.0, program: int = 0,
                         instrument_name: str = "Piano") -> Path:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=480)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=program, name=instrument_name)
    for note in events.get("notes", []):
        onset = max(0.0, float(note["onset"]))
        offset = max(onset + 0.03, float(note["offset"]))
        instrument.notes.append(pretty_midi.Note(
            velocity=max(1, min(127, int(note.get("velocity", 64)))),
            pitch=int(note["pitch"]), start=onset, end=offset))
    for pedal in events.get("pedal_events", []):
        onset = max(0.0, float(pedal["onset"]))
        offset = max(onset + 0.01, float(pedal["offset"]))
        instrument.control_changes.append(pretty_midi.ControlChange(
            number=64, value=127, time=onset))
        instrument.control_changes.append(pretty_midi.ControlChange(
            number=64, value=0, time=offset))
    pm.instruments.append(instrument)
    midi_path = Path(midi_path)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(midi_path))
    return midi_path
