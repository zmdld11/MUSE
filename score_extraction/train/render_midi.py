"""MIDI -> audio + label matrices for supervised training.

Uses FluidSynth (via pyFluidSynth) for realistic piano rendering when the
FluidSynth DLL and SoundFont are available.  Falls back to enhanced additive
synthesis (ADSR + 8 harmonics + stretch tuning + velocity-dependent timbre).

FluidSynth DLL path is auto-configured on first import.
"""
import logging
import os
import sys

import numpy as np
import pretty_midi

from train.config import train_config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-detect: update cfg.FLUIDSYNTH_ENABLED if FluidSynth is usable
# ---------------------------------------------------------------------------
_probe_attempted = False


def _ensure_probe():
    """Run FluidSynth probe once and update config."""
    global _probe_attempted
    if _probe_attempted:
        return
    _probe_attempted = True
    try:
        ok = _probe_fluidsynth_inner()
        if ok:
            cfg.FLUIDSYNTH_ENABLED = True
            logger.info("FluidSynth detected – using realistic piano rendering")
        else:
            logger.info("FluidSynth unavailable – using additive synthesis fallback")
    except Exception as exc:
        logger.debug("FluidSynth probe error: %s", exc)


def _probe_fluidsynth_inner() -> bool:
    """Check whether FluidSynth DLL + SoundFont are usable (no logging)."""
    sf = _find_soundfont()
    if sf is None:
        return False
    try:
        import fluidsynth
        fs = fluidsynth.Synth(gain=0.2, samplerate=22050.0)
        sid = fs.sfload(sf)
        fs.program_select(0, sid, 0, 0)
        fs.noteon(0, 60, 100)
        import time
        time.sleep(0.1)
        fs.noteoff(0, 60)
        samples = fs.get_samples(512)
        fs.delete()
        if samples is not None and len(samples) > 0 and np.abs(samples).max() > 0:
            return True
    except Exception:
        pass
    return False

# ---------------------------------------------------------------------------
# FluidSynth library discovery (run once on import)
# ---------------------------------------------------------------------------
_FLUID_DLL_DIR = os.path.abspath(
    r"d:\program_project\MUSE\score_extraction\fluid_dll"
    r"\fluidsynth-v2.5.6-win10-x64-cpp11\bin"
)

if os.path.isdir(_FLUID_DLL_DIR):
    os.environ["PATH"] = _FLUID_DLL_DIR + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):  # Python 3.8+ on Windows
        os.add_dll_directory(_FLUID_DLL_DIR)


def _find_soundfont() -> str | None:
    """Locate FluidR3_GM SoundFont on disk."""
    candidates = [
        r"D:\program_project\MUSE\score_extraction\data\FluidR3_GM.sf2",
        r"D:\program_project\MUSE\data\FluidR3_GM.sf2",
        os.path.join(cfg.WORKSPACE_DIR, "data", "FluidR3_GM.sf2"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# FluidSynth rendering
# ---------------------------------------------------------------------------

def _render_with_fluidsynth(pm: pretty_midi.PrettyMIDI,
                             sr: int,
                             max_dur_sec: float,
                             program: int = 0) -> np.ndarray:
    """Render a PrettyMIDI object with FluidSynth and return float32 mono audio.

    Parameters
    ----------
    pm : PrettyMIDI
        Parsed MIDI data.
    sr : int
        Target sample rate (passed to synth).
    max_dur_sec : float
        Maximum duration to render.

    Returns
    -------
    np.ndarray (samples,) float32
    """
    import fluidsynth

    sf_path = _find_soundfont()
    if sf_path is None:
        raise FileNotFoundError("FluidR3_GM.sf2 not found")

    # Create synth at target sample rate
    fs = fluidsynth.Synth(gain=0.4, samplerate=float(sr))
    # Do NOT call start() — we use get_samples() for offline rendering.
    sfid = fs.sfload(sf_path)
    fs.program_select(0, sfid, 0, program)  # 0=Acoustic Grand, 4=EPiano, ...
    logger.debug("FluidSynth synth created (sr=%d)", sr)

    total_dur = min(pm.get_end_time(), max_dur_sec)
    if total_dur <= 0:
        fs.delete()
        return np.array([], dtype=np.float32)

    # Build sorted event list: (time, type, pitch, velocity, channel)
    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= total_dur:
                continue
            onset = n.start
            offset = min(n.end, total_dur)
            vel = min(max(int(n.velocity), 0), 127)
            events.append((onset, "on", n.pitch, vel, 0))
            events.append((offset, "off", n.pitch, 0, 0))

    events.sort(key=lambda x: (x[0], 0 if x[1] == "on" else 1))

    total_frames = int(total_dur * sr)
    audio = np.zeros(total_frames, dtype=np.float32)
    cursor = 0  # current write position in samples
    ev_idx = 0
    n_events = len(events)

    while cursor < total_frames and ev_idx < n_events:
        # Time of next event (in seconds)
        next_event_time = events[ev_idx][0]
        next_event_frame = int(next_event_time * sr)

        # If there's a gap before the next event, render it
        if next_event_frame > cursor:
            chunk_frames = min(next_event_frame, total_frames) - cursor
            raw = fs.get_samples(chunk_frames)  # returns flat int16 array
            # raw length should be chunk_frames * 2 (stereo)
            if raw is not None and len(raw) > 0:
                # Convert flat stereo to mono float32
                raw_float = raw.astype(np.float32) / 32768.0
                mono = raw_float[0::2] * 0.5 + raw_float[1::2] * 0.5
                n_copy = min(len(mono), total_frames - cursor)
                audio[cursor:cursor + n_copy] = mono[:n_copy]
            cursor = next_event_frame

        # Process ALL events at this timestamp
        while ev_idx < n_events and events[ev_idx][0] == next_event_time:
            _t, kind, pitch, vel, ch = events[ev_idx]
            if kind == "on":
                fs.noteon(ch, pitch, vel)
            else:
                fs.noteoff(ch, pitch)
            ev_idx += 1

    # Render remaining audio after all events
    if cursor < total_frames:
        remaining = total_frames - cursor
        raw = fs.get_samples(remaining)
        if raw is not None and len(raw) > 0:
            raw_float = raw.astype(np.float32) / 32768.0
            mono = raw_float[0::2] * 0.5 + raw_float[1::2] * 0.5
            n_copy = min(len(mono), total_frames - cursor)
            audio[cursor:cursor + n_copy] = mono[:n_copy]

    fs.delete()
    return audio


# ---------------------------------------------------------------------------
# Fallback: enhanced additive synthesis
# ---------------------------------------------------------------------------

def _synth_note_adsr(pitch_midi: int,
                     dur_sec: float,
                     sr: int,
                     velocity: float = 0.8) -> np.ndarray:
    """Synthesise one note with ADSR + 8 harmonics + stretch tuning.

    Parameters
    ----------
    pitch_midi : int
        MIDI note number 0-127.
    dur_sec : float
        Duration in seconds.
    sr : int
        Sample rate.
    velocity : float
        Normalised velocity 0..1 (affects brightness).

    Returns
    -------
    np.ndarray (samples,) float32
    """
    # Stretch tuning: 0.3 cents per note away from A4
    stretch_cents = 0.3 * (pitch_midi - 69)
    freq = 440.0 * (2 ** ((pitch_midi - 69 + stretch_cents / 100.0) / 12.0))

    n_samp = int(dur_sec * sr)
    if n_samp < 1:
        return np.array([], dtype=np.float32)
    t = np.arange(n_samp, dtype=np.float64) / sr

    # ADSR envelope
    a = min(0.01, dur_sec * 0.5)   # attack
    d = min(0.10, dur_sec * 0.3)   # decay
    s_lvl = 0.65                   # sustain level
    r = min(0.30, dur_sec * 0.3)   # release

    env = np.ones(n_samp, dtype=np.float64)
    a_samp = int(a * sr)
    d_samp = int(d * sr)
    r_start = max(0, n_samp - int(r * sr))

    if a_samp > 0:
        env[:a_samp] = np.linspace(0.0, 1.0, a_samp)
    if d_samp > 0 and (a_samp + d_samp) <= n_samp:
        env[a_samp:a_samp + d_samp] = np.linspace(1.0, s_lvl, d_samp)
    if r_start > 0:
        env[r_start:] = np.linspace(s_lvl, 0.0, n_samp - r_start)

    # Harmonics: (relative_amp, detune_cents)
    harmonics = [
        (1.00, 0.0),
        (0.70, 0.3),
        (0.45, 0.6),
        (0.30, 0.9),
        (0.20, 1.2),
        (0.12, 1.5),
        (0.07, 1.8),
        (0.04, 2.1),
    ]

    brightness = 0.5 + velocity * 0.5  # 0.5 .. 1.0
    wave = np.zeros(n_samp, dtype=np.float64)

    for i, (base_amp, detune_cents) in enumerate(harmonics, start=1):
        partial_freq = freq * i * (2 ** (detune_cents / 1200.0))
        amp = base_amp * (brightness ** (i - 1))
        wave += amp * np.sin(2 * np.pi * partial_freq * t)

    wave *= env
    wave *= velocity * 0.8

    peak = np.abs(wave).max()
    if peak > 0:
        wave /= peak * 1.05

    return wave.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _note_to_midi_bin(pitch):
    """Map MIDI pitch (0-127) to bin index (0-87)."""
    return pitch - cfg.MIDI_OFFSET


def render_midi(midi_path, sr=None, hop_length=None, max_dur_sec=None,
                program: int = 0):
    """Render a MIDI file to audio + training labels.

    Uses FluidSynth when available; falls back to additive synthesis.

    Parameters
    ----------
    midi_path : str
        Path to MIDI file.
    sr : int, optional
        Sample rate (default from config).
    hop_length : int, optional
        Hop length for label matrices.
    max_dur_sec : float, optional
        Truncate audio after this many seconds.

    Returns
    -------
    dict with keys
        audio : np.ndarray (samples,) float32
        onset_labels : np.ndarray (T, 88) float32
        frame_labels : np.ndarray (T, 88) float32
        sr : int
    """
    _ensure_probe()
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
                    "pitch": n.pitch,
                    "velocity": n.velocity / 127.0,
                })

    if not all_notes:
        raise ValueError(f"No valid notes in MIDI: {midi_path}")

    total_dur = min(max(n["offset"] for n in all_notes) + 1.0, max_dur_sec)
    n_samples = int(total_dur * sr)

    # --- Render audio ---
    audio = None

    if cfg.FLUIDSYNTH_ENABLED:
        try:
            audio = _render_with_fluidsynth(pm, sr, max_dur_sec, program)
            logger.debug("FluidSynth rendered %s",
                         os.path.basename(midi_path))
        except Exception as exc:
            logger.warning("FluidSynth failed for %s: %s; using fallback",
                           os.path.basename(midi_path), exc)

    if audio is None or len(audio) == 0:
        # Fallback: additive synthesis
        audio = np.zeros(n_samples, dtype=np.float32)
        for note in all_notes:
            onset_samp = int(note["onset"] * sr)
            offset_samp = int(min(note["offset"], total_dur) * sr)
            if offset_samp <= onset_samp:
                continue
            dur = (offset_samp - onset_samp) / sr
            wave = _synth_note_adsr(note["pitch"], dur, sr, note["velocity"])
            end = min(onset_samp + len(wave), n_samples)
            audio[onset_samp:end] += wave[:end - onset_samp]

        # Normalise
        peak = np.abs(audio).max()
        if peak > 0:
            audio /= peak * 1.1

    # --- Build label matrices ---
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


def probe_fluidsynth(quiet=False) -> bool:
    """Check whether FluidSynth DLL + SoundFont are usable.

    Returns True if rendering via FluidSynth will work.
    """
    ok = _probe_fluidsynth_inner()
    if not quiet:
        if ok:
            logger.info("FluidSynth probe OK – DLL + SoundFont ready")
        else:
            logger.warning("FluidSynth probe failed")
    return ok


# Run probe once at module load time
_ensure_probe()
