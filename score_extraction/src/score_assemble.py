"""Layer 5 (partial): Music21 Score assembly from clean note lists.

All post-processing (HMM, thresholding, harmonic pruning, voice assign)
is handled upstream. This module ONLY handles:
  - Converting notes to music21 objects
  - Adding tempo, key, time signature
  - Voice-aware measure division
  - Dynamics (simple: first-note initial, then on-change)
"""
import logging
from music21 import stream, meter, key, tempo, dynamics, note, harmony

logger = logging.getLogger(__name__)

TEMPO_TERMS = [
    (40, "Grave"), (60, "Largo"), (66, "Adagio"), (76, "Andante"),
    (108, "Moderato"), (120, "Allegro"), (156, "Vivace"), (176, "Presto"), (999, "Prestissimo"),
]

STANDARD_DURATIONS = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def _bpm_to_tempo_term(bpm):
    for t, term in TEMPO_TERMS:
        if bpm < t:
            return term
    return "Allegro"


def _quantize_duration(ql: float) -> float:
    ql = max(ql, 0.125)
    return min(STANDARD_DURATIONS, key=lambda d: abs(d - ql))


def assemble_score(instrument_name: str, notes: list[dict], bpm: float,
                   key_signature: str, time_signature: str = "4/4",
                   chords: list[dict] | None = None) -> stream.Score:
    """
    Build a music21 Score from clean notes.

    Notes are placed at absolute onset times (converted to quarterLength).
    music21 handles measure division automatically via makeMeasures().

    Args:
        notes: list of {"onset": sec, "offset": sec, "pitch": MIDI, "amplitude": 0-1, "voice": int}
    """
    s = stream.Score()
    part = stream.Part()
    part.partName = instrument_name

    # Metadata
    part.append(tempo.MetronomeMark(number=int(bpm), text=_bpm_to_tempo_term(bpm)))
    part.append(meter.TimeSignature(time_signature))
    ks_parts = key_signature.strip().split()
    tonic, mode = ks_parts[0], ks_parts[1] if len(ks_parts) > 1 else "major"
    part.append(key.Key(tonic, mode))

    # Chords (guitar only, from madmom)
    if chords:
        for ch in chords:
            offset_ql = ch["start"] * bpm / 60.0
            cs = harmony.ChordSymbol(ch["label"])
            part.insert(offset_ql, cs)

    # Dynamics: percentile-based, mark only on change
    amps = [n.get("amplitude", 0.1) for n in notes]
    if len(amps) >= 3:
        sorted_amps = sorted(amps)
        n_amps = len(sorted_amps)
        thresholds = {
            "pp": sorted_amps[int(n_amps * 0.10)],
            "p":  sorted_amps[int(n_amps * 0.30)],
            "mp": sorted_amps[int(n_amps * 0.50)],
            "mf": sorted_amps[int(n_amps * 0.70)],
            "f":  sorted_amps[int(n_amps * 0.90)],
            "ff": float("inf"),
        }

        def _amp_to_dyn(amp):
            for label, thresh in thresholds.items():
                if amp <= thresh:
                    return label
            return "ff"
    else:
        def _amp_to_dyn(amp):
            return "mf"

    # Insert notes at absolute offsets
    last_dyn = None
    for n in notes:
        ql_onset_raw = n["onset"] * bpm / 60.0
        # Quantize onset to 1/32-note grid (0.125 ql) so that measure-boundary
        # splits also produce expressible MusicXML durations.
        ql_onset = round(ql_onset_raw * 8) / 8

        dur_sec = n["offset"] - n["onset"]
        ql_dur = _quantize_duration(dur_sec * bpm / 60.0)

        n_obj = note.Note(n["pitch"])
        n_obj.duration.quarterLength = ql_dur

        # Voice assignment (piano: upper/lower)
        voice_id = n.get("voice", 1)
        if voice_id == 2:
            n_obj.stemDirection = "down"

        # Dynamic marking (only on change)
        dyn_label = _amp_to_dyn(n.get("amplitude", 0.1))
        if dyn_label != last_dyn:
            d = dynamics.Dynamic(dyn_label)
            part.insert(ql_onset, d)
            last_dyn = dyn_label

        part.insert(ql_onset, n_obj)

    # Let music21 auto-divide into measures
    part.makeMeasures(inPlace=True)

    s.insert(0, part)
    logger.info(f"Score assembled: {instrument_name} ({len(notes)} notes)")
    return s
