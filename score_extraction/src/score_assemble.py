"""Layer 5: Music21 Score assembly from clean note lists.

All post-processing (HMM, thresholding, harmonic pruning, voice assign)
is handled upstream. This module ONLY handles:
  - Converting notes to music21 objects
  - Adding tempo, key, time signature
  - Voice-aware measure division
  - Dynamics (simple: first-note initial, then on-change)

Uses manual measure construction + voice objects to avoid makeMeasures()
issues with cross-measure note splitting.
"""
import logging
from collections import defaultdict
from music21 import stream, meter, key, tempo, dynamics, note, chord, tie

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


def _quantize_duration_max(ql: float, max_ql: float) -> float:
    """Quantize to a standard duration, but never exceed max_ql."""
    ql = max(ql, 0.125)
    candidates = [d for d in STANDARD_DURATIONS if d <= max_ql + 0.001]
    if not candidates:
        return _quantize_duration(ql)
    return min(candidates, key=lambda d: abs(d - ql))


def _split_note_across_measures(ql_onset: float, ql_dur_quantized: float,
                                 ql_per_measure: float = 4.0):
    """Split a note that crosses measure boundaries into tied fragments.

    Returns list of (measure_idx, offset_in_measure, frag_dur, tie_type).
    tie_type is None (no tie), 'start', 'continue', or 'stop'.

    Each fragment's quantized duration is clamped to the available space
    in its measure, preventing measure overflow.
    """
    if ql_dur_quantized <= 0:
        return []

    measure_idx = int(ql_onset / ql_per_measure)
    measure_start = measure_idx * ql_per_measure
    measure_end = measure_start + ql_per_measure
    note_end = ql_onset + ql_dur_quantized

    # Fits entirely within one measure
    if note_end <= measure_end + 0.001:
        return [(measure_idx, ql_onset - measure_start, ql_dur_quantized, None)]

    # Crosses at least one boundary
    fragments = []
    remaining = ql_dur_quantized
    current_onset = ql_onset

    while remaining > 0.001:
        m_idx = int(current_onset / ql_per_measure)
        _m_start = m_idx * ql_per_measure
        m_end = _m_start + ql_per_measure
        offset_in_m = current_onset - _m_start
        available = m_end - current_onset

        frag_dur = min(remaining, available)
        frag_dur_q = _quantize_duration_max(frag_dur, frag_dur)

        # Determine tie type
        is_first = (len(fragments) == 0)
        remaining_after = remaining - frag_dur_q

        if is_first and remaining_after <= 0.001:
            tie_type = None
        elif is_first:
            tie_type = "start"
        elif remaining_after <= 0.001:
            tie_type = "stop"
        else:
            tie_type = "continue"

        fragments.append((m_idx, offset_in_m, frag_dur_q, tie_type))

        current_onset += frag_dur_q
        remaining -= frag_dur_q

        if remaining < 0.001:
            break
        if frag_dur_q <= 0.001:
            logger.warning(f"  _split_note stuck: onset={ql_onset:.3f}, "
                           f"remaining={remaining:.4f}, frag_dur_q={frag_dur_q:.4f}")
            break

    return fragments


def assemble_score(instrument_name: str, notes: list[dict], bpm: float,
                   key_signature: str, time_signature: str = "4/4",
                   chords: list[dict] | None = None) -> stream.Score:
    """
    Build a music21 Score from clean notes using manual measure construction.

    Notes are placed at absolute onset times (converted to quarterLength).
    Measures are created manually. Uses per-voice streams within each measure
    to correctly represent polyphonic overlapping notes.

    Args:
        notes: list of {"onset": sec, "offset": sec, "pitch": MIDI,
                         "amplitude": 0-1, "voice": int}
    """
    sorted_notes = sorted(notes, key=lambda n: n["onset"])

    s = stream.Score()
    part = stream.Part()
    part.partName = instrument_name

    if "/" in time_signature:
        beats_per_measure = int(time_signature.split("/")[0])
    else:
        beats_per_measure = 4
    ql_per_measure = float(beats_per_measure)

    # Metadata
    ts = meter.TimeSignature(time_signature)
    ks_parts = key_signature.strip().split()
    tonic, mode = ks_parts[0], ks_parts[1] if len(ks_parts) > 1 else "major"
    ks = key.Key(tonic, mode)
    mm = tempo.MetronomeMark(number=int(bpm), text=_bpm_to_tempo_term(bpm))

    # Dynamics: percentile-based
    amps = [n.get("amplitude", 0.1) for n in sorted_notes]
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

    # Phase 1: Pre-compute all fragments with voice grouping
    # Key: (measure_idx, voice) -> list of (offset_in_m, dur, tie, pitch, amp, is_first, frag_idx)
    voice_contents = defaultdict(list)

    for n in sorted_notes:
        ql_onset_raw = n["onset"] * bpm / 60.0
        ql_onset = round(ql_onset_raw * 8) / 8

        dur_sec = n["offset"] - n["onset"]
        ql_dur_raw = dur_sec * bpm / 60.0
        ql_dur = _quantize_duration(ql_dur_raw)

        fragments = _split_note_across_measures(ql_onset, ql_dur, ql_per_measure)

        for frag_idx, (mi, off, d, tt) in enumerate(fragments):
            voice_val = n.get("voice", 1)
            voice_contents[(mi, voice_val)].append({
                "offset": off,
                "dur": d,
                "tie": tt,
                "pitch": n["pitch"],
                "amp": n.get("amplitude", 0.1),
                "is_first": (frag_idx == 0),
            })

    # Phase 2: Build measures with voice streams
    measures_dict = {}
    last_dyn = {}

    for (mi, voice_val), items in sorted(voice_contents.items()):
        if mi not in measures_dict:
            m = stream.Measure()
            m.number = mi + 1
            measures_dict[mi] = m

        m_ref = measures_dict[mi]

        # Sort items by offset for sequential insertion into voice
        items_sorted = sorted(items, key=lambda it: it["offset"])

        # Create a Voice for this voice within the measure
        v = stream.Voice()
        v.id = str(voice_val)

        # Group items at the same offset into chords within this voice
        i = 0
        while i < len(items_sorted):
            same_offset = []
            current_off = items_sorted[i]["offset"]
            while i < len(items_sorted) and abs(items_sorted[i]["offset"] - current_off) < 0.001:
                same_offset.append(items_sorted[i])
                i += 1

            if len(same_offset) == 1:
                it = same_offset[0]
                n_obj = note.Note(it["pitch"])
                n_obj.duration.quarterLength = it["dur"]
                if it["tie"] is not None:
                    n_obj.tie = tie.Tie(it["tie"])
                if voice_val == 2:
                    n_obj.stemDirection = "down"
                v.insert(current_off, n_obj)

                # Dynamic (first fragment of note only)
                if it["is_first"]:
                    dyn_label = _amp_to_dyn(it["amp"])
                    prev = last_dyn.get(voice_val)
                    if dyn_label != prev:
                        d_obj = dynamics.Dynamic(dyn_label)
                        v.insert(current_off, d_obj)
                        last_dyn[voice_val] = dyn_label
            else:
                # Chord: multiple notes at the same offset in the same voice
                note_list = []
                for it in same_offset:
                    n_obj = note.Note(it["pitch"])
                    n_obj.duration.quarterLength = it["dur"]
                    if it["tie"] is not None:
                        n_obj.tie = tie.Tie(it["tie"])
                    if voice_val == 2:
                        n_obj.stemDirection = "down"
                    note_list.append(n_obj)

                c = chord.Chord(note_list)
                v.insert(current_off, c)

                # Dynamic from first item
                first_items = [it for it in same_offset if it["is_first"]]
                if first_items:
                    dyn_label = _amp_to_dyn(first_items[0]["amp"])
                    prev = last_dyn.get(voice_val)
                    if dyn_label != prev:
                        d_obj = dynamics.Dynamic(dyn_label)
                        v.insert(current_off, d_obj)
                        last_dyn[voice_val] = dyn_label

        # Insert voice into measure
        m_ref.insert(0, v)

    # Ensure at least one measure
    if not measures_dict:
        m0 = stream.Measure()
        m0.number = 1
        measures_dict[0] = m0

    # Add metadata to first measure
    m0 = measures_dict[min(measures_dict.keys())]
    m0.insert(0, mm)
    m0.insert(0, ks)
    m0.insert(0, ts)

    # Append measures in order
    for idx in sorted(measures_dict.keys()):
        part.append(measures_dict[idx])

    s.insert(0, part)
    logger.info(f"Score assembled: {instrument_name} ({len(sorted_notes)} notes, "
                f"{len(measures_dict)} measures)")
    return s
