import logging
from music21 import stream, meter, key, tempo, dynamics, note, spanner, harmony, chord
from collections import defaultdict

logger = logging.getLogger(__name__)

TEMPO_TERMS = [
    (40, "Grave"), (60, "Largo"), (66, "Adagio"), (76, "Andante"),
    (108, "Moderato"), (120, "Allegro"), (156, "Vivace"), (176, "Presto"), (999, "Prestissimo"),
]
# Standard music durations in quarterLength
STANDARD_DURATIONS = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

# Quantization grid: multiples of 16th note (0.25 qL) in quarterLength space
# At any BPM, this is the finest rhythmic division we'll use
QUANTIZE_GRID_QL = 0.25  # 16th note


def _quantize_duration(ql: float) -> float:
    """Snap raw quarterLength to nearest standard music duration, floor=0.125."""
    ql = max(ql, 0.125)
    return min(STANDARD_DURATIONS, key=lambda d: abs(d - ql))


def _quantize_onset(seconds: float, seconds_per_qn: float) -> float:
    """Quantize onset (in seconds) to nearest 16th note grid (in quarterLength)."""
    ql = seconds / seconds_per_qn
    # Round to nearest grid point
    grid = QUANTIZE_GRID_QL
    ql_quantized = round(ql / grid) * grid
    return ql_quantized


def _bpm_to_tempo_term(bpm):
    for t, term in TEMPO_TERMS:
        if bpm < t:
            return term
    return "Allegro"


def _assign_relative_dynamics(notes):
    """Assign pp~ff based on percentile rank within this track's own amplitude range."""
    valid_amps = [n.get("amplitude", 0.1) for n in notes if n.get("fret") != -1]
    if len(valid_amps) < 3:
        for n in notes:
            n["dynamic"] = "mf"
        return notes

    sorted_amps = sorted(valid_amps)
    total = len(sorted_amps)
    p10 = sorted_amps[int(total * 0.10)]
    p30 = sorted_amps[int(total * 0.30)]
    p50 = sorted_amps[int(total * 0.50)]
    p70 = sorted_amps[int(total * 0.70)]
    p90 = sorted_amps[int(total * 0.90)]

    for n in notes:
        amp = n.get("amplitude", 0.1)
        if amp <= p10:
            n["dynamic"] = "pp"
        elif amp <= p30:
            n["dynamic"] = "p"
        elif amp <= p50:
            n["dynamic"] = "mp"
        elif amp <= p70:
            n["dynamic"] = "mf"
        elif amp <= p90:
            n["dynamic"] = "f"
        else:
            n["dynamic"] = "ff"

    return notes


def _detect_slides(notes, enable=True):
    """Detect glissando/slides. Only for monophonic instruments."""
    if not enable:
        return notes
    for i in range(1, len(notes)):
        gap = notes[i]["onset"] - notes[i - 1]["offset"]
        pdiff = abs(notes[i]["pitch"] - notes[i - 1]["pitch"])
        if gap < 0.1 and 0.5 < pdiff < 12:
            notes[i]["slide_from"] = notes[i - 1]["pitch"]
    return notes


def _clean_notes(notes, bpm):
    """Remove noise notes and cap unrealistic durations."""
    if not notes:
        return notes
    beat_sec = 60.0 / bpm
    max_dur_sec = 8 * beat_sec
    min_dur_sec = 0.03

    cleaned = []
    for n in notes:
        dur = n["offset"] - n["onset"]
        if dur < min_dur_sec:
            continue
        if dur > max_dur_sec:
            n = dict(n)
            n["offset"] = n["onset"] + max_dur_sec
        cleaned.append(n)
    return cleaned


def _parse_key_signature(ks):
    parts = ks.strip().split()
    tonic = parts[0]
    mode = parts[1] if len(parts) > 1 else "major"
    return tonic, mode


# Instruments where overlapping notes are expected (polyphonic — no slide detection)
POLYPHONIC_INSTRUMENTS = {"piano"}


def assemble_score(instrument_name, notes, bpm, key_signature,
                   time_signature="4/4", chords=None, is_guitar=False):
    s = stream.Score()
    part = stream.Part()
    part.partName = instrument_name
    part.append(tempo.MetronomeMark(number=int(bpm), text=_bpm_to_tempo_term(bpm)))
    part.append(meter.TimeSignature(time_signature))
    tonic, mode = _parse_key_signature(key_signature)
    part.append(key.Key(tonic, mode))

    if chords:
        for ch in chords:
            offset = ch["start"] * bpm / 60.0
            cs = harmony.ChordSymbol(ch["label"])
            part.insert(offset, cs)

    # Clean artifacts + cap durations
    notes = _clean_notes(notes, bpm)
    notes = _assign_relative_dynamics(notes)

    # Slides only for monophonic instruments (not piano)
    monophonic = instrument_name not in POLYPHONIC_INSTRUMENTS
    notes = _detect_slides(notes, enable=monophonic)

    # For monophonic instruments: insert sequentially (no voice assignment needed)
    if monophonic or is_guitar:
        prev_note = None
        for n in notes:
            ql_onset = n["onset"] * bpm / 60.0
            ql_dur = _quantize_duration((n["offset"] - n["onset"]) * bpm / 60.0)

            if n.get("fret") == -1:
                r = note.Rest()
                r.duration.quarterLength = ql_dur
                part.insert(ql_onset, r)
                continue

            n_obj = note.Note(n["pitch"])
            n_obj.duration.quarterLength = ql_dur

            if not is_guitar:
                dyn = n.get("dynamic", "mf")
                d = dynamics.Dynamic(dyn)
                d.displayText = dyn
                n_obj.articulations.append(d)
                if n.get("slide_from") and prev_note is not None:
                    gl = spanner.Glissando()
                    gl.addSpannedElements([prev_note, n_obj])
                    part.append(gl)

            part.insert(ql_onset, n_obj)
            prev_note = n_obj
    else:
        # ===== POLYPHONIC (piano) path — single voice =====
        # Quantize onsets and durations, group by quantized onset,
        # then insert all notes/chords directly into the part (no voice splitting).
        seconds_per_qn = 60.0 / bpm

        # Compute quantized attributes for each note
        for n in notes:
            n["_ql_onset"] = _quantize_onset(n["onset"], seconds_per_qn)

        # Group by quantized onset
        onset_groups = defaultdict(list)
        for n in notes:
            if n.get("fret") == -1:
                continue  # skip rests
            onset_groups[n["_ql_onset"]].append(n)

        # Build chord items and insert directly (single voice)
        for ql_onset in sorted(onset_groups.keys()):
            group = onset_groups[ql_onset]
            pitches = []
            max_ql_dur = 0
            for n in group:
                pitches.append(n["pitch"])
                ql_dur = _quantize_duration((n["offset"] - n["onset"]) * bpm / 60.0)
                if ql_dur > max_ql_dur:
                    max_ql_dur = ql_dur
            if max_ql_dur < 0.125:
                max_ql_dur = 0.125

            if len(pitches) == 1:
                n_obj = note.Note(pitches[0])
            else:
                n_obj = chord.Chord(pitches)
            n_obj.duration.quarterLength = max_ql_dur

            # Add dynamics from the first note in the group
            first_note = group[0]
            dyn = first_note.get("dynamic", "mf")
            d = dynamics.Dynamic(dyn)
            d.displayText = dyn
            n_obj.articulations.append(d)

            part.insert(ql_onset, n_obj)

    # Let music21 auto-divide into measures
    try:
        part.makeMeasures(inPlace=True)
    except Exception as e:
        logger.warning(f"makeMeasures failed: {e}")
        # Fallback: try with finalBarline=False
        try:
            part.makeMeasures(inPlace=True, finalBarline=False)
        except Exception as e2:
            logger.warning(f"makeMeasures fallback failed: {e2}")

    s.insert(0, part)
    logger.info(f"Score assembled: {instrument_name} ({len(notes)} notes)")
    return s
