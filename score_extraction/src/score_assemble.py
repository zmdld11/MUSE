import logging
from music21 import stream, meter, key, tempo, dynamics, note, spanner

logger = logging.getLogger(__name__)

TEMPO_TERMS = [
    (40, "Grave"), (60, "Largo"), (66, "Adagio"), (76, "Andante"),
    (108, "Moderato"), (120, "Allegro"), (156, "Vivace"), (176, "Presto"), (999, "Prestissimo"),
]
DYNAMIC_MAP = [
    (0.01, "pp"), (0.03, "p"), (0.06, "mp"), (0.12, "mf"), (0.25, "f"), (0.5, "ff"),
]


def _bpm_to_tempo_term(bpm):
    for t, term in TEMPO_TERMS:
        if bpm < t:
            return term
    return "Allegro"


def _amp_to_dynamic(amp):
    for t, dyn in DYNAMIC_MAP:
        if amp < t:
            return dyn
    return "ff"


def _detect_dynamics(notes):
    for i in range(1, len(notes)):
        pa = notes[i - 1].get("amplitude", 0.1)
        ca = notes[i].get("amplitude", 0.1)
        if ca > pa * 1.3:
            notes[i]["crescendo"] = True
        elif pa > ca * 1.3:
            notes[i]["diminuendo"] = True
    return notes


def _detect_slides(notes):
    for i in range(1, len(notes)):
        gap = notes[i]["onset"] - notes[i - 1]["offset"]
        pdiff = abs(notes[i]["pitch"] - notes[i - 1]["pitch"])
        if gap < 0.1 and 0.5 < pdiff < 12:
            notes[i]["slide_from"] = notes[i - 1]["pitch"]
    return notes


def _parse_key_signature(ks):
    """Parse a key signature string like 'C major' or 'g minor' into (tonic, mode)."""
    parts = ks.strip().split()
    tonic = parts[0]
    mode = parts[1] if len(parts) > 1 else "major"
    return tonic, mode


def assemble_score(instrument_name, notes, bpm, key_signature,
                   time_signature="4/4", chords=None, is_guitar=False):
    s = stream.Score()
    part = stream.Part()
    part.partName = instrument_name
    part.append(tempo.MetronomeMark(number=int(bpm)))
    part.append(meter.TimeSignature(time_signature))
    tonic, mode = _parse_key_signature(key_signature)
    part.append(key.Key(tonic, mode))

    notes = _detect_dynamics(notes)
    notes = _detect_slides(notes)

    current_offset = 0.0
    prev_note = None
    for n in notes:
        if n.get("fret") == -1:
            r = note.Rest()
            r.duration.quarterLength = max((n["offset"] - n["onset"]) * bpm / 60.0, 0.25)
            part.append(r)
            prev_note = None
            continue

        n_obj = note.Note(n["pitch"])
        n_obj.duration.quarterLength = max((n["offset"] - n["onset"]) * bpm / 60.0, 0.25)

        if not is_guitar:
            # Insert dynamic marking at this note's offset (accumulated)
            dyn = _amp_to_dynamic(n.get("amplitude", 0.1))
            part.insert(current_offset, dynamics.Dynamic(dyn))

            if n.get("crescendo"):
                part.insert(current_offset, dynamics.Crescendo())
            if n.get("diminuendo"):
                part.insert(current_offset, dynamics.Diminuendo())
            if n.get("slide_from") and prev_note is not None:
                gl = spanner.Glissando()
                gl.addSpannedElements([prev_note, n_obj])
                part.append(gl)

        part.append(n_obj)
        current_offset += n_obj.duration.quarterLength
        prev_note = n_obj

    s.insert(0, part)
    logger.info(f"Score assembled: {instrument_name} ({len(notes)} notes)")
    return s
