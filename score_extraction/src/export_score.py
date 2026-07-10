"""MusicXML export with fallback for music21 compatibility issues."""

import logging, os
from copy import deepcopy
from music21 import musicxml, stream, note as note21, dynamics as dyn21, harmony as harm21

logger = logging.getLogger(__name__)

# Element types we consider "non-essential" for stripped export
_NON_ESSENTIAL_CLASSES = (
    dyn21.Dynamic,
    harm21.ChordSymbol,
)


def _strip_to_pure_notes(score):
    """Return a new Score with only notes (no dynamics, chords, tempo markings)."""
    s2 = stream.Score()
    for part in score.parts:
        p2 = stream.Part()
        p2.partName = part.partName
        measures = list(part.recurse().getElementsByClass(stream.Measure))
        if measures:
            for m in measures:
                m2 = deepcopy(m)
                for el in list(m2.recurse().getElementsByClass(_NON_ESSENTIAL_CLASSES)):
                    try:
                        m2.remove(el)
                    except Exception:
                        pass
                p2.append(m2)
        else:
            from music21 import measure
            p2.append(measure.Measure())
        s2.insert(0, p2)
    return s2


def export_score(score, output_path_stem: str) -> str | None:
    """Export a music21 Score to MusicXML.

    Tries normal export first.  Falls back to stripping dynamics/chords
    if music21 9.x serialization crashes.

    Returns the path to the exported file, or None on total failure.
    """
    os.makedirs(os.path.dirname(output_path_stem) or ".", exist_ok=True)
    xml_path = output_path_stem + ".musicxml"

    # Attempt 1: normal export
    try:
        score.write("musicxml", fp=xml_path)
        logger.info(f"Exported: {xml_path}")
        return xml_path
    except Exception as exc:
        logger.warning(
            f"musicxml export failed (will retry without dynamics/chords): {exc}"
        )

    # Attempt 2: strip dynamics/chords and retry
    try:
        stripped = _strip_to_pure_notes(score)
        stripped.write("musicxml", fp=xml_path)
        logger.info(f"Exported (stripped): {xml_path}")
        return xml_path
    except Exception as exc:
        logger.error(f"musicxml export failed even after stripping: {exc}")
        return None
