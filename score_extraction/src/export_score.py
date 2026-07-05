import logging, os
from music21 import musicxml

logger = logging.getLogger(__name__)


def export_score(score, output_path_stem: str) -> str:
    os.makedirs(os.path.dirname(output_path_stem) or ".", exist_ok=True)
    xml_path = output_path_stem + ".musicxml"
    score.write("musicxml", fp=xml_path)
    logger.info(f"Exported: {xml_path}")
    return xml_path
