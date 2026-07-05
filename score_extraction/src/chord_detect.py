import logging

logger = logging.getLogger(__name__)


def detect_chords(wav_path: str) -> list[dict]:
    try:
        import madmom
        from madmom.features.chords import (
            CNNChordFeatureProcessor,
            CRFChordRecognitionProcessor,
        )

        feat_processor = CNNChordFeatureProcessor()
        chord_processor = CRFChordRecognitionProcessor()

        features = feat_processor(wav_path)
        chord_labels = chord_processor(features)

        chords = []
        for start, end, label in chord_labels:
            if label != "N":
                chords.append({
                    "start": round(float(start), 3),
                    "end": round(float(end), 3),
                    "label": str(label),
                })

        logger.info(f"Chords detected: {len(chords)} labels")
        return chords

    except ImportError:
        logger.warning("madmom not installed, skipping chord detection")
        return []
    except Exception as e:
        logger.warning(f"Chord detection failed: {e}")
        return []
