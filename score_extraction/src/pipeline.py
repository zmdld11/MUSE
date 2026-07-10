"""VER2.0 Pipeline: 5-layer AMT architecture.

Layer 1: Acoustic frontend (librosa load)
Layer 2: Transcription model (basic-pitch raw onset/frame probs)
Layer 3: Frame-level post-processing (HMM + threshold + CC)
Layer 4: Note-level post-processing (onset refine + harmonic prune + merge)
Layer 5: Notation assembly (voice + score + export)
"""
import logging
import os
import json

import librosa
import numpy as np

from src.config import config
from src.bpm_detect import detect_bpm
from src.source_separate import separate_tracks
from src.transcriber import transcribe
from src.frame_post import process_frames
from src.note_post import refine_notes
from src.voice_assign import assign_voices
from src.chord_detect import detect_chords
from src.key_estimate import estimate_key
from src.score_assemble import assemble_score
from src.export_score import export_score

logger = logging.getLogger(__name__)


def _process_instrument(audio_path: str, inst_name: str, bpm: float,
                        chords: list[dict] | None = None) -> bool:
    """Run Layers 2-5 on one instrument track. Returns True on success."""
    # Layer 2: Transcription
    logger.info(f"  [{inst_name}] Layer 2: Transcribing...")
    result = transcribe(audio_path)
    if result["frame_probs"].size == 0 or result["frame_probs"].max() < 0.01:
        logger.warning(f"  [{inst_name}] No signal detected, skipping")
        return False

    # Layer 3: Frame-level post-processing
    logger.info(f"  [{inst_name}] Layer 3: Frame post-processing...")
    model_sr = result.get("sr", config.SR)
    model_hop = result.get("hop_length", config.HOP_LENGTH)
    candidates = process_frames(
        result["onset_probs"],
        result["frame_probs"],
        hop_length=model_hop,
        sr=model_sr,
    )
    if len(candidates) == 0:
        logger.warning(f"  [{inst_name}] No candidates after frame post-processing")
        return False

    # Load audio for onset refinement (use model's SR for consistency)
    audio, _sr = librosa.load(audio_path, sr=model_sr, mono=True)

    # Layer 4: Note-level post-processing
    logger.info(f"  [{inst_name}] Layer 4: Note post-processing...")
    notes = refine_notes(candidates, audio, model_sr, model_hop)
    if len(notes) == 0:
        logger.warning(f"  [{inst_name}] No notes after refinement")
        return False

    logger.info(f"  [{inst_name}] {len(notes)} notes after refinement")

    # Layer 5a: Voice assignment
    notes = assign_voices(notes, inst_name)

    # Layer 5b: Key estimation (from all notes accumulated)
    key_sig = estimate_key(notes)

    # Layer 5c: Score assembly
    note_count = len(notes)
    score = assemble_score(
        inst_name, notes, bpm, key_sig,
        config.DEFAULT_TIME_SIG, chords if inst_name == "guitar" else None,
    )

    # Layer 5d: Export
    output_stem = os.path.join(
        config.OUTPUT_DIR,
        os.path.basename(audio_path).rsplit(".", 1)[0],
        inst_name,
    )
    os.makedirs(os.path.dirname(output_stem), exist_ok=True)
    try:
        result_path = export_score(score, output_stem)
        if result_path is not None:
            logger.info(f"  [{inst_name}] Exported ({note_count} notes): {result_path}")
        else:
            logger.error(f"  [{inst_name}] Export returned None (failed)")
            return False
    except Exception as exc:
        logger.error(f"  [{inst_name}] Export crashed: {exc}", exc_info=True)
        return False

    return True


def run_pipeline(audio_path: str, output_dir: str | None = None) -> str:
    """Full 5-layer AMT pipeline."""
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, song_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger.info(f"=== Pipeline VER2.0 start: {song_name} ===")

    # Step 1: BPM
    bpm = detect_bpm(audio_path) or config.DEFAULT_BPM
    if bpm > 120:
        logger.info(f"  Halving BPM: {bpm} → {bpm / 2:.1f}")
        bpm = round(bpm / 2, 1)
    logger.info(f"[1/5] BPM: {bpm}")

    # Step 2: Source separation (Layer 1)
    logger.info("[2/5] Source separation (htdemucs_6s)...")
    tracks = separate_tracks(audio_path, output_dir)

    # Step 3-5: Process each track
    all_notes = {}
    for inst_name, wav_path in tracks.items():
        if inst_name == "drums":
            all_notes[inst_name] = []
            continue

        # Piano: use original audio for transcription (basic-pitch works better)
        src_audio = audio_path if inst_name == "piano" else wav_path
        try:
            _process_instrument(src_audio, inst_name, bpm)
        except Exception as exc:
            logger.error(f"  [{inst_name}] Pipeline crashed: {exc}", exc_info=True)

    # Write info.json
    info = {"song": song_name, "bpm": bpm, "time_signature": config.DEFAULT_TIME_SIG}
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    logger.info(f"=== Pipeline complete: {output_dir} ===")
    return output_dir
