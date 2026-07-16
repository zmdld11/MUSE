"""Analyze timing quantization effect on the canon test file.

Compares onset times before and after quantization, produces statistics
on shift magnitudes, and validates that the quantized output is playable.

Usage:
    conda activate score_build
    python test/analyze_quantization.py
"""
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("analyze_quantization")

# --- Patch refine_notes to capture pre-quantization onsets ------------
from src.note_post import refine_notes as _original_refine
from src.pipeline import _process_instrument as _original_process
from src import pipeline as pipeline_mod

# We'll capture notes before quantization by hooking into the pipeline
PRE_QUANT_NOTES = {}

def _hooked_process(audio_path, inst_name, bpm, chords=None):
    """Wrapper that saves pre-quantization notes."""
    # We need to call the original process but intercept between refine_notes and quantize_onsets
    # Since we can't easily hook in the middle, we'll just run the analysis manually.
    return _original_process(audio_path, inst_name, bpm, chords)

# Actually, let's use a simpler approach: run the pipeline step by step
from src.config import config
from src.transcriber import transcribe
from src.frame_post import process_frames
from src.note_post import refine_notes
from src.quantize_timing import quantize_onsets
from src.voice_assign import assign_voices
import librosa


def analyze_quantization(audio_path, inst_name, bpm):
    """Run quantization on one instrument and report statistics."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {inst_name}")
    logger.info(f"{'='*60}")

    result = transcribe(audio_path)
    if result["frame_probs"].size == 0 or result["frame_probs"].max() < 0.01:
        logger.warning("No signal")
        return

    model_sr = result.get("sr", config.SR)
    model_hop = result.get("hop_length", config.HOP_LENGTH)
    candidates = process_frames(
        result["onset_probs"], result["frame_probs"],
        hop_length=model_hop, sr=model_sr,
    )
    if not candidates:
        logger.warning("No candidates")
        return

    audio, _sr = librosa.load(audio_path, sr=model_sr, mono=True)
    notes = refine_notes(candidates, audio, model_sr, model_hop)
    if not notes:
        logger.warning("No notes")
        return

    pre_onsets = np.array([n["onset"] for n in notes])
    pre_offsets = np.array([n["offset"] for n in notes])
    pre_durs = pre_offsets - pre_onsets

    notes = quantize_onsets(notes, bpm, config.DEFAULT_TIME_SIG)

    post_onsets = np.array([n["onset"] for n in notes])
    post_offsets = np.array([n["offset"] for n in notes])
    post_durs = post_offsets - post_onsets

    shifts = post_onsets - pre_onsets
    abs_shifts = np.abs(shifts)
    dur_changes = post_durs - pre_durs

    # Stats
    n_snapped = np.sum(np.abs(shifts) > 0.0001)
    n_total = len(notes)

    logger.info(f"  Total notes: {n_total}")
    logger.info(f"  Notes shifted: {n_snapped} ({100*n_snapped/n_total:.1f}%)")
    logger.info(f"  Shift magnitude (mean): {np.mean(abs_shifts)*1000:.2f} ms")
    logger.info(f"  Shift magnitude (std):  {np.std(abs_shifts)*1000:.2f} ms")
    logger.info(f"  Shift magnitude (max):  {np.max(abs_shifts)*1000:.2f} ms")
    logger.info(f"  Shift median:           {np.median(abs_shifts)*1000:.2f} ms")

    # Percentiles
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(abs_shifts, p) * 1000
        logger.info(f"  Shift {p}th percentile:     {val:.2f} ms")

    # Within-threshold check
    interval_16th = 60.0 / (bpm * 4.0)
    threshold = interval_16th / 3.0
    within_threshold = np.sum(abs_shifts <= threshold + 0.001)
    beyond_threshold = np.sum(abs_shifts > threshold + 0.001)
    logger.info(f"  Shifts within threshold ({threshold*1000:.1f}ms): {within_threshold}")
    logger.info(f"  Shifts beyond threshold (rubato/ornament): {beyond_threshold}")

    # Duration preservation check
    max_dur_change = np.max(np.abs(dur_changes))
    logger.info(f"  Max duration change: {max_dur_change*1000:.4f} ms")
    logger.info(f"  (should be ~0 — durations should be preserved)")

    # Zero-duration check
    zero_dur = np.sum(post_durs <= 0)
    if zero_dur > 0:
        logger.warning(f"  WARNING: {zero_dur} notes have zero/negative duration after quantization!")

    # Inter-onset interval stats
    intervals = np.diff(post_onsets)
    logger.info(f"\n  Post-quantization inter-onset intervals:")
    logger.info(f"    Mean: {np.mean(intervals)*1000:.2f} ms")
    logger.info(f"    Std:  {np.std(intervals)*1000:.2f} ms")
    logger.info(f"    Min:  {np.min(intervals)*1000:.2f} ms")

    # Grid alignment: how many onsets land exactly on 16th/triplet grid
    t0 = post_onsets[0]
    aligned_16th = 0
    aligned_triplet = 0
    unaligned = 0
    for o in post_onsets:
        rel = o - t0
        dev_16th = abs(rel - round(rel / interval_16th) * interval_16th)
        trip_int = 60.0 / (bpm * 3.0)
        dev_trip = abs(rel - round(rel / trip_int) * trip_int)
        if dev_16th < 0.001:
            aligned_16th += 1
        elif dev_trip < 0.001:
            aligned_triplet += 1
        else:
            unaligned += 1

    logger.info(f"\n  Grid alignment:")
    logger.info(f"    16th grid:     {aligned_16th} ({100*aligned_16th/n_total:.1f}%)")
    logger.info(f"    Triplet grid:  {aligned_triplet} ({100*aligned_triplet/n_total:.1f}%)")
    logger.info(f"    Unaligned:     {unaligned} ({100*unaligned/n_total:.1f}%)")

    return notes


if __name__ == "__main__":
    audio_path = os.path.join(
        os.path.dirname(__file__), "..", "output", "canon_test.wav",
    )
    if not os.path.exists(audio_path):
        logger.error(f"Test file not found: {audio_path}")
        sys.exit(1)

    # Detect BPM
    from src.bpm_detect import detect_bpm
    bpm = detect_bpm(audio_path) or config.DEFAULT_BPM
    if bpm > 120:
        logger.info(f"Halving BPM: {bpm} -> {bpm/2:.1f}")
        bpm = round(bpm / 2, 1)
    logger.info(f"BPM: {bpm}")

    # Analyze piano (main instrument)
    notes = analyze_quantization(audio_path, "piano", bpm)

    # Also analyze separated tracks
    # (Source separation is expensive, so only do it if files exist)
    track_dir = os.path.join(os.path.dirname(__file__), "..", "output", "canon_test")
    for track_name in ["piano", "bass", "vocals", "guitar"]:
        track_path = os.path.join(track_dir, f"canon_test_{track_name}.wav")
        if os.path.exists(track_path):
            analyze_quantization(track_path, track_name, bpm)
