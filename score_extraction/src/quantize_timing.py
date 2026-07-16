"""Timing Quantization Layer: align note onsets to the nearest rhythmic grid.

Grids supported:
  - 16th note: interval = 60 / (bpm * 4) seconds
  - 8th triplet: interval = 60 / (bpm * 3) * 2 seconds

Algorithm:
  1. (Segmented) Split the piece into overlapping segments of
     ``bars_per_segment`` bars (default 8).  Adjacent segments overlap
     by 1 bar so that the grid offset varies smoothly.
  2. For each segment, independently find the best-fit grid offset
     by scanning candidate shifts and minimising the total deviation
     from the nearest 16th-grid point.
  3. For each note (or chord group), if it falls in the overlap region
     of two segments, its snapped position is the average of both
     segments' snapped positions (linear smoothing).
  4. Snap to the nearest grid point (16th or triplet), provided the
     shift is within the allowable threshold (±interval/4).
  5. Notes whose deviation exceeds the threshold are left untouched
     (rubato / ornament preservation).
  6. When a note's onset moves, its offset (end time) shifts by the
     same delta so that the duration is preserved.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def _compute_grids(bpm: float) -> tuple[float, float, float]:
    """Compute grid intervals and the snap threshold.

    Returns
    -------
    interval_16th : float
        Seconds between consecutive 16th-note grid lines.
    interval_triplet : float
        Seconds between consecutive 8th-triplet grid lines.
    snap_threshold : float
        Maximum allowed deviation (seconds) for snapping.
        Currently interval_16th / 4 (~51 ms at 74 BPM).
    """
    interval_16th = 60.0 / (bpm * 4.0)       # e.g. ~0.203 s at 74 BPM
    interval_triplet = 60.0 / (bpm * 3.0)  # ~0.271 s at 74 BPM
    snap_threshold = interval_16th / 4.0      # ~0.051 s at 74 BPM
    return interval_16th, interval_triplet, snap_threshold


def _find_best_grid_offset(
    onsets: np.ndarray, interval_16th: float
) -> float:
    """Find the grid offset that best aligns onsets to the 16th grid.

    Scans candidate shifts from 0 to *interval_16th* (100 steps) and picks
    the one that minimises the total absolute deviation to the nearest
    16th-grid line.

    Parameters
    ----------
    onsets : np.ndarray
        All note onset times in seconds (sorted).
    interval_16th : float
        16th-note grid interval in seconds.

    Returns
    -------
    best_offset : float
        Shift (seconds) to apply to align to the grid.
    """
    if len(onsets) == 0:
        return 0.0

    # Normalise onsets relative to the first onset (remove absolute offset)
    t0 = onsets[0]
    relative = onsets - t0

    # Candidate shifts
    candidates = np.linspace(0.0, interval_16th, 101)

    best_offset = 0.0
    best_error = float("inf")

    for shift in candidates:
        shifted = relative - shift
        # Distance to nearest 16th-grid point
        grid_idx = np.round(shifted / interval_16th)
        aligned = grid_idx * interval_16th
        errors = np.abs(shifted - aligned)
        total_error = float(np.sum(errors))

        if total_error < best_error:
            best_error = total_error
            best_offset = shift

    logger.debug(
        "Grid offset: %.4f s  (total deviation %.4f s over %d onsets)",
        best_offset, best_error, len(onsets),
    )
    return best_offset


def _nearest_grid(
    onset: float,
    offset: float,
    t0: float,
    interval_16th: float,
    interval_triplet: float,
    threshold: float,
) -> tuple[float, str | None]:
    """Snap a single onset to the nearest grid point.

    Parameters
    ----------
    onset : float
        Original onset time in seconds.
    offset : float
        Best-fit grid offset found by :func:`_find_best_grid_offset`.
    t0 : float
        Reference time (first onset of the segment) used to normalise.
    interval_16th, interval_triplet : float
        Grid intervals.
    threshold : float
        Maximum allowed deviation for snapping.

    Returns
    -------
    snapped : float
        Snapped onset time.  Equals *onset* when deviation exceeds threshold.
    grid_type : str or None
        ``"16th"``, ``"triplet"``, or ``None`` if no snap was applied.
    """
    relative = onset - t0 - offset

    # 16th grid
    idx_16th = round(relative / interval_16th)
    pos_16th = t0 + offset + idx_16th * interval_16th
    dev_16th = abs(onset - pos_16th)

    # Triplet grid
    idx_trip = round(relative / interval_triplet)
    pos_trip = t0 + offset + idx_trip * interval_triplet
    dev_trip = abs(onset - pos_trip)

    if dev_16th <= threshold and dev_16th <= dev_trip:
        return pos_16th, "16th"
    elif dev_trip <= threshold:
        return pos_trip, "triplet"
    else:
        return onset, None


def quantize_onsets(
    notes: list[dict],
    bpm: float,
    time_sig: str = "4/4",
    bars_per_segment: int = 8,
) -> list[dict]:
    """Align note onsets to the nearest rhythmic grid point in-place.

    Grids
    -----
    - **16th note**:      ``interval = 60 / (bpm * 4)`` seconds
    - **8th triplet**:    ``interval = 60 / (bpm * 3) * 2`` seconds
    - **Snap threshold**: ``interval_16th / 4`` (more precise, reduces
      misalignment to adjacent grid lines).

    Algorithm
    ---------
    1. The piece is split into overlapping segments (default 8 bars,
       overlapped by 1 bar) so that each section's local tempo drift
       is captured by an independent grid offset.
    2. For each segment, find the best-fit 16th-grid offset by minimising
       total deviation of onsets within that segment.
    3. Group notes that share the same onset (chords) so they move
       together.
    4. For each chord group, snap to the nearest grid point (16th or
       triplet) if the deviation is within the allowable threshold.
       Notes in the 1-bar overlap between adjacent segments use the
       **average** of the two segments' snapped positions to avoid
       a discontinuity.
    5. Notes exceeding the threshold are left untouched (rubato /
       ornament).
    6. When onset moves, the offset (end) is shifted by the same delta.

    Parameters
    ----------
    notes : list[dict]
        Note list as produced by :func:`note_post.refine_notes`.  Each dict
        must contain ``"onset"`` and ``"offset"`` keys (seconds).  Modified
        in-place and returned.
    bpm : float
        Beats per minute (e.g. 73.9).
    time_sig : str
        Time signature string, default ``"4/4"``.  (Only the numerator is
        used for beat calculation.)
    bars_per_segment : int
        Number of bars per segment for local grid offset computation.
        ``≤ 2`` falls back to a single global offset.

    Returns
    -------
    list[dict]
        The same list with updated ``"onset"`` and ``"offset"`` fields.
    """
    if len(notes) == 0:
        return notes

    interval_16th, interval_triplet, snap_threshold = _compute_grids(bpm)
    logger.info(
        "Quantize timing: BPM=%.1f  16th=%.4fs  triplet=%.4fs  threshold=%.4fs  "
        "bars_per_segment=%d",
        bpm, interval_16th, interval_triplet, snap_threshold, bars_per_segment,
    )

    # Parse time signature to get beats per measure
    parts = time_sig.split("/")
    beats_per_measure = int(parts[0])
    beat_duration = 60.0 / bpm

    # Guard: bars_per_segment must be >= 2 (otherwise stride ≤ 0)
    if bars_per_segment <= 1:
        bars_per_segment = max(bars_per_segment, 2)
        logger.warning("bars_per_segment too small, clamped to %d", bars_per_segment)

    # Segment geometry
    segment_len_s = bars_per_segment * beats_per_measure * beat_duration
    segment_stride_s = (bars_per_segment - 1) * beats_per_measure * beat_duration
    overlap_s = beats_per_measure * beat_duration  # 1 bar

    # --- 1. Group notes by onset (chords share the same snap) ----------------
    onset_groups: dict[float, list[dict]] = defaultdict(list)
    for n in notes:
        onset_groups[n["onset"]].append(n)

    unique_onsets = np.array(sorted(onset_groups.keys()), dtype=float)
    if len(unique_onsets) == 0:
        return notes

    global_t0 = unique_onsets[0]
    max_time = unique_onsets[-1]

    # --- 2. Build overlapping segments, find per-segment offsets ------------
    # Each segment: (seg_start, seg_end, seg_t0, seg_offset)
    segments: list[tuple[float, float, float, float]] = []

    seg_idx = 0
    while True:
        seg_start = global_t0 + seg_idx * segment_stride_s
        seg_end = seg_start + segment_len_s

        mask = (unique_onsets >= seg_start) & (unique_onsets < seg_end)
        if np.any(mask):
            seg_onsets = unique_onsets[mask]
            seg_t0 = seg_onsets[0]
            seg_offset = _find_best_grid_offset(seg_onsets, interval_16th)
            segments.append((seg_start, seg_end, seg_t0, seg_offset))
            logger.debug(
                "Segment %d: [%.2f, %.2f)  t0=%.4f  offset=%.4f  notes=%d",
                seg_idx, seg_start, seg_end, seg_t0, seg_offset, len(seg_onsets),
            )

        seg_idx += 1
        if seg_start > max_time:
            break

    if not segments:
        return notes

    # Log segment offsets for diagnosis
    offsets_str = ", ".join(
        f"seg{i}: {o:.4f}"
        for i, (_, _, _, o) in enumerate(segments)
    )
    logger.info("Segment offsets: %s", offsets_str)

    # --- 3. Compute snapped position for each unique onset ------------------
    # For notes in the 1-bar overlap, average the snapped positions from
    # both containing segments (smooth boundary).
    snapped_onset_map: dict[float, float] = {}

    for onset_val in unique_onsets:
        positions: list[float] = []

        for seg_start, seg_end, seg_t0, seg_offset in segments:
            if seg_start <= onset_val < seg_end:
                snapped, _ = _nearest_grid(
                    onset_val, seg_offset, seg_t0,
                    interval_16th, interval_triplet, snap_threshold,
                )
                positions.append(snapped)

        if not positions:
            snapped_onset_map[onset_val] = onset_val
        elif len(positions) == 1:
            snapped_onset_map[onset_val] = positions[0]
        else:
            # Overlap region: average snapped positions from adjacent segments
            avg = sum(positions) / len(positions)
            snapped_onset_map[onset_val] = avg

    # --- 4. Apply snapped onsets to note groups -----------------------------
    snapped_count = 0
    kept_count = 0

    for orig_onset, group in onset_groups.items():
        snapped_onset = snapped_onset_map[orig_onset]
        delta = snapped_onset - orig_onset

        # Safety: ensure every note would still have positive duration
        # after snapping.  Check the actual duration that will result.
        any_bad = False
        for n in group:
            new_dur = (n["offset"] + delta) - snapped_onset
            if new_dur < 1e-4:  # less than 0.1 ms → skip snap
                any_bad = True
                break

        if any_bad:
            delta = 0.0
            snapped_onset = orig_onset

        if abs(delta) > 1e-6:
            snapped_count += 1
        else:
            kept_count += 1

        for n in group:
            n["onset"] = round(snapped_onset, 4)
            n["offset"] = round(n["offset"] + delta, 4)

    logger.info(
        "Quantize timing: %d groups snapped, %d kept (rubato/ornament)",
        snapped_count, kept_count,
    )

    # Re-sort by onset after potential modifications
    notes.sort(key=lambda n: n["onset"])
    return notes
