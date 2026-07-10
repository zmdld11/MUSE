"""DTW alignment between predicted piano roll and MIDI piano roll.

Used in NoteEM's E-step: align the model's transcription of real audio
with the unaligned MIDI score, producing time-warped pseudo-labels.
"""
import logging
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def _piano_roll_to_features(roll, downsample=1):
    """
    Convert piano roll (T, 88) to a feature sequence for DTW.
    Downsampling reduces computation.
    """
    if downsample > 1:
        T = roll.shape[0]
        new_T = T // downsample
        roll = roll[:new_T * downsample].reshape(new_T, downsample, -1).max(axis=1)
    # Binarize
    roll = (roll > 0.3).astype(np.float32)
    return roll


def dtw_align(pred_roll, midi_roll, radius=50):
    """
    Align two piano rolls using DTW with Sakoe-Chiba band constraint.

    Args:
        pred_roll:  (T_pred, 88)  model's frame-level predictions (probabilities)
        midi_roll:  (T_midi, 88)  MIDI-derived frame labels (binary)
        radius:     Sakoe-Chiba band radius (frames). Larger = slower but more flexible.

    Returns:
        warped_midi: (T_pred, 88)  midi_roll warped to match pred_roll's timeline
        path:        list of (i, j) alignment pairs
    """
    pred_feat = _piano_roll_to_features(pred_roll, downsample=2)
    midi_feat = _piano_roll_to_features(midi_roll, downsample=2)

    Tp, Tm = pred_feat.shape[0], midi_feat.shape[0]

    # Cost matrix: Euclidean distance between feature vectors
    cost = cdist(pred_feat, midi_feat, metric="euclidean")  # (Tp, Tm)

    # DTW with band constraint
    path = _dtw_with_band(cost, radius)
    logger.info(f"DTW: pred_len={Tp}, midi_len={Tm}, path_len={len(path)}")

    # Warp MIDI roll to match pred timeline
    warped = np.zeros_like(pred_roll)
    alignment = np.zeros(len(pred_roll), dtype=int)

    # Build mapping: for each pred frame i, find matching MIDI frame j
    for i, j in path:
        i_orig = min(i * 2, len(pred_roll) - 1)
        j_orig = min(j * 2, len(midi_roll) - 1)
        alignment[i_orig] = j_orig

    # Fill gaps by forward-fill
    for i in range(1, len(alignment)):
        if alignment[i] == 0:
            alignment[i] = alignment[i - 1]

    # Map MIDI labels through alignment
    for i in range(len(pred_roll)):
        j = min(alignment[i], len(midi_roll) - 1)
        warped[i] = midi_roll[j]

    return warped, path


def _dtw_with_band(cost, radius):
    """
    DTW with Sakoe-Chiba band constraint. Returns optimal path.

    cost: (Tp, Tm) distance matrix
    radius: max allowed |i - j| deviation
    """
    Tp, Tm = cost.shape
    radius = max(radius, abs(Tp - Tm) + 10)

    # DP matrix initialized to infinity
    D = np.full((Tp, Tm), np.inf)
    D[0, 0] = cost[0, 0]

    # Backtrace
    back = np.zeros((Tp, Tm, 2), dtype=int)

    for i in range(Tp):
        j_start = max(0, i - radius)
        j_end = min(Tm, i + radius + 1)
        for j in range(j_start, j_end):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0:
                candidates.append((D[i - 1, j], i - 1, j))
            if j > 0:
                candidates.append((D[i, j - 1], i, j - 1))
            if i > 0 and j > 0:
                candidates.append((D[i - 1, j - 1], i - 1, j - 1))

            if candidates:
                min_val, pi, pj = min(candidates, key=lambda x: x[0])
                D[i, j] = min_val + cost[i, j]
                back[i, j] = [pi, pj]

    # Backtrack
    path = []
    i, j = Tp - 1, Tm - 1
    while i > 0 or j > 0:
        path.append((i, j))
        i, j = back[i, j]
    path.append((0, 0))
    path.reverse()

    return path


def align_and_label(pred_roll, midi_path, hop_length, sr):
    """
    Full E-step: predict + DTW align → pseudo labels.

    Args:
        pred_roll: model's frame predictions for real audio (T, 88)
        midi_path: path to unaligned MIDI score
        hop_length, sr: audio parameters

    Returns:
        pseudo_labels: (T, 88) time-aligned labels
    """
    import pretty_midi
    from train.render_midi import _note_to_midi_bin

    pm = pretty_midi.PrettyMIDI(midi_path)
    midi_dur = pm.get_end_time()
    n_midi_frames = int(midi_dur * sr / hop_length) + 1
    midi_roll = np.zeros((n_midi_frames, 88), dtype=np.float32)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pb = _note_to_midi_bin(n.pitch)
            if 0 <= pb < 88:
                onset_f = int(n.start * sr / hop_length)
                offset_f = int(n.end * sr / hop_length)
                midi_roll[onset_f:offset_f, pb] = 1.0

    pseudo_labels, path = dtw_align(pred_roll, midi_roll)
    return {"labels": pseudo_labels, "path": path}
