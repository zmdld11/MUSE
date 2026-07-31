"""Layer 3: Frame-level post-processing.

Pipeline:
  1. HMM forward-backward smoothing (per pitch)
  2. Adaptive thresholding (by register)
  3. Binarization
  4. Connected-component labelling → candidate notes
  5. Min/max note-length filter
  6. Onset verification
"""
import logging
import numpy as np
from scipy import ndimage

from src.config import config

logger = logging.getLogger(__name__)

# MIDI pitch ranges for adaptive thresholding
LOW_REGISTER = (21, 50)     # C2 – D4
MID_REGISTER = (50, 72)     # D4 – C6
HIGH_REGISTER = (72, 108)   # C6 – C9


def _hmm_smooth(probs: np.ndarray, p_stay_on: float = 0.88, p_turn_on: float = 0.15) -> np.ndarray:
    """
    Forward-backward HMM smoothing per pitch channel.

    States: 0=OFF, 1=ON
    Transition matrix:  [[p(off→off), p(off→on)],
                         [p(on→off),  p(on→on) ]]
    Emission: frame_probs as likelihood for ON state;
              (1 - frame_probs) as likelihood for OFF state.
    """
    T, P = probs.shape
    p_stay_off = 1 - p_turn_on
    p_turn_off = 1 - p_stay_on

    trans = np.array([[p_stay_off, p_turn_on],
                      [p_turn_off,  p_stay_on]])  # (2, 2)

    # Work in log space for numerical stability
    log_trans = np.log(trans + 1e-12)

    smoothed = np.zeros_like(probs)

    for p in range(P):
        obs = probs[:, p]  # (T,)

        # Emission log-likelihood: P(obs | state)
        # state=OFF (0): likelihood of observing this prob when note is off
        # state=ON  (1): likelihood of observing this prob when note is on
        log_emit_off = np.log(np.clip(1.0 - obs, 1e-12, 1.0))
        log_emit_on = np.log(np.clip(obs, 1e-12, 1.0))
        log_emission = np.column_stack([log_emit_off, log_emit_on])  # (T, 2)

        # Forward pass
        log_alpha = np.zeros((T, 2))
        log_alpha[0] = np.log(0.5) + log_emission[0]  # uniform prior

        for t in range(1, T):
            for s in range(2):
                log_alpha[t, s] = log_emission[t, s] + np.logaddexp(
                    log_alpha[t - 1, 0] + log_trans[0, s],
                    log_alpha[t - 1, 1] + log_trans[1, s],
                )

        # Backward pass
        log_beta = np.zeros((T, 2))
        log_beta[-1] = 0.0  # log(1)

        for t in range(T - 2, -1, -1):
            for s in range(2):
                log_beta[t, s] = np.logaddexp(
                    log_beta[t + 1, 0] + log_trans[s, 0] + log_emission[t + 1, 0],
                    log_beta[t + 1, 1] + log_trans[s, 1] + log_emission[t + 1, 1],
                )

        # Posterior probability of ON state
        log_gamma_on = log_alpha[:, 1] + log_beta[:, 1]
        log_gamma_off = log_alpha[:, 0] + log_beta[:, 0]

        # Normalize: P(ON | obs)
        log_sum = np.logaddexp(log_gamma_on, log_gamma_off)
        gamma_on = np.exp(log_gamma_on - log_sum)
        smoothed[:, p] = gamma_on

    return smoothed


def _adaptive_threshold_per_register(frame_probs: np.ndarray,
                                     percentile: float = 50.0,
                                     fixed_threshold: float = 0.3) -> np.ndarray:
    """
    Binarize frame_probs.

    VER2.1 (2026-07-31): 默认改为固定阈值 0.3.
    诊断发现: 原 50th-percentile 阈值在 HMM 平滑后概率双极化(0/1)时被顶到 ~1.0,
    把所有低置信度帧(含真实短音符)全杀, note recall 仅 0.15.
    固定阈值 0.3 在评测集上 note F1 0.42→0.64. 保留 percentile 模式供参考.
    """
    T, P = frame_probs.shape
    binary = np.zeros_like(frame_probs, dtype=bool)

    registers = [
        (LOW_REGISTER[0] - 21, LOW_REGISTER[1] - 21, "low"),
        (MID_REGISTER[0] - 21, MID_REGISTER[1] - 21, "mid"),
        (HIGH_REGISTER[0] - 21, HIGH_REGISTER[1] - 21, "high"),
    ]

    for start_bin, end_bin, name in registers:
        start_bin = max(0, start_bin)
        end_bin = min(P, end_bin)
        if end_bin <= start_bin:
            continue

        region = frame_probs[:, start_bin:end_bin]
        if fixed_threshold is not None:
            thresh = fixed_threshold
        else:
            thresh = np.percentile(region[region > 0.01], percentile) if region[region > 0.01].size > 0 else 0.3
        binary[:, start_bin:end_bin] = region >= thresh
        logger.debug(f"  {name} register (bin {start_bin}-{end_bin}): threshold = {thresh:.3f}")

    return binary


def _label_connected_components(binary: np.ndarray) -> list[dict]:
    """
    3D connected-component labelling on (T, P) binary map.
    Treats time and pitch as spatial dimensions. 4-connectivity in time direction.
    Returns list of candidate note dicts.
    """
    # 2D connectivity: connect adjacent frames for the same pitch (vertical=time, horizontal=pitch)
    if not binary.any():
        return []
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)  # 4-connectivity
    labeled, num_features = ndimage.label(binary, structure=structure)

    candidates = []
    for label_id in range(1, num_features + 1):
        coords = np.argwhere(labeled == label_id)  # (N, 2) → (frame_idx, pitch_bin)
        if len(coords) == 0:
            continue

        onset_frame = int(coords[:, 0].min())
        offset_frame = int(coords[:, 0].max()) + 1  # exclusive
        pitch_bin = int(round(coords[:, 1].mean()))
        pitch_midi = pitch_bin + 21  # MIDI note number
        confidence = float(len(coords))  # proxy: area in frames

        candidates.append({
            "onset_frame": onset_frame,
            "offset_frame": offset_frame,
            "pitch": pitch_midi,
            "pitch_bin": pitch_bin,
            "confidence": confidence,
        })

    return candidates


def _max_polyphony_filter(candidates: list[dict], max_per_frame: int = 8) -> list[dict]:
    """同一帧最多保留 max_per_frame 个最强的候选音符.

    Groups candidates by onset_frame, sorts by confidence descending,
    and keeps only the top-N per frame.
    """
    from collections import defaultdict

    # Group by onset_frame
    by_frame = defaultdict(list)
    for c in candidates:
        by_frame[c["onset_frame"]].append(c)

    kept = []
    for frame_idx, group in sorted(by_frame.items()):
        # Sort by confidence descending
        group.sort(key=lambda x: x["confidence"], reverse=True)
        kept.extend(group[:max_per_frame])

    logger.info(f"  Max polyphony filter ({max_per_frame}/frame): "
                f"{len(candidates)} → {len(kept)} candidates")
    return kept


def _filter_by_length(candidates: list[dict], hop_length: int, sr: int) -> list[dict]:
    """Drop notes that are too short or too long."""
    frame_duration = hop_length / sr  # seconds per frame
    min_frames = 2    # ~46 ms (2×23.2ms) — 保住快速短音符 (见 2026-07-31 评测诊断)
    max_frames = 600  # ~7 seconds

    filtered = []
    for c in candidates:
        n_frames = c["offset_frame"] - c["onset_frame"]
        if n_frames < min_frames:
            continue
        if n_frames > max_frames:
            c["offset_frame"] = c["onset_frame"] + max_frames
        filtered.append(c)
    return filtered


def _verify_onsets(candidates: list[dict], onset_probs: np.ndarray,
                   window: int = 2, min_onset_prob: float = 0.4) -> list[dict]:
    """Reject candidates whose onset probability at onset frame is too low.
    High-register notes (MIDI > 72) get a relaxed threshold since their
    amplitude is naturally lower — easier to be mistaken for noise."""
    verified = []
    for c in candidates:
        onset_f = c["onset_frame"]
        p_bin = c["pitch_bin"]
        # Relax threshold for high notes (pitch_bin > 50 = MIDI > 71)
        threshold = min_onset_prob * 0.7 if p_bin > 50 else min_onset_prob
        if onset_f < onset_probs.shape[0] and p_bin < onset_probs.shape[1]:
            win_start = max(0, onset_f - window)
            win_end = min(onset_probs.shape[0], onset_f + window + 1)
            onset_support = onset_probs[win_start:win_end, p_bin].max()
            if onset_support >= threshold:
                verified.append(c)
            else:
                logger.debug(f"  dropped: pitch={c['pitch']}, onset_prob={onset_support:.3f}")
    return verified


def process_frames(onset_probs: np.ndarray, frame_probs: np.ndarray,
                   hop_length: int = None, sr: int = None) -> list[dict]:
    """
    Full frame-level post-processing pipeline.

    Args:
        onset_probs:  (T, 88)  onset probabilities
        frame_probs:  (T, 88)  frame (note presence) probabilities

    Returns:
        List of candidate note dicts with keys:
        onset_frame, offset_frame, pitch, pitch_bin, confidence
    """
    hop_length = hop_length or config.HOP_LENGTH
    sr = sr or config.SR

    logger.info(f"Frame post-processing: input shape {frame_probs.shape}")

    # Step 1: HMM smoothing
    smoothed = _hmm_smooth(frame_probs)
    logger.info(f"  HMM smoothed: mean={smoothed.mean():.3f}")

    # Step 2-3: Adaptive threshold + binarize
    binary = _adaptive_threshold_per_register(smoothed)
    active = binary.sum()
    logger.info(f"  Binary: {active} active bins ({active / binary.size * 100:.2f}%)")

    # Step 4: Connected components
    candidates = _label_connected_components(binary)
    logger.info(f"  Connected components: {len(candidates)} candidates")

    # Step 4b: Max polyphony filter (同一帧最多 max_per_frame 个音符)
    candidates = _max_polyphony_filter(candidates, max_per_frame=8)

    # Step 5: Length filter
    candidates = _filter_by_length(candidates, hop_length, sr)
    logger.info(f"  After length filter: {len(candidates)} candidates")

    # Step 6: Onset verification
    candidates = _verify_onsets(candidates, onset_probs)
    logger.info(f"  After onset verify: {len(candidates)} candidates")

    return candidates
