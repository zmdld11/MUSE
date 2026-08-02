"""eval/metrics.py — mir_eval 指标封装.

三个层级指标:
  frame_f1          : 帧级 MPE (mir_eval.multipitch)
  note_f1           : 音符级, onset ≤ ±50ms (mir_eval.transcription)
  note_offset_f1    : 音符级含 offset, onset ≤ 50ms & offset ≤ max(50ms, 20%×符长)

宽容度均按论文标准 (Onsets and Frames / NoteEM 一致).
"""
import logging

import mir_eval
import numpy as np

logger = logging.getLogger(__name__)

ONSET_TOLERANCE = 0.05        # ±50ms
OFFSET_RATIO = 0.2            # 20% × reference length
OFFSET_MIN_TOLERANCE = 0.05   # 至少 ±50ms
MIDI_OFFSET = 21              # 88 音高矩阵中 MIDI 21 = 第一个 bin


# ---------------------------------------------------------------------------
# 帧级指标
# ---------------------------------------------------------------------------

def _binarize_frame_probs(frame_probs: np.ndarray,
                          binarize_threshold: float = None,
                          threshold_cap: float = None,
                          binarize_mode: str = "adaptive",
                          onset_probs: np.ndarray = None) -> np.ndarray:
    """帧级二值化 (VER2.3: 与音符级后处理一致).

    默认 (adaptive): 用 BP 后处理 (process_frames_bp) 生成音符候选,
    音符覆盖的帧标记为活跃 — 与音符级指标同一套逻辑.
    legacy: HMM 平滑 + 自适应阈值 (旧逻辑).
    """
    from src.frame_post import process_frames_bp, _hmm_smooth, _adaptive_threshold_per_register

    if binarize_mode != "legacy":
        # 用 BP 后处理 (逐音高 + melodia) 生成候选, 覆盖帧 = 活跃
        if onset_probs is None:
            onset_probs = np.zeros_like(frame_probs)
        cands = process_frames_bp(onset_probs, frame_probs)
        binary = np.zeros_like(frame_probs, dtype=bool)
        for c in cands:
            s, e, b = c["onset_frame"], c["offset_frame"], c["pitch_bin"]
            if 0 <= b < binary.shape[1]:
                binary[s:min(e, binary.shape[0]), b] = True
        return binary

    # legacy: HMM + 自适应阈值
    smoothed = _hmm_smooth(frame_probs)
    if binarize_threshold is not None:
        return _adaptive_threshold_per_register(
            smoothed, fixed_threshold=float(binarize_threshold))
    return _adaptive_threshold_per_register(smoothed, max_threshold=threshold_cap)


def _frame_probs_to_times_freqs(frame_probs: np.ndarray,
                                hop_length: int, sr: int,
                                binarize_threshold: float = None,
                                threshold_cap: float = None,
                                binarize_mode: str = "adaptive",
                                onset_probs: np.ndarray = None) -> tuple:
    """把 (T, P) 概率矩阵转成 mir_eval.multipitch 需要的 (times, freqs).

    帧级"活跃"定义 = 后处理管线的二值化输出 (默认 BP 后处理).
    times : (T,) 每帧时间戳
    freqs : (T, P) 每帧各音高的频率 (Hz), 无音高处为 NaN
    """
    T, P = frame_probs.shape
    times = np.arange(T) * hop_length / sr

    active = _binarize_frame_probs(
        frame_probs, binarize_threshold, threshold_cap, binarize_mode,
        onset_probs=onset_probs).astype(bool)
    freqs = np.full((T, P), np.nan)
    for t in range(T):
        bins = np.nonzero(active[t])[0]
        if len(bins) > 0:
            midi = bins + MIDI_OFFSET
            freqs[t, :len(bins)] = 440.0 * 2 ** ((midi - 69) / 12.0)

    return times, freqs


def _frame_labels_to_times_freqs(frame_labels: np.ndarray,
                                 hop_length: int, sr: int) -> tuple:
    """把 GT 标签矩阵 (T, P) 转成 mir_eval.multipitch 格式."""
    T, P = frame_labels.shape
    times = np.arange(T) * hop_length / sr

    freqs = np.full((T, P), np.nan)
    for t in range(T):
        bins = np.nonzero(frame_labels[t])[0]
        if len(bins) > 0:
            midi = bins + MIDI_OFFSET
            freqs[t, :len(bins)] = 440.0 * 2 ** ((midi - 69) / 12.0)

    return times, freqs


def _resample_probs_to_grid(est_frame_probs: np.ndarray,
                            est_hop: int, n_ref_frames: int) -> np.ndarray:
    """把 est 的帧概率图重采样到 ref 的帧网格 (按时间线性插值).

    est_frame_probs : (T_est, P)
    est_hop : est 的 hop_length
    n_ref_frames : 目标网格帧数 (= GT 的帧数, 保证时间轴完全对齐)
    Returns : (n_ref_frames, P)
    """
    from scipy import interpolate

    T_est, P = est_frame_probs.shape
    est_times = np.arange(T_est) * est_hop
    # ref 时间轴: 与 GT 覆盖相同的时长 (est_hop*T_est / n_ref_frames = 帧间隔)
    frame_sec = est_hop * T_est / n_ref_frames
    ref_times = np.arange(n_ref_frames) * frame_sec

    out = np.zeros((n_ref_frames, P), dtype=np.float32)
    for p in range(P):
        f = interpolate.interp1d(est_times, est_frame_probs[:, p],
                                 kind="linear", bounds_error=False, fill_value=0.0)
        out[:, p] = f(ref_times)
    return out


def frame_f1(gt_frame_labels: np.ndarray, est_frame_probs: np.ndarray,
             hop_length: int = 512, sr: int = 22050,
             binarize_threshold: float = None,
             threshold_cap: float = None,
             binarize_mode: str = "adaptive",
             onset_probs: np.ndarray = None) -> tuple:
    """帧级 F1 (要求两者已在同一 hop 网格上). GT 二值标签, 预测概率矩阵."""
    ref_times, ref_freqs = _frame_labels_to_times_freqs(gt_frame_labels, hop_length, sr)
    est_times, est_freqs = _frame_probs_to_times_freqs(
        est_frame_probs, hop_length, sr,
        binarize_threshold, threshold_cap, binarize_mode,
        onset_probs=onset_probs)
    try:
        res = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs)
        p, r = float(res["Precision"]), float(res["Recall"])
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return p, r, f
    except Exception as exc:
        logger.warning(f"frame_f1 failed: {exc}")
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# 音符级指标
# ---------------------------------------------------------------------------

def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """MIDI 音高号 → 赫兹 (mir_eval.transcription 的 pitch 单位是 Hz!)."""
    return 440.0 * 2 ** ((np.asarray(midi, dtype=np.float64) - 69) / 12.0)


def _notes_to_mir_eval(notes: list[dict]) -> tuple:
    """把 note dict 列表 (onset/offset/pitch MIDI号) 转成 mir_eval 格式 (pitch→Hz)."""
    notes = [n for n in notes
             if n["pitch"] > 0 and n["offset"] - n["onset"] >= 1e-4]
    if not notes:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.float64)
    intervals = np.array([[n["onset"], n["offset"]] for n in notes])
    pitches_hz = _midi_to_hz(np.array([int(n["pitch"]) for n in notes]))
    return intervals, pitches_hz


def note_f1(gt_intervals: np.ndarray, gt_pitches: np.ndarray,
            est_notes: list[dict]) -> tuple:
    """音符级 F1 — onset-only (offset 忽略). GT pitches 为 MIDI 号, 转 Hz."""
    gt_pitches_hz = _midi_to_hz(gt_pitches)
    est_intervals, est_pitches = _notes_to_mir_eval(est_notes)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_intervals, gt_pitches_hz, est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=None,
    )
    return float(p), float(r), float(f)


def note_offset_f1(gt_intervals: np.ndarray, gt_pitches: np.ndarray,
                   est_notes: list[dict]) -> tuple:
    """音符级 F1 — 含 offset (onset ≤ 50ms & offset ≤ max(50ms, 20%×符长))."""
    gt_pitches_hz = _midi_to_hz(gt_pitches)
    est_intervals, est_pitches = _notes_to_mir_eval(est_notes)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_intervals, gt_pitches_hz, est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=OFFSET_RATIO,
        offset_min_tolerance=OFFSET_MIN_TOLERANCE,
    )
    return float(p), float(r), float(f)


def evaluate(gt: dict, est_frame_probs: np.ndarray, est_notes: list[dict],
             hop_length: int = 512, sr: int = 22050,
             binarize_threshold: float = None,
             threshold_cap: float = None,
             binarize_mode: str = "adaptive",
             est_onset_probs: np.ndarray = None) -> dict:
    """对一个样本计算全部三级指标.

    Parameters
    ----------
    gt : dict with keys frame_labels, intervals, pitches (from dataset.load)
         frame_labels 在 GT 的 hop 网格上 (dataset.py: hop=512)
    est_frame_probs : (T_est, P) 模型输出的帧概率图 (hop = hop_length)
    est_notes : list[dict]  后处理后的音符 (onset/offset/pitch)
    hop_length : est 的 hop_length (与 gt 不同会被重采样到 GT 网格)
    est_onset_probs : (T_est, P) onset 概率 (BP 后处理找 onset 峰值用)

    Returns
    -------
    dict: {frame_f1, note_f1, note_offset_f1, ...}
    """
    # 帧级: 把 est 重采样到 GT 网格 (mir_eval.multipitch 要求同网格)
    n_ref_frames = gt["frame_labels"].shape[0]
    est_on_gt_grid = _resample_probs_to_grid(
        est_frame_probs, est_hop=hop_length, n_ref_frames=n_ref_frames)
    onset_on_gt_grid = None
    if est_onset_probs is not None:
        onset_on_gt_grid = _resample_probs_to_grid(
            est_onset_probs, est_hop=hop_length, n_ref_frames=n_ref_frames)
    fp, fr, ff = frame_f1(gt["frame_labels"], est_on_gt_grid,
                          hop_length=512, sr=sr,
                          binarize_threshold=binarize_threshold,
                          threshold_cap=threshold_cap,
                          binarize_mode=binarize_mode,
                          onset_probs=onset_on_gt_grid)
    np_, nr, nf = note_f1(gt["intervals"], gt["pitches"], est_notes)
    op, or_, of = note_offset_f1(gt["intervals"], gt["pitches"], est_notes)
    return {
        "frame_precision": fp, "frame_recall": fr, "frame_f1": ff,
        "note_precision": np_, "note_recall": nr, "note_f1": nf,
        "offset_precision": op, "offset_recall": or_, "offset_f1": of,
    }
