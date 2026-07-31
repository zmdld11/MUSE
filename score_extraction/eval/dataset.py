"""eval/dataset.py — GiantMIDI 采样 + 渲染 + 缓存 + GT 提取.

评测数据管线:
  data/midi/GiantMIDI-PIano/midis/*.mid
    → 随机采样 N 首
    → train/render_midi.py 渲染成合成音频 (sr=22050, hop=512)
    → GT 三件套: frame_labels (帧级) + intervals/pitches (音符级)
    → 缓存到 eval/cache/{md5}.npz (避免重复渲染)

GT 直接从 MIDI 提取 (理论完美): 每个音符 onset/offset/pitch 精确已知.
"""
import glob
import hashlib
import os

import numpy as np
import pretty_midi

SR = 22050
HOP = 512
MAX_DUR = 30.0

_MIDI_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "midi", "GiantMIDI-PIano", "midis"))
_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))


def list_midis() -> list[str]:
    """所有可用 MIDI 路径 (按文件名排序, 保证可复现)."""
    return sorted(glob.glob(os.path.join(_MIDI_DIR, "*.mid")))


def sample_midis(n: int = 30, seed: int = 42) -> list[str]:
    """随机采样 n 首 MIDI (固定 seed, 可复现)."""
    midis = list_midis()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(midis), size=min(n, len(midis)), replace=False)
    return [midis[i] for i in idx]


def _cache_path(mid_path: str) -> str:
    key = os.path.basename(mid_path)
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"{h}.npz")


def _extract_gt(mid_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 MIDI 提取 GT 音符列表 (只保留 MAX_DUR 内的音符).

    Returns
    -------
    frame_labels : (T, 88) float32  — 帧级 GT (与 render_midi 的 hop/sr 对齐)
    intervals    : (M, 2) float64  — 每个 GT 音符的 [onset, offset] (秒)
    pitches      : (M,)  int       — 每个 GT 音符的 MIDI 音高
    """
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from train.render_midi import render_midi

    r = render_midi(mid_path, sr=SR, hop_length=HOP, max_dur_sec=MAX_DUR)
    frame_labels = r["frame_labels"]

    pm = pretty_midi.PrettyMIDI(mid_path)
    intervals, pitches = [], []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= MAX_DUR:
                continue
            onset = n.start
            offset = min(n.end, MAX_DUR)
            if offset - onset < 1e-4:  # 零时长音符 (mir_eval 拒绝)
                continue
            intervals.append([onset, offset])
            pitches.append(n.pitch)

    return frame_labels, np.asarray(intervals), np.asarray(pitches, dtype=int)


def load(mid_path: str, force_render: bool = False) -> dict:
    """渲染 MIDI → 合成音频 + GT, 带磁盘缓存.

    Returns
    -------
    dict with keys: audio, sr, frame_labels, intervals, pitches
    """
    cp = _cache_path(mid_path)
    if not force_render and os.path.exists(cp):
        d = np.load(cp, allow_pickle=True)
        return {
            "audio": d["audio"],
            "sr": int(d["sr"]),
            "frame_labels": d["frame_labels"],
            "intervals": d["intervals"],
            "pitches": d["pitches"],
        }

    frame_labels, intervals, pitches = _extract_gt(mid_path)

    # 重新渲染拿音频 (frame_labels 里已渲染过一次, 但为了缓存完整性单独存音频)
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from train.render_midi import render_midi
    r = render_midi(mid_path, sr=SR, hop_length=HOP, max_dur_sec=MAX_DUR)
    audio = r["audio"]

    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez(
        cp,
        audio=audio.astype(np.float32),
        sr=SR,
        frame_labels=frame_labels,
        intervals=intervals,
        pitches=pitches,
    )
    return {
        "audio": audio,
        "sr": SR,
        "frame_labels": frame_labels,
        "intervals": intervals,
        "pitches": pitches,
    }


def prewarm(n: int = 30, seed: int = 42) -> list[str]:
    """预渲染 n 首到缓存, 返回路径列表 (评测前先跑, 避免评测时卡在渲染)."""
    midis = sample_midis(n=n, seed=seed)
    for m in midis:
        load(m)
    return midis
