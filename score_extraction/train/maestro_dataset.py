"""MAESTRO 真实录音数据集 (2026-08-04).

MAESTRO v3.0.0: wav 真实钢琴录音 + 对齐 MIDI 标注.
与 PianoSynthDataset (GiantMIDI 合成渲染) 输出格式完全一致:
__getitem__ 返回 (mel, onset, frame) = ((1,229,T), (T,88), (T,88)).

预处理: 整曲 midi → 帧/onset 标签, wav → 60s 段 mel (采样率统一 22050).
每首曲目打包一个 npz 缓存 (含全部段), 首次预处理后直接读缓存.

混合训练: MixedSynthMaestro 偶数索引取合成域, 奇数索引取真实域 (1:1).
"""
import csv
import hashlib
import logging
import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from train.config import train_config as cfg

logger = logging.getLogger(__name__)

DEFAULT_MAESTRO_DIR = os.path.join(cfg.WORKSPACE_DIR, "data", "maestro",
                                   "maestro-v3.0.0")
DEFAULT_CSV = os.path.join(cfg.WORKSPACE_DIR, "data", "maestro",
                           "maestro-v3.0.0.csv")
DEFAULT_CACHE_DIR = os.path.join(cfg.WORKSPACE_DIR, "data", "cache", "maestro")

SR = 22050
HOP = 512
N_MELS = 229
MIDI_OFFSET = 21
MIN_SEG_RATIO = 0.5  # 尾段不足 0.5×max_dur 丢弃


def load_maestro_rows(csv_path=DEFAULT_CSV):
    """读 MAESTRO csv → list[dict] (midi_filename/audio_filename/split/...)."""
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def midi_to_labels(midi_path, sr=SR, hop=HOP):
    """MIDI → (frame_labels, onset_labels) 整曲, (T, 88) float32.

    帧定义: 音符 [s, e] → frame[s_f : e_f+1] = 1; onset[s_f] = 1.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    T = int(np.ceil(pm.get_end_time() * sr / hop))
    frame_labels = np.zeros((T, 88), dtype=np.float32)
    onset_labels = np.zeros((T, 88), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.pitch < MIDI_OFFSET or n.pitch >= MIDI_OFFSET + 88:
                continue
            s_f = int(n.start * sr / hop)
            e_f = int(np.ceil(n.end * sr / hop))
            frame_labels[max(0, s_f):min(T, e_f + 1), n.pitch - MIDI_OFFSET] = 1.0
            if s_f < T:
                onset_labels[s_f, n.pitch - MIDI_OFFSET] = 1.0
    return frame_labels, onset_labels


def _cache_path(midi_path, wav_path, max_dur_sec, cache_dir):
    raw = f"{midi_path}|{wav_path}|{max_dur_sec}".encode("utf-8")
    return os.path.join(cache_dir, hashlib.sha256(raw).hexdigest()[:16] + ".npz")


class MaestroDataset(Dataset):
    """MAESTRO 真实录音数据集 (60s 段缓存)."""

    def __init__(self, split="train", max_files=None, max_dur_sec=60,
                 maestro_dir=DEFAULT_MAESTRO_DIR, csv_path=DEFAULT_CSV,
                 cache_dir=DEFAULT_CACHE_DIR, force_recache=False):
        self.max_dur_sec = max_dur_sec
        self.force_recache = force_recache
        os.makedirs(cache_dir, exist_ok=True)

        rows = [r for r in load_maestro_rows(csv_path) if r["split"] == split]
        if max_files:
            rows = rows[:max_files]
        self.rows = rows
        logger.info(f"MaestroDataset({split}): {len(rows)} 曲目")

        # 预处理全部曲目 → 段索引
        self._segments = []
        for r in rows:
            midi_path = os.path.join(maestro_dir, r["midi_filename"])
            wav_path = os.path.join(maestro_dir, r["audio_filename"])
            cp = _cache_path(midi_path, wav_path, max_dur_sec, cache_dir)
            if not os.path.exists(cp) or force_recache:
                self._preprocess(midi_path, wav_path, cp)
            n_seg = self._count_segments(cp)
            for s in range(n_seg):
                self._segments.append((cp, s))
        logger.info(f"  总段数: {len(self._segments)}")

    def _count_segments(self, cp):
        with np.load(cp) as d:
            return int(d["mels"].shape[0])

    def _preprocess(self, midi_path, wav_path, cp):
        """整曲 → 60s 段 (mel, onset, frame) 打包 npz."""
        import tempfile
        frame_labels, onset_labels = midi_to_labels(midi_path)

        # 音频 → mel 段
        audio, sr = librosa.load(wav_path, sr=SR, mono=True)
        n_hop_per_seg = self.max_dur_sec * SR // HOP
        mels, onsets, frames = [], [], []
        for s in range(0, len(audio) // (self.max_dur_sec * SR)):
            seg = audio[s * self.max_dur_sec * SR:(s + 1) * self.max_dur_sec * SR]
            if len(seg) < MIN_SEG_RATIO * self.max_dur_sec * SR:
                break
            mel = librosa.feature.melspectrogram(
                y=seg, sr=SR, n_mels=N_MELS, hop_length=HOP, fmin=30, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = np.clip((mel_db + 80) / 80, -1, 1).astype(np.float32)

            # 标签段: 与 mel 帧数对齐 (取 min)
            f0 = s * n_hop_per_seg
            n_frames = min(mel_db.shape[1], frame_labels.shape[0] - f0)
            mels.append(mel_db[:, :n_frames])
            onsets.append(onset_labels[f0:f0 + n_frames])
            frames.append(frame_labels[f0:f0 + n_frames])

        if not mels:
            logger.warning(f"  空段: {os.path.basename(midi_path)}")
            mels = [np.zeros((N_MELS, 1), dtype=np.float32)]
            onsets = [np.zeros((1, 88), dtype=np.float32)]
            frames = [np.zeros((1, 88), dtype=np.float32)]

        os.makedirs(os.path.dirname(cp), exist_ok=True)
        tmp = cp + ".tmp.npz"
        np.savez_compressed(tmp, mels=np.stack(mels), onsets=np.stack(onsets),
                            frames=np.stack(frames))
        os.replace(tmp, cp)
        logger.info(f"  缓存: {os.path.basename(cp)} "
                    f"({len(mels)} 段, {os.path.getsize(cp)/1e6:.0f}MB)")

    def __len__(self):
        return len(self._segments)

    def __getitem__(self, idx):
        cp, seg = self._segments[idx]
        with np.load(cp) as d:
            mel = d["mels"][seg]
            onset = d["onsets"][seg]
            frame = d["frames"][seg]
        return (torch.from_numpy(mel).float().unsqueeze(0),
                torch.from_numpy(onset).float(),
                torch.from_numpy(frame).float())


class MixedSynthMaestro(Dataset):
    """合成(GiantMIDI)与真实(MAESTRO) 1:1 混合: 偶索引合成, 奇索引真实."""

    def __init__(self, synth_ds, maestro_ds):
        self.synth_ds = synth_ds
        self.maestro_ds = maestro_ds
        self._n_pairs = min(len(synth_ds), len(maestro_ds))
        logger.info(f"MixedSynthMaestro: synth={len(synth_ds)} "
                    f"maestro={len(maestro_ds)} → {self._n_pairs * 2} 样本")

    def __len__(self):
        return self._n_pairs * 2

    def __getitem__(self, idx):
        pair, side = divmod(idx, 2)
        if side == 0:
            return self.synth_ds[pair % len(self.synth_ds)]
        return self.maestro_ds[pair % len(self.maestro_ds)]
