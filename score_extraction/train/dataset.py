"""PyTorch Dataset for synthetic piano training data from MIDI files.

Caches rendered mel-spectrograms + labels to disk so multi-epoch training
does not re-render MIDI files on every epoch.
"""
import glob
import hashlib
import logging
import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from train.config import train_config as cfg
from train.render_midi import render_midi

logger = logging.getLogger(__name__)

# Cache directory (under score_extraction/data)
CACHE_DIR = os.path.join(cfg.WORKSPACE_DIR, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(midi_path: str, max_dur_sec: float) -> str:
    """Deterministic cache key from MIDI path + duration."""
    raw = f"{midi_path}|{max_dur_sec}".encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.npz")


class PianoSynthDataset(Dataset):
    """
    Dataset that renders MIDI -> audio -> mel-spectrogram + labels.

    Rendered results are cached to ``data/cache/`` so they are only computed
    once across multiple epochs.
    """

    def __init__(self, midi_dir=None, max_files=None, max_dur_sec=None,
                 force_recache=False):
        midi_dir = midi_dir or cfg.MIDI_DIR
        self.max_dur_sec = max_dur_sec or cfg.MAX_DUR_SEC
        self.force_recache = force_recache

        # Collect MIDI files (recursive)
        self.midi_paths = sorted(
            glob.glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True)
        )
        if max_files:
            import random
            random.seed(42)
            self.midi_paths = random.sample(
                self.midi_paths, min(max_files, len(self.midi_paths))
            )

        logger.info("PianoSynthDataset: %d MIDI files", len(self.midi_paths))

        # Pre-compute valid files — cache-first to avoid slow pretty_midi parse
        try:
            cache_dir_entries = set(os.listdir(CACHE_DIR))
        except Exception:
            cache_dir_entries = set()
        self.valid_paths = []
        for p in self.midi_paths:
            cp = os.path.basename(_cache_path(p, self.max_dur_sec))
            if cp in cache_dir_entries:
                self.valid_paths.append(p)
            else:
                try:
                    import pretty_midi
                    pm = pretty_midi.PrettyMIDI(p)
                    if pm.get_end_time() > 5.0:
                        self.valid_paths.append(p)
                except Exception:
                    pass
        logger.info("  Valid (>5s): %d", len(self.valid_paths))

    def __len__(self):
        return len(self.valid_paths)

    def __getitem__(self, idx):
        path = self.valid_paths[idx]
        cpath = _cache_path(path, self.max_dur_sec)

        # Try loading from cache
        if not self.force_recache and os.path.isfile(cpath):
            try:
                npz = np.load(cpath, allow_pickle=False)
                mel_db = npz["mel"]
                onset = npz["onset"]
                frame = npz["frame"]
                return (
                    torch.from_numpy(mel_db).float().unsqueeze(0),
                    torch.from_numpy(onset).float(),
                    torch.from_numpy(frame).float(),
                )
            except Exception as exc:
                logger.debug("Cache read failed for %s: %s", path, exc)

        # Render MIDI -> audio + labels
        try:
            data = render_midi(path, max_dur_sec=self.max_dur_sec)
        except Exception as e:
            logger.warning("Render failed for %s: %s, using next file", path, e)
            return self.__getitem__((idx + 1) % len(self))

        audio = data["audio"]
        onset = data["onset_labels"]
        frame = data["frame_labels"]

        # Compute Mel spectrogram at the dataset's target hop
        mel = librosa.feature.melspectrogram(
            y=audio, sr=cfg.SR, n_mels=cfg.N_MELS,
            hop_length=cfg.HOP_LENGTH, fmin=30, fmax=8000,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T_mel)

        # Align frames
        min_frames = min(mel_db.shape[1], onset.shape[0], frame.shape[0])
        mel_db = mel_db[:, :min_frames]
        onset = onset[:min_frames, :]
        frame = frame[:min_frames, :]

        # Normalize mel to [-1, 1]
        mel_db = np.clip((mel_db + 80) / 80, -1, 1)

        # Save to cache
        try:
            np.savez_compressed(cpath, mel=mel_db, onset=onset, frame=frame)
        except Exception as exc:
            logger.debug("Cache write failed for %s: %s", path, exc)

        return (
            torch.from_numpy(mel_db).float().unsqueeze(0),  # (1, n_mels, T)
            torch.from_numpy(onset).float(),                 # (T, 88)
            torch.from_numpy(frame).float(),                 # (T, 88)
        )


def collate_variable_length(batch):
    """Pad variable-length sequences to max length in batch."""
    specs, onsets, frames = zip(*batch)
    max_len = max(s.shape[-1] for s in specs)

    padded_specs, padded_onsets, padded_frames = [], [], []
    for s, o, f in zip(specs, onsets, frames):
        pad_t = max_len - s.shape[-1]
        padded_specs.append(torch.nn.functional.pad(s, (0, pad_t)))
        padded_onsets.append(torch.nn.functional.pad(o, (0, 0, 0, pad_t)))
        padded_frames.append(torch.nn.functional.pad(f, (0, 0, 0, pad_t)))

    return torch.stack(padded_specs), torch.stack(padded_onsets), torch.stack(padded_frames)
