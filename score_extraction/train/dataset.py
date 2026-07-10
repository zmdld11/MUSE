"""PyTorch Dataset for synthetic piano training data from MIDI files."""
import logging
import glob
import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from train.config import train_config as cfg
from train.render_midi import render_midi

logger = logging.getLogger(__name__)


class PianoSynthDataset(Dataset):
    """
    Dataset that renders MIDI → audio + labels on-the-fly.

    Each item: (mel_spec_tensor, onset_labels_tensor, frame_labels_tensor)
    """

    def __init__(self, midi_dir=None, max_files=None, max_dur_sec=None):
        midi_dir = midi_dir or cfg.MIDI_DIR
        self.max_dur_sec = max_dur_sec or cfg.MAX_DUR_SEC

        # Collect MIDI files (recursive)
        self.midi_paths = sorted(
            glob.glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True)
        )
        if max_files:
            import random
            random.seed(42)
            self.midi_paths = random.sample(self.midi_paths, min(max_files, len(self.midi_paths)))

        logger.info(f"PianoSynthDataset: {len(self.midi_paths)} MIDI files")

        # Pre-compute valid files (quick parse to filter broken ones)
        self.valid_paths = []
        for p in self.midi_paths:
            try:
                import pretty_midi
                pm = pretty_midi.PrettyMIDI(p)
                end = pm.get_end_time()
                if end > 5.0:  # at least 5 seconds
                    self.valid_paths.append(p)
            except Exception:
                pass
        logger.info(f"  Valid (>5s): {len(self.valid_paths)}")

    def __len__(self):
        return len(self.valid_paths)

    def __getitem__(self, idx):
        path = self.valid_paths[idx]
        try:
            data = render_midi(path, max_dur_sec=self.max_dur_sec)
        except Exception as e:
            logger.warning(f"Render failed for {path}: {e}, using next file")
            return self.__getitem__((idx + 1) % len(self))

        audio = data["audio"]
        onset = data["onset_labels"]
        frame = data["frame_labels"]

        # Compute Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=cfg.SR, n_mels=cfg.N_MELS,
            hop_length=cfg.HOP_LENGTH, fmin=30, fmax=8000,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T_mel)

        # Align spectrogram frames with label frames
        # Labels use HOP_LENGTH frames, mel uses the same hop_length
        min_frames = min(mel_db.shape[1], onset.shape[0], frame.shape[0])

        mel_db = mel_db[:, :min_frames]
        onset = onset[:min_frames, :]
        frame = frame[:min_frames, :]

        # Normalize mel to [-1, 1]
        mel_db = np.clip((mel_db + 80) / 80, -1, 1)  # rough normalization

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
