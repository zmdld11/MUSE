# demucs_dataset.py — Demucs 格式数据集
# 从 demucs_format/ 文件夹加载, 现场求和, 增强后返回 (mix, guitar)
import json, math, random, os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class DemucsGuitarDataset(Dataset):
    """每首歌一个文件夹, 含 guitar.wav + other.wav + metadata.json.
    __getitem__ 随机裁剪 segment, 现场求和 mix, 返回 (mix, guitar_target).
    """

    def __init__(self, root, segment=11.0, shift=1.0, sr=22050,
                 augment=True, remix_prob=0.5):
        self.root = root
        self.segment = segment
        self.shift = shift
        self.sr = sr
        self.segment_samples = int(segment * sr)
        self.augment = augment
        self.remix_prob = remix_prob if augment else 0

        # 扫描歌曲
        self.songs = []
        for name in sorted(os.listdir(root)):
            sdir = os.path.join(root, name)
            if not os.path.isdir(sdir):
                continue
            mp = os.path.join(sdir, "metadata.json")
            if not os.path.isfile(mp):
                continue
            with open(mp) as f:
                meta = json.load(f)
            n_frames = meta["length"]
            dur = n_frames / meta["samplerate"]
            if dur < segment:
                continue
            n_examples = max(1, int(math.ceil((dur - segment) / shift) + 1))
            self.songs.append(dict(
                name=name, dir=sdir, meta=meta,
                n_examples=n_examples, dur=dur,
            ))

        self._len = sum(s["n_examples"] for s in self.songs)
        self._offsets = []
        off = 0
        for s in self.songs:
            self._offsets.append(off)
            off += s["n_examples"]
        print(f"DemucsGuitarDataset({root}): {len(self.songs)} songs, {self._len} segments")

    def __len__(self):
        return self._len

    def _find_song(self, idx):
        for i, off in enumerate(self._offsets):
            if i + 1 < len(self._offsets) and idx >= self._offsets[i + 1]:
                continue
            return self.songs[i], idx - off
        return self.songs[-1], idx - self._offsets[-1]

    def __getitem__(self, idx):
        song, sub_idx = self._find_song(idx)

        # 随机裁剪位置
        if self.augment:
            max_start = song["meta"]["length"] - self.segment_samples
            start = random.randint(0, max_start)
        else:
            start = int(sub_idx * self.shift * self.sr)
            start = min(start, song["meta"]["length"] - self.segment_samples)

        # 随机混音: 从另一首歌取 other 声源
        if random.random() < self.remix_prob:
            other_song = random.choice(self.songs)
            other_start = random.randint(0, max(0, other_song["meta"]["length"] - self.segment_samples))
            other_audio = self._load_segment(other_song["dir"], "other", other_start)
        else:
            other_audio = self._load_segment(song["dir"], "other", start)

        gtr = self._load_segment(song["dir"], "guitar", start)

        # 增强
        if self.augment:
            gtr, other_audio = self._augment(gtr, other_audio)

        mix = gtr + other_audio

        # 峰值防削波
        peak = float(np.abs(mix).max())
        if peak > 0.95:
            mix /= peak / 0.95
            gtr /= peak / 0.95

        return torch.from_numpy(mix.copy()), torch.from_numpy(gtr.copy())

    def _load_segment(self, sdir, stem_name, start):
        fp = os.path.join(sdir, f"{stem_name}.wav")
        n = self.segment_samples
        audio, _ = sf.read(fp, start=start, stop=start + n, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        if len(audio) < n:
            audio = np.pad(audio, (0, n - len(audio)))
        return audio

    def _augment(self, gtr, other):
        # 增益缩放
        g_gain = 10 ** (random.uniform(-3, 3) / 20)
        o_gain = 10 ** (random.uniform(-6, 0) / 20)
        gtr = gtr * g_gain
        other = other * o_gain
        return gtr, other
