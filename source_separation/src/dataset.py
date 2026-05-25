# dataset.py — 吉他分离数据集加载
# 从 data/audio/ 加载 WAV pair，返回原始波形
import os, json, random
import torch
from torch.utils.data import Dataset
import soundfile as sf

SR = 22050


class GuitarSeparationDataset(Dataset):
    def __init__(self, data_dir, split="train", augment=False):
        meta_path = os.path.join(data_dir, "metadata.json")
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        audio_dir = os.path.join(data_dir, "audio")
        n_train = meta["num_train"]
        n_val = meta["num_val"]
        total = n_train + n_val

        if split == "train":
            self.indices = list(range(n_train))
        else:
            self.indices = list(range(n_train, total))

        self.audio_dir = audio_dir
        self.augment = augment and split == "train"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        mix_path = os.path.join(self.audio_dir, f"{i:06d}_mix.wav")
        gtr_path = os.path.join(self.audio_dir, f"{i:06d}_gtr.wav")

        mix, _ = sf.read(mix_path)
        gtr, _ = sf.read(gtr_path)

        mix = torch.from_numpy(mix.astype("float32"))
        gtr = torch.from_numpy(gtr.astype("float32"))

        if self.augment:
            gain = 10 ** (random.uniform(-3, 3) / 20)
            mix *= gain; gtr *= gain

        return mix, gtr
