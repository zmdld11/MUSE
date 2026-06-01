# dataset.py — 吉他分离数据集加载
# 支持预计算数据 + 在线随机混音增强
import os, json, random
import torch
from torch.utils.data import Dataset
import soundfile as sf


class GuitarSeparationDataset(Dataset):
    def __init__(self, data_dir, split="train", augment=False,
                 remix_dataset=None, remix_prob=0.5, num_samples=None):
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
        self.remix_dataset = remix_dataset
        self.remix_prob = remix_prob
        self.num_samples = num_samples

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 随机混音增强 (50% 概率, 切断"输出混合"捷径)
        if self.remix_dataset is not None and random.random() < self.remix_prob:
            ri = random.randint(0, len(self.remix_dataset) - 1)
            return self.remix_dataset[ri]

        # 原始预计算数据
        i = self.indices[idx]
        mix_path = os.path.join(self.audio_dir, f"{i:06d}_mix.wav")
        gtr_path = os.path.join(self.audio_dir, f"{i:06d}_gtr.wav")

        mix, _ = sf.read(mix_path)
        gtr, _ = sf.read(gtr_path)

        mix = torch.from_numpy(mix.astype("float32"))
        gtr = torch.from_numpy(gtr.astype("float32"))

        # 裁剪/填充到模型输入长度 (None=保持原长, 用于验证)
        if self.num_samples is not None:
            if len(mix) >= self.num_samples:
                if self.augment:
                    start = random.randint(0, len(mix) - self.num_samples)
                else:
                    start = 0
                mix = mix[start:start + self.num_samples]
                gtr = gtr[start:start + self.num_samples]
            else:
                mix = torch.nn.functional.pad(mix, (0, self.num_samples - len(mix)))
                gtr = torch.nn.functional.pad(gtr, (0, self.num_samples - len(gtr)))

        if self.augment:
            gain = 10 ** (random.uniform(-3, 3) / 20)
            mix *= gain
            gtr *= gain

        return mix, gtr
