"""CRNN 四头人声识谱模型（M1 v1，2026-08-31）。

mel(B,1,128,T) → 4×[Conv3x3(时间 pad 保 100fps) + BN + ReLU + MaxPool(1,2 只池频率)]
→ (B, 384·8, T) → 2×BiGRU(256) → 四头：onset / offset / pitch(46 类含 unvoiced) /
vib_depth。时间全程 1:1（onset ±50ms 口径需要 10ms 分辨率）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_CLASSES = 46  # 0=unvoiced, 1..45 = MIDI 40..84


class VocalCRNN(nn.Module):
    def __init__(self, n_mels: int = 128, conv_ch=(48, 96, 192, 384),
                 gru_hidden: int = 256, gru_layers: int = 2,
                 n_classes: int = N_CLASSES, dropout: float = 0.2,
                 with_vowel: bool = False):
        super().__init__()
        convs, ch = [], 1
        for c in conv_ch:
            convs += [nn.Conv2d(ch, c, kernel_size=(3, 3), padding=(1, 1)),
                      nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=(2, 1))]   # 只池频率 128→8
            ch = c
        self.convs = nn.Sequential(*convs)
        self.feat_dim = ch * (n_mels // (2 ** len(conv_ch)))  # 384*8=3072
        self.gru = nn.GRU(self.feat_dim, gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        out = gru_hidden * 2
        self.head_onset = nn.Linear(out, 1)
        self.head_offset = nn.Linear(out, 1)
        self.head_pitch = nn.Linear(out, n_classes)
        self.head_vib = nn.Linear(out, 1)
        self.with_vowel = with_vowel              # M3：元音起始头
        if with_vowel:
            self.head_vowel = nn.Linear(out, 1)

    def forward(self, mel: torch.Tensor, lengths: torch.Tensor) -> dict:
        """mel (B,1,128,T) → dict of (B,T) / (B,T,46)。"""
        h = self.convs(mel)                       # (B, C, F/16, T)
        h = h.permute(0, 3, 1, 2).flatten(2)      # (B, T, C·F/16)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=mel.shape[-1])  # (B,T,512)
        d = {"onset": self.head_onset(out).squeeze(-1),
             "offset": self.head_offset(out).squeeze(-1),
             "pitch": self.head_pitch(out),
             "vib": self.head_vib(out).squeeze(-1)}
        if self.with_vowel:
            d["vowel"] = self.head_vowel(out).squeeze(-1)
        return d


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


if __name__ == "__main__":
    m = VocalCRNN()
    total, train = count_params(m)
    print(f"VocalCRNN params: total={total/1e6:.2f}M trainable={train/1e6:.2f}M")
    x = torch.randn(2, 1, 128, 500)
    y = m(x, torch.tensor([500, 300]))
    for k, v in y.items():
        print(k, tuple(v.shape))
