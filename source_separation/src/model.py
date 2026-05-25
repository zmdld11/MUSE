# model.py — DemucsLM: 轻量时域 Demucs 音轨分离模型
# VER3.0: 1D Conv U-Net, GLU 门控, LSTM 瓶颈
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout * 2, 8, stride=4, padding=2)

    def forward(self, x):
        x = self.conv(x)
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class DecBlock(nn.Module):
    def __init__(self, cin, cout, skip_ch):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout * 2, 8, stride=4, padding=2)
        if skip_ch != cout * 2:
            self.skip_proj = nn.Conv1d(skip_ch, cout * 2, 1)
        else:
            self.skip_proj = None

    def forward(self, x, skip):
        x = self.conv(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        if self.skip_proj is not None:
            skip = self.skip_proj(skip)
            skip = F.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=False)
        x = x + skip
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class DemucsLM(nn.Module):
    """时域 Demucs 轻量版 — 1D Conv U-Net + LSTM 瓶颈"""
    def __init__(self, channels=(16, 32, 64, 128, 256)):
        super().__init__()
        self.encs = nn.ModuleList()
        cin = 1
        for cout in channels:
            self.encs.append(EncBlock(cin, cout))
            cin = cout

        self.lstm = nn.LSTM(channels[-1], channels[-1], bidirectional=True, batch_first=True)

        # Decoder: 从 LSTM 输出 channels[-1]*2 → 递减到 1
        lstm_out = channels[-1] * 2
        self.decs = nn.ModuleList()
        # 第一层从 LSTM 输出进入
        self.decs.append(DecBlock(lstm_out, channels[-2], channels[-1]))
        # 中间层
        for i in range(len(channels) - 2, 0, -1):
            self.decs.append(DecBlock(channels[i], channels[i - 1], channels[i]))
        # 最后一层输出到 1 通道
        self.decs.append(DecBlock(channels[0], 1, channels[0]))
        self.out = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        orig_len = x.shape[-1]
        skips = []
        for enc in self.encs:
            x = enc(x)
            skips.append(x)
        skips = skips[::-1]

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)

        for dec, skip in zip(self.decs, skips):
            x = dec(x, skip)

        x = self.out(x)
        if x.shape[-1] != orig_len:
            x = F.interpolate(x, size=orig_len, mode='linear', align_corners=False)
        return x
