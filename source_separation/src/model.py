# model.py — DemucsLM: 轻量时域 Demucs 音轨分离模型
# VER5.0: 拼接式跳跃连接 + 4层编码器-解码器 + LSTM 瓶颈
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncBlock(nn.Module):
    """编码器: Conv1d → GLU 门控"""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout * 2, 8, stride=4, padding=2)

    def forward(self, x):
        x = self.conv(x)
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class DecBlock(nn.Module):
    """解码器: ConvTranspose1d → concat(skip) → Conv1x1 融合 → GLU"""
    def __init__(self, cin, cout, skip_ch):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout * 2, 8, stride=4, padding=2)
        # 1x1 卷积学习如何融合上采样特征 + 跳跃连接
        self.fuse = nn.Conv1d(cout * 2 + skip_ch, cout * 2, 1)

    def forward(self, x, skip):
        x = self.conv(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        # 拼接保留全部信息，由 1x1 卷积学习融合权重
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
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

        lstm_out = channels[-1] * 2
        self.decs = nn.ModuleList()
        self.decs.append(DecBlock(lstm_out, channels[-2], channels[-1]))
        for i in range(len(channels) - 2, 0, -1):
            self.decs.append(DecBlock(channels[i], channels[i - 1], channels[i]))
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
