# model.py — DemucsLM: 轻量时域 Demucs 音轨分离模型
# VER6.0: Phase 1 — 输入归一化 + GroupNorm + 权重 rescale
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rescale_conv(conv, reference=0.1):
    """缩放到 reference std, 防止深层网络初始化梯度爆炸 (来自 Demucs)"""
    if conv.weight.numel() <= 1:  # 单元素卷积无法计算 std, 跳过
        return
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def _rescale_module(module, reference=0.1):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            _rescale_conv(sub, reference)


class EncBlock(nn.Module):
    """编码器: Conv1d → GroupNorm → GLU 门控"""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout * 2, 8, stride=4, padding=2)
        self.norm = nn.GroupNorm(1, cout * 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class DecBlock(nn.Module):
    """解码器: ConvTranspose1d → concat(skip) → Conv1x1融合 → GroupNorm → GLU"""
    def __init__(self, cin, cout, skip_ch):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout * 2, 8, stride=4, padding=2)
        self.fuse = nn.Conv1d(cout * 2 + skip_ch, cout * 2, 1)
        self.norm = nn.GroupNorm(1, cout * 2)

    def forward(self, x, skip):
        x = self.conv(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.norm(x)
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class DemucsLM(nn.Module):
    """时域 Demucs 轻量版 — 1D Conv U-Net + LSTM 瓶颈"""
    def __init__(self, channels=(16, 32, 64, 128, 256), rescale=0.1):
        super().__init__()
        self.channels = channels
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

        if rescale:
            _rescale_module(self, reference=rescale)

    def forward(self, x):
        # 输入归一化 (per-sample zero-mean unit-variance, 来自 Demucs)
        mono = x.mean(dim=1, keepdim=True)
        mean = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True).clamp(min=1e-4)
        x = (x - mean) / std

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

        # 反归一化
        x = x * std + mean
        return x
