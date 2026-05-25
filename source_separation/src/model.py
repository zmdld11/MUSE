# model.py — UNet: 频域 2D U-Net 音轨分离模型
# VER2.0: 复数谱输入 → 2D Conv U-Net → 复数掩码
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=(2, 2)):
        super().__init__()
        self.cv = nn.Conv2d(cin, cout, 5, stride=stride, padding=2)
        self.bn = nn.BatchNorm2d(cout)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.rl(self.bn(self.cv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.dc = nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x, target_size):
        x = self.dc(x)
        if x.shape[-2:] != target_size:
            x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.rl(self.bn(x))


class UNet(nn.Module):
    """频域 2D U-Net 复数掩码分离模型

    输入: [B, 2, F, T] 混合音频复数 STFT (real+imag)
    输出: [B, 2, F, T] 复数掩码 [-1, 1] (real+imag)
    """
    def __init__(self, channels=(48, 96, 192)):
        super().__init__()
        c1, c2, c3 = channels

        # Encoder
        self.enc1 = ConvBlock(2, c1, stride=(2, 2))   # [48, 257, 130]
        self.enc2 = ConvBlock(c1, c2, stride=(2, 2))  # [96, 129, 65]
        self.enc3 = ConvBlock(c2, c3, stride=(2, 2))  # [192, 65, 33]

        # Bottleneck
        self.bn1 = nn.Conv2d(c3, c3, 3, padding=1)
        self.bn2 = nn.Conv2d(c3, c3, 3, padding=1)

        # Decoder
        self.dec3 = DeconvBlock(c3 * 2, c2)  # concat skip
        self.dec2 = DeconvBlock(c2 * 2, c1)
        self.dec1 = nn.ConvTranspose2d(c1 * 2, 2, 4, stride=2, padding=1)
        # 初始偏置 +2.0 → tanh(2)≈0.96 → 初始掩码非零，加速收敛
        nn.init.constant_(self.dec1.bias, 2.0)

    def forward(self, x):
        """
        x: [B, 2, F, T] 复数 STFT
        returns: [B, 2, F, T] 复数掩码
        """
        # Encoder
        e1 = self.enc1(x)   # [B, 48, ~257, ~130]
        e2 = self.enc2(e1)  # [B, 96, ~129, ~65]
        e3 = self.enc3(e2)  # [B, 192, ~65, ~33]

        # Bottleneck
        b = self.bn1(e3)
        b = nn.functional.relu_(b)
        b = self.bn2(b)
        b = nn.functional.relu_(b)

        # Decoder with skip
        d3 = self.dec3(torch.cat([b, e3], dim=1), e2.shape[-2:])  # [B, 96, ~129, ~65]
        d2 = self.dec2(torch.cat([d3, e2], dim=1), e1.shape[-2:])  # [B, 48, ~257, ~130]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # [B, 2, F, T]
        d1 = nn.functional.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return torch.tanh(d1)
