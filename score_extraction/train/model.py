"""Onsets-and-Frames with frequency pooling for GPU efficiency."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.config import train_config as cfg


class HarmonicStack(nn.Module):
    def __init__(self, n_shifts=8):
        super().__init__()
        self.n_shifts = n_shifts

    def forward(self, x):
        B, C, T, F = x.shape
        stacks = [x]
        for s in range(1, self.n_shifts + 1):
            rolled = torch.roll(x, shifts=-s, dims=-1)
            rolled[:, :, :, -s:] = 0
            stacks.append(rolled)
        return torch.cat(stacks, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3, 3), dropout=0.1, pool=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding='same')
        self.bn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(pool) if pool != (1, 1) else nn.Identity()

    def forward(self, x):
        return self.pool(self.dropout(F.relu(self.bn(self.conv(x)))))


class OnsetsAndFrames(nn.Module):
    def __init__(self, n_mels=229, n_midi=88):
        super().__init__()
        self.n_midi = n_midi
        n_input_ch = 9

        # Harmonic stack
        self.harmonic = HarmonicStack(n_shifts=8)

        # CNN encoder: frequency pooling 229 → 58 → 29 → 15
        # Channels: 9 → 16 → 32 → 64
        # pool=(time, freq): (1,2) pools frequency by 2, keeps time resolution
        self.enc1 = ConvBlock(n_input_ch, 16, kernel=(5, 5), pool=(1, 2))
        self.enc2 = ConvBlock(16, 16, kernel=(3, 3))
        self.enc3 = ConvBlock(16, 32, kernel=(5, 5), pool=(1, 2))
        self.enc4 = ConvBlock(32, 32, kernel=(3, 3))
        self.enc5 = ConvBlock(32, 64, kernel=(5, 5), pool=(1, 2))
        self.enc6 = ConvBlock(64, 64, kernel=(3, 3))

        # After 3x freq pool: 229 → 115 → 58 → 29
        reduced_mels = n_mels // 8  # 29
        cnn_feat_dim = 64 * reduced_mels  # 1856

        # Project to LSTM input
        self.feat_proj = nn.Linear(cnn_feat_dim, 256)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=128,
            num_layers=1, batch_first=True, bidirectional=True,
        )

        # Heads
        self.onset_head = nn.Sequential(nn.Linear(256, n_midi), nn.Sigmoid())
        self.frame_head = nn.Sequential(nn.Linear(256, n_midi), nn.Sigmoid())

    def forward(self, spec):
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
        # spec: (B, 1, FREQ, TIME) — librosa convention
        # Transpose to (B, 1, TIME, FREQ) for CNN processing
        spec = spec.permute(0, 1, 3, 2)  # (B, 1, T, F)

        x = self.harmonic(spec)  # (B, 9, T, F)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)  # (B, 64, T, F//8)

        # Flatten freq dimension
        B, Ch, T_out, F_out = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T_out, -1)  # (B, T, Ch*F_out)
        x = self.feat_proj(x)  # (B, T, 256)

        x, _ = self.lstm(x)  # (B, T, 256)

        onset = self.onset_head(x)
        frame = self.frame_head(x)
        return {"onset": onset, "frame": frame}


def onset_frame_loss(pred, target):
    o_loss = F.binary_cross_entropy(pred["onset"], target["onset"])
    f_loss = F.binary_cross_entropy(pred["frame"], target["frame"])
    return o_loss + f_loss
