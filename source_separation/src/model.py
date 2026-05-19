# model.py — LightweightUMX: 轻量频域掩码音轨分离模型
# 参照 Open-Unmix 设计，单层 BLSTM + 全连接，~758K 参数
import torch
import torch.nn as nn


class LightweightUMX(nn.Module):
    """轻量频域掩码分离模型，输入混合音频幅度谱，输出目标乐器软掩码"""

    def __init__(self, n_bins=513, hidden=128, num_layers=1):
        super().__init__()
        self.n_bins = n_bins

        self.ln = nn.LayerNorm(n_bins)  # 频率轴归一化

        self.lstm = nn.LSTM(
            input_size=n_bins,
            hidden_size=hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,  # [T, B, F] 输入格式
        )

        lstm_out = hidden * 2  # bidirectional → 2× hidden
        self.fc1 = nn.Sequential(
            nn.Linear(lstm_out, hidden),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, n_bins),
            nn.Sigmoid(),
        )

    def forward(self, mag):
        """
        Args:
            mag: [B, F, T] 混合音频幅度谱 (F=513)
        Returns:
            mask: [B, F, T] 目标乐器软掩码 (0~1)
        """
        # LayerNorm 期望最后一维是特征维 → 转置为 [B, T, F]
        x = mag.permute(0, 2, 1)  # [B, T, F]
        x = self.ln(x)
        # LSTM 期望 [T, B, F]
        x = x.permute(1, 0, 2)  # [T, B, F]
        x, _ = self.lstm(x)
        x = self.fc1(x)  # [T, B, hidden]
        x = self.fc2(x)  # [T, B, F]
        # 转回 [B, F, T]
        x = x.permute(1, 2, 0)  # [B, F, T]
        return x
