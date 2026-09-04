"""VER4.0 全新模型 (2026-08-07, v2 瘦身版)

对比 VER3.x (HarmonicStack + 6 层 CNN + 单层 BiLSTM, ~1M 参数):
  - ResBlock 前端 (通道 32→64→96, 3 次频域池化)
  - 时序后端可选: transformer (2 层, d_model 192, 默认) / 双层 BiLSTM
  - 三头: onset / frame / offset
  - 只实例化所选后端 (避免冗余参数占显存)
参数约 2-2.5M, 4060 8GB 可训练.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


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


class ResBlock(nn.Module):
    """(3,3) 卷积残差块, 输入输出同通道."""

    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, (3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, (3, 3), padding="same")
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(F.relu(self.bn2(self.conv2(h))))
        return F.relu(x + h)


class ConvStage(nn.Module):
    """一个 stage: 升通道卷积 + 2 个 ResBlock + 频域池化."""

    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, (3, 3), padding="same")
        self.bn = nn.BatchNorm2d(out_ch)
        self.blocks = nn.Sequential(ResBlock(out_ch, dropout),
                                    ResBlock(out_ch, dropout))
        self.pool = nn.MaxPool2d((1, 2))

    def forward(self, x):
        x = F.relu(self.bn(self.proj(x)))
        x = self.blocks(x)
        return self.pool(x)


class OnsetsFramesV4(nn.Module):
    def __init__(self, n_mels=229, n_midi=88, backend="transformer"):
        super().__init__()
        self.n_midi = n_midi
        self.backend = backend
        self.harmonic = HarmonicStack(n_shifts=8)  # 9 ch

        self.stage1 = ConvStage(9, 32)     # F: 229 -> 115
        self.stage2 = ConvStage(32, 64)    # F: 115 -> 58
        self.stage3 = ConvStage(64, 96)    # F: 58 -> 29

        reduced_mels = n_mels // 8  # 29
        d_model = 192
        cnn_feat_dim = 96 * reduced_mels  # 2784
        self.feat_proj = nn.Linear(cnn_feat_dim, d_model)

        if backend == "transformer":
            self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=2048)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=768, dropout=0.1,
                batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.lstm = nn.LSTM(input_size=d_model, hidden_size=192,
                                num_layers=2, batch_first=True,
                                bidirectional=True)
            self.lstm_proj = nn.Linear(384, d_model)

        self.onset_head = nn.Sequential(nn.Linear(d_model, n_midi), nn.Sigmoid())
        self.frame_head = nn.Sequential(nn.Linear(d_model, n_midi), nn.Sigmoid())
        self.offset_head = nn.Sequential(nn.Linear(d_model, n_midi), nn.Sigmoid())

    def forward(self, spec):
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
        spec = spec.permute(0, 1, 3, 2)  # (B,1,T,F)
        x = self.harmonic(spec)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)  # (B,96,T,F//8)

        B, Ch, T, F_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        x = F.relu(self.feat_proj(x))
        if self.backend == "transformer":
            x = self.transformer(self.pos_enc(x))
        else:
            x, _ = self.lstm(x)
            x = F.relu(self.lstm_proj(x))

        return {
            "onset": self.onset_head(x),
            "frame": self.frame_head(x),
            "offset": self.offset_head(x),
        }


def frame_to_offset_target(frame):
    """frame (B,T,88) 0/1 → offset 标签 (下降沿)."""
    off = torch.zeros_like(frame)
    off[:, :-1] = (frame[:, :-1] == 1.0) & (frame[:, 1:] == 0.0)
    return off.float()


def v4_loss(pred, target, onset_weight=1.0, offset_weight=2.0,
            frame_w=None, onset_w=None, offset_w=None,
            onset_pos=2000.0, frame_pos=10.0, offset_pos=100.0):
    device = pred["frame"].device
    o = F.binary_cross_entropy(pred["onset"], target["onset"], reduction="none")
    f = F.binary_cross_entropy(pred["frame"], target["frame"], reduction="none")
    off_t = frame_to_offset_target(target["frame"]).to(device)
    of = F.binary_cross_entropy(pred["offset"], off_t, reduction="none")
    # 正样本加权: 标签极稀疏, 不加权模型会把输出全压 0 (onset_f1=0 的退化解)
    o = o * (target["onset"] * onset_pos + (1 - target["onset"]))
    f = f * (target["frame"] * frame_pos + (1 - target["frame"]))
    of = of * (off_t * offset_pos + (1 - off_t))
    if onset_w is not None:
        o = o * onset_w.to(device)
    if frame_w is not None:
        f = f * frame_w.to(device)
    if offset_w is not None:
        of = of * offset_w.to(device)
    return onset_weight * o.mean() + f.mean() + offset_weight * of.mean()
