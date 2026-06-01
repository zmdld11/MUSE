# Demucs 原版代码阅读总结

审查版本: Meta Demucs (demucs-main), GitHub 官方实现
审查日期: 2026-05-30

---

## 一、数据流

### 数据集格式
```
dataset/train/{song_name}/
    drums.wav
    bass.wav
    other.wav
    vocals.wav
    mixture.wav （自动生成，stem 求和）
dataset/valid/ 同上结构
```

- 每首歌一个文件夹，每个声源独立 wav
- 训练集不提供 mixture，在 `__getitem__` 中现场求和
- 验证集预计算 mixture.wav（含全局归一化参数）

### 数据加载 (wav.py `Wavset`)
- 片段长度：**11 秒** (segment=11, shift=1)
- 采样率：**44100 Hz**
- 通道：**立体声 (2ch)**
- `__getitem__` 返回 [sources, channels, samples] 张量
- 使用 `metadata.json` 存储每首歌的 mean/std（基于全曲 mixture 计算）

### 批内混音增强 (augment.py `Remix`)
- **不是在磁盘上混音，是在 batch 内进行**
- 每个样本提供全部 4 个声源
- 按 `group_size=4` 分组，组内随机置换声源
- 新的 mix = 置换后声源求和，目标 = 原始各声源位置
- 零额外 I/O，每个 batch 自动生成全新混合

### 数据增强链
```python
augments = [
    Shift(shift=samplerate*shift),  # 随机时间偏移
    FlipChannels(),                  # 左右声道随机翻转
    FlipSign(),                      # 随机极性翻转
    Remix(proba=1, group_size=4),   # 批内声源置换
    Scale(min=0.25, max=1.25),      # 随机增益缩放
]
```

---

## 二、模型架构 (demucs.py `Demucs`)

### 主干结构
```
Input: [B, 2, T] @ 44100Hz
  → Normalize (per-sample zero-mean unit-variance)
  → Resample 2x (44100→88200 via julius)
  → Encoder ×6:
      Conv1d(k=8, s=4) → GroupNorm → GELU
      → DConv(residual dilated convs + attention + LSTM)
      → Conv1d(1x1 rewrite) → GroupNorm → GLU
  → (可选) BLSTM(2 layers, hidden=channels)
  → Decoder ×6:
      Conv1d(1x1 rewrite, context=1) → GroupNorm → GLU
      → DConv(residual branch, 可选)
      → ConvTranspose1d(k=8, s=4)
      → GroupNorm → GELU (除最后一层)
  → Resample 0.5x (88200→44100)
  → Denormalize (×std + mean)
  → Output: [B, 4, 2, T]
```

### 通道配置
- starting channels: 64
- growth: 2.0 (每层翻倍)
- 6 层: 64 → 128 → 256 → 512 → 1024 → 2048
- 编码器输出通道 = 2 × growth^(n-1) × starting_channels

### DConv 残差块（核心创新）
```python
class DConv(channels, compress=4, depth=2, attn, lstm):
    for d in range(depth):
        dilation = 2^d
        Conv1d(channels→hidden, kernel=3, dilation)
        → GroupNorm → GELU
        → (可选) LocalAttention
        → (可选) BLSTM(2 layers, skip=True)
        → Conv1d(hidden→2*channels, kernel=1)
        → GroupNorm → GLU
        → LayerScale(channels, init=1e-4)
```
- 瓶颈压缩比 4:1（2048ch → 512 hidden）
- 扩张卷积捕捉多尺度时间依赖
- LayerScale 初始化接近零，让残差分支初期不影响主干
- Attention 和 LSTM 按层深度选择性加入（dconv_attn=4, dconv_lstm=4）

### 跳跃连接方式
```python
# Decoder forward:
skip = center_trim(skip, x)  # 精确裁剪对齐
x = decode(x + skip)         # 相加（不是拼接！）
```
- Demucs 使用**相加**融合跳跃连接，与我们旧版相同
- 但在融合前有 rewrite 1×1 卷积处理通道
- `center_trim` 精确裁剪而不是插值

### 关键初始化
```python
def rescale_conv(conv, reference=0.1):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
```
- 所有 Conv1d/ConvTranspose1d 权重的标准差缩放到 ~0.1
- 防止深层网络初始化时梯度爆炸

---

## 三、训练配置 (config.yaml)

| 参数 | 值 |
|------|-----|
| Epochs | **360** |
| Batch Size | **64** (8×GPU) |
| Optimizer | Adam (lr=3e-4, β1=0.9, β2=0.999) |
| Loss | **纯 L1 波形损失**（不用 MRSTFT） |
| Weight Decay | 0 |
| Gradient Clipping | 0 (默认不裁剪) |
| Scheduler | 无（没有 LR scheduler） |
| EMA | epoch: [0.9, 0.95], batch: [0.9995, 0.9999] |
| 输入归一化 | **per-sample 全局均值为 0 方差为 1** |
| 片段长度 | **11s @ 44100Hz** |
| 通道数 | **立体声 2ch** |
| 声源 | drums, bass, other, vocals (4 个) |

### 损失函数
```python
# L1 loss per source, weighted
loss = F.l1_loss(estimate, sources)  # 在 [sources, channels, time] 维度上
loss = loss.mean(dims=(2,3)).mean(0)  # 每声源取平均
loss = (loss * weights).sum() / weights.sum()  # 加权求和
```

---

## 四、我们的实现 vs Demucs — 差距总表

| 维度 | Demucs 原版 | 我们的 VER5.0 | 影响 |
|------|------------|-------------|------|
| **输入归一化** | per-sample mean/std | **无** | 致命 |
| **层数** | 6 | 4 | 大 |
| **首层通道** | 64→2048 | 48→384 | 致命 |
| **DConv 残差块** | 每层都有 | **无** | 致命 |
| **GroupNorm** | 每层都有 | **无** | 大 |
| **权重 rescale** | std→0.1 | 默认 Kaiming | 大 |
| **2× 上采样** | 88.2kHz 内部 | 22.05kHz 原生 | 中 |
| **片段长度** | 11s | 3s | 中 |
| **通道数** | 立体声 2ch | 单声道 1ch | 中 |
| **混音方式** | 批内置换 | 跨歌曲文件读取 | 中 |
| **训练 epoch** | 360 | 30-77 | 中 |
| **Batch size** | 64 | 16 | 小 |
| **EMA** | 多级 EMA | **无** | 小 |
| **Scheduler** | 无 | ReduceLROnPlateau | 小 |

---

## 五、核心结论

Demucs 能只靠 L1 波形损失达到 SDR 7+ dB，不是因为 L1 有多好，而是因为：

1. **6 层 + DConv 残差块**提供了远超我们模型的建模能力
2. **输入归一化**让 L1 损失在所有样本上公平优化
3. **GroupNorm + weight rescale**保证深层网络稳定训练
4. **批内混音**提供了近乎无限的训练数据多样性
5. **360 epochs** 的大量训练让大模型充分收敛

我们的 VER3.0-5.0 在 "小模型 + 无归一化 + 无残差块 + 几十轮训练" 的条件下反复撞墙 0.45 dB，**不是因为损失函数选错了，而是模型和数据 pipeline 与 Demucs 之间存在系统性代差**。
