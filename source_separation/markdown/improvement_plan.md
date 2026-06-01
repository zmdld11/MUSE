# VER6.0 改进计划

基于 Demucs 源码审查（见 `demucs_code_review.md`），制定分阶段改进路线。

当前状态: VER5.0_RemixAug, 7M 参数, SDR 天花板 0.45 dB
当前环境: RTX 4060 8GB + AutoDL RTX 5090 32GB

---

## Phase 1: 关键组件补齐（1-2天，本地 4060）

不改模型架构，只加 Demucs 中缺失的关键训练组件。

### 改动清单

| # | 改动 | 文件 | 效果 |
|---|------|------|------|
| 1 | **输入归一化** | model.py forward() | per-sample zero-mean unit-variance |
| 2 | **GroupNorm** | model.py EncBlock/DecBlock | 每层卷积后加 GroupNorm(1, channels) |
| 3 | **权重 rescale** | model.py __init__ | 所有 Conv1d 权重 std→0.1 |
| 4 | **梯度裁剪** | train.py | clip_grad_norm(max_norm=1.0) |
| 5 | **移除 Scheduler** | train.py | 学 Demucs，不用 ReduceLROnPlateau |
| 6 | **增大片段** | config.py | NUM_SAMPLES: 65536→131072 (~6s, 整除 4^4) |

### 预期效果
- 训练稳定性大幅提升（不再出现 E14 SDR 从 0.45 跳水到 0.20）
- SDR 目标: **突破 1 dB**

### 风险
- GroupNorm 增加少量参数和计算
- 片段加倍导致 batch 可能变小 → 需调整 batch_size

---

## Phase 2: 架构升级 — DConv + 深度增加（2-3天，本地+AutoDL）

### 改动清单

| # | 改动 | 文件 | 效果 |
|---|------|------|------|
| 1 | **DConv 残差块** | model.py 新增 DConv 类 | 每层加扩张卷积残差分支 |
| 2 | **深度 4→5** | config.py | DEMUCS_CHANNELS=(48,96,192,384,768) |
| 3 | **LayerScale** | model.py 新增 | 残差分支初始化为近似恒等 |
| 4 | **增大训练集** | remix_dataset.py | 改为批内混音（减少磁盘 I/O） |
| 5 | **Channel 改立体声** | 全链路 | audio_channels=2（立体声分离更准） |

### DConv 块设计（简化版）
```python
class DConv(nn.Module):
    def __init__(self, channels, compress=4, depth=2):
        hidden = channels // compress
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, hidden, 3, dilation=2**d, padding=2**d),
                nn.GroupNorm(1, hidden), nn.GELU(),
                nn.Conv1d(hidden, 2*channels, 1),
                nn.GroupNorm(1, 2*channels), nn.GLU(1),
                LayerScale(channels, 1e-4),
            ) for d in range(depth)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # 残差连接
        return x
```

### 预期效果
- DConv 提供多尺度时间依赖（扩张 1,2,4,8...）
- 模型参数 ~15-20M
- SDR 目标: **突破 3 dB**

### 风险
- 参数量变大，4060 8GB 可能不够 → 需上 AutoDL
- 训练时间显著增加（每 epoch 可能 5-10 分钟）

---

## Phase 3: 数据 Pipeline 重构（1-2天）

完全对齐 Demucs 数据流。

### 改动清单

| # | 改动 | 说明 |
|---|------|------|
| 1 | **数据集格式重做** | 每首歌一个文件夹，每个声源一个 wav |
| 2 | **11s 片段** | 44100Hz × 11s = ~485k samples |
| 3 | **预计算 metadata** | 全曲 mean/std 用于归一化 |
| 4 | **批内混音** | 替代跨歌曲磁盘读取，大幅加速 |
| 5 | **Shift/Flip 增强** | 随机时间偏移 + 通道翻转 + 极性翻转 |

### 数据集组织
```
data/demucs_format/
  train/
    song_001/
      guitar.wav       （所有吉他轨道求和）
      other.wav        （所有非吉他轨道求和 = mix 减吉他）
      metadata.json    （mean, std, samplerate, length）
    song_002/
      ...
  valid/
    （同上结构）
```

训练时: `[guitar, other]` 两个声源，batch 内随机置换。
实际上对吉他分离只有 2 个"声源"，但批内混音仍然有效——置换吉他轨道跨样本。

### 预期效果
- 训练数据多样性大幅提升
- 每 epoch 时间缩短（减少磁盘 I/O）
- SDR 继续提升

---

## Phase 4: 对标原版完整训练（AutoDL 5090, 2-3天）

### 改动清单

| # | 改动 | 说明 |
|---|------|------|
| 1 | **2× 上采样** | julius.resample_frac 实现内部 88.2kHz |
| 2 | **360 epochs** | 充分训练 |
| 3 | **EMA** | 多级指数滑动平均模型权重 |
| 4 | **纯 L1 loss** | 去掉 MRSTFT（大模型不需要） |
| 5 | **Batch=32** | 5090 32GB 可支撑 |

### 预期效果
- SDR 目标: **> 5 dB**（对标 Demucs 单声源性能）

---

## 优先级总结

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
 (本地)     (AutoDL)    (本地)      (AutoDL)
 1-2天      2-3天       1-2天       2-3天
 目标1dB    目标3dB     数据对齐    目标5dB+
```

### 当前建议

**先做 Phase 1**。不改架构，只加归一化+GroupNorm+rescale+梯度裁剪+6s片段。如果 SDR 能从 0.45 涨到 1 dB 以上，证明方向对了，再投时间做 Phase 2。

如果 Phase 1 做完 SDR 还是不涨，说明问题更深层（可能是数据质量问题），需要重新评估。

---

## 附录: 当前全版本 SDR 记录

| 版本 | Loss | 数据 | 模型 | Epochs | 最佳 SDR |
|------|------|------|------|--------|----------|
| VER1.0 | L1 mask | 预计算 | BLSTM 758K | 22 | -2.65 |
| VER2.0 | L1 mask | 预计算 | UNet 1.98M | 5 | -33.5 |
| VER3.0 | L1 waveform | 预计算 | DemucsLM 11.87M | 29 | 0.37 |
| VER4.0 α=10 | L1+10×MRSTFT | 预计算 | DemucsLM 11.87M | 29 | 0.28 |
| VER4.0 α=0.5 | L1+0.5×MRSTFT | 预计算 | DemucsLM 11.87M | 77 | 0.42 |
| VER5.0 纯remix | L1+0.5×MRSTFT | 100%remix | DemucsLM 1.66M | 11 | -1.59 |
| VER5.0 50/50 | L1+0.5×MRSTFT | 50%real+50%remix | DemucsLM 1.66M | 28 | 0.45 |
| VER5.0 concat | L1+0.5×MRSTFT | 50%real+50%remix | DemucsLM 7.03M | 36 | 0.43 |
