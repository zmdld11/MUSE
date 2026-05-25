# MUSE — 音轨分离 (Source Separation) 模型研发日志

## 项目概述
基于复数频域掩码的乐器音轨分离系统。当前目标：从混合音频中分离吉他音轨。

---

### 版本: VER1.0_LightweightUMX

**日期**: 2026-05-19

**设计理念**:
采用 Open-Unmix 频域掩码方法，以极轻量架构（~758K 参数）实现吉他音轨分离。

**模型结构**:
- 输入：STFT magnitude (513 bins)
- BLSTM(1层, hidden=128) → FC(256→128→513, Sigmoid) → mask [0,1]
- 总参数量：758,531
- 相位重建：复用混合音频相位 —— **致命缺陷**

**训练配置**:
- Loss: L1(mask_pred, mask_target) | Adam(lr=3e-4) | Batch=64 | num_workers=0
- Scheduler: ReduceLROnPlateau(监听 Val Loss) — **Val Loss 不动 → LR 从未下降**

**最终性能**: 22 轮早停 | Val SDR = **-2.65 dB** (时域) / **0.13 dB** (幅度谱) | 过拟合

**VER1.0 根因**:
1. **相位问题**：mix 相位 ≠ guitar 相位，用 mix 相位重建必定失真
2. **LR 无效**：ReduceLROnPlateau 监听 Val Loss，但 Loss 不下降 → LR 从未衰减
3. **容量不足**：1 层 BLSTM，Val Loss 第 1 轮就停滞在 0.28

---

### 版本: VER1.1_ComplexUMX

**日期**: 2026-05-20

**设计理念**:
输出 **复数掩码 (real+imag)** 直接作用于 STFT 复数谱，彻底消除相位重建问题。
同时修正训练策略：加大容量+SDR 调度+GPU STFT。

**模型结构**:
- 输入：STFT magnitude (513 bins)
- LayerNorm(513) → BLSTM(**2层**, 128) → FC(256→128→1026, Tanh)
- 输出：复数掩码 [2, 513, T]，reshape 于 channel 维
- 复数应用：(a+jb)(c+jd) = (ac-bd) + j(ad+bc)
- 总参数量：**~1.21M** (+60%)

**改进点**:
| 维度 | VER1.0 | VER1.1 |
|------|--------|--------|
| 掩码 | 实数 [0,1] → 幅度 | **复数 [-1,1]** → 直接作用复数谱 |
| BLSTM | 1 层 | **2 层** |
| 参数 | 758K | **1.21M** |
| 特征提取 | CPU STFT (num_workers=0) | **GPU STFT** + num_workers=2 |
| 调度器 | 监听 Val Loss (几乎不动) | **监听 -SDR** |
| 初始 LR | 3e-4 | **1e-4** |
| 早停 patience | 15 | **10** |
| 损失函数 | L1(mask_pred, target) | **L1(|X_mix × C_mask|, |X_guitar|)** |

**待验证**:
- [ ] 训练：python -m src.train --epochs 100
- [ ] 目标：SDR ≥ +3 dB（VER1.0 仅 -2.65 dB）

**下一步方向**:
1. VER1.2: 多乐器（bass, drums）

---

### 版本: VER2.0_UNet

**日期**: 2026-05-25

**设计理念**:
根据 init.md 弃用 BLSTM 路线，改用频域 2D U-Net + 复数谱输入。
U-Net 的编码器-解码器+跳跃连接能多尺度捕获时频特征。

**模型结构**:
```
Input: [2, 513, 259] — 复数 STFT (real+imag)
Enc1: Conv2D(2→48, 5×5, stride=2) → BN → ReLU
Enc2: Conv2D(48→96, 5×5, stride=2) → BN → ReLU  
Enc3: Conv2D(96→192, 5×5, stride=2) → BN → ReLU
Bottleneck: 2× Conv2D(192→192, 3×3)
Decoder: ConvTranspose2D + skip connections
Output: Tanh → [2, 513, 259] 复数掩码
```
- 参数量: **1,984,130** (VER1.1 的 1.6 倍)

**数据集变化**: 移入 `source_separation/data/`，仅 MedleyDB (3987+1013 窗口)

**当前状态**:
- E5: L1=0.34, SDR=-33.5 dB — 收敛中但缓慢
- 推测原因：U-Net 输出 Tanh 初始 ≈0（全 0 掩码），第一轮近乎静音

**改进点**:
| 维度 | VER1.1 | VER2.0 |
|------|--------|--------|
| 骨干 | 2层 BLSTM | **3层 2D Conv U-Net** |
| 输入 | 幅度 1ch | **复数谱 2ch** |
| 参数量 | 1.21M | **1.98M** |
| 跳跃连接 | 无 | **有** |

**待解决**:
- [ ] 收敛慢：dec1 偏置初始化
- [ ] 加入 MoisesDB 数据
- [ ] 100 轮完整训练
