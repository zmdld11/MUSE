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

---

### 版本: VER3.0_DemucsLM

**日期**: 2026-05-25

**设计理念**:
U-Net 频域下采样丢失分辨率，改为时域 1D Conv U-Net (Demucs 轻量版)。
核宽通道 (32, 64, 128, 256, 512) + BLSTM 瓶颈。

**模型结构**:
```
Input: [B, 1, 66150] → Conv1D×5 (k=8 s=4 GLU: 1→32→64→128→256→512)
→ BLSTM(512→512, bidirectional) → 1024ch
→ ConvTranspose1D×5 (512→256→128→64→32→1) + skip → [B, 1, 66150]
```
参数量: **11,874,246** | Loss: 波形 L1

**数据**: MedleyDB + MoisesDB 合并数据集, 19937 train / 5063 val (3s 窗口, SR=22050)

**训练 (29 epochs)**:
- L1: 0.0246 → 0.0182 (下降 26%, 但 E10 后基本停滞)
- Val SDR: 0.25 → **0.37 dB** (E1 到 E29 仅涨 0.12 dB, 几乎不动)
- LR: 3e-4 → 1.9e-5 (ReduceLROnPlateau 触发 4 次)
- E8 出现异常波动 (SDR=0.19), E28 最佳 SDR=0.37

**根因分析**:
1. **纯 L1 波形损失与 SDR 不相关**: 模型学会输出"衰减版混合音频"即可持续降低 L1, 但吉他分离效果极差
2. **缺乏频谱监督**: 时域 L1 对频谱掩码误差不敏感, 无法引导模型学习真正的乐器分离
3. **通道过宽**: 11.87M 参数对于 19937 个训练窗口可能过拟合, 但 SDR 不涨说明模型根本没学到信号
4. **Demucs 论文对比**: 原版 Demucs 虽然也用 L1 波形损失, 但模型更大(6层)且训练 360 epochs + MUSDB18 高质量数据

**SDR 度量说明**:
- 当前计算: `SDR = 10*log10(||target||² / ||est-target||²)` per-window 平均
- 与 MUSDB18 BSSEval 有差异: BSSEval 会先做全局缩放对齐, 尺度不变 SI-SDR 更宽松
- 内部比较一致即可, 最终用分离后吉他音轨试听验证

**下一步**: VER4.0 — MRSTFT 损失 + 音频归一化 + 调小模型

---

### 版本: VER4.0_MRSTFT

**日期**: 2026-05-25

**设计理念**:
VER3.0 纯 L1 波形损失与 SDR 不相关，导致模型学会输出"衰减版混合"而非真正分离。
VER4.0 引入多分辨率 STFT (MRSTFT) 损失，在频谱域直接监督分离质量。
模型架构不变 (DemucsLM)，仅改损失函数。

**损失函数**:
```
总损失 = L1_waveform(pred, target) + α × MRSTFT(pred, target)

MRSTFT(pred, target) = 1/3 × Σ_i L1( |STFT_i(pred)|, |STFT_i(target)| )
    i ∈ {2048, 1024, 512}
```

- α = 0.5（MRSTFT 权重，弱辅助引导，避免压倒波形损失）
  - **之前 α=10.0 导致频谱损失梯度占 50-250 倍于波形损失，模型学到"正确频谱的噪声"，输出杂音**
- FFT 2048: 捕捉谐波结构 (93ms 窗口)
- FFT 1024: 捕捉音色纹理 (46ms 窗口)
- FFT 512: 捕捉瞬态细节 (23ms 窗口)

**训练结果 (α=10.0, 失败)**:
- Run 1: 29 epochs, Loss 1.85→1.12 (持续下降), SDR 0.05→0.28 (E19峰值)→0.14 (退化)
- Run 2: 续训1 epoch, SDR=0.18
- **输出音频全为杂音**: 模型学会了匹配频谱形状但相位随机，已证实 α=10.0 过大

**修正 (α=0.5)**:
- 波形 L1 主导梯度方向（相位/时域结构正确），MRSTFT 仅作弱频谱引导
- 保持 VER4.0_MRSTFT 版本号不变（架构未改，仅调超参）

**改进点**:
| 维度 | VER3.0 | VER4.0 |
|------|--------|--------|
| 损失函数 | 纯 L1(波形) | **L1(波形) + 10×MRSTFT** |
| 频谱监督 | 无 | **3 尺度 STFT L1** |
| 梯度信号 | 时域单个值 | **时域 + 频域联合** |
| Epoch 计时 | 无记录 | **记录到 log** |

**预期效果**:
- 频谱 L1 直接惩罚"混淆"，模型被迫学习真正的掩码分离
- 3 个 FFT 尺度防止模型在某一窗口长度上过拟合
- 初级阶段目标: SDR > 3 dB（VER3.0 仅 0.37 dB）

**训练配置**:
- 模型: DemucsLM, 11.87M 参数, 架构不变
- 数据: MedleyDB + MoisesDB, 19937/5063 窗口
- 优化器: Adam(lr=3e-4), ReduceLROnPlateau
- Batch=16, Epochs=100

---

### 版本: VER5.0_RemixAug

**日期**: 2026-05-26

**设计理念**:
VER3.0/4.0 三次尝试（纯 L1、α=10 MRSTFT、α=0.5 MRSTFT）SDR 全部卡在 0.37-0.42 dB。
根本原因不在损失函数，而是**模型总能看到"混合音频中已经有吉他了"**——
输出衰减版混合音频就能降低 L1 和 MRSTFT，不需要真正分离。

VER5.0 两个核心改动：

**1. 在线随机混音 (Random Remixing Augmentation)**
训练时动态生成混合：取歌曲 A 的吉他音轨 + 歌曲 B 的其他乐器 → 随机增益混合。
关键效果：新的混合中不包含目标吉他，输出"衰减版混合"反而会让损失爆炸，
模型**被迫**学会从混合中提取吉他信号。

```
Mix = g_gain × Guitar(Song_A) + Σ o_gain × Other(Song_B)
Target = g_gain × Guitar(Song_A)
```

这是 Demucs 原版论文的核心技巧。从 MoisesDB 240 首 + MedleyDB 74 首的分轨文件中，
可生成的训练样本数量几乎无限，每 epoch 看到的混合都是全新的。

**2. 模型瘦身 + 层数优化**
- 5层(stride=4) → **4层(stride=4)**：瓶颈时间步从 65 → **256 (+293%)**，保留拨弦/击弦瞬态
- 通道 (32,64,128,256,512) → **(24,48,96,192)**：参数量 11.87M → **1,663,830 (-86%)**
- 输入长度 66150→**65536**：整除 4^4=256，解码器完美对齐，无需插值
- 瓶颈 LSTM: 192→256 bidirectional，每步覆盖 ~11.6ms（原 46ms）

**改进点**:
| 维度 | VER4.0 | VER5.0 |
|------|--------|--------|
| 数据增强 | ±3dB 增益 | **在线随机混音（跨歌曲）** |
| 模型层数 | 5 | **4** |
| 跳跃连接 | 相加 (+ skip_proj) | **拼接 + Conv1x1 融合** |
| 通道数 | (32,64,128,256,512) | **(48,96,192,384)** |
| 参数量 | 11.87M | **~6.6M** |
| 输入长度 | 66150 (不可整除) | **65536 (整除 4^4=256)** |
| 瓶颈步数 | 65 | **256** |
| 每步时间分辨率 | ~46ms | **~11.6ms** |
| 插值对齐 | 每层解码器都需插值 | **完美对齐，零插值** |
| 训练集 | 固定 19937 窗口 | **50%真实 + 50%随机混音** |

**DecBlock 架构修正 (2026-05-27)**:
- 原版: ConvTranspose → add(skip) → GLU。问题：相加压缩信息，跳跃连接的浅层特征和深层特征无法有效融合
- 修正: ConvTranspose → **concat(skip)** → **Conv1x1融合** → GLU。拼接保留全部信息，1×1卷积学习最优融合权重
- 通道从 (24,48,96,192) 增大到 (48,96,192,384)，应对拼接后通道数增加的计算量

**训练结果 (VER5.0 初版, 1.66M, 相加跳跃连接)**:
- 28 epochs, 最佳 SDR=0.45 dB
- Loss 持续下降但 SDR 在 0.2-0.45 剧烈振荡，与前四版完全一致的瓶颈

**训练配置**:
- 损失: L1(波形) + 0.5×MRSTFT([2048,1024,512])
- 优化器: Adam(lr=3e-4), ReduceLROnPlateau(patience=3, factor=0.5)
- Batch=16, Epochs=100
- 数据源: MoisesDB(240首) + MedleyDB(74首) 原始分轨文件
- 训练集: RemixDataset (在线随机混音, 25000/epoch)
- 验证集: GuitarSeparationDataset (预计算固定集, 5063 窗口)

**预期**:
- 随机混音切断了"输出混合=降低损失"的捷径
- 4 层模型保留更精细时间结构，减少过拟合
- 目标: SDR > 3 dB（首次突破 0.5 dB 墙）

**实际结果 (VER5.0 三版, 全部失败)**:
| 子版本 | 跳跃连接 | 参数量 | Epochs | 最佳 SDR |
|--------|---------|--------|--------|----------|
| add-skip | 相加 | 1.66M | 28 | 0.45 |
| concat-skip | 拼接 | 7.03M | 36 | 0.43 |
| concat-skip(再次) | 拼接 | 7.03M | 36 | 0.43 |
| 所有版本 SDR 上限 | — | — | — | **0.45 dB** |

六次训练（含 VER3.0/4.0），SDR 天花板确认为 0.45 dB。已确认非损失函数或数据问题，而是模型架构存在系统性差距。

---

### 版本: VER6.0_Phase1

**日期**: 2026-05-30

**设计理念**:
基于 Demucs 源码审查（见 `demucs_code_review.md`），补全我们模型中缺失的关键组件。
不改模型架构（4层 + 拼接式 DecBlock），只加归一化和训练稳定性组件。

**改动清单**:
| # | 改动 | 来源 | 效果 |
|---|------|------|------|
| 1 | 输入归一化 | Demucs forward() | per-sample zero-mean unit-variance |
| 2 | GroupNorm | Demucs EncBlock/DecBlock | 每层 Conv 后加 GroupNorm(1, channels) |
| 3 | 权重 rescale | Demucs rescale_module() | 所有 Conv1d 权重 std→0.1 |
| 4 | 梯度裁剪 | 通用实践 | clip_grad_norm(max_norm=1.0) |
| 5 | 移除 Scheduler | 学 Demucs | 恒定 LR=3e-4, 不衰减 |
| 6 | 增大片段 | — | 65536→131072 (~6s, 整除 4^4) |

**模型结构** (与 VER5.0 相同):
```
EncBlock: Conv1d → GroupNorm → GLU
DecBlock: ConvTranspose1d → concat(skip) → Conv1x1融合 → GroupNorm → GLU
DemucsLM: 4层, 通道(48,96,192,384), 输入归一化+反归一化
```

**训练配置**:
- 损失: L1(波形) + 0.5×MRSTFT([2048,1024,512])
- 优化器: Adam(lr=3e-4), **恒定 LR（无 Scheduler）**
- 梯度裁剪: max_norm=1.0
- Batch=8（因片段加倍，显存翻倍）
- 片段: 131072 样本 (5.94s @ 22050Hz)
- 早停 patience: 20（放宽，因无 LR 衰减后需要更多轮）
- 数据: 50% 预计算真实混音 + 50% 跨歌曲随机混音

**预期**:
- GroupNorm + 输入归一化 → 训练稳定性大幅提升
- 权重 rescale → 深层梯度不爆炸
- 更长片段 → 更多音乐上下文
- 目标: SDR > 1 dB（突破 0.5 dB 天花板）
