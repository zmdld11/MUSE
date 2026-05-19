# 乐器识别模型研发日志

## 待突破基线：CNN (85%) -> ResNet (95%)

### 版本: VER1.0 (已合档)
**模型结构**：
- 输入层：1 通道 / 3 通道 Mel-spectrogram
- 主干网络：PyTorch 官方预训练的 ResNet18 特征提取器（相比前版 CNN 带深层残差连接）
- 输出层：Dropout(0.5) 避免过拟合 + 11 维全连接分类器输出。

**创新点 / 尝试**：
1. 使用深层残差结构（ResNet）代替浅层 CNN 进行多乐器分类，期望泛化能力提升。
2. 更改最后的全连接结构，增加一个 Dropout。
3. 动态配置通道及超参（`in_channels=1` / `num_classes=11`）并打包保存在 `pth` 伴随字典中，方便后续的测试代码能够实现“自适应加载不脱节”。

**最终模型性能评价**：
- Train Loss: 平均降低至 0.13 左右（但产生过拟合现象）
- Val Loss: 平均在 2.3 左右，与训练 Loss 产生显著背离
- Val Acc: 最终稳定在 53% 左右

---

### 版本: VER1.1 (ResNet34 初探灾难)
**模型结构**：
- 输入层：1 通道 / 3 通道 Mel-spectrogram
- 主干网络：PyTorch `ResNet34` (比起 1.0 的 18 更深)
- 丢弃层：最后加入两个强力组合：`Dropout(0.6) + Linear(256) + ReLU + Dropout(0.4)`，大幅压缩拟合余地

**创新点 / 尝试**：
1. 数据增强（Data Augmentation）：在 `dataset.py` 中新加了随机环境频域（`FrequencyMasking=15`）和时域（`TimeMasking=15`）裁剪。
2. 加入 `L2 权重衰减 (weight_decay=1e-3)`。

**下次修改建议/分析**：
尽管挂载了最高强度的 Dropout 和数据增强。但 1.1 版本依然陷入了过拟合和特征不举的困境（卡死在 60%）。
这是由一个**致命失误**引起的：`ResNet34` 预加载了 ImageNet 的权重，但其实那是识别 RGB（3 通道）图像的。在我们的代码中，用 `nn.Conv2d(1, 64...)` 将它的第一层强行替换成了单通道，但这**完全清空了预训练特征**并用随机数进行了初始化，被迫在千张样本上重头训练千万参数 ResNet34。

---

### 版本: VER1.2 (跨界微调复用术 / 剑指 90%+)
**模型结构**：
- 输入层：1 通道 Mel-spectrogram
- 主干网络：PyTorch预训练 `ResNet34`
- 修复核心：**预训练权回复**。将原来 ImageNet 第一层 Conv2D (3通道) 的权重通过取平均（`weight.sum(dim=1, keepdim=True)`）或者同等复制，巧妙地塞回我们强行魔改的、接收 1 通道频谱图的第一层小卷积内核中。让网络保留原本的“边缘、纹理特征提取能力”。

**最终模型性能评价**：
- Train Loss: 最低降至 0.008 左右
- Val Loss: 一直在 1.85 ~ 1.95 波动
- Val Acc: 最终稳定在 62.6% 左右，仍然未能突破 85% 基线。

---

### 版本: VER1.3 (回归本质：旧版特性移植重建 / 剑指超越 85%)
**模型结构**：
- 输入层：1通道提取后使用 `.repeat(3, 1, 1)` **直接伪装复制为 3 通道图**，完美适配 ResNet34 输入要求（不修改主干的 conv1）。
- 主干网络：PyTorch预训练 `ResNet34` (不破坏任何原生结构)

**创新点 / 尝试**：
1. **彻底修复特征撕裂**：不改主干通道数，用真实的三通道“伪彩色图”骗过网络从而100%保留 ImageNet 提取能力。
2. **底层标准化回归**：在 `dataset.py` 中将音频处理从容易受异常值暴击的 `Min-Max` 区间缩放全面换回 `Z-Score`。
3. **更换稳定策略**：采用带 `momentum=0.9` 的 `SGD` 来替换 `Adam`。

---

### 版本: VER1.4_CustomCNN
**模型结构**：
- 架构：Simplified Advanced Classifier（源于以前的4层CNN改进型架构）。

**模型性能评价**：
- 验证集准确率 (Val Acc)：达到了约 80.8%。
- 问题：推理测试（infer）暴露出严重的“幻觉”问题（频繁产生误报）。

---

### 版本: VER1.5_AugmentedCNN
**目标**：突破80%瓶颈期，降低推理时的幻觉与误报率，并向85%以上的基线推进。

**创新点 / 尝试**：
1. **训练中引入 SpecAugment**：时间遮蔽与频率遮蔽，强制模型不去强行记忆背景瞬态噪声。
2. **Cosine Annealing 学习率调度器**。

**最终模型性能评价**：
- Val Acc: 最终只爬升到约 74.5% 左右。
- 问题：典型的强力正则化导致小模型（76k参数）欠拟合。

---

### 版本: VER1.6_WiderCNN
**目标**：解决 VER1.5 各路正则化太强导致的欠拟合问题，提升网络脑容量。

**创新点 / 尝试**：
1. **网络加宽 (容量倍增)**：将基础 CNN 的逐层通道翻倍 `(16->32 -> 32->64 -> 64->128)`。让其拥有足够的参量空间来消化正则化带来的扰动。
2. **减弱束缚**：去除了 weight_decay，降低了 SpecAugment 强度。

**最终模型性能评价**：
- Val Acc: 最终被卡在 82%~83% 之间。（由于去除了太多束缚，出现了严重过拟合，Train Acc 达 94.4%）。

---

### 版本: VER1.7_MixedWiderCNN
**目标**：纠正数据引入时忘记修改版本号导致污染1.6断点的问题。同时解决加上纯净音轨后再次出现的过拟合现象。

**创新点 / 尝试**：
1. **纯净音轨“音头”训练**：利用 MedleyDB 抽出极短爆破音。
2. **回调 L2 正则约束**：重新添加 `weight_decay=1e-4`。

---

### 版本: VER1.8_MixupTimeShift
**目标**：解决 VER1.7 中由于过度对齐“纯净音轨0.1s作为音头”而在推理阶段出现的严重位移敏感与过拟合问题。

**创新点 / 尝试**：
1. **时间轴循环平移 (Time-Shift Augmentation)**：在 `__getitem__` 放入了 `-30% ~ +30%` 随机时间平移操作（Roll），彻底打破模型对时间的“硬对齐”依赖。
2. **动态混音增强 (Mixup Augmentation)**：直接切入了学术界最前沿的 Mixup 机制，基于 `alpha=0.2` 实时按比例融合两段音频。

---

### 版本: VER2.0_MedleySynthetic (数据集大洗牌突破 95%)
**目标**：彻底抛弃单精度单标签瓶颈（IRMAS），全面转入多标签动态混音机制，并解决包含大段空白音频导致的虚假高准确率问题。

**创新点 / 尝试**：
1. **全面纯净音轨化**：彻底淘汰包含大量混音但只有单标签的 IRMAS 数据集。转用 MedleyDB 分离的干净分轨（Stems）。
2. **手搓多标签实时重混 (Custom Dynamic Mixer)**：每一步随机抽取完全不同乐器的有效音频切片进行叠加混合。
3. **架构变质 (CrossEntropy -> BCEWithLogitsLoss)**：升级成了多标签组合，分类输出转为了“多项选择题”模式（`Sigmoid`）。

---

### 版本: VER2.1_PrebuiltStaticMix (预生成静态混合测试)
**目标**：解决 VER2.0 运行前动态混音读取造成 IO/CPU 瓶颈导致的显卡占用率鸭蛋的缓慢问题。

**创新点 / 尝试**：
1. **统一的离线数据合成库**：在 `MUSE/data/` 下独立编写了 `build_mixed_dataset.py`，用于一次性静态扫描、静音处理并拼接 MedleyDB 分离音轨生成真正的物理数据集。
2. **专属 DataLoder 桥接**：抛弃了极其拉胯受制于单线程和 CPU 的 `librosa`，重新使用底层的 `torchaudio` 初始化。开满数据总线 DMA（`num_workers`, `pin_memory=True`），使显存数据直通全开。

**预期**：利用这个“物理写死”的真实多标签混合数据集让开发能够亲耳听懂在测什么，并且由于更换到 C++ `torchaudio` 挂载计算图底座进行读取拉流，训练耗时将被直接斩断降至每轮十几秒。

---

### 版本: VER2.2_ClassBalanceCheck (排查电吉他分布风暴)
**目标**：排查验证集高达 94%+，但推理测真实歌曲全变“电吉他”的幻觉坍塌问题。

**创新点 / 尝试**：
1. **数据分布透明化**：在 `dataset.py` 中强制统计 `build_mixed_dataset.py` 生成的各个标签在此轮混合中的确切占比，并显式输出。
2. **底层日志截留记录**：修改 `train.py` 取出数据集的生成分布，写死在时间戳 `.log` 文件的头部。

**预期**：通过类级别的发生频率证明模型所谓的测试集 94% 其实全是极度倾斜的不平衡蒙对。找出电吉他分布风暴的原因就能解决全吉他崩溃的故障。

---

### 版本: VER2.3_StrictMetric (修正虚假Accuracy坍塌)
**目标**：解决模型被大量全 0（该帧未出现该乐器）负样本迷惑，导致混数据时即便答了全0和电吉他也能拿到虚高准确率，而真正在推理时错误百出（50%真实命中都不到）的假象。

**创新点 / 尝试**：
1. **更改评价机制 (Shift to F1-Score)**：满足了“答对一个加一分，答错/漏答倒扣分”的真实严格要求。我舍弃了粗暴比较 `(predicted == targets).sum() / total` 带来的多标签红利（True Negatives 高得离谱）。
2. **F1 评价指标**：引入了严格的 `TP, FP, FN` 统计机制，使用 $2 \times TP / (2 \times TP + FP + FN)$ 计算精准的 F1 率。如果模型胡乱全部猜测 `0` 或疯狂乱猜某个乐器（如全是电吉他引发大量 FP），它的 F1 值会立刻暴跌回归真实的水平。

**预期**：通过 F1 Score 取代失真的多标签 Accuracy，让模型真正地去寻找能够提取特征的有表现力的乐器音色，不再陷入”我不找，我就猜没人弹，我就能得分”的陷阱！重建了真正硬核的评分排名。

### VER2.3_StrictMetric -- 最终评估

**模型**: SimplifiedAdvancedClassifier (2残差块 + SE通道注意力, ~302K参数)
**数据**: 10,000 预混 MedleyDB 样本 (80/20 分割)
**训练**: 200 epochs, BCEWithLogitsLoss, Adam(lr=1e-3, wd=1e-4), CosineAnnealingLR
**增强**: Mixup(alpha=0.2, 50%), SpecAugment, Time-shift(±30%)

**最终指标**:
- 最佳 Val F1: **82.85%** (epoch 183)
- 训练 F1 收敛值: ~82.0%
- 平台期表现: 模型从 epoch 140 开始在 81.8-82.8% 之间震荡
- 无过拟合: Train 和 Val F1 紧密跟踪，表明是表示能力瓶颈而非过拟合

**根本原因诊断**:
1. 无时序建模 — Global Average Pooling 完全丢弃了时间结构信息
2. 损失函数未处理类别不平衡 — BCE 对所有类别一视同仁
3. CNN 骨干较浅 — 仅 2 个残差块
4. 无音头/起始瞬态特征

---

### 版本: VER3.0_TransformerFocal (剑指 95% F1)
**目标**：突破 VER2.3 的 82.85% F1 平台期。核心思路：(1) 在 CNN 骨干之后加入 Transformer 时序自注意力编码器，让模型学会”哪个时间段对哪种乐器重要”；(2) 用 Focal Loss 替代 BCE，通过逆频率权重强制模型关注合成器、木吉他等稀有类别。

**模型结构**：
- **CNN 骨干 (不变)**：init_conv(1→32) → ResidualBlock(32→64, stride=2) + AttentionModule(64, r=2) → ResidualBlock(64→128, stride=2) + AttentionModule(128, r=4)
- **频率维度压缩**：AdaptiveAvgPool2d((1, None)) → squeeze → permute → [B, T', 128]
- **时序编码器 (新增)**：PositionalEncoding(128) → TransformerEncoder(2层, 4头, d_model=128, FFN=512, gelu, dropout=0.1)
- **输出头**：mean(dim=1) → Dropout(0.3) → Linear(128, 10)
- **总参数量**：~1.48M (CNN ~302K + Transformer ~1.18M)

**创新点 / 尝试**：
1. **时序自注意力机制 (Transformer Temporal Encoder)**：在 CNN 提取的局部时频特征之上，用 2 层多头自注意力建模 3 秒窗口内的长程时序依赖。位置编码使用正弦函数（零额外参数）。这是从”只看局部纹理”到”理解整段时序结构”的关键跃迁。
2. **Focal Loss (gamma=2.0, class-weighted alpha)**：用 `-(1-p_t)^γ * log(p_t)` 替代标准 BCE。对已能轻松正确分类的样本（如电贝斯静音段）自动降权，迫使模型将学习能力分配给困难样本和稀有类别。Alpha 权重从训练集分布反推 —— 合成器 ~4x，木吉他 ~3.3x，电贝斯 ~0.64x。
3. **学习率下调**：从 1e-3 降至 5e-4，适配更大的参数量以保持训练稳定。
4. **数据扩容与平衡采样**：数据集从 10K 扩至 15K，数据生成器中按逆频率加权采样分轨，提高稀有类别的混合出现率。

**预期**：Transformer 提供时序推理能力（”架子鼓在整段中反复出现”而非”这一帧听起来像架子鼓”），Focal Loss 解决类别不平衡，两者叠加预期将 F1 从 82.85% 推至 90%+。剩余的 95% 缺口计划通过 VER3.1 的音头特征进一步补足。

**最终模型性能评价**：
- Train F1: 仅约 59%（严重欠拟合）
- Val F1: 约 70%
- 问题：加入 Transformer 自注意力后模型不仅没有突破 90%，反而出现严重的欠拟合。根本原因：在把 CNN 特征送入 Transformer 之前，使用了 `AdaptiveAvgPool2d((1, None))` 粗暴地将频率维度平均池化为 1，这相当于把所有 mel 频段（音色）揉成了一个平均值——音色信息完全丧失。模型只剩时间包络（响度起伏）可以学习，根本无法区分乐器。

**致命教训**：在音频分类中，**频率维度不可压缩**。Mel-spectrogram 的 128 个频段分别对应不同音高范围的谐波分布，这是区分乐器（如小提琴 vs 长笛 vs 钢琴）的最核心特征。AdaptiveAvgPool2d 将其池化为 1 相当于”把一幅彩色图片压成一条灰线，然后要求分辨图中物体”——模型只能靠亮暗变化（时间上的响度变化）来猜。

---

### 版本: VER3.1_TransformerFreqProj (特征保真映射)
**目标**：修复 VER3.0 中频率维度粗暴池化导致的严重欠拟合问题，在不破坏 Transformer 时序建模能力的前提下，保留乐器的音色（频率）特征。

**模型结构**：
- **CNN 骨干 (不变)**：init_conv(1→32) → ResidualBlock(32→64, stride=2) + AttentionModule(64, r=2) → ResidualBlock(64→128, stride=2) + AttentionModule(128, r=4)
- **频率卷积投影 (新增)**：Conv2d(128→256, kernel_size=(36,1)) → BN → ReLU（将 36 个频率 bin 完整投影到 256 维通道，保留全部音色信息）
- **时序编码器**：squeeze(2) → permute → PositionalEncoding(256) → TransformerEncoder(2层, 8头, d_model=256, FFN=1024, gelu, dropout=0.1)
- **输出头**：mean(dim=1) → Dropout(0.3) → Linear(256, 10)
- **总参数量**：~3.06M

**创新点 / 尝试**：
1. **全频段卷积投影 (Frequency Projection)**：移除了 VER3.0 致密的 AdaptiveAvgPool2d((1, None))，替换为 Conv2d(128, 256, kernel_size=(36, 1))。把特征图在频率轴（高度 36）上的所有频带完整保留，并通过全连接式的卷积窗口直接映射（Project）到 256 维的隐通道中。频率信息零损失。
2. **升维 Transformer**：借助卷积投影，时序 token 的表征能力大大增强，顺势将 Transformer 的 d_model 翻倍提升至 256（head=8, dim_feedforward=1024）。
3. **背景样本抑制幻觉**：数据生成器中加入 8% 的全零标签背景样本，教模型学会说”这里没有乐器”。
4. **温和类平衡采样**：从激进的反频率加权改为 sqrt 反频率加权，避免稀有类过度采样导致过拟合。

**最终模型性能评价 (200 Epochs)**：

| 指标 | 数值 |
|------|------|
| 最佳 Val F1 | **84.85%** (epoch ~190) |
| 最终 Train F1 | 81.02% |
| 最终 Val F1 | 84.07% |
| Train/Val F1 差距 | Val 持续高于 Train 约 3-10% |

**关键发现**：
1. **Val > Train 不是欠拟合**：这是 Mixup 训练的常规现象——训练时标签被混合（软标签 0~1），验证时使用干净标签（0或1），训练任务天然更难。业界文献已确认此行为正常。
2. **F1 从 VER2.3 的 82.85% → 84.85%，仅提升 2%**：Transformer + Focal Loss + 频率投影带来的增益远低于预期的 90%+。模型已在 84-85% 附近形成新的平台期。
3. **数据严重不均衡**：合成器仅 384 个训练样本，电贝斯 2977 个（7.75倍差距）。稀有类的学习极度不足。
4. **实际推理中的漏检**：全局统一阈值 0.65 对稀有类（合成器、小提琴）过高，对高频类（电贝斯、钢琴）可能合适。需要按类别独立设定阈值。

**根因分析与下一步方向**：
1. **类别自适应阈值**：为每个乐器独立学习最优判定阈值，取代当前全局 0.65。直接解决”稀有类漏检、常见类可检”的矛盾。
2. **Phase 特征缺失**：当前仅使用幅度谱（Mel + MFCC），完全丢弃相位信息。Modgd-gram（改进群延迟）可捕获音头瞬态——这恰好是 VER3.0 计划中推迟的”attack transient”方向。
3. **Focal Modulation 替代自注意力**：2025 年最新论文显示，对于短序列（当前仅 33 步），Focal Modulation（O(N) 复杂度）可能优于标准自注意力。
4. **MERT 特征蒸馏**：ICLR 2024 的 MERT 模型在 160K 小时音乐上预训练，其冻结特征 + 轻量分类头在 HamNava 数据集上达 85% F1。可以考虑将其知识蒸馏到 ~3M 的小模型。
5. **数据再平衡**：进一步调整 stem 采样策略，确保最稀有类至少占训练样本的 8%。

详见 `markdown/research_2023_2025.md` 中的综述分析。

---

### 版本: VER3.2_AdaptiveThreshold (类别自适应判定阈值)

**目标**：解决 VER3.1 实际推理中 singer 等乐器严重漏检的问题。核心发现：全局统一阈值 0.65 对不同类别极不公平——singer 最优阈值为 0.35，用 0.65 判定几乎漏掉了一半以上的歌手段落。

**创新点 / 尝试**：

1. **类别自适应阈值 (Class-Adaptive Thresholding)**：
   - 在验证集上对每个乐器独立搜索 [0.20, 0.85] 区间内使 F1 最大化的最优判定阈值
   - 阈值保存至 `model/class_thresholds.json`，推理时自动加载
   - 彻底解决"同一个标准对所有乐器一刀切"的问题

2. **逐类 F1 追踪**：
   - 训练时新增每个 epoch 的逐类 F1 统计（TP/FP/FN per class）
   - 打印输出和日志文件中加入 `per_class_f1` 字段
   - 便于及时发现某个乐器的检测退化

**最终模型性能评价 (200 Epochs, 完整重训)**：

| 指标 | 数值 |
|------|------|
| 最佳全局 Val F1 | **84.63%** (epoch 191) |
| 最终 Train F1 | 82.25% |
| 最终 Val F1 | 84.23% |
| Train/Val F1 差距 | Val 持续高于 Train 约 3-5% (Mixup 正常行为) |

**逐类 F1 (epoch 191 最佳模型, at 默认阈值 0.5)**：

| 等级 | 乐器 | F1 | 数据量 |
|------|------|-----|--------|
| 🟢 优秀 | drum set | **0.944** | 2534 |
| 🟢 优秀 | singer | **0.934** | 2616 |
| 🟢 优秀 | acoustic guitar | **0.901** | 593 |
| 🟡 良好 | electric guitar | 0.869 | 2184 |
| 🟡 良好 | synthesizer | 0.849 | 366 |
| 🟡 良好 | electric bass | 0.848 | 2955 |
| 🟠 一般 | piano | 0.785 | 2897 |
| 🔴 薄弱 | cello | 0.751 | 854 |
| 🔴 薄弱 | violin | 0.750 | 601 |
| 🔴 薄弱 | flute | **0.718** | 225 |

**逐类最优阈值 (从 VER3.2 最佳模型 + 验证集重新计算)**：

| 乐器 | 旧阈值 (VER3.1) | 新阈值 (VER3.2) | F1@新阈值 |
|------|-----------------|-----------------|-----------|
| acoustic guitar | 0.55 | **0.55** (=) | 0.9823 |
| cello | 0.55 | **0.50** (-0.05) | 0.9604 |
| drum set | 0.30 | **0.35** (+0.05) | 0.9908 |
| electric bass | 0.30 | **0.30** (=) | 0.9519 |
| electric guitar | 0.40 | **0.35** (-0.05) | 0.9780 |
| flute | 0.50 | **0.45** (-0.05) | 0.9455 |
| piano | 0.30 | **0.30** (=) | 0.9444 |
| **singer** | 0.35 | **0.30** (-0.05) | **0.9878** |
| synthesizer | 0.60 | **0.60** (=) | 0.9782 |
| violin | 0.35 | **0.40** (+0.05) | 0.9283 |

**关键发现与诊断**：

1. **Singer 漏检问题已解决**：singer 最优阈值降至 0.30，验证 F1 达 0.988。在降阈值 + 自适应判定的双重保障下，singer 检测应当显著改善。

2. **模型顶尖能力被高估**：在各自最优阈值下，所有类别均取得 >0.93 的 F1，但这是在验证集上"查表最优"的结果，不代表真实泛化能力。实际推理中会低于这个值。

3. **三个薄弱环节浮现**：
   - **flute (0.718)**：数据量最少之一（225 val）、频段与人声/小提琴重叠、缺乏锐利音头
   - **violin (0.750)** 和 **cello (0.751)**：同为弓弦乐器，谐波结构相似，区分困难
   - **piano (0.785)**：数据量虽大（2897），但动态范围极大（ppp→fff），模型难以覆盖

4. **数据量不是决定因素**：
   - Synthesizer（仅 366 train / 88 val）却能达 0.849 F1——因为音色极具辨识度
   - Flute（947 train / 225 val）只有 0.718——因为音色易与其他类混淆

**下一步优化方向**：

1. **相位特征 (Modgd-gram)**：当前仅使用幅度谱（Mel + MFCC），丢弃了相位信息。Modgd-gram 是改进群延迟函数，能捕捉乐器的起始瞬态（attack transient）。这对 flute/violin/cello 这类无冲击性音头、靠谐波分布区分的乐器特别有效。

2. **音高感知特征**：小提琴、大提琴、长笛的音高分布不同，但当前模型不知道"音高"概念。加入 CQT（Constant-Q Transform）或 Chroma 特征可以显式告诉模型当前在什么音高区域发声。

3. **类别条件增强**：对薄弱类（flute、violin）在 Mixup 中额外保留更多原始样本比例（降低 λ 上限），防止被其他乐器"淹没"。

4. **Focal Modulation**：2025 年论文建议替代自注意力的方案，对短序列（当前 33 步）可能更有效，且计算量更低。

5. **数据层面**：flute 验证集仅 225 样本（约 2%），可以针对性增加 flute 在混合数据中的出现频率。

---

### 版本: VER3.3_PitchStructured (数据+训练+推理三管齐下)

**目标**：突破 84% F1 平台期，不改变模型架构（保持 TransformerClassifier 不变），从数据分布、训练策略、推理后处理三个层面针对性修复：

1. **Flute/Violin/Cello 漏检严重**（0.718/0.750/0.751）
2. **短促音信号被平均池化稀释**（`mean(dim=1)` 让瞬态消失）
3. **钢琴低音误判为电贝斯**（Canon 纯钢琴版出现幻觉）

#### 修改点

**① 数据生成器 —— 薄弱类频率提升** (`data/build_mixed_dataset.py`)
- 在 `class_weights`（sqrt 逆频率）后施加 Boost 乘数：
  - Flute: **×3.0**（成为最高权重，目标将其验证占比从 ~2% 提至 ~6%）
  - Violin: **×2.0**（提升出现率，帮助区分与 cello 的谐波混淆）
  - Cello: **×1.5**（适度提升）
- 采样率估算（原始数据分布：flute ~2.7%, violin ~6.5%, cello ~10% → boost 后预期：flute ~6-8%, violin ~10-12%, cello ~12-15%）
- 总数保持 15,000 不变，被压缩的份额来自电贝斯和钢琴的冗余出现

**② Conditioned Mixup** (`src/train.py`)
- 当 batch 中包含 cello/flute/violin 任一薄弱类时：
  - Mixup 概率从 **50% 降至 25%**
  - Mixup alpha 从 **0.2 降至 0.08**（Beta(0.08, 0.08) 让 λ 几乎总是接近 0 或 1，混合样本中主样本占绝对主导，薄弱类特征不被稀释）
- 其他 batch 保持原有策略，不影响已优类别的训练效率

**③ Max+Mean 混合池化** (`src/model.py`)
- 输出头从纯 `mean(dim=1)` 改为 `0.7 × mean + 0.3 × max`
- 意图：短促音（打击乐器音头、短音 flute 装饰音）在时间轴上可能只占 1-2 帧，在 33 帧的全局平均池化中被严重稀释。Max 池化保留序列上最强烈的响应——如果某个时间步对一个乐器有高置信度激活，max 路径直接保留这个信号。
- Mean 仍然为主（0.7），保持对持续性乐器的稳定判别；Max 约 0.3，作为"瞬态检测"辅助通道

**④ 频段门控** (`test/infer.py`)
- 推理后处理规则：当 `piano 概率 > electric bass 概率` 且 `piano 概率 > 0.2` 时，将该窗口的 bass 归零
- 经验发现：Canon 钢琴曲中检出 bass 的窗口，piano 概率往往极低（~0.03-0.07），说明并非钢琴低音混淆而是模型在不确定区域的通用幻觉。新规则：如果模型认为钢琴比电贝斯更可信（piano > bass），则 bass 的响应更像是低音泛音而非真实 bass

**⑤ 推理平滑窗口下调** (`src/config.py`)
- `INFER_SMOOTH_WINDOW`: 5 → 3（从 2.5s 平滑降至 1.5s）
- 降低短促音信号被相邻静音窗口摊薄的程度

**⑥ 推理详细日志** (`test/infer.py`)
- 每次推理额外输出 `{filename}_probs.csv`，包含每窗口时间戳和各乐器原始概率
- 位置：`output/VER3.3_PitchStructured/` 目录下，与 PNG 图表同目录
- 格式：CSV 头 `window_start,acoustic guitar,cello,...,violin`，每行一个窗口

#### 不做的改动
- 不改模型架构（保持 `TransformerClassifier` 3.06M 参数）
- 不改学习率、调度器、Loss 函数
- 不新增相位/音高特征（这些留到 VER3.4+）

**最终模型性能评价 (200 Epochs, full training)**：

| 指标 | 数值 |
|------|------|
| 最佳 Val F1 | **86.03%** (epoch 179) |
| 最终 Train F1 | 88.56% |
| 最终 Val F1 | 85.63% |
| 训练集样本 | 15,000 (80/20 split, boosted rare classes) |

**逐类 F1 (epoch 179 最佳模型, 全局阈值 0.5, 未用自适应阈值)**：

| 等级 | 乐器 | F1 | Δ vs VER3.2 |
|------|------|-----|-------------|
| 🟢 优秀 | drum set | **0.950** | +0.006 |
| 🟢 优秀 | singer | **0.931** | -0.003 |
| 🟢 优秀 | electric guitar | **0.896** | +0.027 ✅ |
| 🟢 优秀 | electric bass | **0.887** | +0.039 ✅ |
| 🟡 良好 | synthesizer | 0.839 | -0.010 |
| 🟡 良好 | piano | **0.805** | +0.020 ✅ |
| 🟡 良好 | cello | **0.792** | +0.041 ✅ |
| 🔴 薄弱 | flute | 0.728 | +0.010 |
| 🔴 薄弱 | violin | 0.730 | -0.020 |
| 🔴 薄弱 | acoustic guitar | 0.873 | -0.028 |

**关键发现**：

1. **VER3.3 改进有效 +1.4% absolute**（84.63% → 86.03%），主要贡献来自：
   - Cello +4.1% — 数据 boost + conditioned mixup 效果显著
   - Electric bass +3.9% — Max+Mean 池化强化了低频瞬态信号
   - Piano +2.0% — 混合池化帮助钢琴动态范围覆盖更广
   - Electric guitar +2.7% — 同受益于池化改进

2. **Flute (0.728) 问题未解决**：数据量从 225→253 val，3x boost 未显著提升 F1。说明 flute 的混淆本质是**特征混淆**（与 singer 频段重叠），而非数据量问题。需要相位特征。

3. **Violin (0.730) 轻微倒退**：虽然数据 boost 了 2x，但 violin 与 acoustic guitar 的谐波结构相似性导致 confusion。Conditioned mixup 没有区分能力，只能保留现有特征。

4. **Acoustic guitar (0.873) 下降**：electric guitar 提升的同时 acoustic guitar 下降——模型在两者间有 trade-off。数据 boost 使更多注意力分配到了薄弱类，压缩了 ac.guitar 的容量。

5. **新阈值分布更均匀**：训练后自适应阈值计算显示大多数字段最优阈值为 0.4（VER3.2 为 0.3-0.55），说明 VER3.3 模型输出更"谨慎"。

**推理幻觉修复（后处理门控）**：

在推理阶段添加了频段门控规则（`test/infer.py`）：

| 门控 | 触发条件 | 解决的问题 |
|------|---------|-----------|
| piano → e.bass | piano > e.bass AND piano > 0.2 | 钢琴低音被误判为电贝斯 |
| piano → e.guitar | piano > e.guitar AND piano > 0.2 | 钢琴中频触发失真电吉他的谐波模板 |
| piano → violin | piano > violin AND piano > 0.2 | 钢琴高音泛音触发小提琴 |
| singer → violin | singer > violin AND singer > 0.2 | 人声泛音/颤音触发小提琴 |
| e.guitar → violin | e.guitar > violin AND e.guitar > 0.2 | 电吉他失真谐波与 violin 频段重叠 |

实测效果（VER3.3 模型 + 门控后处理）：

| 歌曲 | 实际乐器 | 修正前幻觉 | 修正后 |
|------|---------|-----------|--------|
| 夜の向日葵（纯钢琴） | piano | e.guitar 112窗, violin 64窗 | e.guitar **37窗**, violin **16窗** |
| 星座になれたら（吉他+贝斯+鼓+人声） | e.bass, e.guitar, drums, singer | violin 48窗 | violin **2窗** |
| Variations On The Canon（纯钢琴） | piano | e.bass 62窗 | e.bass **7窗** |

**下一步优化方向（VER3.4 及以后）**：

当前 86.03% F1 已接近 Mel+MFCC 幅度谱特征的信息上限。要突破 90% 需要：

1. **相位特征 (Modgd-gram)**：加入改进群延迟函数，捕捉起始瞬态。这对 flute/violin/cello 这类无冲击性音头、靠谐波分布区分的乐器特别有效。预计将输入通道从 141 扩展到 141+128=269。

2. **音高感知特征 (CQT/Chroma)**：告知模型当前音高区域，帮助区分小提琴（高音区）vs 大提琴（低音区）、长笛（中高音区）vs 女声。

3. **声学吉他 vs 电吉他区分**：当前模型将大量 acoustic guitar 误判为 electric guitar（星座曲中 0/511 检出 ac.guitar 但 373/511 检出 e.guitar），需要冲击响应特征或包络特征来区分两者。

4. **类别条件后处理**：当前门控规则是硬编码的，可以改为从数据中学到的乐器共现概率矩阵，使抑制更精准。


### 版本: VER3.4_ModgdPhase (相位特征突破)

**目标**：突破 86% F1 平台期。核心思路：在 Mel+MFCC 幅度谱基础上加入 Modgd-gram（修正群延迟相位特征），让模型同时感知幅度和相位信息，解决 flute/violin/cello 等谐波重叠类区分困难的问题。

**输入特征变更 (重要)**：

- 原有：Mel(128) + MFCC(13) = **141 通道**
- 新增：Modgd-gram(128, Mel-scaled) = **128 通道**
- 现在：**269 通道** — 模型首次接入相位信息

**模型结构变更**：

- **CNN 骨干 (不变)**：init_conv(1→32) → ResidualBlock(32→64) + Attention(64, r=2) → ResidualBlock(64→128) + Attention(128, r=4)
- **[新版] 频率自适应池化 (新增)**：`AdaptiveAvgPool2d((36, None))` — 在 freq_proj 前标准化频率维度高度为 36，支持任意输入特征数
- **频率卷积投影 (不变)**：Conv2d(128→256, kernel=(36,1)) → BN → ReLU
- **时序编码器 (不变)**：PositionalEncoding(256) → TransformerEncoder(2层, 8头, d_model=256, FFN=1024)
- **输出头 (不变)**：0.7×mean + 0.3×max → Dropout(0.3) → Linear(256, 10)
- **总参数量：3,062,762**（未变 — AdaptiveAvgPool2d 无参数）

**Modgd-gram 技术细节**：

- 定义：`τ(k) = (X_R·Y_R + X_I·Y_I) / |S(k)|^{2γ}`，其中 X 是 STFT，Y 是加斜坡信号 n·x[n] 的 STFT
- 平滑：3 抽头移动平均平滑幅度谱 |S(k)| 用于分母
- Gamma 钳位：γ=0.3，防止分母过小导致数值不稳定
- 逐帧归一化：每帧独立缩放到 [0, 1]
- Mel 映射：1025 STFT bins → 128 Mel 频带（与 MelSpectrogram 一致）

**修改的文件**：

| 文件 | 修改内容 |
|------|---------|
| `src/config.py` | MODEL_VERSION→VER3.4_ModgdPhase, 新增 N_MODGD=128 |
| `data/dataset.py` | 新增 `compute_modgd()` 方法、MelScale 变换、特征合并从 141→269 通道 |
| `src/model.py` | 新增 `freq_adaptive_pool = AdaptiveAvgPool2d((36, None))` 在 freq_proj 前 |
| `test/infer.py` | 新增 `compute_modgd()` 内联函数、特征合并 141→269、移除 piano→violin 门控 |

**不做的改动**：

- 不改训练超参数（LR=5e-4, FocalLoss gamma=2.0, Mixup, etc.）
- 不改数据生成器（`build_mixed_dataset.py` 无需修改）
- 不改 Loss 函数或优化器

**预期**：Modgd 提供相位信息，使模型能区分谐波结构相似但相位响应不同的乐器（flute vs violin, cello vs piano）。预期 F1 从 86.03% 提升至 90%+。剩余差距计划通过 VER3.5 的 CQT/Chroma 音高感知特征补足。

---

## 待办（训练完成后更新）

- [x] 记录最佳 Val F1 → **84.73%** (epoch 177), 最终 Val F1 收敛至 ~84.1%
- [x] 逐类 F1 (epoch 177): drum set 0.940, singer 0.945, e.bass 0.852, e.guitar 0.877, ac.guitar 0.900, cello 0.767, piano 0.773, synth 0.862, flute 0.667, violin 0.747
- [x] flute F1 0.67 (VER3.3 0.73) — Modgd 未显著改善 flute 混淆
- [x] **标准对照集评估**: Micro F1 **0.539** — 暴露出合成训练集与真实混音的巨大领域差距！Recall 仅 0.39, Precision 0.86
- [x] 核心问题: 合成混音(peak归一化)导致模型输出的概率无法迁移到真实录音 → VER3.5 修复

---

### 版本: VER3.5_RealMix (领域差距弥合)

**目标**: 解决 VER3.4 在标准对照集上暴露的严重领域差距问题 (Val F1 0.84 → GT F1 0.54)。

**根因分析** (来自 6 首标准对照歌曲评估):
1. **Recall 灾难 (0.39)**: 模型在真实混音上严重欠检测 — 合成混音的 peak 归一化使训练样本中乐器音量不真实地一致
2. **跨域降置信**: 模型在真实混音上输出概率系统性偏低 — 合成数据欠缺真实录音的动态范围和频谱平衡
3. **典型幻觉**: electric guitar 在 4/6 首中幻现 (古典钢琴三重奏里检出电吉他!), electric bass 在 3/6 首中幻现
4. **严重漏检**: drum set Recall=0.12, singer Recall=0.13, synthesizer Recall=0.03

**修改内容**:

| 文件 | 修改内容 |
|------|---------|
| `src/model.py` | 新增 `input_norm = InstanceNorm2d(1, affine=True)` 在 init_conv 前, 对每样本独立归一化，消除整体音量和频谱斜度差异 |
| `data/build_mixed_dataset.py` | (1) 用 RMS 归一化替代 peak 归一化，保留乐器间相对响度 (2) 增益范围 0.6~1.0 → 0.25~1.0 (12dB动态) (3) 目标 RMS 随机化: 0.08~0.25 (乐器) / 0.03~0.12 (背景) (4) tanh 软削顶 |
| `src/config.py` | MODEL_VERSION → VER3.5_RealMix, 新增 FEATURE_STATS_PATH |
| `src/train.py` | 恢复 checkpoint 后 `scheduler.T_max = config.EPOCHS` 修复 smoke test 污染 |

**核心创新**:
1. **InstanceNorm2d 输入归一化**: 每个 spectrogram 独立归一化到 0 均值 1 方差，使模型对整体音量、频谱倾斜不敏感。这是弥合合成/真实混音差距的最关键一步。
2. **真实化混音增强**: 不再让所有乐器"平等响亮"，而是模拟真实录音中某些乐器比其他的轻 12dB 的情况。

**预期**: InstanceNorm + 真实化混音应显著提升召回率（目标从 0.39 → 0.65+），同时降低跨域幻觉。

**不做的改动**: 不改架构 (Transformer 不变), 不改 Loss, 不改超参数。

---

### 版本: VER4.0_BinaryEnsemble (逐乐器二分类集成)

**目标**: 放弃单模型多标签分类，改用 10 个独立的逐乐器二分类器。每个模型只学"这个乐器有没有"，从根本上解决类别不平衡和稀有类漏检问题。

**核心架构变更**:
- 旧方案: 一个 TransformerClassifier (3.06M 参数) 输出 10 维 logits → Focal Loss
- 新方案: 10 个 BinaryInstrumentClassifier (72K 参数/个 = 0.73M 总参数，~4x 更小)

**BinaryInstrumentClassifier 结构**:
- `Conv2d(1→16) → BN → ReLU → ResidualBlock(16→32, stride=2) → ResidualBlock(32→64, stride=2) → AdaptiveAvgPool2d(1) → Dropout(0.3) → Linear(64, 1)`
- 参数量: 72,497 / 模型
- 训练: BCEWithLogitsLoss, Adam(lr=1e-3), 30 epochs

**训练策略 —— 三阶段方案**:

1. **纯净音轨训练 (Clean Stem Training)**:
   - 从 MedleyDB 提取 4185 个 3 秒纯净音轨片段 (每类 120-720)
   - 每个二分类器用正样本(该类) + 等量负样本(所有其他类) 1:1 平衡训练
   - 80/20 切分验证，每 epoch ~1 秒

2. **逐类验证 (Per-Instrument Validation)**:
   - 在纯净音轨验证集上所有 10 类 F1 > 0.94
   - drum set 和 singer 达到 1.000

3. **集成推理 (Ensemble Inference)**:
   - 10 个模型并行推理，sigmoid 概率合并
   - 逐窗 3 帧平滑
   - 按类独立阈值

**纯净音轨验证结果 (30 Epochs)**:

| 乐器 | Val F1 |
|------|--------|
| acoustic guitar | **0.969** |
| cello | **0.947** |
| drum set | **1.000** |
| electric bass | **0.984** |
| electric guitar | **0.997** |
| flute | **0.985** |
| piano | **0.987** |
| singer | **1.000** |
| synthesizer | **0.977** |
| violin | **0.959** |

**标准对照集评估 (6 首真实混音)**:

| 歌曲 | VER3.5 | VER4.0 | Δ |
|------|--------|--------|---|
| medleydb_not_for_nothing | 0.440 | **0.589** | +34% |
| medleydb_piano_trio | 0.636 | **0.767** | +21% |
| medleydb_violin_sonata | 0.676 | **0.715** | +6% |
| medleydb_vivaldi | 0.426 | **0.602** | +41% |
| moisesdb_electronic | 0.470 | **0.741** | +58% |
| moisesdb_sunspot | 0.436 | **0.666** | +53% |
| **Global** | **0.527** | **~0.68** | **+29%** |

**关键成功因素**:
1. **每类独立学习**: 合成器从多标签方案的 F1 0.0 跃升至 0.977 (纯净音轨验证)，真实混音中也从几乎不可检测到能检出
2. **平衡训练**: 1:1 正负采样消除了类别不平衡——每类都在同等数据量下训练
3. **纯净音轨作为锚点**: 模型先学会"这乐器的纯音色长什么样"，再在混音中泛化
4. **模块化**: 任何乐器效果不好可单独重训，不影响其他 9 个

**待改进**:
1. 部分乐器 (drum set, piano, electric bass) 在真实混音中 recall 仍偏低——需要微调阶段用真实混音做 domain adaptation
2. 阈值需要针对真实混音重新校准（当前用纯净音轨最优阈值不一定适配混音场景）
3. 评估数据有限 (仅 6 首对照歌)

**总参数量**: 0.73M (vs VER3.5 的 3.06M，缩小 76%)
**总训练时间**: ~5 分钟 (10 类 × 30 秒)
**模型存储**: 2.9 MB (vs 37 MB checkpoint，缩小 92%)

---

### VER4.0_BinaryEnsemble — Stage 2 真实混音微调 + MoisesDB 扩充数据

**目标**: 解决合成/真实混音领域差距 (Ground Truth F1 从 0.54→0.79)，补充合成器训练数据。

**Stage 2 微调**:
- 在 `muse_real_mixed_dataset` (MedleyDB 74 首歌, ~25K 窗口) 上做微调
- 1:1 平衡正负采样，正样本=该类活跃窗口，负样本=其他乐器活跃但该类不活跃的窗口
- 15 epochs + 5 epochs 困难负样本挖掘 (HNM)
- BCEWithLogitsLoss, Adam(lr=1e-4), CosineAnnealingLR
- 混入 20% 纯净音轨防止灾难性遗忘

**MoisesDB 数据扩充 (2026-05-18)**:
- 新增 dataset: `muse_real_mixed_dataset_combined` — MedleyDB + MoisesDB 合并
- 整合 240 首 MoisesDB 真实歌曲 (~204s/首)，各分轨叠加为全混音 + RMS 活动检测
- 最终: **20,000 训练窗口 + 22,752 验证窗口** (sqrt 平衡采样)
- MoisesDB 贡献大量合成器 (2681 vs 原来 36) 和长笛数据

**Stage 2 微调结果 (10 乐器, 合并数据集)**:

| 乐器 | Best Val F1 |
|------|:----------:|
| acoustic guitar | 0.932 |
| cello | **0.980** |
| drum set | 0.969 |
| electric bass | 0.959 |
| electric guitar | 0.907 |
| flute | **0.983** |
| piano | 0.898 |
| singer | 0.939 |
| **synthesizer** | **0.814** (from 0.255) |
| violin | 0.938 |

**标准对照集评估 (6 首真实混音歌曲, 跨曲汇总)**:

| 乐器 | F1 | TP | FP | FN | 分析 |
|------|:--:|:--:|:--:|:--:|------|
| drum set | **0.979** | 1370 | 32 | 27 | 优秀,几乎无幻觉 |
| violin | **0.954** | 2726 | 224 | 38 | 优秀 |
| singer | **0.915** | 1127 | 152 | 57 | 优秀 |
| piano | 0.810 | 2290 | 1033 | 43 | FP偏多(低音泛音触发) |
| electric bass | 0.793 | 978 | 508 | 3 | FP因钢琴低音共混 |
| acoustic guitar | 0.792 | 847 | 404 | 42 | FP来自electric guitar混淆 |
| cello | 0.687 | 1144 | 1038 | 5 | FP严重(与violin混淆) |
| electric guitar | 0.655 | 505 | 503 | 29 | FP/FN均衡偏低 |
| synthesizer | **0.411** | 448 | 1285 | 0 | FP大幅削减仍有1285(之前2604) |
| flute | **0.179** | 72 | 89 | 573 | 严重漏检, 混音中被掩盖 |
| **Global Micro F1** | **0.791** | 11507 | 5268 | 817 | +50% vs VER3.5基线(0.527) |

**与 VER3.5 对比**:
- 全局 F1: 0.527 → **0.791** (+50%)
- synthesizer: 几乎不可检出 → 0.411 (FP从2604降至1285)
- 所有 6 首歌曲均有显著提升

**当前已知问题**:

1. **Flute 严重漏检 (F1=0.179, FN=573)**: 69% 的 flute 窗口未被检出。根源: flute 在混音中能量偏小，且频段与 singer/violin 重叠。模型被其他强能量乐器 (drums、piano) 压制。需要音高感知特征 (CQT) 或专门数据增强。

2. **Cello 误报 (F1=0.687, FP=1038)**: FP 是 TP 的 90%。cello/violin 同为弓弦乐器，谐波结构高度相似。单靠 Mel+MFCC+Modgd 幅度谱难以区分。

3. **Synthesizer FP 仍偏高 (FP=1285)**: 虽然从原来的 2604 降下来了，但 FP 仍是 TP 的 2.9 倍。MoisesDB 数据帮助显著 (F1 0.255→0.411)，但 MoisesDB 的 synth 音色偏音乐化(lead/pad)，与背景噪音混淆。

4. **Piano FP 多 (FP=1033)**: 钢琴低音泛音触发 electric bass，高音泛音触发 violin。频段门控规则已缓解部分，但规则硬编码有时误杀真实检出。

5. **Electric/acoustic guitar 混淆**: 两把吉他音色在混音中互相干扰。clean electric guitar 和 acoustic guitar 的前级特征过于相似。

6. **后处理门控依赖经验规则**: Co-occurrence 和频段门控阈值是手动调的，可能在新歌曲中产生意外抑制。

**下一步方向 (音轨分离)**:
当前模型对吉他 (acoustic + electric + bass) 的综合检测能力已满足作为音轨分离条件信号的需求。建议进入 source_separation 模块开发。

---
