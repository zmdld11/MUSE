# MUSE — 音轨分离 (Source Separation) 模型研发日志

## 项目概述
基于频域掩码的乐器音轨分离系统。当前目标：从混合音频中分离吉他音轨。

---

### 版本: VER1.0_LightweightUMX

**日期**: 2026-05-19

**设计理念**:
采用 Open-Unmix 频域掩码方法，以极轻量架构（~758K 参数）实现吉他音轨分离。
利用已完成的乐器识别模型（VER4.0_BinaryEnsemble）作为前置门控：先用识别模型检测吉他片段，
无吉他段直接静音，有吉他段才送入分离模型——预计节省 60-70% 推理计算量。

**模型结构**:
- 输入：STFT magnitude (513 bins) — 保留完整频率分辨率
- LayerNorm(513) — 频率轴归一化
- BLSTM(1层, hidden=128, bidirectional=1) — 时序建模，513 → 256
- FC1(256 → 128, ReLU) — 全连接投影
- FC2(128 → 513, Sigmoid) — 输出频率掩码 [0,1]
- 总参数量：758,531

**创新点 / 尝试**:
1. 极致轻量化：1层 BLSTM 替代原始 Open-Unmix 的 3 层，参数量仅 ~758K（vs 原始 ~8M）
2. 复用混合音频相位：仅分离幅度掩码，用原始相位重建，避免相位估计困难
3. 目标乐器聚焦：仅分离吉他（acoustic + electric 合并），降低任务复杂度
4. 乐器识别门控集成：非吉他段跳过分离，减少无效计算
5. 吉他合并策略：不区分 acoustic/electric，统一为单一"guitar"目标

**训练配置**:
- 损失函数：L1 magnitude loss
- 优化器：Adam(lr=3e-4, weight_decay=1e-5)
- 批大小：64
- 调度器：ReduceLROnPlateau(patience=5, factor=0.5)
- 最大轮数：100 (early stopping patience=15)
- 数据增强：随机音量缩放 ±3dB

**待完成**:
- [ ] 数据构建：运行 data/build_guitar_separation_dataset.py
- [ ] 训练：运行 python -m src.train
- [ ] 评估：分离后音频 SDR/SIR 指标
- [ ] 集成测试：完整流水线（乐器识别 + 分离）

**下一步方向**:
1. VER1.1: 增加 BLSTM 层数（1→2层, hidden 128→192）提升分离质量
2. VER1.2: 加入音高感知条件输入（pitch-conditioned mask）
3. VER1.3: 多乐器分离（扩展到 bass, drums, vocals）
4. VER1.4: 相位估计网络（不再复用混合音频相位）
