# VER4.0_BinaryEnsemble 模型架构与接口文档

版本状态: **VER4.0** (正式版, 2026-05-18)
代码路径: `d:\program_project\MUSE\instrument_recognition\`

---

## 一、架构总览

采用 **10 个独立二分类器集成** 方案，每个乐器一个 BinaryInstrumentClassifier。
每个模型只判断"这个乐器是否有在响"，总参数量 0.73M（比单 Transformer 方案 3.06M 小 76%）。

```
输入音频 (16/24-bit PCM, 22050Hz)
  │
  ├──→ acoustic_guitar.pth  ─→ sigmoid → [0~1]
  ├──→ cello.pth            ─→ sigmoid → [0~1]
  ├──→ drum_set.pth         ─→ sigmoid → [0~1]
  ├──→ electric_bass.pth    ─→ sigmoid → [0~1]
  ├──→ electric_guitar.pth  ─→ sigmoid → [0~1]
  ├──→ flute.pth            ─→ sigmoid → [0~1]
  ├──→ piano.pth            ─→ sigmoid → [0~1]
  ├──→ singer.pth           ─→ sigmoid → [0~1]
  ├──→ synthesizer.pth      ─→ sigmoid → [0~1]
  └──→ violin.pth           ─→ sigmoid → [0~1]
          │
          ▼
   后处理 (平滑 + 门控 + 去孤立帧)
          │
          ▼
   10 维多标签检出结果 [binary]
```

---

## 二、模型结构

### BinaryInstrumentClassifier (`src/bmodel.py`)

```python
BinaryInstrumentClassifier(
  (net): Sequential(
    (0): Conv2d(1, 16, kernel=3, padding=1) + BatchNorm2d + ReLU
    (1): ResidualBlock(16 → 32, stride=2)  # 下采样
    (2): ResidualBlock(32 → 64, stride=2)  # 下采样
    (3): AdaptiveAvgPool2d(1)               # 全局平均池化
  )
  (head): Sequential(
    (0): Dropout(0.3)
    (1): Linear(64 → 1)                     # 二分类输出
  )
)
```

### ResidualBlock

```python
ResidualBlock(in_ch, out_ch, stride=1):
  conv1(→BN→ReLU) → conv2(→BN) → + shortcut → ReLU
```

- 参数量: **72,497** 个 / 模型 × 10 = **724,970** 总参数
- 存储: ~290 KB / 模型, 共 2.9 MB

### 输入

- 形状: `[B, 1, 269, T]` — (batch, channel, freq, time)
- 269 = Mel(128) + MFCC(13) + Modgd(128)
- T 取决于 3 秒音频 @ 22050Hz, hop_length=512 → ~129 帧

### 特征提取流程 (`src/btrain.py`, `test/binfer.py`)

```python
# 1. Mel-spectrogram (128 bins)
mel = MelSpectrogram(SR, n_mels=128, n_fft=2048, hop=512) → AmplitudeToDB

# 2. MFCC (13 coeffs)
mfcc = MFCC(SR, n_mfcc=13, melkwargs={n_fft:2048, hop:512, n_mels:128})

# 3. Modgd-gram (128 bins, phase features)
modgd = compute_modgd(audio, gamma=0.3)
# τ(k) = (X_R·Y_R + X_I·Y_I) / |S(k)|^{2γ}
# 3-tap smooth, gamma=0.3, per-frame normalize [0,1], Mel scaled

# 合并
features = torch.cat([mel, mfcc, modgd], dim=1)  # [B, 269, T]
```

---

## 三、训练流程

### Stage 1: 纯净音轨预训练 (`src/btrain.py`)

| 参数 | 值 |
|------|----|
| 数据 | MedleyDB 4185 个纯净音轨 (120-720/类) |
| 正负采样 | 1:1 平衡 |
| 损失 | BCEWithLogitsLoss |
| 优化器 | Adam(lr=1e-3) |
| 轮数 | 30 epochs |
| 时间 | ~1s/epoch |
| 验证 | Clean stem 验证集 F1 > 0.94 |

```bash
env/python.exe -m src.btrain --instrument drum_set --epochs 30
```

### Stage 2: 真实混音微调 (`src/bfinetune.py`)

| 参数 | 值 |
|------|----|
| 数据 | MedleyDB + MoisesDB 合成混音 (20000 训练 + 22752 验证) |
| 加载 | Stage 1 预训练权重 |
| 正样本 | 该类在混音中活跃的窗口 |
| 负样本 | 其他乐器活跃但该类不活跃的窗口 |
| 混合 | 20% 纯净音轨防止遗忘 |
| 损失 | BCEWithLogitsLoss |
| 优化器 | Adam(lr=1e-4) + CosineAnnealingLR |
| 轮数 | 15 epochs + 5 epochs HNM |

```bash
# 单乐器
env/python.exe -m src.bfinetune --instrument drum_set --epochs 15

# 全部乐器
env/python.exe -m src.finetune_all
```

### 困难负样本挖掘 (HNM)

Stage 2 的最后 5 轮扫描训练集，找出模型误报 (FP) 最严重的窗口，将它们加入负样本池重新训练。

---

## 四、推理接口

### 1. CLI 推理 (`test/binfer_cli.py`)

```bash
# 扫描 music/ 目录下所有音频
env/python.exe test/binfer_cli.py

# 指定单文件
env/python.exe test/binfer_cli.py 路径/音频.wav
```

输出:
- PNG 甘特图 → `output/VER4.0_BinaryEnsemble/{filename}.png`
- 概率 CSV → `output/VER4.0_BinaryEnsemble/{filename}_probs.csv`

### 2. Python API (`test/binfer.py`)

```python
from binfer import load_ensemble, predict_file, post_process

device = torch.device("cuda")
models = load_ensemble(device)

# 预测: 返回 [N_windows, 10] 概率 + 乐器名列表
probs, inst_names = predict_file("audio.wav", models, device)

# 后处理: 平滑 + 门控 + 去孤立帧
thresholds = [0.35, 0.50, ...]  # 每类独立阈值
cleaned, binary = post_process(probs, inst_names,
                               thresholds=thresholds,
                               min_active_frames=2)
```

### 3. 标准集评估 (`test/beval.py`)

```bash
env/python.exe test/beval.py
```

加载 6 首标准对照歌曲 (`data/ground_truth/`)，输出逐曲和跨曲汇总 F1。

### 后处理流水线 (`post_process`)

1. **阈值二值化**: 每类独立阈值 (默认 acoustic_guitar=0.35, cello=0.50, ...)
2. **共现门控**: 从训练数据预计算 10×10 条件概率矩阵 P(col|row)。若 P(A|B) < 0.03 且 P(B|A) < 0.03，压制低置信度乐器
3. **频段门控**: 已知互斥乐器对 (piano→e.bass, singer→violin, e.guitar→violin 等)，置信度高者压制低者
4. **去孤立帧**: 少于 2 帧连续的激活被清除
5. **概率重建**: 未激活窗口的概率乘以 0.3 衰减

### 默认阈值 (`test/binfer_cli.py`)

| 乐器 | 阈值 |
|------|:----:|
| acoustic guitar | 0.35 |
| cello | 0.50 |
| drum set | 0.40 |
| electric bass | 0.30 |
| electric guitar | 0.50 |
| flute | 0.40 |
| piano | 0.40 |
| singer | 0.35 |
| synthesizer | 0.65 |
| violin | 0.30 |

---

## 五、数据集

### Clean Stems (`data/clean_stems/`)
- MedleyDB 提取的 4185 个 3s 纯净音轨
- 每类 120-720 个片段
- 作用: Stage 1 预训练

### Real Mixes (`data/muse_real_mixed_dataset_combined/`)
- MedleyDB: 74 首歌, ~25K 窗口
- MoisesDB: 240 首歌 + 叠加混音, ~20K 窗口
- 合并后: **20000 训练 + 22752 验证**
- 版本: 2026-05-18
- 总计: 5.27 GB

### Ground Truth (`data/ground_truth/`)
6 首标准对照歌曲:
- `medleydb_not_for_nothing`, `medleydb_piano_trio`, `medleydb_violin_sonata`
- `medleydb_vivaldi`, `moisesdb_electronic`, `moisesdb_sunspot`

---

## 六、文件清单

| 文件 | 用途 |
|------|------|
| `src/bmodel.py` | BinaryInstrumentClassifier 模型定义 |
| `src/btrain.py` | Stage 1 纯净音轨训练 |
| `src/bfinetune.py` | Stage 2 真实混音微调 + HNM |
| `src/finetune_all.py` | 批量运行所有乐器 Stage 2 |
| `src/train_all.py` | 批量运行所有乐器 Stage 1 |
| `src/config.py` | 全局配置 (版本号、采样率等) |
| `test/binfer.py` | 集成推理引擎 (加载 + 预测 + 后处理) |
| `test/binfer_cli.py` | CLI 推理入口 (甘特图 + CSV) |
| `test/beval.py` | 标准对照集评估 |
| `data/build_clean_stems.py` | 从 MedleyDB 提取纯净音轨 |
| `data/build_medleydb_real_mixes.py` | MedleyDB 真实混音构建 |
| `data/build_moisesdb_real_mixes.py` | MoisesDB 真实混音构建 |
| `data/combine_real_mixes.py` | 两个数据集合并脚本 |
| `model/binary/*.pth` | 10 个训练好的二分类权重 |
| `model/log/bfinetune_*.log` | Stage 2 训练日志 |
| `model/log/btrain_*.log` | Stage 1 训练日志 |
| `markdown/model_log.md` | 完整研发日志 |
| `markdown/recall_explanation.md` | Recall/Precision/F1 说明 |
| `CLAUDE.md` | Claude Code 项目规则 |

---

## 七、已知问题

1. **Flute 漏检 (F1=0.179)**: 混音中被其他乐器能量压制，需要 CQT/Chroma 特征或更多数据
2. **Cello/Violin 混淆 (F1=0.687/0.954)**: 弓弦乐器谐波相似，需要相位包络特征
3. **Synthesizer FP (F1=0.411, FP=1285)**: FP 仍偏高，需要更多样化的 synth 训练数据
4. **Piano 低音误触发 Bass**: 后处理门控规则已缓解但未根治
5. **门控规则硬编码**: Co-occurrence 阈值和频段门控需要从数据中学习
