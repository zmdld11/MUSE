# VER2.1 评测闭环设计（AMT 四层级方案 V2）

> 日期：2026-07-31 | 状态：✅ 已实现，评测运行中
> 上一篇设计：[2026-07-06-amt-redesign.md](2026-07-06-amt-redesign.md)

---

## 一、背景：为什么重定方案

上一版（VER2.0，2026-07-06）设计了五层管线架构并落地了代码，但有一个致命缺失——**没有评测闭环**。所有层级都没有量化指标，无法回答"当前转录到底多准"、"改模型前后进步没有"。同时项目文件夹积累了 48 个 `.superpowers/sdd` 任务文件和多份过时设计稿，需要归档收敛。

本方案回答三个问题：
1. 遵守 AMT 四任务层级吗？—— **遵守**（帧级→音符级→流级→记谱级）
2. 每层要训练模型吗？—— **不需要**，一个转录模型 + 规则后处理（详见第三节）
3. 每层用什么指标、宽容度多少？—— 论文标准 + 记谱级自研指标（详见第四节）

## 二、目标收敛

| 优先级 | 目标 | 状态 |
|--------|------|------|
| P0 | 把**音高**识别准确 | 评测闭环衡量（Note F1） |
| P0 | 把**时值**识别准确 | 评测闭环衡量（Note-with-offset F1） |
| P1 | 记谱级可读乐谱 | 自研三指标衡量 |
| 延后 | 力度 velocity | 明确不做 |
| 延后 | 演奏技巧 | 明确不做 |
| 延后 | 跨数据集泛化 | 先保证测试集达标 |

## 三、四层级 vs 模型策略

**结论：不按层级训练多个模型。** 帧级+音符级共用一个转录模型（Onsets and Frames 架构，输出 onset/frame/contour 三张概率图），层级之间用后处理算法连接。

| AMT 层级 | 技术方案 | 需要训练模型？ | 现状 |
|----------|---------|--------------|------|
| **帧级** (MPE) | 转录模型输出 frame probs → HMM 平滑 → 注册自适应阈值 → 二值化 | ❌ | ✅ 已有 |
| **音符级** (NT) | 二值图连通域标注 → onset 精修 → 谐波过滤 → 合并 | ❌ | ✅ 已有 |
| **流级** (MPS) | **htdemucs_6s 分轨绕过**（分离头自带乐器标签） | ❌ | ✅ 已绕过 |
| **记谱级** | 规则组装（music21 + MuseScore CLI） | ❌ | ⚠️ 最弱，需指标 |

**为什么流级能绕过**：htdemucs_6s 输出 6 个语义轨道（bass/drums/guitar/piano/vocals/other），每个轨道自带乐器标签，天然完成了多乐器分流，无需单独的流级模型。

**评测基线双轨**：
- `VER2.0_Bootstrap.pth`：自训 OnsetsAndFrames（2026-07-10 训练 23 epochs，frame_acc=0.996, onset_f1=0.853）
- `basic-pitch`：ICASSP 2022 预训练模型

## 四、评测指标（宽容度按论文标准）

| 层级 | 指标 | 宽容度 | mir_eval 函数 |
|------|------|--------|--------------|
| 帧级 | Frame P/R/F1 | — | `multipitch.evaluate` |
| 音符级 | Note P/R/F1 | onset ≤ ±50ms | `transcription.precision_recall_f1_overlap(offset_ratio=None)` |
| 音符级 | Note-with-offset P/R/F1 | offset ≤ max(50ms, 20%×符长) | 同上（默认参数） |

**帧级二值化一致性**（重要设计决策）：帧级指标不直接用 0.5 硬阈值（两个模型概率分布差异大），而是复用 `frame_post` 的同一套二值化（HMM 平滑 + 注册自适应阈值），保证帧级与音符级可比、两模型可比。

**时间网格对齐**（踩坑记录）：basic-pitch 内部 hop=256，自训模型 hop=512，GT 用 hop=512。mir_eval.multipitch 要求 ref/est 同一时间网格，需把 est 概率图线性重采样到 GT 帧网格（`_resample_probs_to_grid`）。不重采样会直接 ValueError。

## 五、记谱级指标（自研，学术无标准）

三个可量化子指标（V2.0 待实现）：

1. **记谱匹配率**：音符级匹配上的 TP 中，"量化后的时值类别"也一致的占比（全分/二分/四分/八分…）
2. **小节完整性率**：每个小节拍数是否等于拍号要求（4/4 = 4 拍），衡量小节结构正确性
3. **onset 量化误差**：预测 onset 相对最近拍点的平均偏差，衡量"对齐到网格"好不好

> 依据：论文 [2] (AMT Overview 2019) 明确记谱级"尚无标准量化指标"，是本项目可发论文的差异点。

## 六、评测代码结构（已实现）

```
score_extraction/eval/
├── dataset.py   # GiantMIDI 采样 + render_midi 渲染 + npz 缓存 + GT 提取
├── metrics.py   # mir_eval 封装（帧/音符/offset 三指标）+ 网格对齐
├── eval.py      # CLI 入口: --model ours|basic|both --n 40 --seed 42
└── reports/     # 评测报告 JSON + 日志
```

**数据管线**：
```
data/midi/GiantMIDI-PIano/midis/*.mid (10854 首)
  → 随机采样 N 首 (seed 固定可复现)
  → train/render_midi.py 渲染 (FluidSynth, sr=22050, hop=512)
  → GT: frame_labels 矩阵 + intervals/pitches 列表 (从 MIDI 直接提取, 理论完美)
  → 缓存 eval/cache/{md5}.npz (避免重复渲染)
```

**评测流程**：
```
MIDI → 渲染音频 → 模型转录 (onset/frame probs)
  → frame_post.process_frames → note_post.refine_notes → 音符列表
  → mir_eval 计算三指标 vs GT
聚合 → JSON 报告 (逐首明细 + 总体均值)
```

## 七、环境约定（踩坑记录：之前跑不起来的根因）

| 项 | 值 |
|----|-----|
| **唯一可用的 Python** | `C:\Users\ROG\.conda\envs\score_build\python.exe` (Python 3.10) |
| 关键包 | mir_eval 0.8.2 / basic_pitch / pretty_midi 0.2.11 / torch 2.6+cu124 |
| 禁用 | 全局 `C:\Python314`、`D:\program_project\MUSE\env`（缺包） |

**权重恢复**：`model/VER2.0_Bootstrap.pth.bak2` → 拷回 `VER2.0_Bootstrap.pth`（此前被改名导致自训模型不可用，transcriber 回退 basic-pitch）。

## 八、评测结果（首次全量 + 评测修正）

> ⚠️ **评测修正记录**：初次评测发现 pitch 单位 bug（mir_eval.transcription 的 pitch 参数是 **Hz**，误传了 MIDI 号），导致半音错误被误判为匹配、note_f1 被高估约一倍。修正后真实指标见下。教训记入 memory。

**修正后的 40 首全量**（报告 `eval/reports/report_20260731-194519.json`）：

| 模型 | frame F1 | note F1 | offset F1 | note P/R |
|------|----------|---------|-----------|----------|
| VER2.0_Bootstrap | 0.983 | **0.217** | 0.081 | 0.378 / 0.154 |
| basic-pitch | — | ~0.35 | ~0.086 | — |

**诊断结论**：
1. 帧级指标虚高（0.983），印证论文"帧级不反映感知质量"
2. 真实 note_f1 仅 0.217——音高识别远未达标（目标 >0.8）
3. est/gt 比约 0.37，漏检严重

### 后处理参数 A/B（用修正后评测验证）

曾试图改两个参数，**用 A/B 严格验证后只保留有效的一个**：

| 配置 | P | R | note_f1 |
|------|------|------|---------|
| min_frames=4 + 自适应pct50（旧） | 0.525 | 0.182 | 0.271 |
| **min_frames=2 + 自适应pct50（采纳）** | 0.500 | 0.232 | **0.317** |
| min_frames=2 + 固定0.3（曾误改） | 0.307 | 0.150 | 0.201 |

- **min_frames 4→2 有效**：recall 0.182→0.232，note_f1 0.271→0.317
- **阈值改固定 0.3 有害**：precision 从 0.525 砸到 0.307，recall 反降——已回退为自适应 percentile
- 教训：早期"note_f1 0.421"、"固定阈值更优"都是评测 bug 或单曲观察造成的误判，必须全量 A/B 验证

> 注：A/B 用的是独立逐首累加脚本，与 eval.py 的 note_f1=0.217 略有差异（A/B 未走 process_frames 的完整链），两者趋势一致，正式数字以 eval.py 报告为准。

### 音高能力分析（polyphony）

**能测和弦**：poly≤3 时 recall 与单音持平甚至更高（poly=3 达 41.5%），复调上限约 poly=4，超过 5 音同时响才明显崩。但 94% 的 GT 音符在和弦里，整体 recall 仍受限于检测灵敏度。

## 九、下一步计划

1. **诊断音符级瓶颈**：逐首看 est/gt 数量差、onset 偏移分布、误检/漏检归因
2. **调后处理参数**：阈值、HMM 转移概率、谐波过滤强度（不动模型）
3. **视情况训练**：若两模型都 <60%，改进现有 OaF（换 MAESTRO 数据/训更久），按训练规范走
4. **记谱级指标实现**：五节三指标落地

## 十、归档说明

旧方案已归档到 `markdown/archive/`：
- `2026-07-05-score-extraction-design.md` — VER1.0 设计稿
- `2026-07-06-amt-redesign.md` — VER2.0 五层管线设计稿（本方案继承其架构）
- `amt_review_and_strategy.md` — 早期 AMT 综述与策略
- `.superpowers/sdd/`（48 个任务文件）— 历史开发任务记录
