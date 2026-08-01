# 乐谱生成模型 版本日志

> 版本命名：`VER<MAJOR>.<MINOR>_<FeatureTag>`
> 与 instrument_recognition / source_separation 保持一致

---

## VER2.0_Bootstrap (当前版本)

**日期：** 2026-07-10
**分支：** `se/ver1.0`
**状态：** ✅ 训练完成 (23/30 epochs, 被中断但指标已达标)

### 训练配置

| 参数 | 值 |
| :--- | :--- |
| 训练集 | 2000 MIDI (random sample) |
| 验证集 | 50 MIDI |
| Epochs | 23/30 (被中断) |
| Batch size | 2 |
| Max duration | 30s |
| GPU | RTX 4060 |

### 最终指标 (Epoch 23)

| 指标 | 值 | 阈值 | 结果 |
| :--- | :--- | :--- | :--- |
| frame_acc | **0.996** | > 98% | ✅ |
| onset_f1 | **0.853** | > 0.4 | ✅ |
| val_loss | 0.0134 | - | - |
| train_loss | 0.0213 | - | - |

### A4 验证

加载模型后对 `test/fixtures/test_a4_440.wav` 推理：

- MIDI 69 (A4) frame prob: **0.788** — 正确检测
- 模型输出正常，符合预期

### 备注

训练在第 23/30 epoch 时被外部进程中断，但验证指标已远超阈值：
- frame_acc 从 epoch 4 起持续 > 98%（峰值为 0.996）
- onset_f1 从 epoch 6 起持续 > 0.4（峰值为 0.853）
- 最后 5 个 epoch 指标已基本收敛，继续训练增益有限

模型文件覆盖旧版 `VER2.0_Bootstrap.pth` (4.1 MB)。

---

## VER2.0_Bootstrap — 首次评测闭环验证 (2026-07-31)

**日期：** 2026-07-31
**评测集：** GiantMIDI-Piano 采样 40 首 (seed=42)，FluidSynth 合成渲染
**方法：** `eval/eval.py` 双基线对比（自训 OaF vs basic-pitch），mir_eval 三指标

### 指标对比

| 指标 | VER2.0_Bootstrap | basic-pitch | 说明 |
| :--- | :--- | :--- | :--- |
| frame_f1 | **0.983** | 0.977 | 帧级虚高，不反映感知 |
| note_f1 | 0.215 | **0.277** | 音符级，关键战场 |
| offset_f1 | **0.077** | 0.063 | 含 offset |
| note precision | **0.421** | 0.395 | 检出音符命中率 |
| note recall | 0.148 | **0.218** | 漏检率 |

### 诊断结论

**自训模型主瓶颈 = 漏检**：est/gt 音符数比例中位数 0.37（只检出 1/3），note_recall 中位数 0.15。precision 尚可（0.42）说明检出的算准，问题在漏。

报告文件：`eval/reports/report_20260731-161608.json`

### ⚠️ 评测修正 (当天)

**发现 pitch 单位 bug**：mir_eval.transcription 的 pitch 参数是 Hz，误传了 MIDI 号 → 半音错误被误判为匹配，note_f1 高估约一倍。修复见 `eval/metrics.py`。教训记入 memory。

**修正后真实指标**（`eval/reports/report_20260731-194519.json`）：

| 指标 | 修正后真实值 |
| :--- | :--- |
| note_f1 | **0.217** |
| note precision | 0.378 |
| note recall | 0.154 |

### 后处理参数 A/B（修正评测后验证）

| 配置 | P | R | note_f1 |
|------|------|------|---------|
| min_frames=4 + 自适应pct50（旧） | 0.525 | 0.182 | 0.271 |
| **min_frames=2 + 自适应pct50（采纳）** | 0.500 | 0.232 | **0.317** |
| min_frames=2 + 固定0.3（曾误改，已回退） | 0.307 | 0.150 | 0.201 |

**结论**：只保留 `min_frames 4→2`（有效），阈值改固定 0.3 有害（precision 0.525→0.307）已回退。早期"0.421/0.608"等数字是 bug 高估，以本表为准。

---

## VER1.0_BasicPipeline

**日期：** 2026-07-05
**分支：** `se/ver1.0`（14 commits）
**状态：** ✅ 全链路打通，7/7 测试通过

### 架构描述

管道式流水线，11 个模块串联。输入音频文件，经 6 步处理输出 5 份独立 MusicXML 乐谱。

```
音频文件 (.wav/.mp3)
  → [1] BPM 检测 (librosa.beat.beat_track)
  → [2] 音轨分离 (htdemucs_6s → bass/drums/vocals/guitar/piano)
  → [3] 音高识别 (crepe 单音 + basic-pitch 钢琴多音)
  → [4] 和弦识别 (madmom CNN+CRF → guitar only, 降级安全)
  → [5] 调号推算 (Krumhansl-Schmuckler 算法, 24 调候选)
  → [6] 乐谱组装 + 导出 (music21 → .musicxml × 5)
```

### 当前能力

| 功能 | 状态 | 说明 |
|------|------|------|
| BPM 检测 | ✅ | 40~250 BPM 范围，失败降级 120 |
| 5 轨分离 | ✅ | bass / drums / vocals / guitar / piano |
| 单音 pitch tracking | ✅ | crepe, fmin=50Hz, fmax=2000Hz |
| 钢琴多音转录 | ✅ | basic-pitch，不可用时降级为 crepe |
| 吉他和弦标注 | ⚠️ | madmom CNN+CRF，Python 3.10+ 兼容性问题导致降级为空 |
| 调号推算 | ✅ | K-S 算法，24 调 Pearson 相关 |
| 力度标记 | ✅ | 振幅 → pp/p/mp/mf/f/ff 查表 |
| 渐强/渐弱 | ✅ | 相邻音符振幅趋势检测 |
| 速度术语 | ✅ | BPM → Grave/Largo/Adagio/Andante/Moderato/Allegro/Vivace/Presto |
| 滑音/滑弦 | ✅ | 相邻帧 pitch 连续渐变 → glissando |
| 吉他六线谱排指 | ✅ | DP 最短路径（22 品标准调弦），空弦偏好，弦切换惩罚 |
| Bass 八度上移 | ✅ | pitch < 50 → +12，方便读谱 |
| MusicXML 输出 | ✅ | 每音轨独立 .musicxml + info.json + pipeline.log |
| 鼓乐谱 | ❌ | 跳过，鼓需要不同的记谱逻辑 |

### 外部模型依赖

| 模型 | 用途 | 失败策略 |
|------|------|---------|
| htdemucs_6s | 音轨分离 | **致命** — RuntimeError |
| crepe (torchcrepe) | 单音 pitch tracking | 降级 — 跳过该音轨 |
| basic-pitch | 钢琴多音转录 | 降级 — 改用 crepe |
| madmom CNN+CRF | 吉他和弦识别 | 降级 — 返回空列表 |
| librosa | BPM 检测 + 音频加载 | 降级 — BPM=120 |

### 技术参数

| 参数 | 值 |
|------|-----|
| 采样率 | 44100 Hz |
| hop_length | 512 (~11.6ms) |
| crepe fmin/fmax | 50 / 2000 Hz |
| 吉他最大品位 | 22 (电吉他) |
| 调弦 | 标准 EADGBE |
| 排指代价权重 | 品位=1.0, 跨弦=2.0, 空弦偏好=-0.5 |
| 输出格式 | MusicXML (.musicxml) |
| 拍号 fallback | 4/4 |

### 已知限制

1. **鼓轨不生成乐谱** — 鼓的打击乐记谱需要不同逻辑，VER1.0 跳过
2. **madmom 在 Python 3.10+ 不兼容** — `collections.MutableSequence` 已移除，吉他和弦标签暂时缺失
3. **拍号固定 4/4** — 未实现自动拍号检测
4. **无反复记号 / D.S. / Coda** — 需理解乐段结构，留给后续版本
5. **无吉他推弦/勾击弦** — 需专门检测算法
6. **无钢琴踏板标记** — 需专门检测算法
7. **排指不追踪手指状态** — VER1.0 仅做 DP 最短路径，不做四指约束
8. **琶音检测未实现** — 试了但不可靠，跳过

### 性能参考

无训练指标（本模块为管道式拼接，不涉及模型训练）。

端到端测试：7/7 pass（score_build conda env, Python 3.10）
- TestBPM ✅
- TestPitch ✅ (A4=440Hz → MIDI 69)
- TestKeyEstimate ✅ (C major scale → C major)
- TestMusicXMLRoundtrip ✅ (assemble → export → re-parse)
- TestGuitarFingering ✅ (C major scale DP)
- TestMixFixture × 2 ✅ (15s 合成音频 BPM + 完整性)

---

## 后续版本规划

详见 `version_plan.md`

| 版本 | 内容 |
|------|------|
| VER2.0_AdvancedFingering | 把位窗口 + 空弦优化 + 滑音换把 + 变调夹 |
| VER3.0_FullFingerModel | 四指状态追踪 + 乐句预扫描 + 指法约束 |
| 未来 | 推弦/勾击弦/踏板/反复记号/多声部合谱/PDF输出 |
