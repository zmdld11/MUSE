# 乐谱生成 VER2.0 架构重设计

> 基于：NoteEM (ICML 2022) + AMT Overview (IEEE SPM 2019) + Self-Attention AMT (TASLP 2020)
> 日期：2026-07-06

---

## 一、论文架构 vs 我们当前架构

```
我们当前的（幼稚）做法：
  audio → [basic-pitch 黑盒] → 一堆不靠谱的音符 → music21 → 垃圾乐谱
                                  ↑
                            完全不可控！
                            没有中间产物
                            无法调试
                            无法改进

论文里的成熟做法：
  audio → CQT频谱 → [Onset检测器] ──┐
                  → [Frame检测器] ──┤
                  → [Contour检测器] ┘
                          ↓
                  三张概率图 (onset_map, frame_map, contour_map)
                          ↓
                  [后处理管线]
                    ├── 阈值 + HMM平滑
                    ├── 连通域分析 (frame→note)
                    ├── Onset精修
                    ├── 谐波过滤
                    └── 声部分配
                          ↓
                  干净的音符列表
                          ↓
                  music21 → MusicXML
```

核心区别：**中间产物可视、每步可替换、问题可定位**

---

## 二、新架构：五层管线

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: ACOUSTIC FRONTEND                         │
│  audio → CQT → 3D tensor (time × freq × 3)         │
├─────────────────────────────────────────────────────┤
│  Layer 2: TRANSCRIPTION MODEL                       │
│  tensor → [OnsetHead] → onset_probs (T × 88)        │
│         → [FrameHead] → frame_probs (T × 88)        │
│         → [ContourHead]→ contour (T × 88)           │
│  模型可选: basic-pitch / OnsetsAndFrames / MT3       │
├─────────────────────────────────────────────────────┤
│  Layer 3: FRAME-LEVEL POST-PROCESSING               │
│  probs → HMM平滑 → 阈值二值化 → 连通域标注           │
├─────────────────────────────────────────────────────┤
│  Layer 4: NOTE-LEVEL POST-PROCESSING                │
│  连通域 → onset精修 → 谐波过滤 → 音长校正 → 力度估计  │
├─────────────────────────────────────────────────────┤
│  Layer 5: NOTATION ASSEMBLY                         │
│  音符 → 声部分配 → 调号/拍号 → 力度/技巧 → MusicXML  │
└─────────────────────────────────────────────────────┘
```

---

## 三、各层详细设计

### Layer 1: Acoustic Frontend

```
输入: audio.wav (44100 Hz, mono/stereo→mono)

处理:
  CQT (Constant-Q Transform)
    - hop_length = 512 (~11.6ms, 86 fps)
    - n_bins = 352 (覆盖 4 个八度 × 88 个半音)
    - bins_per_octave = 12 × 3 = 36 (每个半音 3 个 bin)
    - fmin = 32.7 Hz (C1), n_bins=264 (88 音符 × 3 bin/音符)

输出: tensor shape (3, T, 88)
  3 通道: magnitude, phase, log-magnitude
  T = 帧数 (~26000 帧 / 5分钟曲目)
  88 = MIDI 21-108 (钢琴全音域)
```

### Layer 2: Transcription Model

**核心改动：不用 basic-pitch 的后处理输出，而是用它的原始模型输出。**

basic-pitch 内部就是一个 Onsets-and-Frames 架构！它的 `predict()` 返回：
```python
model_output   # (onset_probs, frame_probs, contour) — 这才是我们要的！
midi_data      # MIDI-like
note_events    # 后处理过的音符 (不要用这个)
```

`model_output` 包含三个张量：
- `onset_probs`: (T, 88) — 每帧每个音高是 onset 的概率
- `frame_probs`: (T, 88) — 每帧每个音高在响的概率
- `contour`: (T, 88) — 连续音高微调 (用于表达性音高弯曲)

**我们自己做后处理，把黑盒变白盒。**

```
接口:
  transcribe(audio) → {
      "onset_probs": np.ndarray (T × 88),
      "frame_probs": np.ndarray (T × 88),
      "contour": np.ndarray (T × 88),
  }
```

### Layer 3: Frame-Level Post-Processing

这是论文里最核心但我们完全缺失的一步。

```
Step 3.1: HMM 时域平滑
  问题: 神经网络的逐帧预测有"闪烁" — 某帧说G4在响，下一帧又说没有
  方案: 对每个音高通道做 HMM 前向-后向平滑
    - 状态: ON / OFF
    - 转移概率: P(ON→OFF) = 0.1, P(OFF→ON) = 0.05
    - 发射概率: frame_probs 作为 ON 状态的似然
  效果: 消除 < 3 帧 (~35ms) 的闪烁

Step 3.2: 自适应阈值
  问题: 全局阈值 (如 0.3) 对弱音和强音不公平
  方案:
    - 按音高分组 (低/中/高音区)
    - 每组用自己的百分位阈值 (默认 30th percentile)
    - 用户可调 --threshold 参数

Step 3.3: 连通域标注
  问题: frame_probs 是逐帧的，需要连成音符
  方案: 对阈值后的二值图做连通域分析 (scipy.ndimage.label)
    - 每个连通域 = 一个候选音符
    - 连通域位置: (onset_frame, offset_frame, pitch_bin)
  输出: candidate_notes = [(start_frame, end_frame, pitch, confidence)]

Step 3.4: 最小音长过滤
  过滤: 音长 < 3 帧 (~35ms) 的候选音符 → 丢弃
  过滤: 音长 > 600 帧 (~7s) 的候选音符 → 截断并 warn
```

### Layer 4: Note-Level Post-Processing

```
Step 4.1: Onset 精修
  问题: 神经网络的 onset 检测有 ±2 帧 (~23ms) 的抖动
  方案: 在每个候选音符的 onset 附近 ±5 帧窗口内，
        用频谱能量跳变 (spectral flux) 找精确 onset 位置
  输出: onset_time 精确到 ±5ms

Step 4.2: 谐波过滤
  问题: 一个真实音符的泛音经常被误检为独立音符
        (例如 C4 的 2 次泛音 C5，3 次泛音 G5)
  方案:
    - 检查每个音符的 pitch 是否为另一个更强音符的泛音
    - 泛音关系: pitch ± 12 (八度), ± 19 (五度+八度)
    - 如果是泛音且置信度/振幅明显低于基音 → 标记为可疑，丢弃

Step 4.3: 力度估计
  方案: 取候选音符时间窗口内的 RMS 能量
    velocity = clip(RMS / max_RMS * 127, 0, 127)

Step 4.4: 重叠音符合并
  问题: 同音高、几乎同时开始/结束的音符 (duplicate detection)
  方案: pitch 相同 且 onset 差 < 50ms → 合并为同一个音符
```

### Layer 5: Notation Assembly (保留+改进现有)

```
Step 5.1: 声部分配 (新增)
  对于钢琴: 用音高聚类 (高音区→右手, 低音区→左手)
  对于吉他: 用已有的 DP 排指
  输出: 每个音符带 voice_id

Step 5.2: 调号推算 (保留 key_estimate.py)
Step 5.3: 拍号 / 小节线 (保留)
Step 5.4: 力度标注 (保留相对量化方案)
Step 5.5: music21 组装 (保留 score_assemble.py)
Step 5.6: MusicXML 导出 (保留 export_score.py)
```

---

## 四、文件结构重构

```
score_extraction/src/
├── config.py                # 全局配置 (保留，加新参数)
├── frontend.py              # [新] Layer 1: CQT + 特征提取
├── transcriber.py           # [重写] Layer 2: basic-pitch 原始输出
├── frame_post.py            # [新] Layer 3: HMM平滑 + 阈值 + 连通域
├── note_post.py             # [新] Layer 4: onset精修 + 谐波过滤 + 合并
├── voice_assign.py          # [新] 声部分配 (替代 guitar_tab.py 的部分功能)
├── guitar_tab.py            # [保留] 吉他排指 DP
├── key_estimate.py          # [保留] 调号推算
├── score_assemble.py        # [重写] 简化 (只管音符→music21，不管后处理)
├── export_score.py          # [保留]
├── pipeline.py              # [重写] 新版五层管线
├── chord_detect.py          # [保留] 和弦识别
├── source_separate.py       # [保留] demucs 分离
└── bpm_detect.py            # [保留] BPM 检测
```

---

## 五、关键改进对比

| 维度 | VER1.0 | VER2.0 |
|------|--------|--------|
| 转录模型使用方式 | 黑盒（只用 note_events） | **白盒**（用原始 onset/frame probs） |
| 后处理 | 无 | **5 步管线** |
| 单帧噪声 | 大量假音符 | **HMM 平滑消除** |
| 谐波误检 | 无法处理 | **谐波过滤** |
| Onset 精度 | ±50ms | **±5ms**（频谱能量精修） |
| 可调试性 | 只能调 basic-pitch 参数 | **每步中间产物可视化** |
| 声部 | 无 | **音高聚类分配** |

---

## 六、VER1.0 保留的模块

以下模块在论文中没有更好的替代方案，保留：

| 模块 | 理由 |
|------|------|
| demucs 分离 | SOTA 音源分离，论文也未超越 |
| BPM 检测 | 成熟方案 |
| 调号推算 (K-S) | 经典算法，论文无替代 |
| 吉他排指 DP | 我们自研，论文里没有 |
| MusicXML 导出 | 成熟方案 |
