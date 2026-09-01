# 乐谱生成技术白皮书 — 音高识别与节拍检测全流程

> 以 Pachelbel Canon in D (George Winston 钢琴版) 为例，逐层拆解当前管线。

---

## 总览：五层管线

```
音频(.wav/.flac)
  │
  ├─ Layer 0: 预处理 ─ BPM 检测 + 音轨分离
  │
  ├─ Layer 2: 转录模型 ─ basic-pitch (Onsets-and-Frames)
  │     输出：onset 概率图 + frame 概率图 (每帧每音高)
  │
  ├─ Layer 3: 帧级后处理 ─ HMM 平滑 + 自适应阈值 + 连通域 + onset 验证
  │     输出：候选音符列表 {onset_frame, offset_frame, pitch, confidence}
  │
  ├─ Layer 4: 音符级后处理 ─ onset 精修 + 谐波过滤 + 合并去重
  │     输出：干净音符列表 {onset(秒), offset(秒), pitch(MIDI), amplitude}
  │
  ├─ Layer 5: 乐谱组装 ─ 声部分配 + score 构建 + MIDI→XML 导出
  │
  └─→ MusicXML + MIDI
```

---

## 一、用到的技术清单

| 步骤 | 技术 | 论文/来源 |
|------|------|----------|
| BPM 检测 | librosa.beat.beat_track (onset strength + 自相关) | Böck 2014 |
| 音轨分离 | htdemucs_6s (Hybrid Transformer Demucs) | Défossez 2021 |
| 转录模型 | **Onsets-and-Frames** (onset/frame 双头联合预测) | Hawthorne 2018, ICASSP 2022 |
| 帧平滑 | **HMM 前向-后向平滑** (per-pitch 2-state ON/OFF) | Rabiner 1989 |
| 阈值二值化 | **自适应百分位阈值** (低/中/高音区分组, 50th pct) | 自研 |
| 连通域 | scipy.ndimage.label (2D 4-连通) | 经典 CV |
| Onset 验证 | 滑动窗口 onset_probs 最大支持度 | 自研 |
| Onset 精修 | **Spectral Flux** (频谱能量跳变, STFT 差分) | Dixon 2006 |
| 谐波过滤 | 泛音关系匹配 (八度/五八度/双八度/三八三度) | 自研 |
| 和弦归组 | 40ms 窗口内 onset 对齐到最早时间 | 自研 |
| 调性过滤 | C 大调音阶 (pitch classes {0,2,4,5,7,9,11}) + 邻音支持 | 自研 |
| 声部分配 | 中位数音高分割 (上下声部) | 自研 |
| 调号推算 | Krumhansl-Schmuckler 24 调 Pearson 相关 | Krumhansl 1990 |
| 节拍量化 | 16 分 + 8 分三连音双网格, 分段最佳偏移 | 已禁用 |
| 乐谱组装 | music21 Stream + Measure 手动构建 | Cuthbert 2010 |

---

## 二、逐层详解（以 Canon 为例）

### Layer 0: 预处理

**BPM 检测：** `src/bpm_detect.py`
```
Canon 输入: 5'20" 立体声 FLAC
librosa.beat.beat_track → 检测到 BPM = 147.7
自动减半: 147.7 > 120 → BPM = 73.9  ← 更接近实际
```

**音轨分离：** `src/source_separate.py`
```
Canon 是钢琴独奏曲 → demucs 分离出 5 轨:
  piano.wav     (主要信号)
  bass.wav      (低频残留, 基本为空)
  vocals.wav    (空)
  guitar.wav    (空)
  drums.wav     (空)
```

---

### Layer 2: 转录模型 — 音高确定的核心

`src/transcriber.py` 调用 **basic-pitch**（Spotify, ICASSP 2022）。

#### basic-pitch 内部架构：

```
音频 (44100 Hz, mono)
    ↓
Mel 频谱: 229 mel bins, hop=256 (≈11.6ms)
    ↓
HarmonicStack: 在频率轴上 shift 8 次拼接 (捕捉泛音关系)
    ↓
CNN Encoder → LSTM → 两个输出头:
    ├─ Onset Head → onset_probs (T, 88)  每个音高"是不是新音符起点"
    └─ Frame Head → frame_probs (T, 88)  每个音高"现在是不是在响"
```

#### Canon 实际输出：
```
音频长度: 320 秒
帧率: 22050/256 ≈ 86 Hz
总帧数: T = 27821 帧
onset_probs: (27821, 88)   ← 88 = MIDI 21 (A0) ~ 108 (C8)
frame_probs: (27821, 88)
```

**怎么确定一个音是不是出现了？**
basic-pitch 不直接给"C4 在 3.5 秒出现"—它是概率图。每帧 (11.6ms) × 88 音高，每个格子里是 0~1 之间的概率。C4 对应的 bin 是 MIDI 60 - 21 = 39。第 t=302 帧如果 frame_probs[302, 39] = 0.92，说明"在 302 帧 (≈3.5s) C4 在响的概率是 92%"。

**怎么确定一个音什么时候结束？**
frame_probs 降到阈值以下就是结束。但逐帧概率有"闪烁"—同一帧说响下一帧说不响—所以需要 HMM 平滑。

#### 当前参数：
```python
onset_threshold=0.4   # 低于此值的 onset 丢弃（降噪）
frame_threshold=0.2   # frame 检测灵敏度
# minimum_note_length 使用默认 128ms
```

---

### Layer 3: 帧级后处理 — 从概率图到候选音符

`src/frame_post.py`

#### Step 1: HMM 前向-后向平滑

**为什么需要？** CNN 的逐帧预测是独立的，有"闪烁"——某个音高在第 N 帧是 0.8，第 N+1 帧突然变成 0.1，第 N+2 帧又 0.7。人耳听到的音符不存在这种事情。

**怎么做？** 对 88 个音高各自做一条 HMM 链（2 状态：ON / OFF）：

```
状态转移矩阵:
         OFF → OFF: 0.85    OFF → ON:  0.15
         ON  → OFF: 0.15    ON  → ON:  0.85
                              ↑          ↑
                        不太可能突然停  不太可能突然响
```

对每帧做前向-后向计算（Forward-Backward Algorithm），输出后验概率 P(ON | 所有观测)。效果：把相邻帧的概率"拉平"，消除闪烁。

```
平滑前: [0.1, 0.8, 0.1, 0.9, 0.2, 0.1, 0.7, ...]
平滑后: [0.2, 0.6, 0.5, 0.5, 0.3, 0.2, 0.3, ...]
```

#### Step 2: 自适应阈值

对平滑后的概率图做二值化。不同音区的阈值不同（低音/中音/高音各取该区 50th percentile）：
```
低音区 (MIDI 21-50): threshold ≈ 0.25
中音区 (MIDI 50-72): threshold ≈ 0.30
高音区 (MIDI 72-108): threshold ≈ 0.22 (更敏感, 高音弱)
```

#### Step 3: 连通域标注 (Connected Components)

对二值图 (T, 88) 做 4-连通标注。每个连通块 = 一个候选音符：
```
块 #1: 帧 302-315, bin 39 (C4) → 候选音符: onset_frame=302, offset_frame=316, pitch=60
块 #2: 帧 302-308, bin 43 (E4) → 候选音符: onset_frame=302, offset_frame=309, pitch=64
块 #3: 帧 302-320, bin 47 (G4) → 候选音符: onset_frame=302, offset_frame=321, pitch=67
```
↑ 这三块同时开始 (= C 大三和弦), C4+E4+G4 一起响

#### Step 4: 最大复音数限制

同一帧最多 8 个音（泛音不会超过 8 个同时出现的真音）。按 confidence 排序保留 top-8。

#### Step 5: 音长过滤

```
min: 4 帧 (≈46ms) — 比这短的直接丢弃
max: 600 帧 (≈7s) — 比这长的截断
```

#### Step 6: Onset 验证

在每个候选的 onset 帧附近 ±2 帧窗口，检查 onset_probs 的最大值。低于阈值（常规 0.4, 高音 0.28）的候选丢弃。

```
Canon 实际数据:
  连通域: 2094 个候选
  最大复音数过滤: 2094 → 2094 (无变化)
  时长过滤后: ~1500
  Onset 验证后: ~1200
```

---

### Layer 4: 音符级后处理

`src/note_post.py`

#### Step 1: 帧 → 秒转换

```
onset_sec = onset_frame × hop_length / sr
          = onset_frame × 512 / 22050
          ≈ onset_frame × 0.0232s
```

#### Step 2: Onset 精修 (Spectral Flux)

在 onset 附近 ±5 帧窗口，计算频谱能量跳变 (STFT 相邻帧的正能量差分)，找到能量跳变最大的帧 → 更精确的 onset。

#### Step 3: 谐波过滤

检查音高关系。如果一个音是另一个音的泛音（相差 12/19/24/28 个半音），且时间重叠，且更弱 → 标记为谐波误检，丢弃。

```
例: C4=60, G5=79
    79 - 60 = 19 → 八度+五度泛音!
    检查: 是否同时响? 叠加段 G5 是否比 C4 弱?
    如果都满足 → G5 是 C4 的泛音误检, 丢弃
```

#### Step 4: 合并去重

同一音高, onset 差 < 50ms → 合并为一个更长音符。

#### Step 5: 和弦归组

同一和弦内的音（onset 差 < 40ms）→ 全部对齐到最早 onset，避免"分叉感"。

#### Step 6: 调性过滤

不在 C 大调音阶 (C D E F G A B) 里的音，检查周围 ±3 半音、0.5s 内有邻音支持 → 有则保留（可能是装饰音/倚音），无则丢弃。

```
Canon 实际数据:
  谐波过滤后: ~1200 → ~1100
  合并后: ~1100
  和弦归组: 702/827 组对齐
  调性过滤: 删除 2 个孤立非调内音
  → 最终: 1129 个干净音符
```

---

### Layer 5: 乐谱组装 + 导出

`src/score_assemble.py` + `src/voice_assign.py` + `src/pipeline.py`

#### 声部分配

Canon 是钢琴独奏，用音高中位数分割上下声部：
```
中位数音高 ≈ MIDI 62 (D4)
≥ 62 → voice 1 (上声部/右手)
< 62 → voice 2 (下声部/左手)
1129 个音符 → upper: ~580, lower: ~549
```

#### MIDI 写入

`_write_pretty_midi()` 用 pretty_midi 库把音符写成 .mid。每个 voice 一个 Instrument track。显式写入 4/4 拍号 + 73.8 BPM。

#### MusicXML 导出

music21 直接读取 MIDI → 写 .musicxml（正确保留 73.8 BPM 和 4/4 拍号）。

---

## 三、关键技术缺口（为什么效果还不够好）

### 1. 帧率限制（根本瓶颈）

```
帧率: 86 Hz → 每帧 ≈ 11.6ms
16 分音符 @73.9BPM ≈ 203ms → 约 17 帧
onset 检测精度: ±1-2 帧 ≈ ±23ms → onset 有 ±10% 个 16 分音符的抖动
```

在慢段 (8 分音符 ≈ 406ms, ±6% 误差) 基本可接受，快段 (16 分 ≈ 203ms, ±10% 误差) 就乱了。

### 2. 声部分离缺失

目前只有音高频谱分割（中位数），没有真正的多声部跟踪。理想做法：
- Frame-level 每个 pitch 分配一个 stream ID
- 通过 timbre/temporal continuity 跟踪同一声部
- 论文方法：Self-Attention Instance Segmentation 或 HMM 流分配

### 3. 没有音符结束检测

basic-pitch 不直接检测 offset。它只判断"每帧这个音在不在响"。音符结束 = 连续 N 帧 frame_prob < 阈值 → 判定音符结束。但踏板混响让 frame_prob 降得很慢，导致 offset 偏晚 → 音符太长。

论文方法：offset 检测需要专门 head（Onsets and Frames 后续版本加了 offset head）。

### 4. 没有节拍意识

逐帧 transcription 输出的是绝对时间（秒），不知道"现在是第几拍"。论文做法：
- Beat tracking → beat positions
- 把 onset 映射到"第几拍的第几分音符"
- 然后用音乐语言模型 (HMM/Transformer) 做节奏量化

我们试过节拍量化但回退了，因为 basic-pitch 的 onset 误差超过半个网格间隔时，量化反而把音推到错误位置。

### 5. 训练模型泛化

我们训练的 Onsets-and-Frames 在 FluidSynth 合成数据上 frame_acc=99.6%, onset_f1=0.85，但在真实录音上只有 ~385 音符（basic-pitch 是 5700+）。需要 NoteEM 的 EM 迭代才能泛化。

---

## 四、Canon 实例完整数据流

```
输入: Variations On The Canon By Pachelbel - George Winston.flac
      (5'20", 44100 Hz stereo)
      │
      ▼
BPM 检测: librosa → 147.7 → 减半 → 73.9 BPM
      │
      ▼
demucs 分离: htdemucs_6s (CUDA RTX 4060)
      → piano.wav (主要), bass/vocals/guitar/drums.wav (几乎空)
      │
      ▼
basic-pitch on piano.wav: Mel 频谱 229 bins, T=27821 帧
      → onset_probs (27821×88), frame_probs (27821×88)
      │
      ▼
HMM 平滑: 88 路 forward-backward (p_stay_on=0.85)
      │
      ▼
自适应阈值: 低/中/高音区 50th percentile
      │
      ▼
连通域标注: 4-连通 → 2094 块候选
      │
      ▼
时长过滤: 4~600 帧 → ~1500 候选
      │
      ▼
Onset 验证: 常规 0.4, 高音 0.28 → ~1200 候选
      │
      ▼
谐波过滤: 泛音关系匹配 → ~1100
      │
      ▼
和弦归组: 40ms 窗口对齐 → 702/827 组归并
      │
      ▼
调性过滤: pitch class ∈ {0,2,4,5,7,9,11} → 1129 音符
      │
      ▼
声部分配: 音高中位数 MIDI 62
      → upper voice: ~580 notes, lower voice: ~549 notes
      │
      ▼
pretty_midi: .mid (73.9 BPM, 4/4)
      │
      ▼
music21: .mid → .musicxml
      │
      ▼
   piano.musicxml (1207 notes, 73.9 BPM, 4/4) ✅
```

---

## 五、各阶段参数速查

| 参数 | 值 | 位置 |
|------|-----|------|
| onset_threshold | 0.4 | transcriber.py |
| frame_threshold | 0.2 | transcriber.py |
| HMM p_stay_on | 0.85 | frame_post.py |
| HMM p_turn_on | 0.15 | frame_post.py |
| 自适应阈值 pct | 50% | frame_post.py |
| min 帧数 | 4 (~46ms) | frame_post.py |
| max 帧数 | 600 (~7s) | frame_post.py |
| onset 验证 window | ±2 帧 | frame_post.py |
| onset 验证阈值 | 0.4 / 高音 0.28 | frame_post.py |
| 最大复音数 | 8/帧 | frame_post.py |
| spectral flux window | ±5 帧 | note_post.py |
| 谐波区间 | [12, 19, 24, 28] 半音 | note_post.py |
| 合并窗口 | 50ms | note_post.py |
| 和弦归组窗口 | 40ms | note_post.py |
| 调性模板 | C major (D E F G A B) | note_post.py |
| 声部分割 | 中位数音高 | voice_assign.py |
