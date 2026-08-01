# 乐谱生成模型 VER1.0 设计文档

> 日期：2026-07-05
> 分支：`se/ver1.0`

---

## 一、整体方案：管道式流水线

```
音频文件 (wav/mp3)
  │
  ├── ① bpm_detect.py       → BPM值 (float, fallback=120)
  │
  ├── ② source_separate.py  → 调用 htdemucs_6s
  │     产出: bass.wav, drums.wav, vocals.wav, guitar.wav, piano.wav
  │
  ├── ③ pitch_detect.py     → 按音轨并行识别:
  │     钢琴   → basic-pitch (多音)
  │     吉他   → crepe (单音)
  │     人声   → crepe (单音)
  │     贝斯   → crepe (单音，可能需八度下移处理)
  │     鼓     → 跳过，仅记录节拍点
  │
  ├── ④ chord_detect.py     → 仅吉他音轨
  │     madmom 和弦识别 (C/G/Am 等常见开放和弦)，作为乐谱标注
  │
  ├── ⑤ key_estimate.py     → Krumhansl-Schmuckler 算法
  │     汇总所有音轨音符，推算调号
  │
  ├── ⑥ score_assemble.py   → music21 逐音轨构建 Score:
  │     - Stream + PartStaff
  │     - 拍号 (TimeSignature)
  │     - 调号 (KeySignature)
  │     - 音符 + 休止符
  │     - 力度标记 (p/mp/mf/f 根据振幅映射)
  │     - 渐强/渐弱 (相邻音符振幅趋势)
  │     - 速度术语 (BPM 查表映射: Andante/Moderato/Allegro)
  │     - 滑音/滑弦 (crepe 帧间连续渐变 → glissando)
  │     - 琶音(尝试) (密集音符簇 + 和弦结构匹配 → arpeggio)
  │     - 小节线自动划分 (music21 根据拍号处理)
  │     - 终止线
  │
  └── ⑦ export_score.py    → 输出:
        output/<song_name>/
          ├── bass.musicxml
          ├── drums.musicxml
          ├── vocals.musicxml
          ├── guitar.musicxml
          ├── piano.musicxml
          ├── info.json       ← BPM, key, time_sig 等元信息
          └── pipeline.log    ← 运行日志
```

每个音轨独立一份乐谱，不做合并。合并留个前端。

---

## 二、文件结构

```
score_extraction/
├── src/
│   ├── config.py              # 全局配置
│   ├── pipeline.py            # 主流程入口
│   ├── bpm_detect.py          # BPM检测 (librosa)
│   ├── source_separate.py     # 音轨分离 (htdemucs_6s)
│   ├── pitch_detect.py        # 音高/时值识别 (crepe / basic-pitch)
│   ├── chord_detect.py        # 和弦识别 (madmom, 吉他)
│   ├── key_estimate.py        # 调号推算 (K-S算法)
│   ├── score_assemble.py      # music21 乐谱组装
│   └── export_score.py        # 输出 MusicXML
├── data/                      # 数据处理脚本
├── markdown/
│   ├── 需求文档.md
│   ├── version_plan.md        # 版本规划 + 待办
│   └── model_log.md           # 版本日志（与 instrument_recognition 一致）
├── output/                    # 输出目录
└── test/
    ├── fixtures/              # 测试用短音频
    └── test_pipeline.py       # 单元测试 + 端到端
```

---

## 三、config.py 核心参数

```python
# 路径
OUTPUT_DIR = Path("d:/program_project/MUSE/score_extraction/output")
DEMUCS_MODEL_PATH = None        # None=默认路径

# 模型名
DEMUCS_MODEL = "htdemucs_6s"
PITCH_MODEL_PIANO = "basic-pitch"
PITCH_MODEL_MONO = "crepe"

# 音频
SAMPLE_RATE = 44100
HOP_LENGTH = 512

# 乐谱 fallback
DEFAULT_BPM = 120
DEFAULT_TIME_SIG = "4/4"

# 吉他排指 (VER1.0 DP)
MAX_FRET = 22                   # 电吉他 22 品
FRET_WEIGHT = 1.0               # 品位距离
STRING_WEIGHT = 2.0             # 跨弦惩罚
OPEN_STRING_BIAS = -0.5         # 空弦偏好

# 滑音检测
SLIDE_PITCH_THRESHOLD = 0.5     # 半音阈值
SLIDE_MAX_INTERVAL = 5          # 最大滑音帧数 (~50ms)
```

---

## 四、演奏技巧标注（第一阶段）

| 技巧 | 做法 | 第一阶段 |
|------|------|---------|
| 力度标记 (p/mf/f) | crepe amplitude → 查表映射 | ✅ 做 |
| 渐强/渐弱 | 相邻音符振幅趋势检测 | ✅ 做 |
| 速度术语 | BPM → 查表 (Andante/Allegro 等) | ✅ 做 |
| 滑音/滑弦 | crepe 帧间连续渐变 → glissando | ✅ 做 |
| 琶音 | 密集音符簇 + 和弦匹配 → arpeggio | ⚠️ 尝试，不行跳过 |
| 反复记号 | 需理解乐段结构 | ❌ 不做 |
| 吉他推弦/勾击弦 | 需专门检测算法 | ❌ 不做 |
| 钢琴踏板 | 需专门检测算法 | ❌ 不做 |

---

## 五、吉他六线谱排指 — VER1.0 DP 算法

### 目标
给定音符序列和标准调弦(E2 A2 D3 G3 B3 E4)，为每个音符选择最优的(弦, 品位)，使左手移动代价最小。

### 算法
1. **建图**：每个音符枚举 6 弦 × 22 品的所有合法位置（音高正确的品位）
2. **代价函数**：`cost = w1 × |f1-f2| + w2 × |s1-s2| + w3 × (是否空弦偏好)`
3. **DP**：`dp[i][pos] = min(dp[i-1][prev] + cost(prev, pos))`，O(n·k²)，k ≤ 132
4. **回溯**得出最优弦-品位序列
5. **输出** music21 TabStaff (TabNote with `string_number` + `fret_number`)

### 不做（留给 VER2.0+）
- 手指状态追踪（空闲手指预留）
- 把位窗口切分
- 变调夹
- 多指排指约束

### 复杂度
500 音符 × 100 候选位置 × 100 前驱 = 5×10⁶ 次比较 → 毫秒级

---

## 六、错误处理 / 容错策略

| 环节 | 失败策略 |
|------|---------|
| demucs 分离失败 | **致命** — 终止，报原因 |
| BPM 检测失败 | **降级** — fallback=120，warn |
| basic-pitch 不可用 | **降级** — 钢琴改用 crepe 单音 |
| crepe 某音轨空结果 | **跳过该轨** — warn，不阻断其他 |
| madmom 不可用 | **跳过** — 吉他乐谱缺和弦标注 |
| 某音轨识别零音符 | **warn** — 可能是该轨本身无旋律 |

日志同时写终端 + `output/<song>/pipeline.log`

---

## 七、测试

`test/test_pipeline.py`：

1. **BPM 单元测试**：120BPM 节拍器音频 → 偏差 < 2
2. **音高单元测试**：A4=440Hz 正弦波 → crepe 识别 A4
3. **MusicXML 输出测试**：生成简单乐谱 → 能被 music21 重读且音符数一致
4. **端到端测试**：短音频(≤30s) → 全 pipeline 不报错 + output 有 5 个 .musicxml

测试数据放 `test/fixtures/`

---

## 八、依赖

| 库 | 用途 |
|----|------|
| `demucs` | 音轨分离 (htdemucs_6s) |
| `librosa` | 音频处理 + BPM 检测 |
| `crepe` / `torchcrepe` | 单音 pitch tracking |
| `basic-pitch` | 钢琴多音转录 |
| `madmom` | 和弦识别（吉他加分项） |
| `music21` | 乐谱构建 + MusicXML 输出 |
| `numpy`, `scipy` | 通用数值计算 |
| `tqdm` | 进度条 |

---

## 九、后续版本路线

| 版本 | 内容 |
|------|------|
| VER1.0 | 全链路打通，基础 DP 排指 |
| VER2.0 | 把位窗口 + 空弦优化 + 滑音换把 |
| VER3.0 | 完整四指状态追踪排指模型 |
| 未来 | 推弦/勾击弦/踏板/反复记号/变调夹/多声部合谱 |

详见 `version_plan.md`
