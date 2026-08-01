# AMT 技术综述与改进策略

> 2026-07-06 | 基于 score_extraction/article/ 中的 3 篇论文

---

## 一、AMT 的技术层级

论文 2 (Benetos et al., 2019) 将 AMT 分为四个层级：

| 层级 | 任务 | 输出 |
|------|------|------|
| **Frame-level** (MPE) | 每帧（~10ms）判断哪些音在响 | 钢琴卷帘 (piano roll) |
| **Note-level** (NT) | 把帧级音高连接成音符（onset + offset + pitch） | MIDI 音符列表 |
| **Stream-level** (MPS) | 把音符按乐器/声部分组 | 分轨 MIDI |
| **Notation-level** | 输出人类可读乐谱 | MusicXML / 五线谱 |

**我们目前的状态：** basic-pitch 直接做 note-level，但质量不够。缺少 frame-level → note-level 的两阶段管线，也没有 stream-level 的声部分离。

---

## 二、SOTA 架构（2018-2022）

### 2.1 Onsets and Frames（Google Magenta，2018）

这是改变游戏规则的工作。论文 1 (NoteEM) 和论文 3 都在此基础上发展：

```
音频 → CQT频谱 → CNN/Transformer Encoder
                      ├── onset head → onset概率图 (每帧每个音高是否为音符起点)
                      └── frame head → frame概率图 (每帧每个音高是否在响)

onset * frame → 后处理 → 音符列表
```

**核心洞察：** onset 检测比 frame 检测更容易学，两者的 joint prediction 互相正则化。

### 2.2 MT3（Google，2021）

基于 T5 Transformer，把 AMT 建模为 seq2seq 任务：
- 输入：音频频谱序列
- 输出：MIDI-like token 序列（instrument, pitch, velocity, time）
- 支持多乐器

### 2.3 NoteEM（论文 1，2022）

在 Onsets and Frames 架构上加了 EM 训练框架：
- 用合成数据做 bootstrapping
- E-step：用弱转录器 + DTW 对齐 score → 产生伪标签
- M-step：用伪标签重新训练转录器
- 1-2 轮迭代后达到 SOTA（MAPS: 87.3% F1）

### 2.4 Self-Attention 实例分割（论文 3，2020）

把 AMT 视为计算机视觉中的实例分割：
- 频谱图 → 特征提取 → self-attention → 每个"音符实例"作为一个 segment
- 多任务学习：pitch + onset + offset + instrument 联合预测

---

## 三、我们现在的问题

| 问题 | 根因 | 论文中的解法 |
|------|------|------------|
| 音高不准 | basic-pitch 是 2022 年的单模型，没有 onset/frame 分离 | Onsets+Frames 的联合预测架构 |
| 时值不准 | basic-pitch 的 offset 检测很弱 | 用 frame-level 连续性替代 offset 端点检测 |
| 多声部乱 | 无 voice separation | Stream-level 分组（论文 3 的实例分割） |
| 跨曲泛化差 | 无数据增强 | 音高移位增强（论文 1）、合成数据预训练 |
| 噪声音符多 | 无后处理 | HMM 平滑、音乐语言模型 |

---

## 四、推荐改进路线

### 方案 A：换更强的预训练模型（最快，1 天内）

把 basic-pitch 换成 **Google MT3** 或 **Onsets and Frames V3**：
- MT3 支持多乐器，Transfomer-based，比 basic-pitch 强一个量级
- 直接 pip install，API 兼容
- **风险：** MT3 模型很大（~1GB），可能对 GPU 显存有要求

### 方案 B：两阶段管线（中等工作量，2-3 天）

```
音频 → Frame-level MPE (crepe 或 basic-pitch frame模式)
     → Frame 后处理:
        1. HMM 平滑（消除孤立噪声音符）
        2. 贪心匹配（连接同音高的连续帧 → note）
        3. 最小音符时长过滤
     → Onset 精修（用频谱能量跳变检测修正 onset 位置）
     → Note-level 输出
```

**优点：** 在不换模型的前提下大幅提升时值准确性和噪声音符过滤
**缺点：** 仍受限于 basic-pitch 的音高检测能力

### 方案 C：Onsets and Frames + 自制训练（最大工作量，5-7 天）

1. 用合成钢琴数据（如 Maestro 的 MIDI → 渲染 WAV）预训练 Onsets and Frames
2. 用论文 1 的 EM 框架 + pitch shift 增强提升泛化
3. 集成到我们的 pipeline 中

**优点：** 完全可控，最终质量最高
**缺点：** 需要训练（GPU 时间 ~1-2 天），需要准备训练数据

---

## 五、立即执行的改进（不换模型）

无论选哪个长期方案，以下改进可以**立刻**提升当前 basic-pitch 输出的质量：

1. **Frame-level HMM 平滑** — 音符不该在 50ms 内出现又消失，用 HMM 消除"闪烁"
2. **Onset 精修** — 用频谱能量跳变检测把 basic-pitch 的 onset 精确到 ±20ms
3. **最小音长过滤** — 丢弃 < 60ms 的音符（基本是噪声）
4. **谐波过滤** — 如果一个音的所有泛音都被另一个音覆盖，可能是谐波误检
5. **调性一致性检查** — 过滤明显不在调性内的孤立音符

---

## 六、推荐决策

| 方案 | 效果提升 | 工作量 | 建议 |
|------|---------|--------|------|
| A（换 MT3） | 大幅 | 半天 | ⭐ 先试 |
| B（两阶段） | 中幅 | 2-3天 | A 的补充 |
| C（自训练） | 最大 | 5-7天 | 大创最终版 |
| 立即改进 | 小幅 | 1-2h | ⭐ 现在就做 |

**建议：先做"立即改进"稳住当前质量 → 试 MT3 看效果 → 决定是否需要 B/C 方案。**
