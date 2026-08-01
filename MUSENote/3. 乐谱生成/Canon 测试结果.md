# Canon 测试结果

> Pachelbel Canon in D — George Winston 钢琴版
> BPM: 73.9 | Key: C major | Time: 4/4

## 管线输出

| 音轨 | 音符数 | 状态 |
|------|--------|------|
| piano | 1129 | ✅ MusicXML + MIDI |
| bass | 576 | ✅ |
| vocals | 438 | ✅ |
| guitar | 413 | ✅ |

## 技术参数

| 参数 | 值 |
|------|-----|
| 转录后端 | basic-pitch (onset_threshold=0.4) |
| 连通域候选 | 2094 |
| Onset 验证后 | ~1200 |
| 谐波过滤后 | ~1100 |
| 和弦归组 | 702/827 对齐 |
| 最终输出 | 1129 notes, 98 measures |

## 已知问题

- 慢段: 基本准确
- 中段: 节奏偶有抖动
- 快段: 部分音符漏检, 节奏不齐
- 高音区: 偶有被吞 (onset threshold 偏高)

## 根本瓶颈

- basic-pitch 帧率 86Hz → onset 精度 ±23ms
- 16 分音符 @73.9BPM ≈ 203ms → ±11% 误差
- 无 offset 检测头 → 音符结束不精确
- 无节拍意识 → 节奏量化难度大

## 关联

- [[技术白皮书]]
- [[乐谱生成总览]]
- [[transcriber.py]]
