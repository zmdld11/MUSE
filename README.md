# 聆谱 MUSE

> 从一首歌到一份能播放、能跟读的乐谱 —— 多乐器扒带 · 自动记谱 · 交互播放全流程系统。

给定一段歌曲音频（mp3 / flac / wav），MUSE 自动完成乐器分离、多乐器转写、人声旋律提取，再把转写结果量化记谱为五线谱（MusicXML），并在网页播放器里提供「卷帘 + 谱面 + 原曲」三视图对照播放：谱面逐音光标跟播、原始音频与 MIDI 合成音可任意叠加试听。

当前状态：**v0.1.0-pre（预览版）**——单人 demo 闭环已通，模型与记谱规则仍在迭代。

## 功能亮点

- **多乐器分离**：自研 VER-SEP 3.0b 吉他分离模型（GuitarSet F2' 0.74）负责吉他轨，Demucs / MelBand-RoFormer 负责其余乐器与人声。
- **多乐器转写**：ByteDance instrument-agnostic AMT（ia-amt）转写乐器声部；人声旋律由 SOME 专项模型从人声 stem 中提取，器乐曲自动回退混音直推。
- **自动记谱**：节拍/拍速检测 → onset 量化到乐谱格点 → 时值重构（音符/休止/连音）→ 声部分配 → 和弦识别；支持 rubato 曲目的音频提拍吸附。
- **和声分析**：内置卡农进行 / 王道进行等常见和声模板匹配（含变奏与音型折叠），并基于 CoCoPops-Billboard bigram 语料给出和弦常见度 / 原创度指标。
- **交互播放器**：卷帘（piano roll）逐音高亮 + OSMD 五线谱光标跟播，MIDI 音源 / 原曲 / 二者叠加三种音源随时切换，各乐器轨道独立静音独奏。
- **一键管线**：本地桥服务（`pipeline_server.py`）让网页端「选择文件」直接触发全流程转写，进度条分阶段实时反馈。

## 管线架构

```
                    音频 (mp3/flac/wav)
                          │
              ┌───────────┴───────────┐
              │      BPM / 节拍检测    │
              └───────────┬───────────┘
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
  VER-SEP 吉他轨     Demucs 乐器轨      MelBand 人声轨
        │                 │                  │
        ▼                 ▼                  ▼
     ia-amt            ia-amt             SOME
        │                 │              （人声旋律）
        └─────────────────┼──────────────────┘
                          ▼
                 音符层 (notes.json / *.mid)
                          ▼
              记谱层（量化 · 时值 · 声部 · 和弦）
                          ▼
            MusicXML + 谱面 MIDI (score_mid)
                          ▼
           前端播放器（卷帘 / 五线谱 / 音源叠加）
```

## 快速开始

### 环境

- Windows + Python 虚拟环境（仓库根 `env/`，含全部后端依赖：torch / demucs / librosa / pretty_midi 等）
- 前端：Node 18+ 与 pnpm
- `score_extraction/external/` 下的第三方仓库与预训练权重（ia-amt、SOME、MSST、MelBand-RoFormer 权重等）不入库，需自备后放置到该目录

### 后端：一键转写

```bash
cd score_extraction

# 单曲：音频 → 分离 → 转写 → 记谱 → 前端 demo 包
env/python.exe run_one.py <audio> --demo

# 把多个管线输出目录同步为前端曲库（frontend/public/demo）
env/python.exe run_one.py --demo-sync <管线输出目录1> <目录2> …

# 分离模式开关（off / versep_guitar 默认 / versep_demucs）
env/python.exe run_one.py <audio> --sep versep_guitar
```

### 前端：播放器

```bash
cd frontend
pnpm install
pnpm dev        # http://localhost:5173
```

### 网页端一键转写（本地桥）

```bash
cd score_extraction
env/python.exe pipeline_server.py   # 127.0.0.1:8420
```

桥启动后，播放器页「选择文件」选任意音频即自动走完整管线（进度浮层分阶段显示），完成后直接装载卷帘与谱面。

## 仓库结构

```
score_extraction/        后端管线
  run_one.py             一键流 CLI（音频 → 记谱 → demo 包）
  pipeline_server.py     网页端一键转写本地桥（HTTP）
  src/                   分离 / 转写 / 记谱 / 和弦 / 导出各层实现
  eval/                  评测脚本（GuitarSet / MIR-1K / 大横评等）
  train/                 VER-SEP 与转写模型训练
  markdown/              设计文档与路线决策记录
frontend/                网页播放器（React + Vite + Tailwind + OSMD）
  src-tauri/             Tauri 桌面壳（可选）
source_separation/       VER-SEP 分离模型训练（服务器侧）
instrument_recognition/  早期乐器识别模块
MUSENote/                研究笔记（按主题整理的文献综述与结论）
docs/ · 汇报/            项目文档与进度汇报
progress.md findings.md  开发日志与实验结论（持续更新）
```

## 致谢与引用

本系统站在以下开源工作的肩膀上：

- [Demucs](https://github.com/adefossez/demucs)（htdemucs_6s）— 乐器分离
- [MelBand-RoFormer](https://github.com/sucvr/ConvertModel_RoFormer_CaPtions)（KimberleyJSN 权重）与 [MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training) — 人声分离
- [ia-amt](https://github.com/anime-song/instrument-agnostic-amt)（instrument-agnostic AMT）— 多乐器转写
- [SOME](https://github.com/openvpi/SOME)（ISMIR 2022）— 人声音符级转写
- [basic-pitch](https://github.com/spotify/basic-pitch)（Spotify）— 评测基线
- [OpenSheetMusicDisplay](https://github.com/opensheetmusicdisplay/opensheetmusicdisplay) — 五线谱渲染
- [smplr](https://github.com/danigb/smplr) / Tone.js 生态 — 浏览器 MIDI 合成
- 和弦语料：[CoCoPops](https://github.com/CoCoPops-CSM) Billboard 和弦标注数据集

## 路线图

- [x] 一键管线闭环（音频 → 乐谱 → 播放器）
- [x] 人声旋律轨（MelBand + SOME，预览版）
- [ ] 颤音 / 滑音的后处理修正（人声音符过碎问题）
- [ ] choir / 和声轨的区分
- [ ] 谱面编排美感（声部排布 · 间距 · 和弦对齐）
- [ ] 桌面端（Tauri）内置管线

## 许可证

本项目代码采用 [MIT License](LICENSE)。

注意：本项目调用或参考的第三方模型与代码（Demucs、MelBand-RoFormer、ia-amt、SOME、basic-pitch、OSMD 等）遵循其各自原始许可，不在本项目的 MIT 授权范围内。
