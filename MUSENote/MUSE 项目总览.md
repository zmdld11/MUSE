1# MUSE 项目总览

> Music Unmixing & Score Extraction — 大创项目
> 将任意音频自动转化为可读乐谱的完整工具链

## 项目架构

```
音频输入 (.wav/.mp3/.flac)
     │
     ├── [[1. 乐器识别]]     → 识别歌曲中出现的乐器及时间段
     ├── [[2. 音轨分离]]     → 分离出独立乐器音轨
     ├── [[3. 乐谱生成]]     → 音轨 → 五线谱/六线谱
     └── [[4. 前端交互]]     → Vue3 用户界面（未启动）
```

## 技术栈

| 层次 | 技术 |
|------|------|
| 深度学习框架 | PyTorch, CUDA (RTX 4060) |
| 音频处理 | librosa, torchaudio, pretty_midi, music21 |
| 预训练模型 | htdemucs_6s (音轨分离), basic-pitch (转录) |
| 训练数据 | MedleyDB, GiantMIDI-Piano, FluidSynth 渲染 |
| 前端 | Vue 3 + Vite + vue-router（计划） |
| 乐谱标准 | MusicXML, MIDI |

## 对外接口

- [[binfer_cli]] — 乐器识别命令行推理
- [[pipeline.py]] — 乐谱生成主入口
- [[run.py]] — 乐谱生成一键运行脚本

## 关联

- [[技术白皮书]] — 乐谱生成核心技术详解
- [[项目日志]] — 开发时间线
- [[AMT 论文综述]] — 学术背景
