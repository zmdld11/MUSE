# AMT 论文综述

> 自动音乐转录 (Automatic Music Transcription) 学术背景

## 已阅读论文

| 论文 | 出处 | 核心贡献 |
|------|------|---------|
| NoteEM | ICML 2022 | EM 框架：未对齐乐谱 + 合成数据 → 自训练转录器 |
| AMT Overview | IEEE SPM 2019 | 四层 AMT 体系：frame→note→stream→notation |
| Self-Attention AMT | TASLP 2020 | 实例分割范式：self-attention + 多任务学习 |
| ML in AMT Survey | - | 系统性综述 |

## 吉他方向

| 论文 | 核心贡献 |
|------|---------|
| TART | 四阶段吉他转录：MIDI→技巧→弦位→六线谱 |
| Spotify 轻量模型 | 乐器无关 AMT，onset+frame+note 联合预测 |
| High-Res Guitar | 域适应：钢琴模型迁移到吉他 |

## 关键技术概念

- [[Onsets and Frames]] — 分离 onset/frame 检测的双头架构
- [[HMM 平滑]] — 前向-后向消除逐帧闪烁
- [[HarmonicStack]] — 频率轴 shift 拼接捕捉泛音
- [[DTW 对齐]] — 动态时间规整对齐预测与乐谱
- [[NoteEM]] — EM 训练框架（Bootstrap + E-step + M-step）
- [[Spectral Flux]] — 频谱能量跳变检测精确 onset
- [[Krumhansl-Schmuckler]] — 24 调相关性调号推算

## 关联

- [[技术白皮书]]
- [[乐谱生成总览]]
- [[model.py (训练)]]
