# source_separation/ — 分轨模型与训练总目录

MUSE 全工作区的分离（source separation）资产唯一落脚点：自研 VER-SEP 系列、
demucs 微调老线（本项目 2026-05~08 的前身）、下载的开源预训练权重。
管线侧封装在 `score_extraction/src/versep_sep.py`（VER-SEP）与
`score_extraction/src/source_separate.py`（demucs）。

## versep/ — 自研 VER-SEP 系列（mel-band-roformer 吉他/其他二分，现役）

| 文件 | 说明 |
|---|---|
| `VERSEP2.0_roformer_guitar_ep9_sdr11.4248.ckpt` | **管线默认**。comp 硬区定向训练（GuitarSet comp × MoisesDB Other，warm start 1.1）；F2 112 对 SI-SDR +6.2dB、最差四分位 +10.4dB、下游 F1 0.531→0.597，2026-08-24 四口径验收 |
| `VERSEP1.1_roformer_guitar_ep4_sdr5.7463.ckpt` | 上一代（MoisesDB pilot 2338 对），A/B 对照臂 |
| `config_guitar_finetune_v1.yaml` | MSST 微调配置（adamw 2e-5，12ep×500 步，dataset_type 6，两代共用） |
| `config_guitar_finetune_local_smoke.yaml` | 本地冒烟版 |
| `launch_f2_finetune.sh` / `launch_f2_overnight.sh` / `launch_f2v2_finetune.sh` | 服务器训练启动脚本（dyylab） |
| `README_finetune.md` | 微调流水线文档（数据→mixture.wav 规范→训练→验收） |

运行时切换：环境变量 `MUSE_VERSEP_CKPT`（裸文件名在此目录解析，或绝对路径）。

## pretrained/ — 开源预训练权重

| 文件/目录 | 说明 |
|---|---|
| `htdemucs_6s.75fc33f5-1941ce65.th` | Meta htdemucs 六件套（drums/bass/other/vocals/piano/guitar），159MB。**运行时走 torch hub 缓存**（`~/.cache/torch/hub/checkpoints/75fc33f5-1941ce65.th`），此处为自包含副本+来源凭据 |
| `becruily_mel_band_roformer_guitar/` | HF becruily/mel-band-roformer-guitar（Guitar/Other 二分，**VER-SEP 微调基座**）。本地仅 config（`config_guitar_becruily.yaml`），基座权重在 dyylab `~/zmdld11/score_extraction/checkpoints/` |

## 老线（2026-05~08 demucs 微调前身，保留存档）

- `demucs-main/` — demucs 官方仓库完整克隆（训练代码）
- `data/demucs_format/` — pilot 训练数据（9.4G，train/valid）
- `model/` — 老 demucs 微调权重（`guitar.pth`、`checkpoint_latest.pth`，git 已追踪）
- `src/`、`output/`、`autodl_train.py`、`setup_autodl.sh` — AutoDL 时代训练脚手架

## 服务器对应（dyylab `~/zmdld11/score_extraction/`）

- `checkpoints/`：becruily 基座 + VER-SEP 1.1/2.0
- `output/sep_finetune_v2_comp/`：2.0 训练工作区（best = model_mel_band_roformer_ep_9_sdr_11.4248.ckpt）
- `data/sep_comp_v1/`：2.0 定向训练数据（1248 train + 96 valid）

## 历史注记（2026-08-24 整理）

- `score_extraction/separation/`（配置+启动脚本）并入 `versep/`；其中
  `deploy_iaamt_finetune.sh` 是 ia-amt 转写部署脚本，移至 `score_extraction/train/`。
- 本目录新权重（`pretrained/*.th`、`versep/*.ckpt`）不进库；老 `model/*.pth`
  已在库的不动。
