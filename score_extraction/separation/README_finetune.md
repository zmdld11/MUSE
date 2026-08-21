# VER-SEP 1.0：吉他分离器 MoisesDB 域微调（becruily Mel-Band RoFormer）

> **2026-08-19 起默认配方升级为 VER-SEP 1.1**：ckpt 换
> `model_mel_band_roformer_ep_4_sdr_5.7463.ckpt`（失真加权续训；本地
> `output/sep_finetune_v1_local/`，服务器 `output/sep_finetune_v1b/`），推理加
> `--bigshifts 4`（失真 +0.94dB 清音无损；bs8 回退勿用）。

Route B 阶段 14。目标：不区分主音/节奏，把「吉他整体」从真实混音中提取得比
当前默认分离器更干净，供下游吉他 AMT 使用。基线 checkpoint：
`score_extraction/model/external/becruily_mel_band_roformer_guitar/becruily_guitar.ckpt`
（Guitar/Other 二分，22.4M 参数，MSST 官方代码严格加载通过）。

## 数据（本地已生成并校验）

- 根目录：`data/moisesdb_guitar_pilot_v1/`（实际 2338 train + 252 valid chunk，共
  9.82GB；受「同曲窗口互不重叠 + 吉他活跃」的真实可用量限制，低于 3000/300 目标，
  约 7.1h 训练 + 0.8h 验证音频）
- 来源：MoisesDB v0.1 的 222 首含吉他歌曲（按 `stemName == "guitar"` 精确筛选，
  不含 bass_guitar）。20 首按 genre 分层固定为验证集，其余 202 首为训练集。
- 规格：44.1kHz / stereo / PCM_16 / 485100 样本（11s，与模型 `audio.chunk_size`
  一致，训练零 padding）。只选吉他活跃窗口（窗口 RMS ≥ max(0.004, 0.25×该曲
  p75)），同曲窗口互不重叠。
- 每个文件夹：`Guitar.wav` + `Other.wav`（train）；valid 额外含 `mixture.wav`
  （MSST `valid.py` 按 mixture.wav 搜索）。train 的 mixture 由 dataloader 求和。
- `meta.json`（每文件夹）与 `manifest.json`（全局）记录歌曲、起点、缩放因子、
  吉他类型、bleed 标记；缩放仅在峰值>1 时同因子作用于 Guitar+Other，不破坏目标
  对应关系。抽查重算与原始分轨逐样本一致（误差 = 16-bit 1 LSB）。
- 生成/续跑/校验：
  ```
  ./env/python.exe score_extraction/train/generate_separation_pilot.py            # 全量（断点续跑）
  ./env/python.exe score_extraction/train/generate_separation_pilot.py --verify   # 抽查重算比对
  ./env/python.exe score_extraction/train/generate_separation_pilot.py --survey-only
  ```

## 本地冒烟（RTX 4060 8GB，已通过 2026-08-18）

依赖已装入项目 env：wandb、ml_collections、einops、omegaconf、beartype、
prodict、audiomentations、pedalboard、auraloss、torch_log_wmse、
rotary_embedding_torch、prodigyopt。运行时需把 `env/bin` 加进 PATH（nvrtc DLL）。

已打一处补丁（必须，否则 num_stems=1 的内部 multi-STFT loss 维度报错）：
`external/Music-Source-Separation-Training/models/bs_roformer/mel_band_roformer.py`
forward 内 multi-stft 循环对 3 维 recon_audio/target_sel 先补回 stem 维。同步到
服务器时用本地这份 repo，勿重新 clone 覆盖。

```
cd score_extraction/external/Music-Source-Separation-Training
PATH="/d/program_project/MUSE/env/bin:$PATH" WANDB_MODE=disabled \
D:/program_project/MUSE/env/python.exe train.py \
  --model_type mel_band_roformer \
  --config ../../separation/config_guitar_finetune_local_smoke.yaml \
  --data_path ../../../data/moisesdb_guitar_pilot_smoke/train \
  --valid_path ../../../data/moisesdb_guitar_pilot_smoke/valid \
  --results_path ../../../score_extraction/output/sep_finetune_smoke \
  --dataset_type 6 \
  --start_check_point ../../model/external/becruily_mel_band_roformer_guitar/becruily_guitar.ckpt \
  --device_ids 0
```

结果：3 步训练 loss 有限；valid 输出 Guitar SDR 5.74 / k_SDR 92.5；checkpoint
正常保存。mel_band_roformer 默认使用模型内部 loss（L1 + multi-res STFT），与
becruily 原训练一致，不要传 `--loss` 覆盖。

## dyylab 服务器正式微调

1. 上传（校园网可用时，约 10GB，只传派生数据、不传原始 MoisesDB；先 `df -h`
   确认空间，紧张时先 `du -sh` 列清单再决定清理项，不做盲删）：
   ```
   rsync -aP data/moisesdb_guitar_pilot_v1/{train,valid,manifest.json} \
     dyylab:~/zmdld11/score_extraction/data/moisesdb_guitar_pilot_v1/
   ```
2. MSST 代码与依赖（服务器若没有；必须用本地这份含补丁的 repo）：
   ```
   rsync -aP score_extraction/external/Music-Source-Separation-Training dyylab:~/zmdld11/score_extraction/external/
   # 服务器环境: pip install -r Music-Source-Separation-Training/requirements.txt
   ```
   同时上传 `model/external/becruily_mel_band_roformer_guitar/`（ckpt+config）与
   `score_extraction/separation/` 两个 yaml。
3. 启动（单卡 A5000，nohup，日志+pid 落盘；不传 --loss，保持模型内部 loss）：
   ```
   cd ~/zmdld11/score_extraction/external/Music-Source-Separation-Training
   nohup python train.py \
     --model_type mel_band_roformer \
     --config ../../separation/config_guitar_finetune_v1.yaml \
     --data_path ../../data/moisesdb_guitar_pilot_v1/train \
     --valid_path ../../data/moisesdb_guitar_pilot_v1/valid \
     --results_path ../../output/sep_finetune_v1 \
     --dataset_type 6 \
     --start_check_point ../../model/external/becruily_mel_band_roformer_guitar/becruily_guitar.ckpt \
     --num_workers 4 --device_ids 0 > ../../output/sep_finetune_v1.log 2>&1 &
   echo $! > ../../output/sep_finetune_v1.pid
   ```
   规模：12 epoch × 500 step × batch2+accum2 ≈ 6000 步；**adamw lr 2e-5**（prodigy
   lr 1.0 是从零训练配方，微调 1 个 epoch 内即毁权重，勿改回）。增强必须保持
   `enable: false`：此 fork 的 dataset_type 6 先 sum 出 mix 再对 stem 独立增强，
   (mix,target) 不自洽（实测 500 步内 val SDR 4.85→0.04，与 lr 无关）。
   启动务必带 `--pre_valid`：becruily 原始 ckpt 在本验证集 SDR=4.8508，
   重启后 epoch0 SDR 低于此值即说明配置又出问题。

## 评测口径（微调是否有用的判定）

不只看 SI-SDR，按 Route B 的下游标准：

1. MoisesDB 验证 20 首：Guitar SI-SDR / SDR，与基线 ckpt 同口径对比。
2. 东方之空完整曲：微调前后 stem 各跑一次下游评测（固定 -70.5887s 对齐），
   对比 any-pitch onset@50 F1（基线 0.3448）与 strict note@50（基线 0.0843）。
3. 人工 A/B 试听：节奏吉他段的糊度、失真吉他 attack、伪影。
4. 判定：下游 onset/note F1 不降且听感更好 → 采纳为默认分离器；仅 SI-SDR 涨
   而下游不涨 → 记录但不再扩大该方向。

## 已知问题与后续

- 「糊混音」是域问题：录音室混音本身吉他频谱重叠严重（bus 压缩、串音、bleed）。
  MoisesDB 197 条轨带 has_bleed 标记，本 pilot 保留它们（manifest 有记录），
  后续可做 bleed 分层评测。针对性方向：混音风格增广（EQ/压缩/失真链模拟）、
  低码率/MP3 增广、以及最终在下游 AMT 训练里直接吃「糊」stem 的域适应。
- MoisesDB 授权 CC BY-NC-SA：研究用途，派生模型不分发。
- 若 pilot 有收益，v2 扩到 10k-20k chunks 并加入 MedleyDB 等外部混音域。
