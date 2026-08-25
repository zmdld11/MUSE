#!/usr/bin/env bash
# VER-Guitar F2 正式微调：全量打包 -> ByteDance 训练框架 warm start Riley guitar-fl
set -euo pipefail
SE=~/zmdld11/score_extraction
PY=$SE/.venv/bin/python
WS=~/workspaces/f2_v1

echo "[1/3] packing 1792 pairs..."
cd $SE/external/bytedance_pt
$PY pack_f2.py --f2_dir $SE/data/f2_v1 --workspace $WS

echo "[2/3] training (warm start guitar-fl, lr 5e-5, bs 8, 10k iters, ckpt every 5k)..."
cd $SE/external/bytedance_pt/pytorch
$PY main.py train \
  --workspace $WS \
  --model_type Regress_onset_offset_frame_velocity_CRNN \
  --loss_type regress_onset_offset_frame_velocity_bce \
  --augmentation none \
  --max_note_shift 0 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --reduce_iteration 2000 \
  --resume_iteration 0 \
  --early_stop 10001 \
  --cuda \
  --init_model_path $SE/model/external/guitar-fl.pth

echo "F2_FINETUNE_DONE"
