#!/usr/bin/env bash
# VER-Guitar F2 v2：混合 replay 训练（F2 分离数据 + 干净 GuitarSet 重放）
set -euo pipefail
SE=~/zmdld11/score_extraction
PY=$SE/.venv/bin/python
WS=~/workspaces/f2_v2

echo "[1/3] preparing hdf5s (copy F2 v1 + pack clean replay)..."
mkdir -p $WS/hdf5s/maestro
cp -rn ~/workspaces/f2_v1/hdf5s/maestro/2026 $WS/hdf5s/maestro/ 2>/dev/null || true
cd $SE/external/bytedance_pt
$PY pack_clean_replay.py --guitarset_dir $SE/data/guitarset --workspace $WS --copies 6

echo "[2/3] training v2 (warm start guitar-fl, F2+replay mixed, 10k iters)..."
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

echo "F2V2_FINETUNE_DONE"
