#!/usr/bin/env bash
# F2 overnight：生成混音 -> VER-SEP 1.1 + bigshifts4 全量分离（无人值守）
set -euo pipefail
SE=~/zmdld11/score_extraction
PY=$SE/.venv/bin/python
cd $SE
echo "[1/2] generating mixes..."
$PY train/gen_f2_mixes.py --reps 8 --seed 20260819 2>&1
echo "[2/2] separating with VER-SEP 1.1 + bs4..."
cd $SE/external/Music-Source-Separation-Training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY inference.py --model_type mel_band_roformer \
  --config_path $SE/separation/config_guitar_finetune_v1.yaml \
  --start_check_point $SE/output/sep_finetune_v1b/model_mel_band_roformer_ep_4_sdr_5.7463.ckpt \
  --input_folder $SE/data/f2_v1/mix \
  --store_dir $SE/data/f2_v1/sep \
  --bigshifts 4 --device_ids 0
echo "F2_OVERNIGHT_DONE"
