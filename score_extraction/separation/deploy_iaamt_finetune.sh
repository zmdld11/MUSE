#!/bin/bash
# ia-amt 伪影感知微调部署（2026-08-21，贡献②实验）
# 数据：F2 sep stems（VER-SEP 分离伪影域，1232 对训练切分）+ GuitarSet 干净 replay（players 00-03 ×2）
# 热启动：guitar_v1_5 官方检查点
set -e
SE=~/zmdld11/score_extraction
PY=$SE/.venv/bin/python
F2=$SE/data/f2_v1
DAT=$SE/data/f2_iaamt
IA=$SE/external/ia-amt

echo "== [1] 解包 =="
mkdir -p $SE/external $DAT
[ -d $IA ] || (cd $SE/external && tar xzf ia-amt-repo.tar.gz)
mkdir -p $DAT/gs_audio
[ -f $DAT/gs_audio/.done ] || (cd $DAT/gs_audio && tar xzf $SE/data/gs_clean_0003.tar.gz && touch .done)

echo "== [2] staging：F2 sep 训练切分（排除 02_*r7/r8 测试与 03_* 验证）==="
mkdir -p $DAT/stems_sep $DAT/mids_sep $DAT/stems_clean $DAT/mids_clean $DAT/npz
n_sep=0
for d in $F2/sep/*/; do
  pair=$(basename "$d")
  case "$pair" in 02_*r7|02_*r8|03_*) continue;; esac
  ln -sf "$d/Guitar.wav" "$DAT/stems_sep/$pair.wav"
  ln -sf "$F2/gt/$pair.mid" "$DAT/mids_sep/$pair.mid"
  n_sep=$((n_sep+1))
done
echo "sep stems: $n_sep"

echo "== [3] staging：干净 replay ×2 =="
declare -A seen
n_clean=0
for m in $DAT/mids_sep/*.mid; do
  pair=$(basename "$m" .mid)
  clip=$(echo "$pair" | sed 's/_r[0-9]*$//')      # 00_BN1-129-Eb_comp
  [ -n "${seen[$clip]}" ] && continue
  seen[$clip]=1
  src_wav=$DAT/gs_audio/${clip}_mix.wav
  [ -f "$src_wav" ] || { echo "MISSING $src_wav"; continue; }
  for tag in A B; do
    ln -sf "$src_wav" "$DAT/stems_clean/${clip}_${tag}.wav"
    ln -sf "$m" "$DAT/mids_clean/${clip}_${tag}.mid"
    n_clean=$((n_clean+1))
  done
done
echo "clean stems: $n_clean"

echo "== [4] prepare_dataset（npz 标签 + manifest）==="
cd $IA
[ -f $DAT/manifest_sep.csv ] || $PY preprocess/prepare_dataset.py \
  --midis_dir $DAT/mids_sep --stems_dir $DAT/stems_sep \
  --npz_dir $DAT/npz --manifest_path $DAT/manifest_sep.csv --workers 8
[ -f $DAT/manifest_clean.csv ] || $PY preprocess/prepare_dataset.py \
  --midis_dir $DAT/mids_clean --stems_dir $DAT/stems_clean \
  --npz_dir $DAT/npz --manifest_path $DAT/manifest_clean.csv --workers 8
head -1 $DAT/manifest_sep.csv > $DAT/manifest_all.csv
tail -n +2 $DAT/manifest_sep.csv >> $DAT/manifest_all.csv
tail -n +2 $DAT/manifest_clean.csv >> $DAT/manifest_all.csv
wc -l $DAT/manifest_all.csv

echo "== [5] dataset_config + 启动微调 =="
mkdir -p $IA/configs/datasets
printf 'datasets: []\n' > $IA/configs/datasets/f2_finetune.yaml
cd $IA
CUDA_VISIBLE_DEVICES=0 nohup $PY train.py \
  --manifest_path $DAT/manifest_all.csv \
  --dataset_config configs/datasets/f2_finetune.yaml \
  --init_from $SE/external/best_model_guitar_v1_5.pth \
  --lr 5e-5 --warmup_steps 200 --epochs 100 --save_interval 10 \
  --batch_size 8 --num_workers 4 \
  --save_dir $DAT/checkpoints \
  --run_name f2_artifact_finetune_v1 \
  > $SE/data/iaamt_finetune.log 2>&1 &
echo "LAUNCHED pid=$!"
sleep 30
tail -5 $SE/data/iaamt_finetune.log
