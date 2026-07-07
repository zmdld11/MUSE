#!/bin/bash
# AutoDL RTX 5090 环境配置脚本
# 使用: bash setup_autodl.sh

set -e
echo "=== MUSE 音轨分离 - AutoDL 环境配置 ==="

# ====== 1. 系统库 (soundfile 依赖) ======
echo "安装系统依赖..."
apt-get update -qq && apt-get install -y -qq libsndfile1 2>/dev/null

# ====== 2. 检测数据路径 ======
DATA_DIR="/root/autodl-tmp/source_separation"
if [ ! -d "$DATA_DIR" ]; then
    DATA_DIR="/root/source_separation"
fi
if [ ! -d "$DATA_DIR/data/demucs_format" ]; then
    echo "错误: 找不到 demucs_format 数据文件夹！"
    echo "请确保 source_separation/ 上传到了 AutoDL"
    echo "检查: $DATA_DIR"
    exit 1
fi
echo "数据目录: $DATA_DIR"
cd "$DATA_DIR"

# ====== 3. conda 环境 ======
echo "创建 conda 环境 (Python 3.10)..."
conda create -n muse_sep python=3.10 -y 2>/dev/null || echo "(环境已存在)"
source activate muse_sep || conda activate muse_sep

# ====== 4. PyTorch (CUDA 12.1) ======
echo "安装 PyTorch + torchaudio..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# ====== 5. Python 依赖 ======
echo "安装 Python 依赖..."
pip install -r requirements.txt

# ====== 6. 验证 ======
echo ""
echo "--- 环境验证 ---"
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'显存: {torch.cuda.get_device_properties(0).total_mem/1024**3:.0f} GB')
"
python -c "
from src.model import DemucsLM
m = DemucsLM(channels=(48,96,192,384), rescale=0.1)
print(f'模型: {sum(p.numel() for p in m.parameters()):,} 参数')
"
python -c "
from src.demucs_dataset import DemucsGuitarDataset
ds = DemucsGuitarDataset('data/demucs_format/train', segment=6.0, shift=1.0, sr=22050, augment=False)
print(f'数据集: {len(ds)} 片段, {len(ds.songs)} 首歌')
"

echo ""
echo "=== 环境配置完成 ==="
echo ""
echo "本地4060训练:  python -m src.train --epochs 100"
echo "5090全量训练:  python autodl_train.py"
echo ""
echo "后台运行:"
echo "  nohup python autodl_train.py > train.log 2>&1 &"
echo "  tail -f train.log"
