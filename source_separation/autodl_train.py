#!/usr/bin/env python
# autodl_train.py — AutoDL RTX 5090 专用训练入口
# 覆盖 config 以大 GPU 参数运行, 直接: python autodl_train.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.config import config

# ====== RTX 5090 32GB 专属配置 ======
config.MODEL_VERSION = "VER6.0_AutoDL_5090"
config.SEGMENT = 11.0           # 对齐 Demucs 原版
config.SHIFT = 3.0
config.DEMUCS_CHANNELS = (48, 96, 192, 384)  # 当前4层架构
config.BATCH_SIZE = 24          # 5090 32GB 轻松跑
config.EPOCHS = 360             # 对标 Demucs
config.GRAD_CLIP = 1.0

print(f"=== AutoDL RTX 5090 训练 ===")
print(f"版本: {config.MODEL_VERSION}")
print(f"片段: {config.SEGMENT}s, Batch: {config.BATCH_SIZE}, Epochs: {config.EPOCHS}")
print(f"设备: {config.DEVICE}")
if config.DEVICE.type == 'cuda':
    print(f"GPU: {config.DEVICE}")

from src.train import train
import argparse
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=config.EPOCHS)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
config.EPOCHS = a.epochs
train(a)
