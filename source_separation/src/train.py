# train.py — LightweightUMX 吉他分离训练主循环
# 用法: python -m src.train [--epochs 100] [--resume]
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src.config import config
from src.model import LightweightUMX
from src.dataset import GuitarSeparationDataset


def train(args):
    device = config.DEVICE
    print(f"设备: {device}")
    print(f"版本: {config.MODEL_VERSION}")

    # 数据集
    metadata_path = os.path.join(config.DATASET_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"错误: 数据集元数据不存在 {metadata_path}")
        print("请先运行 data/build_guitar_separation_dataset.py")
        return

    train_dataset = GuitarSeparationDataset(
        metadata_path, os.path.join(config.DATASET_DIR, "audio"), augment=True
    )
    # 验证集用原始样本(不含增强)，构造一个不含augment的版本
    val_dataset = GuitarSeparationDataset(
        metadata_path, os.path.join(config.DATASET_DIR, "audio"), augment=False
    )

    # 从 metadata 重新读取以正确切分 train/val
    import json
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    train_dataset.samples = meta["train_samples"]
    val_dataset.samples = meta["val_samples"]

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # 模型
    model = LightweightUMX(
        n_bins=config.N_BINS,
        hidden=config.BLSTM_HIDDEN,
        num_layers=config.BLSTM_LAYERS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    # 损失、优化器、调度器
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE
    )

    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    # 恢复训练
    ckpt_path = os.path.join(config.MODEL_DIR, "checkpoint_latest.pth")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt["best_val_loss"]
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        print(f"从 epoch {start_epoch} 恢复, 最佳 Val Loss: {best_val_loss:.4f}")

    # 日志
    log_path = os.path.join(
        config.LOG_DIR, f"{datetime.now():%Y%m%d-%H%M%S}.log"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{'='*50}\n")
        f.write(f"Model: {config.MODEL_VERSION}\n")
        f.write(f"Target: {config.TARGET_INSTRUMENT}\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Val samples: {len(val_dataset)}\n")
        f.write("Epoch\tTrain_L1\tVal_L1\tLR\n")

    # 训练循环
    for epoch in range(start_epoch, config.EPOCHS):
        # —— Train ——
        model.train()
        train_loss = 0.0
        for mix_mag, _, target_mask in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False
        ):
            mix_mag = mix_mag.to(device)        # [B, F, T]
            target_mask = target_mask.to(device) # [B, F, T]

            optimizer.zero_grad()
            pred_mask = model(mix_mag)           # [B, F, T]
            loss = criterion(pred_mask, target_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # —— Val ——
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix_mag, _, target_mask in val_loader:
                mix_mag = mix_mag.to(device)
                target_mask = target_mask.to(device)
                pred_mask = model(mix_mag)
                val_loss += criterion(pred_mask, target_mask).item()

        val_loss /= len(val_loader)

        # 调度器
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # 日志
        print(f"  Epoch {epoch+1}: Train L1={train_loss:.4f}, Val L1={val_loss:.4f}, LR={current_lr:.2e}")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\t{current_lr:.2e}\n")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "version": config.MODEL_VERSION,
            }, os.path.join(config.MODEL_DIR, "guitar.pth"))
            print(f"    → 保存最佳模型 (Val L1={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # 保存断点
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }, ckpt_path)

        # Early stopping
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {epochs_no_improve} epochs)")
            break

    # 结束
    print(f"\n训练完成! 最佳 Val L1: {best_val_loss:.4f}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Best Val L1: {best_val_loss:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config.EPOCHS = args.epochs
    train(args)
