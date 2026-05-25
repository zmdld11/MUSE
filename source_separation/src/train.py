# train.py — DemucsLM 时域训练主循环
# VER3.0: 端到端波形 L1 损失，无需 STFT
import os, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src.config import config
from src.model import DemucsLM
from src.dataset import GuitarSeparationDataset


def compute_sdr(est, target, eps=1e-8):
    err = est - target
    s = (target ** 2).sum()
    e = (err ** 2).sum()
    if s < eps: return 0.0  # target 全静音 → SDR 无意义
    return (10 * torch.log10(s / torch.clamp(e, min=eps))).item()


def train(args):
    device = config.DEVICE
    print(f"设备: {device}")
    print(f"版本: {config.MODEL_VERSION}")

    data_dir = config.DATASET_DIR
    train_ds = GuitarSeparationDataset(data_dir, "train", augment=True)
    val_ds = GuitarSeparationDataset(data_dir, "val", augment=False)
    print(f"训练: {len(train_ds)}, 验证: {len(val_ds)}")

    tl = DataLoader(train_ds, config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    vl = DataLoader(val_ds, config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = DemucsLM(channels=config.DEMUCS_CHANNELS).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_p:,}")

    criterion = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)

    ckpt_path = os.path.join(config.MODEL_DIR, "checkpoint_latest.pth")
    start_epoch = 0
    best_sdr = -float('inf')
    train_losses, val_sdrs = [], []

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"]
        best_sdr = ck.get("best_val_sdr", -float('inf'))
        train_losses = ck.get("train_losses", [])
        val_sdrs = ck.get("val_sdrs", [])
        print(f"恢复 epoch {start_epoch}, 最佳 SDR={best_sdr:.2f} dB")

    log_path = os.path.join(config.LOG_DIR, f"{datetime.now():%Y%m%d-%H%M%S}.log")
    with open(log_path, "a") as f:
        f.write(f"{'='*50}\nModel: {config.MODEL_VERSION}\nParams: {n_p:,}\nTrain: {len(train_ds)}, Val: {len(val_ds)}\n")
        f.write("Epoch\tTrain_L1\tVal_SDR(dB)\tLR\n")

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_loss = 0.0
        for mix, gtr in tqdm(tl, desc=f"E{epoch+1}/{config.EPOCHS}", leave=False):
            mix, gtr = mix.to(device, non_blocking=True), gtr.to(device, non_blocking=True)
            mix = mix.unsqueeze(1)  # [B, 1, T]
            gtr = gtr.unsqueeze(1)

            opt.zero_grad()
            pred = model(mix)  # [B, 1, T]
            loss = criterion(pred, gtr)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(tl)

        model.eval()
        val_sdr = 0.0
        with torch.no_grad():
            for mix, gtr in tqdm(vl, desc=f"V{epoch+1}", leave=False):
                mix, gtr = mix.to(device, non_blocking=True), gtr.to(device, non_blocking=True)
                pred = model(mix.unsqueeze(1))
                val_sdr += compute_sdr(pred, gtr.unsqueeze(1)) * mix.size(0)
        val_sdr /= len(val_ds)
        lr = opt.param_groups[0]['lr']
        sched.step(-val_sdr)

        print(f"  E{epoch+1}: L1={train_loss:.4f}  SDR={val_sdr:.2f} dB  LR={lr:.1e}")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_sdr:.2f}\t{lr:.1e}\n")

        train_losses.append(train_loss)
        val_sdrs.append(val_sdr)

        if val_sdr > best_sdr:
            best_sdr = val_sdr
            torch.save({"model_state_dict": model.state_dict(), "val_sdr": val_sdr, "version": config.MODEL_VERSION},
                       os.path.join(config.MODEL_DIR, "guitar.pth"))
            print(f"  → 保存 (SDR={val_sdr:.2f})")

        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(),
                    "best_val_sdr": best_sdr, "train_losses": train_losses, "val_sdrs": val_sdrs}, ckpt_path)

        if epoch - max(enumerate(val_sdrs), key=lambda x: x[1])[0] >= config.EARLY_STOPPING_PATIENCE:
            print(f"早停 at epoch {epoch+1}"); break

    print(f"\n最佳 SDR: {best_sdr:.2f} dB")
    with open(log_path, "a") as f:
        f.write(f"Best Val SDR: {best_sdr:.2f} dB\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    config.EPOCHS = a.epochs
    train(a)
