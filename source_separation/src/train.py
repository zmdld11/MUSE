# train.py — UNet 吉他分离训练主循环
# VER2.0: 复数 2D U-Net + 复数谱输入
import os, argparse, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src.config import config
from src.model import UNet
from src.dataset import GuitarSeparationDataset

N_FFT = 1024
HOP = 256
GPU_WINDOW = None


def _stft(audio):
    """GPU 复数 STFT [B, T] → (mag [B, F, T], real [B, F, T], imag [B, F, T])"""
    global GPU_WINDOW
    d = audio.device
    if GPU_WINDOW is None or GPU_WINDOW.device != d:
        GPU_WINDOW = torch.hann_window(N_FFT, device=d)
    X = torch.stft(audio, N_FFT, HOP, window=GPU_WINDOW, return_complex=True)
    mag = torch.sqrt(X.real ** 2 + X.imag ** 2 + 1e-8)
    return mag, X.real, X.imag


def apply_complex_mask(comp_mask, real, imag):
    """(a+jb)(c+jd) = (ac-bd) + j(ad+bc)"""
    r = comp_mask[:, 0] * real - comp_mask[:, 1] * imag
    i = comp_mask[:, 0] * imag + comp_mask[:, 1] * real
    return r, i


def compute_sdr(est_mag, target_mag):
    err = est_mag - target_mag
    return (10 * torch.log10((target_mag ** 2).sum() / torch.clamp((err ** 2).sum(), min=1e-8))).item()


def train(args):
    device = config.DEVICE
    print(f"设备: {device}")
    print(f"版本: {config.MODEL_VERSION}")

    data_dir = config.DATASET_DIR
    if not os.path.exists(os.path.join(data_dir, "metadata.json")):
        print(f"错误: 数据集不存在 {data_dir}")
        print("请先运行 python data/build_dataset.py")
        return

    train_ds = GuitarSeparationDataset(data_dir, "train", augment=True)
    val_ds = GuitarSeparationDataset(data_dir, "val", augment=False)
    print(f"训练: {len(train_ds)}, 验证: {len(val_ds)}")

    tl = DataLoader(train_ds, config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    vl = DataLoader(val_ds, config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = UNet(channels=config.UNET_CHANNELS).to(device)
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
        # —— Train ——
        model.train()
        train_loss = 0.0
        for mix, gtr in tqdm(tl, desc=f"E{epoch+1}/{config.EPOCHS}", leave=False):
            mix, gtr = mix.to(device, non_blocking=True), gtr.to(device, non_blocking=True)

            # GPU 复数 STFT
            mix_mag, mix_r, mix_i = _stft(mix)
            _, gtr_r, gtr_i = _stft(gtr)
            gtr_mag = torch.sqrt(gtr_r ** 2 + gtr_i ** 2 + 1e-8)

            # U-Net 输入: [B, 2, F, T]
            spec_in = torch.stack([mix_r, mix_i], dim=1)
            mask = model(spec_in)

            # 复数掩码 → 重建吉他
            r_hat, i_hat = apply_complex_mask(mask, mix_r, mix_i)
            recon_mag = torch.sqrt(r_hat ** 2 + i_hat ** 2 + 1e-8)

            loss = criterion(recon_mag, gtr_mag)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(tl)

        # —— Val ——
        model.eval()
        val_sdr = 0.0
        with torch.no_grad():
            for mix, gtr in tqdm(vl, desc=f"V{epoch+1}", leave=False):
                mix, gtr = mix.to(device, non_blocking=True), gtr.to(device, non_blocking=True)
                mix_mag, mix_r, mix_i = _stft(mix)
                _, gtr_r, gtr_i = _stft(gtr)
                gtr_mag = torch.sqrt(gtr_r ** 2 + gtr_i ** 2 + 1e-8)

                mask = model(torch.stack([mix_r, mix_i], dim=1))
                r_hat, i_hat = apply_complex_mask(mask, mix_r, mix_i)
                recon_mag = torch.sqrt(r_hat ** 2 + i_hat ** 2 + 1e-8)
                val_sdr += compute_sdr(recon_mag, gtr_mag) * mix.size(0)
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
