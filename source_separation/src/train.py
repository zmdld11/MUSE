# train.py — DemucsLM 时域训练主循环
# VER5.0: 在线随机混音 + MRSTFT 混合损失 + 4层模型
import os, argparse, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src.config import config
from src.model import DemucsLM
from src.dataset import GuitarSeparationDataset
from src.remix_dataset import RemixDataset


def compute_sdr(est, target, eps=1e-8):
    err = est - target
    s = (target ** 2).sum()
    e = (err ** 2).sum()
    if s < eps: return 0.0
    return (10 * torch.log10(s / torch.clamp(e, min=eps))).item()


def compute_mrstft(est, target, fft_sizes, device):
    loss = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=device)
        est_spec = torch.stft(est.squeeze(1), n_fft=n_fft, hop_length=hop,
                              win_length=n_fft, window=window, return_complex=True)
        tgt_spec = torch.stft(target.squeeze(1), n_fft=n_fft, hop_length=hop,
                              win_length=n_fft, window=window, return_complex=True)
        est_mag = torch.sqrt(est_spec.real ** 2 + est_spec.imag ** 2 + 1e-8)
        tgt_mag = torch.sqrt(tgt_spec.real ** 2 + tgt_spec.imag ** 2 + 1e-8)
        loss += F.l1_loss(est_mag, tgt_mag)
    return loss / len(fft_sizes)


def train(args):
    device = config.DEVICE
    print(f"设备: {device}")
    print(f"版本: {config.MODEL_VERSION}")
    print(f"损失: L1 + {config.MRSTFT_WEIGHT}×MRSTFT{config.MRSTFT_FFT_SIZES}")
    print(f"模型: {len(config.DEMUCS_CHANNELS)}层, 通道{config.DEMUCS_CHANNELS}")
    print(f"输入: {config.NUM_SAMPLES} 样本 ({config.NUM_SAMPLES/config.SR:.2f}s)")

    # 构建 RemixDataset 作为增强源 (50% 概率随机混音切断捷径)
    remix_ds = RemixDataset(
        medleydb_dir=config.MEDLEYDB_DIR,
        medleydb_meta_dir=config.MEDLEYDB_META_DIR,
        moisesdb_dir=config.MOISESDB_DIR,
        num_samples=config.NUM_SAMPLES,
        sr=config.SR,
        num_total=config.REMIX_TOTAL,
        cache_path=config.STEM_INDEX_CACHE,
    )
    # 训练集: 50% 真实预计算数据 + 50% 随机混音
    train_ds = GuitarSeparationDataset(
        config.DATASET_DIR, "train", augment=True,
        remix_dataset=remix_ds, remix_prob=0.5,
        num_samples=config.NUM_SAMPLES,
    )
    # 验证集: 纯真实预计算数据
    val_ds = GuitarSeparationDataset(
        config.DATASET_DIR, "val", augment=False,
        num_samples=config.NUM_SAMPLES,
    )
    print(f"训练: {len(train_ds)} (50%真实+50%混音), 验证: {len(val_ds)} (纯真实)")

    tl = DataLoader(train_ds, config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    vl = DataLoader(val_ds, config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = DemucsLM(channels=config.DEMUCS_CHANNELS).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_p:,}")

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
        f.write(f"{'='*50}\nModel: {config.MODEL_VERSION}\n")
        f.write(f"Arch: {len(config.DEMUCS_CHANNELS)}层, 通道{config.DEMUCS_CHANNELS}\n")
        f.write(f"Params: {n_p:,}\n")
        f.write(f"Input: {config.NUM_SAMPLES} samples, Train: {len(train_ds)} (50%real+50%remix), Val: {len(val_ds)}\n")
        f.write(f"Loss: L1 + {config.MRSTFT_WEIGHT}×MRSTFT{config.MRSTFT_FFT_SIZES}\n")
        f.write("Epoch\tTrain_Loss\tVal_SDR(dB)\tTime(s)\tLR\n")

    for epoch in range(start_epoch, config.EPOCHS):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for mix, gtr in tqdm(tl, desc=f"E{epoch+1}/{config.EPOCHS}", leave=False):
            mix, gtr = mix.to(device, non_blocking=True), gtr.to(device, non_blocking=True)
            mix = mix.unsqueeze(1)  # [B, T] → [B, 1, T]
            gtr = gtr.unsqueeze(1)

            opt.zero_grad()
            pred = model(mix)

            l1_loss = F.l1_loss(pred, gtr)
            mrstft_loss = compute_mrstft(pred, gtr, config.MRSTFT_FFT_SIZES, device)
            loss = l1_loss + config.MRSTFT_WEIGHT * mrstft_loss

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

        elapsed = time.time() - t0

        print(f"  E{epoch+1}: Loss={train_loss:.4f}  SDR={val_sdr:.2f} dB  Time={elapsed:.0f}s  LR={lr:.1e}")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_sdr:.2f}\t{elapsed:.0f}\t{lr:.1e}\n")

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
