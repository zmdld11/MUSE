"""全量 GiantMIDI 训练脚本 (2026-08-01 一夜训练).

用法:
  python train/train_overnight.py --n-files 10854 --epochs 50 --batch 8 --save VER2.2_BootstrapFull
  python train/train_overnight.py --resume model/VER2.2_BootstrapFull_latest.pt --epochs 50

特性:
  - 每 epoch 存 checkpoint (latest.pt 含优化器状态, 可断点续训)
  - 按 val_loss 保存 best
  - 绝不覆盖 VER2.0_Bootstrap.pth (基线)
  - 启动即修复 numba 缓存挂起 (同 eval.py)
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# numba 缓存/临时目录修复 (首次 librosa.stft 前必须生效)
_TMP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval", "reports", "tmp"))
os.makedirs(_TMP, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_TMP, "numba_cache"))
os.environ.setdefault("TMP", _TMP)
os.environ.setdefault("TEMP", _TMP)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.config import train_config as cfg
from train.dataset import PianoSynthDataset, collate_variable_length
from train.model import OnsetsAndFrames, onset_frame_loss

_LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "log_overnight.log"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_overnight")

MAX_DUR = 30  # 与现有缓存一致


def compute_metrics(pred, target):
    with torch.no_grad():
        frame_pred = (pred["frame"] > 0.5).float()
        frame_acc = (frame_pred == target["frame"]).float().mean().item()
        onset_pred = (pred["onset"] > 0.5).float()
        onset_precision = (onset_pred * target["onset"]).sum() / (onset_pred.sum() + 1e-8)
        onset_recall = (onset_pred * target["onset"]).sum() / (target["onset"].sum() + 1e-8)
        onset_f1 = 2 * onset_precision * onset_recall / (onset_precision + onset_recall + 1e-8)
    return frame_acc, onset_f1.item()


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = total_fa = total_of = n = 0
    pbar = tqdm(dataloader, desc="Train", disable=True)
    for spec, ot, ft in pbar:
        spec, ot, ft = spec.to(device), ot.to(device), ft.to(device)
        optimizer.zero_grad()
        pred = model(spec)
        loss = onset_frame_loss(pred, {"onset": ot, "frame": ft})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        fa, of1 = compute_metrics(pred, {"onset": ot, "frame": ft})
        total_loss += loss.item()
        total_fa += fa
        total_of += of1
        n += 1
        if n % 200 == 0:
            logger.info(
                f"  step {n}: loss={loss.item():.3f} frame_acc={fa:.3f} onset_f1={of1:.3f}"
            )
    return {"loss": total_loss / n, "frame_acc": total_fa / n, "onset_f1": total_of / n}


def validate(model, dataloader, device):
    model.eval()
    total_loss = total_fa = total_of = n = 0
    with torch.no_grad():
        for spec, ot, ft in dataloader:
            spec, ot, ft = spec.to(device), ot.to(device), ft.to(device)
            pred = model(spec)
            loss = onset_frame_loss(pred, {"onset": ot, "frame": ft})
            fa, of1 = compute_metrics(pred, {"onset": ot, "frame": ft})
            total_loss += loss.item()
            total_fa += fa
            total_of += of1
            n += 1
    return {"loss": total_loss / n, "frame_acc": total_fa / n, "onset_f1": total_of / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=10854)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-size", type=int, default=50)
    ap.add_argument("--save", default="VER2.2_BootstrapFull")
    ap.add_argument("--resume", default=None, help="从 latest.pt 续训")
    args = ap.parse_args()

    # 脱离终端运行时, 把 stdout/stderr 全部并入日志文件, 避免管道断裂静默死亡
    if not sys.stdout.isatty():
        log_fh = open(_LOG_FILE, "a", encoding="utf-8")
        sys.stdout = log_fh
        sys.stderr = log_fh

    device = cfg.DEVICE
    model = OnsetsAndFrames().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"]
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    save_base = os.path.join(cfg.MODEL_SAVE_DIR, args.save)
    latest_path = save_base + "_latest.pt"
    best_path = save_base + "_best.pth"
    final_path = save_base + ".pth"

    logger.info(f"=== Overnight training: {args.n_files} files x {args.epochs} epochs, "
                f"batch={args.batch}, lr={args.lr}, device={device} ===")

    train_ds = PianoSynthDataset(max_files=args.n_files, max_dur_sec=MAX_DUR)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_variable_length)
    val_ds = PianoSynthDataset(max_files=min(args.val_size, args.n_files // 4), max_dur_sec=MAX_DUR)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate_variable_length)
    logger.info(f"Train files: {len(train_ds)}, val files: {len(val_ds)}")

    best_val = float("inf")
    if start_epoch > 0 and os.path.exists(best_path):
        best_val = float("inf")  # 简化: 续训时重新按 val_loss 择优

    t_start = time.time()
    for ep in range(start_epoch, args.epochs):
        t_ep = time.time()
        logger.info(f"=== Epoch {ep + 1}/{args.epochs} start ===")
        tm = train_one_epoch(model, train_loader, optimizer, device)
        vm = validate(model, val_loader, device)
        logger.info(
            f"Epoch {ep + 1}/{args.epochs}: "
            f"train_loss={tm['loss']:.4f} val_loss={vm['loss']:.4f} | "
            f"frame_acc={vm['frame_acc']:.3f} onset_f1={vm['onset_f1']:.3f} | "
            f"{(time.time() - t_ep) / 60:.1f} min"
        )

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": ep + 1,
            "val_loss": vm["loss"],
        }, latest_path)
        if vm["loss"] < best_val:
            best_val = vm["loss"]
            torch.save(model.state_dict(), best_path)
            logger.info(f"  New best (val_loss={vm['loss']:.4f}, onset_f1={vm['onset_f1']:.3f})")

    torch.save(model.state_dict(), final_path)
    logger.info(f"Done in {(time.time() - t_start) / 3600:.1f}h. Saved: {final_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n===== TRAINING CRASHED =====\n")
            traceback.print_exc(file=f)
        raise
