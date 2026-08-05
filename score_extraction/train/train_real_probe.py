"""纯 MAESTRO 真实域微调实验 (2026-08-04 夜).

目的: 验证模型能否从纯合成域 (VER2.0) 出发, 通过纯真实域微调学会 real onset.
用法:
  python train/train_real_probe.py [--resume model/REAL_PROBE_latest.pt]
启动默认从 model/VER2.0_Bootstrap.pth 初始化, 训 3 个 epoch (MAESTRO train 全量 962).
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from train.config import train_config as cfg
from train.maestro_dataset import MaestroDataset
from train.model import OnsetsAndFrames, onset_frame_loss
from train.dataset import collate_variable_length

LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "log_real_probe.log"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("real_probe")

MAX_DUR = 30


def metrics_at(pred, target, th):
    p = (pred["onset"] > th).float()
    tp = (p * target["onset"]).sum().item()
    pp = p.sum().item()
    gp = target["onset"].sum().item()
    prec = tp / pp if pp else 0.0
    rec = tp / gp if gp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def validate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    acc = {th: [0.0, 0.0, 0] for th in (0.1, 0.2, 0.3, 0.5)}  # sumF1, sumP, count
    with torch.no_grad():
        for spec, ot, ft in loader:
            spec, ot, ft = spec.to(device), ot.to(device), ft.to(device)
            pred = model(spec)
            total += onset_frame_loss(pred, {"onset": ot, "frame": ft}).item()
            n += 1
            for th in acc:
                p, r, f = metrics_at(pred, {"onset": ot, "frame": ft}, th)
                acc[th][0] += f
                acc[th][1] += p
                acc[th][2] += 1
    return total / n, {th: (v[0] / v[2], v[1] / v[2]) for th, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    device = cfg.DEVICE
    model = OnsetsAndFrames().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    start_ep = 0

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_ep = ck["epoch"]
        logger.info(f"resumed {args.resume} (epoch {start_ep})")
    else:
        sd = torch.load(r"model\VER2.0_Bootstrap.pth", map_location=device)
        model.load_state_dict(sd)
        logger.info("init from VER2.0_Bootstrap.pth (pure synth)")

    train_ds = MaestroDataset(split="train", max_dur_sec=MAX_DUR)
    val_ds = MaestroDataset(split="validation", max_files=50, max_dur_sec=MAX_DUR)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              collate_fn=collate_variable_length, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            collate_fn=collate_variable_length, num_workers=0)
    logger.info(f"train segs={len(train_ds)} val segs={len(val_ds)} device={device}")

    for ep in range(start_ep, args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        nb = 0
        for spec, ot, ft in train_loader:
            spec, ot, ft = spec.to(device), ot.to(device), ft.to(device)
            pred = model(spec)
            loss = onset_frame_loss(pred, {"onset": ot, "frame": ft})
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            if nb % 500 == 0:
                logger.info(f"  ep{ep + 1} step {nb}: loss={loss.item():.4f}")
        vl, f1s = validate(model, val_loader, device)
        msg = " ".join(f"F1@{th}={v[0]:.3f}(P={v[1]:.3f})" for th, v in f1s.items())
        logger.info(f"EPOCH {ep + 1}/{args.epochs}: train_loss={tot / nb:.4f} "
                    f"val_loss={vl:.4f} | {msg} | {(time.time() - t0) / 60:.1f}min")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": ep + 1, "val_loss": vl},
                   r"model\REAL_PROBE_latest.pt")
    torch.save(model.state_dict(), r"model\REAL_PROBE.pth")
    logger.info("REAL PROBE DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            import traceback
            traceback.print_exc(file=f)
        raise
