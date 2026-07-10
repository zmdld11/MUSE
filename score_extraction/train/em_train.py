"""NoteEM training: Bootstrap on synthetic MIDI → EM iterations."""
import logging
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import train_config as cfg
from train.model import OnsetsAndFrames, onset_frame_loss
from train.dataset import PianoSynthDataset, collate_variable_length

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 2
MAX_DUR = 30  # seconds, keep GPU memory under control


def compute_metrics(pred, target, hop_length=512, sr=22050):
    """Compute frame-level accuracy and approximate note-level F1."""
    with torch.no_grad():
        frame_pred = (pred["frame"] > 0.5).float()
        frame_acc = (frame_pred == target["frame"]).float().mean().item()

        onset_pred = (pred["onset"] > 0.5).float()
        onset_precision = (onset_pred * target["onset"]).sum() / (onset_pred.sum() + 1e-8)
        onset_recall = (onset_pred * target["onset"]).sum() / (target["onset"].sum() + 1e-8)
        onset_f1 = 2 * onset_precision * onset_recall / (onset_precision + onset_recall + 1e-8)

    return {"frame_acc": frame_acc, "onset_f1": onset_f1.item()}


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_frame_acc, total_onset_f1, n = 0, 0, 0, 0
    pbar = tqdm(dataloader, desc="Train")
    for spec, ot, ft in pbar:
        spec, ot, ft = spec.to(device), ot.to(device), ft.to(device)
        optimizer.zero_grad()
        pred = model(spec)
        loss = onset_frame_loss(pred, {"onset": ot, "frame": ft})
        loss.backward()
        optimizer.step()
        m = compute_metrics(pred, {"onset": ot, "frame": ft})
        total_loss += loss.item()
        total_frame_acc += m["frame_acc"]
        total_onset_f1 += m["onset_f1"]
        n += 1
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "f_acc": f"{m['frame_acc']:.3f}", "o_f1": f"{m['onset_f1']:.3f}"})
    return {"loss": total_loss / n, "frame_acc": total_frame_acc / n, "onset_f1": total_onset_f1 / n}


def validate(model, dataloader, device):
    model.eval()
    total_loss, total_frame_acc, total_onset_f1, n = 0, 0, 0, 0
    with torch.no_grad():
        for spec, ot, ft in dataloader:
            spec, ot, ft = spec.to(device), ot.to(device), ft.to(device)
            pred = model(spec)
            loss = onset_frame_loss(pred, {"onset": ot, "frame": ft})
            m = compute_metrics(pred, {"onset": ot, "frame": ft})
            total_loss += loss.item()
            total_frame_acc += m["frame_acc"]
            total_onset_f1 += m["onset_f1"]
            n += 1
    return {"loss": total_loss / n, "frame_acc": total_frame_acc / n, "onset_f1": total_onset_f1 / n}


def train_bootstrap(model, device, n_files=500):
    epochs = cfg.EPOCHS_BOOTSTRAP
    logger.info(f"=== Bootstrap: {epochs} epochs, {n_files} files ===")

    train_ds = PianoSynthDataset(max_files=n_files, max_dur_sec=MAX_DUR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_variable_length)

    val_ds = PianoSynthDataset(max_files=min(50, n_files // 4), max_dur_sec=MAX_DUR)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_variable_length)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    best_val = float("inf")

    for ep in range(epochs):
        tm = train_one_epoch(model, train_loader, optimizer, device)
        vm = validate(model, val_loader, device)
        logger.info(
            f"Epoch {ep+1}/{epochs}: "
            f"train_loss={tm['loss']:.4f} val_loss={vm['loss']:.4f} | "
            f"frame_acc={vm['frame_acc']:.3f} onset_f1={vm['onset_f1']:.3f}"
        )
        if vm["loss"] < best_val:
            best_val = vm["loss"]
            torch.save(model.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, "bootstrap_best.pth"))
            logger.info(f"  Saved best (val_loss={vm['loss']:.4f}, onset_f1={vm['onset_f1']:.3f})")

    logger.info(f"Bootstrap done. Best val={best_val:.4f}")


def run_noteem_training():
    device = cfg.DEVICE
    model = OnsetsAndFrames().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"NoteEM on {device}. Model: {n_params:,} params")

    train_bootstrap(model, device, n_files=500)

    torch.save(model.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, f"{cfg.MODEL_VERSION}.pth"))
    logger.info("Done. Model saved.")


if __name__ == "__main__":
    run_noteem_training()
