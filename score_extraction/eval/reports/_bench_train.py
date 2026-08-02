import os
import sys
import time

sys.path.insert(0, "D:/program_project/MUSE/score_extraction")
os.chdir("D:/program_project/MUSE/score_extraction")

import numpy as np
import torch

from train.config import train_config as cfg
from train.dataset import PianoSynthDataset, collate_variable_length
from train.model import OnsetsAndFrames, onset_frame_loss
from torch.utils.data import DataLoader


def bench_render(n=10):
    import random

    from train.render_midi import render_midi

    paths = sorted(__import__("glob").glob(os.path.join(cfg.MIDI_DIR, "*.mid")))
    random.seed(123)
    picks = random.sample(paths, n)
    t0 = time.time()
    for p in picks:
        render_midi(p, max_dur_sec=30)
    dt = time.time() - t0
    print(f"render: {n} songs in {dt:.1f}s -> {dt / n:.2f}s/song", flush=True)


def bench_steps(n_steps=25, batch=8):
    ds = PianoSynthDataset(max_files=600, max_dur_sec=30)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_variable_length)
    device = cfg.DEVICE
    model = OnsetsAndFrames().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    it = iter(dl)
    t0 = time.time()
    for i in range(n_steps):
        try:
            spec, ot, ft = next(it)
        except StopIteration:
            it = iter(dl)
            spec, ot, ft = next(it)
        spec, ot, ft = spec.to(device), ot.to(device), ft.to(device)
        opt.zero_grad()
        pred = model(spec)
        loss = onset_frame_loss(pred, {"onset": ot, "frame": ft})
        loss.backward()
        opt.step()
    dt = time.time() - t0
    print(f"train: {n_steps} steps batch={batch} in {dt:.1f}s -> {dt / n_steps:.2f}s/step", flush=True)
    print(f"  => {n_steps / dt * 60:.1f} steps/min, 1 epoch(600 songs) = {600 / batch * dt / n_steps / 60:.1f} min", flush=True)


if __name__ == "__main__":
    bench_render(10)
    bench_steps(25, batch=8)
