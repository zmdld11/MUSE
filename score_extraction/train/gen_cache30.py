"""生成 MAESTRO 30s 段缓存 (与 train_overnight MAX_DUR=30 一致)."""
import logging
import sys

sys.path.insert(0, "d:/program_project/MUSE/score_extraction")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from train.maestro_dataset import MaestroDataset

t = MaestroDataset(split="train", max_dur_sec=30)
print(f"CACHE30 train: {len(t)} 段", flush=True)
v = MaestroDataset(split="validation", max_files=50, max_dur_sec=30)
print(f"CACHE30 validation: {len(v)} 段", flush=True)
print("CACHE30 DONE", flush=True)
