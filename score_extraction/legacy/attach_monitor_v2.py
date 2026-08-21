"""Attach monitor v2 for VER3.6_Scratch100: parallel evals (max 2) with checkpoint snapshots."""
import json
import os
import re
import shutil
import subprocess
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.abspath(os.path.join(BASE, "..", "..", "score_extraction"))
PY = os.path.join(MAIN, ".venv", "bin", "python")
LOG = os.path.join(BASE, "train", "log_overnight.log")
EVAL_WORKER = os.path.join(MAIN, "train", "eval_val_worker_v3.py")
LOG_DIR = os.path.join(MAIN, "model", "log")
METRICS = os.path.join(LOG_DIR, "metrics_v36_scratch100.jsonl")
CKPT = os.path.join(BASE, "model", "VER3.6_Scratch100_latest.pt")
SAVE = "VER3.6_Scratch100"
EVAL_EVERY = 5
MAX_CONCURRENT = 2
EVAL_THREADS = 8


def recorded_epochs():
    out = set()
    if os.path.exists(METRICS):
        for line in open(METRICS, encoding="utf-8"):
            try:
                out.add(json.loads(line)["epoch"])
            except Exception:
                pass
    return out


def record(ep, out_json, snap, kind="epoch"):
    rec = {"kind": kind, "epoch": ep, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as jf:
            rec.update(json.load(jf))
    else:
        rec["eval_error"] = "no output json"
    with open(METRICS, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("recorded", kind, ep, flush=True)


def spawn(ep, pending, wanted):
    if len(pending) >= MAX_CONCURRENT:
        return False
    snap = os.path.join(BASE, "model", SAVE + "_e%03d.pt" % ep)
    try:
        shutil.copy(CKPT, snap)
    except Exception as e:
        print("snapshot failed", ep, e, flush=True)
        wanted.discard(ep)
        return False
    out_json = os.path.join(LOG_DIR, "v36_e%03d.json" % ep)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(EVAL_THREADS)
    env["MKL_NUM_THREADS"] = str(EVAL_THREADS)
    env["NUMBA_NUM_THREADS"] = str(EVAL_THREADS)
    proc = subprocess.Popen(
        [PY, EVAL_WORKER, "--ckpt", snap, "--out", out_json, "--seg", "30"],
        cwd=MAIN, env=env)
    pending[ep] = (proc, out_json, snap)
    print("spawned eval", ep, flush=True)
    return True


def main():
    recorded = recorded_epochs()
    pending = {}
    wanted = set()
    pos = os.path.getsize(LOG)
    last_completed = None
    with open(LOG, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Epoch (\d+)/100: .*?([\d.]+) min", line)
            if m:
                last_completed = int(m.group(1))
    if last_completed and last_completed not in recorded and last_completed % EVAL_EVERY == 0:
        wanted.add(last_completed)
    print("monitor v2 start; last_completed =", last_completed, flush=True)
    while True:
        for ep in list(pending):
            proc, out_json, snap = pending[ep]
            if proc.poll() is not None:
                record(ep, out_json, snap)
                try:
                    os.remove(snap)
                except Exception:
                    pass
                del pending[ep]
        with open(LOG, encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            lines = f.readlines()
            pos = f.tell()
        for line in lines:
            m = re.search(r"Epoch (\d+)/100: .*?([\d.]+) min", line)
            if m:
                ep = int(m.group(1))
                if ep not in recorded and ep not in pending and (ep % EVAL_EVERY == 0 or ep >= 98):
                    wanted.add(ep)
        for ep in sorted(wanted):
            if ep not in pending and ep not in recorded:
                if not spawn(ep, pending, wanted):
                    break
                wanted.discard(ep)
        time.sleep(15)


if __name__ == "__main__":
    main()