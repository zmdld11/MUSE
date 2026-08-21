"""VER3.6_Scratch100 runner: train from scratch (mix 1:1, 100 epochs) + per-epoch note eval + system stats."""
import json
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # legacy score_extraction
MAIN = os.path.abspath(os.path.join(BASE, "..", "..", "score_extraction"))  # current repo (eval worker)
PY = os.path.join(MAIN, ".venv", "bin", "python")
SAVE = "VER3.6_Scratch100"
LOG = os.path.join(BASE, "train", "log_overnight.log")
EVAL_WORKER = os.path.join(MAIN, "train", "eval_val_worker_v3.py")
LOG_DIR = os.path.join(MAIN, "model", "log")
METRICS = os.path.join(LOG_DIR, "metrics_v36_scratch100.jsonl")
STATS = os.path.join(LOG_DIR, "stats_v36_scratch100.jsonl")
CKPT = os.path.join(BASE, "model", SAVE + "_latest.pt")
EPOCHS = 100


def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        return [l.strip() for l in out.splitlines()]
    except Exception as e:
        return [str(e)]


def main():
    cmd = [PY, "train/train_overnight.py",
           "--mix", "--n-maestro", "962", "--epochs", str(EPOCHS),
           "--batch", "8", "--lr", "3e-4", "--onset-weight", "5",
           "--workers", "8", "--save", SAVE]
    os.makedirs(LOG_DIR, exist_ok=True)
    print("runner cmd:", " ".join(cmd), flush=True)
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")
    proc = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pos = [0]
    seen = set()
    last_stats = time.time()
    t_start = time.time()
    while proc.poll() is None:
        if os.path.exists(LOG):
            with open(LOG, encoding="utf-8", errors="replace") as f:
                f.seek(pos[0])
                lines = f.readlines()
                pos[0] = f.tell()
            for line in lines:
                m = re.search(r"Epoch (\d+)/" + str(EPOCHS) + r": .*?([\d.]+) min", line)
                if m:
                    ep = int(m.group(1))
                    if ep not in seen:
                        seen.add(ep)
                        rec = {"kind": "epoch", "epoch": ep, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "epoch_min": float(m.group(2)),
                               "wall_since_start_min": round((time.time() - t_start) / 60, 1)}
                        out_json = os.path.join(LOG_DIR, "v36_e%03d.json" % ep)
                        try:
                            r = subprocess.run([PY, EVAL_WORKER, "--ckpt", CKPT, "--out", out_json,
                                                "--seg", "30"], cwd=MAIN, capture_output=True, text=True, timeout=1800)
                            if os.path.exists(out_json):
                                with open(out_json, encoding="utf-8") as jf:
                                    rec.update(json.load(jf))
                            else:
                                rec["eval_error"] = r.stderr[-500:]
                        except Exception as e:
                            rec["eval_error"] = str(e)
                        with open(METRICS, "a", encoding="utf-8") as jf:
                            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        print("recorded epoch", ep, flush=True)
        if time.time() - last_stats >= 300:
            last_stats = time.time()
            stat = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "loadavg": open("/proc/loadavg").read().strip(),
                    "gpu": gpu_stats()}
            with open(STATS, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(stat, ensure_ascii=False) + "\n")
        time.sleep(20)
    print("training exited", proc.returncode, flush=True)


if __name__ == "__main__":
    main()
