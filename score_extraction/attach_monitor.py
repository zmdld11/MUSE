"""Attach monitor for VER3.6_Scratch100: no training spawn, eval every EVAL_EVERY epochs."""
import json
import os
import re
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
EVAL_EVERY = 5


def recorded_epochs():
    out = set()
    if os.path.exists(METRICS):
        for line in open(METRICS, encoding="utf-8"):
            try:
                out.add(json.loads(line)["epoch"])
            except Exception:
                pass
    return out


def run_eval(ep):
    rec = {"kind": "epoch", "epoch": ep, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_json = os.path.join(LOG_DIR, "v36_e%03d.json" % ep)
    try:
        r = subprocess.run([PY, EVAL_WORKER, "--ckpt", CKPT, "--out", out_json,
                            "--seg", "30"], cwd=MAIN, capture_output=True, text=True, timeout=3600)
        if os.path.exists(out_json):
            with open(out_json, encoding="utf-8") as jf:
                rec.update(json.load(jf))
        else:
            rec["eval_error"] = (r.stderr or r.stdout)[-600:]
    except Exception as e:
        rec["eval_error"] = str(e)
    with open(METRICS, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("recorded epoch", ep, flush=True)


def main():
    recorded = recorded_epochs()
    pos = os.path.getsize(LOG)
    last_completed = None
    with open(LOG, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Epoch (\d+)/100: .*?([\d.]+) min", line)
            if m:
                last_completed = int(m.group(1))
    if last_completed and last_completed not in recorded:
        print("startup snapshot eval for epoch", last_completed, flush=True)
        run_eval(last_completed)
        recorded.add(last_completed)
    while True:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            lines = f.readlines()
            pos = f.tell()
        for line in lines:
            m = re.search(r"Epoch (\d+)/100: .*?([\d.]+) min", line)
            if m:
                ep = int(m.group(1))
                if ep not in recorded and (ep % EVAL_EVERY == 0 or ep >= 98):
                    run_eval(ep)
                    recorded.add(ep)
        time.sleep(20)


if __name__ == "__main__":
    main()