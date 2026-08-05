"""训练编排器 (独立于会话存活):
1. 等待 MAESTRO 30s 缓存生成完成 (train 962)
2. 启动混合训练子进程
3. 每 20 分钟监控: 卡死/NaN/崩溃 -> 自动停止并记录
"""
import logging
import os
import subprocess
import sys
import time

WS = "d:/program_project/MUSE/score_extraction"
PY = "C:/Users/ROG/.conda/envs/score_build/python.exe"
LOG = os.path.join(WS, "train", "orchestrator.log")
TRAIN_LOG = os.path.join(WS, "train", "log_overnight.log")
PID_FILE = os.path.join(WS, "train", "train.pid")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8")],
)
log = logging.getLogger("orchestrator")


def wait_cache30():
    """等待 train 962 首 30s 缓存就绪."""
    sys.path.insert(0, WS)
    from train.maestro_dataset import (
        DEFAULT_CACHE_DIR, DEFAULT_CSV, DEFAULT_MAESTRO_DIR,
        _cache_path, load_maestro_rows,
    )

    rows = [r for r in load_maestro_rows(DEFAULT_CSV) if r["split"] == "train"]
    while True:
        done = sum(1 for r in rows if os.path.exists(_cache_path(
            os.path.join(DEFAULT_MAESTRO_DIR, r["midi_filename"]),
            os.path.join(DEFAULT_MAESTRO_DIR, r["audio_filename"]),
            30, DEFAULT_CACHE_DIR)))
        log.info(f"缓存进度: {done}/{len(rows)}")
        if done >= len(rows):
            log.info("30s 缓存就绪")
            return
        time.sleep(60)


def start_training():
    cmd = [
        PY, "-u", "train/train_overnight.py",
        "--epochs", "100", "--lr", "1e-4",
        "--save", "VER3.0_MixedReal",
        "--resume", "model/VER2.2_BootstrapFull_latest.pt",
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=WS, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    log.info(f"训练已启动 PID={proc.pid}")
    return proc


def monitor(proc):
    last_epoch = ""
    while True:
        time.sleep(1200)  # 20 分钟
        ts = time.strftime("%H:%M:%S")
        if proc.poll() is not None:
            tail = ""
            if os.path.exists(TRAIN_LOG):
                tail = open(TRAIN_LOG, encoding="utf-8", errors="ignore").read()[-2000:]
            if "Done in" in tail:
                log.info(f"{ts} 训练正常完成")
            else:
                log.warning(f"{ts} !! 训练进程退出且无完成标志 — 崩溃 (rc={proc.returncode})")
            return
        # 卡死检测: 日志 30 分钟无更新
        if os.path.exists(TRAIN_LOG):
            age = time.time() - os.path.getmtime(TRAIN_LOG)
            if age > 1800:
                log.warning(f"{ts} !! 日志 {age/60:.0f} 分钟无更新 — 卡死, 停止")
                proc.kill()
                return
            tail = open(TRAIN_LOG, encoding="utf-8", errors="ignore").read()[-4000:]
            if "nan" in tail.lower() or "inf" in tail.lower():
                log.warning(f"{ts} !! 检测到 NaN/Inf — 停止")
                proc.kill()
                return
            for line in tail.splitlines():
                if "Epoch " in line and "train_loss=" in line:
                    log.info(f"{ts} {line.strip()}")
                    break


if __name__ == "__main__":
    log.info("===== orchestrator 启动 =====")
    wait_cache30()
    proc = start_training()
    monitor(proc)
    log.info("===== orchestrator 结束 =====")
