"""VER-SEP 2.0 吉他分离（mel-band-roformer，MSST 子进程推理，--bigshifts 4）。

ckpt = VERSEP2.0 ep9（comp 硬区 SDR 11.42 vs 1.1 的 8.36；F2 112 对整体 SI-SDR
+6.2dB、最差四分位 +10.4dB、下游 F1 0.531->0.597，112/112 零回退；干净独奏透传
29.9->38.9dB）。验收记录见 findings 2026-08-25。

产物缓存 cache_dir/<输入名>/Guitar.flac（MSST 默认模板 {file_name}/{instr}），
重跑免分离。ckpt/MSST 缺失返回 None，上层回退 demucs guitar stem。
运行需 env/bin 在 PATH（nvrtc DLL）——本模块在子进程环境里自行补上。
子进程为参数列表形式（shell=False），无任何 shell 拼接。
"""
import glob
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

SE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 分轨模型集中地 = MUSE 根 source_separation/versep/（2026-08-24 整理）。
# MUSE_VERSEP_CKPT 可覆盖（A/B 评测用，取该目录下文件名或绝对路径）。
VERSEP_DIR = os.path.join(os.path.dirname(SE_ROOT), "source_separation", "versep")
CKPT_PATH = os.path.join(
    VERSEP_DIR,
    os.environ.get("MUSE_VERSEP_CKPT",
                   "VERSEP2.0_roformer_guitar_ep9_sdr11.4248.ckpt"))
if not os.path.isabs(CKPT_PATH):
    CKPT_PATH = os.path.join(VERSEP_DIR, CKPT_PATH)
CONFIG_PATH = os.path.join(VERSEP_DIR, "config_guitar_finetune_v1.yaml")
MSST_INFERENCE = os.path.join(SE_ROOT, "external", "Music-Source-Separation-Training", "inference.py")
ENV_BIN = os.path.join(os.path.dirname(SE_ROOT), "env", "bin")


def _fs_resolve(path: str) -> str | None:
    """exists 探测：os.path.exists 偶发误 False（大文件首访扫描锁），glob 兜底。"""
    if os.path.exists(path):
        return path
    hit = glob.glob(path)
    return hit[0] if hit else None


def separate_guitar(audio_path: str, cache_dir: str) -> str | None:
    """分离吉他 stem；返回产物路径，ckpt/MSST 缺失时返回 None（上层回退）。"""
    os.makedirs(cache_dir, exist_ok=True)
    for hit in glob.glob(os.path.join(cache_dir, "*", "Guitar.*")):
        logger.info("  [versep] cached: %s", hit)
        return hit
    ckpt = _fs_resolve(CKPT_PATH)
    msst = _fs_resolve(MSST_INFERENCE)
    if not (ckpt and msst):
        logger.warning("  [versep] VER-SEP ckpt 或 MSST 不在本地，回退 demucs guitar")
        return None

    inp = os.path.join(cache_dir, "input")
    os.makedirs(inp, exist_ok=True)
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in (".wav", ".flac", ".mp3"):
        ext = ".wav"
    local = os.path.join(inp, "mix" + ext)
    if not os.path.exists(local):
        shutil.copyfile(audio_path, local)

    env = dict(os.environ)
    env["PATH"] = ENV_BIN + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    argv = [sys.executable, MSST_INFERENCE,
            "--model_type", "mel_band_roformer",
            "--config_path", CONFIG_PATH,
            "--start_check_point", ckpt,
            "--input_folder", inp,
            "--store_dir", cache_dir,
            "--bigshifts", "4",
            "--device_ids", "0"]
    logger.info("  [versep] running VER-SEP 2.0 + bigshifts4 (首次约 2-4 分钟/曲)...")
    subprocess.run(argv, check=True, env=env, cwd=os.path.dirname(MSST_INFERENCE), shell=False)
    for hit in glob.glob(os.path.join(cache_dir, "*", "Guitar.*")):
        return hit
    raise RuntimeError("VER-SEP 未产出 Guitar stem: " + cache_dir)
