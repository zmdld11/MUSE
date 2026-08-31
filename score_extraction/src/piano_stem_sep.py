"""bs_roformer 钢琴 stem 分离（钢琴混音路径 2026-08-31，08-30 夜间增测接线）。

夜间增测结论（grand_eval 同口径，note@50）：ia-amt@bs_roformer-piano-stem
F2-piano 40 = 0.882（raw 直推 0.729，+15.3pt）、BabySlakh 20 = 0.820
（raw 0.552，+26.8pt）；htdemucs_6s 钢琴 stem 在密集编曲上接近不可用
（SI-SDR 1.4dB vs bs_roformer 7.0）。Mega（78MB 单 stem）与 SW（699MB
6 件套）指标接近，默认取性能占比小者（见 eval/piano_sep_bench.py 实测），
MUSE_PIANO_STEM=mega|sw|off 可覆盖（off=回退 raw 直推钢琴类）。

模式照抄 separate_vocals_melband：MSST 子进程（顶层模块同名互踩必须隔离）、
产物缓存 output_dir/<basename>/piano.wav、失败返回 None 由调用方回退。
"""
import logging
import os

logger = logging.getLogger(__name__)

_SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    "mega": (os.path.join(_SE, "external", "msst_mega_piano"),
             "bs_mega_53stem_piano_mvsep.ckpt", "config.yaml"),
    "sw": (os.path.join(_SE, "external", "msst_sw_6stem"),
           "model_sw.ckpt", "config_sw.yaml"),
}


def piano_stem_model() -> str:
    """当前启用的分离模型（mega|sw|off）。"""
    return os.environ.get("MUSE_PIANO_STEM", "mega")


def separate_piano_bsroformer(audio_path: str, output_dir: str) -> str | None:
    """分离钢琴 stem；返回产物路径，模型缺失/失败返回 None（调用方回退）。"""
    import shutil
    import subprocess
    import sys

    model = piano_stem_model()
    if model == "off":
        return None
    mdir, ckpt_name, cfg_name = MODELS[model]
    msst = os.path.join(_SE, "external", "Music-Source-Separation-Training")
    cfg = os.path.join(mdir, cfg_name)
    ckpt = os.path.join(mdir, ckpt_name)
    if not (os.path.exists(cfg) and os.path.exists(ckpt)):
        logger.warning("  [bspiano] %s 配置或权重缺失，跳过", model)
        return None
    out_wav = os.path.join(os.path.abspath(output_dir),
                           os.path.splitext(os.path.basename(audio_path))[0],
                           "piano.wav")
    if os.path.exists(out_wav):
        logger.info(f"  [bspiano] cached {out_wav}")
        return out_wav
    # MSST 按目录批处理：拷进独立工作目录，避免殃及同目录其它歌
    work_dir = os.path.join(os.path.abspath(output_dir), "_bspiano_input")
    os.makedirs(work_dir, exist_ok=True)
    local_audio = os.path.join(work_dir, os.path.basename(audio_path))
    if not os.path.exists(local_audio):
        shutil.copy2(audio_path, local_audio)
    env = dict(os.environ)
    env["PATH"] = (os.path.join(os.path.dirname(_SE), "env", "bin")
                   + os.pathsep + env.get("PATH", ""))
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 免 HF 心跳 SSL 重试
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [sys.executable, os.path.join(msst, "inference.py"),
           "--model_type", "bs_roformer", "--config_path", cfg,
           "--start_check_point", ckpt,
           "--input_folder", work_dir,
           "--store_dir", os.path.abspath(output_dir),
           "--device_ids", "0"]
    logger.info("  [bspiano] %s 钢琴分离（子进程）...", model)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=msst, env=env)
    if r.returncode != 0 or not os.path.exists(out_wav):
        logger.warning("  [bspiano] 分离失败：%s", (r.stderr or "")[-500:])
        return None
    return out_wav
