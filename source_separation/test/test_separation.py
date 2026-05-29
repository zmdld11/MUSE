# test_separation.py — DemucsLM 分离效果测试
import os, sys, json, time, argparse
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.infer import separate, load_model


def compute_sdr(est, target, eps=1e-8):
    """时域 SDR: 10*log10(||target||² / ||est-target||²)"""
    err = est - target
    tpow = (target ** 2).sum()
    epow = (err ** 2).sum()
    if tpow < eps: return 0.0
    return (10 * torch.log10(tpow / torch.clamp(epow, min=eps))).item()


def compute_sdr_mag(est, target, n_fft=1024, hop=256, eps=1e-8):
    """幅度谱 SDR: 忽略相位，仅比幅度"""
    if isinstance(est, np.ndarray): est = torch.from_numpy(est)
    if isinstance(target, np.ndarray): target = torch.from_numpy(target)
    w = torch.hann_window(n_fft)
    Xe = torch.stft(est, n_fft, hop, window=w, return_complex=True)
    Xt = torch.stft(target, n_fft, hop, window=w, return_complex=True)
    me = torch.sqrt(Xe.real**2 + Xe.imag**2 + eps)
    mt = torch.sqrt(Xt.real**2 + Xt.imag**2 + eps)
    err = me - mt
    tpow = (mt ** 2).sum()
    epow = (err ** 2).sum()
    return (10 * torch.log10(tpow / torch.clamp(epow, min=eps))).item()


def test_inference_speed(model, device, n_runs=20):
    print(f"\n推理速度测试 ({n_runs} 次)...")
    x = torch.randn(66150).to(device)
    for _ in range(5): separate(x, model, device)  # warmup
    times = []
    for _ in tqdm(range(n_runs), desc="速度测试", leave=False):
        t0 = time.perf_counter()
        separate(x, model, device)
        times.append(time.perf_counter() - t0)
    avg = np.mean(times) * 1000
    print(f"  平均延迟: {avg:.1f} ms  ({3000/avg:.1f}x 实时)")


def test_separation_quality(model, device, n_samples=100):
    print(f"\n分离质量测试 ({n_samples} 个样本)...")
    meta_path = os.path.join(config.DATASET_DIR, "metadata.json")
    if not os.path.exists(meta_path): print("  跳过: 无数据集"); return
    with open(meta_path) as f: meta = json.load(f)

    audio_dir = os.path.join(config.DATASET_DIR, "audio")
    total = meta["num_train"] + meta["num_val"]
    val_indices = list(range(meta["num_train"], total))

    sdr_list, sdr_mag_list, l1_list = [], [], []

    for idx in tqdm(val_indices[:n_samples], desc="分离质量"):
        mix, _ = sf.read(os.path.join(audio_dir, f"{idx:06d}_mix.wav"))
        gtr, _ = sf.read(os.path.join(audio_dir, f"{idx:06d}_gtr.wav"))

        mix_t = torch.from_numpy(mix.astype("float32"))
        gtr_t = torch.from_numpy(gtr.astype("float32"))

        pred = separate(mix_t, model, device)
        pred_t = torch.from_numpy(pred)

        sdr_list.append(compute_sdr(pred_t, gtr_t))
        sdr_mag_list.append(compute_sdr_mag(pred, gtr))
        l1_list.append(torch.mean(torch.abs(pred_t - gtr_t)).item())

    if sdr_list:
        print(f"  SDR (时域):   {np.mean(sdr_list):.2f} dB  (中位数 {np.median(sdr_list):.2f})")
        print(f"  SDR (幅度谱): {np.mean(sdr_mag_list):.2f} dB  (中位数 {np.median(sdr_mag_list):.2f})")
        print(f"  L1:           {np.mean(l1_list):.4f}")
        for t in [-5, 0, 3, 6, 10]:
            r = np.mean([s >= t for s in sdr_list]) * 100
            print(f"    ≥ {t:+d} dB: {r:.1f}%")

    return np.mean(sdr_list), np.mean(l1_list)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--speed-only", action="store_true")
    p.add_argument("--quality-only", action="store_true")
    a = p.parse_args()
    print(f"{config.MODEL_VERSION} | {config.DEVICE}")
    model = load_model(device=config.DEVICE)
    if not a.quality_only: test_inference_speed(model, config.DEVICE)
    if not a.speed_only: test_separation_quality(model, config.DEVICE, a.samples)


if __name__ == "__main__":
    main()
