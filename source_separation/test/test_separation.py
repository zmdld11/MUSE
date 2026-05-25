# test_separation.py — 分离效果测试
# 在验证集样本上测试分离质量和速度，基准指标为 SDR
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.infer import separate, load_model, N_FFT, HOP_LENGTH
from src.dataset import compute_mag, get_stft, GuitarSeparationDataset


def compute_sdr(estimate, target, eps=1e-8):
    """
    计算时域 SDR (Signal-to-Distortion Ratio)
    SDR = 10 * log10(||target||² / ||estimate - target||²)
    """
    error = estimate - target
    s_target = torch.sum(target ** 2)
    s_error = torch.sum(error ** 2)
    sdr = 10 * torch.log10(s_target / torch.clamp(s_error, min=eps))
    return sdr.item()


def compute_sdr_mag(est_mag, target_mag, eps=1e-8):
    """
    计算幅度谱 SDR
    """
    error = est_mag - target_mag
    s_target = torch.sum(target_mag ** 2)
    s_error = torch.sum(error ** 2)
    sdr = 10 * torch.log10(s_target / torch.clamp(s_error, min=eps))
    return sdr.item()


def test_inference_speed(model, device, n_runs=20):
    """测试推理速度"""
    print(f"\n推理速度测试 ({n_runs} 次)...")
    x = torch.randn(66150).to(device)  # 3s @ 22050

    # warmup
    for _ in range(5):
        _ = separate(x, model, device)

    times = []
    for _ in tqdm(range(n_runs), desc="速度测试", leave=False):
        t0 = time.perf_counter()
        _ = separate(x, model, device)
        times.append(time.perf_counter() - t0)

    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"  平均延迟: {avg:.1f} ± {std:.1f} ms")
    print(f"  实时比: {3000/avg:.1f}x 实时")
    return avg


def test_separation_quality(model, device, n_samples=100):
    """测试分离质量（SDR 为主指标，L1 为辅）"""
    print(f"\n分离质量测试 ({n_samples} 个样本)...")
    metadata_path = os.path.join(config.DATASET_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print("  跳过: 数据集未构建")
        return

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    val_samples = meta.get("val_samples", meta.get("train_samples", []))
    if not val_samples:
        print("  跳过: 无验证样本")
        return

    audio_dir = os.path.join(config.DATASET_DIR, "audio")
    window = get_stft()

    sdr_list = []
    sdr_mag_list = []
    l1_list = []

    for sample in tqdm(val_samples[:n_samples], desc="分离质量"):
        mix_path = os.path.join(audio_dir, sample["mix"])
        gtr_path = os.path.join(audio_dir, sample["guitar"])

        mix_audio, _ = sf.read(mix_path)
        gtr_audio, _ = sf.read(gtr_path)

        mix_tensor = torch.from_numpy(mix_audio.astype(np.float32))
        gtr_tensor = torch.from_numpy(gtr_audio.astype(np.float32))

        # 分离
        guitar_pred = separate(mix_tensor, model, device)
        guitar_pred_tensor = torch.from_numpy(guitar_pred)

        # L1 loss（时域）
        l1 = torch.mean(torch.abs(guitar_pred_tensor - gtr_tensor)).item()
        l1_list.append(l1)

        # SDR（时域 — 主指标）
        sdr = compute_sdr(guitar_pred_tensor, gtr_tensor)
        sdr_list.append(sdr)

        # SDR（幅度谱 — 辅助参考）
        pred_mag = compute_mag(guitar_pred_tensor, window)
        gtr_mag = compute_mag(gtr_tensor, window)
        min_T = min(pred_mag.shape[1], gtr_mag.shape[1])
        sdr_mag = compute_sdr_mag(pred_mag[:, :min_T], gtr_mag[:, :min_T])
        sdr_mag_list.append(sdr_mag)

    if sdr_list:
        print(f"  SDR (时域):  {np.mean(sdr_list):.2f} dB  (中位数 {np.median(sdr_list):.2f} dB)")
        print(f"  SDR (幅度谱): {np.mean(sdr_mag_list):.2f} dB  (中位数 {np.median(sdr_mag_list):.2f} dB)")
        print(f"  L1 (时域):    {np.mean(l1_list):.4f}")
        # SDR 分层统计
        print(f"  SDR 分布:")
        for threshold in [-5, 0, 3, 6, 10]:
            ratio = np.mean([s >= threshold for s in sdr_list]) * 100
            print(f"    ≥ {threshold:+d} dB: {ratio:.1f}%")

    return np.mean(sdr_list), np.mean(l1_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="验证样本数")
    parser.add_argument("--speed-only", action="store_true", help="仅测速度")
    parser.add_argument("--quality-only", action="store_true", help="仅测质量")
    args = parser.parse_args()

    print(f"测试: {config.MODEL_VERSION}")
    device = config.DEVICE
    print(f"设备: {device}")

    model = load_model(device=device)

    if not args.quality_only:
        test_inference_speed(model, device)
    if not args.speed_only:
        test_separation_quality(model, device, args.samples)


if __name__ == "__main__":
    main()
