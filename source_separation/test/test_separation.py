# test_separation.py — 分离效果测试
# 在验证集样本上测试分离模型的推理质量和速度
import os
import sys
import json
import time
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.model import LightweightUMX
from src.infer import separate, load_model, N_FFT, HOP_LENGTH
from src.dataset import compute_mag, get_stft, GuitarSeparationDataset


def test_inference_speed(model, device, n_runs=20):
    """测试推理速度"""
    print(f"\n推理速度测试 ({n_runs} 次)...")
    x = torch.randn(66150).to(device)  # 3s @ 22050

    # warmup
    for _ in range(5):
        _ = separate(x, model, device)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = separate(x, model, device)
        times.append(time.perf_counter() - t0)

    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    realtime = avg / 3000  # 3s 窗口
    print(f"  平均延迟: {avg:.1f} ± {std:.1f} ms")
    print(f"  实时比: {realtime:.3f}x (窗口) / 整体 ~{3000/avg:.1f}x 实时")
    return avg


def test_separation_quality(model, device, n_samples=5):
    """测试分离质量（L1 loss）"""
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
    losses = []

    for sample in val_samples[:n_samples]:
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
        losses.append(l1)

        print(f"  {sample['mix']}: L1={l1:.4f}")

    if losses:
        print(f"  平均 L1: {np.mean(losses):.4f}")


def main():
    print(f"测试: {config.MODEL_VERSION}")
    device = config.DEVICE
    print(f"设备: {device}")

    model = load_model(device=device)

    test_inference_speed(model, device)
    test_separation_quality(model, device)


if __name__ == "__main__":
    main()
