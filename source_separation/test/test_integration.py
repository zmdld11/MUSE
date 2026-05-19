# test_integration.py — 乐器识别 + 音轨分离集成测试
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.integrate import (
    load_inst_ensemble, detect_guitar_windows, smooth_presence,
    find_guitar_segments, guitar_pipeline,
)


def test_inst_ensemble_loading():
    """测试乐器识别模型加载"""
    print("测试: 乐器识别模型加载")
    models, classes = load_inst_ensemble()
    loaded = sum(1 for m in models.values() if list(m.parameters())[0].requires_grad is not None)
    print(f"  加载: {loaded}/{len(classes)} 模型")
    return loaded == len(classes)


def test_guitar_detection():
    """测试吉他检测逻辑"""
    print("\n测试: 吉他检测逻辑")

    # 模拟 30 秒音频的检测结果
    SR = 22050
    HOP = 11025
    total_samples = 30 * SR
    n_windows = (total_samples - 3 * SR) // HOP + 1  # ~55 windows

    # 模拟概率: 中间 20 个窗口有吉他
    np.random.seed(42)
    fake_probs = np.random.rand(n_windows).astype(np.float32) * 0.1
    fake_probs[15:35] = 0.5 + np.random.rand(20) * 0.3

    smoothed = smooth_presence(fake_probs)
    segments = find_guitar_segments(smoothed)

    print(f"  窗口数: {n_windows}")
    print(f"  概率范围: [{fake_probs.min():.3f}, {fake_probs.max():.3f}]")
    print(f"  检测到段: {segments}")
    return len(segments) > 0


def test_pipeline_on_ground_truth():
    """在标准测试歌曲上运行完整流水线"""
    print("\n测试: 完整流水线（标准测试歌曲）")

    gt_dir = r"D:\program_project\MUSE\data\ground_truth"
    if not os.path.isdir(gt_dir):
        print("  跳过: ground_truth 目录不存在")
        return

    songs = [d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))]
    # 优先选含吉他的歌
    guitar_songs = [s for s in songs if any(
        kw in s.lower() for kw in ["not_for_nothing", "sunspot"]
    )]
    if not guitar_songs:
        guitar_songs = songs[:1]

    song = guitar_songs[0]
    mix_path = os.path.join(gt_dir, song, "mix.wav")
    if not os.path.exists(mix_path):
        print(f"  跳过: {mix_path} 不存在")
        return

    output_path = os.path.join(config.OUTPUT_DIR, f"test_{song}_guitar.wav")

    t0 = time.perf_counter()
    result = guitar_pipeline(mix_path, output_path)
    elapsed = time.perf_counter() - t0

    print(f"  处理时间: {elapsed:.1f}s")
    print(f"  输出: {result}")


def main():
    print(f"集成测试: {config.MODEL_VERSION}")
    device = config.DEVICE
    print(f"设备: {device}")

    test_inst_ensemble_loading()
    test_guitar_detection()
    test_pipeline_on_ground_truth()


if __name__ == "__main__":
    main()
