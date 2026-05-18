"""
二分类集成推理 CLI — 对音频做逐窗口乐器检测 + 生成甘特图

用法:
  python test/binfer_cli.py                          # 扫描 music/ 目录
  python test/binfer_cli.py 路径/音频.wav            # 指定单文件

后处理:
  - 3 帧移动平均平滑
  - 频段门控 (piano→bass, singer→violin 等)
  - 最短连续激活 2 帧 (去孤立误报)
  - 按类独立阈值
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

from src.config import config
from binfer import load_ensemble, predict_file, post_process, NAME_MAP_TO_INST

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CLASSES = config.CLASSES
CLASS_TO_INST = {c: NAME_MAP_TO_INST.get(c, c.replace(' ', '_')) for c in CLASSES}
INST_TO_CLASS = {v: k for k, v in CLASS_TO_INST.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_ensemble(device)

# 从 ground truth 评估得到的最优阈值 (可根据 eval 结果调整)
DEFAULT_THRESHOLDS = {
    'acoustic guitar': 0.35,
    'cello': 0.50,
    'drum set': 0.40,
    'electric bass': 0.30,
    'electric guitar': 0.50,
    'flute': 0.40,
    'piano': 0.40,
    'singer': 0.35,
    'synthesizer': 0.65,
    'violin': 0.30,
}


def analyze_audio(audio_path):
    """分析音频：预测 + 后处理 + 甘特图 + CSV"""
    # 1. 原始预测
    raw_probs, inst_names = predict_file(audio_path, models, device)
    times = np.arange(raw_probs.shape[0]) * 0.5

    # 2. 后处理：门控 + 去孤立帧
    thresholds = [DEFAULT_THRESHOLDS.get(c, 0.4) for c in CLASSES]
    # 将 thresholds 映射到 inst_names 顺序
    inst_thresholds = []
    for inst in inst_names:
        cls = INST_TO_CLASS.get(inst, inst)
        idx = CLASSES.index(cls) if cls in CLASSES else 0
        inst_thresholds.append(thresholds[idx])

    probs, binary = post_process(raw_probs, inst_names, thresholds=inst_thresholds, min_active_frames=2)

    output_dir = os.path.join(config.WORKSPACE_DIR, 'output', 'VER4.0_BinaryEnsemble')
    os.makedirs(output_dir, exist_ok=True)

    # 3. 概率 CSV (去噪后)
    if config.INFER_LOG_OUTPUT:
        csv_name = os.path.splitext(os.path.basename(audio_path))[0] + "_probs.csv"
        csv_path = os.path.join(output_dir, csv_name)
        header = "window_start," + ",".join(CLASSES)
        display_probs = np.zeros((probs.shape[0], len(CLASSES)))
        for i, inst in enumerate(inst_names):
            cls = INST_TO_CLASS.get(inst, inst)
            if cls in CLASSES:
                display_probs[:, CLASSES.index(cls)] = probs[:, i]
        np.savetxt(csv_path, np.column_stack([times, display_probs]),
                   delimiter=",", fmt="%.6f", header=header, comments="")
        print(f"  概率日志: {csv_path}")

    # 4. 甘特图 (用二值化后的结果)
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap('tab20')
    for i, cls_name in enumerate(CLASSES):
        inst = CLASS_TO_INST.get(cls_name)
        if inst is None or inst not in inst_names:
            continue
        idx = inst_names.index(inst)
        active_segments = []
        in_active = False
        start_t = 0
        for t, is_active in enumerate(binary[:, idx]):
            if is_active and not in_active:
                start_t = t
                in_active = True
            elif not is_active and in_active:
                active_segments.append((start_t * 0.5, (t - start_t) * 0.5 + 3))
                in_active = False
        if in_active:
            active_segments.append((start_t * 0.5, (len(binary) - start_t) * 0.5 + 3))
        if active_segments:
            ax.broken_barh(active_segments, (i - 0.4, 0.8), facecolors=cmap(i % 20))

    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("时间 (s)")
    ax.set_title(f"【{os.path.basename(audio_path)}】 乐器激活时间")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    out_name = os.path.splitext(os.path.basename(audio_path))[0] + ".png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  甘特图: {out_path}")


if __name__ == '__main__':
    music_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'music'))

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            print(f"--- 分析: {os.path.basename(audio_file)} ---")
            analyze_audio(audio_file)
        else:
            print(f"文件不存在: {audio_file}")
            sys.exit(1)
    else:
        if not os.path.exists(music_dir):
            print(f"music 目录不存在: {music_dir}")
            sys.exit(1)
        supported = ('.wav', '.mp3', '.flac', '.ogg')
        files = [os.path.join(music_dir, f) for f in os.listdir(music_dir)
                 if f.lower().endswith(supported)]
        if not files:
            print(f"在 {music_dir} 中未找到音频文件")
            sys.exit(1)
        print(f"找到 {len(files)} 个音频文件，开始批量推理...")
        for f in files:
            print(f"\n--- 分析: {os.path.basename(f)} ---")
            analyze_audio(f)
