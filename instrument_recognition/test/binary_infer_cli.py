"""
二分类集成推理 CLI — 对音频做逐窗口乐器检测 + 生成甘特图

用法:
  python test/binary_infer_cli.py                          # 扫描 music/ 目录
  python test/binary_infer_cli.py 路径/音频.wav            # 指定单文件
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
from binary_infer import load_ensemble, predict_file, NAME_MAP

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CLASSES = config.CLASSES
CLASS_TO_INST = {c: c.replace(' ', '_') for c in CLASSES}
INST_TO_CLASS = {v: k for k, v in CLASS_TO_INST.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_ensemble(device)

def analyze_audio(audio_path, thresholds=None):
    """分析音频，生成甘特图和概率CSV"""
    probs, inst_names = predict_file(audio_path, models, device)
    # inst_names: acoustic_guitar, cello, ...
    # CLASSES: acoustic guitar, cello, ...

    times = np.arange(probs.shape[0]) * 0.5  # hop=0.5s

    output_dir = os.path.join(config.WORKSPACE_DIR, 'output', 'VER4.0_BinaryEnsemble')
    os.makedirs(output_dir, exist_ok=True)

    # 保存概率 CSV
    if config.INFER_LOG_OUTPUT:
        csv_name = os.path.splitext(os.path.basename(audio_path))[0] + "_probs.csv"
        csv_path = os.path.join(output_dir, csv_name)
        header = "window_start," + ",".join(CLASSES)
        # 将 inst_names 映射回 display name
        display_probs = np.zeros((probs.shape[0], len(CLASSES)))
        for i, inst in enumerate(inst_names):
            cls = INST_TO_CLASS.get(inst, inst)
            display_probs[:, CLASSES.index(cls)] = probs[:, i]
        np.savetxt(csv_path, np.column_stack([times, display_probs]),
                   delimiter=",", fmt="%.6f", header=header, comments="")
        print(f"  概率日志: {csv_path}")

    # 默认阈值 0.5
    if thresholds is None:
        thresholds = [0.5] * len(CLASSES)

    # 甘特图
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap('tab20')
    for i, cls_name in enumerate(CLASSES):
        inst = CLASS_TO_INST[cls_name]
        if inst in inst_names:
            idx = inst_names.index(inst)
            th = thresholds[i]
            active = []
            for t_idx, t in enumerate(times):
                if probs[t_idx, idx] >= th:
                    active.append((t, config.DURATION))
            if active:
                ax.broken_barh(active, (i - 0.4, 0.8), facecolors=cmap(i % 20))

    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("时间 (s)")
    ax.set_title(f"【{os.path.basename(audio_path)}】 乐器激活时间 (阈值={thresholds[0]:.2f})")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    out_name = os.path.splitext(os.path.basename(audio_path))[0] + ".png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path)
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
