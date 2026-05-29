# test_integration.py — 乐器识别 + 音轨分离集成推理（带进度条）
#
# 用法:
#   python test/test_integration.py                       # 扫描 music/ 目录
#   python test/test_integration.py 路径/音频.wav         # 指定单文件
#
# 输出: output/{MODEL_VERSION}/{文件名}_guitar.wav
import os
import sys
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.integrate import (
    load_inst_ensemble,
    smooth_presence, find_guitar_segments,
    extract_features, SR, SAMPLES_PER_WINDOW, _INFERENCE_HOP,
)
from src.infer import separate, load_model

OUTPUT_DIR = os.path.join(config.WORKSPACE_DIR, 'output', config.MODEL_VERSION)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = config.DEVICE
print(f"设备: {device}")
print(f"模型: {config.MODEL_VERSION}")
print(f"输出目录: {OUTPUT_DIR}")

# 预加载模型
print("加载乐器识别模型...")
inst_models, _ = load_inst_ensemble(device)
print("加载音轨分离模型...")
sep_model = load_model(device=device)


def separate_file(audio_path):
    """对单段音频做完整分离，全程显示进度条"""
    print(f"\n--- 分离: {os.path.basename(audio_path)} ---")

    # 加载音频（先转单声道再降采样）
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    audio = audio.astype(np.float32)
    audio_tensor = torch.from_numpy(audio)

    # Step 1: 吉他检测（带进度条）
    total_samples = len(audio)
    n_windows = max(0, (total_samples - SAMPLES_PER_WINDOW) // _INFERENCE_HOP + 1)
    guitar_probs = np.zeros(n_windows, dtype=np.float32)
    ac_model = inst_models["acoustic guitar"]
    eg_model = inst_models["electric guitar"]

    for w in tqdm(range(n_windows), desc=" 吉他检测", unit="窗"):
        start = w * _INFERENCE_HOP
        end = start + SAMPLES_PER_WINDOW
        window = audio_tensor[start:end]
        feat = extract_features(window.unsqueeze(0)).unsqueeze(1).to(device)
        ac_prob = torch.sigmoid(ac_model(feat)).item()
        eg_prob = torch.sigmoid(eg_model(feat)).item()
        guitar_probs[w] = max(ac_prob, eg_prob)

    # Step 2: 平滑 + 找段
    guitar_probs = smooth_presence(guitar_probs)
    segments = find_guitar_segments(guitar_probs)

    if not segments:
        print("  未检测到吉他片段，输出静音。")
        out_name = os.path.splitext(os.path.basename(audio_path))[0] + "_guitar.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        sf.write(out_path, np.zeros(len(audio), dtype=np.float32), SR)
        return out_path

    # Step 3: 对每个吉他段分离（带进度条）
    output = np.zeros(len(audio), dtype=np.float32)
    fade_len = min(_INFERENCE_HOP // 2, 512)

    for seg_start, seg_end in (
        tqdm(segments, desc=" 音轨分离", unit="段")
    ):
        t_start = seg_start * _INFERENCE_HOP
        t_end = seg_end * _INFERENCE_HOP + SAMPLES_PER_WINDOW
        t_start = max(0, t_start)
        t_end = min(len(audio), t_end)

        seg_audio = audio_tensor[t_start:t_end]
        guitar_stem = separate(seg_audio, sep_model, device)

        # 交叉淡入淡出
        if t_start > 0:
            fade_in = np.linspace(0, 1, min(fade_len, len(guitar_stem)))
            guitar_stem[:len(fade_in)] *= fade_in
        if t_end < len(audio):
            fade_out = np.linspace(1, 0, min(fade_len, len(guitar_stem)))
            guitar_stem[-len(fade_out):] *= fade_out

        output[t_start:t_end] += guitar_stem

    # 归一化
    peak = np.abs(output).max()
    if peak > 0:
        output = output / peak * 0.95

    out_name = os.path.splitext(os.path.basename(audio_path))[0] + "_guitar.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, output, SR)

    guitar_ratio = len(output.nonzero()[0]) / len(output)
    print(f"  完成: {out_path}")
    print(f"  吉他占比: {guitar_ratio*100:.1f}%  (检测到 {len(segments)} 个段落)")
    return out_path


def batch_process():
    """扫描 music/ 目录批量处理"""
    music_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'music')
    )
    if not os.path.isdir(music_dir):
        print(f"music 目录不存在: {music_dir}")
        return

    supported = ('.wav', '.mp3', '.flac', '.ogg')
    files = sorted([os.path.join(music_dir, f) for f in os.listdir(music_dir)
                    if f.lower().endswith(supported)])
    if not files:
        print(f"在 {music_dir} 中未找到音频文件")
        return

    for f in tqdm(files, desc="批量处理", unit="文件"):
        separate_file(f)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            separate_file(audio_file)
        else:
            print(f"文件不存在: {audio_file}")
            sys.exit(1)
    else:
        batch_process()
