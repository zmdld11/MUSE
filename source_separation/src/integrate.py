# integrate.py — 乐器识别 + 音轨分离集成流水线
# 先用 BinaryEnsemble 检测吉他片段 → 仅对吉他段做分离 → 静音拼接
import os
import sys
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np

# 通过文件路径直接导入 BinaryInstrumentClassifier，避免与 source_separation 的 src 包冲突
import importlib.util as _iu
_bmodel_spec = _iu.spec_from_file_location(
    "inst_bmodel",
    r"D:\program_project\MUSE\instrument_recognition\src\bmodel.py"
)
_bmodel = _iu.module_from_spec(_bmodel_spec)
_bmodel_spec.loader.exec_module(_bmodel)
BinaryInstrumentClassifier = _bmodel.BinaryInstrumentClassifier

from src.config import config
from src.model import LightweightUMX
from src.infer import separate, N_FFT, HOP_LENGTH, N_BINS

# —— 特征提取器（与 instrument_recognition/src/btrain.py 保持一致）——
SR = 22050
DURATION = 3
SAMPLES_PER_WINDOW = SR * DURATION  # 66150
_INFERENCE_HOP = int(SR * 0.5)       # 11025

MEL = T.MelSpectrogram(sample_rate=SR, n_mels=128, n_fft=2048, hop_length=512)
DB = T.AmplitudeToDB(stype="power", top_db=80)
MFCC = T.MFCC(sample_rate=SR, n_mfcc=13, melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128})
MODGD_MEL = T.MelScale(n_mels=128, sample_rate=SR, n_stft=2048 // 2 + 1)


def _compute_modgd(audio, gamma=0.3):
    n_fft, hop_len = 2048, 512
    device = audio.device
    window = torch.hann_window(n_fft, device=device)
    X = torch.stft(audio, n_fft=n_fft, hop_length=hop_len, win_length=n_fft,
                   window=window, return_complex=True)
    n = torch.arange(audio.shape[-1], device=device).float() / audio.shape[-1]
    Y = torch.stft(audio * n, n_fft=n_fft, hop_length=hop_len, win_length=n_fft,
                   window=window, return_complex=True)
    tau = X.real * Y.real + X.imag * Y.imag
    S = torch.abs(X)
    S_s = (S[:, :, :-2] + S[:, :, 1:-1] + S[:, :, 2:]) / 3.0
    S_s = nn.functional.pad(S_s, (1, 1), mode='replicate')
    tau = tau / torch.clamp(S_s ** (2 * gamma), min=1e-6)
    mn, mx = tau.min(dim=1, keepdim=True).values, tau.max(dim=1, keepdim=True).values
    tau = (tau - mn) / (mx - mn + 1e-8)
    return MODGD_MEL(tau)


def extract_features(window_audio):
    """提取 3s 窗口的 269 通道特征"""
    mel = DB(MEL(window_audio))        # [1, 128, T]
    mfcc = MFCC(window_audio)          # [1, 13, T]
    modgd = _compute_modgd(window_audio)  # [1, 128, T]
    return torch.cat([mel, mfcc, modgd], dim=1)  # [1, 269, T]


# —— 吉他检测阈值 ——
GUITAR_THRESHOLDS = {
    "acoustic guitar": 0.35,
    "electric guitar": 0.50,
}


def load_inst_ensemble(device=None):
    """加载 10 个乐器识别二分类模型"""
    if device is None:
        device = config.DEVICE

    model_dir = config.INST_MODEL_DIR
    classes = ['acoustic guitar', 'cello', 'drum set', 'electric bass',
               'electric guitar', 'flute', 'piano', 'singer', 'synthesizer', 'violin']

    models = {}
    for cls_name in classes:
        safe_name = cls_name.replace(" ", "_")
        path = os.path.join(model_dir, f"{safe_name}.pth")
        model = BinaryInstrumentClassifier(n_mels=269).to(device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
        else:
            print(f"警告: 模型不存在 {path}")
        models[cls_name] = model

    return models, classes


@torch.no_grad()
def detect_guitar_windows(audio, inst_models, device=None):
    """
    滑动窗口检测吉他片段

    Returns:
        guitar_probs: [n_windows] 每窗口的吉他概率 (acoustic 和 electric 的最大值)
    """
    if device is None:
        device = config.DEVICE

    total_samples = len(audio)
    n_windows = max(0, (total_samples - SAMPLES_PER_WINDOW) // _INFERENCE_HOP + 1)
    guitar_probs = np.zeros(n_windows, dtype=np.float32)

    ac_model = inst_models["acoustic guitar"]
    eg_model = inst_models["electric guitar"]

    for w in range(n_windows):
        start = w * _INFERENCE_HOP
        end = start + SAMPLES_PER_WINDOW
        window = audio[start:end]

        feat = extract_features(window.unsqueeze(0)).unsqueeze(1).to(device)  # [1, 1, 269, T]

        ac_prob = torch.sigmoid(ac_model(feat)).item()
        eg_prob = torch.sigmoid(eg_model(feat)).item()
        guitar_probs[w] = max(ac_prob, eg_prob)

    return guitar_probs


def smooth_presence(probs, window=3):
    """移动平均平滑吉他存在概率"""
    if len(probs) < window:
        return probs
    kernel = np.ones(window) / window
    return np.convolve(probs, kernel, mode='same')


def find_guitar_segments(guitar_probs, min_frames=2):
    """
    从概率序列中提取连续吉他段

    Returns:
        segments: [(start_frame, end_frame), ...] 帧索引（含端点）
    """
    threshold = max(
        GUITAR_THRESHOLDS["acoustic guitar"],
        GUITAR_THRESHOLDS["electric guitar"],
    )  # 使用较高阈值判断

    active = guitar_probs >= threshold
    segments = []
    i = 0
    while i < len(active):
        if active[i]:
            start = i
            while i < len(active) and active[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_frames:
                # 边界扩展 1 窗口
                seg_start = max(0, start - 1)
                seg_end = min(len(active) - 1, end + 1)
                segments.append((seg_start, seg_end))
        else:
            i += 1
    return segments


def guitar_pipeline(audio_path, output_path=None, device=None):
    """
    全自动吉他分离流水线

    1. 加载乐器识别模型 + 分离模型
    2. 滑动窗口检测吉他
    3. 对吉他段执行分离
    4. 拼接输出（非吉他段静音）

    Args:
        audio_path: 输入音频文件路径
        output_path: 输出吉他音轨路径 (默认: output/guitar_stem.wav)
        device: torch device

    Returns:
        output_path: 输出文件路径
    """
    import soundfile as sf

    if device is None:
        device = config.DEVICE
    if output_path is None:
        output_path = os.path.join(config.OUTPUT_DIR, "guitar_stem.wav")

    print(f"输入: {audio_path}")
    print(f"设备: {device}")

    # 加载音频
    audio, sr = sf.read(audio_path)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    audio_tensor = torch.from_numpy(audio)

    print(f"音频: {len(audio)/SR:.1f}s, {SR}Hz")

    # Step 1: 加载乐器识别模型
    print("加载乐器识别集成模型...")
    inst_models, _ = load_inst_ensemble(device)

    # Step 2: 加载分离模型
    print("加载音轨分离模型...")
    sep_model = load_model(device=device)

    # Step 3: 吉他检测
    print("检测吉他片段...")
    guitar_probs = detect_guitar_windows(audio_tensor, inst_models, device)
    guitar_probs = smooth_presence(guitar_probs)
    segments = find_guitar_segments(guitar_probs)

    if not segments:
        print("警告: 未检测到吉他片段！输出静音。")
        sf.write(output_path, np.zeros(len(audio)), SR)
        return output_path

    print(f"检测到 {len(segments)} 个吉他段: {segments}")

    # Step 4: 对每个吉他段执行分离
    output = np.zeros(len(audio), dtype=np.float32)
    fade_len = min(_INFERENCE_HOP // 2, 512)  # 交叉淡入淡出长度

    for seg_start, seg_end in segments:
        # 帧索引 → 采样点
        t_start = seg_start * _INFERENCE_HOP
        t_end = seg_end * _INFERENCE_HOP + SAMPLES_PER_WINDOW
        t_start = max(0, t_start)
        t_end = min(len(audio), t_end)

        seg_audio = audio_tensor[t_start:t_end]
        guitar_stem = separate(seg_audio, sep_model, device)

        # 交叉淡入淡出写入
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

    sf.write(output_path, output, SR)
    print(f"输出: {output_path}")
    print(f"吉他占比: {segments[-1][1] - segments[0][0]}/{len(guitar_probs)} 窗口 ≈ {len(output.nonzero()[0])/len(output)*100:.1f}%")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="吉他分离集成流水线")
    parser.add_argument("audio", help="输入音频路径")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else config.DEVICE
    guitar_pipeline(args.audio, args.output, device)
