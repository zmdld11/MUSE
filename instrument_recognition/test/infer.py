import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure matplotlib to support Chinese characters properly
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.model import SimplifiedAdvancedClassifier, TransformerClassifier


def load_per_class_thresholds(classes):
    """加载逐类最优阈值。若文件不存在则尝试从验证集计算，都不行则返回默认0.5。"""
    threshold_path = os.path.join(config.MODEL_DIR, "class_thresholds.json")

    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            thresholds_dict = json.load(f)
        thresholds = [thresholds_dict.get(c, 0.5) for c in classes]
        print(f"已加载逐类阈值: {dict(zip(classes, thresholds))}")
        return thresholds

    # 尝试验证集计算
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from data.dataset import get_dataloaders
        _, val_loader, val_classes, _, _ = get_dataloaders(val_split=0.2)
        print("正在从验证集计算逐类最优阈值...")
        thresholds = compute_thresholds_from_loader(val_loader, val_classes)
        return thresholds
    except Exception as e:
        print(f"无法计算逐类阈值 ({e})，使用默认阈值 0.5")
        return [0.5] * len(classes)


def compute_thresholds_from_loader(val_loader, classes):
    """在验证集上为每个类别搜索最优F1阈值。"""
    device = config.DEVICE
    model_path = os.path.join(config.MODEL_DIR, "best_model.pth")
    model, _ = load_model(model_path, classes)
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu()
            all_probs.append(probs)
            all_targets.append(targets)

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    per_class_thresholds = {}
    for c, cls_name in enumerate(classes):
        best_f1 = 0.0
        best_thresh = 0.5
        probs_c = all_probs[:, c].numpy()
        targets_c = all_targets[:, c].numpy()
        for thresh in np.arange(0.2, 0.85, 0.05):
            preds = (probs_c >= thresh).astype(float)
            tp = ((preds == 1) & (targets_c == 1)).sum()
            fp = ((preds == 1) & (targets_c == 0)).sum()
            fn = ((preds == 0) & (targets_c == 1)).sum()
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        per_class_thresholds[cls_name] = round(float(best_thresh), 2)
        print(f"  {cls_name}: 最佳阈值={best_thresh:.2f}, F1={best_f1:.4f}")

    # 保存供下次使用
    threshold_path = os.path.join(config.MODEL_DIR, "class_thresholds.json")
    with open(threshold_path, "w") as f:
        json.dump(per_class_thresholds, f, indent=2)
    print(f"逐类阈值已保存至: {threshold_path}")

    return [per_class_thresholds[c] for c in classes]


def load_model(model_path, classes):
    device = config.DEVICE

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    version = ckpt.get('version', '') if isinstance(ckpt, dict) else ''

    # 根据Checkpoint版本自动选择模型结构
    if 'Transformer' in version or 'VER3' in version:
        model = TransformerClassifier(num_classes=len(classes)).to(device)
    else:
        model = SimplifiedAdvancedClassifier(num_classes=len(classes)).to(device)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded Model Version: {ckpt.get('version', 'Unknown')}")
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model, device

def analyze_audio(audio_path, model, classes, device, thresholds=None):
    """分析音频中的乐器分布。
    Args:
        thresholds: 逐类判定阈值列表 [C]，若为None则使用全局 0.5
    """
    if thresholds is None:
        thresholds = [0.5] * len(classes)
    audio, sr = torchaudio.load(audio_path)
    if sr != config.SR:
        audio = torchaudio.functional.resample(audio, sr, config.SR)
    
    # 确保单声道
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        
    y = audio[0] # 形如 (time_len)
    
    window_length = int(config.SR * config.DURATION)
    hop_length = int(config.SR * 0.5) # 0.5 seconds sliding window
    
    if y.shape[0] < window_length:
        y = torch.nn.functional.pad(y, (0, window_length - y.shape[0]))
        
    num_windows = (y.shape[0] - window_length) // hop_length + 1
    if num_windows <= 0:
        num_windows = 1
        
    times = []
    all_probs = []
    
    # 保持和训练集（dataset.py）完全一样的特征提取器，并且都在 CPU 上完成预处理(避免 CUDA nvrtc JIT编译错误)
    mel_transform = T.MelSpectrogram(sample_rate=config.SR, n_mels=config.N_MELS, n_fft=2048, hop_length=512)
    db_transform = T.AmplitudeToDB(stype="power", top_db=80)
    mfcc_transform = T.MFCC(sample_rate=config.SR, n_mfcc=config.N_MFCC, melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": config.N_MELS})
    
    with torch.no_grad():
        for i in tqdm(range(num_windows), desc=f"扫描进度", leave=False):
            start = i * hop_length
            end = start + window_length
            segment = y[start:end]
            
            if segment.shape[0] < window_length:
                segment = torch.nn.functional.pad(segment, (0, window_length - segment.shape[0]))
                
            segment_tensor = segment.unsqueeze(0) # shape: [1, time] 默认在cpu
            
            mel = mel_transform(segment_tensor)
            mel_db = db_transform(mel)
            mfcc = mfcc_transform(segment_tensor)
            
            # 沿着特征维度进行合并: [1, features, time_frames]
            feats = torch.cat([mel_db, mfcc], dim=1)
            
            # 模型需要 Batch 维度： [1, 1, 141, time_frames] 并移动至显卡推理
            input_tensor = feats.unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            # [VER2.0] Softmax 是单选题概率(和为1)，多标签独立存在应使用 Sigmoid (每个乐器从 0 到 1 互不干涉)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
            
            times.append(start / config.SR)
            all_probs.append(probs)
            
    all_probs = np.array(all_probs)
    
    # 对抗幻觉：对概率做更宽窗口的移动平均平滑，孤立短时误报无法通过
    smooth_window = 5
    for i in range(all_probs.shape[1]):
        all_probs[:, i] = np.convolve(all_probs[:, i], np.ones(smooth_window)/smooth_window, mode='same')

    # 逐类自适应判定阈值：每个乐器有自己最优的判断"及格线"
    window_sec = config.DURATION # 每个预测窗口代表的总时长(秒)

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap('tab20')

    for i, instrument in enumerate(classes):
        thresh = thresholds[i]
        active_segments = []
        for t_idx, t in enumerate(times):
            if all_probs[t_idx, i] >= thresh:
                # 绘制从该窗口起始点往后延伸的一整段时间
                active_segments.append((t, window_sec))

        if active_segments:
            ax.broken_barh(active_segments, (i - 0.4, 0.8), facecolors=cmap(i % 20))

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("时间 / Time (s)")
    ax.set_title(f"【{os.path.basename(audio_path)}】 乐器激活时间横轴图 (逐类自适应阈值: {dict(zip(classes, thresholds))})")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_dir = os.path.join(config.WORKSPACE_DIR, 'output', config.MODEL_VERSION)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"[{os.path.basename(audio_path)}] 分析图表已保存至: {output_path}")

def infer(audio_path, model_path):
    # This is a legacy fallback wrapper
    classes = config.CLASSES
    model, device = load_model(model_path, classes)
    thresholds = load_per_class_thresholds(classes)
    analyze_audio(audio_path, model, classes, device, thresholds)

if __name__ == '__main__':
    muse_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    music_dir = os.path.join(muse_root, 'music')
    model_file = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    if not os.path.exists(model_file):
        print(f"找不到模型文件: {model_file}")
        print("请检查模型是否训练完成或路径是否正确。")
        sys.exit(1)

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"找不到音频文件: {audio_file}")
        else:
            print(f"--- 正在预测文件: {os.path.basename(audio_file)} ---")
            infer(audio_file, model_file)
    else:
        # 如果没有传入参数，默认推断 MUSE根目录/music/ 文件夹下的所有测试文件
        if not os.path.exists(music_dir):
            print(f"找不到音乐文件夹: {music_dir}")
            sys.exit(1)
            
        supported_formats = ('.wav', '.mp3', '.flac', '.ogg')
        audio_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir) if f.lower().endswith(supported_formats)]
        
        if not audio_files:
            print(f"警告: 在 {music_dir} 下没有找到支持的音频文件 {supported_formats}。")
            print(f"你可以放一些音频文件进去，或者在运行命令时指定音频文件路径:")
            print(f"python test/infer.py D:/program_project/MUSE/music/你的音频.wav")
        else:
            print(f"在 {music_dir} 中找到 {len(audio_files)} 个音频文件。开始批量推断...")
            classes = config.CLASSES
            model, device = load_model(model_file, classes)
            thresholds = load_per_class_thresholds(classes)
            for idx, audio_file in enumerate(audio_files, 1):
                print(f"\n[{idx}/{len(audio_files)}] 正在分析文件: {os.path.basename(audio_file)} ...")
                analyze_audio(audio_file, model, classes, device, thresholds)
