import os
import torch
import librosa
import numpy as np
import sys
import matplotlib.pyplot as plt

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.model import SimplifiedAdvancedClassifier

def load_model(model_path, classes):
    device = config.DEVICE
    model = SimplifiedAdvancedClassifier(num_classes=len(classes)).to(device)
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded Model Version: {ckpt.get('version', 'Unknown')}")
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    return model, device

def analyze_audio(audio_path, model, classes, device):
    y, sr = librosa.load(audio_path, sr=config.SR)
    window_length = int(config.SR * config.DURATION)
    hop_length = int(config.SR * 0.5) # 0.5 seconds sliding window
    
    if len(y) < window_length:
        pad_length = window_length - len(y)
        y = np.pad(y, (0, pad_length))
        
    num_windows = (len(y) - window_length) // hop_length + 1
    if num_windows <= 0:
        num_windows = 1
        
    times = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(num_windows):
            start = i * hop_length
            end = start + window_length
            segment = y[start:end]
            
            if len(segment) < window_length:
                segment = np.pad(segment, (0, window_length - len(segment)))
                
            mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=config.N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=config.N_MFCC)
            
            features = np.vstack([mel_db, mfcc])[np.newaxis, ...]
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            
            times.append(start / sr)
            all_probs.append(probs)
            
    all_probs = np.array(all_probs)
    
    # Plotting result
    plt.figure(figsize=(14, 6))
    for i in range(len(classes)):
        plt.plot(times, all_probs[:, i], label=classes[i])
        
    plt.title(f"Instrument Probabilities Over Time: {os.path.basename(audio_path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Probability")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.grid(True)
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
    classes = sorted([d for d in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, d))])
    model, device = load_model(model_path, classes)
    analyze_audio(audio_path, model, classes, device)

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
            classes = sorted([d for d in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, d))])
            model, device = load_model(model_file, classes)
            for idx, audio_file in enumerate(audio_files, 1):
                print(f"\n[{idx}/{len(audio_files)}] 正在分析文件: {os.path.basename(audio_file)} ...")
                analyze_audio(audio_file, model, classes, device)
