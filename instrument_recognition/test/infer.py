import os
import torch
import librosa
import numpy as np
import importlib
import argparse

# --- 配置定义 ---
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
SEGMENT_DURATION = 1.0  # 我们按 1 秒为窗口进行切割识别
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
CLASS_NAMES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

def get_mel_spectrogram(y):
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    # 标准化到 [0, 1] 供模型使用
    S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
    return np.expand_dims(S_dB_norm, axis=0)  # Shape: (1, n_mels, t)

def load_adaptive_model(checkpoint_path, device):
    """
    自适应加载功能的实现核心：
    在 train.py 中，我们必须这样保存模型：
    torch.save({
        'state_dict': model.state_dict(),
        'model_name': 'ResNet18Instrument', # 你的类名
        'config': {'num_classes': 11} # 实例化的参数
    }, 'best_model.pth')
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_name' not in checkpoint or 'config' not in checkpoint:
        raise ValueError("模型文件缺少自适应元数据：'model_name' 和 'config'，请检查 train 逻辑。")

    # 动态通过字符串导入并实例化 src.model 中的类
    model_module = importlib.import_module("src.model")
    ModelClass = getattr(model_module, checkpoint['model_name'])
    
    model = ModelClass(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model

def analyze_audio(file_path, model, device, target_instrument=None):
    print(f"Loading '{file_path}'...")
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    num_segments = len(y) // SEGMENT_SAMPLES
    timeline_results = []
    
    print("Analyzing segments...")
    with torch.no_grad():
        for i in range(num_segments):
            start_sample = i * SEGMENT_SAMPLES
            end_sample = start_sample + SEGMENT_SAMPLES
            segment = y[start_sample:end_sample]
            
            mel_spec = get_mel_spectrogram(segment)
            mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).to(device)
            
            outputs = model(mel_tensor)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()  # 假设使用多标签二分类的 Sigmoid
            
            # 或者如果是多分类使用 softmax
            # probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
            # 找到概率大于阈值的乐器，或者得分最高的乐器
            predictions = {CLASS_NAMES[idx]: prob for idx, prob in enumerate(probs) if prob > 0.5}
            
            if predictions:
                timeline_results.append({
                    "time": f"{i}s - {i+1}s",
                    "predictions": predictions
                })

    # 输出结果
    print("\n--- Analysis Results ---")
    if target_instrument:
        print(f"Searching specific instrument: {target_instrument}")
        found = False
        for res in timeline_results:
            if target_instrument in res['predictions']:
                print(f"[{res['time']}] Confidence: {res['predictions'][target_instrument]:.2f}")
                found = True
        if not found:
            print("Target instrument not found in this track.")
    else:
        print("Timeline mapping for all instruments:")
        for res in timeline_results:
            inst_str = ", ".join([f"{k}({v:.2f})" for k, v in res['predictions'].items()])
            print(f"[{res['time']}] Detected: {inst_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Musical Instrument Detection")
    parser.add_argument("--audio", type=str, required=True, help="Path to test audio file in ../../music/")
    parser.add_argument("--model", type=str, default="../model/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--instrument", type=str, default=None, help="Filter by specific instrument (e.g. 'gac')")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = load_adaptive_model(args.model, device)
        analyze_audio(args.audio, model, device, args.instrument)
    except Exception as e:
        print(f"Error during analysis: {e}")
