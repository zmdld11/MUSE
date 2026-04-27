import os
import torch
import librosa
import numpy as np
import sys

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.model import SimplifiedAdvancedClassifier

def infer(audio_path, model_path):
    device = config.DEVICE
    
    classes = sorted([d for d in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, d))])
    
    model = SimplifiedAdvancedClassifier(num_classes=len(classes)).to(device)
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded Model Version: {ckpt.get('version', 'Unknown')}")
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    
    y, sr = librosa.load(audio_path, sr=config.SR, duration=config.DURATION)
    if len(y) < config.SR * config.DURATION:
        pad_length = config.SR * config.DURATION - len(y)
        y = np.pad(y, (0, pad_length))
        
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
    
    features = np.vstack([mel_db, mfcc])[np.newaxis, ...]
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probabilities, 1)
        
    predicted_class = classes[top_idx.item()]
    print(f"Predicted class: {predicted_class} (Probability: {top_prob.item():.4f})")
    return predicted_class

if __name__ == '__main__':
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = 'sample.wav'
        
    model_file = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if os.path.exists(audio_file) and os.path.exists(model_file):
        infer(audio_file, model_file)
    else:
        print('Model or audio file missing.')
