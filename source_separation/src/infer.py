# infer.py — DemucsLM 时域推理引擎
import os, torch
import numpy as np
from src.config import config
from src.model import DemucsLM


def load_model(model_path=None, device=None):
    if device is None: device = config.DEVICE
    if model_path is None: model_path = os.path.join(config.MODEL_DIR, "guitar.pth")
    model = DemucsLM(channels=config.DEMUCS_CHANNELS).to(device)
    if os.path.exists(model_path):
        ck = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        print(f"加载模型: {model_path}")
    return model


@torch.no_grad()
def separate(mix_audio, model, device=None):
    if device is None: device = next(model.parameters()).device
    model.eval()
    mix = mix_audio.to(device)
    x = mix.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    pred = model(x)
    return pred.squeeze().cpu().numpy()
