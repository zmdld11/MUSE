# infer.py — UNet 推理引擎
import os, torch
from src.config import config
from src.model import UNet

N_FFT = 1024
HOP = 256


def load_model(model_path=None, device=None):
    if device is None: device = config.DEVICE
    if model_path is None: model_path = os.path.join(config.MODEL_DIR, "guitar.pth")
    model = UNet(channels=config.UNET_CHANNELS).to(device)
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
    orig_len = len(mix_audio)
    mix = mix_audio.to(device)
    window = torch.hann_window(N_FFT, device=device)

    X = torch.stft(mix, N_FFT, HOP, window=window, return_complex=True)
    r, i = X.real, X.imag
    spec_in = torch.stack([r, i], dim=0).unsqueeze(0)  # [1, 2, F, T]

    mask = model(spec_in).squeeze(0)  # [2, F, T]
    r_hat = mask[0] * r - mask[1] * i
    i_hat = mask[0] * i + mask[1] * r

    out = torch.istft(torch.complex(r_hat, i_hat), N_FFT, HOP, window=window, length=orig_len)
    return out.cpu().numpy()
