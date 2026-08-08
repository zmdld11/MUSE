"""Layer 2: VER2.0 — uses our trained OnsetsAndFrames bootstrap model."""
import logging
import os

import librosa
import numpy as np
import torch

from src.config import config

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    model_path = os.environ.get(
        "MUSE_MODEL_PATH",
        os.path.join(config.WORKSPACE_DIR, "model", "VER2.0_Bootstrap.pth"),
    )
    if not os.path.exists(model_path):
        logger.warning(f"No trained model at {model_path}, falling back to basic-pitch")
        return None

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    from train.model_v4 import OnsetsFramesV4
    if any(k.startswith("transformer.") for k in state):
        _model = OnsetsFramesV4(n_mels=229, n_midi=88, backend="transformer")
        logger.info("V4 Transformer model detected")
    else:
        from train.model import OnsetsAndFrames
        _model = OnsetsAndFrames(n_mels=229, n_midi=88)
        logger.info("VER2/3 LSTM model detected")
    _model.load_state_dict(state)
    _model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    _model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device)
    logger.info(f"Trained model loaded on {device}")
    return _model


def transcribe(audio_path: str) -> dict:
    """Transcribe audio → onset/frame probability maps."""
    model = _load_model()

    if model is not None:
        return _ours_inference(model, audio_path)
    else:
        return _basic_pitch_inference(audio_path)


def _ours_inference(model, audio_path: str) -> dict:
    device = next(model.parameters()).device
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=22050, n_mels=229, hop_length=512, fmin=30, fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip((mel_db + 80) / 80, -1, 1)

    from train.model_v4 import OnsetsFramesV4
    spec = torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0).to(device)
    if isinstance(model, OnsetsFramesV4):
        # V4 Transformer: 位置编码上限 2048 帧 + 注意力 O(T^2), 必须分窗推理
        T = mel_db.shape[1]
        CHUNK, OVERLAP = 1291, 43  # ~30s 窗 + 1s 重叠
        onset_acc = np.zeros((T, 88), dtype=np.float64)
        frame_acc = np.zeros((T, 88), dtype=np.float64)
        wsum = np.zeros((T, 1), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, T, CHUNK - OVERLAP):
                end = min(start + CHUNK, T)
                if end - start < 64:
                    break
                pred = model(spec[:, :, :, start:end])
                o = pred["onset"].squeeze(0).cpu().numpy()[: end - start]
                f = pred["frame"].squeeze(0).cpu().numpy()[: end - start]
                onset_acc[start:end] += o
                frame_acc[start:end] += f
                wsum[start:end] += 1
        onset = (onset_acc / np.maximum(wsum, 1)).astype(np.float32)
        frame = (frame_acc / np.maximum(wsum, 1)).astype(np.float32)
    else:
        with torch.no_grad():
            pred = model(spec)
        onset = pred["onset"].squeeze(0).cpu().numpy()
        frame = pred["frame"].squeeze(0).cpu().numpy()
    logger.info(f"Our model: onset={onset.shape}, frame={frame.shape}")
    return {"onset_probs": onset, "frame_probs": frame, "contour": np.zeros((onset.shape[0], 264), dtype=np.float32), "sr": 22050, "hop_length": 512}

def _basic_pitch_inference(audio_path: str) -> dict:
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        mo, _, _ = predict(audio_path, model_or_model_path=ICASSP_2022_MODEL_PATH,
                           onset_threshold=0.4, frame_threshold=0.2)
        # Handle both tensor (.numpy()) and numpy return types
        onset = mo["onset"].numpy() if hasattr(mo["onset"], "numpy") else np.array(mo["onset"])
        frame = mo["note"].numpy() if hasattr(mo["note"], "numpy") else np.array(mo["note"])
        contour = mo["contour"].numpy() if hasattr(mo["contour"], "numpy") else np.array(mo["contour"])
        logger.info(f"basic-pitch fallback: onset={onset.shape}")
        # basic-pitch uses hop_length=256 at 22050Hz internally
        return {"onset_probs": onset, "frame_probs": frame, "contour": contour, "sr": 22050, "hop_length": 256}
    except Exception as e:
        logger.error(f"basic-pitch failed: {e}")
        return {"onset_probs": np.zeros((1,88),np.float32), "frame_probs": np.zeros((1,88),np.float32), "contour": np.zeros((1,264),np.float32), "sr": 22050}
