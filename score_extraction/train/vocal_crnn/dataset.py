"""synth v1 → 帧级四头目标（M1-CRNN，2026-08-31）。

mel 前端自算（torch.stft + numpy HTK 三角滤波器组）——不依赖 torchaudio
（服务器坏）也不依赖 librosa（mel 计算全程 numpy/torch，跨端一致）。

GT 来源（已核实 gen_vibrato_synth_v1.py + 实测 npz/manifest）：
- manifest.json 音符表 notes[{onset, offset, pitch}]（主 GT，帧目标从这里来）
- clean/{id}.gt.npz: f0 / vib_depth / vib_rate @100fps（vib_depth 回归目标）
- SR=22050，段长 5-10.3s，1000 段 / 14620 音

帧对齐：hop=220 → 帧率 22050/220 ≈ 100.227fps；帧 i 的时间 = i*hop/SR，
音符/颤音 GT 按该时间轴重采样（vib 用线性插值）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 22050
N_FFT = 1024
HOP = 220          # ≈9.98ms/帧
N_MELS = 128
FMIN = 30.0
FMAX = SR / 2      # 11025
HOP_SEC = HOP / SR

PITCH_LO, PITCH_HI = 40, 84          # 冻结口径音域门
N_PITCH_CLS = PITCH_HI - PITCH_LO + 1  # 45
N_CLASSES = N_PITCH_CLS + 1           # +1：class 0 = unvoiced


# ------------------------------ mel 前端 ------------------------------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


_MEL_FB: np.ndarray | None = None


def mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """HTK 式三角滤波器组 (n_mels, n_fft//2+1)，缓存。"""
    global _MEL_FB
    if _MEL_FB is not None:
        return _MEL_FB
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - l) / max(c - l, 1e-9)
        down = (r - freqs) / max(r - c, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    _MEL_FB = fb
    return fb


def wav_to_mel(wav: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """(n,) float wav → (T, 128) log-mel（已逐样本零均值/单位方差）。"""
    y = torch.as_tensor(np.ascontiguousarray(wav), dtype=torch.float32,
                        device=device)
    window = torch.hann_window(N_FFT, device=device)
    spec_c = torch.stft(y, N_FFT, hop_length=HOP, win_length=N_FFT,
                        window=window, center=True,
                        return_complex=True)          # (F, T)
    spec = spec_c.abs().pow(2)                        # 功率谱
    fb = torch.as_tensor(mel_filterbank(), device=device)
    mel = fb @ spec                                     # (128, T)
    mel = torch.log(mel + 1e-5).T                       # (T, 128)
    mel = (mel - mel.mean()) / (mel.std() + 1e-5)
    return mel


# ------------------------------ 帧级目标 ------------------------------

def frame_targets(notes: list[dict], vib_depth: np.ndarray | None,
                  n_frames: int, gt_fps: float = 100.0,
                  vowel_onsets: list[float] | None = None) -> dict:
    """manifest 音符表 + npz 颤音深度 → 帧级目标。

    - onset/offset：音符起/止帧置 1（offset 帧=音符结束所在帧）
    - pitch：voiced 帧 = pitch-PITCH_LO+1（1..45），unvoiced = 0
    - vib：npz vib_depth(100fps) 线性插值到模型帧率
    - vowel（M3）：元音起始帧置 1（无标注段全 0，loss 由 vowel_valid 掩码）
    """
    onset = np.zeros(n_frames, dtype=np.float32)
    offset = np.zeros(n_frames, dtype=np.float32)
    pitch = np.zeros(n_frames, dtype=np.int64)
    for nt in notes:
        if isinstance(nt, dict):                     # synth manifest 格式
            on, off, p = float(nt["onset"]), float(nt["offset"]), int(nt["pitch"])
        else:                                        # 真实数据 [[on,off,pitch]]
            on, off, p = float(nt[0]), float(nt[1]), int(nt[2])
        if not (PITCH_LO <= p <= PITCH_HI):
            continue
        i0 = int(round(on / HOP_SEC))
        i1 = int(round(off / HOP_SEC))
        i0 = max(0, min(i0, n_frames - 1))
        i1 = max(i0 + 1, min(i1, n_frames))
        onset[i0] = 1.0
        offset[min(i1, n_frames - 1)] = 1.0
        pitch[i0:i1] = p - PITCH_LO + 1
    if vib_depth is not None and len(vib_depth):
        src_t = np.arange(len(vib_depth)) / gt_fps
        dst_t = np.arange(n_frames) * HOP_SEC
        vib = np.interp(dst_t, src_t, vib_depth.astype(np.float64),
                        left=0.0, right=0.0).astype(np.float32)
    else:
        vib = np.zeros(n_frames, dtype=np.float32)
    vowel = np.zeros(n_frames, dtype=np.float32)
    for t in (vowel_onsets or []):
        i = int(round(float(t) / HOP_SEC))
        if 0 <= i < n_frames:
            vowel[i] = 1.0
    return {"onset": onset, "offset": offset, "pitch": pitch, "vib": vib,
            "vowel": vowel,
            "voiced": (pitch > 0).astype(np.float32)}


# ------------------------------ Dataset ------------------------------

def load_manifest(data_dir: str | Path) -> list[dict]:
    """manifest.json 优先；缺省则合并 manifest_*.json（GTSinger 按语言增量）。"""
    root = Path(data_dir)
    mf = root / "manifest.json"
    if mf.exists():
        man = json.loads(mf.read_text(encoding="utf-8"))
    else:
        man = []
        for f in sorted(root.glob("manifest_*.json")):
            man += json.loads(f.read_text(encoding="utf-8"))
    return sorted(man, key=lambda m: m["id"])


def split_ids(man: list[dict], val_frac: float = 0.1) -> tuple[list[str], list[str]]:
    """按 segment id 确定性切分：int(id) % 10 == 0 → 验证。"""
    val = [m["id"] for m in man if int(m["id"]) % 10 == 0]
    trn = [m["id"] for m in man if int(m["id"]) % 10 != 0]
    return trn, val


class SynthVocalDataset(Dataset):
    """synth 段 → (mel, onset, offset, pitch, vib)。

    use_mix=True：clean + mix 双变体都进训练（每段两条，GT 同一份）。
    v2 生成脚本已做评测隔离（mir1k 序列池 + 伴奏池均剔除评测 40 曲，
    见 train/gen_vibrato_synth_v1.py「评测隔离（v2, A3）」），mix 可安全使用。
    """

    def __init__(self, data_dir: str | Path, ids: list[str],
                 use_mix: bool = False, mel_device: str = "cpu"):
        self.root = Path(data_dir)
        self.use_mix = use_mix
        self.mel_device = mel_device
        by_id = {m["id"]: m for m in load_manifest(data_dir)}
        self.items: list[dict] = []
        for i in ids:
            if i not in by_id:
                continue
            self.items.append(by_id[i])
            if use_mix and (self.root / "mix" / f"{i}.wav").exists():
                self.items.append({**by_id[i], "audio": "mix"})
        if not self.items:
            raise SystemExit(f"no segments found in {data_dir} for given ids")

    def __len__(self):
        return len(self.items)

    def audio_path(self, sid: str, variant: str | None = None) -> Path:
        sub = "mix" if variant == "mix" else "clean"
        return self.root / sub / f"{sid}.wav"

    def __getitem__(self, k: int) -> dict:
        m = self.items[k]
        wav, sr = sf.read(str(self.audio_path(m["id"], m.get("audio"))),
                          dtype="float32")
        assert sr == SR, f"unexpected sr {sr}"
        # vib 标注仅合成段有（真实段 vib_valid=0 → 头 loss 掩码）
        has_vib = bool(m.get("vib_valid", 1)) and \
            (self.root / "clean" / f"{m['id']}.gt.npz").exists()
        vib = np.load(self.root / "clean" / f"{m['id']}.gt.npz")["vib_depth"] \
            if has_vib else None
        mel = wav_to_mel(wav, device=self.mel_device).cpu()
        tgt = frame_targets(m["notes"], vib, mel.shape[0],
                            vowel_onsets=m.get("vowel_onsets"))
        return {"id": m["id"], "mel": mel, "n_frames": mel.shape[0],
                "vib_valid": 1.0 if has_vib else 0.0,
                "vowel_valid": float(m.get("vowel_valid", 0)), **tgt}

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """右侧 padding 到 batch 内最长；lengths 供 pack_padded / loss 掩码。"""
        T = max(b["n_frames"] for b in batch)
        B = len(batch)
        mel = torch.zeros(B, T, N_MELS)
        onset = torch.zeros(B, T)
        offset = torch.zeros(B, T)
        pitch = torch.zeros(B, T, dtype=torch.long)
        vib = torch.zeros(B, T)
        vowel = torch.zeros(B, T)
        lengths = torch.zeros(B, dtype=torch.long)
        vib_valid = torch.zeros(B)
        vowel_valid = torch.zeros(B)
        for i, b in enumerate(batch):
            t = b["n_frames"]
            mel[i, :t] = b["mel"]
            onset[i, :t] = torch.as_tensor(b["onset"])
            offset[i, :t] = torch.as_tensor(b["offset"])
            pitch[i, :t] = torch.as_tensor(b["pitch"])
            vib[i, :t] = torch.as_tensor(b["vib"])
            vowel[i, :t] = torch.as_tensor(b["vowel"])
            lengths[i] = t
            vib_valid[i] = b.get("vib_valid", 0.0)
            vowel_valid[i] = b.get("vowel_valid", 0.0)
        mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1))  # (B,T)
        return {"ids": [b["id"] for b in batch], "mel": mel, "lengths": lengths,
                "onset": onset, "offset": offset, "pitch": pitch,
                "vib": vib, "vowel": vowel, "mask": mask,
                "vib_valid": vib_valid, "vowel_valid": vowel_valid}
