"""音频节拍跟踪（时值 v3.2）：rubato 曲目格点映射的素材来源。

纯 onset 格点拟合在演奏型录音上无解（canon 实测：George Winston 的
rubato 在 25s 窗内就把 1/12 格相位抹成均匀分布，mean circ dist 0.23
vs 均匀期望 0.25）——必须回到音频信号提拍。librosa.beat_track 的拍间
距 std ~16ms（canon 实测），足以承载分段线性时间→QL 映射。

产物：beat_times（拍时刻序列）+ pulse_sec（拍间距中位数）。脉冲可能
是四分/八分/半音符级——记谱层按 bpm 换算 pulse_ql 并吸附到
{1/4, 1/3, 1/2, 1, 2} QL。
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_beats(audio_path: str) -> dict:
    """音频 → {beat_times, pulse_sec, tempo_pulse}；失败/过短返回 {}。"""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        tempo, beats = librosa.beat.beat_track(
            y=y, sr=sr, units="time", trim=False)
        bt = np.asarray(beats, dtype=float)
        if len(bt) < 16:
            logger.info("  [beat] 拍数 %d < 16，弃用", len(bt))
            return {}
        iv = float(np.median(np.diff(bt)))
        if iv <= 0.05:
            return {}
        return {"beat_times": [round(float(x), 4) for x in bt],
                "pulse_sec": round(iv, 4),
                "tempo_pulse": round(60.0 / iv, 2)}
    except Exception:
        logger.warning("  [beat] librosa 提拍失败（回退格点拟合）",
                       exc_info=True)
        return {}
