"""MAESTRO 数据管线压力测试 (2026-08-03).

量化: 标签生成耗时 / mel 生成耗时 / 缓存大小 / 训练吞吐外推,
找到数据量与训练时间的平衡点. 用 train 分区前 N 首实测, 再外推.

用法: python train/bench_maestro.py [--n 10] [--dur 60]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAESTRO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "maestro", "maestro-v3.0.0")
SR = 22050
HOP = 512
MIDI_OFFSET = 21
N_MELS = 229


def midi_to_labels(midi_path, sr=SR, hop=HOP):
    """MIDI → (frame_labels, onset_labels), 完整曲目不截断."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.pitch < MIDI_OFFSET or n.pitch >= MIDI_OFFSET + 88:
                continue
            notes.append((n.start, n.end, n.pitch))
    end_time = pm.get_end_time()
    T = int(np.ceil(end_time * sr / hop))
    frame_labels = np.zeros((T, 88), dtype=np.float32)
    onset_labels = np.zeros((T, 88), dtype=np.float32)
    for s, e, p in notes:
        s_f, e_f = int(s * sr / hop), int(np.ceil(e * sr / hop))
        frame_labels[max(0, s_f):min(T, e_f + 1), p - MIDI_OFFSET] = 1.0
        if s_f < T:
            onset_labels[s_f, p - MIDI_OFFSET] = 1.0
    return frame_labels, onset_labels


def bench(n_files, dur_sec):
    import librosa
    rows = list(csv.DictReader(
        open(os.path.join(os.path.dirname(MAESTRO_DIR), "maestro-v3.0.0.csv"),
             encoding="utf-8")))
    train_rows = [r for r in rows if r["split"] == "train"][:n_files]

    t_label = 0.0
    t_wav = 0.0
    t_mel = 0.0
    n_segments = 0
    total_audio_sec = 0.0
    mel_sizes = []

    for i, r in enumerate(train_rows):
        midi_path = os.path.join(MAESTRO_DIR, r["midi_filename"])
        wav_path = os.path.join(MAESTRO_DIR, r["audio_filename"])

        # 1) 标签生成
        t0 = time.perf_counter()
        frame_l, onset_l = midi_to_labels(midi_path)
        t_label += time.perf_counter() - t0

        # 2) wav 加载
        t0 = time.perf_counter()
        audio, sr = librosa.load(wav_path, sr=SR, mono=True)
        t_wav += time.perf_counter() - t0

        # 3) mel 生成 (60s 段)
        n_seg = int(np.ceil(len(audio) / (dur_sec * sr)))
        t0 = time.perf_counter()
        for s in range(n_seg):
            seg = audio[s * dur_sec * sr:(s + 1) * dur_sec * sr]
            if len(seg) < 0.5 * dur_sec * sr:
                break
            mel = librosa.feature.melspectrogram(
                y=seg, sr=SR, n_mels=N_MELS, hop_length=HOP, fmin=30, fmax=8000)
            mel_sizes.append(mel.shape)
            n_segments += 1
        t_mel += time.perf_counter() - t0
        total_audio_sec += len(audio) / sr
        assert len(mel_sizes) == n_segments

        if (i + 1) % 2 == 0:
            print(f"  [{i+1}/{n_files}] 标签{t_label:.1f}s 音频{t_wav:.1f}s "
                  f"mel{t_mel:.1f}s 段数{n_segments}")

    per_file_label = t_label / n_files
    per_file_wav = t_wav / n_files
    per_seg_mel = t_mel / max(n_segments, 1)
    avg_dur = total_audio_sec / n_files
    avg_seg = n_segments / n_files

    print(f"\n=== 实测 ({n_files} 首, {dur_sec}s 段) ===")
    print(f"平均曲长: {avg_dur:.0f}s → {avg_seg:.1f} 段/首")
    print(f"标签生成: {per_file_label*1000:.0f} ms/首")
    print(f"wav 加载: {per_file_wav*1000:.0f} ms/首")
    print(f"mel 生成: {per_seg_mel*1000:.0f} ms/段")
    mel_mb = mel_sizes[0][0] * mel_sizes[0][1] * 4 / 1e6 if mel_sizes else 0
    print(f"单段缓存: mel {mel_mb:.1f}MB + labels "
          f"{mel_sizes[0][1]*88*2*4/1e6:.1f}MB ≈ {mel_mb + mel_sizes[0][1]*88*2*4/1e6:.1f}MB")

    # 外推 962 首
    print(f"\n=== 外推 train 全量 962 首 ===")
    total_seg = 962 * avg_seg
    print(f"总段数: ~{total_seg:.0f}")
    print(f"标签生成总耗时: {per_file_label*962/60:.1f} 分钟")
    print(f"mel 生成总耗时: {per_seg_mel*total_seg/60:.1f} 分钟")
    cache_gb = (mel_mb + mel_sizes[0][1] * 88 * 2 * 4 / 1e6) * total_seg / 1024
    print(f"缓存大小: ~{cache_gb:.1f} GB")
    # 训练吞吐: 3.5 step/s × 8 batch (60s 段)
    steps_per_epoch = total_seg / 8
    print(f"\n=== 训练时间 (3.5 step/s, batch=8, {dur_sec}s 段) ===")
    print(f"{steps_per_epoch:.0f} 步/epoch → {steps_per_epoch/3.5/60:.1f} 分钟/epoch")
    for ep in [10, 30, 50]:
        print(f"  {ep} epochs: {steps_per_epoch/3.5/60*ep:.1f} 分钟 "
              f"({steps_per_epoch/3.5/3600*ep:.1f} 小时)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--dur", type=int, default=60)
    args = ap.parse_args()
    bench(args.n, args.dur)
