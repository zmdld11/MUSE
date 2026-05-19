# build_guitar_separation_dataset.py — 从 MedleyDB + MoisesDB 构建吉他分离训练数据
# 输出: separation_dataset/audio/*.wav (mix + guitar 配对) + metadata.json
import os
import sys
import json
import glob
import random
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict

SR = 22050
DURATION = 3.0
HOP = 0.5
SAMPLES_PER_WINDOW = int(SR * DURATION)  # 66150
HOP_SAMPLES = int(SR * HOP)              # 11025
RMS_THRESHOLD_DB = -30

MEDLEYDB_DIR = r"D:\program_project\MUSE\data\MedleyDB\MedleyDB"
MEDLEYDB_META_DIR = r"D:\program_project\MUSE\data\MedleyDB\Metadata"
MOISESDB_DIR = r"D:\program_project\MUSE\data\moisesdb_v0.1"

OUTPUT_DIR = r"D:\program_project\MUSE\data\separation_dataset"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

# 吉他相关关键词 —— 用于匹配 MedleyDB YAML instrument 字段和 MoisesDB trackType 字段
GUITAR_YAML_KEYWORDS = ["acoustic guitar", "electric guitar", "clean electric guitar",
                        "distorted electric guitar"]
GUITAR_MOISES_KEYWORDS = ["acoustic_guitar", "clean_electric_guitar",
                          "distorted_electric_guitar"]


def get_rms(audio):
    return np.sqrt(np.mean(audio ** 2) + 1e-12)


def load_audio(path, target_sr=SR):
    """加载音频到指定采样率，转为 mono float32"""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def process_medleydb():
    """扫描 MedleyDB，返回每首歌的 (mix_audio, guitar_audio, song_name)"""
    results = []
    song_dirs = sorted(os.listdir(MEDLEYDB_DIR))
    print(f"扫描 MedleyDB: {len(song_dirs)} 首歌曲")

    for song_name in tqdm(song_dirs, desc="MedleyDB"):
        song_dir = os.path.join(MEDLEYDB_DIR, song_name)
        if not os.path.isdir(song_dir):
            continue

        # 加载 _MIX.wav
        mix_path = os.path.join(song_dir, f"{song_name}_MIX.wav")
        if not os.path.exists(mix_path):
            # 备选: 直接在目录下找 .wav
            wavs = glob.glob(os.path.join(song_dir, "*.wav"))
            if wavs:
                mix_path = wavs[0]
            else:
                continue

        # 加载 YAML 元数据
        yaml_path = os.path.join(MEDLEYDB_META_DIR, f"{song_name}_METADATA.yaml")
        if not os.path.exists(yaml_path):
            continue

        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)

        stems = metadata.get('stems', {})
        stem_dir = os.path.join(song_dir, metadata.get('stem_dir', f"{song_name}_STEMS"))

        # 查找吉他 stem 文件
        guitar_wavs = []
        for stem_id, stem_info in stems.items():
            instr = stem_info.get('instrument', '').lower()
            if any(kw in instr for kw in GUITAR_YAML_KEYWORDS):
                stem_file = stem_info.get('filename', '')
                stem_path = os.path.join(stem_dir, stem_file)
                if os.path.exists(stem_path):
                    guitar_wavs.append(stem_path)

        if not guitar_wavs:
            continue  # 这首歌没有吉他

        # 加载 mix
        try:
            mix_audio = load_audio(mix_path)
        except Exception as e:
            print(f"  警告: 无法加载 {mix_path}: {e}")
            continue

        # 加载并求和所有吉他 stem
        guitar_audio = None
        for gpath in guitar_wavs:
            try:
                g = load_audio(gpath)
                # 对齐长度到 mix
                if len(g) < len(mix_audio):
                    g = np.pad(g, (0, len(mix_audio) - len(g)))
                else:
                    g = g[:len(mix_audio)]
                if guitar_audio is None:
                    guitar_audio = g
                else:
                    guitar_audio += g
            except Exception as e:
                print(f"  警告: 无法加载 {gpath}: {e}")

        if guitar_audio is None:
            continue

        results.append((mix_audio, guitar_audio, f"medleydb_{song_name}"))

    print(f"  MedleyDB 有效歌曲: {len(results)}")
    return results


def process_moisesdb():
    """扫描 MoisesDB，返回每首歌的 (mix_audio, guitar_audio, song_name)"""
    results = []
    song_dirs = sorted(os.listdir(MOISESDB_DIR))
    print(f"扫描 MoisesDB: {len(song_dirs)} 首歌曲")

    for song_id in tqdm(song_dirs, desc="MoisesDB"):
        song_dir = os.path.join(MOISESDB_DIR, song_id)
        if not os.path.isdir(song_dir):
            continue

        json_path = os.path.join(song_dir, "data.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        stems = data.get('stems', [])

        # 收集所有 stem 的 wav 文件路径，并标记哪些是吉他
        all_wavs = []       # 用于构建总混音
        guitar_wavs = []     # 仅吉他目标
        guitar_track_ids = set()

        # 先标记吉他 track id
        for stem in stems:
            for track in stem.get('tracks', []):
                ttype = track.get('trackType', '')
                if any(kw in ttype for kw in GUITAR_MOISES_KEYWORDS):
                    guitar_track_ids.add(track.get('id', ''))

        # 收集所有 stem 的 wav
        for stem in stems:
            stem_name = stem.get('stemName', '')
            stem_subdir = os.path.join(song_dir, stem_name)
            for track in stem.get('tracks', []):
                tid = track.get('id', '')
                wav_path = os.path.join(stem_subdir, f"{tid}.wav")
                if os.path.exists(wav_path):
                    all_wavs.append(wav_path)
                    if tid in guitar_track_ids:
                        guitar_wavs.append(wav_path)

        if not guitar_wavs:
            continue  # 这首歌没有吉他

        # 构建总混音：求和所有分轨
        mix_audio = None
        for wpath in all_wavs:
            try:
                a = load_audio(wpath)
                if mix_audio is None:
                    mix_audio = a
                else:
                    if len(a) < len(mix_audio):
                        a = np.pad(a, (0, len(mix_audio) - len(a)))
                    else:
                        a = a[:len(mix_audio)]
                    mix_audio += a
            except Exception:
                continue

        if mix_audio is None:
            continue

        # 构建吉他目标：求和吉他分轨
        guitar_audio = None
        for gpath in guitar_wavs:
            try:
                g = load_audio(gpath)
                if len(g) < len(mix_audio):
                    g = np.pad(g, (0, len(mix_audio) - len(g)))
                else:
                    g = g[:len(mix_audio)]
                if guitar_audio is None:
                    guitar_audio = g
                else:
                    guitar_audio += g
            except Exception:
                continue

        if guitar_audio is None:
            continue

        results.append((mix_audio, guitar_audio, f"moisesdb_{song_id}"))

    print(f"  MoisesDB 有效歌曲: {len(results)}")
    return results


def extract_windows(song_data_list):
    """对所有歌曲做滑窗切片，只保留吉他活跃的窗口"""
    all_mix_segments = []
    all_guitar_segments = []
    song_window_counts = []

    for mix, guitar, song_name in tqdm(song_data_list, desc="滑窗切片"):
        total_len = min(len(mix), len(guitar))
        n_windows = max(0, (total_len - SAMPLES_PER_WINDOW) // HOP_SAMPLES + 1)
        if n_windows == 0:
            continue

        # 计算整首歌的吉他活跃阈值
        guitar_rms_global = get_rms(guitar)
        if guitar_rms_global < 1e-6:
            continue

        count = 0
        for w in range(n_windows):
            start = w * HOP_SAMPLES
            end = start + SAMPLES_PER_WINDOW
            g_seg = guitar[start:end]
            g_rms = get_rms(g_seg)
            # 吉他 RMS 相对于全局 RMS 不低于 -30dB
            if g_rms / guitar_rms_global >= 10 ** (RMS_THRESHOLD_DB / 20):
                m_seg = mix[start:end]
                all_mix_segments.append(m_seg)
                all_guitar_segments.append(g_seg)
                count += 1

        if count > 0:
            song_window_counts.append((song_name, count))

    return all_mix_segments, all_guitar_segments, song_window_counts


def train_val_split(song_window_counts, val_ratio=0.2):
    """歌曲级 train/val 切分"""
    random.seed(42)
    random.shuffle(song_window_counts)
    total_windows = sum(c for _, c in song_window_counts)
    val_target = int(total_windows * val_ratio)

    # 贪心分配验证集歌曲，逼近目标比例
    current_val = 0
    val_songs = set()
    for song_name, count in song_window_counts:
        if current_val + count <= val_target + count // 2:
            val_songs.add(song_name)
            current_val += count
        if current_val >= val_target:
            break

    train_songs = set()
    for song_name, _ in song_window_counts:
        if song_name not in val_songs:
            train_songs.add(song_name)

    return train_songs, val_songs


def main():
    parser = argparse.ArgumentParser(description="构建吉他分离数据集")
    parser.add_argument("--medleydb-only", action="store_true")
    parser.add_argument("--moisesdb-only", action="store_true")
    parser.add_argument("--max-train", type=int, default=15000)
    args = parser.parse_args()

    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Step 1: 加载所有音频
    all_data = []
    if not args.moisesdb_only:
        all_data.extend(process_medleydb())
    if not args.medleydb_only:
        all_data.extend(process_moisesdb())

    print(f"\n总有效歌曲: {len(all_data)}")

    if len(all_data) == 0:
        print("错误: 没有找到任何含有吉他的歌曲!")
        return

    # Step 2: 滑窗切片
    mix_segments, guitar_segments, song_window_counts = extract_windows(all_data)
    print(f"总活跃窗口: {len(mix_segments)}")

    if len(mix_segments) == 0:
        print("错误: 没有找到任何吉他活跃窗口!")
        return

    # Step 3: 歌曲级 train/val 切分
    train_songs, val_songs = train_val_split(song_window_counts)
    print(f"训练歌曲: {len(train_songs)}, 验证歌曲: {len(val_songs)}")

    # Step 4: 写入音频文件 + 构建 metadata
    train_samples = []
    val_samples = []
    idx = 0
    train_count = 0
    val_count = 0

    for mix, guitar, song_name in tqdm(all_data, desc="写入文件"):
        total_len = min(len(mix), len(guitar))
        n_windows = max(0, (total_len - SAMPLES_PER_WINDOW) // HOP_SAMPLES + 1)
        guitar_rms_global = get_rms(guitar)
        if guitar_rms_global < 1e-6 or n_windows == 0:
            continue

        for w in range(n_windows):
            start = w * HOP_SAMPLES
            end = start + SAMPLES_PER_WINDOW
            g_seg = guitar[start:end]
            g_rms = get_rms(g_seg)

            if g_rms / guitar_rms_global < 10 ** (RMS_THRESHOLD_DB / 20):
                continue

            m_seg = mix[start:end]

            is_val = song_name in val_songs

            # 训练集下采样
            if not is_val and args.max_train > 0 and train_count >= args.max_train:
                continue

            mix_filename = f"real_{idx:06d}_mix.wav"
            gtr_filename = f"real_{idx:06d}_guitar.wav"

            sf.write(os.path.join(AUDIO_DIR, mix_filename), m_seg, SR, subtype='PCM_16')
            sf.write(os.path.join(AUDIO_DIR, gtr_filename), g_seg, SR, subtype='PCM_16')

            sample = {"mix": mix_filename, "guitar": gtr_filename, "song": song_name}
            if is_val:
                val_samples.append(sample)
                val_count += 1
            else:
                train_samples.append(sample)
                train_count += 1

            idx += 1

    # Step 5: 写入 metadata.json
    metadata = {
        "dataset_type": "guitar_separation",
        "sr": SR,
        "duration": DURATION,
        "hop": HOP,
        "target_instrument": "guitar",
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n写入完成!")
    print(f"  训练窗口: {len(train_samples)}")
    print(f"  验证窗口: {len(val_samples)}")
    print(f"  输出目录: {AUDIO_DIR}")
    print(f"  元数据:   {metadata_path}")


if __name__ == "__main__":
    main()
