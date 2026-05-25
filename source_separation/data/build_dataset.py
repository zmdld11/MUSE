# build_dataset.py — 吉他分离数据集构建 (MedleyDB + MoisesDB)
# 输出: data/audio/*.wav + data/metadata.json
import os, json, glob, random, argparse, yaml
import numpy as np
import soundfile as sf
from tqdm import tqdm

SR = 22050
DURATION = 3.0
HOP = 0.5
WINDOW_LEN = int(SR * DURATION)
HOP_LEN = int(SR * HOP)
RMS_THRESHOLD_DB = -30

MEDLEYDB_DIR = r"D:\program_project\MUSE\data\MedleyDB\MedleyDB"
MEDLEYDB_META = r"D:\program_project\MUSE\data\MedleyDB\Metadata"
MOISESDB_DIR = r"D:\program_project\MUSE\data\moisesdb_v0.1"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GUITAR_YAML = ["acoustic guitar", "electric guitar", "clean electric guitar", "distorted electric guitar"]
GUITAR_MOISES = ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"]


def rms(a):
    return np.sqrt(np.mean(a ** 2) + 1e-12)


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1: audio = np.mean(audio, axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=25000)
    parser.add_argument("--guitar-ratio", type=float, default=0.85)
    parser.add_argument("--skip-moisesdb", action="store_true")
    args = parser.parse_args()

    all_data = []  # [(mix_audio, gtr_audio, song_name)]

    # === MedleyDB ===
    print("MedleyDB...")
    for song in tqdm(sorted(os.listdir(MEDLEYDB_DIR))):
        sdir = os.path.join(MEDLEYDB_DIR, song)
        if not os.path.isdir(sdir): continue
        mix_p = os.path.join(sdir, f"{song}_MIX.wav")
        if not os.path.exists(mix_p): continue
        yaml_p = os.path.join(MEDLEYDB_META, f"{song}_METADATA.yaml")
        if not os.path.exists(yaml_p): continue
        with open(yaml_p, encoding='utf-8') as f:
            meta = yaml.safe_load(f)
        stems = meta.get('stems', {})
        stem_dir = os.path.join(sdir, meta.get('stem_dir', f"{song}_STEMS"))
        gtr_wavs = []
        for sid, si in stems.items():
            if any(k in si.get('instrument', '').lower() for k in GUITAR_YAML):
                sp = os.path.join(stem_dir, si.get('filename', ''))
                if os.path.exists(sp): gtr_wavs.append(sp)
        if not gtr_wavs: continue
        try:
            mix = load_audio(mix_p)
            gtr = np.zeros(len(mix), dtype=np.float32)
            for gw in gtr_wavs:
                g = load_audio(gw)
                if len(g) < len(mix): g = np.pad(g, (0, len(mix) - len(g)))
                else: g = g[:len(mix)]
                gtr += g
            all_data.append((mix, gtr, f"med_{song}"))
        except Exception as e:
            print(f"  err {song}: {e}")
    print(f"  MedleyDB 有效: {len(all_data)}")

    # === MoisesDB ===
    if not args.skip_moisesdb:
        print("MoisesDB (逐首构建，较慢)...")
        for sid in tqdm(sorted(os.listdir(MOISESDB_DIR))):
            sdir = os.path.join(MOISESDB_DIR, sid)
            if not os.path.isdir(sdir): continue
            jp = os.path.join(sdir, "data.json")
            if not os.path.exists(jp): continue
            with open(jp, encoding='utf-8') as f:
                data = json.load(f)
            gids = set()
            for stem in data.get('stems', []):
                for tr in stem.get('tracks', []):
                    if any(k in tr.get('trackType', '') for k in GUITAR_MOISES):
                        gids.add(tr.get('id', ''))
            if not gids: continue
            all_wavs, gtr_wavs = [], []
            for stem in data.get('stems', []):
                sd = os.path.join(sdir, stem.get('stemName', ''))
                for tr in stem.get('tracks', []):
                    tid = tr.get('id', '')
                    wp = os.path.join(sd, f"{tid}.wav")
                    if os.path.exists(wp):
                        all_wavs.append(wp)
                        if tid in gids: gtr_wavs.append(wp)
            if not all_wavs or not gtr_wavs: continue
            try:
                mix = None
                for wp in all_wavs:
                    a = load_audio(wp)
                    if mix is None: mix = a
                    else:
                        if len(a) < len(mix): a = np.pad(a, (0, len(mix) - len(a)))
                        else: a = a[:len(mix)]
                        mix += a
                if mix is None: continue
                gtr = np.zeros(len(mix), dtype=np.float32)
                for wp in gtr_wavs:
                    g = load_audio(wp)
                    if len(g) < len(mix): g = np.pad(g, (0, len(mix) - len(g)))
                    else: g = g[:len(mix)]
                    gtr += g
                all_data.append((mix, gtr, f"moi_{sid[:8]}"))
            except Exception as e:
                print(f"  err {sid[:8]}: {e}")
        print(f"  总歌曲: {len(all_data)}")

    if not all_data:
        print("无有效歌曲!"); return

    # 滑窗提取
    print("提取窗口...")
    pos = []  # 有吉他
    neg = []  # 无吉他
    for mix, gtr, name in tqdm(all_data):
        total = min(len(mix), len(gtr))
        n_win = max(0, (total - WINDOW_LEN) // HOP_LEN + 1)
        thr = rms(gtr) * (10 ** (RMS_THRESHOLD_DB / 20))
        for w in range(n_win):
            s, e = w * HOP_LEN, w * HOP_LEN + WINDOW_LEN
            g = gtr[s:e]
            if rms(g) >= thr:
                pos.append((mix[s:e], g))
            else:
                neg.append((mix[s:e], np.zeros(WINDOW_LEN, dtype=np.float32)))

    print(f"  有吉他: {len(pos)}, 无吉他: {len(neg)}")
    if not pos:
        print("无正样本!"); return

    # 采样
    random.seed(42)
    target_pos = int(args.max_train * args.guitar_ratio)
    target_neg = args.max_train - target_pos
    sel_pos = random.sample(pos, min(len(pos), target_pos))
    sel_neg = random.sample(neg, min(len(neg), target_neg)) if neg else []
    print(f"  采样: {len(sel_pos)} 正 + {len(sel_neg)} 负 = {len(sel_pos)+len(sel_neg)}")

    # 写入 (前 n 个为 train, 剩余为 val)
    split = min(len(sel_pos), int(len(sel_pos) * 0.85))
    train = sel_pos[:split] + sel_neg[:len(sel_neg)//2]
    val = sel_pos[split:] + sel_neg[len(sel_neg)//2:]
    random.shuffle(train)

    print("写入 WAV...")
    idx = 0
    for tag, windows in [("train", train), ("val", val)]:
        for m, g in tqdm(windows, desc=tag):
            sf.write(os.path.join(OUTPUT_DIR, f"{idx:06d}_mix.wav"), m, SR, subtype='PCM_16')
            sf.write(os.path.join(OUTPUT_DIR, f"{idx:06d}_gtr.wav"), g, SR, subtype='PCM_16')
            idx += 1

    meta = {
        "dataset_type": "guitar_separation_v2",
        "sr": SR, "duration": DURATION, "hop": HOP,
        "num_train": len(train), "num_val": len(val),
        "train_songs": [s[-1] for s in all_data],
    }
    with open(os.path.join(os.path.dirname(__file__), "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n完成! 输出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
