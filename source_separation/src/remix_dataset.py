# remix_dataset.py — VER5.0 在线随机混音数据集
# 跨歌曲动态混合：吉他(歌曲A) + 其他乐器(歌曲B) → 模型被迫学习真正分离
import os, json, random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

GUITAR_MEDLEY = ["acoustic guitar", "electric guitar", "clean electric guitar", "distorted electric guitar"]
GUITAR_MOISES = ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"]


def _load_crop(path, start, n_samples, sr):
    audio, _ = sf.read(path, start=start, stop=start + n_samples, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    return audio


class RemixDataset(Dataset):
    """在线随机混音：每样本动态混合不同歌曲的吉他+其他乐器"""

    def __init__(self, medleydb_dir, medleydb_meta_dir, moisesdb_dir,
                 num_samples=65536, sr=22050, num_total=25000,
                 cache_path=None):
        self.num_samples = num_samples
        self.sr = sr
        self.num_total = num_total

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.guitar_stems = data['guitar']
            self.other_stems = data['other']
        else:
            stems = self._scan_all(medleydb_dir, medleydb_meta_dir, moisesdb_dir)
            self.guitar_stems = [s for s in stems if s['is_guitar'] and s['duration'] >= num_samples]
            self.other_stems = [s for s in stems if not s['is_guitar'] and s['duration'] >= num_samples]
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'guitar': self.guitar_stems, 'other': self.other_stems}, f, indent=2)

        assert self.guitar_stems, "无有效吉他分轨！"
        assert self.other_stems, "无有效其他乐器分轨！"
        print(f"RemixDataset: {len(self.guitar_stems)} 吉他, {len(self.other_stems)} 其他乐器片段")

    # ——— 扫描 ————————————————————————————————————————————————

    def _scan_medleydb(self, med_dir, meta_dir):
        stems = []
        if not med_dir or not os.path.isdir(med_dir):
            return stems
        for song in sorted(os.listdir(med_dir)):
            sdir = os.path.join(med_dir, song)
            if not os.path.isdir(sdir):
                continue
            yp = os.path.join(meta_dir, f"{song}_METADATA.yaml")
            if not os.path.exists(yp):
                continue
            import yaml
            with open(yp, encoding='utf-8') as f:
                meta = yaml.safe_load(f)
            stem_dir = os.path.join(sdir, meta.get('stem_dir', f"{song}_STEMS"))
            for info in meta.get('stems', {}).values():
                fp = os.path.join(stem_dir, info.get('filename', ''))
                if not os.path.isfile(fp):
                    continue
                try:
                    dur = sf.info(fp).frames
                except Exception:
                    continue
                stems.append(dict(
                    path=fp, song_id=f"med_{song}", duration=dur,
                    is_guitar=any(k in info.get('instrument', '').lower() for k in GUITAR_MEDLEY),
                ))
        return stems

    def _scan_moisesdb(self, moi_dir):
        stems = []
        if not moi_dir or not os.path.isdir(moi_dir):
            return stems
        for sid in sorted(os.listdir(moi_dir)):
            sdir = os.path.join(moi_dir, sid)
            if not os.path.isdir(sdir):
                continue
            jp = os.path.join(sdir, "data.json")
            if not os.path.exists(jp):
                continue
            with open(jp, encoding='utf-8') as f:
                data = json.load(f)
            gids = set()
            for stem in data.get('stems', []):
                for tr in stem.get('tracks', []):
                    if any(k in tr.get('trackType', '') for k in GUITAR_MOISES):
                        gids.add(tr.get('id', ''))
            for stem in data.get('stems', []):
                sd = os.path.join(sdir, stem.get('stemName', ''))
                for tr in stem.get('tracks', []):
                    tid = tr.get('id', '')
                    wp = os.path.join(sd, f"{tid}.wav")
                    if not os.path.isfile(wp):
                        continue
                    try:
                        dur = sf.info(wp).frames
                    except Exception:
                        continue
                    stems.append(dict(
                        path=wp, song_id=f"moi_{sid[:8]}", duration=dur,
                        is_guitar=tid in gids,
                    ))
        return stems

    def _scan_all(self, med_dir, med_meta, moi_dir):
        stems = self._scan_medleydb(med_dir, med_meta)
        print(f"  MedleyDB: {len(stems)} stems")
        stems += self._scan_moisesdb(moi_dir)
        print(f"  总计: {len(stems)} stems")
        return stems

    # ——— 核心 ——————————————————————————————————————————————————

    def __len__(self):
        return self.num_total

    def __getitem__(self, idx):
        g = random.choice(self.guitar_stems)

        # 吉他片段
        g_start = random.randint(0, g['duration'] - self.num_samples)
        gtr = _load_crop(g['path'], g_start, self.num_samples, self.sr)

        # 其他乐器 (1-3 个分轨, 来自不同歌曲)
        n_o = random.randint(1, 3)
        candidates = [s for s in self.other_stems if s['song_id'] != g['song_id']]
        if len(candidates) < n_o:
            candidates = self.other_stems
        chosen = random.sample(candidates, min(n_o, len(candidates)))

        other = np.zeros(self.num_samples, dtype=np.float32)
        for o in chosen:
            o_start = random.randint(0, o['duration'] - self.num_samples)
            o_audio = _load_crop(o['path'], o_start, self.num_samples, self.sr)
            o_gain = 10 ** (random.uniform(-6, 0) / 20)
            other += o_gain * o_audio

        g_gain = 10 ** (random.uniform(-6, 3) / 20)
        mix = g_gain * gtr + other
        target = g_gain * gtr

        # 峰值归一化防削波
        peak = float(np.abs(mix).max())
        if peak > 0.95:
            scale = 0.95 / peak
            mix *= scale
            target *= scale

        return torch.from_numpy(mix.copy()), torch.from_numpy(target.copy())
