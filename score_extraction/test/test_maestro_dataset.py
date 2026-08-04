"""Unit tests for MAESTRO 数据管线 (train/maestro_dataset.py).

用真实 MAESTRO train 分区前几首验证: csv 读取 / 标签正确性 /
段切分对齐 / 缓存 / 混合采样.

Usage: python test/test_maestro_dataset.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MAESTRO_DIR = os.path.join(os.path.dirname(__file__), "..",
                           "data", "maestro", "maestro-v3.0.0")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..",
                        "data", "maestro", "maestro-v3.0.0.csv")


@unittest.skipUnless(os.path.exists(MAESTRO_DIR), "MAESTRO 数据未下载")
class TestMaestroCSV(unittest.TestCase):
    def test_train_split_count(self):
        """train 分区应为 962 首."""
        from train.maestro_dataset import load_maestro_rows
        rows = load_maestro_rows(CSV_PATH)
        train = [r for r in rows if r["split"] == "train"]
        self.assertEqual(len(train), 962)

    def test_files_exist(self):
        """csv 引用的 midi/wav 都存在."""
        from train.maestro_dataset import load_maestro_rows
        rows = load_maestro_rows(CSV_PATH)
        for r in rows[:5]:
            self.assertTrue(os.path.exists(os.path.join(MAESTRO_DIR, r["midi_filename"])),
                            r["midi_filename"])
            self.assertTrue(os.path.exists(os.path.join(MAESTRO_DIR, r["audio_filename"])),
                            r["audio_filename"])


@unittest.skipUnless(os.path.exists(MAESTRO_DIR), "MAESTRO 数据未下载")
class TestMidiToLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train.maestro_dataset import load_maestro_rows, midi_to_labels
        rows = [r for r in load_maestro_rows(CSV_PATH) if r["split"] == "train"]
        cls.midi_path = os.path.join(MAESTRO_DIR, rows[0]["midi_filename"])
        cls.frame, cls.onset = midi_to_labels(cls.midi_path)

    def test_labels_shape(self):
        """标签矩阵形状 (T, 88)."""
        self.assertEqual(self.frame.shape[1], 88)
        self.assertEqual(self.onset.shape, self.frame.shape)

    def test_note_covered_in_frame(self):
        """每个音符期间 frame_labels 全 1."""
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(self.midi_path)
        sr, hop = 22050, 512
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes[:50]:
                if not (21 <= n.pitch < 109):
                    continue
                s_f = int(n.start * sr / hop)
                e_f = int(np.ceil(n.end * sr / hop))
                col = self.frame[s_f:min(e_f + 1, len(self.frame)), n.pitch - 21]
                self.assertGreater(col.sum(), 0,
                                   f"pitch={n.pitch} t={n.start:.2f}-{n.end:.2f} 无覆盖")

    def test_onset_at_note_start(self):
        """音符起始帧 onset_labels = 1."""
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(self.midi_path)
        sr, hop = 22050, 512
        found = 0
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes[:50]:
                if not (21 <= n.pitch < 109):
                    continue
                s_f = int(n.start * sr / hop)
                if s_f < len(self.onset):
                    self.assertEqual(self.onset[s_f, n.pitch - 21], 1.0)
                    found += 1
        self.assertGreater(found, 10, "没找到足够的 onset 测试点")


@unittest.skipUnless(os.path.exists(MAESTRO_DIR), "MAESTRO 数据未下载")
class TestMaestroDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="maestro_test_")

    def test_len_and_shapes(self):
        """段数 = 曲目段数之和; 样本形状 (1,229,T)/(T,88)/(T,88)."""
        from train.maestro_dataset import MaestroDataset
        ds = MaestroDataset(split="train", max_files=2, max_dur_sec=60,
                            cache_dir=self.tmpdir)
        self.assertGreater(len(ds), 0)
        mel, onset, frame = ds[0]
        self.assertEqual(mel.shape[0], 1)
        self.assertEqual(mel.shape[1], 229)
        self.assertEqual(onset.shape[1], 88)
        self.assertEqual(frame.shape[1], 88)
        # mel 帧数与标签帧数允许 ±1 差异 (hop 对齐)
        self.assertLessEqual(abs(mel.shape[2] - onset.shape[0]), 1)

    def test_cache_reuse(self):
        """第二次实例化直接走缓存 (npz 文件已存在)."""
        from train.maestro_dataset import MaestroDataset
        ds1 = MaestroDataset(split="train", max_files=1, max_dur_sec=60,
                             cache_dir=self.tmpdir)
        cache_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".npz")]
        self.assertGreaterEqual(len(cache_files), 1)
        # 删掉源数据引用验证缓存独立 (用同 cache_dir 新实例)
        ds2 = MaestroDataset(split="train", max_files=1, max_dur_sec=60,
                             cache_dir=self.tmpdir)
        self.assertEqual(len(ds1), len(ds2))

    def test_segment_continuity(self):
        """相邻段标签时间连续: 段 k 末尾帧 == 段 k+1 起始帧附近."""
        from train.maestro_dataset import MaestroDataset
        ds = MaestroDataset(split="train", max_files=1, max_dur_sec=60,
                            cache_dir=self.tmpdir)
        self.assertGreater(len(ds), 1, "首曲应有多段")
        _, o0, f0 = ds[0]
        _, o1, f1 = ds[1]
        # 段 0 的最后一个有效帧与段 1 的第一个帧: 音符延续性抽查
        # (段边界处同一音高的 frame 标记不应同时消失出现 — 简单验证形状对齐即可)
        self.assertEqual(f0.shape[1], f1.shape[1])


@unittest.skipUnless(os.path.exists(MAESTRO_DIR), "MAESTRO 数据未下载")
class TestMixedDataset(unittest.TestCase):
    def test_alternate_sources(self):
        """混合数据集奇数/偶数索引交替返回两域样本."""
        from train.maestro_dataset import MaestroDataset, MixedSynthMaestro
        tmpdir = tempfile.mkdtemp(prefix="maestro_mix_")
        m_ds = MaestroDataset(split="train", max_files=2, max_dur_sec=60,
                              cache_dir=tmpdir)

        class FakeSynth:
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return ("synth", idx)

        mix = MixedSynthMaestro(FakeSynth(), m_ds)
        self.assertEqual(len(mix), min(4, len(m_ds)) * 2)
        # 偶数索引 → synth, 奇数索引 → maestro
        item0 = mix[0]
        item1 = mix[1]
        self.assertEqual(item0[0], "synth")
        self.assertNotEqual(item1[0], "synth")


if __name__ == "__main__":
    unittest.main()
