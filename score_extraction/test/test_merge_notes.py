"""Unit tests for note merging (A1 碎音合并, 2026-08-02).

真实钢琴 frame probs 抖动 → 一个真实音符被 BP 后处理切成多个同音高碎音
(平均 4.7 个/GT 音符). merge_similar_notes 把同音高、相邻 (gap ≤ tol) 的
碎音合并成一个完整音符.

Usage:
    python test/test_merge_notes.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMergeSimilarNotes(unittest.TestCase):
    """Unit tests for src/merge_notes.merge_similar_notes."""

    def _cand(self, onset, offset, pitch, conf=10):
        return {"onset_frame": onset, "offset_frame": offset,
                "pitch": pitch, "pitch_bin": pitch - 21, "confidence": conf}

    def test_merge_adjacent_same_pitch(self):
        """同音高相邻碎音 (gap ≤ tol) → 合并成一个, onset/offset 取两端."""
        from src.merge_notes import merge_similar_notes
        cands = [self._cand(10, 20, 60), self._cand(22, 30, 60)]  # gap=2
        out = merge_similar_notes(cands, gap_tol_frames=4)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["onset_frame"], 10)
        self.assertEqual(out[0]["offset_frame"], 30)
        self.assertEqual(out[0]["pitch"], 60)

    def test_no_merge_wide_gap(self):
        """gap > tol → 不合并 (两个独立音符)."""
        from src.merge_notes import merge_similar_notes
        cands = [self._cand(10, 20, 60), self._cand(30, 40, 60)]  # gap=10 > 4
        out = merge_similar_notes(cands, gap_tol_frames=4)
        self.assertEqual(len(out), 2)

    def test_no_merge_diff_pitch(self):
        """不同音高 → 不合并."""
        from src.merge_notes import merge_similar_notes
        cands = [self._cand(10, 20, 60), self._cand(22, 30, 61)]
        out = merge_similar_notes(cands, gap_tol_frames=4)
        self.assertEqual(len(out), 2)

    def test_merge_overlapping(self):
        """重叠 (后 onset < 前 offset) → 合并, offset 取最大."""
        from src.merge_notes import merge_similar_notes
        cands = [self._cand(10, 30, 60), self._cand(25, 35, 60)]
        out = merge_similar_notes(cands, gap_tol_frames=4)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["onset_frame"], 10)
        self.assertEqual(out[0]["offset_frame"], 35)

    def test_merge_chain(self):
        """3 个碎音链 (每个 gap ≤ tol) → 全部合并成一个."""
        from src.merge_notes import merge_similar_notes
        cands = [self._cand(10, 15, 60), self._cand(16, 22, 60),
                 self._cand(24, 30, 60)]
        out = merge_similar_notes(cands, gap_tol_frames=4)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["onset_frame"], 10)
        self.assertEqual(out[0]["offset_frame"], 30)

    def test_merge_interleaved_pitches(self):
        """多音高交错出现 → 各自独立合并, 不串音高."""
        from src.merge_notes import merge_similar_notes
        cands = [
            self._cand(10, 15, 60), self._cand(12, 18, 64),
            self._cand(16, 22, 60), self._cand(20, 26, 64),
        ]
        out = merge_similar_notes(cands, gap_tol_frames=4)
        self.assertEqual(len(out), 2)
        by_pitch = {c["pitch"]: c for c in out}
        self.assertEqual(by_pitch[60]["onset_frame"], 10)
        self.assertEqual(by_pitch[60]["offset_frame"], 22)
        self.assertEqual(by_pitch[64]["onset_frame"], 12)
        self.assertEqual(by_pitch[64]["offset_frame"], 26)

    def test_unsorted_input(self):
        """输入乱序 → 结果与排序输入一致."""
        from src.merge_notes import merge_similar_notes
        cands = [self._cand(22, 30, 60), self._cand(10, 20, 60)]
        out = merge_similar_notes(cands, gap_tol_frames=4)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["onset_frame"], 10)
        self.assertEqual(out[0]["offset_frame"], 30)

    def test_empty(self):
        """空列表 → 空列表."""
        from src.merge_notes import merge_similar_notes
        self.assertEqual(merge_similar_notes([], 4), [])


if __name__ == "__main__":
    unittest.main()
