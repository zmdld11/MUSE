# test/test_pipeline.py
# Unit + integration tests for score_extraction pipeline.
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# --- availability guards ---
try:
    import torchcrepe  # noqa: F401
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False

try:
    import music21  # noqa: F401
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False


class TestBPM(unittest.TestCase):
    def test_120bpm(self):
        from src.bpm_detect import detect_bpm
        bpm = detect_bpm(os.path.join(FIXTURES, "test_120bpm.wav"))
        self.assertIsNotNone(bpm)
        self.assertAlmostEqual(bpm, 120, delta=3)


class TestPitch(unittest.TestCase):
    @unittest.skipIf(not HAS_CREPE, "torchcrepe not installed (skip pitch test)")
    def test_a4_detected(self):
        from src.pitch_detect import detect_pitch_mono
        notes = detect_pitch_mono(os.path.join(FIXTURES, "test_a4_440.wav"))
        # At least one note with pitch near A4 (MIDI 69)
        self.assertTrue(
            any(68 <= n["pitch"] <= 70 for n in notes),
            "A4 not detected in 440Hz sine wave",
        )


class TestKeyEstimate(unittest.TestCase):
    def test_c_major_scale(self):
        from src.key_estimate import estimate_key
        notes = []
        for p in [60, 62, 64, 65, 67, 69, 71, 72]:
            notes.append({"onset": 0, "offset": 1.0, "pitch": p})
        key = estimate_key(notes * 3)  # repeat for stronger signal
        self.assertIn("C", key)
        self.assertIn("major", key)


class TestMusicXMLRoundtrip(unittest.TestCase):
    @unittest.skipIf(not HAS_MUSIC21, "music21 not installed (skip roundtrip test)")
    def test_roundtrip(self):
        import tempfile
        from src.score_assemble import assemble_score
        from src.export_score import export_score
        import music21

        notes = [
            {"onset": 0, "offset": 1, "pitch": 60, "amplitude": 0.1},
            {"onset": 1, "offset": 2, "pitch": 64, "amplitude": 0.1},
        ]
        score = assemble_score("test", notes, 120, "C major")

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            path = f.name
        try:
            export_score(score, path.replace(".musicxml", ""))
            loaded = music21.converter.parse(path)
            self.assertGreater(len(loaded.parts), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestGuitarFingering(unittest.TestCase):
    def test_c_major_scale(self):
        from src.guitar_tab import assign_guitar_fingering

        notes = [
            {"onset": i, "offset": i + 1, "pitch": p}
            for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])
        ]
        result = assign_guitar_fingering(notes)
        self.assertEqual(len(result), 8)
        for n in result:
            self.assertIn("string", n)
            self.assertIn("fret", n)


class TestMixFixture(unittest.TestCase):
    """Integration test using the multi-instrument mix fixture."""

    def test_mix_15s_exists(self):
        path = os.path.join(FIXTURES, "test_mix_15s.wav")
        self.assertTrue(os.path.exists(path), "test_mix_15s.wav fixture missing")
        self.assertGreater(os.path.getsize(path), 0)

    def test_mix_15s_bpm(self):
        from src.bpm_detect import detect_bpm
        path = os.path.join(FIXTURES, "test_mix_15s.wav")
        bpm = detect_bpm(path)
        # BPM may be detected or None depending on content; should not crash
        self.assertIsNotNone(bpm)


if __name__ == "__main__":
    unittest.main()
