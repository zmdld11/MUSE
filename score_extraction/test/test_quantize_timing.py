"""Test timing quantization module and verify it works on the canon test file.

Usage:
    conda activate score_build
    python test/test_quantize_timing.py              # unit tests
    python test/test_quantize_timing.py --pipeline    # full pipeline integration test
"""
import argparse
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestQuantizeTiming(unittest.TestCase):
    """Unit tests for the quantize_timing module."""

    def setUp(self):
        from src.quantize_timing import _compute_grids
        self.interval_16th, self.interval_triplet, self.threshold = \
            _compute_grids(73.9)
        # At 73.9 BPM:
        # 16th interval  = 60 / (73.9 * 4)  ≈ 0.20298 s
        # Triplet interval ≈ 60 / (73.9 * 3) * 2 ≈ 0.27064 s
        # Threshold      = interval_16th / 4 ≈ 0.05074 s

    def test_grid_intervals(self):
        """Verify grid intervals at canonical BPM."""
        self.assertAlmostEqual(self.interval_16th, 0.20298, places=4)
        self.assertAlmostEqual(self.interval_triplet, 0.27064, places=4)
        # Note: 8th triplet = 60/(bpm*3), NOT 60/(bpm*3)*2.
        # The spec formula with *2 would give 0.541 which is incorrect.
        self.assertAlmostEqual(self.threshold, 0.05074, places=4)

    def test_best_grid_offset_perfect_align(self):
        """Onsets already on grid → offset should be 0."""
        from src.quantize_timing import _find_best_grid_offset
        interval = self.interval_16th
        onsets = np.array([0.0, interval, 2 * interval, 4 * interval])
        offset = _find_best_grid_offset(onsets, interval)
        self.assertAlmostEqual(offset, 0.0, places=4)

    def test_best_grid_offset_shifted(self):
        """Onsets with consistent offset from grid → offset should be ~0.03 s.

        We place onsets at [0.08, 0.08+interval, 0.08+2*interval] but t0
        is the first onset (0.08), so relative = [0, interval, 2*interval],
        and the function should find offset ≈ 0 since the relative positions
        are already grid-aligned.  To test offset detection, use onsets that
        are NOT the first note but have a different offset relationship.
        """
        from src.quantize_timing import _find_best_grid_offset
        interval = self.interval_16th
        # Onsets at fractional offsets: [0.0, interval+0.03, 2*interval+0.03]
        # The majority (last two) are offset by 0.03 → offset should be ~0.03
        onsets = np.array([0.0, interval + 0.03, 2 * interval + 0.03])
        offset = _find_best_grid_offset(onsets, interval)
        self.assertAlmostEqual(offset, 0.03, places=2)

    def test_nearest_grid_16th(self):
        """Note close to 16th grid snaps to it."""
        from src.quantize_timing import _nearest_grid
        interval = self.interval_16th
        t0 = 0.0
        # Onset 0.01 s off from grid → should snap to ~0.0
        snapped, gtype = _nearest_grid(
            0.01, 0.0, t0, interval, self.interval_triplet, self.threshold,
        )
        self.assertEqual(gtype, "16th")
        self.assertAlmostEqual(snapped, 0.0, places=3)

    def test_nearest_grid_triplet(self):
        """Note closer to triplet grid snaps as triplet."""
        from src.quantize_timing import _nearest_grid
        interval_16th = self.interval_16th
        interval_trip = self.interval_triplet
        # Place onset mid-way between 16th grid lines, close to triplet grid
        # 16th grid: 0, 0.203, 0.406, ...
        # Triplet grid: 0, 0.271, ...
        # Onset at 0.14 → 16th dev = 0.140, triplet dev = |0.140 - 0.271| = 0.131
        # Wait, that's closer to 16th. Let me compute properly.
        # At 74bpm: 16th=0.203, triplet=0.271
        # Onset at 0.25: 16th dev = |0.25-0.203|=0.047, trip dev = |0.25-0.271|=0.021
        snapped, gtype = _nearest_grid(
            0.25, 0.0, 0.0, interval_16th, interval_trip, self.threshold,
        )
        self.assertEqual(gtype, "triplet")
        self.assertAlmostEqual(snapped, interval_trip, places=3)

    def test_beyond_threshold(self):
        """Note too far from any grid → stays unchanged."""
        from src.quantize_timing import _nearest_grid
        interval = self.interval_16th
        # Halfway between two 16th grid lines: interval/2 > threshold/3
        onset = interval / 2
        snapped, gtype = _nearest_grid(
            onset, 0.0, 0.0, interval, self.interval_triplet, self.threshold,
        )
        self.assertIsNone(gtype)
        self.assertAlmostEqual(snapped, onset, places=6)

    def test_quantize_onsets_preserves_chords(self):
        """Notes in a chord (same onset) must move together."""
        from src.quantize_timing import quantize_onsets
        interval = self.interval_16th
        notes = [
            {"onset": 0.02, "offset": 1.0, "pitch": 60, "amplitude": 1.0},
            {"onset": 0.02, "offset": 1.0, "pitch": 64, "amplitude": 1.0},
            {"onset": interval + 0.02, "offset": 2.0, "pitch": 67, "amplitude": 1.0},
        ]
        result = quantize_onsets(notes, 73.9)
        # Both chord notes should have the same onset
        self.assertEqual(result[0]["onset"], result[1]["onset"])
        # Duration should be preserved
        self.assertAlmostEqual(result[0]["offset"] - result[0]["onset"], 0.98, places=2)

    def test_duration_preserved(self):
        """Offset shift should equal onset shift."""
        from src.quantize_timing import quantize_onsets
        notes = [
            {"onset": 0.05, "offset": 1.05, "pitch": 60, "amplitude": 1.0},
        ]
        result = quantize_onsets(notes, 73.9)
        onset_delta = result[0]["onset"] - 0.05
        offset_delta = result[0]["offset"] - 1.05
        self.assertAlmostEqual(onset_delta, offset_delta, places=6)

    def test_empty_notes(self):
        """Empty list returns empty list."""
        from src.quantize_timing import quantize_onsets
        result = quantize_onsets([], 73.9)
        self.assertEqual(result, [])

    def test_all_onsets_on_grid(self):
        """Onsets already aligned → no change."""
        from src.quantize_timing import quantize_onsets
        interval = self.interval_16th
        notes = [
            {"onset": 0.0, "offset": 0.5, "pitch": 60, "amplitude": 1.0},
            {"onset": interval, "offset": 0.5 + interval, "pitch": 62,
             "amplitude": 1.0},
            {"onset": 2 * interval, "offset": 0.5 + 2 * interval, "pitch": 64,
             "amplitude": 1.0},
        ]
        result = quantize_onsets(notes, 73.9)
        for orig, new in zip(notes, result):
            assert abs(new["onset"] - orig["onset"]) < 0.001, \
                f"onset {orig['onset']} moved to {new['onset']}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", action="store_true",
                        help="Run full pipeline integration test")
    args, remaining = parser.parse_known_args()

    if args.pipeline:
        print("=" * 60)
        print("Running full pipeline integration test on canon_test.wav")
        print("=" * 60)
        from src.pipeline import run_pipeline

        audio_path = os.path.join(
            os.path.dirname(__file__), "..", "output", "canon_test.wav",
        )
        if not os.path.exists(audio_path):
            print(f"ERROR: Test file not found at {audio_path}")
            sys.exit(1)

        output_dir = run_pipeline(audio_path)
        print(f"\nPipeline output: {output_dir}")

        # Check the generated MIDI exists
        piano_midi = os.path.join(output_dir, "piano.mid")
        if os.path.exists(piano_midi):
            print(f"Piano MIDI: {piano_midi}")
        else:
            print("WARNING: piano.mid not found")

        # Print timing stats
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(piano_midi)
        for inst in pm.instruments:
            onsets = [n.start for n in inst.notes]
            if len(onsets) > 1:
                intervals = np.diff(sorted(onsets))
                print(f"\n  {inst.name}: {len(onsets)} notes")
                print(f"    Mean inter-onset: {np.mean(intervals)*1000:.2f} ms")
                print(f"    Std inter-onset:  {np.std(intervals)*1000:.2f} ms")
                print(f"    Min inter-onset:  {np.min(intervals)*1000:.2f} ms")
                print(f"    Max inter-onset:  {np.max(intervals)*1000:.2f} ms")
    else:
        # Remove --pipeline from argv so unittest doesn't choke
        sys.argv = [sys.argv[0]] + remaining
        unittest.main()
