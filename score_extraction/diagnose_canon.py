"""Diagnose Canon output: compare generated notes against expected structure.
Fixed to use absolute offsets and properly handle chords/voices.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from music21 import converter

# Load generated MusicXML
xml_path = r"d:\program_project\MUSE\score_extraction\output\Variations On The Canon By Pachelbel - George Winston\piano_final.musicxml"
score = converter.parse(xml_path)

# Extract all notes with timing — use direct .offset (absolute position)
notes = []
for part in score.parts:
    for n in part.flatten().notes:
        if n.isNote:
            offset = float(n.offset)      # absolute offset in quarterLength
            dur = float(n.duration.quarterLength)
            notes.append({
                "pitch": n.pitch.midi,
                "pitch_name": n.pitch.nameWithOctave,
                "onset": offset,
                "duration": dur,
            })
        elif n.isChord:
            for p in n.pitches:
                offset = float(n.offset)
                dur = float(n.duration.quarterLength)
                notes.append({
                    "pitch": p.midi,
                    "pitch_name": p.nameWithOctave,
                    "onset": offset,
                    "duration": dur,
                })

notes.sort(key=lambda n: n["onset"])
print(f"Total notes: {len(notes)}")
if notes:
    print(f"Time span: {notes[0]['onset']:.1f} — {notes[-1]['onset']:.1f} beats")
    print(f"Pitch range: {min(n['pitch'] for n in notes)} – {max(n['pitch'] for n in notes)}")

# ===== 1. Pitch class distribution =====
pc_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
pc_counts = np.zeros(12)
for n in notes:
    pc_counts[n["pitch"] % 12] += n["duration"]

total = pc_counts.sum()
print("\n=== Pitch Class Distribution (weighted by duration) ===")
for i, name in enumerate(pc_names):
    bar = "█" * int(pc_counts[i] / total * 50) if total > 0 else ""
    print(f"  {name:3s}: {pc_counts[i]/total*100:5.1f}% {bar}")

# Expected Canon bass line in C major (I-V-vi-iii-IV-I-IV-V)
# In actual Canon in D: D-A-Bm-F#m-G-D-G-A  (I-V-vi-iii-IV-I-IV-V)
# Transposed to C: C-G-Am-Em-F-C-F-G
# Pitch classes relative to tonic (C=0): [0, 7, 9, 4, 5, 0, 5, 7]
canon_bass_pc = [0, 7, 9, 4, 5, 0, 5, 7]

# ===== 2. Detect bass line pattern =====
beat_duration = 1.0
window = beat_duration
bass_notes = []
current_beat = 0
max_onset = notes[-1]["onset"] if notes else 0
while current_beat < max_onset:
    window_notes = [n for n in notes if current_beat <= n["onset"] < current_beat + window]
    if window_notes:
        lowest = min(window_notes, key=lambda n: n["pitch"])
        bass_notes.append(lowest["pitch"] % 12)
    current_beat += window

print(f"\n=== Bass Line (lowest pitch per beat window, first 64 beats) ===")
bass_pc_seq = bass_notes[:64]
bass_str = " ".join(f"{pc_names[pc]:2s}" for pc in bass_pc_seq[:32])
print(f"  Beats  0-31: {bass_str}")
if len(bass_pc_seq) > 32:
    bass_str2 = " ".join(f"{pc_names[pc]:2s}" for pc in bass_pc_seq[32:64])
    print(f"  Beats 32-63: {bass_str2}")

# Check if bass follows the Canon pattern (autocorrelation)
if len(bass_pc_seq) >= 16:
    pattern = np.array(canon_bass_pc)
    matches = 0
    total_windows = len(bass_pc_seq) - 8
    for i in range(total_windows):
        segment = np.array(bass_pc_seq[i:i+8])
        segment_rel = (segment - segment[0]) % 12
        pattern_rel = (pattern - pattern[0]) % 12
        if np.array_equal(segment_rel, pattern_rel):
            matches += 1
    print(f"\n  Canon bass pattern matches (exact): {matches}/{total_windows} windows")

# ===== 3. Onset timing analysis =====
onsets = [n["onset"] for n in notes]
diffs = np.diff(onsets)
print(f"\n=== Timing Analysis ===")
print(f"  Inter-onset intervals: min={diffs.min():.3f} beats, max={diffs.max():.3f}, median={np.median(diffs):.3f}")
print(f"  Mean notes per beat: {1/np.median(diffs):.1f}")

# Polyphony detection using absolute offsets
overlap_count = 0
for i in range(1, len(notes)):
    if notes[i]["onset"] < notes[i-1]["onset"] + notes[i-1]["duration"]:
        overlap_count += 1
print(f"  Overlapping notes (polyphony): {overlap_count}/{len(notes)} ({overlap_count/len(notes)*100:.0f}%)")

# ===== 4. Duration distribution =====
durations = [n["duration"] for n in notes]
print(f"\n=== Duration Distribution ===")
print(f"  min={min(durations):.2f}, max={max(durations):.2f}, median={np.median(durations):.2f}, mean={np.mean(durations):.2f}")
dur_buckets = {"32nd↓": 0, "16th": 0, "8th": 0, "quarter": 0, "half↑": 0}
for d in durations:
    if d <= 0.25: dur_buckets["32nd↓"] += 1
    elif d <= 0.5: dur_buckets["16th"] += 1
    elif d <= 1.0: dur_buckets["8th"] += 1
    elif d <= 2.0: dur_buckets["quarter"] += 1
    else: dur_buckets["half↑"] += 1
for k, v in dur_buckets.items():
    bar = "█" * (v * 50 // len(durations)) if len(durations) > 0 else ""
    print(f"  {k:8s}: {v:5d} ({v/len(durations)*100:5.1f}%) {bar}")

# ===== 5. Metric alignment =====
beat1_hits = sum(1 for n in notes if abs(n["onset"] % 4.0) < 0.1)
beat3_hits = sum(1 for n in notes if abs(n["onset"] % 4.0 - 2.0) < 0.1)
offbeats = sum(1 for n in notes if 0.3 < n["onset"] % 1.0 < 0.7)
print(f"\n=== Metric Alignment ===")
print(f"  Notes on beat 1: {beat1_hits}")
print(f"  Notes on beat 3: {beat3_hits}")
print(f"  Notes on offbeats: {offbeats}")

print("\n=== DIAGNOSIS COMPLETE ===")
