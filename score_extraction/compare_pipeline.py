"""Compare pipeline output against reference score for 夜の向日葵.

Metrics:
  1. Note count per hand vs reference
  2. Chroma similarity (melody contour)
  3. Onset F1 with 50ms tolerance (standard AMT metric)
  4. Note density per measure (rhythm alignment)
  5. Pitch histogram correlation
"""
import sys, os, json
import numpy as np
import pretty_midi

REF_MIDI = r"d:\program_project\MUSE\score_extraction\output\himawari_reference\himawari_reference.mid"
PIPELINE_MIDI = r"d:\program_project\MUSE\score_extraction\output\夜の向日葵 - 松本文紀\piano.mid"
BEAT_SEC = 0.731  # 731ms per beat at 82 BPM
ONSET_TOLERANCE = 0.05  # 50ms (standard AMT)


def load_notes(midi_path):
    """Load MIDI, return sorted notes with instrument info."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            all_notes.append({
                "onset": n.start,
                "offset": n.end,
                "pitch": int(n.pitch),
                "inst": inst.name,
            })
    return sorted(all_notes, key=lambda x: x["onset"])


def split_hands(notes):
    """Split notes into right (high) and left (low) by pitch median."""
    # For reference: use instrument names
    # For pipeline: split by pitch (C4=60 is typical boundary)
    right = [n for n in notes if n["pitch"] >= 60]
    left = [n for n in notes if n["pitch"] < 60]
    return right, left


def chroma_histogram(notes):
    """Compute chroma profile from notes (C, C#, D, ..., B), weighted by duration."""
    chroma = np.zeros(12)
    for n in notes:
        chroma[n["pitch"] % 12] += n["offset"] - n["onset"]
    total = chroma.sum()
    if total > 0:
        chroma /= total
    return chroma


def best_transposition_correlation(ref_chroma, pred_chroma):
    """Find the transposition (0-11 semitones) that maximizes chroma correlation."""
    best_corr = -1
    best_shift = 0
    for shift in range(12):
        shifted = np.roll(pred_chroma, -shift)  # transpose pred down by 'shift'
        corr = np.corrcoef(ref_chroma, shifted)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return best_shift, best_corr


def onset_f1(ref_notes, pred_notes, tolerance=ONSET_TOLERANCE):
    """Compute onset-level precision, recall, F1 with tolerance window."""
    matched_ref = set()
    matched_pred = set()

    for i, rn in enumerate(ref_notes):
        for j, pn in enumerate(pred_notes):
            if j in matched_pred:
                continue
            if rn["pitch"] == pn["pitch"] and abs(rn["onset"] - pn["onset"]) <= tolerance:
                matched_ref.add(i)
                matched_pred.add(j)
                break

    tp = len(matched_ref)
    precision = tp / len(pred_notes) if pred_notes else 0
    recall = tp / len(ref_notes) if ref_notes else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, tp


def pitch_transitions(notes):
    """Get melody direction: +1 (up), -1 (down), 0 (same)."""
    sorted_notes = sorted(notes, key=lambda x: x["onset"])
    directions = []
    for i in range(len(sorted_notes) - 1):
        if sorted_notes[i + 1]["pitch"] > sorted_notes[i]["pitch"]:
            directions.append(1)
        elif sorted_notes[i + 1]["pitch"] < sorted_notes[i]["pitch"]:
            directions.append(-1)
        else:
            directions.append(0)
    return directions


def melody_direction_agreement(ref_notes, pred_notes, n_bins=100):
    """Compare melody contour by binning and correlating direction vectors."""
    if len(ref_notes) < 2 or len(pred_notes) < 2:
        return 0.0

    max_time = max(
        max(n["onset"] for n in ref_notes) if ref_notes else 0,
        max(n["onset"] for n in pred_notes) if pred_notes else 0,
    )
    bin_size = max_time / n_bins

    def direction_bins(notes):
        # For each time bin, take median pitch of notes in that bin
        bins = [[] for _ in range(n_bins)]
        for n in notes:
            bi = min(int(n["onset"] / bin_size), n_bins - 1)
            bins[bi].append(n["pitch"])
        # Get pitch per bin
        pitches = []
        for b in bins:
            pitches.append(int(np.median(b)) if b else None)
        # Get directions between consecutive non-None bins
        dirs = []
        last = None
        for p in pitches:
            if p is not None:
                if last is not None:
                    dirs.append(1 if p > last else (-1 if p < last else 0))
                last = p
        return dirs

    ref_dirs = direction_bins(ref_notes)
    pred_dirs = direction_bins(pred_notes)

    # Align lengths
    min_len = min(len(ref_dirs), len(pred_dirs))
    if min_len < 2:
        return 0.0

    agreement = sum(1 for i in range(min_len) if ref_dirs[i] == pred_dirs[i]) / min_len
    return agreement


def note_density_per_beat(notes, beat_sec, start=0, end=None):
    """Count notes per beat for density profile."""
    if end is None:
        end = max(n["onset"] for n in notes) if notes else 0
    n_beats = int((end - start) / beat_sec) + 1
    density = np.zeros(n_beats)
    for n in notes:
        bi = int((n["onset"] - start) / beat_sec)
        if 0 <= bi < n_beats:
            density[bi] += 1
    return density


def main():
    print("=" * 60)
    print("夜の向日葵 — Pipeline vs Reference Comparison")
    print("=" * 60)

    # Load reference
    ref_notes = load_notes(REF_MIDI)
    ref_right, ref_left = split_hands(ref_notes)
    print(f"\nReference: {len(ref_notes)} total ({len(ref_right)} right, {len(ref_left)} left)")

    if not os.path.exists(PIPELINE_MIDI):
        print(f"Pipeline output not found: {PIPELINE_MIDI}")
        return

    pred_notes = load_notes(PIPELINE_MIDI)
    pred_right, pred_left = split_hands(pred_notes)
    print(f"Pipeline:  {len(pred_notes)} total ({len(pred_right)} right, {len(pred_left)} left)")

    # ---- Metric 1: Note count ----
    print(f"\n{'='*40}")
    print("1. Note Count")
    print(f"  Reference: {len(ref_notes)} ({len(ref_right)} R / {len(ref_left)} L)")
    print(f"  Pipeline:  {len(pred_notes)} ({len(pred_right)} R / {len(pred_left)} L)")
    ratio = len(pred_notes) / len(ref_notes) * 100 if ref_notes else 0
    print(f"  Coverage:  {ratio:.1f}%")

    # ---- Metric 2: Chroma similarity (with key alignment) ----
    print(f"\n{'='*40}")
    print("2. Chroma Profile (melody contour)")
    ref_chroma = chroma_histogram(ref_notes)
    pred_chroma = chroma_histogram(pred_notes)
    raw_corr = np.corrcoef(ref_chroma, pred_chroma)[0, 1]

    shift, aligned_corr = best_transposition_correlation(ref_chroma, pred_chroma)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    print(f"  Raw correlation: {raw_corr:.3f}")
    print(f"  Best transposition: +{shift} semitones → correlation: {aligned_corr:.3f}")
    print(f"  Ref  chroma: {dict(zip(note_names, ref_chroma.round(3)))}")
    print(f"  Pred chroma: {dict(zip(note_names, pred_chroma.round(3)))}")

    # Transpose pred notes for fair pitch comparison
    pred_transposed = []
    for n in pred_notes:
        nt = dict(n)
        nt["pitch"] = n["pitch"] - shift
        pred_transposed.append(nt)
    pred_right_t, pred_left_t = split_hands(pred_transposed)

    # ---- Metric 3: Onset F1 (with transposed pitches) ----
    print(f"\n{'='*40}")
    print(f"3. Onset F1 (tolerance={ONSET_TOLERANCE*1000:.0f}ms, pitch match, pred transposed -{shift}semitones)")
    for name, ref, pred in [("Right", ref_right, pred_right_t), ("Left", ref_left, pred_left_t)]:
        p, r, f1, tp = onset_f1(ref, pred)
        print(f"  {name:6s}: P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp})")

    # ---- Metric 4: Melody direction agreement (transposed) ----
    print(f"\n{'='*40}")
    print("4. Melody Direction Agreement (transposed)")
    for name, ref, pred in [("Right", ref_right, pred_right_t), ("Left", ref_left, pred_left_t)]:
        agree = melody_direction_agreement(ref, pred)
        print(f"  {name:6s}: {agree:.3f}")

    # ---- Metric 5: Note density correlation ----
    print(f"\n{'='*40}")
    print("5. Note Density per Beat (correlation)")
    max_time = max(
        max(n["onset"] for n in ref_notes) if ref_notes else 0,
        max(n["onset"] for n in pred_notes) if pred_notes else 0,
    )
    for name, ref, pred in [("Right", ref_right, pred_right), ("Left", ref_left, pred_left)]:
        ref_den = note_density_per_beat(ref, BEAT_SEC, end=max_time)
        pred_den = note_density_per_beat(pred, BEAT_SEC, end=max_time)
        if len(ref_den) > 1 and len(pred_den) > 1:
            corr = np.corrcoef(ref_den, pred_den)[0, 1]
            print(f"  {name:6s}: {corr:.3f} (ref={ref_den.sum():.0f} pred={pred_den.sum():.0f} notes)")
        else:
            print(f"  {name:6s}: N/A")

    print(f"\n{'='*60}")
    print("Comparison complete.")


if __name__ == "__main__":
    main()
