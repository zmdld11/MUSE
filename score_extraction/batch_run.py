"""Batch run pipeline on all WAV files in tmp_wav."""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

wav_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_wav")
files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))

print(f"Found {len(files)} files to process\n")

already = {"Variations On The Canon By Pachelbel - George Winston.wav"}

for i, f in enumerate(files):
    if os.path.basename(f) in already:
        print(f"[{i+1}/{len(files)}] SKIP: {os.path.basename(f)} (already done)")
        continue
    name = os.path.basename(f)
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(files)}] {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = run_pipeline(f)
        elapsed = time.time() - start
        print(f"OK ({elapsed:.0f}s): {result}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAIL ({elapsed:.0f}s): {e}")

print("\n\n=== ALL DONE ===")
print("Outputs in: d:/program_project/MUSE/score_extraction/output/")
