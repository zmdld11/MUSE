"""Run pipeline on a given audio file."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

audio_path = sys.argv[1]
print(f"Running pipeline on: {audio_path}")
result = run_pipeline(audio_path)
print(f"\nDone! Output: {result}")
