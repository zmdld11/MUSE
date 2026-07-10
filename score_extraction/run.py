"""Entry point for running the pipeline on a single audio file."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import run_pipeline

if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "output/canon_test.wav"
    print(f"Running pipeline on: {audio}")
    result = run_pipeline(audio)
    print(f"Done: {result}")
