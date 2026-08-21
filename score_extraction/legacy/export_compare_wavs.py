"""导出对比音频: VER3.2 原样 vs VER4.0 强平滑, 两首歌.

用法: python -u export_compare_wavs.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pretty_midi
import scipy.io.wavfile as wf
from scipy import ndimage

from src.transcriber import transcribe
from src.frame_post import process_frames_bp

MODEL_DIR = os.path.join(ROOT, "model")
JOBS = [
    ("canon", r"output\canon_ver4\piano_denoised.wav", r"output\canon_ver4"),
    ("himawari", r"output\himawari_ver4\piano_denoised.wav", r"output\himawari_ver4"),
]


def write_midi(notes, path, bpm=73.8):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=70, pitch=int(n["pitch"]),
            start=float(n["onset_time"]), end=float(n["offset_time"])))
    pm.instruments.append(inst)
    pm.write(path)


def render_wav(midi_path, wav_path):
    from train.render_midi import render_midi, probe_fluidsynth
    probe_fluidsynth(quiet=True)
    data = render_midi(midi_path, sr=44100, max_dur_sec=600)
    audio = np.asarray(data["audio"], dtype=np.float64)
    audio = audio / max(1e-9, np.abs(audio).max())
    wf.write(wav_path, 44100, (audio * 30000).astype(np.int16))


def main():
    from src import transcriber as tc
    for tag, model_name in [("v32", "VER3.2_Enhanced.pth"),
                            ("v4", "VER4.0_best.pth")]:
        os.environ["MUSE_MODEL_PATH"] = os.path.join(MODEL_DIR, model_name)
        tc._model = None
        for song, wav, outdir in JOBS:
            print(f"=== {tag} {song} ===", flush=True)
            res = transcribe(wav)
            if tag == "v4":
                f = ndimage.median_filter(res["frame_probs"],
                                          size=(19, 1), mode="nearest")
                notes = process_frames_bp(res["onset_probs"], f, hop_length=512,
                                          sr=22050, onset_thresh=0.4,
                                          frame_thresh=0.2, min_note_len=15,
                                          energy_tol=5, melodia_trick=False)
            else:
                notes = process_frames_bp(res["onset_probs"], res["frame_probs"],
                                          hop_length=512, sr=22050,
                                          onset_thresh=0.4, frame_thresh=0.2,
                                          min_note_len=5, energy_tol=5,
                                          melodia_trick=True)
            durs = np.array([x["offset_time"] - x["onset_time"] for x in notes])
            print(f"  notes={len(notes)} med={np.median(durs)*1000:.0f}ms", flush=True)
            mid = os.path.join(outdir, f"piano_{tag}.mid")
            write_midi(notes, mid)
            render_wav(mid, os.path.join(outdir, f"piano_{tag}.wav"))
            print(f"  -> {mid}", flush=True)


if __name__ == "__main__":
    main()
