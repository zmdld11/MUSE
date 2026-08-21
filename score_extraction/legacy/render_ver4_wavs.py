"""把 VER4.0 输出的两首 MIDI 渲染成 WAV 供试听."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.io.wavfile as wf

from train.render_midi import render_midi, probe_fluidsynth

print("fluidsynth usable:", probe_fluidsynth(quiet=True), flush=True)

JOBS = [
    ("canon_ver4", r"output\canon_ver4\piano.mid",
     r"output\canon_ver4\piano_ver4.wav"),
    ("himawari_ver4", r"output\himawari_ver4\piano.mid",
     r"output\himawari_ver4\piano_ver4.wav"),
]

for name, mid, wav in JOBS:
    data = render_midi(mid, sr=44100, max_dur_sec=600)
    audio = np.asarray(data["audio"], dtype=np.float64)
    audio = audio / max(1e-9, np.abs(audio).max())
    wf.write(wav, 44100, (audio * 30000).astype(np.int16))
    print(f"{name}: {len(audio) / 44100:.1f}s -> {wav}", flush=True)
