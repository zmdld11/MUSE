"""服务器端：MAESTRO 40 首裁前 90s 到暂存目录（纯 stdlib wave，逐样本精确）。"""
import os
import shutil
import wave

SRC = os.path.expanduser("~/zmdld11/score_extraction/data/maestro/maestro-v3.0.0")
STAGE = os.path.expanduser("~/zmdld11/tmp_f2piano_fetch")
SEC = 90

pairs = [l.strip() for l in open("/tmp/f2p_fetch_list.txt") if l.strip()]
audios, midis = pairs[0::2], pairs[1::2]
assert len(audios) == len(midis) == 40, (len(audios), len(midis))

n_wav = 0
for a in audios:
    dst = os.path.join(STAGE, a)
    if os.path.exists(dst):
        n_wav += 1
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with wave.open(os.path.join(SRC, a), "rb") as w:
        p = w.getparams()
        nf = min(w.getnframes(), SEC * p.framerate)
        data = w.readframes(nf)
    with wave.open(dst, "wb") as w:
        w.setparams(p)
        w.setnframes(nf)
        w.writeframes(data)
    n_wav += 1

n_mid = 0
for m in midis:
    dst = os.path.join(STAGE, m)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(SRC, m), dst)
    n_mid += 1

print(f"TRIM_DONE wavs={n_wav} midis={n_mid}")
for root, _, files in os.walk(STAGE):
    for f in files:
        if f.endswith(".wav"):
            sz = os.path.getsize(os.path.join(root, f))
            assert 15.5e6 < sz < 16.5e6, (f, sz)
print("SIZE_OK expect ~15.87MB per 90s stereo 16bit 44.1k")
