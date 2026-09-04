"""F2' 42 SI-SDR：v30b vs tasep_v2（β 项）两臂，mono+FFT 对齐口径（同 v1 版）。"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SE = Path.home() / "zmdld11" / "score_extraction"
GT = SE / "data" / "f2p_v1" / "gt_wav"
V30 = SE / "data" / "sep_v30b_eval" / "out"


def load_mono(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == 44100, (sr, p)
    return x.mean(axis=1)


def si_sdr(ref, est):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    t = a * ref
    return 10 * np.log10((np.dot(t, t) + 1e-12) /
                         (np.dot(est - t, est - t) + 1e-12))


def si_sdr_aligned(ref, est, max_lag=2205):
    n = min(len(ref), len(est))
    r, e = ref[:n], est[:n]
    pad = 1 << (n + 2 * max_lag - 1).bit_length()
    R = np.fft.rfft(r, pad)
    E = np.fft.rfft(e, pad)
    cc = np.fft.irfft(E * np.conj(R), pad)
    lags = np.concatenate([cc[:max_lag + 1], cc[-max_lag:]])
    k = int(np.argmax(lags))
    lag = k if k <= max_lag else k - len(lags)
    if lag > 0:
        rr, ee = r[:-lag], e[lag:]
    elif lag < 0:
        rr, ee = r[-lag:], e[:lag]
    else:
        rr, ee = r, e
    return si_sdr(rr, ee), lag


def find_v2(tag):
    for out in sorted((SE / "data" / "sep_tasep_v2_eval").glob("out_*")):
        p = out / tag / "Guitar.wav"
        if p.exists():
            return p
    return None


rows, miss = [], []
for gtp in sorted(GT.glob("*.wav")):
    tag = gtp.stem
    v30p = V30 / tag / "Guitar.wav"
    v2p = find_v2(tag)
    if not (v30p.exists() and v2p):
        miss.append(tag)
        continue
    ref = load_mono(gtp)
    sv, _ = si_sdr_aligned(ref, load_mono(v30p))
    st, _ = si_sdr_aligned(ref, load_mono(v2p))
    rows.append({"tag": tag, "v30b": round(float(sv), 3), "tasep_v2": round(float(st), 3)})

v = np.array([r["v30b"] for r in rows])
t = np.array([r["tasep_v2"] for r in rows])
out = {"n": len(rows), "missing": miss,
       "v30b_mean": round(float(v.mean()), 3),
       "tasep_v2_mean": round(float(t.mean()), 3),
       "diff_mean": round(float((t - v).mean()), 3),
       "diff_min": round(float((t - v).min()), 3),
       "improved": int((t > v).sum()), "per_pair": rows}
dst = SE / "output" / "f2p_sisdr_tasep_v2_vs_v30b.json"
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("n", "missing", "v30b_mean", "tasep_v2_mean",
                                      "diff_mean", "diff_min", "improved")}, indent=1))
print("->", dst)
