"""F2' 42 SI-SDR：tasep vs v30b 两臂（mono 口径），服务器跑。

输出 ~/zmdld11/score_extraction/output/f2p_sisdr_tasep_vs_v30b.json
"""
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
    """先 FFT 互相关搜 ±50ms 最优滞后再算（MSST 分离链有 ~0.3ms 系统偏移，
    采样级错位会毁掉 SI-SDR——item_35 实测 sdr@0 -15.9 vs 对齐后 +8.5）。"""
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


def find_tasep(tag):
    for out in sorted((SE / "data" / "sep_tasep_v1_eval").glob("out_*")):
        p = out / tag / "Guitar.wav"
        if p.exists():
            return p
    return None


rows, miss = [], []
for gtp in sorted(GT.glob("*.wav")):
    tag = gtp.stem
    v30p = V30 / tag / "Guitar.wav"
    tasp = find_tasep(tag)
    if not (v30p.exists() and tasp):
        miss.append(tag)
        continue
    ref = load_mono(gtp)
    sv, lv = si_sdr_aligned(ref, load_mono(v30p))
    st, lt = si_sdr_aligned(ref, load_mono(tasp))
    rows.append({"tag": tag, "v30b": round(float(sv), 3), "tasep": round(float(st), 3),
                 "lag_v30b": lv, "lag_tasep": lt})

v = np.array([r["v30b"] for r in rows])
t = np.array([r["tasep"] for r in rows])
out = {"n": len(rows), "missing": miss,
       "v30b_mean": round(float(v.mean()), 3),
       "tasep_mean": round(float(t.mean()), 3),
       "diff_mean": round(float((t - v).mean()), 3),
       "diff_min": round(float((t - v).min()), 3),
       "improved": int((t > v).sum()), "per_pair": rows}
dst = SE / "output" / "f2p_sisdr_tasep_vs_v30b.json"
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("n", "missing", "v30b_mean",
                                      "tasep_mean", "diff_mean",
                                      "diff_min", "improved")}, indent=1))
print("->", dst)
