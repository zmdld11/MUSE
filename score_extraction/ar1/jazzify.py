"""AR1 L1 规则版爵士化 demo：canon → 爵士 ballad 改编 v1（2026-08-31）。

内容-风格解耦的最小实现：
- 内容不变量：旋律（score_mid 顶声部原样保留，含 rubato 逐拍时间）
- 风格重写：伴奏层全换 —— 原声贝斯 walking + 钢琴根音less爵士 voicing
  （Charleston 节奏）+ 架子鼓 swing ride/hi-hat/羽毛踢鼓
- 和声：根音取自转写低音声部（2 拍窗众数），九和弦/七和弦按卡农级数表升格
- 人性化：伴奏 ±12ms 抖动 + 速度微曲线（跟 beat_times 走天然带 rubato）

用法（项目根 cwd）: env/python.exe score_extraction/ar1/jazzify.py
输出: score_extraction/output/ar1_demo/canon_jazz_v1.{mid,wav}
"""
from __future__ import annotations

import json
import os
import random
import subprocess

import pretty_midi

SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(SE, "..", "frontend", "public", "demo", "canon")
OUT = os.path.join(SE, "output", "ar1_demo")
FS_EXE = os.path.join(SE, "external", "fluid_dll",
                      "fluidsynth-v2.5.6-win10-x64-cpp11", "bin", "fluidsynth.exe")
SF2 = os.path.join(SE, "data", "FluidR3_GM.sf2")

random.seed(20260901)
HUMAN = 0.012  # 伴奏时值抖动（秒）


def jitter() -> float:
    return random.uniform(-HUMAN, HUMAN)


def onset_groups(notes, tol=0.06):
    groups = []
    for n in notes:
        if groups and n.start - groups[-1][0].start <= tol:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


# ---------------------------------------------------------------- 素材读取
meta = json.load(open(os.path.join(DEMO, "notes.json"), encoding="utf-8"))
beats = meta["beat_times"]
dur = meta["duration"]

pm = pretty_midi.PrettyMIDI(os.path.join(DEMO, "notation", "score_mid", "piano.mid"))
src = sorted((n for i in pm.instruments for n in i.notes),
             key=lambda n: (n.start, -n.pitch))
groups = onset_groups(src)
melody = [sorted(g, key=lambda n: n.pitch)[-1] for g in groups]   # 顶声部=旋律（内容）
bassvox = [sorted(g, key=lambda n: n.pitch)[0] for g in groups]   # 底声部=和声根音线索

# ---------------------------------------------------------------- 和声提取
# 2 拍一窗，根音 = 窗内低音声部的 pitch-class 众数；品质按卡农级数表（C 大调）
QUAL = {0: "maj7", 7: "7", 9: "m7", 4: "m7", 5: "maj7"}
VOICING = {"maj7": [4, 7, 11, 14], "7": [4, 7, 10, 14], "m7": [3, 7, 10, 14]}
CANON_FALLBACK = [0, 7, 9, 4, 5, 0, 5, 7]  # C G Am Em F C F G


def beat_time(i: int) -> float:
    if i < len(beats):
        return beats[i]
    return beats[-1] + (i - len(beats) + 1) * (60.0 / meta["bpm"])


def chord_at_2beat(k: int) -> int:
    """第 k 个 2 拍窗的根音 pitch-class。"""
    t0, t1 = beat_time(2 * k), beat_time(2 * k + 2)
    votes = {}
    for n in bassvox:
        if t0 <= n.start < t1:
            votes[n.pitch % 12] = votes.get(n.pitch % 12, 0) + 1
    if votes:
        return max(votes, key=votes.get)
    return CANON_FALLBACK[k % 8]  # 低音静默窗回退卡农循环


n2beat = int(dur / (60.0 / meta["bpm"])) + 2
roots = [chord_at_2beat(k) for k in range(n2beat // 2 + 1)]

# ---------------------------------------------------------------- 生成
out = pretty_midi.PrettyMIDI(initial_tempo=int(round(meta["bpm"])))

# 1) 旋律：转写内容原样（顶声部），爵士三角钢琴
mel_inst = pretty_midi.Instrument(program=0, name="melody(transcribed)")
for n in melody:
    st = n.start + jitter() * 0.6
    en = max(n.end, st + 0.06)  # 零长音符/抖动越界防御
    mel_inst.notes.append(pretty_midi.Note(
        velocity=min(115, n.velocity + 18), pitch=n.pitch, start=st, end=en))
out.instruments.append(mel_inst)

# 2) 钢琴 comping：根音less voicing（3/5/7/9），Charleston 节奏（1 拍 + 第 2 拍反拍）
comp = pretty_midi.Instrument(program=0, name="comping")
for k, root in enumerate(roots):
    q = QUAL.get(root, "maj7")
    iv = VOICING[q]
    anchor_pc = (root + iv[0]) % 12                     # 三音 pitch-class
    anchor = 52 + ((anchor_pc - 4) % 12)                # 三音锚点落 E3..D#4
    tones = [anchor + x - iv[0] for x in iv]            # 相对三音的根音less堆叠
    b1 = beat_time(2 * k)
    b2 = beat_time(2 * k + 1)
    b25 = b1 + (beat_time(2 * k + 2) - b1) * 0.667  # swing 反拍
    for t, dur_n, vel in ((b1, min(0.9, b25 - b1), 62), (b25, 0.35, 54)):
        for pc in tones:
            comp.notes.append(pretty_midi.Note(
                velocity=max(20, int(vel + random.uniform(-8, 8))),
                pitch=pc, start=t + jitter(), end=t + dur_n))
out.instruments.append(comp)

# 3) 原声贝斯 walking：每 2 拍窗两音（根音→三/五音 或 半音导向下一根音）
bass = pretty_midi.Instrument(program=32, name="bass")
CHORD_TONES = {"maj7": [0, 4, 7, 11], "7": [0, 4, 10, 14], "m7": [0, 3, 7, 10]}
for k in range(len(roots) - 1):
    root, nxt = roots[k], roots[k + 1]
    q = QUAL.get(root, "maj7")
    low = 36 + root if root >= 5 else 48 + root
    nxt_low = 36 + nxt if nxt >= 5 else 48 + nxt
    qt = sorted(set(low + i for i in CHORD_TONES[q]))
    second = qt[1] if qt[1] - low <= 7 else low + 7
    approach = nxt_low - 1 if nxt_low > low else nxt_low + 1
    step = (low, second if k % 2 == 0 else approach)
    for j, p in enumerate(step):
        t = beat_time(2 * k + j)
        tn = beat_time(2 * k + j + 1)
        bass.notes.append(pretty_midi.Note(
            velocity=max(30, int(88 + random.uniform(-8, 6))),
            pitch=p, start=t + jitter() * 0.7,
            end=t + max(0.12, (tn - t) * 0.92)))
out.instruments.append(bass)

# 4) 鼓组（GM ch10）：swing ride + 2/4 闭镲 + 羽毛踢鼓
drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")


def hit(pitch, t, vel):
    drums.notes.append(pretty_midi.Note(
        velocity=max(15, int(vel + random.uniform(-6, 6))),
        pitch=pitch, start=t + jitter(), end=t + 0.18))


for i in range(n2beat):
    t = beat_time(i)
    hit(51, t, 44)                                   # ride 四分
    hit(51, t + (beat_time(i + 1) - t) * 0.667, 34)  # ride swing 反拍
    if i % 4 in (1, 3):
        hit(42, t, 40)                               # 闭镲 2/4
    if i % 2 == 0:
        hit(36, t, 26)                               # 羽毛踢鼓
out.instruments.append(drums)

# ---------------------------------------------------------------- 输出
os.makedirs(OUT, exist_ok=True)
mid_path = os.path.join(OUT, "canon_jazz_v1.mid")
wav_path = os.path.join(OUT, "canon_jazz_v1.wav")
out.write(mid_path)
subprocess.run([FS_EXE, "-ni", "-g", "0.8", "-F", wav_path, SF2, mid_path],
               check=True, capture_output=True, timeout=600)

import soundfile as sf  # noqa: E402
import numpy as np  # noqa: E402
x, sr = sf.read(wav_path)
peak = float(np.abs(x).max()) if x.ndim == 1 else float(np.abs(x).max())
print(f"melody notes={len(mel_inst.notes)} comping={len(comp.notes)} "
      f"bass={len(bass.notes)} drums={len(drums.notes)}")
print(f"roots(cycle head)={[QUAL.get(r, 'maj7') + ':' + str(r) for r in roots[:8]]}")
print(f"wav: {len(x)/sr:.1f}s peak={peak:.3f} -> {wav_path}")
