# 乐谱生成模型 VER1.0 实施计划

> **For agentic workers:** 使用 superpowers:subagent-driven-development 按任务逐个实施。
> 步骤使用 checkbox (`- [ ]`) 语法追踪。

**目标:** 输入音频文件，经 BPM检测→demucs分离→音高识别→调号推算→乐谱组装，输出 5 份独立 MusicXML 乐谱。

**架构:** 管道式流水线，每个模块单一职责，通过明确函数接口串联。主入口 pipeline.py 编排所有步骤，外部模型失败时降级而非崩溃。

**技术栈:** Python 3.10, librosa, demucs (htdemucs_6s), crepe/torchcrepe, basic-pitch, madmom, music21, numpy, tqdm

## 全局约束

- 每个音轨独立一份 MusicXML，不做合并
- 外部模型失败时降级：BPM→120 fallback，钢琴→crepe 单音 fallback，madmom→跳过
- demucs 失败为致命错误，直接终止
- 吉他排指：电吉他 22 品标准调弦，DP 最短路径，不追踪手指状态
- 命名风格：snake_case 函数，中文注释可接受
- 使用 Config 类模式（参考 instrument_recognition/src/config.py）

---

### Task 1: 项目脚手架

**Files:** Create `src/__init__.py`, `src/config.py`, `data/__init__.py`, `test/__init__.py`
**Produces:** `config` 全局单例

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p d:/program_project/MUSE/score_extraction/src
mkdir -p d:/program_project/MUSE/score_extraction/data
mkdir -p d:/program_project/MUSE/score_extraction/test/fixtures
mkdir -p d:/program_project/MUSE/score_extraction/output
mkdir -p d:/program_project/MUSE/score_extraction/model
```

- [ ] **Step 2: 创建空 __init__.py**

```bash
touch d:/program_project/MUSE/score_extraction/src/__init__.py
touch d:/program_project/MUSE/score_extraction/data/__init__.py
touch d:/program_project/MUSE/score_extraction/test/__init__.py
```

- [ ] **Step 3: 写入 `src/config.py`**

```python
import os

class Config:
    def __init__(self):
        self.WORKSPACE_DIR = r"D:\program_project\MUSE\score_extraction"
        self.OUTPUT_DIR = os.path.join(self.WORKSPACE_DIR, "output")
        self.MODEL_DIR = os.path.join(self.WORKSPACE_DIR, "model")
        self.DEMUCS_MODEL = "htdemucs_6s"
        self.DEMUCS_MODEL_PATH = None
        self.PITCH_MODEL_PIANO = "basic-pitch"
        self.PITCH_MODEL_MONO = "crepe"
        self.SR = 44100
        self.HOP_LENGTH = 512
        self.DEFAULT_BPM = 120.0
        self.DEFAULT_TIME_SIG = "4/4"
        self.MAX_FRET = 22
        self.FRET_WEIGHT = 1.0
        self.STRING_WEIGHT = 2.0
        self.OPEN_STRING_BIAS = -0.5
        self.SLIDE_PITCH_THRESHOLD = 0.5
        self.SLIDE_MAX_INTERVAL = 5
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

config = Config()
```

- [ ] **Step 4: 验证 config 导入**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "from src.config import config; print('OK:', config.DEMUCS_MODEL, config.SR)"
```
预期输出: `OK: htdemucs_6s 44100`

### Task 2: BPM 检测 — bpm_detect.py

**Files:** Create `src/bpm_detect.py`
**Produces:** `detect_bpm(audio_path: str) -> float | None`
**Consumes:** `config.SR`

- [ ] **Step 1: 生成 120BPM 节拍器测试音频**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
import numpy as np; import soundfile as sf
sr = 44100; bpm = 120; interval = 60/bpm; dur = 10.0
n_samples = int(sr * dur)
audio = np.zeros(n_samples)
click_len = int(0.05 * sr)
click = np.sin(2 * np.pi * 1000 * np.arange(click_len) / sr) * 0.9
for i in range(int(dur / interval)):
    s = int(i * interval * sr)
    if s + click_len < n_samples:
        audio[s:s + click_len] = click
sf.write('test/fixtures/test_120bpm.wav', audio, sr)
print('fixture generated')
"
```

- [ ] **Step 2: 实现 detect_bpm**

```python
# src/bpm_detect.py
import logging
import librosa

logger = logging.getLogger(__name__)


def detect_bpm(audio_path: str, sr: int = 44100) -> float | None:
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])

        if bpm < 40 or bpm > 250:
            logger.warning(f"BPM {bpm} out of range, using fallback")
            return None

        logger.info(f"BPM detected: {bpm:.1f}")
        return round(bpm, 1)

    except Exception as e:
        logger.warning(f"BPM detection failed: {e}")
        return None
```

- [ ] **Step 3: 验证 BPM 检测**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
from src.bpm_detect import detect_bpm
bpm = detect_bpm('test/fixtures/test_120bpm.wav')
print(f'Detected BPM: {bpm}')
assert bpm is not None and abs(bpm - 120) < 3, f'BPM deviation: {bpm}'
print('PASS')
"
```

### Task 3: 音轨分离 — source_separate.py

**Files:** Create `src/source_separate.py`
**Produces:** `separate_tracks(audio_path: str, output_dir: str) -> dict[str, str]`
**Consumes:** `config.DEMUCS_MODEL`
**Raises:** `RuntimeError` (demucs failure is fatal)

- [ ] **Step 1: 实现 separate_tracks**

```python
# src/source_separate.py
import logging
import os
import subprocess
import sys

from src.config import config

logger = logging.getLogger(__name__)

# htdemucs_6s stem name -> internal instrument name
TRACK_MAP = {
    "bass": "bass",
    "drums": "drums",
    "vocals": "vocals",
    "guitar": "guitar",
    "piano": "piano",
}


def separate_tracks(audio_path: str, output_dir: str) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", config.DEMUCS_MODEL,
        "-o", output_dir,
        audio_path,
    ]

    logger.info(f"Running demucs: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = f"demucs failed (rc={result.returncode}): {result.stderr[-500:]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    demucs_out = os.path.join(output_dir, config.DEMUCS_MODEL, basename)

    tracks = {}
    for stem_name, internal_name in TRACK_MAP.items():
        wav_path = os.path.join(demucs_out, f"{stem_name}.wav")
        if os.path.exists(wav_path):
            tracks[internal_name] = wav_path
            logger.info(f"  {internal_name}: {wav_path}")
        else:
            logger.warning(f"  {internal_name}: MISSING")

    if len(tracks) < 3:
        raise RuntimeError(f"demucs output incomplete: {list(tracks.keys())}")

    return tracks
```

- [ ] **Step 2: 验证 demucs 可用**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "import demucs; print('demucs OK')"
```

If `ModuleNotFoundError`, switch to the conda env that has demucs installed (likely the source_separation env).

### Task 4: 音高识别 — pitch_detect.py

**Files:** Create `src/pitch_detect.py`
**Produces:**
- `detect_pitch_mono(wav_path: str) -> list[dict]` — dict keys: onset, offset, pitch, confidence, amplitude
- `detect_pitch_piano(wav_path: str) -> list[dict]` — same, with basic-pitch multi-note support

- [ ] **Step 1: 生成 A4=440Hz 测试音频**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
import numpy as np; import soundfile as sf
sr = 44100; t = np.arange(3 * sr) / sr
a4 = 0.8 * np.sin(2 * np.pi * 440 * t)
sf.write('test/fixtures/test_a4_440.wav', a4, sr)
print('A4 fixture generated')
"
```

- [ ] **Step 2: 实现 pitch_detect.py**

```python
# src/pitch_detect.py
import logging
import warnings

import numpy as np
import torch
import torchcrepe
import librosa

from src.config import config

logger = logging.getLogger(__name__)


def detect_pitch_mono(wav_path: str) -> list[dict]:
    try:
        audio, sr = librosa.load(wav_path, sr=config.SR, mono=True)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pitch, periodicity = torchcrepe.predict(
                audio_tensor, sr,
                hop_length=config.HOP_LENGTH,
                fmin=50, fmax=2000,
                model="full", batch_size=1024,
                device=device,
                return_periodicity=True,
            )

        pitch = pitch.squeeze().cpu().numpy()
        confidence = periodicity.squeeze().cpu().numpy()
        frame_time = config.HOP_LENGTH / sr

        notes = _frames_to_notes(pitch, confidence, frame_time, audio)

        if len(notes) == 0:
            logger.warning(f"No notes detected: {wav_path}")
        return notes

    except Exception as e:
        logger.warning(f"Mono pitch detection failed ({wav_path}): {e}")
        return []


def detect_pitch_piano(wav_path: str) -> list[dict]:
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        _, _, note_events = predict(wav_path, model_or_model_path=ICASSP_2022_MODEL_PATH)

        notes = []
        for e in note_events:
            notes.append({
                "onset": float(e.start_time),
                "offset": float(e.end_time),
                "pitch": int(e.pitch),
                "velocity": int(e.velocity),
                "confidence": float(getattr(e, "confidence", 1.0)),
            })

        logger.info(f"basic-pitch: {len(notes)} notes (piano)")
        return notes

    except ImportError:
        logger.warning("basic-pitch not installed, piano -> crepe mono")
        return detect_pitch_mono(wav_path)
    except Exception as e:
        logger.warning(f"basic-pitch failed ({e}), piano -> crepe mono")
        return detect_pitch_mono(wav_path)


def _frames_to_notes(
    pitch: np.ndarray, confidence: np.ndarray,
    frame_time: float, audio: np.ndarray
) -> list[dict]:
    MIN_CONF = 0.3
    notes = []
    i = 0
    n = len(pitch)

    while i < n:
        if confidence[i] < MIN_CONF or np.isnan(pitch[i]):
            i += 1
            continue

        start_frame = i
        start_pitch = pitch[i]

        while (
            i < n
            and confidence[i] >= MIN_CONF
            and not np.isnan(pitch[i])
            and abs(pitch[i] - start_pitch) < config.SLIDE_PITCH_THRESHOLD
        ):
            i += 1

        end_frame = i

        start_sample = int(start_frame * frame_time * config.SR)
        end_sample = int(end_frame * frame_time * config.SR)
        if end_sample > start_sample and start_sample < len(audio):
            amp = float(np.sqrt(np.mean(audio[start_sample:end_sample] ** 2)))
        else:
            amp = 0.1

        notes.append({
            "onset": round(start_frame * frame_time, 3),
            "offset": round(end_frame * frame_time, 3),
            "pitch": int(round(start_pitch)),
            "confidence": float(np.mean(confidence[start_frame:end_frame])),
            "amplitude": amp,
        })

    return notes
```

- [ ] **Step 3: 验证 A4 识别**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
from src.pitch_detect import detect_pitch_mono
notes = detect_pitch_mono('test/fixtures/test_a4_440.wav')
print(f'Notes detected: {len(notes)}')
for n in notes[:5]:
    print(f'  pitch={n["pitch"]} onset={n["onset"]}')
has_a4 = any(68 <= n['pitch'] <= 70 for n in notes)
assert has_a4, 'A4 (MIDI 69) not in results!'
print('PASS')
"
```

### Task 5: 和弦识别 — chord_detect.py

**Files:** Create `src/chord_detect.py`
**Produces:** `detect_chords(wav_path: str, bpm: float) -> list[dict]`
Each dict: `{"start": float, "end": float, "label": str}` (e.g. "C", "Am")

- [ ] **Step 1: 实现 detect_chords**

```python
# src/chord_detect.py
import logging

logger = logging.getLogger(__name__)


def detect_chords(wav_path: str, bpm: float) -> list[dict]:
    try:
        import madmom
        from madmom.features.chords import (
            CNNChordFeatureProcessor,
            CRFChordRecognitionProcessor,
        )

        feat_processor = CNNChordFeatureProcessor()
        chord_processor = CRFChordRecognitionProcessor()

        features = feat_processor(wav_path)
        chord_labels = chord_processor(features)

        chords = []
        for start, end, label in chord_labels:
            if label != "N":
                chords.append({
                    "start": round(float(start), 3),
                    "end": round(float(end), 3),
                    "label": str(label),
                })

        logger.info(f"Chords detected: {len(chords)} labels")
        return chords

    except ImportError:
        logger.warning("madmom not installed, skipping chord detection")
        return []
    except Exception as e:
        logger.warning(f"Chord detection failed: {e}")
        return []
```

- [ ] **Step 2: 验证导入和 fallback**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
from src.chord_detect import detect_chords
result = detect_chords('test/fixtures/test_a4_440.wav', 120.0)
assert isinstance(result, list), 'must return list'
print(f'chord_detect OK (returned {len(result)} chords)')
"
```

### Task 6: 调号推算 — key_estimate.py

**Files:** Create `src/key_estimate.py`
**Produces:** `estimate_key(all_notes: list[dict]) -> str`

Krumhansl-Schmuckler algorithm: accumulate pitch-class durations, correlate with major/minor profiles.
Pure algorithm, no external deps — easily unit-testable.

- [ ] **Step 1: 实现**

```python
# src/key_estimate.py
import logging
import numpy as np

logger = logging.getLogger(__name__)

# K-S key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _notes_to_pitch_class_vector(notes):
    vec = np.zeros(12)
    for n in notes:
        pc = n["pitch"] % 12
        duration = n["offset"] - n["onset"]
        vec[pc] += duration
    total = vec.sum()
    return vec / total if total > 0 else vec


def estimate_key(all_notes):
    if len(all_notes) == 0:
        return "C major"

    vec = _notes_to_pitch_class_vector(all_notes)
    best_corr, best_key = -999, "C major"

    for tonic in range(12):
        corr = np.corrcoef(vec, np.roll(MAJOR_PROFILE, tonic))[0, 1]
        if corr > best_corr:
            best_corr, best_key = corr, f"{PITCH_NAMES[tonic]} major"

    for tonic in range(12):
        corr = np.corrcoef(vec, np.roll(MINOR_PROFILE, tonic))[0, 1]
        if corr > best_corr:
            best_corr, best_key = corr, f"{PITCH_NAMES[tonic]} minor"

    logger.info(f"Estimated key: {best_key} (corr={best_corr:.3f})")
    return best_key
```

- [ ] **Step 2: 验证**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
from src.key_estimate import estimate_key
# C major scale notes (heavily weighted C, E, G)
notes = [
    {'onset':0,'offset':1.0,'pitch':60},{'onset':1,'offset':2.0,'pitch':64},
    {'onset':2,'offset':3.0,'pitch':67},{'onset':3,'offset':4.0,'pitch':60},
    {'onset':4,'offset':5.0,'pitch':62},{'onset':5,'offset':6.0,'pitch':62},
]
key = estimate_key(notes)
print(f'Estimated: {key}')
assert 'C' in key and 'major' in key, f'Expected C major, got {key}'
print('PASS')
"
```

### Task 7: 吉他六线谱排指 DP

**Files:** Create `src/guitar_tab.py`
**Produces:** `assign_guitar_fingering(notes: list[dict]) -> list[dict]`
Output dicts add keys: `"string"` (int, 1-6), `"fret"` (int, 0-22)

Standard tuning MIDI: E2=40, A2=45, D3=50, G3=55, B3=59, E4=64 (6th..1st)

- [ ] **Step 1: 实现**

```python
# src/guitar_tab.py
import logging
from src.config import config

logger = logging.getLogger(__name__)

OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]  # low E -> high E


def get_positions(midi_pitch: int) -> list[tuple[int, int]]:
    positions = []
    for s, open_midi in enumerate(OPEN_STRING_MIDI):
        fret = midi_pitch - open_midi
        if 0 <= fret <= config.MAX_FRET:
            positions.append((s, fret))
    return positions


def assign_guitar_fingering(notes: list[dict]) -> list[dict]:
    if len(notes) == 0:
        return []

    candidates = []
    for note in notes:
        pos = get_positions(note["pitch"])
        if not pos:
            pos = [(0, -1)]  # sentinel for unplayable
        candidates.append(pos)

    n = len(notes)
    dp = []
    for i in range(n):
        dp_i = []
        for j, (s, f) in enumerate(candidates[i]):
            if i == 0:
                cost = config.OPEN_STRING_BIAS if f == 0 else 0.0
                dp_i.append((cost, -1))
            else:
                best_cost, best_prev = float("inf"), -1
                for k, (ps, pf) in enumerate(candidates[i - 1]):
                    if pf == -1 or f == -1:
                        step_cost = 5.0
                    else:
                        step_cost = (
                            config.FRET_WEIGHT * abs(f - pf)
                            + config.STRING_WEIGHT * abs(s - ps)
                        )
                        if f == 0:
                            step_cost += config.OPEN_STRING_BIAS
                    total = dp[i - 1][k][0] + step_cost
                    if total < best_cost:
                        best_cost, best_prev = total, k
                dp_i.append((best_cost, best_prev))
        dp.append(dp_i)

    # Backtrack
    result = []
    best_last = min(range(len(dp[-1])), key=lambda j: dp[-1][j][0])
    for i in range(n - 1, -1, -1):
        s, f = candidates[i][best_last]
        note_copy = dict(notes[i])
        note_copy["string"] = int(s) + 1
        note_copy["fret"] = int(f)
        result.append(note_copy)
        _, best_last = dp[i][best_last] if i > 0 else (0, -1)

    result.reverse()
    return result
```

- [ ] **Step 2: 验证**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
from src.guitar_tab import assign_guitar_fingering, get_positions
pos = get_positions(64)
assert len(pos) >= 2
notes = [{'onset':0,'offset':1,'pitch':60},{'onset':1,'offset':2,'pitch':62},{'onset':2,'offset':3,'pitch':64},{'onset':3,'offset':4,'pitch':65}]
r = assign_guitar_fingering(notes)
assert len(r) == 4 and all('string' in n for n in r)
print('PASS')
"
```

---

### Task 8: 乐谱组装

**Files:** Create `src/score_assemble.py`
**Produces:** `assemble_score(instrument_name, notes, bpm, key_sig, time_sig, chords, is_guitar) -> music21.stream.Score`

- [ ] **Step 1: 实现**

```python
# src/score_assemble.py
import logging
from music21 import stream, meter, key, tempo, dynamics, note, articulations

logger = logging.getLogger(__name__)

TEMPO_TERMS = [
    (40,"Grave"),(60,"Largo"),(66,"Adagio"),(76,"Andante"),
    (108,"Moderato"),(120,"Allegro"),(156,"Vivace"),(176,"Presto"),(999,"Prestissimo"),
]
DYNAMIC_MAP = [
    (0.01,"pp"),(0.03,"p"),(0.06,"mp"),(0.12,"mf"),(0.25,"f"),(0.5,"ff"),
]


def _bpm_to_tempo_term(bpm):
    for t, term in TEMPO_TERMS:
        if bpm < t: return term
    return "Allegro"


def _amp_to_dynamic(amp):
    for t, dyn in DYNAMIC_MAP:
        if amp < t: return dyn
    return "ff"


def _detect_dynamics(notes):
    for i in range(1, len(notes)):
        pa = notes[i-1].get("amplitude",0.1)
        ca = notes[i].get("amplitude",0.1)
        if ca > pa*1.3: notes[i]["crescendo"] = True
        elif pa > ca*1.3: notes[i]["diminuendo"] = True
    return notes


def _detect_slides(notes):
    for i in range(1, len(notes)):
        gap = notes[i]["onset"] - notes[i-1]["offset"]
        pdiff = abs(notes[i]["pitch"] - notes[i-1]["pitch"])
        if gap < 0.1 and 0.5 < pdiff < 12:
            notes[i]["slide_from"] = notes[i-1]["pitch"]
    return notes


def assemble_score(instrument_name, notes, bpm, key_signature,
                   time_signature="4/4", chords=None, is_guitar=False):
    s = stream.Score()
    part = stream.Part()
    part.partName = instrument_name
    part.append(tempo.MetronomeMark(number=int(bpm)))
    part.append(meter.TimeSignature(time_signature))
    part.append(key.Key(key_signature))

    notes = _detect_dynamics(notes)
    notes = _detect_slides(notes)

    for n in notes:
        if n.get("fret") == -1:
            r = note.Rest()
            r.duration.quarterLength = max((n["offset"]-n["onset"])*bpm/60.0, 0.25)
            part.append(r)
            continue

        n_obj = note.Note(n["pitch"])
        n_obj.duration.quarterLength = max((n["offset"]-n["onset"])*bpm/60.0, 0.25)

        if not is_guitar:
            dyn = _amp_to_dynamic(n.get("amplitude", 0.1))
            n_obj.articulations.append(dynamics.Dynamic(dyn))
            if n.get("crescendo"):
                n_obj.articulations.append(dynamics.Crescendo())
            if n.get("diminuendo"):
                n_obj.articulations.append(dynamics.Diminuendo())
            if n.get("slide_from"):
                n_obj.articulations.append(articulations.Glissando())

        part.append(n_obj)

    s.insert(0, part)
    logger.info(f"Score assembled: {instrument_name} ({len(notes)} notes)")
    return s
```

- [ ] **Step 2: 验证**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
from src.score_assemble import assemble_score
notes=[{'onset':0,'offset':1,'pitch':60,'amplitude':0.15},
       {'onset':1,'offset':2,'pitch':64,'amplitude':0.20},
       {'onset':2,'offset':3,'pitch':67,'amplitude':0.10}]
score=assemble_score('piano',notes,120,'C major')
assert score is not None and len(score.parts)>0
print('PASS')
"
```

---

### Task 9: 乐谱导出

**Files:** Create `src/export_score.py`
**Produces:** `export_score(score, output_path_stem) -> str` (returns .musicxml path)

- [ ] **Step 1: 实现**

```python
# src/export_score.py
import logging, os
from music21 import musicxml

logger = logging.getLogger(__name__)


def export_score(score, output_path_stem: str) -> str:
    os.makedirs(os.path.dirname(output_path_stem) or ".", exist_ok=True)
    xml_path = output_path_stem + ".musicxml"
    score.write("musicxml", fp=xml_path)
    logger.info(f"Exported: {xml_path}")
    return xml_path
```

- [ ] **Step 2: 验证**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
from src.score_assemble import assemble_score
from src.export_score import export_score
import music21, os
notes=[{'onset':0,'offset':1,'pitch':60,'amplitude':0.1},{'onset':1,'offset':2,'pitch':62,'amplitude':0.1}]
score=assemble_score('test',notes,120,'C major')
path=export_score(score,'output/test_unit')
assert os.path.exists(path)
# Verify it's valid
loaded=music21.converter.parse(path)
assert len(loaded.parts)>0
print('PASS')
os.remove(path)
"
```

---

### Task 10: 主 Pipeline

**Files:** Create `src/pipeline.py`
**Produces:** `run_pipeline(audio_path, output_dir=None) -> str` (returns output folder path)

- [ ] **Step 1: 实现**

```python
# src/pipeline.py
import logging, os, json, tempfile
from src.config import config
from src.bpm_detect import detect_bpm
from src.source_separate import separate_tracks
from src.pitch_detect import detect_pitch_mono, detect_pitch_piano
from src.chord_detect import detect_chords
from src.key_estimate import estimate_key
from src.guitar_tab import assign_guitar_fingering
from src.score_assemble import assemble_score
from src.export_score import export_score

logger = logging.getLogger(__name__)


def run_pipeline(audio_path: str, output_dir: str | None = None) -> str:
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, song_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                                  logging.StreamHandler()])

    logger.info(f"=== Pipeline start: {song_name} ===")

    # 1. BPM
    bpm = detect_bpm(audio_path) or config.DEFAULT_BPM
    logger.info(f"[1/6] BPM: {bpm}")

    # 2. Source separation
    logger.info("[2/6] Source separation (htdemucs_6s)...")
    tracks = separate_tracks(audio_path, output_dir)

    # 3. Pitch detection per track
    logger.info("[3/6] Pitch detection...")
    all_notes = {}
    for inst_name, wav_path in tracks.items():
        if inst_name == "drums":
            all_notes[inst_name] = []  # skip drums
            continue
        elif inst_name == "piano":
            notes = detect_pitch_piano(wav_path)
        else:
            notes = detect_pitch_mono(wav_path)
        all_notes[inst_name] = notes
        logger.info(f"  {inst_name}: {len(notes)} notes")

    # 4. Guitar chords (best-effort)
    chords = []
    if "guitar" in tracks:
        logger.info("[4/6] Chord detection (guitar)...")
        chords = detect_chords(tracks["guitar"], bpm)

    # 5. Key estimation
    logger.info("[5/6] Key estimation...")
    combined_notes = []
    for notes in all_notes.values():
        combined_notes.extend(notes)
    key_sig = estimate_key(combined_notes)
    logger.info(f"  Key: {key_sig}")

    # 6. Score assembly + export
    logger.info("[6/6] Score assembly + export...")
    for inst_name, notes in all_notes.items():
        if inst_name == "drums":
            continue  # drum score requires different logic; skip for VER1.0
        if len(notes) == 0:
            logger.warning(f"  {inst_name}: no notes, skipping score")
            continue

        is_guitar = (inst_name == "guitar")
        if is_guitar:
            notes = assign_guitar_fingering(notes)
            logger.info(f"  {inst_name}: guitar fingering assigned")

        score = assemble_score(
            inst_name, notes, bpm, key_sig,
            config.DEFAULT_TIME_SIG, chords if is_guitar else None,
            is_guitar,
        )

        output_stem = os.path.join(output_dir, inst_name)
        export_score(score, output_stem)

    # Write info.json
    info = {
        "song": song_name,
        "bpm": bpm,
        "key": key_sig,
        "time_signature": config.DEFAULT_TIME_SIG,
        "tracks": {
            name: len(notes) for name, notes in all_notes.items()
        },
        "chord_count": len(chords),
    }
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    logger.info(f"=== Pipeline complete: {output_dir} ===")
    return output_dir
```

- [ ] **Step 2: 验证导入**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
from src.pipeline import run_pipeline
print('pipeline import OK')
"
```

---

### Task 11: 端到端测试

**Files:** Create `test/test_pipeline.py`
**Requires:** test fixtures (`test_a4_440.wav`, `test_120bpm.wav`)

- [ ] **Step 1: 生成短音频 fixture (含多个乐器模拟)**

```bash
cd d:/program_project/MUSE/score_extraction && python -c "
import numpy as np; import soundfile as sf
sr=44100; dur=15.0; t=np.arange(int(sr*dur))/sr
# Simple multi-instrument mix: bass line + melody + chords
bass=0.15*np.sin(2*np.pi*110*t)  # A2
melody=0.2*np.sin(2*np.pi*440*(t%0.5+1)*np.floor(t/0.5)%12*50+440)
chords=0.1*(np.sin(2*np.pi*261*t)+np.sin(2*np.pi*329*t)+np.sin(2*np.pi*392*t))
mix=bass+melody+chords
sf.write('test/fixtures/test_mix_15s.wav', mix/mix.max()*0.9, sr)
print('multi-instrument fixture generated')
"
```

- [ ] **Step 2: 编写测试**

```python
# test/test_pipeline.py
# Unit + integration tests for score_extraction pipeline.
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestBPM(unittest.TestCase):
    def test_120bpm(self):
        from src.bpm_detect import detect_bpm
        bpm = detect_bpm(os.path.join(FIXTURES, "test_120bpm.wav"))
        self.assertIsNotNone(bpm)
        self.assertAlmostEqual(bpm, 120, delta=3)


class TestPitch(unittest.TestCase):
    def test_a4_detected(self):
        from src.pitch_detect import detect_pitch_mono
        notes = detect_pitch_mono(os.path.join(FIXTURES, "test_a4_440.wav"))
        # At least one note with pitch near A4 (MIDI 69)
        self.assertTrue(any(68 <= n["pitch"] <= 70 for n in notes),
                        "A4 not detected in 440Hz sine wave")


class TestKeyEstimate(unittest.TestCase):
    def test_c_major_scale(self):
        from src.key_estimate import estimate_key
        notes = []
        for p in [60, 62, 64, 65, 67, 69, 71, 72]:
            notes.append({"onset": 0, "offset": 1.0, "pitch": p})
        key = estimate_key(notes * 3)  # repeat for stronger signal
        self.assertIn("C", key)
        self.assertIn("major", key)


class TestMusicXMLRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        import music21, tempfile
        from src.score_assemble import assemble_score
        from src.export_score import export_score

        notes = [
            {"onset": 0, "offset": 1, "pitch": 60, "amplitude": 0.1},
            {"onset": 1, "offset": 2, "pitch": 64, "amplitude": 0.1},
        ]
        score = assemble_score("test", notes, 120, "C major")

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            path = f.name
        try:
            export_score(score, path.replace(".musicxml", ""))
            loaded = music21.converter.parse(path)
            self.assertGreater(len(loaded.parts), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestGuitarFingering(unittest.TestCase):
    def test_c_major_scale(self):
        from src.guitar_tab import assign_guitar_fingering
        notes = [{"onset":i,"offset":i+1,"pitch":p}
                 for i, p in enumerate([60,62,64,65,67,69,71,72])]
        result = assign_guitar_fingering(notes)
        self.assertEqual(len(result), 8)
        for n in result:
            self.assertIn("string", n)
            self.assertIn("fret", n)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试**

```bash
cd d:/program_project/MUSE/score_extraction && python -m pytest test/test_pipeline.py -v
```

Expected: all tests pass (or skip gracefully if GPU/models unavailable).
