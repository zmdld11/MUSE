"""LRC 歌词解析与字级对齐（人声专项 v2，2026-08-30）。

歌词是增强层不是门槛：本模块全部产物可选，无 LRC 时管线照常出纯旋律谱。

parse_lrc 处理的现实坑（music/vocal 实测）：
- 编码混杂：UTF-8 BOM / UTF-8 / GBK（不夜城）——字节级嗅探；
- NDJSON 元数据前缀行（Everlasting / Into the Sky 的 {"t":ms,"c":[...]}）；
- 空文本时间戳行（间奏/鸣谢边界，保留在时间轴上但不作词行）；
- 制作人员行（作词/作曲/…）：关键词识别，作词行之外的"非歌词"；
- 占位歌词（花海「暂无歌词」）→ 视为无歌词；
- [offset:±ms] 全局时间平移（正值提前）。

tokenize：CJK/假名一字一 token，拉丁字母连串一词一 token（英文歌词按
词对齐，不拆音节），标点忽略。
"""
from __future__ import annotations

import re
from pathlib import Path

# 时间戳：[mm:ss] / [mm:ss.xx] / [mm:ss.xxx]（一行可多个）
_TS_RE = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_TAG_RE = re.compile(r"^\[([a-zA-Z#]+):([^\]]*)\]\s*$")
# 一行整行只有一个标签（含多时间戳的歌词行不匹配 _TAG_RE）
_CJK_RE = re.compile(
    r"[\u2E80-\u9FFF\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F\uAC00-\uD7AF]"
    r"|[A-Za-z]+(?:'[A-Za-z]+)?|\d+"
)

# 制作人员关键词（词行里罕见，鸣谢行里常见）
_CREDIT_KEYS = (
    "作词", "作曲", "编曲", "混音", "母带", "演唱", "作词作曲", "词曲",
    "监制", "出品", "制作人", "和声", "录音", "调音", "调教", "视频", "插画",
    "封面", "翻译", "发行", "贝斯", "鼓手", "弦乐", "吉他", "钢琴", "策划",
    "特别感谢", "感谢", "歌名", "素材", "字幕", "压制",
)
# 2026-08-30 VII 审计补充：官方 LRC 的鸣谢行变体——©/＠/※ 前缀（
# '©Lyric Present' 曾整行 12 字静音空走）与「词\曲\唱」反斜杠分隔（
# 全词 "作词" 匹配不上）
_CREDIT_PREFIX = ("©", "＠", "@", "※")
_CREDIT_SEP_RE = re.compile(r"^[词曲唱编混录调演奏和]{1,4}\s*[/\\、,，]\s*[词曲唱编混录调演奏和、,，\s：:]+")
_PLACEHOLDER = {"暂无歌词", "纯音乐", "无歌词", "请欣赏", "暂无歌词。", "纯音乐，请欣赏"}


def _decode(raw: bytes) -> tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    for enc in ("utf-8", "gbk", "big5", "shift_jis"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8?replace"


def _parse_ts(token: str) -> float | None:
    m = _TS_RE.fullmatch(token)
    if not m:
        return None
    minutes, seconds, frac = m.group(1), m.group(2), m.group(3)
    t = int(minutes) * 60 + int(seconds)
    if frac:
        t += int(frac) / (10 ** len(frac))
    return float(t)


def tokenize(text: str) -> list[str]:
    return _CJK_RE.findall(text)


def parse_lrc(path: str, audio_duration: float | None = None) -> dict | None:
    """LRC 文件 → 时间轴结构。无有效歌词（缺文件/占位）返回 None。

    返回 {
      "entries":  [{t0, t1, text, kind}],   # 全时间轴（含空行/鸣谢行）
      "lyric_lines": [{t0, t1, text, tokens}],  # kind=="lyric"
      "source_file", "encoding", "offset_applied",
    }
    t1 = 下一时间戳；末行 = min(t0+8s, 音频尾)。
    """
    try:
        raw = open(path, "rb").read()
    except OSError:
        return None
    text_all, encoding = _decode(raw)

    offset_ms = 0.0
    pending: list[tuple[float, str]] = []  # (t0, text) 待定序
    for line in text_all.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            continue  # NDJSON 元数据前缀（Apple Music 式）
        stamps = _TS_RE.findall(line)
        if not stamps:
            m = _TAG_RE.match(line)
            if m and m.group(1).lower() == "offset":
                try:
                    offset_ms = float(m.group(2).strip().replace("+", ""))
                except ValueError:
                    pass
            continue
        body = _TS_RE.sub("", line).strip()
        for stamp in stamps:
            t = _parse_ts(f"[{stamp[0]}:{stamp[1]}{'.' + stamp[2] if stamp[2] else ''}]")
            if t is not None:
                pending.append((t, body))
    if offset_ms:
        pending = [(t - offset_ms / 1000.0, body) for t, body in pending]
    pending.sort(key=lambda x: x[0])
    if not pending:
        return None

    entries: list[dict] = []
    for i, (t0, body) in enumerate(pending):
        t1 = pending[i + 1][0] if i + 1 < len(pending) else t0 + 8.0
        if audio_duration is not None:
            t1 = min(t1, audio_duration)
        if not body:
            kind = "empty"
        elif any(k in body for k in _CREDIT_KEYS) \
                or body.startswith(_CREDIT_PREFIX) \
                or _CREDIT_SEP_RE.match(body):
            kind = "credit"
        else:
            kind = "lyric"
        entries.append({"t0": round(t0, 3), "t1": round(t1, 3),
                        "text": body, "kind": kind})

    lyric_lines = [e for e in entries if e["kind"] == "lyric"]
    texts = {l["text"] for l in lyric_lines}
    if not lyric_lines or texts <= _PLACEHOLDER:
        return None
    for l in lyric_lines:
        l["tokens"] = tokenize(l["text"])
    return {
        "entries": entries,
        "lyric_lines": lyric_lines,
        "source_file": path,
        "encoding": encoding,
        "offset_applied": offset_ms,
    }


# ---------------------------- 检测层增强（B / P1+P2） ----------------------------

_BREATH_MAX_DUR = 0.30   # 呼吸音候选最长时值
_BREATH_PITCH_UP = 7     # 高出邻域中位的半音数（气声高频假音）
_TRIM_DB = -13           # offset 截断的衰减门（相对音符窗内 RMS 峰）
_TRIM_TOL = 0.12         # 截断容差：offset 超出衰减点+此值才收
_TRIM_MIN_DUR = 0.08     # 截断下限（保底时值）


def filter_breath_notes(notes: list[dict], lyric_lines: list[dict],
                        stem_wav: str) -> tuple[list[dict], list[dict]]:
    """行间隙呼吸音过滤（只动歌词行外的音；音头铁律不涉——行内音不动）。

    四条件全满足才删：完全落在歌词行外 ∧ 时值 <0.3s ∧ 音高 ≥ 全曲中位
    +7（气声被推成高频假音）∧ stem 起音能量 < 有声段 90 分位的 15%。
    返回 (保留, 已删清单)——删数进 notes.json 审计。
    """
    import librosa
    import numpy as np
    import statistics
    if not notes or not lyric_lines:
        return notes, []
    y, sr = librosa.load(stem_wav, sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=220)[0]  # 10ms
    voiced_level = float(np.quantile(rms, 0.90))
    med_pitch = statistics.median(n["pitch"] for n in notes)
    kept, removed = [], []
    for n in notes:
        inside = any(l["t0"] - 0.10 <= n["onset"] <= l["t1"] + 0.10
                     for l in lyric_lines)
        dur = n["offset"] - n["onset"]
        if inside or dur >= _BREATH_MAX_DUR or n["pitch"] < med_pitch + _BREATH_PITCH_UP:
            kept.append(n)
            continue
        i0 = int(n["onset"] / 0.01)
        i1 = min(i0 + 20, len(rms))
        level = float(np.mean(rms[i0:i1])) if i1 > i0 else 0.0
        if level >= 0.15 * voiced_level:
            kept.append(n)
            continue
        removed.append({"onset": n["onset"], "offset": n["offset"],
                        "pitch": n["pitch"]})
    return kept, removed


def trim_vocal_offsets(notes: list[dict], stem_wav: str) -> tuple[list[dict], int]:
    """offset 能量截断（P1-γ，2026-08-30 用户"时值拖沓"）：SOME 长音
    offset 系统性拖尾（夏日实测中位 +46ms 但 p90 +455ms、17% 音拖>100ms），
    记谱层 fill-to-next 再放大。修法 = offset 超出窗内 RMS 衰减点
    （峰下 _TRIM_DB，窗不越下一音头防吃字）+ 容差才收。只截尾不动头。

    无歌词依赖，所有人声曲生效。返回 (notes, 截断数)。
    """
    import librosa
    import numpy as np
    if not notes:
        return notes, 0
    y, sr = librosa.load(stem_wav, sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=220)[0]
    out, n_trim = [], 0
    for i, n in enumerate(notes):
        nxt = notes[i + 1]["onset"] if i + 1 < len(notes) else n["offset"] + 1.0
        win_end = min(n["offset"] + 0.6, nxt - 0.02)
        if win_end - n["onset"] >= 0.15:
            i0, i1 = int(n["onset"] / 0.01), int(win_end / 0.01)
            seg = rms[i0:i1]
            peak = float(seg.max())
            if peak > 1e-4:
                above = np.where(seg > peak * 10 ** (_TRIM_DB / 20))[0]
                decay_end = (i0 + int(above[-1])) * 0.01 if len(above) \
                    else n["onset"]
                if n["offset"] > decay_end + _TRIM_TOL:
                    n = {**n, "offset": round(max(decay_end,
                                                  n["onset"] + _TRIM_MIN_DUR), 4)}
                    n_trim += 1
        out.append(n)
    return out, n_trim


_F0_CACHE: dict[str, tuple] = {}


def _tcrepe_f0(stem_wav: str):
    """torchcrepe tiny 帧 F0（缓存 per stem）。返回 (times, f0) numpy。"""
    import torch
    import torchcrepe
    import soundfile as sf
    import librosa
    import numpy as np
    if stem_wav in _F0_CACHE:
        return _F0_CACHE[stem_wav]
    y, sr0 = sf.read(stem_wav, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr0 != 16000:
        y = librosa.resample(y, orig_sr=sr0, target_sr=16000)
    audio = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hop = 256
    f0 = torchcrepe.predict(audio, 16000, hop, 65.0, 1050.0, "tiny",
                            device=torch.device(device), batch_size=1024,
                            return_periodicity=False)
    f0 = f0.reshape(-1).cpu().numpy()
    times = np.arange(len(f0)) * hop / 16000
    _F0_CACHE[stem_wav] = (times, f0)
    return times, f0


def _win_median_pitch(times, f0, t0: float, t1: float,
                      fallback: int | None = None) -> int | None:
    """窗内 voiced F0 中位 → MIDI 取整（音域门 [40,84]）。"""
    import numpy as np
    m = (times >= t0) & (times < t1)
    seg = f0[m]
    seg = seg[(seg > 50) & (seg < 1000)]
    if len(seg) < 4:
        return fallback
    midi = 69.0 + 12.0 * np.log2(np.median(seg) / 440.0)
    p = int(round(midi))
    return p if 40 <= p <= 84 else fallback


def fill_missing_syllables(notes: list[dict], chars: list[dict],
                           stem_wav: str) -> tuple[list[dict], int]:
    """行内补漏 v2（P1-β，2026-08-30 用户"一字一音"先验）：每字必有音。

    v1（tcrepe 音符表找窗内音）实测 1/3 事件仍一音多字——咬字轻的字
    tcrepe 也不给独立音头。v2 = 字窗内无 SOME 音头时**直接合成**：
    onset=字起音（能量峰），offset=字终，pitch=窗内 voiced F0 中位
    （fallback=前一音）。窗下限 0.08s（v1 是 0.18s，快语速行字字清楚）。
    只加不删，补插音 confidence 0.6。
    """
    times, f0 = _tcrepe_f0(stem_wav)
    inserted = []
    prev_pitch = None
    for i, ch in enumerate(chars):
        nxt = chars[i + 1] if i + 1 < len(chars) else None
        if nxt is None or nxt["line_idx"] != ch["line_idx"]:
            w1 = ch["end"]
        else:
            w1 = nxt["onset"] - 0.02
        w0 = ch["onset"] - 0.04
        if w1 - w0 < 0.08:
            continue
        if any(w0 <= n["onset"] < w1 for n in notes):
            prev_pitch = next((n["pitch"] for n in notes
                               if n["onset"] <= w0 + 0.04), prev_pitch)
            continue
        pitch = _win_median_pitch(times, f0, w0, w1, fallback=prev_pitch)
        if pitch is None:
            continue
        inserted.append({
            "onset": round(max(ch["onset"] - 0.02, w0), 4),
            "offset": round(w1, 4),
            "pitch": pitch, "velocity": 100,
            "confidence": 0.6, "instrument_class": "melody",
        })
        prev_pitch = pitch
    if not inserted:
        return notes, 0
    return sorted(notes + inserted,
                  key=lambda n: (n["onset"], n["pitch"])), len(inserted)


def split_melisma(notes: list[dict], chars: list[dict],
                  stem_wav: str) -> tuple[list[dict], int]:
    """拖腔拆音（P1-γ）：一字多音还原。字窗内恰 1 个 SOME 长音（≥0.4s）
    且帧 F0 有两个稳定段（各 ≥0.2s，中位差 ≥2 半音）→ 按段界拆两音
    （首段保原 onset，pitch=各段 F0 中位；转音=真一字多音）。

    只对"单音覆盖整字"的拖腔拆——谱面挂连音线（记谱层 tie 语义），
    合成段 pitch confidence 0.6。返回 (notes, 拆分数)。
    """
    import numpy as np
    times, f0 = _tcrepe_f0(stem_wav)
    out, n_split = list(notes), 0
    for ch in chars:
        w0, w1 = ch["onset"] - 0.03, ch["end"]
        if w1 - w0 < 0.5:
            continue
        cover = [k for k, n in enumerate(out)
                 if n["onset"] >= w0 and n["onset"] < w1]
        if len(cover) != 1:
            continue
        k = cover[0]
        n = out[k]
        if n["offset"] - n["onset"] < 0.4:
            continue
        m = (times >= n["onset"] + 0.05) & (times < n["offset"] - 0.03)
        seg = f0[m]
        seg = seg[(seg > 50) & (seg < 1000)]
        if len(seg) < 30:
            continue
        midi = 69.0 + 12.0 * np.log2(seg / 440.0)
        half = len(midi) // 2
        if half < 8:
            continue
        h1_med, h2_med = float(np.median(midi[:half])), float(np.median(midi[half:]))
        if abs(h2_med - h1_med) < 2.0:
            continue
        # 两半各自内部稳定（段内 std < 0.7 半音）才算"两个稳定段"
        if np.std(midi[:half]) > 0.7 or np.std(midi[half:]) > 0.7:
            continue
        split_t = float(times[m][0] + (half / len(midi))
                        * (n["offset"] - 0.03 - n["onset"] - 0.05))
        p1 = int(round(h1_med)) if 40 <= round(h1_med) <= 84 else n["pitch"]
        p2 = int(round(h2_med)) if 40 <= round(h2_med) <= 84 else n["pitch"]
        out[k] = {**n, "offset": round(split_t, 4), "pitch": p1}
        out.append({"onset": round(split_t, 4), "offset": n["offset"],
                    "pitch": p2, "velocity": 100, "confidence": 0.6,
                    "instrument_class": "melody"})
        n_split += 1
    if n_split:
        out.sort(key=lambda n: (n["onset"], n["pitch"]))
    return out, n_split


def detect_ornaments(notes: list[dict], stem_wav: str) -> list[dict]:
    """技巧标注（P2）：帧 F0 形态 → 每音 ornament 字段。

    vibrato：≥0.35s 音符窗内 cent 轨迹（相对中位）平滑后 |c|>25 过零≥3
    次 ∧ 摆幅 60-350 cent（夏日实测 40% 长音符命中，幅度 70-340）。
    glissando_up/down：首/末 1/3 段中位差 ≥1 半音且中途无明显回摆
    （滑音进/出目标音；谱面呈现=拆音+连音线的音形化，此标记供审计）。
    """
    import numpy as np
    times, f0 = _tcrepe_f0(stem_wav)
    for n in notes:
        n.setdefault("ornament", None)
        dur = n["offset"] - n["onset"]
        if dur < 0.35:
            continue
        m = (times >= n["onset"] + 0.05) & (times < n["offset"] - 0.05)
        seg = f0[m]
        seg = seg[(seg > 50) & (seg < 1000)]
        if len(seg) < 30 or (len(seg) / max(m.sum(), 1)) < 0.5:
            continue
        cents = 1200.0 * np.log2(seg / np.median(seg))
        # 9 帧滑动平均为参考线，取"偏离线"的过零（高通——颤音是摆动分量，
        # 平滑信号本身过零会被 144ms 平滑窗抹平，2026-08-30 实测塌到 0）
        ker = np.ones(9) / 9
        sm = np.convolve(cents, ker, mode="same")
        dev_sign = np.sign(cents - sm)
        big_mask = np.abs(cents) > 25
        crossings = int(np.sum(np.abs(np.diff(dev_sign[big_mask])) > 0)) \
            if big_mask.sum() >= 4 else 0
        ptp = float(np.ptp(cents))
        if crossings >= 3 and 60 <= ptp <= 400:
            n["ornament"] = "vibrato"
            continue
        third = max(len(sm) // 3, 4)
        d = float(np.median(sm[-third:]) - np.median(sm[:third]))
        mid = sm[third:-third]
        if abs(d) >= 100 and (len(mid) == 0 or np.ptp(mid) < 100):
            n["ornament"] = ("glissando_up" if d > 0 else "glissando_down")
    return notes


# ---------------------------- CTC 字界精修（P3 + VII 多语化） ----------------------------

_W2V2_DIRS = {
    "zh": Path(__file__).resolve().parents[1] / "external" / "wav2vec2_zh_cn",
    "ja": Path(__file__).resolve().parents[1] / "external" / "w2v2_ja_xlsr",
    "en": Path(__file__).resolve().parents[1] / "external" / "w2v2_en_xlsr",
}
_CTC_MODELS: dict[str, tuple] = {}   # lang -> (processor, model)
_CTC_LOGP: dict[tuple, tuple] = {}   # (lang, wav) -> (logp, hop_sec, blank)
_KAKASI = None


def _token_units(lang: str, tok: str) -> list[str]:
    """token → CTC 目标单元序列。zh=字本身；en=小写字母（词→字母，
    词首时间=首字母）；ja=汉字转假名展开（pykakasi，无句上下文略有
    误转但强于不修）。"""
    global _KAKASI
    if lang == "zh":
        return [tok]
    if lang == "en":
        return [ch for ch in tok.lower() if ch.isalnum()]
    if _KAKASI is None:
        import pykakasi
        _KAKASI = pykakasi.kakasi()
    units: list[str] = []
    for item in _KAKASI.convert(tok):
        units += list(item.get("hira", ""))
    return units or [tok]


def _ctc_backend(lang: str, stem_wav: str):
    """(logp, hop_sec, blank_id) 全曲前向（按 lang+wav 缓存）。"""
    key = (lang, stem_wav)
    if key in _CTC_LOGP:
        return _CTC_LOGP[key]
    d = _W2V2_DIRS[lang]
    if not (d / "pytorch_model.bin").exists():
        return None
    import numpy as np
    import torch
    import librosa
    if lang not in _CTC_MODELS:
        from transformers import (AutoModelForCTC, AutoProcessor,
                                  Wav2Vec2CTCTokenizer,
                                  Wav2Vec2FeatureExtractor)
        try:
            proc = AutoProcessor.from_pretrained(str(d))
        except ImportError:
            # jonatasgrosman 系仓库带 language_model/ → AutoProcessor 要求
            # pyctcdecode；显式加载 CTC 组件绕过 LM（我们只做 forced_align）
            proc = Wav2Vec2FeatureExtractor.from_pretrained(str(d))
            proc.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(str(d))
        _CTC_MODELS[lang] = (proc, AutoModelForCTC.from_pretrained(str(d)))
    proc, model = _CTC_MODELS[lang]
    y, _ = librosa.load(stem_wav, sr=16000, mono=True)
    inputs = proc(np.asarray(y, dtype=np.float32), sampling_rate=16000,
                  return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    logp = torch.log_softmax(logits, dim=-1)[0]
    hop = y.size / logp.shape[0] / 16000.0
    _CTC_LOGP[key] = (logp, hop, model.config.pad_token_id)
    return _CTC_LOGP[key]


def refine_chars_ctc(chars: list[dict], lyric_lines: list[dict],
                     stem_wav: str) -> tuple[list[dict], int]:
    """逐行 CTC 强制对齐精修字起音（人声专项 v2 P3，2026-08-30）。

    评估口径（eval/vocal_fa_align.py，3 首泠鸢）：字-音头一致性中位
    energy 0.134/0.034/0.083s → CTC 0.058/0.037/0.059s——2/3 首显著更好。
    设计要点：
    - **逐行窗口** Viterbi（行间由 LRC 时间戳锚死）——全曲对齐在副歌
      重复段会整段滑移（不夜城实测中位漂 11.2s）；
    - 行内 OOV 字 / 对齐失败整行 / 本地权重缺失 → 回退 energy onset；
    - **VII 多语化（2026-08-30）**：原 zh 模型对英/日文行 0% 精修
      （Everlasting/Into the Sky 全靠能量峰）；现按行自动选 zh/ja/en
      （覆盖率 ≥80% 的语言；ja=pykakasi 汉字转假名展开、en=词转字母
      展开，子单元对齐后首子单元时间=字时间）。模型 external/w2v2_{lang}
      可选，缺则该语言回退。

    返回 (chars[onset 已精修], 精修成功字数)。
    """
    if not chars:
        return chars, 0
    import torch
    import torchaudio.functional as AF

    out = [dict(c) for c in chars]
    n_refined = 0
    for li, line in enumerate(lyric_lines):
        # 行归属用 line_idx 精确匹配（窗口法在行界 ±0.1s 会双计边界字
        # ——首测 283 精修 > 254 字即此）；chars 与 lyric_lines 同源自枚举
        idxs = [i for i, c in enumerate(out) if c["line_idx"] == li]
        if not idxs:
            continue
        toks = [out[i]["char"] for i in idxs]
        # 语言选择：逐语种算 token 覆盖率，取 ≥80% 的优先序 zh>ja>en
        chosen = None
        for lang in ("zh", "ja", "en"):
            backend = _ctc_backend(lang, stem_wav)
            if backend is None:
                continue
            logp, _hop, _blank = backend
            vocab = _CTC_MODELS[lang][0].tokenizer.get_vocab()
            covered = 0
            for t in toks:
                subs = _token_units(lang, t)
                if subs and all(s in vocab for s in subs):
                    covered += 1
            if covered >= 0.8 * len(toks):
                chosen = lang
                break
        if chosen is None:
            continue
        logp, hop_sec, blank = _ctc_backend(chosen, stem_wav)
        vocab = _CTC_MODELS[chosen][0].tokenizer.get_vocab()
        # 展开 token → 子单元目标序列 + 分组（字时间=组内首个子单元时间）
        ids: list[int] = []
        groups: list[tuple[int, int]] = []  # (token_idx_in_line, n_subs)
        for k, t in enumerate(toks):
            subs = [s for s in _token_units(chosen, t) if s in vocab]
            if not subs:
                groups.append((k, 0))
                continue
            ids += [vocab[s] for s in subs]
            groups.append((k, len(subs)))
        if not ids:
            continue
        w0 = out[idxs[0]]["onset"] - 0.20
        w1 = max(out[i]["end"] for i in idxs) + 0.20
        f0i, f1i = int(w0 / hop_sec), min(int(w1 / hop_sec), logp.shape[0])
        sub = logp[f0i:f1i]
        if f1i - f0i < len(ids):
            continue
        try:
            t_ids = torch.tensor([ids], dtype=torch.int32)
            align, _ = AF.forced_align(
                sub.unsqueeze(0), t_ids,
                input_lengths=torch.tensor([sub.shape[0]]),
                target_lengths=torch.tensor([len(ids)]), blank=blank)
        except Exception:
            continue
        # 逐子单元取首次出现帧 → 组回 token onset
        sub_onsets: list[float] = []
        pos = 0
        for f, lab in enumerate(align[0].tolist()):
            if pos < len(ids) and lab == ids[pos]:
                sub_onsets.append((f0i + f) * hop_sec)
                pos += 1
        if pos < len(ids):
            continue  # 对齐没走完（不应发生）
        tok_onset: dict[int, float] = {}
        cur = 0
        for k, n_subs in groups:
            if n_subs == 0:
                continue
            tok_onset[k] = sub_onsets[cur]
            cur += n_subs
        for k, t0 in tok_onset.items():
            if t0 >= line["t0"] - 0.10:
                out[idxs[k]]["onset"] = round(float(t0), 3)
                n_refined += 1
    # 保序校验（CTC 版首测教训，2026-08-30）：**字序永远是 LRC 文本序**，
    # CTC 行内偶发相邻字错位（"色的蜻蛉"对成 蜻24.68/色24.72）——按
    # onset 重排会把错位固化为乱序挂字；改为违序 onset 单调化（压到前
    # 字 onset），字表原序不动。重算 end 链。
    for li in range(len(out) - 1):
        if out[li + 1]["line_idx"] == out[li]["line_idx"] \
                and out[li + 1]["onset"] < out[li]["onset"]:
            out[li + 1]["onset"] = out[li]["onset"]
    for li, c in enumerate(out):
        nxt = out[li + 1] if li + 1 < len(out) else None
        if nxt is not None and nxt["line_idx"] == c["line_idx"]:
            c["end"] = round(max(nxt["onset"], c["onset"] + 1e-3), 3)
        else:
            c["end"] = round(max(c["end"], c["onset"] + 1e-3), 3)
    return out, n_refined


# ---------------------------- 行锚自校正（VII） ----------------------------

def snap_line_anchors(lrc: dict, stem_wav: str, song_title: str | None = None,
                      notes: list[dict] | None = None,
                      offset_enabled: bool = False) -> float:
    """曲首标题行剔除 + （默认关的）LRC 全局偏移自校正（2026-08-30 VIII）。

    **偏移校正默认关闭（用户拍板 08-30：官方 LRC 行锚=权威）**。首测用
    SOME 首音中位做参照把夏日整体 +0.287s——用户实测"整首歌显示偏慢"：
    LRC 行锚标的是乐句起点（含起唱准备/呼吸），SOME 首个音高天然滞后
    于它，该差值不是 LRC 误差。需要实验时 MUSE_LRC_SNAP=1 打开（保险丝
    ±0.6s、|中位|>0.15s 才动）。

    保留的部分：曲首标题行剔除（text≈歌名 且 t0 早于首个有声段 2s+ 的
    LRC 惯例占位行——整行静音空走）。

    就地修改 lrc["entries"]/["lyric_lines"]，返回应用的平移秒数。
    """
    import librosa
    import numpy as np

    # 曲首标题行剔除（能量法足够粗用）
    y, _ = librosa.load(stem_wav, sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=220)[0]
    thr = 0.10 * float(np.quantile(rms, 0.90))
    if song_title:
        want = song_title.replace(" ", "").lower()
        first_any = next((i * 0.01 for i in range(len(rms) - 8)
                          if float(np.mean(rms[i:i + 8])) > thr), None)
        if first_any is not None:
            drop = [e for e in lrc["entries"] if e["kind"] == "lyric"
                    and e["t0"] < first_any - 2.0
                    and e["text"].replace(" ", "").lower() == want]
            if drop:
                lrc["entries"] = [e for e in lrc["entries"] if e not in drop]
                lrc["lyric_lines"] = [e for e in lrc["lyric_lines"]
                                      if e not in drop]
                for i, e in enumerate(lrc["entries"][:-1]):  # 补链 t1
                    e["t1"] = lrc["entries"][i + 1]["t0"]

    onsets = sorted(n["onset"] for n in (notes or []))
    deltas: list[float] = []
    for L in lrc["lyric_lines"]:
        t0 = L["t0"]
        if onsets:
            import bisect
            k = bisect.bisect_left(onsets, t0 - 0.8)
            if k < len(onsets) and onsets[k] <= t0 + 1.5:
                deltas.append(onsets[k] - t0)
                continue
        # 回退：能量上升沿（无 notes 或窗内无音符的行）
        i0 = max(int((t0 - 1.2) / 0.01), 8)
        i1 = min(int((t0 + 1.5) / 0.01), len(rms) - 8)
        for i in range(i0, i1):
            lvl = float(np.mean(rms[i:i + 8]))
            if lvl > thr:
                pre = float(np.mean(rms[i - 8:i]))
                if pre < 0.6 * lvl:
                    deltas.append(i * 0.01 - t0)
                break
    if len(deltas) < 3 or not offset_enabled:
        return 0.0
    med = float(np.median(deltas))
    if abs(med) <= 0.15 or abs(med) > 0.6:
        return 0.0
    for e in lrc["entries"]:
        e["t0"] = round(e["t0"] + med, 3)
        e["t1"] = round(e["t1"] + med, 3)
    for L in lrc["lyric_lines"]:
        L["t0"] = round(L["t0"] + med, 3)
        L["t1"] = round(L["t1"] + med, 3)
    return med  # 正值 = 行锚整体后移（原 LRC 偏早）


# ---------------------------- 字级对齐 ----------------------------

_SYL_MIN_GAP = 0.12   # 音节起音最小间距（秒）：普通话快速咬字下限
_CAND_DEDUPE = 0.08   # 能量峰与音头融合去重窗


def align_chars(lrc: dict, stem_wav: str,
                notes: list[dict] | None = None) -> list[dict]:
    """歌词行 → 逐字区间（人声专用：melband stem 已除鼓，能量峰=咬字）。

    行内候选取 stem 能量峰（onset_strength 谱通量）∪ 检出音头（少量加权，
    漏检行音头缺失时纯靠能量），按峰强贪心选 N 个（N=行字数，间距
    ≥_SYL_MIN_GAP）；不足 N 个时余下字零宽落在前字末尾（多字共音，显示层
    拼接）。返回 chars: [{char, line_idx, onset, end}]。

    对齐精度 ±100ms 量级——歌词下挂按"字起音时刻在响的音符"归属，对此
    天然鲁棒；此处不追求乐谱级精度。
    """
    import librosa
    import numpy as np
    from scipy.signal import find_peaks

    hop_sr = 220 / 22050
    y, _sr = librosa.load(stem_wav, sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=220)
    note_onsets = [n["onset"] for n in (notes or [])]

    chars: list[dict] = []
    for li, line in enumerate(lrc["lyric_lines"]):
        toks = line["tokens"]
        n = len(toks)
        if n == 0:
            continue
        # 行窗内能量峰（含 0.1s 前伸：字可能抢在 LRC 时间戳前起音）
        i0 = max(int((line["t0"] - 0.10) / hop_sr), 0)
        i1 = min(int(line["t1"] / hop_sr), len(env))
        seg = env[i0:i1]
        cands: list[tuple[float, float]] = []  # (t, weight)
        if len(seg) > 2:
            med = float(np.median(seg)) or 1e-6
            peaks, _props = find_peaks(seg, distance=int(_SYL_MIN_GAP / hop_sr))
            for p in peaks:
                t = (i0 + int(p)) * hop_sr
                cands.append((t, float(seg[p]) / med))
        for t in note_onsets:
            if line["t0"] - 0.10 <= t <= line["t1"]:
                cands.append((float(t), 1.0))  # 音头=中等权重候选
        # 去重（同峰既是能量峰又是音头 → 保留强者）
        cands.sort(key=lambda c: -c[1])
        merged: list[tuple[float, float]] = []
        for t, w in cands:
            if all(abs(t - t2) >= _CAND_DEDUPE for t2, _ in merged):
                merged.append((t, w))
        # DP 选峰（2026-08-30 VII，替代贪心 top-N）：时间序候选上选长度=
        # 字数的单调子序列，最大化权重和（相邻 ≥ _SYL_MIN_GAP）。贪心按
        # 权重挑会挤掉弱咬字字的峰（行尾字漂移主因之一）；O(n·m²) 可忽略。
        merged.sort(key=lambda c: c[0])
        m = len(merged)
        picked: list[float] = []
        if m >= n and n > 0:
            NEG = float("-inf")
            dp = [[NEG] * m for _ in range(n)]
            pre = [[-1] * m for _ in range(n)]
            for i in range(m):
                dp[0][i] = merged[i][1]
            for k in range(1, n):
                for i in range(m):
                    for j in range(i):
                        if merged[i][0] - merged[j][0] >= _SYL_MIN_GAP \
                                and dp[k - 1][j] > dp[k][i] - merged[i][1]:
                            dp[k][i] = dp[k - 1][j] + merged[i][1]
                            pre[k][i] = j
            bi = max(range(m), key=lambda i: dp[n - 1][i])
            if dp[n - 1][bi] > NEG / 2:
                k, path = n - 1, []
                while k >= 0 and bi >= 0:
                    path.append(merged[bi][0])
                    bi = pre[k][bi]
                    k -= 1
                picked = sorted(path)
        # 不足 N 峰：剩余字在"末峰之后剩余时间"均分（P1-α 2026-08-30，
        # 原零宽续前会让 fill 无窗可插 → 1/3 事件一音多字的主因之一；
        # 均分给每字非零窗，补漏层才有落点，一字一音先验才有实体）
        if len(picked) < n:
            last = picked[-1] if picked else line["t0"]
            span = max(line["t1"] - last, 0.1)
            m = n - len(picked)
            for j in range(m):
                picked.append(last + span * (j + 1) / (m + 1))
            picked.sort()
        onsets = picked
        for k, tok in enumerate(toks):
            end = onsets[k + 1] if k + 1 < n else max(line["t1"], onsets[k])
            chars.append({"char": tok, "line_idx": li,
                          "onset": round(float(onsets[k]), 3),
                          "end": round(float(max(end, onsets[k] + 1e-3)), 3)})
    return chars
