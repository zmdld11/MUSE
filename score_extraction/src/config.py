import os

class Config:
    def __init__(self):
        self.WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.OUTPUT_DIR = os.path.join(self.WORKSPACE_DIR, "output")
        self.MODEL_DIR = os.path.join(self.WORKSPACE_DIR, "model")
        self.DEMUCS_MODEL = "htdemucs_6s"
        self.DEMUCS_MODEL_PATH = None
        self.PITCH_MODEL_PIANO = "basic-pitch"
        # Route A frontend: bytedance uses official HR events; ours keeps legacy path.
        self.PIANO_FRONTEND = os.environ.get("MUSE_PIANO_FRONTEND", "bytedance")
        # Route B frontend (2026-08-21): ia-amt = anime-song/instrument-agnostic-amt
        # Semi-CRF（MIT）。基线钉 guitar_v1_5（07-22 版，四口径全面强于 v1：
        # GuitarSet 0.9227 / 東の空 stem 0.3863 / 虚無の先严格 45.9%）。
        self.GUITAR_FRONTEND = os.environ.get("MUSE_GUITAR_FRONTEND", "ia_amt")
        self.IA_AMT_TYPE = os.environ.get("MUSE_IA_AMT_TYPE", "guitar_v1_5")
        # 吉他线输入：raw=原始混音直推（矩阵验证吉他主导曲可行，免 demucs 伪影）；
        # stem=htdemucs_6s 分离轨（暂留，VER-SEP 本地接入后升级 versep 模式）
        self.GUITAR_ENABLE = os.environ.get("MUSE_GUITAR_ENABLE", "1") not in ("0", "false")
        self.GUITAR_INPUT = os.environ.get("MUSE_GUITAR_INPUT", "raw")
        # 多乐器模式（2026-08-22）：ia-amt default 单模型全乐队，按 instrument_class
        # 分轨落盘（每类一个 .mid + notes.json）。开启时替代吉他单线。
        self.MULTI_INSTRUMENT = os.environ.get("MUSE_MULTI_INSTRUMENT", "0") == "1"
        # 记谱层模式（阶段19）：quantized | faithful | both（规则见记谱规则v1.md §4）
        self.NOTATION_MODE = os.environ.get("MUSE_NOTATION_MODE", "both")
        self.PITCH_MODEL_MONO = "crepe"
        self.SR = 44100
        self.MODEL_SR = 22050    # SR for our trained model (mel spectrogram input)
        self.HOP_LENGTH = 512
        self.DEFAULT_BPM = 120.0
        self.DEFAULT_TIME_SIG = "4/4"
        self.MAX_FRET = 22
        self.FRET_WEIGHT = 1.0
        self.STRING_WEIGHT = 2.0
        self.OPEN_STRING_BIAS = -0.5
        self.SLIDE_PITCH_THRESHOLD = 0.5
        self.SLIDE_MAX_INTERVAL = 5
        self.MUSESCORE_PATH = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
        # 2026-08-02: 钢琴曲只转录 piano (跳过 bass/guitar/vocals 等不适配轨道,
        # 也跳过 source separation — piano 直接用原始音频)
        self.ONLY_PIANO = True
        # 2026-08-15 onset-first experiment: preserve attacks, avoid transient-smoothing
        self.PIANO_USE_WIENER = False
        # VER4.2: offset head currently truncates valid sustains; disable by default.
        self.PIANO_USE_MODEL_OFFSET = False
        self.BP_ONSET_THRESHOLD = 0.65
        self.BP_FRAME_THRESHOLD = 0.25
        self.HARD_ECHO_FILTER = False
        self.MIN_NOTE_DURATION = 0.10
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

config = Config()

