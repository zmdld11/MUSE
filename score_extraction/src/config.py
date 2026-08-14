import os

class Config:
    def __init__(self):
        self.WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.OUTPUT_DIR = os.path.join(self.WORKSPACE_DIR, "output")
        self.MODEL_DIR = os.path.join(self.WORKSPACE_DIR, "model")
        self.DEMUCS_MODEL = "htdemucs_6s"
        self.DEMUCS_MODEL_PATH = None
        self.PITCH_MODEL_PIANO = "basic-pitch"
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
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

config = Config()
