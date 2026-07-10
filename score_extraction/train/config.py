"""Training hyperparameters for NoteEM."""
import os
import torch


class TrainConfig:
    def __init__(self):
        # Paths
        self.WORKSPACE_DIR = r"D:\program_project\MUSE\score_extraction"
        self.MIDI_DIR = os.path.join(self.WORKSPACE_DIR, "data", "midi", "GiantMIDI-PIano", "midis")
        self.MODEL_SAVE_DIR = os.path.join(self.WORKSPACE_DIR, "model")
        self.MODEL_VERSION = "VER2.0_NoteEM"

        # Audio
        self.SR = 22050
        self.HOP_LENGTH = 512
        self.N_MELS = 229
        self.N_MIDI = 88          # MIDI 21-108
        self.MIDI_OFFSET = 21     # lowest MIDI note

        # Model
        self.CNN_CHANNELS = [32, 64, 128]
        self.LSTM_HIDDEN = 128

        # Training
        self.BATCH_SIZE = 8
        self.EPOCHS_BOOTSTRAP = 30
        self.EPOCHS_EM = 5
        self.LR = 3e-4
        self.MAX_DUR_SEC = 60     # truncate long pieces
        self.NUM_WORKERS = 0      # dataloader workers

        # EM
        self.EM_ITERATIONS = 2
        self.DTW_RADIUS = 50      # Sakoe-Chiba band for fast DTW

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FluidSynth rendering (auto-detected after config init)
        self.FLUIDSYNTH_ENABLED = False

        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)


train_config = TrainConfig()
