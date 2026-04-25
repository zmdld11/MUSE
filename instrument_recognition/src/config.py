import os

class Config:
    # Base paths
    DATA_PATH = r"D:\program_project\MUSE\data\IRMAS-TrainingData"
    WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_DIR = os.path.join(WORKING_DIR, "model", "log")
    MODEL_DIR = os.path.join(WORKING_DIR, "model")
    
    # Audio preprocessing
    SAMPLE_RATE = 22050
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048
    DURATION = 3.0
    SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
    
    # Training hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3  # SGD learning rate for fine-tuning
    WEIGHT_DECAY = 1e-3
    MOMENTUM = 0.9
    NUM_CLASSES = 11
    
    # Model parameters
    IN_CHANNELS = 3
    
    # Misc
    NUM_WORKERS = 4
