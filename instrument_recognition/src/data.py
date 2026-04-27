# data.py
import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from src.config import config

class IRMASDataset(Dataset):
    def __init__(self, data_dir, transform=True):
        self.data_dir = data_dir
        self.filepaths = []
        self.labels = []
        
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        for c in self.classes:
            class_dir = os.path.join(data_dir, c)
            wav_files = glob.glob(os.path.join(class_dir, "*.wav"))
            for f in wav_files:
                self.filepaths.append(f)
                self.labels.append(c)
                
        self.encoded_labels = self.label_encoder.transform(self.labels)
        self.transform = transform
        
    def _extract_features(self, filepath):
        y, sr = librosa.load(filepath, sr=config.SR, duration=config.DURATION)
        # Pad if shorter
        if len(y) < config.SR * config.DURATION:
            pad_length = config.SR * config.DURATION - len(y)
            y = np.pad(y, (0, pad_length))
        
        # Ext mel
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Ext mfcc
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
        
        features = np.vstack([mel_db, mfcc]) # Stack them channel-wise or height-wise
        # We can treat this as a 1 channel 2D matrix
        features = features[np.newaxis, ...]
        return features

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        feats = self._extract_features(self.filepaths[idx])
        label = self.encoded_labels[idx]
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_dataloaders(val_split=0.2):
    dataset = IRMASDataset(config.DATASET_DIR)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, dataset.classes
