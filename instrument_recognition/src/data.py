# data.py
import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from src.config import config
from tqdm import tqdm

from torchvision import transforms
import torchaudio.transforms as T
import random

class SpecAugment(object):
    def __init__(self, freq_mask_param=12, time_mask_param=12, freq_masks=2, time_masks=2):
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        
    def __call__(self, spec):
        # Apply SpecAugment slightly stronger than 1.6 to prevent overfitting
        if random.random() > 0.5:
            for _ in range(self.freq_masks):
                spec = self.freq_mask(spec)
            for _ in range(self.time_masks):
                spec = self.time_mask(spec)
        return spec

class IRMASDataset(Dataset):
    def __init__(self, data_dir, transform=True, is_train=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        self.spec_aug = SpecAugment() if is_train else None
        
        self.cache_file = os.path.join(config.CACHE_DIR, f"cache_{config.DATASET_VERSION}.pt")
        if os.path.exists(self.cache_file):
            print(f"Loading preprocessed dataset from file cache: {self.cache_file}...")
            cache_data = torch.load(self.cache_file, weights_only=False)
            self.filepaths = cache_data['filepaths']
            self.labels = cache_data['labels']
            self.classes = cache_data['classes']
            self.encoded_labels = cache_data['encoded_labels']
            self.cache = cache_data['cache']
            self.dataset_weights = cache_data.get('dataset_weights', np.ones(len(self.labels)))
            print("Dataset successfully loaded from cache.")
            return
            
        print(f"Cache file {self.cache_file} not found. Re-processing raw dataset (will not auto-delete old isolated versions)...")

        self.filepaths = []
        self.labels = []
        self.dataset_weights = []
        
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        import yaml
        
        medleydb_to_irmas = {
            'female singer': 'voi', 'male singer': 'voi', 'singer': 'voi',
            'piano': 'pia', 'violin': 'vio', 'acoustic guitar': 'gac', 
            'electric guitar': 'gel', 'flute': 'flu', 'clarinet': 'cla', 
            'trumpet': 'tru', 'saxophone': 'sax', 'alto saxophone': 'sax', 
            'tenor saxophone': 'sax', 'baritone saxophone': 'sax', 
            'soprano saxophone': 'sax', 'cello': 'cel', 'electric organ': 'org', 
            'organ': 'org'
        }
        
        # Load IRMAS
        for c in self.classes:
            class_dir = os.path.join(data_dir, c)
            wav_files = glob.glob(os.path.join(class_dir, "*.wav"))
            for f in wav_files:
                self.filepaths.append((f, 'irmas', 0.0)) # (filepath, dataset_type, onset_time_offset)
                self.labels.append(c)
                self.dataset_weights.append(1.0)
                
        # Load MedleyDB Stems (Pure Audio)
        if hasattr(config, 'MEDLEYDB_DIR') and os.path.exists(config.MEDLEYDB_DIR):
            medley_meta_dir = os.path.join(config.MEDLEYDB_DIR, "Metadata")
            medley_audio_dir = os.path.join(config.MEDLEYDB_DIR, "MedleyDB")
            meta_files = glob.glob(os.path.join(medley_meta_dir, "*.yaml"))
            
            medley_counts = {c: 0 for c in self.classes}
            
            for meta_file in tqdm(meta_files, desc="Scanning MedleyDB Stems"):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    try:
                        md = yaml.safe_load(f)
                        stems = md.get('stems', {})
                        track_name = md.get('stem_dir', '')
                        if not stems or not track_name: continue
                        
                        for s_key, s_val in stems.items():
                            inst = s_val.get('instrument', '').lower()
                            if inst in medleydb_to_irmas:
                                irmas_class = medleydb_to_irmas[inst]
                                if medley_counts[irmas_class] < config.MAX_MEDLEY_STEMS_PER_CLASS:
                                    stem_path = os.path.join(medley_audio_dir, track_name, s_val.get('filename', ''))
                                    if os.path.exists(stem_path):
                                        # Use librosa to find onset peaks to grab the "Attack" of the instrument
                                        y, sr = librosa.load(stem_path, sr=config.SR)
                                        # Split to non-silent to speed up onset detection
                                        intervals = librosa.effects.split(y, top_db=40)
                                        if len(intervals) > 0:
                                            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
                                            np.random.shuffle(onsets) # Grab a few random onsets
                                            added_chunks = 0
                                            for onset in onsets:
                                                if added_chunks >= config.MAX_CHUNKS_PER_STEM: break
                                                self.filepaths.append((stem_path, 'medleydb', onset))
                                                self.labels.append(irmas_class)
                                                self.dataset_weights.append(config.MEDLEY_SAMPLE_WEIGHT)
                                                added_chunks += 1
                                            medley_counts[irmas_class] += 1
                    except Exception as e:
                        pass
        
        self.encoded_labels = self.label_encoder.transform(self.labels)
        
        # 预先将所有的音频特征提取并加载到内存缓存中，防止在训练过程中每次都进行耗时的I/O与librosa计算
        self.cache = {}
        print("Pre-loading and extracting all audio features to Memory...")
        
        for idx in tqdm(range(len(self.filepaths)), desc="Caching dataset"):
            fpath, dtype, onset = self.filepaths[idx]
            feats = self._extract_features(fpath, dtype, onset)
            label = self.encoded_labels[idx]
            self.cache[idx] = (torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long))

        print(f"Saving preprocessed dataset to {self.cache_file}...")
        torch.save({
            'filepaths': self.filepaths,
            'labels': self.labels,
            'classes': self.classes,
            'encoded_labels': self.encoded_labels,
            'dataset_weights': self.dataset_weights,
            'cache': self.cache
        }, self.cache_file)
        
    def _extract_features(self, filepath, dtype='irmas', onset=0.0):
        if dtype == 'irmas':
            y, sr = librosa.load(filepath, sr=config.SR, duration=config.DURATION)
        else: # medleydb
            # For medleydb, we grab logic reflecting the "音头 (attack)": onset - 0.1s to onset + 2.9s
            offset_time = max(0, onset - 0.1)
            y, sr = librosa.load(filepath, sr=config.SR, offset=offset_time, duration=config.DURATION)
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
        feats, label = self.cache[idx]
        if self.is_train:
            # 1. 动态时间平移增强 (Time Shift Augmentation)
            # 通过随机移动频谱图的时间轴，打破模型对固定“音头=0.1s”的死记硬背
            _, _, T_dim = feats.shape
            max_shift = int(T_dim * 0.3) # 允许在 30% 范围内平移
            shift = random.randint(-max_shift, max_shift)
            feats = torch.roll(feats, shifts=shift, dims=2)
            
            if getattr(self, 'spec_aug', None):
                # 2. SpecAugment 时频掩码频蔽
                feats = self.spec_aug(feats)
        return feats, label

def get_dataloaders(val_split=0.2):
    dataset = IRMASDataset(config.DATASET_DIR)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # A cleaner approach is to make a copy for train with is_train=True
    import copy
    train_dataset_base = copy.copy(dataset)
    train_dataset_base.is_train = True
    train_dataset_base.spec_aug = SpecAugment()
    train_dataset.dataset = train_dataset_base
    
    # Get weights and labels for Train set to construct WeightedRandomSampler
    train_indices = train_dataset.indices
    
    # We mix inverse class frequency weights with Dataset-Originated sample weights
    class_counts = np.bincount([dataset.encoded_labels[i] for i in train_indices], minlength=len(dataset.classes))
    class_weights = 1.0 / (class_counts + 1e-6)
    
    sample_weights = []
    for i in train_indices:
        label = dataset.encoded_labels[i]
        c_weight = class_weights[label]
        # dataset.dataset_weights[i] contains either 1.0 (IRMAS) or MEDLEY_SAMPLE_WEIGHT (MedleyDB)
        d_weight = dataset.dataset_weights[i]
        sample_weights.append(c_weight * d_weight)
        
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Print dataset details before training starts
    medley_count = sum(1 for i in range(len(dataset)) if dataset.dataset_weights[i] > 1.1)
    irmas_count = len(dataset) - medley_count
    print(f"\n======== 数据集预处理与加载汇总 ========")
    print(f"当前缓存版本: {config.DATASET_VERSION}")
    print(f"IRMAS 切片数量: {irmas_count}")
    print(f"MedleyDB 纯净音轨提取音头片段: {medley_count}")
    print(f"训练集大小: {train_size} | 验证集大小: {val_size}")
    print(f"MedleyDB 专项权重: {config.MEDLEY_SAMPLE_WEIGHT}x")
    print(f"=======================================\n")
    
    return train_loader, val_loader, dataset.classes
