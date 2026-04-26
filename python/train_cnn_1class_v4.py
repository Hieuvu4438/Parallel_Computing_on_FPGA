#!/usr/bin/env python3
"""
================================================================================
CNN Training Script v4 - Binary COPD with Multi-Resolution Wavelet Stacking + Mixup
================================================================================

Features:
- Multi-resolution Wavelet Stacking: 3 scales (Low/Mid/High freq) -> RGB channels
- Mixup Augmentation: alpha=0.2 during training
- 5-Fold GroupKFold (Subject-Independent Split)
- 150 epochs with Early Stopping (patience=20, reset after unfreeze)
- Detailed per-epoch logging
- Comprehensive visualization (2x2 plots per fold)
- Final report with Mean ± Std statistics

Usage:
    python train_cnn_1class_v4.py --data_path /path/to/ICBHI --output_dir ./output_copd_v4
"""

import gc
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import json

import numpy as np
import pywt
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    from torchaudio.transforms import FrequencyMasking, TimeMasking
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS
# ==============================================================================

IMG_SIZE = 224
TARGET_SR = 4000
WAVELET_NAME = 'morl'
NUM_CLASSES = 2
CLASS_NAMES = ['Non-COPD', 'COPD']
N_FOLDS = 5
MIXUP_ALPHA = 0.2  # Mixup augmentation parameter

# Frequency ranges for multi-resolution stacking (50-1500Hz focus)
FREQ_LOW = (50, 400)    # Low frequency band
FREQ_MID = (400, 900)   # Mid frequency band
FREQ_HIGH = (900, 1500) # High frequency band

PATIENT_DIAGNOSIS = {
    101: 'URTI', 102: 'Healthy', 103: 'Asthma', 104: 'COPD', 105: 'URTI',
    106: 'COPD', 107: 'COPD', 108: 'LRTI', 109: 'COPD', 110: 'COPD',
    111: 'Bronchiectasis', 112: 'COPD', 113: 'COPD', 114: 'COPD', 115: 'LRTI',
    116: 'Bronchiectasis', 117: 'COPD', 118: 'COPD', 119: 'URTI', 120: 'COPD',
    121: 'Healthy', 122: 'Pneumonia', 123: 'Healthy', 124: 'COPD', 125: 'Healthy',
    126: 'Healthy', 127: 'Healthy', 128: 'COPD', 129: 'URTI', 130: 'COPD',
    131: 'URTI', 132: 'COPD', 133: 'COPD', 134: 'COPD', 135: 'Pneumonia',
    136: 'Healthy', 137: 'URTI', 138: 'COPD', 139: 'COPD', 140: 'Pneumonia',
    141: 'COPD', 142: 'COPD', 143: 'Healthy', 144: 'Healthy', 145: 'COPD',
    146: 'COPD', 147: 'COPD', 148: 'URTI', 149: 'Bronchiolitis', 150: 'URTI',
    151: 'COPD', 152: 'Healthy', 153: 'Healthy', 154: 'COPD', 155: 'COPD',
    156: 'COPD', 157: 'COPD', 158: 'COPD', 159: 'Healthy', 160: 'COPD',
    161: 'Bronchiolitis', 162: 'COPD', 163: 'COPD', 164: 'URTI', 165: 'URTI',
    166: 'COPD', 167: 'Bronchiolitis', 168: 'Bronchiectasis', 169: 'Bronchiectasis',
    170: 'COPD', 171: 'Healthy', 172: 'COPD', 173: 'Bronchiolitis', 174: 'COPD',
    175: 'COPD', 176: 'COPD', 177: 'COPD', 178: 'COPD', 179: 'Healthy', 180: 'COPD',
    181: 'COPD', 182: 'Healthy', 183: 'Healthy', 184: 'Healthy', 185: 'COPD',
    186: 'COPD', 187: 'Healthy', 188: 'URTI', 189: 'COPD', 190: 'URTI',
    191: 'Pneumonia', 192: 'COPD', 193: 'COPD', 194: 'Healthy', 195: 'COPD',
    196: 'Bronchiectasis', 197: 'URTI', 198: 'COPD', 199: 'COPD', 200: 'COPD',
    201: 'Bronchiectasis', 202: 'Healthy', 203: 'COPD', 204: 'COPD', 205: 'COPD',
    206: 'Bronchiolitis', 207: 'COPD', 208: 'Healthy', 209: 'Healthy', 210: 'URTI',
    211: 'COPD', 212: 'COPD', 213: 'COPD', 214: 'Healthy', 215: 'Bronchiectasis',
    216: 'Bronchiolitis', 217: 'Healthy', 218: 'COPD', 219: 'Pneumonia', 220: 'COPD',
    221: 'COPD', 222: 'COPD', 223: 'COPD', 224: 'Healthy', 225: 'Healthy',
    226: 'Pneumonia',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHASE2_LR = 5e-5
EARLY_STOP_PATIENCE = 20
PHASE2_MIN_EPOCHS = 15
TOTAL_EPOCHS = 150


# ==============================================================================
# MULTI-RESOLUTION WAVELET TRANSFORM
# ==============================================================================

class MultiResolutionWaveletTransform:
    """
    Multi-resolution Wavelet Stacking using Morlet Wavelet.
    Creates 3 spectrograms for Low/Mid/High frequency bands (50-1500Hz),
    then stacks them as RGB channels.
    """
    
    def __init__(self, wavelet=WAVELET_NAME, sample_rate=TARGET_SR, 
                 output_size=IMG_SIZE, num_scales=128):
        self.wavelet = wavelet
        self.sample_rate = sample_rate
        self.output_size = output_size
        self.num_scales = num_scales
        
        # Compute scales for each frequency band
        self.scales_low = self._compute_scales(FREQ_LOW)
        self.scales_mid = self._compute_scales(FREQ_MID)
        self.scales_high = self._compute_scales(FREQ_HIGH)
    
    def _compute_scales(self, freq_range: Tuple[int, int]) -> np.ndarray:
        """Compute wavelet scales for a given frequency range."""
        center_freq = pywt.central_frequency(self.wavelet)
        min_scale = center_freq * self.sample_rate / freq_range[1]
        max_scale = center_freq * self.sample_rate / freq_range[0]
        return np.logspace(np.log10(min_scale), np.log10(max_scale), self.num_scales)
    
    def _transform_band(self, audio: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Transform audio to spectrogram for a specific frequency band."""
        coefficients, _ = pywt.cwt(audio, scales, self.wavelet,
                                  sampling_period=1.0 / self.sample_rate)
        power = np.abs(coefficients) ** 2
        power_db = 10 * np.log10(power + 1e-10)
        # Normalize to [0, 1]
        power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)
        return power_db
    
    def transform(self, audio: np.ndarray) -> np.ndarray:
        """
        Generate multi-resolution wavelet spectrogram.
        Returns: (3, H, W) array representing RGB channels (Low/Mid/High freq)
        """
        # Generate spectrograms for each frequency band
        spec_low = self._transform_band(audio, self.scales_low)
        spec_mid = self._transform_band(audio, self.scales_mid)
        spec_high = self._transform_band(audio, self.scales_high)
        
        # Resize to target size
        specs = [spec_low, spec_mid, spec_high]
        resized_specs = []
        
        for spec in specs:
            zoom_factors = (self.output_size / spec.shape[0], 
                          self.output_size / spec.shape[1])
            spec_resized = zoom(spec, zoom_factors, order=1)
            spec_resized = spec_resized[:self.output_size, :self.output_size]
            resized_specs.append(np.clip(spec_resized, 0, 1).astype(np.float32))
        
        # Stack as RGB channels: [R=Low, G=Mid, B=High]
        stacked = np.stack(resized_specs, axis=0)
        return stacked


# ==============================================================================
# MIXUP AUGMENTATION
# ==============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = MIXUP_ALPHA) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation.
    
    Args:
        x: Input batch (B, C, H, W)
        y: Labels (B,)
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x: Mixed inputs
        y_a: Labels for first sample
        y_b: Labels for second sample
        lam: Mixup lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, 
                   y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute loss for Mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==============================================================================
# SPEC AUGMENT
# ==============================================================================

class SpecAugment:
    def __init__(self, freq_mask_param=20, time_mask_param=20, p=0.5):
        self.p = p
        if HAS_TORCHAUDIO:
            self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_param)
            self.time_mask = TimeMasking(time_mask_param=time_mask_param)
        else:
            self.freq_mask = self.time_mask = None
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return spec
        if HAS_TORCHAUDIO and self.freq_mask is not None:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
            return spec
        return self._manual_mask(spec)
    
    def _manual_mask(self, spec: torch.Tensor) -> torch.Tensor:
        c, h, w = spec.shape
        out = spec.clone()
        if np.random.random() < 0.5:
            f = int(np.random.uniform(0, min(self.freq_mask_param, h)))
            f0 = np.random.randint(0, max(1, h - f))
            out[:, f0:f0 + f, :] = 0
        if np.random.random() < 0.5:
            t = int(np.random.uniform(0, min(self.time_mask_param, w)))
            t0 = np.random.randint(0, max(1, w - t))
            out[:, :, t0:t0 + t] = 0
        return out


# ==============================================================================
# AUDIO AUGMENTER
# ==============================================================================

class AudioAugmenter:
    def __init__(self, noise_level=0.01, shift_ratio=0.15, gain_range=(0.8, 1.2), probability=0.7):
        self.noise_level = noise_level
        self.shift_ratio = shift_ratio
        self.gain_min, self.gain_max = gain_range
        self.probability = probability
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio
        if np.random.random() < 0.5:
            audio = audio + np.random.normal(0, self.noise_level, len(audio))
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-self.shift_ratio, self.shift_ratio) * len(audio))
            audio = np.roll(audio, shift)
        if np.random.random() < 0.5:
            audio = audio * np.random.uniform(self.gain_min, self.gain_max)
        return np.clip(audio, -1.0, 1.0)


# ==============================================================================
# DATASET
# ==============================================================================

class ICBHIBinaryDatasetV4(Dataset):
    def __init__(self, file_list, labels, patient_ids, data_path, transform=None,
                 wavelet_transform=None, augment=False, oversample_minority=False, 
                 spec_augment=False, use_mixup=False):
        self.file_list = file_list
        self.labels = labels
        self.patient_ids = patient_ids
        self.data_path = Path(data_path)
        self.transform = transform
        self.wavelet_transform = wavelet_transform or MultiResolutionWaveletTransform()
        self.augment = augment
        self.augmenter = AudioAugmenter() if augment else None
        self.spec_augment = spec_augment
        self.spec_aug = SpecAugment(20, 20, 0.5) if spec_augment else None
        self.use_mixup = use_mixup  # Note: Mixup is applied in training loop, not here
        if oversample_minority:
            self._oversample_minority()
    
    def _oversample_minority(self):
        class_counts = Counter(self.labels)
        max_count = max(class_counts.values())
        minority_indices = [i for i, l in enumerate(self.labels) if l == 0]
        if not minority_indices:
            return
        n_oversample = max_count - class_counts.get(0, 0)
        if n_oversample <= 0:
            return
        oversample_idx = np.random.choice(minority_indices, n_oversample, replace=True)
        for i in oversample_idx:
            self.file_list.append(self.file_list[i])
            self.labels.append(self.labels[i])
            self.patient_ids.append(self.patient_ids[i])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        wav_path = self.data_path / self.file_list[idx]
        try:
            sr, audio = wavfile.read(wav_path)
        except Exception:
            audio = np.zeros(TARGET_SR * 2, dtype=np.float32)
            sr = TARGET_SR
        
        # Convert to float [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        
        # Resample if needed
        if sr != TARGET_SR:
            num_samples = int(len(audio) * TARGET_SR / sr)
            audio = signal.resample(audio, num_samples)
        
        label = self.labels[idx]
        
        # Audio augmentation
        if self.augment and self.augmenter:
            if label == 0:
                audio = self.augmenter(audio)
            elif np.random.random() < 0.3:
                audio = self.augmenter(audio)
        
        # Generate multi-resolution wavelet spectrogram (already 3-channel RGB)
        spec_image = self.wavelet_transform.transform(audio)
        spec_tensor = torch.from_numpy(spec_image).float()
        
        # Spec augmentation
        if self.spec_augment and self.spec_aug is not None:
            spec_tensor = self.spec_aug(spec_tensor)
        
        # Normalization transform
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
        
        return spec_tensor, label


# ==============================================================================
# DATA LOADING & GROUP K-FOLD
# ==============================================================================

def parse_filename(filename: str) -> Tuple[int, str]:
    basename = Path(filename).stem
    parts = basename.split('_')
    return int(parts[0]) if len(parts) >= 1 else 0, basename


def load_diagnosis_mapping(data_path: str) -> Dict[int, str]:
    f = Path(data_path) / "ICBHI_Challenge_diagnosis.txt"
    if f.exists():
        out = {}
        with open(f, 'r') as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 2:
                    out[int(parts[0])] = parts[1]
        return out
    return PATIENT_DIAGNOSIS.copy()


def get_binary_label(diagnosis: str) -> int:
    return 1 if diagnosis.upper() == 'COPD' else 0


def load_dataset(data_path: str) -> Tuple[List[str], List[int], np.ndarray]:
    data_path = Path(data_path)
    wav_files = sorted(data_path.glob("*.wav"))
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files in {data_path}")
    patient_diagnosis = load_diagnosis_mapping(str(data_path))
    file_list, labels, patient_ids = [], [], []
    for wav_file in wav_files:
        pid, _ = parse_filename(wav_file.name)
        label = get_binary_label(patient_diagnosis.get(pid, ''))
        file_list.append(wav_file.name)
        labels.append(label)
        patient_ids.append(pid)
    patient_ids = np.array(patient_ids)
    print(f"Loaded {len(file_list)} files, {len(np.unique(patient_ids))} patients")
    return file_list, labels, patient_ids


def get_group_kfold_splits(file_list, labels, patient_ids, n_splits=N_FOLDS,
                            val_ratio=0.15, random_state=42):
    X = np.arange(len(file_list))
    y = np.array(labels)
    groups = patient_ids
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for train_val_idx, test_idx in gkf.split(X, y, groups):
        train_val_groups = groups[train_val_idx]
        unique_patients = np.unique(train_val_groups)
        patient_to_label = dict(zip(unique_patients,
            [y[train_val_idx][np.where(train_val_groups == p)[0][0]] for p in unique_patients]))
        plabels = np.array([patient_to_label[p] for p in unique_patients])
        train_patients, val_patients = train_test_split(
            unique_patients, test_size=val_ratio, random_state=random_state, stratify=plabels
        )
        train_idx = train_val_idx[np.isin(train_val_groups, train_patients)]
        val_idx = train_val_idx[np.isin(train_val_groups, val_patients)]
        folds.append((train_idx, val_idx, test_idx))
    return folds


# ==============================================================================
# MODEL
# ==============================================================================

class COPDClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
    
    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ==============================================================================
# TRAIN / VALIDATE
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply Mixup augmentation
        if use_mixup and np.random.random() < 0.5:  # 50% chance to apply Mixup
            mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha=MIXUP_ALPHA)
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            loss = criterion(out, labels)
            running_loss += loss.item() * inputs.size(0)
            _, pred = out.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    loss_avg = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return loss_avg, acc, f1, all_preds, all_labels


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'confusion_matrix': cm
    }


def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ==============================================================================
# PROGRESS LOGGING
# ==============================================================================

def print_epoch_progress(fold_id: int, n_folds: int, phase: str, epoch: int, total_epochs: int,
                         train_loss: float, val_loss: float, train_acc: float, val_acc: float,
                         val_f1: float, lr: float, is_best: bool, patience_counter: int):
    """Detailed per-epoch logging."""
    best_mark = " [BEST]" if is_best else ""
    pat_str = f" | patience={patience_counter}/{EARLY_STOP_PATIENCE}" if phase == "Phase 2" else ""
    print(f"Epoch [{epoch+1:3d}/{total_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | Val F1: {val_f1*100:5.2f}% | "
          f"LR: {lr:.2e}{best_mark}{pat_str}")


def print_fold_banner(fold_id: int, n_folds: int, n_train: int, n_val: int, n_test: int):
    """Banner at start of each fold."""
    print("\n" + "="*70)
    print(f"  FOLD {fold_id + 1}/{n_folds}  |  Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")
    print("="*70)


# ==============================================================================
# TRAIN ONE FOLD
# ==============================================================================

def train_fold(fold_id: int, n_folds: int, train_idx, val_idx, test_idx,
               file_list, labels, patient_ids, data_path: str, output_dir: Path, args,
               normalize_transform, wavelet_transform):
    train_files = [file_list[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_pids = [int(patient_ids[i]) for i in train_idx]
    val_files = [file_list[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_pids = [int(patient_ids[i]) for i in val_idx]
    test_files = [file_list[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_pids = [int(patient_ids[i]) for i in test_idx]
    
    print_fold_banner(fold_id, n_folds, len(train_files), len(val_files), len(test_files))
    
    train_ds = ICBHIBinaryDatasetV4(
        train_files, train_labels, train_pids, data_path,
        normalize_transform, wavelet_transform,
        augment=True, oversample_minority=True, spec_augment=True, use_mixup=True
    )
    val_ds = ICBHIBinaryDatasetV4(
        val_files, val_labels, val_pids, data_path,
        normalize_transform, wavelet_transform, augment=False, spec_augment=False
    )
    test_ds = ICBHIBinaryDatasetV4(
        test_files, test_labels, test_pids, data_path,
        normalize_transform, wavelet_transform, augment=False, spec_augment=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    model = COPDClassifier(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    class_weights = torch.FloatTensor([
        total / (NUM_CLASSES * class_counts.get(i, 1)) for i in range(NUM_CLASSES)
    ]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 
        'val_f1': [], 'val_sensitivity': [], 'val_specificity': [], 'lr': []
    }
    best_val_loss = float('inf')
    patience_counter = 0
    phase1_epochs = min(30, args.epochs // 5)
    total_epochs = args.epochs
    
    # ---------- Phase 1: Freeze backbone ----------
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    for epoch in range(phase1_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, use_mixup=True)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        scheduler.step(val_loss)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / f"best_model_fold_{fold_id}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        print_epoch_progress(fold_id, n_folds, "Phase 1", epoch, total_epochs,
                             train_loss, val_loss, train_acc, val_acc, val_f1, current_lr, is_best, patience_counter)
    
    # ---------- Phase 2: Unfreeze backbone ----------
    model.unfreeze_backbone()
    patience_counter = 0  # Reset patience counter after unfreezing
    optimizer = optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    total_epochs_run = phase1_epochs
    
    for epoch in range(phase1_epochs, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, use_mixup=True)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        scheduler.step()
        total_epochs_run = epoch + 1
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / f"best_model_fold_{fold_id}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        print_epoch_progress(fold_id, n_folds, "Phase 2", epoch, total_epochs,
                             train_loss, val_loss, train_acc, val_acc, val_f1, current_lr, is_best, patience_counter)
        
        phase2_elapsed = (epoch + 1) - phase1_epochs
        if phase2_elapsed >= PHASE2_MIN_EPOCHS and patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  >> Early stopping at epoch {epoch+1} (val_loss, patience={EARLY_STOP_PATIENCE})")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(output_dir / f"best_model_fold_{fold_id}.pth", map_location=DEVICE))
    model.eval()
    
    # Compute validation metrics for history
    _, _, _, val_preds, val_labels_arr = validate(model, val_loader, criterion, DEVICE)
    val_metrics = compute_metrics(val_labels_arr, val_preds)
    history['val_sensitivity'] = [val_metrics['sensitivity']] * len(history['train_loss'])
    history['val_specificity'] = [val_metrics['specificity']] * len(history['train_loss'])
    
    # Test set evaluation
    _, _, _, test_preds, test_labels_arr = validate(model, test_loader, criterion, DEVICE)
    test_metrics = compute_metrics(test_labels_arr, test_preds)
    
    return history, test_metrics, total_epochs_run


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_fold_results(history: Dict, test_metrics: Dict, save_path: str, fold_id: int):
    """
    Create 2x2 plot: Loss, Accuracy, Metrics, Confusion Matrix
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss (Train vs Val)
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', lw=1.5)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', lw=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Fold {fold_id+1} - Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy (Train vs Val)
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', lw=1.5)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', lw=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'Fold {fold_id+1} - Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Metrics (Val F1, Sensitivity, Specificity)
    metrics_data = {
        'Val F1': history['val_f1'][-1] if history['val_f1'] else 0,
        'Val Sensitivity': history['val_sensitivity'][-1] if history['val_sensitivity'] else 0,
        'Val Specificity': history['val_specificity'][-1] if history['val_specificity'] else 0
    }
    axes[1, 0].bar(metrics_data.keys(), [v * 100 for v in metrics_data.values()], 
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_title(f'Fold {fold_id+1} - Validation Metrics')
    axes[1, 0].set_ylim([0, 100])
    for i, (k, v) in enumerate(metrics_data.items()):
        axes[1, 0].text(i, v * 100 + 2, f'{v*100:.2f}%', ha='center', va='bottom')
    
    # 4. Confusion Matrix (Test set)
    cm = test_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title(f'Fold {fold_id+1} - Test Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")


# ==============================================================================
# ONNX EXPORT
# ==============================================================================

def export_to_onnx(model: nn.Module, save_path: str, input_size=(1, 3, IMG_SIZE, IMG_SIZE)):
    model.eval()
    model.to('cpu')
    dummy = torch.randn(input_size)
    torch.onnx.export(
        model, dummy, save_path, export_params=True, opset_version=11,
        do_constant_folding=True, input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX exported to {save_path}")
    try:
        import onnx
        onnx.checker.check_model(onnx.load(save_path))
        print("ONNX check OK")
    except Exception as e:
        print(f"ONNX check: {e}")


# ==============================================================================
# MAIN
# ==============================================================================

def main(args):
    print("\n" + "="*70)
    print("  COPD Binary v4 - Multi-Resolution Wavelet Stacking + Mixup")
    print("="*70)
    print(f"  Device: {DEVICE}  |  Data: {args.data_path}")
    print(f"  Folds: {N_FOLDS}  |  Epochs: {args.epochs}  |  Mixup α: {MIXUP_ALPHA}")
    print(f"  Phase2 LR: {PHASE2_LR}  |  Early-stop patience: {EARLY_STOP_PATIENCE}")
    print("="*70 + "\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_list, labels, patient_ids = load_dataset(args.data_path)
    folds = get_group_kfold_splits(file_list, labels, patient_ids, n_splits=N_FOLDS)
    print(f"GroupKFold: {N_FOLDS} folds (no patient leakage)\n")
    
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    wavelet_transform = MultiResolutionWaveletTransform()
    
    all_histories = []
    all_test_metrics = []
    all_train_metrics = []  # For final report
    all_val_metrics = []    # For final report
    best_f1_fold = -1
    best_f1_score = -1.0
    
    for fold_id in range(N_FOLDS):
        train_idx, val_idx, test_idx = folds[fold_id]
        history, test_metrics, _ = train_fold(
            fold_id, N_FOLDS, train_idx, val_idx, test_idx,
            file_list, labels, patient_ids, args.data_path, output_dir, args,
            normalize_transform, wavelet_transform
        )
        all_histories.append(history)
        all_test_metrics.append(test_metrics)
        
        # Compute train/val metrics for final report
        train_files_fold = [file_list[i] for i in train_idx]
        train_labels_fold = [labels[i] for i in train_idx]
        train_pids_fold = [int(patient_ids[i]) for i in train_idx]
        val_files_fold = [file_list[i] for i in val_idx]
        val_labels_fold = [labels[i] for i in val_idx]
        val_pids_fold = [int(patient_ids[i]) for i in val_idx]
        
        # Load best model to compute train/val metrics
        model_best = COPDClassifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        model_best.load_state_dict(torch.load(output_dir / f"best_model_fold_{fold_id}.pth", map_location=DEVICE))
        model_best.eval()
        
        train_ds_temp = ICBHIBinaryDatasetV4(
            train_files_fold, train_labels_fold, train_pids_fold, args.data_path,
            normalize_transform, wavelet_transform, augment=False, spec_augment=False
        )
        val_ds_temp = ICBHIBinaryDatasetV4(
            val_files_fold, val_labels_fold, val_pids_fold, args.data_path,
            normalize_transform, wavelet_transform, augment=False, spec_augment=False
        )
        train_loader_temp = DataLoader(train_ds_temp, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
        val_loader_temp = DataLoader(val_ds_temp, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
        
        criterion_temp = nn.CrossEntropyLoss()
        _, _, _, train_preds, train_labels_arr = validate(model_best, train_loader_temp, criterion_temp, DEVICE)
        _, _, _, val_preds, val_labels_arr = validate(model_best, val_loader_temp, criterion_temp, DEVICE)
        
        train_metrics = compute_metrics(train_labels_arr, train_preds)
        val_metrics = compute_metrics(val_labels_arr, val_preds)
        
        all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)
        
        # Visualization
        plot_fold_results(history, test_metrics, str(output_dir / f"fold_{fold_id}_results.png"), fold_id)
        
        print(f"  >> Fold {fold_id+1} done | Test Acc: {test_metrics['accuracy']*100:.2f}%  "
              f"F1: {test_metrics['f1_macro']*100:.2f}%  "
              f"Sens: {test_metrics['sensitivity']*100:.2f}%  "
              f"Spec: {test_metrics['specificity']*100:.2f}%")
        
        if test_metrics['f1_macro'] > best_f1_score:
            best_f1_score = test_metrics['f1_macro']
            best_f1_fold = fold_id
        
        del model_best
        cleanup_gpu()
    
    # Final Report
    print("\n" + "="*70)
    print("  5-FOLD CROSS-VALIDATION RESULTS (Mean ± Std)")
    print("="*70)
    
    # Loss
    train_losses = [h['train_loss'][-1] for h in all_histories]
    val_losses = [h['val_loss'][-1] for h in all_histories]
    test_losses = [m.get('loss', 0) for m in all_test_metrics]  # Note: test loss not computed, placeholder
    
    print("\nLoss:")
    print(f"  Train: {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}")
    print(f"  Val:   {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    if any(test_losses):
        print(f"  Test:  {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
    
    # Accuracy
    train_accs = [m['accuracy'] for m in all_train_metrics]
    val_accs = [m['accuracy'] for m in all_val_metrics]
    test_accs = [m['accuracy'] for m in all_test_metrics]
    
    print("\nAccuracy:")
    print(f"  Train: {np.mean(train_accs)*100:.2f}% ± {np.std(train_accs)*100:.2f}%")
    print(f"  Val:   {np.mean(val_accs)*100:.2f}% ± {np.std(val_accs)*100:.2f}%")
    print(f"  Test:  {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")
    
    # Clinical Metrics
    train_f1s = [m['f1_macro'] for m in all_train_metrics]
    val_f1s = [m['f1_macro'] for m in all_val_metrics]
    test_f1s = [m['f1_macro'] for m in all_test_metrics]
    
    test_sens = [m['sensitivity'] for m in all_test_metrics]
    test_spec = [m['specificity'] for m in all_test_metrics]
    test_prec = [m['precision'] for m in all_test_metrics]
    
    print("\nClinical Metrics:")
    print(f"  Macro F1:")
    print(f"    Train: {np.mean(train_f1s)*100:.2f}% ± {np.std(train_f1s)*100:.2f}%")
    print(f"    Val:   {np.mean(val_f1s)*100:.2f}% ± {np.std(val_f1s)*100:.2f}%")
    print(f"    Test:  {np.mean(test_f1s)*100:.2f}% ± {np.std(test_f1s)*100:.2f}%")
    print(f"  Sensitivity (COPD): {np.mean(test_sens)*100:.2f}% ± {np.std(test_sens)*100:.2f}%")
    print(f"  Specificity (Non-COPD): {np.mean(test_spec)*100:.2f}% ± {np.std(test_spec)*100:.2f}%")
    print(f"  Precision: {np.mean(test_prec)*100:.2f}% ± {np.std(test_prec)*100:.2f}%")
    print("="*70)
    
    # Export ONNX from best fold
    print(f"\nExporting ONNX from best F1 fold (fold {best_f1_fold+1})...")
    model_best = COPDClassifier(num_classes=NUM_CLASSES, pretrained=False)
    model_best.load_state_dict(torch.load(output_dir / f"best_model_fold_{best_f1_fold}.pth", map_location='cpu'))
    export_to_onnx(model_best, str(output_dir / "copd_classifier.onnx"))
    del model_best
    cleanup_gpu()
    
    # Save metrics to JSON
    with open(output_dir / "cv_metrics.json", 'w') as f:
        json.dump({
            'train_loss_mean': float(np.mean(train_losses)), 'train_loss_std': float(np.std(train_losses)),
            'val_loss_mean': float(np.mean(val_losses)), 'val_loss_std': float(np.std(val_losses)),
            'train_acc_mean': float(np.mean(train_accs)), 'train_acc_std': float(np.std(train_accs)),
            'val_acc_mean': float(np.mean(val_accs)), 'val_acc_std': float(np.std(val_accs)),
            'test_acc_mean': float(np.mean(test_accs)), 'test_acc_std': float(np.std(test_accs)),
            'train_f1_mean': float(np.mean(train_f1s)), 'train_f1_std': float(np.std(train_f1s)),
            'val_f1_mean': float(np.mean(val_f1s)), 'val_f1_std': float(np.std(val_f1s)),
            'test_f1_mean': float(np.mean(test_f1s)), 'test_f1_std': float(np.std(test_f1s)),
            'test_sensitivity_mean': float(np.mean(test_sens)), 'test_sensitivity_std': float(np.std(test_sens)),
            'test_specificity_mean': float(np.mean(test_spec)), 'test_specificity_std': float(np.std(test_spec)),
            'test_precision_mean': float(np.mean(test_prec)), 'test_precision_std': float(np.std(test_prec)),
            'best_f1_fold': int(best_f1_fold),
            'per_fold': [
                {
                    'train_acc': all_train_metrics[i]['accuracy'],
                    'val_acc': all_val_metrics[i]['accuracy'],
                    'test_acc': all_test_metrics[i]['accuracy'],
                    'train_f1': all_train_metrics[i]['f1_macro'],
                    'val_f1': all_val_metrics[i]['f1_macro'],
                    'test_f1': all_test_metrics[i]['f1_macro'],
                    'test_sensitivity': all_test_metrics[i]['sensitivity'],
                    'test_specificity': all_test_metrics[i]['specificity'],
                    'test_precision': all_test_metrics[i]['precision']
                }
                for i in range(N_FOLDS)
            ],
        }, f, indent=2)
    
    print(f"\nAll outputs saved to {output_dir}. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COPD Binary v4 - Multi-Resolution Wavelet + Mixup")
    parser.add_argument("--data_path", type=str,
                        default="/home/iec/Parallel_Computing_on_FPGA/data/samples/ICBHI_final_database")
    parser.add_argument("--output_dir", type=str, default="./output_copd_v4")
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
