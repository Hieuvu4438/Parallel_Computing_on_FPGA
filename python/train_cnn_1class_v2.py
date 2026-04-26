#!/usr/bin/env python3
"""
================================================================================
CNN Training Script v2 - Binary COPD Classification with 5-Fold GroupKFold
================================================================================

Target: ICBHI 2017 - Binary COPD vs Non-COPD. Goal: approach 98.81% (paper).

Features:
- 5-Fold GroupKFold by patient_id (no data leakage)
- Wavelet CWT (50–1200 Hz, 128 scales) + SpecAugment on spectrogram
- Phase 2: LR 5e-5, CosineAnnealingWarmRestarts; Early stopping on val_loss (patience=20)
- Per-fold and aggregate visualization; best_model_fold_X.pth + ONNX from best F1 fold
- GPU memory cleanup between folds

Usage:
    python train_cnn_1class_v2.py --data_path /path/to/ICBHI_final_database --output_dir ./output_copd_v2
"""

import os
import gc
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import json

import numpy as np
import pandas as pd
import pywt
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_score,
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Optional: SpecAugment via torchaudio
try:
    from torchaudio.transforms import FrequencyMasking, TimeMasking
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS (v2: Wavelet 50–1200 Hz, 128 scales)
# ==============================================================================

IMG_SIZE = 224
TARGET_SR = 4000
WAVELET_NAME = 'morl'
NUM_SCALES = 128
FREQUENCY_RANGE = (50, 1200)  # COPD-focused band

NUM_CLASSES = 2
CLASS_NAMES = ['Non-COPD', 'COPD']
N_FOLDS = 5

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

# Phase 2 learning rate (very low for fine-tuning)
PHASE2_LR = 5e-5
EARLY_STOP_PATIENCE = 20  # on val_loss
PHASE2_MIN_EPOCHS = 15


# ==============================================================================
# WAVELET TRANSFORM (50–1200 Hz, 128 scales)
# ==============================================================================

class WaveletTransform:
    """CWT with Morlet wavelet for time-frequency spectrograms."""

    def __init__(self, wavelet=WAVELET_NAME, num_scales=NUM_SCALES,
                 sample_rate=TARGET_SR, freq_range=FREQUENCY_RANGE,
                 output_size=IMG_SIZE):
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.sample_rate = sample_rate
        self.freq_range = freq_range
        self.output_size = output_size
        self.scales = self._compute_scales()

    def _compute_scales(self) -> np.ndarray:
        center_freq = pywt.central_frequency(self.wavelet)
        min_scale = center_freq * self.sample_rate / self.freq_range[1]
        max_scale = center_freq * self.sample_rate / self.freq_range[0]
        return np.logspace(np.log10(min_scale), np.log10(max_scale), self.num_scales)

    def transform(self, audio: np.ndarray) -> np.ndarray:
        coefficients, _ = pywt.cwt(audio, self.scales, self.wavelet,
                                    sampling_period=1.0 / self.sample_rate)
        power = np.abs(coefficients) ** 2
        power_db = 10 * np.log10(power + 1e-10)
        power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)
        return power_db

    def to_image(self, audio: np.ndarray) -> np.ndarray:
        spec = self.transform(audio)
        zoom_factors = (self.output_size / spec.shape[0],
                        self.output_size / spec.shape[1])
        spec_resized = zoom(spec, zoom_factors, order=1)
        spec_resized = spec_resized[:self.output_size, :self.output_size]
        return np.clip(spec_resized, 0, 1).astype(np.float32)


# ==============================================================================
# SPEC AUGMENT (on spectrogram)
# ==============================================================================

class SpecAugment:
    """Frequency and Time masking on spectrogram (SpecAugment)."""

    def __init__(self, freq_mask_param: int = 20, time_mask_param: int = 20,
                 p: float = 0.5):
        self.p = p
        if HAS_TORCHAUDIO:
            self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_param)
            self.time_mask = TimeMasking(time_mask_param=time_mask_param)
        else:
            self.freq_mask = self.time_mask = None
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: (C, H, W) or (H, W). Apply masking in-place style."""
        if np.random.random() > self.p:
            return spec
        if HAS_TORCHAUDIO and self.freq_mask is not None:
            # (C, H, W) -> last two dims are freq, time
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
            return spec
        # Fallback: manual masking
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
# AUDIO AUGMENTATION (minority class)
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
# DATASET (with SpecAugment option)
# ==============================================================================

class ICBHIBinaryDatasetV2(Dataset):
    """ICBHI binary COPD dataset with optional SpecAugment on spectrogram."""

    def __init__(self, file_list: List[str], labels: List[int], patient_ids: List[int],
                 data_path: str, transform=None, wavelet_transform=None,
                 augment: bool = False, oversample_minority: bool = False,
                 spec_augment: bool = False):
        self.file_list = file_list
        self.labels = labels
        self.patient_ids = patient_ids
        self.data_path = Path(data_path)
        self.transform = transform
        self.wavelet_transform = wavelet_transform or WaveletTransform()
        self.augment = augment
        self.augmenter = AudioAugmenter() if augment else None
        self.spec_augment = spec_augment
        self.spec_aug = SpecAugment(freq_mask_param=20, time_mask_param=20, p=0.5) if spec_augment else None
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
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        if sr != TARGET_SR:
            num_samples = int(len(audio) * TARGET_SR / sr)
            audio = signal.resample(audio, num_samples)
        label = self.labels[idx]
        if self.augment and self.augmenter:
            if label == 0:
                audio = self.augmenter(audio)
            elif np.random.random() < 0.3:
                audio = self.augmenter(audio)
        spec_image = self.wavelet_transform.to_image(audio)
        spec_image = np.stack([spec_image] * 3, axis=0)
        spec_tensor = torch.from_numpy(spec_image).float()
        if self.spec_augment and self.spec_aug is not None:
            spec_tensor = self.spec_aug(spec_tensor)
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
        return spec_tensor, label


# ==============================================================================
# DATA LOADING & GROUP K-FOLD
# ==============================================================================

def parse_filename(filename: str) -> Tuple[int, str]:
    basename = Path(filename).stem
    parts = basename.split('_')
    patient_id = int(parts[0]) if len(parts) >= 1 else 0
    return patient_id, basename


def load_diagnosis_mapping(data_path: str) -> Dict[int, str]:
    diagnosis_file = Path(data_path) / "ICBHI_Challenge_diagnosis.txt"
    if diagnosis_file.exists():
        out = {}
        with open(diagnosis_file, 'r') as f:
            for line in f:
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
        diagnosis = patient_diagnosis.get(pid, '')
        label = get_binary_label(diagnosis)
        file_list.append(wav_file.name)
        labels.append(label)
        patient_ids.append(pid)
    patient_ids = np.array(patient_ids)
    print(f"Loaded {len(file_list)} files, {len(np.unique(patient_ids))} patients")
    return file_list, labels, patient_ids


def get_group_kfold_splits(file_list: List[str], labels: List[int],
                           patient_ids: np.ndarray, n_splits: int = N_FOLDS,
                           val_ratio: float = 0.15, random_state: int = 42
                           ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    GroupKFold by patient_id. For each fold, further split train into train/val by patient.
    Returns list of (train_idx, val_idx, test_idx) so no patient appears in two sets.
    """
    X = np.arange(len(file_list))
    y = np.array(labels)
    groups = patient_ids
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for train_val_idx, test_idx in gkf.split(X, y, groups):
        train_val_groups = groups[train_val_idx]
        unique_patients = np.unique(train_val_groups)
        patient_to_label = dict(zip(unique_patients, [y[train_val_idx][np.where(train_val_groups == p)[0][0]] for p in unique_patients]))
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
# TRAINING & VALIDATION
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, pred = out.max(1)
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': acc, 'f1_macro': f1_macro,
        'sensitivity': sensitivity, 'specificity': specificity,
        'confusion_matrix': cm,
    }


# ==============================================================================
# GPU MEMORY CLEANUP (between folds)
# ==============================================================================

def cleanup_gpu():
    """Free GPU memory between folds to avoid OOM in K-Fold runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ==============================================================================
# TRAIN ONE FOLD
# ==============================================================================

def train_fold(fold_id: int, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
               file_list: List[str], labels: List[int], patient_ids: np.ndarray,
               data_path: str, output_dir: Path, args,
               normalize_transform, wavelet_transform) -> Tuple[Dict, Dict[str, Any], int]:
    """
    Train one fold. Returns (history, test_metrics, total_epochs_run).
    Saves best_model_fold_{fold_id}.pth in output_dir.
    """
    train_files = [file_list[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_pids = [int(patient_ids[i]) for i in train_idx]
    val_files = [file_list[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_pids = [int(patient_ids[i]) for i in val_idx]
    test_files = [file_list[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_pids = [int(patient_ids[i]) for i in test_idx]

    train_ds = ICBHIBinaryDatasetV2(
        train_files, train_labels, train_pids, data_path,
        normalize_transform, wavelet_transform,
        augment=True, oversample_minority=True, spec_augment=True
    )
    val_ds = ICBHIBinaryDatasetV2(
        val_files, val_labels, val_pids, data_path,
        normalize_transform, wavelet_transform, augment=False, spec_augment=False
    )
    test_ds = ICBHIBinaryDatasetV2(
        test_files, test_labels, test_pids, data_path,
        normalize_transform, wavelet_transform, augment=False, spec_augment=False
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = COPDClassifier(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    class_weights = torch.FloatTensor([
        total / (NUM_CLASSES * class_counts.get(i, 1)) for i in range(NUM_CLASSES)
    ]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_loss = float('inf')
    patience_counter = 0
    phase1_epochs = min(30, args.epochs // 5)

    # Phase 1: freeze backbone
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    for epoch in range(phase1_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / f"best_model_fold_{fold_id}.pth")
            patience_counter = 0
        else:
            patience_counter += 1

    # Phase 2: unfreeze, LR=5e-5, CosineAnnealingWarmRestarts, early stop on val_loss (patience=20)
    model.unfreeze_backbone()
    patience_counter = 0  # reset
    phase2_epochs = args.epochs - phase1_epochs
    optimizer = optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    total_epochs_run = phase1_epochs
    for epoch in range(phase1_epochs, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        scheduler.step()
        total_epochs_run = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / f"best_model_fold_{fold_id}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        phase2_elapsed = (epoch + 1) - phase1_epochs
        if phase2_elapsed >= PHASE2_MIN_EPOCHS and patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (val_loss, patience={EARLY_STOP_PATIENCE})")
            break

    # Evaluate on test set
    model.load_state_dict(torch.load(output_dir / f"best_model_fold_{fold_id}.pth", map_location=DEVICE))
    model.eval()
    _, _, _, test_preds, test_labels_arr = validate(model, test_loader, criterion, DEVICE)
    test_metrics = compute_metrics(test_labels_arr, test_preds)
    return history, test_metrics, total_epochs_run


# ==============================================================================
# VISUALIZATION: per-fold + aggregate
# ==============================================================================

def plot_fold_history(history: Dict, save_path: str, fold_id: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', lw=1.5)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', lw=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Fold {fold_id} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', lw=1.5)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', lw=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Fold {fold_id} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def plot_aggregate_curves(all_histories: List[Dict], save_path: str):
    """Average Loss and Accuracy across folds (mean ± std)."""
    max_len = max(len(h['train_loss']) for h in all_histories)
    train_loss = np.full((len(all_histories), max_len), np.nan)
    val_loss = np.full((len(all_histories), max_len), np.nan)
    train_acc = np.full((len(all_histories), max_len), np.nan)
    val_acc = np.full((len(all_histories), max_len), np.nan)
    for i, h in enumerate(all_histories):
        n = len(h['train_loss'])
        train_loss[i, :n] = h['train_loss']
        val_loss[i, :n] = h['val_loss']
        train_acc[i, :n] = h['train_acc']
        val_acc[i, :n] = h['val_acc']
    train_loss_mean = np.nanmean(train_loss, axis=0)
    train_loss_std = np.nanstd(train_loss, axis=0)
    val_loss_mean = np.nanmean(val_loss, axis=0)
    val_loss_std = np.nanstd(val_loss, axis=0)
    train_acc_mean = np.nanmean(train_acc, axis=0)
    train_acc_std = np.nanstd(train_acc, axis=0)
    val_acc_mean = np.nanmean(val_acc, axis=0)
    val_acc_std = np.nanstd(val_acc, axis=0)
    epochs = np.arange(1, max_len + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss_mean, 'b-', label='Train (mean)', lw=1.5)
    axes[0].fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2, color='b')
    axes[0].plot(epochs, val_loss_mean, 'r-', label='Val (mean)', lw=1.5)
    axes[0].fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2, color='r')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Average Loss (Mean ± Std)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, train_acc_mean, 'b-', label='Train (mean)', lw=1.5)
    axes[1].fill_between(epochs, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, alpha=0.2, color='b')
    axes[1].plot(epochs, val_acc_mean, 'r-', label='Val (mean)', lw=1.5)
    axes[1].fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, alpha=0.2, color='r')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Average Accuracy (Mean ± Std)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Aggregate plot saved to {save_path}")


# ==============================================================================
# ONNX EXPORT
# ==============================================================================

def export_to_onnx(model: nn.Module, save_path: str, input_size: Tuple[int, ...] = (1, 3, IMG_SIZE, IMG_SIZE)):
    model.eval()
    model.to('cpu')
    dummy = torch.randn(input_size)
    torch.onnx.export(
        model, dummy, save_path,
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
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
# MAIN: 5-Fold GroupKFold, report Mean ± Std, save best ONNX from best F1 fold
# ==============================================================================

def main(args):
    print("\n" + "="*60)
    print("COPD Binary Classification v2 - 5-Fold GroupKFold")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Data: {args.data_path}")
    print(f"Folds: {N_FOLDS}, Phase2 LR: {PHASE2_LR}, Early stop (val_loss) patience: {EARLY_STOP_PATIENCE}")
    print("="*60 + "\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_list, labels, patient_ids = load_dataset(args.data_path)
    labels_arr = np.array(labels)
    folds = get_group_kfold_splits(file_list, labels, patient_ids, n_splits=N_FOLDS)
    print(f"GroupKFold: {N_FOLDS} folds (no patient leakage)\n")

    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    wavelet_transform = WaveletTransform()

    all_histories = []
    all_test_metrics = []
    best_f1_fold = -1
    best_f1_score = -1.0

    for fold_id in range(N_FOLDS):
        train_idx, val_idx, test_idx = folds[fold_id]
        print(f"\n--- Fold {fold_id + 1}/{N_FOLDS} ---")
        history, test_metrics, _ = train_fold(
            fold_id, train_idx, val_idx, test_idx,
            file_list, labels, patient_ids, args.data_path, output_dir, args,
            normalize_transform, wavelet_transform
        )
        all_histories.append(history)
        all_test_metrics.append(test_metrics)
        plot_fold_history(history, str(output_dir / f"fold_{fold_id}_curves.png"), fold_id)
        print(f"  Fold {fold_id} Test Acc: {test_metrics['accuracy']*100:.2f}%  F1: {test_metrics['f1_macro']*100:.2f}%  Sens: {test_metrics['sensitivity']*100:.2f}%  Spec: {test_metrics['specificity']*100:.2f}%")
        if test_metrics['f1_macro'] > best_f1_score:
            best_f1_score = test_metrics['f1_macro']
            best_f1_fold = fold_id
        cleanup_gpu()

    plot_aggregate_curves(all_histories, str(output_dir / "average_curves.png"))

    # Mean ± Std
    accs = [m['accuracy'] for m in all_test_metrics]
    f1s = [m['f1_macro'] for m in all_test_metrics]
    sens = [m['sensitivity'] for m in all_test_metrics]
    spec = [m['specificity'] for m in all_test_metrics]
    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION RESULTS (Mean ± Std)")
    print("="*60)
    print(f"  Accuracy:    {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  Macro F1:    {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    print(f"  Sensitivity: {np.mean(sens)*100:.2f}% ± {np.std(sens)*100:.2f}%")
    print(f"  Specificity: {np.mean(spec)*100:.2f}% ± {np.std(spec)*100:.2f}%")
    print("="*60)

    # ONNX from best F1 fold
    print(f"\nExporting ONNX from best F1 fold (fold {best_f1_fold})...")
    model_best = COPDClassifier(num_classes=NUM_CLASSES, pretrained=False)
    model_best.load_state_dict(torch.load(output_dir / f"best_model_fold_{best_f1_fold}.pth", map_location='cpu'))
    export_to_onnx(model_best, str(output_dir / "copd_classifier.onnx"))
    del model_best
    cleanup_gpu()

    with open(output_dir / "cv_metrics.json", 'w') as f:
        json.dump({
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'f1_macro_mean': float(np.mean(f1s)),
            'f1_macro_std': float(np.std(f1s)),
            'sensitivity_mean': float(np.mean(sens)),
            'sensitivity_std': float(np.std(sens)),
            'specificity_mean': float(np.mean(spec)),
            'specificity_std': float(np.std(spec)),
            'best_f1_fold': int(best_f1_fold),
            'per_fold': [
                {'accuracy': m['accuracy'], 'f1_macro': m['f1_macro'],
                 'sensitivity': m['sensitivity'], 'specificity': m['specificity']}
                for m in all_test_metrics
            ],
        }, f, indent=2)
    print(f"\nAll outputs in {output_dir}. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COPD Binary v2 - 5-Fold GroupKFold")
    parser.add_argument("--data_path", type=str,
                        default="/home/iec/Parallel_Computing_on_FPGA/data/samples/ICBHI_final_database")
    parser.add_argument("--output_dir", type=str, default="./output_copd_v2")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
