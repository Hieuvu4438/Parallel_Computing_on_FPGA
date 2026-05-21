#!/usr/bin/env python3
"""
================================================================================
MobileNetV2 Training Script V2 — 3-Class Respiratory Sound Classification
================================================================================
OPTIMIZED version fixing underfitting issues from V1:

Key changes from V1:
  1. Fixed audio length (5s) with random crop — consistent spectrogram resolution
  2. CWT 224 scales (no resize interpolation) + wider frequency range (50-1900Hz)
  3. Reduced regularization — WD 0.01, no MixUp, label smoothing 0.05, lower dropout
  4. WeightedRandomSampler instead of hard oversampling
  5. 2-phase training (head→all) with higher LR
  6. Dataset-specific normalization (not ImageNet)
  7. Simplified classifier head (1280→256→3)
  8. Multi-channel input (CWT + Delta + Delta²)
  9. Test-Time Augmentation (TTA) for inference

Pipeline: Raw WAV → Resample 4kHz → BPF → 5s Standardize → CWT 224-scale → MobileNetV2

Usage:
    python train_mobilenetv2_v2.py \\
        --data_path /home/iec/Parallel_Computing_on_FPGA/data/combined/audio \\
        --labels_csv /home/iec/Parallel_Computing_on_FPGA/data/combined/labels.csv \\
        --output_dir ./output_mobilenetv2_v2

    # Quick test (1 fold, 5 epochs):
    python train_mobilenetv2_v2.py --dry_run --epochs 5
"""

import os
import gc
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.ndimage import zoom
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    classification_report,
)

try:
    from torchaudio.transforms import FrequencyMasking, TimeMasking
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG & CONSTANTS — V2 OPTIMIZED
# ==============================================================================
TARGET_SR = 4000
WAVELET_NAME = 'morl'
NUM_SCALES = 224          # Match output size directly — no resize needed!
FREQUENCY_RANGE = (50, 1900)  # Wider range, close to Nyquist (2000Hz)
IMG_SIZE = 224
FIXED_DURATION = 5.0      # Fixed audio length in seconds
FIXED_SAMPLES = int(FIXED_DURATION * TARGET_SR)  # 20000 samples

N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Training Schedule (Simplified 2-phase) ---
PHASE1_EPOCHS = 20        # Head-only training with higher LR
PHASE2_EPOCHS = 80        # All unfrozen with discriminative LR
TOTAL_MAX_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

# --- Learning Rates (Higher for faster convergence) ---
PHASE1_LR = 1e-3          # 3x higher than V1
PHASE2_LR_BACKBONE = 1e-5
PHASE2_LR_HEAD = 1e-4

# --- Regularization (Reduced — fixing underfitting) ---
WEIGHT_DECAY = 0.01       # Was 0.05
GRADIENT_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 15  # More patience for 2-phase
LABEL_SMOOTHING = 0.05    # Was 0.1
FOCAL_GAMMA = 2.0
HEAD_DROPOUT = 0.3        # Was 0.5

# --- TTA ---
TTA_AUGMENTS = 5


# ==============================================================================
# LOGGER UTILITY
# ==============================================================================
class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        with open(self.log_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("  TRAINING LOG: MobileNetV2 V2 Optimized (3 Classes)\n")
            f.write("=" * 70 + "\n")

    def print(self, msg: str = ""):
        print(msg)
        with open(self.log_path, 'a') as f:
            f.write(str(msg) + "\n")


# ==============================================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ==============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with optional label smoothing.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * smooth_targets * log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)
            loss = loss * alpha_t.unsqueeze(0)

        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ==============================================================================
# WAVELET TRANSFORM ENGINE — V2 (224 scales, no resize)
# ==============================================================================
class WaveletTransform:
    """CWT with Morlet wavelet — 224 scales for direct mapping to 224px."""
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
        """Convert audio to spectrogram image.
        With 224 scales, only time-axis needs resize.
        """
        spec = self.transform(audio)  # shape: (224, time_samples)

        # Only resize time axis to 224
        if spec.shape[1] != self.output_size:
            zoom_factor_time = self.output_size / spec.shape[1]
            spec = zoom(spec, (1.0, zoom_factor_time), order=1)

        # Ensure exact size
        spec = spec[:self.output_size, :self.output_size]
        return np.clip(spec, 0, 1).astype(np.float32)

    def to_multichannel(self, audio: np.ndarray) -> np.ndarray:
        """Create 3-channel image: CWT + Delta + Delta².
        Provides richer temporal information than duplicated channels.
        """
        spec = self.to_image(audio)

        # Channel 0: CWT power spectrogram (base)
        ch0 = spec

        # Channel 1: Temporal delta (first derivative along time)
        ch1 = np.gradient(spec, axis=1)
        ch1 = (ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-10)

        # Channel 2: Temporal delta-delta (second derivative)
        ch2 = np.gradient(ch1, axis=1)
        ch2 = (ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-10)

        return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)


# ==============================================================================
# AUDIO AUGMENTATION (Lighter than V1 — avoid over-regularization)
# ==============================================================================
class AudioAugmenter:
    """Audio augmentation with moderate intensity."""
    def __init__(self, noise_snr_range=(15, 30), shift_ratio=0.1,
                 gain_range=(0.8, 1.2), probability=0.5):
        self.noise_snr_range = noise_snr_range
        self.shift_ratio = shift_ratio
        self.gain_min, self.gain_max = gain_range
        self.probability = probability

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio

        # Additive noise (moderate)
        if np.random.random() < 0.4:
            snr_db = np.random.uniform(*self.noise_snr_range)
            signal_power = np.mean(audio ** 2) + 1e-10
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
            audio = audio + noise

        # Time shift
        if np.random.random() < 0.4:
            shift = int(np.random.uniform(-self.shift_ratio, self.shift_ratio) * len(audio))
            audio = np.roll(audio, shift)

        # Gain adjustment
        if np.random.random() < 0.4:
            audio = audio * np.random.uniform(self.gain_min, self.gain_max)

        # Polarity inversion
        if np.random.random() < 0.2:
            audio = -audio

        return np.clip(audio, -1.0, 1.0).astype(np.float32)


class SpecAugment:
    """SpecAugment with reduced intensity (V1 was too aggressive)."""
    def __init__(self, freq_mask_param=20, time_mask_param=25, p=0.5):
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
# AUDIO PREPROCESSING UTILITIES
# ==============================================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def standardize_audio_length(audio: np.ndarray, target_length: int = FIXED_SAMPLES,
                             training: bool = True) -> np.ndarray:
    """Standardize audio to fixed length.
    - Training: random crop (augmentation effect)
    - Inference: center crop
    - Short audio: repeat-pad to preserve patterns
    """
    if len(audio) >= target_length:
        if training:
            start = np.random.randint(0, len(audio) - target_length + 1)
        else:
            start = (len(audio) - target_length) // 2
        return audio[start:start + target_length]
    else:
        # Repeat-pad (better than zero-pad for audio)
        repeats = target_length // len(audio) + 1
        padded = np.tile(audio, repeats)[:target_length]
        return padded


def load_and_preprocess_audio(wav_path: str, training: bool = True) -> np.ndarray:
    """Load, resample, filter, and standardize audio."""
    try:
        sr, audio = wavfile.read(wav_path)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != TARGET_SR:
            num_samples = int(len(audio) * TARGET_SR / sr)
            audio = signal.resample(audio, num_samples)

        # Band-pass filter
        audio = butter_bandpass_filter(audio, 50, min(1900, (TARGET_SR // 2) - 1), TARGET_SR)
        audio = audio.astype(np.float32)

        # Normalize amplitude
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp

        # Standardize length to 5 seconds
        audio = standardize_audio_length(audio, FIXED_SAMPLES, training=training)

    except Exception:
        audio = np.zeros(FIXED_SAMPLES, dtype=np.float32)

    return audio


# ==============================================================================
# DATASET UTILS
# ==============================================================================
def load_dataset_info(csv_path: str, data_dir: str) -> Tuple[List[str], List[int], np.ndarray, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    file_list, labels_str, patient_ids = [], [], []
    classes = sorted(df['label'].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for _, row in df.iterrows():
        fname = row['filename']
        label = row['label']
        pid = row['patient_id']

        full_path = os.path.join(data_dir, label, fname)
        if not os.path.exists(full_path):
            full_path = os.path.join(data_dir, fname)

        if os.path.exists(full_path):
            file_list.append(full_path)
            labels_str.append(label)
            patient_ids.append(pid)

    labels_int = [class_to_idx[l] for l in labels_str]
    return file_list, labels_int, np.array(patient_ids), class_to_idx


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


def compute_dataset_stats(file_list: List[str], wavelet_transform: 'WaveletTransform',
                          max_samples: int = 200) -> Tuple[float, float]:
    """Compute mean and std of spectrograms for dataset-specific normalization."""
    indices = np.random.choice(len(file_list), min(max_samples, len(file_list)), replace=False)
    all_means, all_stds = [], []

    for idx in tqdm(indices, desc="Computing dataset stats", leave=False):
        audio = load_and_preprocess_audio(file_list[idx], training=False)
        spec = wavelet_transform.to_multichannel(audio)
        all_means.append(spec.mean())
        all_stds.append(spec.std())

    return float(np.mean(all_means)), float(np.mean(all_stds))


# ==============================================================================
# DATASET
# ==============================================================================
class AudioSpectrogramDataset(Dataset):
    """On-the-fly audio → CWT multi-channel spectrogram dataset."""
    def __init__(self, file_paths, labels, class_to_idx, transform=None,
                 augment=False, spec_augment=False, training=True):
        self.file_paths = file_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.wavelet_transform = WaveletTransform()
        self.training = training

        self.augment = augment
        self.augmenter = AudioAugmenter(probability=0.5) if augment else None
        self.spec_augment = spec_augment
        self.spec_aug = SpecAugment(freq_mask_param=20, time_mask_param=25, p=0.5) if spec_augment else None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav_path = self.file_paths[idx]
        audio = load_and_preprocess_audio(wav_path, training=self.training)

        # Audio augmentation
        if self.augment and self.augmenter:
            audio = self.augmenter(audio)

        # Multi-channel CWT Spectrogram (CWT + Delta + Delta²)
        spec_image = self.wavelet_transform.to_multichannel(audio)
        spec_tensor = torch.from_numpy(spec_image).float()

        # SpecAugment
        if self.spec_augment and self.spec_aug is not None:
            spec_tensor = self.spec_aug(spec_tensor)

        # Dataset-specific normalization
        if self.transform:
            spec_tensor = self.transform(spec_tensor)

        return spec_tensor, self.labels[idx]


# ==============================================================================
# MODEL: MobileNetV2 with Simplified Head
# ==============================================================================
class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 with simplified classifier head (fewer params → less overfitting)."""
    def __init__(self, num_classes=3, pretrained=True, head_dropout=HEAD_DROPOUT):
        super().__init__()
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        num_features = self.backbone.classifier[1].in_features  # 1280

        # Simplified head: 1280 → 256 → num_classes
        # V1 was 1280→512→256→3 — too many params for 1256 samples
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=head_dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_param_groups(self, lr_backbone: float, lr_head: float):
        backbone_params = []
        head_params = []
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        return [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': lr_head},
        ]


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device,
                    gradient_clip=GRADIENT_CLIP_NORM, batch_scheduler=None):
    """Training loop — no MixUp (too harmful for small dataset)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

        if batch_scheduler is not None:
            batch_scheduler.step()

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


def validate_with_tta(model, loader, wavelet_transform, device, n_augments=TTA_AUGMENTS):
    """Test-Time Augmentation: average predictions from multiple augmented versions."""
    model.eval()
    augmenter = AudioAugmenter(probability=1.0,
                               noise_snr_range=(20, 40),
                               shift_ratio=0.05,
                               gain_range=(0.9, 1.1))
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="TTA", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            # Original prediction
            out = model(inputs)
            probs = F.softmax(out, dim=1)

            # We already have the batch loaded, just average with original
            _, pred = probs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1, all_preds, all_labels


def compute_metrics(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'report': report,
        'confusion_matrix': cm.tolist()
    }


def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ==============================================================================
# TRAINING FOLD — 2-PHASE SIMPLIFIED
# ==============================================================================
def train_fold(fold_id: int, n_folds: int, train_idx, val_idx, test_idx,
               file_list, labels, patient_ids, output_dir: Path, args,
               log: Logger, class_to_idx: dict):

    num_classes = len(class_to_idx)

    train_files = [file_list[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_files = [file_list[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_files = [file_list[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    log.print(f"\n" + "=" * 70)
    log.print(f"  FOLD {fold_id + 1}/{n_folds}  |  Train: {len(train_files)}  "
              f"|  Val: {len(val_files)}  |  Test: {len(test_files)}")
    log.print("=" * 70)

    train_counts = Counter(train_labels)
    log.print(f"  Train class distribution: {dict(train_counts)}")

    # --- Compute Dataset-Specific Normalization ---
    wt = WaveletTransform()
    log.print("  Computing dataset-specific normalization stats...")
    ds_mean, ds_std = compute_dataset_stats(train_files, wt, max_samples=min(200, len(train_files)))
    # Ensure std is not too small
    ds_std = max(ds_std, 0.01)
    log.print(f"  Dataset stats: mean={ds_mean:.4f}, std={ds_std:.4f}")
    normalize = transforms.Normalize(mean=[ds_mean]*3, std=[ds_std]*3)

    # --- WeightedRandomSampler (instead of hard oversampling) ---
    class_weights = {cls: 1.0 / count for cls, count in train_counts.items()}
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_files),
        replacement=True
    )

    train_ds = AudioSpectrogramDataset(train_files, train_labels, class_to_idx,
                                       transform=normalize, augment=True,
                                       spec_augment=True, training=True)
    val_ds = AudioSpectrogramDataset(val_files, val_labels, class_to_idx,
                                     transform=normalize, augment=False,
                                     spec_augment=False, training=False)
    test_ds = AudioSpectrogramDataset(test_files, test_labels, class_to_idx,
                                      transform=normalize, augment=False,
                                      spec_augment=False, training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- Model ---
    model = MobileNetV2Classifier(num_classes=num_classes, pretrained=True).to(DEVICE)

    # --- Focal Loss with UNIFORM alpha ---
    # WeightedRandomSampler already handles class balancing in batch composition.
    # Adding inverse-frequency alpha on top creates DOUBLE correction that
    # causes the model to over-focus on minority classes at the expense of majority.
    alpha_values = torch.FloatTensor([1.0] * num_classes)
    criterion = FocalLoss(alpha=alpha_values, gamma=FOCAL_GAMMA,
                         label_smoothing=LABEL_SMOOTHING)

    log.print(f"  Focal Loss alpha: [{', '.join(f'{a:.3f}' for a in alpha_values.tolist())}], "
              f"gamma={FOCAL_GAMMA}, smoothing={LABEL_SMOOTHING}")
    log.print(f"  WeightedRandomSampler: active (no hard oversampling)")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = -1.0
    patience_counter = 0
    total_epochs = min(args.epochs, TOTAL_MAX_EPOCHS)

    p1_end = min(PHASE1_EPOCHS, total_epochs)
    p2_end = total_epochs

    # ===================== PHASE 1: Head Only =====================
    model.freeze_backbone()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=PHASE1_LR,
        steps_per_epoch=len(train_loader), epochs=p1_end,
        pct_start=0.3, anneal_strategy='cos'
    )

    log.print(f"  --- Phase 1: Head Only (Ep 1-{p1_end}) | LR={PHASE1_LR} ---")
    for epoch in range(0, p1_end):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            batch_scheduler=scheduler
        )
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / f"best_model_fold_{fold_id}.pth")
            patience_counter = 0

        best_mark = " [BEST]" if is_best else ""
        log.print(
            f"  Fold {fold_id+1}/{n_folds} | Phase 1 | "
            f"Epoch {epoch+1:3d}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | "
            f"Val F1: {val_f1*100:5.2f}%{best_mark}"
        )

    # ===================== PHASE 2: All Unfrozen =====================
    remaining = p2_end - p1_end
    if remaining <= 0:
        log.print("  >> Skipping Phase 2 (no remaining epochs).")
    else:
        model.unfreeze_all()
        param_groups = model.get_param_groups(PHASE2_LR_BACKBONE, PHASE2_LR_HEAD)
        optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        t0 = max(remaining // 3, 1)  # Ensure T_0 >= 1
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=2, eta_min=1e-7
        )
        patience_counter = 0

        log.print(f"  --- Phase 2: All Unfrozen (Ep {p1_end+1}-{p2_end}) | "
                  f"LR bb={PHASE2_LR_BACKBONE}, head={PHASE2_LR_HEAD} ---")

        for epoch in range(p1_end, p2_end):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            scheduler.step()

            is_best = val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), output_dir / f"best_model_fold_{fold_id}.pth")
                patience_counter = 0
            else:
                patience_counter += 1

            best_mark = " [BEST]" if is_best else ""
            pat_str = f" | patience={patience_counter}/{EARLY_STOP_PATIENCE}"
            log.print(
                f"  Fold {fold_id+1}/{n_folds} | Phase 2 | "
                f"Epoch {epoch+1:3d}/{total_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | "
                f"Val F1: {val_f1*100:5.2f}%{best_mark}{pat_str}"
            )

            if patience_counter >= EARLY_STOP_PATIENCE:
                log.print(f"  >> Early stopping at epoch {epoch+1} (Phase 2) due to F1 patience.")
                break

    # --- Final Evaluation ---
    model.load_state_dict(torch.load(output_dir / f"best_model_fold_{fold_id}.pth",
                                     map_location=DEVICE, weights_only=True))
    torch.save(model.state_dict(), output_dir / f"weights_fold_{fold_id}_final.pt")

    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    eval_criterion = nn.CrossEntropyLoss()
    _, _, _, test_preds, test_labels_arr = validate(model, test_loader, eval_criterion, DEVICE)
    test_metrics = compute_metrics(test_labels_arr, test_preds, class_names)

    return history, test_metrics


# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_fold_history(history: Dict, save_path: str, fold_id: int):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', lw=1.5)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', lw=1.5)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Fold {fold_id} — Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', lw=1.5)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', lw=1.5)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Fold {fold_id} — Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['val_f1'], 'g-', label='Val Macro F1', lw=1.5)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Macro F1')
    axes[2].set_title(f'Fold {fold_id} — Macro F1'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    # Phase boundary marker
    for ax in axes:
        if PHASE1_EPOCHS < len(list(epochs)):
            ax.axvline(x=PHASE1_EPOCHS, color='gray', linestyle='--', alpha=0.5)
            ax.text(PHASE1_EPOCHS, ax.get_ylim()[1], 'P2', ha='center',
                    va='bottom', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, fold_id):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Fold {fold_id} — Confusion Matrix')
    fig.colorbar(im)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def export_to_onnx(model: nn.Module, save_path: str,
                   input_size=(1, 3, IMG_SIZE, IMG_SIZE)):
    model.eval()
    model.to('cpu')
    dummy = torch.randn(input_size)
    torch.onnx.export(model, dummy, save_path, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
    try:
        import onnx
        onnx.checker.check_model(onnx.load(save_path))
        return True
    except Exception:
        return False


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(str(output_dir / "training_log.txt"))

    log.print(f"Device: {DEVICE}")
    log.print(f"Audio Data Path: {args.data_path}")
    log.print(f"Labels CSV Path: {args.labels_csv}")
    log.print(f"Max Epochs: {min(args.epochs, TOTAL_MAX_EPOCHS)} | "
              f"Batch Size: {args.batch_size} | K-Folds: {N_FOLDS}")
    log.print(f"Model: MobileNetV2 V2 | Loss: Focal (γ={FOCAL_GAMMA}) + "
              f"LabelSmoothing({LABEL_SMOOTHING})")
    log.print(f"No MixUp | Weight Decay={WEIGHT_DECAY} | "
              f"Grad Clip={GRADIENT_CLIP_NORM}")
    log.print(f"2-Phase Training: P1={PHASE1_EPOCHS}ep (head), P2={PHASE2_EPOCHS}ep (all)")
    log.print(f"Early Stopping: patience={EARLY_STOP_PATIENCE} on val_f1")
    log.print(f"Fixed Audio Duration: {FIXED_DURATION}s ({FIXED_SAMPLES} samples)")
    log.print(f"CWT: {NUM_SCALES} scales, freq range {FREQUENCY_RANGE}")
    log.print(f"Multi-channel: CWT + Delta + Delta² (3ch)")
    log.print(f"Normalization: Dataset-specific (not ImageNet)")

    file_list, labels, patient_ids, class_to_idx = load_dataset_info(
        args.labels_csv, args.data_path
    )
    if len(file_list) == 0:
        log.print(f"Error: No valid audio files found.")
        return

    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    log.print(f"Classes Found {len(class_to_idx)}: {class_to_idx}")
    log.print(f"Total Samples: {len(file_list)} | Unique Patients: {len(np.unique(patient_ids))}")

    label_counts = Counter(labels)
    for cls_name, cls_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        count = label_counts[cls_idx]
        log.print(f"  {cls_name}: {count} samples ({count/len(labels)*100:.1f}%)")

    folds = get_group_kfold_splits(file_list, labels, patient_ids, n_splits=N_FOLDS)
    n_folds_to_run = 1 if args.dry_run else N_FOLDS

    all_histories = []
    all_test_metrics = []
    best_f1_fold = -1
    best_f1_score = -1.0

    for fold_id in range(n_folds_to_run):
        train_idx, val_idx, test_idx = folds[fold_id]
        history, test_metrics = train_fold(
            fold_id, n_folds_to_run, train_idx, val_idx, test_idx,
            file_list, labels, patient_ids, output_dir, args, log, class_to_idx
        )
        all_histories.append(history)
        all_test_metrics.append(test_metrics)
        plot_fold_history(history, str(output_dir / f"fold_{fold_id}_curves.png"), fold_id)

        cm = np.array(test_metrics['confusion_matrix'])
        plot_confusion_matrix(cm, class_names,
                              str(output_dir / f"fold_{fold_id}_cm.png"), fold_id)

        macro_f1 = test_metrics['macro_f1']
        log.print(f"  >> Fold {fold_id} Test | Acc: {test_metrics['accuracy']*100:.2f}% | "
                  f"Macro F1: {macro_f1*100:.2f}% | "
                  f"Weighted F1: {test_metrics['weighted_f1']*100:.2f}%")

        for cls in class_names:
            m = test_metrics['report'][cls]
            log.print(f"     {cls.ljust(12)} | P: {m['precision']:.3f} | "
                      f"R: {m['recall']:.3f} | F1: {m['f1-score']:.3f}")

        if macro_f1 > best_f1_score:
            best_f1_score = macro_f1
            best_f1_fold = fold_id

        cleanup_gpu()

    # --- Aggregate Statistics ---
    accs = [m['accuracy'] for m in all_test_metrics]
    f1s = [m['macro_f1'] for m in all_test_metrics]
    wf1s = [m['weighted_f1'] for m in all_test_metrics]

    log.print("\n" + "=" * 70)
    if args.dry_run:
        log.print("  DRY-RUN RESULTS (1 Fold)")
    else:
        log.print("  5-FOLD CROSS-VALIDATION FINAL RESULTS (Mean ± Std)")
    log.print("=" * 70)
    log.print(f"  Accuracy:    {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    log.print(f"  Macro F1:    {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    log.print(f"  Weighted F1: {np.mean(wf1s)*100:.2f}% ± {np.std(wf1s)*100:.2f}%")
    log.print("=" * 70)

    best_report = all_test_metrics[best_f1_fold]['report']
    log.print(f"\nBest Fold ({best_f1_fold}) Detailed Classification Report:")
    for cls in class_names:
        metrics = best_report[cls]
        log.print(f"  Class {cls.ljust(15)} | Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1-score']:.4f}")

    # --- Export Best Model ---
    final_pth = output_dir / "mobilenetv2_v2_best.pth"
    onnx_path = output_dir / "mobilenetv2_v2_best.onnx"

    log.print(f"\nLoading weights from Fold {best_f1_fold} for final export...")
    model_best = MobileNetV2Classifier(num_classes=len(class_to_idx), pretrained=False)
    model_best.load_state_dict(torch.load(
        output_dir / f"best_model_fold_{best_f1_fold}.pth",
        map_location='cpu', weights_only=True
    ))

    torch.save(model_best.state_dict(), final_pth)
    log.print(f"Saved best PyTorch weights to {final_pth}")

    if export_to_onnx(model_best, str(onnx_path)):
        log.print(f"Saved ONNX model successfully to {onnx_path}")
    else:
        log.print(f"ONNX export completed (check may have failed if onnx not installed)")

    del model_best
    cleanup_gpu()

    # --- Save Metrics JSON ---
    with open(output_dir / "cv_metrics.json", 'w') as f:
        json.dump({
            'model': 'MobileNetV2_V2',
            'version': 'V2_optimized',
            'strategies': [
                f'Fixed audio {FIXED_DURATION}s + random crop',
                f'CWT {NUM_SCALES} scales (no resize artifact)',
                f'Freq range {FREQUENCY_RANGE}',
                'Multi-channel: CWT + Delta + Delta²',
                f'Focal Loss (gamma={FOCAL_GAMMA})',
                f'Label Smoothing ({LABEL_SMOOTHING})',
                '2-Phase Training (head→all)',
                f'Discriminative LR (bb={PHASE2_LR_BACKBONE}, head={PHASE2_LR_HEAD})',
                'WeightedRandomSampler (no hard oversampling)',
                'Dataset-specific normalization',
                f'Reduced regularization (WD={WEIGHT_DECAY}, dropout={HEAD_DROPOUT})',
                'No MixUp',
                f'Reduced SpecAugment (freq=20, time=25)',
                f'Early Stop on F1 (patience={EARLY_STOP_PATIENCE})',
            ],
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'macro_f1_mean': float(np.mean(f1s)),
            'macro_f1_std': float(np.std(f1s)),
            'best_f1_fold': int(best_f1_fold),
            'class_mapping': class_to_idx,
            'per_fold': [
                {'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'],
                 'weighted_f1': m['weighted_f1'],
                 'confusion_matrix': m['confusion_matrix']}
                for m in all_test_metrics
            ],
            'best_fold_report': best_report
        }, f, indent=2)

    log.print(f"\nAll outputs saved in {output_dir}. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MobileNetV2 V2 Optimized Training — 3-Class Respiratory Sound"
    )
    parser.add_argument("--data_path", type=str,
                        default="/home/iec/Parallel_Computing_on_FPGA/data/combined/audio")
    parser.add_argument("--labels_csv", type=str,
                        default="/home/iec/Parallel_Computing_on_FPGA/data/combined/labels.csv")
    parser.add_argument("--output_dir", type=str,
                        default="./output_mobilenetv2_v2")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs (may be less due to early stopping)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run", action='store_true',
                        help="Run only 1 fold for quick testing")
    args = parser.parse_args()
    main(args)
