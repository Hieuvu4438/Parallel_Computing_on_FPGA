#!/usr/bin/env python3
"""
================================================================================
EfficientNet-B0 Training Script — 3-Class Respiratory Sound Classification
================================================================================
OPTIMIZED version applying all anti-overfitting strategies:

  1. EfficientNet-B0 (better capacity/params ratio than MobileNetV2)
  2. Focal Loss + Label Smoothing (handles class imbalance + overconfidence)
  3. Gradual Unfreezing (3-stage: head → partial → full backbone)
  4. Discriminative Learning Rates (lower LR for pretrained layers)
  5. MixUp Training (alpha=0.3 — creates virtual examples, reduces memorization)
  6. Enhanced Audio Augmentation (noise, shift, gain, polarity, time-stretch)
  7. Enhanced SpecAugment (freq_mask=30, time_mask=40, p=0.7)
  8. Early Stopping on Macro F1 (not val_loss)
  9. Gradient Clipping (max_norm=1.0)
  10. Higher Weight Decay (0.05)
  11. Augment ALL classes equally at 70% probability

Pipeline: Raw WAV → Resample 4kHz → BPF → CWT Morlet Spectrogram → EfficientNet-B0

Usage:
    python train_efficientnet_b0.py \\
        --data_path /home/iec/Parallel_Computing_on_FPGA/data/combined/audio \\
        --labels_csv /home/iec/Parallel_Computing_on_FPGA/data/combined/labels.csv \\
        --output_dir ./output_efficientnet_b0_3class

    # Quick test (1 fold, 2 epochs):
    python train_efficientnet_b0.py --dry_run --epochs 2
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
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    classification_report, precision_score, recall_score,
)

try:
    from torchaudio.transforms import FrequencyMasking, TimeMasking
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG & CONSTANTS
# ==============================================================================
TARGET_SR = 4000
WAVELET_NAME = 'morl'
NUM_SCALES = 128
FREQUENCY_RANGE = (50, 1200)
IMG_SIZE = 224

N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Training Schedule ---
PHASE1_EPOCHS = 15          # Head-only training
PHASE2A_EPOCHS = 20         # Last 3 blocks unfrozen
PHASE2B_EPOCHS = 20         # Last 6 blocks unfrozen
PHASE2C_EPOCHS = 45         # All unfrozen (total max = 100)
TOTAL_MAX_EPOCHS = PHASE1_EPOCHS + PHASE2A_EPOCHS + PHASE2B_EPOCHS + PHASE2C_EPOCHS

PHASE1_LR = 3e-4
PHASE2A_LR_BACKBONE = 5e-6
PHASE2A_LR_HEAD = 3e-5
PHASE2B_LR_BACKBONE = 2e-6
PHASE2B_LR_HEAD = 1e-5
PHASE2C_LR_BACKBONE = 1e-6
PHASE2C_LR_HEAD = 5e-6

WEIGHT_DECAY = 0.05
GRADIENT_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 12    # Stop on F1, not loss
MIXUP_ALPHA = 0.3
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0


# ==============================================================================
# LOGGER UTILITY
# ==============================================================================
class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        with open(self.log_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("  TRAINING LOG: EfficientNet-B0 Optimized Pipeline (3 Classes)\n")
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
    Focuses training on hard-to-classify samples by down-weighting easy ones.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha          # Per-class weight tensor
        self.gamma = gamma          # Focusing parameter (0 = CE, higher = more focus on hard)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)

        # Apply label smoothing to targets
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * smooth_targets * log_probs

        # Apply per-class alpha weighting
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
# MIXUP UTILITY
# ==============================================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.3):
    """MixUp: creates virtual training examples by linear interpolation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute loss for MixUp-augmented data."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==============================================================================
# WAVELET TRANSFORM ENGINE (same as original)
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
        zoom_factors = (self.output_size / spec.shape[0], self.output_size / spec.shape[1])
        spec_resized = zoom(spec, zoom_factors, order=1)
        spec_resized = spec_resized[:self.output_size, :self.output_size]
        return np.clip(spec_resized, 0, 1).astype(np.float32)


# ==============================================================================
# ENHANCED AUGMENTATION ENGINES
# ==============================================================================
class AudioAugmenter:
    """
    Enhanced audio augmentation applied to ALL classes equally.
    Includes: noise (SNR-based), time-shift, gain, polarity inversion, time-stretch.
    """
    def __init__(self, noise_snr_range=(10, 30), shift_ratio=0.15,
                 gain_range=(0.7, 1.3), probability=0.7):
        self.noise_snr_range = noise_snr_range
        self.shift_ratio = shift_ratio
        self.gain_min, self.gain_max = gain_range
        self.probability = probability

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio

        # SNR-based additive noise (more realistic than fixed noise level)
        if np.random.random() < 0.5:
            snr_db = np.random.uniform(*self.noise_snr_range)
            signal_power = np.mean(audio ** 2) + 1e-10
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
            audio = audio + noise

        # Time shift
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-self.shift_ratio, self.shift_ratio) * len(audio))
            audio = np.roll(audio, shift)

        # Gain adjustment
        if np.random.random() < 0.5:
            audio = audio * np.random.uniform(self.gain_min, self.gain_max)

        # Polarity inversion (simple but effective)
        if np.random.random() < 0.3:
            audio = -audio

        # Time stretch (simple via resampling)
        if np.random.random() < 0.3:
            rate = np.random.uniform(0.85, 1.15)
            original_len = len(audio)
            stretched = signal.resample(audio, int(original_len * rate))
            # Pad or crop back to original length
            if len(stretched) > original_len:
                audio = stretched[:original_len]
            else:
                audio = np.pad(stretched, (0, original_len - len(stretched)), mode='constant')

        return np.clip(audio, -1.0, 1.0).astype(np.float32)


class SpecAugment:
    """Enhanced SpecAugment with stronger masking parameters."""
    def __init__(self, freq_mask_param=30, time_mask_param=40, p=0.7):
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
            # Apply a second pass for stronger augmentation
            if np.random.random() < 0.3:
                spec = self.freq_mask(spec)
                spec = self.time_mask(spec)
            return spec
        return self._manual_mask(spec)

    def _manual_mask(self, spec: torch.Tensor) -> torch.Tensor:
        c, h, w = spec.shape
        out = spec.clone()
        # Frequency mask
        if np.random.random() < 0.7:
            f = int(np.random.uniform(0, min(self.freq_mask_param, h)))
            f0 = np.random.randint(0, max(1, h - f))
            out[:, f0:f0 + f, :] = 0
        # Time mask
        if np.random.random() < 0.7:
            t = int(np.random.uniform(0, min(self.time_mask_param, w)))
            t0 = np.random.randint(0, max(1, w - t))
            out[:, :, t0:t0 + t] = 0
        return out


# ==============================================================================
# DATASET AND LOADER
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


def get_group_kfold_splits(file_list, labels, patient_ids, n_splits=N_FOLDS, val_ratio=0.15, random_state=42):
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


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


class AudioSpectrogramDataset(Dataset):
    """
    On-the-fly audio → CWT spectrogram dataset.
    Key change: Augmentation is applied to ALL classes equally (not just minority).
    """
    def __init__(self, file_paths, labels, class_to_idx, transform=None,
                 augment=False, spec_augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.wavelet_transform = WaveletTransform()

        # Augmentation applied equally to ALL classes
        self.augment = augment
        self.augmenter = AudioAugmenter(probability=0.7) if augment else None
        self.spec_augment = spec_augment
        self.spec_aug = SpecAugment(freq_mask_param=30, time_mask_param=40, p=0.7) if spec_augment else None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav_path = self.file_paths[idx]
        try:
            sr, audio = wavfile.read(wav_path)

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)

            # Handle stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample
            if sr != TARGET_SR:
                num_samples = int(len(audio) * TARGET_SR / sr)
                audio = signal.resample(audio, num_samples)

            # Bandpass Filter (50-1950Hz given 4kHz SR)
            audio = butter_bandpass_filter(audio, 50, min(1950, (TARGET_SR // 2) - 1), TARGET_SR)

            # Ensure float32
            audio = audio.astype(np.float32)

        except Exception:
            audio = np.zeros(TARGET_SR * 2, dtype=np.float32)

        # Audio augmentation — applied to ALL classes equally
        if self.augment and self.augmenter:
            audio = self.augmenter(audio)

        # CWT Spectrogram
        spec_image = self.wavelet_transform.to_image(audio)
        spec_image = np.stack([spec_image] * 3, axis=0)
        spec_tensor = torch.from_numpy(spec_image).float()

        # SpecAugment
        if self.spec_augment and self.spec_aug is not None:
            spec_tensor = self.spec_aug(spec_tensor)

        # ImageNet Normalization
        if self.transform:
            spec_tensor = self.transform(spec_tensor)

        return spec_tensor, self.labels[idx]


def oversample_multiclass(file_paths, labels, pids):
    """Oversample minority classes to match majority class count."""
    counts = Counter(labels)
    max_count = max(counts.values())

    new_files, new_labels, new_pids = [], [], []
    for cls in counts.keys():
        cls_indices = [i for i, l in enumerate(labels) if l == cls]
        for idx in cls_indices:
            new_files.append(file_paths[idx])
            new_labels.append(labels[idx])
            new_pids.append(pids[idx])

        n_oversample = max_count - len(cls_indices)
        if n_oversample > 0:
            oversample_idx = np.random.choice(cls_indices, n_oversample, replace=True)
            for idx in oversample_idx:
                new_files.append(file_paths[idx])
                new_labels.append(labels[idx])
                new_pids.append(pids[idx])

    return new_files, new_labels, new_pids


# ==============================================================================
# MODEL ARCHITECTURE: EfficientNet-B0
# ==============================================================================
class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 with custom multi-layer classifier head.
    Better capacity/params ratio than MobileNetV2 (5.3M params total).
    """
    def __init__(self, num_classes=3, pretrained=True, head_dropout=0.5):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        # EfficientNet-B0 features output: 1280 channels
        num_features = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=head_dropout * 0.6),  # 0.3
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=head_dropout * 0.4),  # 0.2
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all feature extraction layers."""
        for p in self.backbone.features.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        """Gradually unfreeze the last n blocks of the backbone."""
        # First freeze everything
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        # Then unfreeze last n blocks
        total_blocks = len(self.backbone.features)
        start_idx = max(0, total_blocks - n)
        for i in range(start_idx, total_blocks):
            for p in self.backbone.features[i].parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze the entire model."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_param_groups(self, lr_backbone: float, lr_head: float):
        """Create parameter groups with discriminative learning rates."""
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
                    use_mixup=True, gradient_clip=GRADIENT_CLIP_NORM,
                    batch_scheduler=None):
    """Training loop with MixUp, gradient clipping, and optional per-batch scheduler."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        if use_mixup and np.random.random() < 0.5:
            # MixUp path
            mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            optimizer.zero_grad()
            out = model(mixed_inputs)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += (lam * pred.eq(y_a).sum().item() + (1 - lam) * pred.eq(y_b).sum().item())
        else:
            # Standard path
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

        # Step per-batch scheduler (e.g. OneCycleLR)
        if batch_scheduler is not None:
            batch_scheduler.step()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validation / Test evaluation."""
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
# TRAINING FOLD — WITH GRADUAL UNFREEZING
# ==============================================================================
def train_fold(fold_id: int, n_folds: int, train_idx, val_idx, test_idx,
               file_list, labels, patient_ids, output_dir: Path, args, log: Logger,
               class_to_idx: dict):
    num_classes = len(class_to_idx)

    train_files = [file_list[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_pids = [patient_ids[i] for i in train_idx]

    val_files = [file_list[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_files = [file_list[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # Oversample train set
    train_files, train_labels, train_pids = oversample_multiclass(train_files, train_labels, train_pids)

    log.print(f"\n" + "=" * 70)
    log.print(f"  FOLD {fold_id + 1}/{n_folds}  |  Train: {len(train_files)}  |  Val: {len(val_files)}  |  Test: {len(test_files)}")
    log.print("=" * 70)

    # Log class distribution
    train_counts = Counter(train_labels)
    log.print(f"  Train class distribution (after oversample): {dict(train_counts)}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_ds = AudioSpectrogramDataset(train_files, train_labels, class_to_idx,
                                        transform=normalize, augment=True, spec_augment=True)
    val_ds = AudioSpectrogramDataset(val_files, val_labels, class_to_idx,
                                      transform=normalize, augment=False, spec_augment=False)
    test_ds = AudioSpectrogramDataset(test_files, test_labels, class_to_idx,
                                       transform=normalize, augment=False, spec_augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- Model ---
    model = EfficientNetB0Classifier(num_classes=num_classes, pretrained=True).to(DEVICE)

    # --- Focal Loss with class weights ---
    class_counts = Counter(train_labels)
    total_samples = sum(class_counts.values())
    # Alpha inversely proportional to class frequency
    alpha_values = torch.FloatTensor([
        total_samples / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)
    ])
    # Normalize alpha to sum to num_classes
    alpha_values = alpha_values / alpha_values.sum() * num_classes
    criterion = FocalLoss(alpha=alpha_values, gamma=FOCAL_GAMMA,
                         label_smoothing=LABEL_SMOOTHING)

    log.print(f"  Focal Loss alpha: {alpha_values.tolist()}, gamma={FOCAL_GAMMA}, smoothing={LABEL_SMOOTHING}")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = -1.0
    patience_counter = 0
    total_epochs = min(args.epochs, TOTAL_MAX_EPOCHS)

    # Define phase boundaries
    p1_end = PHASE1_EPOCHS
    p2a_end = p1_end + PHASE2A_EPOCHS
    p2b_end = p2a_end + PHASE2B_EPOCHS
    p2c_end = total_epochs

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

    log.print(f"  --- Phase 1: Head Only Training (Ep 1-{p1_end}) | LR={PHASE1_LR} ---")

    for epoch in range(p1_end):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                 DEVICE, use_mixup=False,
                                                 batch_scheduler=scheduler)
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
        log.print(f"  Fold {fold_id+1}/{n_folds} | Phase 1 | Epoch {epoch+1:3d}/{total_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | "
                  f"Val F1: {val_f1*100:5.2f}%{best_mark}")

    # ===================== PHASE 2a: Last 3 blocks =====================
    model.unfreeze_last_n_blocks(3)
    param_groups = model.get_param_groups(PHASE2A_LR_BACKBONE, PHASE2A_LR_HEAD)
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2A_EPOCHS, eta_min=1e-7)
    patience_counter = 0

    log.print(f"  --- Phase 2a: Last 3 Blocks Unfrozen (Ep {p1_end+1}-{p2a_end}) | "
              f"LR backbone={PHASE2A_LR_BACKBONE}, head={PHASE2A_LR_HEAD} ---")

    for epoch in range(p1_end, p2a_end):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                 DEVICE, use_mixup=True)
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
        log.print(f"  Fold {fold_id+1}/{n_folds} | Phase 2a | Epoch {epoch+1:3d}/{total_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | "
                  f"Val F1: {val_f1*100:5.2f}%{best_mark}{pat_str}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            log.print(f"  >> Early stopping at epoch {epoch+1} (Phase 2a) due to F1 patience.")
            break

    # ===================== PHASE 2b: Last 6 blocks =====================
    if patience_counter < EARLY_STOP_PATIENCE:
        model.unfreeze_last_n_blocks(6)
        param_groups = model.get_param_groups(PHASE2B_LR_BACKBONE, PHASE2B_LR_HEAD)
        optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2B_EPOCHS, eta_min=1e-7)
        patience_counter = 0

        log.print(f"  --- Phase 2b: Last 6 Blocks Unfrozen (Ep {p2a_end+1}-{p2b_end}) | "
                  f"LR backbone={PHASE2B_LR_BACKBONE}, head={PHASE2B_LR_HEAD} ---")

        for epoch in range(p2a_end, p2b_end):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                     DEVICE, use_mixup=True)
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
            log.print(f"  Fold {fold_id+1}/{n_folds} | Phase 2b | Epoch {epoch+1:3d}/{total_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | "
                      f"Val F1: {val_f1*100:5.2f}%{best_mark}{pat_str}")

            if patience_counter >= EARLY_STOP_PATIENCE:
                log.print(f"  >> Early stopping at epoch {epoch+1} (Phase 2b) due to F1 patience.")
                break

    # ===================== PHASE 2c: All unfrozen =====================
    if patience_counter < EARLY_STOP_PATIENCE:
        model.unfreeze_all()
        param_groups = model.get_param_groups(PHASE2C_LR_BACKBONE, PHASE2C_LR_HEAD)
        optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        remaining = p2c_end - p2b_end
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-7)
        patience_counter = 0

        log.print(f"  --- Phase 2c: All Blocks Unfrozen (Ep {p2b_end+1}-{p2c_end}) | "
                  f"LR backbone={PHASE2C_LR_BACKBONE}, head={PHASE2C_LR_HEAD} ---")

        for epoch in range(p2b_end, p2c_end):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                     DEVICE, use_mixup=True)
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
            log.print(f"  Fold {fold_id+1}/{n_folds} | Phase 2c | Epoch {epoch+1:3d}/{total_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | "
                      f"Val F1: {val_f1*100:5.2f}%{best_mark}{pat_str}")

            if patience_counter >= EARLY_STOP_PATIENCE:
                log.print(f"  >> Early stopping at epoch {epoch+1} (Phase 2c) due to F1 patience.")
                break

    # --- Final Evaluation ---
    model.load_state_dict(torch.load(output_dir / f"best_model_fold_{fold_id}.pth",
                                     map_location=DEVICE, weights_only=True))
    torch.save(model.state_dict(), output_dir / f"weights_fold_{fold_id}_final.pt")

    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    # Use a simple CE for evaluation (Focal Loss doesn't matter at eval time)
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

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', lw=1.5)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', lw=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Fold {fold_id} — Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', lw=1.5)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', lw=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Fold {fold_id} — Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 Score
    axes[2].plot(epochs, history['val_f1'], 'g-', label='Val Macro F1', lw=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title(f'Fold {fold_id} — Macro F1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Mark phase boundaries
    for ax in axes:
        for boundary, label in [(PHASE1_EPOCHS, '2a'), (PHASE1_EPOCHS + PHASE2A_EPOCHS, '2b'),
                                 (PHASE1_EPOCHS + PHASE2A_EPOCHS + PHASE2B_EPOCHS, '2c')]:
            if boundary < len(epochs):
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                ax.text(boundary, ax.get_ylim()[1], f'P{label}', ha='center',
                        va='bottom', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, fold_id):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Fold {fold_id} — Confusion Matrix')
    fig.colorbar(im)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def export_to_onnx(model: nn.Module, save_path: str, input_size=(1, 3, IMG_SIZE, IMG_SIZE)):
    model.eval()
    model.to('cpu')
    dummy = torch.randn(input_size)
    torch.onnx.export(model, dummy, save_path, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
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
    log.print(f"Max Epochs: {min(args.epochs, TOTAL_MAX_EPOCHS)} | Batch Size: {args.batch_size} | K-Folds: {N_FOLDS}")
    log.print(f"Model: EfficientNet-B0 | Loss: Focal (γ={FOCAL_GAMMA}) + LabelSmoothing({LABEL_SMOOTHING})")
    log.print(f"MixUp α={MIXUP_ALPHA} | Weight Decay={WEIGHT_DECAY} | Grad Clip={GRADIENT_CLIP_NORM}")
    log.print(f"Gradual Unfreezing: Phase1={PHASE1_EPOCHS}ep, Phase2a={PHASE2A_EPOCHS}ep, "
              f"Phase2b={PHASE2B_EPOCHS}ep, Phase2c={PHASE2C_EPOCHS}ep")
    log.print(f"Early Stopping: patience={EARLY_STOP_PATIENCE} on val_f1 (NOT val_loss)")

    file_list, labels, patient_ids, class_to_idx = load_dataset_info(args.labels_csv, args.data_path)
    if len(file_list) == 0:
        log.print(f"Error: No valid audio files mapped from {args.labels_csv} in {args.data_path}")
        return

    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    log.print(f"Classes Found {len(class_to_idx)}: {class_to_idx}")
    log.print(f"Total Samples: {len(file_list)} | Unique Patients: {len(np.unique(patient_ids))}")

    # Log raw class distribution
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

        # Plot confusion matrix
        cm = np.array(test_metrics['confusion_matrix'])
        plot_confusion_matrix(cm, class_names, str(output_dir / f"fold_{fold_id}_cm.png"), fold_id)

        macro_f1 = test_metrics['macro_f1']
        log.print(f"  >> Fold {fold_id} Test | Acc: {test_metrics['accuracy']*100:.2f}% | "
                  f"Macro F1: {macro_f1*100:.2f}% | Weighted F1: {test_metrics['weighted_f1']*100:.2f}%")

        # Per-class detail
        for cls in class_names:
            m = test_metrics['report'][cls]
            log.print(f"     {cls.ljust(12)} | P: {m['precision']:.3f} | R: {m['recall']:.3f} | F1: {m['f1-score']:.3f}")

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
    final_pth = output_dir / "efficientnet_b0_3class_best.pth"
    onnx_path = output_dir / "efficientnet_b0_3class_best.onnx"

    log.print(f"\nLoading weights from Fold {best_f1_fold} for final export...")
    model_best = EfficientNetB0Classifier(num_classes=len(class_to_idx), pretrained=False)
    model_best.load_state_dict(torch.load(output_dir / f"best_model_fold_{best_f1_fold}.pth",
                                           map_location='cpu', weights_only=True))

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
            'model': 'EfficientNet-B0',
            'strategies': [
                'Focal Loss (gamma=2.0)',
                'Label Smoothing (0.1)',
                'Gradual Unfreezing (3 stages)',
                'Discriminative LR',
                'MixUp (alpha=0.3)',
                'Enhanced SpecAugment',
                f'Early Stop on F1 (patience={EARLY_STOP_PATIENCE})',
                f'Weight Decay={WEIGHT_DECAY}',
                f'Gradient Clip={GRADIENT_CLIP_NORM}',
            ],
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'macro_f1_mean': float(np.mean(f1s)),
            'macro_f1_std': float(np.std(f1s)),
            'best_f1_fold': int(best_f1_fold),
            'class_mapping': class_to_idx,
            'per_fold': [
                {'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'],
                 'weighted_f1': m['weighted_f1'], 'confusion_matrix': m['confusion_matrix']}
                for m in all_test_metrics
            ],
            'best_fold_report': best_report
        }, f, indent=2)

    log.print(f"\nAll outputs saved in {output_dir}. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientNet-B0 Optimized Training — 3-Class Respiratory Sound")
    parser.add_argument("--data_path", type=str,
                        default="/home/iec/Parallel_Computing_on_FPGA/data/combined/audio")
    parser.add_argument("--labels_csv", type=str,
                        default="/home/iec/Parallel_Computing_on_FPGA/data/combined/labels.csv")
    parser.add_argument("--output_dir", type=str, default="./output_efficientnet_b0_3class")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs (actual may be less due to gradual unfreezing schedule)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run", action='store_true',
                        help="Run only 1 fold for quick testing")
    args = parser.parse_args()
    main(args)
