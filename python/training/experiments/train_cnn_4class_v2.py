#!/usr/bin/env python3
"""
===============================================================================
CNN Training Script V2 - 4-Class Respiratory Sound Classification
===============================================================================

IMPROVEMENTS:
- Focal Loss for class imbalance
- Label Smoothing
- Mixup Augmentation
- SpecAugment-like data augmentation
- Early Stopping with patience
- Learning Rate Warmup
- Gradient Clipping
- Better regularization

Author: Research Team
Date: 2026
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json
import copy

import numpy as np
import pandas as pd
import pywt
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS 
# ==============================================================================

IMG_SIZE = 224
TARGET_SR = 4000

CLASS_NAMES_4 = ['Normal', 'Crackle', 'Wheeze', 'Both']
NUM_CLASSES_4 = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ==============================================================================
# FOCAL LOSS - Better for Class Imbalance
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reduces loss for well-classified examples, focusing on hard negatives.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute softmax
        p = F.softmax(inputs, dim=1)
        
        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                   label_smoothing=self.label_smoothing)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_weight = focal_weight * alpha_t
        
        # Final loss
        loss = focal_weight * ce_loss
        
        return loss.mean()


# ==============================================================================
# MIXUP AUGMENTATION
# ==============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Mixup: Beyond Empirical Risk Minimization
    Creates virtual training examples by mixing pairs.
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


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==============================================================================
# EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation performance stops improving."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001, 
                 mode: str = 'max', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"  ★ New best score: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  ⚠ Early stopping triggered! Best score: {self.best_score:.4f}")
        
        return self.early_stop


# ==============================================================================
# WAVELET TRANSFORM WITH AUGMENTATION
# ==============================================================================

class WaveletTransform:
    """CWT with Morlet wavelet + SpecAugment-like augmentation."""
    
    def __init__(
        self,
        wavelet: str = 'morl',
        num_scales: int = 128,
        sample_rate: int = TARGET_SR,
        freq_range: Tuple[float, float] = (50, 2000),
        output_size: int = IMG_SIZE
    ):
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.sample_rate = sample_rate
        self.freq_range = freq_range
        self.output_size = output_size
        
        center_freq = pywt.central_frequency(wavelet)
        min_scale = center_freq * sample_rate / freq_range[1]
        max_scale = center_freq * sample_rate / freq_range[0]
        self.scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    
    def to_image(self, audio: np.ndarray, augment: bool = False) -> np.ndarray:
        """Convert audio to spectrogram with optional augmentation."""
        # CWT
        coefficients, _ = pywt.cwt(audio, self.scales, self.wavelet, 
                                    sampling_period=1.0/self.sample_rate)
        
        # Power spectrogram in dB
        power = np.abs(coefficients) ** 2
        power_db = 10 * np.log10(power + 1e-10)
        
        # Normalize to [0, 1]
        power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)
        
        # Resize
        zoom_factors = (self.output_size / power_db.shape[0], 
                       self.output_size / power_db.shape[1])
        spec = zoom(power_db, zoom_factors, order=1)
        spec = spec[:self.output_size, :self.output_size]
        
        # SpecAugment-like augmentation
        if augment:
            spec = self._spec_augment(spec)
        
        return np.clip(spec, 0, 1).astype(np.float32)
    
    def _spec_augment(self, spec: np.ndarray) -> np.ndarray:
        """Apply SpecAugment-like augmentation."""
        h, w = spec.shape
        
        # Frequency masking (horizontal stripes)
        if np.random.random() < 0.5:
            num_masks = np.random.randint(1, 4)
            for _ in range(num_masks):
                f = np.random.randint(0, h // 4)
                f0 = np.random.randint(0, h - f)
                spec[f0:f0+f, :] = 0
        
        # Time masking (vertical stripes)
        if np.random.random() < 0.5:
            num_masks = np.random.randint(1, 4)
            for _ in range(num_masks):
                t = np.random.randint(0, w // 4)
                t0 = np.random.randint(0, w - t)
                spec[:, t0:t0+t] = 0
        
        # Random erasing
        if np.random.random() < 0.3:
            eh = np.random.randint(10, 30)
            ew = np.random.randint(10, 30)
            ey = np.random.randint(0, h - eh)
            ex = np.random.randint(0, w - ew)
            spec[ey:ey+eh, ex:ex+ew] = np.random.uniform(0, 1)
        
        return spec


# ==============================================================================
# DATASET
# ==============================================================================

def parse_annotation_file(annotation_path: str) -> List[Dict]:
    """Parse ICBHI annotation file."""
    cycles = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                start_time = float(parts[0])
                end_time = float(parts[1])
                has_crackle = int(parts[2]) == 1
                has_wheeze = int(parts[3]) == 1
                
                if has_crackle and has_wheeze:
                    label = 3
                elif has_crackle:
                    label = 1
                elif has_wheeze:
                    label = 2
                else:
                    label = 0
                
                cycles.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': label
                })
    return cycles


def extract_cycle_from_audio(audio, sr, start_time, end_time, target_sr=TARGET_SR):
    """Extract and resample breathing cycle."""
    start_sample = max(0, int(start_time * sr))
    end_sample = min(len(audio), int(end_time * sr))
    cycle = audio[start_sample:end_sample]
    
    if sr != target_sr:
        num_samples = int(len(cycle) * target_sr / sr)
        cycle = signal.resample(cycle, num_samples)
    
    return cycle


def load_icbhi_dataset_4class(data_path: str) -> Tuple[List, List, List]:
    """Load ICBHI dataset with 4-class labels."""
    data_path = Path(data_path)
    wav_files = sorted(data_path.glob("*.wav"))
    
    cycles, labels, patient_ids = [], [], []
    
    for wav_file in tqdm(wav_files, desc="Loading annotations"):
        txt_file = wav_file.with_suffix('.txt')
        if not txt_file.exists():
            continue
        
        parts = wav_file.stem.split('_')
        patient_id = int(parts[0]) if parts[0].isdigit() else hash(wav_file.stem) % 1000
        
        cycle_annotations = parse_annotation_file(str(txt_file))
        
        for cycle in cycle_annotations:
            cycles.append({
                'wav_path': str(wav_file),
                'start_time': cycle['start_time'],
                'end_time': cycle['end_time']
            })
            labels.append(cycle['label'])
            patient_ids.append(patient_id)
    
    print(f"\nTotal cycles: {len(cycles)}")
    class_counts = Counter(labels)
    print("Class distribution:")
    for cls_idx in sorted(class_counts.keys()):
        print(f"  {CLASS_NAMES_4[cls_idx]}: {class_counts[cls_idx]} ({class_counts[cls_idx]/len(labels)*100:.1f}%)")
    
    return cycles, labels, patient_ids


class ICBHICycleDataset(Dataset):
    """Dataset with advanced augmentation."""
    
    def __init__(self, cycles, labels, transform=None, wavelet_transform=None, augment=False):
        self.cycles = cycles
        self.labels = labels
        self.transform = transform
        self.wavelet_transform = wavelet_transform or WaveletTransform()
        self.augment = augment
        self._audio_cache = {}
    
    def __len__(self):
        return len(self.cycles)
    
    def _load_audio(self, wav_path: str):
        if wav_path not in self._audio_cache:
            sr, audio = wavfile.read(wav_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
            self._audio_cache[wav_path] = (audio, sr)
        return self._audio_cache[wav_path]
    
    def __getitem__(self, idx: int):
        cycle_info = self.cycles[idx]
        label = self.labels[idx]
        
        audio, sr = self._load_audio(cycle_info['wav_path'])
        cycle_audio = extract_cycle_from_audio(
            audio, sr, cycle_info['start_time'], cycle_info['end_time']
        )
        
        # Audio augmentation
        if self.augment:
            # Time stretching (simulate different speaking rates)
            if np.random.random() < 0.3:
                stretch_factor = np.random.uniform(0.9, 1.1)
                new_len = int(len(cycle_audio) * stretch_factor)
                cycle_audio = signal.resample(cycle_audio, new_len)
            
            # Pitch shift (via resampling trick)
            if np.random.random() < 0.2:
                shift = np.random.uniform(0.95, 1.05)
                cycle_audio = signal.resample(cycle_audio, int(len(cycle_audio) * shift))
            
            # Random gain
            if np.random.random() < 0.5:
                gain = np.random.uniform(0.7, 1.3)
                cycle_audio = cycle_audio * gain
            
            # Add noise
            if np.random.random() < 0.4:
                noise_level = np.random.uniform(0.001, 0.01)
                noise = np.random.normal(0, noise_level, len(cycle_audio))
                cycle_audio = cycle_audio + noise
            
            # Time shift
            if np.random.random() < 0.3:
                shift = int(np.random.uniform(-0.15, 0.15) * len(cycle_audio))
                cycle_audio = np.roll(cycle_audio, shift)
            
            cycle_audio = np.clip(cycle_audio, -1.0, 1.0)
        
        # Convert to spectrogram (with SpecAugment if training)
        spec = self.wavelet_transform.to_image(cycle_audio, augment=self.augment)
        
        # Stack to 3 channels
        spec = np.stack([spec, spec, spec], axis=0)
        spec_tensor = torch.from_numpy(spec)
        
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
        
        return spec_tensor, label


# ==============================================================================
# MODEL WITH ATTENTION
# ==============================================================================

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ca = self.channel_attention(x).unsqueeze(-1).unsqueeze(-1)
        return x * ca


class RespiratoryClassifier4Class(nn.Module):
    """MobileNetV2 with attention and better classifier."""
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        # Get features before classifier
        num_features = self.backbone.classifier[1].in_features  # 1280
        
        # Add attention
        self.attention = CBAM(1280)
        
        # Better classifier with more capacity
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.4),
            nn.Linear(128, NUM_CLASSES_4)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features from backbone
        features = self.backbone.features(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Global pooling
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Classifier
        return self.backbone.classifier(features)
    
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")


# ==============================================================================
# TRAINING WITH MIXED PRECISION
# ==============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler, use_mixup=True, 
                clip_grad=1.0):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Mixup
        if use_mixup and np.random.random() < 0.5:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4)
            
            with autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return total_loss / len(loader.dataset), acc, f1, np.array(all_preds), np.array(all_labels)


def plot_results(history, cm, class_names, output_dir):
    """Plot comprehensive results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val Macro F1', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].set_title('Validation Macro F1')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix (Normalized)')
    axes[1, 1].set_ylabel('True')
    axes[1, 1].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_results.png", dpi=150)
    plt.close()


def subject_split(cycles, labels, patient_ids, train_r=0.7, val_r=0.15):
    """Subject-independent split with stratification attempt."""
    df = pd.DataFrame({'cycle': range(len(cycles)), 'label': labels, 'patient': patient_ids})
    
    unique_patients = df['patient'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    n_train = int(len(unique_patients) * train_r)
    n_val = int(len(unique_patients) * val_r)
    
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train+n_val]
    test_patients = unique_patients[n_train+n_val:]
    
    train_idx = df[df['patient'].isin(train_patients)]['cycle'].tolist()
    val_idx = df[df['patient'].isin(val_patients)]['cycle'].tolist()
    test_idx = df[df['patient'].isin(test_patients)]['cycle'].tolist()
    
    print(f"Train: {len(train_idx)} cycles from {len(train_patients)} patients")
    print(f"Val: {len(val_idx)} cycles from {len(val_patients)} patients")
    print(f"Test: {len(test_idx)} cycles from {len(test_patients)} patients")
    
    return train_idx, val_idx, test_idx


def export_onnx_legacy(model, save_path, device='cpu'):
    """Export using legacy ONNX export (compatible with older PyTorch)."""
    model.eval()
    model.to(device)
    
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    
    # Use legacy export
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            save_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    print(f"Model exported to {save_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except Exception as e:
        print(f"ONNX verification: {e}")


# ==============================================================================
# MAIN
# ==============================================================================

def main(args):
    print("="*70)
    print("CNN TRAINING V2 - 4-CLASS RESPIRATORY CLASSIFICATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Early Stopping Patience: {args.patience}")
    print("="*70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1] Loading dataset...")
    cycles, labels, patient_ids = load_icbhi_dataset_4class(args.data_path)
    
    # Split
    print("\n[2] Splitting dataset...")
    train_idx, val_idx, test_idx = subject_split(cycles, labels, patient_ids)
    
    # Compute class weights for Focal Loss
    train_labels = [labels[i] for i in train_idx]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    # Inverse frequency weighting
    alpha = torch.FloatTensor([
        total_samples / (NUM_CLASSES_4 * class_counts.get(i, 1)) 
        for i in range(NUM_CLASSES_4)
    ])
    alpha = alpha / alpha.sum() * NUM_CLASSES_4
    print(f"Class weights (alpha): {alpha.tolist()}")
    
    # Create datasets
    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    wavelet = WaveletTransform()
    
    train_ds = ICBHICycleDataset(
        [cycles[i] for i in train_idx], [labels[i] for i in train_idx],
        transform, wavelet, augment=True
    )
    val_ds = ICBHICycleDataset(
        [cycles[i] for i in val_idx], [labels[i] for i in val_idx],
        transform, wavelet, augment=False
    )
    test_ds = ICBHICycleDataset(
        [cycles[i] for i in test_idx], [labels[i] for i in test_idx],
        transform, wavelet, augment=False
    )
    
    # Weighted sampler
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Larger batch size for GPU
    batch_size = args.batch_size
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # Model
    print("\n[3] Creating model...")
    model = RespiratoryClassifier4Class(pretrained=True, dropout=args.dropout)
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Focal Loss
    criterion = FocalLoss(alpha=alpha.to(DEVICE), gamma=2.0, label_smoothing=0.1)
    
    # For validation (no focal, just CE)
    val_criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    # ==========================================
    # Phase 1: Train classifier only
    # ==========================================
    print("\n[4] Phase 1: Training classifier (frozen backbone)...")
    model.freeze_backbone()
    
    # Warmup + main training
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1,
        weight_decay=0.01
    )
    
    # Warmup scheduler
    warmup_epochs = 5
    
    for epoch in range(args.phase1_epochs):
        # Warmup learning rate
        if epoch < warmup_epochs:
            lr = args.lr_phase1 * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, use_mixup=False
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, val_criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{args.phase1_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, "
              f"Val Acc={val_acc:.2%}, Val F1={val_f1:.2%}")
        
        if early_stopping(val_f1, model):
            break
    
    # ==========================================
    # Phase 2: Full fine-tuning
    # ==========================================
    print("\n[5] Phase 2: Fine-tuning full model...")
    model.unfreeze_backbone()
    
    # Reset early stopping for phase 2
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_phase2, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )
    
    for epoch in range(args.phase1_epochs, args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, 
            use_mixup=True, clip_grad=1.0
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, val_criterion, DEVICE)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} (lr={current_lr:.2e}): "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, "
              f"Val Acc={val_acc:.2%}, Val F1={val_f1:.2%}")
        
        if early_stopping(val_f1, model):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # ==========================================
    # Final evaluation
    # ==========================================
    print("\n[6] Final evaluation...")
    
    # Load best model
    model.load_state_dict(early_stopping.best_model)
    torch.save(model.state_dict(), output_dir / "best_model.pth")
    
    test_loss, test_acc, test_f1, preds, true_labels = evaluate(
        model, test_loader, val_criterion, DEVICE
    )
    
    cm = confusion_matrix(true_labels, preds)
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {test_acc:.2%}")
    print(f"Macro F1: {test_f1:.2%}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print("\nPer-class metrics:")
    for i, name in enumerate(CLASS_NAMES_4):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {name}: Precision={precision:.2%}, Recall={recall:.2%}, "
              f"Specificity={specificity:.2%}, F1={f1:.2%}")
    
    # Plot results
    plot_results(history, cm, CLASS_NAMES_4, output_dir)
    
    # Save history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # ==========================================
    # Export ONNX
    # ==========================================
    print("\n[7] Exporting to ONNX...")
    
    try:
        export_onnx_legacy(model, str(output_dir / "respiratory_4class.onnx"), device='cpu')
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Saving as TorchScript instead...")
        model.eval()
        model.cpu()
        scripted = torch.jit.trace(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
        scripted.save(str(output_dir / "respiratory_4class.pt"))
        print(f"TorchScript model saved to {output_dir / 'respiratory_4class.pt'}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Best Val F1: {early_stopping.best_score:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test Macro F1: {test_f1:.2%}")
    print(f"Output: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
                        default="D:/PROJECTS/Parallel_Computing_on_FPGA/data/ICBHI_final_database")
    parser.add_argument("--output_dir", type=str, default="./output_4class_v2")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--phase1_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)  # Larger for GPU
    parser.add_argument("--lr_phase1", type=float, default=1e-3)
    parser.add_argument("--lr_phase2", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=25)  # Early stopping patience
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)
