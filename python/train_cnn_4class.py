#!/usr/bin/env python3
"""
===============================================================================
CNN Training Script - 4-Class Respiratory Sound Classification
===============================================================================

Uses per-cycle annotations (Crackle/Wheeze) instead of patient diagnosis.
Classes: Normal (0), Crackle (1), Wheeze (2), Both (3)

This matches the C++ cascaded framework output classes.

Author: Research Team
Date: 2026
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

# 4-class classification based on cycle annotations
CLASS_NAMES_4 = ['Normal', 'Crackle', 'Wheeze', 'Both']
NUM_CLASSES_4 = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# WAVELET TRANSFORM
# ==============================================================================

class WaveletTransform:
    """CWT with Morlet wavelet for spectrogram generation."""
    
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
        
        # Pre-compute scales
        center_freq = pywt.central_frequency(wavelet)
        min_scale = center_freq * sample_rate / freq_range[1]
        max_scale = center_freq * sample_rate / freq_range[0]
        self.scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    
    def to_image(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to 224x224 spectrogram image."""
        # CWT
        coefficients, _ = pywt.cwt(audio, self.scales, self.wavelet, 
                                    sampling_period=1.0/self.sample_rate)
        
        # Power spectrogram in dB
        power = np.abs(coefficients) ** 2
        power_db = 10 * np.log10(power + 1e-10)
        
        # Normalize to [0, 1]
        power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)
        
        # Resize to output size
        zoom_factors = (self.output_size / power_db.shape[0], 
                       self.output_size / power_db.shape[1])
        spec = zoom(power_db, zoom_factors, order=1)
        spec = spec[:self.output_size, :self.output_size]
        
        return np.clip(spec, 0, 1).astype(np.float32)


# ==============================================================================
# DATASET - 4 CLASS (BASED ON CYCLE ANNOTATIONS)
# ==============================================================================

def parse_annotation_file(annotation_path: str) -> List[Dict]:
    """
    Parse ICBHI annotation file.
    
    Format: [Start] [End] [Crackles] [Wheezes]
    Example: 0.036  0.579  0  0
    """
    cycles = []
    
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                start_time = float(parts[0])
                end_time = float(parts[1])
                has_crackle = int(parts[2]) == 1
                has_wheeze = int(parts[3]) == 1
                
                # Determine label
                if has_crackle and has_wheeze:
                    label = 3  # Both
                elif has_crackle:
                    label = 1  # Crackle
                elif has_wheeze:
                    label = 2  # Wheeze
                else:
                    label = 0  # Normal
                
                cycles.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': label
                })
    
    return cycles


def extract_cycle_from_audio(
    audio: np.ndarray, 
    sample_rate: int,
    start_time: float, 
    end_time: float,
    target_sr: int = TARGET_SR
) -> np.ndarray:
    """Extract and resample a breathing cycle from audio."""
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Ensure valid indices
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    cycle = audio[start_sample:end_sample]
    
    # Resample if needed
    if sample_rate != target_sr:
        num_samples = int(len(cycle) * target_sr / sample_rate)
        cycle = signal.resample(cycle, num_samples)
    
    return cycle


def load_icbhi_dataset_4class(data_path: str) -> Tuple[List, List, List]:
    """
    Load ICBHI dataset with 4-class labels from cycle annotations.
    
    Returns:
        cycles: List of (wav_path, start_time, end_time) tuples
        labels: List of labels (0-3)
        patient_ids: List of patient IDs
    """
    data_path = Path(data_path)
    
    # Find all WAV files
    wav_files = sorted(data_path.glob("*.wav"))
    
    cycles = []
    labels = []
    patient_ids = []
    
    for wav_file in tqdm(wav_files, desc="Loading annotations"):
        # Find corresponding annotation file
        txt_file = wav_file.with_suffix('.txt')
        
        if not txt_file.exists():
            continue
        
        # Parse filename for patient ID
        parts = wav_file.stem.split('_')
        patient_id = int(parts[0]) if parts[0].isdigit() else hash(wav_file.stem) % 1000
        
        # Parse annotations
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
    
    # Print class distribution
    class_counts = Counter(labels)
    print("Class distribution:")
    for cls_idx in sorted(class_counts.keys()):
        print(f"  {CLASS_NAMES_4[cls_idx]}: {class_counts[cls_idx]} ({class_counts[cls_idx]/len(labels)*100:.1f}%)")
    
    return cycles, labels, patient_ids


class ICBHICycleDataset(Dataset):
    """Dataset for breathing cycles with 4-class labels."""
    
    def __init__(
        self,
        cycles: List[Dict],
        labels: List[int],
        transform: Optional[callable] = None,
        wavelet_transform: Optional[WaveletTransform] = None,
        augment: bool = False
    ):
        self.cycles = cycles
        self.labels = labels
        self.transform = transform
        self.wavelet_transform = wavelet_transform or WaveletTransform()
        self.augment = augment
        
        # Cache for loaded audio files
        self._audio_cache = {}
    
    def __len__(self) -> int:
        return len(self.cycles)
    
    def _load_audio(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with caching."""
        if wav_path not in self._audio_cache:
            sr, audio = wavfile.read(wav_path)
            
            # Convert to float
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
            
            self._audio_cache[wav_path] = (audio, sr)
        
        return self._audio_cache[wav_path]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cycle_info = self.cycles[idx]
        label = self.labels[idx]
        
        # Load audio
        audio, sr = self._load_audio(cycle_info['wav_path'])
        
        # Extract cycle
        cycle_audio = extract_cycle_from_audio(
            audio, sr, 
            cycle_info['start_time'], 
            cycle_info['end_time']
        )
        
        # Augmentation
        if self.augment:
            # Random gain
            if np.random.random() < 0.5:
                cycle_audio = cycle_audio * np.random.uniform(0.8, 1.2)
            
            # Add noise
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.005, len(cycle_audio))
                cycle_audio = cycle_audio + noise
            
            # Time shift
            if np.random.random() < 0.3:
                shift = int(np.random.uniform(-0.1, 0.1) * len(cycle_audio))
                cycle_audio = np.roll(cycle_audio, shift)
            
            cycle_audio = np.clip(cycle_audio, -1.0, 1.0)
        
        # Convert to spectrogram
        spec = self.wavelet_transform.to_image(cycle_audio)
        
        # Stack to 3 channels
        spec = np.stack([spec, spec, spec], axis=0)
        spec_tensor = torch.from_numpy(spec)
        
        # Apply transforms
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
        
        return spec_tensor, label


# ==============================================================================
# MODEL
# ==============================================================================

class RespiratoryClassifier4Class(nn.Module):
    """MobileNetV2 classifier for 4-class respiratory classification."""
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, NUM_CLASSES_4)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
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
    """Plot training history and confusion matrix."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss')
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Accuracy')
    
    # Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[2])
    axes[2].set_title('Confusion Matrix')
    axes[2].set_ylabel('True')
    axes[2].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_results.png", dpi=150)
    plt.close()


def subject_split(cycles, labels, patient_ids, train_r=0.7, val_r=0.15):
    """Subject-independent split."""
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


# ==============================================================================
# MAIN
# ==============================================================================

def main(args):
    print("="*60)
    print("CNN TRAINING - 4-CLASS RESPIRATORY CLASSIFICATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1] Loading dataset...")
    cycles, labels, patient_ids = load_icbhi_dataset_4class(args.data_path)
    
    # Split
    print("\n[2] Splitting dataset...")
    train_idx, val_idx, test_idx = subject_split(cycles, labels, patient_ids)
    
    # Create datasets
    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    train_labels = [labels[i] for i in train_idx]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # Model
    print("\n[3] Creating model...")
    model = RespiratoryClassifier4Class(pretrained=True, dropout=args.dropout)
    model = model.to(DEVICE)
    
    # Class weights for loss
    class_weights = torch.FloatTensor([1.0/class_counts.get(i, 1) for i in range(NUM_CLASSES_4)])
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES_4
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    # Phase 1: Frozen backbone
    print("\n[4] Phase 1: Training classifier...")
    model.freeze_backbone()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(args.phase1_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.phase1_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, "
              f"Val Acc={val_acc:.2%}, Val F1={val_f1:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")
    
    # Phase 2: Full fine-tuning
    print("\n[5] Phase 2: Fine-tuning...")
    model.unfreeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.phase1_epochs)
    
    for epoch in range(args.phase1_epochs, args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, "
              f"Val Acc={val_acc:.2%}, Val F1={val_f1:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")
    
    # Final evaluation
    print("\n[6] Final evaluation...")
    model.load_state_dict(torch.load(output_dir / "best_model.pth"))
    test_loss, test_acc, test_f1, preds, true_labels = evaluate(model, test_loader, criterion, DEVICE)
    
    cm = confusion_matrix(true_labels, preds)
    
    print(f"\nTest Accuracy: {test_acc:.2%}")
    print(f"Test Macro F1: {test_f1:.2%}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-class metrics
    print("\nPer-class metrics:")
    for i, name in enumerate(CLASS_NAMES_4):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {name}: Precision={precision:.2%}, Recall={recall:.2%}")
    
    # Plot results
    plot_results(history, cm, CLASS_NAMES_4, output_dir)
    
    # Export ONNX
    print("\n[7] Exporting to ONNX...")
    model.eval()
    model.cpu()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, str(output_dir / "respiratory_4class.onnx"),
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11
    )
    print(f"Model exported to {output_dir / 'respiratory_4class.onnx'}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best Val Accuracy: {best_val_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
                        default="D:/PROJECTS/Parallel_Computing_on_FPGA/data/ICBHI_final_database")
    parser.add_argument("--output_dir", type=str, default="./output_4class")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--phase1_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)
