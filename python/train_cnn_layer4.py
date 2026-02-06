#!/usr/bin/env python3
"""
===============================================================================
CNN Training Script for Layer 4 - Respiratory Sound Classification
===============================================================================

Author: Research Team
Date: 2026
Target: ICBHI 2017 Dataset - 5-class classification

Features:
- Wavelet Transform (CWT with Morlet) for spectrogram generation
- MobileNetV2 Transfer Learning with 2-phase fine-tuning
- Subject-Independent Train/Val/Test split
- Class imbalance handling with WeightedRandomSampler
- Comprehensive metrics (Accuracy, F1, Sensitivity, Specificity)
- ONNX export for C++ integration

Requirements:
    pip install torch torchvision torchaudio
    pip install numpy scipy pywavelets scikit-learn
    pip install matplotlib seaborn tqdm pandas
    pip install onnx onnxruntime

Usage:
    python train_cnn_layer4.py --data_path /path/to/ICBHI_final_database --epochs 50
"""

import os
import re
import argparse
import warnings
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
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, f1_score, precision_score, recall_score
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Image size for MobileNetV2
IMG_SIZE = 224

# Target sample rate
TARGET_SR = 4000

# Wavelet parameters
WAVELET_NAME = 'morl'  # Morlet wavelet
NUM_SCALES = 128
FREQUENCY_RANGE = (50, 2000)  # Hz

# Class mapping - ICBHI diagnosis labels
# Based on ICBHI patient diagnosis file
DIAGNOSIS_MAP = {
    'Healthy': 0,
    'URTI': 1,        # Upper Respiratory Tract Infection
    'LRTI': 2,        # Lower Respiratory Tract Infection (Pneumonia, Bronchitis)
    'Asthma': 2,      # Group with LRTI
    'Bronchiectasis': 3,
    'Bronchiolitis': 3,  # Group with Bronchiectasis
    'COPD': 4,
    'Pneumonia': 2,   # Group with LRTI  
}

# 5-class labels
CLASS_NAMES = ['Healthy', 'URTI', 'LRTI/Pneumonia', 'Bronchiectasis', 'COPD']
NUM_CLASSES = 5

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# WAVELET TRANSFORM MODULE
# ==============================================================================

class WaveletTransform:
    """
    Continuous Wavelet Transform (CWT) with Morlet wavelet
    for generating time-frequency spectrograms.
    """
    
    def __init__(
        self,
        wavelet: str = WAVELET_NAME,
        num_scales: int = NUM_SCALES,
        sample_rate: int = TARGET_SR,
        freq_range: Tuple[float, float] = FREQUENCY_RANGE,
        output_size: int = IMG_SIZE
    ):
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.sample_rate = sample_rate
        self.freq_range = freq_range
        self.output_size = output_size
        
        # Pre-compute scales for frequency range
        self.scales = self._compute_scales()
    
    def _compute_scales(self) -> np.ndarray:
        """Compute scales corresponding to frequency range."""
        # For Morlet wavelet: frequency = sample_rate / (scale * center_freq)
        # Center frequency for Morlet ~ 0.8125
        center_freq = pywt.central_frequency(self.wavelet)
        
        min_scale = center_freq * self.sample_rate / self.freq_range[1]
        max_scale = center_freq * self.sample_rate / self.freq_range[0]
        
        # Logarithmic scale spacing (better for audio)
        scales = np.logspace(
            np.log10(min_scale), 
            np.log10(max_scale), 
            self.num_scales
        )
        
        return scales
    
    def transform(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply CWT to audio signal and return spectrogram.
        
        Args:
            audio: 1D numpy array of audio samples
            
        Returns:
            2D numpy array spectrogram (num_scales x time)
        """
        # Compute CWT coefficients
        coefficients, frequencies = pywt.cwt(
            audio, 
            self.scales, 
            self.wavelet, 
            sampling_period=1.0/self.sample_rate
        )
        
        # Take magnitude (power spectrogram)
        power = np.abs(coefficients) ** 2
        
        # Convert to dB scale
        power_db = 10 * np.log10(power + 1e-10)
        
        # Normalize to [0, 1]
        power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)
        
        return power_db
    
    def to_image(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to fixed-size spectrogram image.
        
        Args:
            audio: 1D audio signal
            
        Returns:
            2D image array (output_size x output_size), normalized to [0, 1]
        """
        # Get spectrogram
        spec = self.transform(audio)
        
        # Resize to target size using bilinear interpolation
        zoom_factors = (
            self.output_size / spec.shape[0],
            self.output_size / spec.shape[1]
        )
        spec_resized = zoom(spec, zoom_factors, order=1)
        
        # Ensure exact size
        spec_resized = spec_resized[:self.output_size, :self.output_size]
        
        # Ensure [0, 1] range
        spec_resized = np.clip(spec_resized, 0, 1)
        
        return spec_resized.astype(np.float32)


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class ICBHIDataset(Dataset):
    """
    ICBHI 2017 Dataset for respiratory sound classification.
    
    Loads breathing cycles and converts them to wavelet spectrograms.
    Supports patient-level labels from diagnosis file.
    """
    
    def __init__(
        self,
        file_list: List[str],
        labels: List[int],
        patient_ids: List[int],
        data_path: str,
        transform: Optional[callable] = None,
        wavelet_transform: Optional[WaveletTransform] = None,
        augment: bool = False
    ):
        self.file_list = file_list
        self.labels = labels
        self.patient_ids = patient_ids
        self.data_path = Path(data_path)
        self.transform = transform
        self.wavelet_transform = wavelet_transform or WaveletTransform()
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load audio file
        wav_path = self.data_path / self.file_list[idx]
        
        try:
            sr, audio = wavfile.read(wav_path)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            # Return dummy data
            audio = np.zeros(TARGET_SR * 2)
            sr = TARGET_SR
        
        # Convert to float and normalize
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
        
        # Apply augmentation if training
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Convert to spectrogram image
        spec_image = self.wavelet_transform.to_image(audio)
        
        # Stack to 3 channels (for pretrained model)
        spec_image = np.stack([spec_image, spec_image, spec_image], axis=0)
        
        # Convert to tensor
        spec_tensor = torch.from_numpy(spec_image)
        
        # Apply transforms
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
        
        label = self.labels[idx]
        
        return spec_tensor, label
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentation."""
        # Random gain
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.8, 1.2)
            audio = audio * gain
        
        # Add noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.005, len(audio))
            audio = audio + noise
        
        # Time shift
        if np.random.random() < 0.3:
            shift = int(np.random.uniform(-0.1, 0.1) * len(audio))
            audio = np.roll(audio, shift)
        
        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio


# ==============================================================================
# DATA LOADING & SPLITTING
# ==============================================================================

def parse_filename(filename: str) -> Tuple[int, str, str, str, str]:
    """
    Parse ICBHI filename to extract metadata.
    
    Format: {PatientID}_{RecordingIndex}_{ChestLocation}_{Mode}_{Equipment}.wav
    Example: 101_1b1_Al_sc_Meditron.wav
    """
    basename = Path(filename).stem
    parts = basename.split('_')
    
    if len(parts) >= 5:
        patient_id = int(parts[0])
        recording_idx = parts[1]
        chest_location = parts[2]
        mode = parts[3]
        equipment = parts[4]
    else:
        # Fallback for non-standard names
        patient_id = hash(basename) % 1000
        recording_idx = "unknown"
        chest_location = "unknown"
        mode = "unknown"
        equipment = "unknown"
    
    return patient_id, recording_idx, chest_location, mode, equipment


def load_diagnosis_file(data_path: str) -> Dict[int, str]:
    """
    Load patient diagnosis from ICBHI diagnosis file.
    
    Returns:
        Dict mapping patient_id -> diagnosis string
    """
    diagnosis_file = Path(data_path) / "ICBHI_Challenge_diagnosis.txt"
    
    patient_diagnosis = {}
    
    if diagnosis_file.exists():
        with open(diagnosis_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    patient_id = int(parts[0])
                    diagnosis = parts[1]
                    patient_diagnosis[patient_id] = diagnosis
    else:
        print(f"Warning: Diagnosis file not found at {diagnosis_file}")
        print("Using default 'Healthy' label for all samples")
    
    return patient_diagnosis


def load_dataset(
    data_path: str,
    use_cycle_labels: bool = False
) -> Tuple[List[str], List[int], List[int]]:
    """
    Load ICBHI dataset and extract file list with labels.
    
    Args:
        data_path: Path to ICBHI_final_database
        use_cycle_labels: If True, use per-cycle labels (crackle/wheeze)
                          If False, use patient-level diagnosis
    
    Returns:
        file_list: List of audio file names
        labels: List of class labels (0-4)
        patient_ids: List of patient IDs
    """
    data_path = Path(data_path)
    
    # Get all WAV files
    wav_files = sorted(data_path.glob("*.wav"))
    
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {data_path}")
    
    print(f"Found {len(wav_files)} audio files")
    
    # Load diagnosis mapping
    patient_diagnosis = load_diagnosis_file(data_path)
    
    file_list = []
    labels = []
    patient_ids = []
    
    for wav_file in wav_files:
        patient_id, _, _, _, _ = parse_filename(wav_file.name)
        
        # Get diagnosis for this patient
        if patient_id in patient_diagnosis:
            diagnosis = patient_diagnosis[patient_id]
            
            # Map to class index
            if diagnosis in DIAGNOSIS_MAP:
                label = DIAGNOSIS_MAP[diagnosis]
            else:
                # Unknown diagnosis -> skip or default to Healthy
                label = 0
        else:
            # No diagnosis info -> default to Healthy
            label = 0
        
        file_list.append(wav_file.name)
        labels.append(label)
        patient_ids.append(patient_id)
    
    print(f"Loaded {len(file_list)} samples")
    
    # Print class distribution
    class_counts = Counter(labels)
    print("\nClass distribution:")
    for cls_idx in sorted(class_counts.keys()):
        print(f"  {CLASS_NAMES[cls_idx]}: {class_counts[cls_idx]}")
    
    return file_list, labels, patient_ids


def subject_independent_split(
    file_list: List[str],
    labels: List[int],
    patient_ids: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[List, List, List, List, List, List, List, List, List]:
    """
    Split dataset ensuring no patient appears in multiple sets.
    
    Returns:
        train_files, train_labels, train_pids,
        val_files, val_labels, val_pids,
        test_files, test_labels, test_pids
    """
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'file': file_list,
        'label': labels,
        'patient_id': patient_ids
    })
    
    # Get unique patients
    unique_patients = df['patient_id'].unique()
    print(f"\nTotal unique patients: {len(unique_patients)}")
    
    # First split: train+val vs test
    train_val_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_ratio,
        random_state=random_state
    )
    
    # Second split: train vs val
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=adjusted_val_ratio,
        random_state=random_state
    )
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Filter data by patient sets
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return (
        train_df['file'].tolist(), train_df['label'].tolist(), train_df['patient_id'].tolist(),
        val_df['file'].tolist(), val_df['label'].tolist(), val_df['patient_id'].tolist(),
        test_df['file'].tolist(), test_df['label'].tolist(), test_df['patient_id'].tolist()
    )


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create WeightedRandomSampler to handle class imbalance.
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Compute weight for each class (inverse frequency)
    class_weights = {
        cls: total_samples / count 
        for cls, count in class_counts.items()
    }
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.FloatTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


# ==============================================================================
# MODEL DEFINITION
# ==============================================================================

class RespiratoryClassifier(nn.Module):
    """
    MobileNetV2-based classifier for respiratory sounds.
    
    Uses pretrained ImageNet weights and replaces classifier head.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        # Get the number of features from backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all backbone layers (Phase 1 training)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen - only classifier will be trained")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers (Phase 2 fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - full fine-tuning enabled")
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate model."""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Compute comprehensive metrics."""
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    metrics['per_class'] = {}
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    for i, name in enumerate(class_names):
        # True Positives, False Positives, False Negatives, True Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # F1 Score
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics['per_class'][name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'support': int(cm[i, :].sum())
        }
    
    metrics['confusion_matrix'] = cm
    
    return metrics


def print_metrics(metrics: Dict, class_names: List[str]):
    """Print metrics in formatted table."""
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1-Score:   {metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1:      {metrics['weighted_f1']*100:.2f}%")
    
    print("\nPer-Class Metrics:")
    print("-"*70)
    print(f"{'Class':<20} {'Sensitivity':>12} {'Specificity':>12} {'Precision':>12} {'F1':>10}")
    print("-"*70)
    
    for name in class_names:
        m = metrics['per_class'][name]
        print(f"{name:<20} {m['sensitivity']*100:>11.2f}% {m['specificity']*100:>11.2f}% {m['precision']*100:>11.2f}% {m['f1']*100:>9.2f}%")
    
    print("-"*70)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png"
):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(
    history: Dict,
    save_path: str = "training_history.png"
):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue')
    axes[1].plot(history['val_acc'], label='Val Acc', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")


# ==============================================================================
# ONNX EXPORT
# ==============================================================================

def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_size: Tuple[int, int, int, int] = (1, 3, IMG_SIZE, IMG_SIZE)
):
    """Export model to ONNX format."""
    model.eval()
    model.to('cpu')
    
    # Create dummy input
    dummy_input = torch.randn(input_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {save_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except Exception as e:
        print(f"ONNX verification warning: {e}")


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def main(args):
    """Main training pipeline."""
    print("\n" + "="*70)
    print("RESPIRATORY CNN TRAINING - LAYER 4")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================================
    # 1. LOAD DATA
    # ==================================
    print("[1/6] Loading dataset...")
    
    file_list, labels, patient_ids = load_dataset(args.data_path)
    
    # Subject-independent split
    (train_files, train_labels, train_pids,
     val_files, val_labels, val_pids,
     test_files, test_labels, test_pids) = subject_independent_split(
        file_list, labels, patient_ids
    )
    
    # ==================================
    # 2. CREATE DATA LOADERS
    # ==================================
    print("\n[2/6] Creating data loaders...")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Wavelet transform
    wavelet_transform = WaveletTransform()
    
    # Datasets
    train_dataset = ICBHIDataset(
        train_files, train_labels, train_pids,
        args.data_path, train_transform, wavelet_transform, augment=True
    )
    
    val_dataset = ICBHIDataset(
        val_files, val_labels, val_pids,
        args.data_path, val_transform, wavelet_transform, augment=False
    )
    
    test_dataset = ICBHIDataset(
        test_files, test_labels, test_pids,
        args.data_path, val_transform, wavelet_transform, augment=False
    )
    
    # Weighted sampler for class imbalance
    train_sampler = create_weighted_sampler(train_labels)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ==================================
    # 3. CREATE MODEL
    # ==================================
    print("\n[3/6] Creating model...")
    
    model = RespiratoryClassifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=args.dropout
    )
    model = model.to(DEVICE)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights
    class_counts = Counter(train_labels)
    class_weights = torch.FloatTensor([
        1.0 / class_counts.get(i, 1) for i in range(NUM_CLASSES)
    ])
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = class_weights.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = output_dir / "best_model.pth"
    
    # ==================================
    # 4. PHASE 1: TRAIN CLASSIFIER ONLY
    # ==================================
    print("\n[4/6] Phase 1: Training classifier (backbone frozen)...")
    
    model.freeze_backbone()
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    phase1_epochs = min(args.phase1_epochs, args.epochs // 3)
    
    for epoch in range(phase1_epochs):
        print(f"\nPhase 1 - Epoch {epoch+1}/{phase1_epochs}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        val_loss, val_acc, val_preds, val_labels_arr = validate(
            model, val_loader, criterion, DEVICE
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ★ New best model saved! Val Acc: {val_acc*100:.2f}%")
    
    # ==================================
    # 5. PHASE 2: FINE-TUNE FULL MODEL
    # ==================================
    print("\n[5/6] Phase 2: Fine-tuning full model...")
    
    model.unfreeze_backbone()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr_phase2,
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - phase1_epochs, eta_min=1e-7
    )
    
    for epoch in range(phase1_epochs, args.epochs):
        print(f"\nPhase 2 - Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        val_loss, val_acc, val_preds, val_labels_arr = validate(
            model, val_loader, criterion, DEVICE
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ★ New best model saved! Val Acc: {val_acc*100:.2f}%")
    
    # ==================================
    # 6. FINAL EVALUATION
    # ==================================
    print("\n[6/6] Final evaluation on test set...")
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    test_loss, test_acc, test_preds, test_labels_arr = validate(
        model, test_loader, criterion, DEVICE
    )
    
    # Compute metrics
    metrics = compute_metrics(test_labels_arr, test_preds, CLASS_NAMES)
    print_metrics(metrics, CLASS_NAMES)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        CLASS_NAMES,
        str(output_dir / "confusion_matrix.png")
    )
    
    # Plot training history
    plot_training_history(history, str(output_dir / "training_history.png"))
    
    # Save metrics to JSON
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'macro_f1': float(metrics['macro_f1']),
        'weighted_f1': float(metrics['weighted_f1']),
        'per_class': {
            name: {k: float(v) for k, v in m.items()}
            for name, m in metrics['per_class'].items()
        }
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # ==================================
    # 7. EXPORT TO ONNX
    # ==================================
    print("\n[7/7] Exporting to ONNX...")
    
    onnx_path = output_dir / "respiratory_cnn.onnx"
    export_to_onnx(model, str(onnx_path))
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Macro F1: {metrics['macro_f1']*100:.2f}%")
    print(f"\nOutput files saved to: {output_dir}")
    print("="*70)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN for respiratory sound classification"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="D:/PROJECTS/Parallel_Computing_on_FPGA/data/ICBHI_final_database",
        help="Path to ICBHI dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for models and plots"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total number of epochs"
    )
    
    parser.add_argument(
        "--phase1_epochs",
        type=int,
        default=15,
        help="Number of epochs for Phase 1 (frozen backbone)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    
    parser.add_argument(
        "--lr_phase1",
        type=float,
        default=1e-3,
        help="Learning rate for Phase 1"
    )
    
    parser.add_argument(
        "--lr_phase2",
        type=float,
        default=1e-5,
        help="Learning rate for Phase 2 (fine-tuning)"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    args = parser.parse_args()
    
    main(args)
