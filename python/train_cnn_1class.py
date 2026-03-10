#!/usr/bin/env python3
"""
================================================================================
CNN Training Script - Binary COPD Classification (COPD vs Non-COPD)
================================================================================

Target: ICBHI 2017 Dataset - Binary classification
- Class 1 (COPD): Patients with COPD diagnosis
- Class 0 (Non-COPD): Healthy, Pneumonia, URTI, Bronchiectasis

Features:
- Discrete Wavelet Transform (Morlet) for spectrogram generation
- MobileNetV2 Transfer Learning with 2-phase fine-tuning
- Subject-Independent Train/Val/Test split by Patient ID
- Weighted CrossEntropyLoss for class imbalance handling
- Data Augmentation for minority class (Non-COPD)
- Early Stopping based on val_f1
- Comprehensive metrics: Accuracy, F1, Sensitivity, Specificity
- ONNX export

Usage:
    python train_cnn_1class.py --data_path /path/to/ICBHI_final_database
"""

import os
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
    confusion_matrix, accuracy_score, f1_score, 
    precision_score, recall_score
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS
# ==============================================================================

IMG_SIZE = 224
TARGET_SR = 4000
WAVELET_NAME = 'morl'  # Morlet wavelet
NUM_SCALES = 128
# Focus on COPD/wheeze frequency band (50–1500 Hz)
FREQUENCY_RANGE = (50, 1500)

# Binary classification: COPD (1) vs Non-COPD (0)
NUM_CLASSES = 2
CLASS_NAMES = ['Non-COPD', 'COPD']

# ICBHI Patient Diagnosis Mapping (based on official dataset)
# Source: ICBHI 2017 Challenge demographic file
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


# ==============================================================================
# WAVELET TRANSFORM MODULE
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
                                    sampling_period=1.0/self.sample_rate)
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
# DATA AUGMENTATION - Focus on Minority Class (Non-COPD)
# ==============================================================================

class AudioAugmenter:
    """Audio augmentation: noise, time shift, gain for minority class."""
    
    def __init__(self, noise_level=0.01, shift_ratio=0.15, 
                 gain_range=(0.8, 1.2), probability=0.7):
        self.noise_level = noise_level
        self.shift_ratio = shift_ratio
        self.gain_min, self.gain_max = gain_range
        self.probability = probability
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio
        
        # White noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.noise_level, len(audio))
            audio = audio + noise
        
        # Time shift
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-self.shift_ratio, self.shift_ratio) * len(audio))
            audio = np.roll(audio, shift)
        
        # Gain change
        if np.random.random() < 0.5:
            gain = np.random.uniform(self.gain_min, self.gain_max)
            audio = audio * gain
        
        return np.clip(audio, -1.0, 1.0)


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class ICBHIBinaryDataset(Dataset):
    """ICBHI Dataset for binary COPD classification."""
    
    def __init__(self, file_list, labels, patient_ids, data_path,
                 transform=None, wavelet_transform=None, 
                 augment=False, oversample_minority=False):
        self.file_list = file_list
        self.labels = labels
        self.patient_ids = patient_ids
        self.data_path = Path(data_path)
        self.transform = transform
        self.wavelet_transform = wavelet_transform or WaveletTransform()
        self.augment = augment
        self.augmenter = AudioAugmenter() if augment else None
        
        # Oversample minority class (Non-COPD) if requested
        if oversample_minority:
            self._oversample_minority()
    
    def _oversample_minority(self):
        """Oversample Non-COPD class to balance dataset."""
        class_counts = Counter(self.labels)
        max_count = max(class_counts.values())
        
        new_files, new_labels, new_pids = [], [], []
        for i, (f, l, p) in enumerate(zip(self.file_list, self.labels, self.patient_ids)):
            new_files.append(f)
            new_labels.append(l)
            new_pids.append(p)
        
        # Oversample minority class (Non-COPD = 0)
        minority_indices = [i for i, l in enumerate(self.labels) if l == 0]
        if len(minority_indices) > 0:
            n_oversample = max_count - class_counts[0]
            oversample_indices = np.random.choice(minority_indices, n_oversample, replace=True)
            for idx in oversample_indices:
                new_files.append(self.file_list[idx])
                new_labels.append(self.labels[idx])
                new_pids.append(self.patient_ids[idx])
        
        self.file_list = new_files
        self.labels = new_labels
        self.patient_ids = new_pids
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        wav_path = self.data_path / self.file_list[idx]
        
        try:
            sr, audio = wavfile.read(wav_path)
        except Exception as e:
            audio = np.zeros(TARGET_SR * 2)
            sr = TARGET_SR
        
        # Convert to float
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
        
        # Apply augmentation (especially for minority class Non-COPD)
        label = self.labels[idx]
        if self.augment and self.augmenter:
            # More aggressive augmentation for minority class
            if label == 0:  # Non-COPD (minority)
                audio = self.augmenter(audio)
            elif np.random.random() < 0.3:  # Less augmentation for majority
                audio = self.augmenter(audio)
        
        # Convert to spectrogram
        spec_image = self.wavelet_transform.to_image(audio)
        spec_image = np.stack([spec_image] * 3, axis=0)
        spec_tensor = torch.from_numpy(spec_image)
        
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
        
        return spec_tensor, label


# ==============================================================================
# DATA LOADING & SPLITTING
# ==============================================================================

def parse_filename(filename: str) -> Tuple[int, str]:
    """Extract patient ID from ICBHI filename."""
    basename = Path(filename).stem
    parts = basename.split('_')
    patient_id = int(parts[0]) if len(parts) >= 1 else 0
    return patient_id, basename


def load_diagnosis_mapping(data_path: str) -> Dict[int, str]:
    """Load patient diagnosis from file or use built-in mapping."""
    diagnosis_file = Path(data_path) / "ICBHI_Challenge_diagnosis.txt"
    
    if diagnosis_file.exists():
        patient_diagnosis = {}
        with open(diagnosis_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    patient_id = int(parts[0])
                    diagnosis = parts[1]
                    patient_diagnosis[patient_id] = diagnosis
        return patient_diagnosis
    else:
        print("Using built-in ICBHI patient diagnosis mapping...")
        return PATIENT_DIAGNOSIS


def get_binary_label(diagnosis: str) -> int:
    """Map diagnosis to binary label: COPD=1, Non-COPD=0."""
    if diagnosis.upper() == 'COPD':
        return 1
    return 0  # Healthy, Pneumonia, URTI, Bronchiectasis -> Non-COPD


def load_dataset(data_path: str) -> Tuple[List[str], List[int], List[int]]:
    """Load ICBHI dataset with binary COPD labels."""
    data_path = Path(data_path)
    wav_files = sorted(data_path.glob("*.wav"))
    
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {data_path}")
    
    print(f"Found {len(wav_files)} audio files")
    
    patient_diagnosis = load_diagnosis_mapping(data_path)
    
    file_list, labels, patient_ids = [], [], []
    
    for wav_file in wav_files:
        patient_id, _ = parse_filename(wav_file.name)
        
        if patient_id in patient_diagnosis:
            diagnosis = patient_diagnosis[patient_id]
            label = get_binary_label(diagnosis)
        else:
            # Default to Non-COPD if unknown
            label = 0
        
        file_list.append(wav_file.name)
        labels.append(label)
        patient_ids.append(patient_id)
    
    # Print class distribution
    class_counts = Counter(labels)
    total = len(labels)
    print("\nClass distribution:")
    for cls_idx in sorted(class_counts.keys()):
        pct = 100 * class_counts[cls_idx] / total
        print(f"  {CLASS_NAMES[cls_idx]}: {class_counts[cls_idx]} ({pct:.1f}%)")
    
    return file_list, labels, patient_ids


def subject_independent_split(file_list, labels, patient_ids,
                              train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                              random_state=42):
    """Split dataset by patient ID (subject-independent)."""
    df = pd.DataFrame({
        'file': file_list, 'label': labels, 'patient_id': patient_ids
    })
    
    unique_patients = df['patient_id'].unique()
    print(f"\nTotal unique patients: {len(unique_patients)}")
    
    # Stratify by patient-level labels
    patient_labels = df.groupby('patient_id')['label'].first().values
    
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=test_ratio, random_state=random_state,
        stratify=patient_labels
    )
    
    train_val_labels = df[df['patient_id'].isin(train_val_patients)].groupby('patient_id')['label'].first().values
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=adjusted_val_ratio, random_state=random_state,
        stratify=train_val_labels
    )
    
    print(f"Train patients: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]
    
    print(f"Train samples: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return (
        train_df['file'].tolist(), train_df['label'].tolist(), train_df['patient_id'].tolist(),
        val_df['file'].tolist(), val_df['label'].tolist(), val_df['patient_id'].tolist(),
        test_df['file'].tolist(), test_df['label'].tolist(), test_df['patient_id'].tolist()
    )


# ==============================================================================
# MODEL DEFINITION
# ==============================================================================

class COPDClassifier(nn.Module):
    """MobileNetV2-based binary classifier for COPD detection."""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
    
    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1, np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred):
    """Compute Accuracy, Macro F1, Sensitivity (COPD), Specificity (Non-COPD), Precision."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # For binary: TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1]
    tn, fp, fn, tp = cm.ravel()
    
    # Sensitivity (Recall for COPD) = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity (Recall for Non-COPD) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics['sensitivity'] = sensitivity  # COPD detection rate
    metrics['specificity'] = specificity  # Non-COPD detection rate
    metrics['confusion_matrix'] = cm
    
    return metrics


def print_metrics(metrics):
    """Print detailed metrics table (Accuracy, Macro F1, Sensitivity, Specificity, Precision)."""
    print("\n" + "="*60)
    print("EVALUATION METRICS (Subject-Independent Test Set)")
    print("="*60)
    print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1:    {metrics['f1_macro']*100:.2f}%")
    print(f"  Sensitivity: {metrics['sensitivity']*100:.2f}%  (COPD recall)")
    print(f"  Specificity: {metrics['specificity']*100:.2f}%  (Non-COPD recall)")
    print(f"  Precision:   {metrics['precision']*100:.2f}%  (macro)")
    print("="*60)


def plot_training_history(history: Dict, save_path: str, confusion_matrix: Optional[np.ndarray] = None):
    """Plot training history: Loss, Accuracy, F1-Score, and Confusion Matrix in one figure."""
    n_plots = 4 if confusion_matrix is not None else 3
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss curve
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy curve
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=1.5)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F1-Score curve
    ax = axes[2]
    ax.plot(epochs, history['val_f1'], 'g-', label='Val Macro F1', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Macro F1')
    ax.set_title('F1-Score Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    ax = axes[3]
    if confusion_matrix is not None:
        cm_norm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{confusion_matrix[i, j]}\n({cm_norm[i, j]:.1%})',
                       ha='center', va='center', fontsize=11)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.axis('off')
        ax.set_title('Confusion Matrix (N/A)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - COPD Binary Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ==============================================================================
# ONNX EXPORT
# ==============================================================================

def export_to_onnx(model, save_path, input_size=(1, 3, IMG_SIZE, IMG_SIZE)):
    model.eval()
    model.to('cpu')
    dummy_input = torch.randn(input_size)
    
    torch.onnx.export(
        model, dummy_input, save_path,
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {save_path}")
    
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified!")
    except Exception as e:
        print(f"ONNX verification warning: {e}")


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def main(args):
    print("\n" + "="*60)
    print("COPD BINARY CLASSIFICATION - CNN TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print("="*60 + "\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("[1/5] Loading dataset...")
    file_list, labels, patient_ids = load_dataset(args.data_path)
    
    (train_files, train_labels, train_pids,
     val_files, val_labels, val_pids,
     test_files, test_labels, test_pids) = subject_independent_split(
        file_list, labels, patient_ids
    )
    
    # 2. Create data loaders
    print("\n[2/5] Creating data loaders...")
    
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    wavelet_transform = WaveletTransform()
    
    train_dataset = ICBHIBinaryDataset(
        train_files, train_labels, train_pids, args.data_path,
        normalize_transform, wavelet_transform, augment=True, oversample_minority=True
    )
    val_dataset = ICBHIBinaryDataset(
        val_files, val_labels, val_pids, args.data_path,
        normalize_transform, wavelet_transform, augment=False
    )
    test_dataset = ICBHIBinaryDataset(
        test_files, test_labels, test_pids, args.data_path,
        normalize_transform, wavelet_transform, augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 3. Create model with class weights
    print("\n[3/5] Creating model...")
    model = COPDClassifier(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    
    # Weighted CrossEntropyLoss (higher weight for minority Non-COPD class)
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    class_weights = torch.FloatTensor([
        total / (NUM_CLASSES * class_counts.get(i, 1)) for i in range(NUM_CLASSES)
    ]).to(DEVICE)
    print(f"Class weights: Non-COPD={class_weights[0]:.2f}, COPD={class_weights[1]:.2f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 4. Training
    print("\n[4/5] Training...")
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 15
    
    # Phase 1: Freeze backbone
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    phase1_epochs = min(30, args.epochs // 5)
    
    for epoch in range(phase1_epochs):
        print(f"\nPhase 1 - Epoch {epoch+1}/{phase1_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1*100:.2f}%")
        
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
    
    # Phase 2: Unfreeze backbone — reset early stopping so Phase 2 gets full chance
    model.unfreeze_backbone()
    patience_counter = 0  # Reset so Phase 2 is not stopped by Phase 1 patience
    phase2_min_epochs = 15  # Phase 2 must run at least 15 epochs before early stop
    phase2_epochs = args.epochs - phase1_epochs
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase2_epochs, eta_min=1e-6
    )
    
    for epoch in range(phase1_epochs, args.epochs):
        print(f"\nPhase 2 - Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1*100:.2f}%")
        
        scheduler.step()  # CosineAnnealing: step each epoch
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stop only after Phase 2 has run at least phase2_min_epochs
        phase2_elapsed = (epoch + 1) - phase1_epochs
        if phase2_elapsed >= phase2_min_epochs and patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (Phase 2 ran {phase2_elapsed} epochs)")
            break
    
    # 5. Evaluation (Subject-Independent Test Set)
    print("\n[5/5] Final Evaluation...")
    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=DEVICE))
    model.eval()
    
    _, _, _, test_preds, test_labels_arr = validate(model, test_loader, criterion, DEVICE)
    metrics = compute_metrics(test_labels_arr, test_preds)
    
    print_metrics(metrics)
    plot_training_history(history, str(output_dir / "training_results.png"),
                          confusion_matrix=metrics['confusion_matrix'])
    plot_confusion_matrix(metrics['confusion_matrix'], str(output_dir / "confusion_matrix.png"))
    
    # Export ONNX (eval mode for C++/FPGA integration)
    export_to_onnx(model, str(output_dir / "copd_classifier.onnx"))
    
    # Save metrics (Accuracy, Macro F1, Sensitivity, Specificity, Precision)
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump({
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'sensitivity': float(metrics['sensitivity']),
            'specificity': float(metrics['specificity']),
            'precision': float(metrics['precision'])
        }, f, indent=2)
    
    print(f"\nAll outputs saved to {output_dir}")
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COPD Binary Classification Training")
    parser.add_argument("--data_path", type=str, 
                       default="/home/iec/Parallel_Computing_on_FPGA/data/samples/ICBHI_final_database",
                       help="Path to ICBHI database")
    parser.add_argument("--output_dir", type=str, default="./output_copd_binary",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    
    args = parser.parse_args()
    main(args)
