#!/usr/bin/env python3
"""
================================================================================
Knowledge Distillation Pipeline v2 — 3-Class Respiratory Sound Classification
================================================================================
BTS-d++ inspired: Teacher Ensemble → Student distillation.
Supports BOTH ICBHI 2017 AND combined dataset (ICBHI + DS2).

Pipeline:
  1. Preprocessing: Resample 4kHz, BPF 25-2000Hz, 8s fixed segments
  2. Features: Gammatonegram + Mel-spectrogram fusion (2-channel)
  3. Augmentation: Mixup, SpecAugment, VTLP, SNR noise, time-stretch
  4. Teacher: EfficientNet-B0 ensemble (3 models, different seeds)
  5. Student: MobileNetV2 distilled from teacher soft labels
  6. Loss: Focal Loss + KD Loss (KL-divergence)
  7. Training: Two-stage (balanced → real distribution fine-tune)
  8. Validation: Patient-wise GroupKFold (no data leakage)

Output:
  - Checkpoints (.pt), metrics JSON, training logs
  - Curves: loss, accuracy, F1, precision, recall
  - Grad-CAM visualizations

Usage:
    python distillation_02.py \\
        --data_dir /home/iec/Parallel_Computing_on_FPGA/data/samples/ICBHI_final_database \\
        --labels_file /home/iec/Parallel_Computing_on_FPGA/data/samples/labels.txt \\
        --combined_dir /home/iec/Parallel_Computing_on_FPGA/data/combined/audio \\
        --combined_labels /home/iec/Parallel_Computing_on_FPGA/data/combined/labels.csv \\
        --output_dir ./output_distillation_v2

Target: Accuracy > 95%
"""

import os
import gc
import sys
import json
import time
import csv
import argparse
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, OrderedDict
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.ndimage import zoom
from scipy.signal import gammatone as scipy_gammatone
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

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, precision_score,
    recall_score, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

try:
    from torchaudio.transforms import FrequencyMasking, TimeMasking
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================
TARGET_SR = 4000
SEGMENT_DURATION = 8  # seconds
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_DURATION  # 32000

# Feature extraction
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 128
N_GAMMATONE = 64
IMG_SIZE = 224

# Training
N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3
CLASS_NAMES = ['COPD', 'Healthy', 'Non-COPD']

# Teacher training
TEACHER_EPOCHS_STAGE1 = 40   # balanced oversampling
TEACHER_EPOCHS_STAGE2 = 20   # fine-tune on real distribution
TEACHER_LR_STAGE1 = 1e-3
TEACHER_LR_STAGE2 = 1e-5
TEACHER_ENSEMBLE_SIZE = 3

# Student training (distillation)
STUDENT_EPOCHS_STAGE1 = 50
STUDENT_EPOCHS_STAGE2 = 25
STUDENT_LR_STAGE1 = 3e-4
STUDENT_LR_STAGE2 = 1e-5

# KD params
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.7  # weight for KD loss vs hard label loss

# Regularization
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
MIXUP_ALPHA = 0.4
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0

BATCH_SIZE = 16
NUM_WORKERS = 4

# Disease label mapping to 3 classes
DISEASE_TO_CLASS = {
    'COPD': 'COPD',
    'Healthy': 'Healthy',
    'Asthma': 'Non-COPD',
    'URTI': 'Non-COPD',
    'LRTI': 'Non-COPD',
    'Bronchiectasis': 'Non-COPD',
    'Bronchiolitis': 'Non-COPD',
    'Pneumonia': 'Non-COPD',
}


# ==============================================================================
# LOGGER
# ==============================================================================
class TrainLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.start_time = time.time()
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"  Knowledge Distillation Training Log\n")
            f.write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, msg: str = ""):
        elapsed = time.time() - self.start_time
        timestamp = f"[{elapsed/60:.1f}min]"
        full = f"{timestamp} {msg}"
        print(full)
        with open(self.log_path, 'a') as f:
            f.write(full + "\n")

    def section(self, title: str):
        self.log("\n" + "=" * 80)
        self.log(f"  {title}")
        self.log("=" * 80)


# ==============================================================================
# DATA LOADING — ICBHI format
# ==============================================================================
def parse_labels_file(labels_path: str) -> Dict[int, str]:
    """Parse labels.txt: patient_id<TAB>disease"""
    patient_labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                pid = int(parts[0])
                disease = parts[1].strip()
                patient_labels[pid] = disease
    return patient_labels


def parse_cycle_annotations(txt_path: str) -> List[Tuple[float, float, int, int]]:
    """Parse ICBHI annotation file: start end crackle wheeze"""
    cycles = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 4:
                start = float(parts[0])
                end = float(parts[1])
                crackle = int(parts[2])
                wheeze = int(parts[3])
                cycles.append((start, end, crackle, wheeze))
    return cycles


def build_dataset_from_icbhi(data_dir: str, labels_path: str, logger: TrainLogger):
    """Build dataset: each WAV file → multiple 8s segments, labeled by patient."""
    patient_labels = parse_labels_file(labels_path)
    logger.log(f"Loaded labels for {len(patient_labels)} patients")

    # Count original disease distribution
    disease_counts = Counter(patient_labels.values())
    logger.log(f"Original disease distribution: {dict(disease_counts)}")

    # Map to 3 classes
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    wav_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav')])
    logger.log(f"Found {len(wav_files)} WAV files")

    samples = []  # (wav_path, patient_id, class_idx)
    skipped = 0

    for wav_file in wav_files:
        # Extract patient ID from filename: {patient_id}_{recording}_{location}_{mode}_{device}.wav
        pid = int(wav_file.split('_')[0])
        if pid not in patient_labels:
            skipped += 1
            continue

        disease = patient_labels[pid]
        if disease not in DISEASE_TO_CLASS:
            skipped += 1
            continue

        class_name = DISEASE_TO_CLASS[disease]
        class_idx = class_to_idx[class_name]
        wav_path = os.path.join(data_dir, wav_file)

        samples.append({
            'wav_path': wav_path,
            'patient_id': pid,
            'class_idx': class_idx,
            'class_name': class_name,
            'disease': disease,
        })

    logger.log(f"Valid samples: {len(samples)}, skipped: {skipped}")

    # Class distribution
    class_dist = Counter([s['class_name'] for s in samples])
    logger.log(f"3-class distribution: {dict(class_dist)}")

    return samples, class_to_idx


def build_dataset_from_combined(combined_dir: str, combined_labels: str, logger: TrainLogger):
    """Build dataset from combined directory (ICBHI + DS2).
    
    Combined dir structure:
        combined_dir/COPD/*.wav
        combined_dir/Healthy/*.wav
        combined_dir/Non-COPD/*.wav
    
    labels.csv format: filename,original_label,label,source,patient_id
    """
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    # Build lookup from labels.csv
    label_lookup = {}  # filename -> {label, source, patient_id}
    if combined_labels and os.path.exists(combined_labels):
        with open(combined_labels, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row['filename'].strip()
                label_lookup[fname] = {
                    'label': row['label'].strip(),
                    'source': row.get('source', 'unknown').strip(),
                    'patient_id': row.get('patient_id', '').strip(),
                }
        logger.log(f"Loaded {len(label_lookup)} entries from combined labels.csv")
    
    # Scan subdirectories: COPD, Healthy, Non-COPD
    samples = []
    skipped = 0
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(combined_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.log(f"  Warning: class dir not found: {class_dir}")
            continue
        
        wav_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.wav')])
        logger.log(f"  {class_name}: found {len(wav_files)} WAV files")
        
        class_idx = class_to_idx[class_name]
        
        for wav_file in wav_files:
            wav_path = os.path.join(class_dir, wav_file)
            
            # Try to get patient_id from labels.csv lookup
            patient_id_str = ''
            source = 'unknown'
            original_label = class_name
            
            if wav_file in label_lookup:
                info = label_lookup[wav_file]
                patient_id_str = info['patient_id']
                source = info['source']
                original_label = info.get('label', class_name)
            else:
                # Infer patient_id from filename pattern
                if wav_file.startswith('ICBHI_'):
                    # e.g., ICBHI_104_1b1_Al_sc_Litt3200.wav
                    parts = wav_file.split('_')
                    patient_id_str = f"ICBHI_{parts[1]}" if len(parts) > 1 else wav_file
                    source = 'ICBHI'
                elif wav_file.startswith('DS2_'):
                    # e.g., DS2_BP108_COPD,E W,P R L ,63,M.wav
                    # patient_id pattern: DS2_<number>_<prefix>
                    parts = wav_file.split('_')
                    if len(parts) >= 2:
                        # Extract prefix like BP108, DP108 -> patient DS2_108_BP
                        prefix_part = parts[1]  # e.g., "BP108"
                        patient_id_str = f"DS2_{prefix_part}"
                    source = 'DS2'
                else:
                    patient_id_str = wav_file
            
            # Use patient_id string as group key (hashable)
            # Convert to a unique integer-like key for GroupKFold compatibility
            samples.append({
                'wav_path': wav_path,
                'patient_id': patient_id_str,  # string-based patient ID
                'class_idx': class_idx,
                'class_name': class_name,
                'disease': original_label,
                'source': source,
            })
    
    if not samples:
        logger.log(f"  WARNING: No samples found in combined dir: {combined_dir}")
        return samples, class_to_idx
    
    logger.log(f"Combined dataset: {len(samples)} samples, skipped: {skipped}")
    class_dist = Counter([s['class_name'] for s in samples])
    logger.log(f"  3-class distribution: {dict(class_dist)}")
    source_dist = Counter([s.get('source', 'unknown') for s in samples])
    logger.log(f"  Source distribution: {dict(source_dist)}")
    
    return samples, class_to_idx


def merge_datasets(icbhi_samples, combined_samples, logger: TrainLogger):
    """Merge ICBHI and combined samples, dedup by wav_path, unify patient_id to strings."""
    seen_paths = set()
    merged = []
    
    # Add ICBHI samples first (convert patient_id to string)
    for s in icbhi_samples:
        s_copy = dict(s)
        s_copy['patient_id'] = str(s_copy['patient_id'])
        s_copy.setdefault('source', 'ICBHI')
        norm_path = os.path.normpath(s_copy['wav_path'])
        basename = os.path.basename(norm_path)
        if basename not in seen_paths:
            seen_paths.add(basename)
            merged.append(s_copy)
    
    # Add combined samples (skip duplicates by filename)
    dup_count = 0
    for s in combined_samples:
        s_copy = dict(s)
        s_copy['patient_id'] = str(s_copy['patient_id'])
        norm_path = os.path.normpath(s_copy['wav_path'])
        basename = os.path.basename(norm_path)
        if basename not in seen_paths:
            seen_paths.add(basename)
            merged.append(s_copy)
        else:
            dup_count += 1
    
    logger.log(f"Merged: {len(merged)} unique samples (skipped {dup_count} duplicates)")
    class_dist = Counter([s['class_name'] for s in merged])
    logger.log(f"  Final 3-class distribution: {dict(class_dist)}")
    
    return merged


# ==============================================================================
# AUDIO PREPROCESSING
# ==============================================================================
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def preprocess_audio(wav_path: str, target_sr=TARGET_SR, segment_len=SEGMENT_SAMPLES):
    """Load, resample to 4kHz, BPF 25-2000Hz, normalize, pad/crop to 8s."""
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

        # Resample to 4kHz
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, max(num_samples, 1))

        # Bandpass filter 25-2000 Hz
        b, a = butter_bandpass(25, min(2000, target_sr // 2 - 1), target_sr, order=3)
        audio = signal.filtfilt(b, a, audio).astype(np.float32)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio)) + 1e-10
        audio = audio / max_val

        # Pad or crop to 8 seconds
        if len(audio) < segment_len:
            # Repeat-pad
            repeats = segment_len // len(audio) + 1
            audio = np.tile(audio, repeats)[:segment_len]
        elif len(audio) > segment_len:
            # Center crop
            start = (len(audio) - segment_len) // 2
            audio = audio[start:start + segment_len]

        return audio.astype(np.float32)
    except Exception as e:
        return np.zeros(segment_len, dtype=np.float32)


# ==============================================================================
# FEATURE EXTRACTION — Gammatonegram + Mel-spectrogram fusion
# ==============================================================================
def compute_gammatone_filterbank(sr, n_filters=N_GAMMATONE, fmin=50, fmax=2000):
    """Create gammatone filterbank center frequencies (ERB scale)."""
    ear_q = 9.26449
    min_bw = 24.7
    freqs = -(ear_q * min_bw) + np.exp(
        np.arange(1, n_filters + 1) * (
            -np.log(fmax + ear_q * min_bw) + np.log(fmin + ear_q * min_bw)
        ) / n_filters
    ) * (fmax + ear_q * min_bw)
    freqs = np.flip(freqs)
    return freqs


def compute_gammatonegram(audio, sr=TARGET_SR, n_filters=N_GAMMATONE,
                          n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Compute gammatonegram using FFT-based approximation."""
    # Compute STFT
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Create gammatone-like filterbank weights
    cf = compute_gammatone_filterbank(sr, n_filters, fmin=50, fmax=min(2000, sr // 2 - 1))
    weights = np.zeros((n_filters, len(f)))
    for i, center_freq in enumerate(cf):
        erb = 24.7 * (4.37 * center_freq / 1000 + 1)
        weights[i] = np.exp(-0.5 * ((f - center_freq) / (erb * 0.5)) ** 2)

    # Apply filterbank
    power = np.abs(Zxx) ** 2
    gammatone_spec = np.dot(weights, power)

    # Log compression
    gammatone_spec = np.log10(gammatone_spec + 1e-10)
    return gammatone_spec


def compute_mel_spectrogram(audio, sr=TARGET_SR, n_mels=N_MELS,
                            n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Compute log Mel-spectrogram."""
    if HAS_LIBROSA:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, fmin=50, fmax=min(2000, sr // 2)
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    else:
        # Manual computation
        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        power = np.abs(Zxx) ** 2

        # Create mel filterbank
        mel_min = 2595 * np.log10(1 + 50 / 700)
        mel_max = 2595 * np.log10(1 + min(2000, sr // 2) / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fb = np.zeros((n_mels, len(f)))
        for i in range(n_mels):
            for j in range(bin_points[i], min(bin_points[i + 1], len(f))):
                fb[i, j] = (j - bin_points[i]) / max(bin_points[i + 1] - bin_points[i], 1)
            for j in range(bin_points[i + 1], min(bin_points[i + 2], len(f))):
                fb[i, j] = (bin_points[i + 2] - j) / max(bin_points[i + 2] - bin_points[i + 1], 1)

        mel_spec = np.dot(fb, power)
        mel_spec_db = 10 * np.log10(mel_spec + 1e-10)
        mel_spec_db -= np.max(mel_spec_db)

    return mel_spec_db


def create_hybrid_spectrogram(audio, sr=TARGET_SR, output_size=IMG_SIZE):
    """Create 2-channel hybrid: Gammatonegram + Mel-spectrogram → 224x224."""
    gamma = compute_gammatonegram(audio, sr)
    mel = compute_mel_spectrogram(audio, sr)

    # Normalize each to [0, 1]
    def normalize(x):
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return x

    gamma = normalize(gamma)
    mel = normalize(mel)

    # Resize both to output_size x output_size
    gamma_resized = zoom(gamma, (output_size / gamma.shape[0], output_size / gamma.shape[1]), order=1)
    mel_resized = zoom(mel, (output_size / mel.shape[0], output_size / mel.shape[1]), order=1)

    gamma_resized = np.clip(gamma_resized[:output_size, :output_size], 0, 1)
    mel_resized = np.clip(mel_resized[:output_size, :output_size], 0, 1)

    # Stack as 3-channel (gamma, mel, average) for pretrained models
    avg_channel = (gamma_resized + mel_resized) / 2.0
    hybrid = np.stack([gamma_resized, mel_resized, avg_channel], axis=0)

    return hybrid.astype(np.float32)


# ==============================================================================
# AUGMENTATION
# ==============================================================================
class AudioAugmenter:
    """Audio-domain augmentation: noise, shift, gain, polarity, time-stretch, VTLP."""
    def __init__(self, probability=0.7):
        self.probability = probability

    def __call__(self, audio):
        if np.random.random() > self.probability:
            return audio
        audio = audio.copy()

        # SNR-based noise
        if np.random.random() < 0.5:
            snr_db = np.random.uniform(10, 30)
            sig_power = np.mean(audio ** 2) + 1e-10
            noise_power = sig_power / (10 ** (snr_db / 10))
            audio = audio + np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)

        # Time shift
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-0.15, 0.15) * len(audio))
            audio = np.roll(audio, shift)

        # Gain
        if np.random.random() < 0.5:
            audio = audio * np.random.uniform(0.7, 1.3)

        # Polarity inversion
        if np.random.random() < 0.3:
            audio = -audio

        # Time stretch
        if np.random.random() < 0.3:
            rate = np.random.uniform(0.85, 1.15)
            orig_len = len(audio)
            stretched = signal.resample(audio, int(orig_len * rate))
            if len(stretched) > orig_len:
                audio = stretched[:orig_len]
            else:
                audio = np.pad(stretched, (0, orig_len - len(stretched)), mode='constant')

        # VTLP (Vocal Tract Length Perturbation) — warp frequency axis
        if np.random.random() < 0.3:
            alpha = np.random.uniform(0.9, 1.1)
            n_fft = 512
            stft = np.fft.rfft(audio, n=n_fft)
            freqs = np.arange(len(stft))
            warped_freqs = np.clip((freqs * alpha).astype(int), 0, len(stft) - 1)
            warped_stft = stft[warped_freqs]
            warped_audio = np.fft.irfft(warped_stft, n=n_fft)[:len(audio)]
            if len(warped_audio) < len(audio):
                warped_audio = np.pad(warped_audio, (0, len(audio) - len(warped_audio)))
            audio = warped_audio.astype(np.float32)

        return np.clip(audio, -1.0, 1.0).astype(np.float32)


class SpecAugment:
    """SpecAugment: frequency and time masking on spectrograms."""
    def __init__(self, freq_mask_param=30, time_mask_param=40, p=0.7):
        self.p = p
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        if HAS_TORCHAUDIO:
            self.freq_mask = FrequencyMasking(freq_mask_param)
            self.time_mask = TimeMasking(time_mask_param)
        else:
            self.freq_mask = self.time_mask = None

    def __call__(self, spec):
        if np.random.random() > self.p:
            return spec
        if self.freq_mask is not None:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
            if np.random.random() < 0.3:
                spec = self.freq_mask(spec)
                spec = self.time_mask(spec)
            return spec
        # Manual fallback
        c, h, w = spec.shape
        out = spec.clone()
        f = int(np.random.uniform(0, min(self.freq_mask_param, h)))
        f0 = np.random.randint(0, max(1, h - f))
        out[:, f0:f0 + f, :] = 0
        t = int(np.random.uniform(0, min(self.time_mask_param, w)))
        t0 = np.random.randint(0, max(1, w - t))
        out[:, :, t0:t0 + t] = 0
        return out


# ==============================================================================
# DATASET
# ==============================================================================
class ICBHIDataset(Dataset):
    """On-the-fly: WAV → preprocess → hybrid spectrogram → tensor."""
    def __init__(self, samples, augment=False, spec_augment=False):
        self.samples = samples
        self.augmenter = AudioAugmenter() if augment else None
        self.spec_aug = SpecAugment() if spec_augment else None
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = preprocess_audio(s['wav_path'])

        if self.augmenter:
            audio = self.augmenter(audio)

        spec = create_hybrid_spectrogram(audio)
        spec_tensor = torch.from_numpy(spec).float()

        if self.spec_aug:
            spec_tensor = self.spec_aug(spec_tensor)

        spec_tensor = self.normalize(spec_tensor)
        return spec_tensor, s['class_idx']


def oversample_samples(samples):
    """Oversample minority classes to balance dataset."""
    counts = Counter([s['class_idx'] for s in samples])
    max_count = max(counts.values())
    new_samples = []
    for cls_idx in counts:
        cls_samples = [s for s in samples if s['class_idx'] == cls_idx]
        new_samples.extend(cls_samples)
        n_extra = max_count - len(cls_samples)
        if n_extra > 0:
            extras = [cls_samples[i % len(cls_samples)] for i in range(n_extra)]
            new_samples.extend(extras)
    np.random.shuffle(new_samples)
    return new_samples


# ==============================================================================
# FOCAL LOSS
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            smooth = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth = F.one_hot(targets, num_classes).float()
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        focal_w = (1.0 - probs) ** self.gamma
        loss = -focal_w * smooth * log_probs
        if self.alpha is not None:
            loss = loss * self.alpha.to(inputs.device).unsqueeze(0)
        loss = loss.sum(dim=1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ==============================================================================
# KD LOSS
# ==============================================================================
class DistillationLoss(nn.Module):
    """Combined loss: alpha * KD_loss + (1-alpha) * hard_label_loss."""
    def __init__(self, hard_loss_fn, temperature=KD_TEMPERATURE, alpha=KD_ALPHA):
        super().__init__()
        self.hard_loss_fn = hard_loss_fn
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, targets):
        # KD loss (KL divergence on soft targets)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        # Hard label loss
        hard_loss = self.hard_loss_fn(student_logits, targets)
        return self.alpha * kd_loss + (1 - self.alpha) * hard_loss


# ==============================================================================
# MIXUP
# ==============================================================================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==============================================================================
# MODELS
# ==============================================================================
class TeacherModel(nn.Module):
    """EfficientNet-B0 as teacher with enhanced head."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout * 0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class StudentModel(nn.Module):
    """MobileNetV2 as student with enhanced head."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout * 0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False

    def unfreeze_last_n(self, n):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        total = len(self.backbone.features)
        for i in range(max(0, total - n), total):
            for p in self.backbone.features[i].parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="  Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup and np.random.random() < 0.5:
            mixed, y_a, y_b, lam = mixup_data(inputs, labels)
            out = model(mixed)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += (lam * pred.eq(y_a).sum().item() + (1 - lam) * pred.eq(y_b).sum().item())
        else:
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)


def train_student_epoch(student, teacher_models, loader, kd_criterion, optimizer, device):
    """Train student with distillation from teacher ensemble."""
    student.train()
    for t in teacher_models:
        t.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(loader, desc="  KD-Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Get ensemble teacher logits
        with torch.no_grad():
            teacher_logits = torch.zeros(inputs.size(0), NUM_CLASSES, device=device)
            for t_model in teacher_models:
                teacher_logits += t_model(inputs)
            teacher_logits /= len(teacher_models)

        optimizer.zero_grad()
        student_logits = student(inputs)
        loss = kd_criterion(student_logits, teacher_logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_CLIP)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, pred = student_logits.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for inputs, labels in tqdm(loader, desc="  Eval", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        loss = criterion(out, labels)
        total_loss += loss.item() * inputs.size(0)
        probs = F.softmax(out, dim=1)
        _, pred = out.max(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    prec_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'loss': total_loss / max(n, 1),
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }


# ==============================================================================
# METRICS & VISUALIZATION
# ==============================================================================
def save_all_metrics(metrics_dict, history, fold_id, output_dir, class_names, logger):
    """Save comprehensive metrics: JSON, confusion matrix, curves."""
    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # --- Classification report ---
    y_true = metrics_dict['labels']
    y_pred = metrics_dict['predictions']
    y_prob = metrics_dict['probabilities']
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    logger.log(f"\n{report_text}")

    # --- Save JSON ---
    json_data = {
        'fold': fold_id,
        'accuracy': float(metrics_dict['accuracy']),
        'f1_macro': float(metrics_dict['f1_macro']),
        'f1_weighted': float(metrics_dict['f1_weighted']),
        'precision_macro': float(metrics_dict['precision_macro']),
        'recall_macro': float(metrics_dict['recall_macro']),
        'loss': float(metrics_dict['loss']),
        'per_class': {},
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    for cls in class_names:
        if cls in report:
            json_data['per_class'][cls] = {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1-score': report[cls]['f1-score'],
                'support': report[cls]['support'],
            }

    with open(metrics_dir / f'metrics_fold_{fold_id}.json', 'w') as f:
        json.dump(json_data, f, indent=2)

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'Confusion Matrix — Fold {fold_id}', fontsize=14)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(metrics_dir / f'confusion_matrix_fold_{fold_id}.png', dpi=150)
    plt.close()

    # --- Training curves ---
    if history:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs_range = range(1, len(history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val')
        axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(epochs_range, history['train_acc'], 'b-', label='Train')
        axes[0, 1].plot(epochs_range, history['val_acc'], 'r-', label='Val')
        axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True)

        # F1
        axes[0, 2].plot(epochs_range, history['val_f1'], 'g-', label='Val F1 (macro)')
        axes[0, 2].set_title('F1 Score (macro)'); axes[0, 2].legend(); axes[0, 2].grid(True)

        # Precision
        axes[1, 0].plot(epochs_range, history['val_precision'], 'm-', label='Val Precision')
        axes[1, 0].set_title('Precision (macro)'); axes[1, 0].legend(); axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(epochs_range, history['val_recall'], 'c-', label='Val Recall')
        axes[1, 1].set_title('Recall (macro)'); axes[1, 1].legend(); axes[1, 1].grid(True)

        # LR
        if 'lr' in history:
            axes[1, 2].plot(epochs_range, history['lr'], 'k-')
            axes[1, 2].set_title('Learning Rate'); axes[1, 2].grid(True)
        else:
            axes[1, 2].axis('off')

        for ax in axes.flat:
            ax.set_xlabel('Epoch')
        plt.suptitle(f'Training Curves — Fold {fold_id}', fontsize=16)
        plt.tight_layout()
        plt.savefig(metrics_dir / f'training_curves_fold_{fold_id}.png', dpi=150)
        plt.close()

    # --- ROC curves (per class) ---
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(f'ROC Curves — Fold {fold_id}')
        ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig(metrics_dir / f'roc_curves_fold_{fold_id}.png', dpi=150)
        plt.close()
    except Exception:
        pass

    return json_data


# ==============================================================================
# GRAD-CAM
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class


def save_gradcam_samples(model, dataset, output_dir, device, n_samples=8):
    """Save Grad-CAM visualizations for sample predictions."""
    gradcam_dir = output_dir / 'metrics' / 'gradcam'
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # Find target layer (last conv in MobileNetV2 or EfficientNet)
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'features'):
            target_layer = model.backbone.features[-1]
        else:
            target_layer = list(model.backbone.children())[-3]
    else:
        return

    try:
        cam_gen = GradCAM(model, target_layer)
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

        for i, idx in enumerate(indices):
            spec_tensor, label = dataset[idx]
            input_t = spec_tensor.unsqueeze(0).to(device)
            cam, pred_class = cam_gen.generate(input_t)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            # Original spectrogram
            spec_np = spec_tensor.cpu().numpy()
            axes[0].imshow(spec_np[0], cmap='viridis', aspect='auto')
            axes[0].set_title(f'Gammatonegram\nTrue: {CLASS_NAMES[label]}')
            # Grad-CAM heatmap
            axes[1].imshow(cam, cmap='jet', aspect='auto')
            axes[1].set_title(f'Grad-CAM\nPred: {CLASS_NAMES[pred_class]}')
            # Overlay
            axes[2].imshow(spec_np[0], cmap='viridis', aspect='auto')
            axes[2].imshow(cam, cmap='jet', alpha=0.5, aspect='auto')
            axes[2].set_title('Overlay')

            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(gradcam_dir / f'gradcam_sample_{i}.png', dpi=100)
            plt.close()
    except Exception as e:
        print(f"Grad-CAM error: {e}")


# ==============================================================================
# TRAIN TEACHER ENSEMBLE
# ==============================================================================
def train_teacher_fold(fold_id, train_samples, val_samples, test_samples,
                       output_dir, logger, teacher_idx=0):
    """Train one teacher model with two-stage strategy."""
    logger.section(f"Teacher {teacher_idx+1}/{TEACHER_ENSEMBLE_SIZE} — Fold {fold_id}")

    # Stage 1: Balanced oversampling
    train_balanced = oversample_samples(train_samples)
    logger.log(f"Stage 1 train: {len(train_balanced)} (balanced), val: {len(val_samples)}")

    train_ds = ICBHIDataset(train_balanced, augment=True, spec_augment=True)
    val_ds = ICBHIDataset(val_samples, augment=False, spec_augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = TeacherModel(pretrained=True).to(DEVICE)

    # Class weights for focal loss
    counts = Counter([s['class_idx'] for s in train_balanced])
    total_n = sum(counts.values())
    alpha = torch.FloatTensor([total_n / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)])
    alpha = alpha / alpha.sum() * NUM_CLASSES
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=TEACHER_LR_STAGE1, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS_STAGE1, eta_min=1e-6)

    history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc',
                                'val_f1', 'val_precision', 'val_recall', 'lr']}
    best_f1 = -1
    patience = 0

    # --- Stage 1 ---
    logger.log(f"--- Stage 1: Balanced training ({TEACHER_EPOCHS_STAGE1} epochs) ---")
    for epoch in range(TEACHER_EPOCHS_STAGE1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, use_mixup=True)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['val_precision'].append(val_metrics['precision_macro'])
        history['val_recall'].append(val_metrics['recall_macro'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        is_best = val_metrics['f1_macro'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1_macro']
            ckpt_path = output_dir / 'checkpoints' / f'teacher_{teacher_idx}_fold_{fold_id}_best.pt'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'epoch': epoch,
                        'f1': best_f1, 'acc': val_metrics['accuracy']}, ckpt_path)
            patience = 0
        else:
            patience += 1

        best_mark = " *BEST*" if is_best else ""
        logger.log(f"  T{teacher_idx} Fold{fold_id} S1 Ep{epoch+1:3d}/{TEACHER_EPOCHS_STAGE1} | "
                   f"TrL:{train_loss:.4f} VaL:{val_metrics['loss']:.4f} | "
                   f"TrA:{train_acc*100:.1f}% VaA:{val_metrics['accuracy']*100:.1f}% | "
                   f"F1:{val_metrics['f1_macro']*100:.1f}%{best_mark}")

        if patience >= EARLY_STOP_PATIENCE:
            logger.log(f"  Early stop at epoch {epoch+1}")
            break

    # --- Stage 2: Fine-tune on real distribution ---
    logger.log(f"--- Stage 2: Fine-tune on real distribution ({TEACHER_EPOCHS_STAGE2} epochs) ---")
    # Reload best checkpoint
    ckpt = torch.load(output_dir / 'checkpoints' / f'teacher_{teacher_idx}_fold_{fold_id}_best.pt',
                      map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state'])

    # Real distribution (no oversampling)
    train_ds_real = ICBHIDataset(train_samples, augment=True, spec_augment=True)
    train_loader_real = DataLoader(train_ds_real, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    optimizer_ft = optim.AdamW(model.parameters(), lr=TEACHER_LR_STAGE2, weight_decay=WEIGHT_DECAY)
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=TEACHER_EPOCHS_STAGE2, eta_min=1e-7)
    patience = 0

    for epoch in range(TEACHER_EPOCHS_STAGE2):
        train_loss, train_acc = train_one_epoch(model, train_loader_real, criterion, optimizer_ft, DEVICE, use_mixup=False)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['val_precision'].append(val_metrics['precision_macro'])
        history['val_recall'].append(val_metrics['recall_macro'])
        history['lr'].append(optimizer_ft.param_groups[0]['lr'])
        scheduler_ft.step()

        is_best = val_metrics['f1_macro'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1_macro']
            torch.save({'model_state': model.state_dict(), 'epoch': epoch,
                        'f1': best_f1, 'acc': val_metrics['accuracy']},
                       output_dir / 'checkpoints' / f'teacher_{teacher_idx}_fold_{fold_id}_best.pt')
            patience = 0
        else:
            patience += 1

        best_mark = " *BEST*" if is_best else ""
        logger.log(f"  T{teacher_idx} Fold{fold_id} S2 Ep{epoch+1:3d}/{TEACHER_EPOCHS_STAGE2} | "
                   f"TrL:{train_loss:.4f} VaL:{val_metrics['loss']:.4f} | "
                   f"TrA:{train_acc*100:.1f}% VaA:{val_metrics['accuracy']*100:.1f}% | "
                   f"F1:{val_metrics['f1_macro']*100:.1f}%{best_mark}")

        if patience >= 8:
            logger.log(f"  Stage 2 early stop at epoch {epoch+1}")
            break

    # Reload best
    ckpt = torch.load(output_dir / 'checkpoints' / f'teacher_{teacher_idx}_fold_{fold_id}_best.pt',
                      map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    logger.log(f"  Teacher {teacher_idx} best F1: {best_f1*100:.2f}%")

    return model, history


# ==============================================================================
# TRAIN STUDENT (DISTILLATION)
# ==============================================================================
def train_student_fold(fold_id, teacher_models, train_samples, val_samples,
                       test_samples, output_dir, logger):
    """Train student with KD from teacher ensemble."""
    logger.section(f"Student Distillation — Fold {fold_id}")

    train_balanced = oversample_samples(train_samples)
    train_ds = ICBHIDataset(train_balanced, augment=True, spec_augment=True)
    val_ds = ICBHIDataset(val_samples, augment=False, spec_augment=False)
    test_ds = ICBHIDataset(test_samples, augment=False, spec_augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    student = StudentModel(pretrained=True).to(DEVICE)

    counts = Counter([s['class_idx'] for s in train_balanced])
    total_n = sum(counts.values())
    alpha = torch.FloatTensor([total_n / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)])
    alpha = alpha / alpha.sum() * NUM_CLASSES
    hard_criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    kd_criterion = DistillationLoss(hard_criterion, temperature=KD_TEMPERATURE, alpha=KD_ALPHA)
    eval_criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, label_smoothing=0.0)

    history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc',
                                'val_f1', 'val_precision', 'val_recall', 'lr']}
    best_f1 = -1
    patience = 0

    # --- Stage 1: KD with balanced data, freeze backbone initially ---
    student.freeze_backbone()
    head_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=STUDENT_LR_STAGE1, weight_decay=WEIGHT_DECAY)
    head_warmup = 10

    logger.log(f"--- Stage 1a: Head warmup ({head_warmup} epochs, backbone frozen) ---")
    for epoch in range(head_warmup):
        train_loss, train_acc = train_student_epoch(student, teacher_models, train_loader, kd_criterion, optimizer, DEVICE)
        val_metrics = evaluate(student, val_loader, eval_criterion, DEVICE)

        for k, v in [('train_loss', train_loss), ('train_acc', train_acc),
                      ('val_loss', val_metrics['loss']), ('val_acc', val_metrics['accuracy']),
                      ('val_f1', val_metrics['f1_macro']), ('val_precision', val_metrics['precision_macro']),
                      ('val_recall', val_metrics['recall_macro']), ('lr', optimizer.param_groups[0]['lr'])]:
            history[k].append(v)

        is_best = val_metrics['f1_macro'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1_macro']
            ckpt_path = output_dir / 'checkpoints' / f'student_fold_{fold_id}_best.pt'
            torch.save({'model_state': student.state_dict(), 'f1': best_f1}, ckpt_path)

        logger.log(f"  Student Fold{fold_id} Head Ep{epoch+1:3d}/{head_warmup} | "
                   f"TrA:{train_acc*100:.1f}% VaA:{val_metrics['accuracy']*100:.1f}% | "
                   f"F1:{val_metrics['f1_macro']*100:.1f}%{' *BEST*' if is_best else ''}")

    # --- Stage 1b: Unfreeze all, full KD ---
    student.unfreeze_all()
    optimizer = optim.AdamW(student.parameters(), lr=STUDENT_LR_STAGE1 * 0.3, weight_decay=WEIGHT_DECAY)
    remaining = STUDENT_EPOCHS_STAGE1 - head_warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
    patience = 0

    logger.log(f"--- Stage 1b: Full KD ({remaining} epochs, all unfrozen) ---")
    for epoch in range(remaining):
        train_loss, train_acc = train_student_epoch(student, teacher_models, train_loader, kd_criterion, optimizer, DEVICE)
        val_metrics = evaluate(student, val_loader, eval_criterion, DEVICE)
        scheduler.step()

        for k, v in [('train_loss', train_loss), ('train_acc', train_acc),
                      ('val_loss', val_metrics['loss']), ('val_acc', val_metrics['accuracy']),
                      ('val_f1', val_metrics['f1_macro']), ('val_precision', val_metrics['precision_macro']),
                      ('val_recall', val_metrics['recall_macro']), ('lr', optimizer.param_groups[0]['lr'])]:
            history[k].append(v)

        is_best = val_metrics['f1_macro'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1_macro']
            torch.save({'model_state': student.state_dict(), 'f1': best_f1},
                       output_dir / 'checkpoints' / f'student_fold_{fold_id}_best.pt')
            patience = 0
        else:
            patience += 1

        logger.log(f"  Student Fold{fold_id} KD Ep{epoch+1:3d}/{remaining} | "
                   f"TrA:{train_acc*100:.1f}% VaA:{val_metrics['accuracy']*100:.1f}% | "
                   f"F1:{val_metrics['f1_macro']*100:.1f}%{' *BEST*' if is_best else ''}")

        if patience >= EARLY_STOP_PATIENCE:
            logger.log(f"  Early stop at epoch {epoch+1}")
            break

    # --- Stage 2: Fine-tune on real distribution ---
    logger.log(f"--- Stage 2: Fine-tune on real distribution ({STUDENT_EPOCHS_STAGE2} epochs) ---")
    ckpt = torch.load(output_dir / 'checkpoints' / f'student_fold_{fold_id}_best.pt',
                      map_location=DEVICE, weights_only=True)
    student.load_state_dict(ckpt['model_state'])

    train_ds_real = ICBHIDataset(train_samples, augment=True, spec_augment=True)
    train_loader_real = DataLoader(train_ds_real, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    optimizer_ft = optim.AdamW(student.parameters(), lr=STUDENT_LR_STAGE2, weight_decay=WEIGHT_DECAY)
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=STUDENT_EPOCHS_STAGE2, eta_min=1e-7)
    patience = 0

    for epoch in range(STUDENT_EPOCHS_STAGE2):
        train_loss, train_acc = train_student_epoch(student, teacher_models, train_loader_real, kd_criterion, optimizer_ft, DEVICE)
        val_metrics = evaluate(student, val_loader, eval_criterion, DEVICE)
        scheduler_ft.step()

        for k, v in [('train_loss', train_loss), ('train_acc', train_acc),
                      ('val_loss', val_metrics['loss']), ('val_acc', val_metrics['accuracy']),
                      ('val_f1', val_metrics['f1_macro']), ('val_precision', val_metrics['precision_macro']),
                      ('val_recall', val_metrics['recall_macro']), ('lr', optimizer_ft.param_groups[0]['lr'])]:
            history[k].append(v)

        is_best = val_metrics['f1_macro'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1_macro']
            torch.save({'model_state': student.state_dict(), 'f1': best_f1},
                       output_dir / 'checkpoints' / f'student_fold_{fold_id}_best.pt')
            patience = 0
        else:
            patience += 1

        logger.log(f"  Student Fold{fold_id} FT Ep{epoch+1:3d}/{STUDENT_EPOCHS_STAGE2} | "
                   f"TrA:{train_acc*100:.1f}% VaA:{val_metrics['accuracy']*100:.1f}% | "
                   f"F1:{val_metrics['f1_macro']*100:.1f}%{' *BEST*' if is_best else ''}")

        if patience >= 8:
            break

    # --- Final evaluation on test set ---
    ckpt = torch.load(output_dir / 'checkpoints' / f'student_fold_{fold_id}_best.pt',
                      map_location=DEVICE, weights_only=True)
    student.load_state_dict(ckpt['model_state'])

    test_metrics = evaluate(student, test_loader, eval_criterion, DEVICE)
    logger.log(f"\n  [TEST] Fold {fold_id} | Acc: {test_metrics['accuracy']*100:.2f}% | "
               f"F1: {test_metrics['f1_macro']*100:.2f}% | "
               f"Prec: {test_metrics['precision_macro']*100:.2f}% | "
               f"Rec: {test_metrics['recall_macro']*100:.2f}%")

    # Save metrics and visualizations
    json_data = save_all_metrics(test_metrics, history, fold_id, output_dir, CLASS_NAMES, logger)

    # Grad-CAM
    save_gradcam_samples(student, test_ds, output_dir, DEVICE)

    return student, test_metrics, history, json_data


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    global BATCH_SIZE, NUM_WORKERS, TEACHER_ENSEMBLE_SIZE
    global TEACHER_EPOCHS_STAGE1, TEACHER_EPOCHS_STAGE2
    global STUDENT_EPOCHS_STAGE1, STUDENT_EPOCHS_STAGE2, N_FOLDS

    parser = argparse.ArgumentParser(description='Knowledge Distillation v2 — ICBHI + Combined')
    parser.add_argument('--data_dir', type=str,
                        default='/home/iec/Parallel_Computing_on_FPGA/data/samples/ICBHI_final_database')
    parser.add_argument('--labels_file', type=str,
                        default='/home/iec/Parallel_Computing_on_FPGA/data/samples/labels.txt')
    parser.add_argument('--combined_dir', type=str,
                        default='/home/iec/Parallel_Computing_on_FPGA/data/combined/audio',
                        help='Combined dataset audio dir with COPD/Healthy/Non-COPD subdirs')
    parser.add_argument('--combined_labels', type=str,
                        default='/home/iec/Parallel_Computing_on_FPGA/data/combined/labels.csv',
                        help='Combined dataset labels CSV')
    parser.add_argument('--use_icbhi', action='store_true', default=True,
                        help='Include ICBHI dataset (default: True)')
    parser.add_argument('--use_combined', action='store_true', default=True,
                        help='Include combined dataset (default: True)')
    parser.add_argument('--no_icbhi', action='store_true', default=False,
                        help='Exclude ICBHI dataset')
    parser.add_argument('--no_combined', action='store_true', default=False,
                        help='Exclude combined dataset')
    parser.add_argument('--output_dir', type=str, default='./output_distillation_v2')
    parser.add_argument('--n_folds', type=int, default=N_FOLDS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--n_teachers', type=int, default=TEACHER_ENSEMBLE_SIZE)
    parser.add_argument('--teacher_epochs_s1', type=int, default=TEACHER_EPOCHS_STAGE1)
    parser.add_argument('--teacher_epochs_s2', type=int, default=TEACHER_EPOCHS_STAGE2)
    parser.add_argument('--student_epochs_s1', type=int, default=STUDENT_EPOCHS_STAGE1)
    parser.add_argument('--student_epochs_s2', type=int, default=STUDENT_EPOCHS_STAGE2)
    parser.add_argument('--dry_run', action='store_true', help='Quick test: 1 fold, 2 epochs')
    args = parser.parse_args()

    # Handle dataset flags
    if args.no_icbhi:
        args.use_icbhi = False
    if args.no_combined:
        args.use_combined = False

    # Override globals with CLI args
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    TEACHER_ENSEMBLE_SIZE = args.n_teachers
    TEACHER_EPOCHS_STAGE1 = args.teacher_epochs_s1
    TEACHER_EPOCHS_STAGE2 = args.teacher_epochs_s2
    STUDENT_EPOCHS_STAGE1 = args.student_epochs_s1
    STUDENT_EPOCHS_STAGE2 = args.student_epochs_s2
    N_FOLDS = args.n_folds

    if args.dry_run:
        TEACHER_EPOCHS_STAGE1 = 2
        TEACHER_EPOCHS_STAGE2 = 1
        STUDENT_EPOCHS_STAGE1 = 2
        STUDENT_EPOCHS_STAGE2 = 1
        TEACHER_ENSEMBLE_SIZE = 1
        N_FOLDS = 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)

    logger = TrainLogger(str(output_dir / 'training_log.txt'))
    logger.section("CONFIGURATION")
    logger.log(f"ICBHI data dir: {args.data_dir} (use={args.use_icbhi})")
    logger.log(f"ICBHI labels: {args.labels_file}")
    logger.log(f"Combined dir: {args.combined_dir} (use={args.use_combined})")
    logger.log(f"Combined labels: {args.combined_labels}")
    logger.log(f"Output: {output_dir}")
    logger.log(f"Device: {DEVICE}")
    logger.log(f"Folds: {N_FOLDS}, Teachers: {TEACHER_ENSEMBLE_SIZE}")
    logger.log(f"Teacher epochs: S1={TEACHER_EPOCHS_STAGE1}, S2={TEACHER_EPOCHS_STAGE2}")
    logger.log(f"Student epochs: S1={STUDENT_EPOCHS_STAGE1}, S2={STUDENT_EPOCHS_STAGE2}")
    logger.log(f"KD: T={KD_TEMPERATURE}, alpha={KD_ALPHA}")
    logger.log(f"Batch size: {BATCH_SIZE}, Focal gamma: {FOCAL_GAMMA}, Mixup alpha: {MIXUP_ALPHA}")
    logger.log(f"Dry run: {args.dry_run}")

    # --- Build dataset(s) ---
    logger.section("DATA LOADING")
    
    icbhi_samples = []
    combined_samples = []
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    # Load ICBHI dataset
    if args.use_icbhi and os.path.isdir(args.data_dir):
        logger.log("--- Loading ICBHI dataset ---")
        icbhi_samples, class_to_idx = build_dataset_from_icbhi(
            args.data_dir, args.labels_file, logger
        )
        logger.log(f"ICBHI: {len(icbhi_samples)} samples")
    else:
        logger.log("ICBHI dataset: SKIPPED")
    
    # Load combined dataset
    if args.use_combined and os.path.isdir(args.combined_dir):
        logger.log("--- Loading Combined dataset ---")
        combined_samples, _ = build_dataset_from_combined(
            args.combined_dir, args.combined_labels, logger
        )
        logger.log(f"Combined: {len(combined_samples)} samples")
    else:
        logger.log("Combined dataset: SKIPPED")
    
    # Merge datasets
    if icbhi_samples and combined_samples:
        logger.log("--- Merging ICBHI + Combined ---")
        samples = merge_datasets(icbhi_samples, combined_samples, logger)
    elif icbhi_samples:
        samples = icbhi_samples
        # Ensure patient_id is string
        for s in samples:
            s['patient_id'] = str(s['patient_id'])
            s.setdefault('source', 'ICBHI')
    elif combined_samples:
        samples = combined_samples
    else:
        logger.log("ERROR: No samples loaded from any dataset!")
        sys.exit(1)
    
    logger.log(f"\nTotal samples for training: {len(samples)}")
    class_dist = Counter([s['class_name'] for s in samples])
    logger.log(f"Final class distribution: {dict(class_dist)}")

    # Patient-wise split (patient_ids are strings now for both ICBHI and DS2)
    patient_ids = np.array([s['patient_id'] for s in samples])
    labels_arr = np.array([s['class_idx'] for s in samples])

    try:
        gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        folds = list(gkf.split(np.arange(len(samples)), labels_arr, patient_ids))
    except Exception:
        gkf = GroupKFold(n_splits=N_FOLDS)
        folds = list(gkf.split(np.arange(len(samples)), labels_arr, patient_ids))

    logger.log(f"Created {len(folds)} patient-wise folds")

    # --- Run folds ---
    all_fold_results = []

    for fold_id, (train_val_idx, test_idx) in enumerate(folds):
        logger.section(f"FOLD {fold_id + 1}/{N_FOLDS}")

        # Split train/val within train_val (patient-wise)
        tv_patients = patient_ids[train_val_idx]
        tv_labels = labels_arr[train_val_idx]
        unique_pats = np.unique(tv_patients)
        pat_labels = {p: tv_labels[np.where(tv_patients == p)[0][0]] for p in unique_pats}
        plabels = np.array([pat_labels[p] for p in unique_pats])

        from sklearn.model_selection import train_test_split
        try:
            train_pats, val_pats = train_test_split(
                unique_pats, test_size=0.15, random_state=42, stratify=plabels
            )
        except ValueError:
            train_pats, val_pats = train_test_split(
                unique_pats, test_size=0.15, random_state=42
            )

        train_idx = train_val_idx[np.isin(tv_patients, train_pats)]
        val_idx = train_val_idx[np.isin(tv_patients, val_pats)]

        train_samples_fold = [samples[i] for i in train_idx]
        val_samples_fold = [samples[i] for i in val_idx]
        test_samples_fold = [samples[i] for i in test_idx]

        logger.log(f"Train: {len(train_samples_fold)}, Val: {len(val_samples_fold)}, "
                   f"Test: {len(test_samples_fold)}")
        logger.log(f"Train classes: {Counter([s['class_name'] for s in train_samples_fold])}")
        logger.log(f"Test classes: {Counter([s['class_name'] for s in test_samples_fold])}")

        # --- Train teacher ensemble ---
        teacher_models = []
        for t_idx in range(TEACHER_ENSEMBLE_SIZE):
            np.random.seed(42 + t_idx * 17)
            torch.manual_seed(42 + t_idx * 17)
            teacher, t_history = train_teacher_fold(
                fold_id, train_samples_fold, val_samples_fold,
                test_samples_fold, output_dir, logger, teacher_idx=t_idx
            )
            teacher_models.append(teacher)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # --- Evaluate teacher ensemble on test ---
        test_ds_eval = ICBHIDataset(test_samples_fold, augment=False, spec_augment=False)
        test_loader_eval = DataLoader(test_ds_eval, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=NUM_WORKERS, pin_memory=True)

        # Ensemble predictions
        all_probs = []
        for t_model in teacher_models:
            t_model.eval()
            probs_list = []
            with torch.no_grad():
                for inputs, _ in test_loader_eval:
                    inputs = inputs.to(DEVICE)
                    out = F.softmax(t_model(inputs), dim=1)
                    probs_list.append(out.cpu().numpy())
            all_probs.append(np.concatenate(probs_list))

        ensemble_probs = np.mean(all_probs, axis=0)
        ensemble_preds = ensemble_probs.argmax(axis=1)
        test_labels_np = np.array([s['class_idx'] for s in test_samples_fold])

        teacher_acc = accuracy_score(test_labels_np, ensemble_preds)
        teacher_f1 = f1_score(test_labels_np, ensemble_preds, average='macro', zero_division=0)
        logger.log(f"\n  [TEACHER ENSEMBLE TEST] Fold {fold_id} | "
                   f"Acc: {teacher_acc*100:.2f}% | F1: {teacher_f1*100:.2f}%")

        # --- Train student with KD ---
        np.random.seed(42)
        torch.manual_seed(42)
        student, test_metrics, s_history, json_data = train_student_fold(
            fold_id, teacher_models, train_samples_fold, val_samples_fold,
            test_samples_fold, output_dir, logger
        )

        all_fold_results.append({
            'fold': fold_id,
            'teacher_acc': teacher_acc,
            'teacher_f1': teacher_f1,
            'student_acc': test_metrics['accuracy'],
            'student_f1': test_metrics['f1_macro'],
            'student_precision': test_metrics['precision_macro'],
            'student_recall': test_metrics['recall_macro'],
        })

        # Cleanup
        del teacher_models, student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # --- Summary ---
    logger.section("FINAL SUMMARY")
    logger.log(f"{'Fold':<6} {'T-Acc':>8} {'T-F1':>8} {'S-Acc':>8} {'S-F1':>8} {'S-Prec':>8} {'S-Rec':>8}")
    logger.log("-" * 60)
    for r in all_fold_results:
        logger.log(f"{r['fold']+1:<6} {r['teacher_acc']*100:>7.2f}% {r['teacher_f1']*100:>7.2f}% "
                   f"{r['student_acc']*100:>7.2f}% {r['student_f1']*100:>7.2f}% "
                   f"{r['student_precision']*100:>7.2f}% {r['student_recall']*100:>7.2f}%")

    # Averages
    avg = {k: np.mean([r[k] for r in all_fold_results])
           for k in ['teacher_acc', 'teacher_f1', 'student_acc', 'student_f1',
                      'student_precision', 'student_recall']}
    std = {k: np.std([r[k] for r in all_fold_results])
           for k in ['teacher_acc', 'teacher_f1', 'student_acc', 'student_f1',
                      'student_precision', 'student_recall']}

    logger.log("-" * 60)
    logger.log(f"{'AVG':<6} {avg['teacher_acc']*100:>7.2f}% {avg['teacher_f1']*100:>7.2f}% "
               f"{avg['student_acc']*100:>7.2f}% {avg['student_f1']*100:>7.2f}% "
               f"{avg['student_precision']*100:>7.2f}% {avg['student_recall']*100:>7.2f}%")
    logger.log(f"{'STD':<6} {std['teacher_acc']*100:>7.2f}% {std['teacher_f1']*100:>7.2f}% "
               f"{std['student_acc']*100:>7.2f}% {std['student_f1']*100:>7.2f}% "
               f"{std['student_precision']*100:>7.2f}% {std['student_recall']*100:>7.2f}%")

    # Save summary JSON
    summary = {
        'num_folds': N_FOLDS,
        'num_teachers': TEACHER_ENSEMBLE_SIZE,
        'kd_temperature': KD_TEMPERATURE,
        'kd_alpha': KD_ALPHA,
        'averages': {k: float(v) for k, v in avg.items()},
        'stds': {k: float(v) for k, v in std.items()},
        'fold_results': all_fold_results,
    }
    with open(output_dir / 'metrics' / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.log(f"\n  All outputs saved to: {output_dir}")
    logger.log(f"  Target Acc > 95%: {'ACHIEVED' if avg['student_acc'] > 0.95 else 'NOT YET'} "
               f"(current: {avg['student_acc']*100:.2f}%)")

    # Final summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fold_ids = [r['fold']+1 for r in all_fold_results]
    axes[0].bar(fold_ids, [r['student_acc']*100 for r in all_fold_results], color='steelblue', alpha=0.8)
    axes[0].axhline(y=95, color='r', linestyle='--', label='Target 95%')
    axes[0].axhline(y=avg['student_acc']*100, color='g', linestyle='--', label=f"Avg {avg['student_acc']*100:.1f}%")
    axes[0].set_xlabel('Fold'); axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Student Accuracy by Fold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(fold_ids, [r['student_f1']*100 for r in all_fold_results], color='coral', alpha=0.8)
    axes[1].axhline(y=avg['student_f1']*100, color='g', linestyle='--', label=f"Avg {avg['student_f1']*100:.1f}%")
    axes[1].set_xlabel('Fold'); axes[1].set_ylabel('F1 Score (%)'); axes[1].set_title('Student Macro F1 by Fold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics' / 'summary_chart.png', dpi=150)
    plt.close()

    logger.log("\nDone!")


if __name__ == '__main__':
    main()
