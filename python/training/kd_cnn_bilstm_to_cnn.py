#!/usr/bin/env python3
"""
================================================================================
Knowledge Distillation Pipeline: CNN–BiLSTM Teacher → Pure CNN Student
ICBHI 2017 — 3-Class COPD / Non-COPD / Healthy Classification
FPGA-Optimized Student
================================================================================

Pipeline:
  Phase 1: Train CNN–BiLSTM teacher on ICBHI 2017 (patient-level labels)
  Phase 2: Generate soft labels via temperature-scaled teacher inference
  Phase 3: Distill pure-CNN student using mixed KD + hard-label loss
  Phase 4: Fine-tune student, prepare for FPGA quantization

Key Features:
  - Teacher: 4-block CNN backbone + 2-layer BiLSTM + classifier head
  - Student: CNN6-style pure convolutional (Conv2D-BN-ReLU-Pool-GAP-FC)
  - Loss: Focal Loss (hard) + KL Divergence (soft) with temperature scaling
  - Patient-wise GroupKFold (no data leakage)
  - Comprehensive metrics + W&B logging + matplotlib visualization
  - FPGA-ready student: BN folding, INT8 quantization prep, < 500K params

Output:
  artifacts/training/kd_cnn_bilstm_to_cnn/
  ├── checkpoints/    # Best model weights per fold
  ├── metrics/        # JSON metrics, confusion matrices, ROC curves
  ├── figures/        # Training curves, distillation comparison plots
  └── training_log.txt

Usage:
    python python/training/kd_cnn_bilstm_to_cnn.py \\
        --data_dir data/sample_01/ICBHI_final_database \\
        --labels_file data/sample_01/labels.txt \\
        --output_dir artifacts/training/kd_cnn_bilstm_to_cnn \\
        --wandb_project copd-kd-pipeline

    # Quick dry-run:
    python python/training/kd_cnn_bilstm_to_cnn.py --dry_run
================================================================================
"""

import os
import gc
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, precision_score,
    recall_score, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    balanced_accuracy_score, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from torchaudio.transforms import FrequencyMasking, TimeMasking
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import (
    ICBHI_2017_DIR, ICBHI_2017_LABELS, KD_PIPELINE_ARTIFACTS_DIR,
    ensure_dir,
)

warnings.filterwarnings('ignore')

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
TARGET_SR = 4000
SEGMENT_DURATION = 8  # seconds
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_DURATION  # 32000

# Mel-spectrogram parameters
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 128
FMIN = 50
FMAX = 2000

# Input shape for models: (1, N_MELS, time_frames)
# 8s @ 4kHz with hop=128 -> ~250 time frames; we resize to 224x224 for consistency
INPUT_HEIGHT = 128
INPUT_WIDTH = 224

# Training
N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3
CLASS_NAMES = ['COPD', 'Healthy', 'Non-COPD']

# Teacher (CNN-BiLSTM)
TEACHER_CNN_CHANNELS = [32, 64, 128, 256]
TEACHER_BILSTM_HIDDEN = 128
TEACHER_BILSTM_LAYERS = 2
TEACHER_DROPOUT = 0.4
TEACHER_EPOCHS = 60
TEACHER_LR = 1e-3
TEACHER_WEIGHT_DECAY = 0.01

# Student (CNN6 — lightweight, FPGA-friendly)
STUDENT_CNN_CHANNELS = [16, 32, 64, 128, 128, 128]
STUDENT_DROPOUT = 0.3
STUDENT_EPOCHS = 80
STUDENT_LR = 3e-4
STUDENT_WEIGHT_DECAY = 0.005

# Distillation
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.7  # weight for KD loss (1-alpha for hard loss)

# Regularization
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 20
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0

# Batch
BATCH_SIZE = 32
NUM_WORKERS = 4

# Disease-to-3-class mapping
DISEASE_TO_CLASS = {
    'COPD': 'COPD',
    'Healthy': 'Healthy',
    'URTI': 'Non-COPD',
    'Asthma': 'Non-COPD',
    'LRTI': 'Non-COPD',
    'Bronchiectasis': 'Non-COPD',
    'Bronchiolitis': 'Non-COPD',
    'Pneumonia': 'Non-COPD',
}


# ==============================================================================
# LOGGER
# ==============================================================================
class PipelineLogger:
    def __init__(self, log_path: str, use_wandb: bool = False,
                 wandb_project: str = "copd-kd-pipeline",
                 wandb_run_name: str = None):
        self.log_path = log_path
        self.start_time = time.time()
        self.use_wandb = use_wandb and HAS_WANDB
        self._init_logfile()
        if self.use_wandb:
            run_name = wandb_run_name or f"kd-pipeline-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project=wandb_project, name=run_name, config=self._get_wandb_config())

    def _init_logfile(self):
        ensure_dir(Path(self.log_path).parent)
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("  CNN-BiLSTM → CNN KD Pipeline — ICBHI 2017 COPD Classification\n")
            f.write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Device: {DEVICE}\n")
            f.write("=" * 80 + "\n\n")

    def _get_wandb_config(self):
        return {
            'target_sr': TARGET_SR, 'segment_duration': SEGMENT_DURATION,
            'n_mels': N_MELS, 'n_fft': N_FFT, 'hop_length': HOP_LENGTH,
            'teacher_bilstm_hidden': TEACHER_BILSTM_HIDDEN,
            'teacher_bilstm_layers': TEACHER_BILSTM_LAYERS,
            'student_cnn_channels': STUDENT_CNN_CHANNELS,
            'kd_temperature': KD_TEMPERATURE, 'kd_alpha': KD_ALPHA,
            'focal_gamma': FOCAL_GAMMA, 'label_smoothing': LABEL_SMOOTHING,
            'batch_size': BATCH_SIZE, 'n_folds': N_FOLDS,
            'teacher_epochs': TEACHER_EPOCHS, 'student_epochs': STUDENT_EPOCHS,
        }

    def log(self, msg: str = "", pbar: bool = None):
        elapsed = time.time() - self.start_time
        timestamp = f"[{elapsed/60:.1f}min]"
        full = f"{timestamp} {msg}"
        print(full)
        with open(self.log_path, 'a') as f:
            f.write(full + "\n")

    def log_metrics(self, metrics: dict, step: int = None, prefix: str = ""):
        """Log metrics to console, logfile, and W&B."""
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                self.log(f"  {prefix}{k}: {float(v):.4f}")
        if self.use_wandb:
            wandb_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    wandb_metrics[f"{prefix}{k}"] = float(v)
            if wandb_metrics:
                if step is not None:
                    wandb.log(wandb_metrics, step=step)
                else:
                    wandb.log(wandb_metrics)

    def section(self, title: str):
        self.log("\n" + "=" * 80)
        self.log(f"  {title}")
        self.log("=" * 80)

    def finish(self):
        self.log("\n" + "=" * 80)
        self.log(f"  Pipeline finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        elapsed = time.time() - self.start_time
        self.log(f"  Total time: {elapsed/60:.1f} minutes")
        self.log("=" * 80)
        if self.use_wandb:
            wandb.finish()


# ==============================================================================
# DATA LOADING & LABEL MAPPING
# ==============================================================================
def parse_patient_labels(labels_path: str) -> Dict[int, str]:
    """Parse ICBHI labels file: patient_id<TAB>disease"""
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


def build_icbhi_dataset(data_dir: str, labels_path: str,
                        logger: PipelineLogger) -> List[Dict]:
    """
    Build dataset from ICBHI 2017 directory.
    Each WAV file → assigned its patient's disease label → mapped to 3-class.

    Label mapping strategy:
      - "COPD"                    → COPD
      - "Healthy"                 → Healthy
      - All other diseases        → Non-COPD
        (URTI, Asthma, Bronchiectasis, Bronchiolitis, LRTI, Pneumonia)

    Patient-level labels propagate to all respiratory cycle recordings
    for that patient (patient-wise labeling).

    Notes on edge cases:
      - Patients with missing diagnosis: skipped (logged)
      - Comorbidities not explicitly present in ICBHI labels.txt; if they
        appear (e.g., "Heart Failure + COPD"), they would map to Non-COPD
        since the label doesn't match "COPD" exactly.
      - The original ICBHI diagnostic labels are at patient-level, so all
        recordings from a COPD patient are labeled COPD.
    """
    patient_labels = parse_patient_labels(labels_path)
    logger.log(f"Loaded {len(patient_labels)} patient diagnosis records")

    # Count original disease distribution
    disease_counts = Counter(patient_labels.values())
    logger.log(f"Original disease distribution: {dict(disease_counts)}")

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    wav_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav')])
    logger.log(f"Found {len(wav_files)} WAV files in {data_dir}")

    samples = []
    skipped_no_label = 0
    skipped_unknown_disease = 0

    for wav_file in wav_files:
        pid = int(wav_file.split('_')[0])
        if pid not in patient_labels:
            skipped_no_label += 1
            continue

        disease = patient_labels[pid]
        if disease not in DISEASE_TO_CLASS:
            skipped_unknown_disease += 1
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

    logger.log(f"Valid samples: {len(samples)}")
    logger.log(f"Skipped (no label): {skipped_no_label}")
    logger.log(f"Skipped (unknown disease): {skipped_unknown_disease}")

    class_dist = Counter([s['class_name'] for s in samples])
    logger.log(f"3-class distribution: {dict(class_dist)}")
    for cls_name in CLASS_NAMES:
        cnt = class_dist.get(cls_name, 0)
        logger.log(f"  {cls_name}: {cnt} samples ({cnt/len(samples)*100:.1f}%)")

    return samples, class_to_idx


# ==============================================================================
# AUDIO PREPROCESSING
# ==============================================================================
def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def preprocess_audio(wav_path: str, target_sr: int = TARGET_SR,
                     segment_len: int = SEGMENT_SAMPLES,
                     lowcut: int = 50, highcut: int = 2000) -> np.ndarray:
    """
    Full audio preprocessing pipeline:
      1. Load WAV with original sampling rate
      2. Convert to mono (mean channels if stereo)
      3. Band-pass filter (50–2000 Hz), order 4 Butterworth
      4. Resample to 4 kHz
      5. Normalize amplitude to [-1, 1]
      6. Pad/crop to fixed segment length (8 seconds)

    Returns: float32 np.ndarray of shape (segment_len,)
    """
    try:
        sr, audio = wavfile.read(wav_path)
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        else:
            audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to target SR
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, max(num_samples, 1))

        # Band-pass filter: 50–2000 Hz covers respiratory sound frequency range
        effective_highcut = min(highcut, target_sr // 2 - 1)
        b, a = butter_bandpass(lowcut, effective_highcut, target_sr, order=4)
        audio = signal.filtfilt(b, a, audio).astype(np.float32)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio)) + 1e-10
        audio = audio / max_val

        # Pad or crop to fixed length
        if len(audio) < segment_len:
            repeats = segment_len // len(audio) + 1
            audio = np.tile(audio, repeats)[:segment_len]
        elif len(audio) > segment_len:
            start = (len(audio) - segment_len) // 2
            audio = audio[start:start + segment_len]

        return audio.astype(np.float32)

    except Exception:
        return np.zeros(segment_len, dtype=np.float32)


# ==============================================================================
# FEATURE EXTRACTION: Log-Mel Spectrogram
# ==============================================================================
def compute_log_mel_spectrogram(audio: np.ndarray, sr: int = TARGET_SR,
                                n_mels: int = N_MELS, n_fft: int = N_FFT,
                                hop_length: int = HOP_LENGTH,
                                fmin: int = FMIN, fmax: int = FMAX) -> np.ndarray:
    """
    Compute log-Mel spectrogram.

    Parameters:
      - FFT size: 512 (yields 257 frequency bins at 4kHz)
      - Hop length: 128 (32ms hop at 4kHz)
      - Window length: 512 (implicit in STFT, 128ms at 4kHz)
      - Mel bins: 128
      - Frequency range: 50–2000 Hz (covers respiratory sounds)

    Returns: np.ndarray of shape (n_mels, time_frames)
    """
    if HAS_LIBROSA:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, fmin=fmin, fmax=fmax,
            window='hann'
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
    else:
        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft,
                                noverlap=n_fft - hop_length, window='hann')
        power = np.abs(Zxx) ** 2

        # Mel filterbank
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        bin_points = np.clip(bin_points, 0, len(f) - 1)

        fb = np.zeros((n_mels, len(f)))
        for i in range(n_mels):
            for j in range(bin_points[i], min(bin_points[i + 1], len(f))):
                denom = max(bin_points[i + 1] - bin_points[i], 1)
                fb[i, j] = (j - bin_points[i]) / denom
            for j in range(bin_points[i + 1], min(bin_points[i + 2], len(f))):
                denom = max(bin_points[i + 2] - bin_points[i + 1], 1)
                fb[i, j] = (bin_points[i + 2] - j) / denom

        mel_spec = np.dot(fb, power)
        log_mel = 10 * np.log10(mel_spec + 1e-10)
        log_mel -= np.max(log_mel)

    return log_mel.astype(np.float32)


def spectrogram_to_model_input(spec: np.ndarray,
                                target_height: int = INPUT_HEIGHT,
                                target_width: int = INPUT_WIDTH) -> np.ndarray:
    """
    Resize spectrogram to model input dimensions via zoom interpolation.
    Input: (n_mels, time_frames)
    Output: (target_height, target_width) normalized to [0, 1]
    """
    # Normalize to [0, 1]
    spec_norm = spec - spec.min()
    if spec_norm.max() > 0:
        spec_norm = spec_norm / spec_norm.max()

    # Resize
    h_ratio = target_height / spec_norm.shape[0]
    w_ratio = target_width / spec_norm.shape[1]
    resized = zoom(spec_norm, (h_ratio, w_ratio), order=1)
    resized = resized[:target_height, :target_width]

    return np.clip(resized, 0, 1).astype(np.float32)


# ==============================================================================
# AUGMENTATION
# ==============================================================================
class AudioAugmenter:
    """
    Audio-domain augmentations applied BEFORE spectrogram computation.
    Applied to BOTH teacher and student training.

    Augmentations:
      - Time shift (±15%)
      - Gaussian noise injection (SNR 10–30 dB)
      - Time masking (simulated via time stretch 85%–115%)
    """
    def __init__(self, probability: float = 0.7):
        self.probability = probability

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio
        audio = audio.copy()

        # Time shift
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-0.15, 0.15) * len(audio))
            audio = np.roll(audio, shift)

        # Gaussian noise injection
        if np.random.random() < 0.5:
            snr_db = np.random.uniform(10, 30)
            sig_power = np.mean(audio ** 2) + 1e-10
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
            audio = audio + noise

        # Time stretch
        if np.random.random() < 0.3:
            rate = np.random.uniform(0.85, 1.15)
            orig_len = len(audio)
            stretched = signal.resample(audio, int(orig_len * rate))
            if len(stretched) > orig_len:
                audio = stretched[:orig_len]
            else:
                audio = np.pad(stretched, (0, orig_len - len(stretched)), mode='constant')

        return np.clip(audio, -1.0, 1.0).astype(np.float32)


class SpecAugment:
    """
    SpecAugment applied to spectrogram tensors AFTER feature extraction.

    Applied to BOTH teacher and student training.

    Augmentations:
      - Frequency masking (up to 24 Mel bins, ~19% of 128 bins)
      - Time masking (up to 45 frames, ~20% of 224 frames)
    """
    def __init__(self, freq_mask_param: int = 24, time_mask_param: int = 45,
                 p: float = 0.7):
        self.p = p
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        if HAS_TORCHAUDIO:
            self.freq_mask_fn = FrequencyMasking(freq_mask_param)
            self.time_mask_fn = TimeMasking(time_mask_param)
        else:
            self.freq_mask_fn = self.time_mask_fn = None

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return spec

        if self.freq_mask_fn is not None:
            spec = self.freq_mask_fn(spec)
            spec = self.time_mask_fn(spec)
            # Apply second round with 30% probability for stronger augmentation
            if np.random.random() < 0.3:
                spec = self.freq_mask_fn(spec)
                spec = self.time_mask_fn(spec)
            return spec

        # Manual fallback
        c, h, w = spec.shape
        out = spec.clone()
        # Frequency mask
        f = int(np.random.uniform(0, min(self.freq_mask_param, h)))
        f0 = np.random.randint(0, max(1, h - f))
        out[:, f0:f0 + f, :] = 0
        # Time mask
        t = int(np.random.uniform(0, min(self.time_mask_param, w)))
        t0 = np.random.randint(0, max(1, w - t))
        out[:, :, t0:t0 + t] = 0
        return out


# ==============================================================================
# DATASET
# ==============================================================================
class ICBHIMelDataset(Dataset):
    """On-the-fly preprocessing: WAV → preprocess → log-Mel spectrogram."""
    def __init__(self, samples: List[Dict], augment: bool = False,
                 spec_augment: bool = False):
        self.samples = samples
        self.augmenter = AudioAugmenter() if augment else None
        self.spec_aug = SpecAugment() if spec_augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = preprocess_audio(s['wav_path'])

        if self.augmenter:
            audio = self.augmenter(audio)

        # Compute log-Mel spectrogram
        log_mel = compute_log_mel_spectrogram(audio)
        spec_img = spectrogram_to_model_input(log_mel)

        # Add channel dimension: (1, H, W) for single-channel input
        spec_tensor = torch.from_numpy(spec_img).unsqueeze(0).float()

        if self.spec_aug:
            spec_tensor = self.spec_aug(spec_tensor)

        return spec_tensor, s['class_idx']


def oversample_balance(samples: List[Dict]) -> List[Dict]:
    """Oversample minority classes to match majority class count."""
    counts = Counter([s['class_idx'] for s in samples])
    max_count = max(counts.values())
    balanced = []
    for cls_idx in counts:
        cls_samples = [s for s in samples if s['class_idx'] == cls_idx]
        balanced.extend(cls_samples)
        n_extra = max_count - len(cls_samples)
        if n_extra > 0:
            extras = [cls_samples[i % len(cls_samples)] for i in range(n_extra)]
            balanced.extend(extras)
    np.random.shuffle(balanced)
    return balanced


# ==============================================================================
# TEACHER MODEL: CNN–BiLSTM
# ==============================================================================
class CNNBackbone(nn.Module):
    """
    CNN feature extractor backbone for teacher.

    Architecture:
      Block 1: Conv2D(1→32, 3×3) → BN → ReLU → MaxPool(2×2)
      Block 2: Conv2D(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
      Block 3: Conv2D(64→128, 3×3) → BN → ReLU → MaxPool(2×2)
      Block 4: Conv2D(128→256, 3×3) → BN → ReLU → AdaptiveAvgPool → (H'=4)

    Output shape: (B, 256, 4, W') — preserves time dimension for BiLSTM
    """
    def __init__(self, in_channels: int = 1, channels: List[int] = None,
                 dropout: float = 0.3):
        super().__init__()
        if channels is None:
            channels = TEACHER_CNN_CHANNELS
        self.channels = channels

        layers = []
        prev_c = in_channels
        for i, c in enumerate(channels):
            layers.append(nn.Conv2d(prev_c, c, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout2d(dropout * 0.5))
            prev_c = c

        self.conv_blocks = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, None))

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        return x  # (B, 256, 4, W')


class TeacherCNNBiLSTM(nn.Module):
    """
    CNN–BiLSTM Teacher for Respiratory Sound Classification.

    Architecture:
      Input: (B, 1, 128, 224) log-Mel spectrogram
      ↓
      CNN Backbone: 4 conv blocks → (B, 256, 4, W')
      ↓
      Reshape: (B, 256, 4, W') → (B, W', 1024)  [flatten freq dim]
      ↓
      BiLSTM: 2 layers, hidden=128 → (B, W', 256)
      ↓
      Global Max Pooling over time → (B, 256)
      ↓
      Classifier: FC(256→128) → ReLU → Dropout → FC(128→3)

    Parameters: ~1.5–2M (depending on input dimensions)
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 cnn_channels: List[int] = None,
                 lstm_hidden: int = TEACHER_BILSTM_HIDDEN,
                 lstm_layers: int = TEACHER_BILSTM_LAYERS,
                 dropout: float = TEACHER_DROPOUT):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = TEACHER_CNN_CHANNELS

        self.cnn = CNNBackbone(in_channels=1, channels=cnn_channels,
                               dropout=dropout)

        # After CNN + AdaptiveAvgPool(4, None):
        # channels[-1] = 256, freq_dim = 4
        # Flattened: 256 * 4 = 1024 features per time step
        lstm_input_size = cnn_channels[-1] * 4

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_output_size = lstm_hidden * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # CNN
        features = self.cnn(x)  # (B, C, H, W)

        # Reshape for LSTM: (B, C, H, W) → (B, W, C*H)
        B, C, H, W = features.size()
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
        features = features.view(B, W, C * H)  # (B, W, C*H)

        # BiLSTM
        lstm_out, _ = self.lstm(features)  # (B, W, hidden*2)

        # Global Max Pooling over time dimension
        pooled, _ = lstm_out.max(dim=1)  # (B, hidden*2)

        # Classifier
        return self.classifier(pooled)


# ==============================================================================
# STUDENT MODEL: CNN6 — Pure CNN, FPGA-Friendly
# ==============================================================================
class StudentCNN6(nn.Module):
    """
    CNN6 — Lightweight pure CNN student optimized for FPGA deployment.

    Architecture:
      Input: (B, 1, 128, 224) log-Mel spectrogram
      ↓
      Block 1: Conv2D(1→16, 3×3) → BN → ReLU → MaxPool(2×2)  → (64, 112)
      Block 2: Conv2D(16→32, 3×3) → BN → ReLU → MaxPool(2×2) → (32, 56)
      Block 3: Conv2D(32→64, 3×3) → BN → ReLU → MaxPool(2×2) → (16, 28)
      Block 4: Conv2D(64→128, 3×3) → BN → ReLU → MaxPool(2×2) → (8, 14)
      Block 5: Conv2D(128→128, 3×3) → BN → ReLU → MaxPool(2×2) → (4, 7)
      Block 6: Conv2D(128→128, 3×3) → BN → ReLU                    → (4, 7)
      ↓
      Global Average Pooling → (128)
      ↓
      FC: 128 → 64 → ReLU → 3

    Design principles for FPGA:
      - Conv2D + BN + ReLU only (no DepthwiseConv, no SE, no residual)
      - Small kernel size (3×3) → efficient hardware mapping
      - Global Average Pooling instead of Flatten → minimal FC input
      - BN layers can be folded into Conv weights for deployment
      - ReLU activation → simple threshold logic in hardware
      - Estimated parameters: ~200K–300K

    FPGA-friendly characteristics:
      - Regular data flow, no skip connections
      - All convolutions have stride=1, downsampling via MaxPool only
      - INT8 quantization-friendly architecture
      - Small FC head (128→64→3)
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 channels: List[int] = None,
                 dropout: float = STUDENT_DROPOUT):
        super().__init__()
        if channels is None:
            channels = STUDENT_CNN_CHANNELS

        self.channels = channels
        blocks = []
        prev_c = 1  # single-channel spectrogram input
        for i, c in enumerate(channels):
            blocks.append(nn.Conv2d(prev_c, c, kernel_size=3, padding=1))
            blocks.append(nn.BatchNorm2d(c))
            blocks.append(nn.ReLU(inplace=True))
            # Pool after each block except the last
            if i < len(channels) - 1:
                blocks.append(nn.MaxPool2d(2))
            prev_c = c

        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def fold_batch_norm(self):
        """
        Fold BatchNorm into preceding Conv2d layers for FPGA deployment.
        Conv: y = W*x + b
        BN:   y = gamma*(y - mean)/sqrt(var+eps) + beta
        Folded: W' = gamma*W/sqrt(var+eps), b' = gamma*(b-mean)/sqrt(var+eps) + beta
        """
        folded = StudentCNN6(num_classes=NUM_CLASSES, channels=self.channels)
        folded.eval()

        new_blocks = []
        prev_block = None
        for name, module in self.features.named_children():
            if isinstance(module, nn.Conv2d):
                prev_block = module
            elif isinstance(module, nn.BatchNorm2d) and prev_block is not None:
                # Fold BN into prev Conv
                gamma = module.weight.data
                beta = module.bias.data
                mean = module.running_mean
                var = module.running_var
                eps = module.eps
                std = torch.sqrt(var + eps)

                # New weights
                new_weight = prev_block.weight.data * (gamma / std).view(-1, 1, 1, 1)
                new_bias = gamma * (prev_block.bias.data - mean) / std + beta

                new_conv = nn.Conv2d(
                    prev_block.in_channels, prev_block.out_channels,
                    prev_block.kernel_size, prev_block.stride,
                    prev_block.padding, bias=True
                )
                new_conv.weight.data = new_weight
                new_conv.bias.data = new_bias
                new_blocks.append(new_conv)
                prev_block = None
            elif not isinstance(module, nn.BatchNorm2d):
                if isinstance(module, nn.ReLU):
                    new_blocks.append(nn.ReLU(inplace=True))
                elif isinstance(module, nn.MaxPool2d):
                    new_blocks.append(nn.MaxPool2d(module.kernel_size))

        folded.features = nn.Sequential(*new_blocks)
        folded.classifier = self.classifier
        return folded


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class weights (tensor of shape [num_classes])
        gamma: Focusing parameter (default 2.0)
        label_smoothing: Label smoothing factor (0.1 means 10% smoothing)
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
                smooth = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * smooth * log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).unsqueeze(0)
            loss = loss * alpha_t

        loss = loss.sum(dim=1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class DistillationLoss(nn.Module):
    """
    Combined Knowledge Distillation loss.

    L = alpha * L_KD + (1 - alpha) * L_hard

    where:
      L_KD = KL(softmax(student/T) || softmax(teacher/T)) * T^2
      L_hard = FocalLoss(student, true_labels)

    Temperature scaling formula:
      p_c(T) = exp(z_c / T) / sum_j(exp(z_j / T))

    T selection: Higher T → softer distribution, more info transfer.
    T=4.0 chosen as a balance between softness and discriminability
    for a 3-class task (typical range: 2–8 for audio classification).
    """
    def __init__(self, hard_loss_fn: nn.Module, temperature: float = KD_TEMPERATURE,
                 alpha: float = KD_ALPHA):
        super().__init__()
        self.hard_loss_fn = hard_loss_fn
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # KD loss (KL divergence)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        kd_loss = kd_loss * (self.temperature ** 2)

        # Hard label loss
        hard_loss = self.hard_loss_fn(student_logits, targets)

        total = self.alpha * kd_loss + (1 - self.alpha) * hard_loss
        return total, kd_loss.item(), hard_loss.item()


# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """Single training epoch for teacher (standard classification)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.3f}',
                          'acc': f'{correct/total*100:.1f}%'})
    return total_loss / total, correct / total


def train_distill_epoch(student: nn.Module, teacher: nn.Module,
                        loader: DataLoader, kd_criterion: DistillationLoss,
                        optimizer: optim.Optimizer,
                        device: torch.device) -> Tuple[float, float]:
    """Single training epoch for student with KD from teacher."""
    student.train()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="  KD-Train", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher(inputs)

        optimizer.zero_grad()
        student_logits = student(inputs)
        loss, kd_val, hard_val = kd_criterion(student_logits, teacher_logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_CLIP)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = student_logits.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.3f}',
                          'kd': f'{kd_val:.3f}',
                          'acc': f'{correct/total*100:.1f}%'})
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device,
             return_probs: bool = True) -> Dict:
    """
    Comprehensive evaluation returning all predictions, labels, and probabilities.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for inputs, labels in tqdm(loader, desc="  Eval", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        probs = F.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return {
        'loss': total_loss / len(all_labels),
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
    }


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  y_prob: np.ndarray,
                                  class_names: List[str]) -> Dict:
    """
    Compute ALL required metrics for the 3-class task.

    Primary metric: Macro F1-score (best for imbalanced multi-class)
    Secondary metrics: Balanced Accuracy, Sensitivity, Specificity

    Returns dict with all metrics.
    """
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': cm.tolist(),
        'per_class': {},
    }

    # Sensitivity & Specificity per class
    for i, cls_name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn + 1e-10)  # Recall
        specificity = tn / (tn + fp + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)

        metrics['per_class'][cls_name] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'recall': float(sensitivity),  # same as sensitivity
            'f1_score': float(f1),
            'support': int((y_true == i).sum()),
        }

    # Macro sensitivity & specificity
    metrics['macro_sensitivity'] = float(np.mean(
        [metrics['per_class'][c]['sensitivity'] for c in class_names]))
    metrics['macro_specificity'] = float(np.mean(
        [metrics['per_class'][c]['specificity'] for c in class_names]))

    # AUC (One-vs-Rest)
    try:
        y_bin = label_binarize(y_true, classes=range(n_classes))
        auc_scores = []
        for i in range(n_classes):
            if len(np.unique(y_bin[:, i])) < 2:
                auc_scores.append(0.5)
            else:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                auc_scores.append(float(auc(fpr, tpr)))
        metrics['per_class_auc'] = dict(zip(class_names, auc_scores))
        metrics['macro_auc'] = float(np.mean(auc_scores))
    except Exception:
        metrics['macro_auc'] = None

    # Classification report (for reference)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics['classification_report'] = report

    return metrics


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: str, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                    class_names: List[str], save_path: str):
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{cls} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1.5)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — One-vs-Rest', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_training_curves(history: Dict, save_path: str,
                         title: str = "Training Curves"):
    n_metrics = 4
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', lw=1.5, label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', lw=1.5, label='Val Loss')
    axes[0, 0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('Epoch')

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', lw=1.5, label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', lw=1.5, label='Val Acc')
    axes[0, 1].set_title('Accuracy', fontsize=13, fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('Epoch')

    # F1 Score
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', lw=2, label='Val Macro F1')
    axes[1, 0].plot(epochs, history['val_weighted_f1'], 'm--', lw=1.5, label='Val Weighted F1')
    axes[1, 0].set_title('F1 Scores', fontsize=13, fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Epoch')

    # Learning Rate
    if 'lr' in history:
        axes[1, 1].plot(epochs, history['lr'], 'k-', lw=1.5)
        axes[1, 1].set_title('Learning Rate', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
    else:
        axes[1, 1].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_per_class_metrics(metrics: Dict, class_names: List[str],
                           save_path: str):
    """Bar chart of per-class sensitivity, specificity, precision, F1."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.2

    for i, (metric_name, color) in enumerate(
        [('sensitivity', '#3498db'), ('specificity', '#e74c3c'),
         ('precision', '#2ecc71'), ('f1_score', '#f39c12')]
    ):
        values = [metrics['per_class'][c][metric_name] for c in class_names]
        ax.bar(x + i * width, values, width, label=metric_name.replace('_', ' ').title(),
               color=color, alpha=0.85)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_kd_comparison(teacher_metrics: Dict, student_metrics: Dict,
                       class_names: List[str], save_path: str):
    """Side-by-side comparison of teacher vs student performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    metrics_list = ['sensitivity', 'specificity', 'precision', 'f1_score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    x = np.arange(len(class_names))
    width = 0.2

    for ax, (title, data) in zip(axes, [('Teacher', teacher_metrics),
                                         ('Student', student_metrics)]):
        for i, (m, c) in enumerate(zip(metrics_list, colors)):
            vals = [data['per_class'][cls][m] for cls in class_names]
            ax.bar(x + i * width, vals, width, label=m.replace('_', ' ').title(),
                   color=c, alpha=0.85)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(class_names)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].set_ylabel('Score', fontsize=12)
    plt.suptitle('Teacher vs Student — Per-Class Performance',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def log_plots_to_wandb(figures: Dict[str, plt.Figure], logger: PipelineLogger):
    """Log matplotlib figures to W&B."""
    if not logger.use_wandb:
        return
    for name, fig in figures.items():
        wandb.log({name: wandb.Image(fig)})
        plt.close(fig)


# ==============================================================================
# PHASE 1: TRAIN TEACHER
# ==============================================================================
def train_teacher(fold_id: int, train_samples: List[Dict],
                  val_samples: List[Dict], output_dir: Path,
                  logger: PipelineLogger) -> Tuple[nn.Module, Dict, Dict]:
    """
    Phase 1: Train CNN-BiLSTM teacher.

    Strategy:
      - Focal Loss with class-balanced alpha weights
      - Cosine annealing LR schedule
      - Early stopping on validation macro F1
      - Patient-wise validation
    """
    logger.section(f"PHASE 1: Train Teacher — Fold {fold_id + 1}")

    # Balance training set
    train_balanced = oversample_balance(train_samples)
    logger.log(f"Train (balanced): {len(train_balanced)}, Val: {len(val_samples)}")
    logger.log(f"Train classes: {dict(Counter([s['class_name'] for s in train_balanced]))}")

    train_ds = ICBHIMelDataset(train_balanced, augment=True, spec_augment=True)
    val_ds = ICBHIMelDataset(val_samples, augment=False, spec_augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = TeacherCNNBiLSTM().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Teacher parameters: {n_params:,}")

    # Class-balanced alpha for focal loss
    counts = Counter([s['class_idx'] for s in train_balanced])
    total_n = sum(counts.values())
    alpha = torch.FloatTensor([
        total_n / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)
    ])
    alpha = alpha / alpha.sum() * NUM_CLASSES
    logger.log(f"Focal alpha weights: {alpha.tolist()}")

    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA,
                          label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=TEACHER_LR,
                            weight_decay=TEACHER_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS, eta_min=1e-6)

    history = defaultdict(list)
    best_f1 = -1
    patience_counter = 0
    best_epoch = 0

    for epoch in range(TEACHER_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_results = evaluate(model, val_loader, criterion, DEVICE)
        val_loss = val_results['loss']

        # Quick per-epoch metrics
        val_acc = accuracy_score(val_results['labels'], val_results['predictions'])
        val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                          average='macro', zero_division=0)
        val_wf1 = f1_score(val_results['labels'], val_results['predictions'],
                           average='weighted', zero_division=0)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_weighted_f1'].append(val_wf1)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            ckpt_dir = ensure_dir(output_dir / 'checkpoints')
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'f1': best_f1, 'val_acc': val_acc,
            }, ckpt_dir / f'teacher_fold_{fold_id}_best.pt')
        else:
            patience_counter += 1

        best_mark = " *BEST*" if is_best else ""
        logger.log(
            f"  T-Fold{fold_id+1} Ep{epoch+1:3d}/{TEACHER_EPOCHS} | "
            f"TrL:{train_loss:.4f} VaL:{val_loss:.4f} | "
            f"TrA:{train_acc*100:.1f}% VaA:{val_acc*100:.1f}% | "
            f"F1:{val_f1*100:.1f}% WF1:{val_wf1*100:.1f}%{best_mark}"
        )

        # Log to W&B
        logger.log_metrics({
            'teacher/train_loss': train_loss, 'teacher/val_loss': val_loss,
            'teacher/train_acc': train_acc, 'teacher/val_acc': val_acc,
            'teacher/val_f1': val_f1, 'teacher/val_weighted_f1': val_wf1,
            'teacher/lr': optimizer.param_groups[0]['lr'],
        }, step=epoch, prefix='teacher/fold_' + str(fold_id) + '/')

        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.log(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best checkpoint for final evaluation
    ckpt = torch.load(output_dir / 'checkpoints' / f'teacher_fold_{fold_id}_best.pt',
                      map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    logger.log(f"  Teacher best: epoch={best_epoch+1}, F1={best_f1*100:.2f}%")

    val_results_best = evaluate(model, val_loader, criterion, DEVICE)
    val_metrics = compute_comprehensive_metrics(
        val_results_best['labels'], val_results_best['predictions'],
        val_results_best['probabilities'], CLASS_NAMES)
    logger.log_metrics(val_metrics, prefix='teacher/final_val_')

    return model, history, val_metrics


# ==============================================================================
# PHASE 2 & 3: GENERATE SOFT LABELS & DISTILL STUDENT
# ==============================================================================
def train_student_distillation(fold_id: int, teacher: nn.Module,
                               train_samples: List[Dict],
                               val_samples: List[Dict],
                               test_samples: List[Dict],
                               output_dir: Path,
                               logger: PipelineLogger
                               ) -> Tuple[nn.Module, Dict, Dict]:
    """
    Phase 2+3: Generate soft labels from teacher and distill student.

    Phase 2 (implicit): During each batch, teacher forward pass generates
    soft probabilities with temperature scaling. This is done on-the-fly
    rather than pre-computing, which saves disk space and allows
    consistent augmentation between teacher and student views.

    Phase 3: Student trained with mixed loss:
      L = alpha * KL(softmax(S/T) || softmax(T/T)) * T^2
        + (1-alpha) * FocalLoss(S, y_true)

    Phase 4 (fine-tune): After KD, optional fine-tuning with reduced LR.
    """
    logger.section(f"PHASE 2-4: Student Distillation — Fold {fold_id + 1}")

    # Balance training set
    train_balanced = oversample_balance(train_samples)
    logger.log(f"Train (balanced): {len(train_balanced)}, Val: {len(val_samples)}, "
               f"Test: {len(test_samples)}")

    train_ds = ICBHIMelDataset(train_balanced, augment=True, spec_augment=True)
    val_ds = ICBHIMelDataset(val_samples, augment=False, spec_augment=False)
    test_ds = ICBHIMelDataset(test_samples, augment=False, spec_augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Student model
    student = StudentCNN6().to(DEVICE)
    n_params = student.count_parameters()
    logger.log(f"Student parameters: {n_params:,} (target: < 500K for FPGA)")

    # Loss functions
    counts = Counter([s['class_idx'] for s in train_balanced])
    total_n = sum(counts.values())
    alpha = torch.FloatTensor([
        total_n / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)
    ])
    alpha = alpha / alpha.sum() * NUM_CLASSES

    hard_criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA,
                               label_smoothing=LABEL_SMOOTHING)
    kd_criterion = DistillationLoss(hard_criterion, temperature=KD_TEMPERATURE,
                                    alpha=KD_ALPHA)
    eval_criterion = nn.CrossEntropyLoss()

    # --- Stage 1: Head warmup (10 epochs, freeze CNN features) ---
    # Freeze feature extractor
    for p in student.features.parameters():
        p.requires_grad = False
    head_optimizer = optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=STUDENT_LR, weight_decay=STUDENT_WEIGHT_DECAY)
    head_warmup_epochs = 10

    history = defaultdict(list)
    best_f1 = -1
    patience_counter = 0

    logger.log(f"--- Stage 1: Head warmup ({head_warmup_epochs} epochs, backbone frozen) ---")
    for epoch in range(head_warmup_epochs):
        train_loss, train_acc = train_distill_epoch(
            student, teacher, train_loader, kd_criterion, head_optimizer, DEVICE)
        val_results = evaluate(student, val_loader, eval_criterion, DEVICE)
        val_acc = accuracy_score(val_results['labels'], val_results['predictions'])
        val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                          average='macro', zero_division=0)
        val_wf1 = f1_score(val_results['labels'], val_results['predictions'],
                           average='weighted', zero_division=0)

        _append_history(history, train_loss, train_acc, val_results['loss'],
                        val_acc, val_f1, val_wf1, head_optimizer)

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            patience_counter = 0
            _save_student(student, output_dir, fold_id)
        else:
            patience_counter += 1

        logger.log(
            f"  S-Fold{fold_id+1} Warmup Ep{epoch+1:3d}/{head_warmup_epochs} | "
            f"TrA:{train_acc*100:.1f}% VaA:{val_acc*100:.1f}% | "
            f"F1:{val_f1*100:.1f}%{' *BEST*' if is_best else ''}"
        )

    # --- Stage 2: Full KD training (unfreeze all) ---
    for p in student.parameters():
        p.requires_grad = True
    full_optimizer = optim.AdamW(student.parameters(), lr=STUDENT_LR * 0.3,
                                  weight_decay=STUDENT_WEIGHT_DECAY)
    remaining_epochs = STUDENT_EPOCHS - head_warmup_epochs
    full_scheduler = CosineAnnealingLR(full_optimizer, T_max=remaining_epochs,
                                        eta_min=1e-6)
    patience_counter = 0

    logger.log(f"--- Stage 2: Full KD ({remaining_epochs} epochs, all unfrozen) ---")
    for epoch in range(remaining_epochs):
        train_loss, train_acc = train_distill_epoch(
            student, teacher, train_loader, kd_criterion, full_optimizer, DEVICE)
        val_results = evaluate(student, val_loader, eval_criterion, DEVICE)
        val_acc = accuracy_score(val_results['labels'], val_results['predictions'])
        val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                          average='macro', zero_division=0)
        val_wf1 = f1_score(val_results['labels'], val_results['predictions'],
                           average='weighted', zero_division=0)

        _append_history(history, train_loss, train_acc, val_results['loss'],
                        val_acc, val_f1, val_wf1, full_optimizer)
        full_scheduler.step()

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            patience_counter = 0
            _save_student(student, output_dir, fold_id)
        else:
            patience_counter += 1

        total_epoch = epoch + head_warmup_epochs
        logger.log(
            f"  S-Fold{fold_id+1} KD Ep{total_epoch+1:3d}/{STUDENT_EPOCHS} | "
            f"TrA:{train_acc*100:.1f}% VaA:{val_acc*100:.1f}% | "
            f"F1:{val_f1*100:.1f}%{' *BEST*' if is_best else ''}"
        )

        logger.log_metrics({
            'student/train_loss': train_loss, 'student/val_loss': val_results['loss'],
            'student/train_acc': train_acc, 'student/val_acc': val_acc,
            'student/val_f1': val_f1, 'student/val_weighted_f1': val_wf1,
            'student/lr': full_optimizer.param_groups[0]['lr'],
        }, step=total_epoch, prefix='student/fold_' + str(fold_id) + '/')

        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.log(f"  Early stopping at total epoch {total_epoch+1}")
            break

    # --- Stage 3: Fine-tune on real distribution (optional, 15 epochs) ---
    logger.log(f"--- Stage 3: Fine-tune on real distribution (15 epochs) ---")
    train_ds_real = ICBHIMelDataset(train_samples, augment=True, spec_augment=True)
    train_loader_real = DataLoader(train_ds_real, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    # Reload best
    _load_student(student, output_dir, fold_id)

    ft_optimizer = optim.AdamW(student.parameters(), lr=1e-5,
                                weight_decay=STUDENT_WEIGHT_DECAY)
    ft_scheduler = CosineAnnealingLR(ft_optimizer, T_max=15, eta_min=1e-7)
    patience_counter = 0

    for epoch in range(15):
        train_loss, train_acc = train_distill_epoch(
            student, teacher, train_loader_real, kd_criterion, ft_optimizer, DEVICE)
        val_results = evaluate(student, val_loader, eval_criterion, DEVICE)
        val_acc = accuracy_score(val_results['labels'], val_results['predictions'])
        val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                          average='macro', zero_division=0)
        val_wf1 = f1_score(val_results['labels'], val_results['predictions'],
                           average='weighted', zero_division=0)

        _append_history(history, train_loss, train_acc, val_results['loss'],
                        val_acc, val_f1, val_wf1, ft_optimizer)
        ft_scheduler.step()

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            _save_student(student, output_dir, fold_id)
            patience_counter = 0
        else:
            patience_counter += 1

        logger.log(
            f"  S-Fold{fold_id+1} FT Ep{epoch+1:3d}/15 | "
            f"TrA:{train_acc*100:.1f}% VaA:{val_acc*100:.1f}% | "
            f"F1:{val_f1*100:.1f}%{' *BEST*' if is_best else ''}"
        )
        if patience_counter >= 8:
            break

    # --- Final evaluation on test set ---
    _load_student(student, output_dir, fold_id)
    test_results = evaluate(student, test_loader, eval_criterion, DEVICE)
    test_metrics = compute_comprehensive_metrics(
        test_results['labels'], test_results['predictions'],
        test_results['probabilities'], CLASS_NAMES)

    logger.log(f"\n  [TEST] Fold {fold_id+1} | "
               f"Acc: {test_metrics['accuracy']*100:.2f}% | "
               f"Macro F1: {test_metrics['macro_f1']*100:.2f}% | "
               f"Bal Acc: {test_metrics['balanced_accuracy']*100:.2f}% | "
               f"Sens: {test_metrics['macro_sensitivity']*100:.2f}% | "
               f"Spec: {test_metrics['macro_specificity']*100:.2f}%")
    for cls in CLASS_NAMES:
        m = test_metrics['per_class'][cls]
        logger.log(f"    {cls:12s} | Sens:{m['sensitivity']:.3f} "
                   f"Spec:{m['specificity']:.3f} Prec:{m['precision']:.3f} "
                   f"F1:{m['f1_score']:.3f}")

    logger.log_metrics({
        'student/test_accuracy': test_metrics['accuracy'],
        'student/test_macro_f1': test_metrics['macro_f1'],
        'student/test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'student/test_macro_sensitivity': test_metrics['macro_sensitivity'],
        'student/test_macro_specificity': test_metrics['macro_specificity'],
    }, prefix='student/test_fold_' + str(fold_id) + '/')

    return student, history, test_metrics


def _append_history(history: dict, train_loss: float, train_acc: float,
                    val_loss: float, val_acc: float, val_f1: float,
                    val_wf1: float, optimizer):
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['val_weighted_f1'].append(val_wf1)
    history['lr'].append(optimizer.param_groups[0]['lr'])


def _save_student(model, output_dir, fold_id):
    ckpt_dir = ensure_dir(output_dir / 'checkpoints')
    torch.save({'model_state': model.state_dict()},
               ckpt_dir / f'student_fold_{fold_id}_best.pt')


def _load_student(model, output_dir, fold_id):
    ckpt = torch.load(output_dir / 'checkpoints' / f'student_fold_{fold_id}_best.pt',
                      map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state'])


# ==============================================================================
# SAVE METRICS & FIGURES
# ==============================================================================
def save_fold_outputs(phase: str, fold_id: int, metrics: Dict, history: Dict,
                      output_dir: Path, logger: PipelineLogger):
    """Save metrics JSON, confusion matrix, ROC curves, training curves,
    and per-class metrics plot for a fold."""
    metrics_dir = ensure_dir(output_dir / 'metrics')
    figures_dir = ensure_dir(output_dir / 'figures')
    prefix = f"{phase}_fold_{fold_id}"

    # JSON metrics
    json_out = {
        'fold': fold_id,
        'phase': phase,
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'macro_f1': metrics['macro_f1'],
        'weighted_f1': metrics['weighted_f1'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'macro_sensitivity': metrics.get('macro_sensitivity'),
        'macro_specificity': metrics.get('macro_specificity'),
        'macro_auc': metrics.get('macro_auc'),
        'per_class': metrics['per_class'],
        'confusion_matrix': metrics['confusion_matrix'],
    }
    with open(metrics_dir / f'{prefix}_metrics.json', 'w') as f:
        json.dump(json_out, f, indent=2)

    # Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, CLASS_NAMES,
                          str(figures_dir / f'{prefix}_confusion_matrix.png'),
                          f'{phase.upper()} Fold {fold_id+1} — Confusion Matrix')

    # ROC Curves
    # We need probabilities from evaluate; reload if needed
    # (probabilities are not stored in compute_comprehensive_metrics output)
    # Skip if probabilities not available
    if 'probabilities' in metrics:
        plot_roc_curves(metrics['labels'], metrics['probabilities'],
                        CLASS_NAMES,
                        str(figures_dir / f'{prefix}_roc_curves.png'))

    # Training Curves
    if history and 'train_loss' in history and len(history['train_loss']) > 0:
        plot_training_curves(history,
                             str(figures_dir / f'{prefix}_training_curves.png'),
                             f'{phase.upper()} Fold {fold_id+1} — Training Curves')

    # Per-Class Metrics Bar Chart
    plot_per_class_metrics(metrics, CLASS_NAMES,
                           str(figures_dir / f'{prefix}_per_class_metrics.png'))

    # Log to W&B
    if logger.use_wandb:
        for fig_name in [f'{prefix}_confusion_matrix', f'{prefix}_per_class_metrics']:
            fig_path = figures_dir / f'{fig_name}.png'
            if fig_path.exists():
                wandb.log({fig_name: wandb.Image(str(fig_path))})

    return json_out


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    global BATCH_SIZE, N_FOLDS, TEACHER_EPOCHS, STUDENT_EPOCHS
    global KD_TEMPERATURE, KD_ALPHA

    parser = argparse.ArgumentParser(
        description='CNN-BiLSTM → Pure CNN KD Pipeline — ICBHI 2017 COPD Classification'
    )
    parser.add_argument('--data_dir', type=str,
                        default=str(ICBHI_2017_DIR),
                        help='ICBHI 2017 audio directory')
    parser.add_argument('--labels_file', type=str,
                        default=str(ICBHI_2017_LABELS),
                        help='Patient diagnosis labels file')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--artifact_root', type=str,
                        default=str(KD_PIPELINE_ARTIFACTS_DIR))
    parser.add_argument('--n_folds', type=int, default=N_FOLDS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--teacher_epochs', type=int, default=TEACHER_EPOCHS)
    parser.add_argument('--student_epochs', type=int, default=STUDENT_EPOCHS)
    parser.add_argument('--kd_temperature', type=float, default=KD_TEMPERATURE)
    parser.add_argument('--kd_alpha', type=float, default=KD_ALPHA)
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging (default: enabled unless --no_wandb is specified)')
    parser.add_argument('--wandb_project', type=str, default='copd-kd-pipeline')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--dry_run', action='store_true',
                        help='Quick test: 1 fold, 3 epochs each')
    args = parser.parse_args()

    # Override globals from CLI
    BATCH_SIZE = args.batch_size
    N_FOLDS = args.n_folds
    TEACHER_EPOCHS = args.teacher_epochs
    STUDENT_EPOCHS = args.student_epochs
    KD_TEMPERATURE = args.kd_temperature
    KD_ALPHA = args.kd_alpha

    if args.dry_run:
        TEACHER_EPOCHS = 3
        STUDENT_EPOCHS = 5
        N_FOLDS = 2
        args.no_wandb = True

    if args.output_dir is None:
        args.output_dir = str(args.artifact_root)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dir(output_dir / 'checkpoints')
    ensure_dir(output_dir / 'metrics')
    ensure_dir(output_dir / 'figures')

    # Logger
    use_wandb = not args.no_wandb and HAS_WANDB
    logger = PipelineLogger(
        str(output_dir / 'training_log.txt'),
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # ===================== Configuration =====================
    logger.section("CONFIGURATION")
    logger.log(f"Data dir: {args.data_dir}")
    logger.log(f"Labels: {args.labels_file}")
    logger.log(f"Output: {output_dir}")
    logger.log(f"Device: {DEVICE}")
    logger.log(f"Folds: {N_FOLDS}")
    logger.log(f"Teacher epochs: {TEACHER_EPOCHS}")
    logger.log(f"Student epochs: {STUDENT_EPOCHS}")
    logger.log(f"KD: T={KD_TEMPERATURE}, alpha={KD_ALPHA}")
    logger.log(f"Focal gamma: {FOCAL_GAMMA}, Label smoothing: {LABEL_SMOOTHING}")
    logger.log(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    logger.log(f"Input shape: (1, {INPUT_HEIGHT}, {INPUT_WIDTH})")
    logger.log(f"W&B: {'enabled' if use_wandb else 'disabled'}")

    # ===================== Data Loading =====================
    logger.section("DATA LOADING")
    samples, class_to_idx = build_icbhi_dataset(
        args.data_dir, args.labels_file, logger)
    logger.log(f"Class mapping: {class_to_idx}")

    if len(samples) == 0:
        logger.log("ERROR: No samples loaded!")
        sys.exit(1)

    # ===================== Patient-wise Folds =====================
    patient_ids = np.array([s['patient_id'] for s in samples])
    labels_arr = np.array([s['class_idx'] for s in samples])

    try:
        gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        folds = list(gkf.split(np.arange(len(samples)), labels_arr, patient_ids))
    except Exception:
        gkf = GroupKFold(n_splits=N_FOLDS)
        folds = list(gkf.split(np.arange(len(samples)), labels_arr, patient_ids))

    logger.log(f"Created {len(folds)} patient-wise folds")

    # ===================== Run Pipeline per Fold =====================
    all_results = []

    for fold_id, (train_val_idx, test_idx) in enumerate(folds):
        logger.section(f"FOLD {fold_id + 1}/{N_FOLDS}")

        # Split train/val within train_val (patient-wise)
        tv_patients = patient_ids[train_val_idx]
        tv_labels = labels_arr[train_val_idx]
        unique_pats = np.unique(tv_patients)
        pat_label_map = {p: tv_labels[np.where(tv_patients == p)[0][0]]
                         for p in unique_pats}
        plabels = np.array([pat_label_map[p] for p in unique_pats])

        try:
            train_pats, val_pats = train_test_split(
                unique_pats, test_size=0.15, random_state=42 + fold_id,
                stratify=plabels)
        except ValueError:
            train_pats, val_pats = train_test_split(
                unique_pats, test_size=0.15, random_state=42 + fold_id)

        train_idx = train_val_idx[np.isin(tv_patients, train_pats)]
        val_idx = train_val_idx[np.isin(tv_patients, val_pats)]

        train_samples_fold = [samples[i] for i in train_idx]
        val_samples_fold = [samples[i] for i in val_idx]
        test_samples_fold = [samples[i] for i in test_idx]

        logger.log(f"Train: {len(train_samples_fold)} | Val: {len(val_samples_fold)} "
                   f"| Test: {len(test_samples_fold)}")
        logger.log(f"Train classes: {dict(Counter([s['class_name'] for s in train_samples_fold]))}")
        logger.log(f"Test classes: {dict(Counter([s['class_name'] for s in test_samples_fold]))}")

        # --- Phase 1: Train Teacher ---
        teacher, t_history, t_val_metrics = train_teacher(
            fold_id, train_samples_fold, val_samples_fold, output_dir, logger)

        save_fold_outputs('teacher', fold_id, t_val_metrics, t_history,
                          output_dir, logger)

        # --- Evaluate Teacher on Test ---
        test_ds = ICBHIMelDataset(test_samples_fold, augment=False, spec_augment=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
        eval_criterion = nn.CrossEntropyLoss()
        t_test_results = evaluate(teacher, test_loader, eval_criterion, DEVICE)
        t_test_metrics = compute_comprehensive_metrics(
            t_test_results['labels'], t_test_results['predictions'],
            t_test_results['probabilities'], CLASS_NAMES)
        save_fold_outputs('teacher_test', fold_id, t_test_metrics, {},
                          output_dir, logger)

        logger.log(f"  [TEACHER TEST] Acc: {t_test_metrics['accuracy']*100:.2f}% | "
                   f"Macro F1: {t_test_metrics['macro_f1']*100:.2f}% | "
                   f"Bal Acc: {t_test_metrics['balanced_accuracy']*100:.2f}%")

        # --- Phase 2-4: Distill Student ---
        student, s_history, s_test_metrics = train_student_distillation(
            fold_id, teacher, train_samples_fold, val_samples_fold,
            test_samples_fold, output_dir, logger)

        save_fold_outputs('student', fold_id, s_test_metrics, s_history,
                          output_dir, logger)

        # --- KD Comparison Plot ---
        ensure_dir(output_dir / 'figures')
        plot_kd_comparison(t_test_metrics, s_test_metrics, CLASS_NAMES,
                           str(output_dir / 'figures' /
                               f'kd_comparison_fold_{fold_id}.png'))

        all_results.append({
            'fold': fold_id,
            'teacher': {'accuracy': t_test_metrics['accuracy'],
                        'macro_f1': t_test_metrics['macro_f1'],
                        'balanced_accuracy': t_test_metrics['balanced_accuracy'],
                        'macro_sensitivity': t_test_metrics['macro_sensitivity'],
                        'macro_specificity': t_test_metrics['macro_specificity']},
            'student': {'accuracy': s_test_metrics['accuracy'],
                       'macro_f1': s_test_metrics['macro_f1'],
                       'balanced_accuracy': s_test_metrics['balanced_accuracy'],
                       'macro_sensitivity': s_test_metrics['macro_sensitivity'],
                       'macro_specificity': s_test_metrics['macro_specificity'],
                       'num_params': student.count_parameters()},
        })

        # Cleanup
        del teacher, student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ===================== FINAL SUMMARY =====================
    logger.section("FINAL SUMMARY")

    # Print results table
    header = (f"{'Fold':<6} {'T-Acc':>8} {'T-F1':>8} {'T-Bal':>8} "
              f"{'S-Acc':>8} {'S-F1':>8} {'S-Bal':>8} {'S-Sens':>8} {'S-Spec':>8}")
    logger.log(header)
    logger.log("-" * 85)

    for r in all_results:
        logger.log(
            f"{r['fold']+1:<6} {r['teacher']['accuracy']*100:>7.2f}% "
            f"{r['teacher']['macro_f1']*100:>7.2f}% "
            f"{r['teacher']['balanced_accuracy']*100:>7.2f}% "
            f"{r['student']['accuracy']*100:>7.2f}% "
            f"{r['student']['macro_f1']*100:>7.2f}% "
            f"{r['student']['balanced_accuracy']*100:>7.2f}% "
            f"{r['student']['macro_sensitivity']*100:>7.2f}% "
            f"{r['student']['macro_specificity']*100:>7.2f}%"
        )

    # Averages
    avg = {}
    std = {}
    for key in ['teacher', 'student']:
        for m in ['accuracy', 'macro_f1', 'balanced_accuracy',
                  'macro_sensitivity', 'macro_specificity']:
            vals = [r[key][m] for r in all_results]
            avg[f'{key}_{m}'] = np.mean(vals)
            std[f'{key}_{m}'] = np.std(vals)

    logger.log("-" * 85)
    logger.log(f"{'AVG':<6} {avg['teacher_accuracy']*100:>7.2f}% "
               f"{avg['teacher_macro_f1']*100:>7.2f}% "
               f"{avg['teacher_balanced_accuracy']*100:>7.2f}% "
               f"{avg['student_accuracy']*100:>7.2f}% "
               f"{avg['student_macro_f1']*100:>7.2f}% "
               f"{avg['student_balanced_accuracy']*100:>7.2f}% "
               f"{avg['student_macro_sensitivity']*100:>7.2f}% "
               f"{avg['student_macro_specificity']*100:>7.2f}%")
    logger.log(f"{'STD':<6} ±{std['teacher_accuracy']*100:>6.2f}% "
               f"±{std['teacher_macro_f1']*100:>6.2f}% "
               f"±{std['teacher_balanced_accuracy']*100:>6.2f}% "
               f"±{std['student_accuracy']*100:>6.2f}% "
               f"±{std['student_macro_f1']*100:>6.2f}% "
               f"±{std['student_balanced_accuracy']*100:>6.2f}% "
               f"±{std['student_macro_sensitivity']*100:>6.2f}% "
               f"±{std['student_macro_specificity']*100:>6.2f}%")

    # Save summary JSON
    summary = {
        'config': {
            'data_dir': args.data_dir,
            'labels_file': args.labels_file,
            'teacher': 'CNN-BiLSTM',
            'student': 'CNN6 (FPGA-optimized)',
            'kd_temperature': KD_TEMPERATURE,
            'kd_alpha': KD_ALPHA,
            'focal_gamma': FOCAL_GAMMA,
            'n_folds': N_FOLDS,
            'student_params': all_results[0]['student']['num_params'],
        },
        'averages': {k: float(v) for k, v in avg.items()},
        'stds': {k: float(v) for k, v in std.items()},
        'fold_results': all_results,
    }

    with open(output_dir / 'metrics' / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Summary chart: Student accuracy & F1 by fold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fold_ids = [r['fold'] + 1 for r in all_results]

    axes[0].bar(np.array(fold_ids) - 0.15,
                [r['teacher']['macro_f1'] * 100 for r in all_results],
                0.3, label='Teacher', color='#e74c3c', alpha=0.8)
    axes[0].bar(np.array(fold_ids) + 0.15,
                [r['student']['macro_f1'] * 100 for r in all_results],
                0.3, label='Student', color='#3498db', alpha=0.8)
    axes[0].axhline(y=avg['student_macro_f1'] * 100, color='#3498db',
                    linestyle='--', alpha=0.7,
                    label=f"Student Avg {avg['student_macro_f1']*100:.1f}%")
    axes[0].set_xlabel('Fold'); axes[0].set_ylabel('Macro F1 (%)')
    axes[0].set_title('Teacher vs Student — Macro F1'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(np.array(fold_ids) - 0.15,
                [r['teacher']['balanced_accuracy'] * 100 for r in all_results],
                0.3, label='Teacher', color='#e74c3c', alpha=0.8)
    axes[1].bar(np.array(fold_ids) + 0.15,
                [r['student']['balanced_accuracy'] * 100 for r in all_results],
                0.3, label='Student', color='#3498db', alpha=0.8)
    axes[1].axhline(y=avg['student_balanced_accuracy'] * 100, color='#3498db',
                    linestyle='--', alpha=0.7,
                    label=f"Student Avg {avg['student_balanced_accuracy']*100:.1f}%")
    axes[1].set_xlabel('Fold'); axes[1].set_ylabel('Balanced Accuracy (%)')
    axes[1].set_title('Teacher vs Student — Balanced Accuracy'); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'KD Pipeline Results — CNN-BiLSTM → CNN6 ({N_FOLDS}-Fold CV)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'final_summary.png', dpi=150)
    plt.close()

    if logger.use_wandb:
        wandb.log({'final_summary': wandb.Image(str(output_dir / 'figures' / 'final_summary.png'))})

    # Export best student for FPGA
    logger.section("FPGA DEPLOYMENT PREPARATION")
    best_fold = int(np.argmax([r['student']['macro_f1'] for r in all_results]))
    logger.log(f"Best fold: {best_fold + 1}")

    # Load best student
    best_student = StudentCNN6().to('cpu')
    best_student.eval()
    ckpt = torch.load(output_dir / 'checkpoints' / f'student_fold_{best_fold}_best.pt',
                      map_location='cpu', weights_only=True)
    best_student.load_state_dict(ckpt['model_state'])

    # Save PyTorch weights
    torch.save(best_student.state_dict(), output_dir / 'student_cnn6_best.pt')
    logger.log(f"Saved: {output_dir / 'student_cnn6_best.pt'}")

    # Export ONNX
    try:
        dummy_input = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH)
        torch.onnx.export(best_student, dummy_input,
                          str(output_dir / 'student_cnn6_best.onnx'),
                          export_params=True, opset_version=11,
                          do_constant_folding=True,
                          input_names=['mel_spectrogram'],
                          output_names=['logits'],
                          dynamic_axes={'mel_spectrogram': {0: 'batch_size'},
                                        'logits': {0: 'batch_size'}})
        logger.log(f"ONNX exported: {output_dir / 'student_cnn6_best.onnx'}")
    except Exception as e:
        logger.log(f"ONNX export failed: {e}")

    # Fold BatchNorm for FPGA
    try:
        folded = best_student.fold_batch_norm()
        torch.save(folded.state_dict(), output_dir / 'student_cnn6_folded_bn.pt')
        logger.log(f"BatchNorm-folded model saved: {output_dir / 'student_cnn6_folded_bn.pt'}")
        logger.log(f"Folded model parameters: {folded.count_parameters():,}")
    except Exception as e:
        logger.log(f"BN folding failed: {e}")

    logger.finish()
    return avg, std


if __name__ == '__main__':
    main()
