#!/usr/bin/env python3
"""
================================================================================
NVIDIA pytorch-quantization QAT Pipeline v4
================================================================================
Student: MobileNetV2 (3-class: COPD, Healthy, Non-COPD)
Library: NVIDIA pytorch-quantization (QDQ nodes for TensorRT)

Pipeline:
  1. quant_modules.initialize() → auto-insert QDQ nodes
  2. Load FP32 Student weights
  3. Calibration (Max / Histogram) trên tập nhỏ
  4. QAT fine-tuning (2-5 epochs, LR=1e-5)
  5. Export ONNX với QDQ nodes
  6. Evaluation: Accuracy, Latency, Model Size, SQNR

Usage:
  # Full pipeline
  python quantize_distillation_04.py --mode full --use_wav

  # Calibration only
  python quantize_distillation_04.py --mode calibrate --use_wav

  # QAT only (after calibration)
  python quantize_distillation_04.py --mode qat --use_wav

  # Export ONNX
  python quantize_distillation_04.py --mode export

  # Evaluate FP32 vs INT8
  python quantize_distillation_04.py --mode evaluate --use_wav
================================================================================
"""

import os
import sys
import gc
import copy
import json
import time
import argparse
import logging
import warnings
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from scipy import signal
from scipy.ndimage import zoom

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    print("ERROR: scipy required"); sys.exit(1)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# --- NVIDIA pytorch-quantization ---
try:
    from pytorch_quantization import quant_modules
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib
    from pytorch_quantization.tensor_quant import QuantDescriptor
    HAS_NVIDIA_QUANT = True
except ImportError:
    HAS_NVIDIA_QUANT = False
    print("WARNING: pytorch-quantization not installed. "
          "Install: pip install pytorch-quantization --extra-index-url "
          "https://pypi.ngc.nvidia.com")

warnings.filterwarnings('ignore')

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIG — Consistent with distillation_02.py
# ==============================================================================
NUM_CLASSES = 3
CLASS_NAMES = ['COPD', 'Healthy', 'Non-COPD']
IMG_SIZE = 224

TARGET_SR = 4000
SEGMENT_DURATION = 8
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_DURATION  # 32000

N_MELS = 128
N_FFT = 512
HOP_LENGTH = 128
N_GAMMATONE = 64

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# QAT Hyperparams
QAT_EPOCHS = 5
QAT_LR = 1e-5
QAT_BATCH_SIZE = 8
NUM_CALIB_BATCHES = 32
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0

DISEASE_TO_CLASS = {
    'COPD': 'COPD', 'Healthy': 'Healthy',
    'Asthma': 'Non-COPD', 'URTI': 'Non-COPD', 'LRTI': 'Non-COPD',
    'Bronchiectasis': 'Non-COPD', 'Bronchiolitis': 'Non-COPD',
    'Pneumonia': 'Non-COPD',
}


# ==============================================================================
# AUDIO PREPROCESSING — Identical to distillation_02.py
# ==============================================================================
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def preprocess_audio(wav_path, target_sr=TARGET_SR, segment_len=SEGMENT_SAMPLES):
    """Load, resample 4kHz, BPF 25-2000Hz, normalize, pad/crop 8s."""
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
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, max(num_samples, 1))
        b, a = butter_bandpass(25, min(2000, target_sr // 2 - 1), target_sr, order=3)
        audio = signal.filtfilt(b, a, audio).astype(np.float32)
        max_val = np.max(np.abs(audio)) + 1e-10
        audio = audio / max_val
        if len(audio) < segment_len:
            repeats = segment_len // len(audio) + 1
            audio = np.tile(audio, repeats)[:segment_len]
        elif len(audio) > segment_len:
            start = (len(audio) - segment_len) // 2
            audio = audio[start:start + segment_len]
        return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to process {wav_path}: {e}")
        return np.zeros(segment_len, dtype=np.float32)


# ==============================================================================
# FEATURE EXTRACTION — Identical to distillation_02.py
# ==============================================================================
def compute_gammatone_filterbank(sr, n_filters=N_GAMMATONE, fmin=50, fmax=2000):
    ear_q = 9.26449
    min_bw = 24.7
    freqs = -(ear_q * min_bw) + np.exp(
        np.arange(1, n_filters + 1) * (
            -np.log(fmax + ear_q * min_bw) + np.log(fmin + ear_q * min_bw)
        ) / n_filters
    ) * (fmax + ear_q * min_bw)
    return np.flip(freqs)


def compute_gammatonegram(audio, sr=TARGET_SR, n_filters=N_GAMMATONE,
                          n_fft=N_FFT, hop_length=HOP_LENGTH):
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    cf = compute_gammatone_filterbank(sr, n_filters, fmin=50, fmax=min(2000, sr // 2 - 1))
    weights = np.zeros((n_filters, len(f)))
    for i, center_freq in enumerate(cf):
        erb = 24.7 * (4.37 * center_freq / 1000 + 1)
        weights[i] = np.exp(-0.5 * ((f - center_freq) / (erb * 0.5)) ** 2)
    power = np.abs(Zxx) ** 2
    gammatone_spec = np.dot(weights, power)
    return np.log10(gammatone_spec + 1e-10)


def compute_mel_spectrogram(audio, sr=TARGET_SR, n_mels=N_MELS,
                            n_fft=N_FFT, hop_length=HOP_LENGTH):
    if HAS_LIBROSA:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, fmin=50, fmax=min(2000, sr // 2))
        return librosa.power_to_db(mel_spec, ref=np.max)
    else:
        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        power = np.abs(Zxx) ** 2
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
    """3-channel hybrid: Gammatonegram + Mel-spectrogram + Average → 224x224."""
    gamma = compute_gammatonegram(audio, sr)
    mel = compute_mel_spectrogram(audio, sr)

    def normalize(x):
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return x

    gamma = normalize(gamma)
    mel = normalize(mel)
    gamma_r = np.clip(zoom(gamma, (output_size / gamma.shape[0],
                                    output_size / gamma.shape[1]), order=1)
                      [:output_size, :output_size], 0, 1)
    mel_r = np.clip(zoom(mel, (output_size / mel.shape[0],
                                output_size / mel.shape[1]), order=1)
                    [:output_size, :output_size], 0, 1)
    avg_channel = (gamma_r + mel_r) / 2.0
    return np.stack([gamma_r, mel_r, avg_channel], axis=0).astype(np.float32)


# ==============================================================================
# DATASET
# ==============================================================================
class WavDataset(Dataset):
    """On-the-fly WAV → Hybrid Spectrogram."""
    def __init__(self, samples, normalize=True):
        self.samples = samples
        self.normalize_fn = transforms.Normalize(
            mean=NORM_MEAN, std=NORM_STD) if normalize else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = preprocess_audio(s['wav_path'])
        spec = create_hybrid_spectrogram(audio)
        spec_tensor = torch.from_numpy(spec).float()
        if self.normalize_fn:
            spec_tensor = self.normalize_fn(spec_tensor)
        return spec_tensor, s['class_idx']


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_icbhi_samples(data_dir, labels_path):
    patient_labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                patient_labels[int(parts[0])] = parts[1].strip()

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    samples = []
    for wav_file in sorted(os.listdir(data_dir)):
        if not wav_file.endswith('.wav'):
            continue
        try:
            pid = int(wav_file.split('_')[0])
        except ValueError:
            continue
        if pid not in patient_labels:
            continue
        disease = patient_labels[pid]
        if disease not in DISEASE_TO_CLASS:
            continue
        class_name = DISEASE_TO_CLASS[disease]
        samples.append({
            'wav_path': os.path.join(data_dir, wav_file),
            'class_idx': class_to_idx[class_name],
            'class_name': class_name,
        })
    return samples


def load_combined_samples(combined_dir):
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    samples = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(combined_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for wav_file in sorted(os.listdir(class_dir)):
            if not wav_file.endswith('.wav'):
                continue
            samples.append({
                'wav_path': os.path.join(class_dir, wav_file),
                'class_idx': class_to_idx[class_name],
                'class_name': class_name,
            })
    return samples


def select_balanced_samples(all_samples, num_per_class=67, seed=42):
    np.random.seed(seed)
    selected = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        cls = [s for s in all_samples if s['class_idx'] == class_idx]
        np.random.shuffle(cls)
        n = min(num_per_class, len(cls))
        selected.extend(cls[:n])
        logger.info(f"    {class_name}: {n}/{len(cls)}")
    np.random.shuffle(selected)
    return selected


def build_dataloaders(args, batch_size=8):
    """Build train & calib dataloaders from WAV sources."""
    all_samples = []
    if os.path.isdir(args.icbhi_dir) and os.path.isfile(args.icbhi_labels):
        icbhi = load_icbhi_samples(args.icbhi_dir, args.icbhi_labels)
        logger.info(f"  ICBHI: {len(icbhi)} samples")
        all_samples.extend(icbhi)
    if os.path.isdir(args.combined_dir):
        combined = load_combined_samples(args.combined_dir)
        logger.info(f"  Combined: {len(combined)} samples")
        all_samples.extend(combined)
    if not all_samples:
        logger.error("No WAV samples found!")
        sys.exit(1)

    # Deduplicate
    seen, unique = set(), []
    for s in all_samples:
        bn = os.path.basename(s['wav_path'])
        if bn not in seen:
            seen.add(bn)
            unique.append(s)

    logger.info(f"  Total unique: {len(unique)}")

    # Split: 200 calib, rest for QAT
    num_per_class = args.num_calib // NUM_CLASSES
    calib_samples = select_balanced_samples(unique, num_per_class, seed=42)
    calib_paths = set(s['wav_path'] for s in calib_samples)
    train_samples = [s for s in unique if s['wav_path'] not in calib_paths]

    logger.info(f"  Calib: {len(calib_samples)}, Train: {len(train_samples)}")

    calib_dataset = WavDataset(calib_samples)
    train_dataset = WavDataset(train_samples)

    calib_loader = DataLoader(calib_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    return calib_loader, train_loader


# ==============================================================================
# MODEL — StudentModel (MobileNetV2) identical to distillation_02.py
# ==============================================================================
def create_student_model(num_classes=NUM_CLASSES, pretrained=False, dropout=0.5):
    """Create StudentModel using standard nn modules.

    NOTE: When quant_modules.initialize() has been called, nn.Conv2d/nn.Linear
    are auto-replaced with QuantConv2d/QuantLinear.
    """
    backbone = models.mobilenet_v2(
        weights='IMAGENET1K_V1' if pretrained else None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
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
    return backbone


class StudentModel(nn.Module):
    """Wrapper for MobileNetV2 student (used for FP32 baseline loading)."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None)
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


# ==============================================================================
# LOSS
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        nc = inputs.size(1)
        if self.label_smoothing > 0:
            smooth = torch.full_like(inputs, self.label_smoothing / (nc - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth = F.one_hot(targets, nc).float()
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        focal_w = (1.0 - probs) ** self.gamma
        loss = -focal_w * smooth * log_probs
        if self.alpha is not None:
            loss = loss * self.alpha.to(inputs.device).unsqueeze(0)
        loss = loss.sum(dim=1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ==============================================================================
# CALIBRATION — NVIDIA pytorch-quantization
# ==============================================================================
def collect_stats(model, data_loader, device, num_batches=32):
    """Enable calibrators and collect activation statistics."""
    logger.info(f"  Collecting stats over {num_batches} batches...")

    # Enable calibration mode on all TensorQuantizer modules
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            model(inputs)
            if (i + 1) % 10 == 0:
                logger.info(f"    Calib batch {i + 1}/{num_batches}")

    logger.info("  ✅ Stats collection complete")


def compute_amax(model, method='max', **kwargs):
    """Compute amax values from collected stats.

    Args:
        method: 'max', 'entropy', 'mse', or 'percentile'
    """
    logger.info(f"  Computing amax with method='{method}'...")

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    # HistogramCalibrator supports multiple methods
                    module.load_calib_amax(method, **kwargs)
            module.enable_quant()
            module.disable_calib()

    logger.info("  ✅ Amax computation complete")


def calibrate_model(model, data_loader, device, method='max', num_batches=32):
    """Full calibration pipeline: collect stats → compute amax."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  CALIBRATION — method={method}, batches={num_batches}")
    logger.info(f"{'='*60}")

    collect_stats(model, data_loader, device, num_batches)

    if method == 'max':
        compute_amax(model, method='max')
    elif method in ('entropy', 'mse', 'percentile'):
        compute_amax(model, method=method)
    else:
        # Default: try both max and entropy, pick best
        compute_amax(model, method='max')

    logger.info("  ✅ Calibration complete\n")


# ==============================================================================
# EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate_model(model, dataloader, device, tag=''):
    """Evaluate model and return metrics dict."""
    model.eval()
    all_preds, all_labels = [], []
    total_time = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        start = time.time()
        outputs = model(inputs)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        total_time += time.time() - start
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    valid = all_labels >= 0
    if valid.sum() == 0:
        return {}
    all_preds, all_labels = all_preds[valid], all_labels[valid]
    n = len(all_preds)

    try:
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, classification_report, confusion_matrix)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_val = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        report = classification_report(all_labels, all_preds,
                                       target_names=CLASS_NAMES, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
    except ImportError:
        accuracy = np.mean(all_preds == all_labels)
        f1_macro = f1_weighted = precision = recall_val = 0.0
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        report = "sklearn not available"

    avg_latency_ms = (total_time / n) * 1000

    print(f"\n{'='*60}")
    print(f"  {tag} EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 (macro):      {f1_macro:.4f}")
    print(f"  F1 (weighted):   {f1_weighted:.4f}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall_val:.4f}")
    print(f"  Avg Latency:     {avg_latency_ms:.2f} ms/sample")
    print(f"  Total Inference: {total_time:.3f}s ({n} samples)")
    print(f"\n{report}")
    print(f"{'='*60}\n")

    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision': float(precision),
        'recall': float(recall_val),
        'avg_latency_ms': float(avg_latency_ms),
        'total_inference_s': float(total_time),
        'num_samples': n,
        'confusion_matrix': cm.tolist(),
    }


def compute_model_size_mb(model):
    """Compute model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def compute_sqnr(fp32_outputs, int8_outputs):
    """Compute Signal-to-Quantization-Noise Ratio (dB).

    SQNR = 10 * log10(signal_power / noise_power)
    where noise = fp32_output - int8_output
    """
    signal_power = torch.mean(fp32_outputs ** 2).item()
    noise = fp32_outputs - int8_outputs
    noise_power = torch.mean(noise ** 2).item()
    if noise_power < 1e-12:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


@torch.no_grad()
def collect_outputs(model, dataloader, device, max_batches=50):
    """Collect model outputs for SQNR computation."""
    model.eval()
    all_outputs = []
    for i, (inputs, _) in enumerate(dataloader):
        if i >= max_batches:
            break
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_outputs.append(outputs.cpu())
    return torch.cat(all_outputs, dim=0)


def measure_inference_latency(model, device, input_shape=(1, 3, 224, 224),
                               num_warmup=10, num_runs=100):
    """Measure per-sample inference latency."""
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)

    times = np.array(times)
    return {
        'mean_ms': float(times.mean() * 1000),
        'std_ms': float(times.std() * 1000),
        'median_ms': float(np.median(times) * 1000),
        'p95_ms': float(np.percentile(times, 95) * 1000),
        'p99_ms': float(np.percentile(times, 99) * 1000),
    }


# ==============================================================================
# QAT FINE-TUNING
# ==============================================================================
def qat_finetune(model, train_loader, calib_loader, device, args):
    """QAT fine-tuning loop with low learning rate."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  QAT FINE-TUNING — {args.qat_epochs} epochs, LR={args.qat_lr}")
    logger.info(f"{'='*60}")

    criterion = FocalLoss(
        alpha=torch.FloatTensor([1.0, 1.0, 1.0]),
        gamma=2.0, label_smoothing=0.1
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.qat_lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.qat_epochs, eta_min=1e-7
    )

    best_f1 = -1.0
    best_state = None
    history = {'epoch': [], 'train_loss': [], 'train_acc': [],
               'val_acc': [], 'val_f1': []}

    for epoch in range(args.qat_epochs):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # --- Validate ---
        val_metrics = evaluate_model(model, calib_loader, device,
                                     tag=f'QAT Epoch {epoch + 1}')
        val_f1 = val_metrics.get('f1_macro', 0)
        val_acc = val_metrics.get('accuracy', 0)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

        logger.info(
            f"  Ep {epoch+1:2d}/{args.qat_epochs} | "
            f"TrL: {train_loss:.4f} TrA: {train_acc*100:.1f}% | "
            f"VaA: {val_acc*100:.1f}% F1: {val_f1*100:.1f}%"
            f"{'  *BEST*' if is_best else ''}"
        )

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"  ✅ Restored best QAT model (F1: {best_f1*100:.2f}%)")

    return model, history


# ==============================================================================
# ONNX EXPORT WITH QDQ NODES
# ==============================================================================
def export_onnx(model, output_path, device):
    """Export quantized model to ONNX with QDQ nodes."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  ONNX EXPORT — {output_path}")
    logger.info(f"{'='*60}")

    model.eval()

    # CRITICAL: Enable fake-quantization mode for ONNX compatibility
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        do_constant_folding=True,
        verbose=False,
    )

    # Reset
    quant_nn.TensorQuantizer.use_fb_fake_quant = False

    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  ✅ ONNX exported: {output_path} ({onnx_size:.2f} MB)")
    return onnx_size


# ==============================================================================
# COMPREHENSIVE COMPARISON: FP32 vs Fake-INT8
# ==============================================================================
def run_comparison(fp32_model, quant_model, calib_loader, device, output_dir):
    """Compare FP32 vs Fake-INT8 on all metrics."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  📊 COMPREHENSIVE FP32 vs FAKE-INT8 COMPARISON")
    logger.info(f"{'='*70}")

    # 1. Accuracy metrics
    logger.info("\n  [1/4] Accuracy metrics...")
    fp32_metrics = evaluate_model(fp32_model, calib_loader, device, tag='FP32')
    int8_metrics = evaluate_model(quant_model, calib_loader, device, tag='Fake-INT8')

    # 2. Model size
    logger.info("\n  [2/4] Model size...")
    fp32_size = compute_model_size_mb(fp32_model)
    int8_size = compute_model_size_mb(quant_model)

    # 3. Latency
    logger.info("\n  [3/4] Inference latency...")
    fp32_latency = measure_inference_latency(fp32_model, device)
    int8_latency = measure_inference_latency(quant_model, device)

    # 4. SQNR
    logger.info("\n  [4/4] SQNR computation...")
    fp32_outputs = collect_outputs(fp32_model, calib_loader, device)
    int8_outputs = collect_outputs(quant_model, calib_loader, device)
    sqnr_db = compute_sqnr(fp32_outputs, int8_outputs)

    # --- Print comparison table ---
    print(f"\n{'='*70}")
    print(f"  📊 FP32 vs FAKE-INT8 COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Metric':<25s} {'FP32':>12s} {'Fake-INT8':>12s} {'Diff':>10s} {'Status':>8s}")
    print(f"  {'-'*67}")

    for key in ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']:
        fp = fp32_metrics.get(key, 0)
        i8 = int8_metrics.get(key, 0)
        d = i8 - fp
        sign = '+' if d >= 0 else ''
        st = '✅' if abs(d) < 0.02 else ('⚠️' if abs(d) < 0.05 else '❌')
        print(f"  {key:<25s} {fp:>12.4f} {i8:>12.4f} {sign}{d:>9.4f} {st:>8s}")

    print(f"  {'-'*67}")
    print(f"  {'Model Size (MB)':<25s} {fp32_size:>12.2f} {int8_size:>12.2f} "
          f"{int8_size-fp32_size:>+9.2f}{'':>8s}")
    print(f"  {'Latency mean (ms)':<25s} {fp32_latency['mean_ms']:>12.2f} "
          f"{int8_latency['mean_ms']:>12.2f} "
          f"{int8_latency['mean_ms']-fp32_latency['mean_ms']:>+9.2f}{'':>8s}")
    print(f"  {'Latency p95 (ms)':<25s} {fp32_latency['p95_ms']:>12.2f} "
          f"{int8_latency['p95_ms']:>12.2f} "
          f"{int8_latency['p95_ms']-fp32_latency['p95_ms']:>+9.2f}{'':>8s}")
    print(f"  {'SQNR (dB)':<25s} {'N/A':>12s} {sqnr_db:>12.2f}{'':>10s}{'':>8s}")
    print(f"{'='*70}\n")

    # Save results
    results = {
        'fp32': {
            'metrics': fp32_metrics,
            'model_size_mb': fp32_size,
            'latency': fp32_latency,
        },
        'fake_int8': {
            'metrics': int8_metrics,
            'model_size_mb': int8_size,
            'latency': int8_latency,
        },
        'sqnr_db': sqnr_db,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, 'qat_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  📁 Results saved → {results_path}")

    return results


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_pipeline(args):
    """Main QAT pipeline using NVIDIA pytorch-quantization."""

    if not HAS_NVIDIA_QUANT:
        logger.error("❌ pytorch-quantization not installed!")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ==================================================================
    # STEP 0: Load FP32 baseline model (for comparison later)
    # ==================================================================
    logger.info("\n[0/6] Loading FP32 baseline model...")
    fp32_model = StudentModel(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load(args.student_ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    fp32_model.load_state_dict(state_dict)
    fp32_model.to(device)
    fp32_model.eval()
    logger.info(f"  ✅ FP32 model loaded — F1: {ckpt.get('f1', 'N/A')}")
    logger.info(f"  Params: {sum(p.numel() for p in fp32_model.parameters()):,}")

    # ==================================================================
    # STEP 1: Initialize QDQ nodes (MUST be before model creation)
    # ==================================================================
    logger.info("\n[1/6] Initializing QDQ nodes via quant_modules.initialize()...")

    # Configure calibrator: 'histogram' hoặc 'max'
    if args.calib_method == 'histogram':
        quant_desc_input = QuantDescriptor(
            calib_method='histogram',
            num_bits=8
        )
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        logger.info("  Using Histogram calibrator")
    else:
        quant_desc_input = QuantDescriptor(
            calib_method='max',
            num_bits=8
        )
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        logger.info("  Using Max calibrator")

    # Initialize: replace nn.Conv2d → QuantConv2d, nn.Linear → QuantLinear
    quant_modules.initialize()
    logger.info("  ✅ QDQ nodes inserted")

    # ==================================================================
    # STEP 2: Create quantized model and load FP32 weights
    # ==================================================================
    logger.info("\n[2/6] Creating quantized model and loading FP32 weights...")

    # Create model AFTER initialize() → auto QDQ insertion
    quant_model = StudentModel(num_classes=NUM_CLASSES, pretrained=False)

    # Load FP32 weights (strict=False to handle quantizer parameter mismatches)
    missing, unexpected = quant_model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info(f"  Missing keys (expected for quantizers): {len(missing)}")
    if unexpected:
        logger.info(f"  Unexpected keys: {len(unexpected)}")
    quant_model.to(device)
    logger.info(f"  ✅ Quantized model created with FP32 weights")

    # Deactivate quant_modules to not affect other model creations
    quant_modules.deactivate()

    # ==================================================================
    # STEP 3: Build dataloaders
    # ==================================================================
    logger.info("\n[3/6] Building dataloaders...")
    calib_loader, train_loader = build_dataloaders(args, batch_size=args.batch_size)

    # ==================================================================
    # STEP 4: Calibration
    # ==================================================================
    if args.mode in ('calibrate', 'full'):
        logger.info("\n[4/6] Calibration phase...")
        calibrate_model(
            quant_model, calib_loader, device,
            method=args.calib_method,
            num_batches=args.num_calib_batches
        )

        # Save calibrated state
        calib_ckpt_path = os.path.join(args.output_dir, 'student_calibrated.pt')
        torch.save({
            'model_state': quant_model.state_dict(),
            'calib_method': args.calib_method,
        }, calib_ckpt_path)
        logger.info(f"  💾 Calibrated model saved → {calib_ckpt_path}")

    # ==================================================================
    # STEP 5: QAT Fine-tuning
    # ==================================================================
    if args.mode in ('qat', 'full'):
        logger.info("\n[5/6] QAT fine-tuning phase...")

        # Load calibrated state if available and not coming from calibrate step
        if args.mode == 'qat':
            calib_ckpt = os.path.join(args.output_dir, 'student_calibrated.pt')
            if os.path.isfile(calib_ckpt):
                logger.info(f"  Loading calibrated weights: {calib_ckpt}")
                ckpt_calib = torch.load(calib_ckpt, map_location=device,
                                        weights_only=False)
                quant_model.load_state_dict(ckpt_calib['model_state'])

        quant_model, history = qat_finetune(
            quant_model, train_loader, calib_loader, device, args
        )

        # Save QAT model
        qat_ckpt_path = os.path.join(args.output_dir, 'student_qat_best.pt')
        torch.save({
            'model_state': quant_model.state_dict(),
            'history': history,
            'qat_epochs': args.qat_epochs,
            'qat_lr': args.qat_lr,
        }, qat_ckpt_path)
        logger.info(f"  💾 QAT model saved → {qat_ckpt_path}")

        # Save history
        history_path = os.path.join(args.output_dir, 'qat_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    # ==================================================================
    # STEP 6: Export ONNX + Evaluation
    # ==================================================================
    if args.mode in ('export', 'full'):
        logger.info("\n[6/6] Export & Evaluation phase...")

        # Load QAT model if not in memory
        if args.mode == 'export':
            qat_ckpt = os.path.join(args.output_dir, 'student_qat_best.pt')
            if os.path.isfile(qat_ckpt):
                ckpt_qat = torch.load(qat_ckpt, map_location=device,
                                      weights_only=False)
                quant_model.load_state_dict(ckpt_qat['model_state'])

        # Export ONNX
        onnx_path = os.path.join(args.output_dir, 'student_mobilenetv2_int8_qdq.onnx')
        export_onnx(quant_model, onnx_path, device)

    # ==================================================================
    # EVALUATION: FP32 vs Fake-INT8
    # ==================================================================
    if args.mode in ('evaluate', 'full'):
        logger.info("\n  Running comprehensive evaluation...")
        results = run_comparison(fp32_model, quant_model, calib_loader, device,
                                 args.output_dir)

    # Cleanup
    del fp32_model, quant_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  ✅ NVIDIA QAT PIPELINE COMPLETE!")
    print(f"  Results → {args.output_dir}")
    print(f"{'='*70}\n")


# ==============================================================================
# CLI
# ==============================================================================
def get_project_root():
    if os.path.isdir('/workspace/Parallel_Computing_on_FPGA'):
        return '/workspace/Parallel_Computing_on_FPGA'
    return '/home/iec/Parallel_Computing_on_FPGA'


def parse_args():
    root = get_project_root()
    parser = argparse.ArgumentParser(
        description='NVIDIA pytorch-quantization QAT for MobileNetV2 (3-class)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (calibrate → QAT → export → evaluate)
  python quantize_distillation_04.py --mode full --use_wav

  # Calibrate only
  python quantize_distillation_04.py --mode calibrate --use_wav --calib_method histogram

  # QAT only (requires calibrated model)
  python quantize_distillation_04.py --mode qat --use_wav --qat_epochs 5

  # Export ONNX only
  python quantize_distillation_04.py --mode export

  # Evaluate FP32 vs INT8
  python quantize_distillation_04.py --mode evaluate --use_wav
        """)

    parser.add_argument('--mode', type=str, required=True,
                        choices=['calibrate', 'qat', 'export', 'evaluate', 'full'],
                        help='Pipeline mode')
    parser.add_argument('--student_ckpt', type=str,
                        default=os.path.join(root,
                            'python/output_distillation_v2/checkpoints/student_fold_0_best.pt'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(root, 'quantize_nvidia_qat_result'))

    # QAT
    qat = parser.add_argument_group('QAT')
    qat.add_argument('--qat_epochs', type=int, default=QAT_EPOCHS)
    qat.add_argument('--qat_lr', type=float, default=QAT_LR)
    qat.add_argument('--batch_size', type=int, default=QAT_BATCH_SIZE)

    # Calibration
    cal = parser.add_argument_group('Calibration')
    cal.add_argument('--calib_method', type=str, default='max',
                     choices=['max', 'histogram', 'entropy', 'mse', 'percentile'])
    cal.add_argument('--num_calib', type=int, default=200)
    cal.add_argument('--num_calib_batches', type=int, default=NUM_CALIB_BATCHES)

    # Data
    data = parser.add_argument_group('Data')
    data.add_argument('--use_wav', action='store_true', default=False)
    data.add_argument('--icbhi_dir', type=str,
                      default=os.path.join(root, 'data/samples/ICBHI_final_database'))
    data.add_argument('--icbhi_labels', type=str,
                      default=os.path.join(root, 'data/samples/labels.txt'))
    data.add_argument('--combined_dir', type=str,
                      default=os.path.join(root, 'data/combined/audio'))

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.student_ckpt):
        logger.error(f"❌ Student checkpoint not found: {args.student_ckpt}")
        sys.exit(1)

    if args.use_wav:
        has_data = os.path.isdir(args.icbhi_dir) or os.path.isdir(args.combined_dir)
        if not has_data:
            logger.error("❌ No WAV data found!")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  NVIDIA pytorch-quantization QAT Pipeline v4")
    logger.info("  MobileNetV2 → INT8 (QDQ for TensorRT)")
    logger.info("=" * 70)
    logger.info(f"  Mode:         {args.mode}")
    logger.info(f"  Student:      {args.student_ckpt}")
    logger.info(f"  Output:       {args.output_dir}")
    logger.info(f"  Calib method: {args.calib_method}")
    logger.info(f"  QAT epochs:   {args.qat_epochs}")
    logger.info(f"  QAT LR:       {args.qat_lr}")
    logger.info(f"  Batch size:   {args.batch_size}")
    logger.info("=" * 70)

    run_pipeline(args)


if __name__ == '__main__':
    main()
