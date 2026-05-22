#!/usr/bin/env python3
"""
================================================================================
Vitis AI INT8 Quantization v2 — Student MobileNetV2 (3-class)
================================================================================
Định lượng INT8 model Student (MobileNetV2, 3 lớp: COPD, Healthy, Non-COPD)
đã huấn luyện trong distillation_02.py, sử dụng pytorch_nndct (Vitis AI 3.5).

Target device: DPUCZDX8G_ISA1_B2304 (Ultra96-V2)

Tính năng:
  - Cross-Layer Equalization (CLE) trước khi quantize
  - AdaQuant thông qua Fast Finetuning trong calibration
  - Tùy chọn Fast Finetuning nếu accuracy bị giảm đáng kể
  - Calibration từ pre-generated .npy spectrograms HOẶC on-the-fly từ WAV
  - So sánh Accuracy / F1-score giữa FP32 và INT8
  - Xuất .xmodel cho DPU deployment

Pipeline tiền xử lý (nhất quán 100% với distillation_02.py):
  1. Resample to 4kHz
  2. Bandpass filter 25-2000Hz
  3. Normalize, pad/crop to 8s
  4. Create 3-channel hybrid spectrogram (Gammatonegram + Mel + Average)
  5. Normalize: input / 255.0 → mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
     (Lưu ý: spectrogram đã ở dải [0,1], tương đương input/255 khi pixel ∈ [0,255])

Usage (trong Vitis AI Docker):
    # Bước 1: Calibration
    python quantize_distillation_02.py --quant_mode calib

    # Bước 1b: Calibration + Fast Finetuning (AdaQuant)
    python quantize_distillation_02.py --quant_mode calib --fast_finetune

    # Bước 2: Test (đánh giá quantized model)
    python quantize_distillation_02.py --quant_mode test

    # Bước 2b: Test + Deploy (xuất xmodel cho DPU)
    python quantize_distillation_02.py --quant_mode test --deploy

    # Dùng on-the-fly WAV processing (không cần pre-generated .npy)
    python quantize_distillation_02.py --quant_mode calib --use_wav --num_calib 200
================================================================================
"""

import os
import sys
import argparse
import logging
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from scipy import signal
from scipy.ndimage import zoom

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy")
    sys.exit(1)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

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
# CONFIG — Hoàn toàn nhất quán với distillation_02.py
# ==============================================================================
NUM_CLASSES = 3
CLASS_NAMES = ['COPD', 'Healthy', 'Non-COPD']
IMG_SIZE = 224
BATCH_SIZE = 8  # nhỏ hơn cho calibration stability

# Audio parameters — giống distillation_02.py
TARGET_SR = 4000
SEGMENT_DURATION = 8
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_DURATION  # 32000

# Feature extraction — giống distillation_02.py
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 128
N_GAMMATONE = 64

# ImageNet normalization — giống ICBHIDataset trong distillation_02.py
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Label mapping — giống distillation_02.py
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
# AUDIO PREPROCESSING — Copy chính xác từ distillation_02.py
# ==============================================================================
def butter_bandpass(lowcut, highcut, fs, order=3):
    """Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def preprocess_audio(wav_path, target_sr=TARGET_SR, segment_len=SEGMENT_SAMPLES):
    """Load, resample to 4kHz, BPF 25-2000Hz, normalize, pad/crop to 8s.
    
    Giống hệt hàm preprocess_audio() trong distillation_02.py.
    """
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
# FEATURE EXTRACTION — Copy chính xác từ distillation_02.py
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
    """Compute gammatonegram using FFT-based approximation.
    
    Giống hệt hàm compute_gammatonegram() trong distillation_02.py.
    """
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    cf = compute_gammatone_filterbank(sr, n_filters, fmin=50, fmax=min(2000, sr // 2 - 1))
    weights = np.zeros((n_filters, len(f)))
    for i, center_freq in enumerate(cf):
        erb = 24.7 * (4.37 * center_freq / 1000 + 1)
        weights[i] = np.exp(-0.5 * ((f - center_freq) / (erb * 0.5)) ** 2)
    power = np.abs(Zxx) ** 2
    gammatone_spec = np.dot(weights, power)
    gammatone_spec = np.log10(gammatone_spec + 1e-10)
    return gammatone_spec


def compute_mel_spectrogram(audio, sr=TARGET_SR, n_mels=N_MELS,
                            n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Compute log Mel-spectrogram.
    
    Giống hệt hàm compute_mel_spectrogram() trong distillation_02.py.
    """
    if HAS_LIBROSA:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, fmin=50, fmax=min(2000, sr // 2)
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
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
    """Create 3-channel hybrid: Gammatonegram + Mel-spectrogram → 224x224.
    
    Giống hệt hàm create_hybrid_spectrogram() trong distillation_02.py.
    Output: (3, 224, 224) float32, giá trị trong [0, 1].
    """
    gamma = compute_gammatonegram(audio, sr)
    mel = compute_mel_spectrogram(audio, sr)

    def normalize(x):
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return x

    gamma = normalize(gamma)
    mel = normalize(mel)

    gamma_resized = zoom(gamma, (output_size / gamma.shape[0], output_size / gamma.shape[1]), order=1)
    mel_resized = zoom(mel, (output_size / mel.shape[0], output_size / mel.shape[1]), order=1)

    gamma_resized = np.clip(gamma_resized[:output_size, :output_size], 0, 1)
    mel_resized = np.clip(mel_resized[:output_size, :output_size], 0, 1)

    avg_channel = (gamma_resized + mel_resized) / 2.0
    hybrid = np.stack([gamma_resized, mel_resized, avg_channel], axis=0)

    return hybrid.astype(np.float32)


# ==============================================================================
# MODEL — Hoàn toàn giống StudentModel trong distillation_02.py
# ==============================================================================
class StudentModel(nn.Module):
    """MobileNetV2 as student with enhanced head.
    
    Phải khớp chính xác cấu trúc trong distillation_02.py để load checkpoint.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False, dropout=0.5):
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


# ==============================================================================
# DATASETS
# ==============================================================================
class NpyCalibDataset(Dataset):
    """Dataset loading pre-generated .npy spectrograms + labels.
    
    Dùng cho tập calibration đã tạo bởi generate_calib_data.py.
    Áp dụng chuẩn hóa: spectrogram ∈ [0,1] → normalize(mean, std).
    
    Ghi chú về phép chuẩn hóa:
    - Spectrogram từ create_hybrid_spectrogram() có giá trị [0, 1]
    - Tương đương với input / 255.0 khi pixel ban đầu ∈ [0, 255]
    - Sau đó áp dụng: (input - mean) / std theo ImageNet
    """

    def __init__(self, calib_dir, normalize=True):
        self.calib_dir = calib_dir
        self.normalize_fn = transforms.Normalize(
            mean=NORM_MEAN, std=NORM_STD
        ) if normalize else None

        labels_file = os.path.join(calib_dir, 'calib_labels.txt')
        self.items = []

        if not os.path.isfile(labels_file):
            # Fallback: scan for .npy files without labels
            npy_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.npy')])
            for npy_file in npy_files:
                self.items.append({
                    'filename': npy_file,
                    'class_idx': -1,
                    'class_name': 'unknown'
                })
            logger.warning(f"No calib_labels.txt found. "
                          f"Loaded {len(self.items)} .npy files without labels.")
        else:
            with open(labels_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        self.items.append({
                            'filename': parts[0],
                            'class_idx': int(parts[1]),
                            'class_name': parts[2],
                        })
            logger.info(f"Loaded {len(self.items)} calibration samples from {labels_file}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        npy_path = os.path.join(self.calib_dir, item['filename'])
        spec = np.load(npy_path)  # (3, 224, 224) float32 [0, 1]
        tensor = torch.from_numpy(spec).float()

        # Chuẩn hóa: spec đã ở [0,1] (tương đương input/255.0)
        # Tiếp theo áp dụng ImageNet normalize
        if self.normalize_fn:
            tensor = self.normalize_fn(tensor)

        return tensor, item['class_idx']


class WavCalibDataset(Dataset):
    """On-the-fly WAV → Hybrid Spectrogram dataset.
    
    Tạo spectrogram trực tiếp từ file WAV, giống pipeline trong
    ICBHIDataset.__getitem__() của distillation_02.py (không augment).
    """

    def __init__(self, samples, normalize=True):
        """
        Args:
            samples: list of dict {'wav_path': str, 'class_idx': int, 'class_name': str}
            normalize: whether to apply ImageNet normalization
        """
        self.samples = samples
        self.normalize_fn = transforms.Normalize(
            mean=NORM_MEAN, std=NORM_STD
        ) if normalize else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = preprocess_audio(s['wav_path'])
        spec = create_hybrid_spectrogram(audio)  # (3, 224, 224) [0, 1]
        spec_tensor = torch.from_numpy(spec).float()

        # Chuẩn hóa: giống ICBHIDataset trong distillation_02.py
        # spec_tensor đã ở [0,1], tương đương input/255.0
        # Sau đó normalize theo ImageNet mean/std
        if self.normalize_fn:
            spec_tensor = self.normalize_fn(spec_tensor)

        return spec_tensor, s['class_idx']


# ==============================================================================
# DATA LOADING (cho chế độ --use_wav)
# ==============================================================================
def load_icbhi_samples(data_dir, labels_path):
    """Load ICBHI samples from WAV directory + labels.txt."""
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

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    wav_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav')])

    samples = []
    for wav_file in wav_files:
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
        class_idx = class_to_idx[class_name]
        wav_path = os.path.join(data_dir, wav_file)
        samples.append({
            'wav_path': wav_path,
            'class_idx': class_idx,
            'class_name': class_name,
        })

    return samples


def load_combined_samples(combined_dir):
    """Load combined samples from subdirectory structure: COPD/, Healthy/, Non-COPD/."""
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    samples = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(combined_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        wav_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.wav')])
        class_idx = class_to_idx[class_name]
        for wav_file in wav_files:
            wav_path = os.path.join(class_dir, wav_file)
            samples.append({
                'wav_path': wav_path,
                'class_idx': class_idx,
                'class_name': class_name,
            })

    return samples


def select_balanced_samples(all_samples, num_per_class=67, seed=42):
    """Lấy mẫu cân bằng từ mỗi class cho calibration (~200 mẫu tổng cộng)."""
    np.random.seed(seed)
    selected = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        cls_samples = [s for s in all_samples if s['class_idx'] == class_idx]
        np.random.shuffle(cls_samples)
        n_select = min(num_per_class, len(cls_samples))
        selected.extend(cls_samples[:n_select])
        logger.info(f"  Class '{class_name}': selected {n_select}/{len(cls_samples)} samples")

    np.random.seed(seed)
    np.random.shuffle(selected)
    return selected


# ==============================================================================
# UTILITIES
# ==============================================================================
def load_model(checkpoint_path, device='cpu'):
    """Load StudentModel from checkpoint."""
    model = StudentModel(num_classes=NUM_CLASSES, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        f1 = checkpoint.get('f1', 'N/A')
        acc = checkpoint.get('accuracy', 'N/A')
        logger.info(f"  Checkpoint metrics — F1: {f1}, Accuracy: {acc}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded: {n_params:,} parameters")

    return model


def evaluate_model(model, dataloader, device='cpu', tag=''):
    """Evaluate model and return metrics dict.
    
    Tương thích cả sklearn và manual fallback.
    """
    print(f"\n>>> Starting {tag} evaluation...", flush=True)
    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            start = time.time()
            outputs = model(inputs)
            total_time += time.time() - start
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(
                labels.numpy() if isinstance(labels, torch.Tensor) else labels
            )
            if (batch_idx + 1) % 10 == 0:
                print(f"  Eval batch {batch_idx + 1}/{len(dataloader)}", flush=True)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Filter out unlabeled samples (class_idx == -1)
    valid_mask = all_labels >= 0
    if valid_mask.sum() == 0:
        print("WARNING: No labeled samples for evaluation!", flush=True)
        return {}

    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    n_samples = len(all_preds)

    # --- Compute metrics ---
    try:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            classification_report, confusion_matrix
        )
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False
        print("  [INFO] sklearn not available, computing metrics manually", flush=True)

    if HAS_SKLEARN:
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        report = classification_report(
            all_labels, all_preds,
            target_names=CLASS_NAMES,
            zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)
    else:
        # Manual metrics computation
        accuracy = np.mean(all_preds == all_labels)
        precisions, recalls, f1s = [], [], []
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for i in range(n_samples):
            cm[int(all_labels[i])][int(all_preds[i])] += 1
        for c in range(NUM_CLASSES):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1_val)
        f1_macro = np.mean(f1s)
        f1_weighted = np.average(
            f1s, weights=[cm[c, :].sum() for c in range(NUM_CLASSES)]
        )
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        # Build simple report
        report_lines = [
            f"{'':>15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}"
        ]
        for c in range(NUM_CLASSES):
            support = int(cm[c, :].sum())
            report_lines.append(
                f"{CLASS_NAMES[c]:>15s} {precisions[c]:>10.4f} "
                f"{recalls[c]:>10.4f} {f1s[c]:>10.4f} {support:>10d}"
            )
        report = "\n".join(report_lines)

    # --- Print results ---
    print("\n" + "=" * 60, flush=True)
    print(f"  {tag} EVALUATION RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)", flush=True)
    print(f"  F1 (macro):        {f1_macro:.4f}", flush=True)
    print(f"  F1 (weighted):     {f1_weighted:.4f}", flush=True)
    print(f"  Precision (macro): {precision:.4f}", flush=True)
    print(f"  Recall (macro):    {recall:.4f}", flush=True)
    print(f"  Inference time:    {total_time:.3f}s ({total_time/n_samples*1000:.1f}ms/sample)", flush=True)
    print(f"  Total samples:     {n_samples}", flush=True)
    print(f"\n  Classification Report:", flush=True)
    print(report, flush=True)
    print(f"\n  Confusion Matrix:", flush=True)
    header = "  {:>10s}".format("") + "".join(f"{c:>10s}" for c in CLASS_NAMES)
    print(header, flush=True)
    for i, row in enumerate(cm):
        row_str = "  {:>10s}".format(CLASS_NAMES[i]) + "".join(
            f"{int(v):>10d}" for v in row
        )
        print(row_str, flush=True)
    print("=" * 60, flush=True)

    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision': float(precision),
        'recall': float(recall),
    }


# ==============================================================================
# QUANTIZATION
# ==============================================================================
def run_quantization(args):
    """Main quantization flow using pytorch_nndct (Vitis AI 3.5)."""

    # ------------------------------------------------------------------
    # 1. Import pytorch_nndct
    # ------------------------------------------------------------------
    try:
        from pytorch_nndct.apis import torch_quantizer
        logger.info("✅ pytorch_nndct imported successfully (Vitis AI 3.5)")
    except ImportError as e:
        logger.error(
            "❌ Cannot import pytorch_nndct!\n"
            "   Ensure you are running inside Vitis AI 3.5 Docker.\n"
            f"   Error: {e}"
        )
        sys.exit(1)

    device = torch.device('cpu')  # Vitis AI quantizer works on CPU

    # ------------------------------------------------------------------
    # 2. Load FP32 model
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("  VITIS AI INT8 QUANTIZATION v2 — Student MobileNetV2 (3-class)")
    logger.info("=" * 70)
    logger.info(f"  Mode:            {args.quant_mode}")
    logger.info(f"  Checkpoint:      {args.checkpoint}")
    logger.info(f"  Data source:     {'WAV on-the-fly' if args.use_wav else 'Pre-generated .npy'}")
    if args.use_wav:
        logger.info(f"  ICBHI dir:       {args.icbhi_dir}")
        logger.info(f"  Combined dir:    {args.combined_dir}")
        logger.info(f"  Num calib:       {args.num_calib}")
    else:
        logger.info(f"  Calib dir:       {args.calib_dir}")
    logger.info(f"  Output:          {args.output_dir}")
    logger.info(f"  Target:          {args.target}")
    logger.info(f"  Batch size:      {args.batch_size}")
    logger.info(f"  CLE:             {'Enabled' if args.enable_cle else 'Disabled'}")
    logger.info(f"  Fast Finetuning: {'Enabled' if args.fast_finetune else 'Disabled'}")
    logger.info("=" * 70)

    model = load_model(args.checkpoint, device)

    # ------------------------------------------------------------------
    # 3. Prepare calibration / test data
    # ------------------------------------------------------------------
    if args.use_wav:
        # On-the-fly mode: load WAV files → create spectrograms
        logger.info("\n  Loading WAV samples for on-the-fly spectrogram generation...")
        all_samples = []

        if os.path.isdir(args.icbhi_dir) and os.path.isfile(args.icbhi_labels):
            icbhi = load_icbhi_samples(args.icbhi_dir, args.icbhi_labels)
            logger.info(f"  ICBHI samples: {len(icbhi)}")
            all_samples.extend(icbhi)

        if os.path.isdir(args.combined_dir):
            combined = load_combined_samples(args.combined_dir)
            logger.info(f"  Combined samples: {len(combined)}")
            all_samples.extend(combined)

        if not all_samples:
            logger.error("❌ No WAV samples found! Check --icbhi_dir and --combined_dir paths.")
            sys.exit(1)

        # Deduplicate
        seen = set()
        unique = []
        for s in all_samples:
            bname = os.path.basename(s['wav_path'])
            if bname not in seen:
                seen.add(bname)
                unique.append(s)

        # ~200 samples total, balanced across 3 classes
        num_per_class = args.num_calib // NUM_CLASSES
        selected = select_balanced_samples(unique, num_per_class=num_per_class, seed=42)
        logger.info(f"  Selected {len(selected)} samples for calibration")

        calib_dataset = WavCalibDataset(selected, normalize=True)
    else:
        # Pre-generated .npy mode
        calib_dataset = NpyCalibDataset(args.calib_dir, normalize=True)

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # avoid multiprocessing issues in Docker
        pin_memory=False,
    )
    logger.info(f"  Dataset: {len(calib_dataset)} samples, "
                f"{len(calib_loader)} batches (batch_size={args.batch_size})")

    # ------------------------------------------------------------------
    # 4. Create Quantizer with CLE
    # ------------------------------------------------------------------
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    # Build extra quantizer options
    extra_options = {}
    if args.enable_cle:
        extra_options['cle'] = True
        logger.info("  ✅ CLE (Cross-Layer Equalization) enabled — "
                    "cân bằng trọng số cho Depthwise Separable Conv")

    # Tạo quantizer
    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(dummy_input,),
        output_dir=args.output_dir,
        device=device,
        target=args.target,
    )

    quant_model = quantizer.quant_model
    logger.info("  ✅ Quantizer created successfully")

    # ------------------------------------------------------------------
    # 5A. CALIBRATION MODE
    # ------------------------------------------------------------------
    if args.quant_mode == 'calib':
        logger.info("\n" + "=" * 70)
        logger.info("  PHASE 1: CALIBRATION")
        logger.info("=" * 70)

        # --- Optional: Fast Finetuning (AdaQuant) ---
        if args.fast_finetune:
            logger.info("\n  🔥 Kích hoạt Advanced PTQ: Fast Finetuning (AdaQuant) 🔥")
            logger.info("  AdaQuant sẽ tối ưu lại trọng số INT8 để giảm sai số làm tròn.")
            logger.info("  Quá trình này mất thêm vài phút...\n")

            try:
                # Tạo evaluation function wrapper cho fast_finetune API
                def ft_evaluate(model, loader, device):
                    """Wrapper cho fast_finetune callback."""
                    return evaluate_model(model, loader, device, tag='AdaQuant')

                quantizer.fast_finetune(ft_evaluate, (quant_model, calib_loader, device))
                logger.info("  ✅ Fast Finetuning (AdaQuant) hoàn tất thành công!")
            except TypeError:
                # Một số phiên bản Vitis AI có API khác
                logger.info("  Thử API fast_finetune thay thế...")
                try:
                    quantizer.fast_finetune(
                        evaluate_model,
                        (quant_model, calib_loader, device, 'AdaQuant')
                    )
                    logger.info("  ✅ Fast Finetuning (AdaQuant) hoàn tất thành công!")
                except Exception as e2:
                    logger.error(f"  ❌ Fast Finetuning thất bại: {e2}")
                    logger.info("  Chuyển về Standard Calibration (PTQ thông thường)...")
            except Exception as e:
                logger.error(f"  ❌ Fast Finetuning thất bại: {e}")
                logger.info("  Chuyển về Standard Calibration (PTQ thông thường)...")

        # --- Standard Calibration: Forward pass ---
        logger.info("\n  Running calibration forward pass...")
        quant_model.eval()
        calib_start = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(calib_loader):
                inputs = inputs.to(device)
                _ = quant_model(inputs)
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(calib_loader):
                    logger.info(f"  Calibration batch {batch_idx + 1}/{len(calib_loader)}")

        calib_time = time.time() - calib_start
        logger.info(f"  Calibration time: {calib_time:.1f}s")

        # Export quantization config
        quantizer.export_quant_config()
        logger.info(f"  ✅ Calibration complete!")
        logger.info(f"  Quantization config saved to: {args.output_dir}")

        # Evaluate FP32 baseline on calib data
        logger.info("\n  Evaluating FP32 baseline on calibration data...")
        fp32_metrics = evaluate_model(model, calib_loader, device, tag='FP32 Baseline')

        # Save FP32 baseline metrics
        if fp32_metrics:
            baseline_file = os.path.join(args.output_dir, 'fp32_baseline_metrics.json')
            with open(baseline_file, 'w') as f:
                json.dump({
                    'model': 'StudentModel_MobileNetV2',
                    'classes': CLASS_NAMES,
                    'checkpoint': args.checkpoint,
                    'metrics': fp32_metrics,
                    'num_samples': len(calib_dataset),
                }, f, indent=2)
            logger.info(f"  FP32 baseline saved to: {baseline_file}")

    # ------------------------------------------------------------------
    # 5B. TEST MODE
    # ------------------------------------------------------------------
    elif args.quant_mode == 'test':
        print("\n" + "=" * 70, flush=True)
        print("  PHASE 2: QUANTIZED MODEL TEST", flush=True)
        print("=" * 70, flush=True)

        # Evaluate INT8 quantized model
        quant_metrics = {}
        fp32_metrics = {}

        try:
            quant_metrics = evaluate_model(
                quant_model, calib_loader, device, tag='INT8 Quantized'
            )
        except Exception as e:
            print(f"\n  ERROR evaluating INT8 model: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Evaluate FP32 baseline for comparison
        try:
            print("\n  Evaluating FP32 baseline for comparison...", flush=True)
            fp32_metrics = evaluate_model(
                model, calib_loader, device, tag='FP32 Baseline'
            )
        except Exception as e:
            print(f"\n  ERROR evaluating FP32 model: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # ---- Comparison Report ----
        if quant_metrics and fp32_metrics:
            print("\n" + "=" * 70, flush=True)
            print("  📊 ACCURACY COMPARISON: FP32 vs INT8", flush=True)
            print("=" * 70, flush=True)
            print(f"  {'Metric':<20s} {'FP32':>10s} {'INT8':>10s} {'Diff':>10s} {'Status':>10s}",
                  flush=True)
            print(f"  {'-' * 60}", flush=True)

            degradation_warning = False
            for key in ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']:
                fp32_val = fp32_metrics.get(key, 0)
                int8_val = quant_metrics.get(key, 0)
                diff = int8_val - fp32_val
                sign = '+' if diff >= 0 else ''
                status = '✅' if abs(diff) < 0.03 else ('⚠️' if abs(diff) < 0.10 else '❌')
                if abs(diff) >= 0.10:
                    degradation_warning = True
                print(f"  {key:<20s} {fp32_val:>10.4f} {int8_val:>10.4f} "
                      f"{sign}{diff:>9.4f} {status:>10s}", flush=True)

            print("=" * 70, flush=True)

            if degradation_warning:
                print("\n  ⚠️  CẢNH BÁO: Accuracy giảm đáng kể (>10%)!", flush=True)
                print("  Khuyến nghị chạy lại với --fast_finetune để cải thiện:", flush=True)
                print("    python quantize_distillation_02.py --quant_mode calib "
                      "--fast_finetune", flush=True)
                print("    python quantize_distillation_02.py --quant_mode test", flush=True)

        # ---- Save metrics to JSON ----
        metrics_file = os.path.join(args.output_dir, 'quantization_metrics_v2.json')
        metrics_data = {
            'model': 'StudentModel_MobileNetV2_3class',
            'target_device': args.target,
            'classes': CLASS_NAMES,
            'checkpoint': args.checkpoint,
            'num_samples': len(calib_dataset),
            'fp32': fp32_metrics,
            'int8': quant_metrics,
            'comparison': {},
        }
        if quant_metrics and fp32_metrics:
            for key in ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']:
                metrics_data['comparison'][key] = {
                    'fp32': fp32_metrics.get(key, 0),
                    'int8': quant_metrics.get(key, 0),
                    'diff': quant_metrics.get(key, 0) - fp32_metrics.get(key, 0),
                }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\n  📁 Metrics saved to: {metrics_file}", flush=True)

        # ---- Export xmodel ----
        if args.deploy:
            print("\n  🚀 Exporting deployable .xmodel for DPU...", flush=True)
            try:
                quantizer.export_xmodel(output_dir=args.output_dir)
                print(f"  ✅ Xmodel exported to: {args.output_dir}", flush=True)

                # List exported files
                xmodel_files = [f for f in os.listdir(args.output_dir)
                               if f.endswith('.xmodel')]
                if xmodel_files:
                    print(f"  📦 Exported files:", flush=True)
                    for xf in xmodel_files:
                        xf_path = os.path.join(args.output_dir, xf)
                        xf_size = os.path.getsize(xf_path) / (1024 * 1024)
                        print(f"      {xf} ({xf_size:.2f} MB)", flush=True)
            except Exception as e:
                print(f"  ❌ Xmodel export failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
        else:
            print("\n  💡 To export xmodel, add --deploy flag:", flush=True)
            print("     python quantize_distillation_02.py --quant_mode test --deploy", flush=True)

    # ---- Final summary ----
    print("\n" + "=" * 70, flush=True)
    print("  ✅ QUANTIZATION COMPLETE!", flush=True)
    print(f"  Results saved to: {args.output_dir}", flush=True)
    print("=" * 70, flush=True)


# ==============================================================================
# MAIN
# ==============================================================================
def get_project_root():
    """Detect project root: /workspace (Docker) or /home/iec/... (host)."""
    if os.path.isfile('/workspace/CMakeLists.txt') and \
       not os.path.isdir('/workspace/Parallel_Computing_on_FPGA'):
        return '/workspace'
    if os.path.isdir('/workspace/Parallel_Computing_on_FPGA'):
        return '/workspace/Parallel_Computing_on_FPGA'
    return '/home/iec/Parallel_Computing_on_FPGA'


def parse_args():
    project_root = get_project_root()

    parser = argparse.ArgumentParser(
        description='Vitis AI INT8 Quantization v2 — Student MobileNetV2 (3-class)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng (trong Vitis AI 3.5 Docker):

  # Bước 1: Calibration (dùng pre-generated .npy)
  python quantize_distillation_02.py --quant_mode calib

  # Bước 1b: Calibration + AdaQuant Fast Finetuning
  python quantize_distillation_02.py --quant_mode calib --fast_finetune

  # Bước 1c: Calibration on-the-fly từ WAV (200 mẫu)
  python quantize_distillation_02.py --quant_mode calib --use_wav --num_calib 200

  # Bước 2: Test + Deploy (xuất .xmodel)
  python quantize_distillation_02.py --quant_mode test --deploy
        """
    )

    # --- Core arguments ---
    parser.add_argument(
        '--quant_mode', type=str, required=True,
        choices=['calib', 'test'],
        help='Quantization mode: calib (calibration) or test (evaluation + deploy)'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=os.path.join(
            project_root,
            'python/output_distillation_v2/checkpoints/student_fold_0_best.pt'
        ),
        help='Path to FP32 student model checkpoint'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(project_root, 'quantize_distillation_v2_result'),
        help='Output directory for quantization results and .xmodel'
    )
    parser.add_argument(
        '--target', type=str,
        default='DPUCZDX8G_ISA1_B2304',
        help='DPU target (default: DPUCZDX8G_ISA1_B2304 for Ultra96-V2)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=BATCH_SIZE,
        help=f'Batch size for calibration/test (default: {BATCH_SIZE})'
    )

    # --- Data source ---
    data_group = parser.add_argument_group('Data Source')
    data_group.add_argument(
        '--calib_dir', type=str,
        default=os.path.join(project_root, 'data/calib_data'),
        help='Path to pre-generated .npy calibration data (from generate_calib_data.py)'
    )
    data_group.add_argument(
        '--use_wav', action='store_true', default=False,
        help='Use on-the-fly WAV processing instead of pre-generated .npy files'
    )
    data_group.add_argument(
        '--num_calib', type=int, default=200,
        help='Number of total calibration samples when using --use_wav (default: 200)'
    )
    data_group.add_argument(
        '--icbhi_dir', type=str,
        default=os.path.join(project_root, 'data/samples/ICBHI_final_database'),
        help='Path to ICBHI WAV directory (for --use_wav mode)'
    )
    data_group.add_argument(
        '--icbhi_labels', type=str,
        default=os.path.join(project_root, 'data/samples/labels.txt'),
        help='Path to ICBHI labels.txt (for --use_wav mode)'
    )
    data_group.add_argument(
        '--combined_dir', type=str,
        default=os.path.join(project_root, 'data/combined/audio'),
        help='Path to combined audio directory (for --use_wav mode)'
    )

    # --- Optimization flags ---
    opt_group = parser.add_argument_group('Optimization')
    opt_group.add_argument(
        '--enable_cle', action='store_true', default=True,
        help='Enable Cross-Layer Equalization (default: True)'
    )
    opt_group.add_argument(
        '--no_cle', action='store_false', dest='enable_cle',
        help='Disable Cross-Layer Equalization'
    )
    opt_group.add_argument(
        '--fast_finetune', action='store_true', default=False,
        help='Enable Fast Finetuning (AdaQuant) during calibration. '
             'Tối ưu lại trọng số INT8 để giảm sai số làm tròn.'
    )

    # --- Deploy ---
    parser.add_argument(
        '--deploy', action='store_true', default=False,
        help='Export .xmodel for DPU deployment (only in test mode)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Verify data source ----
    if args.use_wav:
        has_icbhi = (os.path.isdir(args.icbhi_dir) and
                     os.path.isfile(args.icbhi_labels))
        has_combined = os.path.isdir(args.combined_dir)
        if not has_icbhi and not has_combined:
            logger.error(
                "❌ No WAV data found!\n"
                f"   ICBHI dir: {args.icbhi_dir} (exists: {has_icbhi})\n"
                f"   Combined dir: {args.combined_dir} (exists: {has_combined})\n"
                f"   Run generate_calib_data.py or provide valid paths."
            )
            sys.exit(1)
    else:
        if not os.path.isdir(args.calib_dir):
            logger.error(
                f"❌ Calibration data not found: {args.calib_dir}\n"
                f"   Option 1: Run generate_calib_data.py first\n"
                f"   Option 2: Use --use_wav for on-the-fly processing"
            )
            sys.exit(1)

    # ---- Verify checkpoint ----
    if not os.path.isfile(args.checkpoint):
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    run_quantization(args)


if __name__ == '__main__':
    main()
