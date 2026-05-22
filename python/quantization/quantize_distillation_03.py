#!/usr/bin/env python3
"""
================================================================================
Distillation-based Quantization-Aware Training (QAT) v3
================================================================================
Student: MobileNetV2 (3-class: COPD, Healthy, Non-COPD)
Teacher: EfficientNet-B0 Ensemble (3 models)
Target:  Ultra96-V2 DPU B4096 (DPUCZDX8G_ISA1_B4096)
Library: pytorch_nndct (Vitis AI 3.5)

Selected weight: student_fold_0_best.pt (Fold 0)
  - Accuracy: 96.56%, F1 Macro: 94.57%, Loss: 0.1048
  - Best balanced per-class: COPD=98.17%, Healthy=95.89%, Non-COPD=89.66%

Pipeline:
  1. Load FP32 Student + Teacher Ensemble
  2. Cross-Layer Equalization (CLE) for MobileNetV2
  3. Init torch_quantizer(mode='train', bitwidth=8)
  4. Calibration pass (200 samples from Val set)
  5. QAT fine-tune Student with KD loss from Teacher
  6. Export quant config → test mode → export .xmodel

Usage (Vitis AI Docker):
  # Step 1: QAT training
  python quantize_distillation_03.py --mode qat --use_wav

  # Step 2: Test + Deploy
  python quantize_distillation_03.py --mode test --deploy

  # Full pipeline (qat → test → deploy)
  python quantize_distillation_03.py --mode full --use_wav --deploy
================================================================================
"""

import os
import sys
import gc
import argparse
import logging
import time
import json
import copy
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import (
    ARTIFACTS_DIR,
    COMBINED_AUDIO_DIR,
    ICBHI_DIR,
    ICBHI_LABELS,
)

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

# QAT Training hyperparams
QAT_EPOCHS = 20
QAT_LR = 1e-5
QAT_BATCH_SIZE = 8
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.7
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
NUM_CALIB_SAMPLES = 200

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
# MODELS — Identical to distillation_02.py
# ==============================================================================
class TeacherModel(nn.Module):
    """EfficientNet-B0 teacher with enhanced head."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False, dropout=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights='IMAGENET1K_V1' if pretrained else None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256),
            nn.Dropout(p=dropout * 0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class StudentModel(nn.Module):
    """MobileNetV2 student with enhanced head."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256),
            nn.Dropout(p=dropout * 0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ==============================================================================
# LOSS FUNCTIONS — Identical to distillation_02.py
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


class DistillationLoss(nn.Module):
    """KD Loss: alpha * KL_div(soft) + (1-alpha) * FocalLoss(hard)."""
    def __init__(self, hard_loss_fn, temperature=KD_TEMPERATURE, alpha=KD_ALPHA):
        super().__init__()
        self.hard_loss_fn = hard_loss_fn
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, targets):
        soft_s = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_t = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(soft_s, soft_t, reduction='batchmean') * (self.temperature ** 2)
        hard_loss = self.hard_loss_fn(student_logits, targets)
        return self.alpha * kd_loss + (1 - self.alpha) * hard_loss


# ==============================================================================
# DATASETS
# ==============================================================================
class NpyCalibDataset(Dataset):
    """Pre-generated .npy spectrograms + labels."""
    def __init__(self, calib_dir, normalize=True):
        self.calib_dir = calib_dir
        self.normalize_fn = transforms.Normalize(
            mean=NORM_MEAN, std=NORM_STD) if normalize else None
        labels_file = os.path.join(calib_dir, 'calib_labels.txt')
        self.items = []
        if os.path.isfile(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        self.items.append({
                            'filename': parts[0], 'class_idx': int(parts[1]),
                            'class_name': parts[2],
                        })
        else:
            npy_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.npy')])
            for npy_file in npy_files:
                self.items.append({'filename': npy_file, 'class_idx': -1, 'class_name': 'unknown'})
        logger.info(f"  Loaded {len(self.items)} calibration samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        spec = np.load(os.path.join(self.calib_dir, item['filename']))
        tensor = torch.from_numpy(spec).float()
        if self.normalize_fn:
            tensor = self.normalize_fn(tensor)
        return tensor, item['class_idx']


class WavCalibDataset(Dataset):
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
            if not line: continue
            parts = line.split('\t')
            if len(parts) >= 2:
                patient_labels[int(parts[0])] = parts[1].strip()

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    samples = []
    for wav_file in sorted(os.listdir(data_dir)):
        if not wav_file.endswith('.wav'): continue
        try:
            pid = int(wav_file.split('_')[0])
        except ValueError:
            continue
        if pid not in patient_labels: continue
        disease = patient_labels[pid]
        if disease not in DISEASE_TO_CLASS: continue
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
        if not os.path.isdir(class_dir): continue
        for wav_file in sorted(os.listdir(class_dir)):
            if not wav_file.endswith('.wav'): continue
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
    np.random.seed(seed)
    np.random.shuffle(selected)
    return selected


def build_dataloaders(args, num_calib=200, batch_size=8):
    """Build train & calib dataloaders from WAV or .npy sources."""
    if args.use_wav:
        logger.info("  Loading WAV samples for on-the-fly processing...")
        all_samples = []
        if os.path.isdir(args.icbhi_dir) and os.path.isfile(args.icbhi_labels):
            icbhi = load_icbhi_samples(args.icbhi_dir, args.icbhi_labels)
            logger.info(f"    ICBHI: {len(icbhi)} samples")
            all_samples.extend(icbhi)
        if os.path.isdir(args.combined_dir):
            combined = load_combined_samples(args.combined_dir)
            logger.info(f"    Combined: {len(combined)} samples")
            all_samples.extend(combined)
        if not all_samples:
            logger.error("No WAV samples found!"); sys.exit(1)
        # Deduplicate
        seen, unique = set(), []
        for s in all_samples:
            bn = os.path.basename(s['wav_path'])
            if bn not in seen:
                seen.add(bn); unique.append(s)

        # Split: calib_samples for calibration, rest for QAT training
        num_per_class = num_calib // NUM_CLASSES
        calib_samples = select_balanced_samples(unique, num_per_class, seed=42)
        calib_set = set(id(s) for s in calib_samples)
        train_samples = [s for s in unique if id(s) not in calib_set]
        logger.info(f"  Calib: {len(calib_samples)}, Train: {len(train_samples)}")

        calib_dataset = WavCalibDataset(calib_samples)
        train_dataset = WavCalibDataset(train_samples)
    else:
        calib_dataset = NpyCalibDataset(args.calib_dir)
        train_dataset = calib_dataset  # fallback: use calib for training too

    calib_loader = DataLoader(calib_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    return calib_loader, train_loader, calib_dataset, train_dataset


# ==============================================================================
# MODEL LOADING
# ==============================================================================
def load_student(checkpoint_path, device='cpu'):
    model = StudentModel(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    f1 = ckpt.get('f1', 'N/A')
    logger.info(f"  Student loaded — F1: {f1}, Params: {sum(p.numel() for p in model.parameters()):,}")
    return model


def load_teacher_ensemble(checkpoint_dir, fold_id=0, n_teachers=3, device='cpu'):
    teachers = []
    for t_idx in range(n_teachers):
        ckpt_path = os.path.join(checkpoint_dir, f'teacher_{t_idx}_fold_{fold_id}_best.pt')
        if not os.path.isfile(ckpt_path):
            logger.warning(f"  Teacher {t_idx} not found: {ckpt_path}")
            continue
        model = TeacherModel(num_classes=NUM_CLASSES, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        teachers.append(model)
        f1 = ckpt.get('f1', 'N/A')
        logger.info(f"  Teacher {t_idx} loaded — F1: {f1}")
    if not teachers:
        logger.error("No teacher models loaded!"); sys.exit(1)
    logger.info(f"  Teacher ensemble: {len(teachers)} models")
    return teachers


# ==============================================================================
# CROSS-LAYER EQUALIZATION (CLE)
# ==============================================================================
def apply_cle(model):
    """Cross-Layer Equalization for MobileNetV2 depthwise separable convolutions.

    Equalizes weight ranges between consecutive Conv layers to reduce
    quantization error. Critical for INT8 on depthwise separable convs.
    """
    logger.info("  Applying Cross-Layer Equalization (CLE)...")
    equalized_pairs = 0
    features = model.backbone.features

    for block_idx in range(len(features)):
        block = features[block_idx]
        if not hasattr(block, 'conv'):
            continue
        convs = []
        for name, module in block.named_modules():
            if isinstance(module, nn.Conv2d):
                convs.append(module)
        # Equalize consecutive conv pairs
        for i in range(len(convs) - 1):
            conv1, conv2 = convs[i], convs[i + 1]
            if conv1.out_channels != conv2.in_channels:
                continue
            if conv2.groups == conv2.in_channels:  # skip if next is depthwise
                continue
            with torch.no_grad():
                # Compute per-channel scaling factors
                w1 = conv1.weight.data.reshape(conv1.out_channels, -1)
                w2 = conv2.weight.data.reshape(conv2.out_channels, conv2.in_channels, -1)
                range1 = w1.abs().max(dim=1)[0].clamp(min=1e-8)
                range2 = w2.abs().max(dim=2)[0].max(dim=0)[0].clamp(min=1e-8)
                scale = torch.sqrt(range1 / range2).clamp(min=0.01, max=100.0)
                # Apply scaling
                conv1.weight.data /= scale.view(-1, 1, 1, 1)
                if conv1.bias is not None:
                    conv1.bias.data /= scale
                conv2.weight.data *= scale.view(1, -1, 1, 1)
                equalized_pairs += 1

    logger.info(f"  CLE: equalized {equalized_pairs} conv pairs")
    return model


# ==============================================================================
# EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate_model(model, dataloader, device='cpu', tag=''):
    model.eval()
    all_preds, all_labels = [], []
    total_time = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        start = time.time()
        outputs = model(inputs)
        total_time += time.time() - start
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    valid = all_labels >= 0
    if valid.sum() == 0:
        logger.warning("No labeled samples!"); return {}
    all_preds, all_labels = all_preds[valid], all_labels[valid]
    n = len(all_preds)

    try:
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, classification_report, confusion_matrix)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
    except ImportError:
        accuracy = np.mean(all_preds == all_labels)
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for i in range(n):
            cm[int(all_labels[i])][int(all_preds[i])] += 1
        precs, recs, f1s = [], [], []
        for c in range(NUM_CLASSES):
            tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precs.append(p); recs.append(r)
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        f1_macro = np.mean(f1s)
        f1_weighted = np.average(f1s, weights=[cm[c, :].sum() for c in range(NUM_CLASSES)])
        precision = np.mean(precs)
        recall = np.mean(recs)
        report = "sklearn not available"

    print(f"\n{'=' * 60}", flush=True)
    print(f"  {tag} EVALUATION RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Accuracy:      {accuracy:.4f} ({accuracy * 100:.2f}%)", flush=True)
    print(f"  F1 (macro):    {f1_macro:.4f}", flush=True)
    print(f"  F1 (weighted): {f1_weighted:.4f}", flush=True)
    print(f"  Precision:     {precision:.4f}", flush=True)
    print(f"  Recall:        {recall:.4f}", flush=True)
    print(f"  Inference:     {total_time:.3f}s ({total_time / n * 1000:.1f}ms/sample)", flush=True)
    print(f"  Samples:       {n}", flush=True)
    print(f"\n{report}", flush=True)
    print(f"\n  Confusion Matrix:", flush=True)
    print("  " + "".join(f"{c:>10s}" for c in [''] + CLASS_NAMES), flush=True)
    for i, row in enumerate(cm):
        print("  " + f"{CLASS_NAMES[i]:>10s}" + "".join(f"{int(v):>10d}" for v in row), flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return {'accuracy': float(accuracy), 'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted), 'precision': float(precision),
            'recall': float(recall), 'confusion_matrix': cm.tolist(),
            'inference_time_s': float(total_time), 'num_samples': n}


# ==============================================================================
# QAT DISTILLATION TRAINING
# ==============================================================================
def run_qat_distillation(args):
    """Main QAT + Knowledge Distillation pipeline using pytorch_nndct."""

    # 1. Import pytorch_nndct
    try:
        from pytorch_nndct.apis import torch_quantizer
        logger.info("✅ pytorch_nndct imported (Vitis AI)")
    except ImportError as e:
        logger.error(f"❌ Cannot import pytorch_nndct! Run inside Vitis AI Docker.\n   {e}")
        sys.exit(1)

    device = torch.device('cpu')  # Vitis AI quantizer requires CPU

    # 2. Print configuration
    logger.info("=" * 70)
    logger.info("  QAT DISTILLATION v3 — MobileNetV2 → INT8 (DPU B4096)")
    logger.info("=" * 70)
    logger.info(f"  Mode:          {args.mode}")
    logger.info(f"  Student ckpt:  {args.student_ckpt}")
    logger.info(f"  Teacher dir:   {args.teacher_dir}")
    logger.info(f"  DPU target:    {args.target}")
    logger.info(f"  QAT epochs:    {args.qat_epochs}")
    logger.info(f"  QAT LR:        {args.qat_lr}")
    logger.info(f"  KD T={KD_TEMPERATURE}, α={KD_ALPHA}")
    logger.info(f"  Batch size:    {args.batch_size}")
    logger.info("=" * 70)

    # 3. Load FP32 Student
    logger.info("\n[1/7] Loading FP32 Student model...")
    student_fp32 = load_student(args.student_ckpt, device)
    student_fp32.to(device)

    # 4. Apply CLE
    if args.enable_cle:
        logger.info("\n[2/7] Cross-Layer Equalization...")
        student_fp32 = apply_cle(student_fp32)

    # 5. Load Teacher Ensemble
    logger.info("\n[3/7] Loading Teacher Ensemble...")
    teachers = load_teacher_ensemble(
        args.teacher_dir, fold_id=args.fold_id,
        n_teachers=args.n_teachers, device=device)

    # 6. Build dataloaders
    logger.info("\n[4/7] Building dataloaders...")
    calib_loader, train_loader, calib_ds, train_ds = build_dataloaders(
        args, num_calib=args.num_calib, batch_size=args.batch_size)

    # 7. Evaluate FP32 baseline
    logger.info("\n[5/7] FP32 Baseline evaluation...")
    fp32_metrics = evaluate_model(student_fp32, calib_loader, device, tag='FP32 Baseline')

    DUMMY = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    # ==================== QAT MODE ====================
    if args.mode in ('qat', 'full'):
        logger.info("\n[6/7] QAT Training Phase...")

        # Create quantizer in 'train' mode
        quantizer = torch_quantizer(
            quant_mode='train',
            module=student_fp32,
            input_args=(DUMMY,),
            output_dir=args.output_dir,
            device=device,
            target=args.target,
            bitwidth=8,
        )
        quant_model = quantizer.quant_model
        logger.info("  ✅ Quantizer (train mode) created")

        # Calibration pass: initialize quantization scales
        logger.info("  Running calibration forward pass...")
        quant_model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(calib_loader):
                inputs = inputs.to(device)
                _ = quant_model(inputs)
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"    Calib batch {batch_idx + 1}/{len(calib_loader)}")
        logger.info("  ✅ Calibration complete")

        # Setup KD loss & optimizer
        alpha_weights = torch.FloatTensor([1.0, 1.0, 1.0])
        hard_criterion = FocalLoss(alpha=alpha_weights, gamma=FOCAL_GAMMA,
                                   label_smoothing=LABEL_SMOOTHING)
        kd_criterion = DistillationLoss(hard_criterion, temperature=KD_TEMPERATURE,
                                        alpha=KD_ALPHA)
        eval_criterion = FocalLoss(alpha=alpha_weights, gamma=FOCAL_GAMMA)

        # IMPORTANT: Optimizer must use quant_model parameters
        optimizer = optim.AdamW(quant_model.parameters(), lr=args.qat_lr,
                                weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.qat_epochs, eta_min=1e-7)

        best_f1 = -1
        best_state = None
        history = {'epoch': [], 'train_loss': [], 'val_acc': [], 'val_f1': []}

        logger.info(f"\n  Starting QAT fine-tuning ({args.qat_epochs} epochs)...")
        for epoch in range(args.qat_epochs):
            # --- Train ---
            quant_model.train()
            for t in teachers:
                t.eval()
            total_loss, correct, total = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Teacher ensemble logits (no grad)
                with torch.no_grad():
                    teacher_logits = torch.zeros(inputs.size(0), NUM_CLASSES, device=device)
                    for t_model in teachers:
                        teacher_logits += t_model(inputs)
                    teacher_logits /= len(teachers)

                optimizer.zero_grad()
                student_logits = quant_model(inputs)
                loss = kd_criterion(student_logits, teacher_logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(quant_model.parameters(), GRADIENT_CLIP)
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, pred = student_logits.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

            scheduler.step()
            train_loss = total_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            # --- Validate ---
            val_metrics = evaluate_model(quant_model, calib_loader, device,
                                         tag=f'QAT Epoch {epoch + 1}')
            val_f1 = val_metrics.get('f1_macro', 0)
            val_acc = val_metrics.get('accuracy', 0)

            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1
                best_state = copy.deepcopy(quant_model.state_dict())

            logger.info(
                f"  Ep {epoch + 1:3d}/{args.qat_epochs} | "
                f"TrL: {train_loss:.4f} TrA: {train_acc * 100:.1f}% | "
                f"VaA: {val_acc * 100:.1f}% F1: {val_f1 * 100:.1f}%"
                f"{'  *BEST*' if is_best else ''}")

        # Restore best
        if best_state is not None:
            quant_model.load_state_dict(best_state)
            logger.info(f"  Restored best QAT model (F1: {best_f1 * 100:.2f}%)")

        # Export quantization config
        quantizer.export_quant_config()
        logger.info(f"  ✅ Quant config exported to: {args.output_dir}")

        # Save QAT history
        history_path = os.path.join(args.output_dir, 'qat_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Evaluate QAT result
        logger.info("\n  QAT Final evaluation...")
        qat_metrics = evaluate_model(quant_model, calib_loader, device, tag='QAT INT8')

        del quant_model, quantizer
        gc.collect()

    # ==================== TEST MODE ====================
    if args.mode in ('test', 'full'):
        logger.info("\n[7/7] Test Phase — Quantized Model Evaluation...")

        # Reload FP32 model for test quantizer
        student_test = load_student(args.student_ckpt, device)
        if args.enable_cle:
            student_test = apply_cle(student_test)
        student_test.to(device)

        quantizer_test = torch_quantizer(
            quant_mode='test',
            module=student_test,
            input_args=(DUMMY,),
            output_dir=args.output_dir,
            device=device,
            target=args.target,
            bitwidth=8,
        )
        quant_model_test = quantizer_test.quant_model
        logger.info("  ✅ Quantizer (test mode) created")

        # Evaluate INT8
        int8_metrics = evaluate_model(quant_model_test, calib_loader, device, tag='INT8 Quantized')

        # FP32 vs INT8 comparison
        if fp32_metrics and int8_metrics:
            print("\n" + "=" * 70, flush=True)
            print("  📊 FP32 vs INT8 COMPARISON", flush=True)
            print("=" * 70, flush=True)
            print(f"  {'Metric':<20s} {'FP32':>10s} {'INT8':>10s} {'Diff':>10s} {'Status':>8s}",
                  flush=True)
            print(f"  {'-' * 58}", flush=True)
            degrade = False
            for key in ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']:
                fp = fp32_metrics.get(key, 0)
                i8 = int8_metrics.get(key, 0)
                d = i8 - fp
                sign = '+' if d >= 0 else ''
                st = '✅' if abs(d) < 0.03 else ('⚠️' if abs(d) < 0.10 else '❌')
                if abs(d) >= 0.10:
                    degrade = True
                print(f"  {key:<20s} {fp:>10.4f} {i8:>10.4f} {sign}{d:>9.4f} {st:>8s}",
                      flush=True)
            print("=" * 70, flush=True)
            if degrade:
                print("  ⚠️ Significant accuracy degradation detected!", flush=True)
                print("  Try increasing --qat_epochs or adjusting --qat_lr", flush=True)

        # Save comprehensive metrics
        metrics_data = {
            'model': 'StudentModel_MobileNetV2_3class_QAT',
            'target_device': args.target,
            'classes': CLASS_NAMES,
            'student_checkpoint': args.student_ckpt,
            'qat_epochs': args.qat_epochs,
            'qat_lr': args.qat_lr,
            'kd_temperature': KD_TEMPERATURE,
            'kd_alpha': KD_ALPHA,
            'fp32': fp32_metrics if fp32_metrics else {},
            'int8': int8_metrics if int8_metrics else {},
            'timestamp': datetime.now().isoformat(),
        }
        metrics_file = os.path.join(args.output_dir, 'qat_metrics_v3.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"  📁 Metrics → {metrics_file}")

        # Export xmodel
        if args.deploy:
            logger.info("\n  🚀 Exporting .xmodel for DPU deployment...")
            try:
                quantizer_test.export_xmodel(output_dir=args.output_dir)
                logger.info(f"  ✅ Xmodel exported to: {args.output_dir}")
                for xf in os.listdir(args.output_dir):
                    if xf.endswith('.xmodel'):
                        sz = os.path.getsize(os.path.join(args.output_dir, xf)) / (1024 * 1024)
                        logger.info(f"    📦 {xf} ({sz:.2f} MB)")
            except Exception as e:
                logger.error(f"  ❌ Xmodel export failed: {e}")
                import traceback; traceback.print_exc()
        else:
            logger.info("  💡 Add --deploy to export .xmodel")

    # Final summary
    print("\n" + "=" * 70, flush=True)
    print("  ✅ QAT DISTILLATION COMPLETE!", flush=True)
    print(f"  Results → {args.output_dir}", flush=True)
    print("=" * 70, flush=True)


# ==============================================================================
# CLI
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='QAT Distillation v3 — MobileNetV2 INT8 for Ultra96-V2 DPU B4096',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage (inside Vitis AI 3.5 Docker):

  # Step 1: QAT training (Knowledge Distillation + Quantization-Aware Training)
  python quantize_distillation_03.py --mode qat --use_wav

  # Step 2: Test quantized model + export .xmodel
  python quantize_distillation_03.py --mode test --deploy

  # Full pipeline (QAT → Test → Export)
  python quantize_distillation_03.py --mode full --use_wav --deploy

  # Custom epochs/lr
  python quantize_distillation_03.py --mode full --use_wav --deploy \\
      --qat_epochs 30 --qat_lr 5e-6
        """)

    parser.add_argument('--mode', type=str, required=True,
                        choices=['qat', 'test', 'full'],
                        help='qat=train only, test=eval+deploy, full=qat→test→deploy')
    parser.add_argument('--artifact_root', type=str, default=str(ARTIFACTS_DIR),
                        help='Root artifacts directory for default inputs/outputs')
    parser.add_argument('--student_ckpt', type=str, default=None,
                        help='FP32 Student checkpoint (default: fold 0 best)')
    parser.add_argument('--teacher_dir', type=str, default=None,
                        help='Directory with teacher checkpoint files')
    parser.add_argument('--fold_id', type=int, default=0,
                        help='Fold ID for teacher models (default: 0)')
    parser.add_argument('--n_teachers', type=int, default=3,
                        help='Number of teacher models in ensemble')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--target', type=str, default='DPUCZDX8G_ISA1_B4096',
                        help='DPU target (default: B4096 for Ultra96-V2)')

    # QAT hyperparams
    qat_group = parser.add_argument_group('QAT Hyperparameters')
    qat_group.add_argument('--qat_epochs', type=int, default=QAT_EPOCHS)
    qat_group.add_argument('--qat_lr', type=float, default=QAT_LR)
    qat_group.add_argument('--batch_size', type=int, default=QAT_BATCH_SIZE)
    qat_group.add_argument('--num_calib', type=int, default=NUM_CALIB_SAMPLES,
                           help='Number of calibration samples')

    # Data source
    data_group = parser.add_argument_group('Data Source')
    data_group.add_argument('--use_wav', action='store_true', default=False,
                            help='On-the-fly WAV processing')
    data_group.add_argument('--calib_dir', type=str, default=None)
    data_group.add_argument('--icbhi_dir', type=str, default=str(ICBHI_DIR))
    data_group.add_argument('--icbhi_labels', type=str, default=str(ICBHI_LABELS))
    data_group.add_argument('--combined_dir', type=str, default=str(COMBINED_AUDIO_DIR))

    # Flags
    parser.add_argument('--enable_cle', action='store_true', default=True)
    parser.add_argument('--no_cle', action='store_false', dest='enable_cle')
    parser.add_argument('--deploy', action='store_true', default=False,
                        help='Export .xmodel for DPU')

    args = parser.parse_args()
    artifact_root = Path(args.artifact_root)
    distill_dir = artifact_root / 'training' / 'distillation_v2'
    if args.student_ckpt is None:
        args.student_ckpt = str(distill_dir / 'checkpoints' / 'student_fold_0_best.pt')
    if args.teacher_dir is None:
        args.teacher_dir = str(distill_dir / 'checkpoints')
    if args.output_dir is None:
        args.output_dir = str(artifact_root / 'quantization' / 'vitis_qat_v3')
    if args.calib_dir is None:
        args.calib_dir = str(artifact_root / 'quantization' / 'calibration_data')
    return args


def main():
    args = parse_args()

    # Validate paths
    if not os.path.isfile(args.student_ckpt):
        logger.error(f"❌ Student checkpoint not found: {args.student_ckpt}"); sys.exit(1)
    if args.use_wav:
        has_data = (os.path.isdir(args.icbhi_dir) or os.path.isdir(args.combined_dir))
        if not has_data:
            logger.error("❌ No WAV data found!"); sys.exit(1)
    elif not os.path.isdir(args.calib_dir):
        logger.error(f"❌ Calib dir not found: {args.calib_dir}. Use --use_wav"); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    run_qat_distillation(args)


if __name__ == '__main__':
    main()
