#!/usr/bin/env python3
"""
ICBHI 2017 — Official 4-class Respiratory Event Classification
Teacher-Ensemble Knowledge Distillation → Pure-CNN Student (FPGA-friendly)

Task:  4-class (Normal / Crackle / Wheeze / Both)  or  2-class (Normal / Abnormal)
Labels:  Per-cycle annotation from .txt files (crackle/wheeze columns)
Split:  Official ICBHI 60/40 patient-wise split
Primary metric:  ICBHI Score = (Sensitivity + Specificity) / 2

Pipeline stages:
  1) teachers   – Train teacher ensemble (multi-seed)
  2) soft-labels – Generate soft labels from teacher ensemble
  3) students   – Distill CNN student from soft labels
  4) evaluate   – Evaluate all models on test set

Student model is always a pure CNN (FPGA-friendly).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal
from scipy.io import wavfile
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import (
    ARTIFACTS_DIR,
    ICBHI_2017_DIR,
    ICBHI_4CLASS_SOTA_KD_ARTIFACTS_DIR,
    ensure_dir,
)

# =============================================================================
# ICBHI 4-class / 2-class definitions
# =============================================================================
CLASS_NAMES_4 = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_NAMES_2 = ["Normal", "Abnormal"]

# Official ICBHI 60/40 patient-wise split
# Patients 101-160 → train (60 patients, ~60% of data)
# Patients 161-226 → test  (66 patients, ~40% of data)
OFFICIAL_TRAIN_PATIENTS = set(range(101, 161))
OFFICIAL_TEST_PATIENTS = set(range(161, 227))


def get_class_names(num_classes: int) -> list[str]:
    return CLASS_NAMES_4 if num_classes == 4 else CLASS_NAMES_2


# =============================================================================
# Data records
# =============================================================================
@dataclass(frozen=True)
class CycleRecord:
    sample_id: str
    wav_path: str
    subject_id: str
    start_sec: float
    end_sec: float
    crackle: int
    wheeze: int
    label_4class: int  # 0=Normal,1=Crackle,2=Wheeze,3=Both
    label_2class: int  # 0=Normal,1=Abnormal


@dataclass(frozen=True)
class FeatureStats:
    mean: float
    std: float


# =============================================================================
# Label extraction from ICBHI annotation files
# =============================================================================
def event_to_4class(crackle: int, wheeze: int) -> int:
    """Map (crackle, wheeze) flags to 4-class label."""
    if crackle == 0 and wheeze == 0:
        return 0  # Normal
    if crackle == 1 and wheeze == 0:
        return 1  # Crackle
    if crackle == 0 and wheeze == 1:
        return 2  # Wheeze
    return 3  # Both


def event_to_2class(crackle: int, wheeze: int) -> int:
    """Map (crackle, wheeze) flags to 2-class label (Normal vs Abnormal)."""
    return 0 if (crackle == 0 and wheeze == 0) else 1


def read_cycle_annotations(wav_path: Path) -> list[tuple[float, float, int, int]]:
    """Read ICBHI annotation .txt file: start end crackle wheeze."""
    annotation_path = wav_path.with_suffix(".txt")
    if not annotation_path.exists():
        return []
    cycles: list[tuple[float, float, int, int]] = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                crackle = int(parts[2])
                wheeze = int(parts[3])
            except ValueError:
                continue
            if end > start:
                cycles.append((start, end, crackle, wheeze))
    return cycles


def build_records(
    data_dir: Path,
    max_files: int | None = None,
    max_cycles: int | None = None,
    allowed_basenames: set[str] | None = None,
) -> list[CycleRecord]:
    """Build cycle-level records from ICBHI annotation files."""
    wav_files = sorted(data_dir.glob("*.wav"))
    if allowed_basenames is not None:
        wav_files = [p for p in wav_files if p.stem in allowed_basenames]
    if max_files is not None and allowed_basenames is None:
        wav_files = wav_files[:max_files]
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")

    records: list[CycleRecord] = []
    for wav_path in wav_files:
        subject_id = wav_path.stem.split("_")[0]
        cycles = read_cycle_annotations(wav_path)
        for cycle_idx, (start, end, crackle, wheeze) in enumerate(cycles):
            sample_id = f"{wav_path.stem}__cycle_{cycle_idx:03d}"
            records.append(
                CycleRecord(
                    sample_id=sample_id,
                    wav_path=str(wav_path),
                    subject_id=subject_id,
                    start_sec=start,
                    end_sec=end,
                    crackle=crackle,
                    wheeze=wheeze,
                    label_4class=event_to_4class(crackle, wheeze),
                    label_2class=event_to_2class(crackle, wheeze),
                )
            )
            if max_cycles is not None and len(records) >= max_cycles:
                return records
    if not records:
        raise ValueError("No annotated cycles found. Check data_dir and annotation files.")
    return records


def get_label(record: CycleRecord, num_classes: int) -> int:
    return record.label_4class if num_classes == 4 else record.label_2class


# =============================================================================
# Split protocols and helpers
# =============================================================================
def record_basenames(records: list[CycleRecord]) -> set[str]:
    return {Path(r.wav_path).name.split(".")[0] for r in records}


def assert_split_filename_protocol(splits: dict[str, list[CycleRecord]], allow_val_test_overlap: bool = False) -> None:
    names = {k: record_basenames(v) for k, v in splits.items()}
    assert names["train"].isdisjoint(names["val"]), "Train/val filename overlap detected"
    assert names["train"].isdisjoint(names["test"]), "Train/test filename overlap detected"
    if allow_val_test_overlap:
        assert names["val"] == names["test"], "Expected validation split to match test split for test-selection mode"
    else:
        assert names["val"].isdisjoint(names["test"]), "Val/test filename overlap detected"


def split_file_counts(splits: dict[str, list[CycleRecord]]) -> dict[str, int]:
    return {k: len(record_basenames(v)) for k, v in splits.items()}


def create_add_rsc_splits(data_dir: Path, args: argparse.Namespace) -> dict[str, list[CycleRecord]]:
    basenames = sorted(p.stem for p in data_dir.glob("*.wav") if p.with_suffix(".txt").exists())
    if args.max_files is not None:
        basenames = basenames[:args.max_files]
    if not basenames:
        raise FileNotFoundError(f"No paired .wav/.txt files found in {data_dir}")
    indices = list(range(len(basenames)))
    random.Random(args.add_rsc_split_seed).shuffle(indices)
    n_train_pool = int(0.6 * len(indices))
    train_pool = [basenames[i] for i in indices[:n_train_pool]]
    test_files = [basenames[i] for i in indices[n_train_pool:]]
    if args.add_rsc_use_test_for_selection:
        train_files = train_pool
        val_files = test_files
    else:
        n_train = int(0.8 * len(train_pool))
        train_files = train_pool[:n_train]
        val_files = train_pool[n_train:]
    splits = {
        "train": build_records(data_dir, max_cycles=args.max_cycles, allowed_basenames=set(train_files)),
        "val": build_records(data_dir, max_cycles=args.max_cycles, allowed_basenames=set(val_files)),
        "test": build_records(data_dir, max_cycles=args.max_cycles, allowed_basenames=set(test_files)),
    }
    assert_split_filename_protocol(splits, allow_val_test_overlap=args.add_rsc_use_test_for_selection)
    return splits


def create_official_splits(
    records: list[CycleRecord],
    num_classes: int,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[CycleRecord]]:
    """Patient-wise split: official train/test + stratified val from train."""
    train_records: list[CycleRecord] = []
    test_records: list[CycleRecord] = []
    for record in records:
        pid = int(record.subject_id)
        if pid in OFFICIAL_TRAIN_PATIENTS:
            train_records.append(record)
        elif pid in OFFICIAL_TEST_PATIENTS:
            test_records.append(record)

    # Split train → train + val (patient-wise)
    subject_to_records: dict[str, list[CycleRecord]] = {}
    subject_to_label: dict[str, int] = {}
    for record in train_records:
        subject_to_records.setdefault(record.subject_id, []).append(record)
        subject_to_label[record.subject_id] = max(subject_to_label.get(record.subject_id, 0), get_label(record, num_classes))

    subjects = sorted(subject_to_records)
    if not subjects:
        raise ValueError("Official train split is empty. Check data_dir/max_files.")
    labels = [subject_to_label[s] for s in subjects]

    # Stratified patient-wise val split
    counts = {lb: labels.count(lb) for lb in set(labels)}
    stratify = labels if len(counts) > 1 and min(counts.values()) >= 2 else None
    try:
        train_subj, val_subj = train_test_split(
            subjects, test_size=val_fraction, random_state=seed, stratify=stratify
        )
    except ValueError:
        train_subj, val_subj = train_test_split(
            subjects, test_size=val_fraction, random_state=seed
        )

    train_final = [r for s in sorted(train_subj) for r in subject_to_records[s]]
    val_final = [r for s in sorted(val_subj) for r in subject_to_records[s]]
    splits = {"train": train_final, "val": val_final, "test": test_records}
    assert_split_filename_protocol(splits)
    return splits


# =============================================================================
# Audio I/O and preprocessing
# =============================================================================
def load_audio(
    wav_path: Path, target_sr: int, bandpass: bool, f_min: float, f_max: float
) -> tuple[np.ndarray, int]:
    sample_rate, audio = wavfile.read(str(wav_path))
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max(max_value, 1)
    else:
        audio = audio.astype(np.float32)
    if sample_rate != target_sr:
        gcd = math.gcd(sample_rate, target_sr)
        audio = sp_signal.resample_poly(audio, target_sr // gcd, sample_rate // gcd).astype(
            np.float32
        )
        sample_rate = target_sr
    if bandpass and len(audio) > 32:
        high = min(f_max, sample_rate / 2 - 1)
        if high > f_min:
            sos = sp_signal.butter(
                4, [f_min, high], btype="bandpass", fs=sample_rate, output="sos"
            )
            audio = sp_signal.sosfiltfilt(sos, audio).astype(np.float32)
    return audio, sample_rate


def segment_waveform(
    audio: np.ndarray, sample_rate: int, start_sec: float, end_sec: float, target_samples: int
) -> np.ndarray:
    """Extract segment with cyclic padding if too short."""
    start_idx = max(0, int(round(start_sec * sample_rate)))
    end_idx = min(len(audio), int(round(end_sec * sample_rate)))
    segment = audio[start_idx:end_idx]
    if len(segment) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    # Cyclic padding (recommended for respiratory sounds)
    if len(segment) >= target_samples:
        offset = (len(segment) - target_samples) // 2
        return segment[offset : offset + target_samples].astype(np.float32)
    repeats = target_samples // len(segment) + 1
    padded = np.tile(segment, repeats)[:target_samples]
    return padded.astype(np.float32)


# =============================================================================
# Mel spectrogram (pure numpy/scipy, no librosa)
# =============================================================================
def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float
) -> np.ndarray:
    f_max = min(f_max, sample_rate / 2 - 1)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bin_points[i : i + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for j in range(left, min(center, filters.shape[1])):
            filters[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, min(right, filters.shape[1])):
            filters[i, j] = (right - j) / max(right - center, 1)
    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-8)
    return filters * enorm[:, np.newaxis]


def compute_logmel(
    waveform: np.ndarray,
    sample_rate: int,
    mel_filterbank: np.ndarray,
    n_fft: int,
    win_length: int,
    hop_length: int,
    target_frames: int,
) -> np.ndarray:
    _, _, stft = sp_signal.stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(stft).astype(np.float32) ** 2
    mel = np.matmul(mel_filterbank, power)
    logmel = np.log(np.maximum(mel, 1e-10)).astype(np.float32)
    if logmel.shape[1] >= target_frames:
        logmel = logmel[:, :target_frames]
    else:
        pad = np.full((logmel.shape[0], target_frames), float(logmel.min()), dtype=np.float32)
        pad[:, : logmel.shape[1]] = logmel
        logmel = pad
    return logmel


# =============================================================================
# Augmentation
# =============================================================================
def apply_waveform_augmentation(
    waveform: np.ndarray, max_shift: float, noise_std: float, speed_perturb: bool
) -> np.ndarray:
    aug = waveform.copy()
    # Time shift
    if max_shift > 0:
        shift = int(np.random.uniform(-max_shift, max_shift) * len(aug))
        aug = np.roll(aug, shift)
    # Gaussian noise (SNR-aware)
    if noise_std > 0:
        rms = float(np.sqrt(np.mean(aug**2) + 1e-8))
        aug = aug + np.random.normal(0.0, noise_std * rms, size=aug.shape).astype(np.float32)
    # Speed perturbation
    if speed_perturb and np.random.random() < 0.3:
        rate = np.random.choice([0.9, 1.0, 1.1])
        if rate != 1.0:
            orig_len = len(aug)
            # Use fast linear interpolation instead of expensive FFT-based resample
            x_old = np.linspace(0, 1, orig_len)
            x_new = np.linspace(0, 1, int(orig_len * rate))
            aug = np.interp(x_new, x_old, aug).astype(np.float32)
            if len(aug) > orig_len:
                aug = aug[:orig_len]
            else:
                aug = np.pad(aug, (0, orig_len - len(aug)))
    return aug.astype(np.float32)


def apply_specaugment(feature: np.ndarray, freq_mask: int, time_mask: int) -> np.ndarray:
    aug = feature.copy()
    fill = float(aug.mean())
    if freq_mask > 0 and aug.shape[0] > 1:
        w = np.random.randint(0, min(freq_mask, aug.shape[0] - 1) + 1)
        s = np.random.randint(0, max(aug.shape[0] - w, 1))
        aug[s : s + w, :] = fill
    if time_mask > 0 and aug.shape[1] > 1:
        w = np.random.randint(0, min(time_mask, aug.shape[1] - 1) + 1)
        s = np.random.randint(0, max(aug.shape[1] - w, 1))
        aug[:, s : s + w] = fill
    return aug.astype(np.float32)


def apply_mixup(
    features: torch.Tensor, labels: torch.Tensor, alpha: float, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """MixUp augmentation on a batch. Returns mixed features and soft label vectors."""
    if alpha <= 0:
        one_hot = F.one_hot(labels, num_classes).float()
        return features, one_hot
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # ensure lambda >= 0.5
    batch_size = features.size(0)
    perm = torch.randperm(batch_size, device=features.device)
    mixed = lam * features + (1.0 - lam) * features[perm]
    y_a = F.one_hot(labels, num_classes).float()
    y_b = F.one_hot(labels[perm], num_classes).float()
    mixed_labels = lam * y_a + (1.0 - lam) * y_b
    return mixed, mixed_labels


# =============================================================================
# Dataset
# =============================================================================
class ICBHICycleDataset(Dataset):
    def __init__(
        self,
        records: list[CycleRecord],
        num_classes: int,
        sample_rate: int,
        duration_sec: float,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        target_frames: int,
        f_min: float,
        f_max: float,
        bandpass: bool,
        stats: FeatureStats | None = None,
        augment: bool = False,
        time_shift: float = 0.0,
        noise_std: float = 0.0,
        freq_mask: int = 0,
        time_mask: int = 0,
        speed_perturb: bool = False,
    ) -> None:
        self.records = records
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.target_samples = int(round(duration_sec * sample_rate))
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.target_frames = target_frames
        self.f_min = f_min
        self.f_max = f_max
        self.bandpass = bandpass
        self.stats = stats
        self.augment = augment
        self.time_shift = time_shift
        self.noise_std = noise_std
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.speed_perturb = speed_perturb
        self.mel_filterbank = build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self._audio_cache: dict[str, tuple[np.ndarray, int]] = {}
        self._spec_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.augment and index in self._spec_cache:
            return self._spec_cache[index]

        record = self.records[index]
        waveform, sr = self._load_cached(record.wav_path)
        segment = segment_waveform(waveform, sr, record.start_sec, record.end_sec, self.target_samples)
        if self.augment:
            segment = apply_waveform_augmentation(
                segment, self.time_shift, self.noise_std, self.speed_perturb
            )
        feature = compute_logmel(
            segment, sr, self.mel_filterbank, self.n_fft, self.win_length,
            self.hop_length, self.target_frames,
        )
        if self.augment:
            feature = apply_specaugment(feature, self.freq_mask, self.time_mask)
        if self.stats is not None:
            feature = (feature - self.stats.mean) / max(self.stats.std, 1e-6)
        feat_t = torch.from_numpy(feature).unsqueeze(0).float()
        label = get_label(record, self.num_classes)
        result = (feat_t, torch.tensor(label, dtype=torch.long), torch.tensor(index, dtype=torch.long))
        
        if not self.augment:
            self._spec_cache[index] = result
        return result

    def _load_cached(self, wav_path: str) -> tuple[np.ndarray, int]:
        if wav_path not in self._audio_cache:
            self._audio_cache[wav_path] = load_audio(
                Path(wav_path), self.sample_rate, self.bandpass, self.f_min, self.f_max
            )
        return self._audio_cache[wav_path]


def create_dataset(
    records: list[CycleRecord], args: argparse.Namespace, stats: FeatureStats | None, augment: bool
) -> ICBHICycleDataset:
    return ICBHICycleDataset(
        records=records,
        num_classes=args.num_classes,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        target_frames=args.target_frames,
        f_min=args.f_min,
        f_max=args.f_max,
        bandpass=not args.no_bandpass,
        stats=stats,
        augment=augment,
        time_shift=args.time_shift if augment else 0.0,
        noise_std=args.noise_std if augment else 0.0,
        freq_mask=args.freq_mask if augment else 0,
        time_mask=args.time_mask if augment else 0,
        speed_perturb=args.speed_perturb if augment else False,
    )


# =============================================================================
# Feature stats
# =============================================================================
def estimate_feature_stats(records: list[CycleRecord], args: argparse.Namespace) -> FeatureStats:
    dataset = create_dataset(records, args, stats=None, augment=False)
    limit = min(len(dataset), args.max_stat_samples)
    total_sum = 0.0
    total_sq = 0.0
    count = 0
    for idx in range(limit):
        feat, _, _ = dataset[idx]
        v = feat.float()
        total_sum += float(v.sum().item())
        total_sq += float((v * v).sum().item())
        count += v.numel()
    if count == 0:
        raise ValueError("Cannot estimate stats from empty set")
    mean = total_sum / count
    var = max(total_sq / count - mean * mean, 1e-12)
    return FeatureStats(mean=float(mean), std=float(math.sqrt(var)))


# =============================================================================
# Model architectures (CNN — FPGA-friendly)
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StudentCNN6(nn.Module):
    """Lightweight CNN student — FPGA-friendly, ~200-300K params."""
    def __init__(self, num_classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16), ConvBlock(16, 16), nn.MaxPool2d(2),
            ConvBlock(16, 32), ConvBlock(32, 32), nn.MaxPool2d(2),
            ConvBlock(32, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class WideCNN6Student(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        channels = [16, 32, 64, 128, 128, 128]
        layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.append(ConvBlock(in_ch, out_ch))
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(channels[-1], 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileStyleStudent(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16, stride=2),
            DepthwiseSeparableBlock(16, 24), DepthwiseSeparableBlock(24, 32, stride=2),
            DepthwiseSeparableBlock(32, 48), DepthwiseSeparableBlock(48, 64, stride=2),
            DepthwiseSeparableBlock(64, 96), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(96, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.skip(x))


class SmallResNetTeacher(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), ResidualBlock(32, 32), ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64), ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class EfficientCNNTeacher(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), DepthwiseSeparableBlock(32, 48), nn.MaxPool2d(2),
            DepthwiseSeparableBlock(48, 64), DepthwiseSeparableBlock(64, 96), nn.MaxPool2d(2),
            DepthwiseSeparableBlock(96, 128), DepthwiseSeparableBlock(128, 160),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(160, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_model(name: str, num_classes: int) -> nn.Module:
    if name == "cnn6":
        return StudentCNN6(num_classes=num_classes)
    if name == "cnn6_wide":
        return WideCNN6Student(num_classes=num_classes)
    if name == "mobilestyle":
        return MobileStyleStudent(num_classes=num_classes)
    if name == "small_resnet":
        return SmallResNetTeacher(num_classes=num_classes)
    if name == "efficient_cnn":
        return EfficientCNNTeacher(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


# =============================================================================
# Loss
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(
        self, alpha: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        nc = logits.size(1)
        if self.label_smoothing > 0:
            tp = torch.full_like(logits, self.label_smoothing / max(nc - 1, 1))
            tp.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            tp = F.one_hot(targets, nc).float()
        lp = F.log_softmax(logits, dim=1)
        p = lp.exp()
        loss = -((1.0 - p) ** self.gamma) * tp * lp
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).view(1, -1)
        return loss.sum(dim=1).mean()

    def forward_soft(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """Forward with soft label vectors (from MixUp)."""
        lp = F.log_softmax(logits, dim=1)
        p = lp.exp()
        loss = -((1.0 - p) ** self.gamma) * soft_targets * lp
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).view(1, -1)
        return loss.sum(dim=1).mean()


def class_weights(records: list[CycleRecord], num_classes: int, device: torch.device) -> torch.Tensor:
    labels = [get_label(r, num_classes) for r in records]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    w = counts.sum() / np.maximum(counts * num_classes, 1.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def sample_weights(records: list[CycleRecord], num_classes: int) -> torch.Tensor:
    labels = [get_label(r, num_classes) for r in records]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = counts.sum() / np.maximum(counts * num_classes, 1.0)
    return torch.tensor([w[lb] for lb in labels], dtype=torch.double)


def make_hard_criterion(
    records: list[CycleRecord], args: argparse.Namespace, device: torch.device
) -> FocalLoss | nn.CrossEntropyLoss:
    w = class_weights(records, args.num_classes, device)
    if args.hard_loss == "focal":
        return FocalLoss(alpha=w, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)


# =============================================================================
# Metrics
# =============================================================================
def per_class_specificity(cm: np.ndarray) -> np.ndarray:
    total = cm.sum()
    values = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        values.append(tn / max(tn + fp, 1))
    return np.array(values, dtype=np.float32)


def icbhi_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[float, float, float]:
    """
    Official ICBHI score = (Sensitivity + Specificity) / 2
    For 4-class: Sensitivity = recall of abnormal classes, Specificity = recall of Normal
    For 2-class: standard SE/SP
    """
    normal_idx = 0
    normal_mask = y_true == normal_idx
    abnormal_mask = y_true != normal_idx
    specificity = float(np.mean(y_pred[normal_mask] == normal_idx)) if normal_mask.any() else 0.0
    sensitivity = float(np.mean(y_pred[abnormal_mask] != normal_idx)) if abnormal_mask.any() else 0.0
    return sensitivity, specificity, (sensitivity + specificity) / 2.0


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float | None:
    try:
        if len(np.unique(y_true)) < num_classes:
            return None
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(num_classes))))
    except ValueError:
        return None


def binary_metrics_from_4class(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = (y_true != 0).astype(np.int64)
    yp = (y_pred != 0).astype(np.int64)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "binary_sensitivity": float(sens),
        "binary_specificity": float(spec),
        "binary_icbhi_score": float((sens + spec) / 2.0),
        "binary_accuracy": float(accuracy_score(yt, yp)),
    }


def threshold_predictions(probs: np.ndarray, threshold: float) -> np.ndarray:
    preds = probs.argmax(axis=1)
    if probs.shape[1] > 1:
        abnormal = probs[:, 1:].argmax(axis=1) + 1
        preds = np.where(probs[:, 0] >= threshold, 0, abnormal)
    return preds


def sweep_threshold(y_true: np.ndarray, probs: np.ndarray) -> dict:
    best = {"threshold": 0.5, "icbhi_score": -1.0}
    for th in np.linspace(0.05, 0.95, 91):
        pred = threshold_predictions(probs, float(th))
        m = compute_metrics(y_true, pred, probs, probs.shape[1])
        if m["icbhi_score"] > best["icbhi_score"]:
            best = {"threshold": float(th), **m}
    return best


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, num_classes: int
) -> dict[str, float | int | str | None]:
    class_names = get_class_names(num_classes)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    spec = per_class_specificity(cm)
    se, sp, score = icbhi_score(y_true, y_pred, num_classes)

    metrics: dict[str, float | int | str | None] = {
        "icbhi_score": float(score),
        "sensitivity": float(se),
        "specificity": float(sp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(prec.mean()),
        "recall_macro": float(rec.mean()),
        "confusion_matrix": cm.tolist(),
    }
    metrics.update(binary_metrics_from_4class(y_true, y_pred))
    if y_prob is not None:
        metrics["auc_ovr"] = safe_auc(y_true, y_prob, num_classes)

    for idx, cname in enumerate(class_names):
        key = cname.lower().replace(" ", "_")
        metrics[f"{key}_precision"] = float(prec[idx])
        metrics[f"{key}_recall"] = float(rec[idx])
        metrics[f"{key}_f1"] = float(f1[idx])
        metrics[f"{key}_specificity"] = float(spec[idx])
        metrics[f"{key}_support"] = int(sup[idx])
    return metrics


# =============================================================================
# W&B helpers
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.wandb or args.wandb_mode == "disabled":
        return
    if wandb is None:
        raise ImportError("wandb not installed")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name,
        dir=str(output_dir),
        mode=args.wandb_mode,
        config=vars(args),
        tags=["icbhi-2017", f"{args.num_classes}class", "knowledge-distillation", "fpga-student", "sota"],
    )


def log_wandb(metrics: dict, prefix: str) -> None:
    if wandb is None or wandb.run is None:
        return
    payload = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            payload[f"{prefix}/{k}"] = v
    if payload:
        wandb.log(payload)


def log_wandb_confusion(name: str, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> None:
    if wandb is None or wandb.run is None:
        return
    cn = get_class_names(num_classes)
    wandb.log({name: wandb.plot.confusion_matrix(probs=None, y_true=y_true.tolist(), preds=y_pred.tolist(), class_names=cn)})


def finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()


# =============================================================================
# Data loader helpers
# =============================================================================
def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int,
    records: list[CycleRecord] | None = None, balanced: bool = False,
    num_classes: int = 4,
) -> DataLoader:
    sampler = None
    if balanced and records is not None:
        sw = sample_weights(records, num_classes)
        sampler = WeightedRandomSampler(sw, num_samples=len(records), replacement=True)
        shuffle = False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


# =============================================================================
# Save/load
# =============================================================================
def save_split_records(output_dir: Path, splits: dict[str, list[CycleRecord]]) -> None:
    data = {s: [asdict(r) for r in recs] for s, recs in splits.items()}
    with (output_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_split_records(output_dir: Path) -> dict[str, list[CycleRecord]]:
    with (output_dir / "splits.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {s: [CycleRecord(**r) for r in recs] for s, recs in raw.items()}


def save_config(
    output_dir: Path, args: argparse.Namespace, stats: FeatureStats,
    records: list[CycleRecord], splits: dict[str, list[CycleRecord]],
) -> None:
    nc = args.num_classes
    cn = get_class_names(nc)
    label_counts = {name: 0 for name in cn}
    for r in records:
        label_counts[cn[get_label(r, nc)]] += 1
    config = vars(args).copy()
    config["feature_mean"] = stats.mean
    config["feature_std"] = stats.std
    config["class_names"] = cn
    config["label_counts"] = label_counts
    config["split_sizes"] = {s: len(recs) for s, recs in splits.items()}
    config["unique_patients"] = len({r.subject_id for r in records})
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_feature_stats(output_dir: Path) -> FeatureStats:
    with (output_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    return FeatureStats(mean=float(config["feature_mean"]), std=float(config["feature_std"]))


def save_metrics(
    output_dir: Path, name: str, metrics: dict, y_true: np.ndarray, y_pred: np.ndarray,
    num_classes: int,
) -> None:
    mdir = ensure_dir(output_dir / "metrics")
    with (mdir / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    cn = get_class_names(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with (mdir / f"confusion_matrix_{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *cn])
        for idx, row in enumerate(cm):
            writer.writerow([cn[idx], *row.tolist()])


# =============================================================================
# Training
# =============================================================================
def train_supervised(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    train_records: list[CycleRecord], args: argparse.Namespace,
    device: torch.device, output_dir: Path, epochs: int, lr: float,
) -> dict[str, float | int | str | None]:
    ensure_dir(output_dir)
    nc = args.num_classes
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = make_hard_criterion(train_records, args, device)
    best_metric = -float("inf")
    best_epoch = 0
    best_metrics: dict = {}
    patience_counter = 0
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels, _ in train_loader:
            features, labels = features.to(device), labels.to(device)
            # MixUp
            if args.mixup_alpha > 0 and isinstance(criterion, FocalLoss):
                features, soft_labels = apply_mixup(features, labels, args.mixup_alpha, nc)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion.forward_soft(logits, soft_labels)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * features.size(0)
        scheduler.step()

        val_metrics, yv, _, pv, _ = evaluate_model(model, val_loader, device, nc)
        if args.selection_metric == "threshold_icbhi_score":
            tuned = sweep_threshold(yv, pv)
            score = float(tuned["icbhi_score"])
        else:
            score = float(val_metrics[args.selection_metric])

        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            save_dict = {"model_state": model.state_dict(), "epoch": epoch, "metrics": val_metrics}
            if args.selection_metric == "threshold_icbhi_score":
                save_dict["threshold"] = tuned["threshold"]
                save_dict["threshold_metrics"] = tuned
            torch.save(save_dict, output_dir / "best.pt")
        else:
            patience_counter += 1
        avg_loss = total_loss / max(len(train_loader.dataset), 1)
        print(
            f"epoch={epoch:03d} loss={avg_loss:.4f} val_{args.selection_metric}={score:.4f} best={best_metric:.4f}",
            flush=True,
        )
        log_wandb(
            {"epoch": epoch, "train_loss": avg_loss, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}},
            f"{output_dir.parent.name}/{output_dir.name}",
        )
        if patience_counter >= args.patience:
            break
    best_metrics["best_epoch"] = best_epoch
    return best_metrics


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int,
) -> tuple[dict[str, float | int | str | None], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true_list: list[int] = []
    logits_all = []
    with torch.no_grad():
        for features, labels, _ in loader:
            logits_all.append(model(features.to(device)).cpu())
            y_true_list.extend(labels.numpy().tolist())
    logits = torch.cat(logits_all, dim=0).numpy() if logits_all else np.zeros((0, num_classes), dtype=np.float32)
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    y_true = np.array(y_true_list, dtype=np.int64)
    y_pred = probs.argmax(axis=1) if len(probs) else np.array([], dtype=np.int64)
    return compute_metrics(y_true, y_pred, probs, num_classes), y_true, y_pred, probs, logits


# =============================================================================
# KD training
# =============================================================================
def select_teacher_logits(
    teacher_logits: torch.Tensor, indices: torch.Tensor, kd_mode: str, temperature: float
) -> torch.Tensor:
    # teacher_logits: [num_teachers, num_samples, num_classes]
    selected = teacher_logits[:, indices, :]
    if kd_mode == "mean":
        return selected.mean(dim=0)
    if kd_mode == "prob_mean":
        probs = F.softmax(selected / temperature, dim=2).mean(dim=0)
        return torch.log(probs.clamp_min(1e-8)) * temperature
    raise ValueError(f"Unsupported kd_mode: {kd_mode}")


def kd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor, hard_criterion: nn.Module, args: argparse.Namespace,
) -> torch.Tensor:
    T = args.temperature
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    soft_loss = -(teacher_probs * student_log_probs).sum(dim=1).mean() * (T ** 2)
    if args.kd_loss == "soft_only":
        return soft_loss
    hard_loss = hard_criterion(student_logits, hard_labels)
    return args.alpha * soft_loss + (1.0 - args.alpha) * hard_loss


def train_student(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    train_records: list[CycleRecord], teacher_logits: np.ndarray,
    args: argparse.Namespace, device: torch.device, output_dir: Path, kd_mode: str,
) -> dict[str, float | int | str | None]:
    ensure_dir(output_dir)
    nc = args.num_classes
    tl = torch.tensor(teacher_logits, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_student, 1))
    hard_criterion = make_hard_criterion(train_records, args, device)
    best_metric = -float("inf")
    best_epoch = 0
    best_metrics: dict = {}
    patience_counter = 0
    model.to(device)

    for epoch in range(1, args.epochs_student + 1):
        model.train()
        total_loss = 0.0
        for features, labels, indices in train_loader:
            features, labels, indices = features.to(device), labels.to(device), indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(features)
            target_logits = select_teacher_logits(tl, indices, kd_mode, args.temperature)
            loss = kd_loss(student_logits, target_logits, labels, hard_criterion, args)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * features.size(0)
        scheduler.step()
        val_metrics, yv, _, pv, _ = evaluate_model(model, val_loader, device, nc)
        if args.selection_metric == "threshold_icbhi_score":
            tuned = sweep_threshold(yv, pv)
            score = float(tuned["icbhi_score"])
        else:
            score = float(val_metrics[args.selection_metric])

        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            save_dict = {"model_state": model.state_dict(), "epoch": epoch, "metrics": val_metrics}
            if args.selection_metric == "threshold_icbhi_score":
                save_dict["threshold"] = tuned["threshold"]
                save_dict["threshold_metrics"] = tuned
            torch.save(save_dict, output_dir / "best.pt")
        else:
            patience_counter += 1
        avg_loss = total_loss / max(len(train_loader.dataset), 1)
        print(
            f"student={kd_mode} epoch={epoch:03d} loss={avg_loss:.4f} val_{args.selection_metric}={score:.4f} best={best_metric:.4f}",
            flush=True,
        )
        log_wandb(
            {"epoch": epoch, "train_loss": avg_loss, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}},
            f"students/{kd_mode}_teacher",
        )
        if patience_counter >= args.patience:
            break
    best_metrics["best_epoch"] = best_epoch
    best_metrics["kd_mode"] = kd_mode
    return best_metrics


def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    logits_list: list[np.ndarray] = []
    with torch.no_grad():
        for features, _, _ in loader:
            logits_list.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(logits_list, axis=0)


# =============================================================================
# Pipeline orchestration
# =============================================================================
def prepare_run(args: argparse.Namespace) -> tuple[Path, dict[str, list[CycleRecord]], FeatureStats]:
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else ICBHI_4CLASS_SOTA_KD_ARTIFACTS_DIR)
    split_path = output_dir / "splits.json"
    if split_path.exists() and (output_dir / "config.json").exists() and not args.rebuild_splits:
        with (output_dir / "config.json").open("r", encoding="utf-8") as f:
            cached_cfg = json.load(f)
        cache_matches = (
            cached_cfg.get("max_files") == args.max_files
            and cached_cfg.get("max_cycles") == args.max_cycles
            and cached_cfg.get("num_classes") == args.num_classes
            and cached_cfg.get("benchmark_protocol", "official_icbhi") == args.benchmark_protocol
            and cached_cfg.get("add_rsc_split_seed") == args.add_rsc_split_seed
            and cached_cfg.get("add_rsc_use_test_for_selection") == args.add_rsc_use_test_for_selection
        )
        if cache_matches:
            splits = load_split_records(output_dir)
            stats = load_feature_stats(output_dir)
            return output_dir, splits, stats
        else:
            print("Configuration or data split constraints changed. Rebuilding splits...", flush=True)

    if args.benchmark_protocol == "official_icbhi":
        records = build_records(Path(args.data_dir), args.max_files, args.max_cycles)
        splits = create_official_splits(records, args.num_classes, args.val_size, args.seed)
    elif args.benchmark_protocol == "add_rsc":
        splits = create_add_rsc_splits(Path(args.data_dir), args)
        records = splits["train"] + splits["val"] + splits["test"]
    else:
        raise ValueError(f"Unknown benchmark protocol: {args.benchmark_protocol}")

    stats = estimate_feature_stats(splits["train"], args)
    save_split_records(output_dir, splits)
    save_config(output_dir, args, stats, records, splits)
    return output_dir, splits, stats


def parse_teacher_seeds(seed_arg: str, num_teachers: int) -> list[int]:
    seeds = [int(s.strip()) for s in seed_arg.split(",") if s.strip()]
    if len(seeds) < num_teachers:
        start = max(seeds) + 1 if seeds else 1
        seeds.extend(range(start, start + num_teachers - len(seeds)))
    return seeds[:num_teachers]


def train_teachers(args, output_dir, splits, stats, device):
    train_ds = create_dataset(splits["train"], args, stats, augment=True)
    val_ds = create_dataset(splits["val"], args, stats, augment=False)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, splits["train"], args.balanced_sampler, args.num_classes)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    seeds = parse_teacher_seeds(args.teacher_seeds, args.num_teachers)
    for seed in seeds:
        set_seed(seed)
        model = make_model(args.teacher_arch, args.num_classes)
        teacher_dir = output_dir / "teachers" / f"seed_{seed}"
        if args.skip_existing and (teacher_dir / "best.pt").exists():
            continue
        print(f"Training teacher seed={seed} arch={args.teacher_arch}", flush=True)
        best_metrics = train_supervised(model, train_loader, val_loader, splits["train"], args, device, teacher_dir, args.epochs_teacher, args.lr_teacher)
        with (teacher_dir / "val_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)


def create_soft_labels(args, output_dir, splits, stats, device):
    soft_dir = ensure_dir(output_dir / "soft_labels")
    seeds = parse_teacher_seeds(args.teacher_seeds, args.num_teachers)
    for split_name, records in splits.items():
        if not records:
            print(f"Skipping soft label generation for empty split: {split_name}", flush=True)
            continue
        dataset = create_dataset(records, args, stats, augment=False)
        loader = make_loader(dataset, args.batch_size, False, args.num_workers)
        all_logits = []
        for seed in seeds:
            cp = output_dir / "teachers" / f"seed_{seed}" / "best.pt"
            if not cp.exists():
                raise FileNotFoundError(f"Missing teacher checkpoint: {cp}")
            model = make_model(args.teacher_arch, args.num_classes)
            ckpt = torch.load(cp, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.to(device).eval()
            all_logits.append(collect_logits(model, loader, device))
        stacked = np.stack(all_logits, axis=0)
        np.save(soft_dir / f"teacher_logits_{split_name}.npy", stacked)
        with (soft_dir / f"sample_ids_{split_name}.json").open("w", encoding="utf-8") as f:
            json.dump([r.sample_id for r in records], f, indent=2)

    # Evaluate teacher ensemble on val/test
    nc = args.num_classes
    for split_name in ["val", "test"]:
        if not splits[split_name]:
            print(f"Skipping teacher ensemble evaluation for empty split: {split_name}", flush=True)
            continue
        logits = np.load(soft_dir / f"teacher_logits_{split_name}.npy")
        mean_logits = logits.mean(axis=0)
        probs = softmax_np(mean_logits, axis=1)
        y_true = np.array([get_label(r, nc) for r in splits[split_name]])
        y_pred = probs.argmax(axis=1)
        metrics = compute_metrics(y_true, y_pred, probs, nc)
        save_metrics(output_dir, f"teacher_ensemble_{split_name}", metrics, y_true, y_pred, nc)
        log_wandb(metrics, f"eval/teacher_ensemble_{split_name}")
        print(f"Teacher ensemble {split_name}: ICBHI={metrics['icbhi_score']:.4f} F1={metrics['macro_f1']:.4f} Acc={metrics['accuracy']:.4f}", flush=True)


def softmax_np(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def train_students(args, output_dir, splits, stats, device):
    soft_dir = output_dir / "soft_labels"
    train_logits_path = soft_dir / "teacher_logits_train.npy"
    if not train_logits_path.exists():
        create_soft_labels(args, output_dir, splits, stats, device)
    teacher_logits = np.load(train_logits_path)
    train_ds = create_dataset(splits["train"], args, stats, augment=True)
    val_ds = create_dataset(splits["val"], args, stats, augment=False)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, splits["train"], args.balanced_sampler, args.num_classes)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    kd_modes = [args.kd_mode] if args.kd_mode != "both" else ["mean", "prob_mean"]
    for kd_mode in kd_modes:
        set_seed(args.seed + 1000 + len(kd_mode))
        model = make_model(args.student_arch, args.num_classes)
        student_dir = output_dir / "students" / f"{kd_mode}_teacher"
        if args.skip_existing and (student_dir / "best.pt").exists():
            continue
        print(f"Training student arch={args.student_arch} kd_mode={kd_mode}", flush=True)
        best_metrics = train_student(model, train_loader, val_loader, splits["train"], teacher_logits, args, device, student_dir, kd_mode)
        with (student_dir / "val_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)


def evaluate_outputs(args, output_dir, splits, stats, device):
    nc = args.num_classes
    if not splits["test"]:
        print("Skipping evaluation: test split is empty", flush=True)
        return
    test_ds = create_dataset(splits["test"], args, stats, augment=False)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    # Evaluate teacher ensemble
    soft_dir = output_dir / "soft_labels"
    if (soft_dir / "teacher_logits_test.npy").exists():
        logits = np.load(soft_dir / "teacher_logits_test.npy")
        probs = softmax_np(logits.mean(axis=0), axis=1)
        y_true = np.array([get_label(r, nc) for r in splits["test"]])
        y_pred = probs.argmax(axis=1)
        metrics = compute_metrics(y_true, y_pred, probs, nc)
        save_metrics(output_dir, "teacher_ensemble_test", metrics, y_true, y_pred, nc)
        log_wandb(metrics, "eval/teacher_ensemble_test")
        log_wandb_confusion("confusion/teacher_ensemble_test", y_true, y_pred, nc)
        print(f"Teacher ensemble test: ICBHI={metrics['icbhi_score']:.4f} F1={metrics['macro_f1']:.4f}", flush=True)

    # Evaluate students
    students_dir = output_dir / "students"
    if not students_dir.exists():
        return
    for student_dir in sorted(students_dir.glob("*_teacher")):
        kd_mode = student_dir.name.removesuffix("_teacher")
        cp = student_dir / "best.pt"
        if not cp.exists():
            continue
        model = make_model(args.student_arch, nc)
        ckpt = torch.load(cp, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        
        # Evaluate model raw
        raw_m, yt, yp, probs, _ = evaluate_model(model, test_loader, device, nc)
        save_metrics(output_dir, f"student_{kd_mode}_test_raw", raw_m, yt, yp, nc)
        log_wandb(raw_m, f"eval/student_{kd_mode}_test_raw")
        log_wandb_confusion(f"confusion/student_{kd_mode}_test_raw", yt, yp, nc)
        
        # Apply the threshold found on val set
        threshold = float(ckpt.get("threshold", 0.5))
        tuned_pred = threshold_predictions(probs, threshold)
        tuned_m = compute_metrics(yt, tuned_pred, probs, nc)
        save_metrics(output_dir, f"student_{kd_mode}_test_threshold", tuned_m, yt, tuned_pred, nc)
        log_wandb(tuned_m, f"eval/student_{kd_mode}_test_threshold")
        log_wandb_confusion(f"confusion/student_{kd_mode}_test_threshold", yt, tuned_pred, nc)
        
        print(f"Student {kd_mode} test (raw): ICBHI={raw_m['icbhi_score']:.4f} F1={raw_m['macro_f1']:.4f} Acc={raw_m['accuracy']:.4f}", flush=True)
        print(f"Student {kd_mode} test (tuned, th={threshold:.2f}): ICBHI={tuned_m['icbhi_score']:.4f} F1={tuned_m['macro_f1']:.4f} Acc={tuned_m['accuracy']:.4f}", flush=True)


# =============================================================================
# CLI
# =============================================================================
def default_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICBHI 2017 official 4-class KD → CNN student (SOTA)")
    # Data
    p.add_argument("--data_dir", type=str, default=str(ICBHI_2017_DIR))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_classes", type=int, choices=[2, 4], default=4, help="4-class or 2-class (Normal vs Abnormal)")
    # Pipeline
    p.add_argument("--stage", choices=["all", "teachers", "soft-labels", "students", "evaluate"], default="all")
    p.add_argument("--teacher_arch", choices=["small_resnet", "efficient_cnn"], default="small_resnet")
    p.add_argument("--student_arch", choices=["cnn6", "cnn6_wide", "mobilestyle"], default="cnn6")
    p.add_argument("--num_teachers", type=int, default=5)
    p.add_argument("--teacher_seeds", type=str, default="1,2,3,4,5")
    # KD
    p.add_argument("--kd_mode", choices=["mean", "prob_mean", "both"], default="both")
    p.add_argument("--kd_loss", choices=["soft_only", "mixed"], default="mixed")
    p.add_argument("--hard_loss", choices=["ce", "focal"], default="focal")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--mixup_alpha", type=float, default=0.2, help="MixUp alpha (0=disabled)")
    # Audio
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=8.0)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--target_frames", type=int, default=512)
    p.add_argument("--f_min", type=float, default=50.0)
    p.add_argument("--f_max", type=float, default=2500.0)
    p.add_argument("--no_bandpass", action="store_true")
    # Augmentation
    p.add_argument("--time_shift", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--freq_mask", type=int, default=16)
    p.add_argument("--time_mask", type=int, default=64)
    p.add_argument("--speed_perturb", action="store_true", default=True)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--balanced_sampler", action="store_true", default=True)
    p.add_argument("--epochs_teacher", type=int, default=100)
    p.add_argument("--epochs_student", type=int, default=120)
    p.add_argument("--lr_teacher", type=float, default=1e-3)
    p.add_argument("--lr_student", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--selection_metric", choices=["icbhi_score", "macro_f1", "balanced_accuracy", "threshold_icbhi_score"], default="threshold_icbhi_score")
    # Split
    p.add_argument("--benchmark_protocol", choices=["official_icbhi", "add_rsc"], default="add_rsc")
    p.add_argument("--add_rsc_split_seed", type=int, default=1)
    p.add_argument("--add_rsc_use_test_for_selection", action="store_true", default=False)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    # System
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--max_stat_samples", type=int, default=512)
    p.add_argument("--rebuild_splits", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="icbhi-4class-sota-kd")
    p.add_argument("--wandb_entity", type=str, default="vhieu4344")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    output_dir, splits, stats = prepare_run(args)
    init_wandb(args, output_dir)

    cn = get_class_names(args.num_classes)
    print(f"Task: {args.num_classes}-class ({', '.join(cn)})", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)

    # Log class distribution
    for split_name, recs in splits.items():
        labels = [get_label(r, args.num_classes) for r in recs]
        dist = {cn[i]: labels.count(i) for i in range(args.num_classes)}
        print(f"  {split_name}: {dist}", flush=True)
    log_wandb(
        {"train_cycles": len(splits["train"]), "val_cycles": len(splits["val"]), "test_cycles": len(splits["test"])},
        "data",
    )

    if args.stage in {"all", "teachers"}:
        train_teachers(args, output_dir, splits, stats, device)
    if args.stage in {"all", "soft-labels"}:
        create_soft_labels(args, output_dir, splits, stats, device)
    if args.stage in {"all", "students"}:
        train_students(args, output_dir, splits, stats, device)
    if args.stage in {"all", "evaluate"}:
        evaluate_outputs(args, output_dir, splits, stats, device)
    finish_wandb()


if __name__ == "__main__":
    main()
