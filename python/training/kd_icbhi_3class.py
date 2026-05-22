#!/usr/bin/env python3
"""ICBHI 2017 3-class teacher-ensemble knowledge distillation pipeline."""

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
from scipy import signal
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
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import ARTIFACTS_DIR, ICBHI_2017_DIR, ICBHI_2017_LABELS, ICBHI_3CLASS_KD_ARTIFACTS_DIR

CLASS_NAMES = ["COPD", "Non-COPD", "Healthy"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {idx: name for name, idx in LABEL_TO_INDEX.items()}


@dataclass(frozen=True)
class CycleRecord:
    sample_id: str
    wav_path: str
    subject_id: str
    diagnosis: str
    label: str
    label_idx: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class FeatureStats:
    mean: float
    std: float


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StudentCNN6(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 16),
            nn.MaxPool2d(2),
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileStyleStudent(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16, stride=2),
            DepthwiseSeparableBlock(16, 24),
            DepthwiseSeparableBlock(24, 32, stride=2),
            DepthwiseSeparableBlock(32, 48),
            DepthwiseSeparableBlock(48, 64, stride=2),
            DepthwiseSeparableBlock(64, 96),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(96, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.skip(x))


class SmallResNetTeacher(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class EfficientCNNTeacher(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            DepthwiseSeparableBlock(32, 48),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(48, 64),
            DepthwiseSeparableBlock(64, 96),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(96, 128),
            DepthwiseSeparableBlock(128, 160),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(160, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.wandb or args.wandb_mode == "disabled":
        return
    if wandb is None:
        raise ImportError("wandb is not installed. Install python/requirements.txt or run without --wandb.")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name,
        dir=str(output_dir),
        mode=args.wandb_mode,
        config=vars(args),
        tags=["icbhi-2017", "3class", "knowledge-distillation", "fpga-student"],
    )


def log_wandb(metrics: dict[str, float | int | str | None], prefix: str) -> None:
    if wandb is None or wandb.run is None:
        return
    payload = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[f"{prefix}/{key}"] = value
    if payload:
        wandb.log(payload)


def finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def map_diagnosis_to_3class(label: str) -> str:
    normalized = label.strip().lower()
    if normalized == "healthy":
        return "Healthy"
    if normalized == "copd":
        return "COPD"
    return "Non-COPD"


def read_subject_labels(labels_file: Path) -> dict[str, str]:
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    labels: dict[str, str] = {}
    with labels_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            labels[parts[0]] = " ".join(parts[1:])
    if not labels:
        raise ValueError(f"No labels parsed from {labels_file}")
    return labels


def find_wavs(data_dir: Path, max_files: int | None = None) -> list[Path]:
    wavs = sorted(data_dir.glob("*.wav"))
    if max_files is not None:
        wavs = wavs[:max_files]
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")
    return wavs


def subject_id_from_wav(wav_path: Path) -> str:
    return wav_path.stem.split("_")[0]


def read_cycle_annotations(wav_path: Path) -> list[tuple[float, float]]:
    annotation_path = wav_path.with_suffix(".txt")
    if not annotation_path.exists():
        return []
    cycles: list[tuple[float, float]] = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue
            if end > start:
                cycles.append((start, end))
    return cycles


def build_records(
    data_dir: Path,
    labels_file: Path,
    max_files: int | None = None,
    max_cycles: int | None = None,
    fallback_window_sec: float = 8.0,
    fallback_hop_sec: float = 4.0,
) -> list[CycleRecord]:
    subject_labels = read_subject_labels(labels_file)
    records: list[CycleRecord] = []
    for wav_path in find_wavs(data_dir, max_files=max_files):
        subject_id = subject_id_from_wav(wav_path)
        if subject_id not in subject_labels:
            continue
        diagnosis = subject_labels[subject_id]
        label = map_diagnosis_to_3class(diagnosis)
        label_idx = LABEL_TO_INDEX[label]
        cycles = read_cycle_annotations(wav_path)
        if not cycles:
            duration = audio_duration_sec(wav_path)
            starts = np.arange(0.0, max(duration - fallback_window_sec, 0.0) + 1e-6, fallback_hop_sec)
            cycles = [(float(start), float(min(start + fallback_window_sec, duration))) for start in starts]
        for cycle_idx, (start_sec, end_sec) in enumerate(cycles):
            sample_id = f"{wav_path.stem}__cycle_{cycle_idx:03d}"
            records.append(
                CycleRecord(
                    sample_id=sample_id,
                    wav_path=str(wav_path),
                    subject_id=subject_id,
                    diagnosis=diagnosis,
                    label=label,
                    label_idx=label_idx,
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                )
            )
            if max_cycles is not None and len(records) >= max_cycles:
                return records
    if not records:
        raise ValueError("No labeled cycles were created. Check data_dir and labels_file.")
    return records


def audio_duration_sec(wav_path: Path) -> float:
    sample_rate, audio = wavfile.read(str(wav_path))
    return float(len(audio) / sample_rate)


def load_audio(wav_path: Path, target_sr: int, bandpass: bool, f_min: float, f_max: float) -> tuple[np.ndarray, int]:
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
        audio = signal.resample_poly(audio, target_sr // gcd, sample_rate // gcd).astype(np.float32)
        sample_rate = target_sr
    if bandpass and len(audio) > 32:
        sos = signal.butter(4, [f_min, min(f_max, sample_rate / 2 - 1)], btype="bandpass", fs=sample_rate, output="sos")
        audio = signal.sosfiltfilt(sos, audio).astype(np.float32)
    return audio, sample_rate


def segment_waveform(audio: np.ndarray, sample_rate: int, start_sec: float, end_sec: float, target_samples: int) -> np.ndarray:
    start_idx = max(0, int(round(start_sec * sample_rate)))
    end_idx = min(len(audio), int(round(end_sec * sample_rate)))
    segment_audio = audio[start_idx:end_idx]
    if len(segment_audio) >= target_samples:
        offset = (len(segment_audio) - target_samples) // 2
        return segment_audio[offset : offset + target_samples].astype(np.float32)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: len(segment_audio)] = segment_audio.astype(np.float32)
    return padded


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for mel_idx in range(n_mels):
        left, center, right = bin_points[mel_idx : mel_idx + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for fft_bin in range(left, min(center, filters.shape[1])):
            filters[mel_idx, fft_bin] = (fft_bin - left) / max(center - left, 1)
        for fft_bin in range(center, min(right, filters.shape[1])):
            filters[mel_idx, fft_bin] = (right - fft_bin) / max(right - center, 1)
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
    _, _, stft = signal.stft(
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
        padded = np.full((logmel.shape[0], target_frames), float(logmel.min()), dtype=np.float32)
        padded[:, : logmel.shape[1]] = logmel
        logmel = padded
    return logmel


def apply_waveform_augmentation(waveform: np.ndarray, max_shift: float, noise_std: float) -> np.ndarray:
    augmented = waveform.copy()
    if max_shift > 0:
        shift = int(np.random.uniform(-max_shift, max_shift) * len(augmented))
        augmented = np.roll(augmented, shift)
    if noise_std > 0:
        rms = float(np.sqrt(np.mean(augmented**2) + 1e-8))
        augmented = augmented + np.random.normal(0.0, noise_std * rms, size=augmented.shape).astype(np.float32)
    return augmented.astype(np.float32)


def apply_specaugment(feature: np.ndarray, freq_mask: int, time_mask: int) -> np.ndarray:
    augmented = feature.copy()
    if freq_mask > 0 and augmented.shape[0] > 1:
        width = np.random.randint(0, min(freq_mask, augmented.shape[0] - 1) + 1)
        start = np.random.randint(0, max(augmented.shape[0] - width, 1))
        augmented[start : start + width, :] = augmented.mean()
    if time_mask > 0 and augmented.shape[1] > 1:
        width = np.random.randint(0, min(time_mask, augmented.shape[1] - 1) + 1)
        start = np.random.randint(0, max(augmented.shape[1] - width, 1))
        augmented[:, start : start + width] = augmented.mean()
    return augmented.astype(np.float32)


class ICBHICycleDataset(Dataset):
    def __init__(
        self,
        records: list[CycleRecord],
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
    ) -> None:
        self.records = records
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
        self.mel_filterbank = build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self._audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        waveform, sample_rate = self._load_cached_audio(record.wav_path)
        segment_audio = segment_waveform(waveform, sample_rate, record.start_sec, record.end_sec, self.target_samples)
        if self.augment:
            segment_audio = apply_waveform_augmentation(segment_audio, self.time_shift, self.noise_std)
        feature = compute_logmel(
            segment_audio,
            sample_rate,
            self.mel_filterbank,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.target_frames,
        )
        if self.augment:
            feature = apply_specaugment(feature, self.freq_mask, self.time_mask)
        if self.stats is not None:
            feature = (feature - self.stats.mean) / max(self.stats.std, 1e-6)
        feature_tensor = torch.from_numpy(feature).unsqueeze(0).float()
        label_tensor = torch.tensor(record.label_idx, dtype=torch.long)
        index_tensor = torch.tensor(index, dtype=torch.long)
        return feature_tensor, label_tensor, index_tensor

    def _load_cached_audio(self, wav_path: str) -> tuple[np.ndarray, int]:
        if wav_path not in self._audio_cache:
            self._audio_cache[wav_path] = load_audio(Path(wav_path), self.sample_rate, self.bandpass, self.f_min, self.f_max)
        return self._audio_cache[wav_path]


def create_dataset(records: list[CycleRecord], args: argparse.Namespace, stats: FeatureStats | None, augment: bool) -> ICBHICycleDataset:
    return ICBHICycleDataset(
        records=records,
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
    )


def estimate_feature_stats(records: list[CycleRecord], args: argparse.Namespace) -> FeatureStats:
    dataset = create_dataset(records, args, stats=None, augment=False)
    limit = min(len(dataset), args.max_stat_samples)
    values: list[torch.Tensor] = []
    for idx in range(limit):
        feature, _, _ = dataset[idx]
        values.append(feature)
    stacked = torch.stack(values)
    return FeatureStats(mean=float(stacked.mean().item()), std=float(stacked.std().item()))


def create_subject_splits(records: list[CycleRecord], test_size: float, val_size: float, seed: int) -> dict[str, list[CycleRecord]]:
    subject_to_records: dict[str, list[CycleRecord]] = {}
    subject_to_label: dict[str, int] = {}
    for record in records:
        subject_to_records.setdefault(record.subject_id, []).append(record)
        subject_to_label[record.subject_id] = record.label_idx
    subjects = sorted(subject_to_records)
    labels = [subject_to_label[subject] for subject in subjects]
    train_val_subjects, test_subjects = stratified_split(subjects, labels, test_size, seed)
    train_val_labels = [subject_to_label[subject] for subject in train_val_subjects]
    adjusted_val_size = val_size / max(1.0 - test_size, 1e-6)
    train_subjects, val_subjects = stratified_split(train_val_subjects, train_val_labels, adjusted_val_size, seed + 1)
    return {
        "train": flatten_subject_records(train_subjects, subject_to_records),
        "val": flatten_subject_records(val_subjects, subject_to_records),
        "test": flatten_subject_records(test_subjects, subject_to_records),
    }


def stratified_split(subjects: list[str], labels: list[int], test_size: float, seed: int) -> tuple[list[str], list[str]]:
    if len(subjects) < 3:
        return subjects, []
    counts = {label: labels.count(label) for label in set(labels)}
    stratify = labels if min(counts.values()) >= 2 and len(counts) > 1 else None
    try:
        left, right = train_test_split(subjects, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        left, right = train_test_split(subjects, test_size=test_size, random_state=seed, shuffle=True)
    return sorted(left), sorted(right)


def flatten_subject_records(subjects: Iterable[str], subject_to_records: dict[str, list[CycleRecord]]) -> list[CycleRecord]:
    records: list[CycleRecord] = []
    for subject in subjects:
        records.extend(subject_to_records[subject])
    return records


def save_records(path: Path, records: list[CycleRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, indent=2)


def save_split_records(output_dir: Path, splits: dict[str, list[CycleRecord]]) -> None:
    serializable = {split: [asdict(record) for record in records] for split, records in splits.items()}
    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def load_split_records(output_dir: Path) -> dict[str, list[CycleRecord]]:
    with (output_dir / "splits.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {split: [CycleRecord(**record) for record in records] for split, records in raw.items()}


def save_config(output_dir: Path, args: argparse.Namespace, stats: FeatureStats, records: list[CycleRecord]) -> None:
    label_counts = {name: 0 for name in CLASS_NAMES}
    diagnosis_counts: dict[str, int] = {}
    for record in records:
        label_counts[record.label] += 1
        diagnosis_counts[record.diagnosis] = diagnosis_counts.get(record.diagnosis, 0) + 1
    config = vars(args).copy()
    config["feature_mean"] = stats.mean
    config["feature_std"] = stats.std
    config["class_names"] = CLASS_NAMES
    config["label_counts"] = label_counts
    config["diagnosis_counts"] = diagnosis_counts
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (output_dir / "labels_map.json").open("w", encoding="utf-8") as handle:
        json.dump({"class_names": CLASS_NAMES, "label_to_index": LABEL_TO_INDEX}, handle, indent=2)


def load_feature_stats(output_dir: Path) -> FeatureStats:
    with (output_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return FeatureStats(mean=float(config["feature_mean"]), std=float(config["feature_std"]))


def class_weights(records: list[CycleRecord], device: torch.device) -> torch.Tensor:
    counts = np.bincount([record.label_idx for record in records], minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts * len(CLASS_NAMES), 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_model(model_name: str, num_classes: int = 3) -> nn.Module:
    if model_name == "cnn6":
        return StudentCNN6(num_classes=num_classes)
    if model_name == "mobilestyle":
        return MobileStyleStudent(num_classes=num_classes)
    if model_name == "small_resnet":
        return SmallResNetTeacher(num_classes=num_classes)
    if model_name == "efficient_cnn":
        return EfficientCNNTeacher(num_classes=num_classes)
    raise ValueError(f"Unknown model architecture: {model_name}")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def train_supervised_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_records: list[CycleRecord],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    lr: float,
) -> dict[str, float | int | str | None]:
    ensure_dir(output_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_records, device))
    best_metric = -float("inf")
    best_epoch = 0
    best_metrics: dict[str, float | int | str | None] = {}
    patience_counter = 0
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels, _ in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * features.size(0)
        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device, output_dir=None)
        score = float(val_metrics[args.selection_metric])
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": val_metrics}, output_dir / "best.pt")
        else:
            patience_counter += 1
        avg_loss = total_loss / max(len(train_loader.dataset), 1)
        print(
            f"epoch={epoch:03d} loss={avg_loss:.4f} "
            f"val_{args.selection_metric}={score:.4f} best={best_metric:.4f}",
            flush=True,
        )
        log_wandb(
            {"epoch": epoch, "train_loss": avg_loss, **{f"val_{key}": value for key, value in val_metrics.items()}},
            f"{output_dir.parent.name}/{output_dir.name}",
        )
        if patience_counter >= args.patience:
            break
    best_metrics["best_epoch"] = best_epoch
    return best_metrics


def train_student_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_records: list[CycleRecord],
    teacher_logits: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    kd_mode: str,
) -> dict[str, float | int | str | None]:
    ensure_dir(output_dir)
    teacher_logits_tensor = torch.tensor(teacher_logits, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_student, 1))
    hard_criterion = nn.CrossEntropyLoss(weight=class_weights(train_records, device))
    best_metric = -float("inf")
    best_epoch = 0
    best_metrics: dict[str, float | int | str | None] = {}
    patience_counter = 0
    model.to(device)
    for epoch in range(1, args.epochs_student + 1):
        model.train()
        total_loss = 0.0
        for features, labels, indices in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(features)
            target_logits = select_teacher_logits(teacher_logits_tensor, indices, kd_mode)
            loss = kd_loss(student_logits, target_logits, labels, hard_criterion, args)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * features.size(0)
        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device, output_dir=None)
        score = float(val_metrics[args.selection_metric])
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": val_metrics}, output_dir / "best.pt")
        else:
            patience_counter += 1
        avg_loss = total_loss / max(len(train_loader.dataset), 1)
        print(
            f"student={kd_mode} epoch={epoch:03d} loss={avg_loss:.4f} "
            f"val_{args.selection_metric}={score:.4f} best={best_metric:.4f}",
            flush=True,
        )
        log_wandb(
            {"epoch": epoch, "train_loss": avg_loss, **{f"val_{key}": value for key, value in val_metrics.items()}},
            f"students/{kd_mode}_teacher",
        )
        if patience_counter >= args.patience:
            break
    best_metrics["best_epoch"] = best_epoch
    best_metrics["kd_mode"] = kd_mode
    return best_metrics


def select_teacher_logits(teacher_logits: torch.Tensor, indices: torch.Tensor, kd_mode: str) -> torch.Tensor:
    selected = teacher_logits[:, indices, :]
    if kd_mode == "mean":
        return selected.mean(dim=0)
    if kd_mode == "random":
        teacher_ids = torch.randint(0, selected.size(0), (selected.size(1),), device=selected.device)
        sample_ids = torch.arange(selected.size(1), device=selected.device)
        return selected[teacher_ids, sample_ids, :]
    raise ValueError(f"Unsupported kd_mode: {kd_mode}")


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    hard_criterion: nn.Module,
    args: argparse.Namespace,
) -> torch.Tensor:
    temperature = args.temperature
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = -(teacher_probs * student_log_probs).sum(dim=1).mean() * (temperature**2)
    if args.kd_loss == "soft_only":
        return soft_loss
    hard_loss = hard_criterion(student_logits, hard_labels)
    return args.alpha * soft_loss + (1.0 - args.alpha) * hard_loss


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path | None,
    name: str = "metrics",
) -> dict[str, float | int | str | None]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    with torch.no_grad():
        for features, labels, _ in loader:
            logits = model(features.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_pred.extend(probs.argmax(axis=1).tolist())
            y_true.extend(labels.numpy().tolist())
    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    if output_dir is not None:
        save_metrics(output_dir, name, metrics, np.array(y_true), np.array(y_pred))
        log_wandb(metrics, f"eval/{name}")
    return metrics


def evaluate_ensemble(
    teacher_logits: np.ndarray,
    records: list[CycleRecord],
    output_dir: Path | None,
    name: str,
    temperature: float = 1.0,
) -> dict[str, float | int | str | None]:
    mean_logits = teacher_logits.mean(axis=0)
    probs = softmax_np(mean_logits / temperature, axis=1)
    y_true = np.array([record.label_idx for record in records])
    y_pred = probs.argmax(axis=1)
    metrics = compute_metrics(y_true, y_pred, probs)
    if output_dir is not None:
        save_metrics(output_dir, name, metrics, y_true, y_pred)
        log_wandb(metrics, f"eval/{name}")
    return metrics


def softmax_np(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int | str | None]:
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    specificity = per_class_specificity(cm)
    metrics: dict[str, float | int | str | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "icbhi_3class_score": float(icbhi_3class_score(y_true, y_pred)),
        "auc_ovr": safe_auc(y_true, y_prob),
    }
    for idx, class_name in enumerate(CLASS_NAMES):
        key = class_name.lower().replace("-", "_")
        metrics[f"{key}_precision"] = float(precision[idx])
        metrics[f"{key}_recall"] = float(recall[idx])
        metrics[f"{key}_f1"] = float(f1[idx])
        metrics[f"{key}_sensitivity"] = float(recall[idx])
        metrics[f"{key}_specificity"] = float(specificity[idx])
        metrics[f"{key}_support"] = int(support[idx])
    return metrics


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


def icbhi_3class_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    healthy_idx = LABEL_TO_INDEX["Healthy"]
    healthy_mask = y_true == healthy_idx
    abnormal_mask = y_true != healthy_idx
    healthy_specificity = float(np.mean(y_pred[healthy_mask] == healthy_idx)) if healthy_mask.any() else 0.0
    abnormal_sensitivity = float(np.mean(y_pred[abnormal_mask] != healthy_idx)) if abnormal_mask.any() else 0.0
    return (healthy_specificity + abnormal_sensitivity) / 2.0


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < len(CLASS_NAMES):
            return None
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(len(CLASS_NAMES)))))
    except ValueError:
        return None


def save_metrics(output_dir: Path, name: str, metrics: dict[str, float | int | str | None], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    metrics_dir = ensure_dir(output_dir / "metrics")
    with (metrics_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    with (metrics_dir / f"confusion_matrix_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for idx, row in enumerate(cm):
            writer.writerow([CLASS_NAMES[idx], *row.tolist()])


def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    logits_list: list[np.ndarray] = []
    with torch.no_grad():
        for features, _, _ in loader:
            logits_list.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def teacher_checkpoint_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "teachers" / f"seed_{seed}" / "best.pt"


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def prepare_run(args: argparse.Namespace) -> tuple[Path, dict[str, list[CycleRecord]], FeatureStats]:
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else Path(args.artifact_root) / "training" / "icbhi_3class_kd")
    split_path = output_dir / "splits.json"
    if split_path.exists() and (output_dir / "config.json").exists():
        splits = load_split_records(output_dir)
        stats = load_feature_stats(output_dir)
        return output_dir, splits, stats
    records = build_records(
        Path(args.data_dir),
        Path(args.labels_file),
        max_files=args.max_files,
        max_cycles=args.max_cycles,
        fallback_window_sec=args.duration_sec,
        fallback_hop_sec=args.fallback_hop_sec,
    )
    splits = create_subject_splits(records, args.test_size, args.val_size, args.seed)
    stats = estimate_feature_stats(splits["train"], args)
    save_split_records(output_dir, splits)
    save_config(output_dir, args, stats, records)
    return output_dir, splits, stats


def train_teachers(args: argparse.Namespace, output_dir: Path, splits: dict[str, list[CycleRecord]], stats: FeatureStats, device: torch.device) -> None:
    train_dataset = create_dataset(splits["train"], args, stats, augment=True)
    val_dataset = create_dataset(splits["val"], args, stats, augment=False)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    seeds = parse_teacher_seeds(args.teacher_seeds, args.num_teachers)
    for seed in seeds:
        set_seed(seed)
        model = make_model(args.teacher_arch, num_classes=len(CLASS_NAMES))
        teacher_dir = output_dir / "teachers" / f"seed_{seed}"
        if args.skip_existing and (teacher_dir / "best.pt").exists():
            continue
        print(f"Training teacher seed={seed} arch={args.teacher_arch}", flush=True)
        best_metrics = train_supervised_model(
            model,
            train_loader,
            val_loader,
            splits["train"],
            args,
            device,
            teacher_dir,
            epochs=args.epochs_teacher,
            lr=args.lr_teacher,
        )
        with (teacher_dir / "val_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(best_metrics, handle, indent=2)


def create_soft_labels(args: argparse.Namespace, output_dir: Path, splits: dict[str, list[CycleRecord]], stats: FeatureStats, device: torch.device) -> None:
    soft_dir = ensure_dir(output_dir / "soft_labels")
    seeds = parse_teacher_seeds(args.teacher_seeds, args.num_teachers)
    for split_name, records in splits.items():
        dataset = create_dataset(records, args, stats, augment=False)
        loader = make_loader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        all_logits = []
        for seed in seeds:
            checkpoint_path = teacher_checkpoint_path(output_dir, seed)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing teacher checkpoint: {checkpoint_path}")
            model = load_model_checkpoint(make_model(args.teacher_arch, len(CLASS_NAMES)), checkpoint_path, device)
            all_logits.append(collect_logits(model, loader, device))
        stacked = np.stack(all_logits, axis=0)
        np.save(soft_dir / f"teacher_logits_{split_name}.npy", stacked)
        with (soft_dir / f"sample_ids_{split_name}.json").open("w", encoding="utf-8") as handle:
            json.dump([record.sample_id for record in records], handle, indent=2)
    metrics_dir = ensure_dir(output_dir / "metrics")
    val_logits = np.load(soft_dir / "teacher_logits_val.npy")
    test_logits = np.load(soft_dir / "teacher_logits_test.npy")
    val_metrics = evaluate_ensemble(val_logits, splits["val"], output_dir, "teacher_ensemble_val", args.temperature)
    test_metrics = evaluate_ensemble(test_logits, splits["test"], output_dir, "teacher_ensemble_test", args.temperature)
    with (metrics_dir / "teacher_ensemble_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"val": val_metrics, "test": test_metrics}, handle, indent=2)


def train_students(args: argparse.Namespace, output_dir: Path, splits: dict[str, list[CycleRecord]], stats: FeatureStats, device: torch.device) -> None:
    soft_dir = output_dir / "soft_labels"
    train_logits_path = soft_dir / "teacher_logits_train.npy"
    if not train_logits_path.exists():
        create_soft_labels(args, output_dir, splits, stats, device)
    teacher_logits = np.load(train_logits_path)
    train_dataset = create_dataset(splits["train"], args, stats, augment=True)
    val_dataset = create_dataset(splits["val"], args, stats, augment=False)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    kd_modes = [args.kd_mode] if args.kd_mode != "both" else ["mean", "random"]
    for kd_mode in kd_modes:
        set_seed(args.seed + 1000 + len(kd_mode))
        model = make_model(args.student_arch, len(CLASS_NAMES))
        student_dir = output_dir / "students" / f"{kd_mode}_teacher"
        if args.skip_existing and (student_dir / "best.pt").exists():
            continue
        print(f"Training student arch={args.student_arch} kd_mode={kd_mode} kd_loss={args.kd_loss}", flush=True)
        best_metrics = train_student_model(model, train_loader, val_loader, splits["train"], teacher_logits, args, device, student_dir, kd_mode)
        with (student_dir / "val_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(best_metrics, handle, indent=2)


def evaluate_outputs(args: argparse.Namespace, output_dir: Path, splits: dict[str, list[CycleRecord]], stats: FeatureStats, device: torch.device) -> None:
    soft_dir = output_dir / "soft_labels"
    if (soft_dir / "teacher_logits_test.npy").exists():
        test_logits = np.load(soft_dir / "teacher_logits_test.npy")
        evaluate_ensemble(test_logits, splits["test"], output_dir, "teacher_ensemble_test", args.temperature)
    test_dataset = create_dataset(splits["test"], args, stats, augment=False)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    for kd_mode in ["mean", "random"]:
        checkpoint_path = output_dir / "students" / f"{kd_mode}_teacher" / "best.pt"
        if not checkpoint_path.exists():
            continue
        model = load_model_checkpoint(make_model(args.student_arch, len(CLASS_NAMES)), checkpoint_path, device)
        evaluate_model(model, test_loader, device, output_dir, f"student_{kd_mode}_test")


def parse_teacher_seeds(seed_arg: str, num_teachers: int) -> list[int]:
    seeds = [int(seed.strip()) for seed in seed_arg.split(",") if seed.strip()]
    if len(seeds) < num_teachers:
        start = max(seeds) + 1 if seeds else 1
        seeds.extend(range(start, start + num_teachers - len(seeds)))
    return seeds[:num_teachers]


def default_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICBHI 2017 3-class teacher-ensemble KD pipeline")
    parser.add_argument("--data_dir", type=str, default=str(ICBHI_2017_DIR))
    parser.add_argument("--labels_file", type=str, default=str(ICBHI_2017_LABELS))
    parser.add_argument("--artifact_root", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--output_dir", type=str, default=None, help=f"Default: {ICBHI_3CLASS_KD_ARTIFACTS_DIR}")
    parser.add_argument("--stage", choices=["all", "teachers", "soft-labels", "students", "evaluate"], default="all")
    parser.add_argument("--teacher_arch", choices=["small_resnet", "efficient_cnn"], default="small_resnet")
    parser.add_argument("--student_arch", choices=["cnn6", "mobilestyle"], default="cnn6")
    parser.add_argument("--num_teachers", type=int, default=5)
    parser.add_argument("--teacher_seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--kd_mode", choices=["mean", "random", "both"], default="both")
    parser.add_argument("--kd_loss", choices=["soft_only", "mixed"], default="soft_only")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=8.0)
    parser.add_argument("--fallback_hop_sec", type=float, default=4.0)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--win_length", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--target_frames", type=int, default=800)
    parser.add_argument("--f_min", type=float, default=50.0)
    parser.add_argument("--f_max", type=float, default=2500.0)
    parser.add_argument("--no_bandpass", action="store_true")
    parser.add_argument("--time_shift", type=float, default=0.05)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--freq_mask", type=int, default=8)
    parser.add_argument("--time_mask", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_teacher", type=int, default=100)
    parser.add_argument("--epochs_student", type=int, default=100)
    parser.add_argument("--lr_teacher", type=float, default=1e-3)
    parser.add_argument("--lr_student", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--selection_metric", choices=["macro_f1", "balanced_accuracy", "icbhi_3class_score"], default="macro_f1")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_cycles", type=int, default=None)
    parser.add_argument("--max_stat_samples", type=int, default=512)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log losses and metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="icbhi-3class-kd")
    parser.add_argument("--wandb_entity", type=str, default="vhieu4344")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    output_dir, splits, stats = prepare_run(args)
    init_wandb(args, output_dir)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Split sizes: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)
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
