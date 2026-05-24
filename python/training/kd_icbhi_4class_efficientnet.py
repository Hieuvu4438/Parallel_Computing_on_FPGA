#!/usr/bin/env python3
"""ICBHI 2017 3-class disease-level EfficientNet-B0 training/KD pipeline."""

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
import torchvision.models as models
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import (
    ARTIFACTS_DIR,
    ICBHI_2017_DIR,
    ICBHI_2017_LABELS,
    ICBHI_3CLASS_EFFICIENTNET_KD_ARTIFACTS_DIR,
    ensure_dir,
)

CLASS_NAMES = ["COPD", "Non-COPD", "Healthy"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
HEALTHY_INDEX = LABEL_TO_INDEX["Healthy"]

DISEASE_TO_CLASS = {
    "COPD": "COPD",
    "Healthy": "Healthy",
    "URTI": "Non-COPD",
    "Asthma": "Non-COPD",
    "LRTI": "Non-COPD",
    "Bronchiectasis": "Non-COPD",
    "Bronchiolitis": "Non-COPD",
    "Pneumonia": "Non-COPD",
}


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


class ICBHIRespiratoryDataset(Dataset):
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
        input_channels: int,
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
        self.input_channels = input_channels
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
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
        if self.input_channels == 3:
            feature_tensor = feature_tensor.repeat(3, 1, 1)
        return feature_tensor, torch.tensor(record.label_idx, dtype=torch.long), record.sample_id

    def _load_cached_audio(self, wav_path: str) -> tuple[np.ndarray, int]:
        if wav_path not in self._audio_cache:
            self._audio_cache[wav_path] = load_audio(Path(wav_path), self.sample_rate, self.bandpass, self.f_min, self.f_max)
        return self._audio_cache[wav_path]


class TeacherEnsemble:
    def __init__(
        self,
        logits: np.ndarray | torch.Tensor,
        sample_ids: list[str],
        num_classes: int = NUM_CLASSES,
        device: torch.device | str = "cpu",
    ) -> None:
        logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
        if logits_tensor.ndim != 3:
            raise ValueError(f"Teacher logits must be 3D, got shape {tuple(logits_tensor.shape)}")
        if logits_tensor.shape[-1] != num_classes:
            raise ValueError(f"Teacher logits last dim must be {num_classes}, got {logits_tensor.shape[-1]}")
        if logits_tensor.shape[0] == len(sample_ids):
            normalized = logits_tensor
        elif logits_tensor.shape[1] == len(sample_ids):
            normalized = logits_tensor.permute(1, 0, 2).contiguous()
        else:
            raise ValueError(
                "Teacher logits must be [num_samples, k, num_classes] or [k, num_samples, num_classes]; "
                f"got {tuple(logits_tensor.shape)} for {len(sample_ids)} sample ids"
            )
        self.logits = normalized.to(device)
        self.sample_to_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
        self.device = torch.device(device)

    @property
    def num_teachers(self) -> int:
        return int(self.logits.shape[1])

    def get_soft_labels_mean(self, sample_ids: Iterable[str]) -> torch.Tensor:
        selected = self.logits[self._indices(sample_ids)]
        return F.softmax(selected.mean(dim=1), dim=-1)

    def get_soft_labels_random(self, sample_ids: Iterable[str]) -> torch.Tensor:
        selected = self.logits[self._indices(sample_ids)]
        teacher_ids = torch.randint(0, selected.size(1), (selected.size(0),), device=self.device)
        batch_ids = torch.arange(selected.size(0), device=self.device)
        return F.softmax(selected[batch_ids, teacher_ids, :], dim=-1)

    def _indices(self, sample_ids: Iterable[str]) -> torch.Tensor:
        indices: list[int] = []
        for sample_id in sample_ids:
            if sample_id not in self.sample_to_index:
                raise KeyError(f"Sample id {sample_id!r} is missing from teacher logits mapping")
            indices.append(self.sample_to_index[sample_id])
        return torch.tensor(indices, dtype=torch.long, device=self.device)


class EfficientNetB0Student(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3, pretrained: bool = True, input_channels: int = 3) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        if input_channels == 1:
            conv = self.backbone.features[0][0]
            replacement = nn.Conv2d(
                1,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None,
            )
            with torch.no_grad():
                replacement.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
                if conv.bias is not None and replacement.bias is not None:
                    replacement.bias.copy_(conv.bias)
            self.backbone.features[0][0] = replacement
        elif input_channels != 3:
            raise ValueError("EfficientNetB0Student supports input_channels 1 or 3")
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(num_features, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
        raise ValueError(f"No subject labels parsed from {labels_file}")
    return labels


def map_diagnosis_to_3class(diagnosis: str) -> str:
    normalized = diagnosis.strip()
    if normalized in DISEASE_TO_CLASS:
        return DISEASE_TO_CLASS[normalized]
    if normalized.lower() == "healthy":
        return "Healthy"
    if normalized.lower() == "copd":
        return "COPD"
    return "Non-COPD"


def subject_id_from_wav(wav_path: Path) -> str:
    return wav_path.stem.split("_")[0]


def find_wavs(data_dir: Path, max_files: int | None = None) -> list[Path]:
    wavs = sorted(data_dir.glob("*.wav"))
    if max_files is not None:
        wavs = wavs[:max_files]
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")
    return wavs


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


def build_records(data_dir: Path, labels_file: Path, max_files: int | None = None, max_cycles: int | None = None) -> list[CycleRecord]:
    subject_labels = read_subject_labels(labels_file)
    records: list[CycleRecord] = []
    for wav_path in find_wavs(data_dir, max_files=max_files):
        subject_id = subject_id_from_wav(wav_path)
        if subject_id not in subject_labels:
            continue
        diagnosis = subject_labels[subject_id]
        label = map_diagnosis_to_3class(diagnosis)
        label_idx = LABEL_TO_INDEX[label]
        for cycle_idx, (start_sec, end_sec) in enumerate(read_cycle_annotations(wav_path)):
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
        raise ValueError("No labeled ICBHI breathing cycles were parsed. Check data_dir, labels_file, and annotation files.")
    return records


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
        high = min(f_max, sample_rate / 2 - 1)
        if high > f_min:
            sos = signal.butter(4, [f_min, high], btype="bandpass", fs=sample_rate, output="sos")
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
    f_max = min(f_max, sample_rate / 2 - 1)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)
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
    fill_value = float(augmented.mean())
    if freq_mask > 0 and augmented.shape[0] > 1:
        width = np.random.randint(0, min(freq_mask, augmented.shape[0] - 1) + 1)
        start = np.random.randint(0, max(augmented.shape[0] - width, 1))
        augmented[start : start + width, :] = fill_value
    if time_mask > 0 and augmented.shape[1] > 1:
        width = np.random.randint(0, min(time_mask, augmented.shape[1] - 1) + 1)
        start = np.random.randint(0, max(augmented.shape[1] - width, 1))
        augmented[:, start : start + width] = fill_value
    return augmented.astype(np.float32)


def create_dataset(records: list[CycleRecord], args: argparse.Namespace, stats: FeatureStats | None, augment: bool) -> ICBHIRespiratoryDataset:
    return ICBHIRespiratoryDataset(
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
        input_channels=args.input_channels,
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
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for idx in range(limit):
        feature, _, _ = dataset[idx]
        values = feature.float()
        total_sum += float(values.sum().item())
        total_sq_sum += float((values * values).sum().item())
        total_count += values.numel()
    if total_count == 0:
        raise ValueError("Cannot estimate feature statistics from an empty training set")
    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 1e-12)
    return FeatureStats(mean=float(mean), std=float(math.sqrt(variance)))


def flatten_subject_records(subjects: Iterable[str], subject_to_records: dict[str, list[CycleRecord]]) -> list[CycleRecord]:
    records: list[CycleRecord] = []
    for subject in subjects:
        records.extend(subject_to_records[subject])
    return records


def stratified_subject_split(subjects: list[str], labels: list[int], test_size: float, seed: int) -> tuple[list[str], list[str]]:
    if test_size <= 0 or len(subjects) < 2:
        return sorted(subjects), []
    counts = {label: labels.count(label) for label in set(labels)}
    stratify = labels if len(counts) > 1 and min(counts.values()) >= 2 else None
    try:
        left, right = train_test_split(subjects, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        right_count = min(max(1, int(round(len(subjects) * test_size))), len(subjects) - 1)
        shuffled = sorted(subjects)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        right = shuffled[:right_count]
        left = shuffled[right_count:]
    return sorted(left), sorted(right)


def create_patientwise_splits(records: list[CycleRecord], test_size: float, val_size: float, seed: int) -> dict[str, list[CycleRecord]]:
    subject_to_records: dict[str, list[CycleRecord]] = {}
    subject_to_label: dict[str, int] = {}
    for record in records:
        subject_to_records.setdefault(record.subject_id, []).append(record)
        subject_to_label[record.subject_id] = record.label_idx
    subjects = sorted(subject_to_records)
    labels = [subject_to_label[subject] for subject in subjects]
    train_val_subjects, test_subjects = stratified_subject_split(subjects, labels, test_size, seed)
    train_val_labels = [subject_to_label[subject] for subject in train_val_subjects]
    train_subjects, val_subjects = stratified_subject_split(train_val_subjects, train_val_labels, val_size, seed + 1)
    return {
        "train": flatten_subject_records(train_subjects, subject_to_records),
        "val": flatten_subject_records(val_subjects, subject_to_records),
        "test": flatten_subject_records(test_subjects, subject_to_records),
    }


def save_split_records(output_dir: Path, splits: dict[str, list[CycleRecord]]) -> None:
    serializable = {split: [asdict(record) for record in records] for split, records in splits.items()}
    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def load_split_records(output_dir: Path) -> dict[str, list[CycleRecord]]:
    with (output_dir / "splits.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {split: [CycleRecord(**record) for record in records] for split, records in raw.items()}


def save_config(output_dir: Path, args: argparse.Namespace, stats: FeatureStats, records: list[CycleRecord], splits: dict[str, list[CycleRecord]]) -> None:
    label_counts = {name: 0 for name in CLASS_NAMES}
    diagnosis_counts: dict[str, int] = {}
    for record in records:
        label_counts[record.label] += 1
        diagnosis_counts[record.diagnosis] = diagnosis_counts.get(record.diagnosis, 0) + 1
    config = vars(args).copy()
    config["feature_mean"] = stats.mean
    config["feature_std"] = stats.std
    config["class_names"] = CLASS_NAMES
    config["label_to_index"] = LABEL_TO_INDEX
    config["disease_to_class"] = DISEASE_TO_CLASS
    config["label_counts"] = label_counts
    config["diagnosis_counts"] = diagnosis_counts
    config["split_sizes"] = {split: len(split_records) for split, split_records in splits.items()}
    config["unique_patients"] = len({record.subject_id for record in records})
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (output_dir / "labels_map.json").open("w", encoding="utf-8") as handle:
        json.dump({"class_names": CLASS_NAMES, "label_to_index": LABEL_TO_INDEX, "disease_to_class": DISEASE_TO_CLASS}, handle, indent=2)


def load_feature_stats(output_dir: Path) -> FeatureStats:
    with (output_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return FeatureStats(mean=float(config["feature_mean"]), std=float(config["feature_std"]))


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def load_logits_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("logits", "teacher_logits"):
                if key in payload:
                    return np.asarray(payload[key])
        return np.asarray(payload)
    raise ValueError(f"Unsupported logits format: {path}. Use .npy, .pt, or .pth")


def load_sample_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("sample_ids", "ids"):
                if key in payload and isinstance(payload[key], list):
                    return [str(item) for item in payload[key]]
        raise ValueError(f"Cannot parse sample ids from JSON file: {path}")
    sample_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                sample_ids.append(str(row[0]).strip())
    return sample_ids


def make_teacher_ensemble(records: list[CycleRecord], args: argparse.Namespace, device: torch.device) -> TeacherEnsemble:
    if not args.teacher_logits_train:
        raise ValueError("KD modes require --teacher_logits_train. Use --loss_mode supervised to train from hard labels.")
    sample_ids = [record.sample_id for record in records]
    logits = load_logits_file(Path(args.teacher_logits_train))
    if args.teacher_sample_ids_train:
        sample_ids = load_sample_ids(Path(args.teacher_sample_ids_train))
    return TeacherEnsemble(logits, sample_ids, device=device)


def soft_label_cross_entropy(student_logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    return -(soft_labels * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0,
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


def icbhi_disease_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    healthy_mask = y_true == HEALTHY_INDEX
    abnormal_mask = y_true != HEALTHY_INDEX
    specificity = float(np.mean(y_pred[healthy_mask] == HEALTHY_INDEX)) if healthy_mask.any() else 0.0
    sensitivity = float(np.mean(y_pred[abnormal_mask] != HEALTHY_INDEX)) if abnormal_mask.any() else 0.0
    return sensitivity, specificity, (sensitivity + specificity) / 2.0


def metric_key(class_name: str) -> str:
    return class_name.lower().replace("-", "_").replace(" ", "_")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | list[list[int]]]:
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    specificity = per_class_specificity(cm)
    icbhi_se, icbhi_sp, icbhi_score = icbhi_disease_score(y_true, y_pred)
    metrics: dict[str, float | int | list[list[int]]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(icbhi_se),
        "specificity": float(icbhi_sp),
        "icbhi_score": float(icbhi_score),
        "confusion_matrix": cm.tolist(),
    }
    for idx, class_name in enumerate(CLASS_NAMES):
        key = metric_key(class_name)
        metrics[f"{key}_precision"] = float(precision[idx])
        metrics[f"{key}_recall"] = float(recall[idx])
        metrics[f"{key}_f1"] = float(f1[idx])
        metrics[f"{key}_sensitivity"] = float(recall[idx])
        metrics[f"{key}_specificity"] = float(specificity[idx])
        metrics[f"{key}_support"] = int(support[idx])
    return metrics


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float | int | list[list[int]]], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for features, labels, _ in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * features.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)
    metrics = compute_metrics(y_true_arr, y_pred_arr)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics, y_true_arr, y_pred_arr


def save_metrics(metrics_dir: Path, name: str, metrics: dict[str, float | int | list[list[int]]], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    ensure_dir(metrics_dir)
    with (metrics_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    with (metrics_dir / f"confusion_matrix_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for idx, row in enumerate(cm):
            writer.writerow([CLASS_NAMES[idx], *row.tolist()])


def scalar_metrics(metrics: dict[str, float | int | list[list[int]]], prefix: str) -> dict[str, float | int]:
    return {f"{prefix}/{key}": value for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def init_wandb(
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
    teacher: TeacherEnsemble | None,
    param_count: tuple[int, int],
) -> None:
    if not args.wandb or args.wandb_mode == "disabled":
        return
    if wandb is None:
        raise ImportError("wandb is not installed. Install python/requirements.txt or run with --wandb_mode disabled.")
    run_name = args.wandb_run_name or f"icbhi3-effb0-{args.loss_mode}-{args.kd_mode}-seed-{seed}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        dir=str(output_dir),
        mode=args.wandb_mode,
        config={
            **vars(args),
            "seed": seed,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "num_teachers": teacher.num_teachers if teacher is not None else 0,
            "model_total_params": param_count[0],
            "model_trainable_params": param_count[1],
        },
        tags=["icbhi-2017", "3class-disease", "efficientnet-b0", args.loss_mode],
        reinit=True,
    )


def finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def log_wandb(payload: dict[str, float | int], step: int | None = None) -> None:
    if wandb is not None and wandb.run is not None and payload:
        wandb.log(payload, step=step)


def log_wandb_confusion(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if wandb is None or wandb.run is None:
        return
    wandb.log({name: wandb.plot.confusion_matrix(probs=None, y_true=y_true.tolist(), preds=y_pred.tolist(), class_names=CLASS_NAMES)})


def log_checkpoint_artifact(path: Path, seed: int, args: argparse.Namespace) -> None:
    if wandb is None or wandb.run is None:
        return
    artifact = wandb.Artifact(f"icbhi3-effb0-{args.loss_mode}-seed-{seed}", type="model", metadata={"seed": seed, "loss_mode": args.loss_mode})
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def train_one_seed(
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
    splits: dict[str, list[CycleRecord]],
    stats: FeatureStats,
    device: torch.device,
) -> dict[str, float | int | str | None]:
    set_seed(seed)
    train_dataset = create_dataset(splits["train"], args, stats, augment=True)
    val_dataset = create_dataset(splits["val"], args, stats, augment=False)
    test_dataset = create_dataset(splits["test"], args, stats, augment=False)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    teacher = make_teacher_ensemble(splits["train"], args, device) if args.loss_mode in {"kd", "mixed"} else None

    # Calculate class-balanced alpha for focal loss
    train_records = splits["train"]
    counts = {name: 0 for name in CLASS_NAMES}
    for r in train_records:
        counts[r.label] += 1
    class_counts = torch.tensor([counts[name] for name in CLASS_NAMES], dtype=torch.float32)
    beta = 0.999
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-8)
    alpha = weights / weights.sum() * NUM_CLASSES
    hard_criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma).to(device)
    print(f"seed={seed} class_counts={counts} focal_alpha_weights={alpha.tolist()}", flush=True)

    model = EfficientNetB0Student(
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
        input_channels=args.input_channels,
    ).to(device)
    param_count = count_parameters(model)
    init_wandb(args, output_dir, seed, teacher, param_count)
    print(f"seed={seed} total_params={param_count[0]:,} trainable_params={param_count[1]:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    best_path = checkpoints_dir / f"seed_{seed}_best.pt"
    best_score = -float("inf")
    best_epoch = 0
    best_val_metrics: dict[str, float | int | list[list[int]]] = {}
    best_val_true = np.array([], dtype=np.int64)
    best_val_pred = np.array([], dtype=np.int64)
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_soft_loss = 0.0
        total_hard_loss = 0.0
        soft_label_sum = torch.zeros(NUM_CLASSES, device=device)
        soft_label_count = 0
        for features, labels, sample_ids in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            hard_loss = hard_criterion(logits, labels)
            soft_loss = torch.zeros((), device=device)
            if teacher is None:
                loss = hard_loss
            else:
                soft_labels = teacher.get_soft_labels_mean(sample_ids) if args.kd_mode == "mean_teacher" else teacher.get_soft_labels_random(sample_ids)
                soft_loss = soft_label_cross_entropy(logits, soft_labels)
                loss = soft_loss if args.loss_mode == "kd" else args.alpha * soft_loss + (1.0 - args.alpha) * hard_loss
                soft_label_sum += soft_labels.detach().sum(dim=0)
                soft_label_count += features.size(0)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_size = features.size(0)
            total_loss += float(loss.item()) * batch_size
            total_soft_loss += float(soft_loss.item()) * batch_size
            total_hard_loss += float(hard_loss.item()) * batch_size
        scheduler.step()

        train_size = max(len(train_loader.dataset), 1)
        val_metrics, val_true, val_pred = evaluate_model(model, val_loader, device)
        score = float(val_metrics[args.selection_metric])
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_val_true = val_true
            best_val_pred = val_pred
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "metrics": val_metrics,
                    "class_names": CLASS_NAMES,
                    "feature_stats": asdict(stats),
                    "args": vars(args),
                },
                best_path,
            )
        else:
            patience_counter += 1

        log_payload: dict[str, float | int] = {
            "epoch": epoch,
            "train/loss": total_loss / train_size,
            "train/soft_loss": total_soft_loss / train_size,
            "train/hard_loss": total_hard_loss / train_size,
            "train/lr": optimizer.param_groups[0]["lr"],
            "best/val_icbhi_score": best_score,
            **scalar_metrics(val_metrics, "val"),
        }
        if soft_label_count:
            soft_label_mean = (soft_label_sum / soft_label_count).detach().cpu().numpy()
            for idx, class_name in enumerate(CLASS_NAMES):
                log_payload[f"soft_labels/mean_{metric_key(class_name)}"] = float(soft_label_mean[idx])
        log_wandb(log_payload, step=epoch)
        print(
            f"seed={seed} epoch={epoch:03d} train_loss={total_loss/train_size:.4f} "
            f"val_{args.selection_metric}={score:.4f} best={best_score:.4f} patience={patience_counter}/{args.patience}",
            flush=True,
        )
        if patience_counter >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_true, test_pred = evaluate_model(model, test_loader, device)
    save_metrics(metrics_dir, f"seed_{seed}_val_best", best_val_metrics, best_val_true, best_val_pred)
    save_metrics(metrics_dir, f"seed_{seed}_test", test_metrics, test_true, test_pred)
    log_wandb({**scalar_metrics(best_val_metrics, "best_val"), **scalar_metrics(test_metrics, "test")})
    log_wandb_confusion(f"confusion_matrix/seed_{seed}_test", test_true, test_pred)
    log_checkpoint_artifact(best_path, seed, args)
    finish_wandb()
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_icbhi_score": float(best_score),
        "test_icbhi_score": float(test_metrics["icbhi_score"]),
        "test_sensitivity": float(test_metrics["sensitivity"]),
        "test_specificity": float(test_metrics["specificity"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "checkpoint": str(best_path),
    }
    with (metrics_dir / f"seed_{seed}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def prepare_run(args: argparse.Namespace) -> tuple[Path, dict[str, list[CycleRecord]], FeatureStats]:
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else ICBHI_3CLASS_EFFICIENTNET_KD_ARTIFACTS_DIR)
    split_path = output_dir / "splits.json"
    config_path = output_dir / "config.json"
    if split_path.exists() and config_path.exists() and not args.rebuild_splits:
        return output_dir, load_split_records(output_dir), load_feature_stats(output_dir)

    # Try to load existing splits from teacher directory to prevent KD alignment issues
    teacher_split_path = Path("artifacts/training/icbhi_3class_kd/splits.json")
    if teacher_split_path.exists() and not args.rebuild_splits:
        print(f"Loading split records from teacher splits: {teacher_split_path}", flush=True)
        splits = load_split_records(teacher_split_path.parent)
        records = build_records(Path(args.data_dir), Path(args.labels_file), max_files=args.max_files, max_cycles=args.max_cycles)
    else:
        records = build_records(Path(args.data_dir), Path(args.labels_file), max_files=args.max_files, max_cycles=args.max_cycles)
        splits = create_patientwise_splits(records, args.test_size, args.val_size, args.seed)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError(
            f"Invalid split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}. "
            "Increase --max_files/--max_cycles."
        )
    stats = estimate_feature_stats(splits["train"], args)
    save_split_records(output_dir, splits)
    save_config(output_dir, args, stats, records, splits)
    return output_dir, splits, stats


def summarize_runs(output_dir: Path, summaries: list[dict[str, float | int | str | None]]) -> None:
    metrics_dir = ensure_dir(output_dir / "metrics")
    keys = ["test_icbhi_score", "test_sensitivity", "test_specificity", "test_macro_f1", "test_accuracy"]
    aggregate: dict[str, object] = {"runs": summaries}
    for key in keys:
        values = np.array([float(summary[key]) for summary in summaries], dtype=np.float32)
        aggregate[f"{key}_mean"] = float(values.mean())
        aggregate[f"{key}_std"] = float(values.std())
    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)


def parse_seeds(seed_arg: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in seed_arg.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def default_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICBHI 2017 3-class disease-level EfficientNet-B0 training/KD")
    parser.add_argument("--data_dir", type=str, default=str(ICBHI_2017_DIR))
    parser.add_argument("--labels_file", type=str, default=str(ICBHI_2017_LABELS))
    parser.add_argument("--artifact_root", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--output_dir", type=str, default=None, help=f"Default: {ICBHI_3CLASS_EFFICIENTNET_KD_ARTIFACTS_DIR}")
    parser.add_argument("--loss_mode", choices=["supervised", "kd", "mixed"], default="mixed")
    parser.add_argument("--teacher_logits_train", type=str, default="artifacts/training/icbhi_3class_kd/soft_labels/teacher_logits_train.npy")
    parser.add_argument("--teacher_sample_ids_train", type=str, default="artifacts/training/icbhi_3class_kd/soft_labels/sample_ids_train.json")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focusing parameter for Focal Loss")
    parser.add_argument("--kd_mode", choices=["mean_teacher", "random_teacher"], default="mean_teacher")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=8.0)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--target_frames", type=int, default=512)
    parser.add_argument("--f_min", type=float, default=50.0)
    parser.add_argument("--f_max", type=float, default=2500.0)
    parser.add_argument("--input_channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--no_bandpass", action="store_true")
    parser.add_argument("--time_shift", type=float, default=0.05)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--freq_mask", type=int, default=16)
    parser.add_argument("--time_mask", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--selection_metric", choices=["icbhi_score", "macro_f1", "balanced_accuracy"], default="icbhi_score")
    parser.add_argument("--test_size", type=float, default=0.4)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_cycles", type=int, default=None)
    parser.add_argument("--max_stat_samples", type=int, default=512)
    parser.add_argument("--rebuild_splits", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="icbhi-3class-efficientnet-kd")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    set_seed(args.seed)
    device = default_device(args.device)
    output_dir, splits, stats = prepare_run(args)
    print(f"Output directory: {output_dir}", flush=True)
    print(
        f"Split sizes: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} "
        f"stats_mean={stats.mean:.4f} stats_std={stats.std:.4f}",
        flush=True,
    )
    summaries = []
    for seed in parse_seeds(args.seeds):
        summaries.append(train_one_seed(seed, args, output_dir, splits, stats, device))
    summarize_runs(output_dir, summaries)


if __name__ == "__main__":
    main()
