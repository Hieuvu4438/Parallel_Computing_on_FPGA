#!/usr/bin/env python3
"""
ICBHI 2017 official cycle-level KD pipeline — Multiview teacher ensemble.

End-to-end experiment:
  1) Train strong heterogeneous teachers.
  2) Save teacher logits for train/val/test.
  3) Distill reliability-weighted teacher probabilities into a CNN-only student.
  4) Tune the normal-vs-abnormal threshold for ICBHI Score.
  5) Evaluate on the official test split and log everything to W&B.

Default strategy: multiview log-mel/delta/delta-delta teachers + DS-CNN-Res-SE student.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
torch.set_num_threads(2)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import torchvision.models as tv_models
except Exception:
    tv_models = None

try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import ICBHI_2017_DIR, TRAINING_ARTIFACTS_DIR, ensure_dir

CLASS_NAMES_4 = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_NAMES_2 = ["Normal", "Abnormal"]
OFFICIAL_TRAIN_PATIENTS = set(range(101, 161))
OFFICIAL_TEST_PATIENTS = set(range(161, 227))


@dataclass(frozen=True)
class CycleRecord:
    sample_id: str
    wav_path: str
    subject_id: str
    start_sec: float
    end_sec: float
    crackle: int
    wheeze: int
    label_4class: int
    label_2class: int


@dataclass(frozen=True)
class FeatureStats:
    mean: float
    std: float


def get_class_names(nc: int) -> list[str]:
    return CLASS_NAMES_4 if nc == 4 else CLASS_NAMES_2


def event_to_4class(c: int, w: int) -> int:
    if c == 0 and w == 0:
        return 0
    if c == 1 and w == 0:
        return 1
    if c == 0 and w == 1:
        return 2
    return 3


def event_to_2class(c: int, w: int) -> int:
    return 0 if c == 0 and w == 0 else 1


def get_label(r: CycleRecord, nc: int) -> int:
    return r.label_4class if nc == 4 else r.label_2class


def read_cycle_annotations(wav_path: Path) -> list[tuple[float, float, int, int]]:
    ann = wav_path.with_suffix(".txt")
    if not ann.exists():
        return []
    cycles = []
    with ann.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                start, end, crackle, wheeze = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            if end > start:
                cycles.append((start, end, crackle, wheeze))
    return cycles


def build_records(data_dir: Path, max_files: int | None = None, max_cycles: int | None = None, allowed_basenames: set[str] | None = None) -> list[CycleRecord]:
    wavs = sorted(data_dir.glob("*.wav"))
    if allowed_basenames is not None:
        wavs = [p for p in wavs if p.stem in allowed_basenames]
    if max_files is not None and allowed_basenames is None:
        wavs = wavs[:max_files]
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")
    records = []
    for wav_path in wavs:
        sid = wav_path.stem.split("_")[0]
        for idx, (s, e, c, w) in enumerate(read_cycle_annotations(wav_path)):
            records.append(CycleRecord(
                sample_id=f"{wav_path.stem}__cycle_{idx:03d}", wav_path=str(wav_path), subject_id=sid,
                start_sec=s, end_sec=e, crackle=c, wheeze=w,
                label_4class=event_to_4class(c, w), label_2class=event_to_2class(c, w),
            ))
            if max_cycles is not None and len(records) >= max_cycles:
                return records
    if not records:
        raise ValueError("No annotated cycles found")
    return records


def record_basenames(records: list[CycleRecord]) -> set[str]:
    return {Path(r.wav_path).stem for r in records}


def assert_split_filename_protocol(splits: dict[str, list[CycleRecord]], allow_val_test_overlap: bool = False):
    names = {k: record_basenames(v) for k, v in splits.items()}
    assert names["train"].isdisjoint(names["val"]), "Train/val filename overlap detected"
    assert names["train"].isdisjoint(names["test"]), "Train/test filename overlap detected"
    if allow_val_test_overlap:
        assert names["val"] == names["test"], "Expected validation split to match test split for test-selection mode"
    else:
        assert names["val"].isdisjoint(names["test"]), "Val/test filename overlap detected"


def split_file_counts(splits: dict[str, list[CycleRecord]]) -> dict[str, int]:
    return {k: len(record_basenames(v)) for k, v in splits.items()}


def create_add_rsc_splits(data_dir: Path, args) -> dict[str, list[CycleRecord]]:
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


def create_official_splits(records: list[CycleRecord], nc: int, val_frac: float, seed: int) -> dict[str, list[CycleRecord]]:
    train_recs, test_recs = [], []
    for r in records:
        pid = int(r.subject_id)
        if pid in OFFICIAL_TRAIN_PATIENTS:
            train_recs.append(r)
        elif pid in OFFICIAL_TEST_PATIENTS:
            test_recs.append(r)
    s2r: dict[str, list[CycleRecord]] = {}
    s2l: dict[str, int] = {}
    for r in train_recs:
        s2r.setdefault(r.subject_id, []).append(r)
        s2l[r.subject_id] = max(s2l.get(r.subject_id, 0), get_label(r, nc))
    subjects = sorted(s2r)
    if not subjects:
        raise ValueError("Official train split is empty. Check data_dir/max_files.")
    labels = [s2l[s] for s in subjects]
    counts = {lb: labels.count(lb) for lb in set(labels)}
    strat = labels if len(counts) > 1 and min(counts.values()) >= 2 else None
    try:
        tr_s, va_s = train_test_split(subjects, test_size=val_frac, random_state=seed, stratify=strat)
    except ValueError:
        tr_s, va_s = train_test_split(subjects, test_size=val_frac, random_state=seed)
    splits = {"train": [r for s in sorted(tr_s) for r in s2r[s]], "val": [r for s in sorted(va_s) for r in s2r[s]], "test": test_recs}
    assert_split_filename_protocol(splits)
    return splits


def load_audio(wav_path: Path, target_sr: int, bandpass: bool, f_min: float, f_max: float):
    sr, audio = wavfile.read(str(wav_path))
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(np.iinfo(audio.dtype).max, 1)
    else:
        audio = audio.astype(np.float32)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        audio = sp_signal.resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    if bandpass and len(audio) > 32:
        hi = min(f_max, sr / 2 - 1)
        if hi > f_min:
            sos = sp_signal.butter(4, [f_min, hi], btype="bandpass", fs=sr, output="sos")
            audio = sp_signal.sosfiltfilt(sos, audio).astype(np.float32)
    return audio, sr


def segment_waveform(audio: np.ndarray, sr: int, start: float, end: float, target_samples: int):
    s = max(0, int(round(start * sr)))
    e = min(len(audio), int(round(end * sr)))
    seg = audio[s:e]
    if len(seg) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if len(seg) >= target_samples:
        off = (len(seg) - target_samples) // 2
        return seg[off:off + target_samples].astype(np.float32)
    reps = target_samples // len(seg) + 1
    return np.tile(seg, reps)[:target_samples].astype(np.float32)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def build_mel_filterbank(sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float):
    f_max = min(f_max, sr / 2 - 1)
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i:i + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for j in range(left, min(center, fb.shape[1])):
            fb[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, min(right, fb.shape[1])):
            fb[i, j] = (right - j) / max(right - center, 1)
    enorm = 2.0 / np.maximum(hz_pts[2:n_mels + 2] - hz_pts[:n_mels], 1e-8)
    return fb * enorm[:, None]


def compute_logmel(wf, sr, fb, n_fft, win_len, hop_len, target_frames):
    _, _, stft = sp_signal.stft(wf, fs=sr, window="hann", nperseg=win_len, noverlap=win_len - hop_len, nfft=n_fft, boundary=None, padded=False)
    power = np.abs(stft).astype(np.float32) ** 2
    mel = np.matmul(fb, power)
    lm = np.log(np.maximum(mel, 1e-10)).astype(np.float32)
    if lm.shape[1] >= target_frames:
        return lm[:, :target_frames]
    pad = np.full((lm.shape[0], target_frames), float(lm.min()), dtype=np.float32)
    pad[:, :lm.shape[1]] = lm
    return pad


def add_delta_channels(feat: np.ndarray) -> np.ndarray:
    d1 = np.gradient(feat, axis=1).astype(np.float32)
    d2 = np.gradient(d1, axis=1).astype(np.float32)
    return np.stack([feat, d1, d2], axis=0).astype(np.float32)


def apply_wave_aug(wf, shift, noise_std, speed_perturb):
    aug = wf.copy()
    if shift > 0:
        aug = np.roll(aug, int(np.random.uniform(-shift, shift) * len(aug)))
    if noise_std > 0:
        rms = float(np.sqrt(np.mean(aug ** 2) + 1e-8))
        aug += np.random.normal(0, noise_std * rms, aug.shape).astype(np.float32)
    if speed_perturb and np.random.random() < 0.25:
        rate = float(np.random.choice([0.95, 1.0, 1.05]))
        if rate != 1.0:
            orig = len(aug)
            aug = sp_signal.resample(aug, max(1, int(orig * rate))).astype(np.float32)
            aug = aug[:orig] if len(aug) > orig else np.pad(aug, (0, orig - len(aug)))
    return aug.astype(np.float32)


def apply_spec_aug(feat, freq_mask, time_mask):
    aug = feat.copy()
    fill = float(aug.mean())
    if freq_mask > 0 and aug.shape[0] > 1:
        w = np.random.randint(0, min(freq_mask, aug.shape[0] - 1) + 1)
        s = np.random.randint(0, max(aug.shape[0] - w, 1))
        aug[s:s + w, :] = fill
    if time_mask > 0 and aug.shape[1] > 1:
        w = np.random.randint(0, min(time_mask, aug.shape[1] - 1) + 1)
        s = np.random.randint(0, max(aug.shape[1] - w, 1))
        aug[:, s:s + w] = fill
    return aug.astype(np.float32)


class ICBHIDataset(Dataset):
    def __init__(self, records, args, stats: FeatureStats | None, augment: bool, return_sample_id: bool = False):
        self.records = records
        self.args = args
        self.stats = stats
        self.augment = augment
        self.return_sample_id = return_sample_id
        self.target_samples = int(round(args.duration_sec * args.sample_rate))
        self.fb = build_mel_filterbank(args.sample_rate, args.n_fft, args.n_mels, args.f_min, args.f_max)
        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        
        # Setup Cache Directory based on spectrogram config
        import hashlib
        param_str = f"{args.sample_rate}_{args.n_fft}_{args.win_length}_{args.hop_length}_{args.n_mels}_{args.target_frames}_{args.f_min}_{args.f_max}_{args.no_bandpass}_{args.duration_sec}"
        param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
        self.cache_dir = Path("/home/haipd/Parallel_Computing_on_FPGA/data/cache_spectrograms") / param_hash
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        cache_file = self.cache_dir / f"{r.sample_id}.npy"
        
        # Load from cache if it exists, otherwise compute and save
        if cache_file.exists():
            feat = np.load(cache_file)
        else:
            wf, sr = self._load(r.wav_path)
            seg = segment_waveform(wf, sr, r.start_sec, r.end_sec, self.target_samples)
            # Compute base logmel without wave augmentations for caching
            feat = compute_logmel(seg, sr, self.fb, self.args.n_fft, self.args.win_length, self.args.hop_length, self.args.target_frames)
            np.save(cache_file, feat)
            
        # Apply augmentations on the fly if training
        if self.augment:
            # 1. SpecAugment
            feat = apply_spec_aug(feat, self.args.freq_mask, self.args.time_mask)
            # 2. Time shift on spectrogram via rolling
            if self.args.time_shift > 0:
                max_shift = int(self.args.time_shift * self.args.target_frames / self.args.duration_sec)
                if max_shift > 0:
                    shift = np.random.randint(-max_shift, max_shift + 1)
                    feat = np.roll(feat, shift, axis=1)
            # 3. Add noise on spectrogram
            if self.args.noise_std > 0:
                feat += np.random.normal(0, self.args.noise_std, feat.shape).astype(np.float32)
                
        if self.stats is not None:
            feat = (feat - self.stats.mean) / max(self.stats.std, 1e-6)
            
        if self.args.input_view == "logmel_delta":
            tensor = torch.from_numpy(add_delta_channels(feat)).float()
        else:
            tensor = torch.from_numpy(feat).unsqueeze(0).float()
            
        label = torch.tensor(get_label(r, self.args.num_classes), dtype=torch.long)
        ident = r.sample_id if self.return_sample_id else torch.tensor(idx, dtype=torch.long)
        return tensor, label, ident

    def _load(self, wp):
        if wp not in self._cache:
            self._cache[wp] = load_audio(Path(wp), self.args.sample_rate, not self.args.no_bandpass, self.args.f_min, self.args.f_max)
        return self._cache[wp]


def estimate_feature_stats(records, args):
    ds = ICBHIDataset(records, args, None, False)
    limit = min(len(ds), args.max_stat_samples)
    total, total_sq, count = 0.0, 0.0, 0
    for i in range(limit):
        feat, _, _ = ds[i]
        v = feat[0] if feat.ndim == 3 else feat
        total += float(v.sum())
        total_sq += float((v * v).sum())
        count += v.numel()
    if count == 0:
        raise ValueError("Cannot estimate stats from empty dataset")
    mean = total / count
    return FeatureStats(float(mean), float(math.sqrt(max(total_sq / count - mean * mean, 1e-12))))


def class_weights(records, nc, device):
    labels = [get_label(r, nc) for r in records]
    counts = np.bincount(labels, minlength=nc).astype(np.float32)
    weights = counts.sum() / np.maximum(counts * nc, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def sample_weights(records, nc):
    labels = [get_label(r, nc) for r in records]
    counts = np.bincount(labels, minlength=nc).astype(np.float64)
    weights = counts.sum() / np.maximum(counts * nc, 1.0)
    return torch.tensor([weights[y] for y in labels], dtype=torch.double)


def make_loader(ds, args, records=None, balanced=False, shuffle=False):
    sampler = WeightedRandomSampler(sample_weights(records, args.num_classes), len(records), replacement=True) if balanced and records else None
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle and sampler is None, sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


class ConvBNAct(nn.Module):
    def __init__(self, ic, oc, k=3, stride=1, groups=1):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(nn.Conv2d(ic, oc, k, stride=stride, padding=pad, groups=groups, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, ch, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x)


class DSResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1, se=True):
        super().__init__()
        self.body = nn.Sequential(ConvBNAct(ic, ic, 3, stride=stride, groups=ic), ConvBNAct(ic, oc, 1), SEBlock(oc) if se else nn.Identity())
        self.skip = nn.Identity() if ic == oc and stride == 1 else ConvBNAct(ic, oc, 1, stride=stride)
    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x), inplace=True)


class StudentCNN6(nn.Module):
    def __init__(self, nc=4, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(ConvBNAct(in_ch, 16), ConvBNAct(16, 16), nn.MaxPool2d(2), ConvBNAct(16, 32), ConvBNAct(32, 32), nn.MaxPool2d(2), ConvBNAct(32, 64), ConvBNAct(64, 64), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, nc))
    def forward(self, x):
        return self.classifier(self.features(x))


class DSCNNResSEStudent(nn.Module):
    def __init__(self, nc=4, in_ch=1, width=1.0):
        super().__init__()
        c = lambda v: max(8, int(v * width))
        self.features = nn.Sequential(ConvBNAct(in_ch, c(24)), DSResBlock(c(24), c(32), stride=2), DSResBlock(c(32), c(48)), DSResBlock(c(48), c(64), stride=2), DSResBlock(c(64), c(96)), DSResBlock(c(96), c(128), stride=2))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(c(128) * 2, nc)
    def forward(self, x):
        x = self.features(x)
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.fc(self.dropout(torch.cat([avg, mx], dim=1)))


class ResidualBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(ic, oc, 3, stride=stride)
        self.conv2 = nn.Sequential(nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc))
        self.skip = nn.Identity() if ic == oc and stride == 1 else nn.Sequential(nn.Conv2d(ic, oc, 1, stride=stride, bias=False), nn.BatchNorm2d(oc))
    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + self.skip(x), inplace=True)


class ResNetCNNTeacher(nn.Module):
    def __init__(self, nc=4, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(ConvBNAct(in_ch, 32), ResidualBlock(32, 32), ResidualBlock(32, 64, 2), ResidualBlock(64, 64), ResidualBlock(64, 128, 2), ResidualBlock(128, 160, 2), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(160, nc))
    def forward(self, x):
        return self.classifier(self.features(x))


class ResNetCRNNTeacher(nn.Module):
    def __init__(self, nc=4, in_ch=1, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(ConvBNAct(in_ch, 32), ResidualBlock(32, 64, 2), ResidualBlock(64, 96, 2), ResidualBlock(96, 128, (2, 1) if isinstance((2, 1), int) else 1))
        self.reduce = nn.AdaptiveAvgPool2d((8, None))
        self.lstm = nn.LSTM(128 * 8, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(hidden * 2, nc))
    def forward(self, x):
        x = self.cnn(x)
        x = self.reduce(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.lstm(x)
        w = F.softmax(self.attn(out).squeeze(-1), dim=1).unsqueeze(1)
        ctx = torch.bmm(w, out).squeeze(1)
        return self.fc(ctx)


class EfficientNetTeacher(nn.Module):
    def __init__(self, nc=4, in_ch=3, pretrained=True):
        super().__init__()
        if tv_models is None:
            raise ImportError("torchvision is unavailable")
        weights = None
        if pretrained:
            try:
                weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        self.backbone = tv_models.efficientnet_b0(weights=weights)
        if in_ch != 3:
            conv = self.backbone.features[0][0]
            rep = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
            with torch.no_grad():
                rep.weight.copy_(conv.weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1) / max(in_ch, 1))
            self.backbone.features[0][0] = rep
        nf = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(nf, nc))
    def forward(self, x):
        return self.backbone(x)


def make_model(name, nc, in_ch, args):
    if name == "cnn6":
        return StudentCNN6(nc, in_ch)
    if name == "ds_cnn_res_se":
        return DSCNNResSEStudent(nc, in_ch, args.student_width)
    if name == "resnet_cnn":
        return ResNetCNNTeacher(nc, in_ch)
    if name == "resnet_crnn":
        return ResNetCRNNTeacher(nc, in_ch)
    if name == "efficientnet_b0":
        return EfficientNetTeacher(nc, in_ch, pretrained=not args.no_pretrained)
    raise ValueError(f"Unknown model: {name}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls = label_smoothing
    def forward(self, logits, targets):
        nc = logits.size(1)
        if self.ls > 0:
            target = torch.full_like(logits, self.ls / max(nc - 1, 1))
            target.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        else:
            target = F.one_hot(targets, nc).float()
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        loss = -((1 - p) ** self.gamma) * target * logp
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).view(1, -1)
        return loss.sum(dim=1).mean()


def per_class_specificity(cm):
    total = cm.sum()
    vals = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        vals.append(tn / max(tn + fp, 1))
    return np.array(vals, dtype=np.float32)


def icbhi_score(y_true, y_pred):
    normal = y_true == 0
    abnormal = y_true != 0
    sp = float(np.mean(y_pred[normal] == 0)) if normal.any() else 0.0
    se = float(np.mean(y_pred[abnormal] != 0)) if abnormal.any() else 0.0
    return se, sp, (se + sp) / 2.0


def binary_metrics_from_4class(y_true, y_pred):
    yt = (y_true != 0).astype(np.int64)
    yp = (y_pred != 0).astype(np.int64)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"binary_sensitivity": float(sens), "binary_specificity": float(spec), "binary_icbhi_score": float((sens + spec) / 2.0), "binary_accuracy": float(accuracy_score(yt, yp))}


def safe_auc(y_true, y_prob, nc):
    try:
        if len(np.unique(y_true)) < nc:
            return None
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(nc))))
    except ValueError:
        return None


def compute_metrics(y_true, y_pred, y_prob, nc):
    labels = list(range(nc))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    spec = per_class_specificity(cm)
    se, sp, score = icbhi_score(y_true, y_pred)
    metrics = {"icbhi_score": float(score), "sensitivity": float(se), "specificity": float(sp), "accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "confusion_matrix": cm.tolist()}
    metrics.update(binary_metrics_from_4class(y_true, y_pred))
    if y_prob is not None:
        metrics["auc_ovr"] = safe_auc(y_true, y_prob, nc)
    for i, name in enumerate(get_class_names(nc)):
        key = name.lower().replace(" ", "_")
        metrics[f"{key}_precision"] = float(prec[i])
        metrics[f"{key}_recall"] = float(rec[i])
        metrics[f"{key}_f1"] = float(f1[i])
        metrics[f"{key}_specificity"] = float(spec[i])
        metrics[f"{key}_support"] = int(sup[i])
    return metrics


def threshold_predictions(probs, threshold):
    preds = probs.argmax(axis=1)
    if probs.shape[1] > 1:
        abnormal = probs[:, 1:].argmax(axis=1) + 1
        preds = np.where(probs[:, 0] >= threshold, 0, abnormal)
    return preds


def sweep_threshold(y_true, probs):
    best = {"threshold": 0.5, "icbhi_score": -1.0}
    for th in np.linspace(0.05, 0.95, 91):
        pred = threshold_predictions(probs, float(th))
        m = compute_metrics(y_true, pred, probs, probs.shape[1])
        if m["icbhi_score"] > best["icbhi_score"]:
            best = {"threshold": float(th), **m}
    return best


def evaluate_logits(logits, records, nc):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    y_true = np.array([get_label(r, nc) for r in records], dtype=np.int64)
    y_pred = probs.argmax(axis=1)
    return compute_metrics(y_true, y_pred, probs, nc), y_true, y_pred, probs


def evaluate_model(model, loader, device, nc):
    model.eval()
    yt, logits_all = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            logits_all.append(model(x.to(device)).cpu())
            yt.extend(y.numpy().tolist())
    logits = torch.cat(logits_all, dim=0).numpy() if logits_all else np.zeros((0, nc), dtype=np.float32)
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    y_true = np.array(yt, dtype=np.int64)
    y_pred = probs.argmax(axis=1) if len(probs) else np.array([], dtype=np.int64)
    return compute_metrics(y_true, y_pred, probs, nc), y_true, y_pred, probs, logits


def save_metrics(output_dir, name, metrics, y_true=None, y_pred=None, nc=4):
    md = ensure_dir(output_dir / "metrics")
    with (md / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if y_true is not None and y_pred is not None:
        cn = get_class_names(nc)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(nc)))
        with (md / f"confusion_matrix_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true/pred", *cn])
            for i, row in enumerate(cm):
                w.writerow([cn[i], *row.tolist()])


def count_params(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_wandb(payload, prefix=None, step=None):
    if wandb is None or wandb.run is None:
        return
    clean = {}
    for k, v in payload.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            clean[f"{prefix}/{k}" if prefix else k] = v
    if clean:
        wandb.log(clean, step=step)


def init_wandb(args, name, extra=None):
    if not args.wandb or args.wandb_mode == "disabled":
        return
    if wandb is None:
        raise ImportError("wandb is not installed")
    cfg = vars(args).copy()
    if extra:
        cfg.update(extra)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=name, dir=str(args.output_dir), mode=args.wandb_mode, config=cfg, tags=["icbhi-2017", f"{args.num_classes}class", "kd", "student-cnn", args.pipeline_name], reinit=True)


def finish_wandb():
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def prepare_run(args):
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else TRAINING_ARTIFACTS_DIR / args.pipeline_name)
    args.output_dir = str(output_dir)
    split_path = output_dir / "splits.json"
    config_path = output_dir / "config.json"
    if split_path.exists() and config_path.exists() and not args.rebuild_splits:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        cache_matches = (
            cfg.get("max_files") == args.max_files
            and cfg.get("max_cycles") == args.max_cycles
            and cfg.get("num_classes") == args.num_classes
            and cfg.get("input_view") == args.input_view
            and cfg.get("benchmark_protocol", "official_icbhi") == args.benchmark_protocol
            and cfg.get("add_rsc_split_seed") == args.add_rsc_split_seed
            and cfg.get("add_rsc_use_test_for_selection") == args.add_rsc_use_test_for_selection
        )
        if cache_matches:
            with split_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            raw_splits = raw.get("splits", raw)
            splits = {k: [CycleRecord(**r) for r in v] for k, v in raw_splits.items()}
            assert_split_filename_protocol(splits, allow_val_test_overlap=args.benchmark_protocol == "add_rsc" and args.add_rsc_use_test_for_selection)
            return output_dir, splits, FeatureStats(float(cfg["feature_mean"]), float(cfg["feature_std"]))
    if args.benchmark_protocol == "official_icbhi":
        records = build_records(Path(args.data_dir), args.max_files, args.max_cycles)
        splits = create_official_splits(records, args.num_classes, args.val_size, args.seed)
    elif args.benchmark_protocol == "add_rsc":
        splits = create_add_rsc_splits(Path(args.data_dir), args)
    else:
        raise ValueError(f"Unknown benchmark protocol: {args.benchmark_protocol}")
    stats = estimate_feature_stats(splits["train"], args)
    split_meta = {
        "benchmark_protocol": args.benchmark_protocol,
        "add_rsc_split_seed": args.add_rsc_split_seed if args.benchmark_protocol == "add_rsc" else None,
        "add_rsc_use_test_for_selection": args.add_rsc_use_test_for_selection if args.benchmark_protocol == "add_rsc" else None,
        "file_counts": split_file_counts(splits),
        "cycle_counts": {k: len(v) for k, v in splits.items()},
        "filenames": {k: sorted(record_basenames(v)) for k, v in splits.items()},
    }
    with split_path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": split_meta, "splits": {k: [asdict(r) for r in v] for k, v in splits.items()}}, f, indent=2)
    cfg = vars(args).copy()
    cfg.update({"feature_mean": stats.mean, "feature_std": stats.std, "split_sizes": {k: len(v) for k, v in splits.items()}, "split_file_counts": split_file_counts(splits)})
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return output_dir, splits, stats


def train_teacher(arch, seed, args, splits, stats, device, output_dir):
    set_seed(seed)
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    model = make_model(arch, args.num_classes, in_ch, args).to(device)
    teacher_dir = ensure_dir(output_dir / "teachers" / arch / f"seed_{seed}")
    init_wandb(args, f"{args.pipeline_name}-teacher-{arch}-seed-{seed}", {"arch": arch, "seed": seed, "params": count_params(model)[0]})
    train_ds = ICBHIDataset(splits["train"], args, stats, True)
    val_ds = ICBHIDataset(splits["val"], args, stats, False)
    train_loader = make_loader(train_ds, args, splits["train"], balanced=True)
    val_loader = make_loader(val_ds, args)
    criterion = FocalLoss(class_weights(splits["train"], args.num_classes, device), args.focal_gamma, args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_teacher, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs_teacher, 1))
    best_score, best_epoch, patience = -1.0, 0, 0
    best_path = teacher_dir / "best.pt"
    for epoch in range(1, args.epochs_teacher + 1):
        model.train()
        total = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.item()) * x.size(0)
        sched.step()
        val_m, _, _, _, _ = evaluate_model(model, val_loader, device, args.num_classes)
        teacher_metric = "icbhi_score" if args.selection_metric == "threshold_icbhi_score" else args.selection_metric
        score = float(val_m[teacher_metric])
        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save({"model_state": model.state_dict(), "arch": arch, "seed": seed, "args": vars(args), "metrics": val_m}, best_path)
        else:
            patience += 1
        avg = total / max(len(train_ds), 1)
        log_wandb({"epoch": epoch, "train_loss": avg, **{f"val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float))}}, prefix=f"teacher/{arch}", step=epoch)
        print(f"teacher={arch} seed={seed} ep={epoch:03d} loss={avg:.4f} val_{teacher_metric}={score:.4f} best={best_score:.4f}", flush=True)
        if patience >= args.patience:
            break
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    finish_wandb()
    return model, best_epoch, best_score


def collect_and_save_logits(model, arch, seed, args, splits, stats, device, output_dir):
    logits_dir = ensure_dir(output_dir / "teacher_logits")
    for split, records in splits.items():
        if not records:
            continue
        ds = ICBHIDataset(records, args, stats, False, return_sample_id=True)
        loader = make_loader(ds, args)
        _, y_true, y_pred, probs, logits = evaluate_model(model, loader, device, args.num_classes)
        stem = f"{arch}_seed_{seed}_{split}"
        np.save(logits_dir / f"{stem}.npy", logits)
        with (logits_dir / f"{stem}_sample_ids.json").open("w", encoding="utf-8") as f:
            json.dump([r.sample_id for r in records], f, indent=2)
        save_metrics(output_dir, f"teacher_{arch}_seed_{seed}_{split}", compute_metrics(y_true, y_pred, probs, args.num_classes), y_true, y_pred, args.num_classes)


def expected_teacher_names(args):
    return [f"{arch}_seed_{seed}" for arch in parse_csv(args.teacher_arches) for seed in parse_int_csv(args.seeds)]


def load_teacher_logits(args, output_dir, split, records):
    logits = []
    names = []
    expected_ids = [r.sample_id for r in records]
    logits_dir = output_dir / "teacher_logits"
    for name in expected_teacher_names(args):
        path = logits_dir / f"{name}_{split}.npy"
        ids_path = logits_dir / f"{name}_{split}_sample_ids.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing expected teacher logits: {path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing expected teacher sample IDs: {ids_path}")
        arr = np.load(path)
        with ids_path.open("r", encoding="utf-8") as f:
            ids = json.load(f)
        if ids != expected_ids:
            raise ValueError(f"Sample ID mismatch for {path}. Re-run teacher stage with the current split/config.")
        if arr.shape != (len(records), args.num_classes):
            raise ValueError(f"Unexpected logits shape for {path}: got {arr.shape}, expected {(len(records), args.num_classes)}")
        logits.append(arr)
        names.append(name)
    if not logits:
        raise FileNotFoundError(f"No configured teacher logits found for split={split} in {logits_dir}")
    return np.stack(logits, axis=0), names


def reliability_weights(val_logits, val_records, nc):
    y_true = np.array([get_label(r, nc) for r in val_records], dtype=np.int64)
    weights = []
    for t in range(val_logits.shape[0]):
        probs = F.softmax(torch.tensor(val_logits[t]), dim=1).numpy()
        pred = probs.argmax(axis=1)
        _, rec, f1, _ = precision_recall_fscore_support(y_true, pred, labels=list(range(nc)), zero_division=0)
        weights.append(np.maximum(0.05, 0.5 * rec + 0.5 * f1))
    w = np.stack(weights, axis=0).astype(np.float32)
    w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-8)
    return w


def weighted_teacher_probs(logits, class_weights_by_teacher, temperature):
    probs = F.softmax(torch.tensor(logits / temperature, dtype=torch.float32), dim=2).numpy()
    out = (probs * class_weights_by_teacher[:, None, :]).sum(axis=0)
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-8)
    return out.astype(np.float32)


class StudentKDDataset(Dataset):
    def __init__(self, base_ds, teacher_probs):
        self.base_ds = base_ds
        self.teacher_probs = torch.tensor(teacher_probs, dtype=torch.float32)
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        x, y, _ = self.base_ds[idx]
        return x, y, torch.tensor(idx, dtype=torch.long), self.teacher_probs[idx]


def abnormal_logit_from_4class(logits):
    if logits.size(1) == 2:
        return logits[:, 1] - logits[:, 0]
    return torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]


def train_student(args, splits, stats, device, output_dir):
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    val_logits, teacher_names = load_teacher_logits(args, output_dir, "val", splits["val"])
    train_logits, _ = load_teacher_logits(args, output_dir, "train", splits["train"])
    weights = reliability_weights(val_logits, splits["val"], args.num_classes)
    train_probs = weighted_teacher_probs(train_logits, weights, args.temperature)
    student = make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    student_dir = ensure_dir(output_dir / "students" / args.student_arch)
    with (student_dir / "teacher_reliability.json").open("w", encoding="utf-8") as f:
        json.dump({"teacher_names": teacher_names, "class_weights": weights.tolist()}, f, indent=2)
    init_wandb(args, f"{args.pipeline_name}-student-{args.student_arch}", {"student_params": count_params(student)[0], "teacher_names": teacher_names})
    base_train = ICBHIDataset(splits["train"], args, stats, True)
    train_ds = StudentKDDataset(base_train, train_probs)
    sampler = WeightedRandomSampler(sample_weights(splits["train"], args.num_classes), len(splits["train"]), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = make_loader(ICBHIDataset(splits["val"], args, stats, False), args)
    hard = FocalLoss(class_weights(splits["train"], args.num_classes, device), args.focal_gamma, args.label_smoothing)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs_student, 1))
    best_score, best_epoch, patience = -1.0, 0, 0
    best_tiebreak_macro = -1.0
    best_tiebreak_bal = -1.0
    best_tiebreak_both = -1.0
    min_both_f1_guard = 0.05 if args.num_classes == 4 else -1.0
    best_path = student_dir / "best.pt"
    for epoch in range(1, args.epochs_student + 1):
        student.train()
        total = hard_total = kd_total = bin_total = 0.0
        for x, y, _, tprob in train_loader:
            x, y, tprob = x.to(device), y.to(device), tprob.to(device)
            opt.zero_grad(set_to_none=True)
            logits = student(x)
            hard_loss = hard(logits, y)
            kd_loss = -(tprob * F.log_softmax(logits / args.temperature, dim=1)).sum(dim=1).mean() * (args.temperature ** 2)
            hard_bin = (y != 0).float()
            teacher_bin = (1.0 - tprob[:, 0]).clamp(0, 1)
            bin_target = 0.5 * hard_bin + 0.5 * teacher_bin
            bin_loss = F.binary_cross_entropy_with_logits(abnormal_logit_from_4class(logits), bin_target)
            loss = args.hard_weight * hard_loss + args.kd_weight * kd_loss + args.binary_weight * bin_loss
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            n = x.size(0)
            total += float(loss.item()) * n
            hard_total += float(hard_loss.item()) * n
            kd_total += float(kd_loss.item()) * n
            bin_total += float(bin_loss.item()) * n
        sched.step()
        val_m, yv, _, pv, _ = evaluate_model(student, val_loader, device, args.num_classes)
        tuned = sweep_threshold(yv, pv)
        score = float(tuned["icbhi_score"] if args.selection_metric == "threshold_icbhi_score" else val_m[args.selection_metric])
        both_f1 = float(val_m.get("both_f1", 0.0)) if args.num_classes == 4 else 0.0
        meets_guard = both_f1 >= min_both_f1_guard
        macro_f1 = float(val_m.get("macro_f1", 0.0))
        bal_acc = float(val_m.get("balanced_accuracy", 0.0))

        better_primary = score > best_score + 1e-12
        tie_primary = abs(score - best_score) <= 1e-12
        better_tiebreak = tie_primary and (
            (macro_f1 > best_tiebreak_macro + 1e-12)
            or (abs(macro_f1 - best_tiebreak_macro) <= 1e-12 and bal_acc > best_tiebreak_bal + 1e-12)
            or (abs(macro_f1 - best_tiebreak_macro) <= 1e-12 and abs(bal_acc - best_tiebreak_bal) <= 1e-12 and both_f1 > best_tiebreak_both + 1e-12)
        )

        should_save = False
        if meets_guard and (better_primary or better_tiebreak):
            should_save = True
        elif (not meets_guard) and (best_epoch == 0) and better_primary:
            # Bootstrap fallback: if no guarded checkpoint exists yet, allow first improving checkpoint.
            should_save = True

        if should_save:
            best_score, best_epoch, patience = score, epoch, 0
            best_tiebreak_macro = macro_f1
            best_tiebreak_bal = bal_acc
            best_tiebreak_both = both_f1
            torch.save(
                {
                    "model_state": student.state_dict(),
                    "epoch": epoch,
                    "arch": args.student_arch,
                    "threshold": tuned["threshold"],
                    "metrics": val_m,
                    "threshold_metrics": tuned,
                    "args": vars(args),
                    "selection_info": {
                        "score": score,
                        "macro_f1": macro_f1,
                        "balanced_accuracy": bal_acc,
                        "both_f1": both_f1,
                        "meets_both_f1_guard": meets_guard,
                        "both_f1_guard": min_both_f1_guard,
                    },
                },
                best_path,
            )
            np.save(student_dir / "val_probs_best.npy", pv)
        else:
            patience += 1
        denom = max(len(train_ds), 1)
        log_wandb({"epoch": epoch, "loss": total / denom, "hard_loss": hard_total / denom, "kd_loss": kd_total / denom, "binary_loss": bin_total / denom, **{f"val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float))}, "val_threshold_icbhi_score": tuned["icbhi_score"], "val_threshold": tuned["threshold"], "val_meets_both_f1_guard": float(meets_guard), "val_both_f1_guard": float(min_both_f1_guard), "best_score": float(best_score), "best_macro_f1": float(best_tiebreak_macro), "best_balanced_accuracy": float(best_tiebreak_bal), "best_both_f1": float(best_tiebreak_both)}, prefix="student", step=epoch)
        print(f"student ep={epoch:03d} loss={total/denom:.4f} val_icbhi={val_m['icbhi_score']:.4f} tuned={tuned['icbhi_score']:.4f} macro={macro_f1:.4f} bal={bal_acc:.4f} both={both_f1:.4f} guard={int(meets_guard)} best={best_score:.4f}", flush=True)
        if patience >= args.patience:
            break
    finish_wandb()
    return best_path


def evaluate_final(args, splits, stats, device, output_dir):
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    student = make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    ckpt_path = output_dir / "students" / args.student_arch / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["model_state"])
    threshold = float(ckpt.get("threshold", 0.5))
    init_wandb(args, f"{args.pipeline_name}-final-eval", {"student_checkpoint": str(ckpt_path), "threshold": threshold})
    summary = {"student_checkpoint": str(ckpt_path), "threshold": threshold, "student_params": count_params(student)[0]}
    for split in ["val", "test"]:
        if not splits[split]:
            continue
        loader = make_loader(ICBHIDataset(splits[split], args, stats, False), args)
        raw_m, yt, yp, probs, _ = evaluate_model(student, loader, device, args.num_classes)
        tuned_pred = threshold_predictions(probs, threshold)
        tuned_m = compute_metrics(yt, tuned_pred, probs, args.num_classes)
        save_metrics(output_dir, f"student_{split}_raw", raw_m, yt, yp, args.num_classes)
        save_metrics(output_dir, f"student_{split}_threshold", tuned_m, yt, tuned_pred, args.num_classes)
        summary[f"{split}_raw"] = raw_m
        summary[f"{split}_threshold"] = tuned_m
        log_wandb({f"{split}_raw_{k}": v for k, v in raw_m.items() if isinstance(v, (int, float))}, prefix="final")
        log_wandb({f"{split}_threshold_{k}": v for k, v in tuned_m.items() if isinstance(v, (int, float))}, prefix="final")
    with (ensure_dir(output_dir / "metrics") / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (ensure_dir(output_dir / "metrics") / "final_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "mode", "icbhi_score", "sensitivity", "specificity", "macro_f1", "accuracy", "binary_icbhi_score"])
        for split in ["val", "test"]:
            for mode in ["raw", "threshold"]:
                m = summary.get(f"{split}_{mode}")
                if m:
                    w.writerow([split, mode, m.get("icbhi_score"), m.get("sensitivity"), m.get("specificity"), m.get("macro_f1"), m.get("accuracy"), m.get("binary_icbhi_score")])
    if args.export_onnx:
        try:
            dummy = torch.randn(1, in_ch, args.n_mels, args.target_frames, device=device)
            torch.onnx.export(student.eval(), dummy, str(output_dir / "student_final.onnx"), export_params=True, opset_version=11, do_constant_folding=True, input_names=["mel_spectrogram"], output_names=["logits"], dynamic_axes={"mel_spectrogram": {0: "batch"}, "logits": {0: "batch"}})
            print(f"ONNX exported: {output_dir / 'student_final.onnx'}", flush=True)
        except Exception as exc:
            print(f"ONNX export failed: {exc}", flush=True)
    finish_wandb()
    return summary


def parse_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_csv(s):
    return [int(x) for x in parse_csv(s)]


def default_device(arg):
    if arg != "auto":
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="ICBHI 2017 KD full pipeline: multiview teacher ensemble -> CNN student")
    p.add_argument("--pipeline_name", default="icbhi_kd_multiview_ensemble")
    p.add_argument("--data_dir", default=str(ICBHI_2017_DIR))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--num_classes", type=int, choices=[2, 4], default=4)
    p.add_argument("--teacher_arches", default="resnet_cnn,resnet_crnn,efficientnet_b0")
    p.add_argument("--student_arch", choices=["cnn6", "ds_cnn_res_se"], default="ds_cnn_res_se")
    p.add_argument("--student_width", type=float, default=1.0)
    p.add_argument("--stage", choices=["all", "teachers", "student", "evaluate"], default="all")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=8.0)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--target_frames", type=int, default=512)
    p.add_argument("--f_min", type=float, default=50.0)
    p.add_argument("--f_max", type=float, default=4000.0)
    p.add_argument("--input_view", choices=["logmel", "logmel_delta"], default="logmel_delta")
    p.add_argument("--no_bandpass", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--time_shift", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--freq_mask", type=int, default=12)
    p.add_argument("--time_mask", type=int, default=48)
    p.add_argument("--speed_perturb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs_teacher", "--epochs", type=int, default=80)
    p.add_argument("--epochs_student", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_teacher", type=float, default=1e-3)
    p.add_argument("--lr_student", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--hard_weight", type=float, default=0.35)
    p.add_argument("--kd_weight", type=float, default=0.45)
    p.add_argument("--binary_weight", type=float, default=0.20)
    p.add_argument("--selection_metric", choices=["icbhi_score", "macro_f1", "balanced_accuracy", "threshold_icbhi_score"], default="threshold_icbhi_score")
    p.add_argument("--benchmark_protocol", choices=["official_icbhi", "add_rsc"], default="add_rsc")
    p.add_argument("--add_rsc_split_seed", type=int, default=1)
    p.add_argument("--add_rsc_use_test_for_selection", action="store_true", default=False)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--max_stat_samples", type=int, default=512)
    p.add_argument("--rebuild_splits", action="store_true")
    p.add_argument("--export_onnx", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="icbhi-4class-kd-strong")
    p.add_argument("--wandb_entity", default="vhieu4344")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def print_run_header(args, output_dir, splits):
    cn = get_class_names(args.num_classes)
    print(f"Pipeline: {args.pipeline_name}", flush=True)
    print(f"Task: {args.num_classes}-class ({', '.join(cn)})", flush=True)
    print(f"Benchmark protocol: {args.benchmark_protocol}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)
    file_counts = split_file_counts(splits)
    print(f"Split files: train={file_counts['train']} val={file_counts['val']} test={file_counts['test']}", flush=True)
    for name, records in splits.items():
        labels = [get_label(r, args.num_classes) for r in records]
        print(f"  {name}: {{" + ", ".join(f"{cn[i]}={labels.count(i)}" for i in range(args.num_classes)) + "}}", flush=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    output_dir, splits, stats = prepare_run(args)
    print_run_header(args, output_dir, splits)

    if args.stage in {"all", "teachers"}:
        for arch in parse_csv(args.teacher_arches):
            for seed in parse_int_csv(args.seeds):
                try:
                    model, _, _ = train_teacher(arch, seed, args, splits, stats, device, output_dir)
                    collect_and_save_logits(model, arch, seed, args, splits, stats, device, output_dir)
                except Exception as exc:
                    if arch == "efficientnet_b0":
                        print(f"Skipping efficientnet_b0 teacher because it failed: {exc}", flush=True)
                    else:
                        raise
    if args.stage in {"all", "student"}:
        train_student(args, splits, stats, device, output_dir)
    if args.stage in {"all", "evaluate"}:
        evaluate_final(args, splits, stats, device, output_dir)


if __name__ == "__main__":
    main()
