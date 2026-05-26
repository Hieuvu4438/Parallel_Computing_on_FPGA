#!/usr/bin/env python3
"""
ICBHI 2017 — Official 4-class Respiratory Event Classification
CNN-BiLSTM Teacher → Pure-CNN Student KD Pipeline

Task:  4-class (Normal / Crackle / Wheeze / Both) or 2-class (Normal / Abnormal)
Labels:  Per-cycle annotation from .txt files
Split:  Official ICBHI 60/40 patient-wise split
Primary metric:  ICBHI Score = (Sensitivity + Specificity) / 2

Pipeline:
  1) Train CNN-BiLSTM teacher(s) with attention pooling
  2) Online KD: student observes teacher logits in real time
  3) Evaluate on official test set

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
    ICBHI_2017_DIR,
    ICBHI_4CLASS_SOTA_BILSTM_ARTIFACTS_DIR,
    ensure_dir,
)

# =============================================================================
# ICBHI constants
# =============================================================================
CLASS_NAMES_4 = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_NAMES_2 = ["Normal", "Abnormal"]
OFFICIAL_TRAIN_PATIENTS = set(range(101, 161))
OFFICIAL_TEST_PATIENTS = set(range(161, 227))


def get_class_names(nc: int) -> list[str]:
    return CLASS_NAMES_4 if nc == 4 else CLASS_NAMES_2


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
    label_4class: int
    label_2class: int


@dataclass(frozen=True)
class FeatureStats:
    mean: float
    std: float


def event_to_4class(c: int, w: int) -> int:
    if c == 0 and w == 0:
        return 0
    if c == 1 and w == 0:
        return 1
    if c == 0 and w == 1:
        return 2
    return 3


def event_to_2class(c: int, w: int) -> int:
    return 0 if (c == 0 and w == 0) else 1


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
                s, e, c, w = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            if e > s:
                cycles.append((s, e, c, w))
    return cycles


def build_records(data_dir: Path, max_files=None, max_cycles=None) -> list[CycleRecord]:
    wavs = sorted(data_dir.glob("*.wav"))
    if max_files:
        wavs = wavs[:max_files]
    if not wavs:
        raise FileNotFoundError(f"No .wav in {data_dir}")
    records = []
    for wp in wavs:
        sid = wp.stem.split("_")[0]
        for ci, (s, e, c, w) in enumerate(read_cycle_annotations(wp)):
            records.append(CycleRecord(
                sample_id=f"{wp.stem}__cycle_{ci:03d}", wav_path=str(wp), subject_id=sid,
                start_sec=s, end_sec=e, crackle=c, wheeze=w,
                label_4class=event_to_4class(c, w), label_2class=event_to_2class(c, w),
            ))
            if max_cycles and len(records) >= max_cycles:
                return records
    if not records:
        raise ValueError("No annotated cycles found")
    return records


def create_official_splits(records, nc, val_frac=0.2, seed=42):
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
        s2l[r.subject_id] = get_label(r, nc)
    subjects = sorted(s2r)
    labels = [s2l[s] for s in subjects]
    counts = {lb: labels.count(lb) for lb in set(labels)}
    strat = labels if len(counts) > 1 and min(counts.values()) >= 2 else None
    try:
        tr_s, va_s = train_test_split(subjects, test_size=val_frac, random_state=seed, stratify=strat)
    except ValueError:
        tr_s, va_s = train_test_split(subjects, test_size=val_frac, random_state=seed)
    return {
        "train": [r for s in sorted(tr_s) for r in s2r[s]],
        "val": [r for s in sorted(va_s) for r in s2r[s]],
        "test": test_recs,
    }


# =============================================================================
# Audio I/O
# =============================================================================
def load_audio(wav_path, target_sr, bandpass, f_min, f_max):
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


def segment_waveform(audio, sr, start, end, target_samples):
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


# =============================================================================
# Mel spectrogram
# =============================================================================
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max):
    f_max = min(f_max, sr / 2 - 1)
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i:i + 3]
        c = max(c, l + 1); r = max(r, c + 1)
        for j in range(l, min(c, fb.shape[1])):
            fb[i, j] = (j - l) / max(c - l, 1)
        for j in range(c, min(r, fb.shape[1])):
            fb[i, j] = (r - j) / max(r - c, 1)
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


# =============================================================================
# Augmentation
# =============================================================================
def apply_waveform_aug(wf, shift, noise_std, speed_perturb):
    aug = wf.copy()
    if shift > 0:
        s = int(np.random.uniform(-shift, shift) * len(aug))
        aug = np.roll(aug, s)
    if noise_std > 0:
        rms = float(np.sqrt(np.mean(aug ** 2) + 1e-8))
        aug += np.random.normal(0, noise_std * rms, aug.shape).astype(np.float32)
    if speed_perturb and np.random.random() < 0.3:
        rate = np.random.choice([0.9, 1.0, 1.1])
        if rate != 1.0:
            orig = len(aug)
            aug = sp_signal.resample(aug, int(orig * rate)).astype(np.float32)
            if len(aug) > orig:
                aug = aug[:orig]
            else:
                aug = np.pad(aug, (0, orig - len(aug)))
    return aug.astype(np.float32)


def apply_specaugment(feat, freq_mask, time_mask):
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


# =============================================================================
# Dataset
# =============================================================================
class ICBHIDataset(Dataset):
    def __init__(self, records, nc, sr, dur, n_fft, win_len, hop_len, n_mels, target_frames,
                 f_min, f_max, bandpass, stats=None, augment=False,
                 time_shift=0.0, noise_std=0.0, freq_mask=0, time_mask=0, speed_perturb=False):
        self.records = records
        self.nc = nc
        self.sr = sr
        self.target_samples = int(round(dur * sr))
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
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
        self.fb = build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max)
        self._cache: dict[str, tuple[np.ndarray, int]] = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        wf, sr = self._load(r.wav_path)
        seg = segment_waveform(wf, sr, r.start_sec, r.end_sec, self.target_samples)
        if self.augment:
            seg = apply_waveform_aug(seg, self.time_shift, self.noise_std, self.speed_perturb)
        feat = compute_logmel(seg, sr, self.fb, self.n_fft, self.win_len, self.hop_len, self.target_frames)
        if self.augment:
            feat = apply_specaugment(feat, self.freq_mask, self.time_mask)
        if self.stats:
            feat = (feat - self.stats.mean) / max(self.stats.std, 1e-6)
        return (torch.from_numpy(feat).unsqueeze(0).float(),
                torch.tensor(get_label(r, self.nc), dtype=torch.long),
                torch.tensor(idx, dtype=torch.long))

    def _load(self, wp):
        if wp not in self._cache:
            self._cache[wp] = load_audio(Path(wp), self.sr, self.bandpass, self.f_min, self.f_max)
        return self._cache[wp]


def create_dataset(records, args, stats, augment):
    return ICBHIDataset(
        records, args.num_classes, args.sample_rate, args.duration_sec,
        args.n_fft, args.win_length, args.hop_length, args.n_mels, args.target_frames,
        args.f_min, args.f_max, not args.no_bandpass, stats, augment,
        args.time_shift if augment else 0, args.noise_std if augment else 0,
        args.freq_mask if augment else 0, args.time_mask if augment else 0,
        args.speed_perturb if augment else False,
    )


def estimate_feature_stats(records, args):
    ds = create_dataset(records, args, None, False)
    limit = min(len(ds), args.max_stat_samples)
    ts, tsq, cnt = 0.0, 0.0, 0
    for i in range(limit):
        f, _, _ = ds[i]
        v = f.float()
        ts += float(v.sum()); tsq += float((v * v).sum()); cnt += v.numel()
    mean = ts / cnt
    return FeatureStats(mean=float(mean), std=float(math.sqrt(max(tsq / cnt - mean * mean, 1e-12))))


# =============================================================================
# Models
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x):
        return self.block(x)


class CNNBiLSTMTeacher(nn.Module):
    """CNN feature extractor + BiLSTM temporal modeling + attention pooling."""
    def __init__(self, nc=4, dropout=0.3, n_mels=128, hidden_size=128, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 32), nn.MaxPool2d(2),
            ConvBlock(32, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
            ConvBlock(64, 128), ConvBlock(128, 128), nn.MaxPool2d((2, 1)),
        )
        # After CNN: n_mels//8 freq bins, same time frames
        cnn_freq = n_mels // 8
        cnn_channels = 128
        lstm_input = cnn_channels * cnn_freq
        self.lstm = nn.LSTM(
            input_size=lstm_input, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.Tanh(), nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size * 2, nc),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n_mels, T]
        features = self.cnn(x)  # [B, 128, n_mels//8, T']
        B, C, freq_dim, time_dim = features.shape
        features = features.permute(0, 3, 1, 2).reshape(B, time_dim, C * freq_dim)  # [B, T', C*F]
        lstm_out, _ = self.lstm(features)  # [B, T', 2*hidden]
        # Attention pooling
        attn_weights = self.attention(lstm_out).squeeze(-1)  # [B, T']
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # [B, 1, T']
        context = torch.bmm(attn_weights, lstm_out).squeeze(1)  # [B, 2*hidden]
        return self.classifier(context)


class StudentCNN6(nn.Module):
    """Lightweight pure CNN student — FPGA-friendly."""
    def __init__(self, nc=4, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16), ConvBlock(16, 16), nn.MaxPool2d(2),
            ConvBlock(16, 32), ConvBlock(32, 32), nn.MaxPool2d(2),
            ConvBlock(32, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, nc))

    def forward(self, x):
        return self.classifier(self.features(x))


class StudentWideCNN(nn.Module):
    """Wider CNN student for higher capacity."""
    def __init__(self, nc=4, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 32), nn.MaxPool2d(2),
            ConvBlock(32, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
            ConvBlock(64, 128), ConvBlock(128, 128), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, nc))

    def forward(self, x):
        return self.classifier(self.features(x))


def make_model(name, nc, args=None):
    if name == "cnn_bilstm":
        n_mels = args.n_mels if args else 128
        hidden = args.lstm_hidden if args else 128
        layers = args.lstm_layers if args else 2
        return CNNBiLSTMTeacher(nc=nc, n_mels=n_mels, hidden_size=hidden, num_layers=layers)
    if name == "cnn6":
        return StudentCNN6(nc=nc)
    if name == "cnn6_wide":
        return StudentWideCNN(nc=nc)
    raise ValueError(f"Unknown model: {name}")


# =============================================================================
# Loss & Metrics
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ls = label_smoothing
    def forward(self, logits, targets):
        nc = logits.size(1)
        if self.ls > 0:
            tp = torch.full_like(logits, self.ls / max(nc - 1, 1))
            tp.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        else:
            tp = F.one_hot(targets, nc).float()
        lp = F.log_softmax(logits, dim=1)
        p = lp.exp()
        loss = -((1.0 - p) ** self.gamma) * tp * lp
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).view(1, -1)
        return loss.sum(dim=1).mean()


def class_weights(records, nc, device):
    labels = [get_label(r, nc) for r in records]
    c = np.bincount(labels, minlength=nc).astype(np.float32)
    w = c.sum() / np.maximum(c * nc, 1.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def sample_weights(records, nc):
    labels = [get_label(r, nc) for r in records]
    c = np.bincount(labels, minlength=nc).astype(np.float64)
    w = c.sum() / np.maximum(c * nc, 1.0)
    return torch.tensor([w[lb] for lb in labels], dtype=torch.double)


def per_class_specificity(cm):
    total = cm.sum()
    return np.array([cm[i, i] for i in range(cm.shape[0])], dtype=np.float32)  # placeholder - compute below
    vals = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp; tn = total - tp - fp - fn
        vals.append(tn / max(tn + fp, 1))
    return np.array(vals, dtype=np.float32)


def _per_class_specificity(cm):
    total = cm.sum()
    vals = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp; tn = total - tp - fp - fn
        vals.append(tn / max(tn + fp, 1))
    return np.array(vals, dtype=np.float32)


def icbhi_score(y_true, y_pred, nc):
    nm = y_true == 0; am = y_true != 0
    sp = float(np.mean(y_pred[nm] == 0)) if nm.any() else 0.0
    se = float(np.mean(y_pred[am] != 0)) if am.any() else 0.0
    return se, sp, (se + sp) / 2.0


def safe_auc(y_true, y_prob, nc):
    try:
        if len(np.unique(y_true)) < nc:
            return None
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(nc))))
    except ValueError:
        return None


def compute_metrics(y_true, y_pred, y_prob, nc):
    cn = get_class_names(nc)
    labels = list(range(nc))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    spec = _per_class_specificity(cm)
    se, sp, score = icbhi_score(y_true, y_pred, nc)
    m: dict = {
        "icbhi_score": float(score), "sensitivity": float(se), "specificity": float(sp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }
    if y_prob is not None:
        m["auc_ovr"] = safe_auc(y_true, y_prob, nc)
    for i, c in enumerate(cn):
        k = c.lower().replace(" ", "_")
        m[f"{k}_precision"] = float(prec[i]); m[f"{k}_recall"] = float(rec[i])
        m[f"{k}_f1"] = float(f1[i]); m[f"{k}_specificity"] = float(spec[i])
        m[f"{k}_support"] = int(sup[i])
    return m


# =============================================================================
# W&B / Save / Load
# =============================================================================
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def init_wandb(args, od):
    if not args.wandb or args.wandb_mode == "disabled":
        return
    if wandb is None:
        return
    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity or None,
        name=args.wandb_run_name or f"icbhi{args.num_classes}-bilstm-to-cnn",
        dir=str(od), mode=args.wandb_mode, config=vars(args),
        tags=["icbhi-2017", f"{args.num_classes}class", "cnn-bilstm", "knowledge-distillation", "sota"],
    )


def finish_wandb():
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def log_wandb(payload, step=None):
    if wandb is not None and wandb.run is not None:
        clean = {k: v for k, v in payload.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if clean:
            wandb.log(clean, step=step)


def log_wandb_cm(name, y_true, y_pred, nc):
    if wandb is None or wandb.run is None:
        return
    wandb.log({name: wandb.plot.confusion_matrix(probs=None, y_true=y_true.tolist(), preds=y_pred.tolist(), class_names=get_class_names(nc))})


def save_split_records(od, splits):
    with (od / "splits.json").open("w") as f:
        json.dump({s: [asdict(r) for r in recs] for s, recs in splits.items()}, f, indent=2)


def load_split_records(od):
    with (od / "splits.json").open("r") as f:
        raw = json.load(f)
    return {s: [CycleRecord(**r) for r in recs] for s, recs in raw.items()}


def save_config(od, args, stats, records, splits):
    nc = args.num_classes; cn = get_class_names(nc)
    lc = {c: 0 for c in cn}
    for r in records:
        lc[cn[get_label(r, nc)]] += 1
    cfg = vars(args).copy()
    cfg.update({"feature_mean": stats.mean, "feature_std": stats.std, "class_names": cn, "label_counts": lc,
                "split_sizes": {s: len(recs) for s, recs in splits.items()}, "unique_patients": len({r.subject_id for r in records})})
    with (od / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2)


def load_feature_stats(od):
    with (od / "config.json").open("r") as f:
        c = json.load(f)
    return FeatureStats(mean=float(c["feature_mean"]), std=float(c["feature_std"]))


def save_metrics(od, name, metrics, y_true, y_pred, nc):
    md = ensure_dir(od / "metrics")
    with (md / f"{name}.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    cn = get_class_names(nc)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(nc)))
    with (md / f"confusion_matrix_{name}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["true/pred", *cn])
        for i, row in enumerate(cm):
            w.writerow([cn[i], *row.tolist()])


def count_params(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Evaluate
# =============================================================================
def evaluate_model(model, loader, device, nc):
    model.eval()
    yt, yp, yprob = [], [], []
    with torch.no_grad():
        for feats, labels, _ in loader:
            logits = model(feats.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            yprob.extend(probs.tolist())
            yp.extend(probs.argmax(axis=1).tolist())
            yt.extend(labels.numpy().tolist())
    return compute_metrics(np.array(yt), np.array(yp), np.array(yprob), nc), np.array(yt), np.array(yp)


# =============================================================================
# Training loops
# =============================================================================
def train_teacher(args, splits, stats, device, output_dir):
    """Train CNN-BiLSTM teacher model."""
    nc = args.num_classes
    train_ds = create_dataset(splits["train"], args, stats, True)
    val_ds = create_dataset(splits["val"], args, stats, False)

    sw = sample_weights(splits["train"], nc)
    sampler = WeightedRandomSampler(sw, len(splits["train"]), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = make_model("cnn_bilstm", nc, args).to(device)
    tp, tt = count_params(model)
    print(f"Teacher CNN-BiLSTM: {tp:,} params ({tt:,} trainable)", flush=True)

    cw = class_weights(splits["train"], nc, device)
    criterion = FocalLoss(alpha=cw, gamma=args.focal_gamma, label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_teacher, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_teacher)

    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    best_path = ckpt_dir / "teacher_best.pt"
    best_score = -float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs_teacher + 1):
        model.train()
        total_loss = 0.0
        for feats, labels, _ in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * feats.size(0)
        scheduler.step()

        val_m, _, _ = evaluate_model(model, val_loader, device, nc)
        score = float(val_m[args.selection_metric])
        if score > best_score:
            best_score = score; best_epoch = epoch; patience_counter = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": val_m}, best_path)
        else:
            patience_counter += 1

        avg_loss = total_loss / max(len(train_loader.dataset), 1)
        log_wandb({f"teacher/epoch": epoch, "teacher/train_loss": avg_loss, **{f"teacher/val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}, step=epoch)
        if epoch % 10 == 0 or epoch == 1:
            print(f"T ep={epoch:03d} loss={avg_loss:.4f} val_{args.selection_metric}={score:.4f} best={best_score:.4f}", flush=True)
        if patience_counter >= args.patience:
            print(f"Teacher early stop at ep={epoch}", flush=True)
            break

    # Load best and return
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Teacher best: ep={best_epoch} {args.selection_metric}={best_score:.4f}", flush=True)
    return model


def online_kd_student(teacher, args, splits, stats, device, output_dir):
    """Online KD: student observes teacher logits during training."""
    nc = args.num_classes
    train_ds = create_dataset(splits["train"], args, stats, True)
    val_ds = create_dataset(splits["val"], args, stats, False)
    test_ds = create_dataset(splits["test"], args, stats, False)

    sw = sample_weights(splits["train"], nc)
    sampler = WeightedRandomSampler(sw, len(splits["train"]), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    student = make_model(args.student_arch, nc).to(device)
    sp, st = count_params(student)
    print(f"Student {args.student_arch}: {sp:,} params ({st:,} trainable)", flush=True)

    cw = class_weights(splits["train"], nc, device)
    hard_criterion = FocalLoss(alpha=cw, gamma=args.focal_gamma, label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_student)

    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    best_path = ckpt_dir / "student_best.pt"
    best_score = -float("inf")
    best_epoch = 0
    patience_counter = 0
    T = args.temperature
    alpha = args.alpha

    teacher.eval()
    for epoch in range(1, args.epochs_student + 1):
        student.train()
        total_loss = 0.0
        for feats, labels, _ in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            with torch.no_grad():
                t_logits = teacher(feats)
            optimizer.zero_grad(set_to_none=True)
            s_logits = student(feats)
            # KD loss
            t_probs = F.softmax(t_logits / T, dim=1)
            s_log_probs = F.log_softmax(s_logits / T, dim=1)
            soft_loss = -(t_probs * s_log_probs).sum(dim=1).mean() * (T ** 2)
            hard_loss = hard_criterion(s_logits, labels)
            loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * feats.size(0)
        scheduler.step()

        val_m, _, _ = evaluate_model(student, val_loader, device, nc)
        score = float(val_m[args.selection_metric])
        if score > best_score:
            best_score = score; best_epoch = epoch; patience_counter = 0
            torch.save({"model_state": student.state_dict(), "epoch": epoch, "metrics": val_m}, best_path)
        else:
            patience_counter += 1

        avg_loss = total_loss / max(len(train_loader.dataset), 1)
        log_wandb({f"student/epoch": epoch, "student/train_loss": avg_loss, **{f"student/val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}, step=epoch + args.epochs_teacher)
        if epoch % 10 == 0 or epoch == 1:
            print(f"S ep={epoch:03d} loss={avg_loss:.4f} val_{args.selection_metric}={score:.4f} best={best_score:.4f}", flush=True)
        if patience_counter >= args.patience:
            print(f"Student early stop at ep={epoch}", flush=True)
            break

    # Load best and evaluate on test
    ckpt = torch.load(best_path, map_location=device)
    student.load_state_dict(ckpt["model_state"])

    # Teacher and student test
    if splits["test"]:
        t_test_m, t_yt, t_yp = evaluate_model(teacher, test_loader, device, nc)
        save_metrics(ensure_dir(output_dir / "metrics"), "teacher_test", t_test_m, t_yt, t_yp, nc)
        log_wandb({f"teacher/test_{k}": v for k, v in t_test_m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        log_wandb_cm("confusion/teacher_test", t_yt, t_yp, nc)
        print(f"\nTeacher test: ICBHI={t_test_m['icbhi_score']:.4f} SE={t_test_m['sensitivity']:.4f} SP={t_test_m['specificity']:.4f} F1={t_test_m['macro_f1']:.4f}", flush=True)

        s_test_m, s_yt, s_yp = evaluate_model(student, test_loader, device, nc)
        save_metrics(ensure_dir(output_dir / "metrics"), "student_test", s_test_m, s_yt, s_yp, nc)
        log_wandb({f"student/test_{k}": v for k, v in s_test_m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        log_wandb_cm("confusion/student_test", s_yt, s_yp, nc)
        print(f"Student test: ICBHI={s_test_m['icbhi_score']:.4f} SE={s_test_m['sensitivity']:.4f} SP={s_test_m['specificity']:.4f} F1={s_test_m['macro_f1']:.4f}", flush=True)
    else:
        print("\nTest split is empty, skipping test set evaluation", flush=True)
        t_test_m = {"icbhi_score": 0.0, "sensitivity": 0.0, "specificity": 0.0, "macro_f1": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0}
        s_test_m = {"icbhi_score": 0.0, "sensitivity": 0.0, "specificity": 0.0, "macro_f1": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0}

    # Summary
    summary = {
        "teacher": {"params": count_params(teacher)[0], "test": t_test_m},
        "student": {"arch": args.student_arch, "params": count_params(student)[0], "best_epoch": best_epoch, "test": s_test_m},
        "kd_config": {"temperature": T, "alpha": alpha, "selection_metric": args.selection_metric},
    }
    with (ensure_dir(output_dir / "metrics") / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison
    print("\n" + "=" * 70)
    print(f"{'Metric':<25s} {'Teacher':>12s} {'Student':>12s} {'Δ':>10s}")
    print("-" * 70)
    for k in ["icbhi_score", "sensitivity", "specificity", "macro_f1", "accuracy", "balanced_accuracy"]:
        tv = float(t_test_m.get(k, 0)); sv = float(s_test_m.get(k, 0))
        print(f"{k:<25s} {tv*100:>11.2f}% {sv*100:>11.2f}% {(sv-tv)*100:>+9.2f}%")
    print("=" * 70)

    return student, summary


# =============================================================================
# Main
# =============================================================================
def prepare_run(args):
    od = ensure_dir(Path(args.output_dir) if args.output_dir else ICBHI_4CLASS_SOTA_BILSTM_ARTIFACTS_DIR)
    sp = od / "splits.json"
    if sp.exists() and (od / "config.json").exists():
        with (od / "config.json").open("r") as f:
            cached_cfg = json.load(f)
        if (cached_cfg.get("max_files") == args.max_files and
            cached_cfg.get("max_cycles") == args.max_cycles and
            cached_cfg.get("num_classes") == args.num_classes):
            return od, load_split_records(od), load_feature_stats(od)
        else:
            print("Configuration or data split constraints changed. Rebuilding splits...", flush=True)
    records = build_records(Path(args.data_dir), args.max_files, args.max_cycles)
    splits = create_official_splits(records, args.num_classes, args.val_size, args.seed)
    stats = estimate_feature_stats(splits["train"], args)
    save_split_records(od, splits)
    save_config(od, args, stats, records, splits)
    return od, splits, stats


def parse_args():
    p = argparse.ArgumentParser(description="ICBHI 2017 4-class CNN-BiLSTM → CNN student (SOTA)")
    p.add_argument("--data_dir", default=str(ICBHI_2017_DIR))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--num_classes", type=int, choices=[2, 4], default=4)
    p.add_argument("--student_arch", choices=["cnn6", "cnn6_wide"], default="cnn6")
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
    # Teacher
    p.add_argument("--lstm_hidden", type=int, default=128)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--epochs_teacher", type=int, default=100)
    p.add_argument("--lr_teacher", type=float, default=1e-3)
    # Student
    p.add_argument("--epochs_student", type=int, default=120)
    p.add_argument("--lr_student", type=float, default=1e-3)
    # KD
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5)
    # Loss
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--selection_metric", choices=["icbhi_score", "macro_f1", "balanced_accuracy"], default="icbhi_score")
    # Split
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    # System
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--max_stat_samples", type=int, default=512)
    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="icbhi-4class-sota-bilstm")
    p.add_argument("--wandb_entity", default="vhieu4344")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    od, splits, stats = prepare_run(args)
    init_wandb(args, od)
    nc = args.num_classes; cn = get_class_names(nc)
    print(f"Task: {nc}-class ({', '.join(cn)})", flush=True)
    print(f"Output: {od}", flush=True)
    print(f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)
    for sn, recs in splits.items():
        labels = [get_label(r, nc) for r in recs]
        dist = {cn[i]: labels.count(i) for i in range(nc)}
        print(f"  {sn}: {dist}", flush=True)

    # Phase 1: Train teacher
    print("\n" + "=" * 50 + " TEACHER " + "=" * 50, flush=True)
    teacher = train_teacher(args, splits, stats, device, od)

    # Phase 2: Online KD → student
    print("\n" + "=" * 50 + " STUDENT (KD) " + "=" * 50, flush=True)
    student, summary = online_kd_student(teacher, args, splits, stats, device, od)

    # Export best student
    ckpt_dir = ensure_dir(od / "checkpoints")
    ckpt = torch.load(ckpt_dir / "student_best.pt", map_location="cpu")
    student.load_state_dict(ckpt["model_state"])
    student.eval()
    torch.save(student.state_dict(), od / "student_final.pt")

    # ONNX export
    try:
        dummy = torch.randn(1, 1, args.n_mels, args.target_frames)
        torch.onnx.export(
            student.cpu(), dummy, str(od / "student_final.onnx"),
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=["mel_spectrogram"], output_names=["logits"],
            dynamic_axes={"mel_spectrogram": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"ONNX exported: {od / 'student_final.onnx'}", flush=True)
    except Exception as e:
        print(f"ONNX export failed: {e}", flush=True)

    finish_wandb()
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
