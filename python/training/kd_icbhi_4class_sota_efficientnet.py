#!/usr/bin/env python3
"""
ICBHI 2017 — Official 4-class Respiratory Event Classification
EfficientNet-B0 Teacher → Pure-CNN Student KD Pipeline

Task:  4-class (Normal / Crackle / Wheeze / Both) or 2-class (Normal / Abnormal)
Labels:  Per-cycle annotation from .txt files
Split:  Official ICBHI 60/40 patient-wise split
Primary metric:  ICBHI Score = (Sensitivity + Specificity) / 2

Pipeline:
  1) Train EfficientNet-B0 teacher(s) with gradual unfreeze
  2) Generate soft labels from teacher ensemble
  3) Distill to CNN6 student (FPGA-friendly)
  4) Evaluate all on official test set
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
import torchvision.models as models
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
    ICBHI_4CLASS_SOTA_EFFICIENTNET_ARTIFACTS_DIR,
    ensure_dir,
)

# =============================================================================
# ICBHI class definitions
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


def build_records(data_dir: Path, max_files: int | None = None, max_cycles: int | None = None) -> list[CycleRecord]:
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


def create_official_splits(records: list[CycleRecord], nc: int, val_frac: float = 0.2, seed: int = 42):
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
                 f_min, f_max, input_channels, bandpass, stats=None, augment=False,
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
        self.input_channels = input_channels
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
        t = torch.from_numpy(feat).unsqueeze(0).float()
        if self.input_channels == 3:
            t = t.repeat(3, 1, 1)
        return t, torch.tensor(get_label(r, self.nc), dtype=torch.long), r.sample_id

    def _load(self, wp):
        if wp not in self._cache:
            self._cache[wp] = load_audio(Path(wp), self.sr, self.bandpass, self.f_min, self.f_max)
        return self._cache[wp]


def create_dataset(records, args, stats, augment):
    return ICBHIDataset(
        records, args.num_classes, args.sample_rate, args.duration_sec,
        args.n_fft, args.win_length, args.hop_length, args.n_mels, args.target_frames,
        args.f_min, args.f_max, args.input_channels, not args.no_bandpass, stats, augment,
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
    if cnt == 0:
        raise ValueError("Empty set")
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


class StudentCNN6(nn.Module):
    def __init__(self, nc=4, dropout=0.2, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(input_channels, 16), ConvBlock(16, 16), nn.MaxPool2d(2),
            ConvBlock(16, 32), ConvBlock(32, 32), nn.MaxPool2d(2),
            ConvBlock(32, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, nc))
    def forward(self, x):
        return self.classifier(self.features(x))


class EfficientNetTeacher(nn.Module):
    def __init__(self, nc=4, dropout=0.3, pretrained=True, input_channels=3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        if input_channels == 1:
            conv = self.backbone.features[0][0]
            rep = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
            with torch.no_grad():
                rep.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
                if conv.bias is not None and rep.bias is not None:
                    rep.bias.copy_(conv.bias)
            self.backbone.features[0][0] = rep
        nf = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(nf, nc))

    def forward(self, x):
        return self.backbone(x)


def make_model(name, nc, args=None):
    if name == "cnn6":
        ic = args.input_channels if args else 1
        return StudentCNN6(nc=nc, input_channels=ic)
    if name == "efficientnet_b0":
        ic = args.input_channels if args else 3
        return EfficientNetTeacher(nc=nc, pretrained=not (args and args.no_pretrained), input_channels=ic)
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
    spec = per_class_specificity(cm)
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
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def init_wandb(args, output_dir, seed, param_count):
    if not args.wandb or args.wandb_mode == "disabled":
        return
    if wandb is None:
        raise ImportError("wandb not installed")
    rn = args.wandb_run_name or f"icbhi{args.num_classes}-effb0-{args.loss_mode}-seed-{seed}"
    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity or None, name=rn,
        dir=str(output_dir), mode=args.wandb_mode,
        config={**vars(args), "seed": seed, "total_params": param_count[0], "trainable_params": param_count[1]},
        tags=["icbhi-2017", f"{args.num_classes}class", "efficientnet-b0", "sota", args.loss_mode], reinit=True,
    )


def finish_wandb():
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def log_wandb(payload, step=None):
    if wandb is not None and wandb.run is not None and payload:
        wandb.log(payload, step=step)


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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def scalar_metrics(metrics, prefix):
    return {f"{prefix}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


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


def softmax_np(x, axis):
    s = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(s)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


# =============================================================================
# Teacher ensemble & KD
# =============================================================================
class TeacherEnsemble:
    def __init__(self, logits, sample_ids, nc, device="cpu"):
        lt = torch.as_tensor(logits, dtype=torch.float32)
        if lt.ndim != 3:
            raise ValueError(f"logits must be 3D, got {lt.shape}")
        if lt.shape[0] == len(sample_ids):
            lt = lt
        elif lt.shape[1] == len(sample_ids):
            lt = lt.permute(1, 0, 2).contiguous()
        else:
            raise ValueError(f"Shape mismatch: {lt.shape} for {len(sample_ids)} ids")
        self.logits = lt.to(device)
        self.s2i = {sid: i for i, sid in enumerate(sample_ids)}
        self.device = torch.device(device)

    def get_soft_labels_mean(self, sample_ids):
        idx = self._idx(sample_ids)
        return F.softmax(self.logits[idx].mean(dim=1), dim=-1)

    def _idx(self, sample_ids):
        return torch.tensor([self.s2i[s] for s in sample_ids], dtype=torch.long, device=self.device)


def soft_ce(student_logits, soft_labels):
    return -(soft_labels * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()


# =============================================================================
# Training loop
# =============================================================================
def train_one_seed(seed, args, output_dir, splits, stats, device):
    set_seed(seed)
    nc = args.num_classes
    train_ds = create_dataset(splits["train"], args, stats, True)
    val_ds = create_dataset(splits["val"], args, stats, False)
    test_ds = create_dataset(splits["test"], args, stats, False)

    # Balanced sampler
    sw = sample_weights(splits["train"], nc)
    sampler = WeightedRandomSampler(sw, len(splits["train"]), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Teacher ensemble (if KD)
    teacher = None
    if args.loss_mode in {"kd", "mixed"} and args.teacher_logits_dir:
        logits_path = Path(args.teacher_logits_dir) / "teacher_logits_train.npy"
        ids_path = Path(args.teacher_logits_dir) / "sample_ids_train.json"
        if logits_path.exists() and ids_path.exists():
            logits = np.load(logits_path)
            with ids_path.open() as f:
                ids = json.load(f)
            teacher = TeacherEnsemble(logits, ids, nc, device=device)

    # Class-balanced focal loss alpha
    cw = class_weights(splits["train"], nc, device)
    hard_criterion = FocalLoss(alpha=cw, gamma=args.focal_gamma, label_smoothing=args.label_smoothing).to(device)

    # Model
    model = make_model("efficientnet_b0", nc, args).to(device)
    pc = count_parameters(model)
    init_wandb(args, output_dir, seed, pc)
    print(f"seed={seed} params={pc[0]:,} trainable={pc[1]:,}", flush=True)

    # Gradual unfreeze: Phase 1 = head only, Phase 2 = top blocks, Phase 3 = full
    # Phase 1: Warmup head
    for p in model.backbone.features.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(args.warmup_epochs, args.epochs)

    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    best_path = ckpt_dir / f"seed_{seed}_best.pt"
    best_score = -float("inf")
    best_epoch = 0
    best_val_m = {}
    best_vt = np.array([], dtype=np.int64)
    best_vp = np.array([], dtype=np.int64)
    patience_counter = 0

    total_epochs = args.epochs
    for epoch in range(1, total_epochs + 1):
        # Phase transitions
        if epoch == warmup_epochs + 1:
            # Unfreeze top 50% of backbone
            features = list(model.backbone.features.children())
            mid = len(features) // 2
            for i, layer in enumerate(features):
                if i >= mid:
                    for p in layer.parameters():
                        p.requires_grad = True
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 0.3, weight_decay=args.weight_decay)
        elif epoch == warmup_epochs + (total_epochs - warmup_epochs) // 2 + 1:
            # Full unfreeze
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)

        model.train()
        total_loss = 0.0
        for feats, labels, sample_ids in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feats)
            hard_loss = hard_criterion(logits, labels)
            if teacher is not None:
                sl = teacher.get_soft_labels_mean(sample_ids)
                kd_loss = soft_ce(logits, sl)
                loss = args.alpha * kd_loss + (1.0 - args.alpha) * hard_loss if args.loss_mode == "mixed" else kd_loss
            else:
                loss = hard_loss
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * feats.size(0)

        val_m, vt, vp = evaluate_model(model, val_loader, device, nc)
        score = float(val_m[args.selection_metric])
        if score > best_score:
            best_score = score; best_epoch = epoch; best_val_m = val_m; best_vt = vt; best_vp = vp
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "seed": seed, "metrics": val_m, "args": vars(args)}, best_path)
        else:
            patience_counter += 1

        avg_loss = total_loss / max(len(train_loader.dataset), 1)
        log_wandb({"epoch": epoch, "train/loss": avg_loss, "train/lr": optimizer.param_groups[0]["lr"], **scalar_metrics(val_m, "val")}, step=epoch)
        print(f"seed={seed} ep={epoch:03d} loss={avg_loss:.4f} val_{args.selection_metric}={score:.4f} best={best_score:.4f} pat={patience_counter}/{args.patience}", flush=True)
        if patience_counter >= args.patience:
            break

    # Final test evaluation
    if splits["test"]:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_m, tt, tp = evaluate_model(model, test_loader, device, nc)
        save_metrics(metrics_dir, f"seed_{seed}_val_best", best_val_m, best_vt, best_vp, nc)
        save_metrics(metrics_dir, f"seed_{seed}_test", test_m, tt, tp, nc)
        log_wandb({**scalar_metrics(best_val_m, "best_val"), **scalar_metrics(test_m, "test")})
        log_wandb_cm(f"confusion/seed_{seed}_test", tt, tp, nc)
    else:
        test_m = {"icbhi_score": 0.0, "sensitivity": 0.0, "specificity": 0.0, "macro_f1": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0}
        save_metrics(metrics_dir, f"seed_{seed}_val_best", best_val_m, best_vt, best_vp, nc)
        log_wandb({**scalar_metrics(best_val_m, "best_val")})

    # Now distill to CNN6 student if teacher logits available or we self-distill
    # Generate logits from this teacher for student training
    student_results = None
    if args.distill_to_student:
        student_results = distill_to_cnn6(model, splits, args, stats, device, output_dir, seed, nc, hard_criterion)

    finish_wandb()
    summary = {
        "seed": seed, "best_epoch": best_epoch,
        "test_icbhi_score": float(test_m["icbhi_score"]), "test_sensitivity": float(test_m["sensitivity"]),
        "test_specificity": float(test_m["specificity"]), "test_macro_f1": float(test_m["macro_f1"]),
        "test_accuracy": float(test_m["accuracy"]), "test_balanced_accuracy": float(test_m["balanced_accuracy"]),
        "checkpoint": str(best_path),
    }
    if student_results:
        summary["student"] = student_results
    with (metrics_dir / f"seed_{seed}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def distill_to_cnn6(teacher, splits, args, stats, device, output_dir, seed, nc, hard_criterion):
    """Self-distill: use this teacher's logits to train CNN6 student."""
    print(f"  Distilling to CNN6 student (seed={seed})...", flush=True)
    teacher.eval()
    # Collect teacher logits for train set
    train_ds = create_dataset(splits["train"], args, stats, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logits_list = []
    with torch.no_grad():
        for feats, _, _ in train_loader:
            logits_list.append(teacher(feats.to(device)).cpu().numpy())
    teacher_logits = np.concatenate(logits_list, axis=0)  # [N, nc]
    teacher_logits = teacher_logits[np.newaxis, :]  # [1, N, nc] for single teacher

    # Train student with KD
    student = StudentCNN6(nc=nc, input_channels=args.input_channels).to(device)
    T = args.temperature
    alpha = args.alpha
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.student_epochs)

    # Balanced loader for student
    sw = sample_weights(splits["train"], nc)
    sampler = WeightedRandomSampler(sw, len(splits["train"]), replacement=True)
    train_ds_aug = create_dataset(splits["train"], args, stats, True)
    s_train_loader = DataLoader(train_ds_aug, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_ds = create_dataset(splits["val"], args, stats, False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_ds = create_dataset(splits["test"], args, stats, False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    tl_tensor = torch.tensor(teacher_logits, dtype=torch.float32, device=device)  # [1, N, nc]
    best_score = -float("inf")
    best_ep = 0
    s_dir = ensure_dir(output_dir / "student_checkpoints")
    s_best = s_dir / f"student_seed_{seed}_best.pt"

    for ep in range(1, args.student_epochs + 1):
        student.train()
        total_loss = 0.0
        for feats, labels, sample_ids in s_train_loader:
            feats, labels = feats.to(device), labels.to(device)
            # Get index for this batch - need to find indices
            # Since we use sampler, indices come from dataset order
            # We'll compute on-the-fly teacher inference instead
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
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * feats.size(0)
        scheduler.step()

        val_m, _, _ = evaluate_model(student, val_loader, device, nc)
        score = float(val_m[args.selection_metric])
        if score > best_score:
            best_score = score; best_ep = ep
            torch.save({"model_state": student.state_dict(), "epoch": ep}, s_best)
        if ep % 20 == 0:
            print(f"    Student ep={ep} val_{args.selection_metric}={score:.4f} best={best_score:.4f}", flush=True)

    # Test
    if splits["test"]:
        ckpt = torch.load(s_best, map_location=device)
        student.load_state_dict(ckpt["model_state"])
        test_m, tt, tp = evaluate_model(student, test_loader, device, nc)
        save_metrics(ensure_dir(output_dir / "metrics"), f"student_seed_{seed}_test", test_m, tt, tp, nc)
        print(f"  Student test: ICBHI={test_m['icbhi_score']:.4f} F1={test_m['macro_f1']:.4f} Acc={test_m['accuracy']:.4f}", flush=True)
        return {"test_icbhi_score": float(test_m["icbhi_score"]), "test_macro_f1": float(test_m["macro_f1"]),
                "test_accuracy": float(test_m["accuracy"]), "best_epoch": best_ep, "params": sum(p.numel() for p in student.parameters())}
    else:
        return {"test_icbhi_score": 0.0, "test_macro_f1": 0.0, "test_accuracy": 0.0, "best_epoch": best_ep, "params": sum(p.numel() for p in student.parameters())}


# =============================================================================
# Main
# =============================================================================
def prepare_run(args):
    od = ensure_dir(Path(args.output_dir) if args.output_dir else ICBHI_4CLASS_SOTA_EFFICIENTNET_ARTIFACTS_DIR)
    sp = od / "splits.json"
    if sp.exists() and (od / "config.json").exists() and not args.rebuild_splits:
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


def summarize_runs(od, summaries, nc):
    md = ensure_dir(od / "metrics")
    keys = ["test_icbhi_score", "test_sensitivity", "test_specificity", "test_macro_f1", "test_accuracy", "test_balanced_accuracy"]
    agg: dict = {"runs": summaries}
    for k in keys:
        vals = np.array([float(s.get(k, 0)) for s in summaries], dtype=np.float32)
        agg[f"{k}_mean"] = float(vals.mean()); agg[f"{k}_std"] = float(vals.std())
    with (md / "summary.json").open("w") as f:
        json.dump(agg, f, indent=2)


def parse_seeds(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def default_device(a):
    if a != "auto":
        return torch.device(a)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="ICBHI 2017 4-class EfficientNet teacher → CNN student (SOTA)")
    p.add_argument("--data_dir", default=str(ICBHI_2017_DIR))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--num_classes", type=int, choices=[2, 4], default=4)
    p.add_argument("--loss_mode", choices=["supervised", "kd", "mixed"], default="supervised")
    p.add_argument("--teacher_logits_dir", default=None, help="Dir with teacher_logits_train.npy + sample_ids_train.json")
    p.add_argument("--distill_to_student", action="store_true", default=True, help="Also train CNN6 student via self-distillation")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=8.0)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--target_frames", type=int, default=512)
    p.add_argument("--f_min", type=float, default=50.0)
    p.add_argument("--f_max", type=float, default=2500.0)
    p.add_argument("--input_channels", type=int, choices=[1, 3], default=3)
    p.add_argument("--no_bandpass", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--time_shift", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--freq_mask", type=int, default=16)
    p.add_argument("--time_mask", type=int, default=64)
    p.add_argument("--speed_perturb", action="store_true", default=True)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--student_epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_student", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--selection_metric", choices=["icbhi_score", "macro_f1", "balanced_accuracy"], default="icbhi_score")
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--max_stat_samples", type=int, default=512)
    p.add_argument("--rebuild_splits", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="icbhi-4class-sota-efficientnet")
    p.add_argument("--wandb_entity", default="vhieu4344")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    od, splits, stats = prepare_run(args)
    nc = args.num_classes; cn = get_class_names(nc)
    print(f"Task: {nc}-class ({', '.join(cn)})", flush=True)
    print(f"Output: {od}", flush=True)
    print(f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)
    for sn, recs in splits.items():
        labels = [get_label(r, nc) for r in recs]
        dist = {cn[i]: labels.count(i) for i in range(nc)}
        print(f"  {sn}: {dist}", flush=True)

    summaries = []
    for seed in parse_seeds(args.seeds):
        summaries.append(train_one_seed(seed, args, od, splits, stats, device))
    summarize_runs(od, summaries, nc)


if __name__ == "__main__":
    main()
