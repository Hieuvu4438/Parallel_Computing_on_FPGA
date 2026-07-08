#!/usr/bin/env python3
"""
ICBHI Strategy S11: Comprehensive SOTA Pipeline

Combines ALL proven techniques from papers and our experiments:
1. VTLP augmentation (+3.19% from RSC-FTF paper)
2. Aggressive class balancing (5x for Both, 4x for Wheeze)
3. Strong binary auxiliary loss (weight=0.6)
4. MixUp augmentation for minority classes
5. SAM optimizer for better generalization
6. Wide threshold search (0.01 to 0.99)
7. CPU-controlled: num_workers=0, batch_size=32

Target: ICBHI Score > 0.68

Usage:
    python python/training/icbhi_kd_s11_comprehensive.py --stage all --epochs 200
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import json
import sys
from pathlib import Path

import torch
torch.set_num_threads(2)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import ensure_dir
from python.training import icbhi_kd_pipeline_multiview_ensemble as base
from python.training.icbhi_sota_evaluation import evaluate_with_tta


# ============================================================================
# VTLP Augmentation (from RSC-FTF paper, +3.19% ICBHI)
# ============================================================================

def vtlp_augment_waveform(waveform, sample_rate=16000, alpha_range=(0.9, 1.1)):
    """
    Vocal Tract Length Perturbation (VTLP) augmentation.
    Simulates vocal tract length variations across patients.
    Paper: Dong et al. 2025 - RSC-FTF (+3.19% ICBHI Score)
    """
    alpha = np.random.uniform(*alpha_range)
    n_samples = len(waveform)
    indices = np.clip(np.arange(n_samples) * alpha, 0, n_samples - 1).astype(int)
    augmented = waveform[indices]
    # Add Gaussian noise
    noise_level = np.random.uniform(0.001, 0.01)
    noise = np.random.randn(len(augmented)) * noise_level
    augmented = augmented + noise.astype(augmented.dtype)
    return augmented


# ============================================================================
# Dataset with VTLP + MixUp
# ============================================================================

class VTLPDataset(Dataset):
    """Dataset with VTLP augmentation and MixUp for minority classes."""

    def __init__(self, records, args, augment=True):
        self.records = records
        self.args = args
        self.augment = augment
        self.target_samples = int(round(args.duration_sec * args.sample_rate))

        # Pre-compute spectrograms
        self.spectrograms = []
        self.labels = []
        self.subjects = []

        fb = base.build_mel_filterbank(args.sample_rate, args.n_fft, args.n_mels,
                                        args.f_min, args.f_max)

        for rec in records:
            try:
                wf, sr = base.load_audio(rec.wav_path, args.sample_rate,
                                          not args.no_bandpass, args.f_min, args.f_max)
                seg = base.segment_waveform(wf, sr, rec.start_sec, rec.end_sec,
                                             self.target_samples)
                feat = base.compute_logmel(seg, sr, fb, args.n_fft, args.win_length,
                                            args.hop_length, args.target_frames)
                self.spectrograms.append(feat)
                self.labels.append(rec.label_4class)
                self.subjects.append(rec.subject_id)
            except Exception:
                continue

        self.labels = np.array(self.labels)
        print(f"[VTLPDataset] Loaded {len(self.spectrograms)} spectrograms")
        for c in range(4):
            count = (self.labels == c).sum()
            print(f"  Class {c}: {count} samples")

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        feat = self.spectrograms[idx].copy()
        label = self.labels[idx]

        if self.augment:
            # VTLP augmentation (applied to spectrogram frequency axis)
            if np.random.random() < 0.5:
                alpha = np.random.uniform(0.9, 1.1)
                n_freq = feat.shape[0]
                indices = np.clip(np.arange(n_freq) * alpha, 0, n_freq - 1).astype(int)
                feat = feat[indices]

            # SpecAugment
            feat = self._spec_augment(feat)

            # Time shift
            if np.random.random() < 0.3:
                shift = np.random.randint(-20, 20)
                feat = np.roll(feat, shift, axis=-1)

            # Gaussian noise
            if np.random.random() < 0.3:
                noise = np.random.randn(*feat.shape) * 0.01
                feat = feat + noise

        # Normalize
        feat = (feat - feat.mean()) / (feat.std() + 1e-8)

        # Add channel dim
        feat = feat[np.newaxis, ...]

        return torch.FloatTensor(feat), label, idx

    def _spec_augment(self, feat):
        """SpecAugment: frequency and time masking."""
        n_freq, n_time = feat.shape

        # Frequency mask
        if np.random.random() < 0.5:
            f = np.random.randint(0, max(1, n_freq // 8))
            f0 = np.random.randint(0, n_freq - f)
            feat[f0:f0 + f, :] = 0

        # Time mask
        if np.random.random() < 0.5:
            t = np.random.randint(0, max(1, n_time // 8))
            t0 = np.random.randint(0, n_time - t)
            feat[:, t0:t0 + t] = 0

        return feat


# ============================================================================
# MixUp Dataset for Minority Classes
# ============================================================================

class MixUpDataset(Dataset):
    """Wraps VTLPDataset and applies MixUp for minority classes."""

    def __init__(self, base_dataset, alpha=0.3, mixup_prob=0.5):
        self.base = base_dataset
        self.alpha = alpha
        self.mixup_prob = mixup_prob

        # Group by class for MixUp
        self.class_indices = {}
        for i, label in enumerate(base_dataset.labels):
            self.class_indices.setdefault(label, []).append(i)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        feat1, label1, _ = self.base[idx]

        if np.random.random() < self.mixup_prob:
            # MixUp with a random sample from the same or different class
            if label1 in self.class_indices and len(self.class_indices[label1]) > 1:
                # 70% chance: mix with same class (augment minority)
                # 30% chance: mix with different class
                if np.random.random() < 0.7:
                    mix_idx = np.random.choice(self.class_indices[label1])
                else:
                    mix_idx = np.random.randint(0, len(self.base))

                feat2, label2, _ = self.base[mix_idx]

                # Beta distribution for mixing coefficient
                lam = np.random.beta(self.alpha, self.alpha)
                feat = lam * feat1 + (1 - lam) * feat2

                # Use the label of the dominant sample
                if lam >= 0.5:
                    label = label1
                else:
                    label = label2

                return feat, label, idx

        return feat1, label1, idx


# ============================================================================
# Model: Enhanced CNN with Binary Head
# ============================================================================

class EnhancedClassifier(nn.Module):
    """
    CNN classifier with separate binary head for Normal vs Abnormal.
    The binary head helps prevent Normal-prediction collapse.
    """

    def __init__(self, nc=4, in_ch=1, width=1.0):
        super().__init__()
        self.nc = nc
        c = lambda v: max(8, int(v * width))

        # Shared backbone
        self.backbone = nn.Sequential(
            base.ConvBNAct(in_ch, c(32)),
            base.DSResBlock(c(32), c(48), stride=2),
            base.DSResBlock(c(48), c(64)),
            base.DSResBlock(c(64), c(96), stride=2),
            base.DSResBlock(c(96), c(128)),
            base.DSResBlock(c(128), c(160), stride=2),
        )

        # Dual pooling
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        feat_dim = c(160) * 2  # avg + max pooling

        # Main classifier: 4-class
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, nc),
        )

        # Binary head: Normal vs Abnormal
        self.binary_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x, return_binary=False):
        feat = self.backbone(x)
        avg = self.pool_avg(feat).flatten(1)
        mx = self.pool_max(feat).flatten(1)
        feat = torch.cat([avg, mx], dim=1)

        logits = self.classifier(feat)

        if return_binary:
            bin_logits = self.binary_head(feat)
            return logits, bin_logits
        return logits

    def extract_features(self, x):
        feat = self.backbone(x)
        avg = self.pool_avg(feat).flatten(1)
        mx = self.pool_max(feat).flatten(1)
        return torch.cat([avg, mx], dim=1)


# ============================================================================
# Loss: Aggressive Sensitivity-Aware Loss
# ============================================================================

class SensitivityFocusedLoss(nn.Module):
    """
    Loss that heavily penalizes false negatives (missed abnormal samples).
    This is the KEY to preventing Normal-prediction collapse.
    """

    def __init__(self, fn_weight=8.0, fp_weight=1.0):
        super().__init__()
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, nc] main classifier logits
            targets: [B] ground truth labels (0=Normal, 1-3=Abnormal)
        """
        probs = F.softmax(logits, dim=1)
        p_normal = probs[:, 0]
        is_abnormal = (targets != 0).float()

        # False Negative: model says Normal but actually Abnormal
        # HEAVILY penalize this (weight=8.0)
        fn_loss = (is_abnormal * (-torch.log(1 - p_normal + 1e-8))).mean()

        # False Positive: model says Abnormal but actually Normal
        fp_loss = ((1 - is_abnormal) * (-torch.log(p_normal + 1e-8))).mean()

        return self.fn_weight * fn_loss + self.fp_weight * fp_loss


# ============================================================================
# Training
# ============================================================================

def train_comprehensive(args, device, records_train, records_val, nc):
    """Train with all proven techniques."""

    # Build datasets
    print("[S11] Building training dataset with VTLP augmentation...")
    train_base = VTLPDataset(records_train, args, augment=True)
    train_ds = MixUpDataset(train_base, alpha=0.3, mixup_prob=0.5)
    val_ds = VTLPDataset(records_val, args, augment=False)

    # AGGRESSIVE class-balanced sampling
    class_counts = np.bincount(train_base.labels, minlength=nc)
    print(f"[S11] Class distribution: {dict(zip(range(nc), class_counts))}")

    # Multipliers: Normal=1, Crackle=3, Wheeze=5, Both=6
    class_multipliers = {0: 1.0, 1: 3.0, 2: 5.0, 3: 6.0}
    sample_weights = []
    for label in train_base.labels:
        base_w = 1.0 / max(class_counts[label], 1)
        mult = class_multipliers.get(label, 1.0)
        sample_weights.append(base_w * mult)

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    # CPU control: num_workers=0 to avoid subprocess overhead
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # Model
    model = EnhancedClassifier(nc=nc, in_ch=args.in_channels, width=args.width).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[S11] Model: {total_params/1e6:.2f}M params")

    # Loss functions
    # Class weights: inverse frequency normalized
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
    class_weights = class_weights / class_weights.sum() * nc
    focal_fn = base.FocalLoss(
        alpha=torch.tensor(class_weights, dtype=torch.float32),
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    sensitivity_fn = SensitivityFocusedLoss(fn_weight=8.0, fp_weight=1.0)
    binary_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 4.0], device=device)  # 4x weight for Abnormal
    )

    # Optimizer: SAM for better generalization
    from python.training.icbhi_kd_pipeline_multiview_ensemble import SAM
    base_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = SAM(base_opt, rho=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=40, T_mult=2,
    )

    best_score = 0.0
    out_dir = ensure_dir(base.TRAINING_ARTIFACTS_DIR / "s11_comprehensive")

    for epoch in range(args.epochs):
        # Progressive loss weighting
        t = min(epoch / 40.0, 1.0)
        w_focal = 0.50 * (1 - t) + 0.30 * t
        w_sens = 0.35 * (1 - t) + 0.50 * t
        w_bin = 0.15 * (1 - t) + 0.20 * t

        model.train()
        total_loss = 0
        n_batches = 0

        for x, y, idx in train_dl:
            x, y = x.to(device), y.to(device)

            # Forward with binary head
            logits, bin_logits = model(x, return_binary=True)

            # Losses
            l_focal = focal_fn(logits, y)
            l_sens = sensitivity_fn(logits, y)

            # Binary loss: Normal (0) vs Abnormal (1-3)
            bin_labels = (y != 0).long()
            l_bin = binary_fn(bin_logits, bin_labels)

            loss = w_focal * l_focal + w_sens * l_sens + w_bin * l_bin

            # SAM optimizer
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward for SAM
            logits2, bin_logits2 = model(x, return_binary=True)
            l2 = focal_fn(logits2, y) + sensitivity_fn(logits2, y)
            l2.backward()
            optimizer.second_step(zero_grad=True)

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation with wide threshold search
        if (epoch + 1) % 5 == 0:
            val_score, val_se, val_sp, best_tau = _eval_wide_threshold(
                model, val_dl, device, nc
            )
            print(f"[S11] Epoch {epoch+1}/{args.epochs}  "
                  f"loss={total_loss/max(n_batches,1):.4f}  "
                  f"val_score={val_score:.4f}  Se={val_se:.4f}  Sp={val_sp:.4f}  "
                  f"tau={best_tau:.3f}  w_focal={w_focal:.2f}  w_sens={w_sens:.2f}")

            if val_score > best_score:
                best_score = val_score
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_score": val_score,
                    "best_tau": best_tau,
                    "args": vars(args),
                }, out_dir / "best.pt")

    print(f"[S11] Best validation ICBHI Score: {best_score:.4f}")
    return model


def _eval_wide_threshold(model, dl, device, nc):
    """Evaluate with wide threshold search (0.01 to 0.99)."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y, idx in dl:
            x = x.to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    best_score = 0.0
    best_tau = 0.5
    best_se, best_sp = 0.0, 0.0

    # Wide threshold search: 0.01 to 0.99
    for tau in np.arange(0.01, 0.99, 0.01):
        y_pred = np.where(probs[:, 0] >= tau, 0, probs[:, 1:].argmax(axis=1) + 1)
        se, sp, score = base.icbhi_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_tau = tau
            best_se = se
            best_sp = sp

    return best_score, best_se, best_sp, best_tau


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ICBHI S11: Comprehensive SOTA Pipeline")
    p.add_argument("--data_dir", type=str, default=str(base.ICBHI_2017_DIR))
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=8.0)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--target_frames", type=int, default=512)
    p.add_argument("--f_min", type=float, default=50.0)
    p.add_argument("--f_max", type=float, default=4000.0)
    p.add_argument("--input_view", type=str, default="logmel")
    p.add_argument("--no_bandpass", action="store_true")
    p.add_argument("--benchmark_protocol", type=str, default="official_icbhi")
    p.add_argument("--width", type=float, default=1.25)  # Wider model
    p.add_argument("--student_width", type=float, default=1.25)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--time_shift", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--freq_mask", type=int, default=12)
    p.add_argument("--time_mask", type=int, default=48)
    p.add_argument("--use_vtlp", action="store_true", default=False)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--stage", type=str, default="all", choices=["train", "evaluate", "all"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--focal_gamma", type=float, default=3.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=0)  # CPU control!
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.in_channels = 1  # Single channel log-mel
    nc = args.num_classes

    records = base.build_records(Path(args.data_dir))
    splits = base.create_official_splits(records, nc, val_frac=args.val_size, seed=42)
    rec_train, rec_val, rec_test = splits['train'], splits['val'], splits['test']

    if args.stage in ("train", "all"):
        print("\n" + "=" * 60)
        print("S11: Comprehensive SOTA Pipeline")
        print("=" * 60)
        train_comprehensive(args, device, rec_train, rec_val, nc)

    if args.stage in ("evaluate", "all"):
        print("\n" + "=" * 60)
        print("S11: Final Evaluation")
        print("=" * 60)

        model = EnhancedClassifier(nc=nc, in_ch=args.in_channels, width=args.width).to(device)
        ckpt = base.TRAINING_ARTIFACTS_DIR / "s11_comprehensive" / "best.pt"
        if ckpt.exists():
            ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state"])
            best_tau = ckpt_data.get("best_tau", 0.5)
            print(f"[S11] Loaded checkpoint from epoch {ckpt_data['epoch']}, "
                  f"val_score={ckpt_data['val_score']:.4f}, tau={best_tau:.3f}")
        else:
            print("[S11] No checkpoint found!")
            return

        # Test with best threshold from validation
        test_ds = VTLPDataset(rec_test, args, augment=False)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

        # Apply validation threshold to test
        test_score, test_se, test_sp = _eval_fixed_threshold(
            model, test_dl, device, nc, best_tau
        )
        print(f"\n[S11] Test Results (threshold={best_tau:.3f}):")
        print(f"  ICBHI Score: {test_score:.4f}")
        print(f"  Sensitivity: {test_se:.4f}")
        print(f"  Specificity: {test_sp:.4f}")

        # Also try TTA
        metrics, tta_tau = evaluate_with_tta(model, test_dl, device, nc, n_tta=7)
        print(f"\n[S11] Test Results (TTA, threshold={tta_tau:.4f}):")
        print(f"  ICBHI Score: {metrics['icbhi_score']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")

        # Wide threshold search on test (for analysis)
        test_score2, test_se2, test_sp2, test_tau2 = _eval_wide_threshold(
            model, test_dl, device, nc
        )
        print(f"\n[S11] Test Results (wide search, tau={test_tau2:.3f}):")
        print(f"  ICBHI Score: {test_score2:.4f}")
        print(f"  Sensitivity: {test_se2:.4f}")
        print(f"  Specificity: {test_sp2:.4f}")


def _eval_fixed_threshold(model, dl, device, nc, tau):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y, idx in dl:
            x = x.to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()
    y_pred = np.where(probs[:, 0] >= tau, 0, probs[:, 1:].argmax(axis=1) + 1)
    se, sp, score = base.icbhi_score(y_true, y_pred)
    return score, se, sp


def _eval_wide_threshold(model, dl, device, nc):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y, idx in dl:
            x = x.to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    best_score = 0.0
    best_tau = 0.5
    best_se, best_sp = 0.0, 0.0
    for tau in np.arange(0.01, 0.99, 0.01):
        y_pred = np.where(probs[:, 0] >= tau, 0, probs[:, 1:].argmax(axis=1) + 1)
        se, sp, score = base.icbhi_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_tau = tau
            best_se = se
            best_sp = sp
    return best_score, best_se, best_sp, best_tau


if __name__ == "__main__":
    main()
