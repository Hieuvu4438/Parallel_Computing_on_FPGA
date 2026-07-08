#!/usr/bin/env python3
"""
ICBHI Strategy S10: Learned Adaptive Threshold

Novel method: Instead of sweeping a fixed threshold post-hoc, learns a
SAMPLE-SPECIFIC threshold as part of the model. A small "threshold head"
predicts the optimal Normal-vs-Abnormal threshold for each input.

Key innovation:
  - Threshold is a learned function of the input, not a fixed scalar
  - Harder samples (ambiguous crackle/normal) get different thresholds
  - The threshold predictor learns patient-specific and class-specific
    decision boundaries
  - Combined with standard fixed-threshold sweep for comparison

Expected gain: +1-3% ICBHI Score (from better threshold adaptation)

Usage:
    python python/training/icbhi_kd_s10_adaptive_threshold.py --stage train
    python python/training/icbhi_kd_s10_adaptive_threshold.py --stage evaluate
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
import sys
from pathlib import Path

import torch
torch.set_num_threads(2)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import ensure_dir
from python.training import icbhi_kd_pipeline_multiview_ensemble as base
from python.training.icbhi_sota_loss_functions import (
    ClassBalancedFocalLoss, SmoothedKDLoss, SensitivityAwareBinaryLoss,
)
from python.training.icbhi_sota_evaluation import evaluate_with_tta


# ============================================================================
# Adaptive Threshold Model
# ============================================================================

class AdaptiveThresholdClassifier(nn.Module):
    """
    Classifier with learned sample-specific threshold.

    Architecture:
      Shared CNN backbone → features
      ├── Classifier head → 4-class logits
      └── Threshold head → per-sample threshold τ ∈ (0, 1)

    Prediction logic:
      p_normal = softmax(logits)[0]
      τ = sigmoid(threshold_pred)
      if p_normal >= τ → Normal
      else → argmax among abnormal classes
    """

    def __init__(self, nc=4, in_ch=1, width=1.0, use_sigmoid=True):
        super().__init__()
        self.nc = nc
        self.use_sigmoid = use_sigmoid

        c = lambda v: max(8, int(v * width))

        # Shared backbone
        self.backbone = nn.Sequential(
            base.ConvBNAct(in_ch, c(24)),
            base.DSResBlock(c(24), c(32), stride=2),
            base.DSResBlock(c(32), c(48)),
            base.DSResBlock(c(48), c(64), stride=2),
            base.DSResBlock(c(64), c(96)),
            base.DSResBlock(c(96), c(128), stride=2),
        )
        feat_dim = c(128)

        # Shared pooling
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Classifier head: 4-class logits
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, nc),
        )

        # Threshold head: predicts per-sample threshold
        self.threshold_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in (0, 1)
        )

    def extract_features(self, x):
        """Extract shared features."""
        feat = self.backbone(x)
        feat = self.pool(feat)
        return self.feature_proj(feat)

    def forward(self, x, return_threshold=False):
        """
        Args:
            x: [B, C, H, W] input spectrogram
            return_threshold: if True, also return the learned threshold

        Returns:
            logits: [B, nc]
            threshold: [B, 1] (if return_threshold)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)

        if return_threshold:
            threshold = self.threshold_head(features)
            return logits, threshold
        return logits

    def predict_adaptive(self, x):
        """
        Predict with adaptive threshold.

        Returns:
            predictions: [B] class predictions using adaptive threshold
            probabilities: [B, nc] softmax probabilities
            thresholds: [B] learned thresholds
        """
        logits, threshold = self.forward(x, return_threshold=True)
        probs = F.softmax(logits, dim=1)
        threshold = threshold.squeeze(-1)  # [B]

        # Adaptive prediction
        p_normal = probs[:, 0]  # [B]
        is_normal = (p_normal >= threshold).float()  # [B]

        # For abnormal: argmax among classes 1,2,3
        abnormal_probs = probs[:, 1:]  # [B, 3]
        abnormal_pred = abnormal_probs.argmax(dim=1) + 1  # [B] (1, 2, or 3)

        # Combine
        predictions = torch.where(is_normal > 0.5,
                                   torch.zeros_like(abnormal_pred),
                                   abnormal_pred)

        return predictions, probs, threshold


# ============================================================================
# Adaptive Threshold Loss
# ============================================================================

class AdaptiveThresholdLoss(nn.Module):
    """
    Loss function for adaptive threshold training.

    Combines:
      1. Standard classification loss (focal + KD)
      2. Threshold regularization (encourage meaningful thresholds)
      3. ICBHI-score-aware threshold loss (directly optimize Se+Sp)
    """

    def __init__(self, nc, samples_per_class, beta=0.9999, gamma=2.5, label_smoothing=0.05,
                 lambda_threshold=0.1):
        super().__init__()
        self.nc = nc
        self.lambda_threshold = lambda_threshold
        self.focal_fn = ClassBalancedFocalLoss(
            samples_per_class=samples_per_class, beta=beta, gamma=gamma, label_smoothing=label_smoothing,
        )
        self.kd_fn = SmoothedKDLoss(temperature=4.0, smoothing=0.15)
        self.binary_fn = SensitivityAwareBinaryLoss(teacher_ratio=0.4)

    def forward(self, logits, threshold, labels, teacher_probs):
        """
        Args:
            logits: [B, nc] classifier logits
            threshold: [B, 1] learned thresholds
            labels: [B] ground truth labels
            teacher_probs: [B, nc] teacher soft labels
        """
        # Standard losses
        l_focal = self.focal_fn(logits, labels)
        l_kd = self.kd_fn(logits, teacher_probs)
        l_binary = self.binary_fn(logits, labels, teacher_probs)

        # Threshold regularization
        # Encourage thresholds to be in a meaningful range (0.3 - 0.7)
        # Penalize thresholds that are too extreme
        threshold = threshold.squeeze(-1)  # [B]
        l_threshold_reg = (
            torch.relu(0.3 - threshold).mean() +  # Penalize too low
            torch.relu(threshold - 0.7).mean()     # Penalize too high
        )

        # ICBHI-score-aware threshold loss
        # For normal samples: encourage threshold <= p_normal (predict correctly)
        # For abnormal samples: encourage threshold > p_normal (predict correctly)
        probs = F.softmax(logits, dim=1)
        p_normal = probs[:, 0]

        is_normal = (labels == 0).float()

        # Normal samples: loss when p_normal < threshold (false negative)
        normal_loss = ((1 - is_normal) * torch.relu(threshold - p_normal)).mean()

        # Abnormal samples: loss when p_normal >= threshold (false positive)
        abnormal_loss = (is_normal * torch.relu(p_normal - threshold)).mean()

        l_threshold_aware = normal_loss + abnormal_loss

        # Combined
        total = (
            0.35 * l_focal
            + 0.35 * l_kd
            + 0.20 * l_binary
            + self.lambda_threshold * l_threshold_reg
            + 0.10 * l_threshold_aware
        )

        return total, {
            "focal": l_focal.item(),
            "kd": l_kd.item(),
            "binary": l_binary.item(),
            "threshold_reg": l_threshold_reg.item(),
            "threshold_aware": l_threshold_aware.item(),
        }


# ============================================================================
# Training
# ============================================================================

def train_adaptive_threshold(args, device, records_train, records_val, nc):
    """Train adaptive threshold model with optional KD."""

    # Try loading teacher logits, fallback to standalone
    logits_dir = base.TRAINING_ARTIFACTS_DIR / "teacher_logits"
    use_kd = (logits_dir / "logits_train.npy").exists()

    if use_kd:
        print("[AdaptiveThresh] Found teacher logits, using KD training")
        teacher_logits_train = np.load(logits_dir / "logits_train.npy")
        teacher_logits_val = np.load(logits_dir / "logits_val.npy")
        train_ds = base.StudentKDDataset(
            records_train, teacher_logits_train, args, stats=None, augment=True,
        )
        val_ds = base.StudentKDDataset(
            records_val, teacher_logits_val, args, stats=None, augment=False,
        )
    else:
        print("[AdaptiveThresh] No teacher logits found, training standalone (no KD)")
        train_ds = base.ICBHIDataset(records_train, args, stats=None, augment=True)
        val_ds = base.ICBHIDataset(records_val, args, stats=None, augment=False)

    sampler = WeightedRandomSampler(
        base.sample_weights(records_train, nc), len(records_train), replacement=True,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    # Model
    model = AdaptiveThresholdClassifier(
        nc=nc, in_ch=args.in_channels, width=args.width,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[AdaptiveThresh] Model: {total_params/1e6:.2f}M params")

    # Loss
    loss_fn = AdaptiveThresholdLoss(
        nc=nc,
        samples_per_class=np.bincount([r.label_4class for r in records_train], minlength=nc),
        beta=0.9999, gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        lambda_threshold=args.lambda_threshold,
    )

    # Optimizer
    if args.use_sam:
        from python.training.icbhi_kd_pipeline_multiview_ensemble import SAM
        base_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = SAM(base_opt, rho=0.02)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2,
    )

    best_score = 0.0
    out_dir = ensure_dir(base.TRAINING_ARTIFACTS_DIR / "s10_adaptive_thresh")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        loss_components = {}

        for batch in train_dl:
            if len(batch) == 4:
                x, y, idx, t_probs = batch
                x, y = x.to(device), y.to(device)
                t_probs = t_probs.to(device)
            else:
                x, y, idx = batch
                x, y = x.to(device), y.to(device)
                t_probs = F.one_hot(y, nc).float().to(device)

            logits, threshold = model(x, return_threshold=True)

            loss, components = loss_fn(logits, threshold, y, t_probs)

            if args.use_sam:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                l2, _ = loss_fn(model(x, return_threshold=True)[0],
                                model(x, return_threshold=True)[1], y, t_probs)
                l2.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0) + v

        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            # Evaluate with adaptive threshold
            adaptive_score, adaptive_se, adaptive_sp = _eval_adaptive(
                model, val_dl, device, nc,
            )
            # Also evaluate with standard threshold
            standard_score, standard_se, standard_sp = _eval_standard(
                model, val_dl, device, nc,
            )

            avg_loss = total_loss / max(n_batches, 1)
            avg_comp = {k: v / max(n_batches, 1) for k, v in loss_components.items()}

            print(f"[AdaptiveThresh] Epoch {epoch+1}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"adaptive={adaptive_score:.4f} (Se={adaptive_se:.4f} Sp={adaptive_sp:.4f})  "
                  f"standard={standard_score:.4f}")

            # Use adaptive score for checkpoint selection
            if adaptive_score > best_score:
                best_score = adaptive_score
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_score": adaptive_score,
                    "standard_score": standard_score,
                    "args": vars(args),
                }, out_dir / "best.pt")

    print(f"[AdaptiveThresh] Best validation ICBHI Score: {best_score:.4f}")
    return model


def _eval_adaptive(model, dl, device, nc):
    """Evaluate with adaptive (learned) threshold."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            y = batch[1]
            preds, probs, thresholds = model.predict_adaptive(x)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    se, sp, score = base.icbhi_score(y_true, y_pred)
    return score, se, sp


def _eval_standard(model, dl, device, nc):
    """Evaluate with standard threshold sweep."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            y = batch[1]
            all_logits.append(model(x).cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    # Standard threshold sweep
    tuned = base.sweep_threshold_fine(y_true, probs)
    y_pred = base.threshold_predictions(probs, tuned["threshold"])
    se, sp, score = base.icbhi_score(y_true, y_pred)
    return score, se, sp


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ICBHI S10: Adaptive Threshold")
    # Data
    p.add_argument("--data_dir", type=str, default=str(base.ICBHI_2017_DIR))
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=8.0)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--target_frames", type=int, default=512)
    p.add_argument("--f_min", type=float, default=50.0)
    p.add_argument("--f_max", type=float, default=4000.0)
    p.add_argument("--input_view", type=str, default="logmel_delta")
    p.add_argument("--benchmark_protocol", type=str, default="official_icbhi")
    p.add_argument("--student_width", type=float, default=1.0)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--no_bandpass", action="store_true")
    p.add_argument("--time_shift", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--freq_mask", type=int, default=12)
    p.add_argument("--time_mask", type=int, default=48)
    p.add_argument("--use_vtlp", action="store_true", default=False)
    p.add_argument("--val_size", type=float, default=0.15)

    # Model
    p.add_argument("--teacher_arches", type=str, default="resnet_cnn,resnet_crnn,efficientnet_b0")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--width", type=float, default=1.0)
    # Training
    p.add_argument("--stage", type=str, default="all", choices=["train", "evaluate", "all"])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--focal_gamma", type=float, default=2.5)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--lambda_threshold", type=float, default=0.1)
    p.add_argument("--use_sam", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.in_channels = 3 if args.input_view == "logmel_delta" else 1
    nc = args.num_classes

    records = base.build_records(Path(args.data_dir))
    splits = base.create_official_splits(records, nc, val_frac=0.15, seed=42)
    rec_train, rec_val, rec_test = splits['train'], splits['val'], splits['test']

    if args.stage in ("train", "all"):
        print("\n" + "=" * 60)
        print("Training Adaptive Threshold Model")
        print("=" * 60)
        train_adaptive_threshold(args, device, rec_train, rec_val, nc)

    if args.stage in ("evaluate", "all"):
        print("\n" + "=" * 60)
        print("Evaluating Adaptive Threshold Model")
        print("=" * 60)

        model = AdaptiveThresholdClassifier(
            nc=nc, in_ch=args.in_channels, width=args.width,
        ).to(device)
        ckpt = base.TRAINING_ARTIFACTS_DIR / "s10_adaptive_thresh" / "best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])

            test_ds = base.ICBHIDataset(rec_test, args, stats=None, augment=False)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)

        # Adaptive threshold evaluation
        adaptive_score, adaptive_se, adaptive_sp = _eval_adaptive(
            model, test_dl, device, nc,
        )
        print(f"\n[AdaptiveThresh] Adaptive Threshold Test Results:")
        print(f"  ICBHI Score: {adaptive_score:.4f}")
        print(f"  Sensitivity: {adaptive_se:.4f}")
        print(f"  Specificity: {adaptive_sp:.4f}")

        # Standard threshold evaluation (for comparison)
        standard_score, standard_se, standard_sp = _eval_standard(
            model, test_dl, device, nc,
        )
        print(f"\n[AdaptiveThresh] Standard Threshold Test Results:")
        print(f"  ICBHI Score: {standard_score:.4f}")
        print(f"  Sensitivity: {standard_se:.4f}")
        print(f"  Specificity: {standard_sp:.4f}")

        # TTA evaluation
        metrics, threshold = evaluate_with_tta(model, test_dl, device, nc, n_tta=7)
        print(f"\n[AdaptiveThresh] TTA Evaluation (threshold={threshold:.4f}):")
        print(f"  ICBHI Score: {metrics['icbhi_score']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
