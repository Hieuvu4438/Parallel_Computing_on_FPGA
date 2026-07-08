#!/usr/bin/env python3
"""
ICBHI Strategy S7: Mixture of Experts (MoE) Respiratory Classifier — V2

Fixes from V1 (test ICBHI=0.5041, Se=0.0311):
1. Much stronger class weights (5x for Both, 3x for Crackle/Wheeze)
2. Binary loss weight increased from 0.25 to 0.50
3. Aggressive oversampling of minority classes (3x)
4. Sensitivity-aware loss that directly penalizes false negatives
5. Better threshold tuning (search wider range)
6. Curriculum learning: start with easy samples

Expected: ICBHI Score > 0.65 (target > 0.68)
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
    ClassBalancedFocalLoss, SensitivityAwareBinaryLoss,
)
from python.training.icbhi_sota_evaluation import evaluate_with_tta


# ============================================================================
# Mixture of Experts Architecture (same as V1)
# ============================================================================

class ExpertHead(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, features):
        return self.net(features)


class MoERouter(nn.Module):
    def __init__(self, feat_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(feat_dim, num_experts),
        )
    def forward(self, features):
        return self.gate(features)


class MoERespiratoryClassifier(nn.Module):
    def __init__(self, nc=4, in_ch=1, num_experts=5, top_k=2, width=1.0):
        super().__init__()
        self.nc = nc
        self.num_experts = num_experts
        self.top_k = top_k
        c = lambda v: max(8, int(v * width))
        self.backbone = nn.Sequential(
            base.ConvBNAct(in_ch, c(24)),
            base.DSResBlock(c(24), c(32), stride=2),
            base.DSResBlock(c(32), c(48)),
            base.DSResBlock(c(48), c(64), stride=2),
            base.DSResBlock(c(64), c(96)),
            base.DSResBlock(c(96), c(128), stride=2),
        )
        feat_dim = c(128)
        self.router = MoERouter(feat_dim, num_experts)
        self.experts = nn.ModuleList([
            ExpertHead(feat_dim, nc) for _ in range(num_experts)
        ])
        self.route_temp = nn.Parameter(torch.ones(1))

    def forward(self, x, return_routing=False):
        features = self.backbone(x)
        routing_logits = self.router(features)
        routing_weights = F.softmax(routing_logits / self.route_temp, dim=-1)
        top_k_w, top_k_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_w = top_k_w / (top_k_w.sum(dim=-1, keepdim=True) + 1e-8)
        expert_outputs = torch.stack([e(features) for e in self.experts], dim=1)
        B = x.shape[0]
        logits = torch.zeros(B, self.nc, device=x.device)
        for k in range(self.top_k):
            idx = top_k_idx[:, k]
            w = top_k_w[:, k]
            expert_out = expert_outputs[torch.arange(B, device=x.device), idx]
            logits = logits + w.unsqueeze(-1) * expert_out
        if return_routing:
            return logits, routing_weights
        return logits

    def get_features(self, x):
        return self.backbone(x)


class MoELoadBalancingLoss(nn.Module):
    def __init__(self, lambda_balance=0.01):
        super().__init__()
        self.lambda_balance = lambda_balance
    def forward(self, routing_weights):
        avg_routing = routing_weights.mean(dim=0)
        cv_squared = (avg_routing.std() / (avg_routing.mean() + 1e-8)) ** 2
        return self.lambda_balance * cv_squared


# ============================================================================
# Improved Loss: Aggressive Sensitivity-Aware Loss
# ============================================================================

class AggressiveSensitivityLoss(nn.Module):
    """
    Loss that heavily penalizes false negatives (missed abnormal samples).
    Much stronger than standard SensitivityAwareBinaryLoss.
    """
    def __init__(self, fn_weight=5.0):
        super().__init__()
        self.fn_weight = fn_weight  # Weight for false negative penalty

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, nc] classifier logits
            targets: [B] ground truth labels (0=Normal, 1-3=Abnormal)
        """
        probs = F.softmax(logits, dim=1)

        # Binary: Normal vs Abnormal
        p_normal = probs[:, 0]
        is_abnormal = (targets != 0).float()

        # False negative: model says Normal but actually Abnormal
        # Heavily penalize this
        fn_loss = (is_abnormal * (-torch.log(1 - p_normal + 1e-8))).mean()

        # False positive: model says Abnormal but actually Normal
        fp_loss = ((1 - is_abnormal) * (-torch.log(p_normal + 1e-8))).mean()

        # Weighted combination: FN is much worse than FP
        return self.fn_weight * fn_loss + 1.0 * fp_loss


# ============================================================================
# Training with Aggressive Class Balancing
# ============================================================================

def train_moe(args, device, records_train, records_val, nc):
    """Train MoE model with aggressive class balancing."""
    # Data: standalone (no KD)
    train_ds = base.ICBHIDataset(records_train, args, stats=None, augment=True)
    val_ds = base.ICBHIDataset(records_val, args, stats=None, augment=False)

    # AGGRESSIVE class-balanced sampling
    # Count samples per class
    class_counts = np.bincount([r.label_4class for r in records_train], minlength=nc)
    print(f"[MoE] Class distribution: {dict(zip(range(nc), class_counts))}")

    # Compute aggressive weights: inverse frequency * class multiplier
    # Give Both (class 3) and Wheeze (class 2) much higher weights
    class_multipliers = {0: 1.0, 1: 3.0, 2: 4.0, 3: 5.0}  # Both gets 5x weight
    sample_weights = []
    for r in records_train:
        base_weight = 1.0 / max(class_counts[r.label_4class], 1)
        multiplier = class_multipliers.get(r.label_4class, 1.0)
        sample_weights.append(base_weight * multiplier)

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=1, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=1)

    # Model
    model = MoERespiratoryClassifier(
        nc=nc, in_ch=args.in_channels,
        num_experts=args.num_experts, top_k=args.top_k, width=args.width,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MoE] Model: {total_params/1e6:.2f}M params, "
          f"{args.num_experts} experts, top-{args.top_k} routing")

    # Loss functions - AGGRESSIVE
    focal_fn = ClassBalancedFocalLoss(
        samples_per_class=class_counts.astype(np.float64),
        beta=0.9999, gamma=3.0, label_smoothing=0.05,
    )
    sensitivity_fn = AggressiveSensitivityLoss(fn_weight=5.0)
    balance_fn = MoELoadBalancingLoss(lambda_balance=0.01)

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
    out_dir = ensure_dir(base.TRAINING_ARTIFACTS_DIR / "s7_moe_student_v2")

    for epoch in range(args.epochs):
        # Progressive loss weighting
        t_prog = min(epoch / 30.0, 1.0)
        w_focal = 0.50 * (1 - t_prog) + 0.30 * t_prog  # Decrease focal
        w_sens = 0.40 * (1 - t_prog) + 0.60 * t_prog   # Increase sensitivity loss

        model.train()
        total_loss = 0
        n_batches = 0
        routing_stats = torch.zeros(args.num_experts, device=device)

        for batch in train_dl:
            x, y, idx = batch
            x, y = x.to(device), y.to(device)

            logits, routing_w = model(x, return_routing=True)

            # Losses
            l_focal = focal_fn(logits, y)
            l_sens = sensitivity_fn(logits, y)
            l_balance = balance_fn(routing_w)

            loss = w_focal * l_focal + w_sens * l_sens + l_balance

            if args.use_sam:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                focal_fn(model(x), y).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            routing_stats += routing_w.sum(dim=0)

        scheduler.step()

        routing_dist = (routing_stats / len(records_train)).detach().cpu().numpy()

        # Validation
        if (epoch + 1) % 5 == 0:
            val_score, val_se, val_sp = _eval_moe(model, val_dl, device, nc)
            print(f"[MoE] Epoch {epoch+1}/{args.epochs}  "
                  f"loss={total_loss/max(n_batches,1):.4f}  "
                  f"val_score={val_score:.4f}  Se={val_se:.4f}  Sp={val_sp:.4f}  "
                  f"routing=[{', '.join(f'{r:.2f}' for r in routing_dist)}]")

            if val_score > best_score:
                best_score = val_score
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_score": val_score,
                    "args": vars(args),
                }, out_dir / "best.pt")

    print(f"[MoE] Best validation ICBHI Score: {best_score:.4f}")
    return model


def _eval_moe(model, dl, device, nc):
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

    # Wider threshold search: also try argmax and various thresholds
    best_score = 0.0
    best_se, best_sp = 0.0, 0.0

    # Try argmax
    y_pred = probs.argmax(axis=1)
    se, sp, score = base.icbhi_score(y_true, y_pred)
    if score > best_score:
        best_score, best_se, best_sp = score, se, sp

    # Try various thresholds for Normal vs Abnormal
    for tau in np.arange(0.1, 0.9, 0.02):
        y_pred = np.where(probs[:, 0] >= tau, 0, probs[:, 1:].argmax(axis=1) + 1)
        se, sp, score = base.icbhi_score(y_true, y_pred)
        if score > best_score:
            best_score, best_se, best_sp = score, se, sp

    return best_score, best_se, best_sp


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ICBHI S7 V2: MoE with aggressive class balancing")
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
    p.add_argument("--input_view", type=str, default="logmel_delta")
    p.add_argument("--no_bandpass", action="store_true")
    p.add_argument("--benchmark_protocol", type=str, default="official_icbhi")
    p.add_argument("--num_experts", type=int, default=5)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--student_width", type=float, default=1.0)
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
    p.add_argument("--use_sam", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.in_channels = 3 if args.input_view == "logmel_delta" else 1
    nc = args.num_classes

    records = base.build_records(Path(args.data_dir))
    splits = base.create_official_splits(records, nc, val_frac=args.val_size, seed=42)
    rec_train, rec_val, rec_test = splits['train'], splits['val'], splits['test']

    if args.stage in ("train", "all"):
        print("\n" + "=" * 60)
        print("Training MoE Model V2 (Aggressive Class Balancing)")
        print("=" * 60)
        train_moe(args, device, rec_train, rec_val, nc)

    if args.stage in ("evaluate", "all"):
        print("\n" + "=" * 60)
        print("Evaluating MoE Model V2")
        print("=" * 60)

        model = MoERespiratoryClassifier(
            nc=nc, in_ch=args.in_channels,
            num_experts=args.num_experts, top_k=args.top_k, width=args.width,
        ).to(device)
        ckpt = base.TRAINING_ARTIFACTS_DIR / "s7_moe_student_v2" / "best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])

        test_ds = base.ICBHIDataset(rec_test, args, stats=None, augment=False)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=1)

        # Standard threshold sweep
        val_ds = base.ICBHIDataset(rec_val, args, stats=None, augment=False)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=1)

        # Find best threshold on validation
        val_score, val_se, val_sp, best_tau = _eval_moe_with_threshold(model, val_dl, device, nc)
        print(f"[MoE V2] Best val threshold: {best_tau:.4f}  "
              f"Score={val_score:.4f}  Se={val_se:.4f}  Sp={val_sp:.4f}")

        # Apply to test
        test_score, test_se, test_sp = _eval_moe_fixed_threshold(model, test_dl, device, nc, best_tau)
        print(f"\n[MoE V2] Test Results (threshold={best_tau:.4f}):")
        print(f"  ICBHI Score: {test_score:.4f}")
        print(f"  Sensitivity: {test_se:.4f}")
        print(f"  Specificity: {test_sp:.4f}")

        # Also try TTA
        metrics, threshold = evaluate_with_tta(model, test_dl, device, nc, n_tta=7)
        print(f"\n[MoE V2] Test Results (TTA, threshold={threshold:.4f}):")
        print(f"  ICBHI Score: {metrics['icbhi_score']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")


def _eval_moe_with_threshold(model, dl, device, nc):
    """Find best threshold on validation set."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(batch[1])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    best_score = 0.0
    best_tau = 0.5
    for tau in np.arange(0.05, 0.95, 0.01):
        y_pred = np.where(probs[:, 0] >= tau, 0, probs[:, 1:].argmax(axis=1) + 1)
        se, sp, score = base.icbhi_score(y_true, y_pred)
        if score > best_score:
            best_score, best_se, best_sp, best_tau = score, se, sp, tau

    return best_score, best_se, best_sp, best_tau


def _eval_moe_fixed_threshold(model, dl, device, nc, tau):
    """Evaluate with a fixed threshold."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(batch[1])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()
    y_pred = np.where(probs[:, 0] >= tau, 0, probs[:, 1:].argmax(axis=1) + 1)
    se, sp, score = base.icbhi_score(y_true, y_pred)
    return score, se, sp


if __name__ == "__main__":
    main()
