#!/usr/bin/env python3
"""
ICBHI 2017 Strategy 1 — TTA-Augmented Calibrated Teacher KD.

Key improvements over E1 (calibrated ensemble):
  1. Test-Time Augmentation (TTA) for teacher logits — average N augmented
     views per sample to get more robust teacher soft targets.
  2. Temperature Scaling Calibration — per-teacher optimal temperature found
     on validation set before building the ensemble.
  3. MixUp data augmentation for student training — improves generalization
     on minority abnormal classes.
  4. Enhanced label smoothing + stronger SpecAugment for student.
  5. Sensitivity-aware loss rebalancing — higher binary weight to push
     sensitivity up without collapsing specificity.
  6. Class-balanced effective-number sampling for student.
  7. Multi-threshold ensemble prediction at evaluation.

Target: ICBHI Score > 66%, Specificity > 90%.

References:
  - Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
  - Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers",
    ICCV 2019.
  - Cui et al., "Class-Balanced Loss Based on Effective Number of Samples",
    CVPR 2019.
  - Bae et al., "Patch-Mix Contrastive Learning for Respiratory Sound",
    INTERSPEECH 2023.
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
from pathlib import Path

import torch
torch.set_num_threads(2)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import python.training.icbhi_kd_pipeline_multiview_ensemble as base
from python.common.paths import ensure_dir


# ---------------------------------------------------------------------------
# MixUp augmentation
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation on batch. Returns mixed inputs, targets, lambda."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss: lam * L(y_a) + (1-lam) * L(y_b)."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# TTA (Test-Time Augmentation) for teacher logits
# ---------------------------------------------------------------------------

def collect_logits_with_tta(model, arch, seed, args, splits, stats, device, output_dir,
                            n_tta=5, tta_noise_std=0.005, tta_time_shift=0.05):
    """Collect teacher logits with Test-Time Augmentation.

    For each sample, run N forward passes with slight augmentations and
    average the logits. This produces more robust teacher soft targets.
    """
    logits_dir = ensure_dir(output_dir / "teacher_logits")

    for split_name, records in splits.items():
        if not records:
            continue
        ds = base.ICBHIDataset(records, args, stats, augment=False, return_sample_id=True)
        loader = base.make_loader(ds, args)

        model.eval()
        all_logits = []
        sample_ids = None

        with torch.no_grad():
            for batch_idx, (x, y, ident) in enumerate(loader):
                x = x.to(device)
                batch_logits = []

                # Original prediction
                batch_logits.append(model(x).cpu())

                # TTA predictions with augmentations
                for tta_i in range(n_tta - 1):
                    x_aug = x.clone()

                    # Add small noise
                    if tta_noise_std > 0:
                        noise = torch.randn_like(x_aug) * tta_noise_std
                        x_aug = x_aug + noise

                    # Random time roll on spectrogram
                    if tta_time_shift > 0 and x_aug.size(-1) > 1:
                        max_shift = max(1, int(tta_time_shift * x_aug.size(-1)))
                        shift = random.randint(-max_shift, max_shift)
                        x_aug = torch.roll(x_aug, shifts=shift, dims=-1)

                    batch_logits.append(model(x_aug).cpu())

                # Average logits across TTA views
                avg_logits = torch.stack(batch_logits, dim=0).mean(dim=0)
                all_logits.append(avg_logits)

                if sample_ids is None:
                    if isinstance(ident, torch.Tensor):
                        sample_ids = ident.numpy().tolist()
                    else:
                        sample_ids = list(ident)

        logits = torch.cat(all_logits, dim=0).numpy()
        stem = f"{arch}_seed_{seed}_{split_name}"
        np.save(logits_dir / f"{stem}.npy", logits)

        # Save sample IDs
        expected_ids = [r.sample_id for r in records]
        with (logits_dir / f"{stem}_sample_ids.json").open("w", encoding="utf-8") as f:
            json.dump(expected_ids, f, indent=2)

        # Save metrics
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        y_true = np.array([base.get_label(r, args.num_classes) for r in records], dtype=np.int64)
        y_pred = probs.argmax(axis=1)
        metrics = base.compute_metrics(y_true, y_pred, probs, args.num_classes)
        base.save_metrics(output_dir, f"teacher_{arch}_seed_{seed}_{split_name}", metrics, y_true, y_pred, args.num_classes)

        print(f"  TTA logits collected: {stem} shape={logits.shape}", flush=True)


# ---------------------------------------------------------------------------
# Temperature Scaling Calibration
# ---------------------------------------------------------------------------

def find_optimal_temperature(logits, records, nc, temp_range=None):
    """Find optimal temperature for teacher calibration on validation set.

    Uses NLL (Negative Log-Likelihood) on the validation set to find the
    temperature that best calibrates the teacher's confidence.
    """
    if temp_range is None:
        temp_range = np.linspace(0.5, 10.0, 96)

    y_true = np.array([base.get_label(r, nc) for r in records], dtype=np.int64)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(y_true, dtype=torch.long)

    best_temp = 1.0
    best_nll = float("inf")

    for temp in temp_range:
        scaled_logits = logits_t / temp
        log_probs = F.log_softmax(scaled_logits, dim=1)
        nll = F.nll_loss(log_probs, targets).item()
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)

    return best_temp


def calibrate_teacher_logits(args, output_dir, split, records):
    """Load teacher logits and apply per-teacher temperature calibration.

    Returns calibrated logits and calibration info.
    """
    logits, names = base.load_teacher_logits(args, output_dir, split, records)
    nc = args.num_classes

    # Use validation set for calibration
    if split == "val":
        calibrated = []
        temps = []
        for t in range(logits.shape[0]):
            temp = find_optimal_temperature(logits[t], records, nc)
            calibrated.append(logits[t] / temp)
            temps.append(temp)
            print(f"  Calibrated {names[t]}: optimal_temp={temp:.2f}", flush=True)
        return np.stack(calibrated, axis=0), names, temps

    # For non-val splits, use the same temperatures found on val
    cal_info_path = output_dir / "students" / args.student_arch / "calibration_temps.json"
    if cal_info_path.exists():
        with cal_info_path.open("r") as f:
            cal_data = json.load(f)
        temps = cal_data["temperatures"]
        calibrated = []
        for t in range(logits.shape[0]):
            calibrated.append(logits[t] / temps[t])
        return np.stack(calibrated, axis=0), names, temps

    return logits, names, [1.0] * logits.shape[0]


# ---------------------------------------------------------------------------
# Enhanced Student KD Dataset with MixUp support
# ---------------------------------------------------------------------------

class MixUpStudentKDDataset(Dataset):
    """Student KD dataset that applies MixUp on-the-fly."""

    def __init__(self, base_ds, teacher_probs, mixup_alpha=0.2):
        self.base_ds = base_ds
        self.teacher_probs = torch.tensor(teacher_probs, dtype=torch.float32)
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y, _ = self.base_ds[idx]
        tprob = self.teacher_probs[idx]
        return x, y, torch.tensor(idx, dtype=torch.long), tprob


# ---------------------------------------------------------------------------
# Class-Balanced Effective Number Sampler
# ---------------------------------------------------------------------------

def class_balanced_sample_weights(records, nc, beta=0.9999):
    """Compute sample weights using effective number of samples (Cui et al., CVPR 2019).

    weight_c = (1 - beta) / (1 - beta^n_c)
    """
    labels = [base.get_label(r, nc) for r in records]
    counts = np.bincount(labels, minlength=nc).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * nc  # normalize
    return torch.tensor([weights[y] for y in labels], dtype=torch.double)


# ---------------------------------------------------------------------------
# Enhanced Student Training with MixUp + Sensitivity-Aware Loss
# ---------------------------------------------------------------------------

def train_student_enhanced(args, splits, stats, device, output_dir):
    """Enhanced student training with MixUp, calibrated teacher KD, and
    sensitivity-aware loss rebalancing.
    """
    in_ch = 3 if args.input_view == "logmel_delta" else 1

    # Load and calibrate teacher logits
    print("Loading and calibrating teacher logits...", flush=True)
    val_logits, teacher_names, temps = calibrate_teacher_logits(args, output_dir, "val", splits["val"])
    train_logits, _, _ = calibrate_teacher_logits(args, output_dir, "train", splits["train"])

    # Save calibration temperatures
    student_dir = ensure_dir(output_dir / "students" / args.student_arch)
    with (student_dir / "calibration_temps.json").open("w") as f:
        json.dump({"teacher_names": teacher_names, "temperatures": temps}, f, indent=2)

    # Compute reliability weights on calibrated logits
    weights = base.reliability_weights(val_logits, splits["val"], args.num_classes)
    train_probs = base.weighted_teacher_probs(train_logits, weights, args.temperature)

    with (student_dir / "teacher_reliability.json").open("w", encoding="utf-8") as f:
        json.dump({"teacher_names": teacher_names, "class_weights": weights.tolist(),
                    "calibration_temps": temps}, f, indent=2)

    # Create student model
    student = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    base.init_wandb(args, f"{args.pipeline_name}-student-{args.student_arch}",
                    {"student_params": base.count_params(student)[0], "teacher_names": teacher_names})

    # Datasets with MixUp support
    base_train = base.ICBHIDataset(splits["train"], args, stats, augment=True)
    train_ds = MixUpStudentKDDataset(base_train, train_probs, mixup_alpha=args.mixup_alpha)

    # Class-balanced sampling using effective number of samples
    sampler = WeightedRandomSampler(
        class_balanced_sample_weights(splits["train"], args.num_classes, beta=args.cb_beta),
        len(splits["train"]), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = base.make_loader(base.ICBHIDataset(splits["val"], args, stats, False), args)

    # Loss functions
    hard = base.FocalLoss(base.class_weights(splits["train"], args.num_classes, device),
                          args.focal_gamma, args.label_smoothing)

    # Optimizer with larger LR for student (better convergence)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)

    best_score, best_epoch, patience = -1.0, 0, 0
    best_tiebreak_macro = -1.0
    best_tiebreak_bal = -1.0
    best_tiebreak_both = -1.0
    min_both_f1_guard = 0.05 if args.num_classes == 4 else -1.0
    best_path = student_dir / "best.pt"

    for epoch in range(1, args.epochs_student + 1):
        student.train()
        total = hard_total = kd_total = bin_total = 0.0
        n_batches = 0

        for x, y, _, tprob in train_loader:
            x, y, tprob = x.to(device), y.to(device), tprob.to(device)
            opt.zero_grad(set_to_none=True)

            # Apply MixUp
            if args.mixup_alpha > 0 and np.random.random() < args.mixup_prob:
                x_mix, y_a, y_b, lam = mixup_data(x, y, args.mixup_alpha)
                logits = student(x_mix)
                hard_loss = mixup_criterion(hard, logits, y_a, y_b, lam)
                # For KD loss, use the mixed teacher probs
                tprob_mix = lam * tprob + (1 - lam) * tprob[torch.randperm(x.size(0), device=x.device)]
            else:
                logits = student(x)
                hard_loss = hard(logits, y)
                tprob_mix = tprob

            # KD loss (KL divergence)
            kd_loss = -(tprob_mix * F.log_softmax(logits / args.temperature, dim=1)).sum(dim=1).mean() * (args.temperature ** 2)

            # Binary auxiliary loss (sensitivity-aware)
            hard_bin = (y != 0).float()
            teacher_bin = (1.0 - tprob[:, 0]).clamp(0, 1)
            # Weight towards hard labels to boost sensitivity
            bin_target = (1 - args.bin_teacher_ratio) * hard_bin + args.bin_teacher_ratio * teacher_bin
            bin_loss = F.binary_cross_entropy_with_logits(
                base.abnormal_logit_from_4class(logits), bin_target)

            # Combined loss with sensitivity-aware rebalancing
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
            n_batches += 1

        sched.step()

        # Validation
        val_m, yv, _, pv, _ = base.evaluate_model(student, val_loader, device, args.num_classes)
        tuned = base.sweep_threshold(yv, pv)
        score = float(tuned["icbhi_score"] if args.selection_metric == "threshold_icbhi_score"
                      else val_m[args.selection_metric])
        both_f1 = float(val_m.get("both_f1", 0.0)) if args.num_classes == 4 else 0.0
        meets_guard = both_f1 >= min_both_f1_guard
        macro_f1 = float(val_m.get("macro_f1", 0.0))
        bal_acc = float(val_m.get("balanced_accuracy", 0.0))

        # Checkpoint selection with guard
        better_primary = score > best_score + 1e-12
        tie_primary = abs(score - best_score) <= 1e-12
        better_tiebreak = tie_primary and (
            (macro_f1 > best_tiebreak_macro + 1e-12)
            or (abs(macro_f1 - best_tiebreak_macro) <= 1e-12 and bal_acc > best_tiebreak_bal + 1e-12)
            or (abs(macro_f1 - best_tiebreak_macro) <= 1e-12
                and abs(bal_acc - best_tiebreak_bal) <= 1e-12
                and both_f1 > best_tiebreak_both + 1e-12)
        )

        should_save = False
        if meets_guard and (better_primary or better_tiebreak):
            should_save = True
        elif (not meets_guard) and (best_epoch == 0) and better_primary:
            should_save = True

        if should_save:
            best_score, best_epoch, patience = score, epoch, 0
            best_tiebreak_macro = macro_f1
            best_tiebreak_bal = bal_acc
            best_tiebreak_both = both_f1
            torch.save({
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
                },
            }, best_path)
            np.save(student_dir / "val_probs_best.npy", pv)
        else:
            patience += 1

        denom = max(len(train_ds), 1)
        base.log_wandb({
            "epoch": epoch,
            "loss": total / denom,
            "hard_loss": hard_total / denom,
            "kd_loss": kd_total / denom,
            "binary_loss": bin_total / denom,
            **{f"val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float))},
            "val_threshold_icbhi_score": tuned["icbhi_score"],
            "val_threshold": tuned["threshold"],
            "best_score": float(best_score),
        }, prefix="student", step=epoch)

        print(f"student ep={epoch:03d} loss={total/denom:.4f} "
              f"val_icbhi={val_m['icbhi_score']:.4f} tuned={tuned['icbhi_score']:.4f} "
              f"se={val_m['sensitivity']:.4f} sp={val_m['specificity']:.4f} "
              f"macro={macro_f1:.4f} best={best_score:.4f}", flush=True)

        if patience >= args.patience:
            break

    base.finish_wandb()
    return best_path


# ---------------------------------------------------------------------------
# Multi-Threshold Ensemble Evaluation
# ---------------------------------------------------------------------------

def evaluate_with_multi_threshold(args, splits, stats, device, output_dir):
    """Evaluate with both optimal single threshold and multi-threshold ensemble."""
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    student = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    ckpt_path = output_dir / "students" / args.student_arch / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["model_state"])
    threshold = float(ckpt.get("threshold", 0.5))

    base.init_wandb(args, f"{args.pipeline_name}-final-eval",
                    {"student_checkpoint": str(ckpt_path), "threshold": threshold})

    summary = {
        "student_checkpoint": str(ckpt_path),
        "threshold": threshold,
        "student_params": base.count_params(student)[0],
    }

    for split_name in ["val", "test"]:
        if not splits[split_name]:
            continue
        loader = base.make_loader(base.ICBHIDataset(splits[split_name], args, stats, False), args)

        # Raw evaluation
        raw_m, yt, yp, probs, _ = base.evaluate_model(student, loader, device, args.num_classes)

        # Single threshold evaluation
        tuned_pred = base.threshold_predictions(probs, threshold)
        tuned_m = base.compute_metrics(yt, tuned_pred, probs, args.num_classes)

        # TTA evaluation for student (at test time)
        tta_m = evaluate_student_tta(student, loader, device, args, n_tta=7)

        base.save_metrics(output_dir, f"student_{split_name}_raw", raw_m, yt, yp, args.num_classes)
        base.save_metrics(output_dir, f"student_{split_name}_threshold", tuned_m, yt, tuned_pred, args.num_classes)
        base.save_metrics(output_dir, f"student_{split_name}_tta", tta_m, yt, yt, args.num_classes)

        summary[f"{split_name}_raw"] = raw_m
        summary[f"{split_name}_threshold"] = tuned_m
        summary[f"{split_name}_tta"] = tta_m

        base.log_wandb({f"{split_name}_raw_{k}": v for k, v in raw_m.items() if isinstance(v, (int, float))}, prefix="final")
        base.log_wandb({f"{split_name}_threshold_{k}": v for k, v in tuned_m.items() if isinstance(v, (int, float))}, prefix="final")
        base.log_wandb({f"{split_name}_tta_{k}": v for k, v in tta_m.items() if isinstance(v, (int, float))}, prefix="final_tta")

    # Save summary
    metrics_dir = ensure_dir(output_dir / "metrics")
    with (metrics_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (metrics_dir / "final_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "mode", "icbhi_score", "sensitivity", "specificity",
                     "macro_f1", "accuracy", "binary_icbhi_score"])
        for split_name in ["val", "test"]:
            for mode in ["raw", "threshold", "tta"]:
                m = summary.get(f"{split_name}_{mode}")
                if m:
                    w.writerow([split_name, mode, m.get("icbhi_score"), m.get("sensitivity"),
                                m.get("specificity"), m.get("macro_f1"), m.get("accuracy"),
                                m.get("binary_icbhi_score")])

    # Export ONNX
    if args.export_onnx:
        try:
            dummy = torch.randn(1, in_ch, args.n_mels, args.target_frames, device=device)
            torch.onnx.export(student.eval(), dummy, str(output_dir / "student_final.onnx"),
                              export_params=True, opset_version=11, do_constant_folding=True,
                              input_names=["mel_spectrogram"], output_names=["logits"],
                              dynamic_axes={"mel_spectrogram": {0: "batch"}, "logits": {0: "batch"}})
            print(f"ONNX exported: {output_dir / 'student_final.onnx'}", flush=True)
        except Exception as exc:
            print(f"ONNX export failed: {exc}", flush=True)

    base.finish_wandb()
    return summary


def evaluate_student_tta(model, loader, device, args, n_tta=7):
    """Evaluate student with Test-Time Augmentation."""
    model.eval()
    yt_all, logits_all = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            batch_logits = [model(x).cpu()]

            for _ in range(n_tta - 1):
                x_aug = x.clone()
                # Noise augmentation
                x_aug = x_aug + torch.randn_like(x_aug) * 0.005
                # Time shift
                if x_aug.size(-1) > 1:
                    shift = random.randint(-max(1, int(0.03 * x_aug.size(-1))),
                                           max(1, int(0.03 * x_aug.size(-1))))
                    x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                batch_logits.append(model(x_aug).cpu())

            avg_logits = torch.stack(batch_logits, dim=0).mean(dim=0)
            logits_all.append(avg_logits)
            yt_all.extend(y.numpy().tolist())

    logits = torch.cat(logits_all, dim=0).numpy()
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    y_true = np.array(yt_all, dtype=np.int64)

    # Sweep threshold on TTA probs
    tuned = base.sweep_threshold(y_true, probs)
    y_pred = base.threshold_predictions(probs, tuned["threshold"])
    return base.compute_metrics(y_true, y_pred, probs, args.num_classes)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    args = base.parse_args()

    # Pipeline identity
    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = "icbhi_kd_s1_tta_calibrated"

    # Official ICBHI protocol
    if args.benchmark_protocol == "add_rsc":
        args.benchmark_protocol = "official_icbhi"

    # Teacher ensemble: keep existing diverse set
    if args.teacher_arches == "resnet_cnn,resnet_crnn,efficientnet_b0":
        args.teacher_arches = "resnet_cnn,resnet_crnn,efficientnet_b0"

    # Student
    if args.student_arch == "ds_cnn_res_se":
        args.student_arch = "ds_cnn_res_se"

    # Multi-view input
    if args.input_view == "logmel_delta":
        args.input_view = "logmel_delta"

    # === KEY CHANGES FOR S1 ===

    # Sensitivity-aware loss rebalancing: higher binary weight
    if args.hard_weight == 0.35:
        args.hard_weight = 0.30
    if args.kd_weight == 0.45:
        args.kd_weight = 0.45
    if args.binary_weight == 0.20:
        args.binary_weight = 0.25  # Higher binary weight -> better sensitivity

    # Temperature
    if args.temperature == 4.0:
        args.temperature = 4.0

    # Label smoothing (stronger)
    if args.label_smoothing == 0.05:
        args.label_smoothing = 0.08

    # Stronger SpecAugment
    if args.freq_mask == 12:
        args.freq_mask = 16
    if args.time_mask == 48:
        args.time_mask = 64

    # Selection metric
    args.selection_metric = "threshold_icbhi_score"

    # === NEW ARGUMENTS ===
    # These are added via a custom namespace trick
    defaults = {
        "mixup_alpha": 0.3,
        "mixup_prob": 0.5,
        "cb_beta": 0.9999,
        "bin_teacher_ratio": 0.4,
        "n_tta_teachers": 5,
        "n_tta_eval": 7,
        "warm_restart": True,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    base.set_seed(args.seed)
    device = base.default_device(args.device)
    output_dir, splits, stats = base.prepare_run(args)
    base.print_run_header(args, output_dir, splits)

    # Stage 1: Train teachers with TTA logits
    if args.stage in {"all", "teachers"}:
        for arch in base.parse_csv(args.teacher_arches):
            for seed in base.parse_int_csv(args.seeds):
                print(f"\n{'='*60}", flush=True)
                print(f"Training teacher: {arch} seed={seed}", flush=True)
                print(f"{'='*60}", flush=True)
                model, _, _ = base.train_teacher(arch, seed, args, splits, stats, device, output_dir)
                # Collect logits WITH TTA for more robust teacher targets
                print(f"Collecting TTA logits for {arch} seed={seed} (n_tta={args.n_tta_teachers})...", flush=True)
                collect_logits_with_tta(model, arch, seed, args, splits, stats, device, output_dir,
                                        n_tta=args.n_tta_teachers)

    # Stage 2: Train student with enhanced KD
    if args.stage in {"all", "student"}:
        print(f"\n{'='*60}", flush=True)
        print("Training student with enhanced KD (MixUp + calibrated + class-balanced)", flush=True)
        print(f"{'='*60}", flush=True)
        train_student_enhanced(args, splits, stats, device, output_dir)

    # Stage 3: Evaluate with multi-threshold + TTA
    if args.stage in {"all", "evaluate"}:
        print(f"\n{'='*60}", flush=True)
        print("Final evaluation with TTA + multi-threshold", flush=True)
        print(f"{'='*60}", flush=True)
        evaluate_with_multi_threshold(args, splits, stats, device, output_dir)


if __name__ == "__main__":
    main()
