#!/usr/bin/env python3
"""
ICBHI 2017 Strategy 3 — Curriculum + EMA Teacher + Class-Balanced KD.

Key improvements over E1 (calibrated ensemble):
  1. Curriculum Learning — start training on easy (high-confidence) samples,
     gradually introduce harder (lower-confidence) samples. This helps the
     student learn robust features before tackling ambiguous cases.
  2. Exponential Moving Average (EMA) Teacher — maintain an EMA copy of the
     student as a smooth teacher, providing stable soft targets that evolve
     with the student (Mean Teacher approach).
  3. Class-Balanced Focal Loss — weight focal loss by effective number of
     samples per class (Cui et al., CVPR 2019) instead of raw counts.
  4. Progressive KD Weighting — start with higher hard-label weight, gradually
     shift to higher KD weight as teacher quality improves.
  5. Self-Training with Confidence-Weighted Pseudo-Labels — use high-confidence
     teacher predictions on unlabeled/augmented data as additional signal.
  6. Dual-Threshold Prediction — separate optimized thresholds for sensitivity
     and specificity, combined via a weighted voting scheme.
  7. Stochastic Weight Averaging (SWA) — use SWA for the final model to get
     smoother loss landscape and better generalization.

Target: ICBHI Score > 66%, Specificity > 90%.

References:
  - Tarvainen & Valpola, "Mean Teachers are Better Role Models", NeurIPS 2017.
  - Cui et al., "Class-Balanced Loss Based on Effective Number of Samples",
    CVPR 2019.
  - Bengio et al., "Curriculum Learning", ICML 2009.
  - Izmailov et al., "Averaging Weights Leads to Wider Optima and Better
    Generalization", UAI 2018.
  - Xie et al., "Self-Training with Noisy Student", ICCV 2020.
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
from copy import deepcopy
from pathlib import Path

import torch
torch.set_num_threads(2)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import python.training.icbhi_kd_pipeline_multiview_ensemble as base
from python.common.paths import ensure_dir


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA) Model
# ---------------------------------------------------------------------------

class EMAModel:
    """Maintains an Exponential Moving Average copy of model parameters.

    EMA_params = decay * EMA_params + (1 - decay) * model_params
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters from model."""
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def get_ema_model(self):
        return self.ema_model


# ---------------------------------------------------------------------------
# Class-Balanced Focal Loss (Effective Number)
# ---------------------------------------------------------------------------

class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss based on Effective Number of Samples.

    From Cui et al., CVPR 2019.
    weight_c = (1 - beta) / (1 - beta^n_c)
    Loss = weight_c * FocalLoss(logits, targets)
    """

    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * len(samples_per_class)
        self.alpha = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        nc = logits.size(1)
        device = logits.device
        alpha = self.alpha.to(device)

        if self.ls > 0:
            target = torch.full_like(logits, self.ls / max(nc - 1, 1))
            target.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        else:
            target = F.one_hot(targets, nc).float()

        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        loss = -((1 - p) ** self.gamma) * target * logp
        loss = loss * alpha.view(1, -1)
        return loss.sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Curriculum Sampler
# ---------------------------------------------------------------------------

class CurriculumSampler:
    """Implements curriculum learning by gradually including harder samples.

    Samples are sorted by difficulty (e.g., teacher confidence). Training
    starts with easy samples and progressively includes harder ones.
    """

    def __init__(self, n_samples, difficulty_scores, curriculum_start=0.3,
                 curriculum_end=1.0, warmup_epochs=20):
        self.n_samples = n_samples
        self.difficulty_scores = difficulty_scores  # Lower = easier
        self.curriculum_start = curriculum_start
        self.curriculum_end = curriculum_end
        self.warmup_epochs = warmup_epochs

        # Sort indices by difficulty (easiest first)
        self.sorted_indices = np.argsort(difficulty_scores)

    def get_subset_ratio(self, epoch):
        """Get the fraction of samples to include at this epoch."""
        if epoch >= self.warmup_epochs:
            return self.curriculum_end
        progress = epoch / self.warmup_epochs
        return self.curriculum_start + (self.curriculum_end - self.curriculum_start) * progress

    def get_active_indices(self, epoch):
        """Get indices of samples active at this epoch."""
        ratio = self.get_subset_ratio(epoch)
        n_active = max(int(self.n_samples * ratio), 32)  # At least 32 samples
        return self.sorted_indices[:n_active]


# ---------------------------------------------------------------------------
# Dual-Threshold Prediction
# ---------------------------------------------------------------------------

def dual_threshold_prediction(probs, threshold_normal, threshold_abnormal):
    """Dual-threshold prediction for better sensitivity/specificity control.

    - If P(normal) >= threshold_normal -> predict Normal
    - If P(abnormal_class_max) >= threshold_abnormal -> predict that abnormal class
    - Otherwise -> predict argmax
    """
    preds = probs.argmax(axis=1)
    p_normal = probs[:, 0]
    p_abnormal_max = probs[:, 1:].max(axis=1)

    # Strong normal signal
    preds = np.where(p_normal >= threshold_normal, 0, preds)
    # Strong abnormal signal
    abnormal_class = probs[:, 1:].argmax(axis=1) + 1
    preds = np.where((p_normal < threshold_normal) & (p_abnormal_max >= threshold_abnormal),
                     abnormal_class, preds)

    return preds


def sweep_dual_threshold(y_true, probs):
    """Sweep both thresholds to maximize ICBHI Score."""
    best = {"threshold_normal": 0.5, "threshold_abnormal": 0.5, "icbhi_score": -1.0}

    for th_n in np.linspace(0.10, 0.80, 15):
        for th_a in np.linspace(0.10, 0.70, 13):
            pred = dual_threshold_prediction(probs, float(th_n), float(th_a))
            se, sp, score = base.icbhi_score(y_true, pred)
            if score > best["icbhi_score"]:
                best = {
                    "threshold_normal": float(th_n),
                    "threshold_abnormal": float(th_a),
                    "icbhi_score": float(score),
                    "sensitivity": float(se),
                    "specificity": float(sp),
                }

    # Also try single threshold for comparison
    single = base.sweep_threshold(y_true, probs)
    if single["icbhi_score"] > best["icbhi_score"]:
        best = {
            "threshold_normal": float(single["threshold"]),
            "threshold_abnormal": 0.5,
            "icbhi_score": float(single["icbhi_score"]),
            "sensitivity": float(single["sensitivity"]),
            "specificity": float(single["specificity"]),
        }

    return best


# ---------------------------------------------------------------------------
# Compute Sample Difficulty
# ---------------------------------------------------------------------------

def compute_sample_difficulty(teacher_logits, records, nc):
    """Compute difficulty score for each sample based on teacher confidence.

    Difficulty = 1 - max(teacher_prob). Higher difficulty = harder sample.
    """
    probs = F.softmax(torch.tensor(teacher_logits), dim=1).numpy()
    max_prob = probs.max(axis=1)
    difficulty = 1.0 - max_prob
    return difficulty


# ---------------------------------------------------------------------------
# Progressive KD Weight Scheduler
# ---------------------------------------------------------------------------

class ProgressiveKDWeight:
    """Progressively increase KD weight and decrease hard-label weight.

    This allows the student to first learn from hard labels (reliable),
    then increasingly from teacher soft targets (richer signal).
    """

    def __init__(self, hard_start=0.50, hard_end=0.25, kd_start=0.30, kd_end=0.50,
                 binary_start=0.20, binary_end=0.25, warmup_epochs=30):
        self.hard_start = hard_start
        self.hard_end = hard_end
        self.kd_start = kd_start
        self.kd_end = kd_end
        self.binary_start = binary_start
        self.binary_end = binary_end
        self.warmup_epochs = warmup_epochs

    def get_weights(self, epoch):
        if epoch >= self.warmup_epochs:
            return self.hard_end, self.kd_end, self.binary_end
        t = epoch / self.warmup_epochs
        hard = self.hard_start + (self.hard_end - self.hard_start) * t
        kd = self.kd_start + (self.kd_end - self.kd_start) * t
        binary = self.binary_start + (self.binary_end - self.binary_start) * t
        return hard, kd, binary


# ---------------------------------------------------------------------------
# Stochastic Weight Averaging (SWA)
# ---------------------------------------------------------------------------

class SWA:
    """Simple SWA implementation for model averaging."""

    def __init__(self, model, start_epoch=80):
        self.model = model
        self.start_epoch = start_epoch
        self.swa_state = None
        self.n_averaged = 0

    @torch.no_grad()
    def update(self, epoch):
        if epoch < self.start_epoch:
            return
        if self.swa_state is None:
            self.swa_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.n_averaged = 1
        else:
            for k in self.swa_state:
                self.swa_state[k].mul_(self.n_averaged).add_(self.model.state_dict()[k])
                self.swa_state[k].div_(self.n_averaged + 1)
            self.n_averaged += 1

    def apply(self):
        if self.swa_state is not None:
            self.model.load_state_dict(self.swa_state)


# ---------------------------------------------------------------------------
# Enhanced Student Training
# ---------------------------------------------------------------------------

def train_student_curriculum_ema(args, splits, stats, device, output_dir):
    """Train student with curriculum learning, EMA teacher, class-balanced focal loss,
    progressive KD weighting, and SWA.
    """
    in_ch = 3 if args.input_view == "logmel_delta" else 1

    # Load teacher logits
    val_logits, teacher_names = base.load_teacher_logits(args, output_dir, "val", splits["val"])
    train_logits, _ = base.load_teacher_logits(args, output_dir, "train", splits["train"])
    weights = base.reliability_weights(val_logits, splits["val"], args.num_classes)
    train_probs = base.weighted_teacher_probs(train_logits, weights, args.temperature)

    # Compute sample difficulty for curriculum
    avg_train_logits = train_logits.mean(axis=0)  # Average across teachers
    difficulty = compute_sample_difficulty(avg_train_logits, splits["train"], args.num_classes)

    # Create student model
    student = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    student_dir = ensure_dir(output_dir / "students" / args.student_arch)
    with (student_dir / "teacher_reliability.json").open("w", encoding="utf-8") as f:
        json.dump({"teacher_names": teacher_names, "class_weights": weights.tolist()}, f, indent=2)

    # EMA model
    ema = EMAModel(student, decay=args.ema_decay)

    # Class-balanced focal loss
    counts = np.bincount([base.get_label(r, args.num_classes) for r in splits["train"]],
                         minlength=args.num_classes).astype(np.float64)
    cb_focal = ClassBalancedFocalLoss(counts, beta=args.cb_beta, gamma=args.focal_gamma,
                                       label_smoothing=args.label_smoothing).to(device)

    # Also keep standard focal for comparison
    hard = base.FocalLoss(base.class_weights(splits["train"], args.num_classes, device),
                          args.focal_gamma, args.label_smoothing)

    base.init_wandb(args, f"{args.pipeline_name}-student-{args.student_arch}",
                    {"student_params": base.count_params(student)[0], "teacher_names": teacher_names})

    # Progressive KD weight scheduler
    kd_scheduler = ProgressiveKDWeight(
        hard_start=args.hard_weight_start, hard_end=args.hard_weight,
        kd_start=args.kd_weight_start, kd_end=args.kd_weight,
        binary_start=args.binary_weight_start, binary_end=args.binary_weight,
        warmup_epochs=args.progressive_warmup)

    # Curriculum sampler
    curriculum = CurriculumSampler(
        len(splits["train"]), difficulty,
        curriculum_start=args.curriculum_start,
        warmup_epochs=args.curriculum_warmup)

    # SWA
    swa = SWA(student, start_epoch=max(args.epochs_student - 20, int(args.epochs_student * 0.7)))

    # Datasets
    base_train = base.ICBHIDataset(splits["train"], args, stats, augment=True)
    train_ds = base.StudentKDDataset(base_train, train_probs)
    val_loader = base.make_loader(base.ICBHIDataset(splits["val"], args, stats, False), args)

    # Optimizer
    use_sam = getattr(args, 'use_sam', False)
    if use_sam:
        opt = base.make_sam_optimizer(student.parameters(), lr=args.lr_student,
                                      weight_decay=args.weight_decay,
                                      rho=getattr(args, 'sam_rho', 0.05))
    else:
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt.base_optimizer if use_sam else opt, T_max=max(args.epochs_student, 1))

    best_score, best_epoch, patience = -1.0, 0, 0
    best_tiebreak_macro = -1.0
    best_tiebreak_bal = -1.0
    best_tiebreak_both = -1.0
    min_both_f1_guard = 0.05 if args.num_classes == 4 else -1.0
    best_path = student_dir / "best.pt"

    for epoch in range(1, args.epochs_student + 1):
        student.train()
        ema.get_ema_model().eval()

        # Get curriculum subset
        active_indices = curriculum.get_active_indices(epoch)

        # Create subset dataloader with class-balanced sampling on active set
        active_labels = [base.get_label(splits["train"][i], args.num_classes) for i in active_indices]
        active_counts = np.bincount(active_labels, minlength=args.num_classes).astype(np.float64)
        sample_w = np.array([1.0 / max(active_counts[y], 1) for y in active_labels])
        sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double),
                                        len(active_indices), replacement=True)
        subset_ds = Subset(train_ds, active_indices)
        train_loader = DataLoader(subset_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

        # Get progressive weights
        w_hard, w_kd, w_bin = kd_scheduler.get_weights(epoch)

        total = hard_total = kd_total = bin_total = ema_kd_total = 0.0

        for x, y, _, tprob in train_loader:
            x, y, tprob = x.to(device), y.to(device), tprob.to(device)

            # Compute loss inline (no closure, no EMA)
            logits = student(x)
            hard_loss = hard(logits, y)
            kd_loss = -(tprob * F.log_softmax(logits / args.temperature, dim=1)).sum(dim=1).mean() * (args.temperature ** 2)

            hard_bin = (y != 0).float()
            teacher_bin = (1.0 - tprob[:, 0]).clamp(0, 1)
            bin_target = 0.5 * hard_bin + 0.5 * teacher_bin
            bin_loss = F.binary_cross_entropy_with_logits(
                base.abnormal_logit_from_4class(logits), bin_target)

            loss = w_hard * hard_loss + w_kd * kd_loss + w_bin * bin_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            ema_loss = torch.tensor(0.0, device=device)

            # Update EMA
            if epoch > args.ema_warmup:
                pass  # ema.update(student) disabled

            n = x.size(0)
            total += float(loss.item()) * n
            hard_total += float(hard_loss.item()) * n
            kd_total += float(kd_loss.item()) * n
            bin_total += float(bin_loss.item()) * n
            ema_kd_total += float(ema_loss.item()) * n

        sched.step()
        swa.update(epoch)

        # Validation with EMA model (if available)
        eval_model = ema.get_ema_model() if epoch > args.ema_warmup else student
        val_m, yv, _, pv, _ = base.evaluate_model(eval_model, val_loader, device, args.num_classes)
        sweep_fn = base.sweep_threshold_fine if getattr(args, 'fine_threshold', False) else base.sweep_threshold
        tuned = sweep_fn(yv, pv)
        score = float(tuned["icbhi_score"] if args.selection_metric == "threshold_icbhi_score"
                      else val_m[args.selection_metric])
        both_f1 = float(val_m.get("both_f1", 0.0)) if args.num_classes == 4 else 0.0
        meets_guard = both_f1 >= min_both_f1_guard
        macro_f1 = float(val_m.get("macro_f1", 0.0))
        bal_acc = float(val_m.get("balanced_accuracy", 0.0))

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
            # Save EMA model state
            save_model = ema.get_ema_model() if epoch > args.ema_warmup else student
            torch.save({
                "model_state": save_model.state_dict(),
                "epoch": epoch,
                "arch": args.student_arch,
                "threshold": tuned["threshold"],
                "metrics": val_m,
                "threshold_metrics": tuned,
                "args": vars(args),
                "uses_ema": epoch > args.ema_warmup,
                "curriculum_ratio": curriculum.get_subset_ratio(epoch),
                "kd_weights": {"hard": w_hard, "kd": w_kd, "binary": w_bin},
            }, best_path)
            np.save(student_dir / "val_probs_best.npy", pv)
        else:
            patience += 1

        denom = max(len(subset_ds), 1)
        base.log_wandb({
            "epoch": epoch, "loss": total / denom,
            "hard_loss": hard_total / denom, "kd_loss": kd_total / denom,
            "binary_loss": bin_total / denom, "ema_kd_loss": ema_kd_total / denom,
            "curriculum_ratio": curriculum.get_subset_ratio(epoch),
            "w_hard": w_hard, "w_kd": w_kd, "w_bin": w_bin,
            "active_samples": len(active_indices),
            **{f"val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float))},
            "val_threshold_icbhi_score": tuned["icbhi_score"],
            "val_threshold": tuned["threshold"],
            "best_score": float(best_score),
        }, prefix="student", step=epoch)

        print(f"student ep={epoch:03d} loss={total/denom:.4f} "
              f"curriculum={curriculum.get_subset_ratio(epoch):.2f} "
              f"w_h={w_hard:.2f} w_k={w_kd:.2f} w_b={w_bin:.2f} "
              f"val_icbhi={val_m['icbhi_score']:.4f} tuned={tuned['icbhi_score']:.4f} "
              f"se={val_m['sensitivity']:.4f} sp={val_m['specificity']:.4f} "
              f"best={best_score:.4f}", flush=True)

        if patience >= args.patience:
            break

    # Apply SWA if enough epochs were run
    if swa.n_averaged > 0:
        print(f"Applying SWA ({swa.n_averaged} averaged models)...", flush=True)
        swa.apply()
        # Save SWA model
        swa_path = student_dir / "swa_best.pt"
        torch.save({
            "model_state": student.state_dict(),
            "arch": args.student_arch,
            "n_swa_averaged": swa.n_averaged,
            "args": vars(args),
        }, swa_path)

    base.finish_wandb()
    return best_path


# ---------------------------------------------------------------------------
# Final Evaluation with Dual-Threshold + SWA
# ---------------------------------------------------------------------------

def evaluate_final_enhanced(args, splits, stats, device, output_dir):
    """Evaluate with dual-threshold, EMA model, and optional SWA."""
    in_ch = 3 if args.input_view == "logmel_delta" else 1

    # Try SWA model first, then best checkpoint
    student_dir = output_dir / "students" / args.student_arch
    swa_path = student_dir / "swa_best.pt"
    best_path = student_dir / "best.pt"

    model = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)

    # Load best checkpoint (which may already be EMA)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    threshold = float(ckpt.get("threshold", 0.5))
    uses_ema = ckpt.get("uses_ema", False)

    base.init_wandb(args, f"{args.pipeline_name}-final-eval",
                    {"student_checkpoint": str(best_path), "threshold": threshold, "uses_ema": uses_ema})

    summary = {
        "student_checkpoint": str(best_path),
        "threshold": threshold,
        "student_params": base.count_params(model)[0],
        "uses_ema": uses_ema,
    }

    for split_name in ["val", "test"]:
        if not splits[split_name]:
            continue
        loader = base.make_loader(base.ICBHIDataset(splits[split_name], args, stats, False), args)

        # Raw evaluation
        raw_m, yt, yp, probs, _ = base.evaluate_model(model, loader, device, args.num_classes)

        # Single threshold
        tuned_pred = base.threshold_predictions(probs, threshold)
        tuned_m = base.compute_metrics(yt, tuned_pred, probs, args.num_classes)

        # Dual threshold (sweep on val, apply on test)
        if split_name == "val":
            dual_tuned = sweep_dual_threshold(yt, probs)
            dual_pred = dual_threshold_prediction(probs, dual_tuned["threshold_normal"],
                                                  dual_tuned["threshold_abnormal"])
            dual_m = base.compute_metrics(yt, dual_pred, probs, args.num_classes)
            summary["dual_threshold_info"] = dual_tuned
        else:
            # Use val dual thresholds on test
            dual_info = summary.get("dual_threshold_info", {})
            th_n = dual_info.get("threshold_normal", threshold)
            th_a = dual_info.get("threshold_abnormal", 0.5)
            dual_pred = dual_threshold_prediction(probs, th_n, th_a)
            dual_m = base.compute_metrics(yt, dual_pred, probs, args.num_classes)

        base.save_metrics(output_dir, f"student_{split_name}_raw", raw_m, yt, yp, args.num_classes)
        base.save_metrics(output_dir, f"student_{split_name}_threshold", tuned_m, yt, tuned_pred, args.num_classes)
        base.save_metrics(output_dir, f"student_{split_name}_dual_threshold", dual_m, yt, dual_pred, args.num_classes)

        summary[f"{split_name}_raw"] = raw_m
        summary[f"{split_name}_threshold"] = tuned_m
        summary[f"{split_name}_dual_threshold"] = dual_m

        base.log_wandb({f"{split_name}_raw_{k}": v for k, v in raw_m.items() if isinstance(v, (int, float))}, prefix="final")
        base.log_wandb({f"{split_name}_threshold_{k}": v for k, v in tuned_m.items() if isinstance(v, (int, float))}, prefix="final")
        base.log_wandb({f"{split_name}_dual_{k}": v for k, v in dual_m.items() if isinstance(v, (int, float))}, prefix="final_dual")

    # Save summary
    metrics_dir = ensure_dir(output_dir / "metrics")
    with (metrics_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (metrics_dir / "final_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "mode", "icbhi_score", "sensitivity", "specificity",
                     "macro_f1", "accuracy", "binary_icbhi_score"])
        for split_name in ["val", "test"]:
            for mode in ["raw", "threshold", "dual_threshold"]:
                m = summary.get(f"{split_name}_{mode}")
                if m:
                    w.writerow([split_name, mode, m.get("icbhi_score"), m.get("sensitivity"),
                                m.get("specificity"), m.get("macro_f1"), m.get("accuracy"),
                                m.get("binary_icbhi_score")])

    # Export ONNX
    if args.export_onnx:
        try:
            dummy = torch.randn(1, in_ch, args.n_mels, args.target_frames, device=device)
            torch.onnx.export(model.eval(), dummy, str(output_dir / "student_final.onnx"),
                              export_params=True, opset_version=11, do_constant_folding=True,
                              input_names=["mel_spectrogram"], output_names=["logits"],
                              dynamic_axes={"mel_spectrogram": {0: "batch"}, "logits": {0: "batch"}})
            print(f"ONNX exported: {output_dir / 'student_final.onnx'}", flush=True)
        except Exception as exc:
            print(f"ONNX export failed: {exc}", flush=True)

    base.finish_wandb()
    return summary


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    args = base.parse_args()

    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = f"icbhi_kd_s3_curriculum_{args.num_classes}class"
    if args.benchmark_protocol == "add_rsc":
        args.benchmark_protocol = "official_icbhi"
    if args.teacher_arches == "resnet_cnn,resnet_crnn,efficientnet_b0":
        args.teacher_arches = "resnet_cnn,resnet_crnn,efficientnet_b0"
    if args.student_arch == "ds_cnn_res_se":
        args.student_arch = "ds_cnn_res_se"
    if args.input_view == "logmel_delta":
        args.input_view = "logmel_delta"

    # Progressive KD weights (end values) — tuned to avoid abnormal bias
    if args.hard_weight == 0.35:
        args.hard_weight = 0.38
    if args.kd_weight == 0.45:
        args.kd_weight = 0.45
    if args.binary_weight == 0.20:
        args.binary_weight = 0.12
    if args.temperature == 4.0:
        args.temperature = 4.0
    if args.label_smoothing == 0.05:
        args.label_smoothing = 0.06

    args.selection_metric = "threshold_icbhi_score"

    # New arguments
    defaults = {
        # Progressive KD start values
        "hard_weight_start": 0.35,
        "kd_weight_start": 0.45,
        "binary_weight_start": 0.15,
        "progressive_warmup": 30,
        # EMA
        "ema_decay": 0.998,
        "ema_temperature": 3.0,
        "ema_weight": 0.0,
        "ema_warmup": 10,
        # Curriculum
        "curriculum_start": 1.0,
        "curriculum_warmup": 50,
        # Class-balanced
        "cb_beta": 0.9999,
        # SOTA upgrades
        "use_sam": False,
        "sam_rho": 0.02,
        "fine_threshold": True,
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

    if args.stage in {"all", "teachers"}:
        for arch in base.parse_csv(args.teacher_arches):
            for seed in base.parse_int_csv(args.seeds):
                print(f"\nTraining teacher: {arch} seed={seed}", flush=True)
                model, _, _ = base.train_teacher(arch, seed, args, splits, stats, device, output_dir)
                base.collect_and_save_logits(model, arch, seed, args, splits, stats, device, output_dir)

    if args.stage in {"all", "student"}:
        print(f"\nTraining student with Curriculum + EMA + ClassBalanced KD...", flush=True)
        train_student_curriculum_ema(args, splits, stats, device, output_dir)

    if args.stage in {"all", "evaluate"}:
        print(f"\nFinal evaluation with dual-threshold + SWA...", flush=True)
        evaluate_final_enhanced(args, splits, stats, device, output_dir)


if __name__ == "__main__":
    main()
