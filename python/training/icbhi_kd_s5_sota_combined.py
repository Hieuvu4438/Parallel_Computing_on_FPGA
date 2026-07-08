#!/usr/bin/env python3
"""
ICBHI Strategy S5: SOTA-Combined KD Pipeline

Combines the best techniques from papers and analysis:

1. VTLP Frequency Warping (RSC-FTF: +3.19%)
2. Smoothed KD Loss (ADD-RSC: +0.5-1.0%)
3. Curated Teacher Ensemble (KD-Ensembles: +0.2-0.5%)
4. Class-Balanced Focal Loss (Cui et al. CVPR 2019)
5. Sensitivity-Aware Binary Loss (addresses Se << Sp)
6. SAM Optimizer (Foret et al. ICLR 2021: +3-5%)
7. Fine Threshold Sweep (+0.5-1%)
8. Multi-View TTA (RSC-FTF: +1.53%)

Key improvements over S2:
- Better augmentation (VTLP)
- Better loss function (smoothed KD + class-balanced focal)
- Better teacher selection (curated ensemble)
- Better evaluation (multi-view TTA)

Target: >0.69 ICBHI Score on official test set.

Usage:
    python python/training/icbhi_kd_s5_sota_combined.py --stage all
    python python/training/icbhi_kd_s5_sota_combined.py --stage student
    python python/training/icbhi_kd_s5_sota_combined.py --stage evaluate
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
from python.training.icbhi_sota_augmentation import vtlp_frequency_warping
from python.training.icbhi_sota_loss_functions import (
    ClassBalancedFocalLoss, SmoothedKDLoss, SensitivityAwareBinaryLoss,
    MultiTemperatureKDLoss
)
from python.training.icbhi_sota_evaluation import evaluate_with_tta


# ============================================================================
# Feature-Extracting Wrappers (from S2)
# ============================================================================

class FeatureExtractingStudent(nn.Module):
    """Wraps student model to extract intermediate features."""

    def __init__(self, student_model):
        super().__init__()
        self.student = student_model
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        blocks = list(self.student.features.children())
        if len(blocks) >= 6:
            blocks[1].register_forward_hook(self._make_hook("feat_early"))
            blocks[3].register_forward_hook(self._make_hook("feat_mid"))
            blocks[5].register_forward_hook(self._make_hook("feat_late"))

    def _make_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def forward(self, x):
        logits = self.student(x)
        return logits, {k: v for k, v in self._features.items()}


class FeatureExtractingTeacher(nn.Module):
    """Wraps teacher model to extract intermediate features."""

    def __init__(self, teacher_model, arch_name):
        super().__init__()
        self.teacher = teacher_model
        self.arch_name = arch_name
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        if "resnet_cnn" in self.arch_name:
            blocks = list(self.teacher.features.children())
            if len(blocks) >= 6:
                blocks[1].register_forward_hook(self._make_hook("feat_early"))
                blocks[3].register_forward_hook(self._make_hook("feat_mid"))
                blocks[5].register_forward_hook(self._make_hook("feat_late"))
        elif "crnn" in self.arch_name:
            self.teacher.cnn[1].register_forward_hook(self._make_hook("feat_early"))
            self.teacher.cnn[2].register_forward_hook(self._make_hook("feat_mid"))
            self.teacher.cnn[3].register_forward_hook(self._make_hook("feat_late"))
        elif "efficientnet" in self.arch_name:
            backbone = self.teacher.backbone.features
            backbone[3].register_forward_hook(self._make_hook("feat_early"))
            backbone[5].register_forward_hook(self._make_hook("feat_mid"))
            backbone[7].register_forward_hook(self._make_hook("feat_late"))

    def _make_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def forward(self, x):
        logits = self.teacher(x)
        return logits, {k: v for k, v in self._features.items()}


class FeatureProjector(nn.Module):
    """Projects features to a common dimension for distillation."""

    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(8),
                nn.Conv2d(ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for ch in in_channels_list
        ])

    def forward(self, features_list):
        return [proj(feat) for proj, feat in zip(self.projectors, features_list)]


class FeatureDistillationLoss(nn.Module):
    """Cosine similarity loss between projected features."""

    def __init__(self):
        super().__init__()

    def forward(self, s_feats, t_feats):
        loss = torch.tensor(0.0, device=s_feats[0].device)
        for sf, tf in zip(s_feats, t_feats):
            sf_flat = F.adaptive_avg_pool2d(sf, 1).flatten(1)
            tf_flat = F.adaptive_avg_pool2d(tf, 1).flatten(1)
            cos_sim = F.cosine_similarity(sf_flat, tf_flat, dim=1).mean()
            loss = loss + (1 - cos_sim)
        return loss / len(s_feats)


class AttentionTransferLoss(nn.Module):
    """Attention transfer loss between student and teacher features."""

    def __init__(self):
        super().__init__()

    def forward(self, s_feats, t_feats):
        loss = torch.tensor(0.0, device=s_feats[0].device)
        for sf, tf in zip(s_feats, t_feats):
            s_attn = sf.pow(2).sum(dim=1, keepdim=True)
            t_attn = tf.pow(2).sum(dim=1, keepdim=True)
            s_attn = F.normalize(s_attn.flatten(1), dim=1)
            t_attn = F.normalize(t_attn.flatten(1), dim=1)
            loss = loss + (s_attn - t_attn).pow(2).mean()
        return loss / len(s_feats)


# ============================================================================
# Dataset with VTLP augmentation
# ============================================================================

class VTLPStudentKDDataset(base.StudentKDDataset):
    """Student KD dataset with VTLP frequency warping augmentation."""

    def __init__(self, base_ds, teacher_probs, use_vtlp=True):
        super().__init__(base_ds, teacher_probs)
        self.use_vtlp = use_vtlp

    def __getitem__(self, idx):
        x, y, sample_idx, tprob = super().__getitem__(idx)
        if self.use_vtlp and self.base_ds.augment:
            # Apply VTLP at spectrogram level
            feat_np = x.numpy()
            feat_np = vtlp_frequency_warping(feat_np, alpha_range=(0.9, 1.1), prob=0.5)
            x = torch.from_numpy(feat_np.copy())
        return x, y, sample_idx, tprob


# ============================================================================
# Main Training Function
# ============================================================================

def train_student_s5(args, splits, stats, device, output_dir):
    """Train student with S5: SOTA-Combined KD."""
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    nc = args.num_classes

    # Load teacher logits
    val_logits, teacher_names = base.load_teacher_logits(args, output_dir, "val", splits["val"])
    train_logits, _ = base.load_teacher_logits(args, output_dir, "train", splits["train"])

    # Curated ensemble: select top-K teachers
    curated_k = getattr(args, 'curated_k', 5)
    if curated_k > 0 and curated_k < val_logits.shape[0]:
        top_k = base.select_top_teachers(val_logits, splits["val"], nc, k=curated_k)
        val_logits = val_logits[top_k]
        train_logits = train_logits[top_k]
        teacher_names = [teacher_names[i] for i in top_k]
        print(f"  Curated ensemble: {curated_k} teachers: {teacher_names}")

    # Compute reliability weights
    weights = base.reliability_weights(val_logits, splits["val"], nc)
    train_probs = base.weighted_teacher_probs(train_logits, weights, args.temperature)

    # Create student
    student_raw = base.make_model(args.student_arch, nc, in_ch, args).to(device)
    student = FeatureExtractingStudent(student_raw).to(device)

    student_dir = ensure_dir(output_dir / "students" / args.student_arch)
    with (student_dir / "teacher_reliability.json").open("w", encoding="utf-8") as f:
        json.dump({"teacher_names": teacher_names, "class_weights": weights.tolist()}, f, indent=2)

    base.init_wandb(args, f"{args.pipeline_name}-student-{args.student_arch}",
                    {"student_params": base.count_params(student)[0], "teacher_names": teacher_names})

    # Feature projectors
    s_ch = [32, 64, 128]  # DSCNNResSE channels
    proj_dim = 64
    student_projector = FeatureProjector(s_ch, proj_dim).to(device)

    t_ch_map = {
        "resnet_cnn": [32, 64, 160],
        "resnet_crnn": [64, 96, 128],
        "efficientnet_b0": [40, 112, 320],
    }
    unique_archs = list(dict.fromkeys(base.parse_csv(args.teacher_arches)))
    teacher_projectors = nn.ModuleDict()
    for arch in unique_archs:
        ch = t_ch_map.get(arch, [32, 64, 128])
        teacher_projectors[arch] = FeatureProjector(ch[:3], proj_dim).to(device)

    # Loss functions
    counts = np.bincount([base.get_label(r, nc) for r in splits["train"]], minlength=nc).astype(np.float64)
    focal_loss = ClassBalancedFocalLoss(counts, beta=0.9999, gamma=2.5, label_smoothing=0.05)
    kd_loss_fn = SmoothedKDLoss(temperature=args.temperature, smoothing=getattr(args, 'kd_smoothing', 0.15))
    binary_loss_fn = SensitivityAwareBinaryLoss(teacher_ratio=0.4)
    feat_loss_fn = FeatureDistillationLoss()
    attn_loss_fn = AttentionTransferLoss()

    # Datasets with VTLP
    base_train = base.ICBHIDataset(splits["train"], args, stats, augment=True)
    train_ds = VTLPStudentKDDataset(base_train, train_probs, use_vtlp=getattr(args, 'use_vtlp', True))
    sampler = WeightedRandomSampler(
        base.sample_weights(splits["train"], nc), len(splits["train"]), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = base.make_loader(base.ICBHIDataset(splits["val"], args, stats, False), args)

    # Optimizer (SAM)
    use_sam = getattr(args, 'use_sam', True)
    teacher_proj_params = []
    for proj in teacher_projectors.values():
        teacher_proj_params += list(proj.parameters())
    all_params = list(student.parameters()) + list(student_projector.parameters()) + teacher_proj_params

    if use_sam:
        opt = base.make_sam_optimizer(all_params, lr=args.lr_student,
                                      weight_decay=args.weight_decay,
                                      rho=getattr(args, 'sam_rho', 0.02))
    else:
        opt = torch.optim.AdamW(all_params, lr=args.lr_student, weight_decay=args.weight_decay)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt.base_optimizer if use_sam else opt, T_max=max(args.epochs_student, 1))

    # Load teacher models for feature extraction
    teacher_models = []
    for arch in base.parse_csv(args.teacher_arches):
        for seed in base.parse_int_csv(args.seeds):
            ckpt_path = output_dir / "teachers" / arch / f"seed_{seed}" / "best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device)
                ckpt_nc = ckpt.get("args", {}).get("num_classes", nc)
                t_model = base.make_model(arch, ckpt_nc, in_ch, args).to(device)
                t_model.load_state_dict(ckpt["model_state"])
                t_model.eval()
                teacher_models.append(FeatureExtractingTeacher(t_model, arch))

    # Training loop
    best_score, best_epoch, patience = -1.0, 0, 0
    best_tiebreak_macro = -1.0
    best_tiebreak_bal = -1.0
    min_both_f1_guard = 0.05 if nc == 4 else -1.0
    best_path = student_dir / "best.pt"

    for epoch in range(1, args.epochs_student + 1):
        student.train()
        student_projector.train()
        for proj in teacher_projectors.values():
            proj.train()

        total = hard_total = kd_total = feat_total = attn_total = bin_total = 0.0
        n_batches = 0

        for x, y, _, tprob in train_loader:
            x, y, tprob = x.to(device), y.to(device), tprob.to(device)
            batch_size = x.size(0)

            # Student forward
            logits, s_feats = student(x)

            # Hard loss (class-balanced focal)
            hard_loss = focal_loss(logits, y)

            # Smoothed KD loss
            kd_loss = kd_loss_fn(logits, tprob)

            # Feature distillation
            feat_loss_val = torch.tensor(0.0, device=device)
            attn_loss_val = torch.tensor(0.0, device=device)

            if teacher_models and getattr(args, 'feat_weight', 0.1) > 0:
                with torch.no_grad():
                    t_model = teacher_models[epoch % len(teacher_models)]
                    _, t_feats = t_model(x)
                    t_feats_list = [t_feats.get(k, torch.zeros(1)) for k in ["feat_early", "feat_mid", "feat_late"]]

                t_arch = t_model.arch_name
                t_proj = teacher_projectors[t_arch] if t_arch in teacher_projectors else list(teacher_projectors.values())[0]
                t_projected = t_proj(t_feats_list)

                s_projected = student_projector([
                    s_feats.get("feat_early", torch.zeros(1)),
                    s_feats.get("feat_mid", torch.zeros(1)),
                    s_feats.get("feat_late", torch.zeros(1))
                ])

                feat_loss_val = feat_loss_fn(s_projected, t_projected)
                attn_loss_val = attn_loss_fn(s_projected, t_projected)

            # Binary loss
            bin_loss = binary_loss_fn(logits, y, tprob)

            # Total loss
            w_hard = getattr(args, 'hard_weight', 0.35)
            w_kd = getattr(args, 'kd_weight', 0.45)
            w_bin = getattr(args, 'binary_weight', 0.15)
            w_feat = getattr(args, 'feat_weight', 0.10)
            w_attn = getattr(args, 'attn_weight', 0.05)

            loss = (w_hard * hard_loss + w_kd * kd_loss + w_bin * bin_loss +
                    w_feat * feat_loss_val + w_attn * attn_loss_val)

            # Optimizer step
            if use_sam:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                    nn.utils.clip_grad_norm_(student_projector.parameters(), args.grad_clip)
                    for proj in teacher_projectors.values():
                        nn.utils.clip_grad_norm_(proj.parameters(), args.grad_clip)
                opt.first_step(zero_grad=True)

                # SAM second step
                logits2, _ = student(x)
                kd_loss2 = kd_loss_fn(logits2, tprob)
                hard_loss2 = focal_loss(logits2, y)
                loss2 = w_hard * hard_loss2 + w_kd * kd_loss2
                loss2.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                opt.second_step(zero_grad=True)
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                opt.step()

            n = batch_size
            total += float(loss.item()) * n
            hard_total += float(hard_loss.item()) * n
            kd_total += float(kd_loss.item()) * n
            feat_total += float(feat_loss_val.item()) * n
            attn_total += float(attn_loss_val.item()) * n
            bin_total += float(bin_loss.item()) * n
            n_batches += 1

        sched.step()

        # Validation
        val_m, yv, _, pv, _ = base.evaluate_model(student_raw, val_loader, device, nc)
        sweep_fn = base.sweep_threshold_fine if getattr(args, 'fine_threshold', True) else base.sweep_threshold
        tuned = sweep_fn(yv, pv)
        score = float(tuned["icbhi_score"] if args.selection_metric == "threshold_icbhi_score" else val_m[args.selection_metric])
        both_f1 = float(val_m.get("both_f1", 0.0)) if nc == 4 else 0.0
        meets_guard = both_f1 >= min_both_f1_guard
        macro_f1 = float(val_m.get("macro_f1", 0.0))
        bal_acc = float(val_m.get("balanced_accuracy", 0.0))

        # Checkpoint selection
        better_primary = score > best_score + 1e-12
        tie_primary = abs(score - best_score) <= 1e-12
        better_tiebreak = tie_primary and (
            (macro_f1 > best_tiebreak_macro + 1e-12)
            or (abs(macro_f1 - best_tiebreak_macro) <= 1e-12 and bal_acc > best_tiebreak_bal + 1e-12)
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
            torch.save({
                "model_state": student_raw.state_dict(),
                "epoch": epoch, "arch": args.student_arch,
                "threshold": tuned["threshold"],
                "metrics": val_m,
                "threshold_metrics": tuned,
                "args": vars(args),
            }, best_path)
            np.save(student_dir / "val_probs_best.npy", pv)
        else:
            patience += 1

        denom = max(len(train_ds), 1)
        print(f"student ep={epoch:03d} loss={total/denom:.4f} "
              f"hard={hard_total/denom:.4f} kd={kd_total/denom:.4f} "
              f"feat={feat_total/denom:.4f} attn={attn_total/denom:.4f} "
              f"bin={bin_total/denom:.4f} "
              f"val_icbhi={val_m['icbhi_score']:.4f} tuned={tuned['icbhi_score']:.4f} "
              f"se={tuned['sensitivity']:.4f} sp={tuned['specificity']:.4f} "
              f"macro={macro_f1:.4f} best={best_score:.4f}", flush=True)

        if patience >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    base.finish_wandb()
    return best_path


# ============================================================================
# Final Evaluation
# ============================================================================

def evaluate_final_s5(args, splits, stats, device, output_dir):
    """Final evaluation with TTA."""
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    nc = args.num_classes

    student = base.make_model(args.student_arch, nc, in_ch, args).to(device)
    ckpt_path = output_dir / "students" / args.student_arch / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["model_state"])

    summary = {
        "student_checkpoint": str(ckpt_path),
        "threshold": float(ckpt.get("threshold", 0.5)),
        "student_params": base.count_params(student)[0],
    }

    for split in ["val", "test"]:
        if not splits[split]:
            continue
        loader = base.make_loader(base.ICBHIDataset(splits[split], args, stats, False), args)

        # Standard evaluation
        raw_m, yt, yp, probs, _ = base.evaluate_model(student, loader, device, nc)
        threshold = float(ckpt.get("threshold", 0.5))
        tuned_pred = base.threshold_predictions(probs, threshold)
        tuned_m = base.compute_metrics(yt, tuned_pred, probs, nc)

        # TTA evaluation
        tta_m, tta_th = evaluate_with_tta(student, loader, device, nc, n_tta=7)

        base.save_metrics(output_dir, f"student_{split}_raw", raw_m, yt, yp, nc)
        base.save_metrics(output_dir, f"student_{split}_threshold", tuned_m, yt, tuned_pred, nc)
        base.save_metrics(output_dir, f"student_{split}_tta", tta_m, yt,
                          base.threshold_predictions(probs, tta_th), nc)

        summary[f"{split}_raw"] = raw_m
        summary[f"{split}_threshold"] = tuned_m
        summary[f"{split}_tta"] = tta_m

        print(f"\n{split.upper()}:")
        print(f"  Raw:       ICBHI={raw_m['icbhi_score']:.4f} Se={raw_m['sensitivity']:.4f} Sp={raw_m['specificity']:.4f}")
        print(f"  Threshold: ICBHI={tuned_m['icbhi_score']:.4f} Se={tuned_m['sensitivity']:.4f} Sp={tuned_m['specificity']:.4f}")
        print(f"  TTA:       ICBHI={tta_m['icbhi_score']:.4f} Se={tta_m['sensitivity']:.4f} Sp={tta_m['specificity']:.4f}")

    with (ensure_dir(output_dir / "metrics") / "final_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    args = base.parse_args()

    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = "icbhi_kd_s5_sota_combined"

    # S5 defaults
    defaults = {
        # Feature KD weights
        "feat_weight": 0.10,
        "attn_weight": 0.05,
        # SAM
        "use_sam": True,
        "sam_rho": 0.02,
        # VTLP augmentation
        "use_vtlp": True,
        # Smoothed KD
        "kd_smoothing": 0.15,
        # Curated ensemble
        "curated_k": 5,
        # Fine threshold
        "fine_threshold": True,
        # Selection
        "selection_metric": "threshold_icbhi_score",
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    return args


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    base.set_seed(args.seed)
    device = base.default_device(args.device)
    output_dir, splits, stats = base.prepare_run(args)
    base.print_run_header(args, output_dir, splits)

    # Stage 1: Train teachers
    if args.stage in {"all", "teachers"}:
        for arch in base.parse_csv(args.teacher_arches):
            for seed in base.parse_int_csv(args.seeds):
                print(f"\n{'='*60}")
                print(f"Training teacher: {arch} seed={seed}")
                print(f"{'='*60}")
                model, _, _ = base.train_teacher(arch, seed, args, splits, stats, device, output_dir)
                base.collect_and_save_logits(model, arch, seed, args, splits, stats, device, output_dir)

    # Stage 2: Train student
    if args.stage in {"all", "student"}:
        print(f"\n{'='*60}")
        print("Training student with S5: SOTA-Combined KD")
        print(f"{'='*60}")
        train_student_s5(args, splits, stats, device, output_dir)

    # Stage 3: Evaluate
    if args.stage in {"all", "evaluate"}:
        print(f"\n{'='*60}")
        print("Final evaluation with TTA")
        print(f"{'='*60}")
        evaluate_final_s5(args, splits, stats, device, output_dir)


if __name__ == "__main__":
    main()
