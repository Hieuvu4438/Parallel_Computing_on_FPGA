#!/usr/bin/env python3
"""
ICBHI 2017 Strategy 2 — Feature-Level Attention KD with Adversarial Training.

Key improvements over E1 (calibrated ensemble):
  1. Intermediate Feature Distillation — match spatial feature maps between
     teacher and student via learned projection heads (ReviewKD-style).
  2. Attention Transfer (AT) — transfer attention maps from teacher CNN layers
     to student, forcing student to attend to the same spatial regions.
  3. Adversarial Feature Matching — a small discriminator tries to distinguish
     teacher vs student features; student is trained to fool it.
  4. Relational Knowledge Distillation (RKD) — match pairwise distance
     structure in embedding space.
  5. Multi-level KD: logit KD + feature KD + attention KD + relational KD.
  6. Enhanced binary auxiliary loss with focal weighting for hard negatives.

Target: ICBHI Score > 66%, Specificity > 90%.

References:
  - Zagoruyko & Komodakis, "Paying More Attention to Attention", ICLR 2017.
  - Park et al., "Relational Knowledge Distillation", CVPR 2019.
  - Chen et al., "Cross-Layer Distillation with Semantic Calibration", AAAI 2021.
  - Tung & Mori, "Similarity-Preserving Knowledge Distillation", ICCV 2019.
  - Passalis & Tefas, "Learning Deep Representations with Probabilistic
    Knowledge Transfer", ECCV 2018.
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
# Feature-Extracting Model Wrappers
# ---------------------------------------------------------------------------

class FeatureExtractingStudent(nn.Module):
    """Wraps DSCNNResSEStudent to also return intermediate feature maps."""

    def __init__(self, student_model):
        super().__init__()
        self.student = student_model
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        blocks = list(self.student.features.children())
        # Capture features at 3 levels: early, mid, late
        if len(blocks) >= 3:
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


# ---------------------------------------------------------------------------
# Projection Heads for Feature Alignment
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Attention Map Extraction
# ---------------------------------------------------------------------------

def attention_map(feat):
    """Compute spatial attention map from feature tensor.

    Attention = sum_c |feat_c|^2, then normalize.
    Shape: [B, C, H, W] -> [B, 1, H, W]
    """
    am = (feat ** 2).sum(dim=1, keepdim=True)
    am = am / (am.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return am


# ---------------------------------------------------------------------------
# Distillation Losses
# ---------------------------------------------------------------------------

class AttentionTransferLoss(nn.Module):
    """Attention Transfer loss (Zagoruyko & Komodakis, ICLR 2017).

    L_AT = sum_l ||A_s^l - A_t^l||_2^2
    where A^l = normalize(sum_c |F_c^l|^2)
    """

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            am_s = attention_map(sf)
            am_t = attention_map(tf)
            # Resize to match if needed
            if am_s.shape != am_t.shape:
                am_t = F.adaptive_avg_pool2d(am_t, am_s.shape[2:])
            loss += F.mse_loss(am_s, am_t)
        return loss


class RelationalDistillationLoss(nn.Module):
    """Relational Knowledge Distillation (Park et al., CVPR 2019).

    Matches pairwise distance structure in embedding space.
    """

    def __init__(self, dist_type="l2"):
        super().__init__()
        self.dist_type = dist_type

    def forward(self, student_embeds, teacher_embeds):
        """Compute RKD on flattened embeddings."""
        # Flatten spatial dims
        s_flat = [f.flatten(1) for f in student_embeds]
        t_flat = [f.flatten(1) for f in teacher_embeds]

        loss = 0.0
        for sf, tf in zip(s_flat, t_flat):
            # Pairwise distance
            if self.dist_type == "l2":
                pdist_s = torch.cdist(sf, sf, p=2)
                pdist_t = torch.cdist(tf, tf, p=2)
            else:
                pdist_s = F.cosine_similarity(sf.unsqueeze(1), sf.unsqueeze(0), dim=-1)
                pdist_t = F.cosine_similarity(tf.unsqueeze(1), tf.unsqueeze(0), dim=-1)

            # Normalize
            pdist_s = pdist_s / (pdist_s.norm() + 1e-8)
            pdist_t = pdist_t / (pdist_t.norm() + 1e-8)

            loss += F.mse_loss(pdist_s, pdist_t)

        return loss


class FeatureDistillationLoss(nn.Module):
    """Feature-level distillation using learned projection + cosine/MSE loss."""

    def __init__(self, loss_type="cosine"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            if sf.shape != tf.shape:
                tf = F.adaptive_avg_pool2d(tf, sf.shape[2:])
            if self.loss_type == "cosine":
                # Cosine similarity loss (1 - cos_sim)
                sf_flat = sf.flatten(1)
                tf_flat = tf.flatten(1)
                cos_sim = F.cosine_similarity(sf_flat, tf_flat, dim=1)
                loss += (1 - cos_sim).mean()
            else:
                loss += F.mse_loss(sf, tf)
        return loss


# ---------------------------------------------------------------------------
# Adversarial Feature Discriminator
# ---------------------------------------------------------------------------

class FeatureDiscriminator(nn.Module):
    """Small discriminator for adversarial feature matching.

    Takes flattened features and predicts whether they come from teacher or student.
    """

    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Enhanced Student Training with Multi-Level KD
# ---------------------------------------------------------------------------

def get_feature_channels(student_arch, teacher_arches, nc, in_ch):
    """Get channel dimensions for feature projectors."""
    # Student feature channels (DSCNNResSEStudent)
    if student_arch == "ds_cnn_res_se":
        s_ch = [32, 64, 128]  # Approximate channel dims at each level
    else:
        s_ch = [16, 32, 64]  # CNN6

    # Teacher feature channels
    t_ch_map = {
        "resnet_cnn": [32, 64, 160],
        "resnet_crnn": [64, 96, 128],
        "efficientnet_b0": [24, 40, 112],
    }
    t_ch = []
    for arch in teacher_arches:
        t_ch.extend(t_ch_map.get(arch, [32, 64, 128]))

    return s_ch, t_ch


def train_student_with_feature_kd(args, splits, stats, device, output_dir):
    """Train student with multi-level KD: logit + feature + attention + relational + adversarial."""
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    teacher_arches = base.parse_csv(args.teacher_arches)

    # Load teacher logits (standard way)
    val_logits, teacher_names = base.load_teacher_logits(args, output_dir, "val", splits["val"])
    train_logits, _ = base.load_teacher_logits(args, output_dir, "train", splits["train"])
    weights = base.reliability_weights(val_logits, splits["val"], args.num_classes)
    train_probs = base.weighted_teacher_probs(train_logits, weights, args.temperature)

    # Create student model
    student_raw = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    student = FeatureExtractingStudent(student_raw).to(device)

    student_dir = ensure_dir(output_dir / "students" / args.student_arch)
    with (student_dir / "teacher_reliability.json").open("w", encoding="utf-8") as f:
        json.dump({"teacher_names": teacher_names, "class_weights": weights.tolist()}, f, indent=2)

    base.init_wandb(args, f"{args.pipeline_name}-student-{args.student_arch}",
                    {"student_params": base.count_params(student)[0], "teacher_names": teacher_names})

    # Create feature projectors for alignment
    s_ch, t_ch = get_feature_channels(args.student_arch, teacher_arches, args.num_classes, in_ch)
    # Use max channels from student levels as projection target
    proj_dim = 64
    projector = FeatureProjector([s_ch[0], s_ch[1], s_ch[2]], proj_dim).to(device)

    # Adversarial discriminator
    disc_feat_dim = proj_dim * 8 * 8  # After adaptive avg pool to 8x8
    discriminator = FeatureDiscriminator(disc_feat_dim).to(device)

    # Loss functions
    at_loss_fn = AttentionTransferLoss()
    rkd_loss_fn = RelationalDistillationLoss(dist_type="l2")
    feat_loss_fn = FeatureDistillationLoss(loss_type="cosine")
    bce_loss = nn.BCEWithLogitsLoss()

    hard = base.FocalLoss(base.class_weights(splits["train"], args.num_classes, device),
                          args.focal_gamma, args.label_smoothing)

    # Datasets
    base_train = base.ICBHIDataset(splits["train"], args, stats, augment=True)
    train_ds = base.StudentKDDataset(base_train, train_probs)
    sampler = WeightedRandomSampler(
        base.sample_weights(splits["train"], args.num_classes),
        len(splits["train"]), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = base.make_loader(base.ICBHIDataset(splits["val"], args, stats, False), args)

    # Optimizers
    student_params = list(student.parameters()) + list(projector.parameters())
    opt_s = torch.optim.AdamW(student_params, lr=args.lr_student, weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_student * 0.5,
                               weight_decay=args.weight_decay)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=max(args.epochs_student, 1))
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=max(args.epochs_student, 1))

    # Load teacher models for feature extraction
    teacher_models = []
    for arch in teacher_arches:
        for seed in base.parse_int_csv(args.seeds):
            ckpt_path = output_dir / "teachers" / arch / f"seed_{seed}" / "best.pt"
            if ckpt_path.exists():
                t_model = base.make_model(arch, args.num_classes, in_ch, args).to(device)
                ckpt = torch.load(ckpt_path, map_location=device)
                t_model.load_state_dict(ckpt["model_state"])
                t_model.eval()
                teacher_models.append(FeatureExtractingTeacher(t_model, arch))
                print(f"  Loaded teacher: {arch} seed={seed}", flush=True)

    # Training loop
    best_score, best_epoch, patience = -1.0, 0, 0
    best_tiebreak_macro = -1.0
    best_tiebreak_bal = -1.0
    best_tiebreak_both = -1.0
    min_both_f1_guard = 0.05 if args.num_classes == 4 else -1.0
    best_path = student_dir / "best.pt"

    for epoch in range(1, args.epochs_student + 1):
        student.train()
        projector.train()
        # Discriminator: alternate training
        if epoch % 2 == 0:
            discriminator.train()

        total = hard_total = kd_total = feat_total = attn_total = rkd_total = adv_total = bin_total = 0.0

        for x, y, _, tprob in train_loader:
            x, y, tprob = x.to(device), y.to(device), tprob.to(device)
            batch_size = x.size(0)

            # --- Student forward ---
            logits, s_feats = student(x)
            s_feats_projected = projector([s_feats.get("feat_early", torch.zeros(1)),
                                           s_feats.get("feat_mid", torch.zeros(1)),
                                           s_feats.get("feat_late", torch.zeros(1))])

            # --- Hard loss ---
            hard_loss = hard(logits, y)

            # --- Logit KD loss ---
            kd_loss = -(tprob * F.log_softmax(logits / args.temperature, dim=1)).sum(dim=1).mean() * (args.temperature ** 2)

            # --- Feature distillation (from first teacher as reference) ---
            feat_loss_val = torch.tensor(0.0, device=device)
            attn_loss_val = torch.tensor(0.0, device=device)
            rkd_loss_val = torch.tensor(0.0, device=device)

            if teacher_models and args.feat_weight > 0:
                with torch.no_grad():
                    t_model = teacher_models[epoch % len(teacher_models)]
                    _, t_feats = t_model(x)
                    t_feats_list = [t_feats.get(k, torch.zeros(1)) for k in ["feat_early", "feat_mid", "feat_late"]]

                # Project student features to match teacher
                t_projected = projector(t_feats_list)

                # Feature distillation
                feat_loss_val = feat_loss_fn(s_feats_projected, t_projected)

                # Attention transfer
                attn_loss_val = at_loss_fn(s_feats_projected, t_projected)

                # Relational distillation
                if batch_size > 2:  # RKD needs at least 2 samples
                    rkd_loss_val = rkd_loss_fn(s_feats_projected, t_projected)

            # --- Adversarial loss ---
            adv_loss_val = torch.tensor(0.0, device=device)
            if teacher_models and args.adv_weight > 0 and epoch > 5:
                # Get teacher features
                with torch.no_grad():
                    t_model = teacher_models[0]
                    _, t_feats = t_model(x)
                    t_proj = projector([t_feats.get(k, torch.zeros(1)) for k in ["feat_early", "feat_mid", "feat_late"]])
                    t_flat = torch.cat([F.adaptive_avg_pool2d(f, 8).flatten(1) for f in t_proj], dim=1)

                s_flat = torch.cat([F.adaptive_avg_pool2d(f, 8).flatten(1) for f in s_feats_projected], dim=1)

                # Train discriminator
                opt_d.zero_grad(set_to_none=True)
                d_real = discriminator(t_flat.detach())
                d_fake = discriminator(s_flat.detach())
                d_loss = -(torch.log(d_real + 1e-8).mean() + torch.log(1 - d_fake + 1e-8).mean())
                d_loss.backward()
                opt_d.step()

                # Generator (student) loss: fool discriminator
                d_fake_for_g = discriminator(s_flat)
                adv_loss_val = -torch.log(d_fake_for_g + 1e-8).mean()

            # --- Binary auxiliary loss ---
            hard_bin = (y != 0).float()
            teacher_bin = (1.0 - tprob[:, 0]).clamp(0, 1)
            bin_target = 0.5 * hard_bin + 0.5 * teacher_bin
            bin_loss = F.binary_cross_entropy_with_logits(
                base.abnormal_logit_from_4class(logits), bin_target)

            # --- Combined loss ---
            loss = (args.hard_weight * hard_loss
                    + args.kd_weight * kd_loss
                    + args.feat_weight * feat_loss_val
                    + args.attn_weight * attn_loss_val
                    + args.rkd_weight * rkd_loss_val
                    + args.adv_weight * adv_loss_val
                    + args.binary_weight * bin_loss)

            opt_s.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip)
            opt_s.step()

            n = batch_size
            total += float(loss.item()) * n
            hard_total += float(hard_loss.item()) * n
            kd_total += float(kd_loss.item()) * n
            feat_total += float(feat_loss_val.item()) * n
            attn_total += float(attn_loss_val.item()) * n
            rkd_total += float(rkd_loss_val.item()) * n
            adv_total += float(adv_loss_val.item()) * n
            bin_total += float(bin_loss.item()) * n

        sched_s.step()
        sched_d.step()

        # Validation (use raw student without feature extraction)
        val_m, yv, _, pv, _ = base.evaluate_model(student_raw, val_loader, device, args.num_classes)
        tuned = base.sweep_threshold(yv, pv)
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
            torch.save({
                "model_state": student_raw.state_dict(),
                "epoch": epoch,
                "arch": args.student_arch,
                "threshold": tuned["threshold"],
                "metrics": val_m,
                "threshold_metrics": tuned,
                "args": vars(args),
            }, best_path)
            np.save(student_dir / "val_probs_best.npy", pv)
        else:
            patience += 1

        denom = max(len(train_ds), 1)
        base.log_wandb({
            "epoch": epoch, "loss": total / denom,
            "hard_loss": hard_total / denom, "kd_loss": kd_total / denom,
            "feat_loss": feat_total / denom, "attn_loss": attn_total / denom,
            "rkd_loss": rkd_total / denom, "adv_loss": adv_total / denom,
            "binary_loss": bin_total / denom,
            **{f"val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float))},
            "val_threshold_icbhi_score": tuned["icbhi_score"],
            "val_threshold": tuned["threshold"],
            "best_score": float(best_score),
        }, prefix="student", step=epoch)

        print(f"student ep={epoch:03d} loss={total/denom:.4f} "
              f"hard={hard_total/denom:.4f} kd={kd_total/denom:.4f} "
              f"feat={feat_total/denom:.4f} attn={attn_total/denom:.4f} "
              f"val_icbhi={val_m['icbhi_score']:.4f} tuned={tuned['icbhi_score']:.4f} "
              f"se={val_m['sensitivity']:.4f} sp={val_m['specificity']:.4f} "
              f"best={best_score:.4f}", flush=True)

        if patience >= args.patience:
            break

    base.finish_wandb()
    return best_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    args = base.parse_args()

    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = "icbhi_kd_s2_feature_attention"
    if args.benchmark_protocol == "add_rsc":
        args.benchmark_protocol = "official_icbhi"
    if args.teacher_arches == "resnet_cnn,resnet_crnn,efficientnet_b0":
        args.teacher_arches = "resnet_cnn,resnet_crnn,efficientnet_b0"
    if args.student_arch == "ds_cnn_res_se":
        args.student_arch = "ds_cnn_res_se"
    if args.input_view == "logmel_delta":
        args.input_view = "logmel_delta"

    # Loss rebalancing for feature KD
    if args.hard_weight == 0.35:
        args.hard_weight = 0.25
    if args.kd_weight == 0.45:
        args.kd_weight = 0.35
    if args.binary_weight == 0.20:
        args.binary_weight = 0.15
    if args.temperature == 4.0:
        args.temperature = 4.0
    if args.label_smoothing == 0.05:
        args.label_smoothing = 0.06

    args.selection_metric = "threshold_icbhi_score"

    # New arguments for feature-level KD
    defaults = {
        "feat_weight": 0.10,
        "attn_weight": 0.05,
        "rkd_weight": 0.05,
        "adv_weight": 0.03,
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
        print(f"\nTraining student with Feature+Attention+RKD+Adversarial KD...", flush=True)
        train_student_with_feature_kd(args, splits, stats, device, output_dir)

    if args.stage in {"all", "evaluate"}:
        base.evaluate_final(args, splits, stats, device, output_dir)


if __name__ == "__main__":
    main()
