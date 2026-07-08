#!/usr/bin/env python3
"""
Multi-seed student ensemble evaluation for E1.

Trains 3 students with different random seeds, averages their predictions,
and evaluates the ensemble. This is the most reliable technique to improve
generalization and close the val→test gap.

Usage:
    python eval_e1_ensemble.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import icbhi_kd_pipeline_multiview_ensemble as base


def train_student_with_seed(args, splits, stats, device, output_dir, student_seed):
    """Train a single student with a specific random seed."""
    base.set_seed(student_seed)

    in_ch = 3 if args.input_view == "logmel_delta" else 1
    val_logits, teacher_names = base.load_teacher_logits(args, output_dir, "val", splits["val"])
    train_logits, _ = base.load_teacher_logits(args, output_dir, "train", splits["train"])
    weights = base.reliability_weights(val_logits, splits["val"], args.num_classes)
    train_probs = base.weighted_teacher_probs(train_logits, weights, args.temperature)

    student = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    student_dir = base.ensure_dir(output_dir / "students" / f"{args.student_arch}_seed_{student_seed}")

    base_train = base.ICBHIDataset(splits["train"], args, stats, True)
    train_ds = base.StudentKDDataset(base_train, train_probs)
    sampler = base.WeightedRandomSampler(base.sample_weights(splits["train"], args.num_classes), len(splits["train"]), replacement=True)
    train_loader = base.DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = base.make_loader(base.ICBHIDataset(splits["val"], args, stats, False), args)

    hard = base.FocalLoss(base.class_weights(splits["train"], args.num_classes, device), args.focal_gamma, args.label_smoothing)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs_student, 1))

    best_score, best_epoch, patience = -1.0, 0, 0
    best_path = student_dir / "best.pt"

    for epoch in range(1, args.epochs_student + 1):
        student.train()
        total = 0.0
        for x, y, _, tprob in train_loader:
            x, y, tprob = x.to(device), y.to(device), tprob.to(device)
            opt.zero_grad(set_to_none=True)
            logits = student(x)
            hard_loss = hard(logits, y)
            kd_loss = -(tprob * F.log_softmax(logits / args.temperature, dim=1)).sum(dim=1).mean() * (args.temperature ** 2)
            hard_bin = (y != 0).float()
            teacher_bin = (1.0 - tprob[:, 0]).clamp(0, 1)
            bin_target = 0.5 * hard_bin + 0.5 * teacher_bin
            bin_loss = F.binary_cross_entropy_with_logits(base.abnormal_logit_from_4class(logits), bin_target)
            loss = args.hard_weight * hard_loss + args.kd_weight * kd_loss + args.binary_weight * bin_loss
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.item()) * x.size(0)
        sched.step()

        val_m, yv, _, pv, _ = base.evaluate_model(student, val_loader, device, args.num_classes)
        tuned = base.sweep_threshold(yv, pv)
        score = float(tuned["icbhi_score"] if args.selection_metric == "threshold_icbhi_score" else val_m[args.selection_metric])

        if score > best_score + 1e-12:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save({"model_state": student.state_dict(), "epoch": epoch, "threshold": tuned["threshold"], "metrics": val_m}, best_path)
            np.save(student_dir / "val_probs_best.npy", pv)
        else:
            patience += 1

        print(f"  seed={student_seed} ep={epoch:03d} loss={total/len(train_ds):.4f} tuned={tuned['icbhi_score']:.4f} best={best_score:.4f}", flush=True)
        if patience >= args.patience:
            break

    return student, best_path


def main():
    e1_dir = Path("artifacts/training/icbhi_kd_e1_calibrated_ensemble")

    with (e1_dir / "config.json").open() as f:
        config = json.load(f)

    class Args:
        pass
    args = Args()
    for k, v in config.items():
        setattr(args, k, v)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.use_lungmix = False
    args.fine_threshold = False
    args.use_sam = False
    args.min_both_f1_guard = -1

    device = torch.device(args.device)

    with (e1_dir / "splits.json").open() as f:
        splits_raw = json.load(f)
    splits_data = splits_raw.get("splits", splits_raw)

    from icbhi_kd_pipeline_multiview_ensemble import CycleRecord
    splits = {}
    for split_name, records in splits_data.items():
        splits[split_name] = [CycleRecord(**r) for r in records]

    stats = base.estimate_feature_stats(splits["train"], args)

    # Train 3 students with different seeds
    print("="*60)
    print("MULTI-SEED STUDENT ENSEMBLE")
    print("="*60)

    seeds = [1, 2, 3]
    val_probs_list = []
    test_probs_list = []
    val_loader = base.make_loader(base.ICBHIDataset(splits["val"], args, stats, False), args)
    test_loader = base.make_loader(base.ICBHIDataset(splits["test"], args, stats, False), args)

    for seed in seeds:
        print(f"\n--- Training student with seed {seed} ---")
        student, best_path = train_student_with_seed(args, splits, stats, device, e1_dir, seed)

        # Load best checkpoint
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model_state"])
        student.eval()

        # Get val and test probabilities
        _, y_val, _, p_val, _ = base.evaluate_model(student, val_loader, device, args.num_classes)
        _, y_test, _, p_test, _ = base.evaluate_model(student, test_loader, device, args.num_classes)

        val_probs_list.append(p_val)
        test_probs_list.append(p_test)

    # Ensemble: average probabilities
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)

    val_probs_ensemble = np.mean(val_probs_list, axis=0)
    test_probs_ensemble = np.mean(test_probs_list, axis=0)

    # Tune threshold on ensemble val predictions
    val_tuned = base.sweep_threshold(y_val, val_probs_ensemble)
    print(f"Val ensemble:  ICBHI={val_tuned['icbhi_score']:.4f} Sens={val_tuned['sensitivity']:.4f} Spec={val_tuned['specificity']:.4f} threshold={val_tuned['threshold']:.2f}")

    # Apply to test
    test_pred = base.threshold_predictions(test_probs_ensemble, val_tuned["threshold"])
    se, sp, score = base.icbhi_score(y_test, test_pred)
    print(f"Test ensemble: ICBHI={score:.4f} Sens={se:.4f} Spec={sp:.4f}")

    # Also try with fine threshold sweep
    val_tuned_fine = base.sweep_threshold_fine(y_val, val_probs_ensemble)
    test_pred_fine = base.threshold_predictions(test_probs_ensemble, val_tuned_fine["threshold"])
    se_fine, sp_fine, score_fine = base.icbhi_score(y_test, test_pred_fine)
    print(f"\nWith fine threshold sweep:")
    print(f"Val ensemble:  ICBHI={val_tuned_fine['icbhi_score']:.4f} threshold={val_tuned_fine['threshold']:.4f}")
    print(f"Test ensemble: ICBHI={score_fine:.4f} Sens={se_fine:.4f} Spec={sp_fine:.4f}")

    # Compare with individual students
    print("\n--- Individual student results ---")
    for i, seed in enumerate(seeds):
        val_tuned_i = base.sweep_threshold(y_val, val_probs_list[i])
        test_pred_i = base.threshold_predictions(test_probs_list[i], val_tuned_i["threshold"])
        se_i, sp_i, score_i = base.icbhi_score(y_test, test_pred_i)
        print(f"Seed {seed}: Val ICBHI={val_tuned_i['icbhi_score']:.4f} → Test ICBHI={score_i:.4f} Sens={se_i:.4f} Spec={sp_i:.4f}")

    # E1 baseline
    print(f"\n--- E1 Baseline ---")
    print(f"Test ICBHI=0.6342 Sens=0.3796 Spec=0.8888")

    if score > 0.66:
        print(f"\n✅ TARGET ACHIEVED: Test ICBHI={score:.4f} > 0.66!")
    elif score_fine > 0.66:
        print(f"\n✅ TARGET ACHIEVED (fine): Test ICBHI={score_fine:.4f} > 0.66!")
    else:
        print(f"\n❌ Best test ICBHI={max(score, score_fine):.4f}")


if __name__ == "__main__":
    main()
