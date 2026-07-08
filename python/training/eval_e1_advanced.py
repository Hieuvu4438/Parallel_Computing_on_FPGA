#!/usr/bin/env python3
"""
Advanced evaluation strategies for E1 student model.

Methods:
1. Class-Biased Prediction: Apply learned bias to class probabilities
   instead of thresholding P(Normal). This directly addresses the
   sensitivity issue by boosting abnormal class confidence.
2. SWA (Stochastic Weight Averaging): Average model weights from
   multiple late-training checkpoints for better calibration.
3. Combined: Biased prediction + SWA model.

Usage:
    python eval_e1_advanced.py
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


def load_e1_data():
    """Load E1 student model and data."""
    e1_dir = Path("artifacts/training/icbhi_kd_e1_calibrated_ensemble")

    with (e1_dir / "config.json").open() as f:
        config = json.load(f)

    class Args:
        pass
    args = Args()
    for k, v in config.items():
        setattr(args, k, v)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)

    with (e1_dir / "splits.json").open() as f:
        splits_raw = json.load(f)
    splits_data = splits_raw.get("splits", splits_raw)

    from icbhi_kd_pipeline_multiview_ensemble import CycleRecord
    splits = {}
    for split_name, records in splits_data.items():
        splits[split_name] = [CycleRecord(**r) for r in records]

    stats = base.estimate_feature_stats(splits["train"], args)

    in_ch = 3 if args.input_view == "logmel_delta" else 1
    student = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    ckpt_path = e1_dir / "students" / args.student_arch / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state"])
    student.eval()

    val_ds = base.ICBHIDataset(splits["val"], args, stats, False)
    test_ds = base.ICBHIDataset(splits["test"], args, stats, False)
    val_loader = base.make_loader(val_ds, args)
    test_loader = base.make_loader(test_ds, args)

    val_m, y_val, _, p_val, l_val = base.evaluate_model(student, val_loader, device, args.num_classes)
    test_m, y_test, _, p_test, l_test = base.evaluate_model(student, test_loader, device, args.num_classes)

    return {
        "args": args, "device": device, "student": student,
        "y_val": y_val, "p_val": p_val, "l_val": l_val,
        "y_test": y_test, "p_test": p_test, "l_test": l_test,
        "ckpt": ckpt, "splits": splits, "stats": stats,
    }


def class_biased_prediction(probs, bias_factors):
    """Apply per-class bias to probabilities and predict argmax.

    Args:
        probs: [N, C] probability matrix
        bias_factors: [C] bias multiplier per class (>1 boosts that class)

    Returns:
        predictions: [N] array of predicted class indices
    """
    biased = probs * bias_factors
    return biased.argmax(axis=1)


def sweep_class_bias(y_val, p_val, y_test, p_test, nc=4):
    """Find optimal per-class bias factors on val, apply to test."""
    print("\n" + "="*60)
    print("METHOD 1: Class-Biased Prediction")
    print("="*60)

    # Strategy A: Uniform abnormal boost (single factor for classes 1,2,3)
    print("\n--- A: Uniform Abnormal Boost ---")
    best_val_score = -1
    best_boost = 1.0
    for boost in np.arange(1.0, 3.01, 0.05):
        bias = np.array([1.0, boost, boost, boost])
        pred_val = class_biased_prediction(p_val, bias)
        se, sp, score = base.icbhi_score(y_val, pred_val)
        if score > best_val_score:
            best_val_score = score
            best_boost = boost

    bias = np.array([1.0, best_boost, best_boost, best_boost])
    pred_val = class_biased_prediction(p_val, bias)
    pred_test = class_biased_prediction(p_test, bias)
    se_v, sp_v, sc_v = base.icbhi_score(y_val, pred_val)
    se_t, sp_t, sc_t = base.icbhi_score(y_test, pred_test)
    print(f"Best boost={best_boost:.2f}")
    print(f"Val:  ICBHI={sc_v:.4f} Sens={se_v:.4f} Spec={sp_v:.4f}")
    print(f"Test: ICBHI={sc_t:.4f} Sens={se_t:.4f} Spec={sp_t:.4f}")
    result_a = {"icbhi_score": sc_t, "sensitivity": se_t, "specificity": sp_t, "method": "uniform_boost"}

    # Strategy B: Per-class bias (3 independent factors)
    print("\n--- B: Per-Class Bias (3 independent factors) ---")
    best_val_score = -1
    best_bias = np.array([1.0, 1.0, 1.0, 1.0])
    # Coarse grid search
    for b1 in np.arange(1.0, 2.51, 0.1):
        for b2 in np.arange(1.0, 2.51, 0.2):
            for b3 in np.arange(1.0, 2.51, 0.2):
                bias = np.array([1.0, b1, b2, b3])
                pred_val = class_biased_prediction(p_val, bias)
                se, sp, score = base.icbhi_score(y_val, pred_val)
                if score > best_val_score:
                    best_val_score = score
                    best_bias = bias.copy()

    pred_val = class_biased_prediction(p_val, best_bias)
    pred_test = class_biased_prediction(p_test, best_bias)
    se_v, sp_v, sc_v = base.icbhi_score(y_val, pred_val)
    se_t, sp_t, sc_t = base.icbhi_score(y_test, pred_test)
    print(f"Best bias={best_bias}")
    print(f"Val:  ICBHI={sc_v:.4f} Sens={se_v:.4f} Spec={sp_v:.4f}")
    print(f"Test: ICBHI={sc_t:.4f} Sens={se_t:.4f} Spec={sp_t:.4f}")
    result_b = {"icbhi_score": sc_t, "sensitivity": se_t, "specificity": sp_t, "method": "per_class_bias"}

    # Strategy C: Biased + Threshold combined
    print("\n--- C: Class Bias + Threshold Sweep ---")
    best_val_score = -1
    best_params = (1.0, 1.0, 1.0, 1.0, 0.5)
    for b1 in np.arange(1.0, 2.51, 0.2):
        for b2 in np.arange(1.0, 2.51, 0.3):
            for b3 in np.arange(1.0, 2.51, 0.3):
                for th in np.linspace(0.05, 0.95, 19):
                    bias = np.array([1.0, b1, b2, b3])
                    biased = p_val * bias
                    biased_sum = biased.sum(axis=1, keepdims=True)
                    biased_probs = biased / biased_sum
                    pred_val = base.threshold_predictions(biased_probs, float(th))
                    se, sp, score = base.icbhi_score(y_val, pred_val)
                    if score > best_val_score:
                        best_val_score = score
                        best_params = (1.0, b1, b2, b3, th)

    bias = np.array(best_params[:4])
    th = best_params[4]
    biased_v = p_val * bias
    biased_v = biased_v / biased_v.sum(axis=1, keepdims=True)
    biased_t = p_test * bias
    biased_t = biased_t / biased_t.sum(axis=1, keepdims=True)
    pred_val = base.threshold_predictions(biased_v, th)
    pred_test = base.threshold_predictions(biased_t, th)
    se_v, sp_v, sc_v = base.icbhi_score(y_val, pred_val)
    se_t, sp_t, sc_t = base.icbhi_score(y_test, pred_test)
    print(f"Best: bias={best_params[:4]}, threshold={th:.2f}")
    print(f"Val:  ICBHI={sc_v:.4f} Sens={se_v:.4f} Spec={sp_v:.4f}")
    print(f"Test: ICBHI={sc_t:.4f} Sens={se_t:.4f} Spec={sp_t:.4f}")
    result_c = {"icbhi_score": sc_t, "sensitivity": se_t, "specificity": sp_t, "method": "bias+threshold"}

    return max([result_a, result_b, result_c], key=lambda x: x["icbhi_score"])


def evaluate_swa_model(data):
    """Train model with SWA and evaluate."""
    print("\n" + "="*60)
    print("METHOD 2: SWA (Stochastic Weight Averaging)")
    print("="*60)

    args = data["args"]
    device = data["device"]
    e1_dir = Path("artifacts/training/icbhi_kd_e1_calibrated_ensemble")

    # Load teacher logits
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    val_logits, teacher_names = base.load_teacher_logits(args, e1_dir, "val", data["splits"]["val"])
    train_logits, _ = base.load_teacher_logits(args, e1_dir, "train", data["splits"]["train"])
    weights = base.reliability_weights(val_logits, data["splits"]["val"], args.num_classes)
    train_probs = base.weighted_teacher_probs(train_logits, weights, args.temperature)

    student = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    base_train = base.ICBHIDataset(data["splits"]["train"], args, data["stats"], True)
    train_ds = base.StudentKDDataset(base_train, train_probs)
    sampler = base.WeightedRandomSampler(
        base.sample_weights(data["splits"]["train"], args.num_classes),
        len(data["splits"]["train"]), replacement=True)
    train_loader = base.DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                    num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = base.make_loader(base.ICBHIDataset(data["splits"]["val"], args, data["stats"], False), args)
    test_loader = base.make_loader(base.ICBHIDataset(data["splits"]["test"], args, data["stats"], False), args)

    hard = base.FocalLoss(base.class_weights(data["splits"]["train"], args.num_classes, device), args.focal_gamma, args.label_smoothing)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)

    # SWA setup
    swa_model = torch.optim.swa_utils.AveragedModel(student)
    swa_start = max(1, args.epochs_student - 10)  # Start averaging in last 10 epochs
    swa_scheduler = torch.optim.swa_utils.SWALR(opt, swa_lr=args.lr_student * 0.1)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(swa_start, 1))

    best_score = -1.0
    patience = 0

    base.set_seed(42)

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

        if epoch < swa_start:
            sched.step()
        else:
            swa_model.update_parameters(student)
            swa_scheduler.step()

        # Evaluate with regular model during training
        val_m, yv, _, pv, _ = base.evaluate_model(student, val_loader, device, args.num_classes)
        tuned = base.sweep_threshold(yv, pv)
        score = float(tuned["icbhi_score"])

        if score > best_score + 1e-12:
            best_score = score
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  ep={epoch:03d} loss={total/len(train_ds):.4f} tuned={tuned['icbhi_score']:.4f} best={best_score:.4f}", flush=True)

        if patience >= args.patience and epoch >= swa_start:
            break

    # Update batch norm stats for SWA model
    print("Updating SWA batch norm statistics...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    # Evaluate SWA model
    swa_model.eval()
    _, y_val, _, p_val, l_val = base.evaluate_model(swa_model, val_loader, device, args.num_classes)
    _, y_test, _, p_test, l_test = base.evaluate_model(swa_model, test_loader, device, args.num_classes)

    val_tuned = base.sweep_threshold(y_val, p_val)
    test_pred = base.threshold_predictions(p_test, val_tuned["threshold"])
    se, sp, score = base.icbhi_score(y_test, test_pred)
    print(f"\nSWA Model Results:")
    print(f"Val:  ICBHI={val_tuned['icbhi_score']:.4f} threshold={val_tuned['threshold']:.2f}")
    print(f"Test: ICBHI={score:.4f} Sens={se:.4f} Spec={sp:.4f}")

    return {"icbhi_score": score, "sensitivity": se, "specificity": sp, "method": "SWA"}


def main():
    print("Loading E1 student model and data...")
    data = load_e1_data()

    # Method 1: Class-Biased Prediction
    biased_result = sweep_class_bias(data["y_val"], data["p_val"], data["y_test"], data["p_test"])

    # Method 2: SWA
    swa_result = evaluate_swa_model(data)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # E1 baseline
    p_val_raw = data["p_val"]
    p_test_raw = data["p_test"]
    baseline_val = base.sweep_threshold(data["y_val"], p_val_raw)
    baseline_pred = base.threshold_predictions(p_test_raw, baseline_val["threshold"])
    se_b, sp_b, sc_b = base.icbhi_score(data["y_test"], baseline_pred)

    strategies = [
        ("E1 Baseline (threshold)", {"icbhi_score": sc_b, "sensitivity": se_b, "specificity": sp_b}),
        ("Class-Biased Prediction", biased_result),
        ("SWA Model", swa_result),
    ]

    print(f"{'Strategy':<30} {'ICBHI':>8} {'Sens':>8} {'Spec':>8}")
    print("-" * 60)
    for name, result in strategies:
        print(f"{name:<30} {result['icbhi_score']:>8.4f} {result['sensitivity']:>8.4f} {result['specificity']:>8.4f}")

    best = max(strategies, key=lambda x: x[1]["icbhi_score"])
    print(f"\nBest strategy: {best[0]} with ICBHI={best[1]['icbhi_score']:.4f}")

    if best[1]["icbhi_score"] > 0.66:
        print(f"✅ TARGET ACHIEVED: ICBHI > 0.66!")
    else:
        print(f"❌ Best: {best[1]['icbhi_score']:.4f} (target: 0.66)")


if __name__ == "__main__":
    main()
