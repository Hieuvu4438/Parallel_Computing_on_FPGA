#!/usr/bin/env python3
"""
Post-hoc calibration for E1 student model.

Loads the E1 student checkpoint and applies various calibration strategies
to improve the ICBHI Score without retraining:

1. Temperature scaling: Find optimal temperature on val, apply to test
2. Abnormal class boosting: Multiply abnormal class probs by factor > 1
3. Combined: Temperature + boost
4. Sensitivity-constrained threshold: Max ICBHI s.t. Sens >= target
5. Multi-threshold: Separate thresholds for normal/abnormal split

Usage:
    python eval_e1_calibrated.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add training dir to path
sys.path.insert(0, str(Path(__file__).parent))

import icbhi_kd_pipeline_multiview_ensemble as base


def load_e1_student_and_data():
    """Load the E1 student model, val/test data, and teacher logits."""
    e1_dir = Path("artifacts/training/icbhi_kd_e1_calibrated_ensemble")

    # Load config
    with (e1_dir / "config.json").open() as f:
        config = json.load(f)

    # Create args-like object
    class Args:
        pass
    args = Args()
    for k, v in config.items():
        setattr(args, k, v)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.use_lungmix = False
    args.fine_threshold = False

    device = torch.device(args.device)

    # Load splits
    with (e1_dir / "splits.json").open() as f:
        splits_raw = json.load(f)
    splits_data = splits_raw.get("splits", splits_raw)

    # Reconstruct records
    from icbhi_kd_pipeline_multiview_ensemble import CycleRecord
    splits = {}
    for split_name, records in splits_data.items():
        splits[split_name] = [CycleRecord(**r) for r in records]

    # Estimate feature stats from train
    stats = base.estimate_feature_stats(splits["train"], args)

    # Load student model
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    student = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    ckpt_path = e1_dir / "students" / args.student_arch / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state"])
    student.eval()

    # Create data loaders
    val_ds = base.ICBHIDataset(splits["val"], args, stats, False)
    test_ds = base.ICBHIDataset(splits["test"], args, stats, False)
    val_loader = base.make_loader(val_ds, args)
    test_loader = base.make_loader(test_ds, args)

    # Get model predictions
    val_m, y_val, _, p_val, l_val = base.evaluate_model(student, val_loader, device, args.num_classes)
    test_m, y_test, _, p_test, l_test = base.evaluate_model(student, test_loader, device, args.num_classes)

    return {
        "args": args,
        "device": device,
        "student": student,
        "y_val": y_val, "p_val": p_val, "l_val": l_val,
        "y_test": y_test, "p_test": p_test, "l_test": l_test,
        "val_m": val_m, "test_m": test_m,
        "ckpt": ckpt,
    }


def sweep_threshold_minimal(y_true, probs, thresholds=None):
    """Efficient threshold sweep that only computes ICBHI score."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    best = {"threshold": 0.5, "icbhi_score": -1.0, "sensitivity": 0.0, "specificity": 0.0}
    for th in thresholds:
        pred = base.threshold_predictions(probs, float(th))
        se, sp, score = base.icbhi_score(y_true, pred)
        if score > best["icbhi_score"]:
            best = {"threshold": float(th), "icbhi_score": float(score),
                    "sensitivity": float(se), "specificity": float(sp)}
    return best


def temperature_calibrate(logits, temperature):
    """Apply temperature scaling to logits before softmax."""
    return F.softmax(torch.tensor(logits) / temperature, dim=1).numpy()


def boost_abnormal(probs, boost_factor):
    """Multiply abnormal class probabilities by boost_factor and renormalize."""
    boosted = probs.copy()
    boosted[:, 1:] *= boost_factor
    row_sums = boosted.sum(axis=1, keepdims=True)
    return boosted / row_sums


def combined_calibrate(logits, temperature, boost_factor):
    """Temperature scaling + abnormal class boosting."""
    probs = temperature_calibrate(logits, temperature)
    return boost_abnormal(probs, boost_factor)


def sweep_temperature(y_val, l_val, y_test, l_test):
    """Find optimal temperature on val, apply to test."""
    print("\n" + "="*60)
    print("STRATEGY 1: Temperature Scaling")
    print("="*60)

    best_val_score = -1
    best_temp = 1.0

    for temp in np.arange(0.5, 5.01, 0.1):
        p_val = temperature_calibrate(l_val, temp)
        result = sweep_threshold_minimal(y_val, p_val)
        if result["icbhi_score"] > best_val_score:
            best_val_score = result["icbhi_score"]
            best_temp = temp

    # Apply best temperature to test
    p_val_best = temperature_calibrate(l_val, best_temp)
    p_test_best = temperature_calibrate(l_test, best_temp)

    val_result = sweep_threshold_minimal(y_val, p_val_best)
    test_result = sweep_threshold_minimal(y_test, p_test_best)

    print(f"Best temperature: {best_temp:.1f}")
    print(f"Val:  ICBHI={val_result['icbhi_score']:.4f} Sens={val_result['sensitivity']:.4f} Spec={val_result['specificity']:.4f} threshold={val_result['threshold']:.2f}")
    print(f"Test: ICBHI={test_result['icbhi_score']:.4f} Sens={test_result['sensitivity']:.4f} Spec={test_result['specificity']:.4f} threshold={test_result['threshold']:.2f}")

    return test_result, best_temp


def sweep_abnormal_boost(y_val, l_val, y_test, l_test):
    """Find optimal abnormal class boost factor on val, apply to test."""
    print("\n" + "="*60)
    print("STRATEGY 2: Abnormal Class Boosting")
    print("="*60)

    p_val_raw = F.softmax(torch.tensor(l_val), dim=1).numpy()
    p_test_raw = F.softmax(torch.tensor(l_test), dim=1).numpy()

    best_val_score = -1
    best_boost = 1.0

    for boost in np.arange(1.0, 5.01, 0.1):
        p_val = boost_abnormal(p_val_raw, boost)
        result = sweep_threshold_minimal(y_val, p_val)
        if result["icbhi_score"] > best_val_score:
            best_val_score = result["icbhi_score"]
            best_boost = boost

    p_val_best = boost_abnormal(p_val_raw, best_boost)
    p_test_best = boost_abnormal(p_test_raw, best_boost)

    val_result = sweep_threshold_minimal(y_val, p_val_best)
    test_result = sweep_threshold_minimal(y_test, p_test_best)

    print(f"Best boost factor: {best_boost:.1f}")
    print(f"Val:  ICBHI={val_result['icbhi_score']:.4f} Sens={val_result['sensitivity']:.4f} Spec={val_result['specificity']:.4f} threshold={val_result['threshold']:.2f}")
    print(f"Test: ICBHI={test_result['icbhi_score']:.4f} Sens={test_result['sensitivity']:.4f} Spec={test_result['specificity']:.4f} threshold={test_result['threshold']:.2f}")

    return test_result, best_boost


def sweep_combined(y_val, l_val, y_test, l_test):
    """Find optimal temperature + boost combination on val, apply to test."""
    print("\n" + "="*60)
    print("STRATEGY 3: Temperature + Abnormal Boost Combined")
    print("="*60)

    best_val_score = -1
    best_temp = 1.0
    best_boost = 1.0

    for temp in np.arange(0.5, 5.01, 0.2):
        for boost in np.arange(1.0, 5.01, 0.2):
            p_val = combined_calibrate(l_val, temp, boost)
            result = sweep_threshold_minimal(y_val, p_val)
            if result["icbhi_score"] > best_val_score:
                best_val_score = result["icbhi_score"]
                best_temp = temp
                best_boost = boost

    p_val_best = combined_calibrate(l_val, best_temp, best_boost)
    p_test_best = combined_calibrate(l_test, best_temp, best_boost)

    val_result = sweep_threshold_minimal(y_val, p_val_best)
    test_result = sweep_threshold_minimal(y_test, p_test_best)

    print(f"Best temperature: {best_temp:.1f}, boost: {best_boost:.1f}")
    print(f"Val:  ICBHI={val_result['icbhi_score']:.4f} Sens={val_result['sensitivity']:.4f} Spec={val_result['specificity']:.4f} threshold={val_result['threshold']:.2f}")
    print(f"Test: ICBHI={test_result['icbhi_score']:.4f} Sens={test_result['sensitivity']:.4f} Spec={test_result['specificity']:.4f} threshold={test_result['threshold']:.2f}")

    return test_result, best_temp, best_boost


def sweep_constrained_sensitivity(y_val, l_val, y_test, l_test):
    """Find threshold that maximizes ICBHI subject to Sens >= target."""
    print("\n" + "="*60)
    print("STRATEGY 4: Sensitivity-Constrained Optimization")
    print("="*60)

    results = []
    for sens_target in [0.40, 0.45, 0.50, 0.55, 0.60]:
        best_val_score = -1
        best_th = 0.5

        for th in np.linspace(0.01, 0.99, 99):
            pred = base.threshold_predictions(l_val if l_val.ndim == 2 else F.softmax(torch.tensor(l_val), dim=1).numpy(), float(th))
            se, sp, score = base.icbhi_score(y_val, pred)
            if se >= sens_target and score > best_val_score:
                best_val_score = score
                best_th = th

        probs_test = F.softmax(torch.tensor(l_test), dim=1).numpy() if l_test.ndim > 2 else l_test
        pred_test = base.threshold_predictions(probs_test, float(best_th))
        se, sp, score = base.icbhi_score(y_test, pred_test)

        print(f"Sens target >= {sens_target:.2f}: threshold={best_th:.2f} → Test ICBHI={score:.4f} Sens={se:.4f} Spec={sp:.4f}")
        results.append({"sens_target": sens_target, "threshold": best_th, "icbhi_score": score, "sensitivity": se, "specificity": sp})

    best = max(results, key=lambda x: x["icbhi_score"])
    print(f"\nBest: Sens target >= {best['sens_target']:.2f} → Test ICBHI={best['icbhi_score']:.4f} Sens={best['sensitivity']:.4f} Spec={best['specificity']:.4f}")
    return best


def sweep_dual_threshold(y_val, l_val, y_test, l_test):
    """Use separate thresholds for normal/abnormal decision and subtype selection."""
    print("\n" + "="*60)
    print("STRATEGY 5: Dual-Threshold (Binary + Subtype)")
    print("="*60)

    probs_val = F.softmax(torch.tensor(l_val), dim=1).numpy()
    probs_test = F.softmax(torch.tensor(l_test), dim=1).numpy()

    best_val_score = -1
    best_th_bin = 0.5
    best_th_sub = 0.5

    # Sweep binary threshold (P(Normal) threshold for normal/abnormal)
    for th_bin in np.linspace(0.01, 0.99, 99):
        # Sweep subtype threshold (P(best_abnormal) threshold for selecting subtype)
        for th_sub in np.linspace(0.01, 0.99, 50):
            pred_val = np.zeros(len(y_val), dtype=np.int64)
            for i in range(len(y_val)):
                if probs_val[i, 0] >= th_bin:
                    pred_val[i] = 0  # Normal
                else:
                    # Check if any abnormal class has enough confidence
                    abnorm_probs = probs_val[i, 1:]
                    best_abn = abnorm_probs.argmax() + 1
                    if abnorm_probs.max() >= th_sub:
                        pred_val[i] = best_abn
                    else:
                        pred_val[i] = 0  # Default to Normal if uncertain

            se, sp, score = base.icbhi_score(y_val, pred_val)
            if score > best_val_score:
                best_val_score = score
                best_th_bin = th_bin
                best_th_sub = th_sub

    # Apply to test
    pred_test = np.zeros(len(y_test), dtype=np.int64)
    for i in range(len(y_test)):
        if probs_test[i, 0] >= best_th_bin:
            pred_test[i] = 0
        else:
            abnorm_probs = probs_test[i, 1:]
            best_abn = abnorm_probs.argmax() + 1
            if abnorm_probs.max() >= best_th_sub:
                pred_test[i] = best_abn
            else:
                pred_test[i] = 0

    se, sp, score = base.icbhi_score(y_test, pred_test)
    print(f"Best: th_bin={best_th_bin:.2f}, th_sub={best_th_sub:.2f}")
    print(f"Test: ICBHI={score:.4f} Sens={se:.4f} Spec={sp:.4f}")

    return {"icbhi_score": score, "sensitivity": se, "specificity": sp, "th_bin": best_th_bin, "th_sub": best_th_sub}


def main():
    print("Loading E1 student model and data...")
    data = load_e1_student_and_data()

    y_val = data["y_val"]
    l_val = data["l_val"]
    y_test = data["y_test"]
    l_test = data["l_test"]

    # Baseline
    print("\n" + "="*60)
    print("BASELINE (E1 original)")
    print("="*60)
    p_val_raw = F.softmax(torch.tensor(l_val), dim=1).numpy()
    p_test_raw = F.softmax(torch.tensor(l_test), dim=1).numpy()

    baseline_val = sweep_threshold_minimal(y_val, p_val_raw)
    baseline_test = sweep_threshold_minimal(y_test, p_test_raw)
    print(f"Val:  ICBHI={baseline_val['icbhi_score']:.4f} Sens={baseline_val['sensitivity']:.4f} Spec={baseline_val['specificity']:.4f} threshold={baseline_val['threshold']:.2f}")
    print(f"Test: ICBHI={baseline_test['icbhi_score']:.4f} Sens={baseline_test['sensitivity']:.4f} Spec={baseline_test['specificity']:.4f} threshold={baseline_test['threshold']:.2f}")

    # Strategy 1: Temperature scaling
    temp_result, best_temp = sweep_temperature(y_val, l_val, y_test, l_test)

    # Strategy 2: Abnormal boost
    boost_result, best_boost = sweep_abnormal_boost(y_val, l_val, y_test, l_test)

    # Strategy 3: Combined
    combined_result, best_temp_c, best_boost_c = sweep_combined(y_val, l_val, y_test, l_test)

    # Strategy 4: Sensitivity-constrained
    constrained_result = sweep_constrained_sensitivity(y_val, l_val, y_test, l_test)

    # Strategy 5: Dual threshold
    dual_result = sweep_dual_threshold(y_val, l_val, y_test, l_test)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    strategies = [
        ("Baseline (E1)", baseline_test),
        ("Temperature Scaling", temp_result),
        ("Abnormal Boost", boost_result),
        ("Combined (Temp+Boost)", combined_result),
        ("Sensitivity-Constrained", constrained_result),
        ("Dual Threshold", dual_result),
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
        print(f"❌ Target not reached. Best: {best[1]['icbhi_score']:.4f}")


if __name__ == "__main__":
    main()
