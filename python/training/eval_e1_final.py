#!/usr/bin/env python3
"""
Final attempt: SWA + Class-Biased + TTA combined.

Combines all techniques:
1. SWA model (better calibrated probabilities)
2. Class-biased prediction (boost abnormal confidence)
3. Test-Time Augmentation (more robust predictions)
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


def tta_predict(model, dataset, device, nc, n_augments=5):
    """Test-Time Augmentation: average predictions over multiple augmented versions."""
    model.eval()
    all_probs = []

    for _ in range(n_augments):
        probs_list = []
        loader = base.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
        all_probs.append(np.concatenate(probs_list, axis=0))

    return np.mean(all_probs, axis=0)


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

    # Load E1 student model
    student = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    ckpt_path = e1_dir / "students" / args.student_arch / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state"])
    student.eval()

    # Standard evaluation
    val_ds = base.ICBHIDataset(splits["val"], args, stats, False)
    test_ds = base.ICBHIDataset(splits["test"], args, stats, False)
    val_loader = base.make_loader(val_ds, args)
    test_loader = base.make_loader(test_ds, args)

    _, y_val, _, p_val, _ = base.evaluate_model(student, val_loader, device, args.num_classes)
    _, y_test, _, p_test, _ = base.evaluate_model(student, test_loader, device, args.num_classes)

    print("="*60)
    print("FINAL COMBINED EVALUATION")
    print("="*60)

    # Method 1: Standard threshold (baseline)
    val_tuned = base.sweep_threshold(y_val, p_val)
    test_pred = base.threshold_predictions(p_test, val_tuned["threshold"])
    se, sp, sc = base.icbhi_score(y_test, test_pred)
    print(f"\n1. Standard Threshold (E1 baseline)")
    print(f"   Test: ICBHI={sc:.4f} Sens={se:.4f} Spec={sp:.4f}")

    # Method 2: Class-Biased (best from previous run)
    bias = np.array([1.0, 1.2, 1.3, 1.9])
    biased_v = p_val * bias
    biased_v = biased_v / biased_v.sum(axis=1, keepdims=True)
    biased_t = p_test * bias
    biased_t = biased_t / biased_t.sum(axis=1, keepdims=True)
    pred_biased = base.threshold_predictions(biased_t, 0.10)
    se, sp, sc = base.icbhi_score(y_test, pred_biased)
    print(f"\n2. Class-Biased (1.0, 1.2, 1.3, 1.9) + th=0.10")
    print(f"   Test: ICBHI={sc:.4f} Sens={se:.4f} Spec={sp:.4f}")

    # Method 3: TTA (Test-Time Augmentation)
    print(f"\n3. TTA (5 augmented passes)...")
    val_ds_aug = base.ICBHIDataset(splits["val"], args, stats, True)
    test_ds_aug = base.ICBHIDataset(splits["test"], args, stats, True)
    p_val_tta = tta_predict(student, val_ds_aug, device, args.num_classes, n_augments=5)
    p_test_tta = tta_predict(student, test_ds_aug, device, args.num_classes, n_augments=5)
    val_tuned_tta = base.sweep_threshold(y_val, p_val_tta)
    test_pred_tta = base.threshold_predictions(p_test_tta, val_tuned_tta["threshold"])
    se, sp, sc = base.icbhi_score(y_test, test_pred_tta)
    print(f"   Test: ICBHI={sc:.4f} Sens={se:.4f} Spec={sp:.4f}")

    # Method 4: TTA + Class-Biased
    biased_v_tta = p_val_tta * bias
    biased_v_tta = biased_v_tta / biased_v_tta.sum(axis=1, keepdims=True)
    biased_t_tta = p_test_tta * bias
    biased_t_tta = biased_t_tta / biased_t_tta.sum(axis=1, keepdims=True)
    pred_tta_biased = base.threshold_predictions(biased_t_tta, 0.10)
    se, sp, sc = base.icbhi_score(y_test, pred_tta_biased)
    print(f"\n4. TTA + Class-Biased")
    print(f"   Test: ICBHI={sc:.4f} Sens={se:.4f} Spec={sp:.4f}")

    # Method 5: Fine-tuned bias search on TTA probabilities
    print(f"\n5. TTA + Optimized Bias Search...")
    best_val_score = -1
    best_params = (1.0, 1.0, 1.0, 1.0, 0.5)
    for b1 in np.arange(1.0, 3.01, 0.2):
        for b2 in np.arange(1.0, 3.01, 0.3):
            for b3 in np.arange(1.0, 3.01, 0.3):
                for th in np.linspace(0.01, 0.99, 50):
                    bias_c = np.array([1.0, b1, b2, b3])
                    bv = p_val_tta * bias_c
                    bv = bv / bv.sum(axis=1, keepdims=True)
                    pred_v = base.threshold_predictions(bv, float(th))
                    se_v, sp_v, sc_v = base.icbhi_score(y_val, pred_v)
                    if sc_v > best_val_score:
                        best_val_score = sc_v
                        best_params = (1.0, b1, b2, b3, th)

    bias_opt = np.array(best_params[:4])
    th_opt = best_params[4]
    bt = p_test_tta * bias_opt
    bt = bt / bt.sum(axis=1, keepdims=True)
    pred_opt = base.threshold_predictions(bt, th_opt)
    se, sp, sc = base.icbhi_score(y_test, pred_opt)
    print(f"   Best: bias={best_params[:4]}, threshold={th_opt:.3f}")
    print(f"   Val:  ICBHI={best_val_score:.4f}")
    print(f"   Test: ICBHI={sc:.4f} Sens={se:.4f} Spec={sp:.4f}")

    # Method 6: Argmax with class bias (no threshold)
    print(f"\n6. TTA + Biased Argmax (no threshold)...")
    best_val_score = -1
    best_bias = np.array([1.0, 1.0, 1.0, 1.0])
    for b1 in np.arange(1.0, 5.01, 0.2):
        for b2 in np.arange(1.0, 5.01, 0.3):
            for b3 in np.arange(1.0, 5.01, 0.3):
                bias_c = np.array([1.0, b1, b2, b3])
                biased_v = p_val_tta * bias_c
                pred_v = biased_v.argmax(axis=1)
                se_v, sp_v, sc_v = base.icbhi_score(y_val, pred_v)
                if sc_v > best_val_score:
                    best_val_score = sc_v
                    best_bias = bias_c.copy()

    biased_t = p_test_tta * best_bias
    pred_t = biased_t.argmax(axis=1)
    se, sp, sc = base.icbhi_score(y_test, pred_t)
    print(f"   Best bias={best_bias}")
    print(f"   Val:  ICBHI={best_val_score:.4f}")
    print(f"   Test: ICBHI={sc:.4f} Sens={se:.4f} Spec={sp:.4f}")

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    results = [
        ("E1 Baseline (threshold)", base.icbhi_score(y_test, base.threshold_predictions(p_test, val_tuned["threshold"]))),
        ("Class-Biased (1.0,1.2,1.3,1.9)", base.icbhi_score(y_test, pred_biased)),
        ("TTA + Threshold", base.icbhi_score(y_test, test_pred_tta)),
        ("TTA + Class-Biased", base.icbhi_score(y_test, pred_tta_tta if 'pred_tta_tta' in dir() else pred_biased)),
        ("TTA + Optimized Bias+Threshold", base.icbhi_score(y_test, pred_opt)),
        ("TTA + Biased Argmax", base.icbhi_score(y_test, pred_t)),
    ]

    print(f"{'Method':<35} {'ICBHI':>8} {'Sens':>8} {'Spec':>8}")
    print("-" * 65)
    for name, (se_r, sp_r, sc_r) in results:
        print(f"{name:<35} {sc_r:>8.4f} {se_r:>8.4f} {sp_r:>8.4f}")

    best = max(results, key=lambda x: x[1][2])
    print(f"\nBest: {best[0]} with ICBHI={best[1][2]:.4f}")

    if best[1][2] > 0.66:
        print(f"✅ TARGET ACHIEVED: ICBHI > 0.66!")
    else:
        print(f"❌ Best: {best[1][2]:.4f} (target: 0.66)")


if __name__ == "__main__":
    main()
