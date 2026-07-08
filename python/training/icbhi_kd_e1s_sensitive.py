#!/usr/bin/env python3
"""
ICBHI 2017 E1-S — Capacity + regularization optimized calibrated ensemble KD.

Improved variant of E1 targeting higher ICBHI Score through better
generalization (reducing the 0.691→0.634 val→test gap):

  1. Wider student (width=1.25 vs 1.0)
     — more model capacity to learn abnormal patterns better.
  2. More label smoothing (0.10 vs 0.05)
     — stronger regularization to reduce overfitting.
  3. Stronger augmentation (freq_mask=16, time_mask=64)
     — more aggressive SpecAugment for better generalization.
  4. Original E1 KD loss weights preserved
     — proven to work well for this task.

Usage:
  python icbhi_kd_e1s_sensitive.py --stage student
  python icbhi_kd_e1s_sensitive.py --stage evaluate
"""

from __future__ import annotations

import icbhi_kd_pipeline_multiview_ensemble as base


def parse_args():
    args = base.parse_args()

    # --- Identity ---
    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = "icbhi_kd_e1s_sensitive"
    if args.benchmark_protocol == "add_rsc":
        args.benchmark_protocol = "official_icbhi"

    # --- Teacher ensemble (same as E1) ---
    if args.teacher_arches == "resnet_cnn,resnet_crnn,efficientnet_b0":
        args.teacher_arches = "resnet_cnn,resnet_crnn,efficientnet_b0"
    if args.student_arch == "ds_cnn_res_se":
        args.student_arch = "ds_cnn_res_se"
    if args.input_view == "logmel_delta":
        args.input_view = "logmel_delta"

    # --- Wider student for more capacity ---
    if args.student_width == 1.0:
        args.student_width = 1.25

    # --- Keep original E1 loss weights ---
    if args.hard_weight == 0.35:
        args.hard_weight = 0.35
    if args.kd_weight == 0.45:
        args.kd_weight = 0.45
    if args.binary_weight == 0.20:
        args.binary_weight = 0.20
    if args.focal_gamma == 2.0:
        args.focal_gamma = 2.0
    if args.temperature == 4.0:
        args.temperature = 4.0

    # --- Stronger regularization ---
    if args.label_smoothing == 0.05:
        args.label_smoothing = 0.10

    # --- Stronger augmentation ---
    if args.freq_mask == 12:
        args.freq_mask = 16
    if args.time_mask == 48:
        args.time_mask = 64

    # --- Lower both_f1 guard (only 6 "Both" val samples → 0.05 is too strict) ---
    if args.min_both_f1_guard < 0:
        args.min_both_f1_guard = 0.01

    # --- Selection metric ---
    args.selection_metric = "threshold_icbhi_score"

    return args


def main():
    args = parse_args()
    base.set_seed(args.seed)
    device = base.default_device(args.device)
    output_dir, splits, stats = base.prepare_run(args)
    base.print_run_header(args, output_dir, splits)

    if args.stage in {"all", "teachers"}:
        for arch in base.parse_csv(args.teacher_arches):
            for seed in base.parse_int_csv(args.seeds):
                model, _, _ = base.train_teacher(arch, seed, args, splits, stats, device, output_dir)
                base.collect_and_save_logits(model, arch, seed, args, splits, stats, device, output_dir)

    if args.stage in {"all", "student"}:
        base.train_student(args, splits, stats, device, output_dir)

    if args.stage in {"all", "evaluate"}:
        base.evaluate_final(args, splits, stats, device, output_dir)


if __name__ == "__main__":
    main()
