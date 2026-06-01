#!/usr/bin/env python3
"""
ICBHI 2017 E3 — PAFA-inspired patient-safe KD.

Runnable local implementation of Experiment 3 from
`docs/ICBHI_2017_KD_SOTA_EXPERIMENT_PLAN.md`.

NOTE (2026-05): The previous E3 defaults were tuned toward a stronger binary
auxiliary objective + narrower band and ended up underperforming against E1.
This entrypoint now defaults to an **E1-strong baseline** (teacher diversity,
logmel+delta, wider band, and KD-heavy weighting) so we can iterate toward true
PAFA relational KD from a competitive starting point.

Choose the benchmark task with:
  --num_classes 2   # Normal vs Abnormal
  --num_classes 4   # Normal, Crackle, Wheeze, Both
"""

from __future__ import annotations

import icbhi_kd_pipeline_multiview_ensemble as base


def parse_args():
    args = base.parse_args()

    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = "icbhi_kd_e3_pafa_relational"
    if args.benchmark_protocol == "add_rsc":
        args.benchmark_protocol = "official_icbhi"

    # Start from a strong E1-like baseline (better teacher/student ceiling).
    if args.teacher_arches == "resnet_cnn,resnet_crnn,efficientnet_b0":
        args.teacher_arches = "resnet_cnn,resnet_crnn,efficientnet_b0"
    if args.student_arch == "ds_cnn_res_se":
        args.student_arch = "ds_cnn_res_se"
    if args.input_view == "logmel_delta":
        args.input_view = "logmel_delta"
    if args.student_width == 1.0:
        args.student_width = 1.25
    if args.f_max == 4000.0:
        args.f_max = 4000.0

    # Distillation weighting closer to E1 (avoid over-emphasizing binary aux).
    if args.hard_weight == 0.35:
        args.hard_weight = 0.35
    if args.kd_weight == 0.45:
        args.kd_weight = 0.45
    if args.binary_weight == 0.20:
        args.binary_weight = 0.20
    if args.temperature == 4.0:
        args.temperature = 4.0

    # Augmentation stronger by default.
    if args.freq_mask == 12:
        args.freq_mask = 12
    if args.time_mask == 48:
        args.time_mask = 48

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
