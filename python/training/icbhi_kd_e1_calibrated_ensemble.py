#!/usr/bin/env python3
"""
ICBHI 2017 E1 — Calibrated multi-teacher ensemble KD.

Runnable implementation of Experiment 1 from
`docs/ICBHI_2017_KD_SOTA_EXPERIMENT_PLAN.md`:
  - official patient-wise ICBHI split by default
  - heterogeneous teacher ensemble
  - reliability-weighted teacher probabilities from validation metrics
  - CNN-only DS-CNN-Res-SE student
  - binary auxiliary objective and threshold-tuned ICBHI Score
  - full W&B support inherited from the base pipeline

Choose the benchmark task with:
  --num_classes 2   # Normal vs Abnormal
  --num_classes 4   # Normal, Crackle, Wheeze, Both
"""

from __future__ import annotations

import icbhi_kd_pipeline_multiview_ensemble as base


def parse_args():
    args = base.parse_args()

    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = "icbhi_kd_e1_calibrated_ensemble"
    if args.benchmark_protocol == "add_rsc":
        args.benchmark_protocol = "official_icbhi"
    if args.teacher_arches == "resnet_cnn,resnet_crnn,efficientnet_b0":
        args.teacher_arches = "resnet_cnn,resnet_crnn,efficientnet_b0"
    if args.student_arch == "ds_cnn_res_se":
        args.student_arch = "ds_cnn_res_se"
    if args.input_view == "logmel_delta":
        args.input_view = "logmel_delta"
    if args.hard_weight == 0.35:
        args.hard_weight = 0.35
    if args.kd_weight == 0.45:
        args.kd_weight = 0.45
    if args.binary_weight == 0.20:
        args.binary_weight = 0.20
    if args.temperature == 4.0:
        args.temperature = 4.0
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
