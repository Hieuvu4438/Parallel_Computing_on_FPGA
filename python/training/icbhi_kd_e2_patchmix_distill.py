#!/usr/bin/env python3
"""
ICBHI 2017 E2 — Patch-Mix-inspired KD.

Runnable local implementation of Experiment 2 from
`docs/ICBHI_2017_KD_SOTA_EXPERIMENT_PLAN.md`.

This script does not require the external Patch-Mix AST repository or checkpoints.
Instead, it uses the local ICBHI KD base pipeline with defaults that approximate the
Patch-Mix distillation intent:
  - official patient-wise ICBHI split by default
  - heterogeneous teacher ensemble
  - stronger soft-target KD weight
  - stronger spectrogram masking/augmentation than E1
  - CNN-only DS-CNN-Res-SE student
  - threshold-tuned ICBHI Score and W&B logging

Future extension point: replace one local teacher with an external Patch-Mix AST
teacher and save its logits in the base pipeline's `teacher_logits/` format.

Choose the benchmark task with:
  --num_classes 2   # Normal vs Abnormal
  --num_classes 4   # Normal, Crackle, Wheeze, Both
"""

from __future__ import annotations

import icbhi_kd_pipeline_multiview_ensemble as base


def parse_args():
    args = base.parse_args()

    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = "icbhi_kd_e2_patchmix_distill"
    if args.benchmark_protocol == "add_rsc":
        args.benchmark_protocol = "official_icbhi"
    if args.teacher_arches == "resnet_cnn,resnet_crnn,efficientnet_b0":
        args.teacher_arches = "resnet_cnn,resnet_crnn,efficientnet_b0"
    if args.student_arch == "ds_cnn_res_se":
        args.student_arch = "ds_cnn_res_se"
    if args.input_view == "logmel_delta":
        args.input_view = "logmel_delta"
    if args.hard_weight == 0.35:
        args.hard_weight = 0.30
    if args.kd_weight == 0.45:
        args.kd_weight = 0.55
    if args.binary_weight == 0.20:
        args.binary_weight = 0.15
    if args.temperature == 4.0:
        args.temperature = 5.0
    if args.freq_mask == 12:
        args.freq_mask = 16
    if args.time_mask == 48:
        args.time_mask = 64
    if args.time_shift == 0.1:
        args.time_shift = 0.12
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
