#!/usr/bin/env python3
"""
ICBHI 2017 official cycle-level KD pipeline — Binary-auxiliary sensitivity variant.

This is a complete end-to-end KD experiment pipeline built on the same strategy as
`icbhi_kd_pipeline_multiview_ensemble.py`, but with defaults biased toward improving
normal-vs-pathological separation and ICBHI Sensitivity:
  - CRNN/CNN teacher ensemble
  - stronger binary auxiliary loss weight
  - threshold-tuned checkpoint selection
  - CNN-only DS-CNN-Res-SE student
  - full W&B tracking
"""

from __future__ import annotations

import icbhi_kd_pipeline_multiview_ensemble as base


def parse_args():
    args = base.parse_args()
    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = "icbhi_kd_binary_aux_sensitivity"
    if args.teacher_arches == "resnet_cnn,resnet_crnn,efficientnet_b0":
        args.teacher_arches = "resnet_crnn,resnet_cnn"
    if args.input_view == "logmel_delta":
        args.input_view = "logmel"
    if args.f_max == 4000.0:
        args.f_max = 2500.0
    if args.binary_weight == 0.20:
        args.binary_weight = 0.35
    if args.hard_weight == 0.35:
        args.hard_weight = 0.30
    if args.kd_weight == 0.45:
        args.kd_weight = 0.35
    if args.student_width == 1.0:
        args.student_width = 1.25
    if args.freq_mask == 12:
        args.freq_mask = 8
    if args.time_mask == 48:
        args.time_mask = 32
    args.selection_metric = "threshold_icbhi_score"
    return args


def main():
    args = parse_args()
    base.set_seed(args.seed)
    device = base.default_device(args.device)
    output_dir, splits, stats = base.prepare_run(args)
    cn = base.get_class_names(args.num_classes)
    print(f"Pipeline: {args.pipeline_name}", flush=True)
    print(f"Task: {args.num_classes}-class ({', '.join(cn)})", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)
    for name, records in splits.items():
        labels = [base.get_label(r, args.num_classes) for r in records]
        print(f"  {name}: {{" + ", ".join(f"{cn[i]}={labels.count(i)}" for i in range(args.num_classes)) + "}}", flush=True)

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
