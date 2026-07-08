#!/usr/bin/env python3
"""Evaluate S2 student with Test-Time Augmentation."""

import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from python.training import icbhi_kd_pipeline_multiview_ensemble as base
from python.training.icbhi_kd_s1_tta_calibrated import evaluate_student_tta

def main():
    args = base.parse_args()
    args.pipeline_name = "icbhi_kd_s2_featattn_4class"
    args.output_dir = str(base.TRAINING_ARTIFACTS_DIR / args.pipeline_name)
    
    base.set_seed(args.seed)
    device = base.default_device(args.device)
    output_dir, splits, stats = base.prepare_run(args)
    
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    
    # Load student
    student = base.make_model(args.student_arch, args.num_classes, in_ch, args).to(device)
    ckpt_path = output_dir / "students" / args.student_arch / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["model_state"])
    
    print(f"Loaded student from {ckpt_path}")
    print(f"Best val threshold: {ckpt.get('threshold', 'N/A')}")
    
    # Evaluate with TTA
    for split in ["val", "test"]:
        if not splits[split]:
            continue
        loader = base.make_loader(base.ICBHIDataset(splits[split], args, stats, False), args)
        
        print(f"\n=== {split} TTA Evaluation ===")
        tta_m = evaluate_student_tta(student, loader, device, args, n_tta=7)
        
        print(f"ICBHI Score: {tta_m['icbhi_score']:.4f}")
        print(f"Sensitivity: {tta_m['sensitivity']:.4f}")
        print(f"Specificity: {tta_m['specificity']:.4f}")
        print(f"Accuracy: {tta_m['accuracy']:.4f}")
        print(f"Macro F1: {tta_m['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
