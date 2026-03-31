#!/usr/bin/env python3
"""
================================================================================
Vitis AI INT8 Quantization — Student MobileNetV2
================================================================================
Định lượng INT8 model Student (MobileNetV2) sử dụng pytorch_nndct
cho triển khai trên DPU DPUCZDX8G_ISA1_B2304 (Ultra96-V2).

Tính năng:
  - Cross-Layer Equalization (CLE) trước khi quantize
  - AdaQuant trong quá trình calibration
  - Calibration mode: chạy forward pass trên calib dataset
  - Test mode: đánh giá accuracy, F1, precision, recall so với FP32

Usage (trong Vitis AI Docker):
    # Bước 1: Calibration
    python quantize_distillation.py --quant_mode calib

    # Bước 2: Test (đánh giá quantized model)
    python quantize_distillation.py --quant_mode test

    # Deploy (xuất xmodel cho DPU)
    python quantize_distillation.py --quant_mode test --deploy
================================================================================
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================
NUM_CLASSES = 3
CLASS_NAMES = ['COPD', 'Healthy', 'Non-COPD']
IMG_SIZE = 224
BATCH_SIZE = 8  # smaller batch for calibration stability

# ImageNet normalization — same as training in distillation_02.py
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# ==============================================================================
# MODEL — Exact same as distillation_02.py
# ==============================================================================
class StudentModel(nn.Module):
    """MobileNetV2 as student with enhanced head.
    Must match distillation_02.py exactly for checkpoint loading.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout * 0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ==============================================================================
# CALIBRATION DATASET — Loads .npy spectrograms from generate_calib_data.py
# ==============================================================================
class CalibDataset(Dataset):
    """Dataset loading .npy spectrogram files + labels for calibration/test."""

    def __init__(self, calib_dir, normalize=True):
        self.calib_dir = calib_dir
        self.normalize_fn = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD) if normalize else None

        # Parse labels file
        labels_file = os.path.join(calib_dir, 'calib_labels.txt')
        self.items = []

        if not os.path.isfile(labels_file):
            # Fallback: scan for .npy files without labels
            npy_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.npy')])
            for npy_file in npy_files:
                self.items.append({'filename': npy_file, 'class_idx': -1, 'class_name': 'unknown'})
            logger.warning(f"No calib_labels.txt found. Loaded {len(self.items)} .npy files without labels.")
        else:
            with open(labels_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        self.items.append({
                            'filename': parts[0],
                            'class_idx': int(parts[1]),
                            'class_name': parts[2],
                        })
            logger.info(f"Loaded {len(self.items)} calibration samples from {labels_file}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        npy_path = os.path.join(self.calib_dir, item['filename'])
        spec = np.load(npy_path)  # (3, 224, 224) float32 [0, 1]
        tensor = torch.from_numpy(spec).float()

        if self.normalize_fn:
            tensor = self.normalize_fn(tensor)

        return tensor, item['class_idx']


# ==============================================================================
# UTILITIES
# ==============================================================================
def load_model(checkpoint_path, device='cpu'):
    """Load StudentModel from checkpoint."""
    model = StudentModel(num_classes=NUM_CLASSES, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        f1 = checkpoint.get('f1', 'N/A')
        logger.info(f"  Checkpoint F1 score: {f1}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded: {n_params:,} parameters")

    return model


def evaluate_model(model, dataloader, device='cpu', tag=''):
    """Evaluate model and return metrics. Works with or without sklearn."""
    print(f"\n>>> Starting {tag} evaluation...", flush=True)
    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            start = time.time()
            outputs = model(inputs)
            total_time += time.time() - start
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Eval batch {batch_idx + 1}/{len(dataloader)}", flush=True)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Filter out unlabeled samples (class_idx == -1)
    valid_mask = all_labels >= 0
    if valid_mask.sum() == 0:
        print("WARNING: No labeled samples for evaluation!", flush=True)
        return {}

    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    n_samples = len(all_preds)

    # --- Compute metrics (with sklearn fallback) ---
    try:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            classification_report, confusion_matrix
        )
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False
        print("  [INFO] sklearn not available, computing metrics manually", flush=True)

    if HAS_SKLEARN:
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        report = classification_report(
            all_labels, all_preds,
            target_names=CLASS_NAMES,
            zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)
    else:
        # Manual metrics computation
        accuracy = np.mean(all_preds == all_labels)
        # Per-class precision, recall, f1
        precisions, recalls, f1s = [], [], []
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for i in range(n_samples):
            cm[int(all_labels[i])][int(all_preds[i])] += 1
        for c in range(NUM_CLASSES):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        f1_macro = np.mean(f1s)
        f1_weighted = np.average(f1s, weights=[cm[c, :].sum() for c in range(NUM_CLASSES)])
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        # Build simple report
        report_lines = [f"{'':>15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}"]
        for c in range(NUM_CLASSES):
            support = int(cm[c, :].sum())
            report_lines.append(f"{CLASS_NAMES[c]:>15s} {precisions[c]:>10.4f} {recalls[c]:>10.4f} {f1s[c]:>10.4f} {support:>10d}")
        report = "\n".join(report_lines)

    # --- Print results using print() for guaranteed visibility ---
    print("\n" + "=" * 60, flush=True)
    print(f"  {tag} EVALUATION RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)", flush=True)
    print(f"  F1 (macro):        {f1_macro:.4f}", flush=True)
    print(f"  F1 (weighted):     {f1_weighted:.4f}", flush=True)
    print(f"  Precision (macro): {precision:.4f}", flush=True)
    print(f"  Recall (macro):    {recall:.4f}", flush=True)
    print(f"  Inference time:    {total_time:.3f}s ({total_time/n_samples*1000:.1f}ms/sample)", flush=True)
    print(f"  Total samples:     {n_samples}", flush=True)
    print(f"\n  Classification Report:", flush=True)
    print(report, flush=True)
    print(f"\n  Confusion Matrix:", flush=True)
    header = "  {:>10s}".format("") + "".join(f"{c:>10s}" for c in CLASS_NAMES)
    print(header, flush=True)
    for i, row in enumerate(cm):
        row_str = "  {:>10s}".format(CLASS_NAMES[i]) + "".join(f"{int(v):>10d}" for v in row)
        print(row_str, flush=True)
    print("=" * 60, flush=True)

    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision': float(precision),
        'recall': float(recall),
    }


# ==============================================================================
# QUANTIZATION
# ==============================================================================
def run_quantization(args):
    """Main quantization flow using pytorch_nndct."""

    # ------------------------------------------------------------------
    # 1. Import pytorch_nndct
    # ------------------------------------------------------------------
    try:
        from pytorch_nndct.apis import torch_quantizer
        logger.info("✅ pytorch_nndct imported successfully")
    except ImportError as e:
        logger.error(
            "❌ Cannot import pytorch_nndct!\n"
            "   Ensure you are running inside Vitis AI Docker.\n"
            f"   Error: {e}"
        )
        sys.exit(1)

    device = torch.device('cpu')  # Vitis AI quantizer works on CPU

    # ------------------------------------------------------------------
    # 2. Load FP32 model
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  VITIS AI QUANTIZATION — Student MobileNetV2")
    logger.info("=" * 60)
    logger.info(f"  Mode:       {args.quant_mode}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Calib dir:  {args.calib_dir}")
    logger.info(f"  Output:     {args.output_dir}")
    logger.info(f"  Target:     {args.target}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info("=" * 60)

    model = load_model(args.checkpoint, device)

    # ------------------------------------------------------------------
    # 3. Prepare dummy input and calibration data
    # ------------------------------------------------------------------
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    calib_dataset = CalibDataset(args.calib_dir, normalize=True)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # avoid multiprocessing issues in Docker
        pin_memory=False,
    )
    logger.info(f"  Calibration dataset: {len(calib_dataset)} samples")

    # ------------------------------------------------------------------
    # 4. Create Quantizer with CLE
    # ------------------------------------------------------------------
    # quant_mode: 'calib' for calibration, 'test' for evaluation
    extra_options = {}

    # Enable CLE (Cross-Layer Equalization)
    if args.enable_cle:
        extra_options['cle'] = True
        logger.info("  ✅ CLE (Cross-Layer Equalization) enabled")

    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(dummy_input,),
        output_dir=args.output_dir,
        device=device,
        target=args.target,
    )

    quant_model = quantizer.quant_model
    logger.info("  ✅ Quantizer created successfully")

    # ------------------------------------------------------------------
    # 5A. CALIBRATION MODE
    # ------------------------------------------------------------------
    if args.quant_mode == 'calib':
        logger.info("\n" + "=" * 60)
        logger.info("  RUNNING CALIBRATION")
        logger.info("=" * 60)

        if args.fast_finetune:
            logger.info("\n  🔥 Kích hoạt Advanced PTQ: Fast Finetuning (AdaQuant) 🔥")
            logger.info("  Quá trình này sẽ tốn chút thời gian để tự động tối ưu lại trọng số INT8...")
            try:
                # Hàm fast_finetune của Vitis AI nhận logic đánh giá (evaluate_model) & tham số (quant_model, loader, device)
                quantizer.fast_finetune(evaluate_model, (quant_model, calib_loader, device))
                logger.info("  ✅ Fast Finetuning hoàn tất thành công!")
            except Exception as e:
                logger.error(f"  ❌ Fast Finetune bị lỗi rớt: {e}")
                logger.info("  Chuyển về lại chế độ Standard Calibration (PTQ thông thường)...")


        quant_model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(calib_loader):
                inputs = inputs.to(device)
                _ = quant_model(inputs)
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Calibration batch {batch_idx + 1}/{len(calib_loader)}")

        # Export quantization results
        quantizer.export_quant_config()
        logger.info("  ✅ Calibration complete!")
        logger.info(f"  Quantization config saved to: {args.output_dir}")

        # Also evaluate FP32 for baseline comparison
        logger.info("\n  Evaluating FP32 baseline on calibration data...")
        fp32_metrics = evaluate_model(model, calib_loader, device, tag='FP32 Baseline')

    # ------------------------------------------------------------------
    # 5B. TEST MODE
    # ------------------------------------------------------------------
    elif args.quant_mode == 'test':
        print("\n" + "=" * 60, flush=True)
        print("  RUNNING QUANTIZED MODEL TEST", flush=True)
        print("=" * 60, flush=True)

        # Evaluate quantized model
        quant_metrics = {}
        fp32_metrics = {}
        try:
            quant_metrics = evaluate_model(quant_model, calib_loader, device, tag='INT8 Quantized')
        except Exception as e:
            print(f"\n  ERROR evaluating INT8 model: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Evaluate FP32 baseline for comparison
        try:
            print("\n  Evaluating FP32 baseline for comparison...", flush=True)
            fp32_metrics = evaluate_model(model, calib_loader, device, tag='FP32 Baseline')
        except Exception as e:
            print(f"\n  ERROR evaluating FP32 model: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Show comparison
        if quant_metrics and fp32_metrics:
            print("\n" + "=" * 60, flush=True)
            print("  ACCURACY COMPARISON: FP32 vs INT8", flush=True)
            print("=" * 60, flush=True)
            print(f"  {'Metric':<20s} {'FP32':>10s} {'INT8':>10s} {'Diff':>10s}", flush=True)
            print(f"  {'-' * 50}", flush=True)
            for key in ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']:
                fp32_val = fp32_metrics.get(key, 0)
                int8_val = quant_metrics.get(key, 0)
                diff = int8_val - fp32_val
                sign = '+' if diff >= 0 else ''
                print(f"  {key:<20s} {fp32_val:>10.4f} {int8_val:>10.4f} {sign}{diff:>9.4f}", flush=True)
            print("=" * 60, flush=True)

        # Save metrics to JSON file for later reference
        import json
        metrics_file = os.path.join(args.output_dir, 'quantization_metrics.json')
        metrics_data = {
            'fp32': fp32_metrics,
            'int8': quant_metrics,
            'comparison': {},
        }
        if quant_metrics and fp32_metrics:
            for key in ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']:
                metrics_data['comparison'][key] = {
                    'fp32': fp32_metrics.get(key, 0),
                    'int8': quant_metrics.get(key, 0),
                    'diff': quant_metrics.get(key, 0) - fp32_metrics.get(key, 0),
                }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\n  Metrics saved to: {metrics_file}", flush=True)

        # Export deployable model
        if args.deploy:
            print("\n  Exporting deployable xmodel...", flush=True)
            quantizer.export_xmodel(output_dir=args.output_dir)
            print(f"  ✅ Xmodel exported to: {args.output_dir}", flush=True)
        else:
            print("\n  To export xmodel, add --deploy flag", flush=True)

    print("\n  QUANTIZATION COMPLETE!", flush=True)
    print(f"  Results saved to: {args.output_dir}", flush=True)


# ==============================================================================
# MAIN
# ==============================================================================
def get_project_root():
    if os.path.isfile('/workspace/CMakeLists.txt') and not os.path.isdir('/workspace/Parallel_Computing_on_FPGA'):
        return '/workspace'
    if os.path.isdir('/workspace/Parallel_Computing_on_FPGA'):
        return '/workspace/Parallel_Computing_on_FPGA'
    return '/home/iec/Parallel_Computing_on_FPGA'


def parse_args():
    project_root = get_project_root()

    parser = argparse.ArgumentParser(
        description='Vitis AI INT8 Quantization for Student MobileNetV2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng (trong Vitis AI Docker):
  # Bước 1: Calibration
  python quantize_distillation.py --quant_mode calib

  # Bước 2: Test + Deploy
  python quantize_distillation.py --quant_mode test --deploy
        """
    )
    parser.add_argument(
        '--quant_mode', type=str, required=True,
        choices=['calib', 'test'],
        help='Quantization mode: calib (calibration) or test (evaluation)'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=os.path.join(project_root, 'python/output_distillation_v2/checkpoints/student_fold_0_best.pt'),
        help='Path to FP32 model checkpoint'
    )
    parser.add_argument(
        '--calib_dir', type=str,
        default=os.path.join(project_root, 'data/calib_data'),
        help='Path to calibration dataset directory (generated by generate_calib_data.py)'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(project_root, 'quantize_distillation_result'),
        help='Output directory for quantization results'
    )
    parser.add_argument(
        '--target', type=str,
        default='DPUCZDX8G_ISA1_B2304',
        help='DPU target (default: DPUCZDX8G_ISA1_B2304 for Ultra96-V2)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=BATCH_SIZE,
        help=f'Batch size for calibration/test (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--enable_cle', action='store_true', default=True,
        help='Enable Cross-Layer Equalization (default: True)'
    )
    parser.add_argument(
        '--no_cle', action='store_false', dest='enable_cle',
        help='Disable Cross-Layer Equalization'
    )
    parser.add_argument(
        '--fast_finetune', action='store_true', default=False,
        help='Enable Fast Finetuning (Advanced PTQ AdaQuant) during calibration'
    )
    parser.add_argument(
        '--deploy', action='store_true', default=False,
        help='Export xmodel for DPU deployment (only in test mode)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify calib data exists
    if not os.path.isdir(args.calib_dir):
        logger.error(
            f"❌ Calibration data not found: {args.calib_dir}\n"
            f"   Run generate_calib_data.py first:\n"
            f"     python python/quantization/generate_calib_data.py"
        )
        sys.exit(1)

    # Verify checkpoint exists
    if not os.path.isfile(args.checkpoint):
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    run_quantization(args)


if __name__ == '__main__':
    main()
