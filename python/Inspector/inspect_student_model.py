#!/usr/bin/env python3
"""
================================================================================
Vitis AI Model Inspector — Student MobileNetV2
================================================================================
Kiểm tra tính tương thích của model Student (MobileNetV2) với DPU DPUCZDX8G
trên Ultra96-V2 sử dụng Vitis AI 3.5 pytorch_nndct Inspector.

Mục tiêu:
  - Xác định operators nào được hỗ trợ bởi DPU
  - Xác định operators nào phải chạy trên CPU (fallback)
  - Đánh giá tỷ lệ tính toán DPU vs CPU

Usage:
    # Trong Vitis AI 3.5 Docker environment:
    python inspect_student_model.py

    # Hoặc chỉ định đường dẫn checkpoint:
    python inspect_student_model.py --checkpoint /path/to/student_fold_0_best.pt

    # Sử dụng arch.json thay vì target name:
    python inspect_student_model.py --arch /path/to/arch.json

Output:
    - Console log: danh sách operators hỗ trợ/không hỗ trợ
    - File: ./inspect_results/ (inspection report được tạo bởi Inspector)
================================================================================
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import ARTIFACTS_DIR

import torch
import torch.nn as nn
from torchvision import models

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# MODEL DEFINITION (phải khớp chính xác với distillation_02.py)
# ==============================================================================
NUM_CLASSES = 3  # COPD, Healthy, Non-COPD

class StudentModel(nn.Module):
    """MobileNetV2 as student with enhanced head.
    
    Kiến trúc giống hệt với file distillation_02.py
    để đảm bảo load trọng số chính xác.
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
# INSPECTOR
# ==============================================================================
def run_inspector(checkpoint_path: str, target: str, output_dir: str):
    """
    Chạy Vitis AI Inspector để kiểm tra model.
    
    Args:
        checkpoint_path: Đường dẫn đến file .pt checkpoint
        target: DPU target name (ví dụ: 'DPUCZDX8G_ISA1_B2304')
                hoặc đường dẫn đến arch.json
        output_dir: Thư mục lưu kết quả inspection
    """
    # ------------------------------------------------------------------
    # 1. Import Inspector từ Vitis AI
    # ------------------------------------------------------------------
    try:
        from pytorch_nndct.apis import Inspector
        logger.info("✅ pytorch_nndct.apis.Inspector imported thành công")
    except ImportError as e:
        logger.error(
            "❌ Không thể import pytorch_nndct!\n"
            "   Đảm bảo bạn đang chạy trong Vitis AI 3.5 Docker environment.\n"
            "   Lệnh khởi động Docker từ repo root:\n"
            "     docker run --gpus all -it --rm \\\n"
            "       -v \"$PWD\":/workspace \\\n"
            "       xilinx/vitis-ai-pytorch-gpu:3.5.0.001\n"
            f"   Lỗi chi tiết: {e}"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load model + trọng số
    # ------------------------------------------------------------------
    logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
    
    model = StudentModel(num_classes=NUM_CLASSES, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Checkpoint format: {'model_state': ..., 'f1': ...}
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        logger.info(f"   F1 score khi train: {checkpoint.get('f1', 'N/A')}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Trường hợp checkpoint chỉ chứa state_dict trực tiếp
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"✅ Model loaded thành công ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # ------------------------------------------------------------------
    # 3. Tạo dummy input (1, 3, 224, 224)
    # ------------------------------------------------------------------
    dummy_input = torch.randn(1, 3, 224, 224)
    logger.info(f"📐 Dummy input shape: {dummy_input.shape}")

    # ------------------------------------------------------------------
    # 4. Tạo Inspector
    # ------------------------------------------------------------------
    # Nếu target là file arch.json, đọc target name từ bên trong
    if target.endswith('.json') and os.path.isfile(target):
        with open(target, 'r') as f:
            arch_data = json.load(f)
        target_name = arch_data.get('target', target)
        logger.info(f"📋 Đọc từ arch.json: {target} → target = {target_name}")
        inspector = Inspector(target_name)
    else:
        logger.info(f"🎯 Target DPU: {target}")
        inspector = Inspector(target)

    # ------------------------------------------------------------------
    # 5. Chạy inspection
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("  BẮT ĐẦU INSPECTION")
    logger.info("=" * 70)
    logger.info(f"  Model:  StudentModel (MobileNetV2 + custom head)")
    logger.info(f"  Target: {target}")
    logger.info(f"  Input:  (1, 3, 224, 224)")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 70)
    
    # Inspector.inspect() sẽ:
    # - Phân tích computation graph của model
    # - Xác định từng operator thuộc DPU hay CPU
    # - Tạo report trong output_dir
    inspector.inspect(model, (dummy_input,), device=torch.device('cpu'),
                      output_dir=output_dir)
    
    logger.info("=" * 70)
    logger.info("  INSPECTION HOÀN THÀNH")
    logger.info("=" * 70)
    logger.info(f"📁 Kết quả đã lưu tại: {output_dir}")
    
    # ------------------------------------------------------------------
    # 6. Hướng dẫn đọc kết quả
    # ------------------------------------------------------------------
    print_result_guide()


def print_result_guide():
    """In hướng dẫn đọc kết quả Inspector."""
    guide = """
╔══════════════════════════════════════════════════════════════════════╗
║                  HƯỚNG DẪN ĐỌC KẾT QUẢ INSPECTOR                  ║
╚══════════════════════════════════════════════════════════════════════╝

1. FILE KẾT QUẢ:
   - inspect_results/ chứa báo cáo chi tiết
   - Xem file inspect_results/*.txt hoặc *.html (nếu có)

2. CÁCH ĐỌC BÁO CÁO:

   ┌─────────────────────────────────────────────────────────────────┐
   │ ✅ DPU Supported (Màu XANH)                                    │
   │    - Operators chạy được trên DPU hardware                      │
   │    - Ví dụ: Conv2d, ReLU, BatchNorm2d, AvgPool2d               │
   │    - Được tăng tốc hardware (rất nhanh!)                        │
   ├─────────────────────────────────────────────────────────────────┤
   │ ❌ CPU Fallback (Màu ĐỎ)                                       │
   │    - Operators KHÔNG được DPU hỗ trợ                            │
   │    - Phải chạy trên ARM CPU (chậm)                              │
   │    - Ví dụ thường gặp:                                          │
   │      • Dropout       → OK, tự loại bỏ khi eval                  │
   │      • BatchNorm1d   → ⚠️ Không merge được, cần thay thế        │
   │      • Softmax       → Chạy CPU (ít ảnh hưởng, ở cuối mạng)    │
   │      • Linear (fully connected) → Một số DPU hỗ trợ, một số ko │
   ├─────────────────────────────────────────────────────────────────┤
   │ ⚠️ Quantization Warning (Màu VÀNG)                              │
   │    - Operators cần kiểm tra thêm khi quantize INT8              │
   │    - Có thể mất accuracy nếu quantize không đúng cách           │
   └─────────────────────────────────────────────────────────────────┘

3. CÁC VẤN ĐỀ THƯỜNG GẶP VỚI MOBILENETV2:

   ╔═══════════════════════╦════════════╦═══════════════════════════╗
   ║ Operator              ║ DPU B2304  ║ Giải pháp                 ║
   ╠═══════════════════════╬════════════╬═══════════════════════════╣
   ║ Conv2d                ║ ✅ Hỗ trợ  ║ -                         ║
   ║ DepthwiseSeparable    ║ ✅ Hỗ trợ  ║ -                         ║
   ║ BatchNorm2d           ║ ✅ Fused    ║ Merge vào Conv2d          ║
   ║ ReLU6                 ║ ✅ Hỗ trợ  ║ -                         ║
   ║ Linear (classifier)   ║ ✅/⚠️      ║ Có thể cần reshape       ║
   ║ Dropout               ║ ❌ Skip    ║ Tự bỏ khi model.eval()    ║
   ║ BatchNorm1d           ║ ❌ CPU     ║ Thay bằng LayerNorm hoặc  ║
   ║                       ║            ║ bỏ BN trong classifier    ║
   ║ Softmax               ║ ❌ CPU     ║ Bỏ, dùng argmax sau DPU   ║
   ║ AdaptiveAvgPool2d     ║ ✅ Hỗ trợ  ║ -                         ║
   ╚═══════════════════════╩════════════╩═══════════════════════════╝

4. CHỈ SỐ QUAN TRỌNG:
   - "DPU subgraph ratio": Tỷ lệ ops chạy trên DPU (mục tiêu > 90%)
   - "Total ops": Tổng số operations
   - "DPU ops": Số ops chạy trên DPU
   - "CPU ops": Số ops fallback về CPU

5. NẾU TỶ LỆ DPU THẤP (< 80%):
   → Cần sửa lại kiến trúc model (xem phần khuyến nghị bên dưới)
   → Thay thế BatchNorm1d trong custom classifier head
   → Giảm số linear layers nếu DPU không hỗ trợ
"""
    print(guide)


# ==============================================================================
# MAIN
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Vitis AI Model Inspector cho Student MobileNetV2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Trong Vitis AI Docker:
  python inspect_student_model.py

  # Chỉ định checkpoint khác:
  python inspect_student_model.py --checkpoint /path/to/model.pt

  # Sử dụng arch.json:
  python inspect_student_model.py --arch /path/to/arch.json
        """
    )
    parser.add_argument(
        '--artifact_root', type=str,
        default=str(ARTIFACTS_DIR),
        help='Root artifacts directory for default checkpoint/output paths'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=None,
        help='Đường dẫn đến file checkpoint .pt (default: student_fold_0_best.pt)'
    )
    parser.add_argument(
        '--target', type=str, default='DPUCZDX8G_ISA1_B2304',
        help='DPU target name (default: DPUCZDX8G_ISA1_B2304 cho Ultra96-V2)'
    )
    parser.add_argument(
        '--arch', type=str, default=None,
        help='Đường dẫn đến arch.json nếu muốn dùng target từ file'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Thư mục lưu kết quả inspection'
    )
    args = parser.parse_args()
    artifact_root = Path(args.artifact_root)
    if args.checkpoint is None:
        args.checkpoint = str(
            artifact_root / 'training' / 'distillation_v2' / 'checkpoints' / 'student_fold_0_best.pt'
        )
    if args.output_dir is None:
        args.output_dir = str(artifact_root / 'quantization' / 'inspect_student_model')
    return args


def main():
    args = parse_args()
    
    # Xác định target
    target = args.arch if args.arch else args.target
    
    # Kiểm tra checkpoint tồn tại
    if not os.path.isfile(args.checkpoint):
        logger.error(f"❌ Không tìm thấy checkpoint: {args.checkpoint}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("  VITIS AI MODEL INSPECTOR — Student MobileNetV2")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint:  {args.checkpoint}")
    logger.info(f"  Target:      {target}")
    logger.info(f"  Output:      {args.output_dir}")
    logger.info(f"  Num classes: {NUM_CLASSES}")
    logger.info(f"  Input size:  (1, 3, 224, 224)")
    logger.info("=" * 70)
    
    run_inspector(args.checkpoint, target, args.output_dir)


if __name__ == '__main__':
    main()
