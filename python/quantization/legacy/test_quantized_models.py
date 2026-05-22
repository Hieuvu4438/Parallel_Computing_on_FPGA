#!/usr/bin/env python3
import os
import sys
import torch
import json

# Thêm đường dẫn để có thể import từ quantize_distillation_04
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from quantize_distillation_04 import (
    StudentModel, NUM_CLASSES, build_dataloaders, 
    run_comparison, CLASS_NAMES
)

try:
    from pytorch_quantization import quant_modules
except ImportError:
    print("ERROR: NVIDIA pytorch-quantization is required.")
    sys.exit(1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Đường dẫn file
    fp32_ckpt_path = '/home/iec/Parallel_Computing_on_FPGA/python/output_distillation_v2/checkpoints/student_fold_0_best.pt'
    qat_ckpt_path = '/home/iec/Parallel_Computing_on_FPGA/quantize_nvidia_qat_result/student_qat_best.pt'
    output_dir = '/home/iec/Parallel_Computing_on_FPGA/quantize_nvidia_qat_result'
    
    if not os.path.exists(fp32_ckpt_path):
        print(f"Error: FP32 model not found at {fp32_ckpt_path}")
        return
    if not os.path.exists(qat_ckpt_path):
        print(f"Error: QAT model not found at {qat_ckpt_path}")
        return
        
    print("\n[1/4] Loading FP32 Model...")
    fp32_model = StudentModel(num_classes=NUM_CLASSES, pretrained=False)
    fp32_ckpt = torch.load(fp32_ckpt_path, map_location='cpu', weights_only=False)
    fp32_state = fp32_ckpt.get('model_state', fp32_ckpt.get('state_dict', fp32_ckpt))
    fp32_model.load_state_dict(fp32_state)
    fp32_model.to(device)
    fp32_model.eval()
    print("  ✅ FP32 Model loaded.")
    
    print("\n[2/4] Initializing QDQ Nodes for INT8 Model...")
    # Bắt buộc gọi hàm này trước khi khởi tạo model QAT
    quant_modules.initialize()
    print("  ✅ QDQ nodes initialized.")
    
    print("\n[3/4] Creating QAT Model and Loading Quantized Weights...")
    quant_model = StudentModel(num_classes=NUM_CLASSES, pretrained=False)
    qat_ckpt = torch.load(qat_ckpt_path, map_location='cpu', weights_only=False)
    qat_state = qat_ckpt.get('model_state', qat_ckpt.get('state_dict', qat_ckpt))
    
    # strict=False vì các node lượng tử hóa (QuantConv2d, v.v) có chứa thêm tham số amax
    missing, expected = quant_model.load_state_dict(qat_state, strict=False)
    quant_model.to(device)
    quant_model.eval()
    
    # Hủy hiệu ứng qdq auto-insertion cho những khai báo model sau này (nếu có)
    quant_modules.deactivate()
    print("  ✅ QAT Model loaded successfully.")
    
    print("\n[4/4] Building Dataloader for Evaluation...")
    # Khởi tạo mock object giả lập tham số truyền vào từ CLI trong quantize_distillation_04.py
    class DummyArgs:
        def __init__(self):
            root = '/home/iec/Parallel_Computing_on_FPGA'
            self.icbhi_dir = os.path.join(root, 'data/samples/ICBHI_final_database')
            self.icbhi_labels = os.path.join(root, 'data/samples/labels.txt')
            self.combined_dir = os.path.join(root, 'data/combined/audio')
            self.num_calib = 200 # Dùng 200 mẫu để validate
            
    args = DummyArgs()
    # build_dataloaders trả về calib_loader, train_loader. 
    # Ta dùng calib_loader như một validation set để test performance.
    calib_loader, _ = build_dataloaders(args, batch_size=8)
    
    print(f"\n======================================================================")
    print(f"  Bắt đầu chạy so sánh FP32 vs QAT Fake-INT8... ")
    print(f"======================================================================")
    
    # Sử dụng lại hàm run_comparison có sẵn từ quantize_distillation_04.py 
    # (Hàm này đo lường: Accuracy, F1, Model Size, Latency, SQNR và lưu json)
    results = run_comparison(fp32_model, quant_model, calib_loader, device, output_dir)
    
    print(f"\n✅ Hoàn thành test tính năng. Báo cáo chi tiết đã được lưu trong {output_dir}/qat_comparison_results.json")

if __name__ == "__main__":
    main()
