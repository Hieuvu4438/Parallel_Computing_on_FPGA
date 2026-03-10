import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob

# Try to import pytorch_nndct for Vitis AI Quantization
try:
    from pytorch_nndct.apis import torch_quantizer
except ImportError:
    print("Warning: Không tìm thấy thư viện pytorch_nndct. Bạn cần chạy script này bên trong môi trường Docker của Vitis AI 3.5 (PyTorch conda env).")

# ==============================================================================
# HÀM ĐỊNH NGHĨA MODEL TƯƠNG TỰ LÚC TRAIN (BINARY)
# ==============================================================================
class COPDClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ==============================================================================
# HÀM LOAD DỮ LIỆU HIỆU CHUẨN (CALIBRATION)
# ==============================================================================
class ImageDataset(torch.utils.data.Dataset):
    """Đọc ảnh Spectrogram (.png) đã tạo ở bước trước"""
    def __init__(self, data_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        self.transform = transform
        print(f"-> Tìm thấy {len(self.image_paths)} ảnh dùng để Calibration/Evaluation.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        
        # Nhãn giả định vì quá trình Quantization không dùng nhãn để tính loss (chỉ bắt range của weight/activation)
        label = 0 
        
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(data_dir, batch_size=32):
    """
    Chuẩn hóa ảnh y hệt quá trình train (Tỉ lệ và ImageNet Mean/Std).
    Lưu ý: Bạn đã có folder calib_images_02 chứa sẵn ảnh PNG kích thước 224x224
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Đối với ảnh PNG đọc trực tiếp, chỉ cần ToTensor và Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    dataset = ImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# ==============================================================================
# EVALUATION DUMMY FUNCTION
# ==============================================================================
def evaluate(model, dataloader):
    """
    Forward pass toàn bộ data qua Model để VitisAI module 
    ghi nhận khoảng giá trị (min/max) của các tham số.
    """
    model.eval()
    print("[+] Đang đẩy dữ liệu qua Data Loader (Forward Pass)...")
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            _ = model(images)
            if (i+1) % 5 == 0:
                print(f"  -> Processed batch {i+1}/{len(dataloader)}")

# ==============================================================================
# CHỨC NĂNG CHÍNH: QUẢN LÝ QUÁ TRÌNH QUANTIZATION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Vitis AI 3.5 PyTorch Quantize Script (CPU Version)")
    parser.add_argument('--quant_mode', type=str, default='calib', 
                        choices=['calib', 'test'], 
                        help="Chế độ định lượng. 'calib' để tính parameters; 'test' để evaluate và xuất file cấu hình.")
    parser.add_argument('--model_dir', type=str, default="/home/iec/Parallel_Computing_on_FPGA/python/output_copd_v2", 
                        help="Đường dẫn chứa file best_model_fold_0.pth")
    parser.add_argument('--calib_dir', type=str, default="/home/iec/Parallel_Computing_on_FPGA/data/calib_images_02", 
                        help="Đường dẫn chứa khoảng 200 ảnh PNG cho quá trình calibration")
    parser.add_argument('--output_dir', type=str, default="/home/iec/Parallel_Computing_on_FPGA/vitis_ai_flow/quantize_result", 
                        help="Đường dẫn thư mục lưu file kết quả định lượng")
    args = parser.parse_args()

    # 1. Khởi tạo Model Float và load trọng số
    print("="*70)
    print(f" VITIS AI 3.5 PYTORCH QUANTIZATION (Mode: {args.quant_mode.upper()}) ")
    print("="*70)
    
    device = torch.device("cpu") # Chỉ dùng CPU trong môi trường Docker Vitis AI chưa passthrough GPU
    model = COPDClassifier(num_classes=2).to(device)
    
    model_weight_path = os.path.join(args.model_dir, "best_model_fold_0.pth")
    if os.path.exists(model_weight_path):
        print(f"[+] Load trọng số mô hình từ: {model_weight_path}")
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
    else:
        print(f"[ERROR] Không tìm thấy file trọng số tại {model_weight_path}!")
        return

    # 2. Xây dựng DataLoader
    print(f"\n[+] Khởi tạo Dataloader từ: {args.calib_dir}")
    dataloader_calib = get_dataloader(args.calib_dir, batch_size=1)
    if len(dataloader_calib.dataset) == 0:
         print(f"[ERROR] Thư mục chứa Calibration Image trống! Cần ít nhất 100-200 ảnh để định lượng.")
         return

    # 3. Dummy Input để nạp đồ thị (Parse Model Graph)
    dummy_input = torch.randn([1, 3, 224, 224], dtype=torch.float32).to(device)

    # 4. Khởi tạo Vitis AI Quantizer
    print("\n[+] Bắt đầu instance pytorch_nndct (torch_quantizer)...")
    quantizer = torch_quantizer(
        args.quant_mode,
        model,
        (dummy_input),
        output_dir=args.output_dir
    )
    
    quant_model = quantizer.quant_model

    # 5. Phân rã Mode: CALIBRATION / TEST
    if args.quant_mode == 'calib':
        print("\n--- BƯỚC CALIBRATION ---")
        evaluate(quant_model, dataloader_calib)
        
        # Export các file lưu thông số min/max activation sau khi chạy forward pass
        quantizer.export_quant_config()
        print(f"\n[HOÀN THÀNH CALIBRATION]")
        print(f"Hệ thống đã chuẩn hóa số liệu Float32 thành int8 phân vùng xong (Fake quant).")
        print(f"File log và cấu trúc được đặt trong: {args.output_dir}")
        print(f"--> Tiếp tục chạy script với cờ `--quant_mode test` để export mô hình cuối cùng.")
        
    elif args.quant_mode == 'test':
        print("\n--- BƯỚC EVALUATION & DUMP ---")
        # Tuỳ môi trường, ta cần Evaluate để sinh dữ liệu chốt chặn trên node XIR
        evaluate(quant_model, dataloader_calib)
        
        print("\n[+] Đang tạo file Float graph và thư mục Quantized Graph (ONNX)...")
        # Gọi xuất file cấu trúc dành cho compiler (vai_c_xir) -> Folder "quantize_result"
        quantizer.export_xmodel(deploy_check=False) # Hoặc export_quant_config() nếu code VitisAI dưới 2.5
        print(f"\n[HOÀN THÀNH XUẤT XMODEL / DPU MODEL]")
        print(f"Toàn bộ code cấu trúc Quantized đã xả ra tại folder: {args.output_dir}")
        print("Sẵn sàng cho công đoạn Biên dịch (Compilation - vai_c_xir) thành file .xmodel vật lý nạp qua FPGA!")


if __name__ == '__main__':
    main()
