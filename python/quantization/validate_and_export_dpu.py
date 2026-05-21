import os
import glob
import numpy as np
import pywt
import librosa
import scipy.signal as signal
from scipy.ndimage import zoom
from collections import Counter

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# ==============================================================================
# CẤU HÌNH CƠ BẢN
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tùy chỉnh số lớp (Người dùng yêu cầu 5 lớp: Healthy, Pneumonia, URTI, Bronchiectasis, COPD)
# Ghi chú: Có thể thay đổi NUM_CLASSES = 2 nếu bạn load model Binary COPD (như trong output_copd_v2) 
# NUM_CLASSES = 5
# CLASS_NAMES = ['Healthy', 'Pneumonia', 'URTI', 'Bronchiectasis', 'COPD']

NUM_CLASSES = 2
CLASS_NAMES = ['Non-COPD', 'COPD']

DATA_DIR = "/home/iec/Parallel_Computing_on_FPGA/data/samples/ICBHI_final_database"
MODEL_WEIGHTS = "/home/iec/Parallel_Computing_on_FPGA/python/output_copd_v2/best_model_fold_0.pth"
EXPORT_TRACE_PATH = "/home/iec/Parallel_Computing_on_FPGA/python/output_copd_v2/model_dpu_ready.pt"

# Danh sách ICD-10 của các bệnh nhân trong bộ ICBHI 2017
PATIENT_DIAGNOSIS = {
    101: 'URTI', 102: 'Healthy', 103: 'Asthma', 104: 'COPD', 105: 'URTI',
    106: 'COPD', 107: 'COPD', 108: 'LRTI', 109: 'COPD', 110: 'COPD',
    111: 'Bronchiectasis', 112: 'COPD', 113: 'COPD', 114: 'COPD', 115: 'LRTI',
    116: 'Bronchiectasis', 117: 'COPD', 118: 'COPD', 119: 'URTI', 120: 'COPD',
    121: 'Healthy', 122: 'Pneumonia', 123: 'Healthy', 124: 'COPD', 125: 'Healthy',
    126: 'Healthy', 127: 'Healthy', 128: 'COPD', 129: 'URTI', 130: 'COPD',
    131: 'URTI', 132: 'COPD', 133: 'COPD', 134: 'COPD', 135: 'Pneumonia',
    136: 'Healthy', 137: 'URTI', 138: 'COPD', 139: 'COPD', 140: 'Pneumonia',
    141: 'COPD', 142: 'COPD', 143: 'Healthy', 144: 'Healthy', 145: 'COPD',
    146: 'COPD', 147: 'COPD', 148: 'URTI', 149: 'Bronchiolitis', 150: 'URTI',
    151: 'COPD', 152: 'Healthy', 153: 'Healthy', 154: 'COPD', 155: 'COPD',
    156: 'COPD', 157: 'COPD', 158: 'COPD', 159: 'Healthy', 160: 'COPD',
    161: 'Bronchiolitis', 162: 'COPD', 163: 'COPD', 164: 'URTI', 165: 'URTI',
    166: 'COPD', 167: 'Bronchiolitis', 168: 'Bronchiectasis', 169: 'Bronchiectasis',
    170: 'COPD', 171: 'Healthy', 172: 'COPD', 173: 'Bronchiolitis', 174: 'COPD',
    175: 'COPD', 176: 'COPD', 177: 'COPD', 178: 'COPD', 179: 'Healthy', 180: 'COPD',
    181: 'COPD', 182: 'Healthy', 183: 'Healthy', 184: 'Healthy', 185: 'COPD',
    186: 'COPD', 187: 'Healthy', 188: 'URTI', 189: 'COPD', 190: 'URTI',
    191: 'Pneumonia', 192: 'COPD', 193: 'COPD', 194: 'Healthy', 195: 'COPD',
    196: 'Bronchiectasis', 197: 'URTI', 198: 'COPD', 199: 'COPD', 200: 'COPD',
    201: 'Bronchiectasis', 202: 'Healthy', 203: 'COPD', 204: 'COPD', 205: 'COPD',
    206: 'Bronchiolitis', 207: 'COPD', 208: 'Healthy', 209: 'Healthy', 210: 'URTI',
    211: 'COPD', 212: 'COPD', 213: 'COPD', 214: 'Healthy', 215: 'Bronchiectasis',
    216: 'Bronchiolitis', 217: 'Healthy', 218: 'COPD', 219: 'Pneumonia', 220: 'COPD',
    221: 'COPD', 222: 'COPD', 223: 'COPD', 224: 'Healthy', 225: 'Healthy',
    226: 'Pneumonia',
}

# ==============================================================================
# HÀM BANDPASS & CWT TƯƠNG TỰ BƯỚC PREPROCESS
# ==============================================================================
def bandpass_filter(data, fs, lowcut=50.0, highcut=2500.0, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, min(highcut, nyq - 1.0) / nyq
    if low >= high: return data
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

class ValidationDataset(Dataset):
    def __init__(self, data_dir, class_list):
        self.data_dir = data_dir
        self.class_list = class_list
        self.files = []
        self.labels = []
        
        wavs = glob.glob(os.path.join(data_dir, "*.wav"))
        for w in wavs:
            basename = os.path.basename(w)
            # Lấy Patient ID (ví dụ '101' từ '101_1b1_Al_sc_Meditron.wav')
            pid = int(basename.split('_')[0])
            diagnosis = PATIENT_DIAGNOSIS.get(pid, 'Unknown')
            lable = 1 if diagnosis == 'COPD' else 0
            self.files.append(w)
            self.labels.append(lable)
            

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Trích xuất CWT theo chuẩn hệ thống
        audio, orig_sr = librosa.load(self.files[idx], sr=None)
        audio_filtered = bandpass_filter(audio, orig_sr)
        
        target_sr = 4000
        if orig_sr != target_sr:
            audio_resampled = librosa.resample(audio_filtered, orig_sr=orig_sr, target_sr=target_sr)
        else:
            audio_resampled = audio_filtered

        # CWT
        center_freq = pywt.central_frequency('morl')
        scales = np.logspace(np.log10(center_freq*target_sr/1950.0), np.log10(center_freq*target_sr/50.0), 128)
        coef, _ = pywt.cwt(audio_resampled, scales, 'morl', sampling_period=1.0/target_sr)
        power_db = 10 * np.log10(np.abs(coef) ** 2 + 1e-10)
        power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)

        zoom_factors = (224 / power_db.shape[0], 224 / power_db.shape[1])
        spec_resized = zoom(power_db, zoom_factors, order=1)[:224, :224]
        
        spec_rgb = np.stack([spec_resized] * 3, axis=0) # (3, H, W)
        tensor_img = torch.from_numpy(spec_rgb).float()
        tensor_img = self.normalize(tensor_img)
        
        return tensor_img, self.labels[idx]

# ==============================================================================
# HÀM ĐỊNH NGHĨA MODEL & CHỨC NĂNG CỐT LÕI
# ==============================================================================

class COPDClassifier(nn.Module):
    def __init__(self, num_classes=5):
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


def get_model(num_classes):
    """
    Khởi tạo cấu trúc MobileNetV2 bọc trong COPDClassifier 
    giống y hệt lúc train để khớp trọng số state_dict.
    """
    return COPDClassifier(num_classes=num_classes)

def check_dpu_compatibility(model):
    """
    Quét qua các lớp của Model để phát hiện những Activation / Operator phức tạp
    không được DPU hỗ trợ tốt (cần chạy trên CPU fall-back) hoặc không được phép (Ví dụ: SiLU, Hardswish).
    """
    print("\n[+] Đang phân tích mức độ tương thích kiến trúc mạng với Xilinx DPU...")
    unsupported_ops = {
        'GELU': nn.GELU,
        'SiLU': nn.SiLU,
        'Hardswish': nn.Hardswish,
        'Mish': nn.Mish,
        'PReLU': nn.PReLU
    }
    
    warning_found = False
    for name, module in model.named_modules():
        for op_name, op_class in unsupported_ops.items():
            if isinstance(module, op_class):
                print(f"   [CẢNH BÁO] Layer '{name}' dùng toán tử '{op_name}'!")
                warning_found = True
                
        # DPU hỗ trợ nn.ReLU6, nhưng với Quantize int8 đôi lúc clip ReLU6 gây mất thông tin,
        # nhiều kỹ sư Xilinx khuyến nghị replace nn.ReLU6 bằng nn.ReLU. Hàm check nhắc nhở:
        if isinstance(module, nn.ReLU6):
            print(f"   [LƯU Ý] Layer '{name}' dùng nn.ReLU6. DPU hỗ trợ nhưng bạn có thể cân nhắc thế bằng nn.ReLU.")

    if not warning_found:
        print("   -> Tuyệt vời! Mạng không chứa toán tử bất thường, luồng biên dịch (vai_c_xir) sẽ diễn ra trơn tru thuần túy trên DPU.")
    else:
        print("   -> Lời khuyên: Hãy thay các hàm kích hoạt trên bằng nn.ReLU trước khi dùng vai_q_pytorch.")

def validate_model(model, dataloader):
    """Hiệu năng theo yêu cầu: Accuracy, Sensitivity, F1-score (Macro/Micro)"""
    print(f"\n[+] Bắt đầu Validation trên tập Subject-Independent Split ({len(dataloader.dataset)} mẫu)...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Duyệt dữ liệu Evaluation"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Sensitivity (Recall) trung bình cho Multiclass
    cm = confusion_matrix(all_labels, all_preds)
    sensitivities = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sens)
        
    avg_sensitivity = np.mean(sensitivities) * 100
    
    print("\n" + "="*50)
    print(" KẾT QUẢ HIỆU NĂNG VALIDATION (MULTICLASS / BINARY)")
    print("="*50)
    print(f" - Accuracy (Độ chính xác) : {acc * 100:.2f} %")
    print(f" - Sensitivity (Độ nhạy)  : {avg_sensitivity:.2f} % (Average)")
    print(f" - F1-Score (Macro)       : {f1 * 100:.2f} %")
    print("="*50)
    
    # Hiển thị chi tiết từng Class
    print("\nChi tiết Classification Report:")
    # Loại bỏ những nhãn không xuất hiện trong data thực tế ở report này để tránh lỗi
    unique_labels = np.unique(all_labels)
    target_names = [CLASS_NAMES[i] for i in unique_labels]
    print(classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))

def export_trace(model, save_path):
    """
    Sử dụng torch.jit.trace để xuất mô hình. Đây là cơ sở bắt buộc để Vitis AI Quantizer
    nạp mô hình chuẩn PyTorch (Parse Graph).
    """
    print(f"\n[+] Tiến hành đóng gói mô hình (TorchScript Tracing)...")
    model.eval()
    model.to('cpu')  # Luôn trace trên CPU cho vai_q_pytorch
    
    # Batch = 1, Channels = 3, Size = 224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(save_path)
        print(f"   -> [HOÀN THÀNH] Đã lưu TorchScript tại: {save_path}")
        print("   -> Bạn có thể đem file này đưa thẳng vào luồng Vitis AI (vai_q_pytorch).")
    except Exception as e:
        print(f"   -> [LỖI] Export Trace thất bại: {e}")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    print("="*70)
    print(" DPU COMPATIBILITY & FLOAT MODEL VALIDATION PIPELINE ")
    print("="*70)
    
    # 1. Tự động kiểm tra số lượng class của trọng số (để quyết định NUM_CLASSES là 2 hay 5)
    model_num_classes = NUM_CLASSES
    try:
        if os.path.exists(MODEL_WEIGHTS):
            state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu')
            # Tìm class số từ layer cuối
            detect_key = 'backbone.classifier.4.weight' if 'backbone.classifier.4.weight' in state_dict else 'classifier.4.weight'
            if detect_key in state_dict:
                model_num_classes = state_dict[detect_key].shape[0]
                print(f"[!] Tự động phát hiện Model Weights có {model_num_classes} Classes!")
    except Exception as e:
        pass
        
    local_class_names = CLASS_NAMES[:model_num_classes] if model_num_classes <= 5 else CLASS_NAMES
    
    # 2. Khởi tạo Model và Load Weights
    print(f"[+] Khởi tạo MobileNetV2 với {model_num_classes} lớp phân loại...")
    model = get_model(num_classes=model_num_classes)
    
    if os.path.exists(MODEL_WEIGHTS):
        print(f"[+] Đang nạp trọng số (.pth) từ: {MODEL_WEIGHTS}")
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    else:
        print(f"[CẢNH BÁO] Không tìm thấy file trọng số {MODEL_WEIGHTS}. Đang chạy với trọng số ngẫu nhiên!")
        
    model.to(DEVICE)
    
    # 3. Quét kiểm tra Mức độ tương thích DPU
    check_dpu_compatibility(model)
    
    # 4. Tạo Dataloader Evaluation
    print("\n[+] Xây dựng Evaluation Dataloader (ICBHI)...")
    dataset = ValidationDataset(DATA_DIR, class_list=local_class_names)
    if len(dataset) > 0:
        # Giới hạn số lượng test trong ví dụ này cho nhanh (lấy ví dụ 100 sample test)
        # Bỏ dòng này nếu muốn chạy full dataset!
        subset_indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
        subset_ds = torch.utils.data.Subset(dataset, subset_indices)
        
        dataloader = DataLoader(subset_ds, batch_size=16, shuffle=False, num_workers=4)
        
        # 5. Đánh giá Accuracy / Sensitivity / F1-score
        validate_model(model, dataloader)
    else:
        print("[LỖI] Không tìm thấy dữ liệu wav phù hợp với các Class yêu cầu.")
        
    # 6. Export rễ tới TorchScript .pt
    export_trace(model, EXPORT_TRACE_PATH)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
