import os
import random
import glob
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.ndimage import zoom
import pywt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# CẤU HÌNH CƠ BẢN
# ==========================================
TARGET_SR = 4000
IMG_SIZE = 224
TARGET_NUM_CALIB = 200

def bandpass_filter(data, fs, lowcut=50.0, highcut=2500.0, order=5):
    """
    Lọc Bandpass. Lưu ý: Định lý Nyquist chỉ cho phép tần số tối đa = fs / 2.
    Nếu highcut vượt quá Nyquist, ta sẽ tự động giới hạn lại.
    """
    nyq = 0.5 * fs
    # Đảm bảo lowcut và highcut hợp lệ với tần số lấy mẫu hiện tại
    low = min(lowcut, nyq - 20.0) / nyq
    high = min(highcut, nyq - 1.0) / nyq
    
    if low >= high: # Tránh lỗi bộ lọc nếu tín hiệu có fs quá thấp
        return data
        
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y

class WaveletTransform:
    def __init__(self, wavelet='morl', num_scales=128, sample_rate=TARGET_SR, freq_range=(50, 1900), output_size=IMG_SIZE):
        """
        CWT bằng thư viện PyWavelets (pywt)
        Tần số max của CWT khi SR=4000Hz là 2000Hz. Ta chọn range (50, 1900)
        """
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.sample_rate = sample_rate
        self.freq_range = freq_range
        self.output_size = output_size
        self.scales = self._compute_scales()

    def _compute_scales(self) -> np.ndarray:
        center_freq = pywt.central_frequency(self.wavelet)
        min_scale = center_freq * self.sample_rate / self.freq_range[1]
        max_scale = center_freq * self.sample_rate / self.freq_range[0]
        return np.logspace(np.log10(min_scale), np.log10(max_scale), self.num_scales)

    def to_image(self, audio: np.ndarray) -> np.ndarray:
        coefficients, _ = pywt.cwt(audio, self.scales, self.wavelet, sampling_period=1.0 / self.sample_rate)
        power = np.abs(coefficients) ** 2
        power_db = 10 * np.log10(power + 1e-10)
        
        # Min-Max Normalize về khoảng [0, 1]
        power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)
        
        # Resize về (224, 224)
        zoom_factors = (self.output_size / power_db.shape[0], self.output_size / power_db.shape[1])
        spec_resized = zoom(power_db, zoom_factors, order=1)
        spec_resized = spec_resized[:self.output_size, :self.output_size]
        return np.clip(spec_resized, 0, 1).astype(np.float32)

class AudioPreprocessor:
    def __init__(self, target_sr=TARGET_SR, img_size=IMG_SIZE):
        self.target_sr = target_sr
        self.wavelet_transform = WaveletTransform(sample_rate=target_sr, output_size=img_size)
    
    def process(self, wav_path):
        try:
            sr, audio = wavfile.read(wav_path)
            if len(audio) == 0:
                audio = np.zeros(self.target_sr * 2, dtype=np.float32)
                sr = self.target_sr
        except Exception:
            audio = np.zeros(self.target_sr * 2, dtype=np.float32)
            sr = self.target_sr

        # Chuyển về float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        # 1. Bandpass Filtering (50Hz - 2500Hz)
        # THỰC HIỆN TRƯỚC KHI RESAMPLING để tránh vi phạm Nyquist nếu target_sr = 4000Hz (Nyquist = 2000Hz)
        audio = bandpass_filter(audio, fs=sr, lowcut=50.0, highcut=2500.0)

        # 2. Resampling về 4000Hz
        if sr != self.target_sr:
            num_samples = int(len(audio) * self.target_sr / sr)
            audio = signal.resample(audio, num_samples)
            sr = self.target_sr

        # 3. Chuyển đổi sang Wavelet Spectrogram (CWT) và resize (224, 224)
        spec = self.wavelet_transform.to_image(audio)
        
        # Scale về [0, 255] và biểu diễn dưới dạng ảnh RGB giả lập (sao chép 3 kênh) để lưu thành ảnh PNG
        spec_img = (spec * 255).astype(np.uint8)
        spec_rgb = np.stack([spec_img] * 3, axis=-1)
        
        return spec_rgb

# ==========================================
# DATASET CHO VITIS AI CALIBRATION
# ==========================================
class ICBHICalibrationDataset(Dataset):
    """
    DataLoader chuẩn hóa để dùng cho Vitis AI (pytorch-nndct) quantizer
    """
    def __init__(self, calib_dir, transform=None):
        self.calib_dir = calib_dir
        self.image_paths = glob.glob(os.path.join(calib_dir, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Mở ảnh bằng PIL và chuẩn hóa
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image

def main():
    data_dir = "/home/iec/Parallel_Computing_on_FPGA/data/samples/ICBHI_final_database"
    calib_dir = "/home/iec/Parallel_Computing_on_FPGA/data/calib_images"
    os.makedirs(calib_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    if len(wav_files) >= TARGET_NUM_CALIB:
        selected_files = random.sample(wav_files, TARGET_NUM_CALIB)
    else:
        selected_files = wav_files

    preprocessor = AudioPreprocessor()

    print(f"Bắt đầu tiền xử lý {len(selected_files)} file để tạo Tập dữ liệu hiệu chuẩn (Calibration)...")
    for idx, wav_path in enumerate(selected_files):
        # Tiền xử lý âm thanh: Lọc, Resample, CWT, Resize
        spec_rgb = preprocessor.process(wav_path)
        
        # Lưu định dạng PNG để có thể xem được bằng mắt thường và tương thích Vitis AI Datasets
        img = Image.fromarray(spec_rgb)
        filename = os.path.basename(wav_path).replace('.wav', '.png')
        save_path = os.path.join(calib_dir, filename)
        img.save(save_path)
        
        if (idx + 1) % 20 == 0:
            print(f"  -> Đã xử lý {idx + 1}/{len(selected_files)} files")

    print(f"\n[XONG] Đã lưu {len(selected_files)} ảnh PNG (224x224x3) vào thư mục: {calib_dir}")

    # ==========================================
    # KHỞI TẠO DATALOADER (Hướng dẫn phần Quantization)
    # ==========================================
    # Chuẩn hóa Tensor tương tự như ResNet/MobileNet
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    calib_dataset = ICBHICalibrationDataset(calib_dir, transform=normalize_transform)
    calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=False)

    print(f"\n[KIỂM TRA] Định dạng Tensor đầu ra từ DataLoader:")
    for images in calib_loader:
        print(f"Kích thước 1 batch (Shape): {images.shape}")
        print(f"Min value: {images.min():.4f}, Max value: {images.max():.4f}")
        break
    
    print("\nDataset đã sẵn sàng để đưa vào bước Vitis AI (pytorch-nndct) Quantization!")

if __name__ == "__main__":
    main()
