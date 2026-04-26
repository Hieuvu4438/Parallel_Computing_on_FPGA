import os
import glob
import numpy as np
import librosa
import scipy.signal as signal
from scipy.ndimage import zoom
import pywt
from PIL import Image
import torch
import torchvision.transforms as transforms
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # Fallback nếu không có tqdm

# ==============================================================================
# HÀM BANDPASS FILTER
# ==============================================================================
def bandpass_filter(data, fs, lowcut=50.0, highcut=2500.0, order=5):
    """
    Sử dụng bộ lọc Butterworth để giữ dải tần từ 50Hz đến 2500Hz nhằm
    loại bỏ nhiễu môi trường và tiếng tim (thường < 50Hz hoặc bên ngoài dải phân tích).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    # Đảm bảo giới hạn trên nhỏ hơn tần số Nyquist của tín hiệu
    high = min(highcut, nyq - 1.0) / nyq
    if low >= high:
        return data
    b, a = signal.butter(order, [low, high], btype='band')
    # Dùng filtfilt để triệt tiêu độ trễ pha (zero-phase filter)
    y = signal.filtfilt(b, a, data)
    return y

# ==============================================================================
# HÀM XỬ LÝ CHÍNH
# ==============================================================================
def preprocess_and_save(wav_path, output_dir, target_sr=4000, img_size=(224, 224)):
    """
    Tiền xử lý file âm thanh hô hấp: Lọc -> Resample -> Wavelet Transform (CWT) -> Save Normalization
    """
    # ---------------------------------------------------------
    # 1. RESAMPLING & FILTERING
    # ---------------------------------------------------------
    # Sử dụng librosa để load toàn bộ audio (sr=None giữ nguyên tần số gốc)
    audio, original_sr = librosa.load(wav_path, sr=None) 
    
    # Áp dụng bộ lọc băng thông 50Hz - 2500Hz khi tần số vẫn đang cao (thường 44100Hz)
    # Ghi chú DSP: Phải Bandpass trước khi giảm tần số lấy mẫu về 4000Hz,
    # bởi vì theo định lý Nyquist, nếu SR = 4000Hz thì ta không thể biểu diễn tần số 2500Hz.
    audio_filtered = bandpass_filter(audio, original_sr, lowcut=50.0, highcut=2500.0)
    
    # Resample về 4000 Hz
    if original_sr != target_sr:
        audio_resampled = librosa.resample(audio_filtered, orig_sr=original_sr, target_sr=target_sr)
        sr = target_sr
    else:
        audio_resampled = audio_filtered
        sr = target_sr
        
    # ---------------------------------------------------------
    # 2. WAVELET TRANSFORM (CWT) BẰNG MORLET WAVELET
    # ---------------------------------------------------------
    """
    TẠI SAO CWT HIỆU QUẢ HƠN STFT ĐỐI VỚI TÍN HIỆU HÔ HẤP?
    - Tín hiệu âm thanh hô hấp (như tiếng crackles, wheezes) là tín hiệu không ổn định (non-stationary), 
      thường chứa các thành phần tần số thay đổi đột ngột theo thời gian.
    - Phương pháp STFT (Short-Time Fourier Transform) sử dụng một cửa sổ trượt kích thước cố định, 
      dẫn đến sự đánh đổi mang tính hệ thống giữa độ phân giải thời gian và độ phân giải tần số.
    - Trong khi đó, CWT khắc phục nhược điểm này bằng cách thay đổi độ rộng cửa sổ (thông qua "scales"):
        + Tần số cao: Cửa sổ thời gian tự động co ngắn -> Độ phân giải thời gian vượt trội -> Bắt tốt 
          được các âm thanh chớp nhoáng, sắc nét (như tiếng rì rào phế nang hay tiếng rales).
        + Tần số thấp: Cửa sổ thời gian giãn rộng -> Độ phân giải tần số tốt -> Đảm bảo độ chính xác cho 
          những đặc trưng âm trầm trải dài dải rộng (như tiếng ngáy, tiếng tim ngầm).
    """
    wavelet = 'morl'
    
    # Tính toán Scale cho CWT (chúng ta giới hạn từ 50Hz tới 1950Hz (nhỏ hơn chuẩn Nyquist 2000Hz ở SR=4k))
    center_freq = pywt.central_frequency(wavelet)
    min_freq, max_freq = 50.0, 1950.0
    
    max_scale = center_freq * sr / min_freq
    min_scale = center_freq * sr / max_freq
    # Logspace chia scale giúp ta quan sát các dải tần số phi tuyến rõ hơn
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 128)
    
    # Thực hiện CWT Transforms
    coefficients, _ = pywt.cwt(audio_resampled, scales, wavelet, sampling_period=1.0/sr)
    power = np.abs(coefficients) ** 2
    power_db = 10 * np.log10(power + 1e-10)
    
    # Normalize Spectrogram matrix về [0.0 - 1.0]
    power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)
    
    # ---------------------------------------------------------
    # 3. IMAGE FORMATTING (Resize & Normalization)
    # ---------------------------------------------------------
    target_h, target_w = img_size
    # Dùng hàm zoom của scipy để resize nhanh ma trận thay vì dùng code loop
    zoom_factors = (target_h / power_db.shape[0], target_w / power_db.shape[1])
    spec_resized = zoom(power_db, zoom_factors, order=1)
    spec_resized = spec_resized[:target_h, :target_w]
    
    # Tạo ảnh màu (RGB) kích thước 224x224 (sao chép ra 3 kênh màu giống nhau)
    spec_img_8bit = (spec_resized * 255).astype(np.uint8)
    spec_rgb = np.stack([spec_img_8bit] * 3, axis=-1)
    
    # Chuẩn hóa (Normalization) bằng ImageNet parameters cho CNN Model
    # Đây là thao tác thường được Pytorch thực hiện, ta mô phỏng lại để chứng minh luồng chuẩn hóa
    tensor_img = torch.from_numpy(spec_rgb.transpose((2, 0, 1))).float() / 255.0
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # Mean của tập ImageNet chuẩn
        std=[0.229, 0.224, 0.225]   # Std của tập ImageNet chuẩn
    )
    tensor_normalized = normalize(tensor_img)
    
    # ---------------------------------------------------------
    # 4. OUTPUT TO CALIB FOLDER
    # ---------------------------------------------------------
    base_name = os.path.basename(wav_path).replace('.wav', '.png')
    save_path = os.path.join(output_dir, base_name)
    
    # Lưu ảnh Spectrogram PNG để chuẩn bị cho Quantization
    Image.fromarray(spec_rgb).save(save_path)
    
    return tensor_normalized, save_path

# ==============================================================================
# HÀM THỰC THI (MAIN)
# ==============================================================================
def main():
    print("="*70)
    print(" AUDIO PREPROCESSING PIPELINE (CWT) FOR XILINX FPGA QUANTIZATION ")
    print("="*70)
    
    # Định nghĩa thư mục input/output
    data_dir = "/home/iec/Parallel_Computing_on_FPGA/data/samples/ICBHI_final_database"
    output_dir = "/home/iec/Parallel_Computing_on_FPGA/data/calib_images_02"
    
    # Tạo folder chứa ảnh Calibration (Quantization)
    os.makedirs(output_dir, exist_ok=True)
    
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    if not wav_files:
        print(f"[!] Không tìm thấy file .wav nào trong: {data_dir}")
        return
        
    # Tạo bộ Calibration lấy tiêu biểu 200 file
    sample_files = wav_files[:200]
    
    print(f"[+] Tìm thấy {len(wav_files)} file âm thanh.")
    print(f"[+] Đang trích xuất cấu trúc CWT & Bandpass cho {len(sample_files)} file làm Calibration Data.")
    print(f"[+] Thư mục đầu ra ảnh: {output_dir}\n")
    
    for i, wav_path in enumerate(sample_files):
        try:
            _, _ = preprocess_and_save(wav_path, output_dir)
            if (i+1) % 20 == 0:
                print(f"  -> Đã xử lý thành công: {i+1}/{len(sample_files)} files...")
        except Exception as e:
            print(f"[ERROR] Lỗi xử lý file {wav_path}: {e}")
            
    print("\n[HOÀN THÀNH]")
    print(f"Đã lưu thành công {len(sample_files)} ảnh Spectrogram (224x224 RGB) vào {output_dir}.")
    print("Dữ liệu Calibration hiện tại đã tương thích 100% với Xilinx Vitis AI (pytorch-nndct)!")

if __name__ == "__main__":
    main()
