import os
import argparse
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def preprocess_audio(y, sr=4000, segment_len=32000):
    """Tiền xử lý y hệt trong distillation_02.py (BPF 25-2000Hz, Pad/Crop 8s)"""
    # Bandpass filter
    b, a = butter_bandpass(25, min(2000, sr // 2 - 1), sr, order=3)
    audio = signal.filtfilt(b, a, y).astype(np.float32)

    # Normalize
    max_val = np.max(np.abs(audio)) + 1e-10
    audio = audio / max_val

    # Pad / Crop to exactly 8 seconds (32000 samples @ 4kHz)
    if len(audio) < segment_len:
        repeats = segment_len // len(audio) + 1
        audio = np.tile(audio, repeats)[:segment_len]
    elif len(audio) > segment_len:
        start = (len(audio) - segment_len) // 2
        audio = audio[start:start + segment_len]

    return audio.astype(np.float32)

def compute_gammatone_filterbank(sr, n_filters=64, fmin=50, fmax=2000):
    ear_q = 9.26449
    min_bw = 24.7
    freqs = -(ear_q * min_bw) + np.exp(
        np.arange(1, n_filters + 1) * (
            -np.log(fmax + ear_q * min_bw) + np.log(fmin + ear_q * min_bw)
        ) / n_filters
    ) * (fmax + ear_q * min_bw)
    freqs = np.flip(freqs)
    return freqs

def compute_gammatonegram(audio, sr=4000, n_filters=64, n_fft=512, hop_length=256):
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    cf = compute_gammatone_filterbank(sr, n_filters, fmin=50, fmax=min(2000, sr // 2 - 1))
    weights = np.zeros((n_filters, len(f)))
    for i, center_freq in enumerate(cf):
        erb = 24.7 * (4.37 * center_freq / 1000 + 1)
        weights[i] = np.exp(-0.5 * ((f - center_freq) / (erb * 0.5)) ** 2)
    power = np.abs(Zxx) ** 2
    gammatone_spec = np.dot(weights, power)
    gammatone_spec = np.log10(gammatone_spec + 1e-10)
    return gammatone_spec

def compute_mel_spectrogram(audio, sr=4000, n_mels=64, n_fft=512, hop_length=256):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=50, fmax=min(2000, sr // 2)
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def create_hybrid_spectrogram(audio, sr=4000, output_size=224):
    gamma = compute_gammatonegram(audio, sr)
    mel = compute_mel_spectrogram(audio, sr)

    def normalize(x):
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return x

    gamma = normalize(gamma)
    mel = normalize(mel)

    gamma_resized = zoom(gamma, (output_size / gamma.shape[0], output_size / gamma.shape[1]), order=1)
    mel_resized = zoom(mel, (output_size / mel.shape[0], output_size / mel.shape[1]), order=1)

    gamma_resized = np.clip(gamma_resized[:output_size, :output_size], 0, 1)
    mel_resized = np.clip(mel_resized[:output_size, :output_size], 0, 1)

    avg_channel = (gamma_resized + mel_resized) / 2.0
    
    # Stack channels for RGB image: (Gammatone, Mel, Average)
    # Note: axis=-1 to make it (224, 224, 3) for matplotlib imshow.
    hybrid = np.stack([gamma_resized, mel_resized, avg_channel], axis=-1)

    return gamma_resized, mel_resized, avg_channel, hybrid

def extract_and_plot_hybrid(audio_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    print(f"Đang xử lý file cho Hybrid Spectrogram: {audio_path}")
    y, sr = librosa.load(audio_path, sr=4000)
    
    # 1. Preprocess audio theo chuẩn distillation_02.py (Cắt/ghép 8s, Bandpass filter)
    y_processed = preprocess_audio(y, sr)
    
    # 2. Tạo Hybrid Spectrogram (Gammatone + Mel -> 224x224)
    gamma_img, mel_img, avg_img, hybrid_img = create_hybrid_spectrogram(y_processed, sr, output_size=224)
    
    # Lật dọc ảnh vì trục y (tần số) trong ma trận đang ngược so với đồ họa (0 ở trên cùng)
    gamma_img = np.flipud(gamma_img)
    mel_img = np.flipud(mel_img)
    avg_img = np.flipud(avg_img)
    hybrid_img = np.flipud(hybrid_img)
    
    # 3. Vẽ và xuất 4 ảnh riêng biệt
    def save_spec_image(img_data, suffix, cmap=None):
        plt.figure(figsize=(5, 5))
        if cmap is None:
            plt.imshow(img_data, aspect='auto') # Ảnh màu 3 kênh (RGB)
        else:
            plt.imshow(img_data, aspect='auto', cmap=cmap) # Phổ thành phần 1 kênh
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        
        out_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        print(f" -> Đã lưu ảnh {suffix} tại: {out_path}")

    # Xuất các ảnh (dùng phổ màu 'jet' cho các ảnh thành phần 1 kênh để hiển thị rõ nhất)
    save_spec_image(gamma_img, "gammatone_224", cmap='jet')
    save_spec_image(mel_img, "mel_224", cmap='jet')
    save_spec_image(avg_img, "average_224", cmap='jet')
    save_spec_image(hybrid_img, "hybrid_rgb_224") # Ảnh gốc RGB gộp 3 phổ

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Hybrid Gammatone-Mel Spectrogram 224x224")
    parser.add_argument("--audio", type=str, required=True, help="Đường dẫn đến file audio")
    parser.add_argument("--outdir", type=str, default="./visualizations", help="Thư mục xuất ảnh")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"[Lỗi] Không tìm thấy file audio: {args.audio}")
        exit(1)
        
    extract_and_plot_hybrid(args.audio, args.outdir)
