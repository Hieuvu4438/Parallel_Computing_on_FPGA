import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
from scipy.ndimage import zoom

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SAMPLE_WAV = "/home/iec/Parallel_Computing_on_FPGA/data/samples/ICBHI_final_database/101_1b1_Al_sc_Meditron.wav"
OUTPUT_DIR = "/home/iec/Parallel_Computing_on_FPGA/assets/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BPF Parameters from Fixing_Paper.md
BPF_L13 = (50, 2500)  # Layer 1-3
BPF_L4 = (25, 2000)   # Layer 4
TARGET_SR = 4000      # Target sampling rate

# ==============================================================================
# SIGNAL PROCESSING FUNCTIONS
# ==============================================================================

def bandpass_filter(data, fs, lowcut, highcut, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut, nyq - 1.0) / nyq
    if low >= high:
        return data
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def generate_cwt_spectrogram(audio, sr, lowcut, highcut, num_scales=128):
    wavelet = 'morl'
    center_freq = pywt.central_frequency(wavelet)
    
    max_scale = center_freq * sr / lowcut
    min_scale = center_freq * sr / highcut
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    
    coefficients, _ = pywt.cwt(audio, scales, wavelet, sampling_period=1.0/sr)
    power = np.abs(coefficients) ** 2
    power_db = 10 * np.log10(power + 1e-10)
    
    # Normalize
    power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)
    return power_db

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_signal_transformation(wav_path):
    # 1. Load Audio
    audio_raw, sr_orig = librosa.load(wav_path, sr=None)
    
    # Check for NaNs or Infs in raw audio
    if not np.all(np.isfinite(audio_raw)):
        print("[WARNING] Raw audio contains non-finite values. Cleaning...")
        audio_raw = np.nan_to_num(audio_raw)

    duration = len(audio_raw) / sr_orig
    time_orig = np.linspace(0, duration, len(audio_raw))
    
    # 2. Filter & Resample (Layer 4 track)
    audio_filtered = bandpass_filter(audio_raw, sr_orig, BPF_L4[0], BPF_L4[1])
    
    # Check for NaNs or Infs after filtering
    if not np.all(np.isfinite(audio_filtered)):
        print("[WARNING] Filtered audio contains non-finite values. Clipping...")
        audio_filtered = np.nan_to_num(audio_filtered, nan=0.0, posinf=1.0, neginf=-1.0)

    audio_resampled = librosa.resample(audio_filtered, orig_sr=sr_orig, target_sr=TARGET_SR)
    time_resampled = np.linspace(0, duration, len(audio_resampled))
    
    # 3. Generate CWT
    spec = generate_cwt_spectrogram(audio_resampled, TARGET_SR, BPF_L4[0], BPF_L4[1])
    
    # ---------------------------------------------------------
    # Create Figure for Journal Paper
    # ---------------------------------------------------------
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1)
    
    # A. Raw Waveform
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_orig, audio_raw, color='gray', alpha=0.7, linewidth=0.5)
    ax1.set_title("(a) Raw Respiratory Audio Signal", fontweight='bold')
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, duration)
    ax1.grid(True, alpha=0.3)
    
    # B. Filtered & Resampled
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_resampled, audio_resampled, color='tab:blue', linewidth=0.7)
    ax2.set_title(f"(b) Preprocessed Signal (BPF {BPF_L4[0]}-{BPF_L4[1]}Hz, SR={TARGET_SR}Hz)", fontweight='bold')
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(0, duration)
    ax2.grid(True, alpha=0.3)
    
    # C. CWT Spectrogram
    ax3 = fig.add_subplot(gs[2, 0])
    img = ax3.imshow(spec, aspect='auto', origin='lower', extent=[0, duration, BPF_L4[0], BPF_L4[1]], cmap='jet')
    ax3.set_title("(c) Continuous Wavelet Transform (CWT) Spectrogram - Morlet", fontweight='bold')
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_xlabel("Time (seconds)")
    fig.colorbar(img, ax=ax3, label='Normalized Intensity')
    
    # Save results
    save_path = os.path.join(OUTPUT_DIR, "signal_preprocessing_steps.png")
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Figure saved to: {save_path}")
    
    # ---------------------------------------------------------
    # Create Separate Clean CWT for Paper Figure
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, "cwt_clean.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"[SUCCESS] Clean CWT saved for paper figures.")

if __name__ == "__main__":
    if os.path.exists(SAMPLE_WAV):
        plot_signal_transformation(SAMPLE_WAV)
    else:
        print(f"[ERROR] Sample file not found: {SAMPLE_WAV}")
