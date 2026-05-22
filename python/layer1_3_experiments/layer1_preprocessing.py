#!/usr/bin/env python3
"""
================================================================================
Layer 1: Signal Preprocessing + Feature Extraction Pipeline (V2 — Hybrid)
================================================================================
Hệ thống chẩn đoán âm thanh hô hấp — Cascaded Framework (ICBHI 2017)

Chức năng:  1. Tiền xử lý tín hiệu: Load WAV → Resample 4kHz → BPF 50-2500Hz → Normalize
  2. Trích xuất đặc trưng nhẹ (→ Layer 2 Random Forest): ZCR, RMS, MFCCs
  3. Trích xuất Hybrid Spectrogram (→ Layer 3 CNN/DPU):
     Gammatonegram + Mel-spectrogram → 3-channel 224x224 image
     (Thay thế CWT — nhanh hơn 10x, chất lượng tốt hơn cho transfer learning)

Output:
  - features.csv       — Bảng đặc trưng cho Random Forest
  - spectrograms/      — Folder ảnh 3-channel 224x224 (Gammatone, Mel, Average)

Tối ưu cho môi trường nhúng:
  - Xử lý tuần tự, tiết kiệm RAM
  - Sử dụng numpy/scipy thuần (không phụ thuộc pywt)
  - Code module hóa, dễ chuyển sang C++/HLS

Usage:
    python layer1_preprocessing.py
    python layer1_preprocessing.py --data_dir /path/to/audio --output_dir /path/to/output
    python layer1_preprocessing.py --skip_spectrograms   # Chỉ trích xuất features.csv
    python layer1_preprocessing.py --max_samples 50       # Test nhanh với 50 mẫu

Author: Cascaded FPGA Framework
================================================================================
"""

import os
import sys
import csv
import time
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile
from scipy.ndimage import zoom as scipy_zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import COMBINED_AUDIO_DIR, COMBINED_LABELS, LAYER13_ARTIFACTS_DIR

warnings.filterwarnings('ignore')

# ==============================================================================
# Lazy imports — chỉ import khi cần
# ==============================================================================
_librosa = None
_cv2 = None


def _get_librosa():
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


# ==============================================================================
# CẤU HÌNH HỆ THỐNG (System Configuration)
# ==============================================================================
# --- Signal Preprocessing ---
TARGET_SR = 4000          # Tần số lấy mẫu mục tiêu (Hz)
BPF_LOW = 50              # Tần số cắt thấp BPF (Hz)
BPF_HIGH = 2500           # Tần số cắt cao BPF (Hz) — theo paper
BPF_ORDER = 5             # Bậc bộ lọc Butterworth
FIXED_DURATION = 5.0      # Chuẩn hóa độ dài audio (giây)
FIXED_SAMPLES = int(FIXED_DURATION * TARGET_SR)  # = 20000 mẫu

# --- Feature Extraction (Layer 2 — Random Forest) ---
N_MFCC = 13               # Số hệ số MFCC cơ bản
N_FFT = 512               # FFT window size (phù hợp 4kHz SR)
HOP_LENGTH = 128           # Hop length cho MFCC/spectrogram
N_MELS = 128               # Số Mel filter banks (cho Mel spectrogram)
N_GAMMATONE = 64           # Số Gammatone filters

# --- Hybrid Spectrogram (Layer 3 — CNN/DPU) ---
SPEC_SIZE = 224            # Kích thước ảnh spectrogram (pixels)

# --- Classes ---
CLASS_NAMES = ['COPD', 'Healthy', 'Non-COPD']


# ==============================================================================
# MODULE 1: TIỀN XỬ LÝ TÍN HIỆU (Signal Preprocessing)
# ==============================================================================
class SignalPreprocessor:
    """
    Pipeline tiền xử lý tín hiệu âm thanh hô hấp.

    Flow: Load WAV → Mono → Resample 4kHz → BPF 50-2500Hz → Normalize → Fixed Length

    Tối ưu cho embedded:
      - Sử dụng scipy thuần thay vì librosa cho resampling
      - BPF Butterworth bậc 5 (tương thích HLS IP)
      - Normalize [-1, 1] bằng max amplitude
    """

    def __init__(self, target_sr: int = TARGET_SR,
                 bpf_low: float = BPF_LOW,
                 bpf_high: float = BPF_HIGH,
                 bpf_order: int = BPF_ORDER,
                 fixed_duration: float = FIXED_DURATION):
        self.target_sr = target_sr
        self.bpf_low = bpf_low
        self.bpf_high = bpf_high
        self.bpf_order = bpf_order
        self.fixed_samples = int(fixed_duration * target_sr)

        # Pre-compute BPF coefficients (chỉ tính 1 lần)
        nyq = 0.5 * target_sr
        low = bpf_low / nyq
        high = min(bpf_high, nyq - 1) / nyq
        self.b, self.a = scipy_signal.butter(bpf_order, [low, high], btype='band')

    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Nạp file WAV và chuyển về float32 [-1, 1]."""
        sr, audio = wavfile.read(filepath)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.float64:
            audio = audio.astype(np.float32)
        else:
            audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio, sr

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample về TARGET_SR bằng scipy (tương thích embedded)."""
        if orig_sr == self.target_sr:
            return audio
        num_samples = int(len(audio) * self.target_sr / orig_sr)
        return scipy_signal.resample(audio, max(num_samples, 1)).astype(np.float32)

    def bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Bộ lọc thông dải Butterworth 50-2500Hz."""
        return scipy_signal.filtfilt(self.b, self.a, audio).astype(np.float32)

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Chuẩn hóa biên độ về [-1, 1]."""
        max_amp = np.max(np.abs(audio))
        if max_amp > 1e-6:
            return (audio / max_amp).astype(np.float32)
        return audio

    def standardize_length(self, audio: np.ndarray) -> np.ndarray:
        """Chuẩn hóa độ dài audio về FIXED_DURATION giây."""
        target_len = self.fixed_samples
        if len(audio) >= target_len:
            start = (len(audio) - target_len) // 2
            return audio[start:start + target_len]
        else:
            repeats = target_len // len(audio) + 1
            return np.tile(audio, repeats)[:target_len]

    def process(self, filepath: str) -> np.ndarray:
        """Pipeline: Load → Resample → BPF → Normalize → Fixed Length."""
        audio, sr = self.load_audio(filepath)
        audio = self.resample(audio, sr)
        audio = self.bandpass_filter(audio)
        audio = self.normalize(audio)
        audio = self.standardize_length(audio)
        return audio


# ==============================================================================
# MODULE 2: TRÍCH XUẤT ĐẶC TRƯNG NHẸ (cho Layer 2 — Random Forest)
# ==============================================================================
class FeatureExtractor:
    """
    Trích xuất đặc trưng thống kê từ tín hiệu audio.

    Features (tổng 43 chiều):
      - ZCR mean + std           (2)
      - RMS mean + std           (2)
      - MFCCs mean (13) + std (13) = (26)
      - Delta MFCCs mean (13)    (13)
    """

    def __init__(self, sr: int = TARGET_SR, n_mfcc: int = N_MFCC,
                 n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                 n_mels: int = 40, include_deltas: bool = True):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.include_deltas = include_deltas

    def compute_zcr(self, audio: np.ndarray) -> Tuple[float, float]:
        """Zero Crossing Rate — đặc trưng miền thời gian."""
        librosa = _get_librosa()
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        return float(np.mean(zcr)), float(np.std(zcr))

    def compute_rms(self, audio: np.ndarray) -> Tuple[float, float]:
        """Root Mean Square Energy — năng lượng tín hiệu."""
        librosa = _get_librosa()
        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        return float(np.mean(rms)), float(np.std(rms))

    def compute_mfcc(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """MFCCs 13 hệ số + Delta."""
        librosa = _get_librosa()
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        result = {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
        }

        if self.include_deltas:
            delta = librosa.feature.delta(mfccs)
            result['delta_mean'] = np.mean(delta, axis=1)

        return result

    def extract(self, audio: np.ndarray) -> dict:
        """Trích xuất tất cả đặc trưng từ audio đã tiền xử lý."""
        zcr_mean, zcr_std = self.compute_zcr(audio)
        rms_mean, rms_std = self.compute_rms(audio)
        mfcc_data = self.compute_mfcc(audio)

        features = {
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
        }

        for i, val in enumerate(mfcc_data['mfcc_mean']):
            features[f'mfcc_mean_{i+1}'] = float(val)
        for i, val in enumerate(mfcc_data['mfcc_std']):
            features[f'mfcc_std_{i+1}'] = float(val)
        if 'delta_mean' in mfcc_data:
            for i, val in enumerate(mfcc_data['delta_mean']):
                features[f'delta_mfcc_mean_{i+1}'] = float(val)

        return features

    @staticmethod
    def get_feature_names(n_mfcc: int = N_MFCC, include_deltas: bool = True) -> List[str]:
        """Trả về danh sách tên cột cho CSV header."""
        names = ['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std']
        names += [f'mfcc_mean_{i+1}' for i in range(n_mfcc)]
        names += [f'mfcc_std_{i+1}' for i in range(n_mfcc)]
        if include_deltas:
            names += [f'delta_mfcc_mean_{i+1}' for i in range(n_mfcc)]
        return names


# ==============================================================================
# MODULE 3: HYBRID SPECTROGRAM (cho Layer 3 — CNN/DPU)
# Tích hợp từ distillation_02.py: Gammatone + Mel → 3-channel 224x224
# ==============================================================================
class HybridSpectrogramGenerator:
    """
    Tạo Hybrid Spectrogram 3-channel (Gammatone, Mel, Average) kích thước 224x224.

    Thay thế CWT:
      - Nhanh hơn ~10x (FFT-based thay vì CWT)
      - 3-channel có thông tin phong phú hơn (thay vì copy 1 channel)
      - Tương thích pretrained ImageNet models (3-channel input)
      - Phù hợp DPU inference trên FPGA

    Channel:
      0: Gammatonegram  — mô phỏng hệ thống thính giác (ERB scale)
      1: Mel-spectrogram — biểu diễn tần số theo thang Mel
      2: Average         — trung bình 2 channel trên → smoothing
    """

    def __init__(self, sr: int = TARGET_SR, n_gammatone: int = N_GAMMATONE,
                 n_mels: int = N_MELS, n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH, output_size: int = SPEC_SIZE):
        self.sr = sr
        self.n_gammatone = n_gammatone
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.output_size = output_size

    # --- Gammatone Filterbank (ERB scale) ---
    def compute_gammatone_filterbank(self, fmin: float = 50,
                                      fmax: float = 2000) -> np.ndarray:
        """
        Tạo center frequencies cho Gammatone filterbank (ERB scale).
        Mô phỏng hệ thống thính giác con người — tối ưu cho âm thanh hô hấp.
        """
        ear_q = 9.26449
        min_bw = 24.7
        freqs = -(ear_q * min_bw) + np.exp(
            np.arange(1, self.n_gammatone + 1) * (
                -np.log(fmax + ear_q * min_bw) + np.log(fmin + ear_q * min_bw)
            ) / self.n_gammatone
        ) * (fmax + ear_q * min_bw)
        freqs = np.flip(freqs)
        return freqs

    def compute_gammatonegram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute gammatonegram bằng FFT-based approximation.
        Nhanh hơn time-domain gammatone filtering, phù hợp embedded.
        """
        # STFT
        f, t, Zxx = scipy_signal.stft(
            audio, fs=self.sr, nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )

        # Gammatone-like filterbank weights
        cf = self.compute_gammatone_filterbank(
            fmin=50, fmax=min(2000, self.sr // 2 - 1)
        )
        weights = np.zeros((self.n_gammatone, len(f)))
        for i, center_freq in enumerate(cf):
            erb = 24.7 * (4.37 * center_freq / 1000 + 1)
            weights[i] = np.exp(-0.5 * ((f - center_freq) / (erb * 0.5)) ** 2)

        # Apply filterbank
        power = np.abs(Zxx) ** 2
        gammatone_spec = np.dot(weights, power)

        # Log compression
        gammatone_spec = np.log10(gammatone_spec + 1e-10)
        return gammatone_spec

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute log Mel-spectrogram.
        Sử dụng librosa nếu có, fallback sang scipy.
        """
        try:
            librosa = _get_librosa()
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft,
                hop_length=self.hop_length, fmin=50,
                fmax=min(2000, self.sr // 2)
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        except Exception:
            # Manual computation fallback
            f, t, Zxx = scipy_signal.stft(
                audio, fs=self.sr, nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length
            )
            power = np.abs(Zxx) ** 2

            mel_min = 2595 * np.log10(1 + 50 / 700)
            mel_max = 2595 * np.log10(1 + min(2000, self.sr // 2) / 700)
            mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)
            bin_points = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)

            fb = np.zeros((self.n_mels, len(f)))
            for i in range(self.n_mels):
                for j in range(bin_points[i], min(bin_points[i + 1], len(f))):
                    fb[i, j] = (j - bin_points[i]) / max(bin_points[i + 1] - bin_points[i], 1)
                for j in range(bin_points[i + 1], min(bin_points[i + 2], len(f))):
                    fb[i, j] = (bin_points[i + 2] - j) / max(bin_points[i + 2] - bin_points[i + 1], 1)

            mel_spec = np.dot(fb, power)
            mel_spec_db = 10 * np.log10(mel_spec + 1e-10)
            mel_spec_db -= np.max(mel_spec_db)

        return mel_spec_db

    @staticmethod
    def _normalize_0_1(x: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return x

    def create_hybrid_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Tạo Hybrid Spectrogram 3-channel: (Gammatone, Mel, Average) → 224x224.

        Returns:
            hybrid: np.ndarray, shape (3, 224, 224), dtype float32, range [0, 1]
        """
        gamma = self.compute_gammatonegram(audio)
        mel = self.compute_mel_spectrogram(audio)

        # Normalize mỗi channel về [0, 1]
        gamma = self._normalize_0_1(gamma)
        mel = self._normalize_0_1(mel)

        # Resize cả hai về output_size x output_size
        gamma_resized = scipy_zoom(
            gamma,
            (self.output_size / gamma.shape[0], self.output_size / gamma.shape[1]),
            order=1
        )
        mel_resized = scipy_zoom(
            mel,
            (self.output_size / mel.shape[0], self.output_size / mel.shape[1]),
            order=1
        )

        gamma_resized = np.clip(gamma_resized[:self.output_size, :self.output_size], 0, 1)
        mel_resized = np.clip(mel_resized[:self.output_size, :self.output_size], 0, 1)

        # Channel 2: Average — smoothing effect
        avg_channel = (gamma_resized + mel_resized) / 2.0

        # Stack: (3, 224, 224) — CHW format
        hybrid = np.stack([gamma_resized, mel_resized, avg_channel], axis=0)
        return hybrid.astype(np.float32)

    def save_spectrogram(self, audio: np.ndarray, save_path: str) -> bool:
        """
        Tạo và lưu ảnh Hybrid Spectrogram 3-channel.
        Lưu dưới dạng PNG RGB 224x224 (không có trục/lề).

        Args:
            audio: Tín hiệu đã tiền xử lý
            save_path: Đường dẫn lưu file ảnh

        Returns:
            True nếu thành công
        """
        cv2 = _get_cv2()
        try:
            hybrid = self.create_hybrid_spectrogram(audio)  # (3, 224, 224)

            # Chuyển CHW → HWC và scale [0, 255]
            image = (np.transpose(hybrid, (1, 2, 0)) * 255).astype(np.uint8)

            # OpenCV dùng BGR, hybrid channels: (Gammatone, Mel, Average) → BGR
            cv2.imwrite(save_path, image)
            return True
        except Exception as e:
            print(f"    [ERROR] Lưu spectrogram thất bại: {e}")
            return False


# ==============================================================================
# MODULE 4: DATA SCANNER — Quét và tổ chức dataset
# ==============================================================================
class DataScanner:
    """Quét thư mục audio và tạo danh sách file kèm label."""

    def __init__(self, data_dir: str, labels_csv: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.labels_csv = labels_csv

    def scan_by_folder_structure(self) -> List[dict]:
        """Quét files theo cấu trúc thư mục: audio/{COPD,Healthy,Non-COPD}/*.wav"""
        samples = []
        for class_name in CLASS_NAMES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"  [WARN] Không tìm thấy thư mục: {class_dir}")
                continue
            wav_files = sorted(class_dir.glob('*.wav'))
            for wav_file in wav_files:
                samples.append({
                    'filepath': str(wav_file),
                    'filename': wav_file.name,
                    'label': class_name,
                })
        return samples

    def scan_by_csv(self) -> List[dict]:
        """Quét files dựa trên labels.csv."""
        import pandas as pd
        df = pd.read_csv(self.labels_csv)
        samples = []
        for _, row in df.iterrows():
            fname = row['filename']
            label = row['label']
            filepath = self.data_dir / label / fname
            if not filepath.exists():
                filepath = self.data_dir / fname
            if not filepath.exists():
                continue
            samples.append({
                'filepath': str(filepath),
                'filename': fname,
                'label': label,
            })
        return samples

    def scan(self) -> List[dict]:
        """Auto-detect: dùng CSV nếu có, ngược lại quét thư mục."""
        if self.labels_csv and os.path.exists(self.labels_csv):
            print(f"  📋 Sử dụng labels từ CSV: {self.labels_csv}")
            return self.scan_by_csv()
        else:
            print(f"  📁 Quét theo cấu trúc thư mục: {self.data_dir}")
            return self.scan_by_folder_structure()


# ==============================================================================
# PIPELINE CHÍNH (Main Processing Pipeline)
# ==============================================================================
class Layer1Pipeline:
    """
    Pipeline tổng hợp Layer 1:
      1. Quét dataset
      2. Tiền xử lý tín hiệu
      3. Trích xuất đặc trưng → features.csv
      4. Tạo Hybrid Spectrogram → spectrograms/ (3-channel 224x224)
    """

    def __init__(self, data_dir: str, output_dir: str,
                 labels_csv: Optional[str] = None,
                 skip_spectrograms: bool = False,
                 max_samples: Optional[int] = None):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.labels_csv = labels_csv
        self.skip_spectrograms = skip_spectrograms
        self.max_samples = max_samples

        # Khởi tạo các module
        self.preprocessor = SignalPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.hybrid_generator = HybridSpectrogramGenerator()

        # Tạo thư mục output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not skip_spectrograms:
            for cls in CLASS_NAMES:
                (self.output_dir / 'spectrograms' / cls).mkdir(parents=True, exist_ok=True)

    def run(self):
        """Chạy toàn bộ pipeline."""
        print("=" * 70)
        print("  LAYER 1: Signal Preprocessing + Feature Extraction (V2 Hybrid)")
        print("=" * 70)
        print(f"  Data Dir      : {self.data_dir}")
        print(f"  Output Dir    : {self.output_dir}")
        print(f"  Target SR     : {TARGET_SR} Hz")
        print(f"  BPF           : {BPF_LOW}-{BPF_HIGH} Hz (Butterworth order {BPF_ORDER})")
        print(f"  Audio Length  : {FIXED_DURATION}s ({FIXED_SAMPLES} samples)")
        print(f"  MFCC          : {N_MFCC} coefficients (+ Delta)")
        print(f"  Spectrogram   : HYBRID (Gammatone + Mel + Average)")
        print(f"  Gammatone     : {N_GAMMATONE} filters")
        print(f"  Mel Filters   : {N_MELS}")
        print(f"  Output Size   : {SPEC_SIZE}x{SPEC_SIZE} pixels, 3-channel")
        print(f"  Spectrograms  : {'SKIP' if self.skip_spectrograms else 'YES'}")
        print("=" * 70)

        # --- Step 1: Quét dataset ---
        print("\n[1/3] 📂 Quét dataset...")
        scanner = DataScanner(self.data_dir, self.labels_csv)
        samples = scanner.scan()

        if self.max_samples:
            samples = samples[:self.max_samples]

        total = len(samples)
        print(f"  → Tìm thấy {total} file audio")

        class_counts = {}
        for s in samples:
            cls = s['label']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        for cls, count in sorted(class_counts.items()):
            print(f"    {cls}: {count} samples ({count/total*100:.1f}%)")

        # --- Step 2: Xử lý từng file ---
        print(f"\n[2/3] ⚙️  Tiền xử lý + Trích xuất đặc trưng...")
        feature_names = FeatureExtractor.get_feature_names()
        all_rows = []

        errors = 0
        spec_count = 0
        t_start = time.time()

        for idx, sample in enumerate(samples):
            filepath = sample['filepath']
            filename = sample['filename']
            label = sample['label']
            stem = Path(filename).stem

            # Progress
            if (idx + 1) % 50 == 0 or idx == 0 or idx == total - 1:
                elapsed = time.time() - t_start
                speed = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total - idx - 1) / speed if speed > 0 else 0
                print(f"  [{idx+1:4d}/{total}] "
                      f"{speed:.1f} files/s | ETA: {eta:.0f}s | "
                      f"Errors: {errors} | "
                      f"Processing: {filename[:50]}...")

            try:
                # Tiền xử lý
                audio = self.preprocessor.process(filepath)

                # Trích xuất đặc trưng (Layer 2)
                features = self.feature_extractor.extract(audio)

                # Tạo row cho CSV
                row = {
                    'filename': filename,
                    'label': label,
                }
                row.update(features)
                all_rows.append(row)

                # Tạo Hybrid Spectrogram (Layer 3)
                if not self.skip_spectrograms:
                    spec_path = str(
                        self.output_dir / 'spectrograms' / label / f"{stem}.png"
                    )
                    if self.hybrid_generator.save_spectrogram(audio, spec_path):
                        spec_count += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    [ERROR] {filename}: {e}")
                elif errors == 6:
                    print(f"    [ERROR] ... (suppressing further error messages)")

        elapsed_total = time.time() - t_start
        print(f"\n  ✅ Xử lý hoàn tất trong {elapsed_total:.1f}s")
        print(f"     Features: {len(all_rows)} samples extracted")
        print(f"     Spectrograms: {spec_count} images generated")
        print(f"     Errors: {errors}")

        # --- Step 3: Lưu features.csv ---
        print(f"\n[3/3] 💾 Lưu kết quả...")
        csv_path = self.output_dir / 'features.csv'
        self._save_csv(all_rows, feature_names, csv_path)
        print(f"  ✅ Features saved: {csv_path}")
        print(f"     Columns: filename, label, + {len(feature_names)} feature columns")
        print(f"     Total: {len(all_rows)} rows")

        if not self.skip_spectrograms:
            spec_dir = self.output_dir / 'spectrograms'
            print(f"  ✅ Spectrograms saved: {spec_dir}/")
            for cls in CLASS_NAMES:
                cls_dir = spec_dir / cls
                if cls_dir.exists():
                    count = len(list(cls_dir.glob('*.png')))
                    print(f"     {cls}: {count} images")

        # --- Summary ---
        print("\n" + "=" * 70)
        print("  LAYER 1 PIPELINE HOÀN TẤT (V2 Hybrid Spectrogram)")
        print("=" * 70)
        print(f"  📊 features.csv      → Input cho Layer 2 (Random Forest)")
        print(f"  🖼️  spectrograms/     → Input cho Layer 3 (CNN/DPU FPGA)")
        print(f"     Format: 3-channel PNG (Gammatone + Mel + Average) 224x224")
        print(f"  ⏱️  Thời gian:        {elapsed_total:.1f}s "
              f"({elapsed_total/60:.1f} phút)")
        print("=" * 70)

    def _save_csv(self, rows: List[dict], feature_names: List[str],
                  csv_path: Path):
        """Lưu features vào CSV file."""
        header = ['filename', 'label'] + feature_names
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in rows:
                clean_row = {k: row.get(k, 0.0) for k in header}
                writer.writerow(clean_row)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Layer 1: Signal Preprocessing + Hybrid Spectrogram Extraction"
    )
    parser.add_argument(
        '--data_dir', type=str,
        default=str(COMBINED_AUDIO_DIR),
        help='Thư mục chứa audio files (cấu trúc: audio/{COPD,Healthy,Non-COPD}/)'
    )
    parser.add_argument(
        '--labels_csv', type=str,
        default=str(COMBINED_LABELS),
        help='File labels.csv'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=str(LAYER13_ARTIFACTS_DIR / 'layer1'),
        help='Thư mục output cho features.csv và spectrograms/'
    )
    parser.add_argument(
        '--skip_spectrograms', action='store_true',
        help='Bỏ qua tạo spectrogram (chỉ trích xuất features.csv)'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Giới hạn số mẫu xử lý (để test nhanh)'
    )

    args = parser.parse_args()

    pipeline = Layer1Pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        labels_csv=args.labels_csv,
        skip_spectrograms=args.skip_spectrograms,
        max_samples=args.max_samples,
    )
    pipeline.run()


if __name__ == '__main__':
    main()
