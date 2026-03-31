#!/usr/bin/env python3
"""
================================================================================
Calibration Dataset Generator for Vitis AI Quantization
================================================================================
Tạo calibration dataset từ file WAV (ICBHI + Combined) để sử dụng trong
quá trình quantization INT8 với pytorch_nndct.

Pipeline giống hệt distillation_02.py:
  1. Resample to 4kHz
  2. Bandpass filter 25-2000Hz
  3. Normalize, pad/crop to 8s
  4. Create 3-channel hybrid spectrogram (Gammatonegram + Mel + Average)
  5. Save as .npy files (float32 precision)

Usage:
    python generate_calib_data.py
    python generate_calib_data.py --num_per_class 150
    python generate_calib_data.py --output_dir /path/to/output
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import csv
from collections import Counter
from pathlib import Path

from scipy import signal
from scipy.ndimage import zoom

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy")
    sys.exit(1)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ==============================================================================
# CONFIG — Same as distillation_02.py
# ==============================================================================
TARGET_SR = 4000
SEGMENT_DURATION = 8
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_DURATION  # 32000

N_MELS = 128
N_FFT = 512
HOP_LENGTH = 128
N_GAMMATONE = 64
IMG_SIZE = 224

NUM_CLASSES = 3
CLASS_NAMES = ['COPD', 'Healthy', 'Non-COPD']

DISEASE_TO_CLASS = {
    'COPD': 'COPD',
    'Healthy': 'Healthy',
    'Asthma': 'Non-COPD',
    'URTI': 'Non-COPD',
    'LRTI': 'Non-COPD',
    'Bronchiectasis': 'Non-COPD',
    'Bronchiolitis': 'Non-COPD',
    'Pneumonia': 'Non-COPD',
}


# ==============================================================================
# AUDIO PREPROCESSING — Copied from distillation_02.py
# ==============================================================================
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def preprocess_audio(wav_path, target_sr=TARGET_SR, segment_len=SEGMENT_SAMPLES):
    """Load, resample to 4kHz, BPF 25-2000Hz, normalize, pad/crop to 8s."""
    try:
        sr, audio = wavfile.read(wav_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 4kHz
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, max(num_samples, 1))

        # Bandpass filter 25-2000 Hz
        b, a = butter_bandpass(25, min(2000, target_sr // 2 - 1), target_sr, order=3)
        audio = signal.filtfilt(b, a, audio).astype(np.float32)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio)) + 1e-10
        audio = audio / max_val

        # Pad or crop to 8 seconds
        if len(audio) < segment_len:
            repeats = segment_len // len(audio) + 1
            audio = np.tile(audio, repeats)[:segment_len]
        elif len(audio) > segment_len:
            start = (len(audio) - segment_len) // 2
            audio = audio[start:start + segment_len]

        return audio.astype(np.float32)
    except Exception as e:
        print(f"  Warning: Failed to process {wav_path}: {e}")
        return np.zeros(segment_len, dtype=np.float32)


# ==============================================================================
# FEATURE EXTRACTION — Copied from distillation_02.py
# ==============================================================================
def compute_gammatone_filterbank(sr, n_filters=N_GAMMATONE, fmin=50, fmax=2000):
    ear_q = 9.26449
    min_bw = 24.7
    freqs = -(ear_q * min_bw) + np.exp(
        np.arange(1, n_filters + 1) * (
            -np.log(fmax + ear_q * min_bw) + np.log(fmin + ear_q * min_bw)
        ) / n_filters
    ) * (fmax + ear_q * min_bw)
    freqs = np.flip(freqs)
    return freqs


def compute_gammatonegram(audio, sr=TARGET_SR, n_filters=N_GAMMATONE,
                          n_fft=N_FFT, hop_length=HOP_LENGTH):
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


def compute_mel_spectrogram(audio, sr=TARGET_SR, n_mels=N_MELS,
                            n_fft=N_FFT, hop_length=HOP_LENGTH):
    if HAS_LIBROSA:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, fmin=50, fmax=min(2000, sr // 2)
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    else:
        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        power = np.abs(Zxx) ** 2
        mel_min = 2595 * np.log10(1 + 50 / 700)
        mel_max = 2595 * np.log10(1 + min(2000, sr // 2) / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        fb = np.zeros((n_mels, len(f)))
        for i in range(n_mels):
            for j in range(bin_points[i], min(bin_points[i + 1], len(f))):
                fb[i, j] = (j - bin_points[i]) / max(bin_points[i + 1] - bin_points[i], 1)
            for j in range(bin_points[i + 1], min(bin_points[i + 2], len(f))):
                fb[i, j] = (bin_points[i + 2] - j) / max(bin_points[i + 2] - bin_points[i + 1], 1)
        mel_spec = np.dot(fb, power)
        mel_spec_db = 10 * np.log10(mel_spec + 1e-10)
        mel_spec_db -= np.max(mel_spec_db)
    return mel_spec_db


def create_hybrid_spectrogram(audio, sr=TARGET_SR, output_size=IMG_SIZE):
    """Create 3-channel hybrid: Gammatonegram + Mel-spectrogram → 224x224."""
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
    hybrid = np.stack([gamma_resized, mel_resized, avg_channel], axis=0)
    return hybrid.astype(np.float32)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_icbhi_samples(data_dir, labels_path):
    """Load ICBHI samples from WAV directory + labels.txt."""
    patient_labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                pid = int(parts[0])
                disease = parts[1].strip()
                patient_labels[pid] = disease

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    wav_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav')])

    samples = []
    for wav_file in wav_files:
        pid = int(wav_file.split('_')[0])
        if pid not in patient_labels:
            continue
        disease = patient_labels[pid]
        if disease not in DISEASE_TO_CLASS:
            continue
        class_name = DISEASE_TO_CLASS[disease]
        class_idx = class_to_idx[class_name]
        wav_path = os.path.join(data_dir, wav_file)
        samples.append({
            'wav_path': wav_path,
            'class_idx': class_idx,
            'class_name': class_name,
        })

    return samples


def load_combined_samples(combined_dir):
    """Load combined samples from subdirectory structure: COPD/, Healthy/, Non-COPD/."""
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    samples = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(combined_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        wav_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.wav')])
        class_idx = class_to_idx[class_name]
        for wav_file in wav_files:
            wav_path = os.path.join(class_dir, wav_file)
            samples.append({
                'wav_path': wav_path,
                'class_idx': class_idx,
                'class_name': class_name,
            })

    return samples


# ==============================================================================
# MAIN: GENERATE CALIBRATION DATA
# ==============================================================================
def get_project_root():
    """Detect project root: /workspace (Docker) or /home/iec/... (host)."""
    if os.path.isfile('/workspace/CMakeLists.txt') and not os.path.isdir('/workspace/Parallel_Computing_on_FPGA'):
        return '/workspace'
    if os.path.isdir('/workspace/Parallel_Computing_on_FPGA'):
        return '/workspace/Parallel_Computing_on_FPGA'
    return '/home/iec/Parallel_Computing_on_FPGA'


def main():
    project_root = get_project_root()

    parser = argparse.ArgumentParser(
        description='Generate calibration dataset for Vitis AI quantization'
    )
    parser.add_argument('--icbhi_dir', type=str,
                        default=os.path.join(project_root, 'data/samples/ICBHI_final_database'),
                        help='Path to ICBHI WAV directory')
    parser.add_argument('--icbhi_labels', type=str,
                        default=os.path.join(project_root, 'data/samples/labels.txt'),
                        help='Path to ICBHI labels.txt')
    parser.add_argument('--combined_dir', type=str,
                        default=os.path.join(project_root, 'data/combined/audio'),
                        help='Path to combined audio directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(project_root, 'data/calib_data'),
                        help='Output directory for calibration data')
    parser.add_argument('--num_per_class', type=int, default=100,
                        help='Number of samples per class (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print("  CALIBRATION DATA GENERATOR")
    print("=" * 70)
    print(f"  ICBHI dir:     {args.icbhi_dir}")
    print(f"  Combined dir:  {args.combined_dir}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Samples/class: {args.num_per_class}")
    print("=" * 70)

    # ----- Load all samples -----
    all_samples = []

    if os.path.isdir(args.icbhi_dir) and os.path.isfile(args.icbhi_labels):
        icbhi = load_icbhi_samples(args.icbhi_dir, args.icbhi_labels)
        print(f"  ICBHI samples: {len(icbhi)}")
        all_samples.extend(icbhi)
    else:
        print(f"  ICBHI dir not found, skipping")

    if os.path.isdir(args.combined_dir):
        combined = load_combined_samples(args.combined_dir)
        print(f"  Combined samples: {len(combined)}")
        all_samples.extend(combined)
    else:
        print(f"  Combined dir not found, skipping")

    if not all_samples:
        print("ERROR: No samples found!")
        sys.exit(1)

    # ----- Deduplicate by basename -----
    seen = set()
    unique_samples = []
    for s in all_samples:
        basename = os.path.basename(s['wav_path'])
        if basename not in seen:
            seen.add(basename)
            unique_samples.append(s)

    print(f"  Total unique samples: {len(unique_samples)}")
    dist = Counter([s['class_name'] for s in unique_samples])
    print(f"  Distribution: {dict(dist)}")

    # ----- Balanced sampling -----
    selected = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        cls_samples = [s for s in unique_samples if s['class_idx'] == class_idx]
        np.random.shuffle(cls_samples)
        n_select = min(args.num_per_class, len(cls_samples))
        selected.extend(cls_samples[:n_select])
        print(f"  Selected {n_select} samples for class '{class_name}'")

    np.random.shuffle(selected)
    print(f"  Total calibration samples: {len(selected)}")

    # ----- Generate spectrograms -----
    os.makedirs(args.output_dir, exist_ok=True)
    labels_file = os.path.join(args.output_dir, 'calib_labels.txt')

    with open(labels_file, 'w') as lf:
        lf.write("# filename\tclass_idx\tclass_name\n")

        for i, s in enumerate(selected):
            wav_name = os.path.splitext(os.path.basename(s['wav_path']))[0]
            npy_name = f"calib_{i:04d}_{wav_name}.npy"
            npy_path = os.path.join(args.output_dir, npy_name)

            # Process audio → spectrogram
            audio = preprocess_audio(s['wav_path'])
            spec = create_hybrid_spectrogram(audio)  # shape: (3, 224, 224), float32 [0,1]

            np.save(npy_path, spec)
            lf.write(f"{npy_name}\t{s['class_idx']}\t{s['class_name']}\n")

            if (i + 1) % 50 == 0 or (i + 1) == len(selected):
                print(f"  Processed {i + 1}/{len(selected)} samples...")

    print("=" * 70)
    print(f"  DONE! Calibration data saved to: {args.output_dir}")
    print(f"  Total files: {len(selected)}")
    print(f"  Labels file: {labels_file}")
    print(f"  Each file shape: (3, 224, 224) float32")
    print("=" * 70)


if __name__ == '__main__':
    main()
