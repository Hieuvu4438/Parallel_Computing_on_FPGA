#!/usr/bin/env python3
"""
================================================================================
RESPIRATORY SOUND CLASSIFICATION - CNN TRAINING SCRIPT V3
================================================================================

Mục tiêu: Đạt >90% accuracy trên tập Test với 4-class classification
(Normal, Crackle, Wheeze, Both)

Khắc phục các vấn đề:
1. Test Accuracy thấp (23%) -> Tối ưu wavelet spectrogram
2. Normal Recall ~0% -> Class weighting và balanced sampling  
3. Overfitting -> Data augmentation, dropout, early stopping

Tác giả: Research Team
Ngày: 2026

================================================================================
GIẢI THÍCH CÁC THAM SỐ QUAN TRỌNG
================================================================================

EARLY STOPPING:
- patience=15: Số epochs chờ đợi khi val_f1 không cải thiện
- min_delta=0.001: Ngưỡng tối thiểu để coi là "cải thiện"
- Tác dụng: Ngăn overfitting bằng cách dừng training khi model bắt đầu overfit

CLASS WEIGHTS:
- Tính theo công thức: weight[i] = total_samples / (num_classes * count[i])
- Normal (52.8%): weight thấp hơn để model không bias về class này
- Both (7.3%): weight cao hơn để model học tốt class thiểu số
- Ví dụ với 6898 samples:
  - Normal: 6898 / (4 * 3642) = 0.47
  - Crackle: 6898 / (4 * 1864) = 0.93
  - Wheeze: 6898 / (4 * 886) = 1.95
  - Both: 6898 / (4 * 506) = 3.41

================================================================================
"""

import os
import sys
import argparse
import copy
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import Counter
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import pywt
from scipy import signal as scipy_signal
from scipy.io import wavfile
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Cấu hình training - dễ điều chỉnh và tái sử dụng."""
    
    # Data
    img_size: int = 224
    target_sr: int = 4000
    num_classes: int = 4
    
    # Wavelet Transform - Tối ưu cho respiratory sounds
    wavelet_name: str = 'morl'  # Morlet wavelet
    num_scales: int = 128       # Số scales cho CWT
    freq_min: float = 50.0      # Hz - tần số thấp nhất quan tâm
    freq_max: float = 2000.0    # Hz - tần số cao nhất (wheezes lên tới 1600Hz)
    
    # Training
    epochs: int = 150
    phase1_epochs: int = 20     # Epochs cho phase 1 (frozen backbone)
    batch_size: int = 32        # Giảm để ổn định hơn
    
    # Learning rates
    lr_phase1: float = 1e-3     # LR cho classifier head
    lr_phase2: float = 1e-5     # LR rất thấp cho fine-tuning
    
    # Regularization
    dropout: float = 0.5
    weight_decay: float = 0.01
    
    # Early Stopping
    patience: int = 15          # Số epochs chờ khi không cải thiện
    min_delta: float = 0.001    # Ngưỡng cải thiện tối thiểu
    
    # Data Augmentation
    aug_time_shift: float = 0.15    # Max time shift ratio
    aug_noise_level: float = 0.005  # White noise amplitude
    aug_gain_range: Tuple[float, float] = (0.8, 1.2)  # Random gain range
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# Global config
CFG = Config()

# Class names
CLASS_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']

print("="*70)
print("RESPIRATORY CNN TRAINING V3")
print("="*70)
print(f"Device: {CFG.device}")
if CFG.device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("="*70)


# ==============================================================================
# WAVELET SPECTROGRAM GENERATOR - TỐI ƯU CHO RESPIRATORY SOUNDS
# ==============================================================================

class OptimizedWaveletTransform:
    """
    Continuous Wavelet Transform với Morlet wavelet.
    
    Tối ưu cho tín hiệu hô hấp:
    - Crackles: Transient ngắn (10-20ms), tần số 100-2000Hz
    - Wheezes: Liên tục (>100ms), tần số 100-1600Hz, có tính tuần hoàn
    - Normal: Không có các đặc trưng bệnh lý trên
    
    Morlet wavelet được chọn vì:
    1. Có độ phân giải thời gian-tần số tốt
    2. Phù hợp với tín hiệu non-stationary như breathing sounds
    3. Có thể điều chỉnh bandwidth thông qua center frequency
    """
    
    def __init__(
        self,
        wavelet: str = CFG.wavelet_name,
        num_scales: int = CFG.num_scales,
        sample_rate: int = CFG.target_sr,
        freq_range: Tuple[float, float] = (CFG.freq_min, CFG.freq_max),
        output_size: int = CFG.img_size
    ):
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.sample_rate = sample_rate
        self.freq_min, self.freq_max = freq_range
        self.output_size = output_size
        
        # Tính scales tương ứng với dải tần số mong muốn
        self.scales = self._compute_optimal_scales()
        
        print(f"[WaveletTransform] Initialized:")
        print(f"  - Wavelet: {wavelet}")
        print(f"  - Scales: {num_scales} ({self.scales[0]:.2f} to {self.scales[-1]:.2f})")
        print(f"  - Frequency range: {self.freq_min}-{self.freq_max} Hz")
    
    def _compute_optimal_scales(self) -> np.ndarray:
        """
        Tính toán scales tối ưu cho dải tần số mong muốn.
        
        Công thức: scale = (center_freq * sample_rate) / frequency
        
        Với Morlet wavelet, center_frequency ≈ 0.8125
        """
        # Lấy center frequency của wavelet
        try:
            center_freq = pywt.central_frequency(self.wavelet)
        except:
            center_freq = 0.8125  # Default cho Morlet
        
        # Tính scale tương ứng với tần số min và max
        scale_max = center_freq * self.sample_rate / self.freq_min  # Low freq -> high scale
        scale_min = center_freq * self.sample_rate / self.freq_max  # High freq -> low scale
        
        # Tạo scales theo thang logarithmic (tốt hơn cho audio)
        scales = np.geomspace(scale_min, scale_max, self.num_scales)
        
        return scales
    
    def generate_spectrogram(
        self, 
        audio: np.ndarray,
        normalize: bool = True,
        enhance_contrast: bool = True
    ) -> np.ndarray:
        """
        Tạo spectrogram từ tín hiệu audio.
        
        Args:
            audio: 1D numpy array, tín hiệu âm thanh
            normalize: Chuẩn hóa về [0, 1]
            enhance_contrast: Tăng cường độ tương phản
            
        Returns:
            2D numpy array (output_size x output_size)
        """
        # Đảm bảo audio có đủ độ dài
        if len(audio) < 100:
            audio = np.pad(audio, (0, 100 - len(audio)), mode='constant')
        
        # Áp dụng CWT
        coefficients, frequencies = pywt.cwt(
            audio,
            self.scales,
            self.wavelet,
            sampling_period=1.0/self.sample_rate
        )
        
        # Tính power spectrogram (magnitude squared)
        power = np.abs(coefficients) ** 2
        
        # Chuyển sang thang dB với floor để tránh -inf
        power_db = 10 * np.log10(power + 1e-10)
        
        # Clip các giá trị quá thấp (noise floor)
        noise_floor = power_db.max() - 60  # 60dB dynamic range
        power_db = np.clip(power_db, noise_floor, None)
        
        # Normalize to [0, 1]
        if normalize:
            power_db = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)
        
        # Enhance contrast để làm rõ các đặc trưng
        if enhance_contrast:
            # Histogram equalization đơn giản
            power_db = np.power(power_db, 0.8)  # Gamma correction
        
        # Resize về kích thước output
        zoom_factors = (
            self.output_size / power_db.shape[0],
            self.output_size / power_db.shape[1]
        )
        spectrogram = zoom(power_db, zoom_factors, order=1)  # Bilinear interpolation
        
        # Đảm bảo kích thước chính xác
        spectrogram = spectrogram[:self.output_size, :self.output_size]
        
        # Clip final values
        spectrogram = np.clip(spectrogram, 0, 1)
        
        return spectrogram.astype(np.float32)


# ==============================================================================
# DATA AUGMENTATION - TỐI ƯU CHO ÂM THANH Y TẾ
# ==============================================================================

class AudioAugmenter:
    """
    Data augmentation cho tín hiệu âm thanh hô hấp.
    
    Các kỹ thuật được sử dụng:
    1. Time Shifting: Dịch chuyển tín hiệu theo thời gian
    2. White Noise: Thêm nhiễu Gaussian cường độ thấp
    3. Random Gain: Thay đổi biên độ ngẫu nhiên
    4. Time Stretching: Nén/giãn tín hiệu (mô phỏng nhịp thở khác nhau)
    
    Lưu ý: Các augmentation phải giữ nguyên tính chất bệnh lý
    (crackle/wheeze vẫn phải phát hiện được sau augmentation)
    """
    
    def __init__(
        self,
        time_shift_ratio: float = CFG.aug_time_shift,
        noise_level: float = CFG.aug_noise_level,
        gain_range: Tuple[float, float] = CFG.aug_gain_range,
        probability: float = 0.7  # Xác suất apply augmentation
    ):
        self.time_shift_ratio = time_shift_ratio
        self.noise_level = noise_level
        self.gain_min, self.gain_max = gain_range
        self.probability = probability
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline."""
        if np.random.random() > self.probability:
            return audio
        
        # 1. Time Shifting
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-self.time_shift_ratio, self.time_shift_ratio) * len(audio))
            audio = np.roll(audio, shift)
        
        # 2. Random Gain
        if np.random.random() < 0.6:
            gain = np.random.uniform(self.gain_min, self.gain_max)
            audio = audio * gain
        
        # 3. White Noise
        if np.random.random() < 0.4:
            noise = np.random.normal(0, self.noise_level, len(audio))
            audio = audio + noise
        
        # 4. Time Stretching (light)
        if np.random.random() < 0.3:
            stretch_factor = np.random.uniform(0.95, 1.05)
            new_length = int(len(audio) * stretch_factor)
            audio = scipy_signal.resample(audio, new_length)
        
        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio


class SpectrogramAugmenter:
    """
    SpecAugment-style augmentation cho spectrogram.
    
    Các kỹ thuật:
    1. Frequency Masking: Che một số dải tần số
    2. Time Masking: Che một số khoảng thời gian
    
    Tham khảo: "SpecAugment: A Simple Data Augmentation Method for ASR"
    """
    
    def __init__(
        self,
        freq_mask_param: int = 20,   # Max width of freq mask
        time_mask_param: int = 30,   # Max width of time mask
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        probability: float = 0.5
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.probability = probability
    
    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """Apply SpecAugment."""
        if np.random.random() > self.probability:
            return spec
        
        spec = spec.copy()
        h, w = spec.shape
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, h // 4))
            f0 = np.random.randint(0, h - f)
            spec[f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, w // 4))
            t0 = np.random.randint(0, w - t)
            spec[:, t0:t0+t] = 0
        
        return spec


# ==============================================================================
# DATASET
# ==============================================================================

def parse_icbhi_annotation(txt_path: str) -> List[Dict]:
    """
    Parse ICBHI annotation file.
    
    Format: [start_time] [end_time] [crackles] [wheezes]
    
    Labels:
    - Normal (0): crackles=0, wheezes=0
    - Crackle (1): crackles=1, wheezes=0
    - Wheeze (2): crackles=0, wheezes=1
    - Both (3): crackles=1, wheezes=1
    """
    cycles = []
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    start = float(parts[0])
                    end = float(parts[1])
                    crackle = int(parts[2])
                    wheeze = int(parts[3])
                    
                    # Determine label
                    if crackle and wheeze:
                        label = 3  # Both
                    elif crackle:
                        label = 1  # Crackle
                    elif wheeze:
                        label = 2  # Wheeze
                    else:
                        label = 0  # Normal
                    
                    cycles.append({
                        'start': start,
                        'end': end,
                        'label': label,
                        'crackle': crackle,
                        'wheeze': wheeze
                    })
    except Exception as e:
        print(f"Error parsing {txt_path}: {e}")
    
    return cycles


def parse_patient_id(filename: str) -> int:
    """Extract patient ID from ICBHI filename."""
    basename = Path(filename).stem
    parts = basename.split('_')
    try:
        return int(parts[0])
    except:
        return hash(basename) % 10000


def load_icbhi_dataset(data_path: str) -> Tuple[List[Dict], List[int], List[int]]:
    """
    Load ICBHI 2017 dataset.
    
    Returns:
        cycles: List of cycle info dicts
        labels: List of class labels
        patient_ids: List of patient IDs
    """
    data_path = Path(data_path)
    wav_files = sorted(data_path.glob("*.wav"))
    
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {data_path}")
    
    print(f"Found {len(wav_files)} audio files")
    
    cycles = []
    labels = []
    patient_ids = []
    
    for wav_file in tqdm(wav_files, desc="Loading dataset"):
        # Find annotation file
        txt_file = wav_file.with_suffix('.txt')
        if not txt_file.exists():
            continue
        
        # Parse patient ID and annotations
        patient_id = parse_patient_id(wav_file.name)
        annotations = parse_icbhi_annotation(str(txt_file))
        
        for ann in annotations:
            cycles.append({
                'wav_path': str(wav_file),
                'start': ann['start'],
                'end': ann['end']
            })
            labels.append(ann['label'])
            patient_ids.append(patient_id)
    
    # Print statistics
    print(f"\nTotal breathing cycles: {len(cycles)}")
    
    class_counts = Counter(labels)
    print("\nClass distribution:")
    print("-" * 40)
    for i, name in enumerate(CLASS_NAMES):
        count = class_counts.get(i, 0)
        pct = count / len(labels) * 100
        print(f"  {name}: {count:5d} ({pct:5.1f}%)")
    print("-" * 40)
    
    return cycles, labels, patient_ids


class RespiratoryDataset(Dataset):
    """
    Dataset cho ICBHI breathing cycles.
    
    Features:
    - Lazy loading của audio files
    - Wavelet spectrogram generation
    - Configurable augmentation
    """
    
    def __init__(
        self,
        cycles: List[Dict],
        labels: List[int],
        wavelet_transform: OptimizedWaveletTransform,
        audio_augmenter: Optional[AudioAugmenter] = None,
        spec_augmenter: Optional[SpectrogramAugmenter] = None,
        transform: Optional[Callable] = None
    ):
        self.cycles = cycles
        self.labels = labels
        self.wavelet = wavelet_transform
        self.audio_aug = audio_augmenter
        self.spec_aug = spec_augmenter
        self.transform = transform
        
        # Audio cache to avoid repeated disk reads
        self._audio_cache = {}
        self._max_cache_size = 200  # Limit cache size
    
    def __len__(self) -> int:
        return len(self.cycles)
    
    def _load_audio(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """Load and cache audio file."""
        if wav_path not in self._audio_cache:
            # Clear cache if too large
            if len(self._audio_cache) >= self._max_cache_size:
                self._audio_cache.clear()
            
            sr, audio = wavfile.read(wav_path)
            
            # Convert to float [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
            
            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            self._audio_cache[wav_path] = (audio, sr)
        
        return self._audio_cache[wav_path]
    
    def _extract_cycle(
        self, 
        audio: np.ndarray, 
        sr: int, 
        start: float, 
        end: float
    ) -> np.ndarray:
        """Extract breathing cycle from audio."""
        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio), int(end * sr))
        
        cycle = audio[start_sample:end_sample]
        
        # Resample to target SR if needed
        if sr != CFG.target_sr:
            num_samples = int(len(cycle) * CFG.target_sr / sr)
            cycle = scipy_signal.resample(cycle, num_samples)
        
        return cycle
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cycle_info = self.cycles[idx]
        label = self.labels[idx]
        
        # Load audio
        audio, sr = self._load_audio(cycle_info['wav_path'])
        
        # Extract cycle
        cycle = self._extract_cycle(audio, sr, cycle_info['start'], cycle_info['end'])
        
        # Apply audio augmentation
        if self.audio_aug is not None:
            cycle = self.audio_aug(cycle)
        
        # Generate spectrogram
        spectrogram = self.wavelet.generate_spectrogram(cycle)
        
        # Apply spectrogram augmentation
        if self.spec_aug is not None:
            spectrogram = self.spec_aug(spectrogram)
        
        # Convert to 3-channel tensor (repeat grayscale for pretrained model)
        spectrogram = np.stack([spectrogram, spectrogram, spectrogram], axis=0)
        tensor = torch.from_numpy(spectrogram)
        
        # Apply normalization transform
        if self.transform is not None:
            tensor = self.transform(tensor)
        
        return tensor, label


# ==============================================================================
# SUBJECT-INDEPENDENT SPLIT
# ==============================================================================

def subject_independent_split(
    cycles: List[Dict],
    labels: List[int],
    patient_ids: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Chia dữ liệu theo Patient ID để đảm bảo tính khách quan lâm sàng.
    
    QUAN TRỌNG: Một bệnh nhân KHÔNG BAO GIỜ xuất hiện ở cả train và test.
    Điều này ngăn model học đặc điểm riêng của từng bệnh nhân thay vì
    đặc điểm bệnh lý chung.
    """
    np.random.seed(random_seed)
    
    # Get unique patients
    unique_patients = np.array(list(set(patient_ids)))
    np.random.shuffle(unique_patients)
    
    # Split patients
    n_patients = len(unique_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])
    
    # Split indices
    train_idx = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_idx = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
    test_idx = [i for i, pid in enumerate(patient_ids) if pid in test_patients]
    
    print(f"\nSubject-Independent Split:")
    print(f"  Train: {len(train_idx):5d} cycles from {len(train_patients):3d} patients")
    print(f"  Val:   {len(val_idx):5d} cycles from {len(val_patients):3d} patients")
    print(f"  Test:  {len(test_idx):5d} cycles from {len(test_patients):3d} patients")
    
    # Verify no overlap
    assert len(train_patients & val_patients) == 0, "Patient overlap between train and val!"
    assert len(train_patients & test_patients) == 0, "Patient overlap between train and test!"
    assert len(val_patients & test_patients) == 0, "Patient overlap between val and test!"
    print("  ✓ No patient overlap verified")
    
    return train_idx, val_idx, test_idx


# ==============================================================================
# CLASS WEIGHTING
# ==============================================================================

def compute_class_weights(labels: List[int], num_classes: int = 4) -> torch.Tensor:
    """
    Tính class weights để xử lý class imbalance.
    
    Công thức: weight[i] = total_samples / (num_classes * count[i])
    
    Ý nghĩa:
    - Class có ít samples -> weight cao -> loss contribution cao
    - Class có nhiều samples -> weight thấp -> tránh model bias
    
    Ví dụ với ICBHI:
    - Normal (52.8%): weight = 0.47 (thấp nhất)
    - Both (7.3%): weight = 3.41 (cao nhất)
    """
    class_counts = Counter(labels)
    total = len(labels)
    
    weights = []
    print("\nClass Weights:")
    print("-" * 50)
    
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
        
        pct = count / total * 100
        print(f"  {CLASS_NAMES[i]:10s}: count={count:5d} ({pct:5.1f}%) -> weight={weight:.3f}")
    
    print("-" * 50)
    
    return torch.FloatTensor(weights)


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Tạo WeightedRandomSampler để balanced sampling trong training.
    
    Mỗi sample được gán weight = 1/count[class]
    Kết quả: Mỗi batch có phân phối class cân bằng hơn
    """
    class_counts = Counter(labels)
    
    # Weight for each sample = 1 / count of its class
    sample_weights = [1.0 / class_counts[label] for label in labels]
    sample_weights = torch.FloatTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Cho phép lấy lại sample (cần cho minority classes)
    )
    
    return sampler


# ==============================================================================
# MODEL
# ==============================================================================

class RespiratoryClassifier(nn.Module):
    """
    MobileNetV2-based classifier cho respiratory sounds.
    
    Architecture:
    - Backbone: MobileNetV2 pretrained trên ImageNet
    - Classifier: Custom head với BatchNorm và Dropout
    
    Transfer Learning Strategy:
    - Phase 1: Freeze backbone, train classifier
    - Phase 2: Unfreeze all, fine-tune với low LR
    """
    
    def __init__(
        self,
        num_classes: int = CFG.num_classes,
        dropout: float = CFG.dropout,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features  # 1280
        
        # Custom classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.4),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights properly."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Đóng băng backbone cho Phase 1 training."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen (classifier training only)")
    
    def unfreeze_backbone(self):
        """Giải băng toàn bộ cho Phase 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen (full fine-tuning)")
    
    def get_trainable_params(self) -> int:
        """Đếm số parameters có thể train."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """
    Early Stopping để ngăn overfitting.
    
    Parameters:
    - patience: Số epochs chờ khi metric không cải thiện
    - min_delta: Ngưỡng tối thiểu để coi là "cải thiện"
    - mode: 'max' (val_f1, accuracy) hoặc 'min' (val_loss)
    
    Hoạt động:
    1. Theo dõi best score
    2. Nếu score không cải thiện sau `patience` epochs -> stop
    3. Restore model về trạng thái best
    """
    
    def __init__(
        self,
        patience: int = CFG.patience,
        min_delta: float = CFG.min_delta,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_model_state = None
        self.should_stop = False
    
    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        
        # Check if improved
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"  ★ New best: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  ⚠ Early stopping triggered!")
                    print(f"  Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.should_stop
    
    def restore_best_model(self, model: nn.Module):
        """Restore model to best state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"✓ Model restored to epoch {self.best_epoch} (best score: {self.best_score:.4f})")


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler
) -> Tuple[float, float]:
    """Train for one epoch with mixed precision."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward with scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate model on validation/test set."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    epoch_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, accuracy, macro_f1, all_preds, all_labels


def compute_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Compute detailed per-class metrics."""
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': cm,
        'per_class': {}
    }
    
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics['per_class'][name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'support': int((y_true == i).sum())
        }
    
    return metrics


def print_metrics(metrics: Dict):
    """Print formatted metrics."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']*100:6.2f}%")
    print(f"  Macro F1:    {metrics['macro_f1']*100:6.2f}%")
    print(f"  Weighted F1: {metrics['weighted_f1']*100:6.2f}%")
    
    print("\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':12s} {'Sensitivity':>12s} {'Specificity':>12s} {'Precision':>12s} {'F1':>10s} {'Support':>8s}")
    print("-" * 70)
    
    for name in CLASS_NAMES:
        m = metrics['per_class'][name]
        print(f"{name:12s} {m['sensitivity']*100:>11.2f}% {m['specificity']*100:>11.2f}% "
              f"{m['precision']*100:>11.2f}% {m['f1']*100:>9.2f}% {m['support']:>8d}")
    
    print("-" * 70)


def plot_results(history: Dict, cm: np.ndarray, output_dir: Path):
    """Plot training history and confusion matrix."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', label='Val Macro F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation Macro F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix (Normalized)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_results.png', dpi=150)
    plt.close()
    print(f"✓ Results plot saved to {output_dir / 'training_results.png'}")


def export_to_onnx(model: nn.Module, save_path: str):
    """Export model to ONNX format."""
    model.eval()
    model.cpu()
    
    dummy_input = torch.randn(1, 3, CFG.img_size, CFG.img_size)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✓ ONNX model saved to {save_path}")
        
        # Verify
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
        
    except Exception as e:
        print(f"⚠ ONNX export failed: {e}")
        print("Saving as TorchScript instead...")
        
        scripted = torch.jit.trace(model, dummy_input)
        pt_path = save_path.replace('.onnx', '.pt')
        scripted.save(pt_path)
        print(f"✓ TorchScript model saved to {pt_path}")


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def main(args):
    """Main training pipeline."""
    
    start_time = time.time()
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(CFG.device)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("\n" + "="*70)
    print("[1/7] LOADING DATASET")
    print("="*70)
    
    cycles, labels, patient_ids = load_icbhi_dataset(args.data_path)
    
    # =========================================================================
    # 2. SUBJECT-INDEPENDENT SPLIT
    # =========================================================================
    print("\n" + "="*70)
    print("[2/7] SUBJECT-INDEPENDENT SPLIT")
    print("="*70)
    
    train_idx, val_idx, test_idx = subject_independent_split(
        cycles, labels, patient_ids
    )
    
    # =========================================================================
    # 3. COMPUTE CLASS WEIGHTS
    # =========================================================================
    print("\n" + "="*70)
    print("[3/7] COMPUTING CLASS WEIGHTS")
    print("="*70)
    
    train_labels = [labels[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels)
    class_weights = class_weights.to(device)
    
    # =========================================================================
    # 4. CREATE DATA LOADERS
    # =========================================================================
    print("\n" + "="*70)
    print("[4/7] CREATING DATA LOADERS")
    print("="*70)
    
    # Initialize wavelet transform
    wavelet = OptimizedWaveletTransform()
    
    # Augmenters
    audio_aug = AudioAugmenter()
    spec_aug = SpectrogramAugmenter()
    
    # Normalization (ImageNet stats)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Create datasets
    train_dataset = RespiratoryDataset(
        [cycles[i] for i in train_idx],
        [labels[i] for i in train_idx],
        wavelet, audio_aug, spec_aug, normalize
    )
    
    val_dataset = RespiratoryDataset(
        [cycles[i] for i in val_idx],
        [labels[i] for i in val_idx],
        wavelet, None, None, normalize
    )
    
    test_dataset = RespiratoryDataset(
        [cycles[i] for i in test_idx],
        [labels[i] for i in test_idx],
        wavelet, None, None, normalize
    )
    
    # Weighted sampler
    train_sampler = create_weighted_sampler(train_labels)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # =========================================================================
    # 5. CREATE MODEL
    # =========================================================================
    print("\n" + "="*70)
    print("[5/7] CREATING MODEL")
    print("="*70)
    
    model = RespiratoryClassifier(
        num_classes=CFG.num_classes,
        dropout=CFG.dropout,
        pretrained=True
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function với class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # =========================================================================
    # 6. TRAINING
    # =========================================================================
    print("\n" + "="*70)
    print("[6/7] TRAINING")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Phase 1: Train classifier only (backbone frozen)
    # -------------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PHASE 1: Training Classifier (Backbone Frozen)")
    print(f"{'─'*70}")
    
    model.freeze_backbone()
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr_phase1,
        weight_decay=CFG.weight_decay
    )
    
    # Warmup
    warmup_epochs = 3
    
    for epoch in range(CFG.phase1_epochs):
        # Warmup LR
        if epoch < warmup_epochs:
            lr = CFG.lr_phase1 * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1:3d}/{CFG.phase1_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}, F1: {val_f1:.2%}")
    
    # -------------------------------------------------------------------------
    # Phase 2: Fine-tune full model
    # -------------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PHASE 2: Fine-tuning Full Model")
    print(f"{'─'*70}")
    
    model.unfreeze_backbone()
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG.lr_phase2,
        weight_decay=CFG.weight_decay
    )
    
    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Monitor val_f1
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=CFG.patience,
        min_delta=CFG.min_delta,
        mode='max'
    )
    
    for epoch in range(CFG.phase1_epochs, CFG.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{CFG.epochs} (lr={current_lr:.1e}) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}, F1: {val_f1:.2%}")
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Check early stopping
        if early_stopping(val_f1, model, epoch + 1):
            break
    
    # Restore best model
    early_stopping.restore_best_model(model)
    
    # =========================================================================
    # 7. FINAL EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("[7/7] FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    metrics = compute_detailed_metrics(test_labels, test_preds, CLASS_NAMES)
    print_metrics(metrics)
    
    # Save confusion matrix
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Plot results
    plot_results(history, metrics['confusion_matrix'], output_dir)
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'best_model.pth')
    print(f"✓ Model saved to {output_dir / 'best_model.pth'}")
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    
    # Export ONNX
    export_to_onnx(model, str(output_dir / 'respiratory_4class.onnx'))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best Val F1: {early_stopping.best_score:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test Macro F1: {test_f1:.2%}")
    print(f"Output directory: {output_dir}")
    print("="*70)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN for 4-class respiratory sound classification"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="D:/PROJECTS/Parallel_Computing_on_FPGA/data/ICBHI_final_database",
        help="Path to ICBHI dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_v3",
        help="Output directory"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    args = parser.parse_args()
    
    main(args)
