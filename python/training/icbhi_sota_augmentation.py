#!/usr/bin/env python3
"""
ICBHI SOTA Augmentation Strategies

Based on:
- Dong et al. 2025 - RSC-FTF: VTLP-Patch Augmentation (+3.19%)
- Dong et al. 2025 - ADD-RSC: Adaptive denoising augmentation
- Standard SpecAugment extensions

All augmentations work at spectrogram level (compatible with caching).
No additional model parameters needed.
"""

import numpy as np
import torch


def vtlp_frequency_warping(feat, alpha_range=(0.9, 1.1), prob=0.5):
    """
    VTLP (Vocal Tract Length Perturbation) at spectrogram level.

    Simulates vocal tract length variations across patients.
    Applied as frequency-axis warping on Mel spectrogram.

    Paper: Dong et al. 2025 - RSC-FTF
    Ablation: +3.19% ICBHI Score

    Args:
        feat: [n_freq, n_time] Mel spectrogram
        alpha_range: warping factor range
        prob: probability of applying

    Returns:
        augmented spectrogram
    """
    if np.random.random() > prob:
        return feat

    alpha = np.random.uniform(*alpha_range)
    n_freq = feat.shape[0]
    indices = np.clip(np.arange(n_freq) * alpha, 0, n_freq - 1).astype(int)
    return feat[indices]


def time_stretch_spectrogram(feat, rate_range=(0.9, 1.1), prob=0.3):
    """
    Time stretching on spectrogram level.

    Simulates different breathing speeds.
    Applied as time-axis resampling.

    Args:
        feat: [n_freq, n_time] Mel spectrogram
        rate_range: stretch rate range
        prob: probability of applying

    Returns:
        augmented spectrogram (same shape)
    """
    if np.random.random() > prob:
        return feat

    rate = np.random.uniform(*rate_range)
    if abs(rate - 1.0) < 0.01:
        return feat

    n_freq, n_time = feat.shape
    new_n_time = int(n_time * rate)
    if new_n_time < 2:
        return feat

    # Resample along time axis
    indices = np.linspace(0, n_time - 1, new_n_time)
    stretched = np.zeros((n_freq, new_n_time), dtype=feat.dtype)
    for i in range(n_freq):
        stretched[i] = np.interp(indices, np.arange(n_time), feat[i])

    # Crop or pad to original length
    if stretched.shape[1] > n_time:
        stretched = stretched[:, :n_time]
    else:
        pad_width = n_time - stretched.shape[1]
        stretched = np.pad(stretched, ((0, 0), (0, pad_width)), mode='edge')

    return stretched


def mixup_spectrograms(feat1, feat2, alpha=0.3):
    """
    MixUp at spectrogram level.

    Standard MixUp for spectrograms.
    Labels are mixed proportionally.

    Args:
        feat1, feat2: [n_freq, n_time] spectrograms
        alpha: Beta distribution parameter

    Returns:
        mixed spectrogram, mixing coefficient
    """
    lam = np.random.beta(alpha, alpha)
    return lam * feat1 + (1 - lam) * feat2, lam


def cutmix_spectrogram(feat1, feat2, alpha=1.0):
    """
    CutMix at spectrogram level.

    Cuts a rectangular region from one spectrogram and pastes into another.
    More localized augmentation than MixUp.

    Args:
        feat1, feat2: [n_freq, n_time] spectrograms
        alpha: Beta distribution parameter

    Returns:
        mixed spectrogram, mixing coefficient
    """
    n_freq, n_time = feat1.shape
    lam = np.random.beta(alpha, alpha)

    # Cut region size
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(n_freq * cut_ratio)
    cut_w = int(n_time * cut_ratio)

    # Random center
    cx = np.random.randint(n_freq)
    cy = np.random.randint(n_time)

    x1 = max(0, cx - cut_h // 2)
    x2 = min(n_freq, cx + cut_h // 2)
    y1 = max(0, cy - cut_w // 2)
    y2 = min(n_time, cy + cut_w // 2)

    result = feat1.copy()
    result[x1:x2, y1:y2] = feat2[x1:x2, y1:y2]

    # Adjust lambda for actual area
    lam = 1 - (x2 - x1) * (y2 - y1) / (n_freq * n_time)
    return result, lam


def frequency_masking(feat, max_masks=2, max_width=15, prob=0.5):
    """
    Enhanced frequency masking (SpecAugment extension).

    Multiple frequency masks with wider range.

    Args:
        feat: [n_freq, n_time] spectrogram
        max_masks: maximum number of frequency masks
        max_width: maximum mask width
        prob: probability of applying

    Returns:
        augmented spectrogram
    """
    if np.random.random() > prob:
        return feat

    result = feat.copy()
    n_freq = feat.shape[0]
    n_masks = np.random.randint(1, max_masks + 1)

    for _ in range(n_masks):
        width = np.random.randint(1, min(max_width + 1, n_freq))
        start = np.random.randint(0, max(1, n_freq - width))
        result[start:start + width] = result.mean()

    return result


def time_masking(feat, max_masks=2, max_width=30, prob=0.5):
    """
    Enhanced time masking (SpecAugment extension).

    Multiple time masks with wider range.

    Args:
        feat: [n_freq, n_time] spectrogram
        max_masks: maximum number of time masks
        max_width: maximum mask width
        prob: probability of applying

    Returns:
        augmented spectrogram
    """
    if np.random.random() > prob:
        return feat

    result = feat.copy()
    n_time = feat.shape[1]
    n_masks = np.random.randint(1, max_masks + 1)

    for _ in range(n_masks):
        width = np.random.randint(1, min(max_width + 1, n_time))
        start = np.random.randint(0, max(1, n_time - width))
        result[:, start:start + width] = result.mean()

    return result


def gaussian_noise_spectrogram(feat, snr_range=(20, 40), prob=0.3):
    """
    Add Gaussian noise to spectrogram with controlled SNR.

    Args:
        feat: [n_freq, n_time] spectrogram
        snr_range: signal-to-noise ratio range in dB
        prob: probability of applying

    Returns:
        augmented spectrogram
    """
    if np.random.random() > prob:
        return feat

    snr_db = np.random.uniform(*snr_range)
    signal_power = np.mean(feat ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*feat.shape).astype(np.float32) * np.sqrt(noise_power)
    return feat + noise


def apply_all_augmentations(feat, augment_level="medium"):
    """
    Apply a combination of augmentations.

    Levels:
        light: VTLP + mild SpecAugment
        medium: VTLP + time stretch + SpecAugment + noise
        heavy: all augmentations including CutMix

    Args:
        feat: [n_freq, n_time] spectrogram
        augment_level: "light", "medium", or "heavy"

    Returns:
        augmented spectrogram
    """
    result = feat.copy()

    if augment_level == "light":
        result = vtlp_frequency_warping(result, alpha_range=(0.95, 1.05), prob=0.3)
        result = frequency_masking(result, max_masks=1, max_width=10, prob=0.3)
        result = time_masking(result, max_masks=1, max_width=20, prob=0.3)

    elif augment_level == "medium":
        result = vtlp_frequency_warping(result, alpha_range=(0.9, 1.1), prob=0.5)
        result = time_stretch_spectrogram(result, rate_range=(0.95, 1.05), prob=0.3)
        result = frequency_masking(result, max_masks=2, max_width=15, prob=0.5)
        result = time_masking(result, max_masks=2, max_width=30, prob=0.5)
        result = gaussian_noise_spectrogram(result, snr_range=(25, 40), prob=0.2)

    elif augment_level == "heavy":
        result = vtlp_frequency_warping(result, alpha_range=(0.85, 1.15), prob=0.6)
        result = time_stretch_spectrogram(result, rate_range=(0.9, 1.1), prob=0.4)
        result = frequency_masking(result, max_masks=3, max_width=20, prob=0.6)
        result = time_masking(result, max_masks=3, max_width=40, prob=0.6)
        result = gaussian_noise_spectrogram(result, snr_range=(20, 40), prob=0.3)

    return result
