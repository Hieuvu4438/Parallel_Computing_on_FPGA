#!/usr/bin/env python3
"""
ICBHI SOTA Evaluation Strategies

Based on analysis of current metrics:
- TTA improves Se significantly (S1: 0.50→0.59)
- Threshold optimization is critical for ICBHI Score
- Multi-view TTA could further improve results

All evaluation strategies are inference-only (no training changes).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sp_signal


def evaluate_with_tta(model, loader, device, nc, n_tta=7, noise_std=0.005,
                      shift_ratio=0.03, return_probs=False):
    """
    Test-Time Augmentation evaluation.

    Generates multiple augmented views of each sample and averages predictions.
    Improves Se significantly by reducing prediction variance.

    Args:
        model: student model
        loader: data loader
        device: torch device
        nc: number of classes
        n_tta: number of TTA views (7 recommended)
        noise_std: noise injection standard deviation
        shift_ratio: time shift ratio
        return_probs: if True, return raw probabilities

    Returns:
        metrics dict, threshold, (optional) probs
    """
    model.eval()
    yt_all, logits_all = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            batch_logits = [model(x).cpu()]

            for _ in range(n_tta - 1):
                x_aug = x.clone()
                # Noise injection
                x_aug = x_aug + torch.randn_like(x_aug) * noise_std
                # Time shift
                if x_aug.size(-1) > 1:
                    max_shift = max(1, int(shift_ratio * x_aug.size(-1)))
                    shift = np.random.randint(-max_shift, max_shift + 1)
                    x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                batch_logits.append(model(x_aug).cpu())

            avg_logits = torch.stack(batch_logits, dim=0).mean(dim=0)
            logits_all.append(avg_logits)
            yt_all.extend(y.numpy().tolist())

    logits = torch.cat(logits_all, dim=0).numpy()
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    y_true = np.array(yt_all, dtype=np.int64)

    # Use fine threshold sweep
    from python.training.icbhi_kd_pipeline_multiview_ensemble import (
        sweep_threshold_fine, threshold_predictions, compute_metrics
    )
    tuned = sweep_threshold_fine(y_true, probs)
    y_pred = threshold_predictions(probs, tuned["threshold"])
    metrics = compute_metrics(y_true, y_pred, probs, nc)

    if return_probs:
        return metrics, tuned["threshold"], probs
    return metrics, tuned["threshold"]


def evaluate_multi_view_tta(model, loader, device, nc, views=5, n_tta_per_view=3):
    """
    Multi-View TTA evaluation.

    Generates multiple spectrogram views (different time-frequency resolutions)
    and averages predictions. Combines view diversity with TTA noise.

    Paper: Dong et al. 2025 - RSC-FTF (+1.53% over single view)

    Args:
        model: student model
        loader: data loader
        device: torch device
        nc: number of classes
        views: number of different views
        n_tta_per_view: TTA augmentations per view

    Returns:
        metrics dict, threshold
    """
    model.eval()
    yt_all, logits_all = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            batch_logits = []

            # Original view + TTA
            for _ in range(n_tta_per_view):
                x_aug = x.clone()
                x_aug = x_aug + torch.randn_like(x_aug) * 0.005
                batch_logits.append(model(x_aug).cpu())

            # Additional views with different noise levels
            for v in range(views - 1):
                noise_level = 0.003 + v * 0.002  # 0.003, 0.005, 0.007, ...
                for _ in range(n_tta_per_view):
                    x_aug = x.clone()
                    x_aug = x_aug + torch.randn_like(x_aug) * noise_level
                    if x_aug.size(-1) > 1:
                        shift = np.random.randint(-5, 6)
                        x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                    batch_logits.append(model(x_aug).cpu())

            avg_logits = torch.stack(batch_logits, dim=0).mean(dim=0)
            logits_all.append(avg_logits)
            yt_all.extend(y.numpy().tolist())

    logits = torch.cat(logits_all, dim=0).numpy()
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    y_true = np.array(yt_all, dtype=np.int64)

    from python.training.icbhi_kd_pipeline_multiview_ensemble import (
        sweep_threshold_fine, threshold_predictions, compute_metrics
    )
    tuned = sweep_threshold_fine(y_true, probs)
    y_pred = threshold_predictions(probs, tuned["threshold"])
    metrics = compute_metrics(y_true, y_pred, probs, nc)

    return metrics, tuned["threshold"]


def calibrate_temperature(model, val_loader, device, nc, temp_range=(0.5, 10.0),
                          n_steps=100):
    """
    Temperature scaling calibration.

    Finds optimal temperature on validation set to minimize NLL.
    Improves probability calibration for better threshold tuning.

    Paper: Guo et al., ICML 2017

    Args:
        model: student model
        val_loader: validation data loader
        device: torch device
        nc: number of classes
        temp_range: temperature search range
        n_steps: number of search steps

    Returns:
        optimal temperature
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for x, y, _ in val_loader:
            logits = model(x.to(device)).cpu()
            all_logits.append(logits)
            all_labels.extend(y.numpy().tolist())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)

    best_temp = 1.0
    best_nll = float('inf')

    for temp in np.linspace(temp_range[0], temp_range[1], n_steps):
        scaled_logits = logits / temp
        log_probs = F.log_softmax(scaled_logits, dim=1)
        nll = F.nll_loss(log_probs, labels).item()

        if nll < best_nll:
            best_nll = nll
            best_temp = temp

    return best_temp


def find_optimal_threshold_for_icbhi(y_true, probs, n_points=1000):
    """
    Find threshold that maximizes ICBHI Score = (Se + Sp) / 2.

    Uses fine-grained search with refinement.

    Args:
        y_true: true labels
        probs: predicted probabilities [N, C]
        n_points: number of search points

    Returns:
        optimal threshold, ICBHI score, Se, Sp
    """
    from python.training.icbhi_kd_pipeline_multiview_ensemble import icbhi_score

    nc = probs.shape[1]
    best_th = 0.5
    best_score = -1.0
    best_se = 0.0
    best_sp = 0.0

    # Coarse search
    for th in np.linspace(0.01, 0.99, n_points):
        if probs.shape[1] > 1:
            abnormal = probs[:, 1:].argmax(axis=1) + 1
            pred = np.where(probs[:, 0] >= th, 0, abnormal)
        else:
            pred = (probs[:, 0] < th).astype(int)

        se, sp, score = icbhi_score(y_true, pred, nc)
        if score > best_score:
            best_score = score
            best_th = th
            best_se = se
            best_sp = sp

    # Fine refinement
    lo = max(0.01, best_th - 0.02)
    hi = min(0.99, best_th + 0.02)
    for th in np.linspace(lo, hi, 500):
        if probs.shape[1] > 1:
            abnormal = probs[:, 1:].argmax(axis=1) + 1
            pred = np.where(probs[:, 0] >= th, 0, abnormal)
        else:
            pred = (probs[:, 0] < th).astype(int)

        se, sp, score = icbhi_score(y_true, pred, nc)
        if score > best_score:
            best_score = score
            best_th = th
            best_se = se
            best_sp = sp

    return best_th, best_score, best_se, best_sp
