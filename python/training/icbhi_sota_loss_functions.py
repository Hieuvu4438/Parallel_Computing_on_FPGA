#!/usr/bin/env python3
"""
ICBHI SOTA Loss Functions for Knowledge Distillation

Based on analysis of current metrics:
- Problem: Crackle recall < 10%, Se << Sp (model biased toward Normal)
- Root cause: Class imbalance + KD loss dominated by Normal class

Solutions from papers:
1. Focal Loss with class-balanced weights (Cui et al. CVPR 2019)
2. Label smoothing on teacher soft labels (Dong et al. 2025 ADD-RSC)
3. Binary auxiliary loss for abnormal detection
4. Sensitivity-aware loss rebalancing

All loss functions are architecture-agnostic (work with any student model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss based on Effective Number of Samples.

    Addresses class imbalance by weighting classes inversely to their
    effective number of samples.

    Paper: Cui et al., CVPR 2019
    Formula: weight_c = (1 - beta) / (1 - beta^n_c)

    Args:
        samples_per_class: array of samples per class
        beta: hyperparameter for effective number (0.9999 recommended)
        gamma: focal loss gamma (2.0-3.0 recommended)
        label_smoothing: label smoothing factor
    """

    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * len(samples_per_class)
        self.alpha = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        nc = logits.size(1)
        device = logits.device
        alpha = self.alpha.to(device)

        if self.ls > 0:
            target = torch.full_like(logits, self.ls / max(nc - 1, 1))
            target.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        else:
            target = F.one_hot(targets, nc).float()

        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        loss = -((1 - p) ** self.gamma) * target * logp
        loss = loss * alpha.view(1, -1)
        return loss.sum(dim=1).mean()


class SmoothedKDLoss(nn.Module):
    """
    Knowledge Distillation loss with label smoothing on teacher soft labels.

    Prevents overconfident teacher predictions by smoothing the soft labels
    before computing KL divergence.

    Paper: Dong et al. 2025 - ADD-RSC

    Args:
        temperature: softmax temperature (3.0-4.0 recommended)
        smoothing: label smoothing factor on teacher labels (0.1-0.2 recommended)
    """

    def __init__(self, temperature=4.0, smoothing=0.15):
        super().__init__()
        self.temperature = temperature
        self.smoothing = smoothing

    def forward(self, student_logits, teacher_probs):
        nc = teacher_probs.size(1)
        # Smooth teacher labels
        smoothed_teacher = (1 - self.smoothing) * teacher_probs + self.smoothing / nc
        # KL divergence
        kd = -(smoothed_teacher * F.log_softmax(student_logits / self.temperature, dim=1)
               ).sum(dim=1).mean() * (self.temperature ** 2)
        return kd


class SensitivityAwareBinaryLoss(nn.Module):
    """
    Binary auxiliary loss that pushes for higher sensitivity.

    Computes BCE on abnormal logit (logsumexp of non-normal logits minus normal logit).
    Target is blended from hard label and teacher probability.

    Key insight from metrics: current models have Se << Sp.
    This loss helps balance by emphasizing abnormal detection.

    Args:
        teacher_ratio: weight of teacher signal vs hard label (0.3-0.5)
    """

    def __init__(self, teacher_ratio=0.4):
        super().__init__()
        self.teacher_ratio = teacher_ratio

    def forward(self, logits, targets, teacher_probs):
        # Abnormal logit: logsumexp(non-normal) - normal
        if logits.size(1) == 2:
            abnormal_logit = logits[:, 1] - logits[:, 0]
        else:
            abnormal_logit = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]

        # Target: blend hard label and teacher signal
        hard_bin = (targets != 0).float()
        teacher_bin = (1.0 - teacher_probs[:, 0]).clamp(0, 1)
        bin_target = (1 - self.teacher_ratio) * hard_bin + self.teacher_ratio * teacher_bin

        return F.binary_cross_entropy_with_logits(abnormal_logit, bin_target)


class MultiTemperatureKDLoss(nn.Module):
    """
    Multi-temperature Knowledge Distillation.

    Computes KD loss at multiple temperatures simultaneously and averages.
    Different temperatures capture different levels of teacher knowledge:
    - Low temperature (2.0): class-level decisions
    - Medium temperature (4.0): inter-class relationships
    - High temperature (8.0): soft distribution patterns

    Paper: Zhao et al., CVPR 2022 (Decoupled KD)

    Args:
        temperatures: list of temperature values
        smoothing: label smoothing on teacher soft labels
    """

    def __init__(self, temperatures=(2.0, 4.0, 8.0), smoothing=0.1):
        super().__init__()
        self.temperatures = temperatures
        self.smoothing = smoothing

    def forward(self, student_logits, teacher_probs):
        total_kd = torch.tensor(0.0, device=student_logits.device)
        nc = teacher_probs.size(1)

        for T in self.temperatures:
            smoothed_teacher = (1 - self.smoothing) * teacher_probs + self.smoothing / nc
            kd = -(smoothed_teacher * F.log_softmax(student_logits / T, dim=1)
                   ).sum(dim=1).mean() * (T ** 2)
            total_kd = total_kd + kd

        return total_kd / len(self.temperatures)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.

    Pulls features of same class together, pushes different classes apart.
    Helps learn better feature representations in embedding space.

    Paper: Dong et al. 2025 - RSC-FTF (part of 67.55% system)

    Args:
        temperature: temperature for similarity computation (0.1 recommended)
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # L2 normalize
        features = F.normalize(features, dim=1)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask for positive pairs (same class)
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()
        mask.fill_diagonal_(0)

        # Log-softmax
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        n_positives = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (n_positives + 1e-8)

        # Only compute loss for samples with positive pairs
        valid = n_positives > 0
        if valid.any():
            loss = -mean_log_prob[valid].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss


class CombinedKDTotalLoss(nn.Module):
    """
    Combined KD loss function that addresses ICBHI-specific challenges.

    Combines:
    1. Class-balanced focal loss (hard labels)
    2. Smoothed KD loss (teacher soft labels)
    3. Sensitivity-aware binary loss (abnormal detection)
    4. Optional contrastive loss (feature quality)

    Addresses the key problem: Crackle recall < 10%, Se << Sp

    Args:
        hard_weight: weight for focal loss
        kd_weight: weight for KD loss
        binary_weight: weight for binary loss
        contrastive_weight: weight for contrastive loss (0 to disable)
        focal_gamma: focal loss gamma
        kd_smoothing: label smoothing on teacher labels
        kd_temperature: KD temperature
        teacher_ratio: binary loss teacher ratio
    """

    def __init__(self, samples_per_class, hard_weight=0.35, kd_weight=0.45,
                 binary_weight=0.15, contrastive_weight=0.0,
                 focal_gamma=2.5, kd_smoothing=0.15, kd_temperature=4.0,
                 teacher_ratio=0.4):
        super().__init__()
        self.hard_weight = hard_weight
        self.kd_weight = kd_weight
        self.binary_weight = binary_weight
        self.contrastive_weight = contrastive_weight

        self.focal_loss = ClassBalancedFocalLoss(
            samples_per_class, beta=0.9999, gamma=focal_gamma, label_smoothing=0.05
        )
        self.kd_loss = SmoothedKDLoss(
            temperature=kd_temperature, smoothing=kd_smoothing
        )
        self.binary_loss = SensitivityAwareBinaryLoss(teacher_ratio=teacher_ratio)

        if contrastive_weight > 0:
            self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.1)
        else:
            self.contrastive_loss = None

    def forward(self, logits, targets, teacher_probs, features=None):
        """
        Args:
            logits: [B, C] student logits
            targets: [B] hard labels
            teacher_probs: [B, C] teacher soft labels
            features: [B, D] optional features for contrastive loss

        Returns:
            total_loss, loss_dict (for logging)
        """
        hard = self.focal_loss(logits, targets)
        kd = self.kd_loss(logits, teacher_probs)
        binary = self.binary_loss(logits, targets, teacher_probs)

        total = (self.hard_weight * hard +
                 self.kd_weight * kd +
                 self.binary_weight * binary)

        loss_dict = {
            'hard_loss': hard.item(),
            'kd_loss': kd.item(),
            'binary_loss': binary.item(),
        }

        if self.contrastive_loss is not None and features is not None and self.contrastive_weight > 0:
            scl = self.contrastive_loss(features, targets)
            total = total + self.contrastive_weight * scl
            loss_dict['contrastive_loss'] = scl.item()

        loss_dict['total_loss'] = total.item()
        return total, loss_dict
