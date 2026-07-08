"""
Custom loss functions for improved respiratory sound classification.
All losses are training-only and do NOT affect evaluation protocol.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: reduces loss contribution from easy (well-classified) samples
    and focuses on hard (misclassified) samples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Particularly effective for imbalanced datasets where the majority class
    (normal) dominates the gradient signal.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Per-class weight tensor of shape [n_cls].
                   If None, no class weighting is applied.
                   If 'auto', computed from class frequencies.
            gamma: Focusing parameter. Higher gamma = more focus on hard samples.
                   gamma=0 reduces to standard cross-entropy.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None and alpha != 'auto':
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits of shape [B, n_cls]
            targets: Either hard labels [B] or soft labels [B, n_cls]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)
            if targets.dim() == 1:
                # Hard labels
                focal_loss = alpha_t[targets] * focal_loss
            else:
                # Soft labels: weighted average of alpha by soft label distribution
                focal_loss = (targets * alpha_t.unsqueeze(0)).sum(dim=1) * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class KDLoss(nn.Module):
    """
    Proper Knowledge Distillation loss with temperature scaling and
    alpha-blending between soft (KD) and hard (CE) losses.

    L = alpha * T^2 * KL(softmax(z_s/T) || softmax(z_t/T)) + (1-alpha) * CE(z_s, y)

    The temperature T softens the probability distribution, revealing more
    information about inter-class relationships learned by the teacher.
    The alpha parameter controls the balance between mimicking the teacher
    and learning from ground truth labels.

    Reference: Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
    """

    def __init__(self, temperature=4.0, alpha=0.5):
        """
        Args:
            temperature: Softening temperature. Higher T = softer distributions.
                         T=1 is equivalent to no scaling. Typical: 2-8.
            alpha: Weight for KD loss. (1-alpha) is weight for CE loss.
                   alpha=0.5 is a balanced starting point.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, hard_labels=None):
        """
        Args:
            student_logits: Raw logits from student model [B, n_cls]
            teacher_logits: Raw logits (or probabilities) from teacher ensemble [B, n_cls]
            hard_labels: Ground truth hard labels [B]. If None, only KD loss is used.
        """
        T = self.temperature

        # Soft targets from teacher
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)

        if hard_labels is not None and self.alpha < 1.0:
            ce_loss = F.cross_entropy(student_logits, hard_labels)
            return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return kd_loss


class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy with label smoothing.

    Replaces hard one-hot targets with a mixture of the original target
    and a uniform distribution: y_smooth = (1-epsilon) * y + epsilon / K

    This prevents the model from becoming overconfident and improves
    calibration, which is beneficial for noisy medical audio labels.

    Reference: Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016
    """

    def __init__(self, epsilon=0.1, reduction='mean'):
        """
        Args:
            epsilon: Smoothing factor. 0 = no smoothing (standard CE).
                     Typical: 0.05-0.2 for medical audio.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [B, n_cls]
            targets: Hard labels [B] (integer class indices)
        """
        n_cls = inputs.shape[-1]
        log_probs = F.log_softmax(inputs, dim=-1)

        # Hard label part
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Uniform smoothing part
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1 - self.epsilon) * nll_loss + self.epsilon * smooth_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LDAMLoss(nn.Module):
    """
    LDAM (Label-Distribution-Aware Margin) Loss.

    Adds class-aware margins to the logits before computing cross-entropy.
    Minority classes get larger margins, forcing the model to learn more
    discriminative features for under-represented classes.

    The margin for class c is: margin_c = C / (n_c)^(1/4)
    where n_c is the number of samples in class c and C is a hyperparameter.

    Combined with deferred re-weighting (DRW): use standard CE for the first
    N epochs, then switch to class-weighted CE for the remaining epochs.

    Reference: Cao et al., "Learning Imbalanced Data with Label-Distribution-Aware
    Margin Loss", NeurIPS 2019
    """

    def __init__(self, class_counts, max_margin=0.5, scale=30.0):
        """
        Args:
            class_counts: List of sample counts per class, e.g., [2413, 1246, 362, 121]
            max_margin: Maximum margin value (C). Controls the strength of class-aware margins.
            scale: Logit scaling factor (s in the paper). Higher = sharper distributions.
        """
        super().__init__()
        self.scale = scale
        margins = max_margin / (torch.tensor(class_counts, dtype=torch.float32) ** 0.25)
        self.register_buffer('margins', margins)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [B, n_cls]
            targets: Hard labels [B] (integer class indices)
        """
        # Add class-aware margin to the target class logit
        one_hot = F.one_hot(targets, inputs.shape[1]).float()
        margins_expanded = one_hot * self.margins.unsqueeze(0)
        scaled_logits = (inputs - margins_expanded) * self.scale

        return F.cross_entropy(scaled_logits, targets)


def compute_class_weights_from_counts(class_counts, strategy='effective_number', beta=0.999):
    """
    Compute per-class weights for handling class imbalance.

    Args:
        class_counts: List or array of sample counts per class, e.g., [2413, 1246, 362, 121]
        strategy: 'inverse' for simple inverse frequency,
                  'effective_number' for CVPR 2019 method
        beta: Hyperparameter for effective_number strategy (0.99-0.9999)

    Returns:
        alpha: Tensor of shape [n_cls] with normalized weights
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)

    if strategy == 'inverse':
        alpha = 1.0 / counts
    elif strategy == 'effective_number':
        # Class-Balanced Loss Based on Effective Number of Samples (Cui et al., CVPR 2019)
        effective_num = 1.0 - torch.pow(beta, counts)
        alpha = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize so mean weight = 1.0
    alpha = alpha / alpha.sum() * len(class_counts)
    return alpha
