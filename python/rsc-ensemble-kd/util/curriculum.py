"""
Class-Aware Curriculum Knowledge Distillation (CACD)

Core innovations:
1. Two-stage curriculum: Stage 1 (binary KD) → Stage 2 (4-class KD)
2. Class-aware temperature: T_c = T_base * (1 + beta * difficulty_c)
3. Feature alignment: CNN student learns CLAP teacher's feature space

Reference: Our paper "Class-Aware Curriculum Knowledge Distillation..."
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassAwareKDLoss(nn.Module):
    """
    Knowledge Distillation with class-aware temperature scaling.

    Instead of a single temperature T for all classes, each class c has:
        T_c = T_base * (1 + beta * difficulty_c)

    where difficulty_c = 1 - accuracy_c (estimated from teacher ensemble).

    This gives:
    - Easy classes (normal): higher T → softer distribution → less focus
    - Hard classes (both, wheeze): lower T → sharper distribution → more focus

    Combined with class weights w_c to handle imbalance.
    """

    def __init__(self, T_base=4.0, beta=0.5, alpha=0.5, class_weights=None):
        """
        Args:
            T_base: Base temperature for KD
            beta: Difficulty scaling factor. Higher = more differentiation between classes.
            alpha: Weight for KD loss. (1-alpha) for hard CE loss.
            class_weights: Per-class weights [n_cls]. If None, uniform.
        """
        super().__init__()
        self.T_base = T_base
        self.beta = beta
        self.alpha = alpha
        self.n_cls = 4  # ICBHI: normal, crackle, wheeze, both

        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        # Default difficulty scores (will be updated during training)
        # Higher difficulty = lower temperature = sharper distribution
        self.register_buffer('difficulty', torch.zeros(self.n_cls))

    def update_difficulty(self, per_class_accuracy):
        """
        Update class difficulty based on current model performance.

        Args:
            per_class_accuracy: dict or list of per-class accuracy [n_cls]
        """
        if isinstance(per_class_accuracy, list):
            acc = torch.tensor(per_class_accuracy, dtype=torch.float32)
        else:
            acc = torch.tensor(list(per_class_accuracy.values()), dtype=torch.float32)

        # Difficulty = 1 - accuracy (harder classes have higher difficulty)
        self.difficulty = (1.0 - acc).to(self.difficulty.device)
        print(f"  Updated class difficulty: {self.difficulty.tolist()}")

    def get_class_temperatures(self):
        """
        Compute per-class temperatures based on difficulty.

        T_c = T_base * (1 + beta * difficulty_c)
        """
        T = self.T_base * (1.0 + self.beta * self.difficulty)
        return T  # [n_cls]

    def forward(self, student_logits, teacher_logits, hard_labels=None):
        """
        Args:
            student_logits: [B, n_cls] raw logits from student
            teacher_logits: [B, n_cls] raw logits from teacher ensemble
            hard_labels: [B] ground truth labels (optional)
        """
        T = self.get_class_temperatures()  # [n_cls]

        # Expand T for broadcasting: [1, n_cls]
        T = T.unsqueeze(0)

        # Compute per-class KL divergence
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)

        # KL divergence per sample
        kl_per_sample = F.kl_div(soft_student, soft_teacher, reduction='none')  # [B, n_cls]

        # Apply class weights
        if self.class_weights is not None:
            w = self.class_weights.to(kl_per_sample.device).unsqueeze(0)  # [1, n_cls]
            kl_per_sample = kl_per_sample * w

        # Sum over classes, mean over batch
        kd_loss = kl_per_sample.sum(dim=1).mean()

        # Scale by T^2 (standard KD scaling)
        T_mean = T.mean()
        kd_loss = kd_loss * (T_mean ** 2)

        # Combine with hard CE loss
        if hard_labels is not None and self.alpha < 1.0:
            ce_loss = F.cross_entropy(student_logits, hard_labels)
            return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return kd_loss


class BinaryKDLoss(nn.Module):
    """
    Stage 1 Curriculum: Binary KD (normal vs abnormal).

    Collapses 4-class teacher logits into 2-class:
    - Class 0 (normal): stays as class 0
    - Class 1 (abnormal): max of classes 1, 2, 3

    This gives the student a simpler task to learn first.
    """

    def __init__(self, T=4.0, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha

    @staticmethod
    def collapse_to_binary(logits):
        """
        Convert 4-class logits to 2-class (normal vs abnormal).

        Args:
            logits: [B, 4]
        Returns:
            binary_logits: [B, 2]
        """
        normal = logits[:, 0:1]  # [B, 1]
        abnormal = logits[:, 1:].max(dim=1, keepdim=True).values  # [B, 1]
        return torch.cat([normal, abnormal], dim=1)  # [B, 2]

    def forward(self, student_logits, teacher_logits, hard_labels=None):
        """
        Args:
            student_logits: [B, 4] raw logits from student
            teacher_logits: [B, 4] raw logits from teacher
            hard_labels: [B] 4-class ground truth labels
        """
        # Collapse to binary
        student_binary = self.collapse_to_binary(student_logits)
        teacher_binary = self.collapse_to_binary(teacher_logits)

        # KD loss on binary
        T = self.T
        soft_student = F.log_softmax(student_binary / T, dim=1)
        soft_teacher = F.softmax(teacher_binary / T, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)

        # Hard CE on binary labels
        if hard_labels is not None:
            binary_labels = (hard_labels > 0).long()  # 0=normal, 1=abnormal
            ce_loss = F.cross_entropy(student_logits, binary_labels)
            return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return kd_loss


class FeatureAlignmentLoss(nn.Module):
    """
    Feature-level distillation: align CNN student features with CLAP teacher features.

    Uses a learnable projector to map CNN features (2048-dim) to CLAP feature space (1024-dim).
    Loss = MSE(projected_student_features, teacher_features)
    """

    def __init__(self, student_dim=2048, teacher_dim=1024):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.BatchNorm1d(teacher_dim),
            nn.ReLU(),
            nn.Linear(teacher_dim, teacher_dim)
        )

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: [B, student_dim] features from CNN student
            teacher_features: [B, teacher_dim] features from CLAP teacher
        """
        projected = self.projector(student_features)

        # Normalize both
        projected = F.normalize(projected, dim=1)
        teacher_norm = F.normalize(teacher_features, dim=1)

        # Cosine similarity loss (1 - cosine_sim)
        cosine_sim = (projected * teacher_norm).sum(dim=1)
        loss = (1 - cosine_sim).mean()

        return loss


class CACDLoss(nn.Module):
    """
    Complete CACD (Class-Aware Curriculum Distillation) loss.

    Combines:
    1. Binary KD (Stage 1) or Class-Aware KD (Stage 2)
    2. Feature alignment loss
    3. Optional hard label CE loss
    """

    def __init__(self, T_base=4.0, beta=0.5, alpha=0.5,
                 class_weights=None, feat_weight=0.1,
                 student_dim=2048, teacher_dim=1024):
        """
        Args:
            T_base: Base temperature for KD
            beta: Difficulty scaling factor
            alpha: Weight for KD vs CE
            class_weights: Per-class weights for handling imbalance
            feat_weight: Weight for feature alignment loss
            student_dim: CNN student feature dimension
            teacher_dim: CLAP teacher feature dimension
        """
        super().__init__()

        self.binary_kd = BinaryKDLoss(T=T_base, alpha=alpha)
        self.class_aware_kd = ClassAwareKDLoss(
            T_base=T_base, beta=beta, alpha=alpha, class_weights=class_weights
        )
        self.feat_align = FeatureAlignmentLoss(student_dim, teacher_dim)
        self.feat_weight = feat_weight

        self.stage = 1  # Start with standard KD (difficulty=0 for all classes)

    def set_stage(self, stage):
        """Switch between Stage 1 (binary) and Stage 2 (class-aware)."""
        self.stage = stage
        print(f"  CACD stage set to: {stage}")

    def update_difficulty(self, per_class_accuracy):
        """Update class difficulty for class-aware temperature."""
        self.class_aware_kd.update_difficulty(per_class_accuracy)

    def forward(self, student_logits, teacher_logits, hard_labels=None,
                student_features=None, teacher_features=None):
        """
        Args:
            student_logits: [B, 4] from CNN student
            teacher_logits: [B, 4] from teacher ensemble
            hard_labels: [B] ground truth labels
            student_features: [B, 2048] CNN features
            teacher_features: [B, 1024] CLAP features
        """
        # KD loss based on current stage
        if self.stage == 1:
            # Stage 1: Standard KD (not binary!) — preserve class information
            kd_loss = self.class_aware_kd(student_logits, teacher_logits, hard_labels)
        else:
            # Stage 2: Class-aware KD with updated difficulty
            kd_loss = self.class_aware_kd(student_logits, teacher_logits, hard_labels)

        total_loss = kd_loss

        # Feature alignment loss (if features provided)
        if student_features is not None and teacher_features is not None:
            feat_loss = self.feat_align(student_features, teacher_features)
            total_loss = total_loss + self.feat_weight * feat_loss

        return total_loss
