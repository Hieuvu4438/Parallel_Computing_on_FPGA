# ICBHI SOTA Benchmark Strategies — Target ICBHI Score > 68%

**Date:** 2026-06-03
**Objective:** Achieve ICBHI Score > 68% on both 4-class and 2-class classification tasks using the ICBHI 2017 Respiratory Sound Database with the official patient-wise 60/40 split protocol.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Analysis](#2-problem-analysis)
3. [SOTA Landscape Review](#3-sota-landscape-review)
4. [Strategy A: Foundation Model Fine-Tuning with Patient-Aware Training](#4-strategy-a)
5. [Strategy B: Transformer Mega-Ensemble Knowledge Distillation](#5-strategy-b)
6. [Strategy C: Multi-Modal Fusion with Advanced Augmentation](#6-strategy-c)
7. [Strategy D: Combined SOTA Pipeline (Ensemble of Best Components)](#7-strategy-d)
8. [Novel Methods — Differentiating from Existing Papers](#8-novel-methods--differentiating-from-existing-papers)
   - 8.1 What Existing Papers All Do
   - 8.2 Strategy E: Diffusion-Based Minority Class Augmentation
   - 8.3 Strategy F: Mixture of Experts with Class-Specialized Routing
   - 8.4 Strategy G: Patient-Aware Meta-Learning (Patient-as-Task)
   - 8.5 Strategy H: Multi-Task Learning with Auxiliary Respiratory Tasks
   - 8.6 Strategy I: Learned Adaptive Threshold
   - 8.7 Summary of Novel Methods
   - 8.8 Recommended Novel Combinations
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Expected Results & Benchmarks](#10-expected-results--benchmarks)
11. [Ablation Study Plan](#11-ablation-study-plan)
12. [References](#12-references)

---

## 1. Executive Summary

This document presents **4 actionable strategies** to achieve ICBHI Score > 68% on the ICBHI 2017 Respiratory Sound Database benchmark. Each strategy is designed to address the core challenges of the ICBHI dataset: severe class imbalance, inter-patient variability, the Se << Sp imbalance problem, and the difficulty of the "Both" (Crackle+Wheeze) class.

### Current Baseline

| Experiment | Test ICBHI Score | Test Sensitivity | Test Specificity | Val ICBHI Score |
|---|---:|---:|---:|---:|
| E1 (Calibrated Ensemble KD) | 0.634 | 0.380 | 0.889 | 0.691 |
| E2 (PatchMix Distill) | 0.610 | 0.376 | 0.844 | 0.695 |
| E3 (PAFA Relational) | 0.614 | 0.411 | 0.817 | 0.698 |
| S2 (Feature+Attention, TTA) | 0.665 | — | — | — |
| **Current SOTA** | **0.681** | — | — | — |
| **Target** | **>0.680** | **>0.45** | **>0.90** | — |

### Strategy Overview

| Strategy | Core Innovation | Expected ICBHI Score | Risk | Effort |
|---|---|---:|---|---|
| **A: Foundation Model Fine-Tuning** | BEATs/AST pretrained + PAFA losses | 68–72% | Medium | Medium |
| **B: Transformer Mega-Ensemble KD** | 7-teacher ensemble + cross-arch KD | 68–74% | Medium-High | High |
| **C: Multi-Modal Fusion + Augmentation** | Time+Spectral fusion + VTLP + MixUp | 66–70% | Low | Medium |
| **D: Combined SOTA Pipeline** | Best components from A+B+C | 70–76% | Medium | Very High |

---

## 2. Problem Analysis

### 2.1 Dataset Characteristics

- **6,898 respiratory cycles** from **126 patients**
- **4 classes:** Normal (~60%), Crackle (~20%), Wheeze (~10%), Both (~10%)
- **Official split:** Patients 101–160 (train), Patients 161–226 (test)
- **Patient-independent evaluation** — no patient leakage between train/test

### 2.2 Core Challenges

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ROOT CAUSE ANALYSIS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. SEVERE CLASS IMBALANCE                                           │
│     Normal: ~60%, Crackle: ~20%, Wheeze: ~10%, Both: ~10%          │
│     → Model bias toward Normal → Sensitivity << Specificity          │
│                                                                      │
│  2. INTER-PATIENT VARIABILITY                                        │
│     Different recording devices, environments, breathing patterns     │
│     → Domain shift between train/test patients                       │
│                                                                      │
│  3. Se << Sp IMBALANCE                                               │
│     ICBHI Score = (Se + Sp) / 2                                     │
│     Current: Se=0.38, Sp=0.89 → Score=0.634                        │
│     Need: Se=0.45+, Sp=0.90+ → Score=0.675+                        │
│     → Main lever is improving sensitivity by 5-7pp                   │
│                                                                      │
│  4. "BOTH" CLASS BOTTLENECK                                          │
│     Smallest class (~10%), hardest to detect                         │
│     Both = Crackle + Wheeze simultaneously                          │
│     → Persistent F1 bottleneck                                       │
│                                                                      │
│  5. SMALL DATASET SIZE                                               │
│     Only 6,898 cycles from 126 patients                              │
│     → Overfitting risk, need strong regularization                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Metric Optimization Insight

The ICBHI Score is the average of Sensitivity and Specificity for the **binary** task (Normal vs Abnormal):

```
ICBHI Score = (Sensitivity + Specificity) / 2

Where:
  Sensitivity = TP_abnormal / (TP_abnormal + FN_abnormal)
  Specificity = TN_normal / (TN_normal + FP_normal)
```

For 4-class classification, the binary metrics are derived by grouping classes 1,2,3 (Crackle, Wheeze, Both) as "Abnormal" and class 0 (Normal) as "Normal."

**Key insight:** The threshold for Normal vs Abnormal can be tuned independently of the 4-class argmax prediction. This is the single most powerful lever for ICBHI Score optimization.

---

## 3. SOTA Landscape Review

### 3.1 Published Results on ICBHI (Official Patient-Wise Split)

| Method | Year | ICBHI Score | Se | Sp | Task | Notes |
|---|---|---:|---:|---:|---|---|
| Patch-Mix CL (AST) | 2023 | 62.37% | — | — | 4-class | INTERSPEECH 2023 |
| ADD-RSC (AST backbone) | 2025 | 65.53% | — | — | 4-class | Adaptive Differential Denoising |
| RSC-FTF (multi-view) | 2025 | 67.55% | — | — | 4-class | Time+ Spectral fusion + VTLP |
| PAFA (BEATs) | 2025 | 64.84% | — | — | 4-class | Patient-Aware Feature Alignment |
| PAFA (BEATs) | 2025 | 72.08% | — | — | 2-class | Binary classification |
| Arch-Agnostic KD (ensemble) | 2025 | 65.69% | — | — | 4-class | k=5 teacher ensemble |
| Meta-Ensemble | 2026 | 66.49% | — | — | 4-class | Diverse data splits |
| SAM-optimized AST | 2026 | 68.10% | — | — | 4-class | Current project SOTA |

### 3.2 Key Techniques from Literature

| Technique | Source | Reported Gain | Applicability |
|---|---|---:|---|
| VTLP-Patch Augmentation | RSC-FTF (Dong 2025) | +3.19% | High — already implemented |
| Patient-Aware Losses (PCSL+GPAL) | PAFA (Jeong 2025) | +0.4–1.35% | High — training only |
| Multi-View TTA | RSC-FTF (Dong 2025) | +1.53% | High — inference only |
| Label Smoothing KD | ADD-RSC (Dong 2025) | +0.5–1.0% | High — drop-in |
| Curated Teacher Ensemble | KD-Ensembles (Toikkanen 2025) | +0.2–0.5% | High — selection only |
| Supervised Contrastive Learning | RSC-FTF (Dong 2025) | +0.5–1.0% | Medium — needs projection head |
| SAM Optimizer | Foret et al. 2021 | +0.5–1.5% | High — optimizer swap |
| Self-Supervised Pre-training | BEATs/Audio-MAE | +2–5% | Medium — needs pre-trained model |

---

## 4. Strategy A: Foundation Model Fine-Tuning with Patient-Aware Training

### 4.1 Core Idea

Leverage **self-supervised pre-trained audio models** (BEATs, AST, Audio-MAE) that have been trained on millions of audio clips (AudioSet, AudioCaps) and fine-tune them on the small ICBHI dataset. The pre-trained representations capture general audio patterns that transfer well to lung sounds. Combine with **patient-aware losses** to handle inter-patient variability.

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              STRATEGY A: FOUNDATION MODEL FINE-TUNING                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────── PRE-TRAINED BACKBONE ──────────────────────────┐ │
│  │                                                                │ │
│  │  Option 1: BEATs (Microsoft, 2023)                             │ │
│  │  ├─ Self-supervised audio pre-training                         │ │
│  │  ├─ ~90M params, fine-tune top layers                          │ │
│  │  └─ Best for: general audio → medical audio transfer           │ │
│  │                                                                │ │
│  │  Option 2: AST-Tiny (Audio Spectrogram Transformer)            │ │
│  │  ├─ Pre-trained on AudioSet                                    │ │
│  │  ├─ ~5.7M params, lightweight                                  │ │
│  │  └─ Best for: spectrogram-based classification                 │ │
│  │                                                                │ │
│  │  Option 3: Audio-MAE                                           │ │
│  │  ├─ Masked autoencoder pre-training                            │ │
│  │  ├─ ~87M params                                                │ │
│  │  └─ Best for: learning robust representations                  │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────── FINE-TUNING STRATEGY ───────────────────────────┐ │
│  │                                                                │ │
│  │  Phase 1: Linear Probe (freeze backbone, train classifier)     │ │
│  │  ├─ 10-15 epochs, lr=1e-3                                      │ │
│  │  └─ Establishes baseline features                              │ │
│  │                                                                │ │
│  │  Phase 2: Gradual Unfreeze (unfreeze top 2→4→all layers)      │ │
│  │  ├─ 10 epochs per unfreeze stage                               │ │
│  │  ├─ lr=1e-4 for backbone, 1e-3 for classifier                 │ │
│  │  └─ Differential learning rates                                │ │
│  │                                                                │ │
│  │  Phase 3: Full Fine-Tune (all layers, low lr)                  │ │
│  │  ├─ 50-80 epochs, lr=5e-5                                      │ │
│  │  └─ With patient-aware losses + augmentation                   │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────── TRAINING TECHNIQUES ────────────────────────────┐ │
│  │                                                                │ │
│  │  1. Patient-Aware Losses (PCSL + GPAL)                         │ │
│  │     ├─ PCSL: minimize within-patient / between-patient scatter  │ │
│  │     ├─ GPAL: align patient centroids toward global center       │ │
│  │     └─ Projection head removed at inference (zero test params)  │ │
│  │                                                                │ │
│  │  2. VTLP Augmentation                                          │ │
│  │     ├─ Frequency warping α ∈ [0.9, 1.1]                        │ │
│  │     ├─ Gaussian noise injection                                 │ │
│  │     └─ +3.19% ICBHI Score (paper ablation)                     │ │
│  │                                                                │ │
│  │  3. Class-Balanced Focal Loss                                   │ │
│  │     ├─ Effective number weighting (β=0.9999)                    │ │
│  │     ├─ Focal γ=2.0 for hard example focus                      │ │
│  │     └─ Label smoothing ε=0.05                                  │ │
│  │                                                                │ │
│  │  4. Sensitivity-Aware Binary Loss                               │ │
│  │     ├─ BCE on abnormal logit                                    │ │
│  │     ├─ Target: 60% hard label + 40% teacher probability         │ │
│  │     └─ Directly addresses Se << Sp problem                      │ │
│  │                                                                │ │
│  │  5. SAM Optimizer (Sharpness-Aware Minimization)                │ │
│  │     ├─ Finds flatter minima → better generalization             │ │
│  │     ├─ rho=0.05, adaptive=True                                 │ │
│  │     └─ +0.5–1.5% ICBHI Score                                   │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────── INFERENCE ──────────────────────────────────────┐ │
│  │                                                                │ │
│  │  1. Multi-View TTA (7-10 augmented views)                      │ │
│  │  2. Temperature Scaling Calibration                             │ │
│  │  3. Fine Threshold Sweep (coarse 1000pt + fine 500pt)          │ │
│  │  4. Dual-Threshold Prediction (separate Se/Sp optimization)     │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Implementation Details

#### 4.3.1 Model Configuration

```python
# BEATs fine-tuning configuration
model_config = {
    "backbone": "beats_base",           # or "ast_tiny", "audio_mae"
    "pretrained": True,
    "freeze_layers": ["patch_embed", "layer.0", "layer.1"],  # Freeze early layers
    "unfreeze_schedule": [
        {"epochs": 10, "unfreeze": []},           # Linear probe
        {"epochs": 10, "unfreeze": ["layer.2"]},  # Unfreeze layer 2
        {"epochs": 10, "unfreeze": ["layer.3"]},  # Unfreeze layer 3
        {"epochs": 70, "unfreeze": ["all"]},      # Full fine-tune
    ],
    "classifier": {
        "hidden_dim": 256,
        "dropout": 0.3,
        "num_classes": 4,
    },
    "optimizer": "sam",  # Sharpness-Aware Minimization
    "lr_backbone": 5e-5,
    "lr_classifier": 1e-3,
    "weight_decay": 0.01,
}
```

#### 4.3.2 Loss Function

```python
class StrategyALoss(nn.Module):
    """
    Combined loss for Strategy A: Foundation Model Fine-Tuning.
    """
    def __init__(self, feat_dim, num_classes, device):
        super().__init__()
        # Hard label loss
        self.focal_loss = ClassBalancedFocalLoss(
            num_classes=num_classes,
            beta=0.9999,
            gamma=2.0,
            label_smoothing=0.05,
        )
        # Patient-aware loss
        self.patient_loss = PatientAwareLoss(
            feat_dim=feat_dim,
            lambda_pcsl=50.0,
            lambda_gpal=0.0005,
        )
        # Binary auxiliary loss
        self.binary_loss = SensitivityAwareBinaryLoss(
            binary_weight=0.25,
            bin_teacher_ratio=0.4,
        )
        # Supervised contrastive loss
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.1)

    def forward(self, logits, labels, features, patient_ids, teacher_probs):
        # Class-balanced focal loss
        l_focal = self.focal_loss(logits, labels)
        # Patient-aware loss (training only, zero test params)
        l_patient = self.patient_loss(features, patient_ids)
        # Binary auxiliary loss
        l_binary = self.binary_loss(logits, labels, teacher_probs)
        # Contrastive loss
        l_contrast = self.contrastive_loss(F.normalize(features, dim=1), labels)

        # Combined loss
        total = (
            0.35 * l_focal
            + 0.30 * l_binary
            + 0.15 * l_patient
            + 0.10 * l_contrast
            + 0.10 * l_kd  # If using teacher ensemble
        )
        return total
```

#### 4.3.3 Training Schedule

```python
training_schedule = {
    "total_epochs": 100,
    "phase_1_linear_probe": {"epochs": 15, "lr": 1e-3, "frozen": True},
    "phase_2_gradual_unfreeze": {"epochs": 15, "lr": 1e-4, "stages": 3},
    "phase_3_full_finetune": {"epochs": 70, "lr": 5e-5},
    "augmentation": {
        "vtlp_prob": 0.5,
        "vtlp_alpha_range": (0.9, 1.1),
        "mixup_alpha": 0.3,
        "mixup_prob": 0.5,
        "spec_augment": {"freq_mask": 16, "time_mask": 48},
    },
    "scheduler": "cosine_warm_restart",
    "T_0": 30,
    "T_mult": 2,
    "swa_start": 80,  # Start SWA at epoch 80
}
```

### 4.4 Expected Results

| Variant | 4-class ICBHI Score | 2-class ICBHI Score | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| BEATs fine-tune (baseline) | 66–68% | 72–74% | 42–45% | 89–91% |
| BEATs + PAFA losses | 68–70% | 73–75% | 44–47% | 90–92% |
| BEATs + PAFA + VTLP + MixUp | 69–72% | 74–77% | 46–50% | 91–93% |
| BEATs + all techniques + TTA | 70–74% | 76–79% | 48–52% | 92–94% |

### 4.5 Advantages & Risks

| Advantages | Risks |
|---|---|
| Pre-trained features transfer well to medical audio | Large model may not be FPGA-deployable |
| Patient-aware losses reduce inter-patient variability | Fine-tuning on small dataset may overfit |
| Gradual unfreezing prevents catastrophic forgetting | BEATs/Audio-MAE weights need to be downloaded |
| SAM optimizer improves generalization | Higher compute cost for training |

### 4.6 FPGA Deployment Consideration

**Important:** Foundation models like BEATs (~90M params) are too large for FPGA deployment. Strategy A is designed as a **teacher model** for knowledge distillation. The student (DS-CNN-Res-SE or CNN6) is distilled from the fine-tuned foundation model and deployed on FPGA.

```
Training: BEATs (teacher) → DS-CNN-Res-SE (student) via KD
Deployment: DS-CNN-Res-SE only on FPGA
```

---

## 5. Strategy B: Transformer Mega-Ensemble Knowledge Distillation

### 5.1 Core Idea

Build a **heterogeneous teacher ensemble** of 7 models spanning CNN, CRNN, and Transformer architectures. Use **cross-architecture knowledge distillation** to transfer knowledge from the diverse ensemble to a lightweight CNN student. The key innovation is the **cross-architecture feature bridge** that enables Transformer teachers to distill spatial knowledge to CNN students.

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STRATEGY B: TRANSFORMER MEGA-ENSEMBLE KD                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────── TEACHER ENSEMBLE (7 models) ────────────────────────────┐ │
│  │                                                                             │ │
│  │  CNN Teachers (3)          Transformer Teachers (2)    Hybrid Teachers (2) │ │
│  │  ┌─────────────┐          ┌──────────────────┐       ┌──────────────────┐ │ │
│  │  │ ResNet-CNN  │          │ AST-Tiny         │       │ CNN-Transformer  │ │ │
│  │  │ (2.1M)      │          │ (5.7M)           │       │ Hybrid           │ │ │
│  │  │             │          │                  │       │ (8.2M)           │ │ │
│  │  │ ResNet-CRNN │          │ Swin-Tiny        │       │                  │ │ │
│  │  │ (3.4M)      │          │ (28M)            │       │ ResNet-CRNN-     │ │ │
│  │  │             │          │                  │       │ Attention        │ │ │
│  │  │ EfficientNet│          │                  │       │ (5.1M)           │ │ │
│  │  │ B0 (5.3M)   │          │                  │       │                  │ │ │
│  │  └──────┬──────┘          └────────┬──────────┘       └────────┬─────────┘ │ │
│  │         │                          │                           │            │ │
│  │         └──────────┬───────────────┴───────────────────────────┘            │ │
│  │                    ▼                                                        │ │
│  │  ┌──────────── CURATED ENSEMBLE SELECTION ────────────────────────────────┐ │ │
│  │  │  • Train each teacher with 3 seeds = 21 checkpoints                    │ │ │
│  │  │  • Select top-5 by validation ICBHI Score                              │ │ │
│  │  │  • Reliability-weighted ensemble (weight ∝ val Score)                  │ │ │
│  │  │  • TTA-averaged teacher logits (N=5 augmented views)                   │ │ │
│  │  │  • Temperature-scaled calibration per teacher                          │ │ │
│  │  └────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌──────────────────── CROSS-ARCHITECTURE KD BRIDGE ───────────────────────────┐ │
│  │                                                                             │ │
│  │  Level 1: Logit KD ─────────── Smoothed KD + DKD + Class-Balanced          │ │
│  │  Level 2: Feature KD ───────── ReviewKD-style cross-layer alignment         │ │
│  │  Level 3: Attention KD ─────── Transfer Transformer attention → CNN         │ │
│  │  Level 4: Relational KD ────── RKD-Distance + RKD-Angle                    │ │
│  │  Level 5: CLS Token KD ─────── Match [CLS] embedding (Transformer→CNN)     │ │
│  │                                                                             │ │
│  │  Transformer→CNN Feature Bridge:                                            │ │
│  │  ├─ Patch-to-Spatial: Reshape [B, N, D] → [B, D, H', W']                  │ │
│  │  ├─ Project: Conv2d(D, C_target, 1) → match CNN feature dims               │ │
│  │  └─ Attention Map: Average multi-head attn → spatial map                    │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌──────────────────── STUDENT TRAINING ───────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Student: DS-CNN-Res-SE (FPGA-deployable)                                  │ │
│  │  ├─ Depthwise-separable residual blocks with SE attention                   │ │
│  │  ├─ Width multiplier: 1.0–1.25                                             │ │
│  │  └─ ~1.2M params, CNN-only (no LSTM/Transformer)                           │ │
│  │                                                                             │ │
│  │  Training Techniques:                                                       │ │
│  │  ├─ Curriculum Learning (easy→hard, 40%→100% over 25 epochs)               │ │
│  │  ├─ Progressive KD Weighting (hard 0.45→0.28, KD 0.35→0.47)               │ │
│  │  ├─ EMA Mean Teacher (decay=0.998)                                          │ │
│  │  ├─ MixUp (α=0.3, p=0.5) + CutMix                                         │ │
│  │  ├─ SpecAugment (freq_mask=16, time_mask=48)                               │ │
│  │  ├─ Class-Balanced Effective-Number Sampling (β=0.9999)                     │ │
│  │  └─ SWA in last 20 epochs                                                  │ │
│  │                                                                             │ │
│  │  Loss = Σ w_i(t) · L_i  (8 components, progressive weighting)              │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌──────────────────── INFERENCE ──────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  1. Dual-Threshold Prediction (separate Se/Sp optimization)                 │ │
│  │  2. TTA ensemble (10 augmented views)                                       │ │
│  │  3. Temperature Scaling Calibration                                         │ │
│  │  4. Fine threshold sweep (coarse 1000pt + fine 500pt)                       │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Loss Function (8 Components)

```python
class StrategyBLoss(nn.Module):
    """
    8-component loss for Strategy B: Transformer Mega-Ensemble KD.
    Progressive weighting: hard→KD shift over training.
    """
    def __init__(self, feat_dim, num_classes, device):
        super().__init__()
        self.num_classes = num_classes

        # Component losses
        self.focal_loss = ClassBalancedFocalLoss(num_classes, beta=0.9999, gamma=2.0)
        self.kd_loss = SmoothedKDLoss(smoothing=0.15)
        self.dkd_loss = DecoupledKDLoss()  # Target + non-target KD
        self.feat_loss = FeatureDistillationLoss()
        self.attn_loss = AttentionTransferLoss()
        self.rkd_loss = RelationalKDLoss()
        self.binary_loss = SensitivityAwareBinaryLoss(binary_weight=0.25)
        self.ema_loss = EMAKDLoss()  # EMA teacher KD

    def get_weights(self, epoch, total_epochs):
        """Progressive weighting schedule."""
        t = min(epoch / 30.0, 1.0)  # Linear over 30 epochs
        return {
            "focal": 0.45 * (1 - t) + 0.28 * t,
            "kd": 0.35 * (1 - t) + 0.47 * t,
            "dkd": 0.10,
            "feat": 0.05 * t + 0.05,
            "attn": 0.05,
            "rkd": 0.03,
            "binary": 0.20 * (1 - t) + 0.25 * t,
            "ema": 0.15 if epoch > 10 else 0.0,
        }

    def forward(self, logits, labels, features, teacher_logits,
                teacher_features, student_features, ema_logits,
                patient_ids, epoch, total_epochs):
        w = self.get_weights(epoch, total_epochs)

        l_focal = self.focal_loss(logits, labels)
        l_kd = self.kd_loss(logits, teacher_logits)
        l_dkd = self.dkd_loss(logits, teacher_logits, labels)
        l_feat = self.feat_loss(student_features, teacher_features)
        l_attn = self.attn_loss(student_features, teacher_features)
        l_rkd = self.rkd_loss(student_features, teacher_features)
        l_binary = self.binary_loss(logits, labels, teacher_logits)
        l_ema = self.ema_loss(logits, ema_logits) if ema_logits is not None else 0

        total = (
            w["focal"] * l_focal
            + w["kd"] * l_kd
            + w["dkd"] * l_dkd
            + w["feat"] * l_feat
            + w["attn"] * l_attn
            + w["rkd"] * l_rkd
            + w["binary"] * l_binary
            + w["ema"] * l_ema
        )
        return total
```

### 5.4 Teacher Training Strategy

```python
teacher_training_config = {
    "architectures": [
        # CNN Teachers
        {"name": "resnet_cnn", "seeds": [1, 2, 3], "epochs": 100},
        {"name": "resnet_crnn", "seeds": [1, 2, 3], "epochs": 100},
        {"name": "efficientnet_b0", "seeds": [1, 2, 3], "epochs": 100},
        # Transformer Teachers
        {"name": "ast_tiny", "seeds": [1, 2], "epochs": 100},
        {"name": "swin_tiny", "seeds": [1, 2], "epochs": 100},
        # Hybrid Teachers
        {"name": "fusion_teacher", "seeds": [1, 2], "epochs": 100},
        {"name": "crnn_attention", "seeds": [1, 2], "epochs": 100},
    ],
    "total_checkpoints": 17,  # 3*3 + 2*2 + 2*2 = 9+4+4 = 17
    "curated_top_k": 5,       # Select top-5 by val ICBHI Score
    "ensemble_method": "reliability_weighted",  # Weight ∝ val Score
    "tta_for_teacher_logits": 5,  # N=5 augmented views
    "temperature_calibration": True,  # NLL-based optimal T per teacher
}
```

### 5.5 Expected Results

| Variant | 4-class ICBHI Score | 2-class ICBHI Score | Notes |
|---|---:|---:|---|
| Baseline (3 CNN teachers, logit KD) | 63.4% | — | Current E1 |
| + Transformer teachers (AST+Swin) | 66–68% | 72–74% | Richer teacher signals |
| + Cross-arch feature bridge | 67–70% | 73–76% | Feature-level transfer |
| + Curriculum + EMA + Progressive KD | 68–72% | 74–78% | Advanced training dynamics |
| + TTA + Dual-threshold + SWA | 70–74% | 76–80% | Full pipeline |

### 5.6 Advantages & Risks

| Advantages | Risks |
|---|---|
| Diverse teacher ensemble captures complementary patterns | High training compute (17 teacher checkpoints) |
| Cross-architecture KD bridges Transformer→CNN gap | Feature alignment between different architectures is challenging |
| Curriculum learning prevents overwhelming the student | Complex loss balancing (8 components) |
| EMA teacher provides smooth, evolving targets | Transformer teachers may not train well on small ICBHI dataset |
| Progressive weighting adapts over training | Implementation complexity |

---

## 6. Strategy C: Multi-Modal Fusion with Advanced Augmentation

### 6.1 Core Idea

Fuse **time-domain** and **spectral-domain** features in a multi-branch teacher model, combined with the **most aggressive augmentation pipeline** (VTLP + MixUp + CutMix + SpecAugment + noise injection). This strategy focuses on maximizing the information extracted from the limited ICBHI data through better representation learning and data augmentation.

### 6.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              STRATEGY C: MULTI-MODAL FUSION + ADVANCED AUGMENTATION              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────── MULTI-BRANCH FUSION TEACHER ────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Branch 1: Raw Waveform → 1D-CNN                                            │ │
│  │  ├─ Input: [B, 1, 128000] (8s @ 16kHz)                                     │ │
│  │  ├─ Conv1d layers with increasing receptive field                           │ │
│  │  ├─ Captures: temporal patterns, onset/offset characteristics               │ │
│  │  └─ Output: [B, 128] temporal features                                      │ │
│  │                                                                             │ │
│  │  Branch 2: Log-Mel Spectrogram → 2D-CNN                                    │ │
│  │  ├─ Input: [B, 1, 64, 800] (64 mel bins, 800 frames)                       │ │
│  │  ├─ ResNet/EfficientNet backbone                                            │ │
│  │  ├─ Captures: frequency patterns, spectral shape                            │ │
│  │  └─ Output: [B, 128] spectral features                                      │ │
│  │                                                                             │ │
│  │  Branch 3: MFCC + Delta + Delta-Delta → 2D-CNN                             │ │
│  │  ├─ Input: [B, 3, 40, 800] (13 MFCCs + deltas, 3 channels)                │ │
│  │  ├─ Lightweight CNN                                                         │ │
│  │  ├─ Captures: dynamic spectral changes, cepstral features                   │ │
│  │  └─ Output: [B, 128] cepstral features                                      │ │
│  │                                                                             │ │
│  │  ┌────────────── ATTENTION FUSION ──────────────────────────────────────┐   │ │
│  │  │  Concatenate: [B, 384] → Attention → [B, 256]                        │   │ │
│  │  │  Learnable branch importance weights                                  │   │ │
│  │  │  FC → [B, 4] logits                                                   │   │ │
│  │  └──────────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌─────────────── AUGMENTATION PIPELINE (3 levels) ────────────────────────────┐ │
│  │                                                                             │ │
│  │  Level 1: Light (inference TTA)                                             │ │
│  │  ├─ Gaussian noise (SNR 30-40 dB)                                           │ │
│  │  └─ Time shift (±3%)                                                        │ │
│  │                                                                             │ │
│  │  Level 2: Medium (standard training)                                        │ │
│  │  ├─ VTLP frequency warping (α ∈ [0.95, 1.05])                              │ │
│  │  ├─ SpecAugment (freq_mask=12, time_mask=32)                               │ │
│  │  ├─ Gaussian noise (SNR 20-30 dB)                                           │ │
│  │  └─ Speed perturbation (rate ∈ [0.97, 1.03])                               │ │
│  │                                                                             │ │
│  │  Level 3: Heavy (aggressive training)                                       │ │
│  │  ├─ VTLP frequency warping (α ∈ [0.9, 1.1])                                │ │
│  │  ├─ SpecAugment (freq_mask=20, time_mask=64, 2-3 masks each)              │ │
│  │  ├─ MixUp (α=0.3, p=0.5)                                                   │ │
│  │  ├─ CutMix (p=0.3)                                                          │ │
│  │  ├─ Gaussian noise (SNR 10-25 dB)                                           │ │
│  │  ├─ Time stretch (rate ∈ [0.9, 1.1])                                        │ │
│  │  └─ Random gain (±6 dB)                                                     │ │
│  │                                                                             │ │
│  │  ⚠️ Caution: Avoid strong time masks for Crackles (transient events)        │ │
│  │  ⚠️ Caution: Avoid strong frequency masks for Wheezes (tonal bands)         │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌─────────────── STUDENT TRAINING ────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Student: DS-CNN-Res-SE (single-branch, log-mel input only)                 │ │
│  │  ├─ Distilled from multi-branch fusion teacher                              │ │
│  │  ├─ Logit KD + Feature KD + Attention KD                                    │ │
│  │  └─ Class-balanced sampling + sensitivity-aware loss                        │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Augmentation Details

```python
class AdvancedAugmentationPipeline:
    """
    Three-level augmentation pipeline for ICBHI respiratory sound classification.
    """
    def __init__(self, level="medium", sample_rate=16000):
        self.level = level
        self.sr = sample_rate

    def __call__(self, waveform, spectrogram):
        if self.level == "light":
            return self._light_augment(waveform, spectrogram)
        elif self.level == "medium":
            return self._medium_augment(waveform, spectrogram)
        elif self.level == "heavy":
            return self._heavy_augment(waveform, spectrogram)

    def _heavy_augment(self, waveform, spectrogram):
        # Waveform-level
        if random.random() < 0.5:
            waveform = vtlp_augment(waveform, self.sr, alpha_range=(0.9, 1.1))
        if random.random() < 0.3:
            waveform = speed_perturb(waveform, self.sr, rates=[0.9, 0.95, 1.0, 1.05, 1.1])
        if random.random() < 0.4:
            waveform = add_gaussian_noise(waveform, snr_range=(10, 25))

        # Spectrogram-level
        spectrogram = spec_augment(spectrogram, freq_mask=20, time_mask=64,
                                    num_freq_masks=2, num_time_masks=3)
        if random.random() < 0.5:
            spectrogram = mixup(spectrogram, alpha=0.3)
        if random.random() < 0.3:
            spectrogram = cutmix(spectrogram)

        return waveform, spectrogram
```

### 6.4 Expected Results

| Variant | 4-class ICBHI Score | 2-class ICBHI Score | Notes |
|---|---:|---:|---|
| Fusion teacher (baseline) | 66–68% | 72–74% | Multi-branch teacher |
| + VTLP augmentation | 69–71% | 75–77% | +3.19% from paper |
| + MixUp + CutMix | 70–72% | 76–78% | Better minority class |
| + Student KD + TTA | 68–70% | 74–76% | After distillation |

### 6.5 Advantages & Risks

| Advantages | Risks |
|---|---|
| Multi-modal input captures richer information | Fusion teacher is larger and harder to train |
| Aggressive augmentation maximizes limited data | Over-augmentation may hurt Crackle/Wheeze detection |
| VTLP has strong empirical support (+3.19%) | Multi-branch student may not be FPGA-friendly |
| MixUp/CutMix improve minority class recall | Higher preprocessing cost |

---

## 7. Strategy D: Combined SOTA Pipeline (Ensemble of Best Components)

### 7.1 Core Idea

Combine the **best-performing components** from Strategies A, B, and C into a single optimized pipeline. This is the highest-effort but highest-reward strategy, designed to push ICBHI Score to 70%+.

### 7.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STRATEGY D: COMBINED SOTA PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STAGE 1: PRE-TRAINING (from Strategy A)                                        │
│  ├─ Fine-tune BEATs on ICBHI with PAFA losses                                   │
│  ├─ Fine-tune AST on ICBHI with PAFA losses                                     │
│  └─ These become the "foundation teachers"                                      │
│                                                                                  │
│  STAGE 2: TEACHER ENSEMBLE (from Strategy B)                                    │
│  ├─ Foundation teachers: BEATs, AST (from Stage 1)                              │
│  ├─ CNN teachers: ResNet-CNN, ResNet-CRNN, EfficientNet-B0                      │
│  ├─ Hybrid teacher: Fusion teacher (from Strategy C)                            │
│  ├─ Total: 6-7 diverse teachers, 3 seeds each = 18-21 checkpoints              │
│  ├─ Curated selection: top-5 by validation ICBHI Score                          │
│  └─ Reliability-weighted ensemble + TTA-averaged logits                         │
│                                                                                  │
│  STAGE 3: STUDENT DISTILLATION (from Strategies B+C)                            │
│  ├─ Student: DS-CNN-Res-SE (width=1.25, ~1.5M params)                          │
│  ├─ Multi-level KD: Logit + Feature + Attention + Relational                    │
│  ├─ Curriculum learning (easy→hard, 40%→100%)                                   │
│  ├─ Progressive KD weighting (hard 0.45→0.28, KD 0.35→0.47)                    │
│  ├─ EMA mean teacher (decay=0.998)                                              │
│  ├─ Advanced augmentation: VTLP + MixUp + CutMix + SpecAugment                  │
│  ├─ Class-balanced focal loss + sensitivity-aware binary loss                    │
│  ├─ SAM optimizer (rho=0.05)                                                    │
│  └─ SWA in last 20 epochs                                                       │
│                                                                                  │
│  STAGE 4: INFERENCE OPTIMIZATION                                                │
│  ├─ Multi-view TTA (10 views)                                                   │
│  ├─ Temperature scaling calibration                                             │
│  ├─ Dual-threshold prediction                                                   │
│  ├─ Fine threshold sweep (coarse 1000pt + fine 500pt)                           │
│  └─ 2-class metrics derived from 4-class predictions                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Complete Loss Function

```python
class StrategyDLoss(nn.Module):
    """
    Combined loss for Strategy D: all best components.
    10 loss components with progressive weighting.
    """
    def __init__(self, feat_dim, num_classes, device):
        super().__init__()
        # Hard label losses
        self.focal_loss = ClassBalancedFocalLoss(num_classes, beta=0.9999, gamma=2.5)
        # KD losses
        self.smooth_kd = SmoothedKDLoss(smoothing=0.15)
        self.dkd = DecoupledKDLoss()
        self.multi_temp_kd = MultiTemperatureKDLoss(temps=[2.0, 4.0, 8.0])
        # Feature losses
        self.feat_distill = FeatureDistillationLoss()
        self.attn_transfer = AttentionTransferLoss()
        self.rkd = RelationalKDLoss()
        # Specialized losses
        self.patient_loss = PatientAwareLoss(feat_dim, lambda_pcsl=50.0, lambda_gpal=0.0005)
        self.binary_loss = SensitivityAwareBinaryLoss(binary_weight=0.25)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.1)

    def forward(self, batch, epoch, total_epochs):
        t = min(epoch / 30.0, 1.0)

        # Progressive weights
        w = {
            "focal": 0.40 * (1 - t) + 0.25 * t,
            "kd": 0.30 * (1 - t) + 0.40 * t,
            "dkd": 0.08,
            "multi_temp": 0.05,
            "feat": 0.03 * t + 0.05,
            "attn": 0.03 * t + 0.04,
            "rkd": 0.02,
            "patient": 0.05 * t,
            "binary": 0.20 * (1 - t) + 0.25 * t,
            "contrast": 0.05,
        }

        losses = {
            "focal": self.focal_loss(batch.logits, batch.labels),
            "kd": self.smooth_kd(batch.logits, batch.teacher_logits),
            "dkd": self.dkd(batch.logits, batch.teacher_logits, batch.labels),
            "multi_temp": self.multi_temp_kd(batch.logits, batch.teacher_logits),
            "feat": self.feat_distill(batch.student_features, batch.teacher_features),
            "attn": self.attn_transfer(batch.student_features, batch.teacher_features),
            "rkd": self.rkd(batch.student_features, batch.teacher_features),
            "patient": self.patient_loss(batch.student_features, batch.patient_ids),
            "binary": self.binary_loss(batch.logits, batch.labels, batch.teacher_logits),
            "contrast": self.contrastive_loss(
                F.normalize(batch.student_features, dim=1), batch.labels
            ),
        }

        total = sum(w[k] * v for k, v in losses.items())
        return total, losses
```

### 7.4 Expected Results

| Metric | 4-class | 2-class | Notes |
|---|---:|---:|---|
| ICBHI Score | **70–76%** | **76–82%** | With all techniques combined |
| Sensitivity | 48–55% | 70–78% | Main improvement area |
| Specificity | 91–95% | 82–88% | Maintained or improved |
| Macro F1 | 45–55% | 72–80% | Especially improved for Both class |

### 7.5 Training Cost Estimate

| Component | Compute | Time (GPU-hours) |
|---|---|---|
| Foundation model fine-tuning (BEATs + AST) | 2 models × 100 epochs | ~20 |
| CNN teacher training (3 archs × 3 seeds) | 9 models × 100 epochs | ~30 |
| Hybrid teacher training (2 archs × 2 seeds) | 4 models × 100 epochs | ~15 |
| Student distillation | 1 model × 150 epochs | ~10 |
| **Total** | **~17 teacher checkpoints** | **~75 GPU-hours** |

### 7.6 Advantages & Risks

| Advantages | Risks |
|---|---|
| Combines best techniques from all strategies | Very high implementation complexity |
| Foundation model features + diverse ensemble | High training compute cost |
| All augmentation techniques active | Risk of over-engineering |
| All loss components for comprehensive KD | Loss balancing requires careful tuning |
| Most complete pipeline | Longer development time |

---

## 8. Novel Methods — Differentiating from Existing Papers

> **This section presents methods that are NOT yet published for ICBHI respiratory sound classification.** These are designed to differentiate your work from existing papers (ADD-RSC, RSC-FTF, PAFA, Meta-Ensemble, Arch-Agnostic KD, etc.) which all follow similar patterns: CNN/Transformer backbone → augmentation → KD → threshold tuning.

### 8.1 What Existing Papers All Do (The "Standard Pipeline")

Every published ICBHI paper follows essentially the same pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│           EXISTING PAPERS — STANDARD PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│  1. Mel spectrogram extraction                                   │
│  2. CNN/Transformer backbone (ResNet, AST, BEATs, etc.)         │
│  3. Data augmentation (MixUp, SpecAugment, VTLP, noise)         │
│  4. Class-balanced loss (focal loss, weighted sampling)          │
│  5. Optional: KD from ensemble teacher                           │
│  6. Threshold tuning on validation set                           │
│  7. Report Se, Sp, ICBHI Score                                  │
│                                                                  │
│  Innovation surface: backbone choice, augmentation mix,          │
│  loss weighting, threshold strategy                              │
│  → Incremental improvements only (+1-3% per paper)              │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Novel Strategy E: Generative Minority Oversampling with Diffusion Models

**Key Differentiator:** Instead of traditional augmentation (MixUp, noise), use **diffusion models to generate realistic synthetic spectrograms** for the minority classes (Crackle, Wheeze, Both). This is fundamentally different from MixUp/CutMix which are linear interpolations — diffusion models learn the actual data distribution and generate novel samples.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│          STRATEGY E: DIFFUSION-BASED MINORITY CLASS AUGMENTATION                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────── DIFFUSION SPECTROGRAM GENERATOR ────────────────────────────┐ │
│  │                                                                             │ │
│  │  Training Data: ICBHI minority classes (Crackle, Wheeze, Both)             │ │
│  │  ├─ Crackle: ~1,370 samples → generate 500 synthetic                       │ │
│  │  ├─ Wheeze: ~690 samples → generate 800 synthetic                          │ │
│  │  ├─ Both: ~690 samples → generate 800 synthetic                            │ │
│  │  └─ Total synthetic: ~2,100 new minority samples                            │ │
│  │                                                                             │ │
│  │  Architecture: Lightweight 2D UNet on 64×800 log-mel spectrograms           │ │
│  │  ├─ DDPM with 50 diffusion steps                                            │ │
│  │  ├─ Conditional on class label (class-conditional generation)               │ │
│  │  ├─ Train on ~2,850 minority spectrograms (50 epochs)                       │ │
│  │  └─ Generate 2,100 synthetic samples                                        │ │
│  │                                                                             │ │
│  │  Quality Control:                                                           │ │
│  │  ├─ FID score between real and generated spectrograms                       │ │
│  │  ├─ Classifier confidence check (teacher model must classify >0.7)          │ │
│  │  └─ Visual inspection of spectrogram patterns                               │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌─────────────── BALANCED TRAINING SET ───────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Original: Normal=4,096 + Crackle=1,370 + Wheeze=690 + Both=690           │ │
│  │  Synthetic: +0      + Crackle=+500  + Wheeze=+800  + Both=+800            │ │
│  │  ─────────────────────────────────────────────────────────────────────      │ │
│  │  Final:    Normal=4,096 + Crackle=1,870 + Wheeze=1,490 + Both=1,490       │ │
│  │                                                                             │ │
│  │  Class ratio: 2.7:1.3:1:1 (much more balanced than original 5.9:2:1:1)     │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌─────────────── TRAINING WITH SYNTHETIC DATA ───────────────────────────────┐ │
│  │                                                                             │ │
│  │  1. Mix real + synthetic data (no separate treatment)                       │ │
│  │  2. Standard training pipeline with class-balanced loss                     │ │
│  │  3. Teacher ensemble KD (same as other strategies)                          │ │
│  │  4. TTA + threshold tuning                                                  │ │
│  │                                                                             │ │
│  │  Key insight: Synthetic data reduces class imbalance                        │ │
│  │  → Model sees more Crackle/Wheeze/Both during training                      │ │
│  │  → Higher sensitivity for abnormal classes                                   │ │
│  │  → Better ICBHI Score                                                        │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  WHY THIS IS NOVEL:                                                            │
│  ├─ Existing papers use MixUp/CutMix (linear interpolation, not realistic)     │
│  ├─ Existing papers use oversampling (copies, not new samples)                 │
│  ├─ Diffusion generates NOVEL realistic spectrograms that follow the           │
│  │  actual distribution of crackle/wheeze sounds                                │
│  ├─ Conditional generation ensures class-specific patterns                      │
│  └─ No published ICBHI paper has used diffusion-based augmentation              │
│                                                                                  │
│  EXPECTED GAIN: +3-5% ICBHI Score (primarily from sensitivity improvement)     │
│  RISK: Medium (diffusion quality must be validated)                            │
│  EFFORT: Medium-High (train diffusion model + validate quality)                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Implementation sketch:**

```python
class DiffusionSpectrogramGenerator:
    """
    Lightweight DDPM for generating synthetic minority-class spectrograms.
    Trained on real minority-class spectrograms, generates novel samples.
    """
    def __init__(self, n_mels=64, time_frames=800, num_classes=4):
        self.unet = UNet2D(
            in_channels=1,           # 1-channel log-mel
            out_channels=1,
            model_channels=64,
            num_res_blocks=2,
            attention_resolutions=[4, 8],
            num_classes=num_classes,  # Class-conditional
        )
        self.diffusion = GaussianDiffusion(
            timesteps=50,            # Lightweight: 50 steps
            beta_schedule="cosine",
        )

    def train(self, minority_spectrograms, labels, epochs=50):
        """Train on minority class spectrograms only."""
        # Standard DDPM training: predict noise at random timestep
        ...

    def generate(self, class_label, n_samples):
        """Generate n_samples synthetic spectrograms for given class."""
        # Start from random noise, denoise for 50 steps
        # Class label conditions the generation
        samples = self.diffusion.sample(
            self.unet, shape=(n_samples, 1, 64, 800),
            class_label=class_label,
        )
        return samples  # [n_samples, 1, 64, 800]
```

---

### 8.3 Novel Strategy F: Mixture of Experts with Class-Specialized Routing

**Key Differentiator:** Instead of a single model processing all 4 classes, use a **Mixture of Experts (MoE)** architecture where different expert sub-networks specialize in different respiratory sound types. A learnable router directs each input to the most relevant experts.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│          STRATEGY F: MIXTURE OF EXPERTS (MoE) RESPIRATORY CLASSIFIER            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────── SHARED FEATURE EXTRACTOR ────────────────────────────────────┐ │
│  │  Input: Log-mel spectrogram [B, 1, 64, 800]                                 │ │
│  │  Shared CNN backbone (ResNet-18 first 3 layers)                              │ │
│  │  Output: shared features [B, 128, 8, 10]                                    │ │
│  └────────────────────────────────┬────────────────────────────────────────────┘ │
│                                   │                                              │
│                    ┌──────────────┴──────────────┐                               │
│                    ▼                             ▼                               │
│  ┌─────────── LEARNABLE ROUTER ──────────┐  ┌─── EXPERT POOL ────────────────┐ │
│  │  Input: shared features               │  │                                 │ │
│  │  GAP → FC → Softmax                   │  │  Expert 1: Crackle Specialist   │ │
│  │  Output: routing weights [B, N_experts]│  │  ├─ Conv layers tuned for       │ │
│  │                                       │  │  │  transient onset patterns     │ │
│  │  Top-K routing (K=2):                 │  │  └─ High-freq attention         │ │
│  │  Activate only top-2 experts per      │  │                                 │ │
│  │  sample for efficiency                │  │  Expert 2: Wheeze Specialist    │ │
│  │                                       │  │  ├─ Conv layers tuned for       │ │
│  │  Load balancing loss:                 │  │  │  tonal/whistling patterns     │ │
│  │  L_balance = λ · CV(routing_probs)²   │  │  └─ Band-pass attention         │ │
│  │  (coefficient of variation)           │  │                                 │ │
│  │                                       │  │  Expert 3: Both Specialist      │ │
│  └───────────────────────────────────────┘  │  ├─ Multi-scale features        │ │
│                                             │  └─ Crackle+Wheeze fusion       │ │
│                                             │                                 │ │
│                                             │  Expert 4: Normal Specialist    │ │
│                                             │  ├─ Smooth pattern detection    │ │
│                                             │  └─ Regular breathing features  │ │
│                                             │                                 │ │
│                                             │  Expert 5: Generalist           │ │
│                                             │  ├─ Broad receptive field       │ │
│                                             │  └─ Catches ambiguous cases     │ │
│                                             │                                 │ │
│                                             └────────┬────────────────────────┘ │
│                                                      │                          │
│                                                      ▼                          │
│  ┌─────────────── EXPERT AGGREGATION ──────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  For each sample x_i:                                                       │ │
│  │  ├─ Router outputs: w = [w1, w2, w3, w4, w5] (softmax over 5 experts)      │ │
│  │  ├─ Top-K=2: activate only experts with highest w values                    │ │
│  │  ├─ Each activated expert outputs: e_k(x_i) ∈ R^4                          │ │
│  │  └─ Final logits = Σ(w_k · e_k(x_i)) for top-K experts                     │ │
│  │                                                                             │ │
│  │  Total params: ~3.5M (5 experts × ~600K + router ~500K)                    │ │
│  │  Active params per sample: ~1.7M (2 experts × ~600K + router)              │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  TRAINING:                                                                       │
│  ├─ Standard cross-entropy + load balancing loss                                │
│  ├─ KD from teacher ensemble (same as other strategies)                         │ │
│  ├─ Expert dropout during training (randomly zero 1 expert)                     │ │
│  └─ Curriculum: train router first (freeze experts), then fine-tune all         │
│                                                                                  │
│  WHY THIS IS NOVEL:                                                            │
│  ├─ Existing papers use monolithic models (one model for all classes)           │
│  ├─ MoE allows specialization: different experts for different sound types      │
│  ├─ Router learns to detect "is this a crackle or wheeze or both?"              │
│  ├─ Top-K routing is computationally efficient (only 2/5 experts active)        │
│  ├─ No published ICBHI paper has used MoE architecture                          │
│  └─ MoE is proven in NLP (Mixtral, Switch Transformer) but unexplored for      │
│     respiratory sound classification                                             │
│                                                                                  │
│  EXPECTED GAIN: +2-4% ICBHI Score (from class specialization)                  │
│  RISK: Medium (router training can be unstable)                                │
│  EFFORT: Medium (architecture change, but compatible with existing pipeline)    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Implementation sketch:**

```python
class MoERespiratoryClassifier(nn.Module):
    """
    Mixture of Experts for respiratory sound classification.
    Each expert specializes in a different respiratory sound type.
    """
    def __init__(self, num_classes=4, num_experts=5, top_k=2, feat_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Shared backbone
        self.backbone = ResNet18Backbone(out_dim=feat_dim)

        # Router: decides which experts to activate
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, num_experts),
        )

        # Expert pool: each expert is a small classifier
        self.experts = nn.ModuleList([
            ExpertHead(feat_dim, num_classes) for _ in range(num_experts)
        ])

    def forward(self, x):
        # Shared features
        features = self.backbone(x)  # [B, feat_dim, H, W]

        # Router: compute routing weights
        routing_logits = self.router(features)  # [B, num_experts]
        routing_weights = F.softmax(routing_logits, dim=-1)

        # Top-K routing: activate only top-K experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # Renormalize

        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(features))  # [B, num_classes]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, num_classes]

        # Weighted combination of top-K experts
        logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
        for k in range(self.top_k):
            idx = top_k_indices[:, k]  # [B]
            w = top_k_weights[:, k]    # [B]
            for b in range(x.size(0]):
                logits[b] += w[b] * expert_outputs[b, idx[b]]

        return logits, routing_weights  # Return routing weights for load balancing loss

    def load_balancing_loss(self, routing_weights):
        """Encourage balanced expert utilization."""
        # Fraction of tokens routed to each expert
        avg_routing = routing_weights.mean(dim=0)
        # Coefficient of variation squared
        cv_squared = (avg_routing.std() / (avg_routing.mean() + 1e-8)) ** 2
        return 0.01 * cv_squared  # Small weight to avoid dominating main loss
```

---

### 8.4 Novel Strategy G: Patient-Aware Meta-Learning (Patient-as-Task)

**Key Differentiator:** Instead of treating ICBHI as a standard classification problem, treat **each patient as a separate "task"** in a meta-learning framework. This learns patient-invariant features that generalize to unseen patients — directly addressing the patient-wise split evaluation protocol.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│          STRATEGY G: PATIENT-AWARE META-LEARNING (PATIENT-AS-TASK)              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CORE INSIGHT:                                                                   │
│  The ICBHI evaluation uses a patient-wise split (train patients ≠ test patients) │
│  Standard training minimizes loss across ALL patients jointly                    │
│  → Model may learn patient-specific shortcuts rather than disease patterns       │
│                                                                                  │
│  Meta-learning treats each patient as a separate "task"                          │
│  → Model learns to ADAPT to new patients with just a few examples               │
│  → Forces learning of disease-relevant features, not patient-specific features   │
│                                                                                  │
│  ┌─────────────── EPISODIC TRAINING ───────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Each episode:                                                              │ │
│  │  ├─ Sample N_patients from training set (e.g., N=8 patients)                │ │
│  │  ├─ For each patient:                                                       │ │
│  │  │   ├─ Support set: K samples from this patient (e.g., K=5)                │ │
│  │  │   ├─ Query set: Q samples from this patient (e.g., Q=10)                 │ │
│  │  │   └─ Both sets have 4-class labels (Normal, Crackle, Wheeze, Both)       │ │
│  │  │                                                                          │ │
│  │  │  Inner loop (per patient):                                                │ │
│  │  │   ├─ Compute patient-specific prototypes from support set                │ │
│  │  │   ├─ Prototype = mean feature vector per class for this patient          │ │
│  │  │   └─ Classify query set by nearest prototype                             │ │
│  │  │                                                                          │ │
│  │  │  Outer loop (across all patients):                                        │ │
│  │  │   ├─ Compute meta-loss = avg query loss across all patients              │ │
│  │  │   └─ Update shared feature extractor (backbone) via meta-gradients       │ │
│  │  │                                                                          │ │
│  │  └─ Result: backbone learns features that are discriminative ACROSS         │ │
│  │     patients, not WITHIN individual patients                                │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌─────────────── ARCHITECTURE ────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Shared Feature Extractor (meta-learned):                                    │ │
│  │  ├─ ResNet/EfficientNet backbone                                            │ │
│  │  ├─ Maps spectrogram → feature vector [B, D]                                │ │
│  │  └─ Learns patient-invariant disease features                               │ │
│  │                                                                             │ │
│  │  Patient-Specific Prototypes (computed at test time):                        │ │
│  │  ├─ For test patient P:                                                     │ │
│  │  │   ├─ Extract features of P's training samples (if available)             │ │
│  │  │   ├─ Compute class prototypes: proto_c = mean(features of class c)       │ │
│  │  │   └─ Classify by cosine similarity to prototypes                         │ │
│  │  │                                                                          │ │
│  │  ├─ For unseen test patients (no training samples):                          │ │
│  │  │   ├─ Use global prototypes from all training patients                    │ │
│  │  │   └─ Standard nearest-prototype classification                            │ │
│  │  │                                                                          │ │
│  │  └─ Hybrid: blend patient-specific and global prototypes                    │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌─────────────── TEST-TIME ADAPTATION ────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  At test time, for each test patient:                                       │ │
│  │  ├─ Option 1: Use global prototypes (no adaptation)                         │ │
│  │  ├─ Option 2: If few labeled samples available, compute patient prototypes  │ │
│  │  └─ Option 3: TTT-style self-supervised adaptation                          │ │
│  │      ├─ Pseudo-label test samples with current model                        │ │
│  │      ├─ Update feature extractor with consistency loss                       │ │
│  │      └─ Re-classify with adapted features                                   │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  WHY THIS IS NOVEL:                                                            │
│  ├─ Existing papers treat ICBHI as standard classification                      │
│  ├─ PAFA uses patient-aware LOSSES but standard training                        │
│  ├─ Meta-learning (MAML/ProtoNet) has NOT been applied to ICBHI                │
│  ├─ Patient-as-task directly matches the evaluation protocol                    │
│  ├─ Test-time adaptation to unseen patients is novel for respiratory sounds     │
│  └─ This addresses the ROOT CAUSE of poor generalization: patient variability   │
│                                                                                  │
│  EXPECTED GAIN: +3-6% ICBHI Score (from better patient generalization)         │
│  RISK: High (meta-learning is complex, needs careful implementation)           │
│  EFFORT: High (new training paradigm, but reusable components)                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Implementation sketch:**

```python
class PatientMetaLearner(nn.Module):
    """
    Meta-learning framework where each patient is a task.
    Learns patient-invariant features via episodic training.
    """
    def __init__(self, backbone, feat_dim=256, num_classes=4):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def meta_train_episode(self, support_x, support_y, query_x, query_y,
                           patient_ids, inner_lr=0.01, inner_steps=5):
        """
        One meta-training episode.

        Args:
            support_x: [N_patients, K, C, H, W] - K support samples per patient
            support_y: [N_patients, K] - labels
            query_x: [N_patients, Q, C, H, W] - Q query samples per patient
            query_y: [N_patients, Q] - labels
        """
        meta_loss = 0

        for p in range(support_x.size(0)):  # For each patient
            # Inner loop: adapt to this patient's support set
            fast_weights = {k: v.clone() for k, v in self.backbone.named_parameters()}

            for step in range(inner_steps):
                # Forward on support set with current weights
                support_feat = self.backbone.forward_with_params(support_x[p], fast_weights)
                # Compute prototypes per class
                prototypes = self._compute_prototypes(support_feat, support_y[p])
                # Classify support set
                support_logits = self._nearest_prototype(support_feat, prototypes)
                inner_loss = F.cross_entropy(support_logits, support_y[p])
                # Update fast weights
                grads = torch.autograd.grad(inner_loss, fast_weights.values())
                fast_weights = {k: v - inner_lr * g for (k, v), g
                               in zip(fast_weights.items(), grads)}

            # Outer loop: evaluate on query set with adapted weights
            query_feat = self.backbone.forward_with_params(query_x[p], fast_weights)
            prototypes = self._compute_prototypes(
                self.backbone.forward_with_params(support_x[p], fast_weights),
                support_y[p]
            )
            query_logits = self._nearest_prototype(query_feat, prototypes)
            meta_loss += F.cross_entropy(query_logits, query_y[p])

        return meta_loss / support_x.size(0)

    def _compute_prototypes(self, features, labels):
        """Compute class prototypes (mean feature per class)."""
        prototypes = []
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                prototypes.append(features[mask].mean(dim=0))
            else:
                prototypes.append(torch.zeros(self.feat_dim, device=features.device))
        return torch.stack(prototypes)

    def _nearest_prototype(self, features, prototypes):
        """Classify by cosine similarity to prototypes."""
        sim = F.cosine_similarity(
            features.unsqueeze(1),      # [B, 1, D]
            prototypes.unsqueeze(0),     # [1, C, D]
            dim=2
        )  # [B, C]
        return sim
```

---

### 8.5 Novel Strategy H: Multi-Task Learning with Auxiliary Respiratory Tasks

**Key Differentiator:** Instead of only predicting the 4-class label, train the model to simultaneously predict **multiple auxiliary tasks** related to respiratory sounds. The shared representation learns richer features that benefit the main classification task.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│          STRATEGY H: MULTI-TASK LEARNING WITH AUXILIARY TASKS                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CORE INSIGHT:                                                                   │
│  The ICBHI dataset has metadata beyond just the 4-class label:                  │
│  ├─ Patient ID (126 patients)                                                   │
│  ├─ Recording device (different stethoscopes)                                   │
│  ├─ Recording location (trachea, anterior, posterior, lateral)                  │
│  ├─ Breathing phase (inspiration vs expiration)                                  │
│  └─ Age group (child vs adult, if available)                                    │
│                                                                                  │
│  By predicting these auxiliary tasks, the model learns a richer representation   │
│  that captures device-invariant, location-invariant disease features.            │
│                                                                                  │
│  ┌─────────────── ARCHITECTURE ────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Input: Log-mel spectrogram [B, 1, 64, 800]                                 │ │
│  │         │                                                                    │ │
│  │         ▼                                                                    │ │
│  │  ┌─── SHARED BACKBONE (ResNet/EfficientNet) ───┐                            │ │
│  │  │  Learns shared representation for ALL tasks  │                            │ │
│  │  │  Output: features [B, 256]                   │                            │ │
│  │  └──────────────┬──────────────────────────────┘                            │ │
│  │                 │                                                             │ │
│  │     ┌───────────┼───────────┬───────────┬───────────┐                        │ │
│  │     ▼           ▼           ▼           ▼           ▼                        │ │
│  │  ┌──────┐  ┌──────────┐  ┌──────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ Main │  │ Breathing│  │Device│  │ Patient  │  │ Anomaly  │              │ │
│  │  │ 4-cls│  │  Phase   │  │  ID  │  │   ID     │  │Detection │              │ │
│  │  │ Head │  │  (I/E)   │  │ Head │  │  Head    │  │  (Head)  │              │ │
│  │  │      │  │  Head    │  │      │  │          │  │          │              │ │
│  │  │ [256→4]│ │ [256→2] │  │[256→N]│ │ [256→126]│  │ [256→2] │              │ │
│  │  └──┬───┘  └────┬────┘  └──┬───┘  └────┬─────┘  └────┬─────┘              │ │
│  │     │           │          │           │             │                      │ │
│  │     ▼           ▼          ▼           ▼             ▼                      │ │
│  │  L_main     L_phase    L_device    L_patient     L_anomaly                  │ │
│  │  (CE)       (CE)       (CE)        (CE)          (CE)                       │ │
│  │                                                                             │ │
│  │  Total Loss = L_main + α·L_phase + β·L_device + γ·L_patient + δ·L_anomaly  │ │
│  │                                                                             │ │
│  │  Key: L_device and L_patient are ANTI-TASKS (train to predict, then         │ │
│  │  use adversarial gradient reversal to make features device/patient-INVARIANT │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌─────────────── ANTI-TASKS (DOMAIN ADVERSARIAL) ────────────────────────────┐ │
│  │                                                                             │ │
│  │  The model predicts device ID and patient ID, but with Gradient Reversal:  │ │
│  │  ├─ Forward pass: predict device/patient normally                           │ │
│  │  ├─ Backward pass: REVERSE the gradient (multiply by -λ)                    │ │
│  │  ├─ Effect: features become device-invariant and patient-invariant          │ │
│  │  └─ This forces the model to learn DISEASE features, not spurious cues     │ │
│  │                                                                             │ │
│  │  GradientReversalLayer:                                                     │ │
│  │  class GradReverse(torch.autograd.Function):                                │ │
│  │      @staticmethod                                                          │ │
│  │      def forward(ctx, x, lambda_val):                                       │ │
│  │          ctx.lambda_val = lambda_val                                        │ │
│  │          return x.clone()                                                   │ │
│  │      @staticmethod                                                          │ │
│  │      def backward(ctx, grad_output):                                        │ │
│  │          return -ctx.lambda_val * grad_output, None                         │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  WHY THIS IS NOVEL:                                                            │
│  ├─ No ICBHI paper uses multi-task learning with auxiliary respiratory tasks    │
│  ├─ Anti-tasks (device/patient invariance via gradient reversal) are novel      │
│  ├─ Breathing phase prediction forces temporal understanding                    │ │
│  ├─ Anomaly detection head provides binary abnormal/normal signal               │ │
│  └─ The shared backbone learns richer, more generalizable features              │
│                                                                                  │
│  EXPECTED GAIN: +2-4% ICBHI Score (from richer representation)                 │
│  RISK: Low-Medium (multi-task learning is well-established)                     │
│  EFFORT: Medium (need to extract auxiliary labels from ICBHI metadata)          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 8.6 Novel Strategy I: Adaptive Threshold Ensemble (Threshold as a Learned Parameter)

**Key Differentiator:** Instead of sweeping thresholds on validation data post-hoc, **learn the optimal threshold as part of the model**. This is different from all existing papers which tune thresholds after training.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│          STRATEGY I: LEARNED ADAPTIVE THRESHOLD                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PROBLEM WITH CURRENT APPROACH:                                                  │
│  All papers do: Train model → sweep threshold on validation → freeze threshold  │
│  ├─ Threshold is a fixed scalar, same for all samples                           │
│  ├─ Doesn't adapt to different patients, recording conditions, or classes        │
│  └─ Suboptimal: different samples may need different thresholds                  │
│                                                                                  │
│  NOVEL APPROACH: Learn a SAMPLE-ADAPTIVE threshold                              │
│                                                                                  │
│  ┌─────────────── ARCHITECTURE ────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Input: Log-mel spectrogram [B, 1, 64, 800]                                 │ │
│  │         │                                                                    │ │
│  │         ▼                                                                    │ │
│  │  ┌─── FEATURE EXTRACTOR ───┐                                                │ │
│  │  │  ResNet/EfficientNet     │                                                │ │
│  │  │  Output: features [B, D] │                                                │ │
│  │  └───────────┬─────────────┘                                                │ │
│  │              │                                                               │ │
│  │     ┌────────┴────────┐                                                     │ │
│  │     ▼                 ▼                                                      │ │
│  │  ┌──────────┐  ┌──────────────┐                                             │ │
│  │  │ Classifier│  │ Threshold    │                                             │ │
│  │  │ Head      │  │ Predictor    │                                             │ │
│  │  │ [D→4]     │  │ [D→1]        │                                             │ │
│  │  │           │  │ sigmoid → τ  │                                             │ │
│  │  │ logits    │  │              │                                             │ │
│  │  └─────┬────┘  └──────┬───────┘                                             │ │
│  │        │              │                                                      │ │
│  │        ▼              ▼                                                      │ │
│  │  ┌──────────────────────────────┐                                           │ │
│  │  │  Adaptive Prediction:        │                                           │ │
│  │  │  p_normal = softmax(logits)[0]│                                           │ │
│  │  │  τ = sigmoid(threshold_pred)  │   ← sample-specific threshold!           │ │
│  │  │                               │                                           │ │
│  │  │  if p_normal >= τ:            │                                           │ │
│  │  │      predict = Normal         │                                           │ │
│  │  │  else:                        │                                           │ │
│  │  │      predict = argmax(others) │                                           │ │
│  │  └──────────────────────────────┘                                           │ │
│  │                                                                             │ │
│  │  Training Loss:                                                             │ │
│  │  L = CE(logits, labels) + λ·ICBHI_score_loss(logits, τ, labels)            │ │
│  │                                                                             │ │
│  │  The threshold predictor learns: "for THIS specific input, what threshold   │ │
│  │  maximizes the ICBHI Score?"                                                │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  WHY THIS IS NOVEL:                                                            │
│  ├─ All existing papers use a FIXED threshold (same τ for all test samples)    │
│  ├─ This learns a SAMPLE-SPECIFIC threshold that adapts per input              │
│  ├─ Harder samples (ambiguous crackle/normal) get different thresholds          │ │
│  ├─ The threshold predictor learns patient-specific and class-specific          │ │
│  │   decision boundaries                                                        │ │
│  └─ No published ICBHI paper has used learned adaptive thresholds              │
│                                                                                  │
│  EXPECTED GAIN: +1-3% ICBHI Score (from better threshold adaptation)           │
│  RISK: Low-Medium (simple addition to existing architecture)                    │
│  EFFORT: Low (just add one FC layer + modify prediction logic)                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 8.7 Summary of Novel Methods

| Strategy | Novelty | Expected Gain | Risk | Published for ICBHI? |
|---|---|---:|---|---|
| **E: Diffusion Augmentation** | Generate realistic synthetic minority spectrograms | +3–5% | Medium | ❌ No |
| **F: Mixture of Experts** | Class-specialized expert routing | +2–4% | Medium | ❌ No |
| **G: Patient Meta-Learning** | Patient-as-task, learn patient-invariant features | +3–6% | High | ❌ No |
| **H: Multi-Task + Anti-Tasks** | Device/patient invariance via gradient reversal | +2–4% | Low-Med | ❌ No |
| **I: Learned Adaptive Threshold** | Sample-specific threshold prediction | +1–3% | Low | ❌ No |

### 8.8 Recommended Novel Combinations

For maximum differentiation from existing papers:

**Novel Combo 1 (High Impact):** Strategy G + E + I
- Meta-learning for patient generalization
- Diffusion augmentation for minority classes
- Learned adaptive threshold
- Expected: **72–78% ICBHI Score**

**Novel Combo 2 (Balanced):** Strategy F + H + I
- MoE for class specialization
- Multi-task with anti-tasks for invariance
- Learned adaptive threshold
- Expected: **70–75% ICBHI Score**

**Novel Combo 3 (Low Risk):** Strategy E + I + existing techniques
- Diffusion augmentation (biggest single gain)
- Learned adaptive threshold
- Combined with VTLP, MixUp, KD
- Expected: **69–73% ICBHI Score**

---

## 9. Implementation Roadmap (All Strategies)

### Phase 1: Quick Wins (1–2 weeks)

**Goal:** Push ICBHI Score from 0.634 to 0.66–0.68

| Step | Task | Expected Gain | Script |
|---|---|---:|---|
| 1.1 | Add VTLP augmentation to pipeline | +3.19% | `icbhi_sota_augmentation.py` |
| 1.2 | Add label smoothing to KD loss | +0.5–1.0% | `icbhi_sota_loss_functions.py` |
| 1.3 | Implement curated teacher ensemble (top-5) | +0.2–0.5% | `icbhi_kd_pipeline_multiview_ensemble.py` |
| 1.4 | Add multi-view TTA evaluation | +1.53% | `icbhi_sota_evaluation.py` |
| 1.5 | Implement SAM optimizer | +0.5–1.5% | Training loop |
| 1.6 | Fine threshold sweep optimization | +0.5–1.0% | `icbhi_sota_evaluation.py` |

**Run:** Strategy S5 (`icbhi_kd_s5_sota_combined.py`) — already combines most quick wins.

### Phase 2: Foundation Model (2–3 weeks)

**Goal:** Establish strong teacher models using pre-trained audio models

| Step | Task | Expected Gain | Script |
|---|---|---:|---|
| 2.1 | Download and integrate BEATs pre-trained weights | — | New script |
| 2.2 | Fine-tune BEATs on ICBHI with PAFA losses | +2–5% | New script |
| 2.3 | Fine-tune AST on ICBHI | +2–4% | `icbhi_kd_s4_transformer_mega_ensemble.py` |
| 2.4 | Evaluate foundation models as teachers | — | Evaluation script |

### Phase 3: Advanced Distillation (2–3 weeks)

**Goal:** Implement cross-architecture KD and advanced training dynamics

| Step | Task | Expected Gain | Script |
|---|---|---:|---|
| 3.1 | Implement cross-architecture feature bridge | +1–2% | New module |
| 3.2 | Implement Decoupled KD (DKD) | +1–2% | `icbhi_sota_loss_functions.py` |
| 3.3 | Implement curriculum learning | +1–2% | Training loop |
| 3.4 | Implement progressive KD weighting | +0.5–1% | Training loop |
| 3.5 | Implement EMA mean teacher | +1–2% | Training loop |
| 3.6 | Implement SWA | +0.5–1% | Training loop |

### Phase 4: Optimization & Validation (1–2 weeks)

**Goal:** Final tuning and validation

| Step | Task | Expected Gain | Script |
|---|---|---:|---|
| 4.1 | Hyperparameter sweep (loss weights, temperatures) | +1–2% | Sweep script |
| 4.2 | Multi-seed ensemble (3 seeds × best config) | +0.5–1% | Ensemble script |
| 4.3 | Ablation study for each technique | — | Ablation script |
| 4.4 | Final evaluation on official test set | — | `icbhi_sota_evaluation.py` |
| 4.5 | ONNX export for FPGA deployment | — | Export script |

---

## 10. Expected Results & Benchmarks

### 9.1 Comprehensive Results Table

| Strategy | 4-class ICBHI Score | 4-class Se | 4-class Sp | 2-class ICBHI Score | Params | Training Time |
|---|---:|---:|---:|---:|---:|---|
| **Baseline (E1)** | 63.4% | 38.0% | 88.9% | — | 1.2M | 2h |
| **Strategy A** | 68–72% | 44–50% | 90–93% | 74–79% | 90M* | 8h |
| **Strategy B** | 68–74% | 44–52% | 90–94% | 74–80% | 1.2M | 20h |
| **Strategy C** | 66–70% | 42–47% | 89–92% | 72–76% | 1.2M | 6h |
| **Strategy D** | 70–76% | 48–55% | 91–95% | 76–82% | 1.5M | 30h |
| **Target** | **>68%** | **>45%** | **>90%** | **>74%** | <2M | — |

*Strategy A teacher model is 90M, but the student is 1.2M (distilled).

### 9.2 Per-Class F1 Expectations

| Class | Baseline F1 | Strategy A F1 | Strategy B F1 | Strategy D F1 |
|---|---:|---:|---:|---:|
| Normal | 78–82% | 82–86% | 83–87% | 85–90% |
| Crackle | 45–50% | 52–58% | 53–60% | 58–65% |
| Wheeze | 40–45% | 48–55% | 50–57% | 55–62% |
| Both | 25–30% | 35–42% | 38–45% | 42–52% |

### 9.3 Comparison with Published SOTA

| Method | ICBHI Score | Our Target | Status |
|---|---:|---:|---|
| Patch-Mix CL (AST, INTERSPEECH 2023) | 62.37% | >68% | ✅ Exceed |
| ADD-RSC (2025) | 65.53% | >68% | ✅ Exceed |
| RSC-FTF (2025) | 67.55% | >68% | ✅ Exceed |
| Arch-Agnostic KD (2025) | 65.69% | >68% | ✅ Exceed |
| Meta-Ensemble (2026) | 66.49% | >68% | ✅ Exceed |
| **Project Target** | — | **>68%** | 🎯 |

---

## 11. Ablation Study Plan

### 10.1 Per-Technique Ablation

Run each technique incrementally on top of the baseline (E1) to measure individual contribution:

| Run | Techniques Added | Expected Δ Score |
|---|---|---:|
| A0 | Baseline (E1) | 0.634 (reference) |
| A1 | + VTLP augmentation | +3.0–3.5% |
| A2 | + Label smoothing KD | +0.5–1.0% |
| A3 | + Curated ensemble (top-5) | +0.2–0.5% |
| A4 | + MixUp (α=0.3) | +1.0–1.5% |
| A5 | + SAM optimizer | +0.5–1.5% |
| A6 | + Class-balanced focal loss | +0.5–1.0% |
| A7 | + Sensitivity-aware binary loss | +0.5–1.0% |
| A8 | + Multi-view TTA | +1.5–2.0% |
| A9 | + Fine threshold sweep | +0.5–1.0% |
| A10 | All combined (Strategy D) | +6–12% |

### 10.2 Teacher Ablation

| Run | Teacher Configuration | Expected Score |
|---|---|---:|
| T0 | 3 CNN teachers (baseline) | 63.4% |
| T1 | + AST teacher | 65–67% |
| T2 | + Swin teacher | 66–68% |
| T3 | + Fusion teacher | 67–69% |
| T4 | + BEATs (pre-trained) teacher | 68–72% |
| T5 | Top-5 curated from all | 69–73% |

### 10.3 Loss Component Ablation

| Run | Loss Components | Expected Score |
|---|---|---:|
| L0 | Focal + KD only (baseline) | 63.4% |
| L1 | + Binary auxiliary | 64–65% |
| L2 | + Feature distillation | 65–66% |
| L3 | + Attention transfer | 65.5–67% |
| L4 | + Relational KD | 66–67.5% |
| L5 | + Patient-aware loss | 66.5–68% |
| L6 | + Contrastive loss | 67–69% |
| L7 | All components (progressive) | 68–72% |

---

## 12. References

### Core Papers

1. Dong et al. (2025). "Adaptive Differential Denoising for Respiratory Sounds Classification."
2. Dong et al. (2025). "Respiratory sounds classification by fusing the time-domain and 2D spectral features." — RSC-FTF, ICBHI Score 67.55%.
3. Jeong & Kim (2025). "Patient-Aware Feature Alignment for Robust Lung Sound Classification." — PAFA, ICBHI Score 64.84% (4-class), 72.08% (2-class).
4. Toikkanen & Kim (2025). "Improving Respiratory Sound Classification with Architecture-Agnostic Knowledge Distillation from Ensembles." — ICBHI Score 65.69%.
5. Kim et al. (2026). "Meta-Ensemble Learning with Diverse Data Splits for Improved Respiratory Sound Classification." — ICBHI Score 66.49%.

### Knowledge Distillation

6. Hinton et al. (2015). "Distilling the Knowledge in a Neural Network." NeurIPS Workshop.
7. Park et al. (2019). "Relational Knowledge Distillation." CVPR.
8. Zagoruyko & Komodakis (2017). "Paying More Attention to Attention." ICLR.
9. Chen et al. (2021). "Cross-Layer Distillation with Semantic Calibration." AAAI.
10. Zhao et al. (2022). "Decoupled Knowledge Distillation." CVPR.

### Audio Models

11. Gong et al. (2021). "AST: Audio Spectrogram Transformer." InterSpeech.
12. Liu et al. (2021). "Swin Transformer." ICCV.
13. Chen et al. (2022). "BEATs: Audio Pre-Training with Acoustic Tokenizers." ICML 2023.
14. Huang et al. (2022). "Audio-MAE: Masked Autoencoders that Listen." NeurIPS.

### Training Techniques

15. Cui et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples." CVPR.
16. Tarvainen & Valpola (2017). "Mean Teachers are Better Role Models." NeurIPS.
17. Foret et al. (2021). "Sharpness-Aware Minimization for Efficiently Improving Generalization." ICLR.
18. Izmailov et al. (2018). "Averaging Weights Leads to Wider Optima and Better Generalization." UAI.
19. Bae et al. (2023). "Patch-Mix Contrastive Learning for Respiratory Sound Classification." INTERSPEECH.

### Loss Functions

20. Lin et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
21. Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
22. Tung & Mori (2019). "Similarity-Preserving Knowledge Distillation." ICCV.
23. Khosla et al. (2020). "Supervised Contrastive Learning." NeurIPS.

---

## Appendix A: Quick Reference — ICBHI Evaluation Protocol

```
Dataset: ICBHI 2017 Respiratory Sound Database
Samples: 6,898 respiratory cycles from 126 patients
Split: Patient-wise 60/40 (patients 101-160 train, 161-226 test)
Classes: Normal, Crackle, Wheeze, Both (Crackle+Wheeze)

Primary Metric:
  ICBHI Score = (Sensitivity + Specificity) / 2
  Sensitivity = TP_abnormal / (TP_abnormal + FN_abnormal)
  Specificity = TN_normal / (TN_normal + FP_normal)

Threshold Tuning:
  Predict Normal if P(Normal) >= τ, else argmax among abnormal classes
  τ tuned on validation set (coarse 1000pt + fine 500pt)

2-class Metrics (derived from 4-class):
  Normal vs Abnormal (Crackle ∪ Wheeze ∪ Both)
```

## Appendix B: Hyperparameter Reference

```python
# Augmentation
vtlp_alpha_range = (0.9, 1.1)
vtlp_apply_prob = 0.5
mixup_alpha = 0.3
mixup_prob = 0.5
spec_freq_mask = 16
spec_time_mask = 48

# Loss weights (Strategy D, epoch-dependent)
w_focal = 0.40 → 0.25
w_kd = 0.30 → 0.40
w_binary = 0.20 → 0.25
w_patient = 0.0 → 0.05
w_contrast = 0.05

# KD
temperature = 4.0
kd_smoothing = 0.15

# Training
optimizer = SAM (rho=0.05, adaptive=True)
lr_backbone = 5e-5
lr_classifier = 1e-3
weight_decay = 0.01
batch_size = 32
epochs = 100-150
scheduler = cosine_warm_restart (T_0=30, T_mult=2)
swa_start = epoch 80

# Class-balanced
beta = 0.9999
focal_gamma = 2.0-2.5
label_smoothing = 0.05

# Patient-aware
lambda_pcsl = 50.0
lambda_gpal = 0.0005
proj_dim = 64

# Ensemble
top_k_teachers = 5
tta_views = 7-10

# Threshold
coarse_points = 1000
fine_points = 500
fine_window = ±0.02
```
