# ICBHI 2017 KD SOTA Strategies V2 — Achieving >66 ICBHI Score

## 1. Current Results Gap Analysis

### 1.1 Baseline Results (E1-E3)

| Experiment | Test ICBHI Score | Test Sensitivity | Test Specificity | Val ICBHI Score |
|---|---:|---:|---:|---:|
| E1 (Calibrated Ensemble) | 0.634 | 0.380 | 0.889 | 0.691 |
| E2 (PatchMix Distill) | 0.610 | 0.376 | 0.844 | 0.695 |
| E3 (PAFA Relational) | 0.614 | 0.411 | 0.817 | 0.698 |

### 1.2 Targets

| Metric | Target | Current Best (E1) | Gap |
|---|---:|---:|---:|
| ICBHI Score | > 0.66 | 0.634 | +0.026 |
| Specificity | > 0.90 | 0.889 | +0.011 |
| Sensitivity | ~0.43+ | 0.380 | +0.050 |

### 1.3 Root Cause Analysis

**Why sensitivity is low:**
- The model is over-conservative: threshold is set high to maximize specificity, but this severely penalizes sensitivity.
- Teacher ensemble quality may not provide enough signal for abnormal classes (Crackle, Wheeze, Both).
- The `Both` class (Crackle+Wheeze) is the hardest and smallest class — its F1 is a bottleneck.

**Why specificity is near target but not above:**
- 0.889 is close to 0.90, but the threshold tuning is already aggressive.
- Need more robust normal-class representation in the student.

**Key insight:**
> ICBHI Score = (Sensitivity + Specificity) / 2. To reach 0.66, we need approximately Sensitivity=0.42 + Specificity=0.90 = 0.66. The main lever is **improving sensitivity by ~4-5 percentage points** while maintaining or slightly improving specificity.

---

## 2. Literature Review — Q1 Papers and SOTA Techniques

### 2.1 Knowledge Distillation for Imbalanced Medical Audio

| Paper | Venue | Year | Key Technique |
|---|---|---|---|
| Hinton et al., "Distilling Knowledge in a Neural Network" | NeurIPS Workshop | 2015 | Foundation KD with temperature scaling |
| Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" | CVPR | 2019 | Class-balanced weighting for imbalanced data |
| Park et al., "Relational Knowledge Distillation" | CVPR | 2019 | Pairwise distance matching in embedding space |
| Zagoruyko & Komodakis, "Paying More Attention to Attention" | ICLR | 2017 | Attention transfer from teacher to student |
| Tarvainen & Valpola, "Mean Teachers are Better Role Models" | NeurIPS | 2017 | EMA teacher for self-training |
| Guo et al., "On Calibration of Modern Neural Networks" | ICML | 2017 | Temperature scaling for calibration |
| Tung & Mori, "Similarity-Preserving Knowledge Distillation" | ICCV | 2019 | Similarity matrix matching |
| Chen et al., "Cross-Layer Distillation with Semantic Calibration" | AAAI | 2021 | Multi-layer feature distillation |
| Passalis & Tefas, "Learning Deep Representations with Probabilistic Knowledge Transfer" | ECCV | 2018 | Probabilistic KD |
| Mirzadeh et al., "Improved Knowledge Distillation via Teacher Assistant" | AAAI | 2020 | Multi-step distillation |

### 2.2 Respiratory Sound Classification SOTA

| Paper | Venue | Year | ICBHI Score | Method |
|---|---|---|---:|---|
| Bae et al., "Patch-Mix Contrastive Learning" | INTERSPEECH | 2023 | 62.37% | AST + PatchMix + Contrastive |
| PAFA (Patient-Aware Feature Alignment) | — | 2024 | ~72% (2-class) | BEATs + PAFA losses |
| Various AST-based methods | Various | 2023-2025 | 60-66% | Audio Spectrogram Transformer |
| CNN-CRNN ensemble methods | Various | 2022-2024 | 58-64% | CNN + BiLSTM + KD |

### 2.3 Key SOTA Techniques for ICBHI

1. **Audio-pretrained backbones** (BEATs, PANNs, AST) — pretrained on AudioSet
2. **Patient-aware feature alignment** (PAFA) — reduces patient-device bias
3. **Patch-Mix augmentation** — localized spectrogram mixing for robustness
4. **Test-Time Augmentation (TTA)** — multiple augmented inference views averaged
5. **Temperature scaling calibration** — per-model optimal temperature on validation
6. **Class-balanced loss** — effective number of samples weighting
7. **Curriculum learning** — easy-to-hard sample ordering
8. **EMA/Mean Teacher** — smooth teacher from student's own trajectory
9. **Multi-level KD** — logit + feature + attention + relational
10. **Adversarial training** — discriminator-based feature matching
11. **SWA** — stochastic weight averaging for better generalization
12. **Dual-threshold prediction** — separate sensitivity/specificity optimization

---

## 3. Three New KD Strategies

### Strategy 1: TTA-Augmented Calibrated Teacher KD

**File:** `python/training/icbhi_kd_s1_tta_calibrated.py`

**Output:** `artifacts/training/icbhi_kd_s1_tta_calibrated/`

**Core ideas:**
1. **Test-Time Augmentation for Teacher Logits** — When collecting teacher soft targets, run each teacher N=5 times with slight noise/time-shift augmentations and average the logits. This produces more robust and smoother teacher targets.
2. **Temperature Scaling Calibration** — Before building the ensemble, find the optimal temperature for each teacher on the validation set using NLL minimization. This corrects over/under-confident teachers.
3. **MixUp Data Augmentation** — Apply MixUp (alpha=0.3) during student training with 50% probability. MixUp creates virtual training samples between classes, improving generalization on minority abnormal classes.
4. **Class-Balanced Effective Number Sampling** — Use Cui et al.'s effective number of samples (beta=0.9999) for the WeightedRandomSampler, replacing the current raw-count-based sampling.
5. **Sensitivity-Aware Loss Rebalancing** — Increase binary auxiliary loss weight from 0.20 to 0.25, and increase the hard-label component from the teacher's binary target (bin_teacher_ratio=0.4, meaning 60% hard label, 40% teacher). This pushes the model to better detect abnormal samples.
6. **Cosine Annealing with Warm Restarts** — Replace standard cosine annealing with warm restarts (T_0=30, T_mult=2) for better exploration of the loss landscape.
7. **TTA at Evaluation** — Run student inference with 7 augmented views and average for final prediction.

**Expected improvement:**
- TTA logits → smoother teacher targets → better student learning (+1-2% ICBHI)
- MixUp → better abnormal class generalization (+1-2% sensitivity)
- Calibration → more reliable teacher confidence → better threshold tuning (+0.5-1%)
- Class-balanced sampling → better minority class recall (+1% sensitivity)
- **Total expected: ICBHI Score +3-5%, reaching ~0.66-0.68**

**Loss function:**
```
L = 0.30 * FocalLoss_CB
  + 0.45 * KL_Div_KD
  + 0.25 * BCE_Binary_Auxiliary
```

**Run command:**
```bash
python3 python/training/icbhi_kd_s1_tta_calibrated.py \
  --num_classes 4 \
  --benchmark_protocol official_icbhi \
  --teacher_arches resnet_cnn,resnet_crnn,efficientnet_b0 \
  --student_arch ds_cnn_res_se \
  --seeds 1,2,3 \
  --epochs_student 150 \
  --batch_size 32 \
  --lr_student 1e-3 \
  --temperature 4.0 \
  --focal_gamma 2.0 \
  --label_smoothing 0.08 \
  --export_onnx
```

---

### Strategy 2: Feature-Level Attention KD with Adversarial Training

**File:** `python/training/icbhi_kd_s2_feature_attention.py`

**Output:** `artifacts/training/icbhi_kd_s2_feature_attention/`

**Core ideas:**
1. **Intermediate Feature Distillation (ReviewKD-style)** — Instead of only matching final logits, also match spatial feature maps at 3 levels (early, mid, late) between teacher and student using learned projection heads. This transfers richer structural information.
2. **Attention Transfer (AT)** — Compute spatial attention maps (sum of squared activations across channels) from teacher and student features, and minimize their MSE. This forces the student to attend to the same spatial regions as the teacher.
3. **Relational Knowledge Distillation (RKD)** — Match pairwise distance structure in the embedding space. If two samples are close in teacher's feature space, they should also be close in student's feature space.
4. **Adversarial Feature Matching** — A small discriminator tries to distinguish teacher vs student features. The student is trained to fool it, creating a GAN-like training dynamic that pushes student features to match the teacher's distribution.
5. **Multi-Level Loss Combination:**
   ```
   L = 0.25 * FocalLoss
     + 0.35 * KL_KD
     + 0.10 * Feature_Distillation
     + 0.05 * Attention_Transfer
     + 0.05 * Relational_KD
     + 0.03 * Adversarial_Loss
     + 0.15 * Binary_Auxiliary
   ```

**Expected improvement:**
- Feature KD → transfers structural knowledge beyond logits (+2-3% ICBHI)
- Attention transfer → student focuses on correct abnormal regions (+1-2% sensitivity)
- RKD → preserves relative sample relationships (+0.5-1%)
- Adversarial → matches feature distributions (+0.5-1%)
- **Total expected: ICBHI Score +3-6%, reaching ~0.66-0.69**

**Run command:**
```bash
python3 python/training/icbhi_kd_s2_feature_attention.py \
  --num_classes 4 \
  --benchmark_protocol official_icbhi \
  --teacher_arches resnet_cnn,resnet_crnn,efficientnet_b0 \
  --student_arch ds_cnn_res_se \
  --seeds 1,2,3 \
  --epochs_student 150 \
  --batch_size 32 \
  --lr_student 1e-3 \
  --temperature 4.0 \
  --focal_gamma 2.0 \
  --export_onnx
```

---

### Strategy 3: Curriculum + EMA Teacher + Class-Balanced KD

**File:** `python/training/icbhi_kd_s3_curriculum_ema.py`

**Output:** `artifacts/training/icbhi_kd_s3_curriculum_ema/`

**Core ideas:**
1. **Curriculum Learning** — Sort training samples by difficulty (1 - max teacher probability). Start training with the easiest 40% of samples, gradually include all samples over 25 epochs. This prevents the student from being overwhelmed by hard ambiguous samples early in training.
2. **EMA (Mean Teacher)** — Maintain an Exponential Moving Average copy of the student (decay=0.998). After warmup, use the EMA model as an additional teacher alongside the external teacher ensemble. The EMA teacher provides smooth, evolving targets that improve with the student.
3. **Class-Balanced Focal Loss** — Replace standard focal loss with class-balanced focal loss using effective number of samples (Cui et al., CVPR 2019). This gives appropriate weight to the rare `Both` class without over-sampling.
4. **Progressive KD Weighting** — Start with higher hard-label weight (0.45) and lower KD weight (0.35), gradually shift to lower hard (0.28) and higher KD (0.47) over 30 epochs. This lets the student first learn basic classification from hard labels, then increasingly benefit from teacher soft targets.
5. **Dual-Threshold Prediction** — Instead of a single threshold for Normal vs Abnormal, use two thresholds: one for strong normal signal and one for strong abnormal signal. Sweep both on validation to maximize ICBHI Score.
6. **Stochastic Weight Averaging (SWA)** — In the last 20 epochs, average model weights across epochs for a smoother loss landscape and better generalization.

**Expected improvement:**
- Curriculum → stable early training, better convergence (+1-2% ICBHI)
- EMA teacher → smooth evolving targets, self-improving (+1-2%)
- Class-balanced focal → better minority class recall (+1-2% sensitivity)
- Progressive weights → optimal hard/KD balance (+0.5-1%)
- Dual-threshold → better sensitivity/specificity trade-off (+1-2%)
- SWA → better generalization (+0.5-1%)
- **Total expected: ICBHI Score +4-7%, reaching ~0.67-0.70**

**Loss function (at epoch e):**
```
w_hard(e) = 0.45 -> 0.28  (linear over 30 epochs)
w_kd(e)   = 0.35 -> 0.47
w_bin(e)  = 0.20 -> 0.25

L(e) = w_hard(e) * ClassBalancedFocalLoss
     + w_kd(e)   * KL_KD_external_teacher
     + w_bin(e)  * BCE_Binary_Auxiliary
     + 0.15      * KL_KD_EMA_teacher  (after warmup)
```

**Run command:**
```bash
python3 python/training/icbhi_kd_s3_curriculum_ema.py \
  --num_classes 4 \
  --benchmark_protocol official_icbhi \
  --teacher_arches resnet_cnn,resnet_crnn,efficientnet_b0 \
  --student_arch ds_cnn_res_se \
  --seeds 1,2,3 \
  --epochs_student 150 \
  --batch_size 32 \
  --lr_student 1e-3 \
  --temperature 4.0 \
  --focal_gamma 2.0 \
  --export_onnx
```

---

## 4. Comparison of Three Strategies

| Feature | S1: TTA+Calibrated | S2: Feature+Attention | S3: Curriculum+EMA |
|---|---|---|---|
| **Main innovation** | TTA teacher logits + MixUp | Multi-level feature KD | Curriculum + EMA teacher |
| **Teacher quality** | TTA-averaged, calibrated | Standard (feature-extracted) | Standard + EMA student |
| **Student augmentation** | MixUp + stronger SpecAugment | Standard | Standard |
| **KD levels** | Logit only (calibrated) | Logit + Feature + Attention + RKD + Adversarial | Logit (external + EMA) |
| **Sampling** | Class-balanced effective number | Standard weighted | Curriculum + class-balanced |
| **Loss balancing** | Static (sensitivity-aware) | Static (multi-term) | Progressive (adaptive) |
| **Inference** | TTA + threshold sweep | Standard threshold | Dual-threshold + SWA |
| **Complexity** | Low-Medium | High | Medium |
| **Expected ICBHI** | 0.66-0.68 | 0.66-0.69 | 0.67-0.70 |
| **Risk** | Low (mostly additive) | Medium (feature alignment may fail) | Low-Medium (curriculum may be too conservative) |

---

## 5. Recommended Experiment Order

### Phase 1: Quick Wins (S1)
Run S1 first because it has the lowest implementation risk and adds improvements incrementally on top of E1. If TTA logits + MixUp + calibration work, this alone may reach the target.

### Phase 2: Deep KD (S2)
Run S2 to explore multi-level distillation. If feature-level KD shows improvement over logit-only KD, this validates the approach and can be combined with S1's techniques.

### Phase 3: Advanced Training (S3)
Run S3 for the most sophisticated training dynamics. Curriculum + EMA + progressive weighting may provide the largest improvement, especially for the hard `Both` class.

### Phase 4: Combine Best
If any strategy shows promise, combine the best components:
- S1's TTA logits + MixUp
- S2's attention transfer
- S3's EMA teacher + dual-threshold

---

## 6. Verification Checklist

Before claiming SOTA, verify:

- [ ] Patient-wise split is correct (no leakage between train/test)
- [ ] Threshold is tuned on validation only, not on test
- [ ] ICBHI Score is computed correctly: (Sensitivity + Specificity) / 2
- [ ] Sensitivity = correctly predicted Abnormal / all true Abnormal
- [ ] Specificity = correctly predicted Normal / all true Normal
- [ ] Both 4-class and 2-class metrics are reported
- [ ] Mean ± std over multiple seeds is reported
- [ ] Per-class F1 is reported (especially Both class)
- [ ] Confusion matrix is saved
- [ ] Model parameter count is reported
- [ ] ONNX export is verified

---

## 7. SOTA Reference Benchmarks

| Method | Task | ICBHI Score | Sensitivity | Specificity |
|---|---|---:|---:|---:|
| Patch-Mix CL (AST) | 4-class | 62.37% | — | — |
| E1 (Calibrated Ensemble) | 4-class | 63.4% | 38.0% | 88.9% |
| **Target SOTA** | **4-class** | **>66%** | **>42%** | **>90%** |
| S1 (TTA+Calibrated) | 4-class | ~66-68% | ~42-45% | ~89-91% |
| S2 (Feature+Attention) | 4-class | ~66-69% | ~42-47% | ~88-91% |
| S3 (Curriculum+EMA) | 4-class | ~67-70% | ~43-48% | ~89-92% |

---

## 8. References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. NeurIPS Workshop.
2. Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. CVPR.
3. Park, W., Kim, D., Lu, Y., & Cho, M. (2019). Relational Knowledge Distillation. CVPR.
4. Zagoruyko, S., & Komodakis, N. (2017). Paying More Attention to Attention. ICLR.
5. Tarvainen, A., & Valpola, H. (2017). Mean Teachers are Better Role Models. NeurIPS.
6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML.
7. Tung, F., & Mori, G. (2019). Similarity-Preserving Knowledge Distillation. ICCV.
8. Chen, D., et al. (2021). Cross-Layer Distillation with Semantic Calibration. AAAI.
9. Mirzadeh, S. I., et al. (2020). Improved Knowledge Distillation via Teacher Assistant. AAAI.
10. Izmailov, P., et al. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. UAI.
11. Bae, J., et al. (2023). Patch-Mix Contrastive Learning for Respiratory Sound Classification. INTERSPEECH.
12. PAFA: Patient-Aware Feature Alignment for Lung Sound Classification. (2024).
