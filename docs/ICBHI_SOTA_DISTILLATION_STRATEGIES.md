# ICBHI 2017 — Chiến lược Distillation đánh bại SOTA

**Mục tiêu: ICBHI Score > 70% (official 60/40 split)**

---

## 1. Phân tích các Paper SOTA cần đánh bại

### 1.1 Các paper tham khảo chính

| Paper | Phương pháp chính | ICBHI Score (ước tính) | Điểm mạnh | Điểm yếu |
|-------|-------------------|----------------------|-----------|----------|
| **Adaptive Differential Denoising** (2024) | Denoising + CNN/Transformer | ~68-72% | Xử lý nhiễu tốt, giữ tín hiệu bệnh lý | Không tận dụng ensemble, chỉ 1 view |
| **Improving Robustness & Clinical Applicability** (2024) | Audio Enhancement + Robust Training | ~67-71% | Kháng noise, tổng quát hóa tốt | Thiếu KD, chỉ dùng single model |
| **Respiratory Sound by Fusing Time-Domain & 2D Spectral** (2024) | Multi-modal fusion (time + spectral) | ~66-70% | Fusion 2 domain, bắt cả temporal + frequency | Không có KD, không có Transformer |
| **Enhancing with Optimal Feature Ensembles in CNNs** (2024) | Feature ensemble + optimal selection | ~67-71% | Ensemble features, chọn feature tối ưu | Chỉ CNN, không có Transformer teacher |

### 1.2 Các SOTA khác trên leaderboard

| Nhóm phương pháp | ICBHI Score | Ghi chú |
|-----------------|-------------|---------|
| Patch-Mix Contrastive Learning (Bae et al., INTERSPEECH 2023) | ~70-73% | Contrastive + MixUp |
| AST/HTS-AT fine-tuned (Transformer) | ~70-75% | Transformer trực tiếp |
| Dual-path Attention Networks | ~68-72% | Tần số + thời gian riêng biệt |
| Self-supervised (Audio-MAE, SSAST) + fine-tune | ~72-76% | Pre-training lớn |
| Knowledge Distillation ensemble | ~68-72% | Nhiều teacher → student |

---

## 2. Phân tích nguyên nhân tại sao SOTA hiện tại chưa đạt >75%

```
┌─────────────────────────────────────────────────────────────────┐
│                    NGUYÊN NHÂN CHÍH                             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Class imbalance nghiêm trọng                                  │
│    - Normal: ~60%, Crackle: ~20%, Wheeze: ~10%, Both: ~10%    │
│    - Model bias toward Normal → Sensitivity thấp               │
│                                                                 │
│ 2. Noise trong recording                                        │
│    - Ambient noise, breathing pattern variation                 │
│    - Denoising có thể loại bỏ tín hiệu bệnh lý               │
│                                                                 │
│ 3. Single model limitation                                      │
│    - 1 model khó capture tất cả pattern                        │
│    - CNN miss global context, Transformer miss local detail     │
│                                                                 │
│ 4. Official split 60/40 khó                                      │
│    - Patient-independent split                                  │
│    - Domain shift giữa train và test                            │
│                                                                 │
│ 5. Metric đặc thù                                               │
│    - ICBHI Score = (Se + Sp) / 2                               │
│    - Cần CẢ sensitivity VÀ specificity cao                     │
│    - Trade-off Se/Sp rất khó optimize đồng thời                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Tổng hợp TẤT CẢ phương pháp Distillation có thể áp dụng

### 3.1 Phân loại theo loại knowledge truyền

```
┌──────────────────────────────────────────────────────────────────────┐
│                   TAXONOMY OF KNOWLEDGE DISTILLATION                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─── Response-Based (Logit KD) ─────────────────────────────────┐  │
│  │  • Standard KD (Hinton et al., 2015)                          │  │
│  │    L_KD = KL(σ(z_s/T) || σ(z_t/T))                          │  │
│  │  • Born-Again Networks (BAN) — iterative self-KD              │  │
│  │  • Label Smoothing via Teacher                                │  │
│  │  • Decoupled KD (DKD) — tách target/non-target logit         │  │
│  │  • Class-Balanced KD — weight KD loss theo class              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─── Feature-Based (Feature KD) ───────────────────────────────┐  │
│  │  • FitNets (Romero et al., 2015) — match intermediate layer   │  │
│  │  • Attention Transfer (AT) (Zagoruyko, 2017) — match attn map │  │
│  │  • ReviewKD (Chen et al., 2021) — cross-layer feature align  │  │
│  │  • Feature Distillation Loss — cosine/MSE on projected feats  │  │
│  │  • NST (Passalis & Tefas, 2018) — match feature statistics   │  │
│  │  • PKT (Passalis & Tefas, 2018) — probabilistic KT           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─── Relational KD ────────────────────────────────────────────┐  │
│  │  • RKD (Park et al., CVPR 2019) — pairwise distance struct   │  │
│  │  • CRD (Tian et al., 2020) — contrastive representation      │  │
│  │  • IRG (Chen et al., 2021) — instance relation graph          │  │
│  │  • Similarity-Preserving KD (Tung & Mori, ICCV 2019)         │  │
│  │  • CC (Peng et al., 2019) — correlated congruence             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─── Adversarial KD ───────────────────────────────────────────┐  │
│  │  • Feature Discriminator — GAN-style feature matching         │  │
│  │  • Semantic Alignment — match feature distribution            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─── Self-Distillation ────────────────────────────────────────┐  │
│  │  • Born-Again Networks — train student = teacher architecture │  │
│  │  • Be Your Own Teacher (BYOT) — layer-wise self-KD           │  │
│  │  • CS-KD (Yun et al., 2020) — consistency self-KD            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─── Data-Efficient KD ────────────────────────────────────────┐  │
│  │  • TTA KD — test-time augmentation cho teacher logits         │  │
│  │  • Curriculum KD — easy→hard sample scheduling                │  │
│  │  • MixUp/CutMix KD — augmented soft targets                  │  │
│  │  • Noisy Student — add noise during student training          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─── Advanced / SOTA 2023-2024 ────────────────────────────────┐  │
│  │  • DiffKD (2024) — diffusion-based feature refinement         │  │
│  │  • GKD (2024) — generalized KD with on-policy training        │  │
│  │  • CrossKD (2024) — cross-head prediction distillation        │  │
│  │  • CTKD (2024) — curriculum temperature KD                    │  │
│  │  • DistiLLM (2024) — KD for large language models (N/A)       │  │
│  │  • Layer-wise Adaptive KD — per-layer loss weighting          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Chi tiết từng phương pháp áp dụng được

#### A. Response-Based KD (Logit-level)

| Method | Formula | Khi nào dùng | Trọng số đề xuất |
|--------|---------|---------------|------------------|
| **Standard KD** | `L = α·KL(p_s/T \|\| p_t/T)·T²` | Luôn dùng — baseline | 0.35-0.45 |
| **Decoupled KD (DKD)** | Tách `p_t` thành target logit + non-target logit, KD riêng | Khi class imbalance lớn | 0.30-0.40 |
| **Class-Balanced KD** | Weight KD loss theo `1/√n_c` | ICBHI có class imbalance | 0.35-0.45 |
| **Confidence-Weighted KD** | `w_i = 1 - max(p_t_i)` — hard sample focus | Khi teacher tự tin quá | 0.30-0.40 |
| **Multi-Temperature KD** | Dùng nhiều temperature khác nhau | Ensemble teacher đa dạng | 0.25-0.35 |

#### B. Feature-Level KD

| Method | Áp dụng cho | Ghi chú |
|--------|-------------|---------|
| **FitNets** | Teacher→Student feature projection | Cần match spatial dimension |
| **Attention Transfer (AT)** | `A = Σ_c \|F_c\|²` → normalize | Rất phù hợp cho Transformer→CNN |
| **ReviewKD** | Bi-directional cross-layer alignment | SOTA feature KD |
| **NST** | Match feature distribution statistics | Không cần spatial alignment |
| **PKT** | Probabilistic feature matching | Robust hơn MSE |

#### C. Relational KD

| Method | Mô tả | Phù hợp |
|--------|--------|---------|
| **RKD-Distance** | Match pairwise distance trong batch | Khi batch đủ lớn |
| **RKD-Angle** | Match triplet angle trong embedding | Bắt cấu trúc phức tạp hơn |
| **CRD** | Contrastive: positive/negative pairs | Cần memory bank lớn |
| **Similarity-Preserving** | Match similarity matrix `S_s ≈ S_t` | Đơn giản, hiệu quả |

#### D. Transformer-Specific KD

| Method | Mô tả | Tại sao quan trọng |
|--------|--------|--------------------|
| **Attention Map Transfer** | Transfer attention map từ Transformer layers | Transformer có attention map rất giàu informação |
| **CLS Token KD** | Match [CLS] embedding giữa teacher/student | Transformer dùng CLS cho classification |
| **Patch-to-Feature KD** | Map patch embeddings → CNN feature maps | Bridge Transformer-CNN gap |
| **Multi-Head Attention KD** | Transfer từng attention head riêng | Mỗi head học pattern khác nhau |

---

## 4. Kiến trúc đề xuất: Strategy S4 — Transformer-Enhanced Mega-Ensemble KD

### 4.1 Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STRATEGY S4: TRANSFORMER MEGA-ENSEMBLE KD                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────── TEACHER ENSEMBLE (7 models) ────────────────────────────┐ │
│  │                                                                             │ │
│  │  CNN Teachers (3)          Transformer Teachers (2)    Hybrid Teachers (2) │ │
│  │  ┌─────────────┐          ┌──────────────────┐       ┌──────────────────┐ │ │
│  │  │ ResNet-CNN  │          │ AST-Tiny         │       │ CNN-Transformer  │ │ │
│  │  │ ResNet-CRNN │          │ (Audio Spectrogram│       │ Hybrid (CNN     │ │ │
│  │  │ EfficientNet│          │  Transformer)     │       │  backbone +     │ │ │
│  │  │ B0          │          │                   │       │  Self-Attention)│ │ │
│  │  │             │          │ Swin-Tiny         │       │                  │ │ │
│  │  │             │          │ (Shifted Window   │       │ ResNet-CRNN-    │ │ │
│  │  │             │          │  Transformer)     │       │  Attention      │ │ │
│  │  └──────┬──────┘          └────────┬──────────┘       └────────┬─────────┘ │ │
│  │         │                          │                           │            │ │
│  │         └──────────┬───────────────┴───────────────────────────┘            │ │
│  │                    ▼                                                        │ │
│  │           TTA + Temperature Calibration                                     │ │
│  │           Reliability-Weighted Ensemble                                     │ │
│  └─────────────────────────┬───────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌──────────────────── KD TRANSFER ────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Level 1: Logit KD ──────── Standard KD + Decoupled KD + Class-Balanced   │ │
│  │  Level 2: Feature KD ────── ReviewKD-style cross-layer alignment           │ │
│  │  Level 3: Attention KD ──── Transfer Transformer attention → CNN features  │ │
│  │  Level 4: Relational KD ── RKD-Distance + RKD-Angle                       │ │
│  │  Level 5: Adversarial KD ─ Feature discriminator (GAN-style)               │ │
│  │  Level 6: Self-KD ──────── EMA teacher + Born-Again                       │ │
│  │                                                                             │ │
│  └─────────────────────────┬───────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌──────────────────── STUDENT TRAINING ───────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Student: DSCNNResSE (lightweight, FPGA-deployable)                        │ │
│  │                                                                             │ │
│  │  Augmentations:                                                             │ │
│  │  • MixUp + CutMix (on-the-fly)                                             │ │
│  │  • SpecAugment (freq_mask=16, time_mask=64)                                │ │
│  │  • TTA during evaluation (7 augmented views)                                │ │
│  │  • Curriculum Learning (easy→hard)                                          │ │
│  │  • Class-Balanced Effective-Number Sampling                                 │ │
│  │                                                                             │ │
│  │  Loss = w1·L_hard + w2·L_KD + w3·L_feat + w4·L_attn                      │ │
│  │        + w5·L_rkd + w6·L_adv + w7·L_ema + w8·L_bin                        │ │
│  │                                                                             │ │
│  │  Progressive weighting: hard→KD shift over epochs                           │ │
│  │  SWA (Stochastic Weight Averaging) for final model                         │ │
│  │                                                                             │ │
│  └─────────────────────────┬───────────────────────────────────────────────────┘ │
│                            │                                                     │
│                            ▼                                                     │
│  ┌──────────────────── INFERENCE ──────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  • Dual-Threshold Prediction (separate Se/Sp optimization)                  │ │
│  │  • TTA ensemble (7 views)                                                   │ │
│  │  • Multi-threshold voting                                                   │ │
│  │  • ONNX export for FPGA deployment                                         │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Chi tiết Transformer Teachers

#### AST-Tiny (Audio Spectrogram Transformer)

```
Input: Mel Spectrogram [B, 1, n_mels, time]
  │
  ├─ Patch Embedding: Conv2d(1, 192, kernel=16, stride=16)
  │  → [B, num_patches, 192]
  │
  ├─ Position Embedding (learnable)
  │
  ├─ Transformer Encoder × 4 layers
  │  ├─ Multi-Head Self-Attention (3 heads, dim=192)
  │  └─ FFN (192 → 768 → 192)
  │
  ├─ [CLS] Token → Classification Head
  │
  └─ Output: logits [B, nc]

Params: ~5.7M (tiny variant — phù hợp làm teacher)
```

#### Swin-Tiny (Shifted Window Transformer)

```
Input: Mel Spectrogram [B, 1, n_mels, time]
  │
  ├─ Patch Merging (4×4 patches)
  │
  ├─ Stage 1: 2× Swin Block (W-MSA + SW-MSA)
  │  └─ Window size = 7, heads = 3
  │
  ├─ Stage 2: 2× Swin Block
  │  └─ Patch Merging → resolution halved
  │
  ├─ Stage 3: 2× Swin Block
  │
  ├─ Global Average Pooling → FC
  │
  └─ Output: logits [B, nc]

Params: ~28M (tiny — lớn hơn AST, nhưng capture multi-scale tốt hơn)
```

### 4.3 Feature Alignment: Transformer → CNN

Vấn đề lớn nhất khi distill Transformer → CNN: **feature space khác nhau hoàn toàn**.

```
Transformer features: [B, num_patches, embed_dim] — sequential, global context
CNN features:         [B, C, H, W] — spatial, local patterns

Giải pháp:
┌─────────────────────────────────────────────────────────────┐
│           CROSS-ARCHITECTURE FEATURE BRIDGE                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Patch-to-Spatial Projection                              │
│     Transformer patches [B, N, D] → reshape → [B, D, H', W']│
│     Then project: Conv2d(D, C_target, 1) → match CNN dims   │
│                                                              │
│  2. Attention Map Transfer                                   │
│     Transformer attention [B, heads, N, N]                   │
│     → Average across heads → [B, N, N]                      │
│     → Reshape to spatial [B, 1, H', W']                     │
│     → Match with CNN attention map                           │
│                                                              │
│  3. CLS Token → Global Feature                               │
│     Transformer CLS [B, D] → FC → [B, C_global]             │
│     CNN global feature [B, C] → FC → [B, C_global]          │
│     → Match via cosine/MSE loss                              │
│                                                              │
│  4. Cross-Attention Alignment                                │
│     Query = CNN features, Key/Value = Transformer features   │
│     → Learn to attend Transformer knowledge from CNN         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Các technique từ paper tham khảo có thể áp dụng

### 5.1 Từ "Adaptive Differential Denoising"

```
Technique: Adaptive differential signal processing
├── Áp dụng: Pre-processing stage
├── Cách làm:
│   ├── Compute differential signal: d[n] = x[n] - x[n-1]
│   ├── Adaptive threshold based on local SNR
│   ├── Wavelet denoising with level-dependent threshold
│   └── Preserve frequency bands: 100-2000 Hz (lung sound range)
├── Integration: Trước khi extract mel spectrogram
└── Expected impact: +1-2% ICBHI Score (cleaner features)
```

### 5.2 Từ "Improving Robustness & Clinical Applicability"

```
Technique: Robust training with audio enhancement
├── Áp dụng: Data augmentation + training
├── Cách làm:
│   ├── Additive noise augmentation (SNR 5-20 dB)
│   ├── Room impulse response simulation
│   ├── Speed/pitch perturbation
│   ├── MixStyle — mix feature statistics across samples
│   └── Domain adversarial training (generalize across devices)
├── Integration: Augmentation pipeline
└── Expected impact: +2-3% ICBHI Score (better generalization)
```

### 5.3 Từ "Fusing Time-Domain & 2D Spectral Features"

```
Technique: Multi-modal feature fusion
├── Áp dụng: Teacher architecture design
├── Cách làm:
│   ├── Branch 1: 1D-CNN on raw waveform (temporal features)
│   ├── Branch 2: 2D-CNN on mel spectrogram (spectral features)
│   ├── Branch 3: 2D-CNN on MFCC delta (dynamic features)
│   ├── Attention-based fusion: learn branch importance
│   └── Concatenate + shared classifier
├── Integration: New teacher model "FusionTeacher"
└── Expected impact: +2-4% ICBHI Score (richer representation)
```

### 5.4 Từ "Enhancing with Optimal Feature Ensembles"

```
Technique: Optimal feature ensemble selection
├── Áp dụng: Teacher ensemble strategy
├── Cách làm:
│   ├── Extract features: Mel, MFCC, Chroma, Spectral Contrast
│   ├── Train separate CNN for each feature type
│   ├── Feature importance ranking via mutual information
│   ├── Select top-K features per class
│   └── Weighted ensemble based on validation performance
├── Integration: Multi-view teacher ensemble
└── Expected impact: +1-3% ICBHI Score (diverse representations)
```

---

## 6. Ma trận so sánh: KD methods nào nên dùng cùng nhau

```
┌────────────────────────────────────────────────────────────────────────┐
│                    COMPATIBILITY MATRIX                                 │
├──────────────┬─────┬──────┬──────┬──────┬──────┬──────┬──────┬────────┤
│ Method       │KD-L │Feat  │AT    │RKD   │Adv   │Self  │Curric│MixUp   │
├──────────────┼─────┼──────┼──────┼──────┼──────┼──────┼──────┼────────┤
│ Logit KD     │  -  │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓     │
│ Feature KD   │  ✓  │  -   │  ✓   │  ✓   │  △   │  ✓   │  ✓   │  ✓     │
│ Attn Transfer│  ✓  │  ✓   │  -   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓     │
│ Relational   │  ✓  │  ✓   │  ✓   │  -   │  ✓   │  △   │  ✓   │  ✓     │
│ Adversarial  │  ✓  │  △   │  ✓   │  ✓   │  -   │  ✓   │  ✓   │  ✓     │
│ Self-KD/EMA  │  ✓  │  ✓   │  ✓   │  △   │  ✓   │  -   │  ✓   │  ✓     │
│ Curriculum   │  ✓  │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓     │
│ MixUp        │  ✓  │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  -     │
└──────────────┴─────┴──────┴──────┴──────┴──────┴──────┴──────┴────────┘
  ✓ = Compatible  △ = Use with caution  ✗ = Conflicting
```

---

## 7. Roadmap đạt ICBHI Score > 70%

### Phase 1: Foundation (S1-S3 hiện tại → target 66-68%)
```
✓ S1: TTA + Calibrated Teacher + MixUp
✓ S2: Feature Attention + Adversarial KD
✓ S3: Curriculum + EMA + Class-Balanced
```

### Phase 2: Transformer Teacher (S4 → target 68-72%)
```
□ Thêm AST-Tiny teacher
□ Thêm Swin-Tiny teacher
□ Cross-architecture feature bridge (Transformer→CNN)
□ Attention map transfer từ Transformer
□ Multi-teacher ensemble (5-7 teachers)
```

### Phase 3: Advanced Techniques (S5 → target 70-75%)
```
□ Fusion teacher (time-domain + spectral)
□ Decoupled KD (DKD) — tách target/non-target
□ DiffKD-style feature refinement
□ Adaptive differential denoising preprocessing
□ Robust training (noise augmentation, domain adversarial)
□ Dual-threshold + TTA + SWA inference
```

### Phase 4: Optimization & Ensemble (S6 → target 73-78%)
```
□ Hyperparameter tuning (loss weights, temperatures)
□ Multi-seed ensemble (3 seeds × 7 teachers = 21 models)
□ Test-time augmentation with 10+ views
□ Threshold calibration on validation set
□ Knowledge distillation chain: Big→Medium→Small
```

---

## 8. Implementation Priority

### Mức ưu tiên cao (impact/effort ratio lớn)

| # | Technique | Expected gain | Effort |
|---|-----------|---------------|--------|
| 1 | Thêm Transformer teacher (AST-Tiny) | +2-4% | Medium |
| 2 | Multi-teacher ensemble (5+ teachers) | +1-3% | Low |
| 3 | Attention Transfer từ Transformer | +1-2% | Medium |
| 4 | Decoupled KD (DKD) | +1-2% | Low |
| 5 | Class-Balanced KD weighting | +1-2% | Low |

### Mức ưu tiên trung bình

| # | Technique | Expected gain | Effort |
|---|-----------|---------------|--------|
| 6 | Fusion teacher (time + spectral) | +2-3% | High |
| 7 | Curriculum + Progressive KD | +1-2% | Medium |
| 8 | Robust noise augmentation | +1-2% | Medium |
| 9 | Adversarial feature matching | +0.5-1% | Medium |
| 10 | Cross-architecture feature bridge | +1-2% | High |

### Mức ưu tiên thấp (nice-to-have)

| # | Technique | Expected gain | Effort |
|---|-----------|---------------|--------|
| 11 | DiffKD-style refinement | +0.5-1% | High |
| 12 | Adaptive differential denoising | +1-2% | High |
| 13 | Domain adversarial training | +0.5-1% | High |
| 14 | Self-supervised pre-training | +2-4% | Very High |

---

## 9. Loss Function tổng hợp

```python
# S4 Loss = 8 components
L_total = (
    w_hard    * L_focal_class_balanced      # Hard label (class-balanced focal)
  + w_kd      * L_logit_kd                  # Standard logit KD (T-scaled)
  + w_dkd     * L_decoupled_kd              # Decoupled KD (target + non-target)
  + w_feat    * L_feature_distillation      # Feature-level (cosine/MSE)
  + w_attn    * L_attention_transfer        # Attention map matching
  + w_rkd     * L_relational_kd             # Pairwise distance structure
  + w_adv     * L_adversarial               # GAN-style feature matching
  + w_ema     * L_ema_teacher               # EMA self-distillation
  + w_bin     * L_binary_auxiliary           # Binary (normal/abnormal) auxiliary
)

# Progressive weighting schedule
# Epoch 1-30:   hard=0.40, kd=0.25, feat=0.05, attn=0.05, ...
# Epoch 31-80:  hard=0.25, kd=0.35, feat=0.10, attn=0.05, ...
# Epoch 81-150: hard=0.20, kd=0.40, feat=0.10, attn=0.05, ...
```

---

## 10. Benchmark dự kiến

```
┌────────────────────────────────────────────────────────────────────┐
│                    EXPECTED RESULTS                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Strategy          │ Se (%) │ Sp (%) │ Score │ So với SOTA         │
│  ──────────────────┼────────┼────────┼───────┼──────────────────── │
│  Baseline (E1)     │  58.0  │  85.0  │ 66.5  │ Trung bình          │
│  S1 (TTA+Calib)    │  60.0  │  87.0  │ 73.5  │ Khả quan            │
│  S2 (FeatAttn)     │  61.0  │  86.0  │ 73.5  │ Khả quan            │
│  S3 (Curr+EMA)     │  59.0  │  88.0  │ 73.5  │ Khả quan            │
│  ──────────────────┼────────┼────────┼───────┼──────────────────── │
│  S4 (Trans.Ens.)   │  64.0  │  89.0  │ 76.5  │ ★ Beat SOTA ★      │
│  S4 + TTA + SWA    │  66.0  │  90.0  │ 78.0  │ ★★ Beat SOTA ★★    │
│  ──────────────────┼────────┼────────┼───────┼──────────────────── │
│  SOTA target       │  70.0  │  90.0  │ 80.0  │ Ultimate goal       │
│                                                                     │
│  Note: Se = Sensitivity, Sp = Specificity                          │
│  ICBHI Score = (Se + Sp) / 2                                       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 11. Tham khảo

### Papers chính
1. Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
2. Zagoruyko & Komodakis, "Paying More Attention to Attention", ICLR 2017
3. Park et al., "Relational Knowledge Distillation", CVPR 2019
4. Chen et al., "Cross-Layer Distillation with Semantic Calibration", AAAI 2021
5. Gong et al., "AST: Audio Spectrogram Transformer", InterSpeech 2021
6. Liu et al., "Swin Transformer", ICCV 2021
7. Bae et al., "Patch-Mix Contrastive Learning for Respiratory Sound", INTERSPEECH 2023
8. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
9. Tarvainen & Valpola, "Mean Teachers are Better Role Models", NeurIPS 2017

### Papers target cần đánh bại
10. "Adaptive Differential Denoising for Respiratory Sounds Classification"
11. "Improving the Robustness and Clinical Applicability of Respiratory Sound Classification with Audio Enhancement"
12. "Respiratory sounds classification by fusing the time-domain and 2D spectral features"
13. "Enhancing Respiratory Sound Classification with Optimal Feature Ensembles in CNNs"

### SOTA KD 2023-2024
14. "DiffKD: Diffusion-based Knowledge Distillation" (2024)
15. "GKD: Generalized Knowledge Distillation" (2024)
16. "CrossKD: Cross-Head Knowledge Distillation" (2024)
17. "CTKD: Curriculum Temperature Knowledge Distillation" (2024)
18. "Decoupled Knowledge Distillation" (CVPR 2022)
