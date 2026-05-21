# 📊 Báo Cáo Phân Tích Overfitting & Chiến Lược Cải Thiện
## MobileNetV2 — 3-Class Respiratory Sound Classification

---

## 1. Tổng Quan Kết Quả Hiện Tại

| Metric | Mean ± Std (5-Fold CV) |
|--------|----------------------|
| **Accuracy** | 72.21% ± 4.15% |
| **Macro F1** | 57.08% ± 8.08% |
| **Weighted F1** | 72.84% ± 5.86% |

### Per-Class Performance (Best Fold — Fold 3)

| Class | Precision | Recall | F1-Score | Nhận xét |
|-------|-----------|--------|----------|----------|
| **COPD** | 0.979 | 0.870 | 0.921 | ✅ Tốt (majority class) |
| **Healthy** | 0.419 | 0.581 | 0.487 | ❌ Rất tệ |
| **Non-COPD** | 0.539 | 0.593 | 0.565 | ⚠️ Kém |

> [!CAUTION]
> Model đang **thiên vị nặng về class COPD** (F1=0.92) trong khi gần như không phân biệt được Healthy (F1=0.49) và Non-COPD (F1=0.56). Đây là dấu hiệu của class imbalance + overfitting.

---

## 2. Chẩn Đoán Overfitting — Phân Tích Chi Tiết

### 2.1 Biểu Hiện Overfitting Qua Các Fold

| Fold | Train Acc (cuối) | Val Acc (cuối) | **Gap** | Train Loss | Val Loss | **Loss Gap** | Early Stop Epoch |
|------|-----------------|-----------------|---------|------------|----------|-------------|------------------|
| 1 | 90.0% | 60.0% | **30.0%** | 0.284 | 1.385 | **1.101** | 51 |
| 2 | 87.9% | 68.6% | **19.3%** | 0.305 | 1.140 | **0.835** | 50 |
| 3 | 89.3% | 73.0% | **16.3%** | 0.287 | 0.979 | **0.692** | 50 |
| 4 | 89.3% | 68.6% | **20.7%** | 0.277 | 1.000 | **0.723** | 52 |
| 5 | 90.9% | 81.3% | **9.6%** | 0.257 | 0.663 | **0.406** | 55 |

> [!IMPORTANT]
> **Generalization gap trung bình: ~19%** (Train Acc vs Val Acc). Val Loss **tăng liên tục** sau Phase 2 bắt đầu, trong khi Train Loss giảm mạnh → **Overfitting điển hình**.

### 2.2 Phân Tích Theo Phase

**Phase 1 (Freeze Backbone, Epoch 1-30):**
- Train Acc chỉ đạt ~63-67% → Classifier head đơn lẻ **không đủ capacity** để học tốt
- Val Loss **không cải thiện** rõ ràng qua 30 epochs → Head đã **bão hòa** (saturated)
- Val F1 dao động 40-65% → Không ổn định

**Phase 2 (Unfreeze, Epoch 31+):**
- Train Loss giảm nhanh: 0.7 → 0.25 trong ~15 epochs
- Val Loss **tăng ngay**: 0.9 → 1.5 (Fold 1) → **Model memorize training data**
- **Best val_loss thường ở epoch 31-35**, sau đó val loss chỉ tăng liên tục
- Early stopping trigger ở epoch 50-55, nhưng đã quá muộn

### 2.3 Nguyên Nhân Gốc (Root Causes)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    5 NGUYÊN NHÂN GỐC CỦA OVERFITTING               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1️⃣ DATASET QUÁ NHỎ + MẤT CÂN BẰNG NẶNG                          │
│     • Tổng: 1256 samples, 462 patients                             │
│     • COPD: 820 (65.3%) vs Healthy: 140 (11.1%)                   │
│     • Ratio COPD:Healthy = 5.9:1                                   │
│                                                                     │
│  2️⃣ OVERSAMPLING TẠO DATA TRÙNG LẶP                               │
│     • Train set sau oversample: ~1600-1870 → duplicate samples     │
│     • Model "nhớ" exact copies thay vì "học"                       │
│     • Minority class bị duplicate ~4-5x → same sample xuất hiện    │
│       nhiều lần → false sense of learning                          │
│                                                                     │
│  3️⃣ LEARNING RATE PHASE 2 QUÁ CAO + KHÔNG WARM-UP                 │
│     • Phase 2 LR = 5e-5, nhưng unfreeze TOÀN BỘ backbone          │
│     • Cosine Annealing T_0=10 restart quá nhanh                    │
│     • Cần gradual unfreezing thay vì unfreeze all                  │
│                                                                     │
│  4️⃣ REGULARIZATION YẾU                                             │
│     • Dropout 0.4 chỉ ở classifier head                           │
│     • Không có Label Smoothing                                     │
│     • Không MixUp / CutMix                                        │
│     • Weight Decay 0.01 có thể chưa đủ mạnh                      │
│                                                                     │
│  5️⃣ AUGMENTATION CHƯA ĐỦ ĐA DẠNG                                  │
│     • Audio augmentation chỉ có: noise, shift, gain               │
│     • SpecAugment freq_mask=20 / time_mask=20 khá yếu             │
│     • Không có: time-stretch, pitch-shift, speed-perturbation      │
│     • Majority class (COPD) chỉ augment 30% → under-regularized   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chiến Lược Cải Thiện — Xếp Theo Mức Ưu Tiên

### 🔴 Chiến Lược 1: Xử Lý Class Imbalance (CRITICAL — Dự đoán +10-15% F1)

| Vấn đề | Giải pháp | Chi tiết |
|---------|----------|---------|
| Oversampling naive | **Thay bằng SMOTE-like cho audio** hoặc **augmentation-based oversampling** | Mỗi minority sample qua augmentation khác nhau mỗi epoch, KHÔNG copy nguyên bản |
| Class weight chưa tối ưu | **Focal Loss** thay CrossEntropyLoss | `γ=2.0, α=[0.2, 0.5, 0.3]` — Tập trung vào hard samples |
| Val/Test set nhỏ | **Tăng kích thước dataset** bằng external data | Hoặc dùng **Stratified GroupKFold** với tỷ lệ val lớn hơn |

**Focal Loss cụ thể:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        # alpha = [0.15, 0.45, 0.40] cho [COPD, Healthy, Non-COPD]
        # gamma = 2.0: giảm loss cho easy samples, tập trung hard samples
```

### 🟠 Chiến Lược 2: Regularization Mạnh Hơn (HIGH — Dự đoán +5-8% Acc)

| Kỹ thuật | Giá trị đề xuất | Lý do |
|----------|-----------------|-------|
| **Label Smoothing** | `smoothing=0.1` | Giảm overconfidence, cải thiện calibration |
| **MixUp Training** | `alpha=0.3` | Tạo virtual training examples, giảm memorization |
| **Dropout tăng** | `0.5` (head) + `0.1` (backbone) | Thêm Dropout2d sau một số block backbone |
| **Weight Decay tăng** | `0.05` thay `0.01` | Penalize large weights mạnh hơn |
| **Stochastic Depth** | `drop_rate=0.2` | Đã có sẵn trong MobileNetV2 PyTorch |
| **Gradient Clipping** | `max_norm=1.0` | Ổn định gradient khi unfreeze backbone |

### 🟡 Chiến Lược 3: Cải Thiện Training Schedule (MEDIUM — Dự đoán +3-5% Acc)

#### A) Phase 1 — Cải thiện
| Hiện tại | Đề xuất | Lý do |
|----------|---------|-------|
| 30 epoch, LR=1e-3 | **15-20 epoch, LR=3e-4** | Head saturate sớm, giảm epoch Phase 1 |
| ReduceLROnPlateau | **OneCycleLR** | Warm-up + annealing tốt hơn cho head |

#### B) Phase 2 — Cải thiện (QUAN TRỌNG)
| Hiện tại | Đề xuất | Lý do |
|----------|---------|-------|
| Unfreeze ALL | **Gradual Unfreezing** (3 stages) | Unfreeze từng nhóm layer, bắt đầu từ layers gần classifier |
| LR=5e-5 toàn bộ | **Discriminative LR**: backbone=1e-6, head=5e-5 | Lower layers cần LR nhỏ hơn nhiều |
| CosineAnnealing T_0=10 | **CosineAnnealing T_0=20, T_mult=1** | Chu kỳ dài hơn, ổn định hơn |
| Patience=20 | **Patience=10-12** | Dừng sớm hơn khi val_loss tăng |
| Early stop on val_loss | **Early stop on val_f1** | F1 metric phản ánh đúng hơn cho imbalanced data |

**Gradual Unfreezing Schedule Cụ Thể:**
```
Stage 1 (Ep 31-40):  Unfreeze last 3 InvertedResidual blocks
Stage 2 (Ep 41-60):  Unfreeze last 7 blocks  
Stage 3 (Ep 61-80):  Unfreeze all
→ LR cho mỗi stage: [2e-5, 1e-5, 5e-6]
```

### 🟢 Chiến Lược 4: Tăng Cường Data Augmentation (MEDIUM — Dự đoán +3-7% Acc)

#### Audio-level Augmentation (trước CWT)
| Kỹ thuật | Tham số | Lý do |
|----------|---------|-------|
| **Time Stretch** | `rate ∈ [0.8, 1.2]` | Mô phỏng tốc độ thở khác nhau |
| **Pitch Shift** | `±2 semitones` | Mô phỏng khác biệt cá nhân |
| **Gaussian SNR Noise** | `SNR ∈ [10, 30] dB` | Mô phỏng nhiễu môi trường thực tế |
| **Random Crop + Pad** | `±10% length` | Đa dạng hóa độ dài segment |
| **Polarity Inversion** | `50% prob` | Augmentation đơn giản nhưng hiệu quả |

#### Spectrogram-level Augmentation (sau CWT)
| Kỹ thuật | Tham số | Lý do |
|----------|---------|-------|
| **SpecAugment tăng** | `freq_mask=30, time_mask=40, p=0.7` | Param hiện tại (20/20) quá yếu |
| **Freq-Time Warp** | Nhẹ | Biến dạng spectrogram nhẹ |
| **Random Erasing** | `p=0.3, scale=(0.02, 0.2)` | Che random patches |
| **Augment cho ALL classes** | `prob=0.7` cho tất cả | Hiện tại majority class chỉ 30%! |

### 🔵 Chiến Lược 5: Kiến Trúc Model (OPTIONAL — Dự đoán +2-3%)

| Thay đổi | Chi tiết |
|----------|---------|
| **Thêm Attention Module** | Squeeze-and-Excitation (SE) block trước classifier |
| **Multi-Scale Pooling** | Kết hợp GAP + GMP (Global Max Pooling) |
| **Classifier head lớn hơn** | `1280 → 512 → 256 → 3` thay vì `1280 → 256 → 3` |
| **EfficientNet-B0** thay MobileNetV2 | Hiệu quả hơn cùng model size |

---

## 4. Kế Hoạch Hành Động Đề Xuất (Action Plan)

### Phương án A: Quick Fix (1 lần chạy)

> Áp dụng những thay đổi ít impact nhất vào code nhưng cải thiện lớn nhất

```
Ưu tiên:
1. ✅ Focal Loss thay CrossEntropyLoss 
2. ✅ Label Smoothing = 0.1
3. ✅ Gradual Unfreezing (3 stages)
4. ✅ Discriminative LR (backbone vs head)
5. ✅ Early stop on val_f1 thay vì val_loss
6. ✅ Tăng SpecAugment params
7. ✅ Augment 70% cho TẤT CẢ classes (bỏ logic 30% majority)
```

**Dự kiến kết quả:** Accuracy 78-82%, Macro F1 65-72%

### Phương án B: Full Optimization (2-3 lần chạy)

```
Thêm vào Phương án A:
1. ✅ MixUp / CutMix training
2. ✅ Time Stretch + Pitch Shift augmentation  
3. ✅ Gradient Clipping max_norm=1.0
4. ✅ Weight Decay = 0.05
5. ✅ Phase 1 giảm còn 15 epochs
6. ✅ CosineAnnealing T_0=20
7. ✅ Patience giảm còn 12
```

**Dự kiến kết quả:** Accuracy 82-88%, Macro F1 72-80%

### Phương án C: State-of-the-Art Push

```
Thêm vào Phương án B:
1. ✅ Thu thập thêm data (external datasets)
2. ✅ Sử dụng EfficientNet-B0
3. ✅ Knowledge Distillation
4. ✅ Test-Time Augmentation (TTA)
5. ✅ Ensemble nhiều fold
```

**Dự kiến kết quả:** Accuracy 88-93%, Macro F1 80-88%

---

## 5. Tóm Tắt Nhanh — Top 7 Thay Đổi Quan Trọng Nhất

| # | Thay đổi | Impact | Độ khó |
|---|----------|--------|--------|
| 1 | **Focal Loss** thay CrossEntropy | 🔴 Cao | ⭐ Dễ |
| 2 | **Gradual Unfreezing** 3 stages | 🔴 Cao | ⭐⭐ TB |
| 3 | **Discriminative LR** backbone/head | 🟠 Cao | ⭐ Dễ |
| 4 | **Label Smoothing** 0.1 | 🟠 TB | ⭐ Dễ |
| 5 | **Tăng SpecAugment** + augment all classes | 🟡 TB | ⭐ Dễ |
| 6 | **Early stop trên F1** thay val_loss | 🟡 TB | ⭐ Dễ |
| 7 | **MixUp Training** α=0.3 | 🟠 Cao | ⭐⭐ TB |

> [!TIP]
> Chỉ riêng 3 thay đổi đầu tiên (Focal Loss + Gradual Unfreezing + Discriminative LR) đã có thể cải thiện **10-15% Macro F1**. Đây là những "quick wins" nên ưu tiên triển khai trước.

---

## 6. Dataset Analysis — Vấn Đề Cốt Lõi

```
Raw Dataset Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COPD      ████████████████████  820 samples (65.3%)  |  91 patients
Non-COPD  ███████               296 samples (23.6%)  | 240 patients  
Healthy   ████                  140 samples (11.1%)  | 131 patients
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ NGHỊCH LÝ: 
• COPD: 820 samples / 91 patients = ~9 samples/patient (NHIỀU mẫu/ít BN)
• Healthy: 140 samples / 131 patients = ~1.1 samples/patient (ÍT mẫu/nhiều BN)
• Non-COPD: 296 samples / 240 patients = ~1.2 samples/patient

→ COPD patients có NHIỀU recording → model nhớ "giọng" bệnh nhân,
   không phải đặc trưng bệnh → DATA LEAKAGE tiềm ẩn dù dùng GroupKFold
```

> [!WARNING]
> Với ratio COPD:Healthy = **5.9:1**, naive oversampling copy nguyên bản Healthy samples ~5 lần → model **memorize** chính xác những samples đó. Cần **augmentation-based oversampling** để mỗi copy khác nhau đáng kể.
