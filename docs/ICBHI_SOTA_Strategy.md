# Chiến lược đạt >69% ICBHI Score trên Official Benchmark

## Tổng quan tình hình hiện tại

| Metric | Kết quả | Ghi chú |
|--------|---------|---------|
| Best single model (S2 4-class, threshold) | **0.6278** | Lưu trong file metrics |
| Best single model (S2 4-class, TTA) | **0.6647** | Chạy terminal, chưa lưu file |
| Ensemble S1+S2 (TTA) | **0.6740** | Không phải benchmark chuẩn |
| SOTA hiện tại | **0.6810** | SAM-optimized AST (5.7M params) |
| **Mục tiêu** | **>0.6900** | Single model, official test set |

**Gap cần bù: ~2.5-3%**

---

## Phân tích từ 5 Papers nghiên cứu

### Paper 1: Adaptive Differential Denoising (ADD-RSC)
- **Điểm:** 65.53% (AST backbone)
- **Kỹ thuật chính:**
  - Adaptive Frequency Filter (AFF): learnable spectral masks
  - Differential Denoise Layer: differential attention
  - Bias Denoising Loss với label smoothing (ε=0.2)
- **Áp dụng được:** Label smoothing vào KD loss

### Paper 2: Fusing Time-Domain and 2D Spectral Features (RSC-FTF)
- **Điểm:** 67.55% (multi-view)
- **Kỹ thuật chính:**
  - **VTLP-Patch Augmentation: +3.19%** ← Cao nhất!
  - Co-attention fusion: +2.34%
  - Multi-view patch fusion: +1.53%
  - Supervised Contrastive Learning (SCL)
- **Áp dụng được:** VTLP augmentation, multi-view TTA, SCL loss

### Paper 3: Patient-Aware Feature Alignment (PAFA)
- **Điểm:** 64.84% (4-class), 72.08% (2-class)
- **Kỹ thuật chính:**
  - Patient Cohesion-Separation Loss (PCSL)
  - Global Patient Alignment Loss (GPAL)
  - Architecture-agnostic: +0.41% đến +1.35%
- **Áp dụng được:** PCSL + GPAL losses (không thêm params khi inference)

### Paper 4: Meta-Ensemble Learning
- **Điểm:** 66.49%
- **Kỹ thuật chính:**
  - Diverse data splits tăng complementarity
  - Lightweight meta-model (2-hidden FFN)
- **Áp dụng được:** Curated teacher ensemble (top-5 checkpoints)

### Paper 5: Architecture-Agnostic KD from Ensembles
- **Điểm:** 64.39% (single), 65.69% (ensemble)
- **Kỹ thuật chính:**
  - Mean teacher k=5 là near-optimal
  - Curated ensemble: chọn best checkpoints
  - Second-generation distillation
- **Áp dụng được:** Curated ensemble, second-gen distillation

---

## Chiến lược đề xuất (xếp theo mức độ tác động)

### Ưu tiên 1: VTLP-Patch Augmentation (+3.19%)
**Nguồn:** Paper 2 (RSC-FTF), Section 2.2

**Mô tả:**
- Vocal Tract Length Perturbation áp dụng trực tiếp lên raw audio
- Mô phỏng sự biến đổi cấu trúc thanh quản giữa các bệnh nhân
- Công thức: `y(t) = x(αt) + Gaussian noise`
- α ngẫu nhiên, Fhi=2000Hz

**Lợi ích:**
- +3.19% ICBHI Score (theo ablation study)
- Không thêm parameters vào model
- Dễ tích hợp vào preprocessing pipeline hiện tại

**Triển khai:**
```python
def vtlp_augment(audio, sample_rate=16000, alpha_range=(0.9, 1.1)):
    """VTLP-Patch Augmentation"""
    alpha = random.uniform(*alpha_range)
    # Resample with perturbation
    n_samples = len(audio)
    indices = np.arange(n_samples) * alpha
    indices = np.clip(indices, 0, n_samples - 1).astype(int)
    augmented = audio[indices]
    # Add Gaussian noise
    noise = np.random.randn(len(augmented)) * 0.005
    return augmented + noise
```

### Ưu tiên 2: Patient-Aware Loss Functions (+0.4-1.35%)
**Nguồn:** Paper 3 (PAFA)

**Mô tả:**
- PCSL: Cluster features cùng bệnh nhân, tách features khác bệnh nhân
- GPAL: Align tất cả patient centroids về global center
- Projection head chỉ dùng khi train, remove khi inference

**Lợi ích:**
- Giải quyết inter-patient variability (vấn đề lớn của ICBHI)
- Không thêm parameters khi inference
- Architecture-agnostic

**Triển khai:**
```python
class PatientAwareLoss(nn.Module):
    def __init__(self, lambda_pcsl=50.0, lambda_gpal=0.0005):
        super().__init__()
        self.lambda_pcsl = lambda_pcsl
        self.lambda_gpal = lambda_gpal
        self.proj = nn.Linear(feat_dim, 64)  # Training only
    
    def forward(self, features, patient_ids):
        # PCSL: Sw/Sb ratio
        # GPAL: ||mu_p - mu_G||^2
        return self.lambda_pcsl * pcsl + self.lambda_gpal * gpal
```

### Ưu tiên 3: Curated Teacher Ensemble (+0.2-0.5%)
**Nguồn:** Paper 4, Paper 5

**Mô tả:**
- Chọn top-5 teacher checkpoints dựa trên validation Score
- Thay vì dùng tất cả 9 teachers, chỉ dùng 5 tốt nhất
- Mean teacher soft labels từ curated ensemble

**Lợi ích:**
- Soft labels chất lượng cao hơn
- Giảm noise từ teachers kém
- Không thay đổi student architecture

### Ưu tiên 4: Label Smoothing trong KD (+0.5-1.0%)
**Nguồn:** Paper 1 (ADD-RSC)

**Mô tả:**
- Áp dụng label smoothing (ε=0.1-0.2) lên teacher soft labels
- Acts as uncertainty buffer cho noisy labels
- Kết hợp với KD loss hiện tại

**Triển khai:**
```python
def smooth_kd_loss(student_logits, teacher_probs, temperature, smoothing=0.15):
    """KD loss with label smoothing on teacher soft labels"""
    # Smooth teacher labels
    nc = teacher_probs.size(1)
    smoothed = (1 - smoothing) * teacher_probs + smoothing / nc
    # Standard KD
    kd = -(smoothed * F.log_softmax(student_logits / temperature, dim=1)).sum(1).mean()
    return kd * (temperature ** 2)
```

### Ưu tiên 5: Multi-View TTA (+1.53%)
**Nguồn:** Paper 2 (RSC-FTF)

**Mô tả:**
- Test-time: tạo spectrograms với nhiều patch sizes khác nhau
- Average predictions từ tất cả views
- Không cần train lại

**Views đề xuất:**
- Standard: 128×512 (hiện tại)
- Wide: 64×1024
- Tall: 256×256
- Square: 128×128

### Ưu tiên 6: Supervised Contrastive Learning Loss (+0.5-1.0%)
**Nguồn:** Paper 2 (RSC-FTF)

**Mô tả:**
- SCL loss kết hợp với KD loss (beta=0.1)
- Học feature representations tốt hơn trong embedding space
- Pull cùng class, push khác class

---

## Roadmap triển khai

### Phase 1: Quick Wins (1-2 ngày)
1. ✅ Thêm VTLP-Patch Augmentation vào preprocessing
2. ✅ Thêm label smoothing vào KD loss
3. ✅ Curated teacher ensemble (top-5)
4. ✅ Multi-view TTA cho evaluation

### Phase 2: Advanced (3-5 ngày)
5. Implement Patient-Aware Loss (PCSL + GPAL)
6. Implement Supervised Contrastive Learning
7. Hyperparameter tuning

### Phase 3: Validation (1-2 ngày)
8. Ablation study cho từng kỹ thuật
9. Final evaluation trên official test set
10. So sánh với SOTA

---

## Tổng hợp Expected Gains

| Kỹ thuật | Expected Gain | Confidence |
|----------|---------------|------------|
| VTLP-Patch Augmentation | +3.19% | Cao (paper ablation) |
| Patient-Aware Losses | +0.4-1.35% | Trung bình |
| Label Smoothing KD | +0.5-1.0% | Cao |
| Curated Ensemble | +0.2-0.5% | Cao |
| Multi-View TTA | +1.53% | Cao (paper ablation) |
| Contrastive Learning | +0.5-1.0% | Trung bình |
| **Tổng (conservative)** | **+3-5%** | |
| **Projected ICBHI** | **0.69-0.71** | |

---

## Tham khảo

1. Dong et al. (2025). "Adaptive Differential Denoising for Respiratory Sounds Classification."
2. Dong et al. (2025). "Respiratory sounds classification by fusing the time-domain and 2D spectral features."
3. Jeong & Kim (2025). "Patient-Aware Feature Alignment for Robust Lung Sound Classification."
4. Kim et al. (2026). "Meta-Ensemble Learning with Diverse Data Splits for Improved Respiratory Sound Classification."
5. Toikkanen & Kim (2025). "Improving Respiratory Sound Classification with Architecture-Agnostic Knowledge Distillation from Ensembles."
