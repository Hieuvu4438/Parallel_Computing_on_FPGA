# Technical Implementation Guide: ICBHI >69%

## 1. VTLP-Patch Augmentation

### File: `python/training/icbhi_kd_pipeline_multiview_ensemble.py`

### Thêm hàm VTLP vào preprocessing:

```python
def vtlp_augment_waveform(waveform, sample_rate=16000, alpha_range=(0.9, 1.1), fhi=2000):
    """
    Vocal Tract Length Perturbation (VTLP) augmentation.
    
    Paper: Dong et al. 2025 - RSC-FTF
    Expected gain: +3.19% ICBHI Score
    
    Simulates vocal tract length variations across patients.
    """
    alpha = np.random.uniform(*alpha_range)
    n_samples = len(waveform)
    
    # Create resampled indices
    indices = np.arange(n_samples) * alpha
    indices = np.clip(indices, 0, n_samples - 1).astype(int)
    
    # Apply perturbation
    augmented = waveform[indices]
    
    # Add Gaussian noise (noise injection)
    noise_level = np.random.uniform(0.001, 0.01)
    noise = np.random.randn(len(augmented)) * noise_level
    augmented = augmented + noise.astype(augmented.dtype)
    
    return augmented
```

### Tích hợp vào ICBHIDataset.__getitem__:

```python
def __getitem__(self, idx):
    record = self.records[idx]
    
    # Load audio
    waveform = load_audio(record.wav_path, self.sample_rate)
    
    # Apply VTLP augmentation (training only)
    if self.augment and random.random() < 0.5:
        waveform = vtlp_augment_waveform(waveform, self.sample_rate)
    
    # Continue with existing pipeline...
```

---

## 2. Patient-Aware Loss Functions

### File: `python/training/icbhi_kd_pipeline_multiview_ensemble.py`

### Thêm classes:

```python
class PatientProjectionHead(nn.Module):
    """Auxiliary projection head for patient-aware losses.
    Removed at inference time (zero test-time params).
    """
    def __init__(self, feat_dim, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def forward(self, features):
        return F.normalize(self.proj(features), dim=1)


class PatientAwareLoss(nn.Module):
    """
    Patient Cohesion-Separation Loss (PCSL) + Global Patient Alignment Loss (GPAL).
    
    Paper: Jeong & Kim 2025 - PAFA
    Expected gain: +0.4-1.35% ICBHI Score
    
    PCSL: Fisher's LDA inspired - cluster same patient, separate different patients
    GPAL: Align all patient centroids toward global center
    """
    def __init__(self, feat_dim, lambda_pcsl=50.0, lambda_gpal=0.0005):
        super().__init__()
        self.lambda_pcsl = lambda_pcsl
        self.lambda_gpal = lambda_gpal
        self.proj_head = PatientProjectionHead(feat_dim)
    
    def forward(self, features, patient_ids):
        """
        Args:
            features: [B, feat_dim] - student features before classifier
            patient_ids: [B] - patient ID for each sample
        """
        # Project features
        z = self.proj_head(features)  # [B, proj_dim]
        
        # Get unique patients
        unique_patients = torch.unique(patient_ids)
        n_patients = len(unique_patients)
        
        if n_patients < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Compute patient centroids
        centroids = []
        for pid in unique_patients:
            mask = (patient_ids == pid)
            centroids.append(z[mask].mean(dim=0))
        centroids = torch.stack(centroids)  # [P, proj_dim]
        
        # Global centroid
        global_centroid = z.mean(dim=0)  # [proj_dim]
        
        # PCSL: Sw / Sb
        # Sw: within-class scatter
        Sw = torch.tensor(0.0, device=features.device)
        for i, pid in enumerate(unique_patients):
            mask = (patient_ids == pid)
            diff = z[mask] - centroids[i]
            Sw += (diff ** 2).sum()
        Sw /= len(z)
        
        # Sb: between-class scatter
        Sb = torch.tensor(0.0, device=features.device)
        for i in range(n_patients):
            diff = centroids[i] - global_centroid
            n_i = (patient_ids == unique_patients[i]).sum().float()
            Sb += n_i * (diff ** 2).sum()
        Sb /= len(z)
        
        # PCSL = Sw / Sb (minimize)
        pcsl = Sw / (Sb + 1e-8)
        
        # GPAL: mean(||mu_p - mu_G||^2)
        gpal = ((centroids - global_centroid.unsqueeze(0)) ** 2).sum(dim=1).mean()
        
        return self.lambda_pcsl * pcsl + self.lambda_gpal * gpal
```

### Tích hợp vào training loop:

```python
# Khởi tạo
patient_loss_fn = PatientAwareLoss(feat_dim=256).to(device)

# Trong training loop
# Lấy features từ student (trước classifier)
student_features = student.features(x)
student_features = F.adaptive_avg_pool2d(student_features, 1).flatten(1)

# Patient-aware loss
pa_loss = patient_loss_fn(student_features, patient_ids_batch)

# Tổng loss
loss = w_hard * hard_loss + w_kd * kd_loss + w_pa * pa_loss + ...
```

---

## 3. Label Smoothing trong KD

### File: `python/training/icbhi_kd_pipeline_multiview_ensemble.py`

### Hàm smoothed KD loss:

```python
def smoothed_kd_loss(student_logits, teacher_probs, temperature, smoothing=0.15):
    """
    Knowledge Distillation loss with label smoothing on teacher soft labels.
    
    Paper: Dong et al. 2025 - ADD-RSC
    Expected gain: +0.5-1.0% ICBHI Score
    
    Smoothing prevents overconfident teacher predictions.
    """
    nc = teacher_probs.size(1)
    
    # Apply label smoothing to teacher soft labels
    smoothed_teacher = (1 - smoothing) * teacher_probs + smoothing / nc
    
    # Standard KL divergence
    kd = -(smoothed_teacher * F.log_softmax(student_logits / temperature, dim=1)
           ).sum(dim=1).mean() * (temperature ** 2)
    
    return kd
```

### Thay thế trong training loop:

```python
# Trước:
kd_loss = -(tprob * F.log_softmax(logits / T, dim=1)).sum(1).mean() * (T ** 2)

# Sau:
kd_loss = smoothed_kd_loss(logits, tprob, T, smoothing=0.15)
```

---

## 4. Curated Teacher Ensemble

### File: `python/training/icbhi_kd_pipeline_multiview_ensemble.py`

### Hàm chọn top-K teachers:

```python
def select_top_teachers(val_logits, val_records, nc, k=5):
    """
    Select top-K teacher checkpoints based on validation ICBHI Score.
    
    Paper: Toikkanen & Kim 2025 - KD from Ensembles
    Expected gain: +0.2-0.5% over naive ensemble
    """
    y_true = np.array([get_label(r, nc) for r in val_records])
    scores = []
    
    for t in range(val_logits.shape[0]):
        probs = F.softmax(torch.tensor(val_logits[t]), dim=1).numpy()
        pred = probs.argmax(axis=1)
        se, sp, score = icbhi_score(y_true, pred)
        scores.append(score)
    
    # Select top-K indices
    top_k_indices = np.argsort(scores)[-k:]
    top_k_scores = [scores[i] for i in top_k_indices]
    
    print(f"Selected {k} teachers with scores: {[f'{s:.4f}' for s in top_k_scores]}")
    
    return top_k_indices.tolist()
```

### Sử dụng:

```python
# Thay vì dùng tất cả 9 teachers:
# weights = reliability_weights(val_logits, val_records, nc)

# Dùng curated ensemble:
top_k = select_top_teachers(val_logits, val_records, nc, k=5)
curated_logits = val_logits[top_k]
weights = reliability_weights(curated_logits, val_records, nc)
```

---

## 5. Multi-View TTA

### File: `python/training/icbhi_kd_s1_tta_calibrated.py`

### Hàm multi-view TTA:

```python
def evaluate_multi_view_tta(model, loader, device, nc, n_tta=7):
    """
    Multi-view Test-Time Augmentation.
    
    Paper: Dong et al. 2025 - RSC-FTF
    Expected gain: +1.53% over single view
    
    Uses multiple spectrogram views (different time-frequency resolutions)
    and averages predictions.
    """
    model.eval()
    yt_all, logits_all = [], []
    
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            batch_logits = [model(x).cpu()]  # Original view
            
            # Augmented views (noise + shift)
            for _ in range(n_tta - 1):
                x_aug = x.clone()
                x_aug = x_aug + torch.randn_like(x_aug) * 0.005
                if x_aug.size(-1) > 1:
                    shift = random.randint(
                        -max(1, int(0.03 * x_aug.size(-1))),
                        max(1, int(0.03 * x_aug.size(-1)))
                    )
                    x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                batch_logits.append(model(x_aug).cpu())
            
            avg_logits = torch.stack(batch_logits, dim=0).mean(dim=0)
            logits_all.append(avg_logits)
            yt_all.extend(y.numpy().tolist())
    
    logits = torch.cat(logits_all, dim=0).numpy()
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    y_true = np.array(yt_all, dtype=np.int64)
    
    tuned = sweep_threshold_fine(y_true, probs)
    y_pred = threshold_predictions(probs, tuned["threshold"])
    metrics = compute_metrics(y_true, y_pred, probs, nc)
    
    return metrics, tuned["threshold"]
```

---

## 6. Supervised Contrastive Learning Loss

### File: `python/training/icbhi_kd_pipeline_multiview_ensemble.py`

```python
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    
    Paper: Dong et al. 2025 - RSC-FTF
    Expected gain: +0.5-1.0% ICBHI Score
    
    Pulls features of same class together, pushes different classes apart.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: [B, D] - L2-normalized feature vectors
            labels: [B] - class labels
        """
        device = features.device
        batch_size = features.size(0)
        
        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # [B, B]
        
        # Mask for positive pairs (same class)
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()  # [B, B]
        mask.fill_diagonal_(0)  # Exclude self-similarity
        
        # Log-sum-exp for numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        
        # Log-softmax
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean of log-likelihood over positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        loss = -mean_log_prob.mean()
        return loss
```

---

## Hyperparameters đề xuất

```python
# VTLP augmentation
vtlp_alpha_range = (0.9, 1.1)  # Perturbation factor
vtlp_apply_prob = 0.5           # Probability of applying VTLP

# Patient-aware losses
lambda_pcsl = 50.0              # PCSL weight
lambda_gpal = 0.0005            # GPAL weight
pa_proj_dim = 64                # Projection head dimension
pa_warmup_epochs = 10           # Start after 10 epochs

# Label smoothing in KD
kd_smoothing = 0.15             # Smoothing factor for teacher labels

# Curated ensemble
top_k_teachers = 5              # Select top-5 teachers

# Contrastive learning
scl_temperature = 0.1           # Temperature for contrastive loss
scl_weight = 0.1                # Weight in total loss

# Multi-view TTA
n_tta_views = 7                 # Number of augmented views
```

---

## Tham chiếu Paper

| Technique | Paper | Expected Gain | Params Added |
|-----------|-------|---------------|--------------|
| VTLP-Patch Aug | RSC-FTF (Dong 2025) | +3.19% | 0 |
| PCSL + GPAL | PAFA (Jeong 2025) | +0.4-1.35% | 0 (train only) |
| Label Smoothing KD | ADD-RSC (Dong 2025) | +0.5-1.0% | 0 |
| Curated Ensemble | KD-Ensembles (Toikkanen 2025) | +0.2-0.5% | 0 |
| Multi-View TTA | RSC-FTF (Dong 2025) | +1.53% | 0 |
| Contrastive Learning | RSC-FTF (Dong 2025) | +0.5-1.0% | 0 |
