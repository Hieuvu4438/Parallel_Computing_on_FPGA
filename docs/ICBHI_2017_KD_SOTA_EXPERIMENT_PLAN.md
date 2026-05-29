# ICBHI 2017 Knowledge Distillation SOTA Strategy

This document proposes a research and experiment strategy for the **ICBHI 2017 respiratory sound challenge** using **knowledge distillation (KD)** from strong teacher ensembles into a deployable **CNN student**.

Dataset path for this project:

```text
/home/haipd/Parallel_Computing_on_FPGA/data/sample_01/ICBHI_final_database
```

The target is to compete on the official-style **patient-wise 60/40 split** without patient leakage and to report both:

1. **2-class task**: Normal vs Abnormal
2. **4-class task**: Normal, Crackles, Wheezes, Both

---

## 1. Benchmark protocol to lock before modeling

### 1.1 Split rule

Use **subject-level split**, not file-level or cycle-level random split.

Recommended official-style split:

```text
Train/validation subjects: patient IDs 101-160
Test subjects:             patient IDs 161-226/227 depending on available files
```

Key rule:

```text
No patient may appear in both train/validation and test.
```

Inside the official train subjects, create patient-wise validation splits only from train patients. Do not tune hyperparameters or thresholds directly on official test patients.

### 1.2 Cycle-level label mapping

ICBHI cycle annotations contain crackle and wheeze flags. Map each respiratory cycle as:

| Crackle flag | Wheeze flag | 4-class label | 2-class label |
|---:|---:|---|---|
| 0 | 0 | 0: Normal | 0: Normal |
| 1 | 0 | 1: Crackles Only | 1: Abnormal |
| 0 | 1 | 2: Wheezes Only | 1: Abnormal |
| 1 | 1 | 3: Both | 1: Abnormal |

### 1.3 Primary metrics

For the official ICBHI-style score:

```text
Specificity = correctly predicted Normal cycles / all true Normal cycles
Sensitivity = correctly predicted Abnormal cycles / all true Abnormal cycles
ICBHI Score = (Specificity + Sensitivity) / 2
```

For 4-class reporting, abnormal means the union of Crackles, Wheezes, and Both. Therefore, always report both:

- full 4-class confusion matrix and per-class F1
- binary Normal-vs-Abnormal metrics derived from 4-class predictions

### 1.4 SOTA targets from current benchmark references

The user-provided benchmark targets are:

| Task | Specificity | Sensitivity | ICBHI Score | Strong reference direction |
|---|---:|---:|---:|---|
| 4-class | ~89.87% | ~44% | ~65% | high-specificity 4-class SOTA |
| 2-class | 81.66% / 74.87% | 55.77% / 68.29% | 68.71% / 72.08% | Patch-Mix CL and BEATs+PAFA |

Important interpretation:

- 4-class SOTA often has very high specificity but weak sensitivity.
- 2-class BEATs+PAFA reaches stronger sensitivity and score by explicitly improving abnormal detection.
- Our KD strategy should not chase specificity alone; the easiest SOTA path is likely to raise sensitivity while preserving competitive specificity.

---

## 2. Lessons from the provided SOTA repositories

### 2.1 Patch-Mix Contrastive Learning

Repository:

```text
https://github.com/raymin0223/patch-mix_contrastive_learning
```

Useful lessons:

- Uses **Audio Spectrogram Transformer (AST)** on respiratory sound spectrograms.
- Supports official `--test_fold official` split.
- Supports both `--n_cls 4` and binary classification.
- Strong method components:
  - spectrogram patch mixing
  - soft labels based on patch mixing ratio
  - contrastive learning on mixed representations
  - pretrained AST/ImageNet/AudioSet-style initialization

KD implication:

> Use Patch-Mix AST as a teacher family and distill not only logits, but also behavior under patch-level mixing.

### 2.2 PAFA

Repository:

```text
https://github.com/wa976/PAFA
```

Useful lessons:

- Uses **BEATs** audio-pretrained backbone.
- Adds patient-aware feature alignment:
  - Patient Cohesion-Separation Loss (PCSL)
  - Global Patient Alignment Loss (GPAL)
- Strong reported direction for the 2-class task:
  - BEATs+CE: strong baseline
  - BEATs+PAFA: stronger sensitivity and ICBHI score

KD implication:

> Use BEATs+PAFA as a patient-aware teacher and transfer both class probability structure and patient-invariant feature geometry into the CNN student.

---

## 3. Overall proposed method

The core method should be:

```text
Strong multi-family teacher ensemble
        +
patient-wise validation and calibration
        +
reliability-weighted knowledge distillation
        +
CNN-only student with binary auxiliary objective
        +
validation-tuned abnormal threshold
```

### 3.1 Teacher ensemble

Use diverse teachers because crackles, wheezes, and normal breath cycles have different acoustic signatures.

| Teacher family | Input | Why useful |
|---|---|---|
| AST / Patch-Mix AST | log-mel spectrogram | strong global time-frequency modeling |
| BEATs + PAFA | waveform or BEATs features | audio-pretrained, patient-aware representation |
| PANNs CNN14/CNN10 | log-mel | strong AudioSet-pretrained CNN teacher |
| ConvNeXt/EfficientNetV2 spectrogram teacher | 2D spectrogram image | strong CNN visual prior |
| CRNN/CNN-BiGRU teacher | log-mel sequence | temporal wheeze modeling |
| CWT/wavelet CNN teacher | wavelet scalogram | transient crackle modeling |

Minimal practical teacher set:

```text
T1: Patch-Mix AST teacher
T2: BEATs+PAFA teacher
T3: PANNs CNN14 or ConvNeXt/EfficientNetV2 teacher
T4: CRNN or wavelet-CNN specialist teacher
```

### 3.2 Student model constraint

The final student must remain a **CNN model**.

Recommended students:

| Student | Use case |
|---|---|
| CNN6 | minimal FPGA-friendly baseline |
| DS-CNN-Res-SE | main deployable CNN student |
| RepVGG-tiny | strong CNN student that can be re-parameterized to plain convs |
| MobileNetV2/V3-small ReLU variant | good accuracy/efficiency trade-off |

Recommended primary student:

```text
DS-CNN-Res-SE or RepVGG-tiny
```

Why not only CNN6:

- CNN6 is deployable but may be too weak for Crackle/Wheeze/Both separation.
- A slightly stronger CNN-only student is still compatible with FPGA deployment and KD.

---

## 4. Three required KD experiment strategies

## Experiment 1: Logit KD from calibrated multi-teacher ensemble

### Goal

Build the strongest baseline KD pipeline with diverse teachers and a CNN student.

### Teacher models

```text
T1: Patch-Mix AST, official split
T2: BEATs+CE or BEATs+PAFA
T3: ConvNeXt-Tiny or EfficientNetV2-S spectrogram teacher
T4: CRNN/CNN-BiGRU temporal teacher
```

### Student

```text
DS-CNN-Res-SE or RepVGG-tiny
```

### Distillation target

Use calibrated probability ensemble:

```text
p_teacher = sum_t w_t * softmax(logits_t / T_t)
```

where `w_t` is teacher reliability weight from validation performance.

Recommended reliability weighting:

```text
w_t,c = normalized validation recall/F1 of teacher t for class c
p_teacher[c] = sum_t w_t,c * p_t[c]
```

This lets one teacher dominate Crackles if it is best at Crackles, while another teacher can dominate Wheezes.

### Loss

```text
L = 0.35 * FocalCE_4class(y, student_logits)
  + 0.45 * KL(student_probs_T, teacher_probs_T)
  + 0.20 * BCE_NormalAbnormal(student_binary, target_binary)
```

Recommended initial hyperparameters:

| Parameter | Values to sweep |
|---|---|
| KD temperature | 2, 3, 4, 5, 6 |
| KD weight | 0.35, 0.45, 0.60 |
| binary auxiliary weight | 0.10, 0.20, 0.30 |
| focal gamma | 1.0, 1.5, 2.0 |
| label smoothing | 0.00, 0.03, 0.05 |

### Expected benefit

This should improve the student over single-teacher KD by reducing teacher-specific blind spots.

### Main metric target

| Task | Target |
|---|---:|
| 4-class ICBHI Score | >65% |
| 2-class ICBHI Score | >72.08% |

---

## Experiment 2: Patch-Mix contrastive distillation

### Goal

Transfer the Patch-Mix AST teacher’s robustness and representation structure into a CNN student.

### Motivation

Patch-Mix CL is strong because it trains models to understand local spectrogram patches and mixed labels. This is useful for ICBHI because abnormal events are localized:

- crackles: short transient patches
- wheezes: longer tonal patches
- both: mixed abnormal patterns

### Teacher

```text
Patch-Mix AST teacher trained on official ICBHI split
```

Optionally ensemble multiple AST seeds.

### Student

```text
RepVGG-tiny or DS-CNN-Res-SE
```

### Training data transformation

For two samples `(x_i, y_i)` and `(x_j, y_j)`, create patch-mixed spectrogram:

```text
x_mix = PatchMix(x_i, x_j, mask)
y_mix = lambda * y_i + (1 - lambda) * y_j
```

### Distillation losses

Use three terms:

```text
L = L_hard_or_soft_CE(y_mix, student(x_mix))
  + L_KD(student(x_mix), teacher(x_mix))
  + L_feature_contrast(student_feat(x_mix), teacher_feat(x_mix))
```

If teacher features are not easy to extract, use logits-only Patch-Mix KD first:

```text
L = 0.40 * CE(student, y_mix)
  + 0.60 * KL(student_probs, teacher_probs_on_x_mix)
```

### Contrastive distillation option

Use teacher embeddings as anchors:

```text
positive pairs: same class or same mixed-label dominant class
negative pairs: different dominant class
```

For mixed labels, weight the contrastive loss by the patch mixing ratio.

### Expected benefit

- Better generalization on small ICBHI data.
- Stronger abnormal sensitivity because local abnormal patches are emphasized.
- Better CNN student robustness under localized perturbations.

### Main ablation matrix

| ID | Teacher | Student | Patch-Mix | Contrastive KD | Target |
|---|---|---|---|---|---|
| PMKD-1 | AST CE | DS-CNN | no | no | baseline KD |
| PMKD-2 | Patch-Mix AST | DS-CNN | yes | no | patch-logit KD |
| PMKD-3 | Patch-Mix AST | DS-CNN | yes | yes | full strategy |

---

## Experiment 3: Patient-aware feature-alignment KD from BEATs+PAFA

### Goal

Distill PAFA’s patient-aware and patient-invariant representation into a CNN student while preserving official patient-wise evaluation.

### Motivation

ICBHI performance is sensitive to patient identity, recording device, auscultation location, and disease prevalence. A model can overfit patient/device artifacts instead of respiratory events. PAFA directly targets this by organizing embeddings around patient-level structure.

### Teacher

```text
BEATs+PAFA teacher
```

### Student

```text
CNN-only student: DS-CNN-Res-SE, RepVGG-tiny, or MobileNetV2-small
```

### Distillation targets

Use three levels:

1. **Logit KD**

```text
KL(student class probabilities, BEATs+PAFA class probabilities)
```

2. **Embedding KD**

Add a projection head on the student:

```text
z_s = projection(student_embedding)
z_t = teacher_embedding
L_embed = cosine_or_mse(z_s, stopgrad(z_t))
```

3. **Patient-aware relational KD**

Match pairwise similarity structure from teacher embeddings:

```text
S_t[i,j] = cosine(z_t[i], z_t[j])
S_s[i,j] = cosine(z_s[i], z_s[j])
L_rel = MSE(S_s, S_t)
```

This transfers teacher geometry without forcing identical high-dimensional features.

### Patient-aware loss for student

If patient IDs are available during training, add a light PAFA-inspired regularizer:

```text
L_patient = L_same_patient_compactness + L_different_patient_separation
```

Keep this loss small to avoid making the student memorize patient identity.

Recommended total loss:

```text
L = 0.30 * FocalCE_4class
  + 0.35 * KL_KD_from_PAFA
  + 0.20 * RelationalEmbeddingKD
  + 0.15 * BinaryNormalAbnormalLoss
```

### Expected benefit

- Stronger 2-class Normal/Abnormal score.
- Better generalization to unseen test patients.
- Less leakage-like reliance on patient artifacts.

### Main ablation matrix

| ID | KD type | Patient regularization | Target |
|---|---|---|---|
| PAFA-KD-1 | logit KD only | no | baseline BEATs distillation |
| PAFA-KD-2 | logit + embedding KD | no | transfer PAFA representation |
| PAFA-KD-3 | logit + relational KD | yes, weak | best patient-aware student |

---

## 5. Optional Experiment 4: Specialist-teacher mixture distillation

This is recommended if the first three experiments are implemented successfully.

### Goal

Use teachers specialized by acoustic event type.

| Specialist | Best for | Suggested model |
|---|---|---|
| Normal specialist | specificity | EfficientNet/ConvNeXt |
| Crackle specialist | transient events | CWT/wavelet CNN |
| Wheeze specialist | tonal events | CRNN/CNN-BiGRU |
| Both specialist | mixed abnormalities | AST/BEATs ensemble |

### Distillation rule

Use a gating function or validation-derived class reliability:

```text
p_teacher[c] = sum_t reliability[t,c] * p_t[c]
```

No separate gating model is required at first. A validation reliability matrix is simpler and safer.

---

## 6. Training and evaluation pipeline

### 6.1 Data preprocessing

Recommended baseline preprocessing:

| Item | Recommendation |
|---|---|
| sample rate | 16 kHz for teacher, compare 8 kHz for student |
| cycle extraction | use annotation start/end times |
| padding | cyclic padding to fixed duration |
| duration | 6-8 seconds, sweep if needed |
| spectrogram | 128-bin log-mel |
| frequency range | 50 Hz to 4000/6000 Hz depending on sample rate |
| normalization | train-set mean/std only |

Teacher multi-view options:

```text
View A: log-mel
View B: delta + delta-delta log-mel
View C: PCEN mel
View D: CWT/wavelet scalogram
```

Student input options:

```text
Minimal: 1-channel log-mel
Stronger: 3-channel log-mel + delta + delta-delta
```

### 6.2 Augmentation policy

Use strong but clinically safe augmentation:

| Augmentation | Recommended caution |
|---|---|
| time shift | safe |
| random gain | safe |
| low SNR noise | useful but do not drown crackles |
| SpecAugment | use mild time/frequency masks |
| speed perturbation | keep small, e.g. 0.95-1.05 |
| MixUp/Patch-Mix | strong for teacher/student robustness |

Avoid overly strong time masks for Crackles because they can delete short transient events.

Avoid overly strong frequency masks for Wheezes because they can remove tonal bands.

### 6.3 Model selection

For each candidate, save:

```text
config.json
splits.json
best_checkpoint.pt
val_predictions.csv
test_predictions.csv
confusion_matrix.json
metrics.json
```

Validation selection rule:

```text
Primary: maximize validation ICBHI Score
Constraint 1: sensitivity must not collapse
Constraint 2: specificity must remain competitive
Tie-breaker: macro F1 and per-class recall
```

For 4-class task, tune a Normal-vs-Abnormal threshold only on validation:

```text
predict Normal if P(Normal) >= tau
otherwise predict argmax among Crackles/Wheezes/Both
```

Sweep:

```text
tau in [0.20, 0.80]
```

Then freeze `tau` and evaluate once on the official test split.

---

## 7. Concrete experiment table

| Experiment | Teacher | Student | KD type | Main target |
|---|---|---|---|---|
| E0 | none | DS-CNN/RepVGG | hard-label CE/Focal | honest CNN baseline |
| E1 | AST + BEATs + ConvNeXt/CRNN | DS-CNN-Res-SE | calibrated logit KD | strong general KD |
| E2 | Patch-Mix AST | DS-CNN/RepVGG | Patch-Mix logit + contrastive KD | local abnormal robustness |
| E3 | BEATs+PAFA | DS-CNN/RepVGG | logit + relational embedding KD | patient-wise generalization |
| E4 | event-specialist ensemble | DS-CNN/RepVGG | class-reliability mixture KD | maximize per-class recall |

Minimum required for the paper/project:

```text
E1, E2, E3
```

---

## 8. Recommended implementation order

### Phase A: Benchmark correctness

1. Build one canonical cycle-level metadata file:

```text
cycle_id, wav_path, patient_id, start_time, end_time, crackle, wheeze, label_4class, label_2class, split
```

2. Verify patient-wise split.
3. Implement shared metric function for 2-class and 4-class.
4. Save all predictions and thresholds.

### Phase B: Baselines

1. Train hard-label CNN student baseline.
2. Train ConvNeXt/EfficientNetV2 spectrogram teacher baseline.
3. Train or adapt CRNN teacher baseline.
4. Reproduce Patch-Mix AST and BEATs/PAFA as external teacher baselines if resources allow.

### Phase C: KD experiments

1. E1: calibrated ensemble logit KD.
2. E2: Patch-Mix contrastive KD.
3. E3: PAFA relational embedding KD.
4. Compare CNN6 vs DS-CNN-Res-SE vs RepVGG-tiny.

### Phase D: Final test and reporting

1. Select final configs using validation only.
2. Freeze thresholds and hyperparameters.
3. Evaluate once on official test split.
4. Report mean/std over seeds if compute allows.

---

## 9. Success criteria

### 9.1 2-class target

Beat or match:

```text
Specificity: 81.66% or competitive with PAFA 74.87%
Sensitivity: >68.29%
ICBHI Score: >72.08%
```

The most realistic path is E3, because PAFA already improves sensitivity.

### 9.2 4-class target

Beat:

```text
ICBHI Score: >65%
```

Preferred target:

```text
Specificity >= 85-89%
Sensitivity >= 45-50%
ICBHI Score >= 67%
```

The most realistic path is E1 + E2:

```text
calibrated ensemble KD + Patch-Mix local robustness + threshold tuning
```

---

## 10. Risks and controls

| Risk | Control |
|---|---|
| patient leakage | split only by patient ID; assert no overlap |
| test-set overfitting | never tune threshold/hyperparameters on test |
| teacher overfitting | calibrate and validate teachers separately |
| weak teacher hurts student | only distill teachers that beat student on validation |
| high specificity but poor sensitivity | binary auxiliary loss and threshold tuning |
| minority class memorization | augmentation-based balancing, not naive duplication |
| non-reproducible SOTA claim | save splits, configs, predictions, and seeds |

---

## 11. Recommended paper contribution framing

A strong research claim can be framed as:

> We propose a patient-safe knowledge distillation framework for ICBHI 2017 respiratory cycle classification, where heterogeneous audio-pretrained and spectrogram teachers are calibrated and distilled into a compact CNN student. The method combines reliability-weighted ensemble KD, Patch-Mix contrastive distillation, and patient-aware relational distillation to improve both official ICBHI score and deployability.

Potential contribution list:

1. **Teacher diversity**: AST, BEATs/PAFA, CNN spectrogram, CRNN/wavelet specialists.
2. **Reliability-weighted KD**: per-class teacher weighting from validation performance.
3. **Patch-level KD**: transfer Patch-Mix teacher behavior to CNN student.
4. **Patient-aware KD**: transfer PAFA-style feature geometry without leaking test patients.
5. **Deployable CNN student**: maintain FPGA compatibility while approaching SOTA.

---

## 12. References and source links

- ICBHI 2017 respiratory sound database/challenge benchmark protocol.
- Patch-Mix Contrastive Learning repository: `https://github.com/raymin0223/patch-mix_contrastive_learning`
- PAFA repository: `https://github.com/wa976/PAFA`
- User-provided WizWand benchmark pages for 2-class and 4-class ICBHI 60/40 official split.
- Existing project docs:
  - `docs/ICBHI_KD_SOTA_TUNING_STRATEGY.md`
  - `docs/ICBHI_4CLASS_KD_SOTA_ANALYSIS_STRATEGY.md`
  - `docs/icbhi_benchmark_comparison_report.md`
