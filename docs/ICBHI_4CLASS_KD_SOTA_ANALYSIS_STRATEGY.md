# ICBHI 2017 Cycle-Level KD Training Analysis and SOTA Improvement Strategy

This document analyzes the current respiratory sound classification training scripts and proposes a practical strategy to improve the official ICBHI 2017 cycle-level benchmark while keeping the core method: **knowledge distillation from strong teacher models into an FPGA-friendly CNN student**.

Target benchmark to beat for the official 4-class task:

| Task | Metric | SOTA reference |
|---|---:|---:|
| 4-class cycle-level Normal / Crackle / Wheeze / Both | ICBHI Score | 65.53% |
| 4-class cycle-level | Sensitivity | 45.94% |
| 4-class cycle-level | Specificity | 85.13% |

The key objective is not just high accuracy. The main target is **ICBHI Score = (Sensitivity + Specificity) / 2**, where sensitivity measures abnormal-vs-normal detection ability and specificity measures normal detection ability.

Analyzed files:

- `python/training/kd_icbhi_4class_sota.py`
- `python/training/kd_icbhi_4class_sota_bilstm.py`
- `python/training/kd_icbhi_4class_sota_efficientnet.py`

---

## 1. Executive diagnosis

The current code already has a reasonable baseline structure: official patient split, cycle-level labels, class weighting, weighted sampling, focal loss, teacher-to-student KD, and metrics including ICBHI score. However, the current implementation is unlikely to consistently beat a strong SOTA paper because of several structural weaknesses:

1. **Teacher quality is not strong enough.**
   - `kd_icbhi_4class_sota.py` teachers are small CNN/ResNet-style models.
   - `kd_icbhi_4class_sota_bilstm.py` uses a CNN-BiLSTM teacher, but it is trained from scratch on a small imbalanced dataset.
   - `kd_icbhi_4class_sota_efficientnet.py` uses ImageNet EfficientNet-B0, but not audio-pretrained models.

2. **Validation strategy may not reflect the official test target.**
   - Best checkpoint is selected by validation ICBHI score, macro F1, or balanced accuracy, but only from a small validation split carved out of official training patients.
   - The official test split can have different patient/device distribution.

3. **KD is implemented as plain logit matching.**
   - It does not use class-wise temperature, teacher reliability weighting, contrastive feature distillation, intermediate feature matching, or sample difficulty weighting.

4. **The student is too simple for the target.**
   - `cnn6` is FPGA-friendly, but may be under-capacity for 4-class crackle/wheeze/both separation.
   - The wider CNN exists in some files, but the main KD pipeline still treats `cnn6` as the default deployable target.

5. **Preprocessing is narrow and single-view.**
   - All scripts mostly use one log-mel view.
   - Respiratory abnormal events are heterogeneous: crackles are short transient events; wheezes are longer tonal events. A single feature view may miss one of these.

6. **Imbalance handling is incomplete.**
   - Weighted sampler + class-weighted focal loss is useful, but can over-focus minority classes or distort the normal/abnormal balance that ICBHI Score depends on.

7. **Some implementation bugs / code-quality issues can weaken results.**
   - `kd_icbhi_4class_sota_bilstm.py` has a broken `per_class_specificity()` function with unreachable correct code after an early return. Metrics currently use `_per_class_specificity()`, but the duplicate broken function is dangerous.
   - `kd_icbhi_4class_sota_efficientnet.py` computes cached teacher logits in `distill_to_cnn6()` but then ignores them and performs online teacher inference instead.
   - Several argparse flags use `action="store_true", default=True`, which makes the feature always enabled and impossible to disable from CLI.

Recommended direction:

> Keep KD, but upgrade to a **strong audio-pretrained teacher ensemble + multi-view input + reliability-weighted distillation + calibrated student selection for ICBHI Score**.

---

## 2. File-by-file analysis

## 2.1 `kd_icbhi_4class_sota.py`

### What it does well

- Uses official ICBHI patient split: train patients 101-160, test patients 161-226.
- Builds cycle-level labels from ICBHI annotation files.
- Supports 4-class and 2-class labels.
- Implements multiple teacher models and multiple teacher seeds.
- Generates teacher ensemble logits for train/val/test.
- Distills into a CNN student with `mean` or `prob_mean` teacher aggregation.
- Tracks ICBHI score, sensitivity, specificity, macro F1, balanced accuracy, and per-class metrics.
- Keeps student architecture FPGA-friendly.

### Weaknesses

#### 2.1.1 Teacher ensemble is not strong enough

Current teacher choices:

- `small_resnet`
- `efficient_cnn`

These are reasonable local CNN teachers, but they are not SOTA-grade teachers for small medical audio. They are trained from scratch on ICBHI only. The dataset is small and highly imbalanced, so teacher errors become soft-label noise for the student.

**Impact:** KD cannot exceed SOTA if the teacher ensemble is weak or poorly calibrated. A student can sometimes generalize better than a teacher, but only if teacher targets contain useful class structure.

#### 2.1.2 Teacher diversity is limited

The ensemble varies seeds but not enough model families. A 5-seed ensemble of similar CNNs tends to share the same blind spots.

**Better ensemble diversity:**

- Audio Spectrogram Transformer / AST-style teacher
- HTS-AT / Swin-style audio teacher
- PANNs CNN14 / CNN10 teacher
- EfficientNetV2 / ConvNeXt spectrogram teacher
- CNN-BiLSTM or CRNN teacher for temporal dynamics

#### 2.1.3 KD only uses final logits

The KD implementation uses teacher logits only. This is simple but weak. For crackle/wheeze classification, intermediate representations can matter:

- crackle: localized transient representation
- wheeze: harmonic/tonal time-frequency representation
- both: multi-event representation

Final logits may hide the useful structure.

#### 2.1.4 Student input is single-channel log-mel only

The student sees one spectrogram type. The official task needs both transient and tonal discrimination. Single log-mel can work, but a better teacher/student pipeline should consider multi-view inputs.

Possible views:

- log-mel spectrogram
- wavelet scalogram / CWT
- STFT magnitude
- delta + delta-delta log-mel
- PCEN-normalized mel

For FPGA deployment, the final student can still be compact if multi-view channels are limited to 2-3 input channels.

#### 2.1.5 Model selection may not directly optimize target benchmark

The script allows `selection_metric` choices: `icbhi_score`, `macro_f1`, `balanced_accuracy`. That is good, but the training loss itself does not directly optimize sensitivity/specificity balance.

**Risk:** the best validation ICBHI checkpoint may be unstable if validation split is small.

---

## 2.2 `kd_icbhi_4class_sota_bilstm.py`

### What it does well

- CNN-BiLSTM teacher is conceptually appropriate because respiratory cycles have temporal structure.
- Attention pooling can focus on event frames.
- Online KD avoids storing teacher logits and lets the student learn from augmented samples.
- Student remains pure CNN for FPGA friendliness.

### Weaknesses

#### 2.2.1 Teacher is trained from scratch and may be underpowered

The CNN-BiLSTM teacher has a useful inductive bias, but it is not pretrained and uses a fairly small CNN front end. On ICBHI, this can easily underfit minority abnormal patterns or overfit patient/device artifacts.

#### 2.2.2 Online KD with augmented inputs can make teacher targets noisy

The student receives augmented features, and the teacher produces logits on the same augmented features. This is convenient, but if augmentation changes clinically important events, the teacher target may be unstable.

Better strategy:

- Use teacher inference on weakly augmented or clean views.
- Use student on strongly augmented views.
- Distill clean-teacher targets into strong-student augmentations, similar to FixMatch/MeanTeacher logic.

#### 2.2.3 No teacher ensemble

There is only one BiLSTM teacher. This reduces robustness and calibration.

Recommended variant:

- Train 3-5 CNN-BiLSTM/CRNN teachers with different seeds and front ends.
- Combine with transformer/audio-pretrained teachers.
- Distill ensemble probabilities with teacher reliability weighting.

#### 2.2.4 Broken duplicate specificity function

The script contains:

```python
def per_class_specificity(cm):
    total = cm.sum()
    return np.array([cm[i, i] for i in range(cm.shape[0])], dtype=np.float32)
    vals = []
    ...
```

The function returns diagonal values, not specificity, and the correct implementation is unreachable. The script currently uses `_per_class_specificity()` in `compute_metrics()`, so reported metrics are likely safe, but this duplicate broken function should be removed to prevent accidental use.

#### 2.2.5 Student architecture is weak for crackle/wheeze/both

The default `cnn6` has only simple conv blocks and global average pooling. It may lose fine temporal localization needed for crackles.

Recommended student upgrade:

- Keep CNN-only, but use depthwise-separable residual blocks.
- Add lightweight temporal attention or squeeze-excitation if FPGA deployment allows it.
- Use multi-scale kernels: 3x3, 5x3, 3x7 depthwise branches.

---

## 2.3 `kd_icbhi_4class_sota_efficientnet.py`

### What it does well

- EfficientNet-B0 is stronger than the hand-written small CNN teachers.
- Uses gradual unfreezing.
- Supports multi-seed runs.
- Supports self-distillation to CNN6.
- Handles 1-channel or 3-channel input.

### Weaknesses

#### 2.3.1 ImageNet pretraining is suboptimal for respiratory audio

EfficientNet-B0 ImageNet weights can help with generic image edges/textures, but spectrogram medical audio is far from natural images.

Better teacher choices:

- AST pretrained on AudioSet
- HTS-AT pretrained on AudioSet
- PANNs CNN14 pretrained on AudioSet
- BEATs / PaSST / AudioMAE if available
- ConvNeXt-Tiny or EfficientNetV2-S adapted to spectrograms

EfficientNet can remain in the ensemble, but should not be the strongest teacher by default.

#### 2.3.2 `input_channels=3` repeats the same mel spectrogram

Repeating a single channel to 3 channels helps reuse ImageNet conv weights, but it does not add information. Better 3-channel input should be informative:

- channel 1: log-mel
- channel 2: delta log-mel
- channel 3: delta-delta log-mel

or:

- channel 1: log-mel
- channel 2: PCEN mel
- channel 3: CWT/wavelet scalogram

#### 2.3.3 Self-distillation is not enough

`distill_to_cnn6()` trains a CNN student from the same EfficientNet teacher. This is useful, but a single teacher can transfer its biases.

Recommended:

- Use multiple teacher families and average calibrated probabilities.
- Use teacher weighting based on validation per-class recall or per-class F1.
- Use different teachers for different event types, e.g. CRNN strong for wheeze, CWT/ConvNeXt strong for crackle.

#### 2.3.4 Cached teacher logits are computed but not used

`distill_to_cnn6()` collects `teacher_logits` for the train set, creates `tl_tensor`, but then performs online teacher inference in the student loop. This wastes memory/compute and suggests the implementation is unfinished.

#### 2.3.5 CLI flags are hard to disable

Several flags use `action="store_true", default=True`, e.g. `--distill_to_student`, `--speed_perturb`. With this pattern, the option is enabled by default and there is no obvious CLI flag to turn it off.

Recommended:

- Use `BooleanOptionalAction` for Python 3.9+.
- Or define paired flags: `--speed_perturb` and `--no_speed_perturb`.

---

## 3. Metric and benchmark protocol recommendations

## 3.1 Use official test split only for final reporting

For the official ICBHI cycle-level benchmark:

- Train/val: patients 101-160
- Test: patients 161-226

Do not tune hyperparameters directly on official test. Use train patients only for model selection.

Recommended validation protocol:

1. Split official train patients into patient-wise train/val.
2. Repeat with 3-5 different patient-wise validation seeds.
3. Select robust hyperparameters by mean validation ICBHI Score and sensitivity/specificity balance.
4. Retrain final teachers on train+val official-train patients if allowed.
5. Evaluate once on official test.

## 3.2 Track the right metrics

Primary:

- 4-class ICBHI Score
- Sensitivity: abnormal classes predicted as abnormal
- Specificity: normal predicted as normal

Secondary:

- macro F1
- balanced accuracy
- per-class recall: Normal, Crackle, Wheeze, Both
- normal-vs-abnormal binary metrics
- confusion matrix
- AUC OvR
- calibration error / ECE

## 3.3 Add explicit 2-class evaluation from 4-class outputs

Even when training 4-class, also compute binary healthy-vs-pathological by mapping:

- Normal -> Healthy / Normal
- Crackle, Wheeze, Both -> Pathological / Abnormal

This gives:

- binary specificity
- binary sensitivity
- binary ICBHI Score
- binary F1

This is important because a 4-class model can have poor crackle-vs-wheeze discrimination but still strong normal-vs-abnormal screening performance.

---

## 4. Main strategy to beat SOTA

## 4.1 Build a stronger teacher ensemble

The biggest improvement should come from teacher quality. Keep the student CNN, but make teachers much stronger.

Recommended teacher ensemble:

| Teacher | Input | Strength |
|---|---|---|
| AST / PaSST / HTS-AT | log-mel, AudioSet pretrained | global spectrogram context |
| PANNs CNN14 / CNN10 | log-mel, AudioSet pretrained | strong audio CNN baseline |
| ConvNeXt-Tiny / EfficientNetV2-S | 3-channel spectrogram | strong image-style spectrogram model |
| CNN-BiLSTM / CRNN | log-mel sequence | temporal wheeze modeling |
| CWT-CNN / wavelet ConvNeXt | wavelet scalogram | crackle transient modeling |

If implementation time is limited, prioritize:

1. **PANNs CNN14 teacher** pretrained on AudioSet.
2. **AST or HTS-AT teacher** pretrained on AudioSet.
3. Existing **CNN-BiLSTM teacher** as a temporal specialist.
4. Existing **EfficientNet-B0 teacher** as a baseline visual teacher.

The final teacher ensemble should produce train/val/test logits for the same cycle-level samples.

## 4.2 Use multi-view spectrogram inputs for teachers

For teacher models, use richer inputs than the deployable student:

### Recommended 3-channel teacher input

Option A:

- channel 1: log-mel
- channel 2: delta log-mel
- channel 3: delta-delta log-mel

Option B:

- channel 1: log-mel
- channel 2: PCEN mel
- channel 3: CWT/wavelet scalogram resized to mel frame grid

Option C:

- channel 1: low-frequency mel 50-1000 Hz
- channel 2: mid-frequency mel 1000-2500 Hz
- channel 3: full-band mel 50-4000 Hz

For the student, use either:

- single-channel log-mel for simplest FPGA path, or
- 2/3-channel compact input if the FPGA preprocessing budget allows it.

## 4.3 Improve the student architecture while staying CNN-only

The current `cnn6` is deployable but likely too weak. Replace it with a better CNN-only student:

### Recommended student: ResDS-CNN-Attn

Components:

- depthwise-separable convolution blocks
- residual connections
- squeeze-excitation or efficient channel attention
- multi-scale temporal kernels
- global average + global max pooling concatenation
- no recurrent or transformer layers

Example high-level architecture:

```text
Input: [B, C, n_mels, T]
Stem: Conv 3x3, 24 channels
Block1: DS-Res block, 24 channels
Block2: DS-Res block, 32 channels, stride frequency/time
Block3: multi-scale DS block, 48 channels
Block4: multi-scale DS block, 64 channels
Block5: DS-Res-SE block, 96 channels
Pooling: concat(global_avg_pool, global_max_pool)
Classifier: dropout + linear
```

This remains CNN-only and FPGA-friendlier than BiLSTM/Transformer, while being stronger than `cnn6`.

## 4.4 Use reliability-weighted teacher distillation

Current KD averages teacher logits/probabilities equally. Better:

1. Evaluate each teacher on validation set.
2. Compute per-teacher and per-class reliability.
3. Weight teacher probabilities per class.

Example:

```text
teacher_weight[t, c] = normalized(validation_recall_or_f1[t, c])
ensemble_prob[s, c] = sum_t teacher_weight[t, c] * prob[t, s, c]
```

This helps if one teacher is good at crackles while another is good at wheezes.

## 4.5 Use class-aware KD temperature

A single temperature can over-soften minority classes. Recommended:

- Normal: lower temperature, e.g. 2-3
- Crackle/Wheeze/Both: higher temperature, e.g. 4-6

Alternative: tune one global temperature for calibration on validation set, then use it for distillation.

## 4.6 Combine hard loss, soft KD, and binary auxiliary loss

The official ICBHI Score depends strongly on normal-vs-abnormal separation. Add an auxiliary binary head or binary loss during training.

Recommended student objective:

```text
L_total =
  λ_hard * L_focal_4class
+ λ_kd   * L_KL_4class_teacher
+ λ_bin  * L_BCE_normal_vs_abnormal
+ λ_cal  * calibration / confidence regularization optional
```

Suggested starting weights:

```text
λ_hard = 0.35
λ_kd   = 0.45
λ_bin  = 0.20
```

Why binary auxiliary loss matters:

- It directly improves sensitivity/specificity trade-off.
- It discourages pathological samples from being predicted as Normal.
- It can improve ICBHI Score even if 4-class macro F1 changes only slightly.

## 4.7 Use balanced but not excessive sampling

WeightedRandomSampler can help minority classes, but it can also distort the true normal/abnormal balance and cause overfitting.

Recommended alternatives:

- class-balanced loss using effective number of samples
- moderate oversampling only for `Both`
- balanced batch sampler with capped repeat count
- mixup/cutmix/specaugment for repeated minority samples

For ICBHI, `Both` is usually the smallest and hardest class. Avoid training batches where duplicated `Both` samples are repeated without strong augmentation.

## 4.8 Improve augmentation policy

Current augmentations:

- time shift
- Gaussian noise
- speed perturb
- frequency mask
- time mask

Recommended additions:

- random gain
- colored noise / respiratory background noise
- time stretch with smaller range, e.g. 0.95-1.05
- pitch/frequency warping for wheezes carefully
- mixup with low alpha, e.g. 0.1-0.2
- SpecAugment with class-aware strength

Important caution:

- Crackles are short. Strong time masking can delete the discriminative event.
- Wheezes are tonal. Strong frequency masking can delete the discriminative band.

Use weaker augmentation for minority abnormal classes or use clean-teacher / strong-student distillation.

## 4.9 Tune decision threshold for ICBHI Score

Argmax over 4 classes may not maximize ICBHI Score. Since ICBHI Score is based on normal-vs-abnormal sensitivity/specificity, tune a binary abnormal threshold on validation set.

Procedure:

1. Compute `P_normal` from 4-class probabilities.
2. Predict Normal if `P_normal >= threshold`.
3. Otherwise predict the highest abnormal class among Crackle/Wheeze/Both.
4. Sweep threshold on validation set to maximize ICBHI Score.
5. Apply fixed threshold to test set.

This can improve sensitivity/specificity balance without changing the model.

---

## 5. Concrete implementation plan

## Phase 1: Fix measurement and baseline correctness

Priority: high.

1. Remove or fix the broken duplicate `per_class_specificity()` in `kd_icbhi_4class_sota_bilstm.py`.
2. Add binary normal-vs-abnormal metrics to all three scripts.
3. Save validation predictions and probabilities for every best checkpoint.
4. Add threshold sweep for normal-vs-abnormal ICBHI Score.
5. Report both raw argmax metrics and threshold-tuned metrics.
6. Ensure `speed_perturb` and `distill_to_student` can be disabled from CLI.
7. In `kd_icbhi_4class_sota_efficientnet.py`, either use cached teacher logits or remove unused cache logic.

Expected benefit: more reliable benchmark reporting and likely immediate ICBHI Score improvement from threshold tuning.

## Phase 2: Upgrade teacher ensemble

Priority: highest for SOTA.

1. Add support for loading pretrained audio teachers:
   - PANNs CNN14/CNN10
   - AST/PaSST/HTS-AT if dependencies allow
2. Train/fine-tune each teacher on official train patients.
3. Use patient-wise validation inside official train.
4. Save teacher logits for train/val/test.
5. Calibrate each teacher with temperature scaling on validation set.
6. Build reliability-weighted ensemble logits/probabilities.

Expected benefit: largest improvement in student KD quality.

## Phase 3: Upgrade student CNN

Priority: high.

Add a new student architecture, for example:

- `ds_cnn_res`
- `ds_cnn_res_se`
- `multiscale_ds_cnn`

Keep `cnn6` for minimal FPGA baseline, but make the stronger CNN the main benchmark student.

Recommended CLI:

```bash
--student_arch ds_cnn_res_se
```

Expected benefit: better abnormal event representation while still deployable.

## Phase 4: Improve KD objective

Priority: high.

Add:

1. reliability-weighted ensemble probabilities
2. class-aware or calibrated temperature
3. binary auxiliary loss
4. clean-teacher / strong-student augmentation
5. optional feature distillation if teacher/student feature shapes are compatible

Recommended loss:

```text
0.35 * focal_4class
+ 0.45 * KL(student_4class, teacher_ensemble_4class)
+ 0.20 * BCE(student_abnormal, teacher_or_hard_abnormal)
```

Expected benefit: better sensitivity/specificity balance.

## Phase 5: Hyperparameter search focused on ICBHI Score

Tune only on validation patients:

| Parameter | Search values |
|---|---|
| temperature | 2, 3, 4, 5, 6 |
| alpha KD weight | 0.3, 0.45, 0.6, 0.75 |
| binary loss weight | 0.1, 0.2, 0.3 |
| focal gamma | 1.0, 1.5, 2.0 |
| label smoothing | 0.0, 0.03, 0.05 |
| abnormal threshold | sweep 0.20-0.80 |
| student width | 0.75x, 1.0x, 1.25x |

Selection rule:

1. maximize validation ICBHI Score
2. require sensitivity not lower than SOTA sensitivity target
3. require specificity not much lower than SOTA specificity target
4. tie-break by macro F1

---

## 6. Recommended final training recipe

### Data preprocessing

- sample rate: 16 kHz initially; compare 8 kHz and 16 kHz
- duration: 8 seconds with cyclic padding
- f_min: 50 Hz
- f_max: compare 2500 Hz vs 4000 Hz vs 6000 Hz if sample rate allows
- feature: log-mel 128 bins, 512 frames
- teacher input: 3-channel log-mel/delta/delta-delta or log-mel/PCEN/CWT
- student input: 1-channel log-mel or compact 3-channel if feasible

### Teacher ensemble

- 1x PANNs CNN14 or CNN10 pretrained
- 1x AST/PaSST/HTS-AT pretrained
- 1x EfficientNetV2/ConvNeXt spectrogram teacher
- 1x CNN-BiLSTM/CRNN temporal teacher
- 3 seeds for each if compute allows; otherwise 1-2 seeds each

### Student

- `ds_cnn_res_se` or `multiscale_ds_cnn`
- CNN-only
- depthwise separable convs
- residual connections
- global avg + max pooling
- optional SE/ECA attention

### Loss

```text
L = 0.35 * FocalLoss_4class
  + 0.45 * KLDiv_4class_KD
  + 0.20 * BinaryNormalAbnormalLoss
```

### Checkpoint selection

- select by validation threshold-tuned ICBHI Score
- save raw argmax metrics too
- save selected abnormal threshold
- final test uses frozen threshold

### Reporting

Always report:

- 4-class ICBHI Score
- 4-class sensitivity
- 4-class specificity
- 4-class macro F1
- per-class precision/recall/F1
- confusion matrix
- 2-class normal-vs-abnormal sensitivity/specificity/ICBHI Score
- model parameter count
- estimated MACs if possible

---

## 7. Important risks and controls

## 7.1 Risk: test-set overfitting

Do not use official test metrics to pick thresholds or hyperparameters. Tune thresholds on validation only.

## 7.2 Risk: teacher overfitting

Use:

- early stopping by validation ICBHI Score
- calibration
- multi-seed validation
- strong but not destructive augmentation

## 7.3 Risk: high specificity but low sensitivity

The cited SOTA has high specificity but low sensitivity. To beat ICBHI Score, improving sensitivity is likely the easiest path, but do not collapse specificity.

Validation rule:

```text
accept config only if:
  sensitivity improves substantially
  specificity remains competitive
  ICBHI Score improves
```

## 7.4 Risk: stronger student is not FPGA-friendly

Keep two students:

1. `cnn6`: minimal FPGA baseline
2. `ds_cnn_res_se`: main benchmark student

Then export both to ONNX and estimate deployment cost.

---

## 8. Prioritized action list

### Immediate fixes

1. Fix metric cleanup and binary evaluation.
2. Add threshold tuning for normal-vs-abnormal ICBHI Score.
3. Fix CLI boolean flags.
4. Remove unused teacher-logit cache or use it properly.
5. Standardize split/config caching so old splits are not reused incorrectly when preprocessing changes.

### High-impact modeling changes

1. Add audio-pretrained teachers.
2. Add reliability-weighted teacher ensemble.
3. Add stronger CNN-only student.
4. Add binary auxiliary loss.
5. Add multi-view teacher inputs.

### Experiment order

1. Baseline current `kd_icbhi_4class_sota.py` with fixed metrics.
2. Add threshold tuning only.
3. Add stronger student only.
4. Add PANNs/AST teacher logits.
5. Add teacher reliability weighting.
6. Add binary auxiliary loss.
7. Run final multi-seed validation and official test evaluation.

---

## 9. Expected result direction

The current scripts can provide a baseline, but the most likely path to beating the cited 65.53% ICBHI Score is:

```text
strong audio-pretrained teacher ensemble
        +
calibrated/reliability-weighted KD
        +
CNN-only residual depthwise student
        +
binary normal-vs-abnormal auxiliary objective
        +
validation-tuned abnormal threshold
```

This keeps the original research direction — **teacher ensemble KD into a CNN student** — while addressing the main reasons the current scripts may underperform SOTA: weak teachers, single-view features, simple KD, fragile imbalance handling, and lack of threshold optimization for the actual ICBHI metric.
