# ICBHI 2017 3-Class Knowledge Distillation Pipeline

This workflow trains an ICBHI-only respiratory sound classifier for:

1. `COPD`
2. `Non-COPD`
3. `Healthy`

The implementation is `python/training/kd_icbhi_3class.py`. It follows the soft-label teacher-ensemble strategy from `docs/references/Improving Respiratory Sound Classification with Architecture-Agnostic Knowledge Distillation from Ensembles.md`, adapted to a 3-class disease-label task and a lightweight CNN student for FPGA deployment.

## Data layout

```text
data/sample_01/
├── labels.txt
└── ICBHI_final_database/
    ├── 101_1b1_Al_sc_Meditron.wav
    ├── 101_1b1_Al_sc_Meditron.txt
    └── ...
```

`labels.txt` is subject-level diagnosis metadata:

```text
101    URTI
102    Healthy
104    COPD
...
```

The script parses the subject id from the filename prefix before the first underscore and propagates the subject diagnosis to every respiratory cycle from that recording.

## Label mapping

| Original subject diagnosis | 3-class label |
|---|---|
| `COPD` | `COPD` |
| `Healthy` | `Healthy` |
| `URTI`, `Asthma`, `LRTI`, `Bronchiectasis`, `Pneumonia`, `Bronchiolitis`, other non-healthy/non-COPD diagnoses | `Non-COPD` |

Assumption: ICBHI disease diagnosis is subject-level, not cycle-level. All cycles from the same subject inherit the same disease label. If a future metadata file contains multi-disease labels, pure `COPD` should map to `COPD`; any explicitly mixed diagnosis should map to `Non-COPD` unless the experiment intentionally changes the clinical target.

## Preprocessing

Default preprocessing is implemented inside the training script:

| Step | Default |
|---|---|
| Sampling rate | 16 kHz |
| Band-pass | 50-2500 Hz, enabled by default |
| Segmentation | ICBHI cycle `.txt` start/end annotations |
| Fallback segmentation | 8 s windows, 4 s hop |
| Cycle length | pad/crop to 8 s |
| Feature | log-Mel spectrogram |
| FFT | 1024 |
| Window | 400 samples, 25 ms at 16 kHz |
| Hop | 160 samples, 10 ms at 16 kHz |
| Mel bins | 64 |
| Target shape | `1 x 64 x 800` |
| Normalization | train-set mean/std saved in `config.json` |

Training augmentations:

- waveform time shift
- additive Gaussian noise
- frequency masking
- time masking

Augmentation is disabled for validation, test and teacher soft-label generation.

## Teacher ensemble

Default teacher architecture: `small_resnet` over log-Mel spectrograms.

Recommended default:

```bash
python3 python/training/kd_icbhi_3class.py \
  --stage teachers \
  --num_teachers 5 \
  --teacher_seeds 1,2,3,4,5 \
  --teacher_arch small_resnet \
  --wandb
```

Training defaults:

| Hyperparameter | Default |
|---|---|
| Optimizer | AdamW |
| LR | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 100 |
| Scheduler | Cosine annealing |
| Early stopping | patience 15 |
| Selection metric | macro-F1 |

Teachers are saved to:

```text
artifacts/training/icbhi_3class_kd/teachers/seed_*/best.pt
```

## Soft-label generation

For teacher logits `z_i(x)` from `K` teachers and temperature `T`:

Mean teacher:

```text
p_mean(x) = softmax((1/K) * sum_i z_i(x) / T)
```

Random teacher:

```text
i ~ Uniform({1..K})
p_random(x) = softmax(z_i(x) / T)
```

The script saves logits for each split:

```text
artifacts/training/icbhi_3class_kd/soft_labels/teacher_logits_train.npy
artifacts/training/icbhi_3class_kd/soft_labels/teacher_logits_val.npy
artifacts/training/icbhi_3class_kd/soft_labels/teacher_logits_test.npy
```

## Student CNN

Default student: `cnn6`, a pure CNN with:

- Conv2d
- BatchNorm2d
- ReLU
- MaxPool2d
- AdaptiveAvgPool2d
- small Linear classifier

No LSTM, GRU, Transformer or heavy attention is used in the student. BatchNorm can be folded at inference, and the Conv/BN/ReLU/Pooling pattern is suitable for INT8 quantization and FPGA deployment.

Available student options:

```text
--student_arch cnn6
--student_arch mobilestyle
```

## Distillation losses

Paper-faithful soft-only KD:

```text
L = CE_soft(p_teacher, p_student)
```

Practical mixed KD for imbalanced medical data:

```text
L = alpha * T^2 * KLDiv(log_softmax(student_logits/T), teacher_probs_T)
    + (1 - alpha) * CE_weighted(student_logits, hard_label)
```

Defaults:

```text
--kd_mode both
--kd_loss soft_only
--temperature 2.0
--alpha 0.7
```

Use `--kd_loss mixed` when COPD recall/sensitivity is unstable or class imbalance dominates the soft-label signal.

## Metrics

The script reports validation and test metrics:

- Accuracy
- Precision per class
- Recall/Sensitivity per class
- F1 per class
- Macro F1
- Weighted F1
- Specificity per class
- Balanced accuracy
- One-vs-rest AUC when computable
- Confusion matrix
- Adapted ICBHI 3-class score

Primary metric: `macro_f1` by default.

For this medical task, macro-F1 and balanced accuracy are more important than plain accuracy because ICBHI disease labels are imbalanced. COPD sensitivity should be tracked closely if the model is used for screening; specificity is important to avoid excessive false positives.

Adapted ICBHI-style score:

```text
normal specificity = recall of Healthy as Healthy
abnormal sensitivity = fraction of COPD/Non-COPD predicted as non-Healthy
score = (normal specificity + abnormal sensitivity) / 2
```

## Weights & Biases logging

WandB logging is supported with:

```bash
python3 python/training/kd_icbhi_3class.py --stage all --wandb
```

Defaults:

```text
project: icbhi-3class-kd
entity: vhieu4344
```

The script does not store an API key. It uses the existing local WandB login. Make targets enable WandB by default; disable it for local smoke tests with `make train-kd-icbhi WANDB=` or use `--wandb_mode disabled`.

## Full run

```bash
make train-kd-icbhi
```

Equivalent direct command:

```bash
python3 python/training/kd_icbhi_3class.py \
  --stage all \
  --data_dir data/sample_01/ICBHI_final_database \
  --labels_file data/sample_01/labels.txt \
  --output_dir artifacts/training/icbhi_3class_kd \
  --num_teachers 5 \
  --teacher_seeds 1,2,3,4,5 \
  --kd_mode both \
  --wandb
```

## Smoke test

```bash
python3 python/training/kd_icbhi_3class.py \
  --stage all \
  --output_dir artifacts/training/icbhi_3class_kd_smoke \
  --num_teachers 2 \
  --teacher_seeds 1,2 \
  --epochs_teacher 1 \
  --epochs_student 1 \
  --max_files 12 \
  --batch_size 4 \
  --wandb_mode offline
```

## FPGA deployment notes

- Keep the student as `cnn6` or `mobilestyle`.
- Prefer ReLU over SiLU/GELU for DPU friendliness.
- BatchNorm can be folded into Conv during inference/export.
- Use the trained student checkpoint as the input to a later INT8 QAT/export flow.
- Keep generated checkpoints and soft labels under `artifacts/training/icbhi_3class_kd/`; they are ignored by git.
