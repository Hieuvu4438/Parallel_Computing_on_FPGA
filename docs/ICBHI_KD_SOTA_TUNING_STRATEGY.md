# Chiến lược tinh chỉnh để nâng metrics cho các pipeline ICBHI KD

Tài liệu này đề xuất hướng tối ưu cho ba script:

- `python/training/kd_icbhi_4class_efficientnet.py`
- `python/training/kd_icbhi_3class.py`
- `python/training/kd_cnn_bilstm_to_cnn.py`

Mục tiêu là tăng `Macro F1`, `Balanced Accuracy`, `Sensitivity`, `Specificity` và ICBHI-style score, không chỉ tăng plain accuracy. Với bài toán y sinh mất cân bằng lớp, accuracy cao nhưng Macro F1 thấp thường nghĩa là model đang dự đoán tốt lớp đa số nhưng bỏ sót lớp thiểu số.

## 1. Chẩn đoán nhanh kết quả hiện tại

### 1.1 `kd_icbhi_4class_efficientnet.py`

Lưu ý: tên file có `4class`, nhưng code hiện tại đang train 3 lớp: `COPD`, `Non-COPD`, `Healthy`.

| Metric test | Mean | Std | Nhận xét |
|---|---:|---:|---|
| ICBHI score | 80.53% | 1.98% | Tương đối ổn so với các pipeline còn lại |
| Sensitivity | 82.10% | 5.42% | Khá tốt |
| Specificity | 78.97% | 8.68% | Dao động lớn giữa seed |
| Macro F1 | 50.70% | 7.01% | Thấp, model vẫn yếu ở lớp thiểu số |
| Accuracy | 75.62% | 3.32% | Không phản ánh đầy đủ do imbalance |

Vấn đề chính:

- EfficientNet-B0 có backbone mạnh hơn CNN6, nhưng vẫn dùng single split cố định và nhiều seed, chưa có GroupKFold đầy đủ.
- `best_val_icbhi_score` cao nhưng `test_macro_f1` thấp, cho thấy selection metric chưa tối ưu cho mục tiêu phân lớp cân bằng.
- Teacher logits mặc định lấy từ pipeline `icbhi_3class_kd`, nên chất lượng student bị giới hạn bởi chất lượng teacher ensemble hiện tại.
- Cần làm rõ tên file hoặc mục tiêu: nếu là 3-class disease-level thì không nên gọi là `4class`; nếu muốn 4-class official ICBHI respiratory event task thì phải đổi label pipeline.

### 1.2 `kd_icbhi_3class.py`

| Model | ICBHI score | Accuracy | Macro F1 | Balanced Acc | AUC OvR | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| Teacher ensemble | 77.32% | 78.57% | 56.80% | 63.33% | 89.44% | Teacher chưa đủ mạnh |
| Student mean KD | 78.33% | 79.41% | 57.37% | 63.90% | 89.88% | Tốt nhất hiện tại trong file này |
| Student random KD | 59.97% | 75.14% | 50.99% | 54.05% | 83.72% | Random teacher làm giảm Healthy F1 mạnh |

Vấn đề chính:

- `Student mean KD` nhỉnh hơn teacher ensemble, nhưng chênh lệch nhỏ; teacher chưa tạo soft labels đủ tốt.
- `random KD` không ổn định, đặc biệt làm Healthy F1 giảm mạnh.
- Teacher hiện chỉ gồm `small_resnet` hoặc `efficient_cnn`, chưa tiệm cận các teacher hiện đại như AST/HTS-AT/PANNs/ConvNeXt/EfficientNetV2.
- Student `cnn6` phù hợp FPGA nhưng capacity thấp; nên có hai nhánh: nhánh SOTA teacher và nhánh deployable student.

### 1.3 `kd_cnn_bilstm_to_cnn.py`

| Model | Accuracy mean | Macro F1 mean | Balanced Acc mean | Macro Sensitivity mean | Macro Specificity mean |
|---|---:|---:|---:|---:|---:|
| Teacher CNN-BiLSTM | 81.11% | 45.01% | 45.86% | 45.86% | 78.64% |
| Student CNN6 | 81.38% | 45.27% | 48.81% | 48.81% | 77.74% |

Vấn đề chính:

- Accuracy cao nhưng Macro F1/Balanced Acc thấp, dấu hiệu rất rõ của class imbalance.
- Pipeline đang preprocess ở 4 kHz, `FMAX=2000`; có thể làm mất thông tin hữu ích so với 8 kHz hoặc 16 kHz.
- Dùng file-level sample thay vì tận dụng đầy đủ respiratory-cycle annotations như các file KD khác.
- Có oversampling thủ công bằng copy mẫu, dễ làm model memorize minority class.
- Teacher CNN-BiLSTM chưa mạnh hơn student đáng kể; KD không có lợi nhiều nếu teacher yếu.

## 2. Nguyên tắc tối ưu chung

### 2.1 Chuẩn hóa protocol trước khi tối ưu model

Trước khi thay kiến trúc, cần khóa protocol để metrics đáng tin:

| Hạng mục | Khuyến nghị |
|---|---|
| Split | Dùng patient-wise split hoặc StratifiedGroupKFold; tuyệt đối không leak patient giữa train/val/test |
| Primary metric | `macro_f1` hoặc `balanced_accuracy`; không chọn best model chỉ theo accuracy |
| Secondary metric | ICBHI-style score, sensitivity, specificity, per-class F1 |
| Báo cáo | Mean ± std qua 5 folds hoặc official split, kèm per-class metrics |
| Seed | Ít nhất 3 seed cho cấu hình tốt nhất |
| Calibration | Theo dõi confidence, AUC, confusion matrix, threshold nếu cần screening |

Nếu muốn so sánh với SOTA ICBHI official, cần phân biệt rõ:

- Official ICBHI thường là respiratory event classification 4 lớp: normal, crackle, wheeze, both.
- Các script hiện tại chủ yếu là disease-level 3 lớp: COPD, Non-COPD, Healthy.
- Không nên so trực tiếp score 3-class disease-level với SOTA official 4-class nếu label task khác nhau.

### 2.2 Tập trung vào Macro F1/Balanced Accuracy

Plain accuracy dễ bị COPD chi phối. Với mục tiêu thực tế:

- Nếu dùng như screening: ưu tiên sensitivity cho abnormal/COPD.
- Nếu dùng phân biệt Healthy vs bệnh: ưu tiên specificity Healthy và Healthy F1.
- Nếu dùng báo cáo học thuật: ưu tiên Macro F1, Balanced Accuracy, ICBHI score và per-class F1.

### 2.3 Teacher phải mạnh trước, student mới distill tốt

Knowledge Distillation chỉ hiệu quả khi teacher có signal tốt hơn student. Vì vậy lộ trình nên là:

1. Tối ưu teacher mạnh nhất có thể.
2. Tạo soft labels sạch và aligned với split.
3. Distill sang student FPGA-friendly.
4. Dùng QAT/pruning/BN folding sau khi student đã đạt metric tốt.

## 3. Hướng SOTA nên đưa vào repo

Các hướng SOTA/near-SOTA cho respiratory sound hiện nay thường kết hợp một hoặc nhiều thành phần sau:

| Nhóm kỹ thuật | Áp dụng đề xuất |
|---|---|
| Audio Transformer | AST, HTS-AT hoặc PaSST làm teacher mạnh |
| Self-supervised/pretrained audio | BEATs, PANNs/Cnn14, AudioMAE, wav2vec-style embeddings làm teacher/feature extractor |
| Sharpness-Aware Minimization | SAM hoặc ASAM cho teacher để tăng generalization |
| Strong augmentation | cyclic padding, MixUp/CutMix, SpecAugment, noise, time shift, speed perturbation |
| Multi-resolution features | log-Mel nhiều scale, CQT/chromagram/cochleogram nếu có thời gian thử nghiệm |
| Ensemble | ensemble nhiều architecture/fold/seed, sau đó distill sang student |
| Test-time augmentation | crop/shift ensemble, average probability/logits |

Kết quả web lookup nhanh cũng cho thấy hướng AST + SAM + cyclic padding đang được dùng trong các repo tự nhận SOTA cho ICBHI. Vì vậy, nếu mục tiêu là tiệm cận SOTA, nên thêm một teacher transformer/pretrained audio model thay vì chỉ tăng CNN nhỏ.

## 4. Kế hoạch cho từng file

## 4.1 `kd_icbhi_3class.py`: biến thành pipeline teacher-ensemble mạnh nhất

Đây nên là pipeline trung tâm để tạo teacher logits tốt cho các student khác.

### Mục tiêu

- Nâng teacher ensemble trước, sau đó distill sang student deployable.
- Giữ patient-wise split và soft-label alignment ổn định.
- Tăng Macro F1 và Balanced Accuracy, đặc biệt Healthy/Non-COPD F1.

### Thay đổi ưu tiên cao

| Ưu tiên | Thay đổi | Kỳ vọng |
|---:|---|---|
| 1 | Thêm teacher architecture mạnh: `ast`, `htsat`, `panns_cnn14`, `efficientnetv2`, `convnext_tiny` | Tăng chất lượng soft labels |
| 2 | Thêm SAM optimizer option cho teacher | Tăng generalization, giảm overfit |
| 3 | Mặc định dùng `--kd_loss mixed` thay vì `soft_only` khi student yếu | Giữ hard-label signal cho lớp thiểu số |
| 4 | Ưu tiên `prob_mean` hoặc `mean`; bỏ `random` khỏi default sweep | Random KD hiện đang kém ổn định |
| 5 | Thêm class-balanced focal hoặc LDAM/CB loss cho teacher | Cải thiện minority class |
| 6 | Thêm MixUp/CutMix trên spectrogram và class-aware augmentation | Giảm memorize minority class |
| 7 | Thêm TTA khi evaluate teacher/student | Tăng robustness test |

### Kiến trúc teacher đề xuất

| Teacher | Khi dùng | Ghi chú |
|---|---|---|
| AST/PaSST | Mục tiêu score cao nhất | Cần pretrain AudioSet/ImageNet hoặc checkpoint phù hợp |
| HTS-AT | Audio tagging mạnh, tốt cho spectrogram | Teacher nặng nhưng chỉ dùng training |
| PANNs Cnn14 | Audio pretrained mạnh, ổn định trên data nhỏ | Có thể fine-tune classifier head trước |
| EfficientNetV2-S/B0 | Dễ tích hợp vào code hiện tại | Tốt nếu dùng 3-channel log-Mel augmentation |
| ConvNeXt-Tiny | CNN hiện đại, tốt trên spectrogram image | Teacher thay thế small_resnet |

### Student đề xuất

| Student | Mục tiêu | Ghi chú |
|---|---|---|
| `cnn6` hiện tại | FPGA an toàn nhất | Baseline deployable |
| DS-CNN + SE nhẹ | Tăng F1 nhưng vẫn deployable | SE có thể cần cân nhắc FPGA |
| MobileNetV3-Small/ReLU variant | Accuracy tốt hơn CNN6 | Đổi SiLU/HardSwish nếu DPU không hỗ trợ tốt |
| RepVGG-A0 tiny | Train mạnh, deploy thành Conv thuần | Rất đáng thử cho FPGA do re-parameterization |

### Hyperparameter sweep đề xuất

| Nhóm | Giá trị thử |
|---|---|
| Teacher LR | `1e-4`, `3e-4`, `1e-3` |
| Teacher weight decay | `1e-4`, `1e-3`, `1e-2` |
| Hard loss | `ce`, `focal` |
| Focal gamma | `1.0`, `2.0`, `3.0` |
| KD temperature | `1`, `2`, `4`, `6` |
| KD alpha mixed | `0.3`, `0.5`, `0.7` |
| KD mode | `mean`, `prob_mean` |
| Batch size | `16`, `32`, `64` tùy GPU |
| Selection metric | `macro_f1`, `balanced_accuracy`, composite score |

Composite score nên thử:

```text
score = 0.40 * macro_f1
      + 0.30 * balanced_accuracy
      + 0.20 * icbhi_3class_score
      + 0.10 * healthy_f1
```

Lý do: hiện model dễ bỏ qua Healthy/Non-COPD, nên thêm minority-class signal vào checkpoint selection.

### Lộ trình implement

1. Thêm abstraction `make_teacher_model()` riêng với các architecture mạnh.
2. Thêm CLI:
   - `--teacher_arch ast|htsat|panns_cnn14|efficientnetv2|convnext_tiny|small_resnet|efficient_cnn`
   - `--optimizer adamw|sam`
   - `--mixup_alpha`, `--cutmix_alpha`, `--tta_crops`
   - `--selection_metric composite|macro_f1|balanced_accuracy|icbhi_3class_score`
3. Train teacher mạnh trên 5 folds/seeds.
4. Tạo soft labels bằng ensemble logits/probabilities.
5. Distill `cnn6`, `mobilestyle`, `repvgg_tiny`.
6. Chọn student theo Macro F1/Balanced Acc, sau đó mới optimize FPGA.

### Cấu hình khởi đầu nên thử

```bash
python3 python/training/kd_icbhi_3class.py \
  --stage all \
  --output_dir artifacts/training/icbhi_3class_kd_ast_sam \
  --teacher_arch ast \
  --student_arch cnn6 \
  --num_teachers 5 \
  --teacher_seeds 1,2,3,4,5 \
  --kd_mode prob_mean \
  --kd_loss mixed \
  --temperature 4.0 \
  --alpha 0.5 \
  --hard_loss focal \
  --focal_gamma 2.0 \
  --balanced_sampler \
  --selection_metric macro_f1
```

Nếu AST quá nặng hoặc thiếu checkpoint, fallback nên là `convnext_tiny` hoặc `efficientnetv2`.

## 4.2 `kd_icbhi_4class_efficientnet.py`: biến thành strong spectrogram student/teacher

### Mục tiêu

File này hiện là EfficientNet-B0 3-class KD. Nên dùng làm nhánh strong CNN spectrogram model, không nhất thiết FPGA-friendly.

### Vấn đề cần sửa trước

| Vấn đề | Hành động |
|---|---|
| Tên file `4class` nhưng code 3-class | Đổi tên file hoặc đổi task thật sự thành 4-class official; trước mắt ghi rõ là 3-class |
| Test Macro F1 thấp dù ICBHI score cao | Chọn checkpoint theo composite metric hoặc macro-F1, không chỉ ICBHI score |
| Split lấy từ teacher dir cố định | Cho phép truyền `--split_dir` rõ ràng để tránh silent mismatch |
| Chỉ EfficientNet-B0 | Thêm B1/B2/EfficientNetV2/ConvNeXt/Swin-Tiny teacher options |

### Thay đổi ưu tiên cao

| Ưu tiên | Thay đổi | Kỳ vọng |
|---:|---|---|
| 1 | Dùng `selection_metric=macro_f1` hoặc composite thay vì mặc định ICBHI score | Tăng per-class balance |
| 2 | Thêm MixUp/CutMix + label smoothing vào hard loss | Tăng generalization |
| 3 | Thêm SAM/ASAM cho EfficientNet fine-tuning | Giảm overfit theo seed |
| 4 | Fine-tune theo 2 phase: freeze backbone rồi unfreeze gradual | Ổn định pretrained model |
| 5 | Thử EfficientNetV2-S hoặc ConvNeXt-Tiny | Backbone mạnh hơn B0 |
| 6 | TTA nhiều crop/time-shift khi evaluate | Tăng test robustness |
| 7 | Ensemble top-k checkpoints theo seed/fold | Tăng metrics cuối |

### Training schedule đề xuất

| Phase | Epoch | LR | Layer | Ghi chú |
|---|---:|---:|---|---|
| Warmup head | 5-10 | `1e-3` | classifier only | Nhanh, tránh phá pretrained features |
| Fine-tune top blocks | 20-40 | `3e-4` head, `3e-5` backbone top | last 30-50% layers | Dùng cosine + warmup |
| Full fine-tune | 20-60 | `1e-4` head, `1e-5` backbone | all layers | Early stop theo macro-F1/composite |
| SWA/EMA optional | 5-10 | nhỏ | all | Smooth checkpoint |

### Augmentation đề xuất

| Loại | Giá trị |
|---|---|
| Waveform time shift | `0.05-0.15` |
| Noise SNR | `10-30 dB` thay vì chỉ noise std cố định |
| Speed perturbation | `0.9`, `1.0`, `1.1` |
| SpecAugment | `freq_mask=16-32`, `time_mask=64-128` |
| MixUp | `alpha=0.2-0.4` |
| CutMix spectrogram | `alpha=0.5-1.0`, xác suất thấp hơn MixUp |

### Model variants nên thử

| Variant | Mục tiêu |
|---|---|
| EfficientNet-B0 current | Baseline |
| EfficientNet-B1/B2 | Tăng capacity vừa phải |
| EfficientNetV2-S | Strong CNN spectrogram |
| ConvNeXt-Tiny | Strong CNN hiện đại |
| Swin-Tiny/AST | Nếu chấp nhận transformer teacher |

### Cấu hình khởi đầu nên thử

```bash
python3 python/training/kd_icbhi_4class_efficientnet.py \
  --output_dir artifacts/training/icbhi_3class_efficientnetv2_kd_tuned \
  --loss_mode mixed \
  --kd_mode mean_teacher \
  --alpha 0.5 \
  --selection_metric macro_f1 \
  --input_channels 3 \
  --n_mels 128 \
  --target_frames 512 \
  --batch_size 32 \
  --lr 3e-4 \
  --weight_decay 1e-3 \
  --focal_gamma 2.0 \
  --patience 20 \
  --seeds 1,2,3,4,5
```

Sau khi thêm composite metric, ưu tiên chạy:

```text
selection_metric = composite
alpha = 0.5
teacher = best ensemble từ kd_icbhi_3class.py
optimizer = SAM/AdamW
```

## 4.3 `kd_cnn_bilstm_to_cnn.py`: sửa data protocol và làm teacher thực sự mạnh

### Mục tiêu

Pipeline này nên trở thành nhánh CRNN teacher -> pure CNN/FPGA student. Hiện tại teacher chưa đủ mạnh, nên KD không cải thiện rõ.

### Thay đổi bắt buộc

| Ưu tiên | Thay đổi | Lý do |
|---:|---|---|
| 1 | Dùng respiratory-cycle annotations thay vì file-level sample | Tăng số mẫu, đúng hơn với ICBHI |
| 2 | Tăng sample rate lên 8 kHz hoặc 16 kHz | 4 kHz có thể mất chi tiết wheeze/crackle |
| 3 | Bỏ naive oversampling copy, thay bằng WeightedRandomSampler + augmentation-based balancing | Giảm memorization |
| 4 | Đồng bộ class order với hai file còn lại: `COPD`, `Non-COPD`, `Healthy` | Tránh nhầm metric/class |
| 5 | Chọn best checkpoint theo Macro F1/Balanced Acc/composite | Accuracy đang misleading |
| 6 | Lưu per-fold predictions/probabilities để phân tích lỗi | Biết lớp nào đang chết |

### Teacher upgrade

CNN-BiLSTM hiện chưa đủ mạnh. Nên thử:

| Teacher | Chi tiết |
|---|---|
| CNN-BiGRU/BiLSTM + attention pooling | Thêm attention theo time thay vì global max pool |
| CRNN ResNet front-end | ResNet/ConvNeXt feature extractor + BiGRU |
| AST/HTS-AT teacher external | Dùng làm upper-bound và distill vào CNN6 |
| Multi-scale CRNN | Mel 64/128 + hop khác nhau, ensemble logits |

CRNN teacher đề xuất:

```text
Input log-Mel 1 x 128 x T
Backbone: ResNet-like CNN hoặc ConvNeXt tiny blocks
Temporal: BiGRU/BiLSTM hidden 128-256
Pooling: attention pooling + mean/max pooling concat
Classifier: LayerNorm -> Dropout -> Linear
Loss: class-balanced focal + label smoothing
Optimizer: AdamW + SAM optional
```

### Student upgrade

Student vẫn cần FPGA-friendly, nhưng có thể tăng nhẹ capacity:

| Student | Ghi chú |
|---|---|
| CNN6 current | Baseline |
| CNN8 narrow | Thêm 2 conv block nhưng giữ channel nhỏ |
| RepVGG tiny | Train tốt, deploy Conv thuần |
| DS-CNN ReLU | Ít params, nhưng depthwise support FPGA cần kiểm tra |

### Training schedule đề xuất

| Stage | Mô tả |
|---|---|
| 1 | Train teacher trên cycle-level data, GroupKFold, focal loss |
| 2 | Generate clean teacher logits cho train/val/test cycle IDs |
| 3 | Distill student bằng mixed KD, không chỉ on-the-fly teacher logits |
| 4 | Fine-tune student bằng hard labels với LR nhỏ và class-balanced sampler |
| 5 | Export best student theo fold hoặc ensemble-distilled student |

### Cấu hình preprocessing đề xuất

| Tham số | Hiện tại | Đề xuất |
|---|---:|---:|
| `TARGET_SR` | 4000 | 8000 hoặc 16000 |
| `N_MELS` | 128 | 64 và 128 để sweep |
| `N_FFT` | 512 | 1024 nếu 16 kHz |
| `HOP_LENGTH` | 128 | 160 hoặc 256 tùy SR |
| Segment | file center crop 8 s | cycle annotation + cyclic padding |
| Augment | noise/shift/stretch | thêm SNR noise, SpecAugment mạnh, MixUp |

### Cấu hình khởi đầu nên thử

```bash
python3 python/training/kd_cnn_bilstm_to_cnn.py \
  --output_dir artifacts/training/kd_cnn_bilstm_to_cnn_cycle_tuned \
  --n_folds 5 \
  --batch_size 32 \
  --teacher_epochs 100 \
  --student_epochs 120 \
  --kd_temperature 4.0 \
  --kd_alpha 0.5 \
  --no_wandb
```

Trước khi chạy cấu hình này, cần sửa code để hỗ trợ cycle-level records, sampler cân bằng và selection metric không dựa vào accuracy.

## 5. Lộ trình thí nghiệm theo thứ tự ưu tiên

## Phase A: Làm protocol đáng tin

Mục tiêu: metrics không bị nhiễu bởi split/imbalance/leakage.

1. Chuẩn hóa class order trong cả ba file.
2. Dùng StratifiedGroupKFold hoặc một split patient-wise được lưu rõ ràng.
3. Lưu `splits.json`, `config.json`, predictions và confusion matrix cho mọi run.
4. Đổi checkpoint selection sang `macro_f1` hoặc composite.
5. Tạo script/tùy chọn tổng hợp metrics tự động từ output dirs.

Tiêu chí pass:

```text
Mỗi run có đầy đủ:
- per-class precision/recall/F1
- confusion matrix
- macro_f1
- balanced_accuracy
- sensitivity/specificity
- ICBHI-style score
- mean/std qua fold hoặc seed
```

## Phase B: Nâng baseline CNN/EfficientNet

1. `kd_icbhi_4class_efficientnet.py`: thêm composite metric, MixUp, SAM, gradual unfreeze.
2. `kd_icbhi_3class.py`: dùng focal/mixed KD, bỏ random KD khỏi default.
3. `kd_cnn_bilstm_to_cnn.py`: bỏ oversampling copy, tăng SR, dùng cycle-level.

Tiêu chí kỳ vọng:

```text
Macro F1 tăng rõ ở Healthy và Non-COPD.
Balanced Accuracy không còn quanh 45-50% ở CNN-BiLSTM pipeline.
Std giữa seed/fold giảm.
```

## Phase C: Thêm teacher SOTA

1. Thêm `AST` hoặc `HTS-AT` teacher.
2. Nếu thiếu tài nguyên, thêm `PANNs Cnn14`, `ConvNeXt-Tiny`, `EfficientNetV2-S` trước.
3. Train 3-5 teacher khác architecture/seed.
4. Ensemble bằng probability mean và logits mean.
5. Distill sang CNN6/RepVGG tiny.

Tiêu chí kỳ vọng:

```text
Teacher Macro F1 và Balanced Accuracy phải cao hơn student ít nhất 5-10 điểm.
Nếu teacher không hơn student, không nên distill từ teacher đó.
```

## Phase D: Deployable student tốt nhất

1. Chọn top teacher ensemble.
2. Distill các student:
   - CNN6
   - MobileStyle ReLU
   - RepVGG tiny
   - CNN8 narrow
3. Chọn student theo Pareto: Macro F1/Balanced Acc vs params/FLOPs/FPGA compatibility.
4. Sau đó mới làm QAT, pruning, BN folding.

## 6. Ma trận ablation nên chạy

### 6.1 Ablation nhanh

| ID | Pipeline | Thay đổi | Metric chính |
|---|---|---|---|
| A1 | EfficientNet | `selection_metric=macro_f1` | Macro F1 |
| A2 | EfficientNet | A1 + MixUp | Macro F1, Healthy F1 |
| A3 | EfficientNet | A2 + SAM | Mean/std |
| B1 | 3class KD | `kd_loss=mixed`, `temperature=4`, `alpha=0.5` | Student Macro F1 |
| B2 | 3class KD | B1 + focal hard loss | Non-COPD/Healthy F1 |
| B3 | 3class KD | `prob_mean` vs `mean` | Stability |
| C1 | CNN-BiLSTM | no copy oversampling, WeightedRandomSampler | Balanced Acc |
| C2 | CNN-BiLSTM | SR 8k/16k | Macro F1 |
| C3 | CNN-BiLSTM | cycle-level data | All metrics |

### 6.2 Ablation SOTA

| ID | Teacher | Student | Optimizer | KD | Mục tiêu |
|---|---|---|---|---|---|
| S1 | ConvNeXt-Tiny | CNN6 | AdamW | mixed | Baseline strong CNN teacher |
| S2 | EfficientNetV2-S | CNN6 | SAM | mixed | CNN spectrogram upper-bound |
| S3 | PANNs Cnn14 | CNN6 | AdamW | mixed | Audio-pretrained teacher |
| S4 | AST/PaSST | CNN6 | SAM | mixed | SOTA-oriented teacher |
| S5 | AST ensemble | RepVGG tiny | SAM | mixed | Best deployable student |

## 7. Các thay đổi code cụ thể nên thêm

### 7.1 Metrics/selection

Thêm composite metric vào cả ba file:

```python
def selection_score(metrics):
    return (
        0.40 * metrics["macro_f1"]
        + 0.30 * metrics["balanced_accuracy"]
        + 0.20 * metrics.get("icbhi_3class_score", metrics.get("icbhi_score", 0.0))
        + 0.10 * metrics.get("healthy_f1", 0.0)
    )
```

Với `kd_cnn_bilstm_to_cnn.py`, metrics nested trong `per_class`, nên cần map `healthy_f1` từ `metrics["per_class"]["Healthy"]["f1_score"]`.

### 7.2 Loss

Thêm hoặc chuẩn hóa các loss:

- Class-balanced Focal Loss.
- Label smoothing.
- LDAM hoặc Class-Balanced Loss nếu focal chưa đủ.
- Mixed KD thay vì soft-only khi class imbalance nặng.

### 7.3 Augmentation

Thêm CLI chung:

```text
--mixup_alpha
--cutmix_alpha
--specaugment_p
--freq_mask
--time_mask
--speed_perturb
--snr_noise_min
--snr_noise_max
--tta_crops
```

### 7.4 Optimizer

Thêm:

- `AdamW` baseline.
- `SAM` cho teacher/strong model.
- EMA hoặc SWA cho checkpoint cuối.

### 7.5 Model registry

Tách model creation thành registry:

```text
make_teacher_model(name, num_classes)
make_student_model(name, num_classes)
```

Điều này giúp thử AST/HTS-AT/PANNs/EfficientNetV2/ConvNeXt mà không làm rối training loop.

## 8. Tiêu chí chọn cấu hình tốt nhất

Không chọn model chỉ vì accuracy cao. Chọn theo thứ tự:

1. Không có leakage, split đúng patient-wise.
2. Macro F1 cao nhất.
3. Balanced Accuracy cao nhất.
4. Healthy và Non-COPD F1 không bị sụp.
5. ICBHI-style score cao.
6. Std qua folds/seeds thấp.
7. Student đáp ứng params/FLOPs/FPGA constraints.

Ngưỡng mục tiêu thực tế cho repo này sau tuning:

| Pipeline | Mục tiêu ngắn hạn | Mục tiêu mạnh hơn |
|---|---:|---:|
| EfficientNet/ConvNeXt strong CNN | Macro F1 > 60%, Balanced Acc > 68% | Macro F1 > 70% |
| 3class KD student | Macro F1 > 60%, Balanced Acc > 68% | Macro F1 > 68% với CNN6/RepVGG tiny |
| CNN-BiLSTM -> CNN | Macro F1 > 55%, Balanced Acc > 60% | Macro F1 > 65% sau cycle-level + stronger teacher |
| SOTA teacher AST/HTS-AT | Macro F1 > 70% | Dùng làm upper-bound và teacher distillation |

Các mục tiêu này cần kiểm chứng bằng cùng protocol; không nên so với paper khác nếu task/split khác.

## 9. Thứ tự triển khai khuyến nghị

Nếu muốn hiệu quả nhanh nhất:

1. Sửa checkpoint selection sang `macro_f1`/composite trong cả ba file.
2. Với `kd_icbhi_3class.py`, chạy lại `Student mean/prob_mean KD` với `mixed KD + focal + temperature=4 + alpha=0.5`.
3. Với `kd_icbhi_4class_efficientnet.py`, thêm MixUp + SAM + gradual unfreeze, thử EfficientNetV2/ConvNeXt.
4. Với `kd_cnn_bilstm_to_cnn.py`, sửa data sang cycle-level và bỏ oversampling copy trước khi thay model.
5. Thêm teacher mạnh AST/HTS-AT/PANNs vào `kd_icbhi_3class.py`.
6. Distill teacher ensemble tốt nhất sang student FPGA-friendly.
7. Chỉ sau khi student tốt mới làm QAT/export FPGA.

## 10. Checklist trước khi chạy long experiment

- [ ] Xác nhận task là 3-class disease-level hay official 4-class respiratory event.
- [ ] Xác nhận không leak patient giữa train/val/test.
- [ ] Xác nhận class order thống nhất giữa scripts.
- [ ] Xác nhận `splits.json` và teacher logits cùng sample IDs.
- [ ] Log đầy đủ per-class metrics.
- [ ] Chọn checkpoint theo Macro F1/Balanced Acc/composite.
- [ ] Lưu confusion matrix và predictions.
- [ ] Chạy ít nhất 3 seed hoặc 5 folds cho cấu hình tốt nhất.
- [ ] So sánh model theo cùng split/protocol.
- [ ] Ghi lại params/FLOPs nếu student dùng cho FPGA.
