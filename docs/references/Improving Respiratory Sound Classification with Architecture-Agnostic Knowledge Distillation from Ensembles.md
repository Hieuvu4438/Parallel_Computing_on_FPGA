# Improving Respiratory Sound Classification with Architecture-Agnostic Knowledge Distillation from Ensembles

**Authors:** Miika Toikkanen, June-Woo Kim  
**Affiliations:** RSC LAB, MODULABS, Republic of Korea; Department of Psychiatry, Wonkwang University Hospital, Republic of Korea  
**Venue:** arXiv:2505.22027v1 [cs.SD], 28 May 2025  
**Code:** https://github.com/RSC-Toolkit/rsc-ensemble-kd

***

## Abstract

Respiratory sound datasets bị giới hạn về kích thước và chất lượng, khiến việc đạt hiệu suất cao rất khó khăn. Ensemble models giúp cải thiện hiệu suất nhưng tăng chi phí tính toán tại inference. Nghiên cứu này khám phá **soft label distillation** như một phương pháp agnostic về kiến trúc để chắt lọc kiến thức từ ensemble giáo viên vào mô hình học sinh. Kết quả đạt **SOTA Score 64.39 trên ICBHI**, vượt qua kết quả tốt nhất trước đó 0.85, và cải thiện trung bình Score qua các kiến trúc hơn 1.16.

***

## 1. Giới thiệu

### Bối cảnh nghiên cứu

- Respiratory Sound Classification (RSC) là lĩnh vực nghiên cứu tích cực do tiềm năng hỗ trợ chẩn đoán bệnh hô hấp.
- Các công trình trước chủ yếu dùng CNN: ResNet, EfficientNet, CNN6.
- Gần đây, **Audio Spectrogram Transformer (AST)** pretrained trên ImageNet và AudioSet cho thấy lợi thế của self-attention.
- Mô hình **BTS (Bridging Text and Sound)** tận dụng textual metadata prompts, đạt SOTA 63.54% trên ICBHI.

### Vấn đề

- Dataset hô hấp chất lượng cao là tài nguyên khan hiếm.
- Ensemble models tăng hiệu suất nhưng tốn kém tính toán tại inference.
- Ứng dụng knowledge distillation vào RSC vẫn còn ít được khám phá.

### Đóng góp chính

1. Đặt SOTA mới **64.39** trên ICBHI với soft label distillation từ ensemble.
2. Chứng minh rằng chỉ một giáo viên duy nhất đã cải thiện đáng kể hiệu suất học sinh; chỉ cần vài giáo viên là đủ tối ưu.
3. Khám phá **second-generation ensemble** (BTS-d++) tăng ICBHI score từ 64.34 lên **65.45**.
4. Phát hành code công khai để hỗ trợ tái hiện kết quả.

***

## 2. Dataset

### ICBHI Respiratory Sound Dataset

| Thuộc tính | Thông tin |
|---|---|
| Tổng thời lượng | ~5.5 giờ |
| Tổng số breathing cycles | 6,898 |
| Train split | 60% → 4,142 cycles |
| Test split | 40% → 2,756 cycles |
| Số lớp | 4: *normal*, *crackle*, *wheeze*, *both* |
| Điều kiện chia | Không có patient overlap giữa train/test |

### Metadata

- **Age group:** binarized thành *adults* (> 18 tuổi) và *pediatrics* (≤ 18 tuổi)
- **Sex:** giữ nguyên theo annotation gốc
- **Recording location:** giữ nguyên
- **Recording device:** giữ nguyên

***

## 3. Data Preprocessing

### Xử lý âm thanh

- **Trích xuất breathing cycles** từ waveform gốc.
- **Chuẩn hóa độ dài:** mỗi cycle được chuẩn hóa về **8 giây**.
- **Resampling:** 16 kHz (tất cả mô hình), ngoại trừ BTS dùng **48 kHz**.

### Data Augmentation

- **SpecAugment** được áp dụng cho tất cả kiến trúc, **ngoại trừ** BTS và CLAP.

***

## 4. Chi tiết Huấn luyện

### Transformer-based models (BTS, CLAP, AST)

| Tham số | Giá trị |
|---|---|
| Optimizer | Adam |
| Learning rate | 5e-5 |
| LR scheduling | Cosine |
| Batch size | 8 |
| Epochs | 50 |

### Các kiến trúc khác (ResNet, EfficientNet, CNN6...)

| Tham số | Giá trị |
|---|---|
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 200 |

***

## 5. Phương pháp (Method)

### 5.1. BTS Model (Teacher chính)

**BTS (Bridging Text and Sound)** là framework đa phương thức kết hợp âm thanh hô hấp với textual metadata prompts:

- Dựa trên pretrained **LAION-CLAP**.
- **Acoustic encoder:** trích xuất đặc trưng âm thanh.
- **Text encoder:** xử lý metadata (age group, sex, stethoscope, location) dưới dạng text prompts.
- Hai modality được căn chỉnh trong không gian latent chung.
- Chi phí tính toán cao hơn so với các mô hình RSC khác → phù hợp làm teacher cho knowledge distillation.

### 5.2. BTS++: Ensemble of Strong Models

- Huấn luyện **30 mô hình BTS** với các seeds khác nhau.
- Mean ICBHI Score của 30 mô hình đơn lẻ: **63.41 ± 0.77**.
- **Tạo ensemble predictions:** trung bình hóa logits, chọn class có logit lớn nhất.
- Ký hiệu logits: $$z \in \mathbb{R}^{N \times C}$$ với $$N$$ là số mô hình, $$C$$ là số lớp.
- **BTS++[k=5]:** ensemble 5 predictors; **BTS++[k=30]:** ensemble 30 predictors.

### 5.3. BTS-d: Soft Label Distillation

Phương pháp **response-based knowledge distillation**, trong đó teacher là ensemble và student học bắt chước output của teacher.

#### a) Mean Teacher

Tính trung bình logits của tất cả $$k$$ mô hình, sau đó áp dụng softmax:

$$
p_\mu = \text{softmax}\!\left(\frac{1}{k}\sum_{i=1}^{k} z_i\right)
$$

Đại diện cho **ý kiến trung bình** của tất cả $$k$$ teachers.

#### b) Random Teacher

Lấy mẫu ngẫu nhiên một chỉ số teacher $$i$$ từ $$\{1, \ldots, k\}$$ trong mỗi iteration:

$$
p_r = \text{softmax}(z_i), \quad i \sim \text{Uniform}(\{1, \ldots, k\})
$$

Nhãn được tạo **khác nhau ở mỗi epoch** → giúp mô hình học đa dạng hơn.

#### c) Knowledge Distillation Loss

Thay vì dùng hard labels $$y$$, student được tối ưu bằng **cross-entropy với soft labels**:

$$
\mathcal{L}_{CE} = H(y, \hat{y}) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

$$
\mathcal{L}_\mu = H(p_\mu, \hat{y}) \quad \text{(Mean Teacher Loss)}
$$

$$
\mathcal{L}_r = H(p_r, \hat{y}) \quad \text{(Random Teacher Loss)}
$$

> **Lưu ý quan trọng:** Hard labels **không được sử dụng** trong quá trình distillation. Chỉ có soft labels từ teacher làm target huấn luyện.

#### Ký hiệu

- `-d`: distilled student (ví dụ: BTS-d)
- `[k=N]`: số teachers sử dụng (ví dụ: BTS-d[k=5])
- `++`: ensemble version (ví dụ: BTS-d++[k=5])

### 5.4. BTS-d++: Second-Generation Ensemble

- Kết hợp các mô hình BTS-d đã được distill thành **second-generation ensemble**.
- Mỗi predictor trong ensemble đã mạnh hơn → ensemble tổng thể mạnh hơn đáng kể.
- BTS-d++[k=5] đạt **65.45**, cao hơn BTS++[k=5] (64.34) với **cùng chi phí inference**.

***

## 6. Evaluation Metrics

Sử dụng tiêu chuẩn đánh giá chính thức của ICBHI dataset:

| Metric | Định nghĩa |
|---|---|
| **Sensitivity (Se)** | Tỷ lệ các trường hợp bất thường hô hấp được phát hiện đúng |
| **Specificity (Sp)** | Tỷ lệ các trường hợp bình thường được phân loại đúng |
| **ICBHI Score** | Trung bình cộng của Se và Sp: $$\text{Score} = \frac{S_e + S_p}{2}$$ |

### Cách báo cáo

- Báo cáo **mean và variance** của Sp, Se, Score qua **5 lần chạy độc lập** với seeds `{1, 2, 3, 4, 5}`.
- Với ensemble models: báo cáo từ **single run** (không chạy nhiều seeds).

***

## 7. Kết quả Thực nghiệm

### 7.1. Kết quả chính trên ICBHI Dataset

| Method | Backbone | Pretraining | Score (%) |
|---|---|---|---|
| SE+SA | ResNet18 | - | 49.55 |
| LungRN+NL | ResNet-NL | - | 52.26 |
| RespireNet | ResNet34 | IN | 56.20 |
| Nguyen et al. (CoTuning) | ResNet50 | IN | 58.29 |
| Bae et al. (Patch-Mix CL) | AST | IN+AS | 62.37 |
| Daisuke et al. (M2D-X/0.7) | M2D ViT | AS | 63.29 |
| Kim et al. (BTS) *(previous SOTA)* | CLAP | LA | 63.54 |
| **AST-d[k=5] (ours)** | AST | IN+AS | 61.08 ± 1.26 |
| **Audio-CLAP-d[k=5] (ours)** | CLAP | LA | 63.63 ± 0.60 |
| **BTS-d[k=5] mean teacher (ours)** | CLAP | LA | 64.38 ± 0.36 |
| **BTS-d[k=15] random teacher (ours)** | CLAP | LA | **64.39 ± 0.42** |
| BTS++[k=5] (ensemble) | CLAP | LA | 64.34 |
| **BTS-d++[k=5] Second Gen (ours)** | CLAP | LA | **65.45** |
| BTS++[k=30] | CLAP | LA | 65.69 |

*IN = ImageNet, AS = AudioSet, LA = LAION-Audio-630K*

### 7.2. Hiệu quả trên Lightweight Architectures

Distillation được áp dụng cho nhiều kiến trúc với BTS++[k=5] làm mean teacher:

| Model | # Params | Score (Hard Label) | Score (Soft Label) | Gain |
|---|---|---|---|---|
| ResNet18 | 11.7M | 55.09 ± 0.82 | 56.26 ± 0.85 | +1.17 |
| EfficientNet | 5.3M | 56.33 ± 0.43 | 57.68 ± 1.48 | +1.35 |
| CNN6 | 4.8M | 57.17 ± 0.81 | 58.17 ± 0.60 | +1.00 |
| AST | 87.7M | 59.55 ± 0.88 | 61.08 ± 1.26 | +1.53 |
| Audio-CLAP | 28M | 62.56 ± 0.37 | 63.63 ± 0.60 | +1.07 |
| BTS | 153M | 63.54 ± 0.80 | 64.39 ± 0.42 | +0.85 |
| **Average** | - | 59.04 ± 0.69 | 60.20 ± 0.87 | **+1.16** |

**Nhận xét:** Tất cả các kiến trúc đều được hưởng lợi từ soft-label distillation. Specificity tăng nhiều hơn, Sensitivity giảm nhẹ → mô hình phân biệt tốt hơn giữa bình thường và bất thường.

### 7.3. Ablation Study

| Method | Sp (%) | Se (%) | Score (%) |
|---|---|---|---|
| Baseline (hard label) | 81.40 ± 2.57 | 45.67 ± 2.66 | 63.54 ± 0.80 |
| Noised Label (var=0.1) | 81.33 ± 2.90 | 44.10 ± 3.53 | 62.71 ± 0.47 |
| Noised Label (teacher var) | 76.53 ± 3.78 | 47.46 ± 2.36 | 62.00 ± 1.06 |
| Single Teacher | 83.18 ± 1.59 | 44.62 ± 1.56 | 63.90 ± 0.15 |
| Mean Teacher Ensemble (k=5) | 84.93 ± 2.25 | 43.82 ± 2.25 | 64.38 ± 0.36 |
| Random Teacher Ensemble (k=15) | 82.89 ± 2.14 | 45.90 ± 1.89 | 64.39 ± 0.42 |
| Curated Teacher Ensemble (k=5) | 84.28 ± 2.58 | 44.95 ± 2.57 | **64.61 ± 0.75** |
| Remove softmax | - | - | Failed to converge |

**Kết luận từ ablation:**
- Noised labels thực hiện kém hơn baseline → softness của labels không phải lý do cải thiện, mà chính là **thông tin teacher distribution**.
- Single teacher đã cải thiện đáng kể (63.54% → 63.90%).
- Ensemble teacher cho kết quả tốt nhất.
- Curated ensemble (chọn 5 checkpoint tốt nhất) → 64.61%.
- Loại bỏ softmax → không hội tụ được.

### 7.4. Diminishing Returns khi Tăng Số Teachers

- **Mean Teacher:** đạt đỉnh tại k=5, sau đó giảm dần.
- **Random Teacher:** đạt đỉnh muộn hơn tại k=15.
- Random teacher đạt giá trị cao hơn đôi chút, nhưng cần k lớn hơn.
- Tác giả chọn **Mean Teacher k=5 làm default** vì hiệu quả hơn.
- Tại k=5: distilled model có hiệu suất tương đương teacher ensemble dù chỉ bằng 1/5 kích thước.
- Sau điểm hội tụ của validation loss teacher: student bắt đầu **overfit** vào phân phối teacher labels.

***

## 8. Phân tích & Nhận xét

### Ưu điểm của phương pháp

1. **Architecture-agnostic:** áp dụng được cho mọi kiến trúc (CNN, Transformer, lightweight).
2. **Chi phí chỉ tăng tại training**, không tăng tại inference.
3. Mô hình nhỏ (ResNet18, EfficientNet, CNN6) có thể tiếp cận hiệu suất của teacher lớn.
4. Không cần thay đổi kiến trúc hay loss function phức tạp.

### Hạn chế & Hướng phát triển

- Khoảng cách vẫn còn giữa ensemble và distilled model.
- Cần khám phá thêm diverse architectures.
- Có thể áp dụng cho các bài toán tương tự: heart sound, ECG, pathological speech classification.

***

## 9. Kết luận

Nghiên cứu áp dụng **architecture-agnostic knowledge distillation** cho RSC sử dụng soft label training để trích xuất kiến thức từ teacher ensembles. Kết quả chứng minh:

- Distillation từ ensemble chuyển giao kiến thức hiệu quả sang lightweight student models.
- Đạt **SOTA 64.39** trên ICBHI (single model, không tăng inference cost).
- Second-generation ensemble BTS-d++ đạt **65.45** với cùng chi phí inference như BTS++[k=5].
- Kết quả có thể mở rộng sang các bài toán phân loại âm thanh y tế khác trong điều kiện data khan hiếm.

***

## Tài liệu tham khảo Chính

| Ref | Method | Venue |
|---|---|---|
|  Bae et al. | Patch-Mix Contrastive Learning (AST) | INTERSPEECH'23 |
|  Daisuke et al. | M2D-X Masked Modeling Duo | TASLP'24 |
|  Kim et al. | BTS: Bridging Text and Sound | INTERSPEECH'24 |
|  Rocha et al. | ICBHI Dataset | ICBHI 2017 |
|  Hinton et al. | Knowledge Distillation | arXiv 2015 |
|  Wu et al. | LAION-CLAP | ICASSP'23 |