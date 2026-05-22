# Kế Hoạch Chỉnh Sửa Paper — IEEE JSTSP

## Mục tiêu

Chỉnh sửa paper hiện tại để phản ánh **đúng phương pháp đang thực sự triển khai**, đồng thời hoàn thiện các phần còn trống, sửa lỗi nội dung, và chỉn chu về mặt khoa học.

---

## Phân Tích Sự Khác Biệt: Paper ↔ Implementation Thực Tế

### ❶ Sự khác biệt về Task / Target Space

| Hạng mục | Paper Hiện Tại | Implementation Thực Tế |
|---|---|---|
| **Số lớp** | **5 lớp** (Healthy, Pneumonia, URTI, Bronchiectasis, COPD) — đề cập ở Abstract & Conclusion | **3 lớp** (COPD, Healthy, Non-COPD) — toàn bộ code |
| **Class Non-COPD** | Không đề cập | Gộp URTI, Pneumonia, Bronchiectasis, Asthma → Non-COPD |
| **Dataset** | Abstract nói "920 recordings từ ICBHI" + KAUH DB → 1256 recordings | Code dùng ICBHI + combined/DS2, cấu trúc 3-class |

> [!CAUTION]
> **Mâu thuẫn nghiêm trọng nhất**: Abstract và Conclusion nói "5 classes", "98.81% accuracy", nhưng toàn bộ pipeline code chỉ có 3 class. Đây là điểm phải sửa xuyên suốt, nhất quán.

---

### ❷ Sự khác biệt về Feature Extraction

| Hạng mục | Paper (Sec. 3.1.2) | Code Thực Tế |
|---|---|---|
| **Features Layer 1–3** | 43-dim: ZCR(2) + RMS(2) + MFCCs(26) + ΔMFCCs(13) | **Khớp** — `layer1_preprocessing.py` đúng 43 features |
| **Spectrogram Layer 4** | 3-channel: Gammatone + Mel + Avg | **Khớp** — code `create_hybrid_spectrogram()` |
| **Segment Duration** | Không nêu rõ | Code: **8 giây** (distillation_02.py: `SEGMENT_DURATION = 8`) |
| **BPF** | Layer 1: 50-2500Hz | distillation_02.py: **25-2000Hz** (khác!) |
| **BPF** | — | layer1_preprocessing.py: **50-2500Hz** (khớp paper) |

> [!WARNING]
> **BPF không thống nhất**: `layer1_preprocessing.py` dùng 50-2500Hz (khớp paper), nhưng `distillation_02.py` (Layer 4 training) dùng 25-2000Hz. Paper cần nêu rõ 2 track này dùng BPF khác nhau, hoặc thống nhất.

---

### ❸ Layer 4: Mô hình CNN / Distillation

| Hạng mục | Paper (Sec. 3.3) | Code Thực Tế |
|---|---|---|
| **Teacher** | EfficientNet-B0 ensemble (3 models) | **Khớp** — 3 teachers, 4 folds mỗi teacher |
| **Student** | MobileNetV2 | **Khớp** |
| **Loss** | Focal Loss + KL Divergence (temperature scaling) | **Khớp** — T=4, α=0.7 |
| **Số tham số MobileNetV2** | Table 6: `Params=3.5M, FLOPs=300M` | Checkpoint: ~11.6MB → ~3.4M params ✓ |
| **ShuffleNetV1** | Table 6 nêu cả ShuffleNetV1 | **Không có trong code** — chỉ có MobileNetV2 |
| **INT8 Method** | DPU (Vitis-AI implied) | **NVIDIA QAT** (`pytorch-quantization` + ONNX với QDQ nodes) |
| **Precision** | "INT8 (DSQS)" trong table 8 | NVIDIA PTQ/QAT → ONNX INT8 |

> [!IMPORTANT]
> **Hai điểm cần làm rõ trong paper:**
> 1. Loại bỏ hoặc giải thích ShuffleNetV1 trong Table 6 — thực tế chỉ dùng MobileNetV2
> 2. Nêu rõ pipeline INT8: **NVIDIA pytorch-quantization QAT → ONNX QDQ → TensorRT-compatible** (không phải Vitis-AI DPU native INT8)

---

### ❹ Kết quả thực nghiệm — Còn trống hoàn toàn

| Table/Figure | Tình trạng | Dữ liệu thực tế có |
|---|---|---|
| **Table 7** (accuracy) | Toàn `[XX.XX]` | Fold 0: 96.56%, Fold 1: 94.25%, Fold 2: 94.02% |
| **Table 8** (hardware) | Toàn `[XX.XX]` | Chưa có benchmark thực tế trên FPGA |
| **Table 9** (FPGA resources) | Toàn `[XX.XX]` | Chưa synthesis |
| **Fig. Early exit %** | "(data not shown)" | Chưa có |
| **QAT results** | Không đề cập | FP32: 97.98%, Fake-INT8: 97.47% (198 samples) |

---

### ❺ Các vấn đề viết lách / trình bày

| Vị trí | Vấn đề |
|---|---|
| `main.tex` Abstract | "98.81% accuracy" và "five disease classes" — **sai với thực tế 3-class** |
| Introduction (line 15) | "98.81% accuracy rate across five disease classes" |
| Conclusion | "five classes—healthy, COPD, URTI, bronchitis, and pneumonia" — **sai** |
| Table 4 caption | Ghi nhãn `\label{tab3b}` nhưng là Table 4 architecturally (nhầm label) |
| Table 6 label | `\label{tab5}` nhưng về nội dung là table về hyperparams (inconsistent) |
| Table `tab4` | Được gọi trong text qua `\ref{tab4}` nhưng không có table với label đó |
| Sec 3.2 line 64 | `Table \ref{tab4}` — link tới Table với config, nhưng label `tab4` là class distribution |
| References | Một số references (`\cite{b35}` cho MobileNetV2) cần kiểm tra có trong .bib |
| `Data Availability` | "No data was used..." — **sai** (dùng ICBHI 2017 public dataset) |

---

## Chiến Lược Sửa — Chia Theo Priority

### 🔴 PHẢI SỬA (Blocking — Paper Không Thể Submit)

#### 1. Thống nhất số lớp xuyên suốt paper → **3 class**
- **Abstract**: Sửa "five disease classes" → "three diagnostic categories (COPD, Non-COPD, and Healthy)"
- **Introduction** bullet 1: Sửa "98.81%" và "five classes"
- **Conclusion**: Sửa "five classes—healthy, COPD, URTI, bronchitis, and pneumonia" → "three classes"
- **Methodology (Sec 3.1)**: Giữ mô tả 3-class, giải thích mapping từ 8 bệnh → 3 nhóm

#### 2. Điền kết quả thực từ experiments
- **Table 7**: Điền metrics từ distillation metrics JSON (3-fold CV)
- **QAT/INT8**: Thêm bảng/đoạn mô tả kết quả QAT: FP32 97.98% → INT8 97.47%
- **Table 8 & 9**: Nếu chưa có benchmark FPGA thực, phải nêu rõ là "projected/estimated" hoặc cần bổ sung

#### 3. Sửa label cross-references trong LaTeX
- `table_4.tex` đang dùng `\label{tab3b}` — có thể nhầm (là class distribution table)
- `table_5.tex` label `\label{tab4}` — Framework config table
- `table_6.tex` label `\label{tab5}` — Hyperparams table
- Kiểm tra toàn bộ `\ref{}` trong methodology section

#### 4. Sửa "Data Availability"
- Thay "No data was used..." → "The ICBHI 2017 Respiratory Sound Database used in this study is publicly available at [cite]"

---

### 🟡 NÊN SỬA (Methodological Accuracy)

#### 5. Làm rõ 2 track preprocessing với BPF khác nhau
- Layer 1-3 (RF): BPF 50-2500Hz, 5 giây, feature-based
- Layer 4 (CNN): BPF 25-2000Hz, 8 giây, spectrogram-based
- Hoặc unify về một bộ preprocessing duy nhất

#### 6. Cập nhật mô tả INT8 quantization
- Thay thế mô tả chung "INT8 (DSQS)" → **NVIDIA pytorch-quantization QAT + ONNX QDQ export**
- Thêm chi tiết: PTQ calibration (Max calibrator, 32 batches) → QAT fine-tuning (5 epochs, LR=1e-5) → ONNX opset 13 với QDQ nodes → TensorRT deployment
- Kết quả: FP32 97.98% → Fake-INT8 97.47% (accuracy drop chỉ 0.51%)

#### 7. Loại bỏ ShuffleNetV1 khỏi Table 6
- Hoặc thay bằng một comparison thực tế khác (EfficientNet-B0 teacher vs MobileNetV2 student)

#### 8. Cập nhật Table 5 (Framework Config) với params thực tế
- Segment duration: 8s (Layer 4), 5s (Layer 1-3)
- Thêm cột quantization method cho Layer 4

---

### 🟢 HOÀN THIỆN (Polish)

#### 9. Bổ sung subsection về Knowledge Distillation details
- Teacher: 3 × EfficientNet-B0, 4× K-fold mỗi model → 12 training runs
- Student: MobileNetV2 với custom head (512→256→3)
- Loss: α=0.7 × KL(T=4) + 0.3 × FocalLoss(γ=2, smoothing=0.1)
- Augmentation: AudioAugmenter + SpecAugment

#### 10. Bổ sung subsection "NVIDIA QAT Pipeline"
- Mô tả: quant_modules.initialize() → PTQ calibration → QAT fine-tuning → ONNX export
- Results: SQNR = -4.37 dB, size 11.6→11.7 MB, latency drop 177ms→23ms (CPU demo)

#### 11. Figure improvements
- Thêm figure training curves (đã có PNG từ metrics folder)
- Thêm thống kê cross-validation per-fold

#### 12. Câu văn viết lại cho chỉn chu
- Related work: một số chỗ dùng "Auscultation of pulmonary sounds..." — nghe không tự nhiên
- Section labels nhất quán
- Abstract: viết lại để phản ánh đúng contribution (3-class, KD, QAT)

---

## Thứ Tự Thực Hiện (Execution Plan)

```
Phase 1 — Sửa nghiêm trọng (1-2 ngày)
├── Sửa Abstract, Introduction bullet, Conclusion: 3-class
├── Sửa LaTeX label cross-references
├── Sửa Data Availability
└── Điền Table 7 từ metrics JSON

Phase 2 — Cập nhật methodology (2-3 ngày)
├── Rewrite Sec 3.1 Dataset: clarify 3-class mapping + combined dataset
├── Rewrite Sec 3.3.b Layer 4: add KD details + QAT section
├── Update Table 5: remove ShuffleNet, correct params
└── Update Table 8: add QAT results row, note benchmark pending

Phase 3 — Polish (1 ngày)
├── Sửa câu văn
├── Thêm figures từ output
└── Kiểm tra references .bib
```

---

## Summary Kết Quả Thực Tế (Để Điền Vào Paper)

### Distillation (3-class, 3-fold CV, Subject-Independent):

| Fold | Accuracy | F1 Macro | F1 Weighted |
|------|----------|----------|-------------|
| 0 | **96.56%** | 94.57% | 96.46% |
| 1 | 94.25% | 86.32% | 94.32% |
| 2 | 94.02% | 89.09% | 93.94% |
| **Mean** | **94.94%** | **90.00%** | **94.91%** |

### QAT Results (MobileNetV2, 3-class, 198 samples):

| Metric | FP32 | Fake-INT8 (QAT) | Delta |
|--------|------|-----------------|-------|
| Accuracy | 97.98% | **97.47%** | -0.51% |
| F1 Macro | 97.97% | 97.47% | -0.50% |
| Avg Latency | 177.9ms | 30.8ms (CPU) | **-82.7%** |
| Model Size | 11.63 MB | 11.70 MB | ~same |
| SQNR | — | -4.37 dB | — |

> [!NOTE]
> Latency benchmark 177ms vs 30ms là trên **CPU test** (không phải FPGA). Cần benchmark riêng trên Ultra96-V2 DPU.

---

## Các File Cần Chỉnh Sửa

### [MODIFY] Paper Sections
- [main.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/main.tex) — Abstract, keywords
- [01_introduction.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/sections/01_introduction.tex) — Đổi 5-class → 3-class contributions
- [03_methodology.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/sections/03_methodology.tex) — Thêm KD details, QAT section, fix BPF inconsistency
- [04_experiments.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/sections/04_experiments.tex) — Điền kết quả thực
- [05_conclusion.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/sections/05_conclusion.tex) — Sửa 5-class → 3-class

### [MODIFY] Tables
- [table_4.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/tables/table_4.tex) — Sửa label (class distribution)
- [table_5.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/tables/table_5.tex) — Framework config, thêm segment duration
- [table_6.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/tables/table_6.tex) — Loại bỏ ShuffleNetV1, sửa params
- [table_7.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/tables/table_7.tex) — **Điền kết quả thực**
- [table_9.tex](../Paper/Parallel_Computing_on_FPGA___IEEE_Journal_of_Selected_Topics_in_Signal_Processing/tables/table_9.tex) — Kiểm tra trạng thái

### [NEW] (Có thể thêm)
- `tables/table_qat.tex` — Bảng kết quả QAT FP32 vs INT8
