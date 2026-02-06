# 🩺 Respiratory Sound Analysis: Cascaded Framework (Ultra96-V2)

## 📌 Tổng quan dự án (Project Overview)

Dự án này thực hiện lại nghiên cứu từ bài báo: **"Cascaded Framework with Hardware Acceleration for Respiratory Sound Analysis on Heterogeneous FPGA"**. 

Hệ thống hướng tới việc chẩn đoán các bệnh lý hô hấp (**Healthy, Pneumonia, URTI, Bronchiectasis, và COPD**) với:
- 🎯 **Độ chính xác mục tiêu:** 98.81%
- ⚡ **Tiết kiệm năng lượng:** 52.5% so với phương pháp CPU-GPU truyền thống

---

## 🏗️ Kiến trúc hệ thống (4-Layer Cascaded Architecture)

Hệ thống được thiết kế theo cấu trúc phân tầng để tối ưu hiệu suất:

| Layer | Mô tả | Kỹ thuật |
|-------|-------|----------|
| **Layer 1** | Sàng lọc toàn cục | Metadata bệnh nhân + Năng lượng tổng quát |
| **Layer 2** | Phát hiện biến đổi nhanh (transient) | ZCR + Phân phối biên độ (tiếng rale nổ - crackles) |
| **Layer 3** | Sàng lọc phổ âm thanh | Random Forest trên vector đặc trưng MFCC |
| **Layer 4** | Phân tích chuyên sâu | Deep Learning (CNN) + Wavelet Spectrogram |

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO INPUT (4kHz)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Global Screening (Metadata + Energy)                  │
│  ├── Patient metadata analysis                                  │
│  └── Global energy features                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │ Confident?        │
                    │ (τ₁ threshold)    │
                    └─────────┬─────────┘
                        Yes ↙     ↘ No
                           ↓       ↓
                    [EXIT]    ┌────▼────────────────────────────────┐
                              │  LAYER 2: Transient Detection       │
                              │  ├── Zero Crossing Rate (ZCR)       │
                              │  └── Amplitude Distribution         │
                              └─────────────────────────────────────┘
                                            │
                                  ┌─────────┴─────────┐
                                  │ Confident?        │
                                  │ (τ₂ threshold)    │
                                  └─────────┬─────────┘
                                      Yes ↙     ↘ No
                                         ↓       ↓
                                  [EXIT]    ┌────▼────────────────────┐
                                            │  LAYER 3: RF Ensemble   │
                                            │  ├── MFCC 39-dim        │
                                            │  └── Septuple Forest    │
                                            └─────────────────────────┘
                                                        │
                                              ┌─────────┴─────────┐
                                              │ λ ≥ 4 votes?      │
                                              │ (τ₃ threshold)    │
                                              └─────────┬─────────┘
                                                  Yes ↙     ↘ No
                                                     ↓       ↓
                                              [EXIT]    ┌────▼─────────────────┐
                                                        │  LAYER 4: CNN        │
                                                        │  ├── Wavelet Transform│
                                                        │  └── MobileNetV2     │
                                                        └──────────────────────┘
                                                                    │
                                                                    ▼
                                                              [FINAL OUTPUT]
```

---

## 🛠️ Trạng thái triển khai (Current Implementation Status)

### ✅ Phase 1: Tiền xử lý (`SignalPrep.cpp`)

| Bước | Mô tả | Trạng thái |
|------|-------|------------|
| **Resampling** | Đưa toàn bộ tín hiệu về 4kHz | ✅ Done |
| **Band-pass Filter** | Lọc dải thông 50Hz - 2500Hz để loại bỏ nhiễu | ✅ Done |
| **Segmentation** | Cắt tín hiệu theo chu kỳ hô hấp (ICBHI 2017) | ✅ Done |
| **Normalization** | Chuẩn hóa biên độ về dải [-1, 1] | ✅ Done |

### ✅ Phase 2: Trích xuất đặc trưng (`FeatureExtraction.cpp`)

Đã thực hiện trích xuất bộ đặc trưng hỗn hợp (**Hybrid feature set**):

**Đặc trưng miền thời gian:**
- `EED` - Extreme Energy Difference
- `ZCR` - Zero Crossing Rate  
- `RMSE` - Root Mean Square Energy

**Đặc trưng miền tần số:**
- `MFCC 39 chiều` = 13 static + 13 Δ + 13 ΔΔ

### ✅ Phase 3: Cascaded Logic Layer 1-3 (`CascadedLogic.cpp`)

**Hiện trạng:**
- ✅ Đã code mô phỏng bộ **Septuple Forest** (7 cụm rừng song song)
- ✅ Đã implement cơ chế **Majority Voting** với ngưỡng λ ≥ 4
- ✅ **ĐÃ SỬA LỖI OVER-EXIT**: Thắt chặt ngưỡng tin cậy

**✨ Ngưỡng tin cậy mới (giảm over-exit):**

| Layer | Ngưỡng cũ | Ngưỡng mới | Ghi chú |
|-------|----------|----------|--------|
| Layer 1 | 0.75 | **0.90** | Rất cao để tránh over-exit |
| Layer 2 | 0.70 | **0.88** | Cao |
| Layer 3 | 0.65 | **0.85** | Trung bình-cao |

**Logic early-exit mới:**
- Phải có consensus (≥4/7 clusters đồng ý)
- VÀ aggregated confidence vượt ngưỡng
- VÀ ít nhất 1 cluster có confidence > 0.85

---

### ✅ Phase 4: Layer 4 - CNN Integration (MỚI TRIỂN KHAI)

**Trạng thái:** ✅ ĐÃ HOÀN THÀNH

#### 1️⃣ WaveletTransform (`WaveletTransform.hpp`, `WaveletTransform.cpp`)

| Thông số | Giá trị |
|----------|---------|
| Phương pháp | Continuous Wavelet Transform (CWT) |
| Wavelet | Morlet (Gabor) |
| Output | 224x224 Spectrogram (normalized) |
| Normalization | Log-scale, Power-to-dB, Z-score |

**Features:**
- Multi-resolution spectrogram generation
- Bilinear interpolation resize
- NCHW/NHWC format conversion
- OpenMP parallel processing support

#### 2️⃣ CnnInference (`CnnInference.hpp`, `CnnInference.cpp`)

| Thông số | Giá trị |
|----------|---------|
| Model | MobileNetV2 (4.4M params) |
| Framework | ONNX Runtime C++ API |
| Precision | FP32 (simulation) / INT8 (FPGA) |
| Input | 224x224x1 Spectrogram |
| Output | 4-class probabilities |

**Features:**
- PIMPL pattern (hide ONNX Runtime details)
- Simulation mode khi chưa có model
- Custom callback interface cho Vitis-AI DPU
- Batch inference support
- Softmax postprocessing

#### 3️⃣ CascadedController Integration

- `processLayer4()` - Xử lý mẫu ambiguous với CNN
- Tự động tạo spectrogram từ raw signal hoặc features
- Fallback to simulation khi chưa có trained model

---

## 🚀 Nhiệm vụ kế tiếp (Phase 4: Layer 4 Integration)

**Mục tiêu:** Xử lý 20-30% mẫu dữ liệu "khó" (ambiguous samples) bằng Deep Learning để nâng độ chính xác lên **state-of-the-art**.

### 1️⃣ Tinh chỉnh Early-Exit Thresholds

```cpp
// Cần thắt chặt các ngưỡng để giảm tỉ lệ thoát sớm
float tau_1 = 0.95;  // Layer 1 confidence threshold (hiện quá thấp)
float tau_2 = 0.90;  // Layer 2 confidence threshold
float tau_3 = 0.85;  // Layer 3 confidence threshold
int lambda = 4;      // Minimum votes from 7 RF clusters
```

**Yêu cầu:**
- Thắt chặt các ngưỡng τ₁, τ₂, τ₃ để giảm tỉ lệ thoát sớm tại Layer 1
- Chỉ cho phép thoát sớm khi có sự **đồng thuận cao** từ 7 cụm RF

### 2️⃣ Wavelet Transform (`WaveletTransform.cpp`)

| Thông số | Giá trị |
|----------|---------|
| Phương pháp | Discrete Wavelet Transform (DWT) |
| Wavelet | Morlet |
| Output | Spectrogram image |

**Ưu điểm so với STFT:**
- Độ phân giải thời gian-tần số tối ưu hơn
- Phù hợp với tín hiệu non-stationary như âm thanh hô hấp

### 3️⃣ CNN Inference (`CnnInference.cpp`)

| Thông số | Giá trị | Ghi chú |
|----------|---------|---------|
| **Model** | MobileNetV2 (4.4M params) | Hoặc ShuffleNetV1 (3.5M params) |
| **Framework** | ONNX Runtime C++ API | Chạy inference trên CPU/GPU |
| **Precision** | INT8 | Giả lập để tương thích DPU FPGA |

---

## 📊 Thông số kỹ thuật cần tuân thủ (Technical Specs)

| Thông số | Giá trị | Ghi chú |
|----------|---------|---------|
| **Sample Rate** | 4000 Hz | Downsampled from original |
| **Dải thông** | 50Hz - 2500Hz | Band-pass filter |
| **Đặc trưng RF** | 39-dim MFCC + Time features | Hybrid feature set |
| **Ngưỡng đồng thuận (λ)** | ≥ 4 trên 7 cụm | Majority voting |
| **Kiến trúc CNN** | MobileNetV2 | Depthwise Separable Conv |
| **Precision** | INT8 | Cho Layer 4 |

---

## 📝 Yêu cầu cho Model mới

### Nguyên tắc phát triển:

1. **C++17 Standard**
   - Tiếp tục phát triển trên nền tảng C++17
   - Tận dụng các tính năng như structured bindings, `std::optional`, `std::filesystem`

2. **Modular Design**
   ```
   src/
   ├── preprocessing/
   │   └── SignalPrep.cpp
   ├── features/
   │   └── FeatureExtraction.cpp
   ├── layers/
   │   ├── Layer1_GlobalScreen.cpp
   │   ├── Layer2_TransientDetect.cpp
   │   ├── Layer3_RFEnsemble.cpp
   │   └── Layer4_CNN.cpp
   ├── transforms/
   │   └── WaveletTransform.cpp
   └── inference/
       └── CnnInference.cpp
   ```

3. **FPGA Compatibility**
   - Đảm bảo code module hóa để dễ dàng thay thế bằng:
     - **Vitis-AI API** cho DPU inference
     - **HLS IP Cores** cho các bộ lọc và transforms
   - Target platform: **Ultra96-V2**

4. **Layer Routing Logic**
   ```cpp
   // Pseudo-code cho flow điều hướng mẫu
   Result classify(AudioSegment& sample) {
       // Layer 1: Global screening
       auto [pred1, conf1] = layer1_screen(sample);
       if (conf1 >= tau_1) return pred1;
       
       // Layer 2: Transient detection
       auto [pred2, conf2] = layer2_transient(sample);
       if (conf2 >= tau_2) return pred2;
       
       // Layer 3: RF ensemble
       auto [pred3, votes] = layer3_rf_ensemble(sample);
       if (votes >= lambda) return pred3;
       
       // Layer 4: CNN fallback (ambiguous samples)
       auto spectrogram = wavelet_transform(sample);
       return layer4_cnn(spectrogram);
   }
   ```

---

## 📚 References

1. ICBHI 2017 Respiratory Sound Database
2. "Cascaded Framework with Hardware Acceleration for Respiratory Sound Analysis on Heterogeneous FPGA"
3. MobileNetV2: Inverted Residuals and Linear Bottlenecks
4. Vitis-AI User Guide for Ultra96-V2
