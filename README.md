# Respiratory Sound Analysis – Cascaded Framework on FPGA

Triển khai lại nghiên cứu từ bài báo:  
**"Cascaded Framework with Hardware Acceleration for Respiratory Sound Analysis on Heterogeneous FPGA"**

**Target platform:** Ultra96-V2 (Xilinx Zynq UltraScale+ MPSoC)  
**Target accuracy:** 98.81% | **Energy saving:** 52.5% vs CPU-GPU

---

## Cấu trúc thư mục

```
Parallel_Computing_on_FPGA/
├── Makefile                # [MỚI] Bộ điều phối trung tâm cực kỳ tiện lợi
├── CMakeLists.txt          # C++ build system
├── AGENTS.md               # Project rules & architecture
│
├── include/                # C++ public headers
├── src/                    # C++ implementation
│   ├── main.cpp
│   ├── preprocessing/      # SignalPrep
│   ├── features/           # FeatureExtraction, WaveletTransform
│   ├── classifiers/        # CascadedLogic, CnnInference, RandomForestCPU
│   └── utils/              # Logger
├── tests/                  # [TÁI CẤU TRÚC] Thư mục kiểm thử hợp nhất
│   ├── cpp/                # Unit tests cho C++ (test_signal_prep.cpp)
│   ├── python/             # Unit tests cho Python
│   └── cross_validation/   # Kiểm thử đối chuẩn C++ vs Python
│
├── python/                 # Python ML pipeline & Analytics
│   ├── preprocessing/      # Audio preprocessing scripts
│   ├── training/           # Model training (train_*.py, distillation_02.py)
│   │   └── experiments/    # [MỚI] Thực nghiệm cũ & Legacy
│   ├── quantization/       # Quantization & export scripts
│   ├── Inspector/          # Vitis-AI inspector tools
│   ├── layer1_3_experiments/ # Layer 1-3 experiments
│   └── visualizations/     # Trực quan hóa & Vẽ biểu đồ (plot_*.py)
│
├── models/                 # Model artifacts
│   ├── training_outputs/   # Checkpoints từ training
│   ├── quantized/          # INT8 quantized models (Vitis PTQ/QAT, NVIDIA QAT)
│   └── compiled/           # Compiled DPU xmodel
│
├── fpga/                   # FPGA deployment
│   ├── vitis_ai_flow/      # Quantization workflow (arch.json, quantize.py)
│   └── deploy/             # Deployment scripts
│
├── data/                   # Dataset (ICBHI 2017, KAUH)
│
├── docs/                   # Documentation
│   ├── VITIS_AI_FPGA_WORKFLOW.md
│   ├── phase_notes/        # Development notes
│   └── notebook/           # Jupyter notebooks
│
└── artifacts/              # Benchmark outputs, logs, CSV data
```

---

## 🚀 Hướng dẫn Sử dụng Nhanh (Makefile Unified Central Runner)

Dự án cung cấp bộ công cụ điều phối trung tâm `Makefile` giúp bạn chạy tất cả các tác vụ một cách dễ dàng nhất:

### 1. Tác vụ C++ Engine (Build, Run & Test)
```bash
# Build mã nguồn C++ Core (Debug mode + Unit tests)
make build

# Build tối ưu hiệu năng (Release mode)
make build-release

# Chạy toàn bộ 18 Unit Tests để xác minh độ chính xác thuật toán
make test

# Chạy chương trình demo phân tích chính
make run
```

### 2. Tác vụ Python ML Pipeline
```bash
# Thiết lập môi trường ảo venv và cài đặt thư viện tự động
make env

# Huấn luyện mô hình MobileNetV2 (Layer 4 CNN)
make train

# Chạy quy trình Knowledge Distillation
make distill

# Lượng tử hóa mô hình sang INT8 cho FPGA DPU
make quantize
```

### 3. Vẽ đồ thị trực quan hóa cho bài báo (Paper Figures)
```bash
# Vẽ toàn bộ hình ảnh, biểu đồ kiến trúc và đặc trưng
make plot-all

# Vẽ riêng lẻ:
make plot-signal        # Đồ thị tín hiệu & phân chu kỳ audio
make plot-features      # Phân bố ZCR, RMS, MFCC
make plot-spectrogram   # Morlet Wavelet Spectrogram
make plot-kd            # Đồ thị so sánh Teacher vs Student
```

---

## Kiến trúc hệ thống (4-Layer Cascaded)

| Layer | Phương pháp | Mục đích |
|-------|-------------|----------|
| 1 | Metadata + Global Energy | Sàng lọc nhanh |
| 2 | ZCR + Amplitude Distribution | Phát hiện Crackle |
| 3 | Septuple Random Forest (MFCC 39-dim) | Spectral screening |
| 4 | MobileNetV2 + Wavelet Spectrogram | Ambiguous samples |

Early-exit thresholds: τ₁=0.90, τ₂=0.88, τ₃=0.85, λ≥4/7 votes

---

## References

- ICBHI 2017 Respiratory Sound Database
- MobileNetV2: Inverted Residuals and Linear Bottlenecks
- Vitis-AI User Guide for Ultra96-V2
