# Python ML Pipeline

## Cấu trúc

```
python/
├── preprocessing/          # Tiền xử lý audio
│   ├── preprocessing.py            - Pipeline chính (resampling, bandpass, segmentation)
│   ├── preprocess_audio_dsp.py     - DSP-optimized version
│   ├── combine_dataset.py          - Kết hợp ICBHI + KAUH dataset
│   └── wavelet_transform.py        - CWT spectrogram generation
│
├── training/               # Huấn luyện mô hình (Active & Core)
│   ├── train_mobilenetv2.py        - MobileNetV2 (bản huấn luyện chính)
│   ├── train_mobilenetv2_vitis.py  - MobileNetV2 tương thích Vitis-AI
│   ├── train_efficientnet_b0.py    - EfficientNet-B0 làm baseline đối chứng
│   ├── distillation_02.py          - Knowledge Distillation cải tiến (Teacher → Student)
│   └── experiments/                - Thư mục chứa các thực nghiệm cũ và legacy (train_cnn_*.py, v2, v3, v4)
│
├── quantization/           # INT8 Quantization & Export
│   ├── create_calib_dataset.py     - Tạo calibration dataset cho PTQ
│   ├── validate_and_export_dpu.py  - Validate INT8 + export .xmodel
│   └── export_ts.py                - Export TorchScript model
│
├── Inspector/              # Vitis-AI inspector
├── layer1_3_experiments/   # Thực nghiệm Random Forest layers 1-3
└── visualizations/         # Trực quan hóa & Vẽ biểu đồ (plot_*.py, visualize_*.py)
```

## Thứ tự chạy

```bash
# 1. Tiền xử lý
python preprocessing/preprocessing.py

# 2. Training (chọn 1 trong các scripts)
python training/train_mobilenetv2_vitis.py

# 3. Quantization
python quantization/create_calib_dataset.py
python ../fpga/vitis_ai_flow/quantize.py
python quantization/validate_and_export_dpu.py
```
