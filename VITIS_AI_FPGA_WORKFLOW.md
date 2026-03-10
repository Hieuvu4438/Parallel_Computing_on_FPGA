# Tài liệu Luồng Tích hợp Hệ thống (End-to-End FPGA Deployment Pipeline) Khám phá Bệnh lý Hô hấp COPD theo chuẩn Xilinx Vitis AI 3.X

Tài liệu này là báo cáo chi tiết về quy trình hoàn thiện giải pháp phần mềm, được tùy biến từ PyTorch gốc sang kiến trúc định lượng INT8 để tích hợp nhúng (Deployment) lên máy chủ viễn biên hoặc FPGA (Xilinx board `PYNQ-Ultra96-V2`). 
Giải pháp xử lý và chẩn đoán phân loại tín hiệu âm thanh hô hấp từ tập dữ liệu (ICBHI 2017).

---

## 1. Sơ đồ Cấu trúc File & Thư mục Dự án

```text
/home/iec/Parallel_Computing_on_FPGA/
├── data/
│   ├── samples/ICBHI_final_database/   # Tập âm thanh y tế nguyên bản (File .wav)
│   └── calib_images_02/                # Kho chứa 200 ảnh Spectrogram thu nhỏ cho quá trình Vitis AI Calibration
│
├── python/
│   ├── output_copd_v2/
│   │   ├── best_model_fold_0.pth       # Mô hình Weights Float32 đã đạt Accuracy 98.81%
│   │   └── model_dpu_ready.pt          # Đồ thị Trace rễ trành TorchScript (PyTorch Graph cứng) đã được scan DPU
│   │
│   ├── train_cnn_1class_v2.py          # Script huấn luyện (Training Float) với kiến trúc k-Fold (5 folds)
│   ├── preprocess_audio_dsp.py         # Ứng dụng tiền xử lý tín hiệu DSP (Chuyển WAV thành cấu trúc CWT Matrix)
│   ├── create_calib_dataset.py         # Ứng dụng lọc trích ngẫu nhiên Calibration Sample cho FPGA
│   └── validate_and_export_dpu.py      # Kịch bản Evaluation đa năng, đối chiếu độ chính xác / Export Float Graph
│
├── vitis_ai_flow/
│   ├── quantize.py                     # Kế hoạch điều khiển module "pytorch_nndct" (Truy vết Scale Int8)
│   └── quantize_result/                # Thư mục đích đầu ra sau quá trình VAI Định lượng
│       ├── quant_info.json             # Bảng tra Map Min/Max Tensors & Scales Int8
│       ├── COPDClassifier.py           # Kiến trúc tự động của Quantizer để biên dịch
│       └── COPDClassifier_int.xmodel   # Phiên bản lượng tử hóa chuẩn XIR-Level Graph (Quan trọng nhất)
│
└── Vitis-AI/                           # Vitis AI Docker (Chứa môi trường vitis-ai-pytorch và Công cụ Compiler)
    └── docker_run.sh                   # Script khởi động Container Compiler
```

---

## 2. Phân tích nội dung cốt lõi của các File Code đã xây dựng

### 2.1. `python/preprocess_audio_dsp.py` (Lõi DSP - Xử lý tín hiệu)
- **Mục tiêu**: Chuyển ngữ tín hiệu thô (Âm thanh hô hấp `.wav` nhiễu) sang ảnh học sâu 224x224 RGB.
- **Tính toán kỹ thuật (DSP)**:
  - Do định lý Nyquist, hệ thống ép sử dụng chức năng `scipy.signal.butter` để lọc nhiễu môi trường ở dải `50-2500Hz` một cách chặt chẽ trên mẫu tần số gốc > 44.1kHz TRƯỚC khi `resample` về 4000Hz. Nhờ vậy ngăn chặn hoàn toàn sai lệch aliasing (chồng lấn phổ dải).
  - Khởi tạo Phép biến đổi Wavelet (CWT `Morlet`) giúp nắm bắt đồng thời độ phân giải nhỏ ở tần số cao (Tiếng nổ phổi crackle chớp nhoáng) và độ phân giải hẹp ở dải trầm (Tiếng ran rít wheezing dài).

### 2.2. `python/validate_and_export_dpu.py` (Chuẩn hóa Hardware Vitis)
- **Mục tiêu**: Trích xuất trọng số Float sang Float Graph (.pt Trace Mode) sau khi quét độ tương thích Chip.
- **Tính toán kỹ thuật**:
  - Script tự nhận diện Wrapper (định nghĩa PyTorch `MobileNetV2` bên trong `COPDClassifier`) để mount trọng số (Weight Dict) khớp 100% rễ mà không báo lỗi Missing Key.
  - Tự động chạy Evaluator qua file bệnh nhân (Unseen) trong chế độ Binary (`Non-COPD` vs `COPD`) cho ra Macro F1-Score trên **~69%** đối với mẫu Fold_0 thô, giúp theo dõi chính xác sai số Float.
  - Tự động chạy trình Scan Compatibility: Phát hiện sự có mặt của `nn.ReLU6` (do Xilinx DPU đôi lúc dính lỗi Over-Clipping ReLU6 trên 1 số layers, code tự in ra Warning Log) / Quát bỏ `SiLU/Mish`.
  - Cuối cùng, bọc Model bằng `torch.jit.trace` tạo ra `.pt` tĩnh. (Yêu cầu cứng của pytorch_nndct). 

### 2.3. `vitis_ai_flow/quantize.py` (Biên dịch Cấu trúc Int8 Neural Network)
- **Mục tiêu**: Biến thiết kế Float32 thành INT8 (Phù hợp tài nguyên bộ nhớ BRAM/DSP của FPGA Ultra96).
- **Tính toán kỹ thuật**:
  - Giao diện thư viện `pytorch_nndct`.
  - Giảm `batch_size=1` cho Dataloader ở chế độ Test để tương thích hoàn toàn Compiler của DPUCZDX8G.
  - Mode **calib**: Thao tác Forward Pass không Gradient 200 ảnh sinh ra bảng tra (Lookup tables) Min/Max (Fake Quantize).
  - Mode **test**: Test quá trình Forward 1 lần nữa ở Dataloader có `batch_size=1` và tự động phân rã module thành file cấu trúc `.xmodel` nhị phân ra Output.

---

## 3. Hướng dẫn Tích hợp & Chạy DPU (Cách sử dụng từ A - Z)

Để làm lại từ đầu hoặc deploy mô hình mới, vui lòng bật Terminal Server Ubuntu và thực hiện đúng trình tự các bước:

### Bước 1: Trích xuất Bộ Test Hiệu chuẩn (Calibration Dataset)
Môi trường bắt buộc: `Conda` của Host OS hoặc Python Native.
Bạn cần chạy file DSP để quét 200 tấm ảnh `spectrogram_preview` (Lấy từ Dataset ICBHI) từ các file wav nguyên thủy:
```bash
# Ở thư mục gốc: /home/iec/Parallel_Computing_on_FPGA
conda activate fpga # (nếu bạn sử dụng env conda)
python python/preprocess_audio_dsp.py
```

### Bước 2: Khai thác Trọng số `.pth`, Test tương thích và Xuất đồ thị TorchScript `.pt`
Thao tác này kiểm chứng Model của bạn chạy chuẩn trên Test Data và không chứa Op code lạ với DPU:
```bash
python python/validate_and_export_dpu.py
```
*Kết quả xuất ra cảnh báo cấu trúc và tạo file `/home/iec/Parallel_Computing_on_FPGA/python/output_copd_v2/model_dpu_ready.pt`.*

### Bước 3: Đăng nhập vào Hạ tầng Ảo hóa Xilinx (Docker Compilation)
Sử dụng môi trường ảo hóa Vitis Container của Xilinx để đảm bảo thư viện NNDCT được liên kết chuẩn. **BẮT BUỘC CHẠY TỪ THƯ MỤC GỐC của dự án**.
```bash
cd /home/iec/Parallel_Computing_on_FPGA
./Vitis-AI/docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
```
*Sau khi tải môi trường, bạn sẽ thấy dấu nhắc lệnh Bash Shell đổi thành: `(vitis-ai-pytorch) vitis-ai-user@iec:/workspace$`*

### Bước 4: Chạy Quantization Mode: CALIB (Bên trong Docker)
Lệnh phân tách Scale Int8 (Tương tự đào tạo ngắn hạn lấy ngưỡng Min/Max từ Data Hiệu chuẩn):
```bash
python /workspace/vitis_ai_flow/quantize.py \
    --quant_mode calib \
    --model_dir /workspace/python/output_copd_v2 \
    --calib_dir /workspace/data/calib_images_02 \
    --output_dir /workspace/vitis_ai_flow/quantize_result
```

### Bước 5: Chạy Quantization Mode: TEST / EXPORT (Bên trong Docker)
Lệnh chốt Tensors Int8, Fake-Quantize Evaluation và xả Code đồ thị ra file XMODEL (Intermediate):
```bash
python /workspace/vitis_ai_flow/quantize.py \
    --quant_mode test \
    --model_dir /workspace/python/output_copd_v2 \
    --calib_dir /workspace/data/calib_images_02 \
    --output_dir /workspace/vitis_ai_flow/quantize_result
```

### Bước 6: Biên dịch phần cứng ra mã máy DPU (Công cụ VAI_C_XIR)
Đây là bước quyết định để biến Instruction Set ảo sang mã phần cứng thực thi bởi vi mạch FPGA PYNQ-Ultra96-v2 (Board sử dụng nhân DPUCZDX8G). 

*Thực hiện hoàn toàn bên trong Terminal của **Docker** (Trực tiếp gõ sau khi Bước 5 hoàn thành, không cần gọi python):*
Để Ultra96-V2 tương thích, ta cần truyền cấu hình ISA và Kích thước B2304 thông qua cờ `-a`:
```bash
vai_c_xir \
    -x /workspace/quantize_result/COPDClassifier_int.xmodel \
    -a /workspace/vitis_ai_flow/arch.json \
    -o /workspace/compiled_model \
    -n copd_classifier_u96
```

> **🎉 ĐÍCH ĐẾN CUỐI CÙNG:** 
Sau quá trình biên dịch (khoảng 2 phút), thư mục `/workspace/vitis_ai_flow/compiled_model/` (hay ở Host Desktop là phần folder con của Project) sẽ sinh ra tệp mã máy `copd_classifier_u96.xmodel` và `meta.json`. Lúc này, bạn chỉ việc ghim file `.xmodel` này vô thẻ nhớ, đút vào **PYNQ-Ultra96-V2** và lập trình giao diện nhúng là toàn bộ quá trình xử lý AI trên FPGA đã đại thành công!
