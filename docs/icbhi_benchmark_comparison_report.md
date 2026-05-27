# Báo cáo Phân tích & So sánh Benchmark: KD Pipeline vs. ADD-RSC

Tài liệu này phân tích chi tiết sự khác biệt về **phương pháp chia tập dữ liệu (dataset split)**, **quy trình đánh giá (evaluation benchmark)**, **tiền xử lý dữ liệu (preprocessing)** và **định nghĩa metrics** giữa hai phương pháp:
1. **KD Pipeline (Mã nguồn của bạn)**: Đại diện bởi các file `icbhi_kd_pipeline_*.py`.
2. **ADD-RSC (Mã nguồn từ paper đối thủ)**: Đại diện bởi thư mục `ADD-RSC/` (`main.py` và `util/`).

---

## 1. Bảng So sánh Tổng quan (Executive Summary)

| Tiêu chí | KD Pipeline (Phương pháp của bạn) | ADD-RSC (Paper đối thủ) | Đánh giá & Ảnh hưởng |
| :--- | :--- | :--- | :--- |
| **Đơn vị chia split** | **Bệnh nhân (Patient-wise)** | **File âm thanh (File-level)** | **Rất khác nhau**. ADD-RSC bị rò rỉ dữ liệu bệnh nhân (Data Leakage) nghiêm trọng. |
| **Tỷ lệ chia Train/Test** | 60 bệnh nhân train/val (101-160), 66 bệnh nhân test (161-227) | 60% số file làm Train, 40% số file làm Test (ngẫu nhiên) | Chia của KD Pipeline tuân thủ đúng chuẩn của hội thi ICBHI 2017. Chia của ADD-RSC làm bài toán dễ đi rất nhiều. |
| **Quy trình Validation** | Tách 20% bệnh nhân từ tập Train để làm Validation. Tập Test chỉ dùng cuối cùng. | Dùng tập Test làm tập Validation trực tiếp để lưu checkpoint tốt nhất (`best.pth`). | ADD-RSC bị Overfitting tập Test (Test-set overfitting) trong quá trình huấn luyện. |
| **Bộ lọc dải thông** | Áp dụng Butterworth Band-pass (50Hz - 2500Hz/4000Hz) | Định nghĩa argument nhưng **không thực sự áp dụng** trong code xử lý. | KD Pipeline lọc nhiễu tốt hơn trước khi tạo phổ. |
| **Tạo phổ Mel (Fbank)** | Sử dụng STFT của `scipy` và mel filterbank tự dựng. | Sử dụng `torchaudio.compliance.kaldi.fbank` (chuẩn Kaldi). | Khác biệt nhỏ về đặc trưng tần số đầu vào. |
| **Chuẩn hóa phổ** | Tính toán `mean`/`std` động từ tập Train. | Sử dụng hằng số tĩnh (static): `mean = -4.268`, `std = 4.569`. | Khác biệt về phân phối giá trị đầu vào model. |
| **Công thức tính Metrics** | Giống nhau (Sp, Se, Score). Hỗ trợ thêm dò ngưỡng tối ưu (`sweep_threshold`). | Giống nhau (Sp, Se, Score). Chỉ dùng Argmax mặc định. | Công thức giống nhau, nhưng cách lấy dự đoán khác nhau. |

---

## 2. Chi tiết về Cách chia Dataset (Dataset Splitting)

### 2.1 KD Pipeline (Patient-wise Split - Chuẩn Official)
* **Nguyên tắc**: Tập dữ liệu ICBHI 2017 gồm các bệnh nhân từ ID `101` đến `226`. KD Pipeline chia cứng:
  * **Train/Val Patients**: `101` đến `160` (60 bệnh nhân).
  * **Test Patients**: `161` đến `227` (66 bệnh nhân).
* **Validation**: Từ 60 bệnh nhân Train, chia tiếp (ví dụ 20%) làm Validation. 
* **Data Leakage**: **Không có**. Một bệnh nhân chỉ xuất hiện ở duy nhất một tập (Train, Val hoặc Test). Điều này đảm bảo mô hình đánh giá khả năng tổng quát hóa trên bệnh nhân hoàn toàn mới (chưa từng gặp lúc train).

### 2.2 ADD-RSC (Shuffled File-level Split - Không chuẩn)
* **Nguyên tắc**:
  * Thu thập toàn bộ danh sách các file `.wav` và `.txt` trong thư mục dữ liệu.
  * Sắp xếp và xáo trộn ngẫu nhiên danh sách file bằng `random.Random(1).shuffle(indices)`.
  * Lấy **60% số file** làm tập Train, **40% số file** làm tập Test.
* **Data Leakage**: **Nghiêm trọng (Patient-identity Leakage)**.
  * Vì một bệnh nhân thường có nhiều file ghi âm ở các vị trí phổi khác nhau (ví dụ: bệnh nhân `101` có các file `101_1b1_Al_sc_Medusa.wav`, `101_1b1_Pr_sc_Medusa.wav`,...).
  * Việc xáo trộn ngẫu nhiên ở cấp độ file dẫn đến tình trạng: một số file của bệnh nhân `101` nằm ở tập Train, trong khi các file khác của chính bệnh nhân `101` nằm ở tập Test.
  * Khi đó, mô hình sẽ học thuộc lòng các đặc trưng sinh học riêng biệt của bệnh nhân đó, âm học của phòng khám, hoặc thiết bị ghi âm (stethoscope), dẫn đến kết quả Test cao một cách ảo tưởng (thường tăng từ 5% - 15% Score).
* **Lỗi Code (Bug) trong ADD-RSC**:
  Trong file `ADD-RSC/util/icbhi_dataset.py`, nếu người dùng cấu hình `--test_fold` khác với `'official'` (ví dụ: `'0'`, `'1'`,...):
  ```python
  idx = f.split('_')[0] if test_fold in ['0', '1', '2', '3', '4'] else f
  if idx in patient_dict:
      self.filenames.append(f)
  ```
  Biến `idx` lúc này là Patient ID (như `'101'`), nhưng các key của `patient_dict` luôn là tên file đầy đủ (như `'101_1b1_Al_sc_Medusa'`). Do đó điều kiện `idx in patient_dict` luôn trả về `False`, khiến dataset trống rỗng và code bị crash.

---

## 3. Quy trình Đánh giá & Chọn checkpoint (Validation Protocol)

### 3.1 KD Pipeline
* Quá trình train đánh giá trên tập **Validation** độc lập sau mỗi epoch.
* Dùng tập Validation để thực hiện **Sweep Threshold** tìm ngưỡng phân biệt Normal/Abnormal tối ưu nhất nhằm tăng tối đa ICBHI Score.
* Checkpoint tốt nhất được lưu dựa trên hiệu năng của tập Validation này. Tập Test hoàn toàn ẩn và chỉ được đánh giá một lần duy nhất ở bước cuối cùng (`evaluate_final`).

### 3.2 ADD-RSC
* Trong file `main.py`, biến `val_loader` thực chất được load từ `val_dataset = ICBHIDataset(train_flag=False, ...)` (chính là tập **Test** 40%).
* Sau mỗi epoch huấn luyện, mô hình kiểm tra hiệu năng trực tiếp trên tập Test này và cập nhật checkpoint tốt nhất (`best.pth`) nếu Score tăng:
  ```python
  if sc > best_acc[-1] and se > 5:
      save_bool = True
      best_acc = [sp, se, sc]
  ```
* Điều này vi phạm nguyên tắc khoa học máy tính nghiêm trọng (**Test-set Overfitting**), vì tập Test được dùng trực tiếp để chọn mô hình tối ưu.

---

## 4. Tiền xử lý Dữ liệu & Đặc trưng (Preprocessing & Features)

| Thông số tiền xử lý | KD Pipeline | ADD-RSC |
| :--- | :--- | :--- |
| **Sample Rate** | 16,000 Hz | 16,000 Hz |
| **Độ dài tín hiệu (Duration)** | 8.0 giây (lặp lại nếu ngắn hơn) | 8.0 giây (lặp lại nếu ngắn hơn) |
| **Bộ lọc Band-pass** | **Có áp dụng** (mặc định 50Hz - 2500Hz) | **Không áp dụng** (bị bỏ qua trong code thực tế) |
| **Cách tính Mel Spectrogram** | Sử dụng `scipy.signal.stft` thủ công | Sử dụng `torchaudio.compliance.kaldi.fbank` |
| **Normalization** | Chuẩn hóa động: `(x - mean_train) / std_train` | Chuẩn hóa tĩnh: `(x - (-4.267)) / (4.568 * 2)` |
| **Đặc trưng đầu vào** | 1-channel Log-Mel hoặc 3-channel (Mel + Delta + Delta-Delta) | 1-channel Log-Mel |

---

## 5. Metrics & Công thức tính toán

Cả hai bên đều sử dụng định nghĩa của hội thi **ICBHI 2017** cho phân loại 4 lớp (0: Normal, 1: Crackle, 2: Wheeze, 3: Both):

* **Specificity ($S_p$)**: Tỷ lệ dự đoán đúng lớp Normal (lớp 0).
  $$S_p = \frac{\text{Dự đoán đúng Normal}}{\text{Tổng số mẫu thực tế là Normal}}$$
* **Sensitivity ($S_e$)**: Tỷ lệ phát hiện bất thường (lớp 1, 2, 3).
  $$S_e = \frac{\text{Dự đoán Abnormal (1, 2 hoặc 3) trên các mẫu thực tế Abnormal}}{\text{Tổng số mẫu thực tế là Abnormal (1 + 2 + 3)}}$$
* **ICBHI Score**: Trung bình cộng của hai chỉ số trên.
  $$\text{Score} = \frac{S_p + S_e}{2}$$

**Điểm khác biệt duy nhất ở đầu ra**:
* KD Pipeline hỗ trợ điều chỉnh ngưỡng quyết định nhị phân thông qua dò tìm ngưỡng (`sweep_threshold`) giúp cân bằng chủ động giữa $S_p$ và $S_e$.
* ADD-RSC sử dụng hàm `argmax` đơn thuần trên đầu ra mô hình để quyết định nhãn lớp, tương đương với việc cố định ngưỡng phân tách Normal/Abnormal ở mức mặc định (0.5).

---

## 6. Kết luận & Khuyến nghị benchmark song song

Để so sánh **công bằng và chính xác** hiệu năng thuật toán của mô hình của bạn (ví dụ: cấu trúc mạng KD, các loss chức năng) với ADD-RSC:
1. **Không thể so sánh trực tiếp điểm số đã công bố**: Điểm số $65.53\%$ của ADD-RSC chạy trên tập chia ngẫu nhiên 60/40 cấp độ file chứa lỗi rò rỉ dữ liệu nghiêm trọng, trong khi điểm số của KD Pipeline chạy trên tập chia bệnh nhân chính quy (khó hơn rất nhiều).
2. **Cần cấu hình benchmark chung**:
   * **Cách 1 (Khuyên dùng)**: Chạy lại ADD-RSC trên tập chia Patient-wise chính quy của KD Pipeline.
   * **Cách 2**: Chạy KD Pipeline ở chế độ chia ngẫu nhiên cấp độ file 60/40 tương thích với ADD-RSC (sử dụng cấu hình chia ngẫu nhiên file có seed = 1) để xem mô hình KD cải thiện như thế nào trên cùng một "sân chơi dễ".
