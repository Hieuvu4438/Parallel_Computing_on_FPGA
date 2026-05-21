# Bảng giải thích chi tiết các chỉ số (Metrics) trong Paper

Tài liệu này giải thích chi tiết ý nghĩa và cách tính toán các chỉ số (metrics) xuất hiện trong các bảng đánh giá hiệu năng phần cứng (Table XII, XIII, XIV, XV) của bài báo.

---

## TABLE XII: END-TO-END UA-HCI LATENCY MEASUREMENT TEMPLATE ON ULTRA96-V2 AT BATCH SIZE 1

Bảng này trình bày độ trễ (latency) đầu cuối khi xử lý một mẫu âm thanh hô hấp qua từng phân luồng (route) trong kiến trúc phân tầng (cascaded framework) trên board Ultra96-V2.

### 1. Ý nghĩa từng metric:
- **Preprocess (ms):** Thời gian tiền xử lý tín hiệu âm thanh thô trên CPU (như lọc nhiễu dải thông, chuẩn hóa, phân mảnh tín hiệu).
- **Feature/RF (ms):** Thời gian trích xuất đặc trưng thống kê (như năng lượng RMS, ZCR, MFCC) và thời gian thực thi phân loại qua mô hình Rừng ngẫu nhiên (Random Forest - RF) chạy trên CPU (Processing System - PS).
- **Spectrogram (ms):** Thời gian để tạo phổ đồ (Wavelet Hybrid Spectrogram) từ tín hiệu âm thanh. Bước này chỉ được kích hoạt nếu mẫu đi đến Layer 4.
- **DPU + transfer (ms):** Thời gian chuyển đổi (transfer) ma trận ảnh spectrogram từ bộ nhớ CPU xuống bộ nhớ FPGA (từ PS sang PL) và thời gian thực thi suy luận của mô hình CNN (MobileNetV2) trên lõi DPU của FPGA.
- **Mean (ms):** Thời gian trễ trung bình từ lúc đưa âm thanh vào đến khi ra kết quả cuối cùng đối với mẫu rẽ nhánh (exit) tại Layer tương ứng.
- **P95 (ms):** Bách phân vị thứ 95 của độ trễ (95th percentile latency). Nghĩa là 95% số lượng mẫu rẽ nhánh tại layer này có thời gian xử lý nhỏ hơn hoặc bằng giá trị này. Đây là chỉ số quan trọng để đánh giá độ trễ tồi tệ nhất (worst-case latency) trong hệ thống thời gian thực.

### 2. Phương pháp đo lường & Cách tính toán:
- **Preprocess, Feature/RF, Spectrogram (ms):** Các thành phần chạy trên CPU được đo đạc bằng các thư viện đo thời gian có độ chính xác cao trong mã nguồn C/C++ (ví dụ: `std::chrono::high_resolution_clock`). Ta ghi nhận timestamp ngay trước và sau khi gọi hàm, sau đó lấy hiệu số để ra thời gian chạy thực tế (tính bằng mili-giây).
- **DPU + transfer (ms):** Được đo thông qua bộ đếm thời gian thực thi do Xilinx Runtime (XRT API) cung cấp hoặc qua Vitis Profiler. Nó bao gồm thời gian copy bộ nhớ (`sync_bo`) giữa PS và PL, cộng với độ trễ khi kích hoạt nhân DPU (`enqueueTask`).
- **Mean (ms):** Trung bình cộng độ trễ của hàng nghìn lần chạy độc lập. Công thức: $Mean = \frac{1}{N}\sum_{i=1}^{N} Latency_i$ với từng phân luồng (route).
- **P95 (ms):** Được tính bằng cách lưu thời gian chạy của toàn bộ $N$ mẫu thử nghiệm vào một mảng, sắp xếp mảng từ nhỏ đến lớn. Giá trị P95 chính là phần tử nằm ở vị trí thứ $\lfloor 0.95 \times N \rfloor$. Việc lấy chỉ số này giúp loại bỏ 5% các nhiễu đo lường ngẫu nhiên bất thường (outliers) hoặc các độ trễ đột biến do hệ điều hành nền (OS context switch) gây ra.

---

## TABLE XIII: FPGA RESOURCE UTILIZATION PROFILE FOR SYNTHESIZED DPU CORE

Bảng này mô tả lượng tài nguyên phần cứng vật lý trên chip FPGA đã bị tiêu tốn (utilization) để tổng hợp ra (synthesize) kiến trúc lõi xử lý học sâu (DPU).

### 1. Ý nghĩa từng metric:
- **LUT (Look-Up Table):** Khối logic cơ bản nhất trên FPGA, dùng để xây dựng các cổng logic và thực thi các phép toán tổ hợp.
- **FF (Flip-Flop):** Khối nhớ trạng thái (register) trên FPGA, dùng để giữ giá trị qua từng chu kỳ xung nhịp (clock cycle) và thực thi logic tuần tự.
- **BRAM (Block RAM):** Các khối nhớ SRAM được tích hợp sẵn trên chip FPGA với băng thông rất lớn. Chúng được dùng để lưu trữ trọng số của mạng neural, biases, và dữ liệu feature map trung gian của các lớp tích chập (convolution) trong quá trình tính toán.
- **DSP (Digital Signal Processor):** Các lõi xử lý tín hiệu số phần cứng được tối ưu cực tốt cho phép tính nhân-cộng (Multiply-Accumulate hay MAC). DPU sử dụng rất nhiều tài nguyên này để thực hiện cực nhanh các phép tích chập của CNN.
- **Util (%) (Utilization):** Mức độ chiếm dụng của lõi DPU đối với loại tài nguyên tương ứng trên chip so với tổng tài nguyên sẵn có.

### 2. Phương pháp đo lường & Cách tính toán:
- **Used (Tài nguyên đã sử dụng):** Giá trị này hoàn toàn **không** được tính toán thủ công, mà được trích xuất tự động từ các tệp báo cáo tổng hợp và triển khai phần cứng (`Synthesis Report` và `Implementation Report` - các file `.rpt`) do phần mềm Xilinx Vivado / Vitis AI sinh ra sau khi quá trình biên dịch thiết kế (bitstream) hoàn tất.
- **Total (Tổng tài nguyên):** Là thông số kỹ thuật vật lý cố định do nhà sản xuất Xilinx công bố cho từng dòng chip cụ thể (Zynq-7020 cho PYNQ-Z2 và ZU3EG cho Ultra96-V2).
- **Util (%) =** $\frac{\text{Used}}{\text{Total}} \times 100\%$
  *(Ví dụ: Từ report của Vivado cho thấy thiết kế DPU tốn 41,200 LUTs. Thông số kỹ thuật chip ZU3EG có tổng cộng 70,560 LUTs. Vậy mức độ chiếm dụng Util = 41,200 / 70,560 = 58.38%).*

---

## TABLE XIV: ROUTE-LEVEL ENERGY MEASUREMENT TEMPLATE FOR UA-HCI ON ULTRA96-V2

Bảng này thống kê sự tiêu thụ năng lượng theo từng kịch bản rẽ nhánh. Điểm đặc biệt của kiến trúc Cascaded là phần lớn mẫu âm thanh bị dừng sớm ở các lớp RF tiêu tốn cực kỳ ít năng lượng.

### 1. Ý nghĩa từng metric:
- **Exit probability:** Xác suất (tỷ lệ phần trăm) lượng dữ liệu đầu vào sẽ tự tin đạt ngưỡng đồng thuận cao để được xuất kết quả sớm tại layer này mà không phải đi sâu xuống các layer sau.
- **Mean latency (ms):** Thời gian xử lý trung bình của nhánh đó (được lấy từ Table XII).
- **Mean power (W):** Công suất tiêu thụ trung bình (cả trên CPU và FPGA tùy thuộc nhánh đó có kích hoạt FPGA hay không) khi thực thi mẫu ở nhánh đó. Thường được đo tại nguồn của board (board power / rail power).
- **Energy/sample (mJ):** Năng lượng tổng cộng bị tiêu thụ để xử lý trọn vẹn một mẫu trên nhánh (route) đó.

### 2. Phương pháp đo lường & Cách tính toán:
- **Exit probability:** Tính bằng công thức $\frac{\text{Số lượng mẫu dừng tại Layer X}}{\text{Tổng số lượng mẫu trong tập test}}$. Giá trị này được rút ra từ log chạy thử nghiệm mô hình thực tế (empirical evaluation).
- **Mean latency (ms):** Được tham chiếu từ cột Mean trong Table XII đối với mỗi route tương ứng.
- **Mean power (W):** Không thể tính toán trên giấy mà phải **đo lường vật lý**. Mức công suất này được đọc từ các cảm biến nguồn tích hợp trên bo mạch (Power Management IC - PMIC) qua giao thức I2C/PMBus (dùng các thư viện như `pynq.pmbus` trên PYNQ), hoặc bằng cách kết nối đồng hồ vạn năng / máy hiện sóng vào đường điện 12V cấp cho toàn bộ board. Thông số này lấy giá trị trung bình trong suốt quá trình suy luận.
- **Energy/sample (mJ) =** `Mean latency (ms) * Mean power (W)`
  *(Giải thích vật lý: Năng lượng (Joules) = Công suất (Watts) $\times$ Thời gian (Seconds). Do thời gian đo bằng mili-giây (ms), nên tích số tự động có đơn vị đo là mili-Joules (mJ)).*
  *Ví dụ: Ở Layer 1, thời gian chạy là 4.5 ms và công suất mạch đo được là 1.82 W $\rightarrow$ Energy = 4.5 $\times$ 1.82 = 8.19 mJ (làm tròn lên 8.2 mJ trong bảng).*
- **Năng lượng trung bình của toàn hệ thống:** Chính là giá trị kỳ vọng toán học của kiến trúc rẽ nhánh, được tính bằng tổng: $E_{sys} = \sum_{i=1}^{4} (\text{Exit probability}_i \times \text{Energy}_i)$.

---

## TABLE XV: CROSS-PLATFORM HARDWARE PERFORMANCE COMPARISON (EDGE FPGA VS. EDGE GPU)

Bảng này so sánh toàn diện hiệu năng và mức độ tiết kiệm điện năng của kiến trúc sử dụng thiết bị FPGA Edge (PYNQ-Z2, Ultra96-V2) so với đối thủ tiêu chuẩn là Edge GPU (NVIDIA Jetson Nano).

### 1. Ý nghĩa từng metric:
**Latency (Batch=1)**
- **Pre-processing (ms):** Thời gian tiền xử lý âm thanh thô.
- **Inference (ms):** Thời gian chạy tiến trình mô hình suy luận sâu.
- **Total Latency (ms):** Tổng thời gian đầu cuối.

**Power Consumption**
- **Idle Power (W):** Công suất nền của hệ thống khi board bật nguồn, chạy hệ điều hành nhưng không thực thi bất kỳ thuật toán AI nào.
- **Peak Power (W):** Mức công suất đỉnh (cao nhất) đo được đột biến trong quá trình DPU/GPU chạy tải tối đa.
- **Avg. Active Power (W):** Công suất trung bình khi hệ thống liên tục xử lý suy luận các mẫu liên tiếp nhau trong một khoảng thời gian dài.

**Efficiency (Chỉ số hiệu quả phần cứng)**
- **Throughput (FPS - Frames Per Second):** Tốc độ thông lượng, số lượng mẫu (hoặc ảnh spectrogram) được hệ thống phân loại thành công trong vòng 1 giây.
- **Energy (mJ/Inference):** Năng lượng tổng cộng để phân loại xong 1 mẫu. Chỉ số càng thấp càng tốt.
- **Performance/Watt (FPS/W):** Hiệu suất năng lượng. Thể hiện với 1 Watt điện tiêu thụ, thiết bị có thể xử lý bao nhiêu mẫu/giây. Chỉ số này càng cao thì hệ thống phần cứng càng tối ưu và ưu việt cho các ứng dụng chạy pin (như IoT/Wearable).

### 2. Phương pháp đo lường & Cách tính toán:
- **Idle Power / Peak Power / Avg. Active Power (W):** Tương tự bảng XIV, các thông số công suất được đo bằng cảm biến dòng điện của thiết bị (ví dụ dùng sysfs `/sys/class/hwmon/` trên Linux của Jetson Nano hoặc PMBus trên Ultra96-V2).
  - *Idle:* Đo khi bật nguồn nhưng không gọi phần mềm AI.
  - *Peak:* Ghi nhận đỉnh (chỉ số dòng điện tức thời cao nhất) hiển thị trên power monitor khi AI engine chạy full-load.
  - *Avg:* Lấy trung bình cộng của mảng giá trị công suất ghi lại được theo thời gian (ví dụ trong 30 giây thiết bị chạy liên tục).
- **Total Latency (ms):** Đo bằng thư viện C++ `std::chrono` bọc xung quanh toàn bộ hàm `classify()`, tính từ lúc mẫu bắt đầu đi vào cho đến khi có nhãn kết quả dự đoán (end-to-end).
- **Throughput (FPS) =** $\frac{1000}{\text{Total Latency (ms)}}$
  *(Lưu ý: Công thức này đúng với cấu hình thử nghiệm xử lý tuần tự từng mẫu một - Batch Size = 1. Ví dụ: Trên Ultra96-V2, thời gian tổng là 12.0 ms $\Rightarrow$ throughput = 1000 / 12 = 83.3 FPS).*
- **Energy (mJ/Inference) =** $\frac{\text{Avg. Active Power (W)}}{\text{Throughput (FPS)}} \times 1000 \quad \text{hoặc} \quad \text{Avg. Active Power (W)} \times \text{Total Latency (ms)}$
  *(Ví dụ: 3.81 W $\times$ 12.0 ms = 45.72 mJ).*
- **Performance/Watt (FPS/W) =** $\frac{\text{Throughput (FPS)}}{\text{Avg. Active Power (W)}}$
  *(Ví dụ: 83.3 FPS / 3.81 W = 21.86 FPS/W).*
- Cột **Improvement** (Sự cải tiến) tính bằng độ chênh lệch (tỷ số đối với cái tốt hơn, hoặc % giảm bớt):
  - Chênh lệch tốc độ: $36.50 / 12.00 = 3.04\times$ (Ultra96 nhanh hơn gấp 3.04 lần).
  - Phần trăm năng lượng tiết kiệm được: $\frac{211.6 - 45.7}{211.6} \times 100\% = 78.4\%$.
