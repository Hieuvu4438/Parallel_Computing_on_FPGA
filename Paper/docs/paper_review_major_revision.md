# Peer-review nhanh cho manuscript UA-HCI (FPGA respiratory sound)

## Tổng quan quyết định biên tập (mô phỏng)
**Khuyến nghị hiện tại: Major Revision**

Lý do: ý tưởng có giá trị (uncertainty-aware cascade + FPGA co-design), nhưng phần thực nghiệm và báo cáo chưa đủ chặt để khẳng định các claim chính về **energy saving**, **generalization**, và **clinical utility**.

---

## Điểm mạnh
- Bài toán đúng nhu cầu: edge AI cho respiratory sound.
- Kiến trúc có logic: RF early-exit + CNN/DPU cho mẫu khó.
- Có mô tả co-design PS/PL rõ và có benchmark CPU vs DPU cho Layer-4.
- Viết tương đối đầy đủ pipeline huấn luyện (KD, QAT, augmentation).

---

## Các vấn đề chưa ổn (ưu tiên theo mức độ)

## 1) Critical — Claim năng lượng/độ trễ end-to-end chưa được chứng minh bằng số đo thật
Bạn nêu “route-level energy savings” nhưng phần kết quả mới dừng ở:
- latency Layer-4 CPU vs DPU,
- bảng khung đo e2e/energy (định nghĩa trường),
chưa có số đo hoàn chỉnh cho từng route (exit L1/L2/L3/L4), chưa có expected energy thực nghiệm.

**Hệ quả:** Claim “giảm năng lượng nhờ uncertainty-aware routing” chưa đủ bằng chứng.

**Cần sửa:**
1. Báo cáo đầy đủ cho mỗi route:
   - tỷ lệ route \(P_1, P_2, P_3, P_4\),
   - latency route-level (bao gồm preprocessing + RF + spectrogram + transfer + DPU khi có),
   - power trung bình và energy/sample cho từng route.
2. Tính và báo cáo:
   - \(E[C]\), \(E[E]\) của cascade,
   - baseline tĩnh (100% vào CNN),
   - % tiết kiệm với CI (bootstrap hoặc std theo fold/run).
3. Nêu rõ phương pháp đo điện năng (cảm biến nào, sampling rate, đồng bộ thời gian, idle subtraction).

---

## 2) Critical — Nguy cơ leakage/over-optimism từ segmentation và split chưa mô tả đủ chặt
Bài ghi “unit of inference = 8s”, nhưng L1-3 dùng 5s; đồng thời có repeat-padding/center-crop. Chưa rõ:
- segmentation diễn ra trước hay sau group split,
- một recording sinh nhiều segment có bị tràn giữa train/val không,
- threshold \(\tau_i\) tune trên tập nào.

**Hệ quả:** Accuracy cao có thể bị lạc quan nếu quy trình chia/tune chưa nested đúng chuẩn.

**Cần sửa:**
1. Thêm sơ đồ dữ liệu theo thứ tự chuẩn:
   split theo **subject** trước → fit scaler/augment/train chỉ trên train fold → validate trên subject hold-out.
2. Nêu rõ toàn bộ tuning (\(\tau_i\), RF hyperparams, KD/QAT epochs) có dùng validation nội bộ trong train fold hay không.
3. Tốt nhất dùng nested CV hoặc hold-out development set để chọn \(\tau_i\), sau đó khóa tham số và đánh giá test fold.

---

## 3) Major — So sánh SOTA chưa công bằng hoàn toàn
Bảng related work trộn nhiều protocol (random split, patient-independent, subject-independent, số lớp khác nhau). So sánh trực tiếp accuracy dễ gây hiểu nhầm.

**Cần sửa:**
- Tách bảng thành 2 nhóm: cùng protocol vs khác protocol.
- Đánh dấu rõ study nào random split.
- Nếu có thể, thêm baseline nội bộ cùng dữ liệu/split:
  1) CNN-only,
  2) RF-only,
  3) cascade (đề xuất).
- Báo cáo thêm balanced accuracy, macro-F1, per-class recall để xử lý imbalance.

---

## 4) Major — Tính lâm sàng và nhãn “Non-COPD” còn quá gộp
Non-COPD gom nhiều bệnh khác cơ chế (asthma, pneumonia, bronchiectasis, URTI...). Mô hình có thể học biên quyết định “COPD vs phần còn lại” hơn là phân biệt bệnh lý cụ thể.

**Cần sửa:**
- Thảo luận rõ giới hạn clinical claim.
- Bổ sung thí nghiệm ablation:
  - 2-class COPD vs non-COPD,
  - 3-class hiện tại,
  - hoặc tách subgroup trong Non-COPD nếu đủ mẫu.
- Báo cáo calibration (ECE/Brier) nếu định dùng như CDSS screening.

---

## 5) Major — Báo cáo thống kê chưa đầy đủ độ bất định
Hiện nêu best-fold và mean accuracy; cần đầy đủ hơn.

**Cần sửa:**
- Báo cáo mean ± std/95% CI cho accuracy, macro-F1, sensitivity/specificity theo lớp.
- Thêm confusion matrix theo từng fold + trung bình chuẩn hóa.
- Kiểm định ý nghĩa khi so với baseline (McNemar hoặc permutation test theo dự đoán paired).

---

## 6) Major — Mô tả huấn luyện CNN/KD/QAT còn thiếu chi tiết tái lập
Có nhiều thông tin tốt, nhưng thiếu một số khóa để reproducibility:
- seed, số lần chạy, framework version,
- optimizer cụ thể từng stage,
- policy chọn best checkpoint,
- chi tiết teacher ensemble inference/averaging.

**Cần sửa:**
- Thêm bảng “Reproducibility card” (hardware, software, seeds, epochs thực tế, early stop behavior, checkpoint rule).
- Công bố code/inference scripts hoặc pseudo-code đầy đủ cho cascade routing.

---

## 7) Minor — Một số điểm trình bày/kỹ thuật cần làm sạch
- Chính tả/tên thiết bị: “Utra96-V2” → “Ultra96-V2”.
- Tính nhất quán figure file name (có file trùng kiểu “Figure 3.jpg”, “FIG 3 (1).jpg”).
- Thuật ngữ “subject-independent” vs “patient-independent” nên thống nhất một chuẩn.
- Kết luận hiện hơi mạnh so với bằng chứng energy e2e.

---

## Gợi ý roadmap sửa bài (ưu tiên thực thi)
1. **Chốt protocol chống leakage** + viết rõ pipeline split/tune.
2. **Chạy lại evaluation chuẩn** với baseline nội bộ và CI.
3. **Đo e2e latency + energy route-level** trên Ultra96-V2 (và nếu có PYNQ-Z2).
4. **Viết lại Discussion/Conclusion** để khớp bằng chứng đo được.
5. **Làm bảng so sánh công bằng** với prior work.

---

## Đề xuất cấu trúc bổ sung vào manuscript
- Mục mới trong Experiments: “Fair Baselines and Statistical Testing”.
- Mục mới trong Hardware: “Route-level End-to-End Latency and Energy Measurement Protocol”.
- Appendix:
  - pseudo-code cascade inference,
  - full hyperparameters + seed,
  - per-fold confusion matrices.

---

## Kết luận review ngắn
Bài có hướng đi tốt và khả năng chấp nhận sau sửa, nhưng hiện tại thiếu bằng chứng thực nghiệm cốt lõi cho các claim hệ thống (đặc biệt energy/latency end-to-end và fairness của đánh giá). Mức phù hợp hiện tại: **Major Revision**.
