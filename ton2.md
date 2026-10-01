# Báo Cáo Tổng Kết: Task 4.3 - Clean Evaluation & Statistical Reporting

Tài liệu này tổng hợp toàn bộ các công việc đã được thực hiện nhằm đáp ứng yêu cầu "Mọi experiment Non-IID dùng cùng protocol sạch và cho ra bảng kết quả có thể đưa vào report", bảo đảm 100% không có data leakage và chuẩn hóa pipeline báo cáo thống kê.

---

## 1. Những Gì Đã Làm

1. **Chuẩn hóa Giao thức Đánh giá (Evaluation Protocol)**:
   - Xóa bỏ tình trạng code duplicate rải rác giữa các script. Toàn bộ logic chia tập (split), chuẩn hóa (normalization) và tính toán metric được quy về một "nguồn chân lý duy nhất" (single source of truth).
   - **Chronological Split**: Áp dụng chia tập thời gian có khoảng nghỉ (`purge gap = T_in + T_out - 1`) để chặn đứng temporal leakage. Bắt buộc áp dụng cho mọi file.
   - *👉 File tác động:* Tạo mới [`utils/eval_protocol.py`](file:///home/toon/Zone-Aware-GNN/utils/eval_protocol.py), sửa [`scripts/train.py`](file:///home/toon/Zone-Aware-GNN/scripts/train.py) và [`utils/__init__.py`](file:///home/toon/Zone-Aware-GNN/utils/__init__.py).

2. **Vá Lỗ Hổng Non-IID Evaluator**:
   - Sửa lỗi chia tập sai trong `eval_non_iid.py`: Đưa chrono split lên đầu (áp dụng cho toàn dataset) TRƯỚC KHI áp dụng mask (chỉ để tính loss). Điều này giúp baseline so sánh công bằng.
   - Sửa lỗi rò rỉ normalization: Normalizer giờ đây **chỉ được fit trên tập train**.
   - Sửa lỗi scale metric: Bắt buộc `inverse_transform` X/Y trước khi tính toán MAE/RMSE/MAPE/WAPE để trả về đơn vị thực tế (vd: km/h).
   - *👉 File tác động:* Sửa [`scripts/eval_non_iid.py`](file:///home/toon/Zone-Aware-GNN/scripts/eval_non_iid.py).

3. **Hệ thống Multi-seed & Báo cáo Tự động**:
   - Cập nhật runner chạy nhiều seed tự động đánh phiên bản file CSV để không mất dữ liệu (ví dụ: `v1.csv`, `v2.csv`). Gắn kèm mã Git Commit Hash để track code version.
   - Tích hợp cờ test nhanh `--smoke` (2 seeds × 5 epochs).
   - Xây dựng hệ thống tự động sinh report (`.md`, `.tex`) bao gồm tính toán Mean ± Std, kiểm định thống kê (T-test), hiệu chỉnh Holm-Bonferroni và tính Effect Size (Cohen's d).
   - *👉 File tác động:* Sửa [`scripts/run_multi_seed.py`](file:///home/toon/Zone-Aware-GNN/scripts/run_multi_seed.py), tạo mới [`scripts/generate_report.py`](file:///home/toon/Zone-Aware-GNN/scripts/generate_report.py).

4. **Kiểm thử Toàn diện (Unit Tests & CI Mocking)**:
   - Xây dựng 18 test cases bọc lót toàn bộ ngóc ngách của quá trình đánh giá (test overlap, gap bounds, normalizer roundtrip, chia zero target WAPE/MAPE). Mọi test chạy qua 100%.
   - Chạy test thành công pipeline bằng `quick_test.py` sau khi giả lập môi trường clone sạch (không cần data TomTom gốc).
   - *👉 File tác động:* Tạo mới [`tests/test_eval_protocol.py`](file:///home/toon/Zone-Aware-GNN/tests/test_eval_protocol.py).

---

## 2. Báo Cáo / Output Nằm Ở Đâu?

Toàn bộ kết quả chạy tự động được xuất ra tại thư mục: **`data/results/`**

- **Dữ liệu thô của các lần chạy (Raw Multi-seed Logs)**: 
  - `data/results/multiseed_runs_v1.csv` 
  - `data/results/multiseed_runs_v2.csv`
  - `data/results/multiseed_runs_v3.csv` *(có chứa mã commit, thông số config và metric từng fold)*
- **Bảng tổng hợp Statistical (Report Summary)**:
  - `data/results/report_summary_{timestamp}.csv` (Tổng hợp Mean, Std, CI95%)
  - `data/results/report_significance_{timestamp}.csv` (P-values và kết quả kiểm định Holm)
- **Báo cáo định dạng chuẩn Paper (Sẵn sàng copy-paste)**:
  - Bảng Markdown: `data/results/report_chrono_{timestamp}.md`
  - Bảng LaTeX: `data/results/report_chrono_{timestamp}.tex`
  - Báo cáo PDF hoàn chỉnh: `Source_Report/main.pdf` (Tích hợp sẵn bảng kết quả tự động sinh)
- **Tài liệu hướng dẫn (Cheatsheet)**:
  - Bí kíp chạy hệ thống: `run.md` (Chứa 6 lệnh chạy toàn bộ pipeline)

### 🛠️ Các File Được Chỉnh Sửa (Modified)
4. [`scripts/train.py`](file:///home/toon/Zone-Aware-GNN/scripts/train.py) & [`scripts/eval_non_iid.py`](file:///home/toon/Zone-Aware-GNN/scripts/eval_non_iid.py)
   - **Chi tiết**: Ép dùng chung `chronological_split` và chỉ fit normalizer trên tập Train. Cập nhật `inverse_transform` trước khi tính metric.
5. [`scripts/run_multi_seed.py`](file:///home/toon/Zone-Aware-GNN/scripts/run_multi_seed.py)
   - **Chi tiết**: Hỗ trợ `--smoke` test, tự động lưu version output CSV, ghi thêm Git Commit Hash vào kết quả.
6. [`Source_Report/main.tex`](file:///home/toon/Zone-Aware-GNN/Source_Report/main.tex) & `results.tex`
   - **Chi tiết**: Sửa bìa báo cáo (HK261, Logo trường), dọn dẹp các mục include lỗi, tự động compile bảng LaTeX thành file PDF cuối cùng mà không dính lỗi ký tự đặc biệt (`_`).

### 🚀 Cập nhật bổ sung
- **Horizon Breakdown**: Cập nhật logic `compute_metrics` để tính chi tiết `MAE_t`, `RMSE_t` theo từng khung thời gian dự đoán (Horizon).

---
*Xác nhận: Tất cả các công việc trong Track Tôn (Task 4.3) đã Đóng / Hoàn tất 100%.*
