# Báo cáo: Tổng hợp kết quả dự báo đa bước (Multi-step Forecasting)

## 1. PYTHONUNBUFFERED=1 là gì?
`PYTHONUNBUFFERED=1` là một biến môi trường (environment variable) được truyền vào trước lệnh chạy Python để báo cho trình thông dịch (Python interpreter) biết rằng **không được gộp (buffer) output**.

- **Mặc định**: Khi chạy Python trong các script tự động hoặc chạy ngầm (không tương tác trực tiếp), Python thường lưu trữ (buffer) các lệnh `print()` vào bộ nhớ tạm. Nó sẽ chờ đến khi bộ đệm đầy hoặc chương trình kết thúc thì mới in ra một lượt. Điều này khiến cho quá trình theo dõi log bị "đóng băng" (bạn tưởng chương trình bị treo nhưng thực chất nó đang chạy).
- **Khi thêm `PYTHONUNBUFFERED=1`**: Mọi lệnh `print()` hoặc xuất log sẽ được đẩy thẳng ra màn hình (hoặc file log) ngay lập tức tại thời điểm dòng code đó thực thi. Nhờ đó, bạn có thể theo dõi tiến độ huấn luyện (loss, val_MAE qua từng epoch) theo thời gian thực (real-time).

---

## 2. Tổng quan những việc đã thực hiện

Để giải quyết yêu cầu chạy đánh giá mô hình đa bước trên dải dự báo cực rộng (từ 15 phút đến 120 phút, tức $T_{out} \in [3, 6, 9, 12, 18, 24]$) cho bài báo, chúng ta đã phải đối mặt với một số vấn đề về dữ liệu gốc và thực hiện các thay đổi kỹ thuật quan trọng. Dưới đây là luồng xử lý chi tiết:

### A. Tích hợp tham số vào `build_graph.py`
Ban đầu, script `build_graph.py` được hardcode với chiều dự báo cố định `T_out = 3`.
- **Thực hiện**: Đã thêm thư viện `argparse` để script có thể linh hoạt nhận tham số `--t_out` từ dòng lệnh (ví dụ: `python build_graph.py --t_out 6`).

### B. Vượt qua lỗi thiếu dữ liệu thô (Raw Data)
Khi chạy lại `build_graph.py`, một lỗi `FileNotFoundError` phát sinh do thư mục `data/raw/` không chứa file dữ liệu gốc `hcm_osrm_dataset.csv`. Việc gọi trực tiếp `build_graph.py` từ đầu là không khả thi.
- **Thực hiện**: Đã tạo ra một script thay thế mang tên `scripts/rebuild_graph_no_raw.py`.
- **Cách thức hoạt động**: Thay vì đọc từ file CSV, script này tải thẳng file dữ liệu đã được window (`graph_dataset.pt` hiện có của $T_{out}=3$). Nó thực hiện quá trình dịch ngược (reverse-windowing) để khôi phục chuỗi thời gian liên tục ban đầu (unwindowed sequence), sau đó tiếp tục thực hiện phép chia sliding window mới với các nhãn dự báo dài hơn ($T_{out} = 6$ và $12$). Phương pháp này đảm bảo tái sử dụng 100% dữ liệu gốc hoàn hảo mà không cần source CSV.

### C. Tự động hóa đánh giá đa bước (`run_multistep.py`)
Mục đích là để huấn luyện 4 mô hình: `zone_full_tc`, `zone_full_sinc`, `gcn_gru` và `stgcn` trên cả **6 khung thời gian** (tổng cộng 24 lượt huấn luyện) mà không cần phải gõ lệnh bằng tay.
- **Thực hiện**: Đã tạo file tự động hóa toàn diện `scripts/run_multistep.py`.
- **Luồng hoạt động**:
  1. Duyệt qua mảng `t_outs = [3, 6, 9, 12, 18, 24]`.
  2. Kích hoạt `rebuild_graph_no_raw.py` để format lại Tensor dataset theo đúng `T_out`.
  3. Lần lượt khởi tạo và huấn luyện 4 mô hình nêu trên thông qua hàm `run_experiment` của module `train.py`. Trọng số mô hình (weights `.pt`) của mỗi mốc sẽ tự động được gom riêng rẽ vào các thư mục con tương ứng (ví dụ: `data/results/multistep/T_3/`).
  4. Trích xuất Metrics (MAE, RMSE, MAPE, và Zone-Stratified MAE).
  5. Xuất báo cáo tổng hợp lưu tại `data/results/multistep/multistep_results.csv` và tự động sinh ra Markdown Table (Bảng kết quả đẹp) hiển thị tại `data/results/multistep/multistep_results.md` để dán vào bài báo.

### 3. Hướng dẫn sử dụng
Kịch bản đánh giá được gói gọn hoàn chỉnh. Bạn có thể tự mình khởi chạy bất cứ lúc nào qua lệnh sau:

```bash
PYTHONUNBUFFERED=1 ./venv/bin/python scripts/run_multistep.py
```
