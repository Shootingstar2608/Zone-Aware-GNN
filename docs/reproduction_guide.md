# Hướng dẫn Tái lập Thực nghiệm (Reproduction Guide)

Tài liệu này cung cấp quy trình chuẩn hóa từng bước nhằm phục vụ công tác tái lập hoàn toàn từ bản sao chép sạch (**from clean clone**) đối với tập dữ liệu **HCM-Sim v2**, đóng băng giao thức thực nghiệm, sinh phân vùng Non-IID, thực thi kiểm thử nhanh (Smoke Run), và hoàn thành chuẩn đối sánh sạch 05 hạt giống ngẫu nhiên (Clean Benchmark 5-seed) của dự án **Zone-Aware-GNN** trên môi trường máy tính độc lập (PowerShell trên Windows hoặc Bash trên Linux).

---

## 0. Khởi tạo từ Clone Sạch & Cấu hình Môi trường (Clean Clone & Setup)

### 0.1. Sao chép Mã nguồn từ Kho Lưu trữ
Từ môi trường dòng lệnh (PowerShell hoặc Terminal), thực hiện sao chép kho mã nguồn:
```powershell
git clone https://github.com/Shootingstar2608/Zone-Aware-GNN.git
cd Zone-Aware-GNN
```

### 0.2. Thiết lập Môi trường Conda & Bảng mã UTF-8
Khuyến nghị sử dụng Python 3.10 trên môi trường Conda biệt lập:
```powershell
# Cấu hình bảng mã UTF-8 cho PowerShell trên Windows (tránh lỗi charmap cp1252)
$env:PYTHONUTF8 = 1
$env:PYTHONIOENCODING = "utf-8"

# Tạo và kích hoạt môi trường Conda mới
conda create -n dp4 python=3.10 -y
conda activate dp4

# Cài đặt toàn bộ danh mục thư viện đồng bộ từ requirements.txt
pip install -r requirements.txt
```
*(Lưu ý: Tệp `requirements.txt` đã được cấu hình sẵn chỉ mục bánh xe `--extra-index-url https://download.pytorch.org/whl/cu118` nhằm bảo đảm PyTorch tương thích hoàn hảo với phần cứng GPU CUDA 11.8. Nếu máy bạn chạy thuần CPU, bạn có thể chạy: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`).*

### 0.3. Xác thực Sự Tồn tại của Dữ liệu Tĩnh Cơ sở
Từ thư mục gốc dự án, thực thi lệnh xác nhận tính sẵn sàng của 02 tệp dữ liệu tĩnh cơ sở:
```powershell
Test-Path data/raw/hcm_osrm_dataset.csv
Test-Path data/raw/zone_labels.csv
```
*Yêu cầu kỹ thuật bắt buộc:* Cả hai lệnh trên phải trả về giá trị `True`. Tuyệt đối không sử dụng dữ liệu quan sát thực địa từ TomTom API để thay thế cho tập dữ liệu thô OSRM này.

---

## 1. Khởi tạo Dữ liệu Tổng hợp HCM-Sim v2 & Dựng Đồ thị (Build HCM-Sim v2)

### 1.1. Khởi tạo chuỗi lưu lượng tổng hợp (Synthetic Traffic Generation)
Sinh 28 ngày dữ liệu chuỗi thời gian liên tục với bước lấy mẫu 15 phút (tổng cộng 2,688 mốc thời gian thô):
```powershell
New-Item -ItemType Directory -Force data/generated/hcm_sim_v2 | Out-Null

python scripts/dev/generate_synthetic_traffic.py `
  --seed 42 `
  --days 28 `
  --start 2026-10-05 `
  --interval-min 15 `
  --out data/generated/hcm_sim_v2/hcm_sim_traffic.csv
```

### 1.2. Khởi tạo cấu trúc đồ thị chuỗi thời gian (Graph Dataset Construction)
Chuyển đổi chuỗi lưu lượng thành tensor đồ thị phục vụ dự báo đa bước ($T_{in} = 12$, $T_{out} = 24$):
```powershell
New-Item -ItemType Directory -Force data/processed/hcm_sim_v2 | Out-Null

python scripts/build_graph.py `
  --traffic-path data/generated/hcm_sim_v2/hcm_sim_traffic.csv `
  --out-dir data/processed/hcm_sim_v2 `
  --t-in 12 `
  --t-out 24
```

> **[LƯU Ý THỰC THI #1: ĐẶC TẢ THAM SỐ CỬA SỔ DỰ BÁO]**  
> Trong mã nguồn `scripts/build_graph.py`, tham số độ dài cửa sổ dự báo mang giá trị mặc định là `T_out = 3`. Để phục vụ giao thức đánh giá dài hạn 24 bước thời gian của bộ dữ liệu HCM-Sim v2, người thực thi **bắt buộc phải chỉ định tường minh đối số `--t-out 24`**. Việc không thiết lập đối số này sẽ dẫn đến việc định dạng kích thước tensor không tương thích với giao thức chuẩn hóa ($S = 2,653$) và làm sai lệch toàn bộ quy trình đánh giá phía sau.

### 1.3. Xác thực tính toàn vẹn của tập dữ liệu qua các phép kiểm tra logic (Assertions)
Thực thi kiểm tra tính toàn vẹn trước khi chuyển sang bước tiếp theo:
```powershell
@'
import json, torch
from pathlib import Path
p = Path("data/processed/hcm_sim_v2")
meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
ds = torch.load(p / "graph_dataset.pt", map_location="cpu", weights_only=False)
assert meta["data_source"] == "synthetic", meta
assert meta["N"] == 17, meta
assert meta["T_in"] == 12 and meta["T_out"] == 24, meta
assert meta["S"] == 2653, meta
assert ds["X"].shape[0] == 2653 and ds["Y"].shape[-1] == 24
print("✅ HCM-Sim v2 dataset checks passed successfully:", meta)
'@ | python -
```

---

## 2. Đóng băng Giao thức Thực nghiệm (Freeze Protocol)

Khởi tạo tệp đặc tả giao thức phân chia dữ liệu Chronological Train (70% = 1,857) / Val (10% = 265) / Test (20% = 461) cùng khoảng đệm ngắt Purge gap = 35 bước:

```powershell
python scripts/freeze_hcm_sim_protocol.py --data-dir data/processed/hcm_sim_v2 --force
```
*Kết quả kỳ vọng:* Tệp cấu hình chuẩn hóa `data/processed/hcm_sim_v2/benchmark_protocol_v1.json` được tạo lập thành công (cờ `--force` cho phép ghi đè manifest an toàn khi tái lập từ bộ tensor vừa dựng).

---

## 3. Khởi tạo & Xác thực Phân vùng Non-IID (Generate & Verify Partitions)

### 3.1. Khởi tạo phân vùng Non-IID từ tập dữ liệu đã dựng (Generate Partitions)
Sinh các kịch bản phân vùng Non-IID (Dirichlet Quantity Skew, Functional Zone Skew, Temporal Shift):
```powershell
python -m benchmark.partition_gen --data-dir data/processed/hcm_sim_v2
```
*Kết quả kỳ vọng:* Tệp `data/partitions/hcm_sim_v2/partitions_meta.json` được tạo lập hoàn chỉnh với chữ ký fingerprint khớp với tệp tensor `graph_dataset.pt`.

### 3.2. Xác thực tính tái lập bit-for-bit của phân vùng (Verify Partitions)
```powershell
python scripts/verify_partitions.py --data-dir data/processed/hcm_sim_v2
```
*Kết quả kỳ vọng:* Chương trình thông báo:
`✅ Moi partition tai tao duoc bit-for-bit tu metadata.`

---

## 4. Kiểm định Nguyên lý Nhân quả & Tính Hợp lệ (Causality Verification)

Trước khi tiến hành huấn luyện mô hình, người thực thi bắt buộc phải chạy kiểm định tự động nhằm xác nhận mô hình cơ sở Seasonal Naive tuân thủ nghiêm ngặt nguyên lý nhân quả (không đọc trước nhãn tương lai, loại trừ hiện tượng rò rỉ dữ liệu):

```powershell
pytest -q
```

> **[LƯU Ý THỰC THI #2: CẤU HÌNH PHẠM VI KIỂM THỬ TỰ ĐỘNG]**  
> Dự án đã cấu hình tệp `pytest.ini` với quy tắc `testpaths = tests`, đảm bảo khi thực thi lệnh `pytest -q`, Pytest sẽ tự động giới hạn phạm vi quét bài test trong thư mục `tests/` và bỏ qua các file nháp cũ trong `scripts/`. Toàn bộ các bài kiểm tra nhân quả (`test_temporal_benchmark.py`), mô hình (`test_benchmark.py`), giao thức (`test_eval_protocol.py`) và dữ liệu thực tế (`test_hcm_real_static_metadata.py`) đều sẽ được thực thi và đạt chuẩn 100% Pass.

---

## 5. Thực thi Kiểm thử Nhanh (Smoke Run)

Trước khi chạy toàn bộ 40 lượt huấn luyện kéo dài nhiều giờ, thực hiện một lượt chạy kiểm thử nhanh (Smoke Run) trên 08 mô hình với seed 42 và 05 epochs nhằm xác thực toàn bộ pipeline thực thi trơn tru:

```powershell
New-Item -ItemType Directory -Force data/results/hcm_sim_v2 | Out-Null

python scripts/run_multi_seed.py `
  --data-dir data/processed/hcm_sim_v2 `
  --protocol data/processed/hcm_sim_v2/benchmark_protocol_v1.json `
  --models persistence mean_by_node seasonal_naive lstm gcn_gru stgcn baseline_ahgnn zone_full `
  --split-modes chrono `
  --seeds 42 `
  --epochs 5 `
  --out data/results/hcm_sim_v2/clean_smoke_v1.csv
```
*Kết quả kỳ vọng:* 
- Tệp `data/results/hcm_sim_v2/clean_smoke_v1.csv` được ghi nhận với đủ 08 mô hình đều có `status = ok`.
- Mô hình `seasonal_naive` đạt MAE xấp xỉ `0.0663` ngay từ lượt smoke.

---

## 6. Thực thi Clean Benchmark Đa Hạt Giống (Multi-seed Clean Benchmark, 40 Runs)

Tiến hành huấn luyện và đánh giá 08 mô hình trên 05 hạt giống ngẫu nhiên độc lập ($8 \times 5 = 40$ lượt thực nghiệm) kết hợp khoảng đệm phân cách Purge gap = 35 và 100 epochs:

```powershell
python scripts/run_multi_seed.py `
  --data-dir data/processed/hcm_sim_v2 `
  --protocol data/processed/hcm_sim_v2/benchmark_protocol_v1.json `
  --models persistence mean_by_node seasonal_naive lstm gcn_gru stgcn baseline_ahgnn zone_full `
  --split-modes chrono `
  --seeds 1 2 3 4 5 `
  --epochs 100 `
  --save-checkpoints `
  --out data/results/hcm_sim_v2/clean_5seed.csv
```

> **[LƯU Ý THỰC THI #3: CƠ CHẾ BẢO TOÀN DỮ LIỆU THỰC NGHIỆM]**  
> Tiến trình `scripts/run_multi_seed.py` được tích hợp cơ chế bảo vệ tính toàn vẹn: chương trình sẽ từ chối tự động ghi đè lên tệp kết quả nếu tệp đã tồn tại trong thư mục đích. Trong trường hợp tiến trình thực thi bị gián đoạn (chẳng hạn khi chỉ mới hoàn thành một số hạt giống), người thực thi có thể **lặp lại chính xác câu lệnh trên kết hợp với việc bổ sung cờ `--resume`**; hệ thống sẽ tự động đối soát và tiếp tục thực hiện các lượt huấn luyện còn thiếu mà không làm ảnh hưởng đến dữ liệu đã lưu trữ.

---

## 7. Phân tích Thống kê Tự động & Kết xuất Bảng Kết quả (Statistical Analysis & LaTeX Export)

Sau khi toàn bộ 40 lượt thực nghiệm hoàn thành và tệp `clean_5seed.csv` đã được lưu trữ hoàn chỉnh, thực thi script phân tích thống kê:

```powershell
python scripts/stat_analysis.py `
  --runs data/results/hcm_sim_v2/clean_5seed.csv `
  --proposed zone_full `
  --metrics MAE RMSE MAPE WAPE `
  --out-prefix data/results/hcm_sim_v2/clean_5seed_statistics
```
*Kết quả kỳ vọng tạo ra:* 
- `data/results/hcm_sim_v2/clean_5seed_statistics_summary.csv` (Bảng giá trị trung bình $\pm$ độ lệch chuẩn).
- `data/results/hcm_sim_v2/clean_5seed_statistics_significance.csv` (Bảng phân tích kiểm định ý nghĩa thống kê).
- `data/results/hcm_sim_v2/clean_5seed_statistics_tables.tex` (Bảng mã nguồn LaTeX sẵn sàng nhúng vào báo cáo học thuật).

---

## 8. Đánh giá Độ bền vững Non-IID Dirichlet Quantity Skew (Non-IID Robustness Evaluation)

Định lượng độ suy giảm hiệu năng của các kiến trúc mô hình dưới các mức độ biến dịch phân phối số lượng Dirichlet $\alpha \in \{0.1, 0.3, 1.0\}$ trên hạt giống ngẫu nhiên seed 1:

```powershell
New-Item -ItemType Directory -Force data/results/hcm_sim_v2/quantity_skew_seed1 | Out-Null

python scripts/eval_non_iid.py `
  --data-dir data/processed/hcm_sim_v2 `
  --partition-path data/partitions/hcm_sim_v2/partitions_meta.json `
  --models seasonal_naive lstm gcn_gru stgcn baseline_ahgnn zone_full `
  --alphas 0.1 0.3 1.0 `
  --seed 1 `
  --epochs 100 `
  --out-dir data/results/hcm_sim_v2/quantity_skew_seed1
```
*Kết quả kỳ vọng tạo ra:* Tệp dữ liệu CSV tổng hợp kết quả và biểu đồ trực quan hóa mức độ suy giảm hiệu năng theo các mức tham số $\alpha$.

---

## 9. Tự động hóa Kiểm định Toàn trình (Turnkey One-Click Self-Test Script)

Nhằm tối ưu hóa công tác chấm điểm và thẩm định độc lập, repository cung cấp một kịch bản Python tự động hóa khép kín:

```powershell
python scripts/self_test_reproduction.py
```

### Các tác vụ được kịch bản tự động hóa 100%:
1. **Quản lý vòng đời môi trường Conda:** Tự động tạo môi trường mới `dp4` (Python 3.10) và cài đặt đầy đủ `requirements.txt`.
2. **Khởi tạo và kiểm định clone sạch:** Tự tạo thư mục `self_test/`, clone sạch mã nguồn từ GitHub chính thức, kiểm tra file dữ liệu tĩnh.
3. **Thực thi tuần tự toàn bộ pipeline:** Sinh dữ liệu tổng hợp 28 ngày $\rightarrow$ Dựng tensor đồ thị $\rightarrow$ Kiểm định assertions logic $\rightarrow$ Đóng băng protocol (`--force`) $\rightarrow$ Sinh partition Non-IID & Kiểm tra bit-for-bit $\rightarrow$ Chạy kiểm thử nhân quả Pytest $\rightarrow$ Chạy Smoke Run ghi nhận kết quả.
4. **Tự động dọn dẹp sạch sẽ (Clean Up):** Sau khi hoàn tất, kịch bản tự động xóa sạch thư mục `self_test/` và xóa bỏ hoàn toàn môi trường `dp4` khỏi máy tính, không để lại rác trên hệ thống.
*(Nếu muốn giữ lại thư mục hoặc môi trường để thanh tra kết quả, người dùng có thể truyền thêm cờ `--keep`).*

