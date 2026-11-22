# Data Card: HCM-Sim v2 (Synthetic Traffic Controlled Benchmark)


---

## 1. Dataset Overview & Metadata

| Thuộc tính siêu dữ liệu | Chi tiết đặc tả kỹ thuật |
|---|---|
| **Dataset ID** | `hcm_sim_v2` |
| **Version** | `2.0` (Đóng băng phục vụ giao thức đối sánh dự báo dài hạn 24 bước) |
| **Modality / Type** | **Lưu lượng tổng hợp có kiểm soát (Synthetic Traffic Controlled Benchmark)** |
| **Domain** | Dự báo chuỗi thời gian không gian - thời gian (Spatiotemporal Traffic Forecasting) |
| **Target Task** | Dự báo tỷ số tắc nghẽn giao thông đa bước ($T_{out} = 24$ bước, tương đương 06 giờ tới) |
| **Spatial Extent** | $N = 17$ trạm quan sát nút giao thông trọng yếu tại TP. Hồ Chí Minh |
| **Temporal Extent** | 28 ngày liên tục (2026-10-05 đến 2026-11-01), chu kỳ lấy mẫu 15 phút (2,688 mốc thời gian thô $\rightarrow 2,653$ mẫu cửa sổ trượt) |
| **Storage Format** | Tệp tensor PyTorch (`data/processed/hcm_sim_v2/graph_dataset.pt`), tệp siêu dữ liệu JSON (`meta.json`) |
| **Protocol** | `data/processed/hcm_sim_v2/benchmark_protocol_v1.json` |

---

## 2. Stakeholder Bridge

Tài liệu Data Card đóng vai trò cầu nối thông tin hai chiều giữa các nhóm đối tượng liên quan trong vòng đời nghiên cứu:

### 2.1. For Technical Stakeholders
- **Hỗ trợ cộng tác & mở rộng:** Cung cấp thông số cấu hình tensor chính xác ($X \in \mathbb{R}^{2653 \times 17 \times 12 \times F}, Y \in \mathbb{R}^{2653 \times 17 \times 24}$), đặc tả ma trận kề $A \in \mathbb{R}^{17 \times 17}$ và cơ chế đóng gói dữ liệu.
- **Bảo đảm Reproducibility:** Cố định hạt giống ngẫu nhiên (seed 42 cho bộ sinh, seeds 1–5 cho benchmark) và quy định khoảng đệm phân cách Purge gap = 35 bước thời gian nhằm ngăn chặn hiện tượng data leakage.
- **Tuân thủ quy trình kiểm thử:** Cung cấp bộ 49 bài kiểm thử tự động xác thực nguyên lý nhân quả và tính hợp lệ của phân vùng Non-IID.

### 2.2. For Non-Technical Stakeholders
- **Nhận diện chính xác năng lực & rủi ro:** Phân định rõ ràng bản chất mô phỏng của bộ dữ liệu; tránh việc ngụy biện hoặc đánh giá sai lệch năng lực mô hình khi áp dụng vào thực tế.
- **Hỗ trợ ra quyết định khoa học:** Cung cấp bức tranh thực nghiệm khách quan: lý giải nguyên nhân mô hình thống kê cổ điển `seasonal_naive` đạt sai số thấp nhất (MAE = 0.0663) do đặc trưng tuần hoàn của thuật toán sinh dữ liệu.
- **Bảo đảm tính liêm chính học thuật:** Thiết lập cam kết minh bạch về nguồn gốc dữ liệu, không ngụy tạo kết quả và không tuyên bố dữ liệu mô phỏng là số liệu quan sát thực địa.

---

## 3. Dataset Purpose & Usage Boundaries

### 3.1. Intended Uses
- **Chuẩn đối sánh độ bền vững có kiểm soát (Controlled Robustness Benchmark):** Sử dụng làm môi trường kiểm thử cô lập nhằm định lượng mức độ suy giảm hiệu năng của các kiến trúc học sâu không gian - thời gian khi gặp các biến dịch phân phối Non-IID:
  * Biến dịch số lượng Dirichlet (Quantity Skew) với các mức tham số tập trung $\alpha \in \{0.1, 0.3, 1.0\}$.
  * Biến dịch theo vùng chức năng đô thị (Functional Zone Shift).
  * Kiểm thử độ bền vững khi dữ liệu quan sát bị che khuất một phần (Observation Masking).
- **Môi trường thẩm định kiến trúc:** Đánh giá năng lực của cơ chế kết hợp nhãn vùng chức năng so với các mạng nơ-ron đồ thị và hồi quy chuỗi thời gian truyền thống.

### 3.2. Out-of-Scope / Prohibited Uses
- **Tuyệt đối không sử dụng làm căn cứ điều hành giao thông thực tế:** Do chuỗi lưu lượng được tạo lập từ bộ mô phỏng quy tắc toán học, dữ liệu không phản ánh các sự cố bất định thực tế (tai nạn, thời tiết cực đoan, ngập lụt, cấm đường đột xuất).
- **Tuyệt đối không tuyên bố là dữ liệu quan sát thực địa:** Không được mô tả bộ dữ liệu này là "dữ liệu giao thông thu thập thực tế tại TP. Hồ Chí Minh" trong bất kỳ bài báo khoa học hay báo cáo đề tài nào.

---

## 4. Data Provenance & Human Decisions

Dữ liệu HCM-Sim v2 được xây dựng trên sự phối hợp giữa **cấu trúc topo thực địa** và **giả định mô phỏng động học**:

### 4.1. Cấu trúc mạng lưới đường bộ (Spatial Road Network)
- Trích xuất từ Open Source Routing Machine (OSRM) trên bản đồ thực địa TP. Hồ Chí Minh.
- Gồm 17 nút giao thông trọng điểm: `ben_thanh_market`, `district_1`, `district_3`, `district_5`, `binh_thanh`, `tan_son_nhat_airport`, `landmark_81`, `thu_duc`, `linh_trung`, `suoi_tien`, `high_tech_park`, `vnu_hcm`, `hang_xanh`, `saigon_bridge`, `eastern_bus_station`, `thu_thiem_tunnel`, `pham_van_dong`.
- Quyết định thiết kế: Giữ nguyên ma trận khoảng cách OSRM để bảo đảm cấu trúc không gian phản ánh chính xác khoảng cách địa lý thực tế giữa các trung tâm đô thị.

### 4.2. Ngữ nghĩa vùng chức năng đô thị (Functional Zone Semantics)
- Trích xuất từ dữ liệu OpenStreetMap (OSM) theo bán kính ảnh hưởng của từng trạm.
- Phân loại thành 08 nhóm chức năng đa nhãn: Thương mại (Commercial), Cư dân (Residential), Công nghiệp (Industrial), Giáo dục (Education), Y tế (Health), Du lịch (Tourism), Giao thông (Transport), Văn phòng (Office).

### 4.3. Thuật toán sinh chuỗi lưu lượng động (Synthetic Simulation Engine)
- Thực thi qua tập lệnh `scripts/dev/generate_synthetic_traffic.py` (hạt giống cố định seed 42).
- Mô hình toán học kết hợp:
  * Đường cong chu kỳ nhật kỳ (diurnal curve) với 2 đỉnh cao điểm sáng (07:00–08:30) và chiều (17:00–18:30).
  * Hệ số điều biến ngày làm việc vs ngày nghỉ cuối tuần (weekday/weekend factor).
  * Hệ số tương tác vùng chức năng (khu văn phòng/thương mại có lưu lượng đỉnh tập trung cao hơn).
  * Quy luật lan truyền lưu lượng theo đồ thị kề OSRM.
  * Nhiễu ngẫu nhiên phân phối Gauss $\mathcal{N}(0, \sigma^2)$ với $\sigma = 0.05$.

---

## 5. Dataset Composition & Schema Specification

### 5.1. Quy cách Tensor trong tệp `graph_dataset.pt`
- **Tensor đầu vào lịch sử ($X$):**
  $$X \in \mathbb{R}^{S \times N \times T_{in} \times F} = \mathbb{R}^{2653 \times 17 \times 12 \times F}$$
  * $S = 2,653$: Tổng số mẫu cửa sổ trượt hợp lệ sau khi loại trừ biên.
  * $N = 17$: Số lượng trạm quan sát nút giao thông.
  * $T_{in} = 12$: Độ dài cửa sổ lịch sử (tương đương 03 giờ quan sát với bước lấy mẫu 15 phút).
  * $F = 4$: Số chiều đặc trưng (`congestion_ratio`, `traffic_delay_s`, `travel_time_s`, `ff_ratio`).
- **Tensor nhãn mục tiêu ($Y$):**
  $$Y \in \mathbb{R}^{S \times N \times T_{out}} = \mathbb{R}^{2653 \times 17 \times 24}$$
  * $T_{out} = 24$: Độ dài cửa sổ dự báo tương lai (tương đương 06 giờ dự báo đa bước liên tục).
  * Giá trị mục tiêu: Tỷ số tắc nghẽn giao thông (Congestion Ratio), nhận giá trị thực liên tục $\ge 0$.
- **Ma trận kề đồ thị ($A$):**
  $$A \in \mathbb{R}^{17 \times 17}, \quad A_{ii} = 0, \quad \sum_{j} A_{ij} = 1.0$$
  * Chuẩn hóa qua hàm nhân Gauss khoảng cách địa lý và row-normalization.

### 5.2. Nguyên tắc chuẩn hóa và phân chia dữ liệu
- **Chronological Split:**
  * Huấn luyện (Train): 70% (1,857 mẫu cửa sổ trượt).
  * Kiểm định (Validation): 10% (265 mẫu cửa sổ trượt).
  * Thử nghiệm (Test): 20% (461 mẫu cửa sổ trượt).
- **Khoảng đệm phân cách (Purge Gap):**
  $$\text{Purge Gap} = T_{in} + T_{out} - 1 = 12 + 24 - 1 = 35 \text{ bước}$$
  * Thiết lập tại ranh giới Train–Validation và Validation–Test (loại trừ $2 \times 35 = 70$ mẫu gián đoạn: 1857 + 35 + 265 + 35 + 461 = 2653).
- **Ước lượng chuẩn hóa Z-score:** Chỉ ước lượng tham số trung bình $\mu_{\text{train}}$ và phương sai $\sigma_{\text{train}}$ trên tập huấn luyện; áp dụng trực tiếp cho tập kiểm định và thử nghiệm, loại trừ distribution leakage.

---

## 6. Data Quality Audit & Statistical Evaluation

### 6.1. Kiểm định nguyên lý nhân quả (Causality Verification)
- Toàn bộ 49 bài kiểm thử tự động tại `tests/test_temporal_benchmark.py`, `tests/test_benchmark.py`, và `tests/test_eval_protocol.py` đều vượt qua 100% (49 passed in 6.06s).
- Xác nhận mô hình cơ sở Seasonal Naive chỉ đọc giá trị lịch sử tại mốc $t - 24\text{h}$, hoàn toàn không đọc dữ liệu tương lai.

### 6.2. Kết quả Clean Benchmark chính thức qua 05 hạt giống ngẫu nhiên (40 Runs)

| Thứ hạng | Mô hình đối sánh | Số lượng tham số | MAE | RMSE | MAPE (%) | WAPE (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | `seasonal_naive` | 0 | **0.0663 ± 0.0000** | **0.1692 ± 0.0000** | **5.54 ± 0.00** | **5.29 ± 0.00** |
| 2 | `zone_full` (Zone-Aware AH-GNN) | 421,737 | **0.1108 ± 0.0026** | **0.2029 ± 0.0066** | **8.17 ± 0.06** | **8.85 ± 0.21** |
| 3 | `baseline_ahgnn` (không nhãn vùng) | 298,840 | 0.1142 ± 0.0010 | 0.2049 ± 0.0011 | 8.50 ± 0.07 | 9.11 ± 0.08 |
| 4 | `lstm` | 286,104 | 0.1188 ± 0.0011 | 0.2108 ± 0.0022 | 8.71 ± 0.06 | 9.48 ± 0.09 |
| 5 | `gcn_gru` | 31,256 | 0.1274 ± 0.0007 | 0.2157 ± 0.0010 | 9.15 ± 0.05 | 10.17 ± 0.06 |
| 6 | `stgcn` | 31,128 | 0.1503 ± 0.0015 | 0.2293 ± 0.0026 | 10.92 ± 0.14 | 11.99 ± 0.12 |
| 7 | `mean_by_node` | 0 | 0.2512 ± 0.0000 | 0.3194 ± 0.0000 | 19.33 ± 0.00 | 20.05 ± 0.00 |
| 8 | `persistence` | 0 | 0.2638 ± 0.0000 | 0.4146 ± 0.0000 | 19.57 ± 0.00 | 21.05 ± 0.00 |

*Kiểm định ý nghĩa thống kê:* Kiến trúc đề xuất `zone_full` đạt mức cải thiện có ý nghĩa thống kê ($p < 0.001$, kiểm định bắt cặp t-test kết hợp hiệu chỉnh Holm) so với kiến trúc học sâu mạnh nhất trong nhóm đối sánh là `gcn_gru`.

---

## 7. Known Limitations, Uncertainties & Terms of Art

### 7.1. Hạn chế phương pháp luận & Hiện tượng tuần hoàn chiếm ưu thế
- **Ưu thế tuyệt đối của Seasonal Naive:** Thuật toán sinh mô phỏng dựa trên đường cong lặp nhật kỳ có tính quy luật rất cao. Do đó, mô hình thống kê tham chiếu đúng mốc giờ hôm trước ($t - 24\text{h}$) đạt MAE 0.0663, vượt trội hơn toàn bộ các mô hình học sâu. Đây là đặc tính nội tại của bộ dữ liệu mô phỏng, **bắt buộc phải báo cáo trung thực trong bảng kết quả chính của bài báo**.
- **Giá trị thực nghiệm của học sâu:** Vai trò của các mô hình học sâu không phải nhằm cạnh tranh với tính tuần hoàn lý tưởng, mà nhằm duy trì độ ổn định khi xuất hiện nhiễu ngẫu nhiên và biến dịch phân phối dữ liệu Non-IID.

### 7.2. Uncertainties & Unknowns
- Bộ sinh dữ liệu hiện tại chưa mô hình hóa được tác động tương hỗ phức tạp giữa các hiện tượng thời tiết bất thường (mưa lớn gây ngập đường) và hành vi phân luồng tự phát của người tham gia giao thông.

### 7.3. Giải thích thuật ngữ chuyên môn (Terms of Art Explained)
- **Purge Gap:** Khoảng đệm thời gian bị loại bỏ giữa các tập dữ liệu nhằm triệt tiêu tương quan chuỗi bậc cao và ngăn chặn hiện tượng rò rỉ dữ liệu.
- **Diurnal Pattern:** Đường cong dao động mang tính chu kỳ lặp lại theo chu kỳ 24 giờ trong ngày.
- **Non-IID:** Dữ liệu phân bố không độc lập và không đồng nhất giữa các nút quan sát, mô phỏng sự chênh lệch lưu lượng giữa các khu vực đô thị khác nhau.

---
