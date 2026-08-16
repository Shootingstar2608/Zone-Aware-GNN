# Kế Hoạch Tuần 2 — Team 4 Người (21/07 – 27/07/2026)

---

## Tổng kết Tuần 1: Những gì đã có

| Hạng mục | File/Kết quả | Trạng thái |
|---|---|---|
| Dữ liệu TomTom | `tomtom_traffic.csv` (~24 MB, ≈7 ngày) | ✅ |
| Nhãn vùng OSM | `zone_labels.csv` (17 nút × 8 zone types) | ✅ |
| Graph Dataset | `graph_dataset.pt` (658 mẫu, F=4 features) | ✅ |
| 3 Baselines | `models/baselines.py` (LSTM, GCN-GRU, STGCN) — code xong | ✅ |
| Ablation Study | 4 variants đã train, zone_full MAE=0.0795 | ✅ |
| EDA / JSD | Heatmap + Correlation Plot đã sinh | ✅ |
| Collect zones generic | `scripts/collect_zones.py` — tổng quát cho mọi thành phố | ✅ |

**Kết quả ablation đã có:**

| Variant | MAE | RMSE | MAPE% |
|---|---|---|---|
| baseline_ahgnn | 0.2102 | 0.2955 | 15.83 |
| zone_concat | 0.0963 | 0.1536 | 7.11 |
| zone_weight | 0.1057 | 0.1613 | 7.84 |
| **zone_full (Proposed)** | **0.0795** | **0.1419** | **6.14** |

---

## Mục tiêu Tuần 2

> **Tuần 2 = Experiments hoàn chỉnh + Figures cho paper + Bắt đầu viết**

1. Train và đánh giá 3 baselines trên dữ liệu thực → bảng so sánh hoàn chỉnh
2. Triển khai cải tiến Time-Conditioned Zone Embedding (đóng góp khoa học #2)
3. Sinh toàn bộ Figures chất lượng xuất bản cho paper
4. Viết Sections 1 & 2 (Introduction + Related Work)

---

## Người A (Lead — Model & Training)

### Nhiệm vụ A.1: Chạy 3 Baselines trên dữ liệu thực
- Chạy `python scripts/train.py --baselines` → sinh `baseline_results.csv`
- Xác nhận kết quả hợp lệ (không NaN/Inf)
- So sánh MAE của 3 baselines vs `zone_full` → xác nhận proposed model tốt hơn tất cả

### Nhiệm vụ A.2: Triển khai Time-Conditioned Zone Embedding
- Sửa `ZoneEmbedding` trong `models/zone_aware_gnn.py` thành `TimeConditionedZoneEmbedding`
- Công thức: $\tilde{z}^{(v,t)} = g_t \odot \text{MLP}(z^{(v)})$ — thêm cổng thời gian
- Cập nhật pipeline: `ZoneAwareAdjacency` và `ZoneModulatedGraphConv` phải nhận z_embed kích thước `(B, N, d_z)` thay vì `(N, d_z)`
- Tạo variant mới `zone_full_tc` trong `train.py`
- Train và so sánh: `zone_full_tc` vs `zone_full` → kỳ vọng MAE giảm thêm

### Nhiệm vụ A.3: Hyperparameter Tuning (nếu còn thời gian)
- Grid search nhỏ trên các tham số: `zone_embed_dim = [8, 16, 32]`, `alpha = [0.3, 0.5, 0.7]`
- Chọn best config dựa trên validation MAE

### Output mong muốn
- [ ] `data/results/baseline_results.csv` — 3 dòng kết quả
- [ ] `models/zone_aware_gnn.py` — class `TimeConditionedZoneEmbedding` mới
- [ ] `data/results/all_results.csv` — bảng tổng hợp tất cả models (baselines + ablation + TC variant)
- [ ] Model checkpoint `zone_full_tc_best.pt`

---

## Người B (Deep Learning Engineer)

### Nhiệm vụ B.1: Hỗ trợ A triển khai Time-Conditioned Zone Embedding
- Review code của A, test forward pass với dummy input
- Đảm bảo backward pass không bị gradient explosion
- Test interface tương thích: `zone_full_tc` phải chạy được với cùng `train.py`

### Nhiệm vụ B.2: Triển khai Sinusoidal Time Encoding (cải tiến #2)
- Thay thế 4 nhãn thời gian rời rạc bằng mã hóa Sin/Cos liên tục
- Cần sửa `build_graph.py` để lưu thêm trường `hour` và `day_of_week` (float) thay vì chỉ 4 nhãn
- Tạo class `SinusoidalTimeEncoder` → test forward pass
- Tạo variant `zone_full_sine` trong `train.py`

### Nhiệm vụ B.3: Viết Unit Tests cho tất cả models
- Tạo `tests/test_models.py` kiểm tra:
  - Forward pass shape cho tất cả 7 variants (3 baselines + 4 ablation)
  - Gradient flow (backward pass không NaN)
  - Parameter count nhất quán

### Output mong muốn
- [ ] Class `SinusoidalTimeEncoder` trong `models/zone_aware_gnn.py`
- [ ] `build_graph.py` cập nhật — lưu thêm `hour`, `day_of_week`
- [ ] `tests/test_models.py` — unit tests cho tất cả variants
- [ ] (Bonus) Kết quả `zone_full_sine` nếu kịp train

---

## Người C (Data & GIS Engineer)

### Nhiệm vụ C.1: Zone-Stratified Error Analysis (QUAN TRỌNG NHẤT)
- Tạo `scripts/zone_stratified_analysis.py`
- Load kết quả từ các model đã train (best.pt), chạy evaluate trên test set
- Tính MAE riêng cho từng nhóm TAZ theo số lượng zone (1, 2, 3, 4, 5, 6)
- **Sinh Figure 3 — Main Result:** Grouped Bar Chart (Baseline vs Proposed, theo zone_count)
- Kỳ vọng: improvement tăng dần khi zone_count tăng → chứng minh zone-awareness hiệu quả

### Nhiệm vụ C.2: Nâng cấp tất cả Figures lên chất lượng xuất bản
- **Figure 1a** — JSD Heatmap: font serif, DPI≥300, rút gọn tên TAZ, thêm colorbar label
- **Figure 1b** — Correlation JSD vs Zone Similarity: thêm Pearson r, p-value, phân nhóm màu
- **Figure 2** — Zone Entropy Map: bản đồ HCM tô màu theo entropy ($H = -\sum p_k \log_2 p_k$)
- **Figure 3** — Zone-Stratified MAE (Main Result): grouped bar chart
- Tất cả lưu cả `.png` (draft) và `.pdf` (xuất bản) trong `data/results/`

### Nhiệm vụ C.3: Viết Dataset Description
- Viết mô tả chi tiết 3 nguồn dữ liệu (OSRM, TomTom, OSM) cho Section 5.1 của paper
- Bao gồm: số lượng mẫu, khoảng thời gian thu thập, tần suất, 17 node names, 8 zone types
- Giao cho Người D tích hợp vào paper

### Output mong muốn
- [ ] `scripts/zone_stratified_analysis.py`
- [ ] `data/results/fig1a_jsd_heatmap.pdf`
- [ ] `data/results/fig1b_jsd_correlation.pdf`
- [ ] `data/results/fig2_zone_entropy_map.pdf`
- [ ] `data/results/fig3_zone_stratified.pdf` ← **Main Result**
- [ ] File text mô tả dataset (giao cho D)

---

## Người D (Paper Writing & Literature)

### Nhiệm vụ D.1: Viết Section 1 — Introduction (1 trang)
- **Motivation:** Giao thông đô thị có tính Non-IID — phân phối lưu lượng tại các vùng khác nhau rất khác nhau do thành phần chức năng đất đai
- **Gap:** Các mô hình GNN hiện tại (DCRNN, STGCN, Graph WaveNet) chỉ dùng ma trận kề địa lý → bỏ qua ngữ nghĩa vùng chức năng
- **Contribution (3 gạch đầu dòng):**
  1. Đề xuất Zone-Aware AH-GNN với zone embedding và zone-modulated convolution
  2. Chứng minh thực nghiệm rằng zone composition giải thích được tính Non-IID
  3. Đánh giá trên dữ liệu thực HCM với 17 TAZ và 8 loại vùng chức năng

### Nhiệm vụ D.2: Viết Section 2 — Related Work (1.5 trang)
- **2.1 Traffic Flow Prediction:** DCRNN, STGCN, Graph WaveNet, ASTGNN, PDFormer
- **2.2 Heterogeneous GNN:** HAN, HGT — kết nối với việc dùng zone heterogeneity
- **2.3 Urban Zone / Land Use in Traffic:** Urban2Vec, STDN — gap là chưa ai dùng multi-label zone trong GNN adjacency

### Nhiệm vụ D.3: Đọc và tóm tắt 5-7 papers cần cite
Danh sách đề xuất:
1. **DCRNN** (Li et al., 2018) — baseline diffusion conv
2. **STGCN** (Yu et al., 2018) — baseline sandwich architecture
3. **Graph WaveNet** (Wu et al., 2019) — adaptive adjacency (so sánh trực tiếp)
4. **ASTGNN** (Guo et al., 2021) — attention-based
5. **HAN** (Wang et al., 2019) — heterogeneous attention network
6. **Urban2Vec** (Wang et al., 2020) — zone embedding (liên quan trực tiếp)
7. **PDFormer** (Jiang et al., 2023) — state-of-the-art gần nhất

### Output mong muốn
- [ ] Section 1 draft trong `paper/draft_vi.tex` hoặc file riêng
- [ ] Section 2 draft 
- [ ] Bảng tóm tắt related work (model, year, adjacency type, zone-aware?)
- [ ] Danh sách references BibTeX

---

## Lịch trình theo ngày

| Ngày | Người A | Người B | Người C | Người D |
|---|---|---|---|---|
| **T2 21/07** | Chạy 3 baselines | Review baselines code | Zone-Stratified Analysis | Đọc 3 papers đầu |
| **T3 22/07** | Phân tích kết quả baseline | Thiết kế Sinusoidal Enc | Cải thiện JSD Heatmap | Đọc 4 papers còn lại |
| **T4 23/07** | Code TC Zone Emb | Code Sinusoidal Enc | Vẽ Zone Entropy Map | Viết Section 1 draft |
| **T5 24/07** | Tích hợp TC vào pipeline | Test + Unit tests | Vẽ Figure 3 (Main Result) | Viết Section 2 draft |
| **T6 25/07** | Train zone_full_tc | Sửa build_graph.py | Cải thiện Correlation Plot | Hoàn thiện Sections 1&2 |
| **T7 26/07** | So sánh kết quả | (Bonus) Train sine variant | Viết dataset description | Review + BibTeX |
| **CN 27/07** | **Sync nhóm: review tổng thể, plan tuần 3** ||||

---

## Câu hỏi cần thảo luận trong buổi sync

> **Q1:** Kết quả 3 baselines so với zone_full như thế nào? LSTM có tệ hơn rõ rệt không? (Nếu LSTM gần bằng GCN → cần xem lại dữ liệu)

> **Q2:** Time-Conditioned Zone Embedding có cải thiện MAE so với zone_full gốc không? Nếu có → đưa vào paper như đóng góp #2. Nếu không → giữ kiến trúc gốc.

> **Q3:** Figure 3 (Zone-Stratified) có cho thấy improvement tăng theo zone_count không? Nếu không → cần điều chỉnh narrative của paper.

> **Q4:** Nộp venue nào? Cần quyết định trước tuần 3 để định dạng paper đúng template.
