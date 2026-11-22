# BÁO CÁO KỸ THUẬT TỔNG THỂ


### PHẦN I. CHI TIẾT 5 NHIỆM VỤ CỐT LÕI

#### 1. NHIỆM VỤ 1: CHỐT SCHEMA CANONICAL HCM-REAL (20 THUỘC TÍNH)
- **Tập tin đã viết:** `scripts/pipeline/schemas.py`, `data/config/canonical_schema_v1.json`
- **Nội dung code đã xây dựng:**
  1. Thiết kế bảng Data Contract chuẩn gồm **20 trường** (vượt chuẩn tối thiểu 16 trường):
     `timestamp_utc`, `timestamp_local`, `node_id`, `segment_id`, `approved_openlr`, `source`, `current_speed_kmh`, `free_flow_speed_kmh`, `current_travel_time_s`, `free_flow_travel_time_s`, `congestion_ratio`, `confidence`, `road_closure`, `is_missing`, `quality_flag`, `dataset_version`, `time_label`, `traffic_delay_s`, `is_peak_hour`, `missing_reason`.
  2. Ban hành định dạng máy đọc được độc lập: `data/config/canonical_schema_v1.json` (chuẩn `JSON Schema Draft-07`).
  3. Lập trình hàm kiểm định Data Contract `validate_canonical_dataframe()`:
     - Kiểm tra bắt buộc không được null trên các trường định danh và thời gian.
     - Kiểm tra miền giá trị vật lý: $v \ge 0, T \ge 0, 0 \le \text{confidence} \le 1$.
     - Kiểm tra kiểu dữ liệu nghiêm ngặt (`datetime64[ns, UTC]`, `datetime64[ns, Asia/Ho_Chi_Minh]`, `float`, `bool`).
     - Ràng buộc **Khóa chính kết hợp duy nhất (Composite Primary Key)**: Bắt buộc cặp `(node_id, timestamp_utc)` là độc nhất, không cho phép trùng lặp.
     - Ràng buộc **Chuẩn hóa đóng đường (`road_closure = True`)**: Khi đường đóng $\implies$ `current_speed_kmh = 0.0`, `current_travel_time_s = NaN`, `congestion_ratio = NaN`, xem như `is_missing = True` với `missing_reason = 'ROAD_CLOSURE'` và `quality_flag = 'ROAD_CLOSURE'`.
     - Ràng buộc **Số hữu hạn cho dòng hợp lệ (`is_missing = False`)**: Bắt buộc `road_closure = False` và 100% các trường số học giao thông (`current_speed_kmh`, `free_flow_speed_kmh`, `current_travel_time_s`, `free_flow_travel_time_s`, `congestion_ratio`, `traffic_delay_s`, `segment_length_m`, `confidence`) phải là số hữu hạn (`np.isfinite()`), tuyệt đối không null, không NaN, không Infinity.
  4. Viết hàm nạp dữ liệu an toàn `load_hcm_real_canonical()`: Tự động khôi phục đúng kiểu dữ liệu khi đọc file plain-text CSV và tự động đối chiếu mã băm SHA-256 từ `manifest.json`.

---

#### 2. NHIỆM VỤ 2: XÂY DỰNG PIPELINE RAW -> CLEANED -> CANONICAL
- **Tập tin đã viết:** `scripts/pipeline/clean_raw.py`, `scripts/pipeline/resampler.py`, `scripts/pipeline/build_hcm_real.py`
- **Nội dung code đã xây dựng:**
  1. **Tầng Cleaned (`clean_raw.py`):**
     - Chuẩn hóa múi giờ: Nhận diện cả UTC naive/aware, ép về chuẩn UTC ISO-8601, đồng thời sinh cột giờ địa phương `Asia/Ho_Chi_Minh` (+07:00).
     - Chuẩn hóa đơn vị: Đổi tự động vận tốc từ m/s sang km/h nếu dữ liệu đầu vào chưa đúng chuẩn.
     - Phát hiện trùng lặp: Quét trùng `(node_id, timestamp_utc)`, gắn cờ `DUPLICATE_FLAGGED`.
     - Lưu bảng trung gian `traffic_cleaned.csv` và `.parquet` ra đĩa để **bảo toàn 100% vết duplicate cho kiểm toán đối soát**.
  2. **Tầng Resampling (`resampler.py`):**
     - Đưa toàn bộ dữ liệu lên lưới thời gian 1 giờ tròn (`00:00:00Z`).
     - Chiến lược lấy mẫu khoa học: Khi một slot giờ có nhiều bản ghi, tự động chọn bản ghi có **`confidence` cao nhất**; nếu bằng nhau lấy bản ghi muộn nhất.
     - Neo thời gian cố định (`--start-utc`, `--end-utc`): Ngăn chặn mất chuỗi thời gian ở rìa nếu API bị gián đoạn mạng những ngày đầu.
     - Phân loại rõ ràng 5 lý do thiếu dữ liệu trong cột `missing_reason`: `API_ERROR`, `NOT_COLLECTED`, `ROAD_CLOSURE`, `LOW_CONFIDENCE`, `NONE`.
  3. **Bộ điều phối (`build_hcm_real.py`):**
     - Thực thi toàn bộ chu trình chỉ bằng **1 dòng lệnh duy nhất**.
     - Hỗ trợ tính bất biến `--immutable` (tự động đóng dấu timestamp) và cơ chế chống ghi đè khi chưa được phép.

---

#### 3. NHIỆM VỤ 3: GẮN STATIC METADATA ĐA NGUỒN & TOPOLOGY
- **Tập tin đã viết:** `scripts/pipeline/enrich_static_metadata.py`, `data/config/hcm_tomtom_segments.v1.json`
- **Nội dung code đã xây dựng:**
  1. Chuẩn hóa toàn bộ 17 trạm quan trắc TP.HCM với mã **TomTom OpenLR Binary Base64 v1.4** và tọa độ chuẩn `WGS84_EPSG4326`.
  2. Ghép nối đa nguồn với tiền tố minh bạch, tuyệt đối không để các nguồn ghi đè lên nhau:
     - Nhãn sử dụng đất từ OpenStreetMap (`osm_commercial`, `osm_residential`,...).
     - Chỉ số định tuyến từ OSRM (`osrm_avg_outgoing_dist_m`, `osrm_avg_outgoing_duration_s`).
     - Thuộc tính vật lý từ Registry (`registry_lat`, `registry_lon`, `registry_corridor`).
  3. Xây dựng **Ma trận kề $A$ ($17 \times 17$)**:
     - Trọng số tỉ lệ nghịch với thời gian di chuyển: $W_{ij} = \frac{1}{\text{duration}(i, j)}$.
     - Triệt tiêu self-loop ($A_{ii} = 0$), chuẩn hóa tổng hàng $\sum_j A_{ij} = 1.0$.
     - Xuất ra cả file `adjacency.pt` (PyTorch Tensor) và `adjacency.npy` (NumPy).
  4. Xây dựng **Ma trận khoảng cách OSRM tabular $D$ ($17 \times 17$) chuẩn không gian mêtric**:
     - Bảo đảm điều kiện $d(i, j) = 0 \iff i = j$.
     - Với các cặp không có lộ trình OSRM: Gán $\infty$ (`np.inf`), **tuyệt đối không gán 0.0m** để tránh nổ trọng số kết nối trong GNN.
     - Fallback hình học: Nếu vắng mặt file OSRM, tự động tính khoảng cách chim bay Haversine từ GPS trong registry nhân với hệ số uốn khúc đô thị 1.35x.

---

#### 4. NHIỆM VỤ 4: XÂY DỰNG BỘ MÁY ĐÁNH GIÁ CHẤT LƯỢNG (QUALITY ENGINE)
- **Tập tin đã viết:** `benchmark/hcm_real_quality.py`, `scripts/evaluate_hcm_real_quality.py`
- **Nội dung code đã xây dựng:**
  1. Lập trình 5 tiêu chí toán học chuẩn mực:
     - **Completeness:** Triệt tiêu Partial Day Bias bằng cách chia theo số khung giờ thực tế được lập lịch (`scheduled_slots`).
     - **Validity:** Kiểm tra tốc độ, thời gian, độ tin cậy nằm trong miền xác định hợp lệ.
     - **Consistency:** Kiểm tra tính nhất quán vật lý giữa vận tốc đo được và tỷ số quãng đường/thời gian hành trình ($|v - \frac{L}{T} \times 3.6|$).
     - **Freshness:** Kiểm tra tính liên tục theo chuỗi thời gian, phát hiện các khoảng gián đoạn $\ge 2$ giờ.
     - **Source Reliability:** Phân bổ độ tin cậy trung bình từ TomTom API.
  2. Tích hợp thuật toán phát hiện **Cảm biến bị treo / Giá trị đông cứng (Flatline Detector)**: Quét rolling window phát hiện trạm giữ nguyên 1 giá trị vận tốc suốt $\ge 4$ giờ liên tiếp (vận tốc $> 0$).
  3. Phân tích phân phối giao thông **Ngày thường vs Cuối tuần (Weekday vs Weekend Profile)** theo 4 khung giờ: `rush_morning`, `normal`, `rush_evening`, `night`.
  4. Tự động xuất xưởng 2 định dạng báo cáo:
     - File bảng biểu chi tiết: `quality_report.md`.
     - File ma trận điểm số 2 chiều Trạm $\times$ Ngày: `quality_matrix_node_day.json`.
  5. Thiết lập cổng kiểm soát chất lượng tự động **CI/CD Quality Gate (`--fail-under 0.85`)**: Tự động trả về `exit code 1` để chặn đứng dữ liệu bẩn trước khi đưa vào huấn luyện mô hình.

---

#### 5. NHIỆM VỤ 5: PHIÊN BẢN HÓA & MANIFEST KHOA HỌC
- **Tập tin đã viết:** `scripts/pipeline/manifest.py`
- **Nội dung code đã xây dựng:**
  1. Tính toán mã băm mật mã học **SHA-256** cho tất cả các tập tin artifacts xuất xưởng (`traffic_cleaned.csv`, `traffic_canonical.csv`, `nodes.csv`, `osrm_distance_matrix.csv`, `adjacency.pt`).
  2. Băm dấu vết nguồn dữ liệu đầu vào (`source_hashes`) gồm raw input, OSM zone và OSRM.
  3. Ghi nhận dấu vân tay môi trường thực thi khoa học **`environment_fingerprint`** (chi tiết phiên bản Python, OS platform, Pandas, NumPy, PyArrow) để đảm bảo tính tái lập 100%.
  4. Ghi nhận số lượng bản ghi duplicate được gắn cờ vào mục `audit_provenance`.

---

### PHẦN II. BẢNG ĐỐI SOÁT 14 FILE ĐÃ LẬP TRÌNH THEO `git status`

Toàn bộ 14 file staged trong Git phản ánh trọn vẹn kiến trúc mà Thành viên B đã dày công xây dựng (+3,682 dòng code):

| STT | Tập tin trong `git status` | Dòng code | Trách nhiệm kỹ thuật trong dự án |
|:---:|---|:---:|---|
| 1 | `.gitignore` | `+7` | Chặn rò rỉ dữ liệu thô, API key và kết quả chạy tạm thời theo quy tắc Mục 8 |
| 2 | `benchmark/hcm_real_quality.py` | `+423` | Engine đo 5 tiêu chí chất lượng, Flatline detector, báo cáo MD và ma trận JSON |
| 3 | `data/config/canonical_schema_v1.json` | `+130` | Bản đặc tả hợp đồng dữ liệu chuẩn JSON Schema Draft-07 (20 trường) |
| 4 | `data/config/hcm_tomtom_segments.v1.json` | `+216` | Cấu hình 17 trạm quan trắc chuẩn hóa TomTom OpenLR Binary Base64 v1.4 |
| 5 | `scripts/dev/generate_mock_raw_tomtom.py` | `+173` | Công cụ sinh dữ liệu mô phỏng 48h cấy sẵn lỗi phục vụ test offline |
| 6 | `scripts/evaluate_hcm_real_quality.py` | `+112` | CLI kiểm định chất lượng tích hợp Quality Gate CI/CD (`--fail-under 0.85`) |
| 7 | `scripts/pipeline/build_hcm_real.py` | `+233` | Bộ điều phối toàn diện Raw $\rightarrow$ Cleaned $\rightarrow$ Canonical $\rightarrow$ Metadata $\rightarrow$ Manifest |
| 8 | `scripts/pipeline/clean_raw.py` | `+134` | Tầng 1: Chuẩn hóa múi giờ UTC/Local, đơn vị, phát hiện duplicate |
| 9 | `scripts/pipeline/resampler.py` | `+258` | Tầng 2: Resampling lưới 1h, chọn confidence cao nhất, phân loại 5 lý do thiếu |
| 10 | `scripts/pipeline/enrich_static_metadata.py` | `+235` | Tầng 3: Ghép OSM/OSRM, ma trận kề $A$, sửa lỗi metric space ma trận khoảng cách $D$ |
| 11 | `scripts/pipeline/manifest.py` | `+174` | Sinh manifest.json, băm SHA-256 artifacts/inputs, ghi environment fingerprint |
| 12 | `scripts/pipeline/schemas.py` | `+366` | Định nghĩa Enums, 20 cột canonical, Composite Primary Key, loader `load_hcm_real_canonical()` |
| 13 | `tests/test_hcm_real_pipeline.py` | `+1,017` | Bộ kiểm thử toàn diện với **32 bài unit test độc lập** (PASS 100%) |
| 14 | `scripts/pipeline/__init__.py` | `+25` | Khởi tạo module package `scripts.pipeline` |

---

### PHẦN III. KẾT QUẢ NGHIỆM THU 4 ĐẦU RA KỸ THUẬT

1. **Dataset HCM-Real canonical & Script build chạy lại được:**
   - Script `build_hcm_real.py` tự động hóa 100%, sẵn sàng chạy lại bất cứ lúc nào bằng 1 lệnh.
2. **Bảng `nodes`, bảng dynamic traffic, bảng/JSON quality report:**
   - Đã sinh đầy đủ cấu trúc: `nodes.csv`, `traffic_canonical.csv`, `traffic_cleaned.csv`, `osrm_distance_matrix.csv`, `adjacency.pt`, `manifest.json`, `quality_report.md`, `quality_matrix_node_day.json`.
3. **Unit test cho schema, timestamp, range, duplicate và missing flag:**
   - Bộ test suite gồm **32/32 bài test tự động PASS 100%** trong 3.40s (cả standalone runner và `pytest`).
4. **Báo cáo chất lượng theo từng node và từng ngày:**
   - Đã lập trình thuật toán ma trận 2 chiều Trạm $\times$ Ngày hoàn chỉnh. Hiện đang kiểm chứng trên 2 ngày mock data và sẵn sàng mở rộng lên 28 ngày ngay khi Thành viên A bàn giao file thật.

---

### PHẦN IV. HƯỚNG DẪN CHI TIẾT CÁC LỆNH CHẠY THỰC THI

#### A. Lệnh Chạy Kiểm Thử Tự Động (Unit Tests)
```powershell
pytest tests/test_hcm_real_pipeline.py -v
```

#### B. Lệnh Chạy Pipeline Build Dataset HCM-Real
```powershell
python scripts/pipeline/build_hcm_real.py --raw-path data/raw/mock_tomtom_raw.csv --out-dir data/processed/hcm_real_v1
```

#### C. Lệnh Chạy Đánh Giá Chất Lượng & Quality Gate CI/CD
```powershell
# Chạy với kiểm soát khắt khe từng tiêu chí thành phần:
# Completeness >= 0.80, Consistency >= 0.85, Điểm tổng >= 0.85
python scripts/evaluate_hcm_real_quality.py `
  --canonical-path data/processed/hcm_real_v1/traffic_canonical.csv `
  --out-dir data/processed/hcm_real_v1 `
  --fail-under 0.85 `
  --fail-under-completeness 0.80 `
  --fail-under-consistency 0.85
```