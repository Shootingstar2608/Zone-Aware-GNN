# Data Card: HCM-Real-TomTom (Observed Traffic Flow Segment Snapshot v1-pilot)


---

## 1. Dataset Overview & Metadata

| Thuộc tính siêu dữ liệu | Chi tiết đặc tả kỹ thuật |
|---|---|
| **Dataset ID** | `hcm_real_tomtom_segments_v1` |
| **Snapshot Version** | `v1-pilot` |
| **Schema Version** | `1.1.0` (Quy định tại `docs/hcm_real_canonical_schema.md`, kiểm tra tự động bằng `benchmark/hcm_real_schema.py`) |
| **Modality / Type** | **Dữ liệu lưu lượng giao thông quan sát thực nghiệm (Empirical Observed Traffic Flow Data)** |
| **Đơn vị cung cấp dữ liệu nguồn** | TomTom Traffic Flow API (Phân đoạn Segment Data endpoint, Đơn vị cung cấp: TomTom NV) |
| **Geographic Scope** | 17 hành lang giao thông đã được phê chuẩn tại TP. Hồ Chí Minh |
| **Khoảng thời gian thu thập (Temporal Extent)** | `2026-10-24T12:00Z` đến `2026-11-19T10:00Z` (26 ngày, tương ứng 623 mốc giờ liên tục) |
| **Resolution** | Lưới thời gian tròn giờ (60-minute bucket), đối chiếu múi giờ địa phương Việt Nam (Asia/Ho_Chi_Minh, UTC+07:00) và chuẩn quốc tế (UTC+00:00) |
| **Matrix Size** | $623 \text{ mốc giờ} \times 17 \text{ trạm} =$ **10,591 bản ghi dữ liệu** |
| **Registry SHA-256** | `b8811c5a6da777123d71d1184a4fe5b1b7ab57cdfdf92cf1482d221977684178` |
| **Canonical SHA-256** | `e5432e8f136fe04e8ac89a6ed749a9f483e3c176accb26ec9ccca60d3006ff51` |
| **Reproducibility** | Đã được xác minh tính tất định và khả năng tái lập độc lập trên cả hai môi trường hệ điều hành Linux và Windows |

---

## 2. Stakeholder Bridge

### 2.1. For Technical Stakeholders
- **Data Contract:** Quy định chính xác 23 trường dữ liệu của Schema 1.1.0 với các ràng buộc kiểu dữ liệu, miền giá trị và quy tắc logic cross-field.
- **Tính tự động hóa và toàn vẹn:** Cung cấp pipeline xử lý tự động (`scripts/pipeline/`) kèm 21 bài kiểm thử chất lượng dữ liệu, bảo đảm tính tất định và ngăn chặn biến đổi dữ liệu trái phép.
- **Định lượng suy giảm chuỗi thời gian:** Minh bạch hóa tỷ lệ khuyết thiếu dữ liệu (42.29%) và xác nhận số lượng cửa sổ trượt liên tục 36 giờ bằng 0, làm cơ sở định hướng nghiên cứu xử lý dữ liệu khuyết thiếu.

### 2.2. For Non-Technical Stakeholders
- **Nhận diện đúng mức độ sẵn sàng:** Hiểu rõ bản snapshot v1-pilot là kết quả thử nghiệm sơ bộ nhằm hoàn thiện quy trình công nghệ thu thập dữ liệu, chưa phải là cơ sở dữ liệu hoàn chỉnh để áp dụng ra thực địa.
- **Tuân thủ quy định pháp lý và bản quyền:** Đảm bảo toàn bộ quy trình nghiên cứu tuân thủ nghiêm ngặt thỏa thuận cấp phép thương mại của TomTom NV, không vi phạm phát tán dữ liệu thô.
- **Phân định ranh giới nghiên cứu:** Cung cấp bằng chứng thực địa khách quan để đối chứng với bộ dữ liệu mô phỏng HCM-Sim v2, ngăn chặn các tuyên bố sai lệch trong báo cáo khoa học.

---

## 3. Dataset Purpose & Usage Boundaries

### 3.1. Intended Uses
- **Xác thực quy trình thu thập dữ liệu:** Đánh giá độ ổn định của hệ thống kết nối API, cơ chế tự động xử lý lỗi mạng và quy trình kiểm tra phản hồi.
- **Kiểm định tính tương thích của lược đồ chuẩn hóa:** Kiểm thử quy trình chuẩn hóa dữ liệu từ phản hồi thô dạng JSON sang bảng dữ liệu chuẩn hóa 23 trường.
- **Phân tích khám phá đặc trưng giao thông thực nghiệm:** Khảo sát dải tốc độ thực tế (5–48 km/h), phân tích các điểm nghẽn giao thông và đánh giá sự biến động giữa các khung giờ trong ngày.

### 3.2. Out-of-Scope / Prohibited Uses
- **Chưa thể áp dụng trực tiếp cho giao thức dự báo liên tục 36 bước ($T_{in}=12, T_{out}=24$):** Do chuỗi liên tục dài nhất trong toàn bộ snapshot chỉ đạt 35 giờ (0 cửa sổ trượt 36 giờ liên tục), bộ dữ liệu này chưa thể dùng để huấn luyện hoặc đánh giá các mô hình dự báo dài hạn nếu không tích hợp cơ chế bù dữ liệu khuyết thiếu (Imputation hoặc Masking).
- **Tuyệt đối không sử dụng thay thế hoặc quy đồng bản chất với HCM-Sim v2:** Dữ liệu HCM-Real phản ánh chuỗi đo thực địa có khuyết thiếu; không được trộn lẫn hoặc đánh đồng với dữ liệu tổng hợp hoàn chỉnh của HCM-Sim.

---

## 4. Data Provenance & Human Decisions

### 4.1. Quy trình tuyển chọn và phê duyệt hành lang (Corridor Selection Policy)
- Phương pháp tuyển chọn: Ánh xạ từ các nút giao thông trung tâm sang phân đoạn đường TomTom gần nhất có hỗ trợ đầy đủ dịch vụ dữ liệu lưu lượng (Flow Segment Data).
- Rà soát thủ công: Thực hiện kiểm định đối soát bản đồ OpenStreetMap (OSM) vào ngày 2026-10-19, loại trừ các đoạn đường đi bộ (như phố đi bộ Nguyễn Huệ) và bảo đảm mỗi trạm quan sát mang một mã định danh vị trí OpenLR duy nhất.
- Đóng băng danh mục phê chuẩn: Gồm đúng 17 hành lang trọng điểm lưu trữ tại `data/config/hcm_tomtom_segments.v1.json` (mã băm SHA-256: `b8811c5a...`).

### 4.2. Truy vết nguồn gốc phản hồi thô (Audit Trail)
- Mỗi bản ghi chuẩn hóa đều lưu trữ đường dẫn tương đối trỏ đến tệp phản hồi JSON thô gốc (`raw_response_file`) được lưu trữ tại `data/raw/tomtom_hcm_segments/`.

---

## 5. Dataset Composition & Schema Specification

### 5.1. Quy cách Lược đồ Dữ liệu Chuẩn hóa 23 trường (Canonical Schema 1.1.0)

Tập dữ liệu chuẩn hóa dạng bảng gồm **23 trường dữ liệu** phân định thành 07 nhóm chức năng kỹ thuật:

| Nhóm chức năng | Tên trường | Kiểu dữ liệu | Ý nghĩa & Quy tắc nghiệp vụ |
|---|---|:---:|---|
| **Phiên bản** | `dataset_id` | String | Định danh tập dữ liệu (`hcm_real_tomtom_segments_v1`) |
| | `dataset_version` | String | Định danh phiên bản snapshot (`v1-pilot`) |
| | `schema_version` | String | Phiên bản hợp đồng dữ liệu (`1.1.0`) |
| **Thời gian** | `timestamp_utc` | String | Thời gian chuẩn quốc tế ISO-8601 múi giờ UTC (`+00:00`) |
| | `timestamp_local` | String | Thời gian địa phương Việt Nam ISO-8601 (`+07:00`) |
| **Định danh** | `node_id` | String | Khóa trạm quan sát (thuộc danh mục 17 trạm đã được phê chuẩn) |
| | `segment_id` | String | Định danh phân đoạn đường TomTom tương ứng |
| | `approved_openlr` | String | Chuỗi mã hóa vị trí OpenLR chuẩn hóa |
| **Nguồn dữ liệu** | `source_name` | String | Tên nguồn dữ liệu (`tomtom`) |
| | `source_type` | String | Phân loại API (`observed_traffic_flow_segment_api`) |
| | `raw_response_file` | String | Đường dẫn tương đối đến tệp dữ liệu JSON thô phục vụ kiểm tra truy vết |
| **Lưu lượng động** | `current_speed_kmh` | Float / Null | Tốc độ thực tế quan sát được trên phân đoạn (km/h) |
| | `free_flow_speed_kmh`| Float / Null | Tốc độ dòng tự do theo tiêu chuẩn thiết kế của TomTom (km/h) |
| | `current_travel_time_s`| Float / Null | Thời gian hành trình thực tế qua phân đoạn (giây) |
| | `free_flow_travel_time_s`| Float / Null | Thời gian hành trình theo tốc độ dòng tự do (giây) |
| | `congestion_ratio` | Float / Null | Tỷ số tắc nghẽn giao thông ($T_{\text{current}} / T_{\text{free\_flow}}$) |
| | `confidence` | Float / Null | Hệ số tin cậy của phép đo do TomTom cung cấp $\in [0.0, 1.0]$ |
| | `road_closure` | Bool / Null | Cờ trạng thái đóng đường phát sinh do sự cố hoặc bảo trì hạ tầng |
| **Trạng thái & Chất lượng**| `observation_status` | Enum | `observed`, `road_closed`, `missing`, `invalid_payload` |
| | `is_missing` | Boolean | True nếu dữ liệu bị khuyết thiếu hoặc cấu trúc phản hồi không hợp lệ |
| | `missing_reason` | Enum / String | Nguyên nhân khuyết thiếu: `not_collected`, `network_error`, `api_timeout`,... |
| | `quality_flag` | String | Cờ cảnh báo chất lượng: `low_confidence`, `road_closure`,... |
| **Quản trị Provenance** | `registry_config_sha256`| String | Mã băm SHA-256 của tệp cấu hình registry sử dụng trong đợt thu thập |

---

## 6. Data Quality Audit & Missingness Breakdown

Kết quả kiểm định chính thức dựa trên báo cáo chất lượng dữ liệu của Thành viên B (`tailieu/hcm_real_quality_report.md`):

### 6.1. Phân bố trạng thái quan sát thực tế (Coverage Breakdown)
- **Tổng số bản ghi kỳ vọng:** 10,591 bản ghi ($623 \text{ mốc giờ} \times 17 \text{ trạm}$).
- **Hiện trạng phân bố quan sát:**
  * `observed`: **5,395 bản ghi** — Tỷ lệ độ phủ lưu lượng (Traffic Coverage) đạt **50.94%**.
  * `road_closed`: **717 bản ghi** — Tỷ lệ ghi nhận trạng thái đóng đường chiếm **6.77%**.
  * `missing`: **4,479 bản ghi** — Tỷ lệ khuyết thiếu dữ liệu chiếm **42.29%**.
  * `invalid_payload`: **0 bản ghi** (0.0%).
- **Retrieval Coverage:** 6,112 / 10,591 = **57.71%** (`observed` + `road_closed`).

### 6.2. Phân tích nguyên nhân khuyết thiếu dữ liệu (Missingness Root Causes)
- $94.5\%$ tổng số bản ghi thiếu bắt nguồn từ nguyên nhân **`not_collected`** (4,233 bản ghi): Tiến trình thu thập tạm ngừng hoạt động hoặc tác vụ định kỳ không được kích hoạt tại thời điểm tương ứng.
- $5.5\%$ tổng số bản ghi thiếu phát sinh từ sự cố mạng và hạ tầng kết nối API (246 bản ghi):
  * `network_error`: 204 bản ghi.
  * `api_timeout`: 42 bản ghi (trùng khớp hoàn toàn với nhật ký sự cố tại `collection_errors.jsonl`).

### 6.3. Gián đoạn thời gian và tính liên tục chuỗi thời gian (Continuity Constraints)
- **Mốc thời gian gián đoạn:**
  * 347 mốc giờ thu thập đầy đủ dữ liệu cho toàn bộ 17 trạm quan sát.
  * 15 mốc giờ xuất hiện tình trạng khuyết thiếu dữ liệu cục bộ tại một số trạm.
  * **261 mốc giờ bị khuyết thiếu hoàn toàn dữ liệu tại cả 17 trạm** (chiếm 41.9% tổng số mốc giờ trong chuỗi snapshot).
- **Phân bố theo chu kỳ 24 giờ:** Độ phủ thấp nhất tập trung vào ban đêm và sáng sớm (02:00–09:00 chỉ đạt 31%–46%), độ phủ cao nhất vào buổi tối (21:00–00:00 đạt 76%–78%).
- **Dị thường trạm quan sát:** Hai trạm khu vực trung tâm (`ben_thanh_market` và `district_1`) ghi nhận cờ `road_closure = True` tại 100% số bản ghi thu thập thành công (358 bản ghi `road_closed`, 0 bản ghi `observed`). Do đó, trên thực tế chỉ có **15/17 trạm quan sát** cung cấp chuỗi số liệu tốc độ hợp lệ.
- **Tính liên tục ngặt:** Đoạn chuỗi thời gian liên tục dài nhất trong toàn bộ snapshot chỉ đạt **35 giờ** $\rightarrow$ **Số lượng cửa sổ trượt liên tục 36 giờ ($T_{in}=12, T_{out}=24$) bằng 0**.

---

## 7. Known Limitations, Uncertainties & Terms of Art

### 7.1. Uncertainties & Unknowns
- **Ẩn số về tiến trình thu thập:** Sự gián đoạn trong khung giờ 02:00–09:00 phản ánh tiến trình collector daemon chạy trên môi trường cục bộ chưa có cơ chế tự phục hồi sau khi hệ điều hành tạm nghỉ hoặc gặp sự cố nguồn điện.
- **Ẩn số về trạng thái đóng đường trung tâm:** Trạng thái 100% `road_closed` tại hai trạm Bến Thành và Quận 1 phản ánh phân đoạn đường được TomTom cấu hình thi công hạ tầng đô thị kéo dài, dẫn đến việc các chỉ số tốc độ bị gán giá trị `null` theo đúng lược đồ chuẩn mực.

### 7.2. Giải thích thuật ngữ chuyên môn (Terms of Art Explained)
- **OpenLR (Open Location Referencing):** Tiêu chuẩn mã hóa vị trí độc lập với bản đồ nền, cho phép định danh chính xác phân đoạn đường mà không phụ thuộc vào hệ thống tọa độ của một nhà cung cấp cụ thể.
- **Free-flow Speed:** Tốc độ dòng tự do theo thiết kế kỹ thuật của tuyến đường trong điều kiện không bị cản trở bởi mật độ giao thông.
- **Congestion Ratio:** Tỷ số giữa thời gian hành trình thực tế so với thời gian hành trình theo dòng tự do ($T_{\text{current}} / T_{\text{free\_flow}}$).
- **Hourly Bucket:** Khoảng thời gian gom mẫu chuẩn hóa tròn 60 phút.

---
