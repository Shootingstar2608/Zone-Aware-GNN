# Data Card — Zone-Aware GNN Traffic Benchmark

**Phiên bản:** 1.0
**Người soạn:** [Người 4 — Evaluation / Docs]
**Ngày cập nhật:** [điền ngày cập nhật lần cuối]
**Liên hệ:** [email/kênh liên lạc của nhóm]

> Tài liệu này mô tả nguồn gốc, cách thu thập, giấy phép sử dụng, rủi ro
> quyền riêng tư, và hướng dẫn tái lập cho bộ dữ liệu dùng trong benchmark
> Zone-Aware GNN (dự đoán tắc nghẽn giao thông theo vùng). Tham khảo
> checklist trước khi công bố/chia sẻ dữ liệu ra ngoài nhóm nghiên cứu.

---

## 1. Tổng quan

Dữ liệu benchmark được tổng hợp từ **3 nguồn độc lập**, mỗi nguồn phục vụ
một vai trò khác nhau trong đồ thị giao thông theo vùng (zone-aware graph):

| Nguồn | Vai trò | Loại dữ liệu |
|---|---|---|
| **OSM** (OpenStreetMap) | Cấu trúc mạng lưới đường + phân loại đất/vùng (zone labels) | Bản đồ vector, POI, land-use tags |
| **OSRM** (Open Source Routing Machine) | Khoảng cách/thời gian di chuyển tham chiếu (free-flow) giữa các node | Ma trận khoảng cách, thời gian tuyến đường |
| **TomTom** (Traffic API) | Chỉ số giao thông thời gian thực (congestion, delay, travel time) | Time-series traffic metrics |

Ba nguồn được ghép nối qua `node_id`/toạ độ địa lý để tạo ra bản ghi
chuẩn hoá theo schema chung (xem `schema/models.py`).

### Thống kê dataset thực tế (nguồn: `data/processed/meta.json`)

| Ký hiệu | Giá trị | Ý nghĩa |
|---|---|---|
| `N` | 17 | Số node (địa điểm) trong đồ thị |
| `K` | 8 | Số chiều zone label (khớp đúng 8 loại đã thiết kế) |
| `F` | 4 | Số feature traffic mỗi node mỗi timestep |
| `T_in` | 12 | Số bước thời gian đầu vào (lịch sử) |
| `T_out` | 24 | Số bước thời gian dự đoán (horizon) |
| `S` | 637 | Số cửa sổ trượt (sliding window) sinh ra từ chuỗi thời gian |

**17 node thực tế** (đều thuộc khu vực TP.HCM): Ben Thanh Market, Binh Thanh,
District 1, District 3, District 5, Eastern Bus Station, Hang Xanh, High Tech
Park, Landmark 81, Linh Trung, Pham Van Dong, Saigon Bridge, Suoi Tien, Tan
Son Nhat Airport, Thu Duc, Thu Thiem Tunnel, VNU HCM.

**4 feature traffic thực tế** (`feature_names` trong `meta.json`) —
thay cho mô tả chung chung "congestion/delay/speed/free-flow" trước đây:

| Feature | Nguồn tính | Ý nghĩa |
|---|---|---|
| `congestion_ratio` | TomTom | Mức tắc nghẽn tương đối (0 = thông thoáng) |
| `traffic_delay_s` | TomTom | Độ trễ do tắc nghẽn, đơn vị giây |
| `travel_time_s` | TomTom | Thời gian di chuyển thực tế quan sát, đơn vị giây |
| `ff_ratio` | TomTom ÷ OSRM | Tỷ lệ `travel_time_s` (TomTom, thực tế) trên thời gian free-flow lý thuyết (OSRM) — đây chính là điểm **kết hợp trực tiếp 2 nguồn** OSRM + TomTom thành 1 feature duy nhất |

✅ **Đã xác nhận:** `meta.json` là nguồn số liệu chính thức cho dataset hiện
tại (`T_out=24`, `S=637`). Số liệu khác (`T_in=12, T_out=3, S=658`) từng xuất
hiện trong `MULTISEED_FINDINGS.md` thuộc về một cấu hình/run khác thời điểm
trước đó, không dùng để đối chiếu — mọi tài liệu kể từ đây lấy `meta.json`
làm chuẩn.

---

## 2. Chi tiết từng nguồn dữ liệu

### 2.1. OpenStreetMap (OSM)

- **Nội dung sử dụng:** topology mạng lưới đường (nodes, edges), tag phân
  loại đất (`landuse`, `amenity`, `building`) dùng để suy ra 8 chiều
  `zone_labels` (residential, commercial, industrial, school, university,
  hospital, transport, park).
- **Cách thu thập:** trích xuất qua Overpass API / file `.pbf` (Planet
  OSM extract theo khu vực), xử lý bằng `collect_zones.py`.
- **Tần suất cập nhật:** OSM là dữ liệu cộng đồng, cập nhật liên tục —
  cần ghi rõ **timestamp snapshot** đã dùng để đảm bảo tái lập được (xem
  mục 5).
- **License:** [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/).
  Yêu cầu chính: **Attribution** (ghi công © OpenStreetMap contributors)
  và **Share-Alike** — nếu công bố database phái sinh có chứa dữ liệu OSM
  gốc (không chỉ kết quả tổng hợp/thống kê), bản phái sinh đó cũng phải
  mở theo ODbL.
- **Lưu ý:** ⚠️ điều khoản ODbL có thể thay đổi — kiểm tra lại văn bản
  chính thức tại odbl link phía trên trước khi công bố dữ liệu phái sinh
  ra ngoài phạm vi nghiên cứu nội bộ.

### 2.2. OSRM (Open Source Routing Machine)

- **Nội dung sử dụng:** thời gian di chuyển "free-flow" (không tắc, tính
  theo tốc độ giới hạn đường) giữa các cặp node — dùng làm mẫu số để
  tính feature `ff_ratio` (mục 1): `ff_ratio = travel_time_s (TomTom,
  thực tế) / free-flow time (OSRM, lý thuyết)`. Đây KHÔNG chỉ là baseline
  tham khảo mà là **input trực tiếp của 1 trong 4 feature train model**.
- **Cách thu thập:** self-host OSRM engine, route dựa trên **cùng bản
  OSM extract** đã dùng ở mục 2.1 (đảm bảo nhất quán giữa 2 nguồn).
- **License:** bản thân OSRM (phần mềm engine) dùng giấy phép
  [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) — được tự
  do dùng, sửa, phân phối lại. **Tuy nhiên**, dữ liệu bản đồ mà OSRM xử
  lý vẫn là OSM → mọi output tính toán từ OSRM (route, khoảng cách) vẫn
  kế thừa nghĩa vụ ODbL của OSM gốc (mục 2.1), không phải giấy phép riêng
  của OSRM.

### 2.3. TomTom Traffic API

- **Nội dung sử dụng:** 3/4 feature train model lấy trực tiếp từ TomTom —
  `congestion_ratio`, `traffic_delay_s`, `travel_time_s` (xem bảng feature
  ở mục 1) — thu thập tại từng trong 17 node theo thời gian thực.
- **Cách thu thập:** gọi qua TomTom Developer API (cần API key), thu
  thập định kỳ bởi `tomtom_collector.py`.
- **License:** dữ liệu TomTom là **dữ liệu thương mại/độc quyền**, sử
  dụng theo [TomTom Developer Terms of Use](https://developer.tomtom.com/terms-and-conditions)
  — **KHÔNG** phải open data. Các điểm cần lưu ý:
  - Yêu cầu tài khoản + API key cá nhân/tổ chức, có giới hạn số lượng
    request (rate limit) theo gói dùng (free tier vs trả phí).
  - **Không được redistribute** dữ liệu thô ra ngoài phạm vi ứng dụng/
    nghiên cứu được cấp phép — nếu công bố dataset công khai (paper,
    Hugging Face, GitHub), cần dùng dữ liệu đã qua xử lý/tổng hợp
    (aggregated/derived) và **trích dẫn TomTom rõ ràng**, không đính kèm
    raw response từ API.
  - ⚠️ Điều khoản dịch vụ (ToS) của TomTom **có thể thay đổi theo thời
    gian** — bắt buộc kiểm tra lại bản ToS hiện hành tại thời điểm công
    bố dữ liệu, không dựa vào tóm tắt trong tài liệu này để đưa ra quyết
    định pháp lý.

---

## 3. Quyền riêng tư (Privacy)

- Dữ liệu **không chứa thông tin cá nhân** (PII) trực tiếp — không có
  biển số xe, ID thiết bị, hay dữ liệu định vị cá nhân. Traffic metrics
  từ TomTom là dữ liệu **tổng hợp theo đoạn đường** (segment-level
  aggregate), không truy vết được cá nhân cụ thể.
- Rủi ro gián tiếp cần lưu ý: kết hợp `zone_labels` (vd: gần trường học,
  bệnh viện) với dữ liệu traffic tần suất cao theo thời gian **có thể**
  suy luận ra pattern hoạt động của khu vực nhạy cảm (giờ tan trường, giờ
  cao điểm bệnh viện) — không phải rủi ro định danh cá nhân, nhưng nên
  cân nhắc khi công bố dữ liệu ở độ phân giải thời gian rất mịn (vd:
  theo phút) cho các zone loại `school`/`hospital`.
- Khuyến nghị: khi公 bố dataset ra ngoài, cân nhắc làm tròn timestamp
  (vd: theo giờ thay vì theo phút) cho các node thuộc zone nhạy cảm.

---

## 4. Tính minh bạch (Transparency)

- **Pipeline xử lý** đầy đủ (từ raw → processed → partition) đã được
  script hoá và version-controlled trong repo (`build_graph.py`,
  `collect_zones.py`, `tomtom_collector.py`, `benchmark/partition_gen.py`)
  — không có bước xử lý thủ công/không ghi lại.
- **Chất lượng dữ liệu** được đo và báo cáo tự động qua
  `benchmark/data_quality.py` → `data/results/data_quality_report.md`
  (5 chỉ số: Completeness, Validity, Consistency, Freshness, Source
  Reliability) — tham khảo báo cáo này để biết giới hạn thực tế của dữ
  liệu tại thời điểm thu thập.
- **Giới hạn đã biết** (một phần đã xác nhận từ `MULTISEED_FINDINGS.md`,
  phần còn lại vẫn là suy đoán hợp lý dựa trên đặc điểm chung của nguồn —
  cần Người 3 xác nhận lại bằng `data_quality_report.md` khi có):
  - OSM có độ phủ/độ chính xác không đồng đều giữa các khu vực (khu vực
    ít người đóng góp có thể thiếu tag `landuse` → `zone_labels` kém
    tin cậy hơn). *(suy đoán, chưa đo thật)*
  - OSRM free-flow time là ước lượng lý thuyết (tốc độ giới hạn đường),
    không phải quan sát thực tế. *(suy đoán, chưa đo thật)*
  - **Đã xác nhận thật:** kết quả single-seed (`all_results.csv`) là
    NHIỄU — độ lệch chuẩn giữa các seed (0.004–0.012 MAE) lớn hơn khoảng
    cách giữa các model đang so sánh (0.002–0.005 MAE). Bất kỳ ai dùng
    dataset này để benchmark **bắt buộc chạy nhiều seed** (≥10, xem
    `scripts/run_multi_seed.py`) rồi kiểm định thống kê (Holm-corrected
    paired t-test), không được kết luận từ 1 lần chạy.

---

## 5. ⚠️ Vấn đề nghiêm trọng đã phát hiện: Rò rỉ thời gian (Temporal Leakage)

**Nguồn:** `MULTISEED_FINDINGS.md`, mục 3. Đây là phát hiện quan trọng nhất
cần bất kỳ ai dùng lại dataset/pipeline này phải biết trước khi đánh giá model.

`train.py` (bản gốc) chia train/test bằng `random_split` áp lên các cửa sổ
trượt (sliding window, `S=637` cửa sổ từ chuỗi thời gian liên tục). Vì các
cửa sổ liền kề chồng lấn tới `T_in-1=11` bước trên `T_in=12` bước, chia ngẫu
nhiên khiến một cửa sổ test có thể gần như trùng hoàn toàn với 1 cửa sổ train
đứng ngay cạnh nó theo thời gian → **model gần như đã "nhìn thấy" đáp án**,
kết quả (MAE ≈ 0.08, MAPE ≈ 6%) bị thổi phồng, không phản ánh khả năng dự
đoán thật ra tương lai.

**Hệ quả cho bất kỳ ai dùng lại dữ liệu/pipeline này:**
- **KHÔNG dùng `random_split`** để chia train/test trên dữ liệu chuỗi thời
  gian này. Dùng chia **theo thứ tự thời gian** (chronological split, ví dụ
  70/10/20 theo mốc thời gian) — pipeline đã hỗ trợ sẵn qua
  `python scripts/run_multi_seed.py --split-modes chrono`.
- Nếu so sánh với số liệu benchmark cũ (`all_results.csv`, split ngẫu
  nhiên) → **không so sánh trực tiếp được** với kết quả chạy theo chrono
  split, vì 2 protocol đo những thứ khác nhau (memorization vs forecasting
  thật).
- Đây là lý do các paper cùng lĩnh vực (DCRNN, STGCN, Graph WaveNet) đều
  dùng chia theo thời gian — bất kỳ dataset traffic forecasting nào theo
  dạng sliding-window đều có rủi ro leakage tương tự nếu không cẩn thận
  cách chia.

---

## 6. Hướng dẫn tái lập (Reproducibility)

Để tái tạo lại đúng bộ dữ liệu đã dùng trong benchmark:

1. **Ghi lại snapshot OSM** đã dùng: lưu ngày tải `.pbf` hoặc hash của
   Overpass query, vì OSM thay đổi liên tục. → điền vào
   `data/raw/osm_snapshot_meta.json` (ngày tải, vùng địa lý, nguồn tải).
2. **Cố định seed** cho mọi bước có yếu tố ngẫu nhiên (lấy mẫu, chia
   partition Non-IID, khởi tạo model) — xem `benchmark/partition_gen.py`,
   seed mặc định = 42, ghi log seed thực tế đã dùng vào
   `data/partitions/partitions_meta.json`.
3. **Lưu phiên bản API**: ghi rõ TomTom API version + thời điểm gọi
   (timestamp range) trong metadata thu thập, vì response schema của
   TomTom có thể thay đổi giữa các version API.
4. **Container hoá môi trường xử lý** (OSRM self-host + Python
   pipeline) qua `requirements.txt` / Docker (nếu có) để tránh sai lệch
   do phiên bản thư viện khác nhau.
5. Toàn bộ config trên nên được đóng gói cùng dataset khi chia sẻ, để
   người dùng lại có thể tái tạo pipeline mà không cần đoán.

---

## 7. Checklist trước khi công bố dữ liệu/kết quả ra ngoài nhóm

- [ ] Đã trích dẫn OSM (© OpenStreetMap contributors, ODbL) đầy đủ
- [ ] Không đính kèm raw response gốc từ TomTom API (chỉ dữ liệu đã xử lý)
- [ ] Đã kiểm tra lại ToS hiện hành của TomTom tại thời điểm công bố
- [ ] Đã chạy `data_quality.py` và đính kèm báo cáo chất lượng
- [ ] Đã ghi lại snapshot OSM + seed + API version để đảm bảo tái lập
- [ ] Đã cân nhắc làm tròn thời gian cho các zone nhạy cảm (mục 3)
- [ ] **Đã dùng chrono split (`--split-modes chrono`), KHÔNG dùng
      `random_split`, khi báo cáo kết quả benchmark** (mục 5)
- [ ] **Đã chạy ≥10 seed + kiểm định thống kê**, không kết luận từ 1 lần
      chạy (mục 4)

---

## 8. Ghi chú phiên bản

| Ngày | Người sửa | Nội dung |
|---|---|---|
| [ngày] | [tên] | Khởi tạo Data Card |
| [ngày] | [tên] | Cập nhật số liệu thật (N/K/F/T_in/T_out/S, tên node, tên feature) từ `meta.json`; thêm mục cảnh báo temporal leakage từ `MULTISEED_FINDINGS.md` |
| [ngày] | [tên] | Xác nhận `meta.json` (T_out=24, S=637) là số liệu chính thức, gỡ ghi chú nghi ngờ |