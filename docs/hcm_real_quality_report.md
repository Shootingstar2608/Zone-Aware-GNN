# HCM-Real canonical: báo cáo chất lượng dữ liệu (v1-pilot)

Phạm vi: bản canonical dựng từ raw TomTom của 17 segment đã approved.
Báo cáo chỉ chứa số liệu tổng hợp, không chứa dữ liệu TomTom gốc.

## 1. Thông tin build

| Mục | Giá trị |
|---|---|
| Dataset / version | `hcm_real_tomtom_segments_v1` / `v1-pilot` |
| Schema version | 1.1.0 |
| Bucket | 60 phút |
| Lưới thời gian | 2026-10-24T12:00Z → 2026-11-19T10:00Z (623 mốc giờ) |
| Số node | 17 |
| Lưới kỳ vọng | 623 × 17 = 10,591 row |
| Registry sha256 | `b8811c5a6da777123d71d1184a4fe5b1b7ab57cdfdf92cf1482d221977684178` |
| Canonical sha256 | `e5432e8f136fe04e8ac89a6ed749a9f483e3c176accb26ec9ccca60d3006ff51` |

Tái lập: build lại trên hai máy khác nhau (Linux và Windows) cho cùng canonical sha256.

## 2. Số row theo trạng thái

| observation_status | Số row | Tỷ lệ |
|---|---|---|
| observed | 5,395 | 50.94% |
| road_closed | 717 | 6.77% |
| missing | 4,479 | 42.29% |
| invalid_payload | 0 | 0% |

- Retrieval coverage (observed + road_closed): 6,112 / 10,591 = 57.71%.
- Traffic coverage (chỉ observed): 50.94%.

## 3. Missing theo nguyên nhân

| missing_reason | Số row |
|---|---|
| not_collected | 4,233 |
| network_error | 204 |
| api_timeout | 42 |

- `network_error` + `api_timeout` = 246, khớp số lần fail trong `collection_errors.jsonl`.
- `not_collected`: không có bản ghi nào trong raw lẫn error log cho (giờ, node) đó.
- Theo snapshot (một mốc giờ): 347 đủ 17 node, 15 thiếu một phần, 261 trống hoàn toàn.
- Các khoảng trống dài nhất: khoảng 62 giờ đầu kỳ thu (24/10 → 27/10) và khoảng 30 giờ rồi 20 giờ vào 16/11 → 19/11.

## 4. Theo node

| node_id | observed | road_closed | missing |
|---|---|---|---|
| ben_thanh_market | 0 | 358 | 265 |
| district_1 | 0 | 358 | 265 |
| binh_thanh | 359 | 0 | 264 |
| district_3 | 357 | 0 | 266 |
| district_5 | 358 | 0 | 265 |
| eastern_bus_station | 360 | 0 | 263 |
| hang_xanh | 359 | 0 | 264 |
| high_tech_park | 360 | 0 | 263 |
| landmark_81 | 360 | 0 | 263 |
| linh_trung | 360 | 0 | 263 |
| pham_van_dong | 361 | 0 | 262 |
| saigon_bridge | 359 | 0 | 264 |
| suoi_tien | 359 | 1 | 263 |
| tan_son_nhat_airport | 360 | 0 | 263 |
| thu_duc | 359 | 0 | 264 |
| thu_thiem_tunnel | 362 | 0 | 261 |
| vnu_hcm | 362 | 0 | 261 |

## 5. Kiểm tra chất lượng đã chạy

- Không có cặp (timestamp, node) trùng trong canonical. Raw có 34 cặp trùng, đã gộp và gắn cờ `duplicate_observation`.
- Mọi mốc thời gian tròn giờ.
- `is_missing` nhất quán với `observation_status` ở mọi row.
- Speed observed: 5 → 48 km/h (trung bình 30.2). Không row nào có tốc độ hiện tại vượt free-flow.
- Confidence: nhỏ nhất 0.536, trung bình 0.990; không row nào dưới ngưỡng 0.5.
- 3 file test HCM-Real: 21 passed.

## 6. Phát hiện cần lưu ý

1. **Coverage chỉ ~51%**, chủ yếu do không có bản ghi thu thập (not_collected), không phải do lỗi API. Missing gần như đều nhau giữa các node (261–266 row/node), tức các giờ thiếu là chung cho cả 17 node.
2. **`ben_thanh_market` và `district_1` có road_closure = True ở mọi row quan sát được**, nên canonical không có row observed nào cho hai node này. Trong raw, hai node này vẫn có tốc độ khác 0 (ví dụ 23 và 16 km/h) khi cờ đóng đường bật. Chưa rõ nguyên nhân; cần kiểm tra segment/OpenLR đã approved. Registry không bị thay đổi trong bản build này.
3. Chỉ **15/17 node** có số liệu tốc độ dùng được.
4. `suoi_tien` có 1 row road_closed đơn lẻ.
5. Cờ zone OSM (bán kính 800 m) bật gần hết 8 loại ở nhiều node nên khả năng phân biệt node yếu.

## 7. Giới hạn

- Đây là bản pilot (khoảng 26 ngày, chưa đủ mục tiêu 28 ngày liên tục).
- Dữ liệu canonical và raw TomTom không đưa vào Git vì chưa duyệt điều khoản phân phối lại.