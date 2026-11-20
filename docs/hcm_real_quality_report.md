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

## 3. Missing

### 3.1. Theo nguyên nhân ghi trong dữ liệu

| missing_reason | Số row |
|---|---|
| not_collected | 4,233 |
| network_error | 204 |
| api_timeout | 42 |

- `network_error` + `api_timeout` = 246, khớp số lần fail trong `collection_errors.jsonl`.
- `not_collected`: không có bản ghi nào trong raw lẫn error log cho (giờ, node) đó.
- Theo snapshot (một mốc giờ): 347 đủ 17 node, 15 thiếu một phần, 261 trống hoàn toàn.

### 3.2. Các đoạn thiếu dữ liệu hoàn toàn (cả 17 node, giờ Việt Nam)

| Từ (giờ VN) | Đến (giờ VN) | Số giờ |
|---|---|---|
| 2026-10-24 21:00 | 2026-10-27 10:00 | 62 |
| 2026-10-29 01:00 | 2026-10-29 07:00 | 7 |
| 2026-10-30 03:00 | 2026-10-30 08:00 | 6 |
| 2026-10-31 02:00 | 2026-10-31 07:00 | 6 |
| 2026-11-01 02:00 | 2026-11-01 07:00 | 6 |
| 2026-11-02 03:00 | 2026-11-02 07:00 | 5 |
| 2026-11-02 11:00 | 2026-11-02 15:00 | 5 |
| 2026-11-03 02:00 | 2026-11-03 17:00 | 16 |
| 2026-11-04 02:00 | 2026-11-04 06:00 | 5 |
| 2026-11-05 02:00 | 2026-11-05 07:00 | 6 |
| 2026-11-06 03:00 | 2026-11-06 09:00 | 7 |
| 2026-11-07 02:00 | 2026-11-07 07:00 | 6 |
| 2026-11-07 19:00 | 2026-11-07 20:00 | 2 |
| 2026-11-08 08:00 | 2026-11-08 19:00 | 12 |
| 2026-11-09 03:00 | 2026-11-09 09:00 | 7 |
| 2026-11-10 17:00 | 2026-11-10 17:00 | 1 |
| 2026-11-10 19:00 | 2026-11-10 19:00 | 1 |
| 2026-11-11 09:00 | 2026-11-11 11:00 | 3 |
| 2026-11-11 20:00 | 2026-11-11 20:00 | 1 |
| 2026-11-12 02:00 | 2026-11-12 16:00 | 15 |
| 2026-11-13 02:00 | 2026-11-13 16:00 | 15 |
| 2026-11-13 18:00 | 2026-11-13 18:00 | 1 |
| 2026-11-14 09:00 | 2026-11-14 20:00 | 12 |
| 2026-11-16 08:00 | 2026-11-16 08:00 | 1 |
| 2026-11-16 23:00 | 2026-11-17 01:00 | 3 |
| 2026-11-17 07:00 | 2026-11-18 12:00 | 30 |
| 2026-11-18 15:00 | 2026-11-19 10:00 | 20 |

Tổng: 261 giờ trống hoàn toàn trên 623 mốc giờ.
Mỗi thời điểm là giờ bắt đầu của mốc giờ, tính cả hai đầu.

### 3.3. Các giờ thiếu một phần (một số node thiếu)

- 2026-10-27 22:00 → 2026-10-27 23:00 (2 giờ)
- 2026-10-28 20:00 → 2026-10-28 22:00 (3 giờ)
- 2026-11-02 22:00 → 2026-11-02 22:00 (1 giờ)
- 2026-11-03 21:00 → 2026-11-03 22:00 (2 giờ)
- 2026-11-04 21:00 → 2026-11-04 22:00 (2 giờ)
- 2026-11-09 20:00 → 2026-11-09 22:00 (3 giờ)
- 2026-11-12 21:00 → 2026-11-12 21:00 (1 giờ)
- 2026-11-12 23:00 → 2026-11-12 23:00 (1 giờ)

Tổng: 15 giờ.

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

## 5. Tính liên tục của chuỗi thời gian

Một giờ được tính là "đủ" khi cả 15 node có số liệu tốc độ (loại hai node `ben_thanh_market` và `district_1` vì luôn ở trạng thái road_closed) đều ở trạng thái observed.
Đây là định nghĩa chặt: mọi giờ trong cửa sổ đều đủ 15 node, không nội suy và không dùng mask.

| Chỉ số | Giá trị |
|---|---|
| Số giờ đủ 15 node | 349 / 623 (56.0%) |
| Số đoạn liên tục | 36 |
| Đoạn liên tục dài nhất | 35 giờ |
| 5 đoạn dài nhất (giờ) | 35, 21, 19, 19, 19 |
| Số cửa sổ trượt 36 giờ liên tục (T_in = 12, T_out = 24) | **0** |

Với cấu hình cửa sổ 12 giờ đầu vào và 24 giờ dự báo, bản pilot này không có cửa sổ nào đủ dữ liệu liên tục.

## 6. Coverage theo giờ trong ngày (giờ Việt Nam)

Tỷ lệ cặp (giờ, node) có bản ghi (observed hoặc road_closed), trên toàn bộ 17 node.

| Giờ | Coverage | Giờ | Coverage |
|---|---|---|---|
| 00h | 77% | 12h | 62% |
| 01h | 73% | 13h | 65% |
| 02h | 46% | 14h | 65% |
| 03h | 31% | 15h | 62% |
| 04h | 31% | 16h | 65% |
| 05h | 31% | 17h | 69% |
| 06h | 31% | 18h | 72% |
| 07h | 31% | 19h | 69% |
| 08h | 46% | 20h | 73% |
| 09h | 46% | 21h | 78% |
| 10h | 54% | 22h | 76% |
| 11h | 58% | 23h | 76% |

Coverage thấp nhất ở khoảng 02h–09h (thấp nhất 31% ở 03h–07h) và cao nhất ở khoảng 21h–00h.
Trong mục 3.2, từ 29/10 đến 9/11 hầu như mỗi đêm có một đoạn trống 5–7 giờ trong khoảng 01h–09h.

## 7. Kiểm tra chất lượng đã chạy

- Không có cặp (timestamp, node) trùng trong canonical. Raw có 34 cặp trùng, đã gộp và gắn cờ `duplicate_observation`.
- Mọi mốc thời gian tròn giờ.
- `is_missing` nhất quán với `observation_status` ở mọi row.
- Speed observed: 5 → 48 km/h (trung bình 30.2). Không row nào có tốc độ hiện tại vượt free-flow.
- Confidence: nhỏ nhất 0.536, trung bình 0.990; không row nào dưới ngưỡng 0.5.
- 3 file test HCM-Real: 21 passed.

## 8. Phát hiện cần lưu ý

1. **Coverage chỉ ~51%**, chủ yếu là các giờ không có bản ghi thu thập (not_collected). Log lỗi chỉ ghi nhận 246 lần fail.
   Missing gần như đều nhau giữa các node (261–266 row/node), tức các giờ thiếu là chung cho cả 17 node.
   Chi tiết xem mục 3.
2. **Không có cửa sổ 36 giờ liên tục** (đoạn dài nhất 35 giờ), nên cấu hình T_in = 12, T_out = 24 chưa áp dụng được trên bản pilot này (mục 5).
3. **Coverage lệch theo giờ trong ngày**: thấp nhất ở 02h–09h (31–46%), nên giờ cao điểm sáng đang bị thiếu nhiều hơn các giờ khác (mục 6).
4. **`ben_thanh_market` và `district_1` có road_closure = True ở mọi row quan sát được**, nên canonical không có row observed nào cho hai node này. Trong raw, hai node này vẫn có tốc độ khác 0 (ví dụ 23 và 16 km/h) khi cờ đóng đường bật. Chưa rõ nguyên nhân; người phụ trách thu thập đang kiểm tra lại segment/OpenLR. Registry không bị thay đổi trong bản build này.
5. Chỉ **15/17 node** có số liệu tốc độ dùng được.
6. `suoi_tien` có 1 row road_closed đơn lẻ.
7. Cờ zone OSM (bán kính 800 m) bật gần hết 8 loại ở nhiều node nên khả năng phân biệt node yếu.

## 9. Giới hạn

- Đây là bản pilot (khoảng 26 ngày, chưa đủ mục tiêu 28 ngày liên tục).