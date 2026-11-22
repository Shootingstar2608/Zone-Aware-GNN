# HCM-Real-TomTom v1: schema canonical

## Mục đích

Đây là hợp đồng dữ liệu chung cho HCM-Real-TomTom. Một record biểu diễn một
segment tại một thời điểm: quan sát traffic hợp lệ, trạng thái đường đóng, hoặc
một quan sát thiếu được ghi rõ nguyên nhân. Schema này không chứa API key và không yêu cầu
công bố raw response TomTom.

Raw collector ghi dữ liệu vào `data/raw/tomtom_hcm_segments/`. Pipeline kế tiếp
sẽ tạo canonical dataset theo schema này ở kho nội bộ; raw và canonical data
không tự động được đưa lên GitHub.

Schema máy đọc được nằm tại
`data/schema/hcm_real_tomtom_v1.schema.json`; validator chuẩn Python nằm tại
`benchmark/hcm_real_schema.py`.

## Các nhóm trường

| Nhóm | Trường | Ý nghĩa |
|---|---|---|
| Phiên bản | `dataset_id`, `dataset_version`, `schema_version` | Xác định dataset và contract nào đã sinh record. |
| Thời gian | `timestamp_utc`, `timestamp_local` | Cùng một thời điểm ở UTC và Asia/Ho\_Chi\_Minh; không dùng timestamp không timezone. |
| Định danh | `node_id`, `segment_id`, `approved_openlr` | Liên kết record với registry 17 segment đã được duyệt. `approved_openlr` tránh mơ hồ do API trả đoạn đường gần tọa độ. |
| Nguồn | `source_name`, `source_type`, `raw_response_file` | Nêu rõ nguồn TomTom và vị trí artifact raw local để audit. |
| Traffic động | `current_speed_kmh`, `free_flow_speed_kmh`, `current_travel_time_s`, `free_flow_travel_time_s`, `congestion_ratio`, `confidence`, `road_closure` | Giá trị do TomTom trả về hoặc được tính trực tiếp từ các trường trả về. |
| Trạng thái & chất lượng | `observation_status`, `is_missing`, `missing_reason`, `quality_flag` | Tách đường đóng đã biết khỏi lỗi thu thập. Nhiều cờ trong `quality_flag` dùng dấu `|`. |
| Provenance | `registry_config_sha256` | Hash config registry dùng khi thu; phát hiện record bị trộn giữa các registry khác nhau. |

## Quy tắc bắt buộc

1. `timestamp_utc` phải mang offset `+00:00`; `timestamp_local` phải mang
   `+07:00`; hai trường phải cùng một thời điểm.
2. `observation_status=observed` có `is_missing=false`, `road_closure=false`
   và **không được phép có `null`, `NaN`
   hay `Infinity` ở bất kỳ trường traffic số nào**: `current_speed_kmh`,
   `free_flow_speed_kmh`, `current_travel_time_s`,
   `free_flow_travel_time_s`, `congestion_ratio`, `confidence`. Mọi giá trị
   phải hữu hạn và dương, trừ `confidence` thuộc `[0,1]`; `road_closure` phải
   là boolean; `missing_reason` phải rỗng. Ngoài ra,
   `congestion_ratio = current_travel_time_s / free_flow_travel_time_s`.
3. `observation_status=road_closed` có `is_missing=false`,
   `road_closure=true`; tất cả metric traffic là `null`. `null` ở đây nghĩa là
   *không áp dụng* cho đường đã đóng, không có nghĩa speed bằng 0 hoặc dữ liệu
   bị mất. `quality_flag` phải nêu `road_closure|not_applicable_traffic_metrics`.
4. `observation_status=missing` hoặc `invalid_payload` có `is_missing=true`;
   mọi trường traffic số là `null` và phải
   ghi `missing_reason`; `road_closure` cũng phải là `null`. Vocabulary v1.1 gồm `api_timeout`, `api_http_error`,
   `network_error`, `invalid_payload`, `low_confidence`, `road_closure`,
   `not_collected`. Với `invalid_payload`, `raw_response_file` có thể vẫn có
   đường dẫn để audit payload HTTP 200 bị thiếu/NaN; các trường traffic vẫn
   là `null`, không được gắn nhầm là observed.
5. `quality_flag` không xóa dữ liệu. Ví dụ: `low_confidence`,
   `road_closure`, `ratio_below_one`, `partial_snapshot`.
6. Tập record canonical có thể materialize một hàng missing cho mọi node tại
   một timestamp bị thiếu. Vì vậy benchmark biết “không có dữ liệu” thay vì
   nhầm đó là không tồn tại node.

## Ví dụ record quan sát

```json
{
  "dataset_id": "hcm_real_tomtom_segments_v1",
  "dataset_version": "v1-pilot",
  "schema_version": "1.1.0",
  "timestamp_utc": "2026-10-28T01:00:49+00:00",
  "timestamp_local": "2026-10-28T08:00:49+07:00",
  "node_id": "district_1",
  "segment_id": "district_1",
  "approved_openlr": "...",
  "source_name": "tomtom",
  "source_type": "observed_traffic_flow_segment_api",
  "current_speed_kmh": 22.0,
  "free_flow_speed_kmh": 32.0,
  "current_travel_time_s": 296.0,
  "free_flow_travel_time_s": 203.0,
  "congestion_ratio": 1.458128,
  "confidence": 0.99535,
  "road_closure": false,
  "observation_status": "observed",
  "is_missing": false,
  "missing_reason": "",
  "quality_flag": "",
  "registry_config_sha256": "<64-char SHA-256>",
  "raw_response_file": "raw/20261028T010049Z_district_1.json"
}
```

## Ví dụ đường đóng

```json
{
  "observation_status": "road_closed",
  "is_missing": false,
  "road_closure": true,
  "current_speed_kmh": null,
  "current_travel_time_s": null,
  "congestion_ratio": null,
  "quality_flag": "road_closure|not_applicable_traffic_metrics"
}
```

Đây là một tín hiệu có giá trị: hệ thống biết segment đang đóng. Pipeline/model
sau này phải dùng `observation_status` hoặc `road_closure` làm mask, không thay
`null` bằng 0 km/h. Tương tự,
`confidence` thấp phải được gắn cờ trước khi một experiment chọn ngưỡng lọc;
không được xóa raw record để làm dữ liệu trông đẹp hơn.

## Tạo dataset canonical từ raw collector (B2)

Sau khi collector đã có ít nhất một snapshot, chạy từ root repository:

```bash
python scripts/build_hcm_real_canonical.py --dataset-version v1-pilot
```

Script chỉ **đọc** raw CSV và error log; không sửa raw data. Nó tạo ba artifact
trong `data/processed/hcm_real_tomtom_v1/`:

| File | Nội dung |
|---|---|
| `hcm_real_tomtom_canonical.csv` | Lưới đầy đủ `timestamp x 17 node`, gồm cả row observed lẫn row missing. |
| `quality_summary.json` | Coverage, timeout, snapshot đầy đủ/một phần, cờ chất lượng, speed/confidence summary. |
| `manifest.json` | Hash raw input, registry, output và tham số build để audit/tái tạo. |

Ví dụ nếu lúc 22:00 API timeout ở 5 node, raw CSV chỉ có 12 dòng. Canonical
CSV vẫn có 17 dòng tại 22:00: 12 dòng `is_missing=false` và 5 dòng
`is_missing=true, missing_reason=api_timeout`. Nếu máy không thu trong một
giờ nào đó, grid cũng tạo đủ 17 dòng với `missing_reason=not_collected`.

Artifact canonical hiện bị `.gitignore` cùng raw TomTom, vì điều khoản phân
phối dữ liệu nguồn chưa được rà soát. Git chỉ lưu schema, builder, test và
metadata tổng hợp không nhạy cảm.
