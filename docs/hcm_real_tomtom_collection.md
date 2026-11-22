# HCM-Real TomTom Segment Collection v1

## Mục đích

Thu thập **traffic quan sát thực** ở 17 neo đoạn đường tại TP.HCM, tương ứng
với 17 node HCM-Sim. Đây không phải dữ liệu synthetic và không thay thế
HCM-Sim: HCM-Sim là controlled benchmark; collector này tạo evidence thực để
kiểm tra/calibrate benchmark.

## Vì sao không query 272 cặp OD

272 cặp $\times$ 96 snapshot/ngày ở khoảng 15 phút cần 783.360 request/tháng,
vượt xa free tier. Collector dùng 17 điểm neo road segment. Với khoảng 60 phút,
nó dùng khoảng 12.240 request trong 30 ngày.

## Cấu hình segment

`data/config/hcm_tomtom_segments.v1.json` chứa 17 anchor. Chúng có trạng thái
`pending_api_validation`, không được gọi là segment đã chốt cho đến khi phản hồi
TomTom đầu tiên được kiểm tra. Quy trình chốt:

1. Xuất API key cục bộ, không commit:

   ```bash
   export TOMTOM_API_KEY='...'
   ```

2. Chạy kiểm tra một snapshot:

   ```bash
   python scripts/collect_tomtom_segments.py --validate-only
   ```

3. Mở `data/raw/tomtom_hcm_segments/segment_validation_pending_review.json`
   cùng các raw response. Kiểm tra hình học `returned_segment_coordinates`,
   functional road class và mã `openlr` (nếu TomTom trả về) có nằm trên road
   segment mong muốn không.

4. Chỉ sau khi kiểm tra, đổi từng `status` thành `approved` và bổ sung tên
   đường đã đối chiếu bản đồ, thời điểm xác thực và OpenLR (nếu có) vào object
   segment. Flow Segment Data không bảo đảm trả tên đường, nên tên đường là kết
   quả của bước đối chiếu hình học, không được tự suy diễn từ tên node. Commit
   **config**, không commit API key hay raw response.

## Thu thập chính thức

```bash
python scripts/collect_tomtom_segments.py --hours 24 --interval-min 60
```

Chạy định kỳ từ scheduler/cron của máy có API key. Output local:

```text
data/raw/tomtom_hcm_segments/raw/*.json
data/raw/tomtom_hcm_segments/tomtom_hcm_segments_observed.csv
data/raw/tomtom_hcm_segments/collection_errors.jsonl
```

CSV chuẩn hóa có `current_speed_kmh`, `free_flow_speed_kmh`, travel time,
confidence, road closure và `congestion_ratio = current/free-flow travel time`.
Không công bố raw response hay dữ liệu đã thu trước khi rà điều khoản TomTom.

## Chuyển raw data sang canonical dataset

Raw CSV không nên được dùng trực tiếp cho benchmark vì một API timeout làm mất
hẳn dòng của node. Sau khi có dữ liệu mới, chạy:

```bash
python scripts/build_hcm_real_canonical.py --dataset-version v1-pilot
```

Builder tạo lưới thời gian x 17 node, thêm row missing cho timeout/gap, kiểm
tra duplicate/range/ratio, gắn quality flag và sinh manifest/quality summary.
Chi tiết contract: `docs/hcm_real_canonical_schema.md`.
