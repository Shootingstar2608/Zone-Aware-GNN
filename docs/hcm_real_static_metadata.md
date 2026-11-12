# HCM-Real v1: metadata tĩnh (B3)

## Vì sao phải tách metadata tĩnh khỏi traffic động?

Traffic động là câu trả lời cho câu hỏi: **lúc 08:00 tại node A xe đang chạy
bao nhanh?** Nó thay đổi theo từng snapshot TomTom. Ngược lại, tọa độ node,
nhãn ngữ cảnh khu vực và khoảng cách/thời gian route tham chiếu không thay đổi
ở mỗi giờ thu. Vì vậy B3 tạo ba bảng riêng, nối với traffic động bằng
`node_id`; không lặp lại toàn bộ metadata trong từng record traffic.

| Artifact | Số dòng mong đợi | Nguồn | Nó nói lên điều gì? |
|---|---:|---|---|
| `static_nodes.csv` | 17 | Registry segment đã duyệt | 17 node nào được nghiên cứu, tọa độ neo và OpenLR nào đã được kiểm tra. |
| `osm_zone_labels.csv` | 17 | OpenStreetMap qua Overpass | Các loại POI/land-use xuất hiện trong bán kính đã công bố quanh từng node. |
| `osrm_od_reference.csv` | 272 = 17 x 16 | OSRM Table API | khoảng cách và thời lượng route khi lái xe từ node A đến B, theo router tại thời điểm truy xuất. |
| `static_metadata_manifest.json` | 1 | Pipeline B3 | hash, số dòng và registry mà ba bảng cùng tham chiếu. |

`OpenLR` là mã định danh vị trí đường do TomTom trả về. Nó quan trọng hơn chỉ
dùng tên đường vì một tên đường có thể có nhiều đoạn. `OD` là
*origin--destination*: cặp điểm xuất phát--điểm đến. 272 là **có hướng**:
A→B và B→A là hai route riêng, vì đường một chiều hay cách router chọn đường
có thể khác nhau.

## Ý nghĩa của zone OSM

Script dùng bán kính mặc định 800 m quanh **tọa độ anchor đã approved**, rồi
gắn nhãn 0/1 cho `commercial`, `residential`, `industrial`, `school`,
`university`, `hospital`, `transport`, `park`. Ví dụ `school=1` nghĩa là OSM
có ít nhất một đối tượng được gắn thẻ trường học/mẫu giáo trong bán kính này;
nó không có nghĩa node chính là cổng trường, cũng không chứng minh có ùn tắc.

Đây là nhãn ngữ cảnh có thể dùng để xây scenario *zone skew* (các client nhận
nhiều node thuộc loại khu vực khác nhau). Nó **không** được dùng để tự khẳng
định một mô hình sẽ tốt hơn: cần chạy ablation và báo cáo giới hạn đó.

OSM là dữ liệu cộng đồng và có thể thay đổi theo thời gian. Output lưu endpoint,
thời điểm lấy, bán kính, và hash registry để có provenance (dấu vết nguồn gốc).
Khi viết paper/repository công khai, phải ghi attribution OpenStreetMap và rà
soát điều khoản ODbL trước khi phân phối lại artifact.

## Ý nghĩa của OSRM OD reference

`duration_s` của OSRM là thời lượng route tham chiếu từ routing engine; nó
không phải tốc độ đo được bởi TomTom và không được gọi là giao thông thực.
Nó hữu ích để tạo trọng số khoảng cách/di chuyển cho đồ thị hoặc kiểm tra tính
hợp lý của các cặp node. Bảng này không tự biến đồ thị thành mạng đoạn đường:
nó là đồ thị OD dày đặc gồm các cặp node nghiên cứu.

## Cách nối với bảng canonical sau này

Ví dụ record dynamic có `node_id=district_1`, `timestamp=08:00`:

1. nối với `static_nodes.csv` theo `node_id` để biết tọa độ/segment approved;
2. nối với `osm_zone_labels.csv` theo `node_id` nếu experiment cần feature
   zone;
3. khi cần quan hệ node i→j, lấy đúng hàng trong `osrm_od_reference.csv` với
   `origin_node_id=i` và `destination_node_id=j`.

Không nối 272 cặp OD vào từng dòng traffic vì sẽ nhân số dòng lên 16 lần và dễ
nhầm free-flow/reference với quan sát TomTom.
