# Kế hoạch triển khai nhóm 4 người hướng đến benchmark và bài báo Q3
# Kế hoạch triển khai nhóm 4 người hướng đến benchmark và bài báo Q3

## 1. Mục tiêu và phạm vi

Mục tiêu của giai đoạn này là hoàn thiện một nền tảng dữ liệu giao thông đa
nguồn và benchmark Non-IID có thể tái lập cho kiến trúc Node-Agent. Kết quả
phải tách bạch hai dataset:

- **HCM-Sim:** dữ liệu mô phỏng có kiểm soát, dùng để tạo và đo các kịch bản
  Non-IID.
- **HCM-Real-TomTom:** dữ liệu giao thông quan sát tại TP.HCM, thu theo 17
  segment đã duyệt bằng TomTom Flow Segment API.

OSM cung cấp ngữ nghĩa không gian/zone; OSRM cung cấp routing, khoảng cách và
thời gian free-flow; TomTom cung cấp đặc trưng giao thông động. Không được mô
tả HCM-Sim là phép đo TomTom hay dữ liệu giao thông thực.

CityFlow không thuộc phạm vi bắt buộc của kế hoạch này. CityFlow chỉ là hướng
mở rộng sau này cho mô phỏng vi mô, can thiệp đèn tín hiệu hoặc sự cố; nó không
làm dữ liệu TomTom thực trở nên chính xác hơn và không thay thế việc thu thập
HCM-Real.

## 2. Mốc khởi đầu đã hoàn thành

Registry `data/config/hcm_tomtom_segments.v1.json` đã có 17 segment ở trạng
thái `approved`. Mỗi segment cần giữ lại `node_id`, tọa độ, đường/corridor,
OpenLR đã duyệt, lý do chọn và manifest validation. Đây là **danh mục điểm
quan sát**, chưa phải chuỗi dữ liệu đủ dài để huấn luyện hay kết luận khoa học.

## 3. Phân công thành viên A -- thu thập dữ liệu và provenance

### Mục tiêu

Tạo chuỗi HCM-Real quan sát được, có thể truy vết nguồn gốc và đủ dài để đánh
giá theo thời gian.

### Công việc

1. **Đóng băng registry 17 segment.** Không thay đổi tọa độ, `node_id`, đường
   mục tiêu hoặc OpenLR trong phiên bản v1. Nếu buộc phải thay đổi, tạo phiên
   bản registry mới và chạy validation lại.
2. **Thu thử trong 24 giờ.** Chạy collector mỗi 60 phút; kỳ vọng có 408 bản
   ghi (= 17 node x 24 giờ). Rà lỗi API, retry, node bị thiếu, confidence thấp
   và road closure.
3. **Thu tự động tối thiểu 28 ngày liên tiếp.** Dùng `systemd timer` hoặc
   `cron`, không phụ thuộc vào một terminal đang mở. Với tần suất giờ, mục tiêu
   là 11.424 bản ghi (= 17 x 24 x 28), trước khi trừ các bản ghi thiếu hợp lệ.
4. **Bảo vệ khóa API.** Chỉ đọc `TOMTOM_API_KEY` từ biến môi trường; không ghi
   vào config, log hoặc Git.
5. **Lập source register.** Với TomTom, OSM và OSRM, ghi vai trò nguồn, URL/
   phiên bản, ngày truy xuất, license/terms, giới hạn phát hành raw data,
   privacy và cách tái lập.

### Đầu ra nghiệm thu

- Registry v1 và manifest validation được liên kết rõ ràng.
- Log thu thập, raw response theo ngày và báo cáo pilot 24 giờ.
- Cơ chế scheduler chạy được sau khi reboot.
- Báo cáo coverage theo node/ngày và source/license register.

## 4. Phân công thành viên B -- data engineering và chất lượng dữ liệu

### Mục tiêu

Chuyển API response thô thành HCM-Real có schema thống nhất và có cờ chất
lượng, không xóa im lặng dữ liệu bất thường.

### Công việc

1. **Chốt schema canonical HCM-Real.** Tối thiểu gồm:
   `timestamp_utc`, `timestamp_local`, `node_id`, `segment_id`,
   `approved_openlr`, `source`, `current_speed_kmh`, `free_flow_speed_kmh`,
   `current_travel_time_s`, `free_flow_travel_time_s`, `congestion_ratio`,
   `confidence`, `road_closure`, `is_missing`, `quality_flag` và
   `dataset_version`.
2. **Viết pipeline raw -> cleaned -> canonical.** Chuẩn hóa múi giờ và đơn vị,
   loại/đánh dấu duplicate, resample về từng giờ, và lưu lý do thiếu dữ liệu:
   API lỗi, chưa thu được, closure hoặc confidence thấp.
3. **Gắn static metadata.** Liên kết bảng node/segment của registry với zone
   từ OSM và free-flow, distance, adjacency từ OSRM. Mỗi trường phải nêu rõ
   nguồn; không để dữ liệu các nguồn ghi đè lên nhau.
4. **Chạy quality checks.** Đo completeness, validity, consistency, freshness
   và source reliability theo node/ngày. Ví dụ consistency cần phát hiện
   `current_travel_time_s < free_flow_travel_time_s` nếu điều đó không phù hợp
   với định nghĩa dữ liệu đã chọn.
5. **Phiên bản hóa dataset.** Manifest phải ghi config hash, thời gian thu,
   số record, coverage, schema version và hash của artifact.

### Đầu ra nghiệm thu

- Dataset HCM-Real canonical cùng script build chạy lại được.
- Bảng `nodes`, bảng dynamic traffic, bảng/JSON quality report.
- Unit test cho schema, timestamp, range, duplicate và missing flag.
- Báo cáo chất lượng theo từng node và từng ngày.

## 5. Phân công thành viên C -- benchmark, Non-IID và mô hình

### Mục tiêu

Tạo protocol công bằng, chống rò rỉ thời gian và đánh giá được tác động của
Non-IID thay vì chỉ có một bảng metric một lần chạy.

### Công việc

1. **Kiểm tra split.** Dùng chronological split train -> validation -> test;
   đặt purge gap phù hợp với $T_{in}=12$, $T_{out}=24$; scaler chỉ fit trên
   train. Viết test xác nhận các cửa sổ train/test không chồng input/target.
2. **Sửa baseline theo trục thời gian thật.** Mô hình nhận tensor
   `(batch, node, T_in, feature)`. LSTM/GRU phải quét 12 timestep, không được
   flatten 12 snapshot thành một bước giả. Bổ sung persistence, mean-by-node
   và seasonal naive.
3. **Hoàn thiện scenario HCM-Sim.** Xác nhận và chạy end-to-end quantity skew,
   feature/zone skew, temporal shift, concept drift và input masking. Input
   masking phải che input thực, không chỉ che loss.
4. **Chuẩn hóa experiment.** Mỗi scenario có config, intensity, seed, mask/
   partition hash và dataset fingerprint. Chạy ít nhất 5 seed để debug; bảng
   chính ưu tiên 10 seed nếu đủ tài nguyên.
5. **Báo cáo thống kê.** MAE, RMSE, MAPE/WAPE; mean, standard deviation,
   CI95%; effect size và kiểm định có điều chỉnh đa so sánh khi tuyên bố một
   model tốt hơn.

### Đầu ra nghiệm thu

- Baseline tuần tự đúng và test shape/leakage.
- Fixed splits, scenario configs, seed manifests.
- Bảng HCM-Sim tái tạo được bằng command rõ ràng.
- Khi HCM-Real đã đủ dài: bảng pilot/đánh giá trên HCM-Real theo cùng protocol.

## 6. Phân công thành viên D -- tài liệu, tái lập và paper

### Mục tiêu

Đóng gói code, dữ liệu, protocol và kết quả thành research artifact mà người
khác có thể hiểu, kiểm tra và tái tạo.

### Công việc

1. **Viết hai Data Card riêng.** Data Card HCM-Sim phải nêu rõ synthetic và
   generator; Data Card HCM-Real phải nêu rõ TomTom observed, coverage, API
   terms, missingness và giới hạn phân phối raw data.
2. **Rà claim và tài liệu cũ.** Sửa mọi câu lẫn lộn giữa HCM-Sim/TomTom.
   Claim paper nên là nền tảng dữ liệu đa nguồn + benchmark Non-IID tái lập,
   không phải tuyên bố Zone-Aware GNN vượt trội trên giao thông thực khi chưa
   có bằng chứng HCM-Real đầy đủ.
3. **Viết hướng dẫn tái lập.** README phải hướng dẫn từ clone sạch đến build
   dataset, tạo partition, chạy baseline, tái tạo bảng chính; ghi rõ seed,
   version, hash, config và yêu cầu API key.
4. **Kiểm tra clone sạch.** Một thành viên không tham gia code phải clone mới,
   làm theo README và tái tạo được artifact/kết quả chính.
5. **Viết paper.** Hoàn thiện motivation, related work, methodology, Data Card
   summary, kết quả, limitations, ethics/license và appendix cấu hình.

### Đầu ra nghiệm thu

- Hai Data Card và source/license register hoàn chỉnh.
- README tái lập và log “clean clone reproduced successfully”.
- Paper draft, appendix config/seed và checklist submission.

## 7. Phụ thuộc và lịch triển khai gợi ý

| Tuần | A: thu thập | B: pipeline | C: benchmark | D: tài liệu |
|---|---|---|---|---|
| 1 | Freeze registry, pilot 24 giờ | Chốt schema, parser raw | Sửa temporal baseline | Tách Data Card, source register |
| 2 | Bắt đầu scheduler 28 ngày | Clean/resample, quality report | Baseline HCM-Sim | Viết methodology/protocol |
| 3 | Theo dõi coverage/API | Build HCM-Real v1 | Scenario Non-IID nhiều seed | Viết kết quả HCM-Sim |
| 4 | Tiếp tục tích lũy dữ liệu | Freeze snapshot nếu đủ | HCM-Real pilot | Clean-clone, hoàn thiện paper |

Thu thập 28 ngày là ràng buộc thời gian thực, không thể rút ngắn bằng cách
chạy nhanh hơn. Trong thời gian chờ, các thành viên B, C, D phải hoàn thiện
pipeline, benchmark và artifact HCM-Sim song song.

## 8. Quy tắc phối hợp Git

- A sở hữu collector, config segment và tài liệu nguồn dữ liệu.
- B sở hữu schema, adapter, preprocessing và quality checks.
- C sở hữu partition, model, runner và result configs.
- D sở hữu Data Card, README và paper/report.
- Mỗi nhánh/PR chỉ giải quyết một mục tiêu; không dùng `git add .` khi có thay
  đổi không liên quan.
- Raw response TomTom và API key không được commit nếu `.gitignore`/terms
  không cho phép.

## 9. Checklist trước khi gửi bài Q3

- [ ] Registry có 17 segment riêng biệt, approved và provenance rõ.
- [ ] Có chuỗi HCM-Real liên tục cùng báo cáo coverage/missingness/chất lượng.
- [ ] Có schema chung phân biệt vai trò TomTom, OSM và OSRM.
- [ ] HCM-Sim và HCM-Real được tách bạch về thư mục, Data Card và claim.
- [ ] Split theo thời gian, purge gap và train-only normalization đã được test.
- [ ] Baseline xử lý đúng trục thời gian và có baseline naive.
- [ ] Các scenario Non-IID tái lập bằng config, seed và fingerprint.
- [ ] Kết quả chính có nhiều seed và thống kê bất định.
- [ ] License, privacy, ethics, limitations và khả năng phát hành dữ liệu được
  công bố rõ.
- [ ] Clone sạch có thể tái tạo kết quả chính theo README.

Hoàn thành checklist trên là điều kiện cần để bài benchmark/data có phạm vi
hẹp nhưng có thể bảo vệ ở mức Q3; không phải cam kết chắc chắn tạp chí sẽ nhận
bài, vì quyết định còn phụ thuộc phản biện và mức phù hợp của venue.
