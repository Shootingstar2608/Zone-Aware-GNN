# Data Card — HCM-Sim v1

**Phiên bản:** `HCM-Sim v1.0-rc1` (bản nháp của Thành viên D)
**Loại dữ liệu:** synthetic / rule-based. **Không phải** dữ liệu giao thông quan sát.
**Contract nguồn:** [`hcm_sim_v1_spec.md`](hcm_sim_v1_spec.md). Nếu hai tài liệu mâu thuẫn, spec là chuẩn.
**Dataset liên quan:** HCM-Real-TomTom có Data Card riêng (`DATA_CARD_HCM_REAL.md`). Hai dataset không được trộn.
**Ngày cập nhật:** [TODO-D]
**Liên hệ:** [TODO-D]

> Quy ước: mục `[TODO-D]` là thông tin chưa được xác nhận từ file hoặc code thật.
> Không điền bằng phỏng đoán. Gỡ hết `[TODO-D]` trước khi mở PR.

---

## 1. Định vị

HCM-Sim là benchmark dự báo giao thông **mô phỏng**, xây trên đồ thị đường và
ngữ nghĩa vùng chức năng của TP.HCM. Mục đích là kiểm soát và đo các kịch bản
Non-IID (quantity skew, zone skew, temporal shift, concept drift, input masking)
theo một protocol không rò rỉ thời gian và tái lập được.

**HCM-Sim là:**

- một bộ dữ liệu mô phỏng có seed, sinh lại được từ code;
- nền tảng để so sánh phương pháp dưới điều kiện Non-IID có kiểm soát.

**HCM-Sim không phải:**

- phép đo từ TomTom hay bất kỳ cảm biến giao thông nào;
- bằng chứng về hiệu năng trên giao thông thực tế.

Mọi bảng, hình và câu trong paper dùng dữ liệu này phải gọi nó là _synthetic_.

## 2. Thành phần và nguồn

| Thành phần          | Nguồn                                       | Cách tạo                                                                                              | Vai trò                                                       |
| ------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Đồ thị tĩnh `A`     | `data/raw/hcm_osrm_dataset.csv`             | Routing OSRM trên dữ liệu OSM                                                                         | Adjacency có hướng, chuẩn hóa theo hàng; tham chiếu free-flow |
| Zone labels `Z`     | `data/raw/zone_labels.csv`                  | [TODO-D: `collect_zones.py` (Overpass, bán kính ? m) hay `scripts/dev/create_zone_labels_manual.py`?] | Ma trận multi-label `(N, K)`                                  |
| Chuỗi động `X`, `Y` | `scripts/dev/generate_synthetic_traffic.py` | Simulator rule-based, xác định theo seed                                                              | 4 feature đầu vào và target tương lai                         |

Chuỗi động do simulator sinh ra, `data_source: synthetic`.
[TODO-D: xác nhận generator không dùng tham số hiệu chỉnh từ dữ liệu TomTom.]

## 3. Cấu trúc dữ liệu

### 3.1 Hình học

| Tham số          | Giá trị                       | Ghi chú                                        |
| ---------------- | ----------------------------- | ---------------------------------------------- |
| `N` (node)       | 17                            | Danh sách ở 3.2                                |
| `K` (zone types) | 8                             | Danh sách ở 3.3                                |
| `F` (feature)    | 4                             | Danh sách ở 3.5                                |
| Khoảng lấy mẫu   | 15 phút                       | 96 snapshot/ngày                               |
| Thời lượng       | 7 ngày, bắt đầu thứ Hai 00:00 | `T = 672`                                      |
| `T_in` / `T_out` | 12 / 24                       | Mặc định của benchmark                         |
| `S` (số cửa sổ)  | 637                           | `S = T − T_in − T_out + 1 = 672 − 12 − 24 + 1` |
| Seed             | 42                            |                                                |

Các giá trị trên lấy từ spec và Data Card cũ. [TODO-D: đối chiếu với
`data/processed/hcm_sim_v1/meta.json`, không dùng `data/processed/meta.json` vì đó là bản legacy.]

### 3.2 17 node

Ben Thanh Market, Binh Thanh, District 1, District 3, District 5, Eastern Bus
Station, Hang Xanh, High Tech Park, Landmark 81, Linh Trung, Pham Van Dong,
Saigon Bridge, Suoi Tien, Tan Son Nhat Airport, Thu Duc, Thu Thiem Tunnel, VNU HCM.

### 3.3 8 zone types

`commercial`, `residential`, `industrial`, `school`, `university`, `hospital`,
`transport`, `park`. Nhãn là nhị phân và **multi-label**: một node có thể thuộc
nhiều zone. [TODO-D: nhãn nghĩa là "có POI loại đó trong bán kính R" hay "land use
của chính node"? Điền sau khi xác nhận nguồn ở mục 2.]

### 3.4 Tensor trong `graph_dataset.pt`

| Field                                  | Shape            | Mô tả                                       |
| -------------------------------------- | ---------------- | ------------------------------------------- |
| `A`                                    | `(N, N)`         | Adjacency có hướng, chuẩn hóa theo hàng     |
| `Z`                                    | `(N, K)`         | Zone labels nhị phân                        |
| `X`                                    | `(S, N, T_in·F)` | Cửa sổ đầu vào                              |
| `Y`                                    | `(S, N, T_out)`  | Target: congestion ratio các bước tương lai |
| `time_labels`                          | `(S,)`           | Nhãn thời gian rời rạc (3.6)                |
| `hour`                                 | `(S,)`           | Giờ `[0, 23]`                               |
| `dow`                                  | `(S,)`           | Thứ `[0, 6]`, thứ Hai = 0                   |
| `time_sinc`                            | `(S, 8)`         | Mã hóa chu kỳ giờ/ngày                      |
| `nodes`, `feature_names`, `zone_types` | list             | Thứ tự khớp với tensor                      |

### 3.5 Bốn feature động (theo thứ tự)

| #   | Feature            | Ý nghĩa                                                                                  |
| --- | ------------------ | ---------------------------------------------------------------------------------------- |
| 0   | `congestion_ratio` | Mức tắc nghẽn tương đối. Cũng là target `Y`                                              |
| 1   | `traffic_delay_s`  | Độ trễ do tắc nghẽn (giây)                                                               |
| 2   | `travel_time_s`    | Thời gian di chuyển (giây)                                                               |
| 3   | `ff_ratio`         | Tỉ lệ `travel_time_s` so với free-flow tham chiếu. [TODO-D: xác nhận mẫu số lấy từ OSRM] |

Cả bốn feature đều **do simulator sinh ra**, không có feature nào lấy từ TomTom.

### 3.6 Nhãn thời gian

| Nhãn         | Giá trị | Giờ             |
| ------------ | ------: | --------------- |
| night        |       0 | 00:00–05:59     |
| rush_morning |       1 | 07:00–09:59     |
| rush_evening |       2 | 16:00–19:59     |
| normal       |       3 | các giờ còn lại |

`time_labels`, `hour`, `dow` được suy từ cùng một timestamp. Dataset thiếu một
trong bốn nhãn không đạt release validation.

## 4. Cơ chế sinh dữ liệu

Đã biết từ spec: simulator rule-based, xác định theo `--seed`, nhận `--days` và
`--out`, và ghi vào metadata: `data_source`, tên và phiên bản simulator, seed,
thời điểm bắt đầu, khoảng lấy mẫu, số ngày.

Phần mô tả cơ chế chưa viết, vì cần đọc generator:

| Khía cạnh                               | Mô tả    |
| --------------------------------------- | -------- |
| Nhịp theo giờ trong ngày (rush profile) | [TODO-D] |
| Khác biệt ngày thường / cuối tuần       | [TODO-D] |
| Ảnh hưởng của zone lên congestion       | [TODO-D] |
| Lan truyền giữa các node kề nhau        | [TODO-D] |
| Nhiễu                                   | [TODO-D] |
| Sự cố / concept drift                   | [TODO-D] |

Phần "ảnh hưởng của zone" quan trọng nhất. Nó là giả định của simulator, và là
nguồn của thiên lệch ở mục 9.

## 5. Tái lập

Sinh dữ liệu động:

```bash
python scripts/dev/generate_synthetic_traffic.py \
  --seed 42 --days 7 \
  --out data/generated/hcm_sim_v1/hcm_sim_traffic.csv
```

Build dataset, không ghi đè dữ liệu legacy:

```bash
python scripts/build_graph.py \
  --traffic-path data/generated/hcm_sim_v1/hcm_sim_traffic.csv \
  --out-dir data/processed/hcm_sim_v1 \
  --t-in 12 --t-out 24
```

Đường dẫn chuẩn:

```text
data/generated/hcm_sim_v1/hcm_sim_traffic.csv
data/generated/hcm_sim_v1/hcm_sim_traffic.meta.json
data/processed/hcm_sim_v1/graph_dataset.pt
data/processed/hcm_sim_v1/meta.json
data/processed/hcm_sim_v1/manifest.json
```

`manifest.json` ghi SHA-256 của mọi input, Git commit, kích thước output, `T_in`, `T_out`.

| Thông tin bản release               | Giá trị  |
| ----------------------------------- | -------- |
| SHA-256 `graph_dataset.pt`          | [TODO-D] |
| Git commit lúc build                | [TODO-D] |
| SHA-256 input OSRM / zone / traffic | [TODO-D] |

Mọi partition gắn với dataset fingerprint. Khi hash dataset đổi, phải sinh lại partition.

## 6. Giao thức đánh giá

Mọi kết quả chính phải:

1. dùng chronological train/validation/test split;
2. đặt purge gap `T_in + T_out − 1 = 35` giữa các split liền kề;
3. fit normalizer chỉ trên train, áp cùng normalizer cho validation và test;
4. inverse-transform prediction và target trước khi tính MAE, RMSE, MAPE, WAPE;
5. với bảng cuối: tối thiểu 10 seed, báo mean, std, CI95%, effect size và kiểm
   định có hiệu chỉnh Holm.

Split ngẫu nhiên gây rò rỉ thời gian: các cửa sổ liền kề chồng lấn `T_in − 1`
bước, nên cửa sổ test gần như trùng một cửa sổ train. Kết quả random split chỉ
được giữ ở thư mục `legacy` để tham khảo lịch sử, **không** đưa vào bảng chính.

## 7. Kịch bản benchmark

| Kịch bản          | Yếu tố kiểm soát                    | Cần đo                                |
| ----------------- | ----------------------------------- | ------------------------------------- |
| IID chronological | Split thời gian chuẩn               | MAE, RMSE, MAPE, WAPE                 |
| Quantity skew     | Lệch lượng quan sát theo node       | Gini, coverage, metrics               |
| Zone skew         | Dịch chuyển phân phối theo cụm zone | Độ phân kỳ giữa cụm, metrics          |
| Temporal shift    | Ngày thường → cuối tuần             | Mức suy giảm tương đối                |
| Concept drift     | Cường độ, thời lượng, độ dốc sự cố  | Suy giảm và phục hồi                  |
| Input masking     | Thiếu quan sát theo node/thời gian  | Tỉ lệ thiếu, cách imputation, metrics |

Mỗi kịch bản ghi dataset fingerprint, tham số partition, seed, chỉ số split,
purge gap, Git commit của generator, và `mask_hash` khi có.

## 8. Mục đích sử dụng

**Phù hợp:** so sánh phương pháp dưới Non-IID có kiểm soát; kiểm tra protocol
đánh giá; kiểm thử pipeline và tính tái lập.

**Không phù hợp:** kết luận về hiệu năng trên giao thông thực; ước lượng tác
động chính sách hay quy hoạch giao thông; huấn luyện hệ thống vận hành thật.

## 9. Giới hạn đã biết

1. **Không chứng minh hiệu năng trên giao thông thực.** Cần kiểm chứng ngoài
   trên dữ liệu quan sát có giấy phép phù hợp trước khi claim mạnh.
2. **Thiên lệch do simulator.** Hiệu ứng zone là giả định của generator, nên
   phương pháp zone-aware có thể hưởng lợi từ inductive bias trùng với generator.
   Cần báo thêm cơ chế held-out của simulator.
3. **Chỉ có một tuần, một cuối tuần.** Kịch bản weekday→weekend có test set trùng
   hoàn toàn với "hai ngày cuối chuỗi", nên không tách được hiệu ứng cuối tuần
   khỏi hiệu ứng vị trí trong chuỗi.
4. **Quy mô nhỏ:** 17 node, 7 ngày, 637 cửa sổ.
5. **Nhãn zone:** nghĩa của nhãn phụ thuộc cách tạo (mục 2 và 3.3). [TODO-D]

## 10. Giấy phép, quyền riêng tư, đạo đức

- **Dữ liệu động** là dữ liệu tổng hợp, không chứa thông tin cá nhân hay định vị cá nhân.
- **Đồ thị và zone labels** dẫn xuất từ OpenStreetMap (© OpenStreetMap
  contributors, ODbL 1.0). OSRM engine dùng BSD-2-Clause, nhưng kết quả routing
  trên dữ liệu OSM vẫn kế thừa nghĩa vụ ODbL. Khi phát hành ngoài nhóm phải ghi attribution.
- **TomTom:** HCM-Sim v1 không chứa dữ liệu TomTom. [TODO-D: chốt sau khi xác nhận mục 2.]
- Việc dataset dẫn xuất có thuộc "Derivative Database" theo ODbL hay không cần
  lead hoặc giảng viên xác nhận. Tài liệu này không phải tư vấn pháp lý. [TODO-D]

## 11. Artifact legacy

Dataset legacy (`data/processed/graph_dataset.pt`, `data/raw/tomtom_traffic.csv`)
và kết quả random-split (`data/results/_legacy_random_split/`) **không** thuộc
benchmark HCM-Sim v1. Không đưa số liệu legacy vào bảng chính.

## 12. Lịch sử phiên bản

| Ngày     | Người sửa | Nội dung                                                          |
| -------- | --------- | ----------------------------------------------------------------- |
| [TODO-D] | Bảo (D)   | Tách Data Card HCM-Sim khỏi `DATA_CARD.md`; gỡ mô tả nguồn TomTom |
