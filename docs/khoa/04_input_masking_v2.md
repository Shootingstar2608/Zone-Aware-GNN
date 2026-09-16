# 04 — Input Masking v2

**Ngày:** 2026-09-15
**Phạm vi:** `apply_input_mask()` trong `benchmark/partition_gen.py`, `tests/test_benchmark.py`
**Mức độ:** 🟠 Cao — đổi số kênh đầu vào ⇒ **vỡ toàn bộ checkpoint cũ**

---

## 1. v1 làm gì, và vì sao chưa đủ

`quantity_skew` sinh ra mask `(S, N)`: ô nào `False` nghĩa là nút đó không có dữ liệu ở
cửa sổ đó.

**v1 (loss masking)** chỉ bỏ chấm điểm những ô đó. Nhưng **đặc trưng của nút vẫn nằm
nguyên trong input** — model vẫn nhìn thấy đầy đủ 12 bước lịch sử của nút "thiếu dữ liệu".
Nó chỉ không bị phạt khi dự báo sai nút đó.

Nói cách khác: v1 tạo ra **lệch gradient** giữa các nút (đúng bản chất quantity skew trong
FL), nhưng **không tạo ra bài toán thiếu thông tin**. Model không hề bị ép phải suy từ
hàng xóm.

**v2 (input masking)** che luôn đặc trưng đầu vào. Giờ nút tối om thật sự tối, và cách duy
nhất để dự báo nó là qua **message passing từ các nút lân cận** — tức đúng chỗ
zone-awareness phải chứng minh giá trị.

---

## 2. ⚠️ Chỗ plan tuần này nói ngược

Plan viết:

> *"khi nút bị drop, đặc trưng của nó được thay thế bằng cờ ẩn **hoặc giá trị nội suy**,
> ép mô hình phải khai thác liên kết vùng (Zone Adjacency) từ các nút lân cận"*

**Nội suy không ép được model làm gì cả — nó làm bài toán DỄ đi.**

Nếu ta nội suy giá trị của nút thiếu từ hàng xóm rồi đưa vào input, thì **ta đã làm hộ
model đúng cái việc ta muốn nó học**. Model nhận được một tensor đầy đủ, không có lỗ hổng
nào, và không có động lực nào để dùng đồ thị — thông tin hàng xóm đã được trộn sẵn vào
rồi.

Chỉ **cờ ẩn** (flag) mới thật sự tạo ra lỗ hổng buộc model phải tự bắc cầu qua đồ thị.

### Cách xử lý: biến nội suy thành baseline thay vì điều kiện thí nghiệm

| Chiến lược | Vai trò |
|---|---|
| **`flag`** (mặc định) | **Điều kiện thí nghiệm chính.** Thêm 1 kênh availability, đặc trưng bị che đặt về `fill_value`. Model phân biệt được "không có dữ liệu" với "đường thoáng", và phải tự suy phần thiếu |
| `neighbor` | **Baseline: nội suy thủ công theo đồ thị.** Đây là *cận trên* của việc bắc cầu bằng tay — model zone-aware phải vượt được nó mới có ý nghĩa |
| `last_seen` | **Baseline: nội suy theo thời gian.** Lấy lại quan sát gần nhất của chính nút đó |
| `zero` | **Baseline ngây thơ.** Chỉ điền `fill_value`, không có kênh báo ⇒ model không phân biệt được thiếu dữ liệu với giá trị thật |

Cách này còn **tốt hơn** cho paper: thay vì một con số, ta có một bảng bốn dòng cho thấy
*cách xử lý dữ liệu thiếu* ảnh hưởng thế nào — và nếu `flag` thắng `neighbor`, đó là bằng
chứng model học được thứ mà nội suy thủ công không làm được.

---

## 3. Quyết định thiết kế

### 3.1. Che ở mức **mẫu**, không phải mức snapshot

Mask là `(S, N)` — theo *cửa sổ*, không theo *snapshot*. Nhưng các cửa sổ **chồng lấn
nhau**: cửa sổ 100 và 101 dùng chung 11/12 snapshot.

Hệ quả: cùng một snapshot có thể **hiện** ở cửa sổ này và **ẩn** ở cửa sổ kia. Xét về câu
chuyện vật lý "sensor chết" thì không nhất quán.

**Vì sao vẫn chọn mức mẫu:**

1. **Nhất quán với v1.** Loss mask cũng là `(S, N)`. Nếu input mask theo snapshot còn loss
   mask theo mẫu thì hai tầng nói hai chuyện khác nhau.
2. **Diễn giải vẫn hợp lệ:** mỗi mẫu huấn luyện là một "cơ hội quan sát" độc lập, mask nói
   nút nào báo cáo trong lần đó.
3. Mức snapshot cần sinh mask ở tầng khác hẳn (`(T_total, N)` rồi ánh xạ ngược), tức đổi
   API của cả 4 cơ chế.

**Đây là hạn chế đã biết**, ghi vào Data Card. Nếu reviewer hỏi thì trả lời thẳng: mask
theo mẫu, không mô phỏng sensor chết liên tục theo thời gian thực.

*(Tham số `block_len` của `quantity_skew` đã bù một phần: nó cho các cửa sổ bị che nằm
thành khối liên tiếp thay vì rải rác.)*

### 3.2. `flag` đổi số kênh: `F: 4 → 5`

```
in_channels: T_in * F = 12 * 4 = 48   ->   12 * 5 = 60
```

**Hệ quả: mọi checkpoint `.pt` cũ không load được nữa.** Đây là lý do v2 bị đẩy sang sau
v1 ngay từ đầu (`docs/khoa/03` §4.1).

Người 4 phải truyền `meta` đã cập nhật `F=5` vào `build_model()`, nếu không sẽ lệch chiều
ở lớp đầu tiên.

### 3.3. ⚠️ `fill_value` phụ thuộc vào việc Tôn có chuẩn hoá hay chưa

Đây là chỗ **hai người phải nói chuyện với nhau**.

Hiện tại dữ liệu **chưa chuẩn hoá**:

| Kênh | Miền giá trị thật |
|---|---|
| `congestion_ratio` | 0.96 – 2.5 |
| `traffic_delay_s` | 0.03 – 1471 |
| `travel_time_s` | 535 – 2414 |
| `ff_ratio` | 0.96 – 2.5 |

Với `fill_value = 0`, ô bị che có `travel_time_s = 0` — **nằm ngoài miền giá trị thật hoàn
toàn**. Model nhận ra ngay lập tức. Nghĩa là baseline `zero` đang vô tình hoạt động gần
giống `flag`, và phép so sánh giữa hai cái mất ý nghĩa.

**Sau khi Tôn triển khai z-score**, `0` trở thành **giá trị trung bình** — rất khó phân
biệt với dữ liệu thật. Lúc đó `zero` mới thật sự là baseline ngây thơ, và bảng so sánh 4
chiến lược mới đọc được.

> 🔗 **Kết luận: bảng so sánh 4 chiến lược chỉ nên chạy SAU khi z-score đã vào.**
> Chạy trước thì phải ghi rõ dữ liệu chưa chuẩn hoá, và không so `zero` với `flag`.

### 3.4. Phát hiện phụ: `ff_ratio` gần như trùng `congestion_ratio`

```
max |ff_ratio - congestion_ratio| = 0.00027   (chỉ là sai số làm tròn)
```

Vì generator tính `travel_time = base_dur × cong_ratio` và `free_flow = base_dur`, nên
`ff_ratio = travel_time / free_flow = cong_ratio` **theo định nghĩa**.

Nghĩa là $F = 4$ nhưng chỉ có **3 kênh mang thông tin độc lập**. Không chí mạng, nhưng:

- Nên nói rõ trong Data Card
- Liên quan tới việc chuẩn hoá của Tôn (chuẩn hoá một kênh trùng lặp)
- Và giải thích một phần vì sao model có 358k tham số lại overfit trên 17 nút

---

## 4. Số liệu

```
alpha  frac_masked  node tối om  in_channels
 0.1        0.500            4           60
 0.5        0.500            0           60
 1.0        0.500            0           60
 5.0        0.500            0           60
```

`frac_masked` đúng `0.500` ở **mọi** $\alpha$ — đó là bất biến của `mode="fixed_coverage"`
hiện lên ở tầng input: tổng lượng dữ liệu không đổi, chỉ có *phân bố* đổi. Đúng như thiết
kế ở `docs/khoa/03` §9.1.

Ở $\alpha = 0.1$ có **4 nút tối om hoàn toàn** — đó là thí nghiệm ăn tiền: dự báo một nút
chưa từng được quan sát, chỉ bằng nhãn zone và cấu trúc đồ thị.

---

## 5. `tests/test_benchmark.py` — 27 test, pass hết

Chạy được cả hai kiểu, không bắt buộc cài pytest:

```bash
python tests/test_benchmark.py          # runner thuần
pytest tests/test_benchmark.py -v       # nếu có pytest
```

| Nhóm | Số test | Kiểm gì |
|---|---|---|
| Gini | 2 | Biên (đều / lệch hết / rỗng / chia 0) + đơn điệu theo $\alpha$ |
| `quantity_skew` | 5 | Trần $S$, giữ ngân sách, stats khớp mask, `block_len` không chồng lấn, mode sai raise |
| `zone_skew` | 3 | **Gini = 0** (trực giao với quantity), JSD tăng khi `off_band_weight` giảm, `labels` thủ công |
| `temporal_shift` | 5 | **Không rò rỉ 3 cặp**, val là ngày thường, 3 tập rời nhau, gap scale theo horizon, `normal_to_rush` bị từ chối |
| `concept_drift` | 2 | Nằm trong test, profile hình thang |
| Tái lập | 3 | Cùng seed → cùng hash, khác seed → khác mask, **không đọc rng toàn cục** |
| Schema | 1 | Bản ghi đủ 12 trường, serialize được |
| Input masking | 5 | `flag` thêm kênh đúng, đặc trưng thật sự bị xoá, `neighbor` không để lỗ, giữ shape, bắt lỗi shape |

Ba test đáng chú ý:

**`test_quantity_skew_stats_khop_mask`** — đếm lại `n_per_node` từ chính mask. Chính test
kiểu này đã bắt được lỗi mask rỗng hồi P1 (stats đẹp long lanh trong khi mask toàn `False`).

**`test_khong_dung_rng_toan_cuc`** — gọi `np.random.seed()` với hai giá trị khác nhau rồi
kiểm mask có đổi không. Vì `train.py` có `set_seed(42)`, nếu module lỡ dùng rng toàn cục
thì thứ tự gọi hàm sẽ đổi kết quả mà **không báo lỗi gì**.

**`test_temporal_shift_khong_ro_ri`** — kiểm cả 3 cặp train–test, train–val, val–test. Chỉ
kiểm train–test là chưa đủ.

---

## 6. Việc bàn giao

**Cho Người 4:**
- `apply_input_mask(X, mask, meta, strategy="flag")` → `(X_masked, info)`
- Nhớ truyền `meta` có `F=5` vào `build_model()` khi dùng `flag`
- Checkpoint cũ **không load được** — phải train lại từ đầu

**Cho Tôn:**
- Bảng so sánh 4 chiến lược **chờ z-score**. Xem §3.3
- `ff_ratio` gần như trùng `congestion_ratio` — cân nhắc lúc chuẩn hoá

**Câu hỏi còn treo cho nhóm:**
- Che ở mức mẫu hay mức snapshot? (§3.1 — hiện chọn mức mẫu, có lý do, nhưng là hạn chế đã biết)
- Nút tối om hoàn toàn: nếu training ra NaN thì đặt sàn tối thiểu hay xử ở tầng eval?
