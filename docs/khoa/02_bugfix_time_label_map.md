# 02 — Bugfix: `TIME_LABEL_MAP` nuốt mất toàn bộ nhãn giờ cao điểm

**Mức độ:** 🟠 Cao — làm hỏng đóng góp khoa học #2, chặn kịch bản Temporal Shift
**Ảnh hưởng tới:** `zone_full_tc` (Tân), `AH_GNN`, Người 2 (Temporal Shift), mọi kết quả đã có
**Chi phí sửa:** 1 hàm, nhưng **kéo theo rebuild dataset**

---

## 1. Triệu chứng

Đếm nhãn thời gian trong dataset thật:

```python
>>> Counter(d["time_labels"].tolist())
Counter({3: 480, 0: 157})
```

Chỉ có **2** giá trị. Nhãn `1` (cao điểm sáng) và `2` (cao điểm chiều) **chưa bao giờ xuất hiện**,
dù mọi model đều được khởi tạo với `num_time_labels=4`.

---

## 2. Nguyên nhân

`scripts/build_graph.py` dòng 46–51:

```python
TIME_LABEL_MAP = {
    **{h: 0 for h in range(0, 6)},    # 0-5   -> night
    **{h: 1 for h in range(7, 10)},   # 7-9   -> rush_morning
    **{h: 2 for h in range(16, 20)},  # 16-19 -> rush_evening
    **{h: 3 for h in range(6, 24)},   # 6-23  -> normal   <-- THU PHAM
}
```

Dict literal trong Python gộp **trái sang phải, key sau ghi đè key trước**.
`range(6, 24)` bao trùm `7,8,9` và `16,17,18,19` ⇒ dòng cuối **xoá sạch** hai dòng trên nó.

Kết quả: map thực tế chỉ còn `{0-5: 0, 6-23: 3}` — một cái đồng hồ **ngày/đêm**, không hơn.

### Bug thứ cấp: hai nguồn sự thật khác nhau

`scripts/dev/generate_synthetic_traffic.py` có hàm nhãn **riêng**, và hàm đó **đúng**:

```python
def get_time_label(hour: int) -> str:
    if hour in range(0, 6):   return "night"
    if hour in range(7, 10):  return "rush_morning"
    if hour in range(16, 20): return "rush_evening"
    return "normal"
```

Nên file CSV **có sẵn cột `time_label` đúng** (`"rush_morning"`, `"rush_evening"`...).
Nhưng `build_graph.py` **vứt cột đó đi** và tự tính lại từ `hour` bằng cái map hỏng:

```python
time_list.append(TIME_LABEL_MAP.get(hour, 3))   # bo qua row["time_label"]
```

Hai định nghĩa cho cùng một khái niệm, đặt ở hai file — một cái đúng, một cái sai, và
pipeline dùng đúng cái sai.

---

## 3. Hậu quả

1. **`zone_full_tc` (Tân) bị vô hiệu một nửa.** Time-Conditioned Zone Embedding
   $\tilde{z}^{(v,t)} = g_t \odot \text{MLP}(z^{(v)})$ lấy $g_t$ từ đúng nhãn rời rạc này.
   Với chỉ 2 nhãn, cổng thời gian chỉ biết "đêm" vs "không đêm" — không thể phân biệt
   cao điểm sáng với cao điểm chiều, tức **mất đúng thông tin mà nó sinh ra để nắm bắt**.
   Điều này khớp với `MULTISEED_FINDINGS §2.3`: `zone_full_tc` kém ổn định hơn
   `zone_full_sinc` (bản sinusoidal của Bảo lấy `hour`/`dow` liên tục nên không dính bug).

2. **2 trong 4 embedding vector thời gian là tham số chết.** Mọi model khai
   `num_time_labels=4` nhưng chỉ 2 nhãn từng được nhìn thấy ⇒ 2 vector không bao giờ nhận gradient.

3. **Chặn Người 2.** Kịch bản *Temporal Shift: Normal hours → Rush hours* trong plan tuần này
   **không thể thực hiện** — không có nhãn nào phân biệt được giờ cao điểm.

---

## 4. Cách sửa

Thay dict literal bằng hàm tường minh, đặt đúng chỗ cũ trong `scripts/build_graph.py`:

```python
def time_label_of(hour: int) -> int:
    """0 = night, 1 = rush_morning, 2 = rush_evening, 3 = normal.

    Khop dung ngu nghia cua get_time_label() trong
    scripts/dev/generate_synthetic_traffic.py — DUNG de hai noi lech nhau.
    Luu y: hour == 6 va 10-15, 20-23 deu roi vao 'normal', giong ben generator.
    """
    if 0 <= hour < 6:   return 0
    if 7 <= hour < 10:  return 1
    if 16 <= hour < 20: return 2
    return 3
```

Rồi đổi **hai** chỗ gọi (dòng ~130 trong `build_node_features_tomtom` và dòng ~163 trong
`build_node_features_osrm_proxy`):

```python
# cu:  time_list.append(TIME_LABEL_MAP.get(hour, 3))
time_list.append(time_label_of(hour))
```

Xoá hẳn `TIME_LABEL_MAP` để không ai lỡ tay dùng lại.

### Kiểm chứng ngay tại chỗ, trước khi rebuild

```python
>>> Counter(time_label_of(h) for h in range(24))
Counter({3: 12, 0: 6, 2: 4, 1: 3})
```

6 giờ đêm + 3 giờ cao điểm sáng + 4 giờ cao điểm chiều + 11 giờ thường + giờ 6 = 24. ✔

---

## 5. Vì sao sửa bằng hàm chứ không phải sửa `range`

Sửa tối thiểu sẽ là đổi dòng cuối thành `range(10, 16)` + `range(20, 24)` + thêm `6`.
Chạy được, nhưng **để nguyên cái bẫy**:

- Cấu trúc `{**a, **b, **c}` với các khoảng chồng lấn là một **cái bẫy im lặng** — không
  warning, không error, chỉ âm thầm sai. Người sau thêm một khoảng nữa là dính lại y hệt.
- Đọc `range(10,16)` không nói lên "đây là giờ hành chính"; đọc `if 16 <= hour < 20: return 2`
  thì nói lên ngay.
- Hàm cho phép viết docstring neo vào generator ⇒ chống lệch giữa hai file, tức **sửa luôn
  bug thứ cấp ở §2**, thứ mà sửa `range` không đụng tới.

**Phương án tốt hơn nữa (chưa làm bây giờ):** đọc thẳng cột `time_label` từ CSV và map
chuỗi → số, để chỉ còn **một** nguồn sự thật. Chưa làm vì nhánh fallback
`build_node_features_osrm_proxy` không có cột đó, nên vẫn cần hàm. Ghi vào việc sau.

---

## 6. ⚠️ Hệ quả: sửa xong là kết quả cũ hết hiệu lực

Fix này đổi nội dung `graph_dataset.pt` ⇒ **mọi số liệu đã có không còn so sánh được**:
`ablation_results.csv`, `baseline_results.csv`, `all_results.csv`,
`multiseed_runs.csv` (180 runs), `multistep_results.csv` (24 runs).

Quy tắc khi commit:

1. **Commit riêng một mình**, không gộp chung với thay đổi khác
2. Message nói rõ nó invalidate cái gì, ví dụ:

```
fix(build_graph): TIME_LABEL_MAP nuot mat nhan rush hour

Dict literal merge khien range(6,24) ghi de nhan 1 va 2.
time_labels chi con {0, 3} thay vi {0,1,2,3}.

BREAKING: doi noi dung graph_dataset.pt. Moi ket qua trong
data/results/ sinh truoc commit nay khong con so sanh duoc.
Can chay lai truoc khi dua so vao paper.
```

3. Rebuild dataset trước khi ai chạy experiment mới:

```bash
python scripts/dev/generate_synthetic_traffic.py   # tai sinh raw (seed 42, deterministic)
python scripts/build_graph.py --t_out 3            # rebuild dataset
```

> **Thứ tự quan trọng:** làm fix này **trước** khi Người 4 chạy `eval_non_iid.py`.
> Sửa sau = Người 4 phải chạy lại toàn bộ experiment.

---

## 7. Liên quan

- `docs/00_dataset_provenance.md` §2 — bug này chính là manh mối đầu tiên dẫn tới phát hiện synthetic
- `data/results/MULTISEED_FINDINGS.md §2.3` — `zone_full_tc` kém ổn định, nay đã có một phần lời giải
