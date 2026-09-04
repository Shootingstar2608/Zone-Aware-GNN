# 00 — Nguồn gốc dataset: `graph_dataset.pt` là dữ liệu SYNTHETIC

**Ngày phát hiện:** 2026-09-04
**Mức độ:** 🔴 Chặn — ảnh hưởng tới Data Card, phần Experiments, và narrative của toàn bộ paper
**Trạng thái:** cần cả nhóm quyết định (xem §6)

---

## 1. TL;DR

Toàn bộ kết quả thực nghiệm của nhóm (ablation, multi-seed 180 runs, multi-step 24 runs,
zone-stratified, JSD Non-IID evidence) được tính trên `data/processed/graph_dataset.pt`.

File đó **không phải dữ liệu TomTom**. Nó được sinh ra bởi
`scripts/dev/generate_synthetic_traffic.py` — một script mô phỏng viết tay.

Đã xác minh bằng cách **tái tạo lại dataset từ script và so sánh từng ô**:

```
shape so sanh : (637, 17) vs (637, 17)
max abs diff  : 0.000000
mean abs diff : 0.000000
khop hoan toan (< 1e-6) : True
```

Khớp **bit-for-bit** trên toàn bộ 10.829 giá trị. Đây là xác định, không phải suy đoán.

---

## 2. Phát hiện ra bằng cách nào

Chuỗi suy luận, ghi lại để ai cũng kiểm chứng được:

### Bước 1 — `time_labels` chỉ có 2 giá trị

```python
Counter({3: 480, 0: 157})
```

Nhãn `1` (cao điểm sáng) và `2` (cao điểm chiều) **không tồn tại**.
Nguyên nhân là bug `TIME_LABEL_MAP` (xem `docs/02`). Nhưng điều đáng chú ý hơn:
nhãn `0` ứng với giờ 0–5, nhãn `3` ứng với giờ 6–23 — nghĩa là `time_labels`
thực chất là một **đồng hồ ngày/đêm**.

### Bước 2 — Cấu trúc thời gian hoàn hảo một cách bất thường

Phân tích run-length của `time_labels`:

```
[(0,13), (3,72), (0,24), (3,72), (0,24), (3,72), (0,24),
 (3,72), (0,24), (3,72), (0,24), (3,72), (0,24), (3,48)]
```

- Mỗi khối đêm đúng **24** snapshot = 6 giờ x 4 → **15 phút/snapshot**
- Mỗi khối ngày đúng **72** snapshot = 18 giờ x 4 → nhất quán
- Lặp lại **7 lần**, không sai một snapshot nào

Dữ liệu API thật **không bao giờ** đều như vậy: luôn có jitter, timeout, snapshot thiếu.
(So sánh: file TomTom thật trên máy có timestamp kiểu `2026-07-28 15:44:20` — lệch giây.)

### Bước 3 — Khôi phục đồng hồ tuyệt đối

Khối đêm đầu tiên chỉ có 13 snapshot (thay vì 24) → chuỗi bắt đầu giữa khối đêm.
Window `s` mang nhãn của snapshot `s + T_in - 1 = s + 11`. Giải ra:

$$\text{snapshot } 0 = 00\text{:}00, \qquad \text{hour}(j) = \left\lfloor j/4 \right\rfloor \bmod 24$$

Tức chuỗi bắt đầu **đúng 00:00**, 672 snapshot = **đúng 7 ngày**.

### Bước 4 — Kiểm tra dấu vân tay ngữ nghĩa

Giả định ngày 0 = thứ Hai, tính congestion trung bình:

| Khung giờ | Ngày thường | Cuối tuần | Chênh |
|---|---|---|---|
| Đêm 0–5h | 1.0133 | 1.0126 | +0.0007 |
| Cao điểm sáng 7–9h | **1.7947** | **1.0122** | **+0.7825** |
| Hành chính 9–16h | 1.2154 | 1.2570 | −0.0416 |
| Cao điểm chiều 16–19h | 1.9464 | 1.4851 | +0.4614 |
| Tối 19–22h | 1.2284 | 1.4827 | −0.2543 |

Cuối tuần **hoàn toàn không có** đỉnh sáng (1.0122 = đúng mức free-flow), còn tối cuối
tuần lại **cao hơn** ngày thường. Đây chính xác là logic viết tay trong generator.

### Bước 5 — Tái tạo bit-exact

Chạy lại generator (`np.random.seed(42)`, `start_date = 2026-05-18` = thứ Hai) từ
`hcm_osrm_dataset.csv` + `zone_labels.csv`, gộp theo node đúng cách `build_graph.py` làm,
rồi so với `.pt`. Kết quả ở §1. Script tái lập ở §5.

---

## 3. Đối chiếu các con số

| Dấu hiệu | `graph_dataset.pt` | `generate_synthetic_traffic.py` |
|---|---|---|
| Số snapshot | 672 | `steps_per_day=96 x num_days=7 = 672` |
| Khoảng cách | đúng 15 phút, không jitter | `interval_min = 15` |
| Mốc bắt đầu | 00:00 | `datetime(2026, 5, 18, 0, 0, 0)` (thứ Hai) |
| Congestion đêm 0–5h | 1.0133 | `if 0 <= hour < 5: cong_ratio = 1.0` |
| Đỉnh sáng cuối tuần | 1.0122 (không có) | nhánh weekend không có đỉnh sáng |
| Miền giá trị | [0.9608, 2.5426] | `clip(0.95, 3.5)` |
| Nhiễu | — | `np.random.normal(0, 0.08)` |

Còn `data/raw/tomtom_traffic.csv` **thật** trên máy (126 snapshot, 2026-07-28 15:44 →
2026-07-29 18:55, ~27 giờ) là một lần gọi API khác và **chưa từng được dùng để build dataset**.

> ⚠️ Lưu ý về tên file: generator ghi output ra **đúng đường dẫn**
> `data/raw/tomtom_traffic.csv`. Nghĩa là dữ liệu mô phỏng đã **ghi đè lên đúng chỗ**
> mà dữ liệu TomTom thật lẽ ra phải nằm. Đây nhiều khả năng là cách sự nhầm lẫn phát sinh —
> không ai cố tình, nhưng hệ quả thì vẫn vậy.

---

## 4. Hệ quả

### 4.1. Claim bị chết

> "Zone-Aware GNN thắng baselines trên dữ liệu giao thông thực tại TP.HCM"

Generator viết thẳng quan hệ mà model được giao nhiệm vụ "khám phá":

```python
if 7 <= hour < 9:
    if z_v["school"] or z_v["university"]: cong_ratio += 0.40
    if z_v["industrial"]:                  cong_ratio += 0.35
    if z_u["residential"]:                 cong_ratio += 0.20
```

Model được cho xem chính `z_v` đó. **Đây là lập luận vòng tròn** — model đang khám phá lại
bộ sinh dữ liệu của chính nó. Phần "JSD chứng minh tính Non-IID" cũng vậy: nó đo độ lệch
do chính rule viết tay tạo ra, không phải đặc tính của giao thông TP.HCM.

### 4.2. Nghịch lý ablation được giải thích — KHÔNG PHẢI BUG

`data/results/MULTISEED_FINDINGS.md §2.2` ghi nhận điều khó hiểu: `zone_concat` (đơn giản
nhất) thắng, còn `zone_weight` và `zone_adj` làm model **tệ đi có ý nghĩa thống kê**
(−18.9% và −17.0%, p = 0.030). Findings đề xuất "nhiều khả năng có bug thật".

**Không có bug.** Generator là hàm **cộng thuần** theo zone one-hot:

$$c = 1 + \sum_k \beta_k(t) \cdot z_k + \varepsilon$$

- **Nối** zone embedding vào feature ⇒ khớp đúng dạng hàm này ⇒ thắng
- **Điều biến nhân** (`zone_weight`) và **bias adjacency** (`zone_adj`) ⇒ sai dạng hàm,
  chỉ thêm 358k tham số trên 17 node ⇒ overfit ⇒ thua

> 🔧 **Hành động:** đừng ai bỏ thời gian debug `zone_weight` / `zone_adj`. Chúng không hỏng.
> Chúng chỉ giả định sai về cấu trúc của dữ liệu — mà dữ liệu đó là do ta tự viết ra.

### 4.3. Cái được cứu

Dữ liệu synthetic **không tự động giết** một bài benchmark. Rất nhiều benchmark Non-IID
dùng dữ liệu mô phỏng hoặc bán mô phỏng. Cái giết là **gọi sai tên nó**.

Hướng đi tuần này (xây benchmark Non-IID) thực ra là lối thoát đúng:

- Có generator ⇒ độ heterogeneity trở thành **núm vặn có ground truth**
- Có thể sinh nhiều kịch bản, nhiều thành phố giả lập, lặp lại được 100%
- Concept Drift làm ở **tầng generator** thay vì hack vào tensor ⇒ nhất quán vật lý

Điều kiện: gọi nó là **simulator**, không phải TomTom.

### 4.4. Blocker tự tan

Trong bản đánh giá trước có nêu "mất file raw, không rebuild được dataset" là lỗi chí mạng.
**Không còn nữa:**

- File raw **tái sinh được** — chạy lại generator, `np.random.seed(42)` cố định ⇒ ra đúng file cũ
- `hour` / `dow` **khôi phục chính xác**, không cần rebuild:

$$\text{hour}(j) = \left\lfloor j/4 \right\rfloor \bmod 24, \qquad
\text{dow}(j) = \left\lfloor j/96 \right\rfloor \bmod 7, \qquad \text{ngày } 0 = \text{thứ Hai}$$

⇒ Kịch bản **Temporal Shift (weekday → weekend)** của Người 2 unblock hoàn toàn.

---

## 5. Script tái lập (ai cũng chạy lại được)

Lưu thành `scripts/dev/verify_provenance.py`, chạy từ thư mục gốc repo:

```python
import pandas as pd, numpy as np, torch
from datetime import datetime, timedelta

df_osrm = pd.read_csv("data/raw/hcm_osrm_dataset.csv")
edges = df_osrm.groupby(["origin","destination"])[["distance_m","duration_s"]].mean().reset_index()
zn = pd.read_csv("data/raw/zone_labels.csv", index_col="node")

start, n_steps = datetime(2026,5,18,0,0,0), 672
np.random.seed(42)
noise = np.random.normal(0.0, 0.08, size=(n_steps, len(edges)))

u, v = edges["origin"].values, edges["destination"].values
zc = lambda names, col: zn.loc[names, col].values.astype(bool)
zu_res, zu_sch, zu_ind = zc(u,"residential"), zc(u,"school")|zc(u,"university"), zc(u,"industrial")
zv_sch, zv_ind, zv_res = zc(v,"school")|zc(v,"university"), zc(v,"industrial"), zc(v,"residential")
zv_tra, zv_com, zv_hos, zv_park = zc(v,"transport"), zc(v,"commercial"), zc(v,"hospital"), zc(v,"park")

out = np.zeros((n_steps, len(edges)))
for s in range(n_steps):
    ts = start + timedelta(minutes=15*s); h, wk = ts.hour, ts.weekday() >= 5
    c = np.ones(len(edges))
    if not wk:
        if   7 <= h < 9:  c += 0.25 + 0.40*zv_sch + 0.35*zv_ind + 0.20*zu_res + 0.20*zv_tra
        elif 16 <= h < 19: c += 0.30 + 0.35*zu_sch + 0.45*zu_ind + 0.25*zv_res + 0.25*zv_com
        elif 9 <= h < 16:  c += 0.10 + 0.15*(zv_com | zv_hos)
        elif 19 <= h < 22: c += 0.25*zv_com + 0.15*zv_park
    else:
        if   10 <= h < 14: c += 0.15 + 0.35*zv_com
        elif 17 <= h < 21: c += 0.20 + 0.45*zv_com + 0.25*zv_park + 0.15*zv_res
    if 0 <= h < 5: c = np.ones(len(edges))
    out[s] = np.clip(np.round(c + noise[s], 3), 0.95, 3.5)

nodes = sorted(df_osrm["origin"].unique().tolist()); n2i = {n:i for i,n in enumerate(nodes)}
src = np.array([n2i[x] for x in u])
cnt = np.bincount(src, minlength=len(nodes))
node_c = np.zeros((n_steps, len(nodes)))
for e in range(len(edges)):
    node_c[:, src[e]] += out[:, e]
node_c /= np.maximum(cnt, 1)

d = torch.load("data/processed/graph_dataset.pt", weights_only=False)
X = d["X"].numpy(); S, N, _ = X.shape
real = X.reshape(S, N, 12, 4)[:, :, 0, 0]
diff = np.abs(real - node_c[:S, :])
print("max abs diff :", diff.max())
print("khop hoan toan:", bool(diff.max() < 1e-6))
```

> Ghi chú: script dùng `X[:, :, 0, 0]` — tức `congestion_ratio` tại bước đầu của mỗi cửa sổ,
> ứng đúng snapshot `j`. Ba feature còn lại đều là hàm của `congestion_ratio` và `base_dur`
> nên không cần kiểm riêng.

---

## 6. Việc phải quyết (cả nhóm, trước khi Người 4 viết Data Card)

Người 4 sắp soạn `docs/DATA_CARD.md` mô tả "3 nguồn dữ liệu: TomTom, OSRM, OSM" kèm phần
Licensing và Privacy. Nếu viết "TomTom Routing API, 672 snapshots" thì nhóm **đưa số liệu
không có thật vào paper** — thứ không sửa được sau khi nộp.

Ba câu phải chốt:

1. **Dataset gọi là gì trong paper?**
   Đề xuất: `HCM-Sim` — *a rule-based traffic simulator over a real 17-node OSRM road graph
   of Ho Chi Minh City*. Nhấn: cấu trúc đồ thị (OSRM) và nhãn vùng (OSM) **là thật**;
   chuỗi traffic động là **mô phỏng**.

2. **Bảng kết quả cũ: giữ hay bỏ?**
   Đề xuất: **giữ nhưng đổi vai trò** — không còn là "bằng chứng model tốt", mà là
   *sanity check on simulated data*, và nói thẳng rằng zone effect là do generator đặt vào.
   Kèm §4.2 như một quan sát về matched vs mismatched inductive bias.

3. **Có thu dữ liệu TomTom thật không?**
   Collector đã có sẵn (`scripts/tomtom_collector.py`) và đã chạy được 27 giờ.
   Nếu muốn giữ claim về dữ liệu thật thì phải chạy đủ ≥ 2 tuần → cần quyết **ngay hôm nay**
   vì nó chạy nền mất nhiều ngày.

---

## 7. Liên quan

- `docs/02_bugfix_time_label_map.md` — bug làm mất nhãn giờ cao điểm (phát hiện trong cùng lượt)
- `docs/03_benchmark_partition_design.md` — vì sao thiết kế partition không bị dính vòng tròn này
- `data/results/MULTISEED_FINDINGS.md §2.2` — nghịch lý ablation, nay đã có lời giải
