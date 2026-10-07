# 05 — Nhật ký công việc: Người 2 (Khoa) — Benchmark Non-IID

**Giai đoạn:** 04/09/2026 → 21/09/2026
**Phạm vi:** `benchmark/`, `tests/`, `scripts/`, `.github/workflows/`, 2 bugfix chặn ở `scripts/`
**Trạng thái:** 4 cơ chế phân vùng + input masking v2 + 27 unit test + CLI (31 partition) + CI
3/3 job xanh — **đóng toàn bộ DoD §4.2 của plan tuần**

---

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Điều tra: dataset là synthetic](#2-điều-tra-dataset-là-synthetic)
3. [Ba bugfix chặn](#3-ba-bugfix-chặn)
4. [Thiết kế benchmark](#4-thiết-kế-benchmark)
5. [Bốn cơ chế phân vùng](#5-bốn-cơ-chế-phân-vùng)
6. [Input Masking v2](#6-input-masking-v2)
7. [Unit tests](#7-unit-tests)
8. [Ba chỗ lệch so với plan](#8-ba-chỗ-lệch-so-với-plan)
9. [Issues còn treo](#9-issues-còn-treo)
10. [Câu hỏi cho từng thành viên](#10-câu-hỏi-cho-từng-thành-viên)
11. [Lịch sử commit](#11-lịch-sử-commit)
12. [CLI sinh partition + CI (21/09)](#12-cli-sinh-partition--ci-2109)

---

## 1. Bối cảnh

Nhóm 4 người, đề tài Zone-Aware AH-GNN dự báo giao thông đô thị. Tuần này pivot từ
*"đề xuất một model mới"* sang *"xây benchmark Non-IID"*. Vai trò:

| Người | Nhiệm vụ |
|---|---|
| 1 | Schema chuẩn hoá + Data Access Interface (Node-Agent) |
| **2 (Khoa)** | **Bộ sinh phân vùng Non-IID** |
| 3 | Đánh giá chất lượng dữ liệu (5 tiêu chí) |
| 4 | Eval benchmark + Data Card + QA |

Trạng thái lúc bắt đầu: dataset 17 nút × 8 zone type × 637 cửa sổ, 9 model đã train,
180 runs multi-seed có kiểm định Holm, 24 runs multi-step qua 6 horizon.

---

## 2. Điều tra: dataset là synthetic

### 2.1. Kết luận

`data/processed/graph_dataset.pt` — **nền tảng của toàn bộ kết quả thực nghiệm** — không
phải dữ liệu TomTom. Nó được sinh bởi `scripts/dev/generate_synthetic_traffic.py`.

Xác minh bằng cách tái tạo lại từ script rồi so từng ô:

```
shape so sánh : (637, 17) vs (637, 17)
max abs diff  : 0.000000
mean abs diff : 0.000000
khớp hoàn toàn (< 1e-6) : True
```

Khớp **bit-for-bit** trên 10.829 giá trị.

### 2.2. Chuỗi phát hiện

| Bước | Manh mối |
|---|---|
| 1 | `time_labels` chỉ có 2 giá trị `{0: 157, 3: 480}` thay vì 4 |
| 2 | Run-length hoàn hảo: 24 snapshot đêm / 72 snapshot ngày, lặp đúng 7 lần, không sai một cái |
| 3 | Suy ngược đồng hồ: bắt đầu đúng 00:00, 672 snapshot = đúng 7 ngày, lưới 15 phút không jitter |
| 4 | Cuối tuần **không có** đỉnh cao điểm sáng (1.0122 = đúng free-flow) — trùng khít logic viết tay trong generator |
| 5 | Tái tạo bit-exact |

Đối chiếu:

| Dấu hiệu | `graph_dataset.pt` | Generator |
|---|---|---|
| Số snapshot | 672 | `96 × 7 = 672` |
| Khoảng cách | đúng 15 phút, không jitter | `interval_min = 15` |
| Mốc bắt đầu | 00:00 | `datetime(2026, 5, 18, 0, 0, 0)` (thứ Hai) |
| Congestion đêm | 1.0133 | `if 0 <= hour < 5: cong_ratio = 1.0` |
| Miền giá trị | [0.9608, 2.5426] | `clip(0.95, 3.5)` |

> ⚠️ Generator ghi output ra **đúng đường dẫn** `data/raw/tomtom_traffic.csv` — tức dữ liệu
> mô phỏng ghi đè lên đúng chỗ dữ liệu thật lẽ ra phải nằm. Đây là cách sự nhầm lẫn phát
> sinh. Không ai cố tình.

### 2.3. Hệ quả

**Claim bị chết:** *"Zone-Aware GNN thắng baselines trên dữ liệu giao thông thực HCM"*.
Generator viết thẳng quan hệ mà model được giao nhiệm vụ khám phá:

```python
if 7 <= hour < 9:
    if z_v["school"] or z_v["university"]: cong_ratio += 0.40
    if z_v["industrial"]:                  cong_ratio += 0.35
```

Model được cho xem chính `z_v` đó → **lập luận vòng tròn**.

**Nghịch lý ablation được giải thích — KHÔNG PHẢI BUG.** `MULTISEED_FINDINGS §2.2` ghi nhận
`zone_concat` (đơn giản nhất) thắng, còn `zone_weight` và `zone_adj` làm model tệ đi có ý
nghĩa thống kê (−18.9% và −17.0%, p = 0.030), và đề xuất *"nhiều khả năng có bug thật"*.

Không có bug. Generator là hàm **cộng thuần** theo zone one-hot:

$$c = 1 + \sum_k \beta_k(t) \cdot z_k + \varepsilon$$

- **Nối** zone embedding vào feature ⇒ khớp đúng dạng hàm ⇒ thắng
- **Điều biến nhân** và **bias adjacency** ⇒ sai dạng hàm, thêm 358k tham số trên 17 nút ⇒ overfit

🔧 **Đừng ai bỏ thời gian debug `zone_weight` / `zone_adj`.** Chúng không hỏng.

**Cái được cứu:** dữ liệu synthetic không tự động giết một bài benchmark. Cái giết là **gọi
sai tên nó**. Hướng benchmark thực ra là lối thoát đúng — có generator nghĩa là độ
heterogeneity trở thành **núm vặn có ground truth**.

**Blocker tự tan:** file raw tái sinh được (`np.random.seed(42)` cố định), và `hour`/`dow`
khôi phục chính xác:

$$\text{hour}(j) = \lfloor j/4 \rfloor \bmod 24, \qquad \text{dow}(j) = \lfloor j/96 \rfloor \bmod 7$$

⇒ kịch bản Temporal Shift unblock hoàn toàn, không cần rebuild dataset.

---

## 3. Ba bugfix chặn

### 3.1. `train.py` crash ở mọi chế độ

```python
def run_experiment(variant_name, meta, dataset_dict, ablation_cfg):   # thiếu lambda_cos
    effective_lambda_cos = lambda_cos if ...                          # NameError
...
    result = run_experiment(..., lambda_cos=args.lambda_cos)          # TypeError
```

Vết khâu từ lần merge `origin/ton` + `origin/khoa`. Lọt lưới vì mọi người chạy qua
`run_multistep.py` / `run_multi_seed.py`, không ai gọi `train.py` trực tiếp.

**Sửa:** thêm `lambda_cos=LAMBDA_COS` vào signature — 1 dòng, vá cả 2 lỗi, giữ nguyên hành
vi mặc định nên kết quả cũ vẫn so sánh được.

### 3.2. `TIME_LABEL_MAP` nuốt mất nhãn giờ cao điểm

```python
TIME_LABEL_MAP = {
    **{h: 0 for h in range(0, 6)},
    **{h: 1 for h in range(7, 10)},   # morning rush
    **{h: 2 for h in range(16, 20)},  # evening rush
    **{h: 3 for h in range(6, 24)},   # ← ghi đè cả hai dòng trên
}
```

Dict literal gộp trái→phải, key sau thắng. `range(6,24)` nuốt trọn `7,8,9` và `16..19`.
Kết quả: `time_labels` chỉ còn `{0: 157, 3: 480}` — một cái đồng hồ ngày/đêm.

**Hậu quả:** `zone_full_tc` (Tân) mất đúng thông tin nó sinh ra để nắm bắt — khớp với
`MULTISEED_FINDINGS §2.3` (tc kém ổn định hơn sinc, vì sinc dùng `hour`/`dow` liên tục
nên không dính bug). Và 2/4 embedding thời gian là tham số chết.

**Sửa bằng hàm** `time_label_of()` chứ không sửa `range`, vì:
- `{**a, **b, **c}` với khoảng chồng lấn là **bẫy im lặng** — không warning, người sau thêm khoảng nữa là dính lại
- Hàm cho viết docstring neo vào generator ⇒ sửa luôn bug thứ cấp *hai file định nghĩa khác nhau cho cùng một khái niệm*

### 3.3. 32 file dính CRLF

`git status` báo 34 file modified. Kiểm toán bằng `--ignore-all-space`:

```
THỰC SỰ: scripts/build_graph.py  (51 dòng)
THỰC SỰ: scripts/train.py        (13 dòng)
--- hết ---
```

32 file còn lại: **0 dòng thay đổi thật**, toàn `^M`. `core.autocrlf` chưa set.

Nếu push nguyên trạng: commit **+6473 / −6463 dòng**, trong đó ~64 dòng là thật. Hậu quả:
review không thấy gì, và **Bảo với Tân sẽ conflict trên từng file một** kể cả file họ
không đụng.

**Sửa:** `core.autocrlf true` + `git add --renormalize .` + `.gitattributes` với
`* text=auto` (để áp cho cả nhóm, không chỉ máy mình).

---

## 4. Thiết kế benchmark

### 4.1. Nguyên tắc bất di bất dịch

> **partition = index/mask + config + seed. KHÔNG BAO GIỜ copy tensor.**

Cách ngây thơ: mỗi kịch bản đẻ một bản sao `.pt` đã lọc. Nhưng 4 cơ chế × 4 mức × 10 seed
= **160 bản × 3 MB ≈ nửa GB**, repo phình, và không ai diff được hai partition.

Dữ liệu gốc không bị đụng một byte. Tất cả "Non-IID" nằm ở **hình dạng của mask**.

### 4.2. Câu hỏi thiết kế cốt lõi: client là gì?

Trong FL kinh điển mỗi sample thuộc về đúng một client. Ở đây:

```
X: (S=637, N=17, T_in*F=48)   ← 1 sample = 1 cửa sổ, CHỨA CẢ 17 NÚT
```

Model là GNN, một forward pass nuốt cả đồ thị. Nếu client = nút thì client **không sở hữu
sample nào** — nó sở hữu **một cột** của mọi sample.

Bốn phương án đã cân nhắc:

| # | Client | Dirichlet chia gì | Đụng chiều nút? | Người 4 chạy được? |
|---|---|---|---|---|
| **1** ✅ | 1 nút | **độ phủ thời gian mỗi nút** → mask `(S,N)` | ✅ | ✅ sửa ~5 dòng loss |
| 2 | 1 nút + FedAvg | cửa sổ mỗi agent | ✅ | ❌ chưa ai code FL |
| 3 | 1 cụm zone | cửa sổ mỗi cụm | ✅ | trùng cơ chế Zone Skew |
| 4 | K client ảo | sample theo class (FL kinh điển) | ❌ | vô nghĩa với paper |

**Chọn #1.** Bốn lý do, theo thứ tự quan trọng:

1. **Biến lựa chọn kỹ thuật thành thí nghiệm thật.** Nút ít dữ liệu buộc phải dựa vào
   message passing từ hàng xóm — đúng chỗ zone-awareness phải chứng minh giá trị. Và
   finding này **không dính vòng tròn**: generator mô phỏng quan hệ zone→congestion nhưng
   **không hề mô phỏng chuyện thiếu dữ liệu**.
2. **Có câu chuyện vật lý thật** — *sensor coverage skew*, hiện tượng citable trong mạng
   giao thông thực.
3. Đúng nguyên văn plan.
4. Người 4 tích hợp được trong một buổi chiều.

### 4.3. Tầng nền — 9 helper

| Hàm | Vai trò |
|---|---|
| `dataset_fingerprint()` + `check_fingerprint()` | SHA-256 của `.pt`. Nếu ai rebuild dataset, chỉ số cũ trỏ sai → **fail to tiếng** thay vì chạy im ra số sai |
| `gini(counts)` | Trục $x$ của figure chính. $G=0$ đều tuyệt đối, $G \to 1$ một nút ôm hết |
| `overlap_gap(meta)` | $T_{in} + T_{out} - 1 = 35$. **Đọc từ `meta.json`, không hardcode** |
| `assert_no_leakage()` | Kiểm khoảng cách nhỏ nhất giữa hai tập bằng `searchsorted` |
| `recover_clock(S, t_in)` | Khôi phục `(hour, dow)` — món quà từ vụ điều tra |
| `mask_hash()` | Verify tái lập mà không cần lưu mask vào JSON |
| `_water_fill()`, `_cluster_zones()`, `_jsd()` | Xem từng cơ chế |

**Vì sao `overlap_gap` quan trọng nhất:** dữ liệu là cửa sổ trượt — cửa sổ 0 dùng khung
`0–11`, cửa sổ 1 dùng `1–12`. Hai cửa sổ cạnh nhau **trùng 11/12 dữ liệu đầu vào**.
`train.py` dùng `random_split` → cửa sổ 100 ở train, 101 ở test → model **đã nhìn thấy gần
hết đáp án**. Đó là lý do MAE ≈ 0.08 đẹp bất thường, và là lỗ hổng `MULTISEED_FINDINGS §3`
cảnh báo nhưng chưa ai vá.

---

## 5. Bốn cơ chế phân vùng

### 5.1. `quantity_skew` — Coverage Skew

$$p \sim \text{Dir}(\alpha \cdot \mathbf{1}_N), \qquad n_v = \text{water-fill}(p, S, B)$$

**Vấn đề phát hiện lúc code:** clip $n_v$ ở trần $S$ làm **bốc hơi ngân sách**. Ở $\alpha$
nhỏ, nút top đòi $n_v \gg S$, phần thừa bị cắt và vứt → $\alpha$ nhỏ vừa *lệch hơn* vừa *ít
dữ liệu hơn* → trục $x$ của figure trộn hai biến.

**Water-filling:** phần vượt trần **chia lại** cho các nút chưa đầy, **lặp** đến khi không
ai vượt (chia lại có thể đẩy nút khác chạm trần). Làm tròn largest-remainder để tổng khớp
$B$ (`floor` thuần hụt tới $N$ đơn vị).

Đã thử **cả 3 cách đọc** rồi so bằng số:

```
mode             alpha         coverage             gini   n_zero
budget             0.1      1.000±0.000      0.000±0.000      0.0
budget             5.0      1.000±0.000      0.000±0.000      0.0   ← thoái hoá

relative           0.1      0.113±0.031      0.829±0.045      8.3
relative           5.0      0.556±0.081      0.199±0.049      0.0   ← confounded

fixed_coverage     0.1      0.500±0.000      0.487±0.011      3.7
fixed_coverage     0.5      0.500±0.000      0.416±0.031      0.3
fixed_coverage     1.0      0.500±0.000      0.387±0.032      0.0
fixed_coverage     5.0      0.500±0.000      0.198±0.049      0.0   ← CHỌN
```

- `budget` ($B = NS$): ngân sách đúng bằng trần tuyệt đối → water-filling buộc mọi nút phải
  đầy → Gini = 0 ở mọi $\alpha$. Cách đọc này **tự sụp** khi có phân phối lại.
- `relative`: coverage chạy song song Gini → MAE tăng thì không biết do biến nào.
- `fixed_coverage`: coverage `0.500 ± 0.000` ở mọi $\alpha$, Gini vẫn trải 0.198 → 0.487.
  **Đúng một biến thay đổi.**

Hai mode kia giữ lại làm **robustness check** cho appendix.

**Ba quyết định nhỏ:**

- **Cho phép nút 0 mẫu, không đặt sàn.** Đó là thí nghiệm mạnh nhất: một nút *chưa từng
  được giám sát*, model chỉ dự báo được bằng nhãn zone + cấu trúc đồ thị.
- **`block_len`** — sensor thật chết theo khối, không rụng lẻ tẻ. Khối cuối được phép ngắn
  hơn để tổng sức chứa vẫn đúng $S$.
- **$\bar c$ là núm điều khiển độ lệch tối đa:** $\lceil \bar c \cdot N \rceil$ = số nút
  tối thiểu phải có dữ liệu. Với $\bar c = 0.5, N = 17$ → tối đa 8 nút tối om → Gini chặn
  trên ~0.5. **Cố định cho toàn paper**, biến sweep là $\alpha$.

### 5.2. `zone_skew` — Feature Skew

Cụm nút theo vector zone, chia ngày thành $k$ dải giờ, cụm $c$ chủ yếu quan sát dải $c$.

**Quyết định then chốt: giữ số mẫu mỗi nút BẰNG NHAU TUYỆT ĐỐI** (318 cửa sổ, đo được
`gini = 0.00000`), chỉ khác **cửa sổ nào**.

⇒ **Trực giao** với `quantity_skew`: một cái đổi *bao nhiêu*, một cái đổi *loại gì*. Người
4 quét từng cái mà không lẫn.

```
off_band_weight   JSD giữa cụm
1.0 (không lệch)  0.0050
0.1 (mặc định)    0.2010
0.01              0.2478
0.0               ValueError — dải giờ không đủ chứa 318 cửa sổ
```

**`_cluster_zones` tự viết bằng numpy** (agglomerative average-linkage, cosine distance)
thay vì sklearn: 17 nút nên chi phí không đáng kể, đổi lại không phụ thuộc phiên bản
sklearn (API `metric`/`affinity` đã đổi vài lần) và tất định tuyệt đối.

**Vấn đề thật:** cụm tự động rất mất cân bằng.

```
k=2  sizes=[16, 1]       jsd=0.3500
k=3  sizes=[13, 3, 1]    jsd=0.2010
k=4  sizes=[11, 3, 1, 2] jsd=0.1231
```

High Tech Park (chỉ có mỗi `industrial`) luôn tách thành cụm 1 nút, vì 17 nút có vector
zone quá giống nhau (đa số là hỗn hợp commercial + residential).

→ Thêm tham số `labels` cho gán thủ công. Kết quả tốt hơn hẳn: `[14, 3]`, JSD `0.345`, cụm
có nghĩa rõ ràng (High Tech Park, Suoi Tien, Tan Son Nhat = nút công nghiệp/giao thông).
**Khuyến nghị dùng bản thủ công**, ghi tiêu chí gán vào paper.

### 5.3. `temporal_shift`

```
train  0..338    (339)  thứ 2–5
       ← gap 36 →
val    374..433  (60)   thứ 6
       ← gap 36 →
test   469..636  (168)  thứ 7 + CN
burn   70 cửa sổ vào 2 khoảng trống
```

**Ba quyết định:**

1. **Val lấy từ phân phối TRAIN (ngày thường).** Nếu val là cuối tuần thì early-stopping và
   chọn hyperparameter đang **nhìn thấy phân phối test** — leakage kiểu khác, tinh vi hơn,
   và phá đúng thứ benchmark distribution-shift muốn đo.
2. **Dựng lùi từ test.** Test là tài nguyên khan hiếm cố định (168 cửa sổ). Dựng xuôi thì
   train ăn hết chỗ.
3. **Lọc theo `dow` chứ không `np.arange`.** 1 tuần thì liền mạch, nhưng 3 tuần sẽ đứt quãng.

**Scale theo horizon** — phải ghi vào paper, đừng so MAE giữa các $T_{out}$ mà lờ đi:

| $T_{out}$ | gap | train | val | test | burned |
|---|---|---|---|---|---|
| 3 | 14 | 381 | 60 | 168 | 28 |
| 12 | 23 | 363 | 60 | 168 | 46 |
| 24 | 35 | 339 | 60 | 168 | 70 |

### 5.4. `concept_drift`

**Trả về SPEC, không sửa tensor.**

```json
{"kind": "incident",
 "target_nodes": [1, 11, 12],
 "window_start": 579, "window_end": 618,
 "profile": [1.0, 1.1, ..., 1.8, 1.8, ..., 1.1],
 "applies_to": "congestion_ratio (generator suy ra delay và travel_time)"}
```

**Vì sao:** `congestion_ratio`, `traffic_delay_s`, `travel_time_s` **ràng buộc nhau về vật
lý** (`travel_time = base_dur × cong_ratio`, `delay = travel_time − base_dur`). Sửa mỗi
delay mà không sửa hai cái kia → quan hệ input-output phi vật lý → model học được cũng vô
nghĩa. Vì dữ liệu là synthetic nên có generator để áp ở tầng sinh, giữ nhất quán tự động.

Sự cố đặt **trong test set** — drift phải là thứ model chưa từng thấy lúc train.

---

## 6. Input Masking v2

### 6.1. v1 chưa đủ

**v1 (loss masking)** chỉ bỏ chấm điểm nút thiếu dữ liệu, nhưng **đặc trưng vẫn nằm nguyên
trong input**. Model vẫn nhìn thấy đầy đủ 12 bước lịch sử, chỉ không bị phạt. Tạo ra lệch
gradient (đúng bản chất quantity skew), nhưng **không tạo ra bài toán thiếu thông tin**.

**v2** che luôn đặc trưng. Nút tối om thật sự tối, cách duy nhất để dự báo là qua message
passing từ hàng xóm.

### 6.2. ⚠️ Chỗ plan nói ngược

Plan viết: *"thay thế bằng cờ ẩn **hoặc giá trị nội suy**, ép mô hình phải khai thác Zone
Adjacency"*.

**Nội suy không ép được gì — nó làm bài toán DỄ đi.** Nếu ta nội suy giá trị nút thiếu từ
hàng xóm rồi đưa vào input, ta **đã làm hộ model đúng cái việc muốn nó học**. Model nhận
tensor đầy đủ, không lỗ hổng, không có động lực dùng đồ thị.

**Cách xử lý:** biến nội suy thành **baseline** thay vì điều kiện thí nghiệm.

| Chiến lược | Vai trò |
|---|---|
| **`flag`** (mặc định) | **Điều kiện chính.** Thêm kênh availability, model phân biệt được "không có dữ liệu" với "đường thoáng", phải tự suy phần thiếu |
| `neighbor` | **Baseline: nội suy theo đồ thị.** Cận trên của bắc cầu thủ công — model zone-aware phải vượt được |
| `last_seen` | **Baseline: nội suy theo thời gian** |
| `zero` | **Baseline ngây thơ.** Không có kênh báo |

Cách này **tốt hơn cho paper**: thay vì một con số, có bảng bốn dòng. Nếu `flag` thắng
`neighbor` thì đó là bằng chứng model học được thứ nội suy thủ công không làm được.

### 6.3. Quyết định + cảnh báo

**Che ở mức mẫu, không mức snapshot.** Các cửa sổ chồng lấn nên cùng một snapshot có thể
hiện ở cửa sổ này, ẩn ở cửa sổ kia. Không nhất quán với câu chuyện "sensor chết", nhưng:
nhất quán với v1, diễn giải vẫn hợp lệ (mỗi mẫu là một "cơ hội quan sát"), và mức snapshot
cần đổi API của cả 4 cơ chế. **Hạn chế đã biết**, ghi vào Data Card.

**`flag` đổi `in_channels: 48 → 60`** ⇒ **mọi checkpoint cũ không load được.**

**`fill_value` phụ thuộc việc Tôn có chuẩn hoá chưa:**

| Kênh | Miền thật |
|---|---|
| `congestion_ratio` | 0.96 – 2.5 |
| `traffic_delay_s` | 0.03 – 1471 |
| `travel_time_s` | 535 – 2414 |

Hiện `fill_value = 0` cho `travel_time_s` **nằm ngoài miền thật hoàn toàn** → model nhận ra
ngay → `zero` vô tình hoạt động gần giống `flag` → phép so sánh mất ý nghĩa. Sau z-score,
`0` = giá trị trung bình, lúc đó `zero` mới thật sự là baseline ngây thơ.

> 🔗 **Bảng so 4 chiến lược chỉ nên chạy SAU khi z-score vào.**

### 6.4. Phát hiện phụ: `ff_ratio` gần trùng `congestion_ratio`

```
max |ff_ratio − congestion_ratio| = 0.00027   (chỉ là sai số làm tròn)
```

Vì generator tính `travel_time = base_dur × cong_ratio` và `free_flow = base_dur`, nên
`ff_ratio = cong_ratio` **theo định nghĩa**. ⇒ $F = 4$ nhưng chỉ **3 kênh độc lập**.

Liên quan: chuẩn hoá của Tôn (chuẩn hoá một kênh trùng lặp), và giải thích một phần vì sao
358k tham số overfit trên 17 nút.

### 6.5. Số liệu

```
alpha  frac_masked  nút tối om  in_channels
 0.1        0.500           4           60
 0.5        0.500           0           60
 1.0        0.500           0           60
 5.0        0.500           0           60
```

`frac_masked` đúng `0.500` ở mọi $\alpha$ — bất biến `fixed_coverage` hiện lên ở tầng input.

---

## 7. Unit tests

`tests/test_benchmark.py` — **27 test, pass hết**. Chạy được cả hai kiểu, không bắt buộc pytest:

```bash
python tests/test_benchmark.py
pytest tests/test_benchmark.py -v
```

| Nhóm | Số | Kiểm gì |
|---|---|---|
| Gini | 2 | Biên + đơn điệu theo $\alpha$ |
| `quantity_skew` | 5 | Trần $S$, giữ ngân sách, stats khớp mask, `block_len`, mode sai raise |
| `zone_skew` | 3 | **Gini = 0**, JSD tăng khi `off_band_weight` giảm, `labels` thủ công |
| `temporal_shift` | 5 | **Không rò rỉ 3 cặp**, val là ngày thường, 3 tập rời, gap scale, `normal_to_rush` bị từ chối |
| `concept_drift` | 2 | Nằm trong test, profile hình thang |
| Tái lập | 3 | Cùng seed → cùng hash, **không đọc rng toàn cục** |
| Schema | 1 | Bản ghi đủ 12 trường, serialize được |
| Input masking | 5 | `flag` thêm kênh đúng, đặc trưng thật sự bị xoá, `neighbor` không để lỗ |

**Ba test đáng chú ý:**

- `test_quantity_skew_stats_khop_mask` — đếm lại từ chính mask. Chính kiểu test này bắt
  được lỗi mask rỗng (stats đẹp long lanh trong khi mask toàn `False`).
- `test_khong_dung_rng_toan_cuc` — gọi `np.random.seed()` hai giá trị khác nhau rồi kiểm
  mask có đổi không. Vì `train.py` có `set_seed(42)`, nếu module lỡ dùng rng toàn cục thì
  thứ tự gọi hàm sẽ đổi kết quả mà **không báo lỗi gì**.
- `test_temporal_shift_khong_ro_ri` — kiểm cả 3 cặp, không chỉ train–test.

---

## 8. Ba chỗ lệch so với plan

| Plan gốc | Thực tế | Lý do |
|---|---|---|
| "Dirichlet phân bổ số mẫu giữa các nút" | Diễn giải thành **coverage skew** + water-filling | Câu gốc không có nghĩa hiển nhiên vì client không chia nhau pool rời rạc |
| "Temporal Shift: Normal → Rush hours" | **Bỏ**, thay bằng `rush_mask()` làm metric | **Bất khả thi**: `0/449` cửa sổ normal sống sót. Rush/normal xen kẽ chu kỳ 6 giờ = 24 cửa sổ, nhỏ hơn gap 35 |
| "Concept Drift: tạo kịch bản đột biến" | Trả về **spec** cho generator | Sửa tensor phá ràng buộc vật lý giữa 3 trường |
| "thay bằng cờ ẩn **hoặc nội suy** để ép model dùng đồ thị" | `flag` là điều kiện chính, nội suy **hạ xuống baseline** | Nội suy hộ model thì model không cần học nữa |

Về `normal_to_rush`: ngay cả ở $T_{out} = 3$, 112 cửa sổ sống sót **toàn bộ là đêm sâu**
(00:00–03:30 và 23:30–24:00) — chỗ duy nhất cách rush đủ xa. Kịch bản biến thành *"đêm →
cao điểm"*, claim khác hẳn và yếu hơn nhiều.

Cả 4 chỗ lệch đều **có số liệu chống lưng**, ghi trong `docs/khoa/03` §9 và `04`.

---

## 9. Issues còn treo

### 🔴 P0

**Dataset chưa rebuild sau fix `TIME_LABEL_MAP`.** `meta.json` vẫn `S=637, T_out=24`,
`time_labels` vẫn `{0, 3}`. Fix nằm trong git nhưng **chưa có hiệu lực** — cả nhóm vẫn chạy
trên dữ liệu còn bug.

**`data/raw/tomtom_traffic.csv` giờ là bản THẬT 126 snapshot** (27 giờ), không phải synthetic
672. Ai chạy `build_graph.py` sẽ ra $S = 126 - 12 - 24 + 1 = 91$ thay vì 637 — dataset khác
hoàn toàn, mọi kết quả đã commit thành vô nghĩa, và **không ai biết vì sao** vì tên file y hệt.
→ **Đổi tên ngay**: `tomtom_real_2026-07-28.csv` và `tomtom_sim_672.csv`. `build_graph.py`
nên nhận `--source` thay vì hardcode.

**Ba câu quyết định dataset** (`docs/khoa/00` §6) — chặn Data Card của Người 4.

### 🟠 P1

- ~~`partitions_meta.json` vẫn còn 3 `PLACEHOLDER`~~ → **xong 21/09**, xem §12.1: CLI sinh
  31 bản ghi thật, `verify_partitions.py` xác nhận tái lập bit-for-bit
- ~~Chưa ai chạy `chrono` split~~ → **xong 20–21/09**. 10 seed × 9 model dưới chronological
  split có purge gap. Kết quả đảo thứ hạng: `lstm` ngang `zone_full_sinc` về MAE, mọi so sánh
  đều không đạt ý nghĩa sau Holm. Phần thực nghiệm của paper phải viết lại
- Nhiệm vụ Người 3 gần như vô nghĩa trên dữ liệu synthetic (xem §10)

### 🟡 P2

| Issue | Ghi chú |
|---|---|
| Ablation paradox chưa chốt hướng | `docs/khoa/00` §4.2 giải thích tại sao **không nên debug** |
| `train.py` vẫn `random_split`, chưa có gap | Việc của Tôn tuần này |
| Chưa bật collector TomTom | Việc duy nhất có deadline vật lý |
| `from torchgen import model` — `train.py` dòng 26 | Import thừa, máy khác có thể `ImportError` |
| Nút 0 mẫu chưa quyết xử lý | Nếu training NaN thì đặt sàn hay xử ở tầng eval? |

### Câu hỏi thiết kế còn treo

- **Node-Agent có nghĩa federated thật không?** Nếu có thì partition phải thiết kế lại và
  `train.py` centralized phải viết mới. Càng hỏi muộn càng đắt.
- **Che ở mức mẫu hay snapshot?** Hiện mức mẫu, có lý do, nhưng là hạn chế đã biết.

---

## 10. Câu hỏi cho từng thành viên

### 🔴 Người 3 — quan trọng nhất, chưa ai nhận ra

| Tiêu chí | Trên dữ liệu synthetic |
|---|---|
| Completeness | luôn **100%** — generator không bỏ sót snapshot |
| Freshness | luôn **đúng 15 phút** — lưới đều tuyệt đối |
| Consistency (OSRM ↔ TomTom) | **hoàn hảo by construction** — `travel_time = base_dur × cong_ratio` |
| Validity | luôn hợp lệ — có `clip(0.95, 3.5)` |
| Source Reliability | chỉ **một** nguồn, không đối chiếu được |

**Cả 5 chỉ số đều vô nghĩa.** Không phải lỗi của Người 3, nhưng họ đang chuẩn bị làm một
tuần cho thứ sẽ ra toàn số hoàn hảo.

*Gợi ý:* viết module trước, chạy trên 27 giờ dữ liệu thật đã có, coi như smoke test.

### 🟠 Người 1 — schema

1. Schema mới có `timestamp` thật không? Nếu có thì bỏ `recover_clock` đi, đọc thẳng cho chắc.
2. `client.get_local_features(time_range)` — `time_range` theo timestamp hay chỉ số cửa sổ?
   Partition làm việc bằng **chỉ số cửa sổ**.
3. "Node-Agent" có nghĩa chuyển sang federated thật không?

### 🟡 Người 4

1. Đồng ý loss masking cho v1 không? (v2 `flag` đã sẵn sàng nhưng vỡ checkpoint)
2. Bao nhiêu seed? Ước tính ~12 giây/run trên CPU 2 nhân — tính tổng trước khi bấm chạy.
3. Split protocol nào là chuẩn — `chrono` chạy trước hay sau benchmark?

### 🟡 Tân & Bảo

1. `zone_full_tc` phụ thuộc `time_labels` bị bug. Sau khi fix, có train lại không? Ai train?
2. Ai còn giữ file `tomtom_traffic.csv` bản 672 snapshot không?

### 🔴 Cả nhóm — 3 câu chặn Data Card

| Câu | Option khuyên dùng |
|---|---|
| **Tên dataset** | `HCM-Sim` — *rule-based traffic simulator over a real 17-node OSRM road graph of HCMC*. Nhấn: đồ thị (OSRM) và nhãn vùng (OSM) **là thật**, chuỗi traffic là mô phỏng. Và **đổi contribution** sang benchmark |
| **Bảng kết quả cũ** | **Giữ, đổi vai trò** — không còn là "bằng chứng model tốt" mà là *sanity check on simulated data*. Biến §4.2 thành finding về **matched vs mismatched inductive bias** — chỉ nói được vì ta biết DGP |
| **Thu TomTom thật?** | **Bật collector hôm nay**, quyết sau. Bất đối xứng: bắt đầu tốn gần như không gì và lúc nào cũng vứt được; không bắt đầu thì 3 tuần nữa **không có cách nào bù** |

**Không thương lượng:** không được ghi *"TomTom Routing API, 672 snapshots"* vào paper.

---

## 11. Lịch sử commit

| Commit | Nội dung |
|---|---|
| `329d929` | Contract bộ sinh phân vùng + 2 bugfix chặn (`train.py`, `TIME_LABEL_MAP`) + `.gitattributes` |
| `7b169a3` | checkpoint |
| `2977660` | Hiện thực cả 4 cơ chế Non-IID + xoá stub `concept_drift` trùng + sweep script |
| `7050640` | Input masking v2 + 27 unit test + `docs/khoa/04` |
| `4afea6e` | `docs/khoa/05` — worklog giai đoạn 04/09–20/09 |
| `ef09b60` | CLI sinh partition + `dataset_id` + track raw inputs (bỏ 2 file khỏi `.gitignore`) |
| `07d86c4` | GitHub Actions 3 job + `scripts/verify_partitions.py` + exit code cho `quick_test` |
| `63e385f` | Import lười `benchmark/__init__.py` — sửa 2 job CI đỏ vì torch |

### Tài liệu đã viết

| File | Nội dung |
|---|---|
| `docs/README.md` | Index + 3 việc cần quyết sớm nhất |
| `docs/khoa/00_dataset_provenance.md` | Bằng chứng synthetic — 5 bước suy luận, script tái lập, hệ quả |
| `docs/khoa/01_bugfix_train_lambda_cos.md` | Crash `train.py` — 2 lỗi chồng nhau, bảng so 4 cách sửa |
| `docs/khoa/02_bugfix_time_label_map.md` | `TIME_LABEL_MAP` — cơ chế dict merge, bug thứ cấp |
| `docs/khoa/03_benchmark_partition_design.md` | Thiết kế partition — 4 phương án, spec chống rò rỉ, §9 kết quả triển khai |
| `docs/khoa/04_input_masking_v2.md` | Input masking v2 |
| `docs/khoa/05_worklog.md` | File này |

---

## 12. CLI sinh partition + CI (21/09)

**Trạng thái:** đóng toàn bộ DoD §4.2 của plan tuần. CI xanh 3/3 job ở commit `63e385f`.

### 12.1. CLI — `python -m benchmark.partition_gen`

Trước đó module chỉ có hàm, không có đường chạy. `partitions_meta.json` vẫn là file mẫu viết
tay với 3 `PLACEHOLDER`. Thêm `main()` + argparse, sinh **31 partition thật**:

| Scenario | Số bản ghi | Sinh thế nào |
|---|---|---|
| `quantity_skew` | 20 | 4 alpha × 5 seed |
| `zone_skew` | 5 | 5 seed |
| `temporal_shift` | 1 | tất định, không phụ thuộc seed |
| `concept_drift` | 5 | 5 seed |

Bốn quyết định đáng ghi:

**Sweep alpha × seed chứ không mỗi alpha một điểm.** `docs/khoa/03` §4.4 đã lập luận: một lần
bốc Dirichlet ở $\alpha = 0.1$ có thể ra rất lệch hoặc tình cờ gần đều. 20 điểm cho scatter
MAE-vs-Gini, không phải 4 cột bấp bênh.

**`load_zone_matrix` đọc `zone_labels.csv`, không đọc `.pt`.** $Z$ có trong dataset nhưng mở nó
phải `import torch`. Đọc CSV thì CLI chạy được ở nơi không có torch — điều kiện để CI
numpy-only tồn tại được. Thứ tự node lấy từ `meta["nodes"]` chứ không `sorted()`: sai thứ tự
thì mask gán nhầm zone cho node và **không có gì báo lỗi cả**.

**`concept_drift` gọi `temporal_shift` trước** để lấy `test_idx` — sự cố phải nằm trong test
set (`docs/khoa/03` §9.4). Trả về `perturbation_spec` gắn ngoài `build_record`, không sinh
mask, không sửa tensor.

**`--append` gộp theo `partition_id`**, nên chạy lại cùng cấu hình thì ghi đè đúng bản ghi cũ
chứ không nhân đôi.

Schema thêm trường `dataset_id`, mặc định lấy tên file dataset (`graph_dataset`) — **không**
đặt sẵn `hcm_sim_v1`, vì dataset hiện tại vẫn là bản legacy và gọi nó là hcm_sim là khai sai.

### 12.2. `scripts/verify_partitions.py`

DoD ghi *"mọi partition tái tạo được từ JSON metadata"* — script này chứng minh điều đó thay
vì tuyên bố suông. Đọc `partitions_meta.json`, sinh lại từng partition từ
`(scenario, params, seed)`, rồi so:

| Kiểu | Số bản ghi | So cái gì |
|---|---|---|
| mask | 25 | `mask_hash` |
| splits | 1 | từng list `train` / `val` / `test` |
| spec | 5 | `perturbation_spec` |

Kết quả: **31/31 khớp bit-for-bit.**

Nhánh `else` cuối bắt scenario lạ và **fail**. Ai thêm cơ chế thứ 5 mà quên cập nhật script
thì CI kêu to, thay vì lặng lẽ bỏ qua rồi báo xanh — cùng tinh thần *fail to tiếng* ở
`docs/khoa/03` §6.

### 12.3. CI — `.github/workflows/tests.yml`

Ba job tách rời, chạy trên mọi nhánh (`branches: ["**"]`):

| Job | Cài gì | Thời gian |
|---|---|---|
| Unit test benchmark | numpy, pytest | 11s |
| Partition tái lập được từ metadata | numpy | 9s |
| quick_test trên clone sạch | numpy, pandas, scipy, torch CPU | 47s |

Tách ba job để job logic trả lời trong ~10 giây, không phải đợi torch tải xong. Không có
training nào trong CI, đúng ràng buộc plan.

### 12.4. Hai lỗi CI bắt được

**`quick_test.py` không thể fail.** Nó in ✅/❌ cho 6 hạng mục rồi kết thúc, không có
`sys.exit()` nào trong cả script. Đưa nguyên trạng vào CI thì được một dấu tick xanh **không
chứng minh điều gì** — tệ hơn là không có CI, vì nó tạo cảm giác an toàn giả. Đã thêm 4 dòng
exit code.

**`benchmark/__init__.py` kéo torch vào mọi import.** Commit `4b22dbc` (Người 3) thêm
`from benchmark.data_quality import DataQualityAssessor` ở cấp package, mà `data_quality.py`
`import torch` ở module level. Hệ quả:

```
import benchmark.partition_gen
  → Python chạy benchmark/__init__.py TRƯỚC
     → import data_quality → import torch   ← chết ở đây
```

`partition_gen.py` chỉ cần numpy nhưng **không import nổi nếu thiếu torch**. Hai job numpy-only
đỏ ngay ở bước import. Trên máy dev không lộ vì venv có sẵn torch — đúng loại lỗi chỉ hiện trên
clone sạch, tức chính cái DoD đang muốn chứng minh.

Sửa bằng module `__getattr__` (PEP 562): torch chỉ nạp khi thực sự chạm tới
`DataQualityAssessor`. Đường dùng `from benchmark import DataQualityAssessor` của Người 3 giữ
nguyên, không phải sửa gì.

> ⚠️ **Bẫy cho cả nhóm:** `benchmark/__init__.py` phải giữ nguyên import lười. Ai thêm một dòng
> `from benchmark.X import Y` ở cấp package mà `X` kéo theo torch là **CI đỏ lại ngay**, và lỗi
> sẽ hiện ra ở chỗ chẳng liên quan gì (`import benchmark.partition_gen`).

### 12.5. Raw inputs được track

`data/raw/hcm_osrm_dataset.csv` (2.9 MB) và `tomtom_traffic.csv` (1.8 MB) trước bị
`.gitignore`. Hệ quả: `quick_test.py` không chạy nổi trên clone sạch, và máy thứ hai phải copy
tay. Đã bỏ khỏi `.gitignore` và commit — tổng 4.7 MB, chưa cần Git LFS (GitHub chỉ cảnh báo từ
50 MB). Chỉ checkpoint `.pt` là vẫn không track.

### 12.6. Nợ kỹ thuật mới phát sinh

`partitions_meta.json` giờ **2.2 MB**. Thiết kế ở `docs/khoa/03` §6 viết *"chỉ lưu index ⇒ file
nhỏ, đọc được bằng mắt, diff được bằng git"* — ở 2.2 MB thì cả hai vế sau không còn đúng. Thủ
phạm là `node_windows`, liệt kê từng chỉ số cửa sổ cho từng node.

Về lý thuyết nó **thừa**: bản ghi đã có `seed` + `params` (tái lập được) và `mask_hash` (kiểm
chứng được). Nhưng `scripts/eval_non_iid.py` dòng 262 đọc thẳng `partition["node_windows"]`, bỏ
bây giờ là làm vỡ code Người 4. Đề xuất: thêm cờ `--no-node-windows`, bàn với Người 4 trước.

---

## Phụ lục: bài học rút ra

**`stats` phải đếm từ `mask`, không từ `n`.** Chính điều này bắt được lỗi mask rỗng — nếu
đếm từ `n` thì stats vẫn đẹp long lanh trong khi mask toàn `False`, và Người 4 sẽ train
trên mask trống mà không hiểu tại sao mọi model đều như nhau.

**Bug tệ nhất là bug không crash.** `TIME_LABEL_MAP`, mask rỗng, CRLF — cả ba đều chạy êm
ru và ra số sai. Cái crash thì sửa trong 5 phút.

**Một cái tên file trùng đủ để phá cả paper.** Generator ghi output ra đúng chỗ dữ liệu
thật — đó là toàn bộ nguyên nhân vụ synthetic. Và nó vừa lặp lại theo chiều ngược.

**Trước khi debug, hỏi dữ liệu từ đâu ra.** Nhóm suýt bỏ một tuần debug `zone_weight` cho
một thứ không hỏng.

**Môi trường dev che lỗi của môi trường sạch.** `benchmark/__init__.py` kéo torch vào mọi
import — chạy êm suốt 3 tuần trên máy có venv, đỏ ngay giây đầu trên CI. Cách duy nhất phát
hiện là thật sự chạy ở nơi không có sẵn thư viện.

**Một test không thể fail thì tệ hơn không có test.** `quick_test.py` in ❌ rồi vẫn exit 0 —
đưa vào CI là có một dấu tick xanh bảo chứng cho không điều gì.
