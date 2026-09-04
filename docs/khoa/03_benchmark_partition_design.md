# 03 — Thiết kế bộ sinh phân vùng Non-IID (Người 2)

**Phạm vi:** `benchmark/partition_gen.py`, `data/partitions/partitions_meta.json`
**Người dùng đầu ra:** Người 4 (`scripts/eval_non_iid.py`, `tests/test_benchmark.py`)
**Trạng thái:** contract đã chốt, cơ chế #1 (Quantity Skew) đang làm

---

## 1. Câu hỏi thiết kế cốt lõi

Plan viết: *"Dùng phân phối Dirichlet để phân bổ số lượng mẫu dữ liệu lệch nhau giữa các node."*

Câu đó **không có nghĩa hiển nhiên** với hình dạng dữ liệu của ta:

```
X: (S=637, N=17, T_in*F=48)     <- 1 sample = 1 cua so thoi gian, CHUA CA 17 NODE
Y: (S=637, N=17, T_out=24)
A: (17,17), Z: (17,8)            <- chia se chung
```

Model là GNN: `model(X_b, Z, T_b, A)` — một forward pass nuốt **cả đồ thị**.

> **Mâu thuẫn:** trong FL kinh điển (CIFAR, FEMNIST) mỗi sample thuộc về đúng một client.
> Ở đây, nếu client = node thì client **không sở hữu sample nào** — nó sở hữu **một cột**
> của mọi sample. Nên phải chọn trục chia một cách có chủ đích.

---

## 2. Các phương án đã cân nhắc

| # | Client là gì | Dirichlet chia gì | Đụng chiều node? | Người 4 chạy được trong tuần? |
|---|---|---|---|---|
| **1** | 1 node = 1 client | **cửa sổ thời gian mỗi node có dữ liệu** → mask `(S,N)` | ✅ | ✅ sửa ~5 dòng ở loss |
| 2 | 1 node = 1 agent + FedAvg | cửa sổ mỗi agent | ✅ | ❌ chưa ai code FL |
| 3 | 1 cụm zone = 1 client | cửa sổ mỗi cụm | ✅ | ✅ nhưng trùng cơ chế Zone Skew |
| 4 | K client ảo | sample theo tỉ lệ class (FL kinh điển) | ❌ | ✅ nhưng vô nghĩa với paper |

### Vì sao loại

- **#4** — chuẩn sách giáo khoa nhất, nhưng **không đụng chiều node**. Toàn bộ luận điểm
  của paper là *zone/không gian gây ra Non-IID*; chia kiểu này thì benchmark không kiểm
  chứng được chính giả thuyết của nhóm. Loại.
- **#2** — đúng tinh thần "Node-Agent" trong task của Người 1 nhất, nhưng `train.py` đang
  **centralized**. Chọn nó ⇒ Người 4 phải viết FedAvg từ đầu ⇒ tuần sau vẫn chưa có figure.
  Ghi vào Future Work.
- **#3** — trùng cơ chế Feature/Zone Skew của chính module này. Dư thừa.

---

## 3. Quyết định: Phương án 1

> **Client = node. Dirichlet phân bổ *số cửa sổ thời gian mà mỗi node có dữ liệu*.
> Đầu ra là mask nhị phân $M \in \{0,1\}^{S \times N}$.**

$$p \sim \text{Dir}(\alpha \cdot \mathbf{1}_N), \qquad n_v = \lfloor p_v \cdot S \rfloor$$

### Lý do, theo thứ tự quan trọng

**① Biến lựa chọn kỹ thuật thành thí nghiệm thật.**
Node ít dữ liệu buộc phải dựa vào **message passing từ hàng xóm**. Đó đúng là chỗ
zone-awareness phải chứng minh giá trị: nếu model zone-aware **suy giảm chậm hơn** khi
coverage lệch tăng, đó là finding thật.

Và quan trọng hơn — **finding này không bị dính lập luận vòng tròn** như bảng kết quả cũ
(xem `docs/00` §4.1). Generator mô phỏng quan hệ zone→congestion, nhưng nó **không hề mô
phỏng chuyện thiếu dữ liệu**. Cơ chế Non-IID này nằm ngoài những gì generator viết ra,
nên kết luận rút ra từ nó không phải là "khám phá lại chính mình".

**② Có câu chuyện vật lý thật.**
*Sensor coverage skew* — trong mạng giao thông thật, độ phủ dữ liệu giữa các nút cực kỳ
không đều: ngã tư lớn có camera + loop detector, đường nhánh gần như không báo gì.
Đây là hiện tượng citable, không phải trò synthetic gán ghép.

**③ Đúng nguyên văn plan** — "phân bổ số lượng mẫu lệch nhau **giữa các node**".

**④ Người 4 tích hợp được trong một buổi chiều** — không đổi kiến trúc, không đổi
`TensorDataset`, chỉ nhân mask vào loss.

---

## 4. Bốn quyết định con

### 4.1. Mask ở đâu: loss hay input?

| | Ý nghĩa | Sức nặng khoa học | Chi phí |
|---|---|---|---|
| **Loss masking** (v1) | node $v$ vẫn có trong input, nhưng không bị chấm điểm ở cửa sổ thiếu | vừa — gradient lệch giữa các node, đúng bản chất FL quantity skew | ~5 dòng |
| **Input masking** (v2) | feature của $v$ bị che ⇒ model phải **suy ra $v$ từ hàng xóm** | mạnh — đây mới là bài test thật | phải quyết cách impute |

**Chốt: v1 loss masking, v2 input masking.**
v1 ship trong ngày ⇒ Người 4 chạy được ngay; v2 là thí nghiệm ăn tiền, làm sau.

⚠️ Khi lên v2: zero-fill xong thì model **không phân biệt được "không có dữ liệu" với
"đường thoáng"**. Phải thêm kênh availability ⇒ `F: 4 → 5` ⇒ **vỡ toàn bộ checkpoint cũ**.
Đó là lý do nữa để đẩy nó sang v2 chứ không nhét vào v1.

### 4.2. Chọn cửa sổ ngẫu nhiên hay theo khối?

Sensor thật chết **theo khối** (mất 6 tiếng), không rụng lẻ tẻ từng khung 15 phút.
⇒ Để tham số `block_len`, mặc định `= 1` cho v1 (ngẫu nhiên, đơn giản), tăng sau.
**Không hardcode.**

### 4.3. Node nhận 0 mẫu thì sao?

Với $\alpha = 0.1$, $N = 17$: chắc chắn có node $p_v \approx 0$.

**Chốt: CHO PHÉP 0, không đặt sàn.** Vì đó là thí nghiệm mạnh nhất trong cả benchmark:

> Một node **chưa từng được giám sát một lần nào** — model dự báo nó chỉ bằng *nhãn zone*
> + *cấu trúc đồ thị*. Nếu model zone-aware làm được còn GCN-GRU thì không ⇒ bằng chứng
> trực tiếp cho giá trị của zone semantics.

Ghi `n_zero_nodes` và `zero_nodes` vào metadata để Người 4 lọc ra vẽ figure riêng.

### 4.4. Trục $x$ của figure: $\alpha$ hay Gini?

**Không dùng $\alpha$.** Một lần bốc Dirichlet ở $\alpha = 0.1$ có thể ra rất lệch hoặc gần
như đều — bốc là ngẫu nhiên. 4 giá trị $\alpha$ = 4 điểm bấp bênh.

**Dùng Gini thực tế đo được** của phân bố $n_v$:

$$G = \frac{\sum_i \sum_j |n_i - n_j|}{2 N \sum_i n_i}, \qquad
G \to 0 \ (\alpha \to \infty), \quad G \to 1 \ (\alpha \to 0)$$

Rồi bốc **nhiều seed cho mỗi $\alpha$** (5–10 seed, đúng chuẩn multi-seed nhóm đã làm ở
`MULTISEED_FINDINGS.md`). Figure của Người 4 thành **scatter MAE vs Gini với 20–40 điểm +
đường fit**, thay vì 4 cột. Đẹp hơn, trung thực hơn, và nối liền mạch với văn hoá
multi-seed nhóm đã xây.

---

## 5. Spec chống rò rỉ (Người 4 viết test dựa vào đây)

Với sliding window, hai cửa sổ $i, j$ **chồng lấn về mặt dữ liệu** khi:

$$|i - j| < T_{in} + T_{out}$$

Nên mọi split theo thời gian **bắt buộc** có purge gap:

$$\text{gap} \ge T_{in} + T_{out} - 1$$

Với `meta.json` hiện tại ($T_{in} = 12$, $T_{out} = 24$): **gap ≥ 35 cửa sổ**.

> ⚠️ `T_out` **không cố định** — nó là 3, 6, 9, 12, 18 hoặc 24 tuỳ lần chạy multistep,
> và `meta.json` hiện đang để lại ở 24. **Đọc từ `meta.json`, không hardcode.**

Đây chính là lỗ hổng mà `MULTISEED_FINDINGS.md §3` cảnh báo nhưng chưa ai vá:
`train.py` dùng `random_split` trên chuỗi thời gian ⇒ cửa sổ test có thể trùng 11/12 bước
input với cửa sổ train ⇒ model gần như đã nhìn thấy đáp án. Module này là chỗ vá nó.

**Invariant mà `tests/test_benchmark.py` phải kiểm:**

1. `set(train) & set(test) == empty`
2. `min(|i - j| for i in train for j in test) >= gap` (với mọi partition theo thời gian)
3. Cùng `seed` + cùng `params` ⇒ index list giống **hệt bit-for-bit**
4. `sum(n_per_node)` khớp đúng con số metadata khai báo
5. `dataset_fingerprint` khớp file `.pt` hiện tại, **fail to tiếng** nếu lệch

---

## 6. Contract

```python
def quantity_skew(S, N, alpha, seed, block_len=1) -> (mask, stats)
def zone_skew(Z, S, n_clusters, seed, ...)       -> (mask, stats)
def temporal_shift(S, meta, scenario, seed)      -> (splits, stats)
def concept_drift(...)                            -> (perturbation_spec, stats)
```

Nguyên tắc bất di bất dịch: **partition = index/mask + config + seed. KHÔNG BAO GIỜ copy tensor.**

Lý do: `graph_dataset.pt` là 3 MB; 4 cơ chế × 4 mức $\alpha$ × 10 seed = 160 bản copy ⇒ repo
phình lên gần nửa GB, và Người 4 không thể diff hai partition với nhau. Chỉ lưu index ⇒
`partitions_meta.json` nhỏ, đọc được bằng mắt, diff được bằng git.

### Schema `partitions_meta.json`

```
partition_id            : chuoi duy nhat, vd "qskew_a0.5_s42"
scenario                : quantity_skew | zone_skew | temporal_shift | concept_drift
params                  : {} tham so sinh ra no
seed                    : int
dataset_fingerprint     : sha256 cua graph_dataset.pt
meta_snapshot           : {N, S, T_in, T_out} tai thoi diem sinh
git_commit              : hash luc sinh
mask_hash               : sha256 cua mask -> de verify khong can luu mask
node_windows            : {node_idx: [window_idx, ...]}
stats                   : {n_per_node, gini, n_zero_nodes, zero_nodes, ...}
```

`dataset_fingerprint` là thứ cứu ta sau này: nếu ai rebuild `.pt` với `T_out` khác, index cũ
trở thành vô nghĩa — phải **fail to tiếng** chứ không được chạy im lặng ra số sai.

---

## 7. Thứ tự làm (ưu tiên theo "Người 4 chờ cái gì")

| Ưu tiên | Việc | Người 4 mở khoá được gì |
|---|---|---|
| 🔴 P0 | Contract + stub + schema mẫu | Viết được **cả** `eval_non_iid.py` lẫn `test_benchmark.py` song song |
| 🟠 P1 | Quantity Skew + Gini | Chạy được **figure chính** Performance vs Heterogeneity |
| 🟡 P2 | Temporal Shift | Thêm 2 kịch bản (unblock nhờ `docs/00` §4.4) |
| 🟢 P3 | Zone Skew | Rẻ nhưng không cho trục sweep ⇒ cần muộn nhất |
| ⚪ P4 | Concept Drift | Làm ở **tầng generator**, không hack tensor (xem dưới) |

**Điểm mấu chốt: P0 mới là thứ mở khoá Người 4, không phải code partition.**
Contract tốn 1 giờ, code tốn 4 ngày — mà Người 4 chỉ chờ cái contract.

### Ghi chú cho P4 — Concept Drift

Vì dữ liệu là synthetic (`docs/00`), **đừng hack spike vào tensor**. Thêm drift mode vào
chính `generate_synthetic_traffic.py`. Như vậy `traffic_delay_s`, `travel_time_s`,
`congestion_ratio` di chuyển **nhất quán về mặt vật lý**.

Nếu hack thẳng vào tensor: sửa `traffic_delay_s` mà không sửa `congestion_ratio` (= label $Y$)
sẽ tạo ra quan hệ input–output **phi vật lý**, model học được cũng vô nghĩa. Còn nếu sửa cả
hai bằng tay thì ta đang viết lại generator ở chỗ sai — cứ sửa ở generator cho gọn.

---

## 8. Giới hạn phải ghi vào Data Card

Người 4 cần biết trước, đừng để phát hiện lúc đang viết:

- **Chỉ có đúng 1 cuối tuần** trong 7 ngày dữ liệu ⇒ test set weekday→weekend chiếm 2/7 và
  **confounded hoàn toàn** với "2 ngày cuối chuỗi". Không tách được hai hiệu ứng.
- **17 node / 637 cửa sổ** là quy mô nhỏ cho một benchmark. Với $\alpha = 0.1$, một node có
  thể còn < 10 cửa sổ ⇒ metric trên node đó nhiễu rất mạnh. Cần report khoảng tin cậy,
  không report điểm.
- Toàn bộ chuỗi traffic là **mô phỏng**, không phải đo đạc (xem `docs/00`).
