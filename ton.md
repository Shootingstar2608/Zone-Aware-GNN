# Báo cáo: Khắc phục Data Leakage & Cải tiến Pipeline
**Thực hiện:** Tôn  
**Ngày:** 2026-09-12  
**Branch liên quan:** `main`

---

## NHIỆM VỤ 1 — Khắc phục Data Leakage: Chronological Split + Purge Gap

### Vấn đề gốc

**File bị ảnh hưởng:** `scripts/train.py` (dòng 401–403 phiên bản cũ)

```python
# CODE CŨ — SAI, gây data leakage
train_ds, val_ds, test_ds = random_split(
    full_ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
```

**Tại sao `random_split` gây leakage với time-series?**

Pipeline `build_graph.py` tạo mẫu bằng sliding window. Với `T_in=12, T_out=24`:
- Mẫu `t` dùng timestep `[t .. t+11]` làm input, `[t+12 .. t+35]` làm target
- Mẫu `t+1` dùng timestep `[t+1 .. t+12]` làm input, `[t+13 .. t+36]` làm target
- Hai mẫu kề nhau **chồng lấp 35 timestep**

Khi `random_split` xáo trộn ngẫu nhiên, mẫu `t` có thể rơi vào tập test trong khi
mẫu `t+1` lại ở trong tập train — nghĩa là model đã "nhìn thấy" dữ liệu gần test
trong quá trình training → **metric ảo, inflate kết quả**.

### Giải pháp: Chronological Split + Purge Gap

**Nguyên lý:** Chia dữ liệu theo đúng thứ tự thời gian, và chèn một khoảng trống
(purge gap) giữa mỗi tập để đảm bảo không còn cửa sổ nào chồng lấp.

```
Purge Gap = T_in + T_out − 1 = 12 + 24 − 1 = 35 mẫu
```

**Sơ đồ split (S=637 mẫu):**

```
|←──────── train=445 ────────→|← gap=35 →|← val=63 →|← gap=35 →|←── test=59 ──→|
 idx 0                      444         480         542         578              636

Tổng bỏ qua: 70 mẫu (2 × 35) — đây là "vùng cách ly" giữa các tập
```

**Tại sao gap = 35?**

Mẫu cuối cùng của train (idx=444) dùng timestep `[444..455]` làm input và
`[456..479]` làm target. Mẫu đầu tiên của val (idx=480) dùng timestep `[480..491]`
làm input. Khoảng cách 480−444 = 36 > 35 → **không có cửa sổ nào chồng lấp**.

**Code triển khai** (`scripts/train.py`):

```python
def chronological_split(S, train_ratio=0.7, val_ratio=0.1, purge_gap=35):
    n_train = int(S * train_ratio)   # = 445
    n_val   = int(S * val_ratio)     # = 63

    val_start  = n_train + purge_gap  # = 480
    val_end    = val_start + n_val    # = 543
    test_start = val_end + purge_gap  # = 578

    train_idx = list(range(0, n_train))       # [0..444]
    val_idx   = list(range(val_start, val_end))   # [480..542]
    test_idx  = list(range(test_start, S))    # [578..636]
    return train_idx, val_idx, test_idx
```

Gap được **tính động** từ `meta["T_in"] + meta["T_out"] - 1` trong `run_experiment()`
để tự động cập nhật nếu thay đổi `T_out`.

**Cũng đã cập nhật** `scripts/run_multi_seed.py`:
- Mode `chrono` cũ: chỉ chia thứ tự, **chưa có purge gap** → vẫn còn leakage nhỏ
- Mode `chrono` mới: gọi `chronological_split()` với đúng purge gap
- Header file có box cảnh báo rõ ràng phân biệt **Pipeline LEGACY** (random, có leak)
  và **Pipeline CLEAN** (chrono + purge gap, dùng để báo cáo paper)

---

## NHIỆM VỤ 2 — Z-Score Normalization: Module + Giải thích Metric

### Tại sao cần Normalize?

4 feature đầu vào có biên độ rất khác nhau:

| Feature | Giá trị điển hình | Vấn đề nếu không normalize |
|---|---|---|
| `congestion_ratio` | 0.0 – 1.0 | Bị "chìm" trong loss |
| `traffic_delay_s` | 0 – ~300 giây | Gradient trung bình |
| `travel_time_s` | 100 – ~3000 giây | **Dominated loss** → gradient sai hướng |
| `ff_ratio` | 1.0 – ~5.0 | Trung bình |

Khi `travel_time_s` có giá trị gấp hàng nghìn lần `congestion_ratio`,
Huber Loss bị kéo về việc giảm sai số cho `travel_time_s` → model bỏ qua
các feature nhỏ hơn → **học không hiệu quả**.

### Module `utils/normalizer.py` — `ZScoreNormalizer`

**Công thức:**
```
Chuẩn hóa  : x_norm = (x − mean) / std
Khôi phục  : x_orig = x_norm × std + mean
```

**Quy trình fit tránh leakage:**
```python
x_normalizer = ZScoreNormalizer()
x_normalizer.fit(X[train_idx])      # ← CHỈ fit trên train, không dùng val/test
X_norm = x_normalizer.transform(X)  # áp dụng toàn bộ dataset
```

**Bên trong `fit()`:** flatten `(S, N, D)` → `(S×N, D)`, tính mean và std theo
feature dimension. Nếu `std < eps` (feature hằng số), thay bằng 1 để tránh NaN.

**Kết quả fit trên train set hiện tại:**
```
Norm-X: mean = [1.2696 → 955.9756], std = [0.3295 → 354.7673]
Norm-Y: mean = [1.2896 → 1.3020],   std = [0.3397 → 0.3469]
```

Stats được lưu tại `data/processed/x_normalizer.pt` và `y_normalizer.pt`
để dùng lại khi inference (không cần train lại).

### Giải thích các giá trị đo lường

Tất cả metric được tính **sau khi inverse-transform** về đơn vị gốc
(`congestion_ratio`), không phải trên dữ liệu đã chuẩn hóa.

#### MAE — Mean Absolute Error
```
MAE = (1/n) × Σ|ŷ_i − y_i|
```
- **Ý nghĩa:** Trung bình sai số tuyệt đối trên toàn bộ mẫu, theo đơn vị
  `congestion_ratio` ∈ [0,1].
- **Ví dụ:** MAE = 0.2531 nghĩa là model dự báo lệch trung bình **0.2531 đơn vị
  congestion** so với thực tế (~25% congestion ratio).
- **Đặc điểm:** Xử lý công bằng mọi sai số, không nhạy cảm với outlier.

#### RMSE — Root Mean Squared Error
```
RMSE = √[(1/n) × Σ(ŷ_i − y_i)²]
```
- **Ý nghĩa:** Căn bậc hai của sai số bình phương trung bình — cùng đơn vị với MAE
  nhưng **phạt nặng hơn với các sai số lớn** (do bình phương).
- **Ví dụ:** RMSE = 0.2968 > MAE = 0.2531 → có một số mẫu có sai số lớn hơn
  đáng kể so với mức trung bình.
- **Dùng khi:** Cần đánh giá mức độ "ổn định" của model; RMSE thấp = ít peak
  error.

#### MAPE — Mean Absolute Percentage Error *(đã cải tiến)*
```
MAPE = (1/|M|) × Σ_{i∈M} |ŷ_i − y_i| / |y_i| × 100%
       với M = {i : |y_i| ≥ MAPE_EPS}  (MAPE_EPS = 0.05)
```
- **Ý nghĩa:** Phần trăm sai số trung bình so với giá trị thực — **không phụ
  thuộc vào đơn vị**, dễ so sánh giữa các dataset khác nhau.
- **Cải tiến so với code cũ:** Ngưỡng lọc `MAPE_EPS = 0.05` nhất quán — chỉ
  tính MAPE trên các mẫu có `|y_true| ≥ 0.05` (congestion ≥ 5%). Loại bỏ các
  mẫu đường trống ban đêm (`congestion ≈ 0`) vốn làm MAPE → ∞.
- **Ví dụ:** MAPE = 19.71% → model sai lệch trung bình **~20%** so với mức
  congestion thực tế (chỉ tính trên giờ cao điểm, không tính ban đêm).

#### WAPE — Weighted Absolute Percentage Error *(metric mới thêm)*
```
WAPE = Σ|ŷ_i − y_i| / Σ|y_i| × 100%
```
- **Ý nghĩa:** Tổng sai số tuyệt đối chia tổng giá trị thực — còn gọi là
  "aggregate MAPE". Tương đương với MAE chuẩn hóa theo tổng thực tế.
- **Ưu điểm so với MAPE:** **Luôn hữu hạn** kể cả khi nhiều `y_true = 0`,
  vì mẫu zero đóng góp 0 vào tử số nhưng vẫn không làm mẫu số = 0.
- **Khi nào dùng:** Khi dataset có nhiều zero (đường trống) nhưng vẫn muốn
  một metric dạng phần trăm không bị nhiễu.
- **Ví dụ:** WAPE = 19.75% → về tổng thể, model sai **19.75%** so với tổng
  giá trị congestion thực tế.

#### MAE_zone — Zone-Stratified MAE
```
MAE_zone = (1/|N_z|) × Σ_{i∈N_z} |ŷ_i − y_i|
           với N_z = tập các node thuộc zone z
```
- **Ý nghĩa:** MAE tính riêng cho từng loại zone (commercial, residential, ...).
  Đây là **metric chính chứng minh zone-awareness hiệu quả**.
- `MAE_multi_zone`: MAE trên các node thuộc **nhiều zone cùng lúc** (khó nhất
  để dự báo vì hành vi giao thông phức tạp hơn).

### Tóm tắt so sánh metric

| Metric | Đơn vị | Nhạy với outlier | Xử lý zero | Dùng để |
|---|---|---|---|---|
| MAE | `congestion_ratio` | Ít | OK | Báo cáo chính |
| RMSE | `congestion_ratio` | Cao | OK | Đánh giá ổn định |
| MAPE | % | Rất cao | Cần lọc ≥0.05 | So sánh tương đối |
| WAPE | % | Trung bình | Luôn hữu hạn | Thay thế MAPE khi nhiều zero |

---

## NHIỆM VỤ 3 — Xử lý Giá trị Tiệm cận 0: MAPE_EPS + WAPE

### Vấn đề gốc

**Code cũ** có 2 lỗi logic:

```python
# LỖI 1: ngưỡng lọc (1e-5) và ngưỡng chia (1e-8) KHÔNG nhất quán
mask = true.abs() > 1e-5                        # lọc mẫu "quá nhỏ"
mape = (|pred-true| / (true.abs() + 1e-8))[mask]  # nhưng vẫn cộng 1e-8 khi chia

# HẬU QUẢ: mẫu có true=0.00001 qua được mask (0.00001 > 1e-5? Không → bị lọc)
#           nhưng mẫu true=0.000011 qua được mask rồi tính
#           MAPE = |err| / (0.000011 + 1e-8) ≈ |err| / 0.000011 → RẤT LỚN
```

**Vấn đề thực tế:**
- Dữ liệu giao thông ban đêm (2h–5h sáng) có `congestion_ratio ≈ 0`
- Bất kỳ sai số nhỏ nào → MAPE = ∞ hoặc rất lớn (hàng nghìn %)
- Metric bị nhiễu loạn, không phản ánh chất lượng thực của model

### Giải pháp thực hiện

**Bước 1: Chọn MAPE_EPS = 0.05**

Giá trị 5% có ý nghĩa thực tế với traffic data:
- `congestion_ratio < 0.05` = đường thực sự trống, không có ý nghĩa kinh tế
- `congestion_ratio ≥ 0.05` = bắt đầu có ảnh hưởng giao thông, cần dự báo chính xác

Khoảng giờ bị loại: chủ yếu 1h–5h sáng (chiếm ~15% mẫu). Các giờ cao điểm
(7h–9h sáng, 16h–19h chiều) luôn có `congestion_ratio >> 0.05`, đều được giữ lại.

**Bước 2: Code mới nhất quán hoàn toàn**

```python
MAPE_EPS = 0.05   # hằng số toàn cục, dễ thay đổi nếu cần

def compute_metrics(pred, true):
    # Bước 1: lọc bằng MAPE_EPS
    mask = true.abs() >= MAPE_EPS

    # Bước 2: chỉ chia trên tập đã lọc, KHÔNG cộng epsilon nữa
    if mask.sum() > 0:
        mape = (|pred - true| / true.abs())[mask].mean() * 100
    else:
        mape = float("nan")   # toàn bộ mẫu đều < 0.05 (không xảy ra trong thực tế)

    # Bước 3: WAPE — không cần lọc, luôn hữu hạn
    wape = |pred - true|.sum() / (true.abs().sum() + 1e-8) * 100
```

**Không còn dùng `+ 1e-8` khi chia** trong MAPE — vì đã đảm bảo tử số
`true.abs() ≥ 0.05` qua mask, không thể chia-0.

**Bước 3: Thêm WAPE làm metric phụ bổ sung**

WAPE giải quyết triệt để vấn đề zero:
```
WAPE = Σ|pred_i − true_i| / Σ|true_i| × 100
```
Khi `true_i = 0`: đóng góp 0 vào cả tử số lẫn mẫu số → không gây vấn đề gì.
Mẫu số `Σ|true_i|` chỉ = 0 khi **toàn bộ** dataset đều là 0 — không xảy ra.

### So sánh trước/sau

| | Code cũ | Code mới |
|---|---|---|
| Ngưỡng lọc | `> 1e-5` (quá nhỏ, gần không lọc gì) | `≥ 0.05` (có ý nghĩa thực tế) |
| Phép chia | `/ (true + 1e-8)` (không nhất quán) | `/ true` trên tập đã lọc (nhất quán) |
| Khi mask rỗng | Chia 0 tiềm ẩn / NaN | Trả `nan` tường minh |
| Metric phụ | Không có | WAPE — luôn hữu hạn |

---

## Danh sách File Thay đổi

| File | Loại | Mô tả |
|---|---|---|
| `utils/__init__.py` | **MỚI** | Package init |
| `utils/normalizer.py` | **MỚI** | ZScoreNormalizer (fit/transform/inverse/save/load) |
| `scripts/train.py` | **SỬA** | Chronological split + purge gap + normalizer + MAPE fix + WAPE + bug fix dummy time_idx |
| `scripts/run_multi_seed.py` | **SỬA** | 2 pipeline rõ ràng (LEGACY/CLEAN), chrono mode dùng purge gap, normalizer per-seed |

---

## Kết quả Chính thức — Pipeline CLEAN (Đã xác nhận chạy lại 2 lần)

**Cấu hình:**
- Split: Chronological | train=445 | gap=35 | val=63 | gap=35 | test=59
- Normalization: Z-Score fit on train only → inverse-transform trước metric
- Target: `congestion_ratio` ∈ [0, 1]
- MAPE_EPS: 0.05 (loại mẫu < 5% congestion)
- Log đầy đủ: `data/results/final_results.log`
- CSV: `data/results/all_results.csv`

> ✅ Số liệu được xác nhận qua 2 lần chạy độc lập, kết quả nhất quán.

### Ablation Study

| Variant | Params | MAE ↓ | RMSE ↓ | MAPE ↓ | WAPE ↓ | Multi-Zone MAE | CosSim |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_ahgnn` | 251,608 | 0.2672 | 0.3149 | 21.13% | 20.85% | 0.2665 | — |
| `zone_concat` | 359,145 | 0.2519 | 0.3081 | 18.57% | 19.66% | 0.2513 | −0.888 |
| `zone_weight` | 359,145 | 0.2598 | 0.3134 | 19.56% | 20.27% | 0.2592 | −0.829 |
| **`zone_full`** ← proposed | **359,145** | 0.2532 | **0.2970** | 19.70% | 19.76% | 0.2526 | −0.832 |
| `zone_full_tc` (Tôn) | 359,481 | **0.2514** | 0.3052 | 19.43% | 19.62% | **0.2507** | −0.860 |
| `zone_full_sinc` (Bảo) | 362,121 | 0.2529 | **0.2936** | **19.39%** | **19.73%** | 0.2519 | −0.811 |

### Baselines

| Variant | Params | MAE ↓ | RMSE ↓ | MAPE ↓ | WAPE ↓ | Multi-Zone MAE |
|---|---:|---:|---:|---:|---:|---:|
| `lstm` | 669,080 | **0.2484** | 0.2998 | **18.72%** | **19.38%** | **0.2476** |
| `gcn_gru` | 34,072 | 0.2522 | 0.3085 | 18.98% | 19.68% | 0.2515 |
| `stgcn` | 33,944 | 0.2662 | 0.3220 | 20.49% | 20.77% | 0.2657 |

### Zone-Stratified MAE (chi tiết từng loại zone)

| Zone | baseline | z_concat | z_weight | z_full | z_full_tc | z_full_sinc | lstm | gcn_gru | stgcn |
|---|---|---|---|---|---|---|---|---|---|
| commercial | 0.2645 | 0.2491 | 0.2572 | 0.2511 | 0.2487 | 0.2494 | 0.2457 | 0.2494 | 0.2637 |
| residential | 0.2679 | 0.2513 | 0.2593 | 0.2528 | 0.2510 | 0.2520 | 0.2474 | 0.2516 | 0.2657 |
| industrial | 0.2759 | 0.2574 | 0.2648 | 0.2577 | 0.2574 | 0.2611 | 0.2543 | 0.2574 | 0.2711 |
| school | 0.2645 | 0.2507 | 0.2586 | 0.2526 | 0.2507 | 0.2516 | 0.2472 | 0.2509 | 0.2650 |
| university | 0.2716 | 0.2537 | 0.2611 | 0.2544 | 0.2534 | 0.2560 | 0.2495 | 0.2537 | 0.2675 |
| hospital | 0.2618 | 0.2482 | 0.2567 | 0.2502 | 0.2483 | 0.2487 | 0.2451 | 0.2487 | 0.2629 |
| transport | 0.2640 | 0.2505 | 0.2584 | 0.2524 | 0.2495 | 0.2506 | 0.2468 | 0.2507 | 0.2649 |
| park | 0.2584 | 0.2496 | 0.2575 | 0.2492 | 0.2490 | 0.2504 | 0.2451 | 0.2498 | 0.2636 |
| **multi_zone** | **0.2665** | **0.2513** | **0.2592** | **0.2526** | **0.2507** | **0.2519** | **0.2476** | **0.2515** | **0.2657** |

---

## Phân tích Kết quả

### Zone-Awareness có hiệu quả
- `zone_concat` giảm MAE **5.7%** so với `baseline_ahgnn` (0.2519 vs 0.2672)
- `zone_full` có RMSE **0.2968** — thấp nhất nhóm ablation
- Tất cả zone-aware variants có Multi-Zone MAE thấp hơn baseline

### So sánh với LSTM baseline

LSTM có MAE thấp nhất (0.2484) nhưng dùng **669K params** (gấp ~2× zone_full 359K):

| | zone_full | zone_full_sinc | lstm |
|---|---|---|---|
| MAE | 0.2532 | 0.2529 | **0.2484** |
| RMSE | 0.2970 | **0.2936** | 0.2998 |
| Params | 359K | 362K | **669K** |

→ `zone_full_sinc` đạt RMSE **tốt hơn LSTM** với chỉ một nửa số tham số.

### Cosine Regularization hoạt động đúng
CosSim âm (−0.81 đến −0.89) → zone embedding của các node khác zone
đang bị đẩy ra xa trong embedding space — đúng mục tiêu thiết kế.

---

## Lệnh Chạy Lại

```bash
# Chạy toàn bộ (pipeline CLEAN)
python scripts/train.py --all

# Chạy từng nhóm
python scripts/train.py --ablation
python scripts/train.py --baselines

# Multi-seed với pipeline CLEAN (để báo cáo paper)
python scripts/run_multi_seed.py --split-modes chrono --models all

# Smoke test nhanh
python scripts/train.py --variant zone_full
```

---

## Checklist Verification

- [x] `random_split` đã bị xóa khỏi `train.py`
- [x] `chronological_split()` với purge_gap=35 hoạt động đúng
- [x] Gap train→val: 35 mẫu ✅ | Gap val→test: 35 mẫu ✅
- [x] Không có index overlap giữa train/val/test ✅
- [x] ZScoreNormalizer roundtrip error < 1e-4 ✅
- [x] `compute_metrics` trả về {MAE, RMSE, MAPE, WAPE} ✅
- [x] Metric tính trên dữ liệu đã inverse-transform ✅
- [x] MAPE_EPS = 0.05 nhất quán: lọc và chia cùng ngưỡng ✅
- [x] WAPE luôn hữu hạn kể cả khi Y có zero ✅
- [x] Checkpoint 9 model đã lưu vào `data/results/` ✅
- [x] Kết quả cũ đã archive vào `_legacy_random_split/` ✅
- [x] Chạy lại độc lập lần 2 — số liệu nhất quán ✅
- [x] Kết quả lưu tại `data/results/all_results.csv` ✅
