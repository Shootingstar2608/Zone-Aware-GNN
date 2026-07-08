# Tuần 1: Yêu cầu Chi tiết cho Thành viên B & C + Phân tích Kiến trúc Embedding

---

## Phần I: Yêu cầu & Output cho Thành viên B (Deep Learning Engineer)

### Mục tiêu Tuần 1
Triển khai **3 mô hình baseline chuẩn** (LSTM, GCN-GRU, STGCN) trên tập dữ liệu HCM-Zone hiện tại, với interface thống nhất để so sánh trực tiếp với Zone-Aware AH-GNN.

---

### Yêu cầu 1: Interface mô hình thống nhất

Tất cả baseline phải tuân theo cùng một giao diện `forward()` để tích hợp với pipeline `train.py` hiện tại:

```python
class BaselineModel(nn.Module):
    def forward(self, X, Z=None, time_idx=None, A_static=None):
        """
        X:        (B, N, in_channels)   — in_channels = T_in * F = 12 * 4 = 48
        Z:        (N, K)                — zone labels (bỏ qua cho baseline)
        time_idx: (B,)                  — time label index [0..3]
        A_static: (N, N)                — OSRM adjacency (bỏ qua cho LSTM)
        → output: (B, N, T_out)         — T_out = 3
        """
```

> [!IMPORTANT]
> **Lý do:** `train.py` gọi `model(X_b, Z, T_b, A)` cho tất cả variant. Baseline phải chấp nhận (và bỏ qua) các tham số không dùng đến.

---

### Yêu cầu 2: Danh sách 3 Baseline cần triển khai

#### Baseline 1: LSTM (Temporal-only, không dùng graph)
* **Kiến trúc:** Flatten tất cả node features → LSTM 2-layer → FC → output
* **Input:** `X` shape `(B, N, 48)` → reshape thành `(B, N*48)` rồi unsqueeze thành `(B, 1, N*48)` cho LSTM
* **Output:** `(B, N, T_out)`
* **Mục đích:** Chứng minh rằng cấu trúc đồ thị (GNN) là cần thiết — nếu LSTM đạt kết quả tốt hơn GNN thì GNN không có giá trị.

#### Baseline 2: GCN-GRU (Standard GCN + GRU, zone-blind)
* **Kiến trúc:** GCN layer (shared weight W cho tất cả nodes) → GRU temporal → FC
* **Adjacency:** Dùng `A_static` (OSRM) — cố định, không adaptive
* **Công thức GCN chuẩn:** $H' = \sigma(\hat{A} \cdot H \cdot W)$ — chung một $W$ cho mọi nút
* **Mục đích:** Chứng minh rằng adjacency adaptive + node-specific weight là cần thiết

#### Baseline 3: STGCN (Spatio-Temporal Graph Convolutional Network)
* **Kiến trúc:** Sandwich Temporal Conv1D → GCN → Temporal Conv1D
* **Adjacency:** Dùng `A_static` (OSRM) — cố định
* **Mục đích:** So sánh với kiến trúc state-of-the-art chuẩn (Yu et al., 2018)

---

### Yêu cầu 3: File & thư mục output mong muốn

```
models/
├── ah_gnn.py              # (đã có) Baseline AH-GNN
├── zone_aware_gnn.py      # (đã có) Proposed model
└── baselines.py           # ← [MỚI] Chứa 3 class: LSTMBaseline, GCNGRUBaseline, STGCNBaseline
```

```
data/results/
├── ablation_results.csv   # (đã có) Kết quả ablation 4 variant zone-aware
└── baseline_results.csv   # ← [MỚI] Kết quả 3 baseline + 1 proposed (zone_full)
```

**Format của `baseline_results.csv`:**

| model | MAE | RMSE | MAPE | MAE_multi_zone | n_params |
|---|---|---|---|---|---|
| LSTM | ? | ? | ? | ? | ? |
| GCN_GRU | ? | ? | ? | ? | ? |
| STGCN | ? | ? | ? | ? | ? |
| zone_full | 0.0795 | 0.1419 | 6.14 | 0.0793 | 358,452 |

---

### Yêu cầu 4: Code tham khảo

File tham khảo đã có sẵn tại [references/paper_traffic_flow/baseline.py](file:///home/peter/Research/references/paper_traffic_flow/baseline.py). Tuy nhiên, file đó dùng interface khác (edge-level prediction với `edge_index` và `edge_feats`). Thành viên B cần **chuyển đổi sang node-level prediction** với interface `(B, N, in_channels) → (B, N, T_out)`.

**Điểm khác biệt quan trọng cần lưu ý:**

| Yếu tố | File tham khảo (baseline.py) | Cần triển khai |
|---|---|---|
| Prediction target | Edge-level ETA `(B, E)` | Node-level congestion `(B, N, T_out)` |
| Input format | `(B, T, N, F)` sequence | `(B, N, T_in*F)` flattened |
| Adjacency | Load từ `adj_physical.npy` | Nhận `A_static` từ tham số forward |
| Forward signature | `forward(X_seq, edge_index, edge_feats)` | `forward(X, Z, time_idx, A_static)` |

---

### Yêu cầu 5: Tiêu chí nghiệm thu (Acceptance Criteria)

- [ ] File `models/baselines.py` chạy được với `venv/bin/python -c "from models.baselines import *; print('OK')"`
- [ ] Mỗi model forward pass thành công với dummy input: `X=(32, 17, 48), Z=(17,8), time_idx=(32,), A=(17,17)` → output `(32, 17, 3)`
- [ ] Chạy `venv/bin/python scripts/train.py` với mỗi baseline tạo ra kết quả MAE/RMSE/MAPE hợp lệ (không NaN, không Inf)
- [ ] File `data/results/baseline_results.csv` được tạo ra đúng format

---
---

## Phần II: Yêu cầu & Output cho Thành viên C (Data & GIS Engineer)

### Mục tiêu Tuần 1
Xây dựng **pipeline tự động** crawl nhãn vùng đa nhãn từ OpenStreetMap cho **bất kỳ thành phố nào**, dựa trên tọa độ GPS của các nút giao thông. Pipeline phải hoạt động như một công cụ tổng quát, không hardcode cho HCM.

---

### Yêu cầu 1: Input specification

Thành viên C sẽ nhận đầu vào là file CSV mô tả các nút của một thành phố mới:

**File input: `nodes_config.csv`**
```csv
node,lat,lon
Ben Thanh Market,10.7725,106.6980
District 1,10.7756,106.7009
...
```

**Tham số cấu hình:**
* `--radius`: Bán kính tìm kiếm OSM xung quanh mỗi nút (mặc định 800m)
* `--output`: Đường dẫn file CSV output
* `--city`: Tên thành phố (dùng cho logging/metadata)

---

### Yêu cầu 2: Output specification

**File output: `zone_labels.csv`** — cùng format với file hiện tại:

```csv
node,commercial,residential,industrial,school,university,hospital,transport,park
Ben Thanh Market,1,0,0,1,0,1,1,0
District 1,1,1,0,0,0,1,1,1
```

**Thêm file metadata: `zone_meta.json`**
```json
{
  "city": "Ho Chi Minh City",
  "radius_m": 800,
  "num_nodes": 17,
  "num_zone_types": 8,
  "zone_types": ["commercial", "residential", ...],
  "crawl_timestamp": "2026-07-08T12:00:00",
  "overpass_api": "https://overpass-api.de/api/interpreter",
  "multi_zone_count": 13,
  "single_zone_count": 1
}
```

---

### Yêu cầu 3: Cấu trúc file & thư mục mong muốn

```
scripts/
├── collect_zones.py           # (đã có) Script hiện tại, hardcode 17 nút HCM
└── collect_zones_generic.py   # ← [MỚI] Script tổng quát cho mọi thành phố

data/
├── raw/
│   ├── zone_labels.csv        # (đã có) HCM zones
│   └── nodes_config.csv       # ← [MỚI] Template input cho script tổng quát
```

---

### Yêu cầu 4: Code tham khảo & Lưu ý kỹ thuật

File tham khảo hiện tại: [scripts/collect_zones.py](file:///home/peter/Research/scripts/collect_zones.py)

**Các vấn đề kỹ thuật cần xử lý:**

1. **Rate limiting:** Overpass API giới hạn 1 request/giây. Cần `time.sleep(1.5)` giữa các query.
2. **Error handling:** Overpass có thể trả 429 (Too Many Requests) hoặc 504 (Gateway Timeout). Cần retry logic với exponential backoff.
3. **Fallback:** Nếu Overpass không trả về kết quả cho một nút (vùng nông thôn, dữ liệu OSM thiếu), ghi nhận `all_zeros` và log cảnh báo.
4. **Idempotent:** Nếu chạy lại script, kết quả phải giống nhau (deterministic).

---

### Yêu cầu 5: Tiêu chí nghiệm thu (Acceptance Criteria)

- [ ] Script chạy được với input mẫu (17 nút HCM) và sinh ra file `zone_labels.csv` tương đương file hiện tại
- [ ] Script chạy được với input thành phố khác (ví dụ: 10 nút Hà Nội) mà không cần sửa code
- [ ] Output CSV có đúng 8 cột zone + cột `node`
- [ ] Mỗi nút có ít nhất 1 zone label (nếu không, log warning rõ ràng)
- [ ] File `zone_meta.json` được tạo ra chứa metadata đầy đủ

---
---

## Phần III: Phân tích Sâu Kiến trúc 3-Vector Embedding

### Tổng quan Kiến trúc Hiện tại

Mô hình Zone-Aware AH-GNN hiện tại sử dụng **3 vector embedding** chính để xây dựng ma trận kề động và điều biến trọng số graph convolution:

```
┌─────────────────────────────────────────────────────────────────┐
│                   3 EMBEDDING VECTORS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. E ∈ R^{N×d_e}     Node Spatial Embedding                   │
│     └─ nn.Parameter(randn(17, 32))                              │
│     └─ Hoàn toàn học được (learnable), khởi tạo ngẫu nhiên     │
│     └─ Đại diện cho "vị trí ẩn" của nút trong không gian đồ thị│
│                                                                 │
│  2. z̃ ∈ R^{N×d_z}    Zone Semantic Embedding                   │
│     └─ MLP(Z)  với Z ∈ {0,1}^{N×8}  (multi-hot zone labels)    │
│     └─ Chuyển đổi vector đa nhãn rời rạc → không gian liên tục │
│     └─ z̃ là TĨNH — không thay đổi theo thời gian               │
│                                                                 │
│  3. W_t ∈ R^{4×d_e×d_e}  Time-Varying Weight                   │
│     └─ nn.Parameter(randn(4, 32, 32))                           │
│     └─ 4 nhãn thời gian: night / rush_am / rush_pm / normal    │
│     └─ Biến đổi không gian embedding E theo ngữ cảnh thời gian │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Fusion: Ẽ_v = E_v + W_proj(z̃_v)         ← Cộng tính (additive)
        Ã_t = Softmax(ReLU(Ẽ·W_t·Ẽᵀ))  ← Ma trận kề động
        A   = α·Ã_t + (1-α)·A_osrm      ← Pha trộn với tĩnh
```

---

### Đánh giá Hiệu quả: Điểm mạnh

| Khía cạnh | Đánh giá | Giải thích |
|---|---|---|
| **Phân tách quan tâm (Separation of Concerns)** | ✅ Tốt | 3 vector phản ánh 3 nguồn thông tin độc lập: vị trí, chức năng, thời gian |
| **Multi-hot → Dense** | ✅ Tốt | MLP chuyển đổi zone labels rời rạc sang không gian liên tục, cho phép các tổ hợp zone tương tự nằm gần nhau |
| **Learnable blend α** | ✅ Tốt | Cho phép mô hình tự cân bằng giữa cấu trúc vật lý (OSRM) và cấu trúc ngữ nghĩa (zone) |
| **Kết quả ablation** | ✅ Tốt | zone_full (MAE=0.0795) >> baseline (MAE=0.2102), chứng minh zone embedding thực sự hữu ích |

---

### Đánh giá Hiệu quả: 4 Điểm yếu & Cải tiến đề xuất

#### Điểm yếu 1: Zone embedding TĨNH theo thời gian

**Vấn đề:** Hiện tại `z̃` được tính một lần từ `Z` và dùng chung cho mọi thời điểm. Nhưng trong thực tế, ảnh hưởng của vùng chức năng lên giao thông **phụ thuộc rất mạnh vào thời gian**:
- Vùng `school` chỉ gây tắc nghẽn lúc **7:00-7:30** (giờ vào học) và **16:30-17:00** (giờ tan học), nhưng hoàn toàn bình thường lúc 2:00 sáng.
- Vùng `industrial` tắc nghẽn lúc **17:00-18:30** (tan ca), nhưng cuối tuần gần như free-flow.

Hiện tại mô hình xử lý sự khác biệt thời gian này **chỉ qua $W_t$** (4 ma trận biến đổi rời rạc), nhưng $z̃$ giữ nguyên. Điều này có nghĩa: mô hình "biết" thời gian thay đổi nhưng **không biết rằng ảnh hưởng của zone cũng thay đổi theo thời gian**.

**Cải tiến đề xuất — Time-Conditioned Zone Embedding:**
```python
# Hiện tại (tĩnh):
z_embed = self.zone_emb(Z)              # (N, d_z) — giống nhau mọi lúc

# Đề xuất (động theo thời gian):
z_embed = self.zone_emb(Z, time_idx)    # (B, N, d_z) — khác nhau theo thời điểm
```

Cách thực hiện: thêm một lớp **gating** cho mỗi nhãn thời gian:
$$\tilde{\mathbf{z}}^{(v,t)} = \mathbf{g}_t \odot \text{MLP}(\mathbf{z}^{(v)})$$
trong đó $\mathbf{g}_t \in \mathbb{R}^{d_z}$ là "cổng thời gian" học được, tắt/bật ảnh hưởng từng loại zone theo thời điểm.

> [!TIP]
> **Mức độ ưu tiên: CAO.** Đây là cải tiến có tác động lớn nhất vì nó giải quyết một giới hạn cốt lõi của kiến trúc hiện tại.

---

#### Điểm yếu 2: Nhãn thời gian quá thô (4 danh mục rời rạc)

**Vấn đề:** Chỉ có 4 nhãn `[night, rush_am, rush_pm, normal]`. Sự phân chia này quá thô:
- "normal" bao trùm từ 6:00 đến 16:00 và 20:00 đến 24:00 — một khoảng rất rộng
- Không phân biệt ngày trong tuần vs. cuối tuần (thứ Hai rất khác Chủ Nhật)
- Mỗi nhãn chia sẻ chung một ma trận `W_t` cố định — mất tính mịn (smoothness)

**Cải tiến đề xuất — Sinusoidal Time Encoding:**
```python
# Hiện tại:
W = self.W_t[time_idx]    # Lookup bảng 4 nhãn → (B, d_e, d_e)

# Đề xuất: Mã hóa liên tục
hour_enc = sinusoidal_encoding(hour, day_of_week)  # (B, d_time)
W = self.time_proj(hour_enc)  # MLP → (B, d_e, d_e)
```

Mã hóa sin/cos cho phép mô hình học các mẫu hình tuần hoàn mịn (chu kỳ 24h, chu kỳ 7 ngày) thay vì nhảy bậc giữa 4 danh mục.

> [!TIP]
> **Mức độ ưu tiên: TRUNG BÌNH.** Cải tiến này dễ triển khai và cải thiện khả năng nắm bắt mẫu hình thời gian.

---

#### Điểm yếu 3: Phép cộng tính (Additive Fusion) có thể không đủ biểu diễn

**Vấn đề:** Hiện tại, zone embedding được kết hợp với node embedding bằng phép cộng:
$$\tilde{\mathbf{E}}_v = \mathbf{E}_v + \mathbf{W}_{\text{proj}} \tilde{\mathbf{z}}^{(v)}$$

Phép cộng giả định rằng spatial và semantic là **độc lập tuyến tính** — mỗi chiều spatial bị dịch chuyển (shift) một lượng cố định bởi zone. Nhưng trong thực tế, ảnh hưởng của zone lên cấu trúc không gian có thể phức tạp hơn (ví dụ: zone "transport" ở gần zone "commercial" có ảnh hưởng khác hẳn zone "transport" ở gần zone "industrial").

**Cải tiến đề xuất — Bilinear hoặc Gated Fusion:**
```python
# Phương án A: Gated fusion
gate = torch.sigmoid(self.W_gate(torch.cat([E, z_embed], dim=-1)))
E_fused = gate * E + (1 - gate) * self.W_proj(z_embed)

# Phương án B: Cross-attention
E_fused = CrossAttention(query=E, key=z_embed, value=z_embed)
```

> [!NOTE]
> **Mức độ ưu tiên: THẤP.** Kiến trúc cộng tính hiện tại đã cho kết quả tốt. Cải tiến này nên thử sau khi giải quyết Điểm yếu 1 & 2.

---

#### Điểm yếu 4: Node Embedding E khởi tạo ngẫu nhiên, không có prior vật lý

**Vấn đề:** `E = nn.Parameter(randn(17, 32))` khởi tạo hoàn toàn ngẫu nhiên. Với chỉ 17 nút và 658 mẫu huấn luyện, không gian dữ liệu khá nhỏ. E có thể overfit hoặc mất thông tin hình học ban đầu.

**Cải tiến đề xuất — Khởi tạo từ tọa độ GPS:**
```python
# Khởi tạo E với prior vật lý thay vì ngẫu nhiên
coords = torch.tensor([[10.7725, 106.698], ...])  # (N, 2) GPS
E_init = self.coord_proj(coords)  # Linear(2, d_e) → (N, d_e)
self.E = nn.Parameter(E_init)     # Vẫn learnable, nhưng khởi đầu tốt hơn
```

> [!NOTE]
> **Mức độ ưu tiên: THẤP.** Là một cải tiến nhỏ nhưng dễ thực hiện và hợp lý về mặt khoa học.

---

### Tóm tắt: Ma trận Ưu tiên Cải tiến

| # | Cải tiến | Ưu tiên | Effort | Impact | Ai làm? |
|---|---|---|---|---|---|
| 1 | Time-Conditioned Zone Embedding | 🔴 CAO | Trung bình | Lớn — giải quyết giới hạn cốt lõi | Thành viên A + B |
| 2 | Sinusoidal Time Encoding | 🟡 TB | Thấp | Trung bình — mịn hóa temporal | Thành viên B |
| 3 | Gated/Attention Fusion | 🟢 Thấp | Trung bình | Nhỏ-TB — đã hoạt động tốt | Tuần 2-3 |
| 4 | GPS-initialized E | 🟢 Thấp | Rất thấp | Nhỏ — tăng convergence | Thành viên B |

---

### Kết luận: Kiến trúc 3-vector có hiệu quả không?

**Có, kiến trúc 3-vector là một thiết kế tốt** vì nó phản ánh đúng 3 nguồn thông tin độc lập trong bài toán (không gian vật lý, ngữ nghĩa chức năng đất, ngữ cảnh thời gian). Kết quả ablation study (MAE giảm 62% so với baseline) xác nhận hiệu quả rõ ràng.

Tuy nhiên, **điểm yếu lớn nhất hiện tại là zone embedding tĩnh**. Trong giao thông thực tế, ảnh hưởng của loại vùng đất lên lưu lượng thay đổi rất mạnh theo thời gian, và kiến trúc hiện tại chưa nắm bắt được điều này. Nếu nhóm triển khai **Time-Conditioned Zone Embedding** (Cải tiến #1), đây sẽ trở thành một **đóng góp khoa học bổ sung mạnh mẽ** cho bài báo.
