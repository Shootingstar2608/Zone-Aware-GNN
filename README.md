# Zone-Aware AH-GNN: Urban Traffic Prediction under Non-IID Conditions

> Giải quyết bài toán dự báo lưu lượng giao thông đô thị với dữ liệu không đồng nhất (Non-IID) bằng cách tích hợp ngữ nghĩa vùng chức năng đất đai (TAZ Zone Labels) vào kiến trúc GNN thích ứng.

> **Trạng thái benchmark hiện hành:** `HCM-Sim v1` là benchmark **synthetic**,
> không phải quan sát giao thông thực từ TomTom. Dùng nó để kiểm chứng protocol
> Non-IID có thể tái lập; không diễn giải kết quả như hiệu năng trên traffic thực.
> Contract chính thức: [`docs/hcm_sim_v1_spec.md`](docs/hcm_sim_v1_spec.md).

---

## 📐 Kiến trúc Mô hình

```
Z (Multi-label Zones) ──► ZoneEmbedding (MLP)
                                │  z̃ (N, d_z)
                                ▼
E (Spatial Embedding) ──► ZoneAwareAdjacency ──► Ã_t (B, N, N)
                                │                      │
                                ▼                      │
                       ZoneModulatedConv ◄─────────────┘
                         (W_v per node)
                                │
                                ▼
                            GRU / FC ──► Ŷ (B, N, T_out)
```

**3 đổi mới chính** so với AH-GNN chuẩn:
1. **Zone Embedding** — mã hóa nhãn vùng đa nhãn `{0,1}^K → R^{d_z}`
2. **Zone-Modulated Weight Generation** — trọng số tích chập `W_v` riêng cho từng nút
3. **Zone-Biased Adaptive Adjacency** — ma trận kề động ưu tiên nút cùng ngữ nghĩa vùng

---

## 🗂️ Cấu trúc Dự án

```
Research/
├── paper/
│   ├── main.tex          # Bản thảo bài báo (English — bản submit)
│   ├── draft_vi.tex      # Bản thảo tiếng Việt (working draft nhóm)
│   └── figures/          # Biểu đồ JSD heatmap, scatter
│
├── models/
│   ├── zone_aware_gnn.py # Model đề xuất chính
│   ├── ah_gnn.py         # Baseline AH-GNN (không zone)
│   └── baselines.py      # [WIP] LSTM, GCN-GRU, STGCN
│
├── scripts/
│   ├── train.py          # Huấn luyện + ablation study (4 variants)
│   ├── build_graph.py    # Xây dựng đồ thị PyG từ 3 nguồn dữ liệu
│   ├── run_eda.py        # Phân tích JSD & Non-IID evidence
│   ├── collect_zones.py  # Crawl nhãn vùng từ OSM/Overpass API
│   ├── tomtom_collector.py # Thu thập dữ liệu giao thông TomTom
│   ├── quick_test.py     # Kiểm tra toàn pipeline (5 tests)
│   └── dev/              # Scripts phát triển/debug (không production)
│
├── data/
│   ├── raw/              # Dữ liệu thô gốc
│   ├── processed/        # graph_dataset.pt + meta.json
│   └── results/          # Kết quả thực nghiệm, model checkpoints
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🚀 Quickstart

### 1. Cài đặt môi trường

```bash
python3 -m venv venv
source venv/bin/activate  |  venv\Scripts\Activate.ps1
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Tạo và kiểm tra HCM-Sim v1 (không cần API key)

```bash
venv/bin/python scripts/dev/generate_synthetic_traffic.py \
  --seed 42 --days 7 --out data/generated/hcm_sim_v1/hcm_sim_traffic.csv
venv/bin/python scripts/build_graph.py \
  --traffic-path data/generated/hcm_sim_v1/hcm_sim_traffic.csv \
  --out-dir data/processed/hcm_sim_v1 --t-in 12 --t-out 24
venv/bin/python scripts/quick_test.py --data-dir data/processed/hcm_sim_v1
```

Sinh và kiểm chứng metadata partition độc lập:

```bash
venv/bin/python -m benchmark.partition_gen --data-dir data/processed/hcm_sim_v1
venv/bin/python scripts/verify_partitions.py --data-dir data/processed/hcm_sim_v1
```

### 3. Cấu hình API Key (chỉ cho pipeline TomTom lịch sử)

```bash
cp .env.example .env
# Điền TOMTOM_API_KEY=your_key vào file .env
```

### 4. Kiểm tra pipeline legacy

```bash
venv/bin/python scripts/quick_test.py
```

Kết quả mong đợi: `✅` cho tất cả 5 tests — OSRM, Adjacency, Speed Proxy, Zone Labels, Model Forward Pass.

### 5. Xây dựng đồ thị legacy (build graph dataset)

```bash
# Tạo nhãn vùng từ OSM (cần internet)
venv/bin/python scripts/collect_zones.py

# Xây dựng dataset PyG
venv/bin/python scripts/build_graph.py
# Output: data/processed/graph_dataset.pt + meta.json
```

### 6. Phân tích thống kê Non-IID (EDA)

```bash
venv/bin/python scripts/run_eda.py
# Output: data/results/eda_jsd_heatmap.png + eda_jsd_correlation.png
```

### 7. Huấn luyện & Ablation Study

```bash
# Huấn luyện mô hình đề xuất
venv/bin/python scripts/train.py --data-dir data/processed/hcm_sim_v1 \
  --out-dir data/results/hcm_sim_v1 --epochs 100 --variant zone_full

# Chạy toàn bộ 4 variants ablation study
venv/bin/python scripts/train.py --ablation
# Output: data/results/ablation_results.csv
```

---

## 📊 Kết quả lịch sử (không dùng làm claim paper)

| Variant | Zone Embed | Zone Weight | Zone Adj | MAE | RMSE | MAPE |
|---|:---:|:---:|:---:|---:|---:|---:|
| AH-GNN (Baseline) | ❌ | ❌ | ❌ | 0.2102 | 0.2955 | 15.83% |
| + Zone Concat | ✅ | ❌ | ❌ | 0.0963 | 0.1536 | 7.11% |
| + Zone Weight | ✅ | ✅ | ❌ | 0.1057 | 0.1613 | 7.84% |
| **Zone-Aware (Đề xuất)** | ✅ | ✅ | ✅ | **0.0795** | **0.1419** | **6.14%** |

Các con số trên là kết quả lịch sử, chưa đủ điều kiện làm claim paper: phải
chạy lại trên chrono split, nhiều seed, và báo cáo độ bất định/kiểm định.

---

## 📁 Dataset chính: HCM-Sim v1

| Thành phần | Giá trị |
|---|---|
| Bản chất | Synthetic, sinh xác định từ seed 42 |
| Node / zone | 17 node TP.HCM, 8 multi-label zones từ OSM/OSRM artifacts |
| Chuỗi | 7 ngày, interval 15 phút, 672 snapshots |
| Input / target | `T_in=12`, `T_out=24`, 4 feature động |
| Cửa sổ | `S=637`, kèm `time_labels` 4 lớp |
| Provenance | raw metadata + processed manifest + SHA-256 hash |

Dataset OSRM/TomTom bên dưới là pipeline lịch sử, cần audit nguồn và license
trước khi được dùng để công bố.

## 📁 Dataset lịch sử: HCM-Zone

| Nguồn | Vai trò | Kích thước |
|---|---|---|
| OSRM (OpenStreetMap) | Ma trận kề tĩnh `A` | 20,377 rows, 17 nút |
| TomTom Routing API | Đặc trưng động `X_t` | 672 snapshots × 272 cặp |
| Overpass API (OSM) | Nhãn vùng `Z` | 17 nút × 8 loại vùng |

---

## 🗓️ Kế hoạch Nhóm (4 người, 4 tuần)

| Thành viên | Tuần 1 | Tuần 2 | Tuần 3 | Tuần 4 |
|---|---|---|---|---|
| A (Writing) | Viết Section 3-4 | Granger Causality | Viết Section 5 | Review & Submit |
| B (DL Eng) | Triển khai baselines | Fine-tune hyperparams | Cross-city validation | Review |
| C (Data/GIS) | Generic OSM crawler | Mở rộng dữ liệu | Cross-city data prep | Review |
| D (Simulation) | Setup CityFlow | Tích hợp GNN → đèn tín hiệu | Chạy kịch bản cực đoan | Review |

---

## 📚 Citation

```bibtex
@article{zone_aware_ahgnn_2026,
  title   = {Multi-Label Zone-Aware Adaptive Heterogeneous GNN
             for Urban Traffic Flow Prediction under Non-IID Conditions},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {2026}
}
```
