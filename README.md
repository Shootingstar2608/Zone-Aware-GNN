# Zone-Aware AH-GNN: Urban Traffic Prediction under Non-IID Conditions

> Giải quyết bài toán dự báo lưu lượng giao thông đô thị với dữ liệu không đồng nhất (Non-IID) bằng cách tích hợp ngữ nghĩa vùng chức năng đất đai (TAZ Zone Labels) vào kiến trúc GNN thích ứng.

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
source venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Cấu hình API Key (cần có TomTom API Key)

```bash
cp .env.example .env
# Điền TOMTOM_API_KEY=your_key vào file .env
```

### 3. Kiểm tra pipeline

```bash
venv/bin/python scripts/quick_test.py
```

Kết quả mong đợi: `✅` cho tất cả 5 tests — OSRM, Adjacency, Speed Proxy, Zone Labels, Model Forward Pass.

### 4. Xây dựng đồ thị (build graph dataset)

```bash
# Tạo nhãn vùng từ OSM (cần internet)
venv/bin/python scripts/collect_zones.py

# Xây dựng dataset PyG
venv/bin/python scripts/build_graph.py
# Output: data/processed/graph_dataset.pt + meta.json
```

### 5. Phân tích thống kê Non-IID (EDA)

```bash
venv/bin/python scripts/run_eda.py
# Output: data/results/eda_jsd_heatmap.png + eda_jsd_correlation.png
```

### 6. Huấn luyện & Ablation Study

```bash
# Huấn luyện mô hình đề xuất
venv/bin/python scripts/train.py --variant zone_full

# Chạy toàn bộ 4 variants ablation study
venv/bin/python scripts/train.py --ablation
# Output: data/results/ablation_results.csv
```

---

## 📊 Kết quả Thực nghiệm (HCM-Zone Dataset)

| Variant | Zone Embed | Zone Weight | Zone Adj | MAE | RMSE | MAPE |
|---|:---:|:---:|:---:|---:|---:|---:|
| AH-GNN (Baseline) | ❌ | ❌ | ❌ | 0.2102 | 0.2955 | 15.83% |
| + Zone Concat | ✅ | ❌ | ❌ | 0.0963 | 0.1536 | 7.11% |
| + Zone Weight | ✅ | ✅ | ❌ | 0.1057 | 0.1613 | 7.84% |
| **Zone-Aware (Đề xuất)** | ✅ | ✅ | ✅ | **0.0795** | **0.1419** | **6.14%** |

MAE giảm **62.2%** so với baseline. Multi-Zone MAE giảm **61.7%**.

---

## 📁 Dataset: HCM-Zone

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
