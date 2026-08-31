# 📊 Data Quality Report — HCM-Zone Dataset

> **Generated:** 2026-08-31 17:49:32

## Nguồn dữ liệu đã xử lý

| File | Trạng thái |
|---|---|
| `graph_dataset.pt` | ✅ Có |
| `meta.json` | ✅ Có |
| `hcm_osrm_dataset.csv (raw)` | ❌ Không có |
| `tomtom_traffic.csv (raw)` | ❌ Không có |
| `zone_labels.csv (raw)` | ✅ Có |

## Summary Dashboard

| Tiêu chí | Score | Status | Các giá trị thành phần (Sub-scores) |
|---|:---:|:---:|---|
| Completeness | 1.0000 | ✅ | `processed: 1.0000`, `zone_tensor: 1.0000` |
| Validity | 1.0000 | ✅ | `X_features: 1.0000`, `adjacency: 1.0000`, `zone_tensor: 1.0000`, `Y_target: 1.0000` |
| Consistency | 0.9717 | ✅ | `adjacency_consistency: 0.8866`, `feature_consistency: 1.0000`, `cross_source: 1.0000` (bỏ qua), `zone_node_consistency: 1.0000` |
| Freshness | 0.8507 | ✅ | `time_label_distribution: 0.4028`, `hour_dow_coverage: 1.0000` (bỏ qua), `temporal_coverage: 1.0000`, `raw_timestamps: 1.0000` (bỏ qua) |
| Source Reliability | 0.9123 | ✅ | `osrm: 0.8361`, `tomtom: 0.9352`, `osm_zones: 0.9658` |
| **Overall** | **0.9469** | | (Trung bình cộng 5 tiêu chí trên) |

---

## 1. Completeness (Score: 1.0000)

### ℹ️ Info

- Dataset: 637 samples, 17 nodes
- Zone tensor: 17 nodes × 8 zone types

### 📋 Chi tiết

#### processed

| Metric | Value |
|---|---|
| n_samples | 637 |
| n_nodes | 17 |
| **nan_rates** | |
| — X | 0.000000 |
| — Y | 0.000000 |
| — A | 0.000000 |
| — Z | 0.000000 |
| **zero_rates** | |
| — X | 0.000000 |
| — Y | 0.000000 |
| **inf_rates** | |
| — X | 0.000000 |
| — Y | 0.000000 |
| shape_valid | True |
| missing_keys | [] |

#### zone_tensor

| Metric | Value |
|---|---|
| n_nodes | 17 |
| n_zone_types | 8 |
| nan_rate | 0.000000 |
| nodes_no_zone | 0 |
| coverage | 1.000000 |

---

## 2. Validity (Score: 1.0000)

### 📋 Chi tiết

#### X_features

| Metric | Value |
|---|---|
| has_nan | False |
| has_inf | False |
| **feature_violations** | |
| — congestion_ratio | {'range': [0.0, 5.0], 'total': 129948, 'below_min': 0, 'above_max': 0, 'violation_rate': 0.0, 'actual_range': [0.960812509059906, 2.542562484741211]} |
| — traffic_delay_s | {'range': [0.0, 7200.0], 'total': 129948, 'below_min': 0, 'above_max': 0, 'violation_rate': 0.0, 'actual_range': [0.03125, 1471.268798828125]} |
| — travel_time_s | {'range': [0.0, 10800.0], 'total': 129948, 'below_min': 0, 'above_max': 0, 'violation_rate': 0.0, 'actual_range': [535.2437744140625, 2414.324951171875]} |
| — ff_ratio | {'range': [0.5, 10.0], 'total': 129948, 'below_min': 0, 'above_max': 0, 'violation_rate': 0.0, 'actual_range': [0.9608085751533508, 2.5425519943237305]} |

#### adjacency

| Metric | Value |
|---|---|
| is_square | True |
| range | [0.0, 0.27715563774108887] |
| in_01_range | True |
| diagonal_mean | 0.000000 |
| has_self_loops | False |
| density | 0.941176 |
| n_nonzero | 272 |

#### zone_tensor

| Metric | Value |
|---|---|
| is_binary | True |
| unique_values | [0.0, 1.0] |
| nodes_without_zone | 0 |
| min_zones_per_node | 1 |
| max_zones_per_node | 6 |
| mean_zones_per_node | 3.470588 |

#### Y_target

| Metric | Value |
|---|---|
| has_nan | False |
| has_inf | False |
| range | [0.960812509059906, 2.542562484741211] |
| mean | 1.266125 |
| n_negative | 0 |
| n_extreme_high | 0 |

---

## 3. Consistency (Score: 0.9717)

### ℹ️ Info

- ⚠️ Bỏ qua kiểm tra cross-source (OSRM vs TomTom). Lý do: Không tìm thấy dữ liệu thô (hcm_osrm_dataset.csv, tomtom_traffic.csv). Pipeline hiện tại thiết kế chỉ nhận được dữ liệu đã xử lý cuối cùng (graph_dataset.pt).

### 📋 Chi tiết

#### adjacency_consistency

| Metric | Value |
|---|---|
| mean_asymmetry | 0.020368 |
| max_asymmetry | 0.110198 |
| **triangle_inequality** | |
| — violations | 94 |
| — total_triplets | 4080 |
| — violation_rate | 0.023039 |
| **row_sum_stats** | |
| — mean | 1.000000 |
| — std | 0.000000 |
| — min | 1.000000 |
| — max | 1.000000 |

#### feature_consistency

| Metric | Value |
|---|---|
| cr_ff_correlation | 1.000000 |
| cr_ff_consistent | True |
| **temporal_smoothness** | |
| — mean_step_change | 0.048065 |
| — max_step_change | 1.324400 |

#### cross_source

| Metric | Value |
|---|---|
| info | ['⚠️ Bỏ qua kiểm tra cross-source (OSRM vs TomTom). Lý do: Không tìm thấy dữ liệu thô (hcm_osrm_dataset.csv, tomtom_traffic.csv). Pipeline hiện tại thiết kế chỉ nhận được dữ liệu đã xử lý cuối cùng (graph_dataset.pt).'] |

#### zone_node_consistency

| Metric | Value |
|---|---|
| n_matched_nodes | 17 |
| n_total_nodes | 17 |

---

## 4. Freshness (Score: 0.8507)

### ℹ️ Info

- ⚠️ Bỏ qua đo khoảng cách timestamp giữa các snapshot. Lý do: Thiếu file dữ liệu thô tomtom_traffic.csv. Thông tin timestamp gốc đã bị mất trong quá trình tạo tensor (chỉ còn mảng time_labels).

### 📋 Chi tiết

#### time_label_distribution

| Metric | Value |
|---|---|
| **label_distribution** | |
| — night | {'count': 157, 'ratio': 0.24646781789638933} |
| — rush_morning | {'count': 0, 'ratio': 0.0} |
| — rush_evening | {'count': 0, 'ratio': 0.0} |
| — normal | {'count': 480, 'ratio': 0.7535321821036107} |
| total_samples | 637 |
| entropy | 0.805600 |
| max_entropy | 2.000000 |
| entropy_ratio | 0.402800 |

#### hour_dow_coverage

| Metric | Value |
|---|---|
| info | ["⚠️ Bỏ qua đo hour/dow coverage do graph_dataset.pt không có trường 'hour' và 'dow'."] |

#### temporal_coverage

| Metric | Value |
|---|---|
| estimated_total_timesteps | 672 |
| estimated_duration_hours | 56.000000 |
| daily_coverage_ratio | 1.000000 |
| estimated_days | 2.300000 |

#### raw_timestamps

| Metric | Value |
|---|---|
| info | ['⚠️ Bỏ qua đo khoảng cách timestamp giữa các snapshot. Lý do: Thiếu file dữ liệu thô tomtom_traffic.csv. Thông tin timestamp gốc đã bị mất trong quá trình tạo tensor (chỉ còn mảng time_labels).'] |

---

## 5. Source Reliability (Score: 0.9123)

### ℹ️ Info

- OSRM reliability: 0.84
- TomTom reliability: 0.94
- OSM zone reliability: 0.97

### 📋 Chi tiết

#### osrm

| Metric | Value |
|---|---|
| mean_relative_asymmetry | 0.300800 |
| n_isolated_nodes | 0 |
| min_out_degree | 16 |
| max_out_degree | 16 |
| weight_cv | 0.572900 |

#### tomtom

| Metric | Value |
|---|---|
| **per_node_stats** | |
| — mean_cr | 1.253782 |
| — mean_std | 0.322879 |
| — max_std | 0.343657 |
| mean_cv_per_node | 0.257500 |
| repeated_value_rate | 0.000900 |

#### osm_zones

| Metric | Value |
|---|---|
| coverage | 1.000000 |
| zone_diversity | 0.931600 |
| **zone_count_distribution** | |
| — 1 | 1 |
| — 2 | 3 |
| — 3 | 5 |
| — 4 | 4 |
| — 5 | 3 |
| — 6 | 1 |
| mean_zones_per_node | 3.470000 |
| **zone_type_coverage** | |
| — commercial | 13 |
| — residential | 13 |
| — industrial | 3 |
| — school | 8 |
| — university | 4 |
| — hospital | 4 |
| — transport | 10 |
| — park | 4 |

---

## 💡 Recommendations

- ✅ Dữ liệu đầy đủ, không cần bổ sung
