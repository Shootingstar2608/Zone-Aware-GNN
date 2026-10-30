"""
scripts/pipeline/enrich_static_metadata.py
==========================================
Tầng 3: Gắn Static Metadata Đa Nguồn (Multi-source Linking).
- Liên kết bảng 17 segments của registry với:
  + OSM Land-use zones (data/raw/zone_labels.csv) -> prefix osm_*
  + OSRM routing metrics (data/raw/hcm_osrm_dataset.csv) -> prefix osrm_*
  + Registry provenance -> prefix registry_*
- Sinh ma trận kề adjacency.pt (17 x 17).
- Xuất bảng nodes.csv và nodes.parquet.
- Đảm bảo 100% không có trường của các nguồn ghi đè lên nhau.
"""

from __future__ import annotations

import json
import os
import pandas as pd
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


ZONE_TYPES = [
    "commercial",
    "residential",
    "industrial",
    "school",
    "university",
    "hospital",
    "transport",
    "park",
]


def compute_haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Tính khoảng cách Great-Circle (Haversine) giữa hai tọa độ WGS-84 (mét)."""
    R = 6371000.0  # Bán kính Trái Đất (mét)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)


def build_static_metadata(
    registry_path: str,
    zone_path: str,
    osrm_path: str,
    out_dir: str,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Tạo bảng nodes static và ma trận kề A không ghi đè dữ liệu giữa các nguồn.
    Đảm bảo tính chuẩn tắc toán học cho ma trận khoảng cách:
      - d(i, i) = 0.0
      - d(i, j) > 0.0 với mọi i != j (Tuyệt đối không gán 0.0 cho cặp không có kết nối!)
      - Cặp không có đường đi OSRM: d(i, j) = np.inf (vô cùng)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Đọc Registry 17 segments
    with open(registry_path, "r", encoding="utf-8") as f:
        registry_data = json.load(f)

    segments = registry_data["segments"]
    nodes_order = [s["node_id"] for s in segments]
    node2idx = {name: idx for idx, name in enumerate(nodes_order)}
    node_coords = {
        s["node_id"]: (float(s.get("lat", 0.0)), float(s.get("lon", 0.0)))
        for s in segments
    }
    N = len(nodes_order)

    # DataFrame khởi tạo từ registry
    df_nodes = pd.DataFrame([
        {
            "node_id": s["node_id"],
            "segment_id": s["segment_id"],
            "approved_openlr": s["approved_openlr"],
            "registry_lat": float(s["lat"]),
            "registry_lon": float(s["lon"]),
            "registry_corridor": s["corridor_name"],
            "registry_selection_reason": s["selection_reason"],
            "registry_status": s["status"],
            "tomtom_static_length_m": float(s.get("segment_length_m", 1000.0)),
            "tomtom_static_free_flow_speed_kmh": float(s.get("default_free_flow_speed_kmh", 45.0)),
        }
        for s in segments
    ])

    # 2. Ghép nhãn vùng OSM (zone_labels.csv) với prefix osm_*
    if os.path.exists(zone_path):
        df_zones = pd.read_csv(zone_path)
        # Chuẩn hóa tên cột tham chiếu
        node_col = "node" if "node" in df_zones.columns else df_zones.columns[0]
        df_zones = df_zones.set_index(node_col)

        for z in ZONE_TYPES:
            if z in df_zones.columns:
                df_nodes[f"osm_{z}"] = df_nodes["node_id"].map(df_zones[z]).fillna(0.0).astype(float)
            else:
                df_nodes[f"osm_{z}"] = 0.0
        df_nodes["osm_source"] = "OpenStreetMap_2026"
    else:
        for z in ZONE_TYPES:
            df_nodes[f"osm_{z}"] = 0.0
        df_nodes["osm_source"] = "missing"

    # 3. Tính toán routing metrics từ OSRM (hcm_osrm_dataset.csv) với prefix osrm_*
    if os.path.exists(osrm_path):
        df_osrm = pd.read_csv(osrm_path)
        # Tính khoảng cách và free-flow trung bình tới các node lân cận
        dist_agg = df_osrm.groupby("origin")["distance_m"].mean().to_dict()
        dur_agg = df_osrm.groupby("origin")["duration_s"].mean().to_dict()

        df_nodes["osrm_avg_outgoing_dist_m"] = df_nodes["node_id"].map(dist_agg).fillna(0.0).round(1)
        df_nodes["osrm_avg_outgoing_duration_s"] = df_nodes["node_id"].map(dur_agg).fillna(0.0).round(1)
        df_nodes["osrm_source"] = "OSRM_Car_Routing_v1"

        # Xây dựng ma trận kề A (17 x 17) và ma trận khoảng cách OSRM tabular
        A_raw = np.zeros((N, N), dtype=np.float32)
        cnt = np.zeros((N, N), dtype=np.int32)
        dist_sum = np.zeros((N, N), dtype=np.float32)
        dist_cnt = np.zeros((N, N), dtype=np.int32)

        for _, row in df_osrm.iterrows():
            orig = row.get("origin")
            dest = row.get("destination")
            dur = float(row.get("duration_s", 0.0))
            dist = float(row.get("distance_m", 0.0))
            if orig in node2idx and dest in node2idx:
                i = node2idx[orig]
                j = node2idx[dest]
                A_raw[i, j] += dur
                cnt[i, j] += 1
                dist_sum[i, j] += dist
                dist_cnt[i, j] += 1

        # XÂY DỰNG MA TRẬN KHOẢNG CÁCH CHUẨN TOÁN HỌC:
        # Bất biến metric space:
        # 1. d(i, i) = 0.0
        # 2. d(i, j) > 0 với mọi i != j (Identity of indiscernibles)
        # 3. Nếu không có kết nối OSRM (dist_cnt == 0): gán np.inf (vô cùng), TUYỆT ĐỐI KHÔNG ĐƯỢC GÁN 0.0!
        dist_matrix_mean = np.full((N, N), np.inf, dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i == j:
                    dist_matrix_mean[i, j] = 0.0
                elif dist_cnt[i, j] > 0:
                    dist_matrix_mean[i, j] = dist_sum[i, j] / dist_cnt[i, j]
                else:
                    dist_matrix_mean[i, j] = np.inf

        df_dist_matrix = pd.DataFrame(dist_matrix_mean, index=nodes_order, columns=nodes_order)

        # XÂY DỰNG MA TRẬN KỀ A_norm:
        # W_ij = 1.0 / duration nếu có kết nối (cnt > 0 và dur > 0), ngược lại = 0.0
        A_inv = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j and cnt[i, j] > 0 and A_raw[i, j] > 0:
                    avg_dur = A_raw[i, j] / cnt[i, j]
                    A_inv[i, j] = 1.0 / avg_dur

        # Chuẩn hóa hàng
        row_sums = A_inv.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        A_norm = A_inv / row_sums
        if HAS_TORCH:
            adj_tensor = torch.tensor(A_norm, dtype=torch.float32)
        else:
            adj_tensor = A_norm
    else:
        # Fallback khi file OSRM không tồn tại:
        # Tính khoảng cách hình học Haversine từ GPS trong registry thay vì điền 0.0 mù quáng!
        df_nodes["osrm_avg_outgoing_dist_m"] = 0.0
        df_nodes["osrm_avg_outgoing_duration_s"] = 0.0
        df_nodes["osrm_source"] = "missing_osrm_fallback_haversine"

        dist_matrix_mean = np.full((N, N), np.inf, dtype=np.float32)
        A_inv = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(N):
                if i == j:
                    dist_matrix_mean[i, j] = 0.0
                else:
                    lat_i, lon_i = node_coords.get(nodes_order[i], (0.0, 0.0))
                    lat_j, lon_j = node_coords.get(nodes_order[j], (0.0, 0.0))
                    if lat_i != 0.0 and lat_j != 0.0:
                        # Khoảng cách chim bay nhân hệ số uốn khúc đường phố đô thị TP.HCM (~1.35)
                        d_hav = compute_haversine_distance_m(lat_i, lon_i, lat_j, lon_j) * 1.35
                        dist_matrix_mean[i, j] = d_hav
                        A_inv[i, j] = 1.0 / d_hav if d_hav > 0 else 0.0
                    else:
                        dist_matrix_mean[i, j] = np.inf

        row_sums = A_inv.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        A_norm = A_inv / row_sums

        df_dist_matrix = pd.DataFrame(dist_matrix_mean, index=nodes_order, columns=nodes_order)
        if HAS_TORCH:
            adj_tensor = torch.tensor(A_norm, dtype=torch.float32)
        else:
            adj_tensor = A_norm

    # 4. Xuất file artifacts
    nodes_csv_path = os.path.join(out_dir, "nodes.csv")
    nodes_parquet_path = os.path.join(out_dir, "nodes.parquet")
    adj_path = os.path.join(out_dir, "adjacency.pt")
    adj_npy_path = os.path.join(out_dir, "adjacency.npy")
    dist_csv_path = os.path.join(out_dir, "osrm_distance_matrix.csv")

    df_nodes.to_csv(nodes_csv_path, index=False)
    df_dist_matrix.to_csv(dist_csv_path)
    try:
        df_nodes.to_parquet(nodes_parquet_path, index=False)
    except Exception:
        pass

    if HAS_TORCH:
        torch.save(adj_tensor, adj_path)
    else:
        np.save(adj_npy_path, A_norm)
        with open(adj_path, "wb") as f:
            np.save(f, A_norm)

    return df_nodes, adj_tensor
