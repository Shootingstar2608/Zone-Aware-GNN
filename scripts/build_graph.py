"""
Step 2: build_graph.py
======================
Kết hợp 3 nguồn dữ liệu thành PyTorch Geometric dataset:
  - OSRM  → Adjacency Matrix A
  - TomTom → Node Features X_t (dynamic, per timestep)
  - OSM    → Zone Labels Z (static, multi-hot)

Chạy: python scripts/build_graph.py
Output: data/processed/graph_dataset.pt
        data/processed/meta.json
"""

import os
import json
import numpy as np
import pandas as pd
import torch

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
OSRM_PATH    = "data/raw/hcm_osrm_dataset.csv"
TOMTOM_PATH  = "data/raw/tomtom_traffic.csv"   # Sẽ có sau khi thu thập
ZONE_PATH    = "data/raw/zone_labels.csv"
OUT_DIR      = "data/processed"

ZONE_TYPES   = ["commercial","residential","industrial",
                "school","university","hospital","transport","park"]

T_IN  = 12   # 12 timesteps input  (ví dụ: 12 × 5min = 1 giờ)
T_OUT = 3    # 3  timesteps predict (15 phút)

TIME_LABEL_MAP = {   # giờ → time label index cho W_t
    **{h: 0 for h in range(0, 6)},      # night
    **{h: 1 for h in range(7, 10)},     # rush morning
    **{h: 2 for h in range(16, 20)},    # rush evening
    **{h: 3 for h in range(6, 24)},     # normal (fallback)
}


# ══════════════════════════════════════════════
# MODULE 1: Build Adjacency từ OSRM
# ══════════════════════════════════════════════
def build_adjacency(df_osrm: pd.DataFrame, node2idx: dict) -> np.ndarray:
    """
    A[i][j] = 1 / mean_duration(i→j)
    Không có dữ liệu → 0
    """
    N = len(node2idx)
    A = np.zeros((N, N))
    cnt = np.zeros((N, N))

    for _, row in df_osrm.iterrows():
        i = node2idx.get(row["origin"])
        j = node2idx.get(row["destination"])
        if i is not None and j is not None:
            A[i][j] += row["duration_s"]
            cnt[i][j] += 1

    cnt[cnt == 0] = 1
    A = A / cnt
    A_norm = np.where(A > 0, 1.0 / A, 0.0)

    # Row normalize
    row_sum = A_norm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return A_norm / row_sum


# ══════════════════════════════════════════════
# MODULE 2: Build Node Features từ TomTom
# ══════════════════════════════════════════════
def build_node_features_tomtom(df_tt: pd.DataFrame, node2idx: dict,
                                nodes: list) -> tuple:
    """
    Pivot edge-level TomTom data → node-level features per timestep.

    TomTom schema (tomtom_collector.py output):
      timestamp, time_label, src_node, dst_node,
      src_name, dst_name, src_poi, dst_poi,
      travel_time_s, free_flow_time_s, traffic_delay_s,
      length_m, congestion_ratio

    Returns:
      X:      np.array shape (T, N, F) — F=4 features
      times:  list of (timestamp, time_label_idx)
    """
    df_tt = df_tt.copy()
    df_tt["timestamp"] = pd.to_datetime(df_tt["timestamp"])
    df_tt["hour"] = df_tt["timestamp"].dt.hour

    # Map src_name / dst_name về node index
    df_tt["src_idx"] = df_tt["src_name"].map(node2idx)
    df_tt["dst_idx"] = df_tt["dst_name"].map(node2idx)
    df_tt = df_tt.dropna(subset=["src_idx", "dst_idx"])

    # Tập hợp theo snapshot (timestamp)
    snapshots = sorted(df_tt["timestamp"].unique())
    N = len(nodes)
    F = 4   # [congestion_ratio, traffic_delay_s, travel_time_s, ff_ratio]

    X_list, time_list = [], []

    for ts in snapshots:
        snap = df_tt[df_tt["timestamp"] == ts]
        feat = np.zeros((N, F))
        cnt  = np.zeros((N, 1))

        for _, row in snap.iterrows():
            i = int(row["src_idx"])
            ff = row["free_flow_time_s"] or 1
            ff_ratio = row["travel_time_s"] / ff if ff > 0 else 1.0
            feat[i] += [
                row["congestion_ratio"],
                row["traffic_delay_s"],
                row["travel_time_s"],
                ff_ratio,
            ]
            cnt[i] += 1

        cnt[cnt == 0] = 1
        feat = feat / cnt

        hour = pd.Timestamp(ts).hour
        tl = TIME_LABEL_MAP.get(hour, 3)

        X_list.append(feat)
        time_list.append(tl)

    X = np.stack(X_list, axis=0)   # (T, N, F)
    return X, time_list


def build_node_features_osrm_proxy(df_osrm: pd.DataFrame,
                                    node2idx: dict, nodes: list) -> tuple:
    """
    FALLBACK khi chưa có TomTom.
    Dùng speed proxy từ OSRM: speed = distance/duration (km/h)
    Node feature = avg outgoing speed, aggregated theo giờ.

    Returns:
      X:      np.array shape (T, N, 1)
      times:  list of time_label_idx
    """
    df = df_osrm.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]  = df["timestamp"].dt.hour
    df["speed"] = (df["distance_m"] / df["duration_s"]) * 3.6

    N  = len(nodes)
    X_list, time_list = [], []

    for hour in range(24):
        sub = df[df["hour"] == hour]
        feat = np.zeros((N, 1))
        cnt  = np.zeros((N, 1))
        for _, row in sub.iterrows():
            i = node2idx.get(row["origin"])
            if i is not None:
                feat[i, 0] += row["speed"]
                cnt[i, 0]  += 1
        cnt[cnt == 0] = 1
        feat = feat / cnt
        tl = TIME_LABEL_MAP.get(hour, 3)
        X_list.append(feat)
        time_list.append(tl)

    X = np.stack(X_list, axis=0)  # (24, N, 1)
    return X, time_list


# ══════════════════════════════════════════════
# MODULE 3: Load Zone Labels từ OSM
# ══════════════════════════════════════════════
def load_zone_labels(zone_path: str, nodes: list) -> np.ndarray:
    """
    Returns Z: np.array shape (N, K) — multi-hot zone matrix
    """
    df = pd.read_csv(zone_path, index_col="node")
    Z = df.loc[nodes, ZONE_TYPES].values.astype(float)
    return Z   # (N, K=8)


# ══════════════════════════════════════════════
# MODULE 4: Tạo Sliding Window Samples
# ══════════════════════════════════════════════
def create_samples(X: np.ndarray, times: list,
                   t_in: int, t_out: int) -> tuple:
    """
    X:     (T, N, F)
    times: list of T time_label indices

    Returns:
      X_win: (S, N, t_in * F)  ← flatten time dim vào feature
      Y_win: (S, N, t_out)     ← predict t_out steps ahead (feature 0 = congestion)
      T_win: (S,)              ← time label của timestep cuối
    """
    T = X.shape[0]
    X_w, Y_w, T_w = [], [], []

    for t in range(T - t_in - t_out + 1):
        x_window = X[t : t + t_in]             # (t_in, N, F)
        x_flat   = x_window.transpose(1, 0, 2) # (N, t_in, F)
        x_flat   = x_flat.reshape(x_flat.shape[0], -1)  # (N, t_in*F)

        y_window = X[t + t_in : t + t_in + t_out, :, 0]  # (t_out, N) — congestion
        y_flat   = y_window.T   # (N, t_out)

        X_w.append(x_flat)
        Y_w.append(y_flat)
        T_w.append(times[t + t_in - 1])

    return (np.stack(X_w), np.stack(Y_w), np.array(T_w))


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load OSRM ---
    print("📂 Loading OSRM data...")
    df_osrm = pd.read_csv(OSRM_PATH)
    nodes = sorted(df_osrm["origin"].unique().tolist())
    node2idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    print(f"  {N} nodes, {len(df_osrm):,} rows")

    # --- Adjacency (OSRM) ---
    print("🔗 Building adjacency matrix...")
    A = build_adjacency(df_osrm, node2idx)
    print(f"  A shape: {A.shape}, non-zero: {np.count_nonzero(A)}")

    # --- Node Features ---
    if os.path.exists(TOMTOM_PATH):
        print("⚡ TomTom data found — using real traffic features (F=4)")
        df_tt = pd.read_csv(TOMTOM_PATH)
        X, time_labels = build_node_features_tomtom(df_tt, node2idx, nodes)
        feature_names = ["congestion_ratio","traffic_delay_s","travel_time_s","ff_ratio"]
    else:
        print("⚠️  TomTom data NOT found — using OSRM speed proxy (F=1)")
        print("   → Collect TomTom data first for better results")
        X, time_labels = build_node_features_osrm_proxy(df_osrm, node2idx, nodes)
        feature_names = ["speed_kmh"]

    print(f"  X shape: {X.shape} (T, N, F)")

    # --- Zone Labels (OSM) ---
    if os.path.exists(ZONE_PATH):
        print("🏙️  Loading zone labels...")
        Z = load_zone_labels(ZONE_PATH, nodes)
        print(f"  Z shape: {Z.shape} (N, K={len(ZONE_TYPES)})")
        multi_zone_nodes = [nodes[i] for i in range(N) if Z[i].sum() > 1]
        print(f"  Multi-zone nodes ({len(multi_zone_nodes)}): {multi_zone_nodes}")
    else:
        print("⚠️  Zone labels NOT found — run collect_zones.py first")
        Z = np.zeros((N, len(ZONE_TYPES)))

    # --- Sliding Windows ---
    print(f"🪟  Creating windows (T_in={T_IN}, T_out={T_OUT})...")
    X_win, Y_win, T_win = create_samples(X, time_labels, T_IN, T_OUT)
    print(f"  Samples: {X_win.shape[0]}")

    # --- Convert to Tensors ---
    A_t = torch.tensor(A, dtype=torch.float32)
    Z_t = torch.tensor(Z, dtype=torch.float32)
    X_t = torch.tensor(X_win, dtype=torch.float32)
    Y_t = torch.tensor(Y_win, dtype=torch.float32)
    T_t = torch.tensor(T_win, dtype=torch.long)

    # --- Save ---
    dataset = {
        "A": A_t,             # (N, N)       — adjacency
        "Z": Z_t,             # (N, K)       — zone labels
        "X": X_t,             # (S, N, t_in*F)
        "Y": Y_t,             # (S, N, t_out)
        "time_labels": T_t,   # (S,)
        "nodes": nodes,
        "feature_names": feature_names,
        "zone_types": ZONE_TYPES,
    }
    torch.save(dataset, os.path.join(OUT_DIR, "graph_dataset.pt"))

    meta = {
        "N": N, "K": len(ZONE_TYPES), "F": len(feature_names),
        "T_in": T_IN, "T_out": T_OUT,
        "S": X_win.shape[0],
        "nodes": nodes,
        "feature_names": feature_names,
        "zone_types": ZONE_TYPES,
        "has_tomtom": os.path.exists(TOMTOM_PATH),
        "has_zones":  os.path.exists(ZONE_PATH),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved to {OUT_DIR}/graph_dataset.pt")
    print(f"   Meta: {meta}")


if __name__ == "__main__":
    main()
