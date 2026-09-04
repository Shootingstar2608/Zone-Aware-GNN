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

[Bảo - tuần 2] Thêm hour_win, dow_win để SeasonalTimeEncoder
dùng thông tin thời gian chính xác thay vì 4 nhãn rời rạc.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import torch

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
OSRM_PATH = "data/raw/hcm_osrm_dataset.csv"
TOMTOM_PATH = "data/raw/tomtom_traffic.csv"
ZONE_PATH = "data/raw/zone_labels.csv"
OUT_DIR = "data/processed"

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

T_IN = 12
T_OUT = 3

def time_label_of(hour: int) -> int:
    """0 = night, 1 = rush_morning, 2 = rush_evening, 3 = normal.

    Khop dung ngu nghia cua get_time_label() trong
    scripts/dev/generate_synthetic_traffic.py -- DUNG de hai noi lech nhau.
    hour == 6 va 10-15, 20-23 deu roi vao 'normal', giong ben generator.
    """
    if 0 <= hour < 6:
        return 0
    if 7 <= hour < 10:
        return 1
    if 16 <= hour < 20:
        return 2
    return 3


# ══════════════════════════════════════════════
# MODULE 1: Build Adjacency từ OSRM
# ══════════════════════════════════════════════
def build_adjacency(df_osrm: pd.DataFrame, node2idx: dict) -> np.ndarray:
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
    row_sum = A_norm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return A_norm / row_sum


# ══════════════════════════════════════════════
# MODULE 2: Build Node Features từ TomTom
# [Bảo] Thêm trả về hour_list và dow_list
# ══════════════════════════════════════════════
def build_node_features_tomtom(
    df_tt: pd.DataFrame, node2idx: dict, nodes: list
) -> tuple:
    """
    Returns:
      X         : (T, N, F)
      time_list : list[int] — 4 nhãn rời rạc (giữ lại cho backward compat)
      hour_list : list[int] — giờ thực tế 0-23  ← MỚI (Bảo)
      dow_list  : list[int] — ngày trong tuần 0-6 ← MỚI (Bảo)
    """
    df_tt = df_tt.copy()
    df_tt["timestamp"] = pd.to_datetime(df_tt["timestamp"])
    df_tt["hour"] = df_tt["timestamp"].dt.hour
    df_tt["dow"] = df_tt["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun

    df_tt["src_idx"] = df_tt["src_name"].map(node2idx)
    df_tt["dst_idx"] = df_tt["dst_name"].map(node2idx)
    df_tt = df_tt.dropna(subset=["src_idx", "dst_idx"])

    snapshots = sorted(df_tt["timestamp"].unique())
    N = len(nodes)
    F = 4  # [congestion_ratio, traffic_delay_s, travel_time_s, ff_ratio]

    X_list, time_list, hour_list, dow_list = [], [], [], []

    for ts in snapshots:
        snap = df_tt[df_tt["timestamp"] == ts]
        feat = np.zeros((N, F))
        cnt = np.zeros((N, 1))

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

        ts_obj = pd.Timestamp(ts)
        hour = ts_obj.hour
        dow = ts_obj.dayofweek

        # time_list.append(TIME_LABEL_MAP.get(hour, 3))
        time_list.append(time_label_of(hour))
        hour_list.append(hour)  # ← MỚI
        dow_list.append(dow)  # ← MỚI
        X_list.append(feat)

    X = np.stack(X_list, axis=0)  # (T, N, F)
    return X, time_list, hour_list, dow_list


def build_node_features_osrm_proxy(
    df_osrm: pd.DataFrame, node2idx: dict, nodes: list
) -> tuple:
    """FALLBACK — trả thêm hour_list, dow_list để giữ interface đồng nhất"""
    df = df_osrm.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["speed"] = (df["distance_m"] / df["duration_s"]) * 3.6

    N = len(nodes)
    X_list, time_list, hour_list, dow_list = [], [], [], []

    for hour in range(24):
        sub = df[df["hour"] == hour]
        feat = np.zeros((N, 1))
        cnt = np.zeros((N, 1))
        for _, row in sub.iterrows():
            i = node2idx.get(row["origin"])
            if i is not None:
                feat[i, 0] += row["speed"]
                cnt[i, 0] += 1
        cnt[cnt == 0] = 1
        feat = feat / cnt
        # time_list.append(TIME_LABEL_MAP.get(hour, 3))
        time_list.append(time_label_of(hour))
        hour_list.append(hour)
        dow_list.append(2)  # fallback: Wednesday (mid-week)
        X_list.append(feat)

    X = np.stack(X_list, axis=0)
    return X, time_list, hour_list, dow_list


# ══════════════════════════════════════════════
# MODULE 2b (MỚI - Bảo): Sinusoidal Encoding
# Thay 4 nhãn rời rạc bằng sin/cos liên tục
# ══════════════════════════════════════════════
def sinusoidal_time_encode(hour: int, dow: int, d: int = 4) -> np.ndarray:
    """
    Encode (hour, day_of_week) → sinusoidal vector (2*d,)

    hour : 0-23  → chu kỳ 24
    dow  : 0-6   → chu kỳ 7

    d=2 → [sin_h, cos_h, sin_d, cos_d]  (4 chiều)
    d=4 → 8 chiều (nhiều tần số hơn)
    """
    enc = []
    for k in range(1, d // 2 + 1):
        enc.append(math.sin(2 * math.pi * hour / 24 * k))
        enc.append(math.cos(2 * math.pi * hour / 24 * k))
    for k in range(1, d // 2 + 1):
        enc.append(math.sin(2 * math.pi * dow / 7 * k))
        enc.append(math.cos(2 * math.pi * dow / 7 * k))
    return np.array(enc, dtype=np.float32)  # (2*d,)


# ══════════════════════════════════════════════
# MODULE 3: Load Zone Labels
# ══════════════════════════════════════════════
def load_zone_labels(zone_path: str, nodes: list) -> np.ndarray:
    df = pd.read_csv(zone_path, index_col="node")
    Z = df.loc[nodes, ZONE_TYPES].values.astype(float)
    return Z


# ══════════════════════════════════════════════
# MODULE 4: Sliding Window
# [Bảo] Thêm hour_win, dow_win, sinc_win
# ══════════════════════════════════════════════
def create_samples(
    X: np.ndarray, times: list, hour_list: list, dow_list: list, t_in: int, t_out: int
) -> tuple:
    """
    Returns:
      X_win   : (S, N, t_in*F)
      Y_win   : (S, N, t_out)
      T_win   : (S,)         — 4 nhãn rời rạc (backward compat)
      H_win   : (S,)         — giờ thực tế 0-23        ← MỚI
      D_win   : (S,)         — ngày trong tuần 0-6      ← MỚI
      Sinc_win: (S, 8)       — sinusoidal encoding      ← MỚI
    """
    T = X.shape[0]
    X_w, Y_w, T_w, H_w, D_w, Sinc_w = [], [], [], [], [], []

    for t in range(T - t_in - t_out + 1):
        x_window = X[t : t + t_in]
        x_flat = x_window.transpose(1, 0, 2).reshape(x_window.shape[1], -1)

        y_window = X[t + t_in : t + t_in + t_out, :, 0]
        y_flat = y_window.T

        idx = t + t_in - 1
        hour = hour_list[idx]
        dow = dow_list[idx]
        sinc = sinusoidal_time_encode(hour, dow, d=4)  # (8,)

        X_w.append(x_flat)
        Y_w.append(y_flat)
        T_w.append(times[idx])
        H_w.append(hour)
        D_w.append(dow)
        Sinc_w.append(sinc)

    return (
        np.stack(X_w),
        np.stack(Y_w),
        np.array(T_w),
        np.array(H_w),
        np.array(D_w),
        np.stack(Sinc_w),
    )


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
import argparse

def main():
    # ===================================================================================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_out", type=int, default=3, help="Prediction horizon")
    args = parser.parse_args()
    global T_OUT
    T_OUT = args.t_out
    # ===================================================================================================================================

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- OSRM ---
    print(f"📂 Loading OSRM data... (T_out={T_OUT})")
    df_osrm = pd.read_csv(OSRM_PATH)
    nodes = sorted(df_osrm["origin"].unique().tolist())
    node2idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    print(f"  {N} nodes, {len(df_osrm):,} rows")

    print("🔗 Building adjacency matrix...")
    A = build_adjacency(df_osrm, node2idx)
    print(f"  A shape: {A.shape}, non-zero: {np.count_nonzero(A)}")

    # --- Node Features ---
    if os.path.exists(TOMTOM_PATH):
        print("⚡ TomTom data found — using real traffic features (F=4)")
        df_tt = pd.read_csv(TOMTOM_PATH)
        X, time_labels, hour_list, dow_list = build_node_features_tomtom(
            df_tt, node2idx, nodes
        )
        feature_names = [
            "congestion_ratio",
            "traffic_delay_s",
            "travel_time_s",
            "ff_ratio",
        ]
    else:
        print("⚠️  TomTom data NOT found — using OSRM speed proxy (F=1)")
        X, time_labels, hour_list, dow_list = build_node_features_osrm_proxy(
            df_osrm, node2idx, nodes
        )
        feature_names = ["speed_kmh"]

    print(f"  X shape: {X.shape} (T, N, F)")

    # --- Zone Labels ---
    if os.path.exists(ZONE_PATH):
        print("🏙️  Loading zone labels...")
        Z = load_zone_labels(ZONE_PATH, nodes)
        print(f"  Z shape: {Z.shape}")
        multi_zone_nodes = [nodes[i] for i in range(N) if Z[i].sum() > 1]
        print(f"  Multi-zone nodes ({len(multi_zone_nodes)}): {multi_zone_nodes}")
    else:
        print("⚠️  Zone labels NOT found")
        Z = np.zeros((N, len(ZONE_TYPES)))

    # --- Sliding Windows ---
    print(f"🪟  Creating windows (T_in={T_IN}, T_out={T_OUT})...")
    X_win, Y_win, T_win, H_win, D_win, Sinc_win = create_samples(
        X, time_labels, hour_list, dow_list, T_IN, T_OUT
    )
    print(f"  Samples     : {X_win.shape[0]}")
    print(f"  hour range  : {H_win.min()}-{H_win.max()}")
    print(f"  dow  range  : {D_win.min()}-{D_win.max()}")
    print(f"  sinc shape  : {Sinc_win.shape}")

    # --- Tensors ---
    A_t = torch.tensor(A, dtype=torch.float32)
    Z_t = torch.tensor(Z, dtype=torch.float32)
    X_t = torch.tensor(X_win, dtype=torch.float32)
    Y_t = torch.tensor(Y_win, dtype=torch.float32)
    T_t = torch.tensor(T_win, dtype=torch.long)
    H_t = torch.tensor(H_win, dtype=torch.long)  # ← MỚI
    D_t = torch.tensor(D_win, dtype=torch.long)  # ← MỚI
    Sinc_t = torch.tensor(Sinc_win, dtype=torch.float32)  # ← MỚI

    # --- Save ---
    dataset = {
        "A": A_t,  # (N, N)
        "Z": Z_t,  # (N, K)
        "X": X_t,  # (S, N, T_in*F)
        "Y": Y_t,  # (S, N, T_out)
        "time_labels": T_t,  # (S,)   — 4 nhãn rời rạc (backward compat)
        "hour": H_t,  # (S,)   — giờ thực 0-23
        "dow": D_t,  # (S,)   — ngày trong tuần 0-6
        "time_sinc": Sinc_t,  # (S, 8) — sinusoidal encoding
        "nodes": nodes,
        "feature_names": feature_names,
        "zone_types": ZONE_TYPES,
    }
    torch.save(dataset, os.path.join(OUT_DIR, "graph_dataset.pt"))

    meta = {
        "N": N,
        "K": len(ZONE_TYPES),
        "F": len(feature_names),
        "T_in": T_IN,
        "T_out": T_OUT,
        "S": X_win.shape[0],
        "nodes": nodes,
        "feature_names": feature_names,
        "zone_types": ZONE_TYPES,
        "has_tomtom": os.path.exists(TOMTOM_PATH),
        "has_zones": os.path.exists(ZONE_PATH),
        "time_fields": [
            "time_labels (discrete)",
            "hour (0-23)",
            "dow (0-6)",
            "time_sinc (8-dim sinusoidal)",
        ],
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved to {OUT_DIR}/graph_dataset.pt")
    print(f"   New fields: hour, dow, time_sinc")
    print(
        f"   SeasonalTimeEncoder can now use exact hour/dow instead of 4 discrete labels"
    )


if __name__ == "__main__":
    main()
