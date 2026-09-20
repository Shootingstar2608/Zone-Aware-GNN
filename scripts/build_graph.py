"""
Step 2: build_graph.py
======================
Kết hợp 3 nguồn dữ liệu thành PyTorch Geometric dataset:
  - OSRM   → Adjacency Matrix A
  - TomTom/HCM-Sim → Node Features X_t (dynamic, per timestep)
  - OSM    → Zone Labels Z (static, multi-hot)

Chạy (legacy, dữ liệu TomTom thật):
    python scripts/build_graph.py
    → data/processed/graph_dataset.pt, data/processed/meta.json

Chạy cho HCM-Sim (Non-IID benchmark, dữ liệu synthetic — Bảo, tuần 2):
    python scripts/build_graph.py \
        --traffic-path data/raw/hcm_sim_traffic.csv \
        --out-dir data/processed/hcm_sim_v1/ \
        --t_in 12 --t_out 3
    → data/processed/hcm_sim_v1/graph_dataset.pt, meta.json, manifest.json

⚠️ Mặc định KHÔNG đổi (traffic-path=data/raw/tomtom_traffic.csv,
   out-dir=data/processed) để không ghi đè dataset legacy. Chạy cho
   HCM-Sim thì PHẢI truyền rõ --traffic-path/--out-dir như trên.

[Bảo - tuần 2] Thêm hour_win, dow_win để SeasonalTimeEncoder
dùng thông tin thời gian chính xác thay vì 4 nhãn rời rạc.

[Bảo - tuần Non-IID benchmark] Thêm CLI --traffic-path/--out-dir/--t_in/--t_out,
hash input (sha256) + git commit hash trong manifest.json, đánh dấu
data_source (synthetic/unknown) trong meta.json — phục vụ provenance
cho HCM-Sim benchmark, không đổi hành vi mặc định của pipeline cũ.
"""

import os
import json
import math
import hashlib
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

# Windows console mac dinh dung cp1252, khong encode duoc cac ky tu Unicode
# (vd: "✓", "✅", "📂") -> crash khi chay qua subprocess (khong co TTY that).
# Ep stdout/stderr sang UTF-8 neu co the, bo qua neu moi truong khong ho tro.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ──────────────────────────────────────────────
# CONFIG (mặc định — giữ nguyên để backward-compat)
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
# MODULE 0 (MỚI — Bảo, tuần Non-IID benchmark): Provenance helpers
# ══════════════════════════════════════════════
def compute_file_hash(path: str, algo: str = "sha256") -> str:
    """Hash nội dung file (theo chunk, an toàn với file lớn)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash() -> str:
    """Lấy commit hash hiện tại. Trả về 'unknown' nếu không phải git repo
    hoặc git không có sẵn (ví dụ chạy trong CI container tối giản)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_traffic_meta(traffic_path: str) -> dict | None:
    """Đọc <traffic_path>.meta.json do generator ghi (nếu có).
    File real TomTom (legacy) không có meta này → trả về None."""
    meta_path = os.path.splitext(traffic_path)[0] + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


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
# MODULE 2: Build Node Features từ TomTom / HCM-Sim
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

        time_list.append(time_label_of(hour))
        hour_list.append(hour)
        dow_list.append(dow)
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

    Công thức số lượng sample: S = T - t_in - t_out + 1
    (test riêng ở tests/test_data_pipeline.py::test_sliding_window_sample_count_formula)
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
    parser = argparse.ArgumentParser()
    # --t_out / --t-out: giữ alias cũ (--t_out) để backward-compat với
    # script/CI hiện có, đồng thời hỗ trợ dạng dash theo spec HCM-Sim.
    parser.add_argument(
        "--t_out",
        "--t-out",
        dest="t_out",
        type=int,
        default=T_OUT,
        help="Prediction horizon",
    )
    parser.add_argument(
        "--t_in",
        "--t-in",
        dest="t_in",
        type=int,
        default=T_IN,
        help="Input window length",
    )
    parser.add_argument(
        "--traffic-path",
        dest="traffic_path",
        type=str,
        default=TOMTOM_PATH,
        help="Duong dan file traffic dong (TomTom that HOAC HCM-Sim synthetic). "
        "Mac dinh la file legacy (TomTom that) de KHONG doi hanh vi cu.",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        default=OUT_DIR,
        help="Thu muc output. Mac dinh la thu muc legacy. Voi HCM-Sim, truyen "
        "vi du data/processed/hcm_sim_v1/ de KHONG ghi de dataset cu.",
    )
    args = parser.parse_args()

    traffic_path = args.traffic_path
    out_dir = args.out_dir
    t_in = args.t_in
    t_out = args.t_out

    os.makedirs(out_dir, exist_ok=True)

    # --- OSRM ---
    print(f"📂 Loading OSRM data... (T_in={t_in}, T_out={t_out})")
    df_osrm = pd.read_csv(OSRM_PATH)
    nodes = sorted(df_osrm["origin"].unique().tolist())
    node2idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    print(f"  {N} nodes, {len(df_osrm):,} rows")

    print("🔗 Building adjacency matrix...")
    A = build_adjacency(df_osrm, node2idx)
    print(f"  A shape: {A.shape}, non-zero: {np.count_nonzero(A)}")

    # --- Node Features ---
    traffic_meta = load_traffic_meta(traffic_path)
    data_source = (traffic_meta or {}).get("data_source", "unknown")

    if os.path.exists(traffic_path):
        tag = "SYNTHETIC (HCM-Sim)" if data_source == "synthetic" else "traffic"
        print(
            f"⚡ Traffic data found ({tag}) — using real-shaped features (F=4): {traffic_path}"
        )
        df_tt = pd.read_csv(traffic_path)
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
        print(
            f"⚠️  Traffic data NOT found at {traffic_path} — using OSRM speed proxy (F=1)"
        )
        X, time_labels, hour_list, dow_list = build_node_features_osrm_proxy(
            df_osrm, node2idx, nodes
        )
        feature_names = ["speed_kmh"]
        data_source = data_source if data_source != "unknown" else "osrm_proxy"

    print(f"  X shape: {X.shape} (T, N, F)")
    print(f"  data_source: {data_source}")

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
    print(f"🪟  Creating windows (T_in={t_in}, T_out={t_out})...")
    X_win, Y_win, T_win, H_win, D_win, Sinc_win = create_samples(
        X, time_labels, hour_list, dow_list, t_in, t_out
    )
    print(f"  Samples     : {X_win.shape[0]}")
    print(f"  hour range  : {H_win.min()}-{H_win.max()}")
    print(f"  dow  range  : {D_win.min()}-{D_win.max()}")
    print(f"  sinc shape  : {Sinc_win.shape}")
    print(f"  time labels : {sorted(set(T_win.tolist()))} (kỳ vọng đủ [0,1,2,3])")

    # --- Tensors ---
    A_t = torch.tensor(A, dtype=torch.float32)
    Z_t = torch.tensor(Z, dtype=torch.float32)
    X_t = torch.tensor(X_win, dtype=torch.float32)
    Y_t = torch.tensor(Y_win, dtype=torch.float32)
    T_t = torch.tensor(T_win, dtype=torch.long)
    H_t = torch.tensor(H_win, dtype=torch.long)
    D_t = torch.tensor(D_win, dtype=torch.long)
    Sinc_t = torch.tensor(Sinc_win, dtype=torch.float32)

    # --- Save dataset ---
    dataset = {
        "A": A_t,  # (N, N)
        "Z": Z_t,  # (N, K)
        "X": X_t,  # (S, N, T_in*F)
        "Y": Y_t,  # (S, N, T_out)
        "time_labels": T_t,  # (S,)
        "hour": H_t,  # (S,)
        "dow": D_t,  # (S,)
        "time_sinc": Sinc_t,  # (S, 8)
        "nodes": nodes,
        "feature_names": feature_names,
        "zone_types": ZONE_TYPES,
    }
    dataset_path = os.path.join(out_dir, "graph_dataset.pt")
    torch.save(dataset, dataset_path)

    # --- Save meta.json (đánh dấu rõ data_source, kể cả synthetic) ---
    meta = {
        "N": N,
        "K": len(ZONE_TYPES),
        "F": len(feature_names),
        "T_in": t_in,
        "T_out": t_out,
        "S": X_win.shape[0],
        "nodes": nodes,
        "feature_names": feature_names,
        "zone_types": ZONE_TYPES,
        "has_traffic": os.path.exists(traffic_path),
        "has_zones": os.path.exists(ZONE_PATH),
        "data_source": data_source,  # "synthetic" | "unknown" | "osrm_proxy"
        "time_fields": [
            "time_labels (discrete)",
            "hour (0-23)",
            "dow (0-6)",
            "time_sinc (8-dim sinusoidal)",
        ],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Save manifest.json (provenance: hash input + git commit) ---
    manifest = {
        "dataset_id": os.path.basename(os.path.normpath(out_dir)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "data_source": data_source,
        "osrm_path": OSRM_PATH,
        "osrm_sha256": (
            compute_file_hash(OSRM_PATH) if os.path.exists(OSRM_PATH) else None
        ),
        "zone_path": ZONE_PATH,
        "zone_sha256": (
            compute_file_hash(ZONE_PATH) if os.path.exists(ZONE_PATH) else None
        ),
        "traffic_path": traffic_path,
        "traffic_sha256": (
            compute_file_hash(traffic_path) if os.path.exists(traffic_path) else None
        ),
        "traffic_meta": traffic_meta,  # None nếu traffic thật (legacy, không có .meta.json)
        "t_in": t_in,
        "t_out": t_out,
        "num_nodes": N,
        "num_samples": int(X_win.shape[0]),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Saved to {out_dir}/")
    print(f"   graph_dataset.pt, meta.json, manifest.json")
    print(f"   data_source = {data_source}")


if __name__ == "__main__":
    main()
