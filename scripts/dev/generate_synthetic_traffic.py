"""
scripts/dev/generate_synthetic_traffic.py
==========================================
Sinh dữ liệu giao thông động mô phỏng cho các node của TP.HCM (HCM-Sim).
Kết hợp cấu trúc OSRM tĩnh và đặc trưng vùng của từng nút để tạo ra
các mẫu hình tắc nghẽn đặc thù (Non-IID) theo thời gian.

Thiết kế lại (tuần Non-IID benchmark — Bảo):
  - Tách logic sinh dữ liệu (generate_traffic_dataframe) khỏi việc ghi file
    (run_and_save) → có thể unit-test mà không cần ghi CSV thật ra đĩa.
  - Dùng np.random.default_rng(seed) thay vì np.random.seed() toàn cục
    → deterministic thật sự theo seed, không phụ thuộc thứ tự gọi hàm khác
      trong tiến trình.
  - Thêm CLI: --seed, --days, --out (+ --start, --interval-min để linh hoạt).
  - Ghi kèm metadata (<out>.meta.json) đánh dấu rõ data_source=synthetic,
    simulator, seed, start_time, interval, số ngày — dùng cho manifest.json
    của build_graph.py và cho các checklist "không claim dữ liệu thật".

⚠️ get_time_label() ở đây PHẢI khớp 100% với time_label_of() trong
   scripts/build_graph.py — hai hàm này định nghĩa cùng một ngữ nghĩa nhãn
   thời gian, chỉ khác kiểu trả về (string vs int). Có test riêng
   (tests/test_data_pipeline.py::test_time_label_matches_between_modules)
   kiểm tra khớp trên toàn bộ 24 giờ để tránh hai nơi lệch nhau.

Output mặc định: data/raw/hcm_sim_traffic.csv
                  data/raw/hcm_sim_traffic.meta.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Windows console mac dinh dung cp1252, khong encode duoc cac ky tu Unicode
# (vd: "✓", "✅") -> crash khi chay qua subprocess (khong co TTY that).
# Ep stdout/stderr sang UTF-8 neu co the, bo qua neu moi truong khong ho tro.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
OSRM_PATH = "data/raw/hcm_osrm_dataset.csv"
ZONE_PATH = "data/raw/zone_labels.csv"
DEFAULT_OUT = "data/raw/hcm_sim_traffic.csv"

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

SIMULATOR_NAME = "hcm_sim_traffic_generator"
SIMULATOR_VERSION = "1.1.0"  # bump: refactor testable + CLI + metadata synthetic

DEFAULT_START = "2026-05-18"  # Thứ Hai
DEFAULT_INTERVAL_MIN = 15
DEFAULT_DAYS = 7
DEFAULT_SEED = 42

# Gợi ý POI để giữ cấu trúc TomTom tương thích
NODES_POI_HINT = {
    "Ben Thanh Market": "commercial",
    "District 1": "mixed",
    "District 3": "residential",
    "District 5": "residential",
    "Binh Thanh": "residential",
    "Tan Son Nhat Airport": "transport",
    "Landmark 81": "commercial",
    "Thu Duc": "mixed",
    "Linh Trung": "industrial",
    "Suoi Tien": "commercial",
    "High Tech Park": "industrial",
    "VNU HCM": "university",
    "Hang Xanh": "arterial",
    "Saigon Bridge": "bridge",
    "Eastern Bus Station": "transport",
    "Thu Thiem Tunnel": "transport",
    "Pham Van Dong": "arterial",
}


def get_time_label(hour: int) -> str:
    """PHẢI khớp 100% với time_label_of() trong scripts/build_graph.py."""
    if hour in range(0, 6):
        return "night"
    if hour in range(7, 10):
        return "rush_morning"
    if hour in range(16, 20):
        return "rush_evening"
    return "normal"


# ══════════════════════════════════════════════
# I/O helpers (đọc dữ liệu thật từ đĩa — dùng cho CLI)
# ══════════════════════════════════════════════
def load_edges_and_zones(osrm_path: str = OSRM_PATH, zone_path: str = ZONE_PATH):
    """
    Đọc OSRM + zone labels thật từ đĩa.
    KHÔNG dùng hàm này trong unit test — test tự dựng DataFrame nhỏ, tự tạo
    (xem tests/test_data_pipeline.py) để không phụ thuộc dữ liệu thật.
    """
    assert os.path.exists(osrm_path), f"Missing OSRM data: {osrm_path}"
    df_osrm = pd.read_csv(osrm_path)
    edges_base = (
        df_osrm.groupby(["origin", "destination"])[["distance_m", "duration_s"]]
        .mean()
        .reset_index()
    )

    assert os.path.exists(zone_path), f"Missing zone labels: {zone_path}"
    df_zone = pd.read_csv(zone_path, index_col="node")
    return edges_base, df_zone


# ══════════════════════════════════════════════
# Core simulation logic — THUẦN (pure), có thể unit-test
# ══════════════════════════════════════════════
def generate_traffic_dataframe(
    seed: int,
    days: int,
    edges_base: pd.DataFrame,
    df_zone: pd.DataFrame,
    start: str = DEFAULT_START,
    interval_min: int = DEFAULT_INTERVAL_MIN,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Sinh DataFrame giao thông động, deterministic hoàn toàn theo `seed`.

    Cùng seed + cùng input (edges_base, df_zone, days, start, interval_min)
    → LUÔN trả về DataFrame giống hệt nhau (kiểm tra ở test_data_pipeline.py).
    Khác seed → khác kết quả (nhiễu ngẫu nhiên khác nhau).
    """
    rng = np.random.default_rng(seed)

    start_date = datetime.strptime(start, "%Y-%m-%d")
    steps_per_day = 24 * (60 // interval_min)
    total_steps = steps_per_day * days
    timestamps = [
        start_date + timedelta(minutes=i * interval_min) for i in range(total_steps)
    ]

    if verbose:
        print(
            f"✓ Sinh {len(timestamps)} time steps ({days} ngày, interval {interval_min} phút)"
        )

    records = []
    for step_idx, ts in enumerate(timestamps):
        hour = ts.hour
        dow = ts.weekday()  # 0=Mon ... 6=Sun
        is_weekend = int(dow >= 5)
        time_label = get_time_label(hour)

        if verbose and ((step_idx + 1) % 100 == 0 or step_idx == 0):
            print(f"  Processing step {step_idx + 1}/{total_steps} | {ts}")

        for _, edge in edges_base.iterrows():
            u = edge["origin"]
            v = edge["destination"]
            base_dist = edge["distance_m"]
            base_dur = edge["duration_s"]

            z_u = df_zone.loc[u]
            z_v = df_zone.loc[v]

            cong_ratio = 1.0

            if not is_weekend:
                # NGÀY THƯỜNG
                if 7 <= hour < 9:
                    cong_ratio += 0.25
                    if z_v["school"] or z_v["university"]:
                        cong_ratio += 0.40
                    if z_v["industrial"]:
                        cong_ratio += 0.35
                    if z_u["residential"]:
                        cong_ratio += 0.20
                    if z_v["transport"]:
                        cong_ratio += 0.20
                elif 16 <= hour < 19:
                    cong_ratio += 0.30
                    if z_u["school"] or z_u["university"]:
                        cong_ratio += 0.35
                    if z_u["industrial"]:
                        cong_ratio += 0.45
                    if z_v["residential"]:
                        cong_ratio += 0.25
                    if z_v["commercial"]:
                        cong_ratio += 0.25
                elif 9 <= hour < 16:
                    cong_ratio += 0.10
                    if z_v["commercial"] or z_v["hospital"]:
                        cong_ratio += 0.15
                elif 19 <= hour < 22:
                    if z_v["commercial"]:
                        cong_ratio += 0.25
                    if z_v["park"]:
                        cong_ratio += 0.15
            else:
                # CUỐI TUẦN
                if 10 <= hour < 14:
                    cong_ratio += 0.15
                    if z_v["commercial"]:
                        cong_ratio += 0.35
                elif 17 <= hour < 21:
                    cong_ratio += 0.20
                    if z_v["commercial"]:
                        cong_ratio += 0.45
                    if z_v["park"]:
                        cong_ratio += 0.25
                    if z_v["residential"]:
                        cong_ratio += 0.15

            if 0 <= hour < 5:
                cong_ratio = 1.0

            noise = rng.normal(0.0, 0.08)
            cong_ratio += noise
            cong_ratio = max(0.95, min(3.5, cong_ratio))

            travel_time = base_dur * cong_ratio
            delay = max(0.0, travel_time - base_dur)

            records.append(
                {
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "time_label": time_label,
                    "src_node": u,
                    "dst_node": v,
                    "src_name": u,
                    "dst_name": v,
                    "src_poi": NODES_POI_HINT.get(u, "mixed"),
                    "dst_poi": NODES_POI_HINT.get(v, "mixed"),
                    "travel_time_s": round(travel_time, 1),
                    "free_flow_time_s": round(base_dur, 1),
                    "traffic_delay_s": round(delay, 1),
                    "length_m": round(base_dist, 1),
                    "congestion_ratio": round(cong_ratio, 3),
                }
            )

    return pd.DataFrame(records)


def build_meta(seed, days, start, interval_min, num_edges, total_rows) -> dict:
    start_date = datetime.strptime(start, "%Y-%m-%d")
    return {
        "data_source": "synthetic",
        "simulator": SIMULATOR_NAME,
        "simulator_version": SIMULATOR_VERSION,
        "seed": seed,
        "start_time": start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "interval_min": interval_min,
        "num_days": days,
        "total_steps": days * (24 * (60 // interval_min)),
        "num_edges": num_edges,
        "total_rows": total_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_and_save(
    out_path: str,
    seed: int,
    days: int,
    edges_base: pd.DataFrame,
    df_zone: pd.DataFrame,
    start: str = DEFAULT_START,
    interval_min: int = DEFAULT_INTERVAL_MIN,
    verbose: bool = True,
) -> str:
    """Sinh dữ liệu rồi ghi CSV + <tên>.meta.json cạnh nhau. Trả về out_path."""
    df_out = generate_traffic_dataframe(
        seed=seed,
        days=days,
        edges_base=edges_base,
        df_zone=df_zone,
        start=start,
        interval_min=interval_min,
        verbose=verbose,
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    meta = build_meta(seed, days, start, interval_min, len(edges_base), len(df_out))
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\n✅ Đã sinh dữ liệu SYNTHETIC và lưu vào {out_path}")
        print(f"   Metadata (data_source=synthetic): {meta_path}")
        print(f"   Total rows: {len(df_out):,}")
        print(
            f"   Congestion Ratio summary:\n{df_out['congestion_ratio'].describe().to_string()}"
        )

    return out_path


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sinh dữ liệu giao thông động mô phỏng (HCM-Sim) — KHÔNG phải dữ liệu thật."
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed cho reproducibility"
    )
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Số ngày mô phỏng")
    p.add_argument(
        "--out", type=str, default=DEFAULT_OUT, help="Đường dẫn file CSV output"
    )
    p.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START,
        help="Ngày bắt đầu (YYYY-MM-DD), nên là Thứ Hai",
    )
    p.add_argument(
        "--interval-min",
        type=int,
        default=DEFAULT_INTERVAL_MIN,
        help="Khoảng cách giữa các snapshot (phút)",
    )
    p.add_argument("--osrm-path", type=str, default=OSRM_PATH)
    p.add_argument("--zone-path", type=str, default=ZONE_PATH)
    return p


def main():
    args = build_argparser().parse_args()

    print("=" * 55)
    print("  Generating HCM-Sim Synthetic Dynamic Traffic Data")
    print(f"  seed={args.seed}  days={args.days}  out={args.out}")
    print("=" * 55)

    edges_base, df_zone = load_edges_and_zones(args.osrm_path, args.zone_path)
    print(f"✓ Loaded {len(edges_base)} base road segments from OSRM")
    print(f"✓ Loaded zone labels for {len(df_zone)} TAZs")

    run_and_save(
        out_path=args.out,
        seed=args.seed,
        days=args.days,
        edges_base=edges_base,
        df_zone=df_zone,
        start=args.start,
        interval_min=args.interval_min,
        verbose=True,
    )


if __name__ == "__main__":
    main()
