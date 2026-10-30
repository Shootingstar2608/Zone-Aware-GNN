"""
scripts/dev/generate_mock_raw_tomtom.py
=======================================
Bộ sinh dữ liệu thô mô phỏng (Mock TomTom Flow Segment API responses).
Dùng để kiểm thử pipeline end-to-end của Thành viên B độc lập với tiến độ thu thập của Thành viên A.

Bao gồm đầy đủ các ca thực tế:
  - Giờ cao điểm / bình thường / đêm
  - Trường hợp tắc đường (congestion_ratio > 1.0)
  - Độ tin cậy thấp (confidence < 0.5)
  - Đóng đường (road_closure = True)
  - Mẫu số bằng 0 (free_flow_travel_time_s = 0)
  - Vận tốc vượt lý thuyết (current_travel_time_s < free_flow_travel_time_s * 0.8)
  - Bản ghi trùng lặp (duplicate timestamps)
  - Mất kết nối / thiếu dữ liệu (missing slots)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
import random
import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def generate_mock_raw_traffic(
    registry_path: str = "data/config/hcm_tomtom_segments.v1.json",
    days: int = 2,
    seed: int = 42,
    out_csv: str = "data/raw/mock_tomtom_raw.csv",
) -> pd.DataFrame:
    """Sinh tập dữ liệu thô mô phỏng TomTom Flow Segment API."""
    random.seed(seed)
    np.random.seed(seed)

    with open(registry_path, "r", encoding="utf-8") as f:
        registry_data = json.load(f)
    segments = registry_data["segments"]

    # Thời gian bắt đầu: ví dụ 2026-10-25 00:00:00 UTC
    start_time = datetime(2026, 10, 25, 0, 0, 0, tzinfo=timezone.utc)
    total_hours = days * 24

    records = []

    for h in range(total_hours):
        current_time = start_time + timedelta(hours=h)
        # Giờ địa phương TP.HCM (+7)
        hour_local = (current_time.hour + 7) % 24

        # Mô phỏng xác suất mất mạng toàn cục 2% (cả 17 trạm đều mất)
        if random.random() < 0.02:
            continue

        for seg in segments:
            node_id = seg["node_id"]
            seg_id = seg["segment_id"]
            openlr = seg["approved_openlr"]
            length_m = float(seg.get("segment_length_m", 1000.0))
            ff_speed = float(seg.get("default_free_flow_speed_kmh", 45.0))

            # Mô phỏng xác suất trạm này bị lỗi lẻ tẻ (3%)
            # Trong đó có 1% là API Error (HTTP 500/Timeout), 2% là collector không thu được (omitted)
            err_roll = random.random()
            if err_roll < 0.01:
                records.append({
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "node_id": node_id,
                    "segment_id": seg_id,
                    "approved_openlr": openlr,
                    "currentSpeed": np.nan,
                    "freeFlowSpeed": ff_speed,
                    "currentTravelTime": np.nan,
                    "freeFlowTravelTime": ff_time_s,
                    "confidence": 0.0,
                    "roadClosure": False,
                    "lengthInMeters": length_m,
                    "http_status": 500,
                    "error_message": "Gateway Timeout from TomTom Flow API",
                })
                continue
            elif err_roll < 0.03:
                continue

            # Free flow travel time lý thuyết
            ff_time_s = round((length_m / (ff_speed * 1000.0 / 3600.0)), 1)

            # Mô phỏng hành vi giao thông theo giờ địa phương
            is_rush = (7 <= hour_local <= 9) or (16 <= hour_local <= 19)
            is_night = (0 <= hour_local <= 5)

            # 1. Edge-case: Free-flow travel time = 0 (1% xác suất)
            if random.random() < 0.01:
                ff_time_s = 0.0

            # 2. Edge-case: Road closure (1% xác suất)
            road_closure = False
            if random.random() < 0.01:
                road_closure = True
                curr_speed = 0.0
                curr_time_s = 0.0
                confidence = 1.0
            else:
                # 3. Edge-case: Vận tốc vượt lý thuyết (xe chạy nhanh hơn lý tưởng, 3% xác suất)
                if random.random() < 0.03 and ff_time_s > 0:
                    curr_time_s = round(ff_time_s * random.uniform(0.65, 0.78), 1)
                    curr_speed = round(min(120.0, ff_speed * random.uniform(1.2, 1.4)), 1)
                elif is_rush:
                    # Giờ cao điểm: tắc đường, ratio từ 1.2 đến 2.4
                    congestion_mult = random.uniform(1.2, 2.4)
                    curr_time_s = round(ff_time_s * congestion_mult, 1)
                    curr_speed = round(max(5.0, ff_speed / congestion_mult), 1)
                elif is_night:
                    # Đêm: đường rất thoáng
                    curr_time_s = round(ff_time_s * random.uniform(0.9, 1.05), 1)
                    curr_speed = round(ff_speed * random.uniform(0.95, 1.1), 1)
                else:
                    # Bình thường: ratio từ 1.0 đến 1.25
                    congestion_mult = random.uniform(1.0, 1.25)
                    curr_time_s = round(ff_time_s * congestion_mult, 1)
                    curr_speed = round(max(10.0, ff_speed / congestion_mult), 1)

                # 4. Confidence: bình thường 0.8 - 1.0, thỉnh thoảng < 0.5 (2% xác suất)
                if random.random() < 0.02:
                    confidence = round(random.uniform(0.1, 0.45), 2)
                else:
                    confidence = round(random.uniform(0.75, 1.0), 2)

            rec = {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "node_id": node_id,
                "segment_id": seg_id,
                "approved_openlr": openlr,
                "currentSpeed": curr_speed,
                "freeFlowSpeed": ff_speed,
                "currentTravelTime": curr_time_s,
                "freeFlowTravelTime": ff_time_s,
                "confidence": confidence,
                "roadClosure": road_closure,
                "lengthInMeters": length_m,
                "http_status": 200,
            }
            records.append(rec)

            # 5. Edge-case: Duplicate record (1% xác suất sinh thêm 1 bản ghi cùng timestamp)
            if random.random() < 0.01:
                dup_rec = rec.copy()
                records.append(dup_rec)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Đã sinh {len(df)} bản ghi raw mô phỏng ({days} ngày) -> {out_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw/mock_tomtom_raw.csv")
    args = parser.parse_args()
    generate_mock_raw_traffic(days=args.days, seed=args.seed, out_csv=args.out)
