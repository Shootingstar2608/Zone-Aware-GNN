"""
scripts/pipeline/resampler.py
=============================
Tầng 2: Canonical Resampling & Quality Flag Engine.
- Tạo lưới thời gian tròn từng giờ (Hourly Grid: 00:00 -> 23:00) cho 17 segments.
- Chèn các khung giờ thiếu với is_missing=True, ghi rõ missing_reason, KHÔNG xóa dòng.
- Tính congestion_ratio = current_travel_time_s / free_flow_travel_time_s (chuẩn giao thông).
- Bẫy an toàn khi free_flow_travel_time_s == 0 -> gán cờ ANOMALY_FREE_FLOW_ZERO, congestion_ratio=null.
- Tự động gán time_label (night, rush_morning, rush_evening, normal) đồng bộ với build_graph.py.
- Đảm bảo output có đủ 20 cột canonical theo thứ tự chuẩn.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from .schemas import (
    QualityFlag,
    MissingReason,
    CANONICAL_COLUMNS,
    FLOAT_CANONICAL_COLUMNS,
    BOOL_CANONICAL_COLUMNS,
    STRING_CANONICAL_COLUMNS,
    get_time_label,
    validate_canonical_dataframe,
)

LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def load_registry(registry_path: str) -> dict[str, Any]:
    """Load registry 17 segments và trả về dict map theo node_id."""
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {seg["node_id"]: seg for seg in data["segments"]}
    if "planned_collection" in data:
        result["__planned_collection__"] = data["planned_collection"]
    return result


def resample_to_hourly_canonical(
    df_cleaned: pd.DataFrame,
    registry_path: str,
    dataset_version: str = "hcm_real_v1.0",
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> pd.DataFrame:
    """
    Chuyển đổi dữ liệu đã làm sạch sang Canonical HCM-Real với lưới giờ cố định.
    - Bảo vệ chống Edge Truncation: Nếu được cung cấp start_utc / end_utc hoặc planned_collection,
      lưới giờ sẽ được mở rộng từ mốc bắt đầu kế hoạch, đảm bảo không bị mất các ngày lỗi ở đầu/cuối.
    """
    registry = load_registry(registry_path)
    planned_sched = registry.get("__planned_collection__", {})
    nodes_in_registry = [k for k in registry.keys() if k != "__planned_collection__"]

    # Xác định khung thời gian chạy resampling: Ưu tiên tham số truyền vào > kế hoạch registry > dữ liệu thực tế
    if start_utc is None and "start_utc" in planned_sched:
        start_utc = pd.to_datetime(planned_sched["start_utc"], utc=True).to_pydatetime()
    if end_utc is None and "end_utc" in planned_sched:
        end_utc = pd.to_datetime(planned_sched["end_utc"], utc=True).to_pydatetime()

    if df_cleaned.empty:
        if start_utc is None or end_utc is None:
            raise ValueError("df_cleaned rỗng và không truyền start_utc, end_utc!")
    else:
        min_ts = df_cleaned["timestamp_utc"].min()
        max_ts = df_cleaned["timestamp_utc"].max()
        calc_start = min_ts.replace(minute=0, second=0, microsecond=0)
        calc_end = max_ts.replace(minute=0, second=0, microsecond=0)
        start_utc = start_utc or calc_start
        end_utc = end_utc or calc_end

    # Sinh danh sách các mốc giờ tròn UTC
    curr = start_utc
    hourly_slots = []
    while curr <= end_utc:
        hourly_slots.append(curr)
        curr += timedelta(hours=1)

    # Chuẩn bị DataFrame cleaned: làm tròn timestamp_utc về đầu giờ gần nhất
    df_work = df_cleaned.copy()
    if not df_work.empty:
        # Làm tròn timestamp về giờ gần nhất
        df_work["hourly_slot_utc"] = df_work["timestamp_utc"].apply(
            lambda t: t.replace(minute=0, second=0, microsecond=0)
        )
        if "is_api_error" not in df_work.columns:
            df_work["is_api_error"] = False

        # Loại bỏ các bản ghi duplicate để lấy 1 bản ghi chính cho mỗi (node, slot)
        non_dup = df_work[df_work["quality_flag"] != QualityFlag.DUPLICATE_FLAGGED.value].copy()
        
        # Tính khoảng cách thời gian tuyệt đối tới đầu giờ :00
        non_dup["time_diff_s"] = (non_dup["timestamp_utc"] - non_dup["hourly_slot_utc"]).dt.total_seconds().abs()

        # Thuật toán chọn mẫu thông minh:
        # 1. Ưu tiên bản ghi hợp lệ (không phải API error)
        # 2. Ưu tiên độ tin cậy cao nhất (confidence giảm dần: 1.0 -> 0.0)
        # 3. Ưu tiên mốc thời gian sát đầu giờ :00 nhất (time_diff_s tăng dần)
        non_dup_sorted = non_dup.sort_values(
            by=["node_id", "hourly_slot_utc", "is_api_error", "confidence", "time_diff_s"],
            ascending=[True, True, True, False, True],
        )
        grouped = non_dup_sorted.groupby(["node_id", "hourly_slot_utc"]).first().reset_index()
        record_map = {(row["node_id"], row["hourly_slot_utc"]): row for _, row in grouped.iterrows()}
    else:
        record_map = {}

    rows = []
    for slot_utc in hourly_slots:
        slot_local = slot_utc.astimezone(LOCAL_TZ)
        hour_local = slot_local.hour
        time_lbl = get_time_label(hour_local)

        for node_id in nodes_in_registry:
            seg_info = registry[node_id]
            seg_id = seg_info["segment_id"]
            openlr = seg_info["approved_openlr"]
            length_m = float(seg_info.get("segment_length_m", 1000.0))

            key = (node_id, slot_utc)
            if key in record_map:
                obs = record_map[key]
                if obs.get("is_api_error", False):
                    # Collector có gửi request nhưng API trả về lỗi (HTTP >= 400 hoặc timeout)
                    rows.append({
                        "timestamp_utc": slot_utc,
                        "timestamp_local": slot_local,
                        "node_id": node_id,
                        "segment_id": seg_id,
                        "approved_openlr": openlr,
                        "source": "tomtom_flow_segment_api",
                        "current_speed_kmh": np.nan,
                        "free_flow_speed_kmh": np.nan,
                        "current_travel_time_s": np.nan,
                        "free_flow_travel_time_s": np.nan,
                        "congestion_ratio": np.nan,
                        "traffic_delay_s": np.nan,
                        "segment_length_m": length_m,
                        "time_label": time_lbl,
                        "confidence": 0.0,
                        "road_closure": False,
                        "is_missing": True,
                        "missing_reason": MissingReason.API_ERROR.value,
                        "quality_flag": QualityFlag.MISSING_API_ERROR.value,
                        "dataset_version": dataset_version,
                    })
                    continue

                # Có dữ liệu quan sát tại khung giờ này
                speed_curr = obs["current_speed_kmh"]
                speed_ff = obs["free_flow_speed_kmh"]
                tt_curr = obs["current_travel_time_s"]
                tt_ff = obs["free_flow_travel_time_s"]
                conf = float(obs.get("confidence", 1.0))
                closure = bool(obs.get("road_closure", False))

                # Phân loại missing_reason & quality_flag
                if closure:
                    is_missing = False
                    missing_reason = MissingReason.ROAD_CLOSURE.value
                    quality_flag = QualityFlag.ROAD_CLOSURE.value
                    speed_curr = 0.0  # Chuẩn hóa đóng đường -> vận tốc bằng 0
                    delay_s = np.nan
                    congestion_ratio = np.nan
                elif conf < 0.5:
                    is_missing = False
                    missing_reason = MissingReason.LOW_CONFIDENCE.value
                    quality_flag = QualityFlag.VALID_LOW_CONFIDENCE.value
                else:
                    missing_reason = MissingReason.NONE.value
                    quality_flag = QualityFlag.VALID.value

                if not closure:
                    # Tính delay
                    if pd.notna(tt_curr) and pd.notna(tt_ff):
                        delay_s = max(0.0, float(tt_curr) - float(tt_ff))
                    else:
                        delay_s = np.nan

                    # Tính congestion_ratio và bẫy mẫu số == 0
                    if pd.isna(tt_ff) or tt_ff <= 0.0:
                        congestion_ratio = np.nan
                        quality_flag = QualityFlag.ANOMALY_FREE_FLOW_ZERO.value
                    elif pd.isna(tt_curr):
                        congestion_ratio = np.nan
                    else:
                        congestion_ratio = round(float(tt_curr) / float(tt_ff), 4)
                        # Kiểm tra tốc độ vượt lý thuyết (T_curr < 0.8 * T_ff)
                        if tt_curr < tt_ff * 0.8 and quality_flag == QualityFlag.VALID.value:
                            quality_flag = QualityFlag.ANOMALY_FASTER_THAN_FREE_FLOW.value

                rows.append({
                    "timestamp_utc": slot_utc,
                    "timestamp_local": slot_local,
                    "node_id": node_id,
                    "segment_id": seg_id,
                    "approved_openlr": openlr,
                    "source": "tomtom_flow_segment_api",
                    "current_speed_kmh": speed_curr,
                    "free_flow_speed_kmh": speed_ff,
                    "current_travel_time_s": tt_curr,
                    "free_flow_travel_time_s": tt_ff,
                    "congestion_ratio": congestion_ratio,
                    "traffic_delay_s": delay_s,
                    "segment_length_m": length_m,
                    "time_label": time_lbl,
                    "confidence": conf,
                    "road_closure": closure,
                    "is_missing": False,
                    "missing_reason": missing_reason,
                    "quality_flag": quality_flag,
                    "dataset_version": dataset_version,
                })
            else:
                # KHÔNG có dữ liệu: Chèn bản ghi missing, KHÔNG xóa dòng
                rows.append({
                    "timestamp_utc": slot_utc,
                    "timestamp_local": slot_local,
                    "node_id": node_id,
                    "segment_id": seg_id,
                    "approved_openlr": openlr,
                    "source": "tomtom_flow_segment_api",
                    "current_speed_kmh": np.nan,
                    "free_flow_speed_kmh": np.nan,
                    "current_travel_time_s": np.nan,
                    "free_flow_travel_time_s": np.nan,
                    "congestion_ratio": np.nan,
                    "traffic_delay_s": np.nan,
                    "segment_length_m": length_m,
                    "time_label": time_lbl,
                    "confidence": 0.0,
                    "road_closure": False,
                    "is_missing": True,
                    "missing_reason": MissingReason.NOT_COLLECTED.value,
                    "quality_flag": QualityFlag.MISSING_NOT_COLLECTED.value,
                    "dataset_version": dataset_version,
                })

    df_canonical = pd.DataFrame(rows)[CANONICAL_COLUMNS]

    # Ép kiểu nghiêm ngặt (Strict Dtype Enforcement)
    df_canonical["timestamp_utc"] = pd.to_datetime(df_canonical["timestamp_utc"], utc=True)
    df_canonical["timestamp_local"] = pd.to_datetime(df_canonical["timestamp_local"])
    for col in FLOAT_CANONICAL_COLUMNS:
        df_canonical[col] = pd.to_numeric(df_canonical[col], errors="coerce").astype(float)
    for col in BOOL_CANONICAL_COLUMNS:
        df_canonical[col] = df_canonical[col].astype(bool)
    for col in STRING_CANONICAL_COLUMNS:
        df_canonical[col] = df_canonical[col].astype(str)

    return df_canonical

