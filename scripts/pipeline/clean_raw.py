"""
scripts/pipeline/clean_raw.py
=============================
Tầng 1: Làm sạch và chuẩn hóa dữ liệu thô (Raw -> Cleaned).
- Chuẩn hóa timestamp sang UTC và Local (Asia/Ho_Chi_Minh, +07:00).
- Chuẩn hóa đơn vị đo: km/h, s, m.
- Phát hiện và đánh dấu các bản ghi trùng lặp (DUPLICATE_FLAGGED) - tuyệt đối không xóa im lặng.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from .schemas import QualityFlag, MissingReason

LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def clean_raw_traffic(
    df_raw: pd.DataFrame,
    default_dataset_version: str = "hcm_real_v1.0",
    raw_timezone: str = "auto",
) -> pd.DataFrame:
    """
    Làm sạch bảng thô từ TomTom Flow Segment API collector.
    
    Kỳ vọng các trường đầu vào thô tối thiểu:
      - timestamp (chuỗi hoặc datetime, có thể là UTC hoặc local)
      - node_id (hoặc segment_id)
      - current_speed_kmh (hoặc currentSpeed)
      - free_flow_speed_kmh (hoặc freeFlowSpeed)
      - current_travel_time_s (hoặc currentTravelTime)
      - free_flow_travel_time_s (hoặc freeFlowTravelTime)
      - confidence (nếu có, default 1.0)
      - road_closure (nếu có, default False)
      - http_status / status_code / error (nếu có để nhận diện API error)
    """
    df = df_raw.copy()

    # 1. Đổi tên cột đồng nhất nếu input dùng camelCase của TomTom API
    rename_map = {
        "currentSpeed": "current_speed_kmh",
        "freeFlowSpeed": "free_flow_speed_kmh",
        "currentTravelTime": "current_travel_time_s",
        "freeFlowTravelTime": "free_flow_travel_time_s",
        "roadClosure": "road_closure",
        "length": "segment_length_m",
        "lengthInMeters": "segment_length_m",
        "statusCode": "http_status",
        "status_code": "http_status",
        "errorMessage": "error_message",
        "error": "error_message",
    }
    df = df.rename(columns=rename_map)

    # 2. Chuẩn hóa timestamp sang UTC và Asia/Ho_Chi_Minh (+07:00)
    if "timestamp" in df.columns:
        ts_series = pd.to_datetime(df["timestamp"])
    elif "timestamp_utc" in df.columns:
        ts_series = pd.to_datetime(df["timestamp_utc"])
    else:
        raise ValueError("Raw data thiếu trường timestamp!")

    parsed_utc = []
    parsed_local = []
    tz_choice = raw_timezone.strip().upper()
    for t in ts_series:
        if t.tzinfo is None:
            if tz_choice in ("UTC", "Z", "+00:00", "+00"):
                dt_utc = t.replace(tzinfo=timezone.utc)
                dt_local = dt_utc.astimezone(LOCAL_TZ)
            else:
                # Mặc định collector chạy tại VN ghi giờ local
                dt_local = t.replace(tzinfo=LOCAL_TZ)
                dt_utc = dt_local.astimezone(timezone.utc)
        else:
            dt_utc = t.astimezone(timezone.utc)
            dt_local = t.astimezone(LOCAL_TZ)
        parsed_utc.append(dt_utc)
        parsed_local.append(dt_local)

    df["timestamp_utc"] = parsed_utc
    df["timestamp_local"] = parsed_local

    # Phát hiện cờ API Error (HTTP >= 400 hoặc có thông báo lỗi)
    if "http_status" in df.columns:
        df["http_status"] = pd.to_numeric(df["http_status"], errors="coerce")
    else:
        df["http_status"] = np.nan

    if "error_message" not in df.columns:
        df["error_message"] = None

    status_error = df["http_status"].notna() & (df["http_status"] >= 400)
    msg_error = df["error_message"].notna() & (df["error_message"].astype(str).str.strip().isin(["", "nan", "None"]) == False)
    df["is_api_error"] = status_error | msg_error

    # 3. Chuẩn hóa các trường số học
    for num_col in ["current_speed_kmh", "free_flow_speed_kmh", "current_travel_time_s", "free_flow_travel_time_s"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").astype(float)
        else:
            df[num_col] = np.nan

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0).astype(float)
    else:
        df["confidence"] = 1.0

    if "road_closure" in df.columns:
        df["road_closure"] = df["road_closure"].fillna(False).astype(bool)
    else:
        df["road_closure"] = False

    if "source" not in df.columns:
        df["source"] = "tomtom_flow_segment_api"

    if "dataset_version" not in df.columns:
        df["dataset_version"] = default_dataset_version

    # 4. Phát hiện và đánh dấu Duplicate (cùng node_id và cùng timestamp)
    # Tuyệt đối không drop dòng, đánh dấu cờ DUPLICATE_FLAGGED cho các bản ghi lặp
    df["is_duplicate"] = df.duplicated(subset=["node_id", "timestamp_utc"], keep="first")
    df["quality_flag"] = np.where(
        df["is_duplicate"],
        QualityFlag.DUPLICATE_FLAGGED.value,
        np.where(df["is_api_error"], QualityFlag.MISSING_API_ERROR.value, QualityFlag.VALID.value),
    )
    
    return df

