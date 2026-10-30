"""
scripts/pipeline/schemas.py
===========================
Định nghĩa Schema chuẩn hóa (Canonical Schema) và Data Contract cho dataset HCM-Real.
Tuân thủ nghiêm ngặt 20 trường chuẩn (16 trường bắt buộc + 4 trường kỹ thuật đã chốt).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd


class QualityFlag(str, Enum):
    """Danh mục cờ chất lượng dữ liệu."""
    VALID = "VALID"
    VALID_LOW_CONFIDENCE = "VALID_LOW_CONFIDENCE"
    ANOMALY_FREE_FLOW_ZERO = "ANOMALY_FREE_FLOW_ZERO"
    ANOMALY_FASTER_THAN_FREE_FLOW = "ANOMALY_FASTER_THAN_FREE_FLOW"
    ROAD_CLOSURE = "ROAD_CLOSURE"
    MISSING_API_ERROR = "MISSING_API_ERROR"
    MISSING_NOT_COLLECTED = "MISSING_NOT_COLLECTED"
    DUPLICATE_FLAGGED = "DUPLICATE_FLAGGED"


class MissingReason(str, Enum):
    """Danh mục lý do thiếu dữ liệu theo §4.2."""
    NONE = "NONE"
    API_ERROR = "API_ERROR"
    NOT_COLLECTED = "NOT_COLLECTED"
    ROAD_CLOSURE = "ROAD_CLOSURE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class TimeLabel(str, Enum):
    """4 nhãn khung giờ thống nhất với build_graph.py."""
    NIGHT = "night"
    RUSH_MORNING = "rush_morning"
    RUSH_EVENING = "rush_evening"
    NORMAL = "normal"


CANONICAL_COLUMNS = [
    "timestamp_utc",
    "timestamp_local",
    "node_id",
    "segment_id",
    "approved_openlr",
    "source",
    "current_speed_kmh",
    "free_flow_speed_kmh",
    "current_travel_time_s",
    "free_flow_travel_time_s",
    "congestion_ratio",
    "traffic_delay_s",
    "segment_length_m",
    "time_label",
    "confidence",
    "road_closure",
    "is_missing",
    "missing_reason",
    "quality_flag",
    "dataset_version",
]

MANDATORY_NON_NULL_COLUMNS = [
    "timestamp_utc",
    "timestamp_local",
    "node_id",
    "segment_id",
    "approved_openlr",
    "source",
    "segment_length_m",
    "time_label",
    "confidence",
    "road_closure",
    "is_missing",
    "missing_reason",
    "quality_flag",
    "dataset_version",
]

FLOAT_CANONICAL_COLUMNS = [
    "current_speed_kmh",
    "free_flow_speed_kmh",
    "current_travel_time_s",
    "free_flow_travel_time_s",
    "congestion_ratio",
    "traffic_delay_s",
    "segment_length_m",
    "confidence",
]

BOOL_CANONICAL_COLUMNS = [
    "road_closure",
    "is_missing",
]

DATETIME_CANONICAL_COLUMNS = [
    "timestamp_utc",
    "timestamp_local",
]

STRING_CANONICAL_COLUMNS = [
    "node_id",
    "segment_id",
    "approved_openlr",
    "source",
    "time_label",
    "missing_reason",
    "quality_flag",
    "dataset_version",
]


def get_time_label(hour: int) -> str:
    """
    Quy đổi giờ trong ngày (0-23) thành 4 nhãn khung giờ.
    Khớp 100% ngữ nghĩa của time_label_of() trong scripts/build_graph.py:
      - 0 <= hour < 6   : night
      - 7 <= hour < 10  : rush_morning
      - 16 <= hour < 20 : rush_evening
      - Còn lại (6h, 10h-15h, 20h-23h) : normal
    """
    if 0 <= hour < 6:
        return TimeLabel.NIGHT.value
    elif 7 <= hour < 10:
        return TimeLabel.RUSH_MORNING.value
    elif 16 <= hour < 20:
        return TimeLabel.RUSH_EVENING.value
    else:
        return TimeLabel.NORMAL.value


def validate_canonical_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Kiểm tra tính hợp lệ toàn diện của DataFrame canonical HCM-Real.
    Trả về dict { "is_valid": bool, "errors": list[str], "warnings": list[str] }.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Kiểm tra đủ 20 cột và đúng thứ tự
    if list(df.columns) != CANONICAL_COLUMNS:
        missing_cols = set(CANONICAL_COLUMNS) - set(df.columns)
        extra_cols = set(df.columns) - set(CANONICAL_COLUMNS)
        if missing_cols:
            errors.append(f"Thiếu các cột bắt buộc: {sorted(missing_cols)}")
        if extra_cols:
            errors.append(f"Thừa các cột ngoài schema: {sorted(extra_cols)}")
        if not missing_cols and not extra_cols:
            errors.append("Thứ tự các cột trong DataFrame không khớp chuẩn CANONICAL_COLUMNS.")

    # 2. Kiểm tra non-null trên các cột bắt buộc
    for col in MANDATORY_NON_NULL_COLUMNS:
        if col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                errors.append(f"Cột bắt buộc '{col}' có {null_count} giá trị null.")

    # 2b. Kiểm tra ràng buộc Khóa chính duy nhất (Composite Primary Key: node_id + timestamp_utc)
    if "node_id" in df.columns and "timestamp_utc" in df.columns:
        dup_pk_mask = df.duplicated(subset=["node_id", "timestamp_utc"], keep=False)
        dup_pk_count = int(dup_pk_mask.sum())
        if dup_pk_count > 0:
            errors.append(
                f"Vi phạm ràng buộc Khóa chính (Composite Primary Key Constraint): Có {dup_pk_count} dòng bị trùng lặp cặp (node_id, timestamp_utc)."
            )

    # 3. Kiểm tra kiểu dữ liệu nghiêm ngặt (Strict Dtype Validation)
    for col in FLOAT_CANONICAL_COLUMNS:
        if col in df.columns:
            if df[col].dtype == object or not (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col])):
                errors.append(
                    f"Cột số học '{col}' có kiểu dữ liệu '{df[col].dtype}' không hợp lệ (kỳ vọng float, không chấp nhận chuỗi/object)."
                )

    for col in BOOL_CANONICAL_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_bool_dtype(df[col]):
                errors.append(
                    f"Cột logic '{col}' có kiểu dữ liệu '{df[col].dtype}' không hợp lệ (kỳ vọng bool)."
                )

    for col in DATETIME_CANONICAL_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if not isinstance(first_val, (pd.Timestamp, datetime)):
                    errors.append(
                        f"Cột thời gian '{col}' có kiểu dữ liệu '{df[col].dtype}' không phải datetime. (Khuyến nghị dùng load_hcm_real_canonical() để bảo toàn kiểu dữ liệu)."
                    )

    # 4. Kiểm tra Enum values
    if "quality_flag" in df.columns:
        valid_qf = {f.value for f in QualityFlag}
        invalid_qf = set(df["quality_flag"].dropna().unique()) - valid_qf
        if invalid_qf:
            errors.append(f"Cột 'quality_flag' chứa giá trị không hợp lệ: {invalid_qf}")

    if "missing_reason" in df.columns:
        valid_mr = {r.value for r in MissingReason}
        invalid_mr = set(df["missing_reason"].dropna().unique()) - valid_mr
        if invalid_mr:
            errors.append(f"Cột 'missing_reason' chứa giá trị không hợp lệ: {invalid_mr}")

    if "time_label" in df.columns:
        valid_tl = {t.value for t in TimeLabel}
        invalid_tl = set(df["time_label"].dropna().unique()) - valid_tl
        if invalid_tl:
            errors.append(f"Cột 'time_label' chứa giá trị không hợp lệ: {invalid_tl}")

    # 4. Kiểm tra miền giá trị (Value Ranges)
    if "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce").dropna()
        if ((conf < 0.0) | (conf > 1.0)).any():
            errors.append("Cột 'confidence' vượt ngoài miền [0.0, 1.0].")

    if "current_speed_kmh" in df.columns:
        speed = pd.to_numeric(df["current_speed_kmh"], errors="coerce").dropna()
        if (speed < 0.0).any():
            errors.append("Cột 'current_speed_kmh' chứa giá trị âm.")
        if (speed > 160.0).any():
            warnings.append(f"Có {int((speed > 160.0).sum())} bản ghi có vận tốc > 160 km/h.")

    if "current_travel_time_s" in df.columns:
        tt = pd.to_numeric(df["current_travel_time_s"], errors="coerce").dropna()
        if (tt < 0.0).any():
            errors.append("Cột 'current_travel_time_s' chứa giá trị âm.")

    if "free_flow_travel_time_s" in df.columns:
        ff = pd.to_numeric(df["free_flow_travel_time_s"], errors="coerce").dropna()
        if (ff < 0.0).any():
            errors.append("Cột 'free_flow_travel_time_s' chứa giá trị âm.")

    # 5. Kiểm tra quy tắc bẫy mẫu số bằng 0 và consistency
    if "free_flow_travel_time_s" in df.columns and "congestion_ratio" in df.columns:
        ff_num = pd.to_numeric(df["free_flow_travel_time_s"], errors="coerce")
        ff_zero_mask = ff_num == 0.0
        if ff_zero_mask.any():
            invalid_ratio = df.loc[ff_zero_mask, "congestion_ratio"].notna()
            if invalid_ratio.any():
                errors.append(
                    f"Có {int(invalid_ratio.sum())} bản ghi có free_flow_travel_time_s=0 nhưng congestion_ratio không phải null."
                )
            bad_flags = df.loc[ff_zero_mask, "quality_flag"] != QualityFlag.ANOMALY_FREE_FLOW_ZERO.value
            if bad_flags.any():
                errors.append(
                    f"Có {int(bad_flags.sum())} bản ghi có free_flow_travel_time_s=0 nhưng chưa được gán cờ ANOMALY_FREE_FLOW_ZERO."
                )

    # 6. Kiểm tra tính đồng bộ của is_missing
    if "is_missing" in df.columns and "missing_reason" in df.columns:
        missing_mask = df["is_missing"] == True
        # Nếu is_missing=True thì missing_reason không được là NONE
        bad_missing_reason = df.loc[missing_mask, "missing_reason"] == MissingReason.NONE.value
        if bad_missing_reason.any():
            errors.append(
                f"Có {int(bad_missing_reason.sum())} bản ghi có is_missing=True nhưng missing_reason='NONE'."
            )

    # 7. Ràng buộc logic chéo (Cross-field Validation):
    # 7a. Ràng buộc chuẩn hóa đóng đường (Road Closure Invariant):
    # Khi road_closure == True thì is_missing bắt buộc = True, missing_reason='ROAD_CLOSURE',
    # quality_flag='ROAD_CLOSURE', current_speed_kmh=0.0, current_travel_time_s=0.0, congestion_ratio=null
    if "road_closure" in df.columns:
        closure_mask = df["road_closure"] == True
        if closure_mask.any():
            if "is_missing" in df.columns:
                bad_is_missing = ~df.loc[closure_mask, "is_missing"]
                if bad_is_missing.any():
                    errors.append(
                        f"Có {int(bad_is_missing.sum())} bản ghi có road_closure=True nhưng is_missing=False (Quy ước: road_closure=True bắt buộc is_missing=True)."
                    )
            if "missing_reason" in df.columns:
                bad_closure_reason = df.loc[closure_mask, "missing_reason"] != MissingReason.ROAD_CLOSURE.value
                if bad_closure_reason.any():
                    errors.append(
                        f"Có {int(bad_closure_reason.sum())} bản ghi có road_closure=True nhưng missing_reason khác 'ROAD_CLOSURE'."
                    )
            if "current_speed_kmh" in df.columns:
                closure_speeds = pd.to_numeric(df.loc[closure_mask, "current_speed_kmh"], errors="coerce").dropna()
                bad_closure_speed = closure_speeds > 0.0
                if bad_closure_speed.any():
                    errors.append(
                        f"Có {int(bad_closure_speed.sum())} bản ghi bị road_closure=True nhưng current_speed_kmh > 0."
                    )
            if "current_travel_time_s" in df.columns:
                bad_closure_tt = df.loc[closure_mask, "current_travel_time_s"].notna()
                if bad_closure_tt.any():
                    errors.append(
                        f"Có {int(bad_closure_tt.sum())} bản ghi bị road_closure=True nhưng current_travel_time_s không phải null/NaN (Quy ước: road_closure=True bắt buộc current_travel_time_s=NaN)."
                    )
            if "congestion_ratio" in df.columns:
                bad_closure_cr = df.loc[closure_mask, "congestion_ratio"].notna()
                if bad_closure_cr.any():
                    errors.append(
                        f"Có {int(bad_closure_cr.sum())} bản ghi bị road_closure=True nhưng congestion_ratio không phải null/NaN."
                    )

    # 7b. Ràng buộc thép cho dòng hợp lệ (Non-missing Finite Numbers Invariant):
    # Khi is_missing == False thì:
    # 1. road_closure BẮT BUỘC = False
    # 2. Toàn bộ các trường số học giao thông BẮT BUỘC PHẢI LÀ SỐ HỮU HẠN (finite, không null, không NaN, không Inf)
    if "is_missing" in df.columns:
        valid_mask = df["is_missing"] == False
        if valid_mask.any():
            if "road_closure" in df.columns:
                bad_closure_in_valid = df.loc[valid_mask, "road_closure"] == True
                if bad_closure_in_valid.any():
                    errors.append(
                        f"Có {int(bad_closure_in_valid.sum())} bản ghi có is_missing=False nhưng road_closure=True."
                    )

            traffic_finite_cols = [
                "current_speed_kmh",
                "free_flow_speed_kmh",
                "current_travel_time_s",
                "free_flow_travel_time_s",
                "congestion_ratio",
                "traffic_delay_s",
                "segment_length_m",
                "confidence",
            ]
            for col in traffic_finite_cols:
                if col in df.columns:
                    series = df.loc[valid_mask, col]
                    # Cho phép ngoại lệ: Nếu là ANOMALY_FREE_FLOW_ZERO thì congestion_ratio = null
                    if col == "congestion_ratio" and "quality_flag" in df.columns:
                        non_anomaly_mask = valid_mask & (df["quality_flag"] != QualityFlag.ANOMALY_FREE_FLOW_ZERO.value)
                        series = df.loc[non_anomaly_mask, col]

                    is_invalid = series.isna() | ~np.isfinite(pd.to_numeric(series, errors="coerce"))
                    if is_invalid.any():
                        errors.append(
                            f"Cột số học '{col}' có {int(is_invalid.sum())} giá trị null/NaN/Inf ở các dòng is_missing=False. (Yêu cầu: is_missing=False bắt buộc 100% số hữu hạn)."
                        )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_records": len(df),
    }


def load_hcm_real_canonical(path_or_dir: str, verify_manifest: bool = False) -> pd.DataFrame:
    """
    Hàm đọc chuẩn hóa dữ liệu HCM-Real Canonical (Bảo vệ downstream cho Thành viên C).
    - Tự động nhận diện thư mục hoặc đường dẫn file (.parquet hoặc .csv).
    - Ưu tiên đọc .parquet để bảo toàn 100% kiểu dữ liệu gốc.
    - Nếu đọc .csv: Tự động parse timestamp_utc (UTC), timestamp_local (Asia/Ho_Chi_Minh),
      ép kiểu float cho các cột số học và bool cho các cột cờ logic.
    - Nếu verify_manifest=True: Tự động kiểm tra đối chiếu mã SHA-256 từ manifest.json
      để đảm bảo file không bị chỉnh sửa trái phép (Dataset Integrity Verification).
    """
    import os
    import json
    import hashlib
    from zoneinfo import ZoneInfo
    local_tz = ZoneInfo("Asia/Ho_Chi_Minh")

    target_path = path_or_dir
    manifest_dir = path_or_dir if os.path.isdir(path_or_dir) else os.path.dirname(path_or_dir)

    if os.path.isdir(path_or_dir):
        parquet_candidate = os.path.join(path_or_dir, "traffic_canonical.parquet")
        csv_candidate = os.path.join(path_or_dir, "traffic_canonical.csv")
        if os.path.exists(parquet_candidate):
            target_path = parquet_candidate
        elif os.path.exists(csv_candidate):
            target_path = csv_candidate
        else:
            raise FileNotFoundError(f"Không tìm thấy traffic_canonical.parquet hoặc .csv trong {path_or_dir}")

    # Kiểm tra tính toàn vẹn thông qua manifest.json nếu được yêu cầu
    if verify_manifest:
        manifest_file = os.path.join(manifest_dir, "manifest.json")
        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
            artifact_hashes = manifest_data.get("artifact_hashes", {})
            file_basename = os.path.basename(target_path)
            if file_basename in artifact_hashes:
                expected_hash_str = artifact_hashes[file_basename]
                expected_hash = expected_hash_str.split(":")[-1]
                h = hashlib.sha256()
                with open(target_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                actual_hash = h.hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"DatasetIntegrityError: Tệp '{file_basename}' đã bị can thiệp trái phép hoặc sai lệch hash! "
                        f"(Kỳ vọng: {expected_hash[:12]}..., Thực tế: {actual_hash[:12]}...)"
                    )

    if target_path.endswith(".parquet"):
        df = pd.read_parquet(target_path)
    else:
        df = pd.read_csv(target_path)

    # Ép kiểu bảo vệ downstream
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    if "timestamp_local" in df.columns:
        ts_loc = pd.to_datetime(df["timestamp_local"])
        if hasattr(ts_loc.dt, "tz") and ts_loc.dt.tz is not None:
            df["timestamp_local"] = ts_loc.dt.tz_convert(local_tz)
        else:
            df["timestamp_local"] = ts_loc.dt.tz_localize(local_tz)

    for col in FLOAT_CANONICAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in BOOL_CANONICAL_COLUMNS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.lower().isin(["true", "1", "t"])
            else:
                df[col] = df[col].astype(bool)

    for col in STRING_CANONICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df[CANONICAL_COLUMNS] if all(c in df.columns for c in CANONICAL_COLUMNS) else df

