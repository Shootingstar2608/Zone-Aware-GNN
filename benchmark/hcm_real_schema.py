"""HCM-Real v1 canonical-record contract.

This module intentionally depends only on the Python standard library.  It is
the shared contract between the TomTom collector, the cleaning pipeline, data
quality checks and the benchmark runner; it does not read a private raw file.
"""
from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any, Mapping


DATASET_ID = "hcm_real_tomtom_segments_v1"
SCHEMA_VERSION = "1.1.0"
MISSING_REASONS = frozenset({"", "api_timeout", "api_http_error", "network_error", "invalid_payload", "low_confidence", "road_closure", "not_collected"})
OBSERVATION_STATUSES = frozenset({"observed", "road_closed", "missing", "invalid_payload"})

CANONICAL_COLUMNS = (
    "dataset_id", "dataset_version", "schema_version",
    "timestamp_utc", "timestamp_local",
    "node_id", "segment_id", "approved_openlr",
    "source_name", "source_type",
    "current_speed_kmh", "free_flow_speed_kmh",
    "current_travel_time_s", "free_flow_travel_time_s",
    "congestion_ratio", "confidence", "road_closure", "observation_status",
    "is_missing", "missing_reason", "quality_flag",
    "registry_config_sha256", "raw_response_file",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_FIELDS = (
    "current_speed_kmh", "free_flow_speed_kmh", "current_travel_time_s",
    "free_flow_travel_time_s", "congestion_ratio", "confidence",
)


def _parse_iso(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} phải là chuỗi ISO-8601")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} không phải ISO-8601 hợp lệ: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} phải có timezone offset")
    return parsed


def validate_canonical_record(record: Mapping[str, Any]) -> None:
    """Raise ValueError when one canonical HCM-Real record violates v1."""
    missing_fields = [field for field in CANONICAL_COLUMNS if field not in record]
    if missing_fields:
        raise ValueError(f"Thiếu trường canonical: {', '.join(missing_fields)}")
    if record["dataset_id"] != DATASET_ID:
        raise ValueError(f"dataset_id phải là {DATASET_ID!r}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version phải là {SCHEMA_VERSION!r}")
    if not isinstance(record["dataset_version"], str) or not record["dataset_version"]:
        raise ValueError("dataset_version không được rỗng")
    utc = _parse_iso(record["timestamp_utc"], "timestamp_utc")
    local = _parse_iso(record["timestamp_local"], "timestamp_local")
    if utc.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp_utc phải dùng offset +00:00")
    if local.utcoffset().total_seconds() != 7 * 3600:
        raise ValueError("timestamp_local phải dùng offset +07:00 (Asia/Ho_Chi_Minh)")
    if abs((utc - local.astimezone(utc.tzinfo)).total_seconds()) > 1:
        raise ValueError("timestamp_utc và timestamp_local không cùng thời điểm")
    for field in ("node_id", "segment_id", "approved_openlr", "source_name", "source_type"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{field} không được rỗng")
    if not isinstance(record["is_missing"], bool):
        raise ValueError("is_missing phải là boolean")
    status = record["observation_status"]
    if status not in OBSERVATION_STATUSES:
        raise ValueError(f"observation_status không thuộc vocabulary v1.1: {status!r}")
    if not isinstance(record["quality_flag"], str):
        raise ValueError("quality_flag phải là chuỗi; nhiều cờ phân cách bằng dấu |")
    if not isinstance(record["registry_config_sha256"], str) or not _SHA256.fullmatch(record["registry_config_sha256"]):
        raise ValueError("registry_config_sha256 phải là SHA-256 hex 64 ký tự")
    if not isinstance(record["raw_response_file"], str):
        raise ValueError("raw_response_file phải là chuỗi (có thể rỗng khi record missing)")

    reason = record["missing_reason"]
    if reason not in MISSING_REASONS:
        raise ValueError(f"missing_reason không thuộc vocabulary v1: {reason!r}")
    if status in {"missing", "invalid_payload"}:
        if not record["is_missing"]:
            raise ValueError(f"status={status} phải có is_missing=true")
        if not reason:
            raise ValueError("record missing phải có missing_reason")
        if status == "invalid_payload" and reason != "invalid_payload":
            raise ValueError("status=invalid_payload phải có missing_reason=invalid_payload")
        if any(record[field] is not None for field in _NUMERIC_FIELDS):
            raise ValueError("record missing phải để các trường traffic số là null")
        if record["road_closure"] is not None:
            raise ValueError("record missing phải để road_closure là null (trạng thái không biết)")
        return

    if record["is_missing"]:
        raise ValueError(f"status={status} phải có is_missing=false")
    if reason:
        raise ValueError("record available phải có missing_reason rỗng")
    if status == "road_closed":
        if record["road_closure"] is not True:
            raise ValueError("status=road_closed phải có road_closure=true")
        if any(record[field] is not None for field in _NUMERIC_FIELDS):
            raise ValueError("status=road_closed phải để traffic metrics là null (not applicable)")
        flags = set(filter(None, record["quality_flag"].split("|")))
        required_flags = {"road_closure", "not_applicable_traffic_metrics"}
        if not required_flags.issubset(flags):
            raise ValueError("status=road_closed phải có quality flags road_closure và not_applicable_traffic_metrics")
        return

    if record["road_closure"] is not False:
        raise ValueError("status=observed phải có road_closure=false")
    for field in _NUMERIC_FIELDS:
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{field} của record observed phải là số hữu hạn")
        if field == "confidence" and not 0 <= value <= 1:
            raise ValueError("confidence phải nằm trong [0, 1]")
        if field != "confidence" and value <= 0:
            raise ValueError(f"{field} phải dương")
    expected_ratio = record["current_travel_time_s"] / record["free_flow_travel_time_s"]
    if abs(record["congestion_ratio"] - expected_ratio) > 1e-4:
        raise ValueError("congestion_ratio không khớp current/free-flow travel time")

