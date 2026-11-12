"""Build a canonical, quality-annotated HCM-Real dataset from TomTom raw CSV.

The script never mutates the raw collector output.  It materialises the full
time x approved-node grid so an API timeout becomes an explicit missing row,
not a silently absent observation.

Example:
    python scripts/build_hcm_real_canonical.py --dataset-version v1-pilot
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.hcm_real_schema import (  # noqa: E402
    CANONICAL_COLUMNS, DATASET_ID, SCHEMA_VERSION, validate_canonical_record,
)

LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEFAULT_RAW = ROOT / "data/raw/tomtom_hcm_segments/tomtom_hcm_segments_observed.csv"
DEFAULT_ERRORS = ROOT / "data/raw/tomtom_hcm_segments/collection_errors.jsonl"
DEFAULT_CONFIG = ROOT / "data/config/hcm_tomtom_segments.v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/hcm_real_tomtom_v1"

RAW_REQUIRED_COLUMNS = {
    "timestamp_local", "timestamp_utc", "node_id", "road_closure",
    "current_speed_kmh", "free_flow_speed_kmh", "current_travel_time_s",
    "free_flow_travel_time_s", "confidence", "congestion_ratio",
    "source_name", "source_type", "config_sha256", "raw_response_file",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def floor_interval(value: datetime, interval_min: int) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Timestamp phải có timezone")
    value = value.astimezone(timezone.utc)
    minute = (value.minute // interval_min) * interval_min
    return value.replace(minute=minute, second=0, microsecond=0)


def parse_bool(value: str) -> bool:
    if value.strip().lower() == "true":
        return True
    if value.strip().lower() == "false":
        return False
    raise ValueError(f"Boolean không hợp lệ: {value!r}")


def parse_number(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} không phải số: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} phải là số hữu hạn, không phải NaN/Infinity: {value!r}")
    return parsed


def load_registry(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    approved = [segment for segment in registry.get("segments", []) if segment.get("status") == "approved"]
    if len(approved) != 17:
        raise ValueError(f"Registry phải có đúng 17 segment approved, hiện có {len(approved)}")
    ids = [segment.get("node_id") for segment in approved]
    openlrs = [segment.get("approved_openlr") for segment in approved]
    if len(set(ids)) != 17 or any(not node for node in ids):
        raise ValueError("Registry approved chứa node_id thiếu hoặc trùng")
    if len(set(openlrs)) != 17 or any(not code for code in openlrs):
        raise ValueError("Registry approved chứa approved_openlr thiếu hoặc trùng")
    return registry, approved


def load_raw_rows(path: Path, interval_min: int, approved_ids: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy raw CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        absent = sorted(RAW_REQUIRED_COLUMNS - headers)
        if absent:
            raise ValueError(f"Raw CSV thiếu cột: {', '.join(absent)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Raw CSV rỗng")
    for row in rows:
        if row["node_id"] not in approved_ids:
            raise ValueError(f"Raw CSV có node không thuộc registry approved: {row['node_id']}")
        row["_observed_utc"] = datetime.fromisoformat(row["timestamp_utc"]).astimezone(timezone.utc)
        row["_bucket_utc"] = floor_interval(row["_observed_utc"], interval_min)
    return rows


def failure_reason(failure: dict[str, Any]) -> str:
    error = str(failure.get("error", "")).lower()
    if "timed out" in error or "timeout" in error:
        return "api_timeout"
    if failure.get("status") is not None:
        return "api_http_error"
    return "network_error"


def load_failures(path: Path, interval_min: int, approved_ids: set[str]) -> dict[tuple[datetime, str], str]:
    failures: dict[tuple[datetime, str], str] = {}
    if not path.exists():
        return failures
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            stamp = datetime.fromisoformat(entry["timestamp_utc"]).astimezone(timezone.utc)
            bucket = floor_interval(stamp, interval_min)
            for failure in entry.get("failures", []):
                node_id = failure.get("node_id")
                if node_id not in approved_ids:
                    raise ValueError(f"Error log dòng {line_no} có node lạ: {node_id!r}")
                failures[(bucket, node_id)] = failure_reason(failure)
    return failures


def flags_for_observed(raw: dict[str, Any], low_confidence_threshold: float, duplicate: bool) -> str:
    flags: list[str] = []
    confidence = parse_number(raw["confidence"], "confidence")
    if confidence < low_confidence_threshold:
        flags.append("low_confidence")
    if duplicate:
        flags.append("duplicate_observation")
    ratio = parse_number(raw["congestion_ratio"], "congestion_ratio")
    if ratio < 1:
        flags.append("ratio_below_one")
    return "|".join(flags)


def observed_record(raw: dict[str, Any], segment: dict[str, Any], bucket: datetime,
                    dataset_version: str, low_confidence_threshold: float,
                    duplicate: bool) -> dict[str, Any]:
    local_bucket = bucket.astimezone(LOCAL_TZ)
    return {
        "dataset_id": DATASET_ID,
        "dataset_version": dataset_version,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": bucket.isoformat(),
        "timestamp_local": local_bucket.isoformat(),
        "node_id": segment["node_id"],
        "segment_id": segment["node_id"],
        "approved_openlr": segment["approved_openlr"],
        "source_name": raw["source_name"],
        "source_type": raw["source_type"],
        "current_speed_kmh": parse_number(raw["current_speed_kmh"], "current_speed_kmh"),
        "free_flow_speed_kmh": parse_number(raw["free_flow_speed_kmh"], "free_flow_speed_kmh"),
        "current_travel_time_s": parse_number(raw["current_travel_time_s"], "current_travel_time_s"),
        "free_flow_travel_time_s": parse_number(raw["free_flow_travel_time_s"], "free_flow_travel_time_s"),
        "congestion_ratio": parse_number(raw["congestion_ratio"], "congestion_ratio"),
        "confidence": parse_number(raw["confidence"], "confidence"),
        "road_closure": parse_bool(raw["road_closure"]),
        "observation_status": "observed",
        "is_missing": False,
        "missing_reason": "",
        "quality_flag": flags_for_observed(raw, low_confidence_threshold, duplicate),
        "registry_config_sha256": raw["config_sha256"],
        "raw_response_file": raw["raw_response_file"],
    }


def road_closed_record(raw: dict[str, Any], segment: dict[str, Any], bucket: datetime,
                       dataset_version: str, duplicate: bool) -> dict[str, Any]:
    """Represent a confirmed closure without inventing speed=0 or time=infinity."""
    flags = ["road_closure", "not_applicable_traffic_metrics"]
    if duplicate:
        flags.append("duplicate_observation")
    return {
        "dataset_id": DATASET_ID,
        "dataset_version": dataset_version,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": bucket.isoformat(),
        "timestamp_local": bucket.astimezone(LOCAL_TZ).isoformat(),
        "node_id": segment["node_id"],
        "segment_id": segment["node_id"],
        "approved_openlr": segment["approved_openlr"],
        "source_name": raw["source_name"],
        "source_type": raw["source_type"],
        "current_speed_kmh": None,
        "free_flow_speed_kmh": None,
        "current_travel_time_s": None,
        "free_flow_travel_time_s": None,
        "congestion_ratio": None,
        "confidence": None,
        "road_closure": True,
        "observation_status": "road_closed",
        "is_missing": False,
        "missing_reason": "",
        "quality_flag": "|".join(flags),
        "registry_config_sha256": raw["config_sha256"],
        "raw_response_file": raw["raw_response_file"],
    }


def missing_record(segment: dict[str, Any], bucket: datetime, dataset_version: str,
                   config_hash: str, reason: str, raw_response_file: str = "") -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_version": dataset_version,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": bucket.isoformat(),
        "timestamp_local": bucket.astimezone(LOCAL_TZ).isoformat(),
        "node_id": segment["node_id"],
        "segment_id": segment["node_id"],
        "approved_openlr": segment["approved_openlr"],
        "source_name": "tomtom",
        "source_type": "observed_traffic_flow_segment_api",
        "current_speed_kmh": None,
        "free_flow_speed_kmh": None,
        "current_travel_time_s": None,
        "free_flow_travel_time_s": None,
        "congestion_ratio": None,
        "confidence": None,
        "road_closure": None,
        "observation_status": "invalid_payload" if reason == "invalid_payload" else "missing",
        "is_missing": True,
        "missing_reason": reason,
        "quality_flag": f"{reason}|partial_snapshot",
        "registry_config_sha256": config_hash,
        # An invalid HTTP-success payload still has a raw response worth
        # auditing. API/network failures and not_collected intervals do not.
        "raw_response_file": raw_response_file,
    }


def time_buckets(start: datetime, end: datetime, interval_min: int) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(minutes=interval_min)


def build_canonical(raw_rows: list[dict[str, Any]], failures: dict[tuple[datetime, str], str],
                    segments: list[dict[str, Any]], config_hash: str, interval_min: int,
                    dataset_version: str, low_confidence_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[datetime, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_rows:
        by_key[(raw["_bucket_utc"], raw["node_id"])].append(raw)
    for candidates in by_key.values():
        candidates.sort(key=lambda row: row["_observed_utc"])

    start = min(row["_bucket_utc"] for row in raw_rows)
    end = max(row["_bucket_utc"] for row in raw_rows)
    output: list[dict[str, Any]] = []
    missing_by_reason: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    duplicate_keys = 0
    snapshot_counts: list[int] = []
    for bucket in time_buckets(start, end, interval_min):
        retrieved_in_snapshot = 0
        for segment in segments:
            key = (bucket, segment["node_id"])
            candidates = by_key.get(key, [])
            if candidates:
                duplicate = len(candidates) > 1
                if duplicate:
                    duplicate_keys += 1
                try:
                    if parse_bool(candidates[-1]["road_closure"]):
                        candidate = road_closed_record(
                            candidates[-1], segment, bucket, dataset_version, duplicate,
                        )
                    else:
                        candidate = observed_record(
                            candidates[-1], segment, bucket, dataset_version,
                            low_confidence_threshold, duplicate,
                        )
                    validate_canonical_record(candidate)
                except ValueError:
                    record = missing_record(
                        segment, bucket, dataset_version, config_hash,
                        "invalid_payload", candidates[-1]["raw_response_file"],
                    )
                    missing_by_reason["invalid_payload"] += 1
                    flag_counts["invalid_payload"] += 1
                    flag_counts["partial_snapshot"] += 1
                else:
                    record = candidate
                    retrieved_in_snapshot += 1
                    for flag in filter(None, record["quality_flag"].split("|")):
                        flag_counts[flag] += 1
            else:
                reason = failures.get(key, "not_collected")
                record = missing_record(segment, bucket, dataset_version, config_hash, reason)
                missing_by_reason[reason] += 1
                flag_counts[reason] += 1
                flag_counts["partial_snapshot"] += 1
            validate_canonical_record(record)
            output.append(record)
        snapshot_counts.append(retrieved_in_snapshot)

    observed = [record for record in output if record["observation_status"] == "observed"]
    road_closed = [record for record in output if record["observation_status"] == "road_closed"]
    retrieved = observed + road_closed
    speeds = [record["current_speed_kmh"] for record in observed]
    confidences = [record["confidence"] for record in observed]
    summary = {
        "dataset_id": DATASET_ID,
        "dataset_version": dataset_version,
        "schema_version": SCHEMA_VERSION,
        "interval_min": interval_min,
        "grid_start_utc": start.isoformat(),
        "grid_end_utc": end.isoformat(),
        "n_nodes": len(segments),
        "n_snapshots": len(snapshot_counts),
        "expected_grid_rows": len(output),
        "observed_rows": len(observed),
        "road_closed_rows": len(road_closed),
        "retrieved_rows": len(retrieved),
        "missing_rows": len(output) - len(retrieved),
        "observed_coverage": len(observed) / len(output),
        "retrieval_coverage": len(retrieved) / len(output),
        "observation_status_counts": dict(sorted(Counter(record["observation_status"] for record in output).items())),
        "complete_snapshots": sum(count == len(segments) for count in snapshot_counts),
        "partial_snapshots": sum(0 < count < len(segments) for count in snapshot_counts),
        "empty_snapshots": sum(count == 0 for count in snapshot_counts),
        "duplicate_timestamp_node_keys": duplicate_keys,
        "missing_by_reason": dict(sorted(missing_by_reason.items())),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "speed_kmh": {"min": min(speeds), "max": max(speeds), "mean": sum(speeds) / len(speeds)} if speeds else None,
        "confidence": {"min": min(confidences), "max": max(confidences), "mean": sum(confidences) / len(confidences)} if confidences else None,
        "observed_config_hashes": sorted({raw["config_sha256"] for raw in raw_rows}),
    }
    return output, summary


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical HCM-Real dataset from ignored TomTom raw data.")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--errors-jsonl", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-version", default="v1-pilot")
    parser.add_argument("--interval-min", type=int, default=None)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()
    registry, segments = load_registry(args.config)
    interval = args.interval_min or int(registry["intended_interval_min"])
    if interval <= 0 or 60 % interval != 0:
        raise ValueError("interval-min phải là ước số dương của 60")
    if not 0 <= args.low_confidence_threshold <= 1:
        raise ValueError("low-confidence-threshold phải thuộc [0, 1]")
    approved_ids = {segment["node_id"] for segment in segments}
    raw_rows = load_raw_rows(args.raw_csv, interval, approved_ids)
    failures = load_failures(args.errors_jsonl, interval, approved_ids)
    config_hash = file_sha256(args.config)
    records, summary = build_canonical(raw_rows, failures, segments, config_hash, interval, args.dataset_version, args.low_confidence_threshold)

    out_csv = args.output_dir / "hcm_real_tomtom_canonical.csv"
    out_summary = args.output_dir / "quality_summary.json"
    out_manifest = args.output_dir / "manifest.json"
    write_csv(out_csv, records)
    summary["canonical_csv_sha256"] = file_sha256(out_csv)
    summary["raw_csv_sha256"] = file_sha256(args.raw_csv)
    summary["error_log_sha256"] = file_sha256(args.errors_jsonl) if args.errors_jsonl.exists() else None
    summary["registry_config_sha256"] = config_hash
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_version": args.dataset_version,
        "schema_version": SCHEMA_VERSION,
        "canonical_csv": out_csv.name,
        "quality_summary": out_summary.name,
        "raw_csv": str(args.raw_csv),
        "errors_jsonl": str(args.errors_jsonl),
        "registry_config": str(args.config),
        "raw_csv_sha256": summary["raw_csv_sha256"],
        "error_log_sha256": summary["error_log_sha256"],
        "registry_config_sha256": config_hash,
        "canonical_csv_sha256": summary["canonical_csv_sha256"],
        "build_parameters": {"interval_min": interval, "low_confidence_threshold": args.low_confidence_threshold},
        "note": "Derived TomTom artifact is ignored by Git pending redistribution review; publish only schema, code and non-sensitive aggregate metadata.",
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} canonical rows to {out_csv}")
    print(
        f"Traffic-observed={summary['observed_rows']}; road-closed={summary['road_closed_rows']}; "
        f"missing={summary['missing_rows']}; traffic coverage={summary['observed_coverage']:.2%}; "
        f"retrieval coverage={summary['retrieval_coverage']:.2%}"
    )
    print(f"Quality summary: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
