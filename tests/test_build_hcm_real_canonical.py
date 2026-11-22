"""Tests for raw TomTom -> canonical HCM-Real materialisation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_hcm_real_canonical.py"
SPEC = importlib.util.spec_from_file_location("build_hcm_real_canonical", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def raw(node_id: str, utc: str, local: str, speed: str = "20"):
    return {
        "timestamp_utc": utc, "timestamp_local": local, "node_id": node_id,
        "road_closure": "False", "current_speed_kmh": speed, "free_flow_speed_kmh": "40",
        "current_travel_time_s": "180", "free_flow_travel_time_s": "120",
        "confidence": "0.9", "congestion_ratio": "1.5", "source_name": "tomtom",
        "source_type": "observed_traffic_flow_segment_api", "config_sha256": "a" * 64,
        "raw_response_file": f"raw/{node_id}.json",
    }


def segments():
    return [
        {"node_id": "a", "approved_openlr": "openlr-a"},
        {"node_id": "b", "approved_openlr": "openlr-b"},
    ]


def test_materialises_missing_timeout_and_duplicate_observation():
    rows = [
        raw("a", "2026-10-28T01:00:49+00:00", "2026-10-28T08:00:49+07:00"),
        raw("a", "2026-10-28T02:00:10+00:00", "2026-10-28T09:00:10+07:00", "21"),
        raw("a", "2026-10-28T02:00:50+00:00", "2026-10-28T09:00:50+07:00", "22"),
        raw("b", "2026-10-28T02:00:49+00:00", "2026-10-28T09:00:49+07:00"),
    ]
    for row in rows:
        row["_observed_utc"] = builder.datetime.fromisoformat(row["timestamp_utc"])
        row["_bucket_utc"] = builder.floor_interval(row["_observed_utc"], 60)
    failures = {(builder.datetime.fromisoformat("2026-10-28T01:00:00+00:00"), "b"): "api_timeout"}

    records, summary = builder.build_canonical(rows, failures, segments(), "a" * 64, 60, "v1-test", 0.5)

    assert len(records) == 4  # 2 timestamps x 2 approved nodes
    missing = next(record for record in records if record["node_id"] == "b" and record["is_missing"])
    assert missing["missing_reason"] == "api_timeout"
    duplicate = next(record for record in records if record["node_id"] == "a" and record["timestamp_utc"].startswith("2026-10-28T02"))
    assert duplicate["current_speed_kmh"] == 22.0  # newest raw observation wins
    assert "duplicate_observation" in duplicate["quality_flag"]
    assert summary["observed_rows"] == 3
    assert summary["missing_rows"] == 1
    assert summary["missing_by_reason"] == {"api_timeout": 1}


def test_failure_reason_classification():
    assert builder.failure_reason({"status": None, "error": "<urlopen error timed out>"}) == "api_timeout"
    assert builder.failure_reason({"status": 429, "error": "Too many requests"}) == "api_http_error"
    assert builder.failure_reason({"status": None, "error": "network down"}) == "network_error"


def test_nan_http_success_payload_becomes_explicit_invalid_payload_missing_record():
    rows = [raw("a", "2026-10-28T01:00:49+00:00", "2026-10-28T08:00:49+07:00")]
    rows[0]["current_travel_time_s"] = "NaN"
    for row in rows:
        row["_observed_utc"] = builder.datetime.fromisoformat(row["timestamp_utc"])
        row["_bucket_utc"] = builder.floor_interval(row["_observed_utc"], 60)

    records, summary = builder.build_canonical(rows, {}, segments(), "a" * 64, 60, "v1-test", 0.5)

    invalid = next(record for record in records if record["node_id"] == "a")
    assert invalid["is_missing"] is True
    assert invalid["observation_status"] == "invalid_payload"
    assert invalid["missing_reason"] == "invalid_payload"
    assert invalid["current_travel_time_s"] is None
    assert invalid["raw_response_file"] == "raw/a.json"
    assert summary["missing_by_reason"]["invalid_payload"] == 1


def test_road_closure_becomes_available_closed_status_not_missing_or_zero_speed():
    rows = [raw("a", "2026-10-28T01:00:49+00:00", "2026-10-28T08:00:49+07:00")]
    rows[0].update({"road_closure": "True", "current_travel_time_s": "NaN", "current_speed_kmh": "0"})
    for row in rows:
        row["_observed_utc"] = builder.datetime.fromisoformat(row["timestamp_utc"])
        row["_bucket_utc"] = builder.floor_interval(row["_observed_utc"], 60)

    records, summary = builder.build_canonical(rows, {}, segments(), "a" * 64, 60, "v1-test", 0.5)
    closed = next(record for record in records if record["node_id"] == "a")
    assert closed["observation_status"] == "road_closed"
    assert closed["is_missing"] is False
    assert closed["road_closure"] is True
    assert closed["current_travel_time_s"] is None
    assert summary["road_closed_rows"] == 1
