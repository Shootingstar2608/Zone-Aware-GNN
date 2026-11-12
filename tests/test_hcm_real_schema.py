"""Tests for the public HCM-Real canonical schema contract."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.hcm_real_schema import CANONICAL_COLUMNS, DATASET_ID, SCHEMA_VERSION, validate_canonical_record


def observed_record():
    return {
        "dataset_id": DATASET_ID, "dataset_version": "v1-pilot", "schema_version": SCHEMA_VERSION,
        "timestamp_utc": "2026-10-28T01:00:49+00:00", "timestamp_local": "2026-10-28T08:00:49+07:00",
        "node_id": "district_1", "segment_id": "district_1", "approved_openlr": "C0v-example",
        "source_name": "tomtom", "source_type": "observed_traffic_flow_segment_api",
        "current_speed_kmh": 22.0, "free_flow_speed_kmh": 32.0,
        "current_travel_time_s": 296.0, "free_flow_travel_time_s": 203.0,
        "congestion_ratio": 296.0 / 203.0, "confidence": 0.99535, "road_closure": False,
        "observation_status": "observed",
        "is_missing": False, "missing_reason": "", "quality_flag": "",
        "registry_config_sha256": "a" * 64, "raw_response_file": "raw/example.json",
    }


def test_observed_record_satisfies_contract():
    row = observed_record()
    assert tuple(row) == CANONICAL_COLUMNS
    validate_canonical_record(row)


def test_missing_record_requires_explicit_reason_and_null_traffic():
    row = observed_record()
    row.update({"is_missing": True, "observation_status": "missing", "missing_reason": "api_timeout", "raw_response_file": "", "road_closure": None})
    for field in ("current_speed_kmh", "free_flow_speed_kmh", "current_travel_time_s", "free_flow_travel_time_s", "congestion_ratio", "confidence"):
        row[field] = None
    validate_canonical_record(row)


@pytest.mark.parametrize("field,value", [
    ("confidence", 1.1),
    ("congestion_ratio", 2.0),
    ("timestamp_local", "2026-10-28T08:00:49"),
    ("registry_config_sha256", "not-a-hash"),
])
def test_invalid_observed_records_are_rejected(field, value):
    row = deepcopy(observed_record())
    row[field] = value
    with pytest.raises(ValueError):
        validate_canonical_record(row)


@pytest.mark.parametrize("field,value", [
    ("current_travel_time_s", None),
    ("current_travel_time_s", math.nan),
    ("current_travel_time_s", math.inf),
    ("free_flow_travel_time_s", None),
    ("current_speed_kmh", None),
    ("confidence", None),
])
def test_observed_record_never_allows_null_nan_or_infinity(field, value):
    row = deepcopy(observed_record())
    row[field] = value
    with pytest.raises(ValueError):
        validate_canonical_record(row)


def test_closed_road_is_available_status_with_not_applicable_metrics():
    row = observed_record()
    row.update({"observation_status": "road_closed", "road_closure": True,
                "quality_flag": "road_closure|not_applicable_traffic_metrics"})
    for field in ("current_speed_kmh", "free_flow_speed_kmh", "current_travel_time_s", "free_flow_travel_time_s", "congestion_ratio", "confidence"):
        row[field] = None
    validate_canonical_record(row)


def test_closed_road_cannot_be_mislabelled_missing_or_zero_speed():
    row = observed_record()
    row.update({"observation_status": "road_closed", "road_closure": True,
                "quality_flag": "road_closure|not_applicable_traffic_metrics", "current_speed_kmh": 0.0})
    with pytest.raises(ValueError):
        validate_canonical_record(row)
