"""Tests for candidate-anchor probing; no network or API key needed."""

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_tomtom_candidates.py"
SPEC = importlib.util.spec_from_file_location("tomtom_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_candidate_config_is_loadable_and_has_unique_ids():
    candidates = probe.load_candidates(probe.DEFAULT_CANDIDATES)
    assert len(candidates) >= 6
    assert len({item["candidate_id"] for item in candidates}) == len(candidates)


def test_summary_rejects_a_segment_used_by_another_node():
    candidate = {"candidate_id": "candidate", "replace_node_id": "node_a",
                 "target_road_name": "Road", "lat": 10.0, "lon": 106.0}
    result = {"ok": True, "status": 200, "payload": {"flowSegmentData": {
        "openlr": "same", "coordinates": {"coordinate": [{"latitude": 10.0, "longitude": 106.0}]}}}}
    summary = probe.summarise_candidate(candidate, result, {"same": ["node_a", "node_b"]})
    assert summary["unique_against_baseline"] is False
    assert summary["conflicts_with_node_ids"] == ["node_b"]


def test_summary_accepts_the_replaced_node_only():
    candidate = {"candidate_id": "candidate", "replace_node_id": "node_a",
                 "target_road_name": "Road", "lat": 10.0, "lon": 106.0}
    result = {"ok": True, "status": 200, "payload": {"flowSegmentData": {
        "openlr": "replacement", "coordinates": {"coordinate": [{"latitude": 10.0, "longitude": 106.0}]}}}}
    summary = probe.summarise_candidate(candidate, result, {"replacement": ["node_a"]})
    assert summary["unique_against_baseline"] is True
    assert summary["min_distance_to_segment_km"] == 0.0


def test_alternatives_for_the_same_node_are_not_cross_candidate_conflicts():
    rows = [
        {"candidate_id": "a", "replace_node_id": "thu_duc", "openlr": "same"},
        {"candidate_id": "b", "replace_node_id": "thu_duc", "openlr": "same"},
        {"candidate_id": "c", "replace_node_id": "linh_trung", "openlr": "same"},
    ]
    assert probe.cross_candidate_conflicts(rows[0], rows) == ["c"]


def test_cli_accepts_a_relative_candidate_path_without_an_api_key():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--candidates",
         "data/config/hcm_tomtom_replacement_candidates_thu_duc_round2.v1.json"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "Missing TOMTOM_API_KEY" in result.stderr
