"""Unit tests for the HCM TomTom segment collector; no network/API key needed."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/collect_tomtom_segments.py"
SPEC = importlib.util.spec_from_file_location("tomtom_segments", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def test_checked_config_contains_exactly_17_unique_hcm_anchors():
    config = collector.load_config(collector.DEFAULT_CONFIG)
    assert len(config["segments"]) == 17
    assert len({s["node_id"] for s in config["segments"]}) == 17


def test_ratio_prefers_travel_times_then_speed_fallback():
    assert collector.ratio_from_flow({"currentTravelTime": 180, "freeFlowTravelTime": 120}) == 1.5
    assert collector.ratio_from_flow({"currentSpeed": 20, "freeFlowSpeed": 40}) == 2.0
    assert collector.ratio_from_flow({"currentSpeed": 0, "freeFlowSpeed": 40}) is None


def test_request_asks_tomtom_for_openlr(monkeypatch):
    requested = {}

    class Response:
        status = 200

        def read(self):
            return b'{"flowSegmentData": {}}'

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_urlopen(url, timeout):
        requested["url"] = url
        return Response()

    monkeypatch.setattr(collector, "urlopen", fake_urlopen)
    result = collector.request_segment("secret", {"lat": 10.0, "lon": 106.0})
    assert result["ok"] is True
    assert "openLr=true" in requested["url"]


def test_normalized_row_preserves_observed_provenance(tmp_path):
    segment = {"node_id": "test", "node_name": "Test", "lat": 10.0, "lon": 106.0}
    payload = {"flowSegmentData": {"currentSpeed": 20, "freeFlowSpeed": 40,
                                    "currentTravelTime": 180, "freeFlowTravelTime": 120,
                                    "confidence": 0.9, "roadClosure": False}}
    row = collector.normalise_response(
        segment, payload, collector.datetime(2026, 1, 1, tzinfo=collector.timezone.utc), "abc", "raw/x.json"
    )
    assert row["source_type"] == "observed_traffic_flow_segment_api"
    assert row["congestion_ratio"] == 1.5
    assert set(row) == set(collector.NORMALIZED_COLUMNS)


def test_append_csv_writes_header_only_once(tmp_path):
    path = tmp_path / "out.csv"
    row = {key: "x" for key in collector.NORMALIZED_COLUMNS}
    collector.append_csv(path, [row])
    collector.append_csv(path, [row])
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
