from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_hcm_real_osm_zones import labels_from_elements
from collect_hcm_real_osrm_od import rows_from_table


def segment(node_id: str, openlr: str) -> dict:
    return {"node_id": node_id, "approved_openlr": openlr, "lat": 10.0, "lon": 106.0}


def test_osm_labels_map_expected_tags() -> None:
    labels = labels_from_elements([
        {"tags": {"landuse": "commercial"}},
        {"tags": {"amenity": "school"}},
        {"tags": {"aeroway": "aerodrome"}},
    ])
    assert labels["commercial"] == 1
    assert labels["school"] == 1
    assert labels["transport"] == 1
    assert labels["residential"] == 0


def test_osrm_rows_are_directed_and_exclude_diagonal() -> None:
    segments = [segment("a", "openlr-a"), segment("b", "openlr-b")]
    payload = {"durations": [[0, 20.0], [25.0, 0]], "distances": [[0, 100.0], [120.0, 0]]}
    rows = rows_from_table(segments, payload, "https://example.test", "hash", "2026-01-01T00:00:00+00:00")
    assert {(row["origin_node_id"], row["destination_node_id"]) for row in rows} == {("a", "b"), ("b", "a")}
    assert rows[0]["distance_m"] == 100.0


def test_osrm_rows_reject_unroutable_pair() -> None:
    segments = [segment("a", "openlr-a"), segment("b", "openlr-b")]
    payload = {"durations": [[0, None], [25.0, 0]], "distances": [[0, None], [120.0, 0]]}
    with pytest.raises(ValueError, match="thiếu route"):
        rows_from_table(segments, payload, "https://example.test", "hash", "2026-01-01T00:00:00+00:00")
