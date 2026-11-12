"""Collect one complete 17 x 17 OSRM OD reference matrix for HCM-Real.

The output is a routing/free-flow reference, not observed traffic and not a
road-segment adjacency claim.  It is kept separate from TomTom dynamic data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from build_hcm_real_static_nodes import DEFAULT_CONFIG, load_approved_segments, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/hcm_real_tomtom_v1/osrm_od_reference.csv"
DEFAULT_ENDPOINT = "https://router.project-osrm.org"
OUTPUT_COLUMNS = (
    "origin_node_id", "destination_node_id", "origin_approved_openlr", "destination_approved_openlr",
    "distance_m", "duration_s", "osrm_profile", "osrm_retrieved_at_utc", "osrm_endpoint", "registry_config_sha256",
)


def table_url(segments: list[dict], endpoint: str) -> str:
    coordinates = ";".join(f"{segment['lon']},{segment['lat']}" for segment in segments)
    return f"{endpoint.rstrip('/')}/table/v1/driving/{coordinates}?{urlencode({'annotations': 'duration,distance'})}"


def fetch_table(segments: list[dict], endpoint: str = DEFAULT_ENDPOINT) -> dict[str, Any]:
    request = Request(table_url(segments, endpoint), headers={"User-Agent": "HCM-Real-Research/1.0 (academic multi-source benchmark)"})
    with urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("code") != "Ok":
        raise ValueError(f"OSRM không trả Ok: {payload}")
    return payload


def rows_from_table(segments: list[dict], payload: dict[str, Any], endpoint: str, config_hash: str,
                    retrieved_at: str) -> list[dict]:
    durations, distances = payload.get("durations"), payload.get("distances")
    n = len(segments)
    if not isinstance(durations, list) or not isinstance(distances, list) or len(durations) != n or len(distances) != n:
        raise ValueError("OSRM table không có matrix kích thước N x N")
    rows: list[dict] = []
    for i, origin in enumerate(segments):
        if len(durations[i]) != n or len(distances[i]) != n:
            raise ValueError("OSRM table có hàng matrix không đúng N")
        for j, destination in enumerate(segments):
            if i == j:
                continue
            duration, distance = durations[i][j], distances[i][j]
            if not isinstance(duration, (int, float)) or not isinstance(distance, (int, float)) or duration <= 0 or distance <= 0:
                raise ValueError(f"OSRM thiếu route hợp lệ: {origin['node_id']} -> {destination['node_id']}")
            rows.append({
                "origin_node_id": origin["node_id"], "destination_node_id": destination["node_id"],
                "origin_approved_openlr": origin["approved_openlr"], "destination_approved_openlr": destination["approved_openlr"],
                "distance_m": distance, "duration_s": duration, "osrm_profile": "driving",
                "osrm_retrieved_at_utc": retrieved_at, "osrm_endpoint": endpoint,
                "registry_config_sha256": config_hash,
            })
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect full OSRM OD reference for approved HCM-Real nodes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()
    segments = load_approved_segments(args.config)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    payload = fetch_table(segments, args.endpoint)
    rows = rows_from_table(segments, payload, args.endpoint, sha256(args.config), retrieved_at)
    if len(rows) != 17 * 16:
        raise AssertionError(f"Expected 272 directed OD rows, got {len(rows)}")
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} directed OSRM OD reference rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
