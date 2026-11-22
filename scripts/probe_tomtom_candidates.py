"""Probe replacement road-segment anchors before changing the 17-node config.

This is a one-shot validation helper.  It queries TomTom for each candidate
point, compares the returned OpenLR to the current validation manifest, and
writes a reviewable JSON report.  It never modifies the final segment config.

Example:
  export TOMTOM_API_KEY='...'
  python scripts/probe_tomtom_candidates.py \
      --candidates data/config/hcm_tomtom_replacement_candidates.v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = ROOT / "scripts/collect_tomtom_segments.py"
COLLECTOR_SPEC = importlib.util.spec_from_file_location("tomtom_segment_collector", COLLECTOR_PATH)
if COLLECTOR_SPEC is None or COLLECTOR_SPEC.loader is None:
    raise ImportError(f"Cannot load collector module: {COLLECTOR_PATH}")
COLLECTOR = importlib.util.module_from_spec(COLLECTOR_SPEC)
COLLECTOR_SPEC.loader.exec_module(COLLECTOR)
request_segment = COLLECTOR.request_segment


DEFAULT_CANDIDATES = ROOT / "data/config/hcm_tomtom_replacement_candidates.v1.json"
DEFAULT_BASELINE = ROOT / "data/raw/tomtom_hcm_segments/segment_validation_pending_review.json"
DEFAULT_OUT_DIR = ROOT / "data/raw/tomtom_hcm_segments"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2)
    return 2 * radius_km * math.asin(math.sqrt(a))


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError("Candidate config phải có mảng candidates không rỗng")
    ids = set()
    for item in candidates:
        for field in ("candidate_id", "replace_node_id", "lat", "lon", "target_road_name"):
            if field not in item:
                raise ValueError(f"Candidate thiếu trường {field}: {item}")
        if item["candidate_id"] in ids:
            raise ValueError(f"candidate_id bị trùng: {item['candidate_id']}")
        ids.add(item["candidate_id"])
        if not (-90 <= float(item["lat"]) <= 90 and -180 <= float(item["lon"]) <= 180):
            raise ValueError(f"Tọa độ không hợp lệ: {item['candidate_id']}")
    return candidates


def baseline_openlr_index(manifest: dict[str, Any]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for row in manifest.get("validated_responses", []):
        openlr = row.get("openlr")
        if openlr:
            index.setdefault(openlr, []).append(row["node_id"])
    return index


def min_distance_to_geometry(candidate: dict[str, Any], flow: dict[str, Any]) -> float | None:
    points = flow.get("coordinates", {}).get("coordinate", [])
    if not points:
        return None
    return min(
        haversine_km(float(candidate["lat"]), float(candidate["lon"]),
                     float(point["latitude"]), float(point["longitude"]))
        for point in points
    )


def summarise_candidate(candidate: dict[str, Any], result: dict[str, Any],
                        openlr_index: dict[str, list[str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "replace_node_id": candidate["replace_node_id"],
        "target_road_name": candidate["target_road_name"],
        "anchor_lat": candidate["lat"],
        "anchor_lon": candidate["lon"],
        "selection_note": candidate.get("selection_note"),
        "http_ok": result["ok"],
    }
    if not result["ok"]:
        summary.update({"http_status": result["status"], "error": result["error"]})
        return summary

    flow = result["payload"].get("flowSegmentData", {})
    openlr = flow.get("openlr")
    conflict_ids = [node_id for node_id in openlr_index.get(openlr, [])
                    if node_id != candidate["replace_node_id"]]
    summary.update({
        "http_status": result["status"],
        "openlr": openlr,
        "conflicts_with_node_ids": conflict_ids,
        "unique_against_baseline": bool(openlr) and not conflict_ids,
        "min_distance_to_segment_km": min_distance_to_geometry(candidate, flow),
        "functional_road_class": flow.get("frc"),
        "confidence": flow.get("confidence"),
        "current_speed_kmh": flow.get("currentSpeed"),
        "free_flow_speed_kmh": flow.get("freeFlowSpeed"),
        "returned_segment_coordinates": flow.get("coordinates", {}).get("coordinate"),
    })
    return summary


def cross_candidate_conflicts(row: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    """Return collisions with candidates that replace *other* final nodes.

    Several alternatives for the same replace_node_id are expected to sometimes
    resolve to the same OpenLR.  They are alternatives, not simultaneous nodes.
    """
    return [
        peer["candidate_id"]
        for peer in rows
        if peer["candidate_id"] != row["candidate_id"]
        and peer.get("openlr") == row.get("openlr")
        and peer.get("replace_node_id") != row.get("replace_node_id")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe candidate TomTom anchors and detect duplicate OpenLR segments.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--api-key", default=os.getenv("TOMTOM_API_KEY"))
    args = parser.parse_args()

    # CLI users naturally pass relative paths; resolve them before using
    # Path.relative_to(ROOT) in the audit metadata below.
    args.candidates = args.candidates.resolve()
    args.baseline_manifest = args.baseline_manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    if not args.api_key:
        print("Missing TOMTOM_API_KEY. Set it in the shell or pass --api-key.", file=sys.stderr)
        return 2
    candidates = load_candidates(args.candidates)
    baseline = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    openlr_index = baseline_openlr_index(baseline)
    rows = []
    for candidate in candidates:
        probe_segment = {
            "node_id": candidate["candidate_id"],
            "node_name": candidate["target_road_name"],
            "lat": candidate["lat"],
            "lon": candidate["lon"],
        }
        rows.append(summarise_candidate(candidate, request_segment(args.api_key, probe_segment), openlr_index))

    # A batch can contain two brand-new candidates that collide with each other
    # while remaining different from every segment in the baseline manifest.
    for row in rows:
        peers = cross_candidate_conflicts(row, rows)
        row["conflicts_with_candidate_ids"] = peers
        if peers:
            row["unique_against_baseline"] = False

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"candidate_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_config": str(args.candidates.relative_to(ROOT)),
        "candidate_config_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "baseline_manifest": str(args.baseline_manifest.relative_to(ROOT)),
        "baseline_config_sha256": baseline.get("config_sha256"),
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rows:
        distance = row.get("min_distance_to_segment_km")
        distance_text = "n/a" if distance is None else f"{distance:.3f} km"
        conflicts = ",".join(row.get("conflicts_with_node_ids", [])
                             + row.get("conflicts_with_candidate_ids", [])) or "none"
        status = "PASS" if row.get("unique_against_baseline") else "REJECT"
        print(f"{status:6} {row['candidate_id']:<34} distance={distance_text:<9} conflicts={conflicts}")
    print(f"Wrote {output}")
    return 0 if all(row.get("unique_against_baseline") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
