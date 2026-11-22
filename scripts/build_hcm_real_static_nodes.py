"""Create the static node table for the approved HCM-Real registry.

This is intentionally an offline transformation: no API key or network access
is needed.  It turns the approved TomTom segment registry into the primary-key
table to which OSM zones and TomTom dynamic observations can be joined.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "data/config/hcm_tomtom_segments.v1.json"
DEFAULT_OUTPUT = ROOT / "data/processed/hcm_real_tomtom_v1/static_nodes.csv"

NODE_COLUMNS = (
    "node_id", "node_name", "anchor_lat", "anchor_lon", "target_road_name",
    "expected_corridor", "approved_openlr", "selection_rationale",
    "registry_config_sha256",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_approved_segments(path: Path) -> list[dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    segments = [item for item in config.get("segments", []) if item.get("status") == "approved"]
    ids = [item.get("node_id") for item in segments]
    openlrs = [item.get("approved_openlr") for item in segments]
    if len(segments) != 17 or len(set(ids)) != 17 or len(set(openlrs)) != 17:
        raise ValueError("Cần đúng 17 segment approved có node_id và OpenLR riêng biệt")
    return segments


def make_rows(segments: list[dict], config_hash: str) -> list[dict]:
    return [{
        "node_id": item["node_id"],
        "node_name": item["node_name"],
        "anchor_lat": item["lat"],
        "anchor_lon": item["lon"],
        "target_road_name": item["target_road_name"],
        "expected_corridor": item["expected_corridor"],
        "approved_openlr": item["approved_openlr"],
        "selection_rationale": item["selection_rationale"],
        "registry_config_sha256": config_hash,
    } for item in segments]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NODE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static HCM-Real nodes table from approved registry.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    segments = load_approved_segments(args.config)
    write_rows(args.output, make_rows(segments, sha256(args.config)))
    print(f"Wrote {len(segments)} approved static nodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
