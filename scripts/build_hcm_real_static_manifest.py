"""Validate and fingerprint the three static HCM-Real metadata tables.

Static metadata deliberately remains separate from the dynamic TomTom time
series.  This script verifies that all three tables describe the same approved
17-node registry before writing a small provenance manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from build_hcm_real_static_nodes import DEFAULT_CONFIG, load_approved_segments, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/processed/hcm_real_tomtom_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy artifact bắt buộc: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_same_registry(rows: list[dict[str, str]], expected_ids: set[str], config_hash: str,
                          table_name: str, node_column: str = "node_id") -> None:
    ids = [row.get(node_column, "") for row in rows]
    if set(ids) != expected_ids or len(ids) != len(expected_ids):
        raise ValueError(f"{table_name} phải có đúng một row cho mỗi trong 17 node approved")
    hashes = {row.get("registry_config_sha256", "") for row in rows}
    if hashes != {config_hash}:
        raise ValueError(f"{table_name} không khớp hash của registry approved hiện tại")


def validate_od(rows: list[dict[str, str]], expected_ids: set[str], config_hash: str) -> None:
    expected_pairs = {(source, target) for source in expected_ids for target in expected_ids if source != target}
    pairs = {(row.get("origin_node_id", ""), row.get("destination_node_id", "")) for row in rows}
    if pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise ValueError("osrm_od_reference.csv phải có đủ 17 x 16 = 272 directed OD pairs")
    hashes = {row.get("registry_config_sha256", "") for row in rows}
    if hashes != {config_hash}:
        raise ValueError("OSRM OD reference không khớp hash của registry approved hiện tại")
    for row in rows:
        if float(row["distance_m"]) <= 0 or float(row["duration_s"]) <= 0:
            raise ValueError("OSRM OD reference có distance/duration không dương")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and manifest HCM-Real static metadata.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.static_dir / "static_metadata_manifest.json"
    segments = load_approved_segments(args.config)
    expected_ids = {segment["node_id"] for segment in segments}
    config_hash = sha256(args.config)
    nodes_path = args.static_dir / "static_nodes.csv"
    zones_path = args.static_dir / "osm_zone_labels.csv"
    od_path = args.static_dir / "osrm_od_reference.csv"
    nodes, zones, od_rows = read_csv(nodes_path), read_csv(zones_path), read_csv(od_path)
    require_same_registry(nodes, expected_ids, config_hash, "static_nodes.csv")
    require_same_registry(zones, expected_ids, config_hash, "osm_zone_labels.csv")
    validate_od(od_rows, expected_ids, config_hash)

    manifest = {
        "dataset_id": "hcm_real_tomtom_segments_v1",
        "static_metadata_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_config": str(args.config.relative_to(ROOT)),
        "registry_config_sha256": config_hash,
        "node_count": len(nodes),
        "zone_label_count": len(zones),
        "directed_od_pair_count": len(od_rows),
        "artifacts": {
            path.name: {"sha256": sha256(path), "rows": len(rows)}
            for path, rows in ((nodes_path, nodes), (zones_path, zones), (od_path, od_rows))
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated 17 nodes, 17 zone-label rows, and 272 directed OD pairs.")
    print(f"Wrote static metadata manifest to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
