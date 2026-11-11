"""Collect OSM zone labels for the 17 approved HCM-Real anchors.

Usage:
    python scripts/collect_hcm_real_osm_zones.py

The script queries Overpass one node at a time to respect shared-service use.
It aborts on a failed query instead of silently replacing an unknown zone with
all-zero labels.  Re-run after a transient failure; it resumes completed nodes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from build_hcm_real_static_nodes import DEFAULT_CONFIG, load_approved_segments, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/hcm_real_tomtom_v1/osm_zone_labels.csv"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ZONE_TYPES = ("commercial", "residential", "industrial", "school", "university", "hospital", "transport", "park")
OUTPUT_COLUMNS = ("node_id", "approved_openlr", "anchor_lat", "anchor_lon", *ZONE_TYPES,
                  "zone_radius_m", "osm_retrieved_at_utc", "osm_source", "registry_config_sha256")


def build_query(lat: float, lon: float, radius_m: int) -> str:
    return f"""
    [out:json][timeout:45];
    (
      nwr["landuse"~"commercial|retail"](around:{radius_m},{lat},{lon});
      nwr["landuse"="residential"](around:{radius_m},{lat},{lon});
      nwr["landuse"="industrial"](around:{radius_m},{lat},{lon});
      nwr["amenity"~"school|kindergarten"](around:{radius_m},{lat},{lon});
      nwr["amenity"~"university|college"](around:{radius_m},{lat},{lon});
      nwr["amenity"~"hospital|clinic"](around:{radius_m},{lat},{lon});
      nwr["aeroway"](around:{radius_m},{lat},{lon});
      nwr["landuse"="transportation"](around:{radius_m},{lat},{lon});
      nwr["amenity"="bus_station"](around:{radius_m},{lat},{lon});
      nwr["leisure"="park"](around:{radius_m},{lat},{lon});
      nwr["landuse"="grass"](around:{radius_m},{lat},{lon});
    );
    out tags;
    """


def labels_from_elements(elements: list[dict[str, Any]]) -> dict[str, int]:
    labels = {zone: 0 for zone in ZONE_TYPES}
    for element in elements:
        tags = element.get("tags", {})
        landuse, amenity, leisure = tags.get("landuse", ""), tags.get("amenity", ""), tags.get("leisure", "")
        labels["commercial"] |= int(landuse in {"commercial", "retail"})
        labels["residential"] |= int(landuse == "residential")
        labels["industrial"] |= int(landuse == "industrial")
        labels["school"] |= int(amenity in {"school", "kindergarten"})
        labels["university"] |= int(amenity in {"university", "college"})
        labels["hospital"] |= int(amenity in {"hospital", "clinic"})
        labels["transport"] |= int(landuse == "transportation" or "aeroway" in tags or amenity == "bus_station")
        labels["park"] |= int(leisure == "park" or landuse == "grass")
    return labels


def fetch_elements(lat: float, lon: float, radius_m: int, endpoint: str = OVERPASS_URL) -> list[dict[str, Any]]:
    query = build_query(lat, lon, radius_m)
    request = Request(f"{endpoint}?{urlencode({'data': query})}", headers={
        "User-Agent": "HCM-Real-Research/1.0 (academic multi-source benchmark)",
        "Accept": "application/json",
    })
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("elements", [])


def read_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["node_id"]: row for row in csv.DictReader(handle)}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect OSM zones using the approved HCM-Real registry.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radius-m", type=int, default=800)
    parser.add_argument("--pause-s", type=float, default=2.0)
    parser.add_argument("--endpoint", default=OVERPASS_URL)
    args = parser.parse_args()
    if args.radius_m <= 0:
        raise ValueError("radius-m phải dương")
    segments = load_approved_segments(args.config)
    config_hash = sha256(args.config)
    existing = read_existing(args.output)
    rows: list[dict] = []
    for index, segment in enumerate(segments, start=1):
        old = existing.get(segment["node_id"])
        if (old and old.get("registry_config_sha256") == config_hash
                and old.get("zone_radius_m") == str(args.radius_m)
                and old.get("osm_source") == args.endpoint):
            rows.append(old)
            print(f"[{index}/17] {segment['node_id']}: reused existing label")
            continue
        print(f"[{index}/17] {segment['node_id']}: querying OSM Overpass")
        elements = fetch_elements(float(segment["lat"]), float(segment["lon"]), args.radius_m, args.endpoint)
        row = {
            "node_id": segment["node_id"], "approved_openlr": segment["approved_openlr"],
            "anchor_lat": segment["lat"], "anchor_lon": segment["lon"],
            **labels_from_elements(elements), "zone_radius_m": args.radius_m,
            "osm_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "osm_source": args.endpoint, "registry_config_sha256": config_hash,
        }
        rows.append(row)
        print(f"         labels={{{', '.join(k for k in ZONE_TYPES if row[k]) or 'none'}}}")
        # Persist progress after every successful node.  If the public Overpass
        # service is temporarily unavailable, the next invocation reuses these
        # rows and only queries the unfinished anchors.
        write_rows(args.output, rows)
        time.sleep(args.pause_s)
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} OSM zone rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
