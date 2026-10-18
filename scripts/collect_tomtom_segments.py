"""Collect observed HCM road-segment traffic from TomTom at a sustainable rate.

This script deliberately collects 17 fixed road anchors, rather than all
17 x 16 OD pairs.  At the default hourly interval it uses approximately
12,240 Traffic Flow Segment Data requests in a 30-day month.

Usage:
  export TOMTOM_API_KEY='...'
  python scripts/collect_tomtom_segments.py --validate-only
  python scripts/collect_tomtom_segments.py --hours 24 --interval-min 60

Raw responses stay in data/raw/tomtom_hcm_segments/ (ignored by Git).  The
normalised CSV is sufficient for the later schema adapter; neither file may
contain an API key.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "data/config/hcm_tomtom_segments.v1.json"
DEFAULT_OUT_DIR = ROOT / "data/raw/tomtom_hcm_segments"
LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

NORMALIZED_COLUMNS = [
    "timestamp_local", "timestamp_utc", "node_id", "node_name",
    "anchor_lat", "anchor_lon", "road_closure", "current_speed_kmh",
    "free_flow_speed_kmh", "current_travel_time_s", "free_flow_travel_time_s",
    "confidence", "congestion_ratio", "source_name", "source_type",
    "config_sha256", "raw_response_file",
]


def config_sha256(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    segments = config.get("segments", [])
    if len(segments) != 17:
        raise ValueError(f"Config phải có đúng 17 segment, đang có {len(segments)}")
    ids = [s.get("node_id") for s in segments]
    if len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError("node_id phải tồn tại và không được trùng")
    for s in segments:
        if not (-90 <= float(s["lat"]) <= 90 and -180 <= float(s["lon"]) <= 180):
            raise ValueError(f"Toạ độ không hợp lệ: {s['node_id']}")
    return config


def request_segment(api_key: str, segment: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    query = urlencode({
        "point": f"{segment['lat']},{segment['lon']}",
        "unit": "KMPH",
        # Ask TomTom for a stable location reference when it is available.
        # It makes the one-time anchor validation auditable without relying
        # only on a point visually inspected on a map.
        "openLr": "true",
        "key": api_key,
    })
    try:
        with urlopen(f"{FLOW_URL}?{query}", timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return {"ok": True, "status": response.status, "payload": json.loads(body)}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "error": body[:500]}
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "status": None, "error": str(exc)}


def ratio_from_flow(flow: dict[str, Any]) -> float | None:
    current = flow.get("currentTravelTime")
    free = flow.get("freeFlowTravelTime")
    if isinstance(current, (int, float)) and isinstance(free, (int, float)) and free > 0:
        return round(float(current) / float(free), 6)
    current_speed = flow.get("currentSpeed")
    free_speed = flow.get("freeFlowSpeed")
    if isinstance(current_speed, (int, float)) and isinstance(free_speed, (int, float)) and current_speed > 0:
        return round(float(free_speed) / float(current_speed), 6)
    return None


def normalise_response(segment: dict[str, Any], payload: dict[str, Any], now: datetime,
                       config_hash: str, raw_response_file: str) -> dict[str, Any]:
    flow = payload.get("flowSegmentData", {})
    return {
        "timestamp_local": now.astimezone(LOCAL_TZ).isoformat(),
        "timestamp_utc": now.astimezone(timezone.utc).isoformat(),
        "node_id": segment["node_id"],
        "node_name": segment["node_name"],
        "anchor_lat": segment["lat"],
        "anchor_lon": segment["lon"],
        "road_closure": flow.get("roadClosure"),
        "current_speed_kmh": flow.get("currentSpeed"),
        "free_flow_speed_kmh": flow.get("freeFlowSpeed"),
        "current_travel_time_s": flow.get("currentTravelTime"),
        "free_flow_travel_time_s": flow.get("freeFlowTravelTime"),
        "confidence": flow.get("confidence"),
        "congestion_ratio": ratio_from_flow(flow),
        "source_name": "tomtom",
        "source_type": "observed_traffic_flow_segment_api",
        "config_sha256": config_hash,
        "raw_response_file": raw_response_file,
    }


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def collect_snapshot(api_key: str, config: dict[str, Any], config_hash: str,
                     out_dir: Path, validate_only: bool = False) -> tuple[int, int]:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows, failures, validation = [], [], []

    for segment in config["segments"]:
        result = request_segment(api_key, segment)
        raw_file = raw_dir / f"{stamp}_{segment['node_id']}.json"
        raw_file.write_text(json.dumps({"segment": segment, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
        if not result["ok"]:
            failures.append({"node_id": segment["node_id"], "status": result["status"], "error": result["error"]})
            continue
        row = normalise_response(segment, result["payload"], now, config_hash, str(raw_file.relative_to(out_dir)))
        rows.append(row)
        flow = result["payload"].get("flowSegmentData", {})
        validation.append({
            "node_id": segment["node_id"], "node_name": segment["node_name"],
            "anchor_lat": segment["lat"], "anchor_lon": segment["lon"],
            "status": "validated_response_received",
            "functional_road_class": flow.get("frc"),
            "openlr": flow.get("openlr"),
            "returned_segment_coordinates": flow.get("coordinates", {}).get("coordinate"),
            "current_speed_kmh": flow.get("currentSpeed"),
            "free_flow_speed_kmh": flow.get("freeFlowSpeed"),
            "confidence": flow.get("confidence"),
            "raw_response_file": str(raw_file.relative_to(out_dir)),
        })

    if validate_only:
        validation_path = out_dir / "segment_validation_pending_review.json"
        validation_path.write_text(json.dumps({
            "generated_at_utc": now.isoformat(), "config_sha256": config_hash,
            "instruction": "Review each raw response and replace pending_api_validation with approved only after checking that the returned segment is the intended road.",
            "validated_responses": validation, "failures": failures,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        append_csv(out_dir / "tomtom_hcm_segments_observed.csv", rows)
        if failures:
            with (out_dir / "collection_errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestamp_utc": now.isoformat(), "failures": failures}, ensure_ascii=False) + "\n")
    return len(rows), len(failures)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect hourly observed TomTom traffic for 17 HCM road anchors.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--api-key", default=os.getenv("TOMTOM_API_KEY"))
    parser.add_argument("--interval-min", type=int, default=60)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("Missing TOMTOM_API_KEY. Set it in the shell or pass --api-key.", file=sys.stderr)
        return 2
    if args.interval_min < 15:
        print("Use an interval of at least 15 minutes; hourly is recommended for the free tier.", file=sys.stderr)
        return 2
    config = load_config(args.config)
    config_hash = config_sha256(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        ok, failed = collect_snapshot(args.api_key, config, config_hash, args.out_dir, validate_only=True)
        print(f"Validation responses: {ok}/17 success, {failed} failed. Review {args.out_dir / 'segment_validation_pending_review.json'}.")
        return 0 if ok == 17 else 1

    snapshots = max(1, int(args.hours * 60 / args.interval_min))
    requests_per_month = len(config["segments"]) * 24 * 30 * 60 // args.interval_min
    print(f"Collecting {snapshots} snapshot(s), {len(config['segments'])} anchors each; estimated monthly requests at this interval: {requests_per_month:,}.")
    for i in range(snapshots):
        ok, failed = collect_snapshot(args.api_key, config, config_hash, args.out_dir)
        print(f"snapshot {i + 1}/{snapshots}: {ok}/17 written; {failed} failures")
        if i + 1 < snapshots:
            time.sleep(args.interval_min * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
