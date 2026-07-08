"""
tomtom_collector_v4.py  (17 nodes thống nhất)
=============================================
Thu thập dữ liệu giao thông thực từ TomTom API cho
17 nodes đồng bộ với OSRM dataset.

Cài: pip install requests pandas python-dotenv
Dùng: python scripts/tomtom_collector_v4.py --hours 24 --interval_min 5
      (khuyến nghị: chạy cả ngày để có rush hour data)
"""

import requests, pandas as pd, os, time, argparse
from datetime import datetime
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# 17 NODES — đồng bộ với OSRM dataset
# poi_type là HINT ban đầu, zone thực sẽ lấy từ OSM
# ──────────────────────────────────────────────
NODES = {
    "Ben Thanh Market":     {"lat": 10.7725, "lon": 106.6980, "poi_type": "commercial"},
    "District 1":           {"lat": 10.7756, "lon": 106.7009, "poi_type": "mixed"},
    "District 3":           {"lat": 10.7842, "lon": 106.6800, "poi_type": "residential"},
    "District 5":           {"lat": 10.7550, "lon": 106.6664, "poi_type": "residential"},
    "Binh Thanh":           {"lat": 10.8106, "lon": 106.7091, "poi_type": "residential"},
    "Tan Son Nhat Airport": {"lat": 10.8188, "lon": 106.6519, "poi_type": "transport"},
    "Landmark 81":          {"lat": 10.7949, "lon": 106.7218, "poi_type": "commercial"},
    "Thu Duc":              {"lat": 10.8496, "lon": 106.7530, "poi_type": "mixed"},
    "Linh Trung":           {"lat": 10.8715, "lon": 106.7830, "poi_type": "industrial"},
    "Suoi Tien":            {"lat": 10.8641, "lon": 106.8018, "poi_type": "commercial"},
    "High Tech Park":       {"lat": 10.8428, "lon": 106.8099, "poi_type": "industrial"},
    "VNU HCM":              {"lat": 10.8801, "lon": 106.8054, "poi_type": "university"},
    "Hang Xanh":            {"lat": 10.8038, "lon": 106.7108, "poi_type": "arterial"},
    "Saigon Bridge":        {"lat": 10.7992, "lon": 106.7230, "poi_type": "bridge"},
    "Eastern Bus Station":  {"lat": 10.8266, "lon": 106.7148, "poi_type": "transport"},
    "Thu Thiem Tunnel":     {"lat": 10.7697, "lon": 106.7085, "poi_type": "transport"},
    "Pham Van Dong":        {"lat": 10.8278, "lon": 106.7212, "poi_type": "arterial"},
}

NODE_NAMES = list(NODES.keys())
EDGES      = list(permutations(NODE_NAMES, 2))   # 17×16 = 272 cặp
OUT_CSV    = "data/raw/tomtom_traffic.csv"

COLS = [
    "timestamp", "time_label",
    "src_name", "dst_name",
    "src_poi",  "dst_poi",
    "travel_time_s", "free_flow_time_s",
    "traffic_delay_s", "length_m",
    "congestion_ratio",
]


def get_time_label(hour: int) -> str:
    if hour in range(0, 6):   return "night"
    if hour in range(7, 10):  return "rush_morning"
    if hour in range(16, 20): return "rush_evening"
    return "normal"


def fetch_pair(api_key: str, src_name: str, dst_name: str) -> dict | None:
    src = NODES[src_name]
    dst = NODES[dst_name]
    url = (
        f"https://api.tomtom.com/routing/1/calculateRoute/"
        f"{src['lat']},{src['lon']}:{dst['lat']},{dst['lon']}/json"
    )
    params = {
        "routeType":  "fastest",
        "traffic":    "true",
        "travelMode": "car",
        "key":        api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        s  = r.json()["routes"][0]["summary"]
        tt = s.get("travelTimeInSeconds", 0)
        ff = s.get("noTrafficTravelTimeInSeconds") or 0
        dl = s.get("trafficDelayInSeconds", 0)
        lm = s.get("lengthInMeters", 0)
        ratio = round(tt / ff, 3) if ff > 0 else round(1 + dl / tt, 3) if tt > 0 else 1.0
        return {
            "src_name": src_name, "dst_name": dst_name,
            "src_poi":  src["poi_type"], "dst_poi": dst["poi_type"],
            "travel_time_s":    tt,
            "free_flow_time_s": ff,
            "traffic_delay_s":  dl,
            "length_m":         lm,
            "congestion_ratio": ratio,
        }
    except Exception:
        return None


def collect_snapshot(api_key: str, n_workers: int = 10) -> int:
    now        = datetime.now()
    timestamp  = now.strftime("%Y-%m-%d %H:%M:%S")
    time_label = get_time_label(now.hour)
    t0         = time.time()

    records, errors = [], 0
    batch_size = 20
    batches = [EDGES[i:i+batch_size] for i in range(0, len(EDGES), batch_size)]

    for batch in batches:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(fetch_pair, api_key, s, d): (s, d) for s, d in batch}
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    r["timestamp"]  = timestamp
                    r["time_label"] = time_label
                    records.append(r)
                else:
                    errors += 1
        time.sleep(0.3)

    if records:
        os.makedirs("data/raw", exist_ok=True)
        header = not os.path.exists(OUT_CSV)
        pd.DataFrame(records)[COLS].to_csv(OUT_CSV, mode="a", header=header, index=False)

    elapsed = time.time() - t0
    print(
        f"  [{timestamp}] ({time_label}) "
        f"✓ {len(records)}/{len(EDGES)}  ✗ {errors}  ⏱ {elapsed:.1f}s"
    )
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",      default=None)
    parser.add_argument("--interval_min", type=int,   default=5)
    parser.add_argument("--hours",        type=float, default=24.0)
    parser.add_argument("--workers",      type=int,   default=10)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("TOMTOM_API_KEY")
    if not api_key:
        print("❌ Thiếu API key!\n   Dùng: --api_key KEY  hoặc set TOMTOM_API_KEY trong .env")
        return

    # Test connection
    print("🔍 Testing API connection...")
    test = fetch_pair(api_key, "Ben Thanh Market", "District 1")
    if not test:
        print("❌ API key không hoạt động hoặc hết quota!")
        return
    print(f"✅ API OK! Sample: BenThanh→D1 = {test['travel_time_s']}s "
          f"(congestion={test['congestion_ratio']}x)\n")

    total_snaps  = int((args.hours * 60) / args.interval_min)
    interval_s   = args.interval_min * 60
    est_records  = total_snaps * len(EDGES)

    print("=" * 60)
    print("  TomTom Collector v4 — 17 Nodes HCM (Unified)")
    print("=" * 60)
    print(f"  Nodes   : {len(NODES)}")
    print(f"  Edges   : {len(EDGES)} directed pairs")
    print(f"  Interval: {args.interval_min} min | Duration: {args.hours}h")
    print(f"  Snapshots: {total_snaps} | ~{est_records:,} records")
    print(f"  Output  : {OUT_CSV}")
    print("=" * 60)

    total_rec = 0
    for i in range(total_snaps):
        print(f"\n[Snapshot {i+1}/{total_snaps}]")
        total_rec += collect_snapshot(api_key, args.workers)
        if i < total_snaps - 1:
            print(f"  ⏳ Chờ {args.interval_min} phút...")
            time.sleep(interval_s)

    print(f"\n✅ Xong! Tổng: {total_rec:,} records → {OUT_CSV}")


if __name__ == "__main__":
    main()
