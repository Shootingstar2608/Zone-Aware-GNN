"""
Step 1: collect_zones.py
========================
Query OpenStreetMap (Overpass API) để lấy multi-label zone
cho tất cả 17 nodes trong node set thống nhất.

Chạy: python scripts/collect_zones.py
Output: data/raw/zone_labels.csv

Fix 406: Dùng GET + proper User-Agent
"""

import requests
import pandas as pd
import time
import json
import os

# ──────────────────────────────────────────────
# NODE SET THỐNG NHẤT (17 nodes từ OSRM dataset)
# ──────────────────────────────────────────────
NODES = {
    "Ben Thanh Market":     (10.7725, 106.6980),
    "District 1":           (10.7756, 106.7009),
    "District 3":           (10.7842, 106.6800),
    "District 5":           (10.7550, 106.6664),
    "Binh Thanh":           (10.8106, 106.7091),
    "Tan Son Nhat Airport": (10.8188, 106.6519),
    "Landmark 81":          (10.7949, 106.7218),
    "Thu Duc":              (10.8496, 106.7530),
    "Linh Trung":           (10.8715, 106.7830),
    "Suoi Tien":            (10.8641, 106.8018),
    "High Tech Park":       (10.8428, 106.8099),
    "VNU HCM":              (10.8801, 106.8054),
    "Hang Xanh":            (10.8038, 106.7108),
    "Saigon Bridge":        (10.7992, 106.7230),
    "Eastern Bus Station":  (10.8266, 106.7148),
    "Thu Thiem Tunnel":     (10.7697, 106.7085),
    "Pham Van Dong":        (10.8278, 106.7212),
}

# 8 loại zone — mỗi node sẽ có vector multi-hot 8 chiều
ZONE_TYPES = [
    "commercial",   # Thương mại (chợ, trung tâm thương mại)
    "residential",  # Dân cư
    "industrial",   # Công nghiệp / khu chế xuất
    "school",       # Trường phổ thông
    "university",   # Đại học
    "hospital",     # Bệnh viện
    "transport",    # Giao thông (sân bay, bến xe, cầu)
    "park",         # Công viên / cây xanh
]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RADIUS_M = 800  # Bán kính tìm kiếm quanh mỗi node (mét)


def build_overpass_query(lat: float, lon: float, radius: int) -> str:
    return f"""
    [out:json][timeout:30];
    (
      way["landuse"="commercial"](around:{radius},{lat},{lon});
      way["landuse"="retail"](around:{radius},{lat},{lon});
      way["landuse"="residential"](around:{radius},{lat},{lon});
      way["landuse"="industrial"](around:{radius},{lat},{lon});
      way["amenity"="school"](around:{radius},{lat},{lon});
      way["amenity"="kindergarten"](around:{radius},{lat},{lon});
      way["amenity"="university"](around:{radius},{lat},{lon});
      way["amenity"="college"](around:{radius},{lat},{lon});
      way["amenity"="hospital"](around:{radius},{lat},{lon});
      way["amenity"="clinic"](around:{radius},{lat},{lon});
      way["aeroway"](around:{radius},{lat},{lon});
      way["landuse"="transportation"](around:{radius},{lat},{lon});
      way["amenity"="bus_station"](around:{radius},{lat},{lon});
      way["leisure"="park"](around:{radius},{lat},{lon});
      way["landuse"="grass"](around:{radius},{lat},{lon});
    );
    out tags;
    """


def parse_zone_labels(elements: list) -> dict:
    z = {k: 0 for k in ZONE_TYPES}
    for elem in elements:
        tags = elem.get("tags", {})
        lu = tags.get("landuse", "")
        am = tags.get("amenity", "")
        leis = tags.get("leisure", "")

        if lu in ("commercial", "retail"):          z["commercial"] = 1
        if lu == "residential":                     z["residential"] = 1
        if lu == "industrial":                      z["industrial"] = 1
        if am in ("school", "kindergarten"):        z["school"] = 1
        if am in ("university", "college"):         z["university"] = 1
        if am in ("hospital", "clinic"):            z["hospital"] = 1
        if lu == "transportation" or \
           "aeroway" in tags or \
           am == "bus_station":                     z["transport"] = 1
        if leis == "park" or lu == "grass":         z["park"] = 1
    return z


def query_node_zones(node_name: str, lat: float, lon: float) -> dict:
    query = build_overpass_query(lat, lon, RADIUS_M)
    # Dùng GET request — tránh lỗi 406 của một số Overpass mirrors
    headers = {
        "User-Agent": "ResearchBot/1.0 (traffic-zone-research)",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(
            OVERPASS_URL,
            params={"data": query},
            headers=headers,
            timeout=35,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        labels = parse_zone_labels(elements)
        print(f"  ✓ {node_name}: {labels}")
        return labels
    except Exception as e:
        print(f"  ✗ {node_name}: ERROR — {e}")
        return {k: 0 for k in ZONE_TYPES}


def main():
    out_path = "data/raw/zone_labels.csv"
    os.makedirs("data/raw", exist_ok=True)

    print("=" * 55)
    print("  Collecting OSM Zone Labels via Overpass API")
    print(f"  {len(NODES)} nodes | radius={RADIUS_M}m")
    print("=" * 55)

    records = []
    for node_name, (lat, lon) in NODES.items():
        print(f"\n[{node_name}]  lat={lat}, lon={lon}")
        labels = query_node_zones(node_name, lat, lon)
        row = {"node": node_name, "lat": lat, "lon": lon, **labels}
        records.append(row)
        time.sleep(1.5)  # Overpass rate limit

    df = pd.DataFrame(records).set_index("node")
    df.to_csv(out_path)
    print(f"\n✅ Saved to {out_path}")
    print(df[ZONE_TYPES])

    # Thống kê nhanh
    multi_zone = df[ZONE_TYPES].sum(axis=1)
    print(f"\n📊 Nodes với nhiều hơn 1 zone:")
    print(multi_zone[multi_zone > 1].sort_values(ascending=False))


if __name__ == "__main__":
    main()
