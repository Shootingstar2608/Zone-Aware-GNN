import requests
import pandas as pd
import random
import time
import os
from datetime import datetime

# ======================================
# DANH SÁCH TỌA ĐỘ HCM + THỦ ĐỨC
# (lat, lng)
# ======================================

locations = {

    # ===== TRUNG TÂM =====
    "Ben Thanh Market": (10.7725, 106.6980),
    "District 1": (10.7756, 106.7009),
    "District 3": (10.7842, 106.6800),
    "District 5": (10.7550, 106.6664),
    "Binh Thanh": (10.8106, 106.7091),
    "Tan Son Nhat Airport": (10.8188, 106.6519),
    "Landmark 81": (10.7949, 106.7218),

    # ===== THỦ ĐỨC =====
    "Thu Duc": (10.8496, 106.7530),
    "Linh Trung": (10.8715, 106.7830),
    "Suoi Tien": (10.8641, 106.8018),
    "High Tech Park": (10.8428, 106.8099),
    "VNU HCM": (10.8801, 106.8054),

    # ===== TRAFFIC HOTSPOTS =====
    "Hang Xanh": (10.8038, 106.7108),
    "Saigon Bridge": (10.7992, 106.7230),
    "Eastern Bus Station": (10.8266, 106.7148),
    "Thu Thiem Tunnel": (10.7697, 106.7085),
    "Pham Van Dong": (10.8278, 106.7212)
}

# ======================================
# OUTPUT FILE
# ======================================

CSV_FILE = "hcm_osrm_dataset.csv"

if not os.path.exists(CSV_FILE):

    df = pd.DataFrame(columns=[
        "timestamp",
        "origin",
        "destination",
        "distance_m",
        "duration_s"
    ])

    df.to_csv(CSV_FILE, index=False)

# ======================================
# OSRM API
# ======================================

if os.path.exists(CSV_FILE):
    existing_df = pd.read_csv(CSV_FILE)
    counter = len(existing_df)
else:
    counter = 0

print("START COLLECTING FREE DATA...")

while True:

    origin_name = random.choice(list(locations.keys()))
    destination_name = random.choice(list(locations.keys()))

    while origin_name == destination_name:
        destination_name = random.choice(list(locations.keys()))

    o_lat, o_lng = locations[origin_name]
    d_lat, d_lng = locations[destination_name]

    # OSRM dùng lng,lat
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{o_lng},{o_lat};{d_lng},{d_lat}"
        f"?overview=false"
    )

    try:

        response = requests.get(url)

        data = response.json()

        if data["code"] == "Ok":

            route = data["routes"][0]

            distance = route["distance"]
            duration = route["duration"]

            row = {
                "timestamp": datetime.now(),
                "origin": origin_name,
                "destination": destination_name,
                "distance_m": distance,
                "duration_s": duration
            }

            pd.DataFrame([row]).to_csv(
                CSV_FILE,
                mode="a",
                header=False,
                index=False
            )

            counter += 1

            print(
                f"[{counter}] "
                f"{origin_name} -> {destination_name} | "
                f"{round(distance)}m | "
                f"{round(duration)}s"
            )

        else:
            print("FAILED:", data)

    except Exception as e:
        print("ERROR:", e)

    time.sleep(1)