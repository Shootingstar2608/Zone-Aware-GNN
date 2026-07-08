"""
generate_synthetic_traffic.py
=============================
Sinh dữ liệu giao thông động mô phỏng cho 17 nodes của TP.HCM.
Kết hợp cấu trúc OSRM tĩnh và đặc trưng vùng của từng nút để tạo ra
các mẫu hình tắc nghẽn đặc thù (Non-IID) theo thời gian.

Output: data/raw/tomtom_traffic.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Cấu hình
OSRM_PATH = "data/raw/hcm_osrm_dataset.csv"
ZONE_PATH = "data/raw/zone_labels.csv"
OUT_PATH  = "data/raw/tomtom_traffic.csv"

# Thiết lập seed để đảm bảo khả năng tái tạo
np.random.seed(42)

def get_time_label(hour: int) -> str:
    if hour in range(0, 6):   return "night"
    if hour in range(7, 10):  return "rush_morning"
    if hour in range(16, 20): return "rush_evening"
    return "normal"

def main():
    print("=" * 55)
    print("  Generating Synthetic Dynamic Traffic Data")
    print("=" * 55)

    # 1. Load OSRM to get base edges (distance and static duration)
    assert os.path.exists(OSRM_PATH), f"Missing OSRM data: {OSRM_PATH}"
    df_osrm = pd.read_csv(OSRM_PATH)
    
    # Tính trung bình khoảng cách và thời gian tĩnh cho từng cặp origin -> destination
    edges_base = df_osrm.groupby(["origin", "destination"])[["distance_m", "duration_s"]].mean().reset_index()
    print(f"✓ Loaded {len(edges_base)} base road segments from OSRM")

    # 2. Load Zone Labels
    assert os.path.exists(ZONE_PATH), f"Missing zone labels: {ZONE_PATH}"
    df_zone = pd.read_csv(ZONE_PATH, index_col="node")
    zone_types = ["commercial","residential","industrial","school","university","hospital","transport","park"]
    print("✓ Loaded zone labels for 17 TAZs")

    # 3. Định nghĩa thời gian mô phỏng: 7 ngày (thứ Hai đến Chủ nhật)
    start_date = datetime(2026, 5, 18, 0, 0, 0) # Thứ Hai
    interval_min = 15
    steps_per_day = 24 * (60 // interval_min) # 96
    num_days = 7
    total_steps = steps_per_day * num_days # 672 snapshots

    timestamps = [start_date + timedelta(minutes=i * interval_min) for i in range(total_steps)]
    print(f"✓ Generated {len(timestamps)} time steps (7 days, 15-minute interval)")

    # 4. Sinh dữ liệu giao thông
    records = []
    
    for step_idx, ts in enumerate(timestamps):
        hour = ts.hour
        dow = ts.weekday() # 0 = Monday, 6 = Sunday
        is_weekend = int(dow >= 5)
        time_label = get_time_label(hour)
        
        # In tiến độ
        if (step_idx + 1) % 100 == 0 or step_idx == 0:
            print(f"  Processing step {step_idx + 1}/{total_steps} | {ts}")

        for _, edge in edges_base.iterrows():
            u = edge["origin"]
            v = edge["destination"]
            base_dist = edge["distance_m"]
            base_dur = edge["duration_s"]
            
            # Lấy thông tin zone của source u và destination v
            z_u = df_zone.loc[u]
            z_v = df_zone.loc[v]
            
            # Khởi tạo congestion ratio cơ sở (free flow = 1.0)
            cong_ratio = 1.0
            
            # Thêm các yếu tố tắc nghẽn động học
            if not is_weekend:
                # NGÀY THƯỜNG (Thứ Hai - Thứ Sáu)
                # Cao điểm sáng (7:00 - 9:00)
                if 7 <= hour < 9:
                    cong_ratio += 0.25 # Nền chung giờ cao điểm sáng
                    if z_v["school"] or z_v["university"]:
                        cong_ratio += 0.40 # Học sinh/sinh viên đến trường
                    if z_v["industrial"]:
                        cong_ratio += 0.35 # Công nhân đến khu công nghiệp
                    if z_u["residential"]:
                        cong_ratio += 0.20 # Người dân rời nhà
                    if z_v["transport"]:
                        cong_ratio += 0.20 # Bến xe, sân bay đông đúc
                
                # Cao điểm chiều (16:30 - 18:30)
                elif 16 <= hour < 19:
                    cong_ratio += 0.30 # Nền chung giờ cao điểm chiều
                    if z_u["school"] or z_u["university"]:
                        cong_ratio += 0.35 # Tan học
                    if z_u["industrial"]:
                        cong_ratio += 0.45 # Tan ca
                    if z_v["residential"]:
                        cong_ratio += 0.25 # Người dân về nhà
                    if z_v["commercial"]:
                        cong_ratio += 0.25 # Đi mua sắm, ăn uống
                        
                # Giờ hành chính hoặc chiều tối thường
                elif 9 <= hour < 16:
                    cong_ratio += 0.10
                    if z_v["commercial"] or z_v["hospital"]:
                        cong_ratio += 0.15
                elif 19 <= hour < 22:
                    if z_v["commercial"]:
                        cong_ratio += 0.25 # Ăn tối, vui chơi thương mại
                    if z_v["park"]:
                        cong_ratio += 0.15
            else:
                # CUỐI TUẦN (Thứ Bảy & Chủ Nhật)
                # Không có đỉnh trường học/khu công nghiệp
                # Thương mại và công viên là đỉnh chính
                if 10 <= hour < 14: # Đỉnh trưa cuối tuần
                    cong_ratio += 0.15
                    if z_v["commercial"]:
                        cong_ratio += 0.35
                elif 17 <= hour < 21: # Đỉnh tối cuối tuần
                    cong_ratio += 0.20
                    if z_v["commercial"]:
                        cong_ratio += 0.45
                    if z_v["park"]:
                        cong_ratio += 0.25
                    if z_v["residential"]:
                        cong_ratio += 0.15
            
            # Ban đêm giảm thiểu
            if 0 <= hour < 5:
                cong_ratio = 1.0 # free flow
            
            # Thêm nhiễu ngẫu nhiên nhỏ (random fluctuations)
            noise = np.random.normal(0.0, 0.08)
            cong_ratio += noise
            
            # Giới hạn dưới là 1.0 (không thể chạy nhanh hơn free-flow lý thuyết quá nhiều)
            # Giới hạn trên là 3.5 (kẹt xe nghiêm trọng)
            cong_ratio = max(0.95, min(3.5, cong_ratio))
            
            # Tính toán travel time, delay dựa trên congestion ratio
            travel_time = base_dur * cong_ratio
            delay = max(0.0, travel_time - base_dur)
            
            records.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "time_label": time_label,
                "src_node": u,
                "dst_node": v,
                "src_name": u,
                "dst_name": v,
                "src_poi": NODES_POI_HINT.get(u, "mixed"),
                "dst_poi": NODES_POI_HINT.get(v, "mixed"),
                "travel_time_s": round(travel_time, 1),
                "free_flow_time_s": round(base_dur, 1),
                "traffic_delay_s": round(delay, 1),
                "length_m": round(base_dist, 1),
                "congestion_ratio": round(cong_ratio, 3)
            })

    # Ghi dữ liệu ra CSV
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Successfully generated synthetic traffic data and saved to {OUT_PATH}")
    print(f"   Total rows: {len(df_out):,}")
    print(f"   Congestion Ratio summary:\n{df_out['congestion_ratio'].describe().to_string()}")

# Gợi ý POI để giữ cấu trúc TomTom tương thích
NODES_POI_HINT = {
    "Ben Thanh Market":     "commercial",
    "District 1":           "mixed",
    "District 3":           "residential",
    "District 5":           "residential",
    "Binh Thanh":           "residential",
    "Tan Son Nhat Airport": "transport",
    "Landmark 81":          "commercial",
    "Thu Duc":              "mixed",
    "Linh Trung":           "industrial",
    "Suoi Tien":            "commercial",
    "High Tech Park":       "industrial",
    "VNU HCM":              "university",
    "Hang Xanh":            "arterial",
    "Saigon Bridge":        "bridge",
    "Eastern Bus Station":  "transport",
    "Thu Thiem Tunnel":     "transport",
    "Pham Van Dong":        "arterial",
}

if __name__ == "__main__":
    main()
