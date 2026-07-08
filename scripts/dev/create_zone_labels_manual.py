"""
create_zone_labels_manual.py
============================
Tạo zone_labels.csv dựa trên kiến thức địa lý thực tế của TP.HCM.
Dùng khi Overpass API bị timeout/lỗi.

Đây cũng là cách được dùng trong nhiều paper thực tế:
gán nhãn zone dựa trên quy hoạch đô thị chính thức
hoặc POI data từ Google Maps.
"""

import pandas as pd
import os

ZONE_TYPES = ["commercial","residential","industrial",
              "school","university","hospital","transport","park"]

# ──────────────────────────────────────────────
# ZONE LABELS THỦ CÔNG
# Dựa trên: quy hoạch đô thị HCM, Google Maps POI, thực tế địa bàn
# 1 = zone này tồn tại trong bán kính 800m quanh node
# ──────────────────────────────────────────────
#                                           comm  resid indus school univ  hosp  transp park
ZONE_DATA = {
    "Ben Thanh Market":     [1,     0,     0,     1,    0,    1,    1,    0],
    # Chợ Bến Thành = commercial; gần BV Từ Dũ=hospital; gần bến xe buýt=transport; có trường

    "District 1":           [1,     1,     0,     0,    0,    1,    1,    1],
    # Trung tâm tài chính, khu phố Tây, BV Chợ Rẫy gần đó, Công viên 23/9

    "District 3":           [1,     1,     0,     1,    0,    1,    0,    1],
    # Khu dân cư + trường học dày đặc, BV Bình Dân, có công viên Lê Văn Tám

    "District 5":           [1,     1,     1,     1,    1,    1,    0,    0],
    # Chợ Lớn (commercial+industrial), ĐH Y Dược (university), nhiều trường, BV Chợ Rẫy

    "Binh Thanh":           [1,     1,     0,     1,    0,    0,    0,    0],
    # Khu dân cư đông đúc, nhiều trường phổ thông, một số điểm thương mại

    "Tan Son Nhat Airport": [1,     0,     0,     0,    0,    0,    1,    0],
    # Sân bay (transport) + khu thương mại xung quanh

    "Landmark 81":          [1,     1,     0,     0,    0,    0,    0,    0],
    # Khu Vinhomes Central Park: commercial + residential cao cấp

    "Thu Duc":              [1,     1,     0,     1,    1,    0,    1,    0],
    # TP Thủ Đức: commercial + residential + trường ĐH + bến xe

    "Linh Trung":           [0,     1,     1,     0,    1,    0,    0,    0],
    # KCX Linh Trung (industrial) + KTX ĐH (university + residential)

    "Suoi Tien":            [1,     0,     0,     0,    0,    0,    1,    1],
    # Khu du lịch Suối Tiên (park/commercial) + bến xe miền Đông (transport)

    "High Tech Park":       [0,     0,     1,     0,    0,    0,    0,    0],
    # Khu Công nghệ Cao TP.HCM: thuần industrial/R&D

    "VNU HCM":              [0,     1,     0,     1,    1,    0,    0,    1],
    # ĐH Quốc gia: university + school + residential (KTX) + công viên nội bộ

    "Hang Xanh":            [1,     1,     0,     1,    0,    0,    1,    0],
    # Ngã tư Hàng Xanh: giao thông lớn (transport) + dân cư + thương mại

    "Saigon Bridge":        [0,     1,     0,     0,    0,    0,    1,    0],
    # Cầu Sài Gòn: transport + dân cư 2 bên bờ

    "Eastern Bus Station":  [1,     1,     0,     0,    0,    0,    1,    0],
    # Bến xe miền Đông: transport + commercial + residential xung quanh

    "Thu Thiem Tunnel":     [1,     1,     0,     0,    0,    0,    1,    0],
    # Hầm Thủ Thiêm: transport + Thủ Thiêm đang phát triển commercial + residential

    "Pham Van Dong":        [1,     1,     0,     1,    0,    0,    1,    0],
    # Đường PVĐ: arterial (transport) + commercial + trường học dọc đường
}

os.makedirs("data/raw", exist_ok=True)
df = pd.DataFrame(ZONE_DATA, index=ZONE_TYPES).T
df.index.name = "node"
df = df.astype(int)
df.to_csv("data/raw/zone_labels.csv")

print("✅ Zone labels saved to data/raw/zone_labels.csv")
print(f"\n{'Node':<25} {'Zones':<60} {'Count'}")
print("-" * 90)
for node, row in df.iterrows():
    active = [z for z in ZONE_TYPES if row[z] == 1]
    star = "⭐" if len(active) > 2 else "  "
    print(f"{star} {node:<23} {', '.join(active):<60} {len(active)}")

print(f"\n📊 Multi-zone nodes (>2 zones): {(df.sum(axis=1) > 2).sum()}")
print(f"   Single-zone nodes: {(df.sum(axis=1) == 1).sum()}")
