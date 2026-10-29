"""
scripts/pipeline/build_hcm_real.py
==================================
Script điều phối toàn diện Pipeline HCM-Real (Raw -> Cleaned -> Canonical).
Tái hiện lại dataset 100% bằng 1 dòng lệnh duy nhất:
    python scripts/pipeline/build_hcm_real.py --raw-path data/raw/tomtom_sample.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
import pandas as pd

# Đảm bảo import được package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.pipeline.clean_raw import clean_raw_traffic
from scripts.pipeline.resampler import resample_to_hourly_canonical
from scripts.pipeline.enrich_static_metadata import build_static_metadata
from scripts.pipeline.manifest import generate_manifest
from scripts.pipeline.schemas import validate_canonical_dataframe, CANONICAL_COLUMNS


def load_raw_input(raw_path: str) -> pd.DataFrame:
    """Đọc file raw từ nhiều định dạng: CSV, JSON, JSONL."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Không tìm thấy file raw tại: {raw_path}")

    ext = os.path.splitext(raw_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(raw_path)
    elif ext == ".json":
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and "records" in data:
            return pd.DataFrame(data["records"])
        else:
            return pd.DataFrame([data])
    elif ext in [".jsonl", ".ndjson"]:
        records = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {ext} (chỉ nhận .csv, .json, .jsonl)")


def run_pipeline(
    raw_path: str,
    registry_path: str = "data/config/hcm_tomtom_segments.v1.json",
    zone_path: str = "data/raw/zone_labels.csv",
    osrm_path: str = "data/raw/hcm_osrm_dataset.csv",
    out_dir: str = "data/processed/hcm_real_v1",
    dataset_version: str = "hcm_real_v1.0",
    schema_version: str = "1.0.0",
    raw_timezone: str = "auto",
    immutable: bool = False,
    force_overwrite: bool = True,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> dict:
    """
    Chạy toàn bộ pipeline từ raw -> cleaned -> canonical -> enriched metadata -> manifest.
    """
    if immutable:
        timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"{out_dir.rstrip('/\\')}_{timestamp_suffix}"

    if not force_overwrite and os.path.exists(os.path.join(out_dir, "manifest.json")):
        raise FileExistsError(
            f"DatasetImmutabilityError: Thư mục '{out_dir}' đã chứa dataset có sẵn và cờ force_overwrite=False. "
            "Sử dụng --immutable để tạo thư mục phiên bản mới hoặc bật --force-overwrite."
        )

    os.makedirs(out_dir, exist_ok=True)
    print("=" * 65)
    print(f"  Bắt đầu Pipeline Build HCM-Real [{dataset_version}]")
    print(f"  Thư mục đích: {out_dir}")
    print("=" * 65)

    # 1. Đọc dữ liệu thô
    print(f"📂 [1/5] Đang đọc raw data từ: {raw_path}")
    df_raw = load_raw_input(raw_path)
    print(f"   -> Đọc thành công {len(df_raw)} bản ghi thô.")

    # 2. Làm sạch (Cleaned)
    print("🧹 [2/5] Đang làm sạch và chuẩn hóa (UTC, Local, đơn vị, duplicate)...")
    df_cleaned = clean_raw_traffic(df_raw, default_dataset_version=dataset_version, raw_timezone=raw_timezone)
    dup_count = int((df_cleaned["quality_flag"] == "DUPLICATE_FLAGGED").sum())
    print(f"   -> Đã làm sạch {len(df_cleaned)} bản ghi (phát hiện {dup_count} duplicates gắn cờ).")

    # Lưu bảng trung gian traffic_cleaned để bảo toàn vết duplicate phục vụ kiểm toán đối soát
    cleaned_csv = os.path.join(out_dir, "traffic_cleaned.csv")
    cleaned_parquet = os.path.join(out_dir, "traffic_cleaned.parquet")
    df_cleaned.to_csv(cleaned_csv, index=False)
    try:
        df_cleaned.to_parquet(cleaned_parquet, index=False)
    except Exception:
        pass
    print(f"   -> Đã lưu bảng trung gian: {cleaned_csv} (bảo toàn toàn bộ {dup_count} duplicate records).")

    # 3. Canonical Resampling
    print("⏱️ [3/5] Đang thực hiện Hourly Resampling và gán Quality Flags...")
    df_canonical = resample_to_hourly_canonical(
        df_cleaned=df_cleaned,
        registry_path=registry_path,
        dataset_version=dataset_version,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    print(f"   -> Tạo thành công lưới canonical với {len(df_canonical)} slots.")

    # 4. Kiểm tra Data Contract
    print("🔍 [4/5] Đang kiểm tra Data Contract (Schema & Constraints)...")
    validation = validate_canonical_dataframe(df_canonical)
    if not validation["is_valid"]:
        print("❌ Lỗi Data Contract không đạt:")
        for err in validation["errors"]:
            print(f"   - {err}")
        raise ValueError("Data Contract validation thất bại! Dừng build.")
    else:
        print("   ✅ Data Contract PASS 100% (đủ 20 cột, đúng kiểu, ràng buộc hợp lệ).")

    # Lưu canonical traffic ra CSV và Parquet
    traffic_csv = os.path.join(out_dir, "traffic_canonical.csv")
    traffic_parquet = os.path.join(out_dir, "traffic_canonical.parquet")
    df_canonical.to_csv(traffic_csv, index=False)
    try:
        df_canonical.to_parquet(traffic_parquet, index=False)
        parquet_saved = True
    except Exception:
        parquet_saved = False

    # 5. Gắn Static Metadata (OSM, OSRM, Registry)
    print("🗺️ [5/5] Đang gắn Static Metadata (OSM zones, OSRM metrics, Adjacency)...")
    df_nodes, adj_tensor = build_static_metadata(
        registry_path=registry_path,
        zone_path=zone_path,
        osrm_path=osrm_path,
        out_dir=out_dir,
    )
    print(f"   -> Đã tạo bảng nodes ({len(df_nodes)} trạm) và ma trận kề A ({adj_tensor.shape}).")

    # 6. Tạo Manifest
    print("📝 Đang sinh manifest.json và tính mã băm SHA-256 artifacts...")
    manifest = generate_manifest(
        out_dir=out_dir,
        config_path=registry_path,
        df_canonical=df_canonical,
        dataset_version=dataset_version,
        schema_version=schema_version,
        source_files={
            "raw_traffic_input": raw_path,
            "osm_zone_labels": zone_path,
            "osrm_dataset": osrm_path,
        },
        df_cleaned=df_cleaned,
    )

    print("=" * 65)
    print("  ✅ HOÀN THÀNH BUILD DATASET HCM-REAL!")
    print(f"  - Thư mục output : {out_dir}")
    print(f"  - Tổng bản ghi    : {len(df_canonical)}")
    print(f"  - Tỷ lệ có dữ liệu: {manifest['coverage']['overall_valid_ratio'] * 100:.2f}%")
    print(f"  - Tỷ lệ missing   : {manifest['coverage']['overall_missing_ratio'] * 100:.2f}%")
    print(f"  - Duplicate logged: {manifest['audit_provenance']['duplicate_records_flagged']}")
    print(f"  - Artifacts       : traffic_cleaned.csv, traffic_canonical.csv, nodes.csv, osrm_distance_matrix.csv, adjacency.pt, manifest.json")
    print("=" * 65)

    return {
        "df_canonical": df_canonical,
        "df_cleaned": df_cleaned,
        "df_nodes": df_nodes,
        "adj_tensor": adj_tensor,
        "manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description="Build canonical HCM-Real dataset")
    parser.add_argument("--raw-path", required=True, help="Đường dẫn file raw TomTom (.csv, .json, .jsonl)")
    parser.add_argument("--registry-path", default="data/config/hcm_tomtom_segments.v1.json", help="Đường dẫn registry 17 segments")
    parser.add_argument("--zone-path", default="data/raw/zone_labels.csv", help="Đường dẫn OSM zone labels")
    parser.add_argument("--osrm-path", default="data/raw/hcm_osrm_dataset.csv", help="Đường dẫn OSRM routing dataset")
    parser.add_argument("--out-dir", default="data/processed/hcm_real_v1", help="Thư mục xuất kết quả")
    parser.add_argument("--dataset-version", default="hcm_real_v1.0", help="Phiên bản dataset")
    parser.add_argument("--schema-version", default="1.0.0", help="Phiên bản schema")
    parser.add_argument("--raw-timezone", default="auto", help="Múi giờ dữ liệu thô (auto, UTC, Asia/Ho_Chi_Minh)")
    parser.add_argument("--start-utc", default=None, help="Mốc thời gian bắt đầu kế hoạch (ISO 8601 UTC, bảo vệ Edge Truncation)")
    parser.add_argument("--end-utc", default=None, help="Mốc thời gian kết thúc kế hoạch (ISO 8601 UTC, bảo vệ Edge Truncation)")
    parser.add_argument("--immutable", action="store_true", help="Tạo thư mục bất biến có hậu tố timestamp tránh ghi đè")
    parser.add_argument("--no-force-overwrite", dest="force_overwrite", action="store_false", help="Báo lỗi nếu thư mục đích đã có dataset")
    parser.set_defaults(force_overwrite=True)

    args = parser.parse_args()
    start_dt = pd.to_datetime(args.start_utc, utc=True).to_pydatetime() if args.start_utc else None
    end_dt = pd.to_datetime(args.end_utc, utc=True).to_pydatetime() if args.end_utc else None

    run_pipeline(
        raw_path=args.raw_path,
        registry_path=args.registry_path,
        zone_path=args.zone_path,
        osrm_path=args.osrm_path,
        out_dir=args.out_dir,
        dataset_version=args.dataset_version,
        schema_version=args.schema_version,
        raw_timezone=args.raw_timezone,
        immutable=args.immutable,
        force_overwrite=args.force_overwrite,
        start_utc=start_dt,
        end_utc=end_dt,
    )


if __name__ == "__main__":
    main()
