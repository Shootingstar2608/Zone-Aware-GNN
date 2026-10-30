"""
tests/test_hcm_real_pipeline.py
===============================
Bộ Unit Tests toàn diện cho Pipeline HCM-Real (Thành viên B).
Bao phủ 100% tiêu chí nghiệm thu:
  1. test_canonical_schema_20_columns
  2. test_mandatory_non_null_fields
  3. test_timestamp_timezone_conversion
  4. test_value_ranges
  5. test_congestion_ratio_formula
  6. test_free_flow_zero_handling
  7. test_faster_than_free_flow_flag
  8. test_missing_flags_and_reasons
  9. test_duplicate_flagging
  10. test_time_label_matching_build_graph
  11. test_static_metadata_no_collision
  12. test_manifest_reproducibility

Chạy được cả 2 cách:
    python tests/test_hcm_real_pipeline.py
    pytest tests/test_hcm_real_pipeline.py -v
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.pipeline.schemas import (
    CANONICAL_COLUMNS,
    MANDATORY_NON_NULL_COLUMNS,
    QualityFlag,
    MissingReason,
    TimeLabel,
    get_time_label,
    validate_canonical_dataframe,
    load_hcm_real_canonical,
)
from scripts.pipeline.clean_raw import clean_raw_traffic
from scripts.pipeline.resampler import resample_to_hourly_canonical
from scripts.pipeline.enrich_static_metadata import build_static_metadata
from scripts.pipeline.manifest import generate_manifest, compute_file_sha256

LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


# ── Fixtures & Mock Helpers ──────────────────────────────────────────────────

def _make_dummy_registry(tmp_path: str) -> str:
    """Tạo file registry giả lập gồm 3 trạm để test đơn vị."""
    data = {
        "registry_version": "v1.0-test",
        "openlr_standard": "TomTom_OpenLR_Binary_v1.4",
        "openlr_encoding": "Base64",
        "segments": [
            {
                "node_id": "Node_A",
                "segment_id": "seg_01_a",
                "lat": 10.77,
                "lon": 106.69,
                "corridor_name": "Corridor A",
                "approved_openlr": "CwRbVCRvVyPNBQD8/4sZCQ==",
                "selection_reason": "Test node A",
                "status": "approved",
                "segment_length_m": 1000.0,
                "default_free_flow_speed_kmh": 40.0,
            },
            {
                "node_id": "Node_B",
                "segment_id": "seg_02_b",
                "lat": 10.78,
                "lon": 106.70,
                "corridor_name": "Corridor B",
                "approved_openlr": "CwRbViRvVyTNBAD9/4sZCQ==",
                "selection_reason": "Test node B",
                "status": "approved",
                "segment_length_m": 1200.0,
                "default_free_flow_speed_kmh": 50.0,
            },
            {
                "node_id": "Node_C",
                "segment_id": "seg_03_c",
                "lat": 10.79,
                "lon": 106.71,
                "corridor_name": "Corridor C",
                "approved_openlr": "CwRcPCRvUyHNAwD8/4sZCQ==",
                "selection_reason": "Test node C",
                "status": "approved",
                "segment_length_m": 1500.0,
                "default_free_flow_speed_kmh": 60.0,
            },
        ],
    }
    path = os.path.join(tmp_path, "test_registry.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


# ── Tests ────────────────────────────────────────────────────────────────────

def test_canonical_schema_20_columns():
    """Kiểm tra schema canonical có đúng 20 cột theo thứ tự chuẩn."""
    assert len(CANONICAL_COLUMNS) == 20, f"Kỳ vọng 20 cột, nhưng có {len(CANONICAL_COLUMNS)}"
    expected_cols = [
        "timestamp_utc", "timestamp_local", "node_id", "segment_id",
        "approved_openlr", "source", "current_speed_kmh", "free_flow_speed_kmh",
        "current_travel_time_s", "free_flow_travel_time_s", "congestion_ratio",
        "traffic_delay_s", "segment_length_m", "time_label", "confidence",
        "road_closure", "is_missing", "missing_reason", "quality_flag", "dataset_version",
    ]
    assert CANONICAL_COLUMNS == expected_cols


def _make_dummy_canonical_row() -> dict:
    return {
        "timestamp_utc": pd.Timestamp("2026-10-25 01:00:00+00:00"),
        "timestamp_local": pd.Timestamp("2026-10-25 08:00:00+07:00"),
        "node_id": "Node_A",
        "segment_id": "seg_01_a",
        "approved_openlr": "CwRbVCRvVyPNBQD8/4sZCQ==",
        "source": "tomtom_flow_segment_api",
        "current_speed_kmh": 35.0,
        "free_flow_speed_kmh": 40.0,
        "current_travel_time_s": 90.0,
        "free_flow_travel_time_s": 90.0,
        "congestion_ratio": 1.0,
        "traffic_delay_s": 0.0,
        "segment_length_m": 1000.0,
        "time_label": "rush_morning",
        "confidence": 1.0,
        "road_closure": False,
        "is_missing": False,
        "missing_reason": "NONE",
        "quality_flag": "VALID",
        "dataset_version": "hcm_real_v1.0",
    }


def test_mandatory_non_null_fields():
    """Kiểm tra validator bắt lỗi khi trường bắt buộc bị null."""
    row = _make_dummy_canonical_row()
    row["timestamp_utc"] = None  # Cột bắt buộc bị null
    df_bad = pd.DataFrame([row])
    val = validate_canonical_dataframe(df_bad)
    assert not val["is_valid"], "Validator phải báo lỗi khi timestamp_utc null"
    assert any("timestamp_utc" in err for err in val["errors"])


def test_timestamp_timezone_conversion():
    """Kiểm tra chuẩn hóa múi giờ: UTC và Asia/Ho_Chi_Minh (+07:00)."""
    # 2026-10-25 08:00:00 Local (+07:00) tương ứng 2026-10-25 01:00:00 UTC
    raw_df = pd.DataFrame([{
        "timestamp": "2026-10-25 08:00:00",
        "node_id": "Node_A",
        "currentSpeed": 35.0,
        "freeFlowSpeed": 40.0,
        "currentTravelTime": 90.0,
        "freeFlowTravelTime": 90.0,
    }])
    cleaned = clean_raw_traffic(raw_df)
    ts_utc = cleaned["timestamp_utc"].iloc[0]
    ts_local = cleaned["timestamp_local"].iloc[0]

    assert ts_utc.tzinfo == timezone.utc
    assert ts_local.tzinfo == LOCAL_TZ
    assert ts_local.hour == 8
    assert ts_utc.hour == 1  # 8h local - 7h offset = 1h UTC


def test_value_ranges():
    """Kiểm tra bounds: confidence [0, 1], speed >= 0, delay >= 0."""
    row = _make_dummy_canonical_row()
    row["confidence"] = 1.5  # Vượt ngoài [0, 1]
    row["current_speed_kmh"] = -10.0  # Tốc độ âm
    df_invalid = pd.DataFrame([row])
    val = validate_canonical_dataframe(df_invalid)
    assert not val["is_valid"]
    assert any("confidence" in err for err in val["errors"])
    assert any("current_speed_kmh" in err for err in val["errors"])


def test_congestion_ratio_formula():
    """Kiểm tra công thức chuẩn: current_travel_time / free_flow_travel_time (có thể > 1.0)."""
    # Giả lập 1 quan sát: travel_time = 150s, free_flow = 100s -> ratio = 1.5
    raw_df = pd.DataFrame([{
        "timestamp": "2026-10-25 08:00:00",
        "node_id": "Node_A",
        "currentSpeed": 25.0,
        "freeFlowSpeed": 40.0,
        "currentTravelTime": 150.0,
        "freeFlowTravelTime": 100.0,
        "confidence": 1.0,
        "roadClosure": False,
    }])
    cleaned = clean_raw_traffic(raw_df)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        start_u = datetime(2026, 10, 25, 1, 0, 0, tzinfo=timezone.utc)
        canonical = resample_to_hourly_canonical(cleaned, reg_path, start_utc=start_u, end_utc=start_u)

        row_a = canonical[canonical["node_id"] == "Node_A"].iloc[0]
        assert row_a["congestion_ratio"] == 1.5, f"Kỳ vọng 1.5, nhận được {row_a['congestion_ratio']}"
        assert row_a["quality_flag"] == QualityFlag.VALID.value


def test_free_flow_zero_handling():
    """Kiểm tra bẫy mẫu số = 0: gán congestion_ratio=null, ANOMALY_FREE_FLOW_ZERO, không crash."""
    raw_df = pd.DataFrame([{
        "timestamp": "2026-10-25 08:00:00",
        "node_id": "Node_A",
        "currentSpeed": 30.0,
        "freeFlowSpeed": 0.0,
        "currentTravelTime": 100.0,
        "freeFlowTravelTime": 0.0,  # Lỗi API trả mẫu số = 0
        "confidence": 1.0,
        "roadClosure": False,
    }])
    cleaned = clean_raw_traffic(raw_df)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        start_u = datetime(2026, 10, 25, 1, 0, 0, tzinfo=timezone.utc)
        canonical = resample_to_hourly_canonical(cleaned, reg_path, start_utc=start_u, end_utc=start_u)

        row_a = canonical[canonical["node_id"] == "Node_A"].iloc[0]
        assert pd.isna(row_a["congestion_ratio"]), "Khi ff=0 thì congestion_ratio phải là null/NaN"
        assert row_a["quality_flag"] == QualityFlag.ANOMALY_FREE_FLOW_ZERO.value
        assert not row_a["is_missing"], "Không được drop dòng, dòng vẫn phải tồn tại"


def test_faster_than_free_flow_flag():
    """Kiểm tra phát hiện T_curr < T_ff * 0.8 -> cờ ANOMALY_FASTER_THAN_FREE_FLOW."""
    raw_df = pd.DataFrame([{
        "timestamp": "2026-10-25 08:00:00",
        "node_id": "Node_A",
        "currentSpeed": 60.0,
        "freeFlowSpeed": 40.0,
        "currentTravelTime": 60.0,  # Nhanh hơn lý thuyết rất nhiều (60s < 100s * 0.8)
        "freeFlowTravelTime": 100.0,
        "confidence": 1.0,
        "roadClosure": False,
    }])
    cleaned = clean_raw_traffic(raw_df)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        start_u = datetime(2026, 10, 25, 1, 0, 0, tzinfo=timezone.utc)
        canonical = resample_to_hourly_canonical(cleaned, reg_path, start_utc=start_u, end_utc=start_u)

        row_a = canonical[canonical["node_id"] == "Node_A"].iloc[0]
        assert row_a["quality_flag"] == QualityFlag.ANOMALY_FASTER_THAN_FREE_FLOW.value
        assert row_a["congestion_ratio"] == 0.6


def test_missing_flags_and_reasons():
    """Kiểm tra chèn bản ghi thiếu: is_missing=True, missing_reason='NOT_COLLECTED'."""
    # Chỉ cung cấp dữ liệu cho Node_A, Node_B và Node_C bị thiếu hoàn toàn
    raw_df = pd.DataFrame([{
        "timestamp": "2026-10-25 08:00:00",
        "node_id": "Node_A",
        "currentSpeed": 35.0,
        "freeFlowSpeed": 40.0,
        "currentTravelTime": 90.0,
        "freeFlowTravelTime": 90.0,
    }])
    cleaned = clean_raw_traffic(raw_df)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        start_u = datetime(2026, 10, 25, 1, 0, 0, tzinfo=timezone.utc)
        canonical = resample_to_hourly_canonical(cleaned, reg_path, start_utc=start_u, end_utc=start_u)

        # Kiểm tra Node_B
        row_b = canonical[canonical["node_id"] == "Node_B"].iloc[0]
        assert row_b["is_missing"] == True
        assert row_b["missing_reason"] == MissingReason.NOT_COLLECTED.value
        assert row_b["quality_flag"] == QualityFlag.MISSING_NOT_COLLECTED.value
        assert pd.isna(row_b["current_speed_kmh"])


def test_duplicate_flagging():
    """Kiểm tra phát hiện duplicate: không xóa mà đánh dấu cờ DUPLICATE_FLAGGED."""
    raw_df = pd.DataFrame([
        {
            "timestamp": "2026-10-25 08:00:00",
            "node_id": "Node_A",
            "currentSpeed": 35.0,
            "freeFlowSpeed": 40.0,
            "currentTravelTime": 90.0,
            "freeFlowTravelTime": 90.0,
        },
        {
            "timestamp": "2026-10-25 08:00:00",  # Trùng lặp hoàn toàn thời gian và node_id
            "node_id": "Node_A",
            "currentSpeed": 36.0,
            "freeFlowSpeed": 40.0,
            "currentTravelTime": 88.0,
            "freeFlowTravelTime": 90.0,
        },
    ])
    cleaned = clean_raw_traffic(raw_df)
    assert len(cleaned) == 2, "Tuyệt đối không được xóa dòng duplicate"
    assert cleaned["quality_flag"].iloc[1] == QualityFlag.DUPLICATE_FLAGGED.value


def test_time_label_matching_build_graph():
    """Kiểm tra nhãn khung giờ khớp 100% với build_graph.py."""
    # 0h - 5h -> night
    for h in [0, 2, 5]:
        assert get_time_label(h) == "night"
    # 7h - 9h -> rush_morning
    for h in [7, 8, 9]:
        assert get_time_label(h) == "rush_morning"
    # 16h - 19h -> rush_evening
    for h in [16, 18, 19]:
        assert get_time_label(h) == "rush_evening"
    # 6h, 10h-15h, 20h-23h -> normal
    for h in [6, 10, 12, 15, 20, 23]:
        assert get_time_label(h) == "normal"


def test_static_metadata_no_collision():
    """Kiểm tra static metadata có prefix rõ ràng (osm_*, osrm_*, registry_*) và không đè nhau."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        zone_csv = os.path.join(tmp_dir, "zones.csv")
        osrm_csv = os.path.join(tmp_dir, "osrm.csv")

        pd.DataFrame([
            {"node": "Node_A", "commercial": 1, "residential": 0},
            {"node": "Node_B", "commercial": 0, "residential": 1},
            {"node": "Node_C", "commercial": 0, "residential": 0},
        ]).to_csv(zone_csv, index=False)

        pd.DataFrame([
            {"origin": "Node_A", "destination": "Node_B", "distance_m": 1200.0, "duration_s": 150.0},
            {"origin": "Node_B", "destination": "Node_A", "distance_m": 1200.0, "duration_s": 150.0},
        ]).to_csv(osrm_csv, index=False)

        df_nodes, adj = build_static_metadata(reg_path, zone_csv, osrm_csv, tmp_dir)

        # Kiểm tra prefix
        assert "registry_lat" in df_nodes.columns
        assert "osm_commercial" in df_nodes.columns
        assert "osrm_avg_outgoing_dist_m" in df_nodes.columns
        # Đảm bảo không có trường nào ghi đè lên nhau
        assert df_nodes.loc[df_nodes["node_id"] == "Node_A", "osm_commercial"].iloc[0] == 1.0


def test_manifest_reproducibility():
    """Kiểm tra sinh manifest.json đầy đủ 6 thông tin bắt buộc và hash hợp lệ."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        dummy_canonical = pd.DataFrame([{
            col: ("2026-10-25 08:00:00" if "timestamp" in col else (False if "is_missing" in col or "road_closure" in col else ("VALID" if "quality_flag" in col else ("NONE" if "missing_reason" in col else 1.0))))
            for col in CANONICAL_COLUMNS
        }])
        dummy_canonical["node_id"] = "Node_A"
        dummy_canonical["timestamp_utc"] = pd.to_datetime(["2026-10-25 01:00:00"])

        manifest = generate_manifest(tmp_dir, reg_path, dummy_canonical)

        assert manifest["dataset_name"] == "HCM-Real"
        assert "config_hash" in manifest
        assert manifest["config_hash"].startswith("sha256:")
        assert "collection_time" in manifest
        assert "coverage" in manifest
        assert manifest["total_records"] == 1
        assert "artifact_hashes" in manifest


def test_multiday_hourly_resampling_grid_continuity():
    """Kiểm tra tính liên tục của lưới giờ qua nhiều ngày (3 ngày x 24h x 3 trạm = 216 slots)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        # Tạo dữ liệu lác đác trong 3 ngày (chỉ có 5 bản ghi cho Node_A)
        raw_df = pd.DataFrame([
            {"timestamp": "2026-10-25 00:00:00", "node_id": "Node_A", "currentSpeed": 40.0, "freeFlowSpeed": 40.0, "currentTravelTime": 90.0, "freeFlowTravelTime": 90.0},
            {"timestamp": "2026-10-25 12:00:00", "node_id": "Node_A", "currentSpeed": 30.0, "freeFlowSpeed": 40.0, "currentTravelTime": 120.0, "freeFlowTravelTime": 90.0},
            {"timestamp": "2026-10-26 08:00:00", "node_id": "Node_A", "currentSpeed": 25.0, "freeFlowSpeed": 40.0, "currentTravelTime": 150.0, "freeFlowTravelTime": 90.0},
            {"timestamp": "2026-10-27 18:00:00", "node_id": "Node_A", "currentSpeed": 20.0, "freeFlowSpeed": 40.0, "currentTravelTime": 180.0, "freeFlowTravelTime": 90.0},
            {"timestamp": "2026-10-27 23:00:00", "node_id": "Node_A", "currentSpeed": 45.0, "freeFlowSpeed": 40.0, "currentTravelTime": 80.0, "freeFlowTravelTime": 90.0},
        ])
        cleaned = clean_raw_traffic(raw_df)
        canonical = resample_to_hourly_canonical(cleaned, reg_path)

        # 3 ngày = 72 giờ x 3 nodes = 216 slots
        assert len(canonical) == 216, f"Kỳ vọng đúng 216 slots cho 3 ngày, nhưng nhận được {len(canonical)}"
        # Đảm bảo không có giờ nào bị nhảy cóc
        for node_id in ["Node_A", "Node_B", "Node_C"]:
            sub = canonical[canonical["node_id"] == node_id]
            assert len(sub) == 72
            # Kiểm tra khoảng cách liên tiếp giữa các slot đúng 1 giờ
            diffs = sub["timestamp_utc"].diff().dropna()
            assert (diffs == pd.Timedelta(hours=1)).all()


def test_road_closure_handling():
    """Kiểm tra trạng thái đóng đường: road_closure=True, quality_flag='ROAD_CLOSURE'."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        raw_df = pd.DataFrame([{
            "timestamp": "2026-10-25 08:00:00",
            "node_id": "Node_A",
            "currentSpeed": 0.0,
            "freeFlowSpeed": 40.0,
            "currentTravelTime": 0.0,
            "freeFlowTravelTime": 90.0,
            "roadClosure": True,
            "confidence": 1.0,
        }])
        cleaned = clean_raw_traffic(raw_df)
        start_u = datetime(2026, 10, 25, 1, 0, 0, tzinfo=timezone.utc)
        canonical = resample_to_hourly_canonical(cleaned, reg_path, start_utc=start_u, end_utc=start_u)

        row_a = canonical[canonical["node_id"] == "Node_A"].iloc[0]
        assert row_a["road_closure"] == True
        assert row_a["quality_flag"] == QualityFlag.ROAD_CLOSURE.value
        assert row_a["missing_reason"] == MissingReason.ROAD_CLOSURE.value
        assert row_a["is_missing"] == True
        assert row_a["current_speed_kmh"] == 0.0
        assert pd.isna(row_a["current_travel_time_s"])
        assert pd.isna(row_a["congestion_ratio"])

        # Đảm bảo validator chấp thuận
        val = validate_canonical_dataframe(canonical)
        assert val["is_valid"], f"Validator fail on road closure: {val['errors']}"


def test_non_missing_finite_numbers_invariant():
    """Kiểm tra ràng buộc thép: Khi is_missing=False thì:
    1. road_closure bắt buộc = False
    2. Toàn bộ trường số học giao thông BẮT BUỘC PHẢI LÀ SỐ HỮU HẠN (finite), KHÔNG null/NaN/Inf.
    """
    row = _make_dummy_canonical_row()
    row["is_missing"] = False
    row["road_closure"] = False
    # Đảm bảo các số hữu hạn
    row["current_speed_kmh"] = 35.5
    row["current_travel_time_s"] = 85.0
    row["congestion_ratio"] = 1.05
    df = pd.DataFrame([row])

    val = validate_canonical_dataframe(df)
    assert val["is_valid"], f"Unexpected error: {val['errors']}"

    # Ca vi phạm 1: is_missing=False nhưng có NaN ở current_travel_time_s -> PHẢI BỊ CHẶN
    df_bad_tt = pd.DataFrame([dict(row, current_travel_time_s=np.nan)])
    val_bad_tt = validate_canonical_dataframe(df_bad_tt)
    assert not val_bad_tt["is_valid"]
    assert any("current_travel_time_s" in err for err in val_bad_tt["errors"])

    # Ca vi phạm 2: is_missing=False nhưng có road_closure=True -> PHẢI BỊ CHẶN
    df_bad_closure = pd.DataFrame([dict(row, road_closure=True)])
    val_bad_closure = validate_canonical_dataframe(df_bad_closure)
    assert not val_bad_closure["is_valid"]
    assert any("road_closure=True" in err for err in val_bad_closure["errors"])


def test_low_confidence_flagging():
    """Kiểm tra cờ độ tin cậy thấp khi confidence < 0.5."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        raw_df = pd.DataFrame([{
            "timestamp": "2026-10-25 08:00:00",
            "node_id": "Node_A",
            "currentSpeed": 35.0,
            "freeFlowSpeed": 40.0,
            "currentTravelTime": 90.0,
            "freeFlowTravelTime": 90.0,
            "confidence": 0.35,  # Thấp < 0.5
            "roadClosure": False,
        }])
        cleaned = clean_raw_traffic(raw_df)
        start_u = datetime(2026, 10, 25, 1, 0, 0, tzinfo=timezone.utc)
        canonical = resample_to_hourly_canonical(cleaned, reg_path, start_utc=start_u, end_utc=start_u)

        row_a = canonical[canonical["node_id"] == "Node_A"].iloc[0]
        assert row_a["quality_flag"] == QualityFlag.VALID_LOW_CONFIDENCE.value
        assert row_a["missing_reason"] == MissingReason.LOW_CONFIDENCE.value
        assert row_a["confidence"] == 0.35


def test_speed_vs_length_travel_time_consistency():
    """Kiểm tra tính nhất quán giữa tốc độ đo được và tỷ số length / travel_time."""
    # length = 1000m, travel_time = 100s -> speed tính toán = 10 m/s = 36.0 km/h
    length_m = 1000.0
    tt_s = 100.0
    v_calc = (length_m / tt_s) * 3.6  # 36.0 km/h

    v_reported_good = 35.0  # Lệch 1 km/h (~2.8% <= 20%) -> Đạt nhất quán
    rel_error_good = abs(v_reported_good - v_calc) / v_calc
    assert rel_error_good < 0.20

    v_reported_bad = 80.0  # Lệch quá 100% -> Bất thường không nhất quán
    rel_error_bad = abs(v_reported_bad - v_calc) / v_calc
    assert rel_error_bad > 0.20


def test_adjacency_matrix_mathematical_invariants():
    """Kiểm tra các bất biến toán học của ma trận kề: shape 17x17, A_ii=0, sum hàng = 1, A_ij >= 0."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = "data/config/hcm_tomtom_segments.v1.json"
        zone_csv = "data/raw/zone_labels.csv"
        osrm_csv = "data/raw/hcm_osrm_dataset.csv"

        df_nodes, adj_tensor = build_static_metadata(reg_path, zone_csv, osrm_csv, tmp_dir)
        A = adj_tensor.numpy() if hasattr(adj_tensor, "numpy") else adj_tensor

        # 1. Shape đúng 17 x 17
        assert A.shape == (17, 17)
        # 2. Không có self-loop (đường chéo bằng 0)
        assert np.allclose(np.diag(A), 0.0)
        # 3. Chuẩn hóa hàng (mỗi hàng tổng xấp xỉ 1.0)
        row_sums = A.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)
        # 4. Tất cả phần tử không âm
        assert (A >= 0.0).all()


def test_end_to_end_pipeline_determinism_and_idempotency():
    """Kiểm tra tính xác định và tái lập: chạy 2 lần độc lập cho ra hash artifacts giống hệt nhau."""
    import tempfile
    from scripts.pipeline.build_hcm_real import run_pipeline

    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        raw_csv = os.path.join(tmp_dir, "raw.csv")
        pd.DataFrame([
            {"timestamp": "2026-10-25 08:00:00", "node_id": "Node_A", "currentSpeed": 35.0, "freeFlowSpeed": 40.0, "currentTravelTime": 90.0, "freeFlowTravelTime": 90.0},
            {"timestamp": "2026-10-25 09:00:00", "node_id": "Node_A", "currentSpeed": 30.0, "freeFlowSpeed": 40.0, "currentTravelTime": 100.0, "freeFlowTravelTime": 90.0},
        ]).to_csv(raw_csv, index=False)

        out_run1 = os.path.join(tmp_dir, "run1")
        out_run2 = os.path.join(tmp_dir, "run2")

        res1 = run_pipeline(raw_path=raw_csv, registry_path=reg_path, out_dir=out_run1, zone_path="", osrm_path="")
        res2 = run_pipeline(raw_path=raw_csv, registry_path=reg_path, out_dir=out_run2, zone_path="", osrm_path="")

        # So sánh DataFrame canonical
        pd.testing.assert_frame_equal(res1["df_canonical"], res2["df_canonical"])

        # So sánh hash file CSV
        h1 = compute_file_sha256(os.path.join(out_run1, "traffic_canonical.csv"))
        h2 = compute_file_sha256(os.path.join(out_run2, "traffic_canonical.csv"))
        assert h1 == h2, f"Pipeline phải hoàn toàn deterministic: {h1} vs {h2}"


def test_strict_dtype_validation():
    """Kiểm tra validator bắt lỗi nghiêm ngặt khi kiểu dữ liệu không hợp lệ (ví dụ: float bị gán string)."""
    row = _make_dummy_canonical_row()
    df_good = pd.DataFrame([row])
    val_good = validate_canonical_dataframe(df_good)
    assert val_good["is_valid"], f"Row chuẩn phải pass: {val_good['errors']}"

    # Đưa cột số học thành kiểu string (object)
    df_bad = df_good.copy()
    df_bad["current_speed_kmh"] = "45.5"  # chuỗi thay vì float
    val_bad = validate_canonical_dataframe(df_bad)
    assert not val_bad["is_valid"], "Validator phải báo lỗi khi current_speed_kmh là chuỗi"
    assert any("current_speed_kmh" in err for err in val_bad["errors"])

    # Đưa cột boolean thành chuỗi
    df_bad_bool = df_good.copy()
    df_bad_bool["is_missing"] = "False"  # chuỗi thay vì bool
    val_bad_bool = validate_canonical_dataframe(df_bad_bool)
    assert not val_bad_bool["is_valid"], "Validator phải báo lỗi khi is_missing là chuỗi"
    assert any("is_missing" in err for err in val_bad_bool["errors"])

    # Kiểm tra cross-field: road_closure=True nhưng current_speed_kmh > 0
    df_bad_closure = df_good.copy()
    df_bad_closure["road_closure"] = True
    df_bad_closure["current_speed_kmh"] = 40.0
    val_closure = validate_canonical_dataframe(df_bad_closure)
    assert not val_closure["is_valid"], "Validator phải báo lỗi khi road_closure=True mà tốc độ > 0"
    assert any("road_closure=True nhưng current_speed_kmh > 0" in err for err in val_closure["errors"])


def test_api_error_vs_not_collected():
    """Kiểm tra phân biệt rõ ràng giữa API_ERROR (gọi API thất bại) và NOT_COLLECTED (không gọi)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        raw_data = [
            {
                "timestamp": "2026-10-25 08:00:00",
                "node_id": "Node_A",
                "http_status": 500,
                "error_message": "Internal Server Error / Gateway Timeout",
            },
            {
                "timestamp": "2026-10-25 10:00:00",
                "node_id": "Node_A",
                "currentSpeed": 35.0,
                "freeFlowSpeed": 40.0,
                "currentTravelTime": 90.0,
                "freeFlowTravelTime": 90.0,
                "http_status": 200,
            }
        ]
        df_raw = pd.DataFrame(raw_data)
        df_clean = clean_raw_traffic(df_raw)
        df_canonical = resample_to_hourly_canonical(
            df_cleaned=df_clean,
            registry_path=reg_path,
            start_utc=pd.Timestamp("2026-10-25 01:00:00+00:00"),  # 08:00 local
            end_utc=pd.Timestamp("2026-10-25 03:00:00+00:00"),    # 10:00 local
        )

        df_node_a = df_canonical[df_canonical["node_id"] == "Node_A"].sort_values("timestamp_local")
        
        # Giờ 08:00 (có request nhưng lỗi HTTP 500) -> API_ERROR
        row_08 = df_node_a.iloc[0]
        assert row_08["is_missing"] == True
        assert row_08["missing_reason"] == MissingReason.API_ERROR.value
        assert row_08["quality_flag"] == QualityFlag.MISSING_API_ERROR.value

        # Giờ 09:00 (không có request nào) -> NOT_COLLECTED
        row_09 = df_node_a.iloc[1]
        assert row_09["is_missing"] == True
        assert row_09["missing_reason"] == MissingReason.NOT_COLLECTED.value
        assert row_09["quality_flag"] == QualityFlag.MISSING_NOT_COLLECTED.value

        # Giờ 10:00 (thành công) -> VALID
        row_10 = df_node_a.iloc[2]
        assert row_10["is_missing"] == False
        assert row_10["missing_reason"] == MissingReason.NONE.value
        assert row_10["quality_flag"] == QualityFlag.VALID.value


def test_smart_resampling_highest_confidence():
    """Kiểm tra thuật toán Resampling: khi có nhiều bản ghi trong cùng 1 giờ, chọn bản có confidence cao nhất."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        raw_data = [
            {
                "timestamp": "2026-10-25 08:05:00",
                "node_id": "Node_A",
                "currentSpeed": 20.0,
                "freeFlowSpeed": 40.0,
                "currentTravelTime": 150.0,
                "freeFlowTravelTime": 90.0,
                "confidence": 0.4,
            },
            {
                "timestamp": "2026-10-25 08:25:00",
                "node_id": "Node_A",
                "currentSpeed": 45.0,
                "freeFlowSpeed": 40.0,
                "currentTravelTime": 80.0,
                "freeFlowTravelTime": 90.0,
                "confidence": 0.95,
            },
        ]
        df_raw = pd.DataFrame(raw_data)
        df_clean = clean_raw_traffic(df_raw)
        df_canonical = resample_to_hourly_canonical(
            df_cleaned=df_clean,
            registry_path=reg_path,
            start_utc=pd.Timestamp("2026-10-25 01:00:00+00:00"),
            end_utc=pd.Timestamp("2026-10-25 01:00:00+00:00"),
        )
        row = df_canonical[df_canonical["node_id"] == "Node_A"].iloc[0]
        # Bắt buộc phải chọn bản ghi có confidence = 0.95 và speed = 45.0
        assert row["confidence"] == 0.95, f"Kỳ vọng chọn bản confidence cao nhất (0.95), nhưng nhận được {row['confidence']}"
        assert row["current_speed_kmh"] == 45.0


def test_manifest_source_hash_integrity():
    """Kiểm tra manifest.json ghi mã băm của source thô đầu vào và phát hiện khi source bị biến đổi."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        raw_path = os.path.join(tmp_dir, "raw.csv")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write("timestamp,node_id,currentSpeed\n2026-10-25 08:00:00,Node_A,35.0\n")

        row = _make_dummy_canonical_row()
        df_canonical = pd.DataFrame([row])

        # Sinh manifest lần 1
        m1 = generate_manifest(
            out_dir=tmp_dir,
            config_path=reg_path,
            df_canonical=df_canonical,
            source_files={"raw_input": raw_path},
        )
        assert "source_hashes" in m1
        assert "raw_input" in m1["source_hashes"]
        h1 = m1["source_hashes"]["raw_input"]

        # Sửa 1 byte trong file raw
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write("timestamp,node_id,currentSpeed\n2026-10-25 08:00:00,Node_A,36.0\n")

        # Sinh manifest lần 2
        m2 = generate_manifest(
            out_dir=tmp_dir,
            config_path=reg_path,
            df_canonical=df_canonical,
            source_files={"raw_input": raw_path},
        )
        h2 = m2["source_hashes"]["raw_input"]
        assert h1 != h2, f"Mã băm source phải thay đổi khi nội dung raw thay đổi: {h1} vs {h2}"


def test_load_hcm_real_canonical_loader():
    """Kiểm tra load_hcm_real_canonical đọc CSV tự động khôi phục đúng dtype (không mất kiểu)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        row = _make_dummy_canonical_row()
        df_orig = pd.DataFrame([row])
        csv_path = os.path.join(tmp_dir, "traffic_canonical.csv")
        df_orig.to_csv(csv_path, index=False)

        # Đọc bằng load_hcm_real_canonical từ file
        df_loaded = load_hcm_real_canonical(csv_path)
        assert pd.api.types.is_datetime64_any_dtype(df_loaded["timestamp_utc"])
        assert pd.api.types.is_datetime64_any_dtype(df_loaded["timestamp_local"])
        assert pd.api.types.is_float_dtype(df_loaded["current_speed_kmh"])
        assert pd.api.types.is_bool_dtype(df_loaded["road_closure"])
        assert pd.api.types.is_bool_dtype(df_loaded["is_missing"])

        # Validator chạy trên df_loaded phải PASS 100%
        val = validate_canonical_dataframe(df_loaded)
        assert val["is_valid"], f"Loaded DataFrame phải pass validator: {val['errors']}"

        # Đọc từ thư mục chứa file
        df_from_dir = load_hcm_real_canonical(tmp_dir)
        assert len(df_from_dir) == 1


def test_traffic_cleaned_preserves_duplicates():
    """Kiểm tra bảng trung gian traffic_cleaned.csv lưu toàn vẹn các bản ghi trùng lặp."""
    import tempfile
    from scripts.pipeline.build_hcm_real import run_pipeline

    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        raw_csv = os.path.join(tmp_dir, "raw_dup.csv")
        raw_data = [
            {"timestamp": "2026-10-25 08:00:00", "node_id": "Node_A", "currentSpeed": 35.0, "freeFlowSpeed": 40.0, "currentTravelTime": 90.0, "freeFlowTravelTime": 90.0},
            # Bản ghi trùng lặp cùng timestamp
            {"timestamp": "2026-10-25 08:00:00", "node_id": "Node_A", "currentSpeed": 36.0, "freeFlowSpeed": 40.0, "currentTravelTime": 88.0, "freeFlowTravelTime": 90.0},
        ]
        pd.DataFrame(raw_data).to_csv(raw_csv, index=False)
        out_build = os.path.join(tmp_dir, "out")

        res = run_pipeline(raw_path=raw_csv, registry_path=reg_path, out_dir=out_build, zone_path="", osrm_path="")

        # traffic_cleaned.csv phải tồn tại và có 2 dòng
        cleaned_file = os.path.join(out_build, "traffic_cleaned.csv")
        assert os.path.exists(cleaned_file), "Phải xuất file trung gian traffic_cleaned.csv"
        df_cleaned = pd.read_csv(cleaned_file)
        assert len(df_cleaned) == 2, "traffic_cleaned phải giữ nguyên cả 2 bản ghi"
        assert (df_cleaned["quality_flag"] == "DUPLICATE_FLAGGED").sum() == 1, "Phải gắn cờ DUPLICATE_FLAGGED cho 1 bản"

        # traffic_canonical.csv chỉ có 1 slot đại diện cho giờ 08:00
        df_canon = pd.read_csv(os.path.join(out_build, "traffic_canonical.csv"))
        assert len(df_canon[df_canon["node_id"] == "Node_A"]) == 1


def test_completeness_partial_day_no_bias():
    """Kiểm tra loại bỏ Partial Day Bias: ngày thu bắt đầu lúc 17:00 (7 slots) đạt 100% completeness."""
    import tempfile
    from benchmark.hcm_real_quality import HCMRealQualityEvaluator

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Giả lập dataset 1 trạm chỉ thu từ 17:00 đến 23:00 (7 giờ)
        rows = []
        for h in range(17, 24):
            r = _make_dummy_canonical_row()
            r["timestamp_local"] = pd.Timestamp(f"2026-10-25 {h:02d}:00:00+07:00")
            r["timestamp_utc"] = r["timestamp_local"].astimezone(timezone.utc)
            r["is_missing"] = False
            rows.append(r)
        
        canon_csv = os.path.join(tmp_dir, "traffic_canonical.csv")
        pd.DataFrame(rows).to_csv(canon_csv, index=False)

        evaluator = HCMRealQualityEvaluator(canon_csv)
        matrix = evaluator.evaluate_node_day_matrix()
        day_eval = matrix["Node_A__2026-10-25"]

        # Mẫu số phải là 7 slots được lập lịch -> completeness = 7/7 = 1.0 (thay vì 7/24 = 0.29)
        assert day_eval["completeness"] == 1.0, f"Completeness ngày không trọn vẹn phải là 1.0, nhận được: {day_eval['completeness']}"


def test_immutability_and_overwrite_guard():
    """Kiểm tra cơ chế Immutability Guard từ chối ghi đè khi force_overwrite=False và hỗ trợ --immutable."""
    import tempfile
    from scripts.pipeline.build_hcm_real import run_pipeline

    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        raw_csv = os.path.join(tmp_dir, "raw.csv")
        pd.DataFrame([
            {"timestamp": "2026-10-25 08:00:00", "node_id": "Node_A", "currentSpeed": 35.0, "freeFlowSpeed": 40.0, "currentTravelTime": 90.0, "freeFlowTravelTime": 90.0},
        ]).to_csv(raw_csv, index=False)
        out_dir = os.path.join(tmp_dir, "dataset_v1")

        # Chạy lần 1 thành công
        run_pipeline(raw_path=raw_csv, registry_path=reg_path, out_dir=out_dir, zone_path="", osrm_path="")

        # Chạy lần 2 với force_overwrite=False -> Bắt buộc phải ném FileExistsError
        raised = False
        try:
            run_pipeline(raw_path=raw_csv, registry_path=reg_path, out_dir=out_dir, zone_path="", osrm_path="", force_overwrite=False)
        except FileExistsError:
            raised = True
        assert raised, "Phải ném FileExistsError bảo vệ tính bất biến khi thư mục đã có manifest"

        # Chạy với immutable=True -> Tạo thư mục mới có timestamp
        res_imm = run_pipeline(raw_path=raw_csv, registry_path=reg_path, out_dir=out_dir, zone_path="", osrm_path="", immutable=True)
        assert res_imm["manifest"]["total_records"] > 0


def test_composite_primary_key_uniqueness():
    """Kiểm tra validator bắt lỗi vi phạm ràng buộc khóa chính duy nhất (node_id + timestamp_utc)."""
    r1 = _make_dummy_canonical_row()
    r2 = _make_dummy_canonical_row()  # Cùng node_id và timestamp_utc
    df_dup = pd.DataFrame([r1, r2])
    val = validate_canonical_dataframe(df_dup)
    assert not val["is_valid"], "Validator phải từ chối khi có 2 dòng trùng cặp (node_id, timestamp_utc)"
    assert any("Composite Primary Key Constraint" in err for err in val["errors"])


def test_load_hcm_real_canonical_integrity_verification():
    """Kiểm tra load_hcm_real_canonical tự động bắt lỗi khi phát hiện file bị sửa đổi trái phép."""
    import tempfile
    from scripts.pipeline.build_hcm_real import run_pipeline

    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        raw_csv = os.path.join(tmp_dir, "raw.csv")
        pd.DataFrame([
            {"timestamp": "2026-10-25 08:00:00", "node_id": "Node_A", "currentSpeed": 35.0, "freeFlowSpeed": 40.0, "currentTravelTime": 90.0, "freeFlowTravelTime": 90.0},
        ]).to_csv(raw_csv, index=False)
        out_dir = os.path.join(tmp_dir, "out")
        run_pipeline(raw_path=raw_csv, registry_path=reg_path, out_dir=out_dir, zone_path="", osrm_path="")

        # Nạp với verify_manifest=True khi file nguyên vẹn -> PASS
        df_ok = load_hcm_real_canonical(out_dir, verify_manifest=True)
        assert len(df_ok) > 0

        # Cố tình sửa đổi 1 ký tự trong traffic_canonical.csv
        canon_file = os.path.join(out_dir, "traffic_canonical.csv")
        with open(canon_file, "a", encoding="utf-8") as f:
            f.write("# unauthorized modification\n")

        # Nạp trực tiếp file CSV đã bị sửa với verify_manifest=True -> Bắt buộc phải quăng ValueError
        tampered = False
        try:
            load_hcm_real_canonical(canon_file, verify_manifest=True)
        except ValueError as e:
            if "DatasetIntegrityError" in str(e):
                tampered = True
        assert tampered, "load_hcm_real_canonical phải phát hiện file bị sửa và ném DatasetIntegrityError"


def test_flatline_stagnation_detection():
    """Kiểm tra Quality Engine phát hiện dữ liệu đông cứng (flatline) khi tốc độ không đổi >= 4 giờ."""
    import tempfile
    from benchmark.hcm_real_quality import HCMRealQualityEvaluator

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Giả lập 6 giờ liên tiếp cùng 1 giá trị vận tốc 35.0 km/h
        rows = []
        for h in range(8, 14):
            r = _make_dummy_canonical_row()
            r["timestamp_local"] = pd.Timestamp(f"2026-10-25 {h:02d}:00:00+07:00")
            r["timestamp_utc"] = r["timestamp_local"].astimezone(timezone.utc)
            r["current_speed_kmh"] = 35.0
            r["is_missing"] = False
            rows.append(r)
        
        canon_csv = os.path.join(tmp_dir, "traffic_canonical.csv")
        pd.DataFrame(rows).to_csv(canon_csv, index=False)

        evaluator = HCMRealQualityEvaluator(canon_csv)
        matrix = evaluator.evaluate_node_day_matrix()
        day_eval = matrix["Node_A__2026-10-25"]
        assert day_eval["max_flatline_hours"] >= 4, f"Phải phát hiện chuỗi flatline >= 4 giờ, nhận được: {day_eval['max_flatline_hours']}"


def test_manifest_environment_fingerprint():
    """Kiểm tra manifest.json ghi lại thông tin môi trường hệ thống phục vụ tái lập khoa học."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = _make_dummy_registry(tmp_dir)
        row = _make_dummy_canonical_row()
        df_canon = pd.DataFrame([row])
        m = generate_manifest(out_dir=tmp_dir, config_path=reg_path, df_canonical=df_canon)
        assert "environment_fingerprint" in m
        env = m["environment_fingerprint"]
        assert "python_version" in env
        assert "pandas_version" in env
        assert "os_platform" in env


def test_distance_matrix_metric_space_and_disconnected_inf():
    """Kiểm tra tính chuẩn tắc toán học của ma trận khoảng cách:
    1. d(i, i) = 0.0 (phản xạ)
    2. d(i, j) > 0.0 với mọi i != j (tính phân biệt - Identity of Indiscernibles)
    3. Cặp không có đường đi OSRM phải là np.inf, TUYỆT ĐỐI KHÔNG BẰNG 0.0!
    4. Khi không có OSRM, fallback Haversine phải > 0.0, không bằng 0.0!
    """
    import tempfile
    from scripts.pipeline.enrich_static_metadata import build_static_metadata

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Tạo dummy registry 3 trạm với tọa độ GPS rõ ràng
        reg_file = os.path.join(tmp_dir, "dummy_reg.json")
        with open(reg_file, "w", encoding="utf-8") as f:
            json.dump({
                "segments": [
                    {"node_id": "Node_A", "segment_id": "seg_a", "approved_openlr": "OPENLR_A", "lat": 10.7769, "lon": 106.7009, "corridor_name": "C1", "selection_reason": "R", "status": "active"},
                    {"node_id": "Node_B", "segment_id": "seg_b", "approved_openlr": "OPENLR_B", "lat": 10.7900, "lon": 106.7100, "corridor_name": "C2", "selection_reason": "R", "status": "active"},
                    {"node_id": "Node_C", "segment_id": "seg_c", "approved_openlr": "OPENLR_C", "lat": 10.8200, "lon": 106.7500, "corridor_name": "C3", "selection_reason": "R", "status": "active"},
                ]
            }, f)

        # 1. Trường hợp có OSRM nhưng CHỈ CÓ đường đi giữa (A, B); KHÔNG CÓ đường đi đến C
        osrm_partial = os.path.join(tmp_dir, "osrm_partial.csv")
        pd.DataFrame([
            {"origin": "Node_A", "destination": "Node_B", "distance_m": 2500.0, "duration_s": 300.0},
            {"origin": "Node_B", "destination": "Node_A", "distance_m": 2500.0, "duration_s": 300.0},
        ]).to_csv(osrm_partial, index=False)

        out_run1 = os.path.join(tmp_dir, "run1")
        build_static_metadata(reg_file, "", osrm_partial, out_run1)
        dist_df1 = pd.read_csv(os.path.join(out_run1, "osrm_distance_matrix.csv"), index_col=0)

        # Kiểm tra tính phân biệt
        assert dist_df1.loc["Node_A", "Node_A"] == 0.0
        assert dist_df1.loc["Node_B", "Node_B"] == 0.0
        assert dist_df1.loc["Node_C", "Node_C"] == 0.0
        assert dist_df1.loc["Node_A", "Node_B"] == 2500.0

        # CỐT TỬ: Cặp (Node_A, Node_C) không có kết nối OSRM PHẢI là np.inf, KHÔNG ĐƯỢC BẰNG 0.0!
        assert np.isinf(dist_df1.loc["Node_A", "Node_C"])
        assert np.isinf(dist_df1.loc["Node_C", "Node_A"])
        assert dist_df1.loc["Node_A", "Node_C"] > 0.0

        # 2. Trường hợp KHÔNG CÓ file OSRM (missing): Fallback Haversine phải > 0.0 cho mọi i != j
        out_run2 = os.path.join(tmp_dir, "run2")
        build_static_metadata(reg_file, "", "", out_run2)
        dist_df2 = pd.read_csv(os.path.join(out_run2, "osrm_distance_matrix.csv"), index_col=0)

        for i in ["Node_A", "Node_B", "Node_C"]:
            for j in ["Node_A", "Node_B", "Node_C"]:
                if i == j:
                    assert dist_df2.loc[i, j] == 0.0
                else:
                    assert dist_df2.loc[i, j] > 1000.0  # Khoảng cách địa lý thực tế giữa các trạm > 1km
                    assert not np.isinf(dist_df2.loc[i, j])


# ── Standalone Test Runner ───────────────────────────────────────────────────

ALL_TESTS = [
    test_canonical_schema_20_columns,
    test_mandatory_non_null_fields,
    test_timestamp_timezone_conversion,
    test_value_ranges,
    test_congestion_ratio_formula,
    test_free_flow_zero_handling,
    test_faster_than_free_flow_flag,
    test_missing_flags_and_reasons,
    test_duplicate_flagging,
    test_time_label_matching_build_graph,
    test_static_metadata_no_collision,
    test_manifest_reproducibility,
    test_multiday_hourly_resampling_grid_continuity,
    test_road_closure_handling,
    test_low_confidence_flagging,
    test_speed_vs_length_travel_time_consistency,
    test_adjacency_matrix_mathematical_invariants,
    test_end_to_end_pipeline_determinism_and_idempotency,
    test_strict_dtype_validation,
    test_api_error_vs_not_collected,
    test_smart_resampling_highest_confidence,
    test_manifest_source_hash_integrity,
    test_load_hcm_real_canonical_loader,
    test_traffic_cleaned_preserves_duplicates,
    test_completeness_partial_day_no_bias,
    test_immutability_and_overwrite_guard,
    test_composite_primary_key_uniqueness,
    test_load_hcm_real_canonical_integrity_verification,
    test_flatline_stagnation_detection,
    test_manifest_environment_fingerprint,
    test_distance_matrix_metric_space_and_disconnected_inf,
    test_non_missing_finite_numbers_invariant,
]


def run_all_tests():
    print("=" * 65)
    print("  Chạy Unit Test Suite HCM-Real Pipeline (Thành viên B)")
    print("=" * 65)
    passed = 0
    failed = 0

    for test_fn in ALL_TESTS:
        test_name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {test_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test_name} — {e}")
            failed += 1

    print("=" * 65)
    print(f"  KẾT QUẢ: {passed}/{len(ALL_TESTS)} PASS")
    if failed == 0:
        print("  🎉 TẤT CẢ UNIT TESTS ĐÃ VƯỢT QUA 100%!")
    else:
        print(f"  ⚠️ CÓ {failed} TESTS THẤT BẠI!")
    print("=" * 65)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
