"""
tests/test_data_pipeline.py
============================
Test cho phần 4.1 (Bảo) — Data simulator & dataset provenance, theo
DoD trong kế hoạch tuần Non-IID benchmark:

  - hash deterministic theo seed
  - đủ 4 time labels
  - label khớp hour
  - đúng công thức S = T - T_in - T_out + 1
  - metadata đánh dấu synthetic

Test không phụ thuộc dữ liệu thật trong data/raw/: các test đơn vị tự
dựng DataFrame nhỏ (tiny_edges_and_zones fixture). Chỉ 1 test tích hợp
end-to-end (test_end_to_end_pipeline_creates_manifest_and_all_labels)
cần data/raw/hcm_osrm_dataset.csv + zone_labels.csv thật — test này tự
skip nếu không có, để clone sạch / CI fixture nhỏ vẫn chạy được các
test còn lại không cần raw private.
"""

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module(name: str, rel_path: str):
    """Load module theo đường dẫn file, không cần scripts/ là package
    (không cần __init__.py) — tránh phụ thuộc cấu trúc repo cụ thể."""
    full_path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_graph = _load_module("hcmsim_build_graph", "scripts/build_graph.py")
gen_traffic = _load_module(
    "hcmsim_gen_traffic", "scripts/dev/generate_synthetic_traffic.py"
)


# ══════════════════════════════════════════════
# Fixtures — dữ liệu nhỏ, tự tạo, KHÔNG cần raw thật
# ══════════════════════════════════════════════
@pytest.fixture
def tiny_edges_and_zones():
    edges_base = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "destination": ["B", "C", "A"],
            "distance_m": [1000.0, 1500.0, 1200.0],
            "duration_s": [120.0, 180.0, 150.0],
        }
    )
    df_zone = pd.DataFrame(
        {
            "commercial": [1, 0, 0],
            "residential": [0, 1, 0],
            "industrial": [0, 0, 1],
            "school": [0, 0, 0],
            "university": [0, 0, 0],
            "hospital": [0, 0, 0],
            "transport": [0, 0, 0],
            "park": [0, 0, 0],
        },
        index=["A", "B", "C"],
    )
    df_zone.index.name = "node"
    return edges_base, df_zone


# ══════════════════════════════════════════════
# 1) Hash / kết quả deterministic theo seed
# ══════════════════════════════════════════════
def test_generator_deterministic_by_seed(tiny_edges_and_zones):
    edges_base, df_zone = tiny_edges_and_zones

    df1 = gen_traffic.generate_traffic_dataframe(
        seed=123, days=2, edges_base=edges_base, df_zone=df_zone
    )
    df2 = gen_traffic.generate_traffic_dataframe(
        seed=123, days=2, edges_base=edges_base, df_zone=df_zone
    )
    pd.testing.assert_frame_equal(df1, df2)

    df3 = gen_traffic.generate_traffic_dataframe(
        seed=999, days=2, edges_base=edges_base, df_zone=df_zone
    )
    assert not df1["congestion_ratio"].equals(
        df3["congestion_ratio"]
    ), "Seed khac nhau nhung cho ra cung mot ket qua — nhieu dang bi seed sai cach"


def test_generator_file_hash_deterministic_by_seed(tmp_path, tiny_edges_and_zones):
    """Cung seed -> hash file CSV giong het nhau (quan trong vi manifest.json
    dung hash file nay de provenance)."""
    edges_base, df_zone = tiny_edges_and_zones

    out1 = tmp_path / "run1.csv"
    out2 = tmp_path / "run2.csv"
    out3 = tmp_path / "run3.csv"

    gen_traffic.run_and_save(
        str(out1), seed=7, days=1, edges_base=edges_base, df_zone=df_zone, verbose=False
    )
    gen_traffic.run_and_save(
        str(out2), seed=7, days=1, edges_base=edges_base, df_zone=df_zone, verbose=False
    )
    gen_traffic.run_and_save(
        str(out3), seed=8, days=1, edges_base=edges_base, df_zone=df_zone, verbose=False
    )

    h1 = build_graph.compute_file_hash(str(out1))
    h2 = build_graph.compute_file_hash(str(out2))
    h3 = build_graph.compute_file_hash(str(out3))

    assert h1 == h2, "Cung seed phai cho ra file CSV giong het nhau (hash khop)"
    assert h1 != h3, "Khac seed phai cho ra file khac nhau"


def test_compute_file_hash_changes_with_content(tmp_path):
    f = tmp_path / "dummy.csv"
    f.write_text("a,b\n1,2\n")
    h_before = build_graph.compute_file_hash(str(f))
    h_again = build_graph.compute_file_hash(str(f))
    assert h_before == h_again

    f.write_text("a,b\n1,3\n")
    h_after = build_graph.compute_file_hash(str(f))
    assert h_before != h_after


# ══════════════════════════════════════════════
# 2) Đủ 4 time labels
# ══════════════════════════════════════════════
def test_generator_produces_all_four_time_labels(tiny_edges_and_zones):
    edges_base, df_zone = tiny_edges_and_zones
    # 7 ngay du de cover het cac khung gio trong ngay (dem, sang, chieu, binh thuong)
    df = gen_traffic.generate_traffic_dataframe(
        seed=42, days=7, edges_base=edges_base, df_zone=df_zone
    )
    labels = set(df["time_label"].unique())
    assert labels == {
        "night",
        "rush_morning",
        "rush_evening",
        "normal",
    }, f"Thieu nhan thoi gian, chi thay: {labels}"


def test_time_label_of_covers_all_four_indices_over_24h():
    indices = {build_graph.time_label_of(h) for h in range(24)}
    assert indices == {0, 1, 2, 3}


# ══════════════════════════════════════════════
# 3) Label khớp hour (giữa generator và build_graph, và trong chính generator)
# ══════════════════════════════════════════════
def test_time_label_matches_between_modules_for_every_hour():
    """get_time_label() (generator, tra ve string) va time_label_of()
    (build_graph, tra ve int) PHAI khop 100% tren toan bo 24 gio."""
    label_to_idx = {"night": 0, "rush_morning": 1, "rush_evening": 2, "normal": 3}
    for hour in range(24):
        gen_label = gen_traffic.get_time_label(hour)
        graph_idx = build_graph.time_label_of(hour)
        assert label_to_idx[gen_label] == graph_idx, (
            f"Lech nhan tai hour={hour}: generator='{gen_label}' "
            f"(idx ky vong={label_to_idx[gen_label]}) vs build_graph idx={graph_idx}"
        )


def test_generated_rows_time_label_matches_timestamp_hour(tiny_edges_and_zones):
    edges_base, df_zone = tiny_edges_and_zones
    df = gen_traffic.generate_traffic_dataframe(
        seed=1, days=1, edges_base=edges_base, df_zone=df_zone
    )
    hours = pd.to_datetime(df["timestamp"]).dt.hour
    expected = hours.apply(gen_traffic.get_time_label)
    assert (df["time_label"] == expected).all()


# ══════════════════════════════════════════════
# 4) Công thức S = T - T_in - T_out + 1
# ══════════════════════════════════════════════
@pytest.mark.parametrize("T,t_in,t_out", [(50, 12, 3), (30, 5, 2), (100, 24, 6)])
def test_sliding_window_sample_count_formula(T, t_in, t_out):
    N, F = 5, 4
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, N, F))
    hours = [h % 24 for h in range(T)]
    times = [build_graph.time_label_of(h) for h in hours]
    dows = [0] * T

    X_w, Y_w, T_w, H_w, D_w, Sinc_w = build_graph.create_samples(
        X, times, hours, dows, t_in, t_out
    )

    expected_S = T - t_in - t_out + 1
    assert X_w.shape[0] == expected_S
    assert Y_w.shape[0] == expected_S
    assert T_w.shape[0] == expected_S
    assert H_w.shape[0] == expected_S
    assert D_w.shape[0] == expected_S
    assert Sinc_w.shape[0] == expected_S

    assert X_w.shape[1] == N
    assert X_w.shape[2] == t_in * F
    assert Y_w.shape[2] == t_out


# ══════════════════════════════════════════════
# 5) Metadata đánh dấu synthetic
# ══════════════════════════════════════════════
def test_generator_metadata_marks_synthetic(tmp_path, tiny_edges_and_zones):
    edges_base, df_zone = tiny_edges_and_zones
    out_csv = tmp_path / "hcm_sim_traffic.csv"

    gen_traffic.run_and_save(
        out_path=str(out_csv),
        seed=7,
        days=1,
        edges_base=edges_base,
        df_zone=df_zone,
        verbose=False,
    )

    meta_path = tmp_path / "hcm_sim_traffic.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())

    assert meta["data_source"] == "synthetic"
    assert meta["seed"] == 7
    assert meta["num_days"] == 1
    assert "simulator" in meta
    assert "start_time" in meta
    assert "interval_min" in meta


def test_build_graph_loads_traffic_meta(tmp_path, tiny_edges_and_zones):
    edges_base, df_zone = tiny_edges_and_zones
    out_csv = tmp_path / "hcm_sim_traffic.csv"
    gen_traffic.run_and_save(
        out_path=str(out_csv),
        seed=3,
        days=1,
        edges_base=edges_base,
        df_zone=df_zone,
        verbose=False,
    )
    meta = build_graph.load_traffic_meta(str(out_csv))
    assert meta is not None
    assert meta["data_source"] == "synthetic"

    # File khong ton tai (vd traffic that, khong co .meta.json) -> None
    assert build_graph.load_traffic_meta(str(tmp_path / "khong_ton_tai.csv")) is None


def test_get_git_commit_hash_never_raises():
    h = build_graph.get_git_commit_hash()
    assert isinstance(h, str) and len(h) > 0


# ══════════════════════════════════════════════
# 6) Tích hợp end-to-end (chỉ chạy nếu có raw data thật)
# ══════════════════════════════════════════════
RAW_OSRM = os.path.join(ROOT, "data/raw/hcm_osrm_dataset.csv")
RAW_ZONE = os.path.join(ROOT, "data/raw/zone_labels.csv")


@pytest.mark.skipif(
    not (os.path.exists(RAW_OSRM) and os.path.exists(RAW_ZONE)),
    reason="Can data/raw/hcm_osrm_dataset.csv + zone_labels.csv that de chay end-to-end",
)
def test_end_to_end_pipeline_creates_manifest_and_all_labels(tmp_path):
    import torch

    traffic_csv = tmp_path / "hcm_sim_traffic.csv"
    out_dir = tmp_path / "hcm_sim_v1"

    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts/dev/generate_synthetic_traffic.py"),
            "--seed",
            "7",
            "--days",
            "3",
            "--out",
            str(traffic_csv),
        ],
        check=True,
        cwd=ROOT,
    )
    assert traffic_csv.exists()
    assert (tmp_path / "hcm_sim_traffic.meta.json").exists()

    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts/build_graph.py"),
            "--traffic-path",
            str(traffic_csv),
            "--out-dir",
            str(out_dir),
            "--t_in",
            "4",
            "--t_out",
            "2",
        ],
        check=True,
        cwd=ROOT,
    )

    dataset_path = out_dir / "graph_dataset.pt"
    manifest_path = out_dir / "manifest.json"
    meta_path = out_dir / "meta.json"
    assert dataset_path.exists()
    assert manifest_path.exists()
    assert meta_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["data_source"] == "synthetic"
    assert manifest["t_in"] == 4
    assert manifest["t_out"] == 2
    assert "git_commit" in manifest
    assert manifest["traffic_sha256"] is not None

    meta = json.loads(meta_path.read_text())
    assert meta["data_source"] == "synthetic"

    ds = torch.load(dataset_path, weights_only=False)
    labels_present = set(ds["time_labels"].tolist())
    assert labels_present == {
        0,
        1,
        2,
        3,
    }, f"Thieu label trong dataset: {labels_present}"
