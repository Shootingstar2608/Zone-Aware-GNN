"""
scripts/pipeline/manifest.py
============================
Tầng 4: Phiên bản hóa Dataset & Manifest Builder (Công việc 5).
Manifest ghi nhận đầy đủ 6 thông tin bắt buộc:
  - config hash
  - thời gian thu
  - số record
  - coverage
  - schema version
  - hash của artifact
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd


def compute_file_sha256(filepath: str) -> str:
    """Tính mã băm SHA-256 của file."""
    if not os.path.exists(filepath):
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash() -> str:
    """Lấy commit SHA hiện tại của repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def generate_manifest(
    out_dir: str,
    config_path: str,
    df_canonical: pd.DataFrame,
    dataset_version: str = "hcm_real_v1.0",
    schema_version: str = "1.0.0",
    source_files: dict[str, str] | None = None,
    df_cleaned: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Tạo và ghi file manifest.json cho dataset HCM-Real.
    """
    total_records = len(df_canonical)
    if total_records > 0:
        min_ts = df_canonical["timestamp_utc"].min()
        max_ts = df_canonical["timestamp_utc"].max()
        start_utc_str = min_ts.isoformat() if hasattr(min_ts, "isoformat") else str(min_ts)
        end_utc_str = max_ts.isoformat() if hasattr(max_ts, "isoformat") else str(max_ts)
        valid_records = int((df_canonical["is_missing"] == False).sum())
        valid_ratio = round(valid_records / total_records, 4)
        missing_ratio = round(1.0 - valid_ratio, 4)

        # Coverage theo node
        by_node_coverage = {}
        for node_id, group in df_canonical.groupby("node_id"):
            node_total = len(group)
            node_valid = int((group["is_missing"] == False).sum())
            by_node_coverage[node_id] = round(node_valid / max(node_total, 1), 4)

        # Thống kê phân loại cờ
        qf_counts = df_canonical["quality_flag"].value_counts().to_dict()
        mr_counts = df_canonical["missing_reason"].value_counts().to_dict()
    else:
        start_utc_str = None
        end_utc_str = None
        valid_ratio = 0.0
        missing_ratio = 0.0
        by_node_coverage = {}
        qf_counts = {}
        mr_counts = {}

    dup_count = 0
    cleaned_total = 0
    if df_cleaned is not None and not df_cleaned.empty:
        cleaned_total = len(df_cleaned)
        if "quality_flag" in df_cleaned.columns:
            dup_count = int((df_cleaned["quality_flag"] == "DUPLICATE_FLAGGED").sum())
        elif "is_duplicate" in df_cleaned.columns:
            dup_count = int((df_cleaned["is_duplicate"] == True).sum())

    # Danh sách các artifact cần băm
    target_artifacts = [
        "traffic_cleaned.csv",
        "traffic_cleaned.parquet",
        "traffic_canonical.csv",
        "traffic_canonical.parquet",
        "nodes.csv",
        "nodes.parquet",
        "osrm_distance_matrix.csv",
        "adjacency.pt",
    ]
    artifact_hashes = {}
    for fname in target_artifacts:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            artifact_hashes[fname] = f"sha256:{compute_file_sha256(fpath)}"

    # Tính mã băm các file raw nguồn để đảm bảo truy vết 2 chiều (Provenance)
    source_hashes = {}
    if source_files:
        for src_name, src_path in source_files.items():
            if src_path and os.path.exists(src_path):
                source_hashes[src_name] = f"sha256:{compute_file_sha256(src_path)}"

    import platform
    import sys
    try:
        import pyarrow
        pyarrow_ver = pyarrow.__version__
    except ImportError:
        pyarrow_ver = "not_installed"

    env_fingerprint = {
        "python_version": sys.version.split()[0],
        "os_platform": platform.platform(),
        "os_system": platform.system(),
        "architecture": platform.architecture()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "pyarrow_version": pyarrow_ver,
    }

    manifest_data = {
        "dataset_name": "HCM-Real",
        "dataset_version": dataset_version,
        "schema_version": schema_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "config_hash": f"sha256:{compute_file_sha256(config_path)}",
        "environment_fingerprint": env_fingerprint,
        "source_hashes": source_hashes,
        "collection_time": {
            "start_utc": start_utc_str,
            "end_utc": end_utc_str,
        },
        "total_records": total_records,
        "audit_provenance": {
            "cleaned_records_total": cleaned_total,
            "duplicate_records_flagged": dup_count,
            "audit_trail_file": "traffic_cleaned.csv",
        },
        "coverage": {
            "overall_valid_ratio": valid_ratio,
            "overall_missing_ratio": missing_ratio,
            "by_node": by_node_coverage,
            "quality_flag_distribution": qf_counts,
            "missing_reason_distribution": mr_counts,
        },
        "artifact_hashes": artifact_hashes,
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    return manifest_data

