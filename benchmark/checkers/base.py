"""
benchmark/data_quality.py
=========================
Module đo lường chất lượng dữ liệu đã xử lý theo 5 tiêu chí:
  1. Completeness  — Tính đầy đủ (tỷ lệ thiếu dữ liệu)
  2. Validity      — Tính hợp lệ (dữ liệu nằm trong miền giá trị hợp lý)
  3. Consistency   — Tính nhất quán (đối chiếu giữa các thành phần trong dataset)
  4. Freshness     — Độ tươi mới (mức độ bao phủ thời gian)
  5. Source Reliability — Độ tin cậy nguồn (đánh giá từng nguồn OSRM/TomTom/OSM)

Đo lường trên dữ liệu đã xử lý:
  - data/processed/graph_dataset.pt  (tensors A, Z, X, Y, time_labels, ...)
  - data/processed/meta.json         (metadata: N, K, F, T_in, T_out, S, ...)
  - data/raw/zone_labels.csv         (zone labels gốc để đối chiếu)

Cách dùng:
    from benchmark.data_quality import DataQualityAssessor
    assessor = DataQualityAssessor(project_root=".")
    report = assessor.run_all()
    print(report["overall_score"])
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

# ═══════════════════════════════════════════════════
# Data classes cho kết quả
# ═══════════════════════════════════════════════════

@dataclass
class CheckResult:
    """Kết quả của một tiêu chí đánh giá."""
    name: str
    score: float              # ∈ [0, 1], 1 = hoàn hảo
    details: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)


# ═══════════════════════════════════════════════════
# Helper: Load dữ liệu đã xử lý
# ═══════════════════════════════════════════════════

class DataLoader:
    """
    Load dữ liệu đã xử lý từ data/processed/.

    Các file được load:
      - graph_dataset.pt : chứa tensors A, Z, X, Y, time_labels, nodes, ...
      - meta.json        : metadata mô tả dataset (N, K, F, T_in, T_out, S)
      - zone_labels.csv  : file zone labels gốc để đối chiếu với tensor Z
    """

    def __init__(self, project_root: str = "."):
        self.root = project_root
        self.processed_dir = os.path.join(project_root, "data", "processed")
        self.raw_dir = os.path.join(project_root, "data", "raw")

        # Paths
        self.dataset_path = os.path.join(self.processed_dir, "graph_dataset.pt")
        self.meta_path = os.path.join(self.processed_dir, "meta.json")
        self.osrm_path = os.path.join(self.raw_dir, "hcm_osrm_dataset.csv")
        self.tomtom_path = os.path.join(self.raw_dir, "tomtom_traffic.csv")
        self.zone_path = os.path.join(self.raw_dir, "zone_labels.csv")

        # Loaded data (lazy — chỉ đọc file khi lần đầu truy cập)
        self._dataset = None
        self._meta = None
        self._df_zones = None

    # ── Kiểm tra file tồn tại ──

    @property
    def has_dataset(self) -> bool:
        """graph_dataset.pt có tồn tại không?"""
        return os.path.exists(self.dataset_path)

    @property
    def has_meta(self) -> bool:
        """meta.json có tồn tại không?"""
        return os.path.exists(self.meta_path)

    @property
    def has_raw_osrm(self) -> bool:
        return os.path.exists(self.osrm_path)

    @property
    def has_raw_tomtom(self) -> bool:
        return os.path.exists(self.tomtom_path)

    @property
    def has_raw_zone(self) -> bool:
        return os.path.exists(self.zone_path)

    @property
    def has_zones(self) -> bool:
        """zone_labels.csv có tồn tại không?"""
        return os.path.exists(self.zone_path)

    # ── Lazy loaders (chỉ đọc file 1 lần, cache lại) ──

    @property
    def dataset(self) -> dict:
        """Load graph_dataset.pt → dict chứa các tensors."""
        if self._dataset is None and self.has_dataset:
            self._dataset = torch.load(self.dataset_path, weights_only=False)
        return self._dataset

    @property
    def meta(self) -> dict:
        """Load meta.json → dict metadata."""
        if self._meta is None and self.has_meta:
            with open(self.meta_path) as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def df_zones(self) -> pd.DataFrame | None:
        """Load zone_labels.csv → DataFrame."""
        if self._df_zones is None and self.has_zones:
            self._df_zones = pd.read_csv(self.zone_path)
        return self._df_zones

    def summary(self) -> dict:
        """Trả về tóm tắt các file dữ liệu đã xử lý có sẵn."""
        available = {
            "graph_dataset.pt": self.has_dataset,
            "meta.json": self.has_meta,
            "hcm_osrm_dataset.csv (raw)": self.has_raw_osrm,
            "tomtom_traffic.csv (raw)": self.has_raw_tomtom,
            "zone_labels.csv (raw)": self.has_raw_zone,
        }
        return {"available": available}
