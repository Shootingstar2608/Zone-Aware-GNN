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

from .base import CheckResult, DataLoader

# ═══════════════════════════════════════════════════
# TIÊU CHÍ 1: COMPLETENESS — Tính đầy đủ
# Đo tỷ lệ dữ liệu bị thiếu (NaN, Inf, zero bất thường)
# và kiểm tra cấu trúc dataset đúng với mô tả.
# ═══════════════════════════════════════════════════

class CompletenessChecker:
    """
    Đo tỷ lệ thiếu dữ liệu trong dataset đã xử lý.

    Các phép kiểm tra:
      - NaN rate: tỷ lệ giá trị NaN trong mỗi tensor (X, Y, A, Z)
      - Zero rate: tỷ lệ giá trị = 0 (quá nhiều zero → dữ liệu bị thiếu nhưng gán 0)
      - Inf rate: tỷ lệ giá trị vô cực (lỗi tính toán)
      - Shape validation: kích thước tensor phải khớp với meta.json
      - Required keys: dataset phải chứa đủ các key cần thiết
      - Zone coverage: mỗi nút phải có ít nhất 1 nhãn vùng
    """

    def check(self, loader: DataLoader) -> CheckResult:
        details = {}
        warnings = []
        info = []
        sub_scores = []

        # ── Kiểm tra graph_dataset.pt ──
        if loader.has_dataset:
            ds = loader.dataset
            proc_result = self._check_processed(ds, loader.meta)
            details["processed"] = proc_result
            sub_scores.append(proc_result["score"])

            if proc_result.get("warnings"):
                warnings.extend(proc_result["warnings"])
            info.append(f"Dataset: {proc_result['n_samples']} samples, "
                        f"{proc_result['n_nodes']} nodes")
        else:
            warnings.append("⚠️ graph_dataset.pt không tồn tại!")
            sub_scores.append(0.0)

        # ── Kiểm tra zone labels trong dataset (tensor Z) ──
        if loader.has_dataset:
            zone_result = self._check_zone_completeness(loader.dataset["Z"])
            details["zone_tensor"] = zone_result
            sub_scores.append(zone_result["score"])
            info.append(f"Zone tensor: {zone_result['n_nodes']} nodes × "
                        f"{zone_result['n_zone_types']} zone types")

        score = float(np.mean(sub_scores)) if sub_scores else 0.0
        return CheckResult(
            name="Completeness",
            score=score,
            details=details,
            warnings=warnings,
            info=info,
        )

    def _check_processed(self, ds: dict, meta: dict | None) -> dict:
        """
        Kiểm tra tính đầy đủ của graph_dataset.pt.

        Đo:
          - nan_rates: tỷ lệ NaN trong mỗi tensor
          - zero_rates: tỷ lệ zero trong X và Y
          - inf_rates: tỷ lệ Inf trong X và Y
          - shape_valid: tensor shapes có khớp với meta.json không
          - missing_keys: các key bắt buộc bị thiếu
        """
        result = {"warnings": []}

        X = ds["X"]  # (S, N, T_in*F) — features đầu vào
        Y = ds["Y"]  # (S, N, T_out)  — target đầu ra
        A = ds["A"]  # (N, N)         — ma trận kề
        Z = ds["Z"]  # (N, K)         — zone labels

        result["n_samples"] = X.shape[0]
        result["n_nodes"] = A.shape[0]

        # Tỷ lệ NaN trong mỗi tensor
        x_nan_rate = float(torch.isnan(X).float().mean())
        y_nan_rate = float(torch.isnan(Y).float().mean())
        a_nan_rate = float(torch.isnan(A).float().mean())
        z_nan_rate = float(torch.isnan(Z).float().mean())
        result["nan_rates"] = {
            "X": x_nan_rate, "Y": y_nan_rate,
            "A": a_nan_rate, "Z": z_nan_rate,
        }

        # Tỷ lệ zero — nếu > 50% thì cảnh báo vì có thể là dữ liệu thiếu
        x_zero_rate = float((X == 0).float().mean())
        y_zero_rate = float((Y == 0).float().mean())
        result["zero_rates"] = {"X": x_zero_rate, "Y": y_zero_rate}
        if x_zero_rate > 0.5:
            result["warnings"].append(
                f"⚠️ X có {x_zero_rate:.1%} giá trị = 0 — có thể là dữ liệu thiếu"
            )

        # Tỷ lệ Inf — giá trị vô cực do lỗi tính toán
        x_inf = float(torch.isinf(X).float().mean())
        y_inf = float(torch.isinf(Y).float().mean())
        result["inf_rates"] = {"X": x_inf, "Y": y_inf}

        # Kiểm tra kích thước tensor khớp với meta.json
        shape_ok = True
        if meta:
            expected_N = meta.get("N", -1)
            expected_K = meta.get("K", -1)
            expected_S = meta.get("S", -1)
            if A.shape[0] != expected_N:
                result["warnings"].append(
                    f"⚠️ A shape {A.shape[0]} ≠ meta N={expected_N}")
                shape_ok = False
            if Z.shape[1] != expected_K:
                result["warnings"].append(
                    f"⚠️ Z shape[1] {Z.shape[1]} ≠ meta K={expected_K}")
                shape_ok = False
            if X.shape[0] != expected_S:
                result["warnings"].append(
                    f"⚠️ X samples {X.shape[0]} ≠ meta S={expected_S}")
                shape_ok = False
        result["shape_valid"] = shape_ok

        # Kiểm tra dataset chứa đủ keys bắt buộc
        required_keys = {"A", "Z", "X", "Y", "time_labels", "nodes",
                         "feature_names", "zone_types"}
        present_keys = set(ds.keys())
        missing_keys = required_keys - present_keys
        result["missing_keys"] = list(missing_keys)
        if missing_keys:
            result["warnings"].append(
                f"⚠️ Thiếu keys trong dataset: {missing_keys}")

        # Score tổng hợp: trung bình 4 sub-metrics
        nan_score = 1.0 - np.mean([x_nan_rate, y_nan_rate, a_nan_rate, z_nan_rate])
        inf_score = 1.0 - np.mean([x_inf, y_inf])
        shape_score = 1.0 if shape_ok else 0.5
        key_score = 1.0 - len(missing_keys) / len(required_keys)
        result["score"] = float(np.mean([nan_score, inf_score, shape_score, key_score]))
        return result

    def _check_zone_completeness(self, Z: torch.Tensor) -> dict:
        """
        Kiểm tra tính đầy đủ của tensor zone labels Z.

        Đo:
          - n_nodes, n_zone_types: kích thước
          - nan_rate: tỷ lệ NaN
          - nodes_no_zone: số nút không có zone nào (tổng dòng = 0)
          - coverage: tỷ lệ nút có ít nhất 1 zone
        """
        result = {}
        result["n_nodes"] = Z.shape[0]
        result["n_zone_types"] = Z.shape[1]

        nan_rate = float(torch.isnan(Z).float().mean())
        result["nan_rate"] = nan_rate

        row_sums = Z.sum(dim=1)
        n_empty = int((row_sums == 0).sum())
        result["nodes_no_zone"] = n_empty
        result["coverage"] = 1.0 - n_empty / max(Z.shape[0], 1)

        result["score"] = (1.0 - nan_rate) * result["coverage"]
        return result


