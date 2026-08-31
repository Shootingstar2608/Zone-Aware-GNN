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
# TIÊU CHÍ 5: SOURCE RELIABILITY — Độ tin cậy nguồn
# Đánh giá mức độ khả tín của từng nguồn dữ liệu
# dựa trên dấu vết trong dữ liệu đã xử lý.
# ═══════════════════════════════════════════════════

class SourceReliabilityChecker:
    """
    Đánh giá mức độ khả tín của từng nguồn dữ liệu dựa trên
    dấu vết còn lại trong dữ liệu đã xử lý.

    3 nguồn được đánh giá:
      - OSRM (qua ma trận A): đối xứng, connectivity, weight distribution
      - TomTom (qua features X): stability per node, repeated values
      - OSM Zones (qua tensor Z): coverage, diversity, multi-label distribution
    """

    def check(self, loader: DataLoader) -> CheckResult:
        details = {}
        warnings = []
        info = []
        sub_scores = []

        if not loader.has_dataset:
            return CheckResult(name="Source Reliability", score=0.0,
                               warnings=["⚠️ Không có graph_dataset.pt"])

        ds = loader.dataset

        # Đánh giá OSRM qua ma trận kề A
        osrm_result = self._assess_osrm_reliability(ds["A"])
        details["osrm"] = osrm_result
        sub_scores.append(osrm_result["score"])
        info.append(f"OSRM reliability: {osrm_result['score']:.2f}")

        # Đánh giá TomTom qua features X
        tt_result = self._assess_tomtom_reliability(ds)
        details["tomtom"] = tt_result
        sub_scores.append(tt_result["score"])
        info.append(f"TomTom reliability: {tt_result['score']:.2f}")
        if tt_result.get("warnings"):
            warnings.extend(tt_result["warnings"])

        # Đánh giá OSM zone labels qua tensor Z
        zone_result = self._assess_zone_reliability(ds)
        details["osm_zones"] = zone_result
        sub_scores.append(zone_result["score"])
        info.append(f"OSM zone reliability: {zone_result['score']:.2f}")

        score = float(np.mean(sub_scores)) if sub_scores else 0.0
        return CheckResult(
            name="Source Reliability", score=score,
            details=details, warnings=warnings, info=info,
        )

    def _assess_osrm_reliability(self, A: torch.Tensor) -> dict:
        """
        Đánh giá độ tin cậy nguồn OSRM qua ma trận kề.

        Đo 3 sub-metrics:
          1. symmetry_score: mức đối xứng tương đối giữa A[i,j] và A[j,i]
             Traffic graph có thể bất đối xứng (đường 1 chiều) nhưng
             bất đối xứng quá lớn → dữ liệu không tin cậy
          2. connectivity_score: mỗi nút phải kết nối tới ít nhất 1 nút khác
             Nút cô lập = 0 out-degree → dữ liệu bị thiếu
          3. weight_score: phân phối trọng số không quá lệch
             Dùng Coefficient of Variation (CV = std/mean)
        """
        result = {}
        N = A.shape[0]
        A_np = A.numpy()

        # 1. Đối xứng tương đối
        asym = np.abs(A_np - A_np.T)
        max_A = np.maximum(A_np, A_np.T)
        max_A[max_A == 0] = 1
        relative_asym = asym / max_A
        edge_mask = (A_np > 1e-8) | (A_np.T > 1e-8)
        mean_rel_asym = float(relative_asym[edge_mask].mean()) if edge_mask.sum() > 0 else 0.0
        result["mean_relative_asymmetry"] = round(mean_rel_asym, 4)
        symmetry_score = max(0, 1.0 - mean_rel_asym)

        # 2. Connectivity
        out_degree = (A_np > 1e-8).sum(axis=1)
        n_isolated = int((out_degree == 0).sum())
        result["n_isolated_nodes"] = n_isolated
        result["min_out_degree"] = int(out_degree.min())
        result["max_out_degree"] = int(out_degree.max())
        connectivity_score = 1.0 - n_isolated / max(N, 1)

        # 3. Weight distribution
        nonzero_weights = A_np[A_np > 1e-8]
        if len(nonzero_weights) > 0:
            cv = float(np.std(nonzero_weights) / np.mean(nonzero_weights))
            result["weight_cv"] = round(cv, 4)
            weight_score = max(0, 1.0 - cv / 3)
        else:
            weight_score = 0.0

        result["score"] = float(np.mean([symmetry_score, connectivity_score,
                                         weight_score]))
        return result

    def _assess_tomtom_reliability(self, ds: dict) -> dict:
        """
        Đánh giá độ tin cậy TomTom qua features trong tensor X.

        Đo 2 sub-metrics:
          1. stability_score: CV (coefficient of variation) trung bình per node
             cho congestion_ratio. CV thấp = ổn định = tin cậy.
             Công thức: CV = std / mean. Score = 1 - CV/2
          2. repeat_score: tỷ lệ giá trị lặp liên tiếp (|X[t] - X[t-1]| < ε)
             Nếu quá nhiều giá trị giống hệt nhau liên tiếp → API có thể
             trả về cached data thay vì dữ liệu real-time.
        """
        result = {"warnings": []}
        X = ds["X"]
        feature_names = ds.get("feature_names",
                               ["congestion_ratio", "traffic_delay_s",
                                "travel_time_s", "ff_ratio"])
        F = len(feature_names)
        S, N, TF = X.shape
        T_in = TF // F

        try:
            X_r = X.reshape(S, N, T_in, F)

            # 1. Stability per node
            cr = X_r[:, :, :, 0]       # congestion_ratio: (S, N, T_in)
            cr_flat = cr.reshape(-1, N)  # (S*T_in, N)
            node_std = cr_flat.std(dim=0)
            node_mean = cr_flat.mean(dim=0)

            result["per_node_stats"] = {
                "mean_cr": float(node_mean.mean()),
                "mean_std": float(node_std.mean()),
                "max_std": float(node_std.max()),
            }

            cv_per_node = node_std / (node_mean + 1e-8)
            mean_cv = float(cv_per_node.mean())
            result["mean_cv_per_node"] = round(mean_cv, 4)

            # 2. Repeated values detection
            cr_seq = X_r[:, :, :, 0]
            diffs = cr_seq[:, :, 1:] - cr_seq[:, :, :-1]
            n_repeated = int((diffs.abs() < 1e-8).sum())
            total_diffs = diffs.numel()
            repeat_rate = n_repeated / max(total_diffs, 1)
            result["repeated_value_rate"] = round(repeat_rate, 4)

            if repeat_rate > 0.3:
                result["warnings"].append(
                    f"⚠️ {repeat_rate:.1%} giá trị lặp liên tiếp — "
                    f"có thể do API cache"
                )

            stability_score = max(0, 1.0 - mean_cv / 2)
            repeat_score = max(0, 1.0 - repeat_rate)
            result["score"] = float(np.mean([stability_score, repeat_score]))

        except Exception as e:
            result["error"] = str(e)
            result["score"] = 0.5

        return result

    def _assess_zone_reliability(self, ds: dict) -> dict:
        """
        Đánh giá độ tin cậy của zone labels qua tensor Z.

        Đo 2 sub-metrics:
          1. coverage: % nút có ≥ 1 zone label (100% = tốt)
          2. diversity: Shannon entropy trên phân phối zone types
             Nếu mọi zone type đều được sử dụng đều → entropy cao = diversity tốt
             Nếu chỉ 1-2 zone type chiếm hết → entropy thấp = diversity kém
        """
        result = {}
        Z = ds["Z"]
        N, K = Z.shape

        # Coverage
        row_sums = Z.sum(dim=1)
        coverage = float((row_sums > 0).float().mean())
        result["coverage"] = round(coverage, 4)

        # Diversity: Shannon entropy
        zone_ratios = Z.mean(dim=0)  # tỷ lệ nút có zone type k
        if zone_ratios.sum() > 0:
            zone_ratios = zone_ratios / zone_ratios.sum()  # Chuẩn hóa về phân phối xác suất
        nonzero_ratios = zone_ratios[zone_ratios > 0]
        if len(nonzero_ratios) > 0:
            entropy = -float((nonzero_ratios * torch.log2(nonzero_ratios)).sum())
            max_entropy = np.log2(K)
            diversity = min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            diversity = 0.0
        result["zone_diversity"] = round(diversity, 4)

        # Multi-label distribution
        result["zone_count_distribution"] = {
            int(i): int((row_sums == i).sum())
            for i in range(int(row_sums.max()) + 1)
            if (row_sums == i).sum() > 0
        }
        result["mean_zones_per_node"] = round(float(row_sums.mean()), 2)

        # Zone type usage
        zone_types = ds.get("zone_types",
                            ["commercial", "residential", "industrial", "school",
                             "university", "hospital", "transport", "park"])
        result["zone_type_coverage"] = {
            zone_types[i] if i < len(zone_types) else f"zone_{i}":
                int(Z[:, i].sum())
            for i in range(K)
        }

        result["score"] = float(np.mean([coverage, diversity]))
        return result


