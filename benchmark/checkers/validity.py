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
# TIÊU CHÍ 2: VALIDITY — Tính hợp lệ
# Kiểm tra mỗi giá trị dữ liệu có nằm trong miền
# giá trị hợp lý hay không.
# ═══════════════════════════════════════════════════

class ValidityChecker:
    """
    Kiểm tra dữ liệu nằm trong miền giá trị hợp lý.

    Kiểm tra trên các tensor trong graph_dataset.pt:
      - X features: congestion_ratio, traffic_delay_s, travel_time_s, ff_ratio
      - A: ma trận kề ∈ [0, 1], đường chéo = 0
      - Z: zone labels ∈ {0, 1} (binary)
      - Y: target (congestion_ratio) ≥ 0

    Miền giá trị hợp lệ được định nghĩa trong FEATURE_RANGES.
    Có thể chỉnh sửa trực tiếp dict này để thay đổi ngưỡng.
    """

    # ── CẤU HÌNH: Miền giá trị hợp lệ cho từng feature ──
    # Cấu trúc: "tên_feature": (giá_trị_tối_thiểu, giá_trị_tối_đa)
    # Các feature này tương ứng với 4 cột trong tensor X
    FEATURE_RANGES = {
        "congestion_ratio": (0.0, 5.0),      # Tỷ lệ tắc nghẽn: 1.0 = bình thường
        "traffic_delay_s": (0.0, 7200.0),     # Delay tối đa 2 giờ (nội thành HCM)
        "travel_time_s": (0.0, 10800.0),      # Thời gian di chuyển tối đa 3 giờ
        "ff_ratio": (0.5, 10.0),              # Free-flow ratio: < 0.5 bất thường
    }

    def check(self, loader: DataLoader) -> CheckResult:
        details = {}
        warnings = []
        info = []
        sub_scores = []

        if not loader.has_dataset:
            return CheckResult(name="Validity", score=0.0,
                               warnings=["⚠️ Không có graph_dataset.pt"])

        ds = loader.dataset

        # Kiểm tra miền giá trị của tensor X (features đầu vào)
        x_result = self._check_tensor_X(ds)
        details["X_features"] = x_result
        sub_scores.append(x_result["score"])
        if x_result.get("warnings"):
            warnings.extend(x_result["warnings"])

        # Kiểm tra ma trận kề A
        a_result = self._check_adjacency(ds["A"])
        details["adjacency"] = a_result
        sub_scores.append(a_result["score"])
        if a_result.get("warnings"):
            warnings.extend(a_result["warnings"])

        # Kiểm tra zone labels Z
        z_result = self._check_zone_tensor(ds["Z"])
        details["zone_tensor"] = z_result
        sub_scores.append(z_result["score"])

        # Kiểm tra target Y
        y_result = self._check_target_Y(ds["Y"])
        details["Y_target"] = y_result
        sub_scores.append(y_result["score"])
        if y_result.get("warnings"):
            warnings.extend(y_result["warnings"])

        score = float(np.mean(sub_scores)) if sub_scores else 0.0
        return CheckResult(
            name="Validity", score=score,
            details=details, warnings=warnings, info=info,
        )

    def _check_tensor_X(self, ds: dict) -> dict:
        """
        Kiểm tra từng feature trong tensor X có nằm trong miền hợp lệ.

        Tensor X có shape (S, N, T_in*F). Reshape thành (S, N, T_in, F)
        rồi kiểm tra từng feature (cột F) theo FEATURE_RANGES.

        Output:
          - has_nan, has_inf: có NaN/Inf không
          - feature_violations: dict mỗi feature → {range, total, below_min,
            above_max, violation_rate, actual_range}
          - score: tỷ lệ giá trị hợp lệ (trừ penalty nếu có NaN/Inf)
        """
        X = ds["X"]
        result = {"warnings": []}

        has_nan = bool(torch.isnan(X).any())
        has_inf = bool(torch.isinf(X).any())
        result["has_nan"] = has_nan
        result["has_inf"] = has_inf
        if has_nan:
            result["warnings"].append("⚠️ Tensor X chứa NaN")
        if has_inf:
            result["warnings"].append("⚠️ Tensor X chứa Inf")

        S, N, TF = X.shape
        feature_names = ds.get("feature_names", list(self.FEATURE_RANGES.keys()))
        F = len(feature_names)
        T_in = TF // F if F > 0 else 12

        violations = {}
        valid_count = 0
        total_count = 0

        try:
            # Reshape X từ (S, N, T_in*F) → (S, N, T_in, F)
            # để tách riêng từng feature ra kiểm tra
            X_reshaped = X.reshape(S, N, T_in, F)

            for f_idx, f_name in enumerate(feature_names):
                if f_name in self.FEATURE_RANGES:
                    lo, hi = self.FEATURE_RANGES[f_name]
                    feat = X_reshaped[:, :, :, f_idx]

                    # Bỏ qua NaN/Inf khi đếm vi phạm
                    valid_mask = ~torch.isnan(feat) & ~torch.isinf(feat)
                    feat_valid = feat[valid_mask]

                    n_total = int(feat_valid.numel())
                    n_below = int((feat_valid < lo).sum())  # dưới giá trị min
                    n_above = int((feat_valid > hi).sum())  # trên giá trị max
                    n_out = n_below + n_above

                    violations[f_name] = {
                        "range": [lo, hi],
                        "total": n_total,
                        "below_min": n_below,
                        "above_max": n_above,
                        "violation_rate": n_out / max(n_total, 1),
                        "actual_range": [float(feat_valid.min()),
                                         float(feat_valid.max())] if n_total > 0 else None,
                    }
                    valid_count += n_total - n_out
                    total_count += n_total

                    # Cảnh báo nếu > 5% giá trị ngoài miền
                    if n_out / max(n_total, 1) > 0.05:
                        result["warnings"].append(
                            f"⚠️ {f_name}: {n_out/n_total:.1%} ngoài [{lo}, {hi}]"
                        )
        except Exception as e:
            result["warnings"].append(f"⚠️ Không reshape được X: {e}")
            total_count = int(X.numel())
            valid_count = total_count

        result["feature_violations"] = violations
        # Nếu có NaN/Inf thì nhân penalty 0.9
        result["score"] = valid_count / max(total_count, 1) if not (has_nan or has_inf) else \
            (valid_count / max(total_count, 1)) * 0.9
        return result

    def _check_adjacency(self, A: torch.Tensor) -> dict:
        """
        Kiểm tra ma trận kề A.

        Kiểm tra:
          - is_square: A phải là ma trận vuông (N×N)
          - in_01_range: tất cả giá trị ∈ [0, 1] (đã row-normalized)
          - has_self_loops: đường chéo nên = 0 (không có tự kết nối)
          - density: tỷ lệ ô khác 0 (đo mật độ đồ thị)
        """
        result = {"warnings": []}
        N = A.shape[0]

        result["is_square"] = (A.shape[0] == A.shape[1])

        a_min = float(A.min())
        a_max = float(A.max())
        result["range"] = [a_min, a_max]
        in_range = (a_min >= -1e-6) and (a_max <= 1.0 + 1e-6)
        result["in_01_range"] = in_range
        if not in_range:
            result["warnings"].append(
                f"⚠️ A range [{a_min:.4f}, {a_max:.4f}] ngoài [0, 1]")

        diag = torch.diag(A)
        diag_mean = float(diag.mean())
        result["diagonal_mean"] = diag_mean
        result["has_self_loops"] = diag_mean > 1e-6

        n_nonzero = int((A > 1e-6).sum())
        result["density"] = n_nonzero / (N * N)
        result["n_nonzero"] = n_nonzero

        scores = [
            1.0 if result["is_square"] else 0.0,
            1.0 if in_range else 0.5,
            1.0 if not result["has_self_loops"] else 0.8,
        ]
        result["score"] = float(np.mean(scores))
        return result

    def _check_zone_tensor(self, Z: torch.Tensor) -> dict:
        """
        Kiểm tra tensor zone labels Z.

        Kiểm tra:
          - is_binary: tất cả giá trị phải là 0 hoặc 1
          - nodes_without_zone: không nút nào nên có tổng = 0
          - zones_per_node: thống kê số zone mỗi nút
        """
        result = {}

        unique_vals = torch.unique(Z).tolist()
        is_binary = all(v in [0.0, 1.0] for v in unique_vals)
        result["is_binary"] = is_binary
        result["unique_values"] = unique_vals

        row_sums = Z.sum(dim=1)
        n_empty = int((row_sums == 0).sum())
        result["nodes_without_zone"] = n_empty
        result["min_zones_per_node"] = int(row_sums.min())
        result["max_zones_per_node"] = int(row_sums.max())
        result["mean_zones_per_node"] = float(row_sums.mean())

        score = 1.0
        if not is_binary:
            score *= 0.5
        if n_empty > 0:
            score *= (1.0 - n_empty / Z.shape[0])
        result["score"] = score
        return result

    def _check_target_Y(self, Y: torch.Tensor) -> dict:
        """
        Kiểm tra target Y (congestion_ratio — giá trị dự báo).

        Kiểm tra:
          - has_nan, has_inf
          - n_negative: congestion_ratio không thể âm
          - n_extreme_high: giá trị > 5.0 là cực đoan bất thường
        """
        result = {"warnings": []}

        has_nan = bool(torch.isnan(Y).any())
        has_inf = bool(torch.isinf(Y).any())
        result["has_nan"] = has_nan
        result["has_inf"] = has_inf

        valid_Y = Y[~torch.isnan(Y) & ~torch.isinf(Y)]
        if valid_Y.numel() > 0:
            result["range"] = [float(valid_Y.min()), float(valid_Y.max())]
            result["mean"] = float(valid_Y.mean())

            n_negative = int((valid_Y < 0).sum())
            n_extreme = int((valid_Y > 5.0).sum())
            result["n_negative"] = n_negative
            result["n_extreme_high"] = n_extreme

            if n_negative > 0:
                result["warnings"].append(f"⚠️ Y có {n_negative} giá trị âm")

            valid_rate = 1.0 - (n_negative + n_extreme) / valid_Y.numel()
        else:
            valid_rate = 0.0

        score = valid_rate
        if has_nan:
            score *= 0.8
        if has_inf:
            score *= 0.7
        result["score"] = score
        return result


