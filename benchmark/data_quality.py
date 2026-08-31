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

from .checkers import CheckResult, DataLoader, CompletenessChecker, ValidityChecker, ConsistencyChecker, FreshnessChecker, SourceReliabilityChecker

# ═══════════════════════════════════════════════════
# ORCHESTRATOR — Chạy tất cả 5 tiêu chí
# ═══════════════════════════════════════════════════

class DataQualityAssessor:
    """
    Điều phối chạy tất cả 5 tiêu chí đánh giá chất lượng dữ liệu
    trên dữ liệu đã xử lý.

    Cách dùng:
        assessor = DataQualityAssessor(project_root=".")
        report = assessor.run_all()
        print(report["overall_score"])

    Output:
        - Console: in tóm tắt kết quả
        - report dict: chứa overall_score, results, data_summary
        - generate_markdown_report(): sinh file .md chi tiết
    """

    def __init__(self, project_root: str = "."):
        self.loader = DataLoader(project_root)
        self.checkers = {
            "completeness": CompletenessChecker(),
            "validity": ValidityChecker(),
            "consistency": ConsistencyChecker(),
            "freshness": FreshnessChecker(),
            "source_reliability": SourceReliabilityChecker(),
        }
        self._results: dict[str, CheckResult] = {}

    def run_all(self) -> dict:
        """Chạy tất cả 5 tiêu chí và trả về kết quả tổng hợp."""
        print("=" * 60)
        print("  Data Quality Assessment — Zone-Aware-GNN")
        print("=" * 60)

        # Hiển thị các file dữ liệu có sẵn
        summary = self.loader.summary()
        print(f"\n📦 Dữ liệu đã xử lý:")
        for source, available in summary["available"].items():
            icon = "✅" if available else "❌"
            print(f"  {icon} {source}")
        print()

        # Chạy từng checker
        for key, checker in self.checkers.items():
            print(f"🔍 Checking {key}...")
            try:
                result = checker.check(self.loader)
                self._results[key] = result
                icon = "✅" if result.score >= 0.8 else "⚠️" if result.score >= 0.6 else "❌"
                print(f"  {icon} {result.name}: {result.score:.4f}")
                for w in result.warnings:
                    print(f"    {w}")
            except Exception as e:
                print(f"  ❌ {key}: ERROR — {e}")
                self._results[key] = CheckResult(
                    name=key, score=0.0, warnings=[f"Error: {e}"]
                )

        # Overall score = trung bình cộng 5 scores
        overall = float(np.mean([r.score for r in self._results.values()]))

        print(f"\n{'=' * 60}")
        print(f"  Overall Quality Score: {overall:.4f}")
        print(f"{'=' * 60}\n")

        return {
            "overall_score": overall,
            "results": self._results,
            "data_summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

    def get_results(self) -> dict[str, CheckResult]:
        """Trả về kết quả đã chạy."""
        return self._results

    def generate_markdown_report(self) -> str:
        """Sinh báo cáo dạng Markdown."""
        if not self._results:
            self.run_all()

        overall = float(np.mean([r.score for r in self._results.values()]))
        summary = self.loader.summary()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append("# 📊 Data Quality Report — HCM-Zone Dataset\n")
        lines.append(f"> **Generated:** {now}\n")

        # Nguồn dữ liệu
        lines.append("## Nguồn dữ liệu đã xử lý\n")
        lines.append("| File | Trạng thái |")
        lines.append("|---|---|")
        for source, avail in summary["available"].items():
            icon = "✅ Có" if avail else "❌ Không có"
            lines.append(f"| `{source}` | {icon} |")
        lines.append("")

        # Summary Dashboard
        lines.append("## Summary Dashboard\n")
        lines.append("| Tiêu chí | Score | Status | Các giá trị thành phần (Sub-scores) |")
        lines.append("|---|:---:|:---:|---|")

        for key, result in self._results.items():
            icon = "✅" if result.score >= 0.8 else "⚠️" if result.score >= 0.6 else "❌"
            
            # Khai thác sub-scores
            sub_scores_str = []
            if result.details:
                for sub_key, sub_val in result.details.items():
                    if isinstance(sub_val, dict) and "score" in sub_val:
                        sub_score = sub_val["score"]
                        info_str = str(sub_val.get("info", ""))
                        if sub_score == 1.0 and "bỏ qua" in info_str.lower():
                            sub_scores_str.append(f"`{sub_key}: {sub_score:.4f}` (bỏ qua)")
                        else:
                            sub_scores_str.append(f"`{sub_key}: {sub_score:.4f}`")
            
            sub_scores_cell = ", ".join(sub_scores_str) if sub_scores_str else "-"
            
            lines.append(f"| {result.name} | {result.score:.4f} | {icon} | {sub_scores_cell} |")
            
        lines.append(f"| **Overall** | **{overall:.4f}** | | (Trung bình cộng 5 tiêu chí trên) |")
        lines.append("")

        # Chi tiết từng tiêu chí
        for i, (key, result) in enumerate(self._results.items(), 1):
            lines.append(f"---\n")
            lines.append(f"## {i}. {result.name} (Score: {result.score:.4f})\n")

            if result.warnings:
                lines.append("### ⚠️ Warnings\n")
                for w in result.warnings:
                    lines.append(f"- {w}")
                lines.append("")

            if result.info:
                lines.append("### ℹ️ Info\n")
                for info_item in result.info:
                    lines.append(f"- {info_item}")
                lines.append("")

            if result.details:
                lines.append("### 📋 Chi tiết\n")
                for detail_key, detail_val in result.details.items():
                    lines.append(f"#### {detail_key}\n")
                    if isinstance(detail_val, dict):
                        lines.append("| Metric | Value |")
                        lines.append("|---|---|")
                        for k, v in detail_val.items():
                            if k in ("warnings", "score"):
                                continue
                            if isinstance(v, dict):
                                lines.append(f"| **{k}** | |")
                                for kk, vv in v.items():
                                    vv_str = (f"{vv:.6f}" if isinstance(vv, float)
                                              else str(vv))
                                    lines.append(f"| — {kk} | {vv_str} |")
                            elif isinstance(v, list):
                                lines.append(f"| {k} | {v} |")
                            else:
                                v_str = (f"{v:.6f}" if isinstance(v, float)
                                         else str(v))
                                lines.append(f"| {k} | {v_str} |")
                        lines.append("")
                    else:
                        lines.append(f"{detail_val}\n")

        # Recommendations
        lines.append("---\n")
        lines.append("## 💡 Recommendations\n")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        return "\n".join(lines)

    def _generate_recommendations(self) -> list[str]:
        """Sinh gợi ý cải thiện dựa trên kết quả."""
        recs = []

        for key, result in self._results.items():
            if result.score < 0.6:
                recs.append(
                    f"🔴 **{result.name}** cần cải thiện gấp "
                    f"(score={result.score:.2f})")
            elif result.score < 0.8:
                recs.append(
                    f"🟡 **{result.name}** có thể cải thiện thêm "
                    f"(score={result.score:.2f})")

        comp = self._results.get("completeness")
        if comp and comp.score >= 0.9:
            recs.append("✅ Dữ liệu đầy đủ, không cần bổ sung")

        fresh = self._results.get("freshness")
        if fresh and fresh.score < 0.7:
            recs.append(
                "📅 Nên thu thập thêm dữ liệu để cải thiện temporal coverage "
                "(khuyến nghị: ≥ 7 ngày, bao phủ cả weekday và weekend)")

        rel = self._results.get("source_reliability")
        if rel and rel.details.get("tomtom", {}).get("repeated_value_rate", 0) > 0.3:
            recs.append(
                "🔄 TomTom API trả cached data — tăng interval giữa snapshots")

        if not recs:
            recs.append("✅ Chất lượng dữ liệu tổng thể tốt!")

        return recs
