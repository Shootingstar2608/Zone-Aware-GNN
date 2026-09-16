"""
benchmark — Module đánh giá chất lượng dữ liệu cho Zone-Aware-GNN.

Cung cấp 5 tiêu chí đo lường:
  1. Completeness  — Tính đầy đủ
  2. Validity      — Tính hợp lệ
  3. Consistency   — Tính nhất quán
  4. Freshness     — Độ tươi mới
  5. Source Reliability — Độ tin cậy nguồn
"""

from benchmark.data_quality import DataQualityAssessor

__all__ = ["DataQualityAssessor"]
