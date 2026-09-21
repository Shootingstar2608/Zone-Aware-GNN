"""
benchmark — Module đánh giá chất lượng dữ liệu cho Zone-Aware-GNN.

Cung cấp 5 tiêu chí đo lường:
  1. Completeness  — Tính đầy đủ
  2. Validity      — Tính hợp lệ
  3. Consistency   — Tính nhất quán
  4. Freshness     — Độ tươi mới
  5. Source Reliability — Độ tin cậy nguồn
"""

# Import LUOI (PEP 562). `benchmark.data_quality` keo theo torch + pandas,
# trong khi `benchmark.partition_gen` chi can numpy. Neu import thang o day
# thi `import benchmark.partition_gen` cung doi hoi torch => CI do tren clone
# sach, va pha DoD "chay test khong can raw private".
# Duong dung cu `from benchmark import DataQualityAssessor` van giu nguyen.

__all__ = ["DataQualityAssessor"]


def __getattr__(name):
    if name == "DataQualityAssessor":
        from benchmark.data_quality import DataQualityAssessor
        return DataQualityAssessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")