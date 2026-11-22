"""
scripts/evaluate_hcm_real_quality.py
====================================
CLI script chạy đánh giá chất lượng dữ liệu HCM-Real và xuất báo cáo.
Chạy:
    python scripts/evaluate_hcm_real_quality.py
    python scripts/evaluate_hcm_real_quality.py --canonical-path data/processed/hcm_real_v1/traffic_canonical.csv
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from benchmark.hcm_real_quality import HCMRealQualityEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate HCM-Real data quality per-node / per-day")
    parser.add_argument(
        "--canonical-path",
        default="data/processed/hcm_real_v1/traffic_canonical.csv",
        help="Đường dẫn file traffic_canonical.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/results/hcm_real_v1",
        help="Thư mục xuất báo cáo JSON và Markdown",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.85,
        help="Ngưỡng điểm chất lượng tối thiểu toàn cục để pass CI/CD gate (mặc định 0.85)",
    )
    parser.add_argument(
        "--fail-under-completeness",
        type=float,
        default=0.80,
        help="Ngưỡng completeness tối thiểu (mặc định 0.80)",
    )
    parser.add_argument(
        "--fail-under-consistency",
        type=float,
        default=0.85,
        help="Ngưỡng consistency tối thiểu (mặc định 0.85)",
    )
    args = parser.parse_args()

    evaluator = HCMRealQualityEvaluator(args.canonical_path)
    res = evaluator.run_all(args.out_dir)

    matrix = res["matrix"]
    if matrix:
        import numpy as np
        mean_overall = float(np.mean([m["overall_score"] for m in matrix.values()]))
        mean_comp = float(np.mean([m["completeness"] for m in matrix.values()]))
        mean_cons = float(np.mean([m["consistency"] for m in matrix.values()]))
    else:
        mean_overall = 0.0
        mean_comp = 0.0
        mean_cons = 0.0

    print("=" * 65)
    print("  BÁO CÁO CHẤT LƯỢNG DỮ LIỆU HCM-REAL ĐÃ ĐƯỢC TẠO THÀNH CÔNG")
    print(f"  - JSON Matrix : {res['json_path']}")
    print(f"  - Markdown Rep: {res['markdown_path']}")
    print(f"  - Điểm tổng thể: {mean_overall:.4f} (Ngưỡng yêu cầu: >= {args.fail_under})")
    print(f"  - Completeness : {mean_comp:.4f} (Ngưỡng yêu cầu: >= {args.fail_under_completeness})")
    print(f"  - Consistency  : {mean_cons:.4f} (Ngưỡng yêu cầu: >= {args.fail_under_consistency})")
    print("=" * 65)

    gate_failed = False
    failure_reasons = []

    if mean_overall < args.fail_under:
        gate_failed = True
        failure_reasons.append(f"Overall Quality ({mean_overall:.4f}) < {args.fail_under}")

    if mean_comp < args.fail_under_completeness:
        gate_failed = True
        failure_reasons.append(f"Completeness ({mean_comp:.4f}) < {args.fail_under_completeness}")

    if mean_cons < args.fail_under_consistency:
        gate_failed = True
        failure_reasons.append(f"Consistency ({mean_cons:.4f}) < {args.fail_under_consistency}")

    if gate_failed:
        print("❌ CI/CD QUALITY GATE THẤT BẠI:")
        for r in failure_reasons:
            print(f"   - {r}")
        print("Dừng pipeline để bảo vệ mô hình downstream!")
        sys.exit(1)
    else:
        print("✅ CI/CD QUALITY GATE PASSED: Tất cả các tiêu chí đều đạt chuẩn yêu cầu!")
        sys.exit(0)


if __name__ == "__main__":
    main()
