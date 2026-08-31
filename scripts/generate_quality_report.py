#!/usr/bin/env python3
"""
scripts/generate_quality_report.py
==================================
Tự động quét toàn bộ dữ liệu đã xử lý và xuất báo cáo
chất lượng dữ liệu ra Markdown.

Chạy:
    python scripts/generate_quality_report.py
    # hoặc
    ./venv/bin/python scripts/generate_quality_report.py

Output:
    data/results/data_quality_report.md

Options:
    --project-root PATH    Thư mục gốc dự án (default: tự phát hiện)
    --output PATH          Đường dẫn file report (default: data/results/data_quality_report.md)
    --verbose              In chi tiết kết quả từng tiêu chí
"""

import argparse
import os
import sys

# Thêm project root vào sys.path để import benchmark package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.data_quality import DataQualityAssessor


def main():
    parser = argparse.ArgumentParser(
        description="Sinh báo cáo chất lượng dữ liệu cho Zone-Aware-GNN"
    )
    parser.add_argument(
        "--project-root",
        default=PROJECT_ROOT,
        help="Thư mục gốc dự án (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Đường dẫn file report output "
             "(default: data/results/data_quality_report.md)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="In chi tiết kết quả từng tiêu chí",
    )
    args = parser.parse_args()

    project_root = args.project_root
    output_path = args.output or os.path.join(
        project_root, "data", "results", "data_quality_report.md"
    )

    # ── Kiểm tra dữ liệu tồn tại ──
    dataset_path = os.path.join(project_root, "data", "processed", "graph_dataset.pt")
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy {dataset_path}")
        print("   Hãy chạy 'python scripts/build_graph.py' trước.")
        sys.exit(1)

    # ── Chạy đánh giá ──
    print(f"\n🚀 Bắt đầu đánh giá chất lượng dữ liệu...")
    print(f"   Project root: {project_root}")
    print(f"   Output: {output_path}\n")

    assessor = DataQualityAssessor(project_root=project_root)
    report_data = assessor.run_all()

    # ── Verbose output ──
    if args.verbose:
        print("\n" + "=" * 60)
        print("  CHI TIẾT KẾT QUẢ")
        print("=" * 60)
        for key, result in report_data["results"].items():
            print(f"\n{'─' * 50}")
            print(f"📌 {result.name} — Score: {result.score:.4f}")
            if result.warnings:
                for w in result.warnings:
                    print(f"   {w}")
            if result.info:
                for info_item in result.info:
                    print(f"   {info_item}")
            if result.details:
                import json as _json
                for dk, dv in result.details.items():
                    print(f"\n   [{dk}]")
                    if isinstance(dv, dict):
                        for k, v in dv.items():
                            if k in ("warnings",):
                                continue
                            print(f"     {k}: {v}")

    # ── Sinh Markdown report ──
    markdown = assessor.generate_markdown_report()

    # ── Lưu file ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\n✅ Báo cáo đã được lưu tại: {output_path}")
    print(f"   Overall Score: {report_data['overall_score']:.4f}")

    # ── Return code dựa trên quality ──
    if report_data["overall_score"] < 0.5:
        print("\n🔴 Chất lượng dữ liệu THẤP — cần kiểm tra lại!")
        sys.exit(2)
    elif report_data["overall_score"] < 0.7:
        print("\n🟡 Chất lượng dữ liệu TRUNG BÌNH — cần cải thiện.")
    else:
        print("\n🟢 Chất lượng dữ liệu TỐT!")


if __name__ == "__main__":
    main()
