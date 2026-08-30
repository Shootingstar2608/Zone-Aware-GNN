"""
zone_stratified_analysis.py
============================
Task Bảo — Tuần 3

Vẽ biểu đồ phân tích lỗi chi tiết theo từng loại vùng (zone_stratified_analysis).
Chứng minh mô hình mới (zone_full/zone_full_tc/zone_full_sinc) giảm sai số
vượt trội ở các vùng phức tạp (trường học, bệnh viện, multi-zone).

Input : data/results/all_results.csv hoặc ablation_results.csv
Output: data/results/zone_stratified_analysis.png

Dùng: python scripts/zone_stratified_analysis.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUT_DIR = "data/results"
META_PATH = "data/processed/meta.json"

ZONE_DISPLAY = {
    "commercial": "🏬 Thương mại",
    "residential": "🏘 Dân cư",
    "industrial": "🏭 Công nghiệp",
    "school": "🏫 Trường học",
    "university": "🎓 Đại học",
    "hospital": "🏥 Bệnh viện",
    "transport": "🚌 Giao thông",
    "park": "🌳 Công viên",
    "multi_zone": "🔀 Đa vùng",
}

MODEL_COLORS = {
    "baseline_ahgnn": "#E74C3C",
    "zone_concat": "#E67E22",
    "zone_weight": "#F39C12",
    "zone_full": "#27AE60",
    "zone_full_tc": "#2980B9",
    "zone_full_sinc": "#9B59B6",
    "lstm": "#95A5A6",
    "gcn_gru": "#7F8C8D",
    "stgcn": "#BDC3C7",
}

MODEL_LABELS = {
    "baseline_ahgnn": "AH-GNN (baseline)",
    "zone_concat": "+ Zone Concat",
    "zone_weight": "+ Zone Weight",
    "zone_full": "Zone-Full ★",
    "zone_full_tc": "Zone-TC (Tân)",
    "zone_full_sinc": "Zone-Sinc (Bảo)",
    "lstm": "LSTM",
    "gcn_gru": "GCN-GRU",
    "stgcn": "STGCN",
}


def load_results():
    """Load kết quả từ CSV — ưu tiên all_results.csv."""
    for fname in ["all_results.csv", "ablation_results.csv", "baseline_results.csv"]:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ Loaded: {path} ({len(df)} rows)")
            return df
    raise FileNotFoundError("Không tìm thấy results CSV trong data/results/")


def extract_zone_mae_cols(df: pd.DataFrame) -> list:
    """Lấy danh sách các cột MAE_<zone>."""
    return [c for c in df.columns if c.startswith("MAE_") and c != "MAE"]


def plot_zone_stratified(df: pd.DataFrame, out_path: str):
    zone_cols = extract_zone_mae_cols(df)
    zone_names = [c.replace("MAE_", "") for c in zone_cols]
    n_zones = len(zone_cols)

    if n_zones == 0:
        print("⚠️  Không có cột MAE_<zone> trong CSV.")
        print("   Cần chạy train.py với compute_zone_stratified_metrics trước.")
        _plot_placeholder(df, out_path)
        return

    # Filter chỉ lấy zone-aware models để so sánh chính
    priority_models = ["baseline_ahgnn", "zone_full", "zone_full_tc", "zone_full_sinc"]
    available = df["variant"].tolist()
    plot_models = [m for m in priority_models if m in available]
    if not plot_models:
        plot_models = available[:6]

    df_plot = df[df["variant"].isin(plot_models)].copy()

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 24), facecolor="#0D1117")
    gs = gridspec.GridSpec(
        3, 1, figure=fig, hspace=0.5, top=0.93, bottom=0.06, left=0.07, right=0.97
    )

    title_color = "#ECF0F1"
    label_color = "#BDC3C7"
    grid_color = "#1E2D3D"

    fig.suptitle(
        "Phân tích Sai số theo Loại Vùng Chức năng\n" "Zone-Stratified MAE Analysis",
        color=title_color,
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    # ── Plot 1: Grouped bar chart — MAE theo zone x model ───────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0D1117")
    _style_ax(ax1, grid_color, label_color)

    x = np.arange(n_zones)
    n_m = len(plot_models)
    bar_w = 0.72 / n_m

    for mi, model in enumerate(plot_models):
        row = df_plot[df_plot["variant"] == model]
        if row.empty:
            continue
        vals = [float(row[c].values[0]) if c in row.columns else 0 for c in zone_cols]
        offset = (mi - n_m / 2 + 0.5) * bar_w
        color = MODEL_COLORS.get(model, "#95A5A6")
        label = MODEL_LABELS.get(model, model)
        bars = ax1.bar(
            x + offset,
            vals,
            bar_w,
            label=label,
            color=color,
            alpha=0.88,
            edgecolor="#0D1117",
            linewidth=0.5,
        )

        # Thêm giá trị lên đỉnh bar nếu là model đề xuất
        if model in ["zone_full", "zone_full_tc", "zone_full_sinc"]:
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color=color,
                        fontweight="bold",
                    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [ZONE_DISPLAY.get(z, z) for z in zone_names],
        color=label_color,
        fontsize=10,
        rotation=15,
        ha="right",
    )
    ax1.set_ylabel("MAE (thấp hơn = tốt hơn)", color=label_color, fontsize=11)
    ax1.set_title(
        "MAE theo từng Loại Vùng — So sánh các mô hình\n"
        "(Vùng phức tạp: 🏫 Trường học, 🏥 Bệnh viện, 🔀 Đa vùng)",
        color=title_color,
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(
        facecolor="#1A252F",
        edgecolor="#2C3E50",
        labelcolor=label_color,
        fontsize=9,
        loc="upper right",
    )

    # ── Plot 2: Improvement heatmap (% giảm MAE so với baseline) ────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#0D1117")

    baseline_row = df_plot[df_plot["variant"] == "baseline_ahgnn"]
    compare_models = [m for m in plot_models if m != "baseline_ahgnn"]

    if not baseline_row.empty and compare_models:
        improvement_matrix = []
        baseline_vals = np.array(
            [
                float(baseline_row[c].values[0]) if c in baseline_row.columns else 0
                for c in zone_cols
            ]
        )

        for model in compare_models:
            row = df_plot[df_plot["variant"] == model]
            if row.empty:
                improvement_matrix.append([0] * n_zones)
                continue
            vals = np.array(
                [float(row[c].values[0]) if c in row.columns else 0 for c in zone_cols]
            )
            improv = np.where(
                baseline_vals > 0, (baseline_vals - vals) / baseline_vals * 100, 0
            )
            improvement_matrix.append(improv)

        imp_arr = np.array(improvement_matrix)
        vmax = max(abs(imp_arr).max(), 1)

        im = ax2.imshow(imp_arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

        ax2.set_xticks(range(n_zones))
        ax2.set_yticks(range(len(compare_models)))
        ax2.set_xticklabels(
            [ZONE_DISPLAY.get(z, z) for z in zone_names],
            color=label_color,
            fontsize=9,
            rotation=20,
            ha="right",
        )
        ax2.set_yticklabels(
            [MODEL_LABELS.get(m, m) for m in compare_models],
            color=label_color,
            fontsize=10,
        )

        # Giá trị trong ô
        for i in range(len(compare_models)):
            for j in range(n_zones):
                v = imp_arr[i, j]
                txt_color = "white" if abs(v) > vmax * 0.5 else "#2C3E50"
                ax2.text(
                    j,
                    i,
                    f"{v:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_color,
                    fontweight="bold",
                )

        cbar = plt.colorbar(im, ax=ax2, fraction=0.015, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=label_color, labelsize=8)
        cbar.set_label(
            "% Cải thiện so với AH-GNN baseline", color=label_color, fontsize=9
        )
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=label_color)

        ax2.set_title(
            "% Cải thiện MAE so với AH-GNN Baseline theo từng Loại Vùng\n"
            "(Xanh = tốt hơn baseline, Đỏ = kém hơn)",
            color=title_color,
            fontsize=12,
            fontweight="bold",
        )
        _style_ax(ax2, grid_color, label_color)
    else:
        ax2.text(
            0.5,
            0.5,
            "Cần có baseline_ahgnn trong results để vẽ improvement heatmap",
            ha="center",
            va="center",
            color=label_color,
            fontsize=12,
            transform=ax2.transAxes,
        )

    # ── Plot 3: Radar chart — profile lỗi theo zone ─────────────────────────
    ax3 = fig.add_subplot(gs[2], projection="polar")
    ax3.set_facecolor("#0D1117")

    angles = np.linspace(0, 2 * np.pi, n_zones, endpoint=False).tolist()
    angles += angles[:1]

    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(
        [ZONE_DISPLAY.get(z, z) for z in zone_names],
        color=label_color,
        fontsize=9,
    )
    ax3.yaxis.set_tick_params(labelcolor=label_color, labelsize=7)
    ax3.grid(color=grid_color, linewidth=0.8)
    ax3.spines["polar"].set_color("#2C3E50")

    highlight = ["baseline_ahgnn", "zone_full", "zone_full_sinc"]
    for model in highlight:
        row = df_plot[df_plot["variant"] == model]
        if row.empty:
            continue
        vals = [float(row[c].values[0]) if c in row.columns else 0 for c in zone_cols]
        vals += vals[:1]
        color = MODEL_COLORS.get(model, "#95A5A6")
        label = MODEL_LABELS.get(model, model)
        ax3.plot(angles, vals, color=color, linewidth=2.5, label=label)
        ax3.fill(angles, vals, color=color, alpha=0.12)

    ax3.set_title(
        "Radar Chart — Profile Sai số theo Loại Vùng\n"
        "(Diện tích nhỏ hơn = mô hình tốt hơn toàn diện)",
        color=title_color,
        fontsize=12,
        fontweight="bold",
        pad=25,
    )
    ax3.legend(
        facecolor="#1A252F",
        edgecolor="#2C3E50",
        labelcolor=label_color,
        fontsize=9,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.1),
    )

    fig.text(
        0.99,
        0.01,
        "Zone-Aware AH-GNN — Bảo, Tuần 3",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4A5568",
        style="italic",
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"✅ Saved → {out_path}")


def _style_ax(ax, grid_color, label_color):
    ax.grid(color=grid_color, linewidth=0.6, axis="y")
    ax.tick_params(colors=label_color)
    ax.yaxis.label.set_color(label_color)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2C3E50")


def _plot_placeholder(df: pd.DataFrame, out_path: str):
    """Khi không có zone MAE cols — vẽ overall metrics comparison."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    label_color = "#BDC3C7"
    title_color = "#ECF0F1"

    variants = df["variant"].tolist()
    mae_vals = df["MAE"].tolist()
    colors = [MODEL_COLORS.get(v, "#95A5A6") for v in variants]
    labels = [MODEL_LABELS.get(v, v) for v in variants]

    bars = ax.barh(labels, mae_vals, color=colors, alpha=0.85, edgecolor="#0D1117")
    for bar, v in zip(bars, mae_vals):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            ha="left",
            color=label_color,
            fontsize=9,
        )

    ax.set_xlabel("MAE (thấp hơn = tốt hơn)", color=label_color, fontsize=11)
    ax.set_title(
        "Overall MAE Comparison\n(Chưa có zone-level data — cần chạy train.py để có zone MAE)",
        color=title_color,
        fontsize=13,
        fontweight="bold",
    )
    ax.tick_params(colors=label_color)
    ax.xaxis.label.set_color(label_color)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2C3E50")
    ax.grid(color="#1E2D3D", linewidth=0.6, axis="x")

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"✅ Saved (placeholder) → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=os.path.join(OUT_DIR, "zone_stratified_analysis.png")
    )
    args = parser.parse_args()

    print("📂 Loading results...")
    df = load_results()
    print(f"   Variants: {df['variant'].tolist()}")
    print(f"   Columns : {df.columns.tolist()}")

    print("🎨 Plotting zone stratified analysis...")
    plot_zone_stratified(df, args.out)


if __name__ == "__main__":
    main()
