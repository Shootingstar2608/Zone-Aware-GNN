"""
visualize_adjacency.py
======================
Task Bảo — Tuần 3

Visualize sự thay đổi của ZoneAwareAdjacency tại các thời điểm khác nhau:
  - 8h sáng (rush_morning, time_idx=2)
  - 17h chiều (rush_evening, time_idx=3)
  - 23h đêm (night, time_idx=0)
  - 11h trưa (normal, time_idx=1)

Output: data/results/adj_visualization.png

Dùng: python scripts/visualize_adjacency.py
      python scripts/visualize_adjacency.py --model_path data/results/zone_full_best.pt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.zone_aware_gnn import ZoneAwareAHGNN

# ─── CONFIG ──────────────────────────────────────────────────────────────────
META_PATH = "data/processed/meta.json"
ZONE_PATH = "data/raw/zone_labels.csv"
DATASET_PATH = "data/processed/graph_dataset.pt"
OUT_DIR = "data/results"

TIME_SLOTS = [
    {"label": "8h sáng\n(Cao điểm đi làm)", "time_idx": 2, "color": "#E74C3C"},
    {"label": "17h chiều\n(Cao điểm tan tầm)", "time_idx": 3, "color": "#E67E22"},
    {"label": "11h trưa\n(Bình thường)", "time_idx": 1, "color": "#27AE60"},
    {"label": "23h đêm\n(Ban đêm)", "time_idx": 0, "color": "#2980B9"},
]

# Colormap tối giản cho paper
CMAP = LinearSegmentedColormap.from_list(
    "adj_cmap", ["#FFFFFF", "#3498DB", "#1A252F"], N=256
)
# ─────────────────────────────────────────────────────────────────────────────


def load_meta_and_zone():
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    nodes = meta["nodes"]
    zone_types = meta["zone_types"]
    N, K = meta["N"], meta["K"]

    # Load zone labels
    import pandas as pd

    Z = np.zeros((N, K), dtype=np.float32)
    if os.path.exists(ZONE_PATH):
        df_z = pd.read_csv(ZONE_PATH, index_col="node")
        for i, node in enumerate(nodes):
            if node in df_z.index:
                Z[i] = df_z.loc[node, zone_types].values.astype(float)

    return meta, nodes, zone_types, Z


def build_model_from_meta(meta):
    N, K = meta["N"], meta["K"]
    F = meta["F"]
    T_in = meta["T_in"]
    T_out = meta["T_out"]
    return ZoneAwareAHGNN(
        num_nodes=N,
        num_zones=K,
        in_channels=T_in * F,
        hidden_channels=64,
        out_channels=T_out,
        node_embed_dim=32,
        zone_embed_dim=16,
        num_time_labels=4,
        num_layers=2,
    )


@torch.no_grad()
def get_adjacency_at_time(model, Z_tensor, time_idx: int) -> np.ndarray:
    """Lấy ma trận kề tại 1 thời điểm cụ thể."""
    model.eval()
    t = torch.tensor([time_idx], dtype=torch.long)
    z_embed = model.zone_emb(Z_tensor)  # (N, d_z)
    A = model.adj_module(z_embed, t, A_static=None)  # (1, N, N)
    return A[0].cpu().numpy()


def plot_adjacency_comparison(matrices, time_slots, nodes, zone_types, Z, out_path):
    """
    Vẽ heatmap ma trận kề 4 thời điểm + diff plot + bar chart độ mạnh kết nối.
    """
    N = len(nodes)
    fig = plt.figure(figsize=(22, 20), facecolor="#0D1117")

    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        top=0.92,
        bottom=0.08,
        left=0.06,
        right=0.97,
    )

    title_color = "#ECF0F1"
    label_color = "#BDC3C7"

    fig.suptitle(
        "Phân tích Ma trận Kề Động (ZoneAwareAdjacency)\n"
        "So sánh cường độ kết nối giữa các khu vực theo khung giờ",
        color=title_color,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    short_nodes = [n.replace(" ", "\n") for n in nodes]

    # ── Hàng 1: 4 heatmap ──────────────────────────────────────────────────
    axes_heat = []
    for col, (slot, mat) in enumerate(zip(time_slots, matrices)):
        ax = fig.add_subplot(gs[0, col])
        axes_heat.append(ax)
        ax.set_facecolor("#0D1117")

        im = ax.imshow(mat, cmap=CMAP, vmin=0, vmax=mat.max(), aspect="auto")
        ax.set_title(
            slot["label"], color=slot["color"], fontsize=11, fontweight="bold", pad=8
        )

        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(short_nodes, rotation=90, fontsize=5.5, color=label_color)
        ax.set_yticklabels(short_nodes, fontsize=5.5, color=label_color)
        ax.tick_params(colors=label_color, length=2)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2C3E50")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=label_color, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=label_color)

    # ── Hàng 2: Diff heatmap (8h vs 17h) + (8h vs đêm) + zone bar ─────────
    # Diff 1: 8h sáng vs 17h chiều
    ax_diff1 = fig.add_subplot(gs[1, :2])
    ax_diff1.set_facecolor("#0D1117")
    diff1 = matrices[0] - matrices[1]  # rush_morning - rush_evening
    vmax_d = max(abs(diff1).max(), 0.01)
    im_d1 = ax_diff1.imshow(
        diff1, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d, aspect="auto"
    )
    ax_diff1.set_title(
        "Δ Kết nối: 8h sáng − 17h chiều\n"
        "(Đỏ = mạnh hơn buổi sáng, Xanh = mạnh hơn buổi chiều)",
        color=title_color,
        fontsize=10,
        fontweight="bold",
    )
    ax_diff1.set_xticks(range(N))
    ax_diff1.set_yticks(range(N))
    ax_diff1.set_xticklabels(short_nodes, rotation=90, fontsize=5.5, color=label_color)
    ax_diff1.set_yticklabels(short_nodes, fontsize=5.5, color=label_color)
    ax_diff1.tick_params(colors=label_color, length=2)
    cbar_d1 = plt.colorbar(im_d1, ax=ax_diff1, fraction=0.023, pad=0.04)
    cbar_d1.ax.yaxis.set_tick_params(color=label_color, labelsize=7)
    plt.setp(cbar_d1.ax.yaxis.get_ticklabels(), color=label_color)

    # Diff 2: 8h sáng vs đêm
    ax_diff2 = fig.add_subplot(gs[1, 2:])
    ax_diff2.set_facecolor("#0D1117")
    diff2 = matrices[0] - matrices[3]  # rush_morning - night
    vmax_d2 = max(abs(diff2).max(), 0.01)
    im_d2 = ax_diff2.imshow(
        diff2, cmap="RdBu_r", vmin=-vmax_d2, vmax=vmax_d2, aspect="auto"
    )
    ax_diff2.set_title(
        "Δ Kết nối: 8h sáng − 23h đêm\n"
        "(Chứng minh kết nối thay đổi theo thời gian — Non-I)",
        color=title_color,
        fontsize=10,
        fontweight="bold",
    )
    ax_diff2.set_xticks(range(N))
    ax_diff2.set_yticks(range(N))
    ax_diff2.set_xticklabels(short_nodes, rotation=90, fontsize=5.5, color=label_color)
    ax_diff2.set_yticklabels(short_nodes, fontsize=5.5, color=label_color)
    ax_diff2.tick_params(colors=label_color, length=2)
    cbar_d2 = plt.colorbar(im_d2, ax=ax_diff2, fraction=0.023, pad=0.04)
    cbar_d2.ax.yaxis.set_tick_params(color=label_color, labelsize=7)
    plt.setp(cbar_d2.ax.yaxis.get_ticklabels(), color=label_color)

    # ── Hàng 3: Bar chart — độ mạnh kết nối trung bình theo zone ───────────
    ax_bar = fig.add_subplot(gs[2, :])
    ax_bar.set_facecolor("#0D1117")
    ax_bar.spines["bottom"].set_color("#2C3E50")
    ax_bar.spines["left"].set_color("#2C3E50")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(colors=label_color)

    # Tính mean outgoing strength theo node, group theo zone
    zone_strengths = {slot["label"].split("\n")[0]: [] for slot in time_slots}
    zone_labels_plot = zone_types

    x = np.arange(len(zone_labels_plot))
    bar_w = 0.18
    colors_bar = [s["color"] for s in time_slots]

    for ci, (slot, mat) in enumerate(zip(time_slots, matrices)):
        out_strength = mat.sum(axis=1)  # [N] tổng kết nối đi ra
        means = []
        for k, zone in enumerate(zone_types):
            node_mask = Z[:, k] == 1
            if node_mask.sum() > 0:
                means.append(out_strength[node_mask].mean())
            else:
                means.append(0.0)
        offset = (ci - 1.5) * bar_w
        bars = ax_bar.bar(
            x + offset,
            means,
            bar_w,
            label=slot["label"].replace("\n", " "),
            color=slot["color"],
            alpha=0.85,
            edgecolor="#0D1117",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f"🏷 {z}" for z in zone_labels_plot],
        color=label_color,
        fontsize=10,
    )
    ax_bar.set_ylabel(
        "Tổng cường độ kết nối đi ra (mean)", color=label_color, fontsize=10
    )
    ax_bar.set_title(
        "Cường độ kết nối trung bình theo loại vùng & khung giờ\n"
        "(Chứng minh Zone-Awareness: vùng khác nhau có pattern kết nối khác nhau)",
        color=title_color,
        fontsize=11,
        fontweight="bold",
    )
    ax_bar.yaxis.label.set_color(label_color)
    ax_bar.tick_params(axis="y", colors=label_color)
    ax_bar.yaxis.set_tick_params(labelcolor=label_color)
    legend = ax_bar.legend(
        fontsize=9,
        facecolor="#1A252F",
        edgecolor="#2C3E50",
        labelcolor=label_color,
        loc="upper right",
    )

    # Watermark
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to .pt checkpoint (optional — dùng random weights nếu không có)",
    )
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "adj_visualization.png"))
    args = parser.parse_args()

    print("📂 Loading meta & zone labels...")
    meta, nodes, zone_types, Z = load_meta_and_zone()
    Z_tensor = torch.FloatTensor(Z)

    print("🔧 Building model...")
    model = build_model_from_meta(meta)

    if args.model_path and os.path.exists(args.model_path):
        print(f"   Loading weights from {args.model_path}")
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print("   ✅ Weights loaded")
    else:
        # Thử tìm bất kỳ .pt nào trong results/
        pt_files = [f for f in os.listdir(OUT_DIR) if f.endswith(".pt")]
        if pt_files:
            pt_path = os.path.join(OUT_DIR, pt_files[0])
            print(f"   Auto-found: {pt_path}")
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)
        else:
            print("   ⚠️  Không tìm thấy checkpoint → dùng random weights")
            print("   (Chạy train.py trước để có kết quả thực tế)")

    print("📊 Computing adjacency matrices...")
    matrices = []
    for slot in TIME_SLOTS:
        A = get_adjacency_at_time(model, Z_tensor, slot["time_idx"])
        matrices.append(A)
        print(
            f"   {slot['label'].split(chr(10))[0]}: "
            f"mean={A.mean():.4f}, max={A.max():.4f}, "
            f"non-zero={(A > 0.01).sum()}"
        )

    print("🎨 Plotting...")
    plot_adjacency_comparison(matrices, TIME_SLOTS, nodes, zone_types, Z, args.out)


if __name__ == "__main__":
    main()
