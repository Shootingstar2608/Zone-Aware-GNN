"""
scripts/eval_non_iid.py
========================
Task Người 4 — Evaluation / Docs

Tích hợp các bộ partition từ benchmark/partition_gen.py vào pipeline huấn luyện.
Đánh giá mô hình Zone-Aware vs Baseline thay đổi như thế nào khi độ lệch
Non-IID tăng dần.

Output:
  data/results/non_iid_eval.csv         — bảng kết quả đầy đủ
  data/results/non_iid_performance.png  — biểu đồ Performance vs Heterogeneity

Chạy:
  python scripts/eval_non_iid.py
  python scripts/eval_non_iid.py --alphas 0.1 0.5 1.0 5.0 --models zone_full gcn_gru lstm
  python scripts/eval_non_iid.py --skip_train  # chỉ vẽ biểu đồ từ CSV đã có
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.train import (
    ABLATION_VARIANTS,
    BASELINE_NAMES,
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LR,
    PATIENCE,
    ZONE_AWARE_VARIANTS,
    build_model,
    compute_metrics,
    compute_zone_stratified_metrics,
    evaluate,
    set_seed,
    train_one_epoch,
)
from benchmark.partition_gen import (
    assert_no_leakage,
    check_fingerprint,
    dataset_fingerprint,
    gini,
    load_meta,
    load_partitions,
    overlap_gap,
    quantity_skew,
    save_partitions,
    build_record,
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_PATH = "data/processed/graph_dataset.pt"
META_PATH = "data/processed/meta.json"
PARTITION_PATH = "data/partitions/partitions_meta.json"
OUT_DIR = "data/results"

DEFAULT_ALPHAS = [0.1, 0.5, 1.0, 5.0]
DEFAULT_MODELS = ["zone_full", "zone_full_tc", "gcn_gru", "stgcn", "lstm"]
SEEDS = [42]  # dùng 1 seed; Người 3 (Khoa) sẽ sweep 5 seeds
LAMBDA_COS = 0.1

# Màu sắc cho biểu đồ
MODEL_COLORS = {
    "zone_full": "#27AE60",
    "zone_full_tc": "#2980B9",
    "zone_full_sinc": "#9B59B6",
    "baseline_ahgnn": "#E74C3C",
    "gcn_gru": "#E67E22",
    "stgcn": "#F39C12",
    "lstm": "#95A5A6",
}
MODEL_LABELS = {
    "zone_full": "Zone-Full ★",
    "zone_full_tc": "Zone-TC (Tân)",
    "zone_full_sinc": "Zone-Sinc (Bảo)",
    "baseline_ahgnn": "AH-GNN (baseline)",
    "gcn_gru": "GCN-GRU",
    "stgcn": "STGCN",
    "lstm": "LSTM",
}
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# 1. SINH PARTITION (nếu chưa có)
# ══════════════════════════════════════════════════════════════
def generate_quantity_partitions(
    dataset: dict, meta: dict, alphas: list[float], seed: int = 42
) -> list[dict]:
    """
    Sinh partition Quantity Skew cho mỗi alpha trong `alphas`.
    Lưu vào PARTITION_PATH. Trả về list records.
    """
    S = meta["S"]
    N = meta["N"]
    gap = overlap_gap(meta)
    records = []

    for alpha in alphas:
        partition_id = f"quantity_skew_a{alpha}_s{seed}"
        mask, stats = quantity_skew(S, N, alpha=alpha, seed=seed)

        # Từ mask → train/val/test index (theo node trung bình)
        # Dùng union: sample s vào train nếu ít nhất 1 node có dữ liệu tại s
        has_data = mask.any(axis=1)  # (S,) bool
        valid_idx = np.where(has_data)[0]

        n_tr = int(len(valid_idx) * 0.7)
        n_va = int(len(valid_idx) * 0.1)
        train_idx = valid_idx[:n_tr]
        val_idx = valid_idx[n_tr : n_tr + n_va]
        test_idx = valid_idx[n_tr + n_va :]

        # Kiểm tra leakage
        try:
            assert_no_leakage(train_idx, test_idx, gap)
        except ValueError as e:
            print(f"  ⚠️  Partition {partition_id}: {e} — purging border samples")
            test_idx = test_idx[test_idx > train_idx[-1] + gap]

        splits = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        }

        rec = build_record(
            partition_id=partition_id,
            scenario="quantity_skew",
            params={"alpha": alpha, "mode": "fixed_coverage", "c_bar": 0.5},
            seed=seed,
            meta=meta,
            mask=mask,
            splits=splits,
            stats=stats,
        )
        records.append(rec)
        print(
            f"  ✓ partition {partition_id}: "
            f"gini={stats['gini']:.3f}, "
            f"zero_nodes={stats['n_zero_nodes']}, "
            f"train={len(train_idx)}/val={len(val_idx)}/test={len(test_idx)}"
        )

    save_partitions(records, PARTITION_PATH)
    return records


# ══════════════════════════════════════════════════════════════
# 2. TRAIN VÀ EVALUATE TRÊN 1 PARTITION
# ══════════════════════════════════════════════════════════════
def train_on_partition(
    variant: str,
    meta: dict,
    dataset: dict,
    partition: dict,
    seed: int = 42,
) -> dict:
    """
    Train model `variant` trên partition đã cho, trả về metrics + stats.
    """
    set_seed(seed)

    A = dataset["A"].to(DEVICE)
    Z = dataset["Z"].to(DEVICE)
    X = dataset["X"]  # (S, N, F*T_in)
    Y = dataset["Y"]  # (S, N, T_out)
    TL = dataset["time_labels"]

    splits = partition["splits"]
    train_idx = torch.tensor(splits["train"], dtype=torch.long)
    val_idx = torch.tensor(splits["val"], dtype=torch.long)
    test_idx = torch.tensor(splits["test"], dtype=torch.long)

    # Nếu partition quá nhỏ để train → skip
    if len(train_idx) < BATCH_SIZE:
        return {
            "skip": True,
            "reason": f"train_size={len(train_idx)} < batch_size={BATCH_SIZE}",
        }

    train_ds = TensorDataset(X[train_idx], Y[train_idx], TL[train_idx])
    val_ds = TensorDataset(X[val_idx], Y[val_idx], TL[val_idx])
    test_ds = TensorDataset(X[test_idx], Y[test_idx], TL[test_idx])

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    ablation_cfg = ABLATION_VARIANTS.get(variant, (False, False, False))
    use_zone_emb, use_zone_weight, use_zone_adj = ablation_cfg
    model = build_model(variant, meta, use_zone_emb, use_zone_weight, use_zone_adj)
    model = model.to(DEVICE)

    lambda_cos = LAMBDA_COS if variant in ZONE_AWARE_VARIANTS else 0.0
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_mae = float("inf")
    patience_cnt = 0
    best_state = None

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(
            model, train_loader, optimizer, A, Z, DEVICE, lambda_cos=lambda_cos
        )
        val_preds, val_trues = evaluate(model, val_loader, A, Z, DEVICE)
        val_metrics = compute_metrics(val_preds, val_trues)
        scheduler.step(val_metrics["MAE"])

        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    test_preds, test_trues = evaluate(model, test_loader, A, Z, DEVICE)
    test_metrics = compute_metrics(test_preds, test_trues)
    zone_metrics = compute_zone_stratified_metrics(
        test_preds, test_trues, Z.cpu(), meta["zone_types"]
    )

    stats = partition.get("stats", {})
    return {
        "variant": variant,
        "partition_id": partition["partition_id"],
        "alpha": partition["params"].get("alpha", float("nan")),
        "gini": stats.get("gini", float("nan")),
        "n_zero_nodes": stats.get("n_zero_nodes", 0),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
        **test_metrics,
        **zone_metrics,
    }


# ══════════════════════════════════════════════════════════════
# 3. VẼ BIỂU ĐỒ Performance vs Heterogeneity
# ══════════════════════════════════════════════════════════════
def plot_non_iid(df: pd.DataFrame, out_path: str) -> None:
    """
    3 panel:
      A. MAE vs Gini coefficient  (trục X = mức độ Non-IID)
      B. MAPE vs Alpha (Dirichlet concentration)
      C. Heatmap: MAE theo (model x alpha)
    """
    fig = plt.figure(figsize=(20, 18), facecolor="#0D1117")
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        top=0.92,
        bottom=0.07,
        left=0.07,
        right=0.97,
    )

    bg = "#0D1117"
    title_color = "#ECF0F1"
    label_color = "#BDC3C7"
    grid_color = "#1E2D3D"

    fig.suptitle(
        "Performance vs. Non-IID Heterogeneity\n"
        "Đánh giá mô hình Zone-Aware khi độ lệch dữ liệu tăng dần (Quantity Skew)",
        color=title_color,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    models_in_df = [m for m in DEFAULT_MODELS if m in df["variant"].unique()]

    # ── Panel A: MAE vs Gini ─────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _style(ax_a, bg, grid_color, label_color)
    for variant in models_in_df:
        sub = df[df["variant"] == variant].sort_values("gini")
        if sub.empty:
            continue
        color = MODEL_COLORS.get(variant, "#95A5A6")
        label = MODEL_LABELS.get(variant, variant)
        ax_a.plot(
            sub["gini"],
            sub["MAE"],
            "o-",
            color=color,
            label=label,
            linewidth=2.2,
            markersize=7,
        )
        # Shade nếu có nhiều seeds
        if "MAE_std" in sub.columns:
            ax_a.fill_between(
                sub["gini"],
                sub["MAE"] - sub["MAE_std"],
                sub["MAE"] + sub["MAE_std"],
                color=color,
                alpha=0.15,
            )

    ax_a.set_xlabel("Gini Coefficient (0=đều, →1=lệch)", color=label_color, fontsize=11)
    ax_a.set_ylabel("MAE (thấp hơn = tốt hơn)", color=label_color, fontsize=11)
    ax_a.set_title(
        "MAE vs Mức độ Lệch Dữ liệu (Gini)\n"
        "Mô hình Zone-Aware suy giảm chậm hơn khi Non-IID tăng",
        color=title_color,
        fontsize=11,
        fontweight="bold",
    )
    ax_a.legend(
        facecolor="#1A252F", edgecolor="#2C3E50", labelcolor=label_color, fontsize=9
    )

    # ── Panel B: MAPE vs Alpha ───────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _style(ax_b, bg, grid_color, label_color)
    alphas_sorted = sorted(df["alpha"].dropna().unique())
    for variant in models_in_df:
        sub = df[df["variant"] == variant].groupby("alpha")["MAPE"].mean().reset_index()
        sub = sub.sort_values("alpha")
        color = MODEL_COLORS.get(variant, "#95A5A6")
        label = MODEL_LABELS.get(variant, variant)
        ax_b.plot(
            sub["alpha"],
            sub["MAPE"],
            "s--",
            color=color,
            label=label,
            linewidth=2.2,
            markersize=7,
        )

    ax_b.set_xscale("log")
    ax_b.set_xlabel(
        "Alpha Dirichlet (nhỏ=lệch mạnh, lớn=đều)", color=label_color, fontsize=11
    )
    ax_b.set_ylabel("MAPE (%)", color=label_color, fontsize=11)
    ax_b.set_title(
        "MAPE vs Alpha Dirichlet\n" "Alpha nhỏ → Non-IID mạnh → mô hình yếu kém hơn",
        color=title_color,
        fontsize=11,
        fontweight="bold",
    )
    ax_b.legend(
        facecolor="#1A252F", edgecolor="#2C3E50", labelcolor=label_color, fontsize=9
    )

    # ── Panel C: Heatmap MAE (model x alpha) ────────────────────────────────
    ax_c = fig.add_subplot(gs[1, :])
    _style(ax_c, bg, grid_color, label_color)

    pivot = df.groupby(["variant", "alpha"])["MAE"].mean().unstack(level="alpha")
    pivot = pivot.reindex([m for m in DEFAULT_MODELS if m in pivot.index])

    if not pivot.empty:
        import matplotlib.colors as mcolors

        cmap = plt.cm.RdYlGn_r
        im = ax_c.imshow(
            pivot.values,
            cmap=cmap,
            aspect="auto",
            vmin=pivot.values.min(),
            vmax=pivot.values.max(),
        )

        ax_c.set_xticks(range(len(pivot.columns)))
        ax_c.set_xticklabels(
            [f"α={a}" for a in pivot.columns],
            color=label_color,
            fontsize=11,
        )
        ax_c.set_yticks(range(len(pivot.index)))
        ax_c.set_yticklabels(
            [MODEL_LABELS.get(m, m) for m in pivot.index],
            color=label_color,
            fontsize=11,
        )

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    txt_color = "white" if v > (pivot.values.max() * 0.6) else "#1A252F"
                    ax_c.text(
                        j,
                        i,
                        f"{v:.4f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=txt_color,
                        fontweight="bold",
                    )

        cbar = plt.colorbar(im, ax=ax_c, fraction=0.015, pad=0.02)
        cbar.set_label("MAE", color=label_color, fontsize=10)
        cbar.ax.yaxis.set_tick_params(color=label_color, labelsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=label_color)

    ax_c.set_title(
        "Heatmap MAE: Model × Alpha (Đỏ=xấu, Xanh=tốt)\n"
        "Zone-Aware giữ MAE thấp ngay cả khi α nhỏ (Non-IID mạnh)",
        color=title_color,
        fontsize=12,
        fontweight="bold",
    )

    fig.text(
        0.99,
        0.01,
        "Zone-Aware AH-GNN — Người 4, Tuần 3",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4A5568",
        style="italic",
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close()
    print(f"✅ Plot saved → {out_path}")


def _style(ax, bg, grid_color, label_color):
    ax.set_facecolor(bg)
    ax.grid(color=grid_color, linewidth=0.6, axis="both")
    ax.tick_params(colors=label_color)
    ax.xaxis.label.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2C3E50")


# ══════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Eval Non-IID benchmark")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
        help="Danh sách alpha Dirichlet cần thử",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Danh sách variant cần đánh giá",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Bỏ qua train, chỉ vẽ biểu đồ từ CSV đã có",
    )
    parser.add_argument("--out_csv", default=os.path.join(OUT_DIR, "non_iid_eval.csv"))
    parser.add_argument(
        "--out_plot", default=os.path.join(OUT_DIR, "non_iid_performance.png")
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("data/partitions", exist_ok=True)

    # ── Load dataset + meta ──────────────────────────────────────────────────
    print(f"📂 Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset not found. Run: python scripts/build_graph.py")
        return

    dataset = torch.load(DATASET_PATH, weights_only=False)
    meta = load_meta(META_PATH)
    print(
        f"   N={meta['N']}, S={meta['S']}, T_in={meta['T_in']}, T_out={meta['T_out']}"
    )

    if args.skip_train:
        # Chỉ vẽ từ CSV đã có
        if not os.path.exists(args.out_csv):
            print(f"❌ CSV không tồn tại: {args.out_csv}")
            return
        df = pd.read_csv(args.out_csv)
        print(f"   Loaded {len(df)} rows from {args.out_csv}")
        plot_non_iid(df, args.out_plot)
        return

    # ── Sinh partition ───────────────────────────────────────────────────────
    print(f"\n📐 Generating {len(args.alphas)} quantity-skew partitions...")
    partitions = generate_quantity_partitions(
        dataset, meta, args.alphas, seed=args.seed
    )

    # ── Verify fingerprint ───────────────────────────────────────────────────
    fp_now = dataset_fingerprint(DATASET_PATH)
    for rec in partitions:
        if rec["dataset_fingerprint"] != fp_now:
            print(f"⚠️  Fingerprint mismatch for {rec['partition_id']} — skipping")

    # ── Train + evaluate ─────────────────────────────────────────────────────
    all_rows = []
    total = len(partitions) * len(args.models)
    done = 0

    print(
        f"\n🚀 Training {len(args.models)} models × {len(partitions)} partitions "
        f"= {total} experiments (device={DEVICE})\n"
    )

    for partition in partitions:
        alpha = partition["params"]["alpha"]
        gini_val = partition["stats"]["gini"]
        print(f"\n{'─'*55}")
        print(f"  Partition: {partition['partition_id']}")
        print(
            f"  Alpha={alpha}, Gini={gini_val:.3f}, "
            f"zero_nodes={partition['stats']['n_zero_nodes']}"
        )
        print(f"{'─'*55}")

        for variant in args.models:
            done += 1
            print(f"\n  [{done}/{total}] {variant} | alpha={alpha}")
            row = train_on_partition(variant, meta, dataset, partition, seed=args.seed)

            if row.get("skip"):
                print(f"  ⚠️  Skipped: {row['reason']}")
                continue

            print(
                f"  → MAE={row['MAE']:.4f} | RMSE={row['RMSE']:.4f} "
                f"| MAPE={row['MAPE']:.2f}% | elapsed={row['elapsed_s']}s"
            )
            all_rows.append(row)

    if not all_rows:
        print("❌ Không có kết quả nào. Kiểm tra lại partition size.")
        return

    # ── Lưu CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\n✅ Results saved → {args.out_csv}")
    print(f"\n{'='*60}")
    print("  SUMMARY — MAE theo Alpha & Model")
    print(f"{'='*60}")
    pivot = df.groupby(["variant", "alpha"])["MAE"].mean().unstack("alpha")
    print(pivot.to_string())

    # ── Vẽ biểu đồ ───────────────────────────────────────────────────────────
    print(f"\n🎨 Plotting...")
    plot_non_iid(df, args.out_plot)
    print(f"\n✅ Done! Outputs:")
    print(f"   {args.out_csv}")
    print(f"   {args.out_plot}")


if __name__ == "__main__":
    main()
