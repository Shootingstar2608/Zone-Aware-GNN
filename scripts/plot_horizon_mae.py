"""
scripts/plot_horizon_mae.py
===========================
Vẽ MAE theo từng bước dự báo t = 1..T_out cho các mô hình (trung bình qua seed)
và đặt cạnh các tham chiếu ngây thơ (persistence, seasonal naive) tính trên
CÙNG test split chrono.

Đọc kết quả mô hình từ CSV của run_multi_seed.py (cần các cột MAE_1..MAE_{T_out});
tham chiếu ngây thơ được tính lại tại chỗ bằng scripts/naive_baselines.py.

Chạy:
  # Hình 4.x trong báo cáo (bản legacy, 2 seed — nguồn của report_chrono_20260930_2109):
  python scripts/plot_horizon_mae.py --runs data/results/multiseed_runs_v4.csv --data-dir data/processed

  # Sau khi chạy lại trên HCM-Sim v1:
  python scripts/plot_horizon_mae.py --runs data/results/multiseed_runs_hcm_sim_v1.csv

Output: paper/figures/horizon_mae_<dataset_id>.pdf (+ .png)   (bản legacy → horizon_mae_legacy.pdf)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.naive_baselines import (  # noqa: E402
    TARGET_FEATURE,
    dataset_tag,
    load_dataset,
    make_predictions,
    steps_per_day,
)
from utils.eval_protocol import (  # noqa: E402
    TRAIN_RATIO,
    VAL_RATIO,
    chronological_split,
    compute_metrics,
)

# Thứ tự màu cố định (palette đã kiểm tra phân biệt được với người mù màu).
# Màu gắn theo THỨ TỰ trong --models, không theo thứ hạng kết quả.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P"]
DISPLAY_NAMES = {
    "lstm": "LSTM",
    "gcn_gru": "GCN-GRU",
    "stgcn": "STGCN",
    "baseline_ahgnn": "AH-GNN",
}
INK, INK_MUTED, GRID = "#3d3c39", "#8a8984", "#e4e3df"


def model_step_mae(runs_csv: str, models: list[str], t_out: int, split_mode: str) -> dict[str, tuple[np.ndarray, int]]:
    df = pd.read_csv(runs_csv)
    if "split_mode" in df.columns:
        df = df[df["split_mode"] == split_mode]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    step_cols = [f"MAE_{t}" for t in range(1, t_out + 1)]
    missing = [c for c in step_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{runs_csv} thiếu cột theo bước (vd. {missing[0]}). "
                         "Cần CSV từ run_multi_seed.py bản có horizon breakdown.")
    out = {}
    for m in models:
        sub = df[df["variant"] == m]
        if sub.empty:
            print(f"⚠️  Không có run nào của '{m}' trong {runs_csv} — bỏ qua.")
            continue
        out[m] = (sub[step_cols].mean().to_numpy(), len(sub))
    return out


def naive_step_mae(data_dir: str) -> tuple[dict[str, np.ndarray], int, str]:
    dataset, meta, manifest = load_dataset(data_dir)
    X = dataset["X"].numpy().astype(np.float64)
    Y = dataset["Y"].numpy().astype(np.float64)
    S, _, t_out = Y.shape
    t_in, n_feat = meta["T_in"], meta["F"]
    f_idx = meta["feature_names"].index(TARGET_FEATURE)
    train_idx, _, test_idx = chronological_split(S, TRAIN_RATIO, VAL_RATIO, purge_gap=t_in + t_out - 1)
    preds = make_predictions(X, Y, train_idx, t_in, n_feat, f_idx, steps_per_day(manifest))
    true_t = torch.from_numpy(Y[test_idx])
    res = {}
    for name in ("persistence", "seasonal_naive"):
        m = compute_metrics(torch.from_numpy(preds[name][test_idx]), true_t)
        res[name] = np.array([m[f"MAE_{t}"] for t in range(1, t_out + 1)])
    return res, t_out, dataset_tag(data_dir, manifest)


def main():
    p = argparse.ArgumentParser(description="MAE theo bước dự báo: mô hình vs tham chiếu ngây thơ.")
    p.add_argument("--runs", required=True, help="CSV kết quả của run_multi_seed.py (có cột MAE_1..MAE_T).")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "processed", "hcm_sim_v1"),
                   help="Dataset dùng để sinh --runs (phải cùng bản, nếu không tập test sẽ lệch).")
    p.add_argument("--models", nargs="+", default=["lstm", "gcn_gru", "zone_full_sinc"],
                   help=f"Tối đa {len(SERIES_COLORS)} mô hình; nên giữ ≤ 3 cho dễ đọc.")
    p.add_argument("--split-mode", default="chrono")
    p.add_argument("--out", default=None, help="Đường dẫn .pdf output (kèm .png cùng tên).")
    args = p.parse_args()

    if len(args.models) > len(SERIES_COLORS):
        raise ValueError(f"Tối đa {len(SERIES_COLORS)} mô hình mỗi hình — tách thành nhiều hình.")

    data_dir = os.path.abspath(args.data_dir)
    naive, t_out, tag = naive_step_mae(data_dir)
    models = model_step_mae(args.runs, args.models, t_out, args.split_mode)
    t = np.arange(1, t_out + 1)

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK_MUTED, "xtick.color": "#52514e", "ytick.color": "#52514e",
    })
    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    for i, m in enumerate(args.models):
        if m not in models:
            continue
        y, n = models[m]
        ax.plot(t, y, color=SERIES_COLORS[i], lw=2, marker=SERIES_MARKERS[i], ms=4, markevery=3,
                label=f"{DISPLAY_NAMES.get(m, m)} ({n} run)")
    ax.plot(t, naive["persistence"], color=INK, lw=2, ls="--", label="Persistence (lặp giá trị cuối)")
    ax.plot(t, naive["seasonal_naive"], color=INK_MUTED, lw=2, ls=":",
            label="Seasonal naive (cùng giờ hôm trước)")

    ymax = max([naive["persistence"].max()] + [v[0].max() for v in models.values()])
    ax.set_xlim(1, t_out + 0.5)
    ax.set_ylim(0, ymax * 1.08)
    ax.set_xticks(sorted({1, *range(4, t_out + 1, 4), t_out}))
    ax.set_xlabel("Bước dự báo $t$ (mỗi bước 15 phút)")
    ax.set_ylabel("MAE (congestion ratio)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8.5, ncol=3)
    fig.tight_layout()

    out = args.out or os.path.join(ROOT, "paper", "figures", f"horizon_mae_{tag}.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=200, bbox_inches="tight")

    # Bảng số ngắn để đối chiếu với text của báo cáo
    probe = [k for k in (1, t_out // 2, t_out) if k >= 1]
    print("  " + f"{'series':<16}" + "".join(f"{'t=' + str(k):>9}" for k in probe))
    for m, (y, n) in models.items():
        print("  " + f"{m:<16}" + "".join(f"{y[k - 1]:>9.4f}" for k in probe))
    for name, y in naive.items():
        print("  " + f"{name:<16}" + "".join(f"{y[k - 1]:>9.4f}" for k in probe))
    lead = models.get(args.models[0])
    if lead is not None:
        worse = np.where(naive["persistence"] > lead[0])[0]
        if len(worse):
            print(f"\n  persistence tốt hơn {args.models[0]} ở các bước t < {worse[0] + 1}")
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
