"""
scripts/naive_baselines.py
==========================
Tham chiếu ngây thơ (không huấn luyện) trên ĐÚNG test split của protocol sạch.

Mục đích: mọi bảng kết quả mô hình cần một mốc "không học gì" để biết mô hình
thực sự học được bao nhiêu. Script này dùng lại nguyên `chronological_split` và
`compute_metrics` trong utils/eval_protocol.py, nên số liệu so sánh trực tiếp
được với output của run_multi_seed.py (cùng split, cùng định nghĩa metric,
cùng MAPE_EPS, cùng đơn vị gốc).

Ba tham chiếu (dự báo congestion_ratio cho T_out bước):
  persistence        — lặp lại giá trị quan sát cuối cùng của cửa sổ đầu vào.
  node_train_mean    — hằng số theo node = trung bình target của node trên train.
  seasonal_naive     — giá trị cùng giờ của chu kỳ trước (mặc định trễ 1 ngày).
                       ⚠️ Dùng quan sát NẰM NGOÀI cửa sổ T_in bước mà mô hình
                       được thấy → KHÔNG phải đối thủ ngang hàng; nó đo lượng
                       thông tin có sẵn trong quá khứ của chuỗi (≈ mức nhiễu
                       nếu chu kỳ lặp lại đúng). Chỉ dùng quan sát TRƯỚC thời
                       điểm dự báo, nên không rò rỉ tương lai.

Chạy:
  python scripts/naive_baselines.py                            # HCM-Sim v1
  python scripts/naive_baselines.py --data-dir data/processed  # bản legacy
  python scripts/naive_baselines.py --season-lag 672           # trễ 1 tuần (cần chuỗi ≥ 2 tuần)

Output:
  data/results/naive_baselines_<dataset_id>.csv   (bản legacy → naive_baselines_legacy.csv)
  — mỗi dòng một baseline; cột MAE/RMSE/MAPE/WAPE và MAE_t, RMSE_t, ... theo
    đúng tên cột của run_multi_seed.py để ghép bảng / vẽ hình chung được.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.eval_protocol import (  # noqa: E402
    TRAIN_RATIO,
    VAL_RATIO,
    chronological_split,
    compute_metrics,
    get_git_commit_hash,
)

TARGET_FEATURE = "congestion_ratio"  # Y trong HCM-Sim là congestion ratio tương lai
DEFAULT_INTERVAL_MIN = 15
BASELINES = ["persistence", "node_train_mean", "seasonal_naive"]


# ══════════════════════════════════════════════════════════════
# 1. Đọc dữ liệu
# ══════════════════════════════════════════════════════════════
def load_dataset(data_dir: str) -> tuple[dict, dict, dict]:
    """Trả về (dataset, meta, manifest). manifest rỗng nếu là bản legacy."""
    dataset = torch.load(os.path.join(data_dir, "graph_dataset.pt"), weights_only=False)
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    manifest_path = os.path.join(data_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    return dataset, meta, manifest


def dataset_tag(data_dir: str, manifest: dict) -> str:
    """Tên ngắn của dataset: dataset_id trong manifest; bản legacy (không manifest) → 'legacy'."""
    if manifest.get("dataset_id"):
        return manifest["dataset_id"]
    base = os.path.basename(os.path.normpath(data_dir))
    return "legacy" if base == "processed" else base


def steps_per_day(manifest: dict) -> int:
    """Số bước mỗi ngày, lấy interval từ manifest (bản legacy không có → 15 phút)."""
    interval = manifest.get("traffic_meta", {}).get("interval_min", DEFAULT_INTERVAL_MIN)
    return 24 * 60 // int(interval)


# ══════════════════════════════════════════════════════════════
# 2. Dựng lại chuỗi gốc từ các cửa sổ trượt
# ══════════════════════════════════════════════════════════════
def reconstruct_series(X: np.ndarray, Y: np.ndarray, t_in: int, n_feat: int, f_idx: int) -> np.ndarray:
    """
    Dựng lại chuỗi c[t] (T_total, N) của feature f_idx từ X và Y.

    Layout của X (xem build_graph.py): x_window (T_in, N, F) → transpose(1,0,2)
    → reshape(N, T_in*F), tức chỉ số cột = t * F + f.
    Cửa sổ i: đầu vào c[i : i+T_in], target c[i+T_in : i+T_in+T_out].
    """
    S, N, _ = X.shape
    t_out = Y.shape[2]
    Xr = X.reshape(S, N, t_in, n_feat)
    c = np.full((S + t_in + t_out - 1, N), np.nan, dtype=np.float64)
    c[:S] = Xr[:, :, 0, f_idx]
    for i in range(S):
        c[i + t_in: i + t_in + t_out] = Y[i].T

    # Kiểm tra layout: bước cuối của cửa sổ i phải trùng c[i+T_in-1].
    # Sai layout (vd. feature-major) sẽ làm phép so này lệch lớn → báo lỗi to.
    last_obs = Xr[:, :, -1, f_idx]
    max_dev = np.nanmax(np.abs(last_obs - c[t_in - 1: t_in - 1 + S]))
    if not max_dev < 1e-4:
        raise ValueError(
            f"Layout X không khớp giả định (lệch tối đa {max_dev:.3g}). "
            "Kiểm tra lại cách build_graph.py reshape cửa sổ."
        )
    return c


# ══════════════════════════════════════════════════════════════
# 3. Ba tham chiếu ngây thơ
# ══════════════════════════════════════════════════════════════
def make_predictions(X, Y, train_idx, t_in, n_feat, f_idx, season_lag) -> dict[str, np.ndarray]:
    S, N, t_out = Y.shape
    Xr = X.reshape(S, N, t_in, n_feat)
    preds = {}

    # (a) persistence
    preds["persistence"] = np.repeat(Xr[:, :, -1, f_idx][:, :, None], t_out, axis=2)

    # (b) hằng số theo node, chỉ học từ train
    node_mean = Y[train_idx].mean(axis=(0, 2))  # (N,)
    preds["node_train_mean"] = np.broadcast_to(node_mean[None, :, None], Y.shape).copy()

    # (c) seasonal naive: target tại τ = i+T_in+k  ←  c[τ - season_lag]
    c = reconstruct_series(X, Y, t_in, n_feat, f_idx)
    sn = np.full(Y.shape, np.nan, dtype=np.float64)
    origin = np.arange(S) + t_in - 1  # thời điểm quan sát cuối của cửa sổ i
    for k in range(t_out):
        tau = np.arange(S) + t_in + k
        src = tau - season_lag
        ok = src >= 0
        # Không dùng quan sát sau thời điểm dự báo (season_lag phải ≥ T_out)
        assert np.all(src[ok] <= origin[ok]), "season_lag < T_out sẽ nhìn thấy tương lai"
        sn[ok, :, k] = c[src[ok]]
    preds["seasonal_naive"] = sn
    return preds


# ══════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Tham chiếu ngây thơ trên test split chrono.")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "processed", "hcm_sim_v1"),
                   help="Thư mục chứa graph_dataset.pt + meta.json (mặc định HCM-Sim v1).")
    p.add_argument("--season-lag", type=int, default=None,
                   help="Độ trễ cho seasonal naive, tính bằng số bước (mặc định = 1 ngày).")
    p.add_argument("--out", default=None, help="Đường dẫn CSV output.")
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    dataset, meta, manifest = load_dataset(data_dir)
    X = dataset["X"].numpy().astype(np.float64)
    Y = dataset["Y"].numpy().astype(np.float64)
    S, N, t_out = Y.shape
    t_in, n_feat = meta["T_in"], meta["F"]
    assert t_out == meta["T_out"] and S == meta["S"], "meta.json không khớp tensor"
    f_idx = meta["feature_names"].index(TARGET_FEATURE)
    season_lag = args.season_lag or steps_per_day(manifest)
    if season_lag < t_out:
        raise ValueError(f"--season-lag ({season_lag}) phải ≥ T_out ({t_out}).")

    purge_gap = t_in + t_out - 1
    train_idx, val_idx, test_idx = chronological_split(S, TRAIN_RATIO, VAL_RATIO, purge_gap=purge_gap)
    preds = make_predictions(X, Y, train_idx, t_in, n_feat, f_idx, season_lag)

    dataset_id = dataset_tag(data_dir, manifest)
    commit = get_git_commit_hash()
    true_t = torch.from_numpy(Y[test_idx])
    rows = []
    for name in BASELINES:
        pred = preds[name][test_idx]
        if np.isnan(pred).any():
            print(f"⚠️  {name}: thiếu lịch sử cho một phần test (season_lag quá dài) — bỏ qua.")
            continue
        m = compute_metrics(torch.from_numpy(pred), true_t)
        rows.append({"variant": name, "dataset_id": dataset_id, "split_mode": "chrono",
                     "n_test": len(test_idx), "season_lag": season_lag if name == "seasonal_naive" else "",
                     "commit": commit, **m})

    df = pd.DataFrame(rows)
    out = args.out or os.path.join(ROOT, "data", "results", f"naive_baselines_{dataset_id}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    # Tóm tắt ra màn hình
    if "dow" in dataset:
        dows = sorted(set(dataset["dow"][test_idx].tolist()))
        print(f"  dow trong tập test : {dows}  (0 = thứ Hai)")
    print(f"  dataset            : {dataset_id}  |  S={S}, T_in={t_in}, T_out={t_out}")
    print(f"  test windows       : {test_idx[0]}..{test_idx[-1]} (n={len(test_idx)})")
    print(f"  seasonal lag       : {season_lag} bước\n")
    print(f"  {'baseline':<18}{'MAE':>8}{'RMSE':>8}{'MAPE':>8}{'WAPE':>8}{'MAE_1':>8}{f'MAE_{t_out}':>8}")
    for r in rows:
        print(f"  {r['variant']:<18}{r['MAE']:>8.4f}{r['RMSE']:>8.4f}{r['MAPE']:>8.2f}"
              f"{r['WAPE']:>8.2f}{r['MAE_1']:>8.4f}{r[f'MAE_{t_out}']:>8.4f}")
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
