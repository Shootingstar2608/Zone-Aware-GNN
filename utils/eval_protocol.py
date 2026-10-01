"""
utils/eval_protocol.py
=======================
Shared evaluation protocol cho Zone-Aware-GNN.

Mọi script (train.py, eval_non_iid.py, run_multi_seed.py) PHẢI dùng
module này để đảm bảo:
  1. Chronological split + purge gap (không temporal leakage)
  2. Normalization fit CHỈ trên train
  3. Metrics tính trên dữ liệu đã inverse-transform
  4. Zone-stratified metrics

Không thay đổi kiến trúc model. Giữ tương thích X/Y/A/Z interface.
"""

from __future__ import annotations

import subprocess
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from utils.normalizer import ZScoreNormalizer

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
PURGE_GAP_DEFAULT = 35  # = T_in(12) + T_out(24) - 1

MAPE_EPS = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHRONOLOGICAL SPLIT WITH PURGE GAP
# ══════════════════════════════════════════════════════════════════════════════
def chronological_split(
    S: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    purge_gap: int = PURGE_GAP_DEFAULT,
) -> tuple[list[int], list[int], list[int]]:
    """
    Chia S mẫu theo thứ tự thời gian với purge gap giữa các tập.

    Purge gap = T_in + T_out - 1 loại bỏ các mẫu có cửa sổ trượt
    chồng lấp với tập liền trước, ngăn model thấy future data.

    Sơ đồ (S=637, T_in=12, T_out=24, gap=35):
      |←── train=445 ──→|← 35 →|← val=64 →|← 35 →|←── test=58 ──→|
       idx 0          444       480       543       578            636

    Args:
        S          : Tổng số mẫu.
        train_ratio: Tỷ lệ train (0.7).
        val_ratio  : Tỷ lệ val   (0.1).
        purge_gap  : = meta["T_in"] + meta["T_out"] - 1.

    Returns:
        train_idx, val_idx, test_idx — 3 list index không chồng lấp.
    """
    n_train = int(S * train_ratio)
    n_val   = int(S * val_ratio)

    val_start  = n_train + purge_gap
    val_end    = val_start + n_val
    test_start = val_end + purge_gap

    if test_start >= S:
        raise ValueError(
            f"Dataset quá nhỏ ({S} mẫu) với purge_gap={purge_gap}. "
            f"Cần ít nhất {n_train + n_val + 2 * purge_gap + 1} mẫu."
        )

    train_idx = list(range(0, n_train))
    val_idx   = list(range(val_start, val_end))
    test_idx  = list(range(test_start, S))
    return train_idx, val_idx, test_idx


# ══════════════════════════════════════════════════════════════════════════════
# 2. NORMALIZATION — FIT ONLY ON TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def fit_normalizers(
    X: Tensor,
    Y: Tensor,
    train_idx: list[int],
) -> tuple[ZScoreNormalizer, ZScoreNormalizer, Tensor, Tensor]:
    """
    Fit Z-Score normalizer chỉ trên train_idx, transform toàn bộ dataset.

    Returns:
        x_normalizer, y_normalizer, X_norm, Y_norm
    """
    x_normalizer = ZScoreNormalizer()
    x_normalizer.fit(X[train_idx])

    y_normalizer = ZScoreNormalizer()
    y_normalizer.fit(Y[train_idx])

    X_norm = x_normalizer.transform(X)
    Y_norm = y_normalizer.transform(Y)

    return x_normalizer, y_normalizer, X_norm, Y_norm


# ══════════════════════════════════════════════════════════════════════════════
# 3. METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(pred: Tensor, true: Tensor, mape_eps: float = MAPE_EPS) -> dict:
    """
    pred, true: (S, N, T_out) — ĐÃ inverse-transform về đơn vị gốc.

    MAE  : Mean Absolute Error
    RMSE : Root Mean Squared Error
    MAPE : Mean Absolute Percentage Error — chỉ tính trên mẫu |true| >= mape_eps
    WAPE : Weighted Absolute Percentage Error = Σ|err| / Σ|true| × 100
    """
    mae  = (pred - true).abs().mean().item()
    rmse = ((pred - true) ** 2).mean().sqrt().item()

    mask = true.abs() >= mape_eps
    if mask.sum() > 0:
        mape = ((pred - true).abs() / true.abs())[mask].mean().item() * 100
    else:
        mape = float("nan")

    denom = true.abs().sum().item()
    wape  = (pred - true).abs().sum().item() / (denom + 1e-8) * 100

    res = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape}

    # Bóc tách theo Horizon (nếu đầu vào là 3D: Batch x Node x T_out)
    if pred.ndim == 3 and pred.shape[2] > 1:
        T_out = pred.shape[2]
        for t in range(T_out):
            pt, tt = pred[:, :, t], true[:, :, t]
            res[f"MAE_{t+1}"] = (pt - tt).abs().mean().item()
            res[f"RMSE_{t+1}"] = ((pt - tt) ** 2).mean().sqrt().item()
            
            mask_t = tt.abs() >= mape_eps
            if mask_t.sum() > 0:
                res[f"MAPE_{t+1}"] = ((pt - tt).abs() / tt.abs())[mask_t].mean().item() * 100
            else:
                res[f"MAPE_{t+1}"] = float("nan")
                
            denom_t = tt.abs().sum().item()
            res[f"WAPE_{t+1}"] = (pt - tt).abs().sum().item() / (denom_t + 1e-8) * 100

    return res


def compute_zone_stratified_metrics(
    pred: Tensor, true: Tensor, Z: Tensor, zone_types: list[str]
) -> dict:
    """
    Tính MAE riêng cho từng zone type.
    pred, true: (S, N, T_out) — ĐÃ inverse-transform.
    Z: (N, K) — multi-hot zone label.
    """
    results = {}
    Z_np = Z.cpu().numpy()
    for k, zone in enumerate(zone_types):
        node_mask = Z_np[:, k] == 1
        if node_mask.sum() == 0:
            continue
        pred_z = pred[:, node_mask, :]
        true_z = true[:, node_mask, :]
        results[f"MAE_{zone}"] = (pred_z - true_z).abs().mean().item()

    multi_mask = Z_np.sum(axis=1) > 1
    if multi_mask.sum() > 0:
        pred_m = pred[:, multi_mask, :]
        true_m = true[:, multi_mask, :]
        results["MAE_multi_zone"] = (pred_m - true_m).abs().mean().item()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE WITH INVERSE-TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_with_inverse(
    model: nn.Module,
    loader: DataLoader,
    A: Tensor,
    Z: Tensor,
    y_normalizer: ZScoreNormalizer,
    device: str,
) -> tuple[Tensor, Tensor]:
    """
    Evaluate model, inverse-transform cả predictions và targets về đơn vị gốc.
    Buộc mọi script phải inverse-transform trước khi tính metrics.
    """
    model.eval()
    preds, trues = [], []
    for batch in loader:
        X_b = batch[0].to(device)
        Y_b = batch[1]
        T_b = batch[2].to(device)
        pred = model(X_b, Z, T_b, A)
        preds.append(pred.cpu())
        trues.append(Y_b)
    preds = torch.cat(preds)
    trues = torch.cat(trues)

    preds_real = y_normalizer.inverse_transform(preds)
    trues_real = y_normalizer.inverse_transform(trues)

    return preds_real, trues_real


# ══════════════════════════════════════════════════════════════════════════════
# 5. UTILITY
# ══════════════════════════════════════════════════════════════════════════════
def get_git_commit_hash() -> str:
    """Lấy git commit hash hiện tại."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
