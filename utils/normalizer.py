"""
utils/normalizer.py
===================
Z-Score Normalization module cho Zone-Aware-GNN.

MỤC ĐÍCH
---------
Chuẩn hóa 4 kênh đầu vào (congestion_ratio, traffic_delay_s,
travel_time_s, ff_ratio) về mean=0, std=1 để:
  1. Gradient ổn định hơn khi train (các feature có biên độ rất khác nhau).
  2. Loss không bị dominated bởi feature có đơn vị lớn (travel_time_s).
  3. Metric MAE/RMSE sau inverse-transform phản ánh đúng đơn vị thực tế.

QUY TẮC FIT ĐỂ TRÁNH DATA LEAKAGE
------------------------------------
  normalizer.fit(X_train)   ← CHỈ fit trên train set
  normalizer.transform(X_val)    ← áp dụng stats đã fit
  normalizer.transform(X_test)   ← áp dụng stats đã fit (KHÔNG fit lại)

  Inverse khi report metric:
    pred_real = y_norm.inverse_transform(pred_norm)
    true_real = y_norm.inverse_transform(true_norm)
    metrics   = compute_metrics(pred_real, true_real)  # đơn vị gốc

USAGE EXAMPLE
-------------
    from utils.normalizer import ZScoreNormalizer

    x_norm = ZScoreNormalizer()
    x_norm.fit(X[train_idx])
    X_tr = x_norm.transform(X[train_idx])
    X_va = x_norm.transform(X[val_idx])
    X_te = x_norm.transform(X[test_idx])

    y_norm = ZScoreNormalizer()
    y_norm.fit(Y[train_idx])
    Y_tr = y_norm.transform(Y[train_idx])
    # ... train ...
    pred_real = y_norm.inverse_transform(pred_norm)

SAVE / LOAD (dùng lại lúc inference)
--------------------------------------
    x_norm.save("data/processed/x_normalizer.pt")
    x_norm2 = ZScoreNormalizer.load("data/processed/x_normalizer.pt")
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import Tensor


class ZScoreNormalizer:
    """
    Z-Score (Standard Score) Normalizer.

    Công thức:
      Chuẩn hóa : x_norm = (x - mean) / std
      Khôi phục : x_orig = x_norm * std + mean

    Mean và std được tính theo chiều cuối (feature dimension) để
    broadcast tự động qua các chiều (S, N, ...).

    Attributes
    ----------
    mean   : Tensor — mean của từng feature, tính trên tập train
    std    : Tensor — std  của từng feature, tính trên tập train
    eps    : float  — ngưỡng để thay std=0 (feature hằng số) bằng 1
    fitted : bool   — True sau khi fit() được gọi
    """

    def __init__(self, eps: float = 1e-8):
        self.eps: float = eps
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None
        self.fitted: bool = False

    # ------------------------------------------------------------------ FIT
    def fit(self, X: Tensor) -> "ZScoreNormalizer":
        """
        Tính mean và std trên tập train.

        Parameters
        ----------
        X : Tensor, shape (S, N, D) hoặc (S, N, T)
            CHỈ truyền vào X[train_idx], không bao giờ truyền val/test.

        Returns
        -------
        self (để chain: normalizer.fit(X).transform(X))
        """
        if X.dim() < 2:
            raise ValueError(
                f"X phải có ít nhất 2 chiều, nhận được shape {X.shape}"
            )

        last_dim = X.shape[-1]
        # Flatten về (S*N, D) để tính thống kê trên toàn bộ train set
        X_flat = X.reshape(-1, last_dim).float()

        self.mean = X_flat.mean(dim=0)  # (D,)
        self.std  = X_flat.std(dim=0)   # (D,)

        # Tránh chia-0 với feature hằng số (std ≈ 0)
        self.std = torch.where(
            self.std < self.eps,
            torch.ones_like(self.std),
            self.std,
        )

        self.fitted = True
        return self

    # --------------------------------------------------- TRANSFORM / INVERSE
    def transform(self, X: Tensor) -> Tensor:
        """
        Chuẩn hóa X dùng mean/std đã fit trên train.

        Parameters
        ----------
        X : Tensor — cùng số chiều và chiều cuối như lúc fit.
        """
        self._check_fitted()
        mean = self.mean.to(X.device)
        std  = self.std.to(X.device)
        return (X.float() - mean) / std

    def inverse_transform(self, X_norm: Tensor) -> Tensor:
        """
        Khôi phục về đơn vị gốc. Dùng trước compute_metrics().

        Parameters
        ----------
        X_norm : Tensor đã được transform().
        """
        self._check_fitted()
        mean = self.mean.to(X_norm.device)
        std  = self.std.to(X_norm.device)
        return X_norm.float() * std + mean

    # ------------------------------------------------------- SAVE / LOAD
    def save(self, path: str) -> None:
        """Lưu stats để dùng lại lúc inference (không cần train lại)."""
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"mean": self.mean, "std": self.std, "eps": self.eps}, path)

    @classmethod
    def load(cls, path: str) -> "ZScoreNormalizer":
        """Load normalizer từ file đã save()."""
        ckpt = torch.load(path, weights_only=True)
        obj = cls(eps=ckpt["eps"])
        obj.mean   = ckpt["mean"]
        obj.std    = ckpt["std"]
        obj.fitted = True
        return obj

    # ------------------------------------------------------------- INTERNAL
    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(
                "ZScoreNormalizer chưa được fit(). "
                "Gọi normalizer.fit(X_train) trước khi transform."
            )

    def __repr__(self) -> str:
        if self.fitted:
            return (
                f"ZScoreNormalizer(fitted=True, "
                f"mean=[{self.mean.min():.4f}..{self.mean.max():.4f}], "
                f"std=[{self.std.min():.4f}..{self.std.max():.4f}])"
            )
        return "ZScoreNormalizer(fitted=False)"
