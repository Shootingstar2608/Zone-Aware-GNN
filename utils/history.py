"""Shared shape and naive-forecast helpers for the HCM-Sim benchmark.

The stored graph dataset keeps a history window flattened for compactness
(``S,N,T_in*F``).  Models must never mistake that storage layout for a
one-step sequence.  This module restores the explicit time axis and provides
the non-learned reference forecasters used in every benchmark run.
"""
from __future__ import annotations

import torch
from utils.normalizer import ZScoreNormalizer


def unflatten_history(X: torch.Tensor, meta: dict) -> torch.Tensor:
    """Convert persisted ``(S,N,T_in*F)`` history into ``(S,N,T_in,F)``."""
    if X.ndim == 4:
        if X.shape[2:] != (meta["T_in"], meta["F"]):
            raise ValueError(f"4D X has incompatible shape {tuple(X.shape)}")
        return X
    if X.ndim != 3:
        raise ValueError(f"X must be 3D or 4D, got {tuple(X.shape)}")
    expected = meta["T_in"] * meta["F"]
    if X.shape[-1] != expected:
        raise ValueError(f"X last dimension {X.shape[-1]} != T_in*F={expected}")
    return X.reshape(X.shape[0], X.shape[1], meta["T_in"], meta["F"])


def apply_history_input_mask(
    X: torch.Tensor, mask: torch.Tensor, *, strategy: str = "flag", fill_value: float = 0.0
) -> tuple[torch.Tensor, dict]:
    """Mask the actual history seen by a model.

    ``mask[s,v]=False`` hides all T_in observations for node ``v`` in sample
    ``s``.  With ``flag`` an availability channel is appended, so zero-filled
    values cannot be confused with genuine zero traffic features.
    """
    if X.ndim != 4:
        raise ValueError("apply_history_input_mask requires X=(S,N,T_in,F)")
    if mask.shape != X.shape[:2]:
        raise ValueError(f"mask shape {tuple(mask.shape)} != X sample/node shape {tuple(X.shape[:2])}")
    if strategy not in {"flag", "zero"}:
        raise ValueError("Benchmark supports only masking strategy 'flag' or 'zero'")
    available = mask.to(device=X.device, dtype=torch.bool)
    out = X.clone()
    out[~available] = fill_value
    if strategy == "flag":
        flag = available[:, :, None, None].expand(*X.shape[:3], 1).to(dtype=X.dtype)
        out = torch.cat((out, flag), dim=-1)
    return out, {
        "strategy": strategy,
        "fill_value": float(fill_value),
        "masked_fraction": float((~available).float().mean()),
        "feature_dim_before": X.shape[-1],
        "feature_dim_after": out.shape[-1],
    }


def fit_masked_normalizers(
    X: torch.Tensor, Y: torch.Tensor, train_idx: list[int], mask: torch.Tensor
) -> tuple[ZScoreNormalizer, ZScoreNormalizer, torch.Tensor, torch.Tensor]:
    """Fit X/Y statistics only from observed train node-windows.

    A quantity-skew mask must affect not merely the loss but also the
    statistics available to the learner.  Otherwise a withheld node can leak
    its mean and standard deviation into training.
    """
    if X.ndim != 4 or Y.ndim != 3:
        raise ValueError("Expected X=(S,N,T,F) and Y=(S,N,T_out)")
    if mask.shape != X.shape[:2]:
        raise ValueError("mask must have shape (S,N)")
    observed = mask[train_idx].bool()
    if not observed.any():
        raise ValueError("No observed training node-windows; cannot fit scaler")

    x_values = X[train_idx][observed]  # (observed_windows,T,F)
    y_values = Y[train_idx][observed]  # (observed_windows,T_out)
    x_norm, y_norm = ZScoreNormalizer(), ZScoreNormalizer()
    x_norm.fit(x_values)
    y_norm.fit(y_values)
    return x_norm, y_norm, x_norm.transform(X), y_norm.transform(Y)


def persistence_forecast(X: torch.Tensor, t_out: int, target_feature: int = 0) -> torch.Tensor:
    """Repeat the last observed target-like feature across the forecast horizon."""
    last = X[:, :, -1, target_feature]
    return last.unsqueeze(-1).expand(-1, -1, t_out).clone()


def mean_by_node_forecast(Y: torch.Tensor, train_idx: list[int], test_idx: list[int]) -> torch.Tensor:
    """Forecast each node's train-only target mean for every future horizon."""
    if not train_idx:
        raise ValueError("train_idx cannot be empty")
    node_mean = Y[train_idx].mean(dim=(0, 2))
    return node_mean[None, :, None].expand(len(test_idx), -1, Y.shape[-1]).clone()


def seasonal_naive_forecast(
    Y: torch.Tensor, test_idx: list[int], *, steps_per_day: int, fallback: torch.Tensor
) -> torch.Tensor:
    """Use the same forecast window one day earlier, with a causal fallback.

    For a test window s, ``Y[s-steps_per_day]`` lies strictly in the past. If
    the first test day has no prior-day window, persistence is used instead.
    """
    if fallback.shape != (len(test_idx), Y.shape[1], Y.shape[2]):
        raise ValueError("fallback must align with test prediction shape")
    pred = fallback.clone()
    for out_i, sample_i in enumerate(test_idx):
        lag_i = sample_i - steps_per_day
        if lag_i >= 0:
            pred[out_i] = Y[lag_i]
    return pred


def naive_forecast(
    name: str, X: torch.Tensor, Y: torch.Tensor, train_idx: list[int], test_idx: list[int], meta: dict
) -> torch.Tensor:
    """Return one of the mandatory non-learned benchmark references."""
    if name == "persistence":
        return persistence_forecast(X[test_idx], Y.shape[-1])
    if name == "mean_by_node":
        return mean_by_node_forecast(Y, train_idx, test_idx)
    if name == "seasonal_naive":
        fallback = persistence_forecast(X[test_idx], Y.shape[-1])
        # HCM-Sim v1 uses 15-minute snapshots. The field is configurable so a
        # later dataset can declare another cadence explicitly.
        interval_min = int(meta.get("interval_min") or 15)
        if 1440 % interval_min:
            raise ValueError("interval_min must divide one day for seasonal_naive")
        return seasonal_naive_forecast(
            Y, test_idx, steps_per_day=1440 // interval_min, fallback=fallback
        )
    raise ValueError(f"Unknown naive baseline: {name}")
