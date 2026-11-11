"""Tests for the explicit-time benchmark model/data contract."""
from __future__ import annotations

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baselines import GCNGRUBaseline, LSTMBaseline, STGCNBaseline
from scripts.train import build_model
from utils.history import (
    apply_history_input_mask,
    fit_masked_normalizers,
    naive_forecast,
    unflatten_history,
)


META = {"N": 4, "K": 3, "F": 2, "T_in": 5, "T_out": 2, "interval_min": 15}


def test_unflatten_and_real_temporal_models_have_expected_shape():
    flat = torch.randn(7, 4, 10)
    X = unflatten_history(flat, META)
    assert X.shape == (7, 4, 5, 2)
    Z, A, time = torch.ones(4, 3), torch.eye(4), torch.zeros(7, dtype=torch.long)
    models = [
        LSTMBaseline(4, 2, 2),
        GCNGRUBaseline(4, 2, 2),
        STGCNBaseline(4, 10, 2, T_in=5, F_dim=2),
        build_model("zone_full", META),
    ]
    for model in models:
        assert model(X, Z, time, A).shape == (7, 4, 2)


def test_input_mask_hides_history_and_adds_availability_flag():
    X = torch.ones(2, 3, 3, 2)
    M = torch.tensor([[True, False, True], [False, True, True]])
    masked, info = apply_history_input_mask(X, M, strategy="flag")
    assert masked.shape == (2, 3, 3, 3)
    assert torch.all(masked[0, 1, :, :2] == 0)
    assert torch.all(masked[0, 1, :, 2] == 0)
    assert torch.all(masked[0, 0, :, 2] == 1)
    assert abs(info["masked_fraction"] - 2 / 6) < 1e-6


def test_masked_normalizer_does_not_use_hidden_node_values():
    X = torch.zeros(3, 2, 2, 1)
    X[:, 1] = 10_000  # hidden node would leak a very large mean if included
    Y = torch.zeros(3, 2, 2)
    M = torch.tensor([[True, False], [True, False], [True, False]])
    x_norm, _, _, _ = fit_masked_normalizers(X, Y, [0, 1, 2], M)
    assert x_norm.mean.item() == 0


def test_naive_forecasts_are_causal_and_have_target_shape():
    X = torch.arange(10 * 2 * 3, dtype=torch.float32).reshape(10, 2, 3, 1)
    Y = torch.arange(10 * 2 * 2, dtype=torch.float32).reshape(10, 2, 2)
    for name in ("persistence", "mean_by_node", "seasonal_naive"):
        pred = naive_forecast(name, X, Y, [0, 1, 2, 3], [4, 5], META)
        assert pred.shape == (2, 2, 2)
    assert torch.equal(naive_forecast("persistence", X, Y, [0], [4], META)[0, :, 0], X[4, :, -1, 0])
