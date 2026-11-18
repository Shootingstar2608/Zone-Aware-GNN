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


def test_seasonal_naive_uses_only_previous_day_targets_and_is_causal():
    """Perturbing current/future targets must not change seasonal predictions."""
    from utils.history import seasonal_naive_forecast

    # 4 samples/day keeps the test small: sample 6 uses sample 2, sample 7 uses sample 3.
    Y = torch.arange(12 * 2 * 2, dtype=torch.float32).reshape(12, 2, 2)
    test_idx = [6, 7]
    fallback = torch.full((2, 2, 2), -999.0)

    pred = seasonal_naive_forecast(
        Y, test_idx, steps_per_day=4, fallback=fallback
    )
    expected = torch.stack([Y[2], Y[3]])
    assert torch.equal(pred, expected)

    # Changing all targets at/after the current test sample cannot affect its forecast.
    Y_changed = Y.clone()
    Y_changed[6:] += 1_000_000
    pred_changed = seasonal_naive_forecast(
        Y_changed, test_idx, steps_per_day=4, fallback=fallback
    )
    assert torch.equal(pred, pred_changed)


def test_seasonal_naive_uses_causal_fallback_when_previous_day_is_unavailable():
    from utils.history import seasonal_naive_forecast

    Y = torch.arange(8 * 2 * 2, dtype=torch.float32).reshape(8, 2, 2)
    fallback = torch.full((2, 2, 2), 123.0)
    pred = seasonal_naive_forecast(
        Y, [1, 5], steps_per_day=4, fallback=fallback
    )
    assert torch.equal(pred[0], fallback[0])  # 1 - 4 < 0
    assert torch.equal(pred[1], Y[1])          # 5 - 4 = 1


def test_seasonal_naive_respects_quantity_skew_observation_mask():
    X = torch.ones(8, 2, 3, 1)
    Y = torch.arange(8 * 2 * 2, dtype=torch.float32).reshape(8, 2, 2)
    observed = torch.ones(8, 2, dtype=torch.bool)
    observed[2, 0] = False  # yesterday's value for node 0 was not observed
    observed[3, 0] = False
    observed[7, 0] = False  # current input for node 0 is also unavailable

    pred = naive_forecast(
        "seasonal_naive", X, Y, [0, 1, 2, 3], [6, 7],
        {**META, "interval_min": 360}, observed_mask=observed,
    )
    # sample 6: node 0 falls back to persistence; node 1 uses Y[2].
    assert torch.equal(pred[0, 0], torch.ones(2))
    assert torch.equal(pred[0, 1], Y[2, 1])
    # sample 7: node 0 has neither current input nor yesterday's observation,
    # so it falls back to the mean of visible training targets for that node.
    expected_train_mean = Y[[0, 1], 0].mean()
    assert torch.allclose(pred[1, 0], expected_train_mean.expand(2))
    assert torch.equal(pred[1, 1], Y[3, 1])
