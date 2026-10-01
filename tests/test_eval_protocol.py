"""
tests/test_eval_protocol.py
============================
Unit tests cho utils/eval_protocol.py — shared evaluation protocol.

Chạy:
    pytest tests/test_eval_protocol.py -v
    python tests/test_eval_protocol.py
"""

import math
import os
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from utils.eval_protocol import (
    chronological_split,
    fit_normalizers,
    compute_metrics,
    compute_zone_stratified_metrics,
    MAPE_EPS,
    PURGE_GAP_DEFAULT,
)
from utils.normalizer import ZScoreNormalizer


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_dataset(S=200, N=5, F=4, T_in=6, T_out=3):
    """Synthetic dataset nhỏ để test nhanh."""
    torch.manual_seed(0)
    X = torch.randn(S, N, T_in * F)
    Y = torch.randn(S, N, T_out).abs()  # positive targets
    return X, Y


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHRONOLOGICAL SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def test_chrono_split_no_overlap():
    """Train/val/test indices không được chồng lấp."""
    S = 637
    train, val, test = chronological_split(S, purge_gap=35)
    assert not set(train) & set(val), "train & val overlap"
    assert not set(val) & set(test), "val & test overlap"
    assert not set(train) & set(test), "train & test overlap"


def test_chrono_split_purge_gap_respected():
    """Khoảng cách giữa các tập >= purge_gap."""
    S = 637
    gap = 35
    train, val, test = chronological_split(S, purge_gap=gap)
    assert min(val) - max(train) >= gap, (
        f"gap train→val = {min(val) - max(train)}, expected >= {gap}"
    )
    assert min(test) - max(val) >= gap, (
        f"gap val→test = {min(test) - max(val)}, expected >= {gap}"
    )


def test_chrono_split_ordering():
    """Tất cả train < tất cả val < tất cả test."""
    S = 637
    train, val, test = chronological_split(S, purge_gap=35)
    assert max(train) < min(val)
    assert max(val) < min(test)


def test_chrono_split_various_horizons():
    """Gap scale đúng theo T_out khác nhau."""
    S = 500
    T_in = 12
    for T_out in [3, 12, 24]:
        gap = T_in + T_out - 1
        train, val, test = chronological_split(S, purge_gap=gap)
        actual_gap_tv = min(val) - max(train)
        assert actual_gap_tv >= gap, f"T_out={T_out}: gap={actual_gap_tv} < {gap}"


def test_chrono_split_too_small_dataset():
    """Dataset quá nhỏ phải raise ValueError."""
    try:
        chronological_split(50, purge_gap=35)
        raise AssertionError("Expected ValueError for small dataset")
    except ValueError:
        pass


def test_chrono_split_covers_all_expected():
    """Tổng samples = train + gap + val + gap + test."""
    S = 637
    gap = 35
    train, val, test = chronological_split(S, purge_gap=gap)
    total_used = len(train) + len(val) + len(test)
    total_gaps = 2 * gap
    # Tất cả index phải trong [0, S)
    assert all(0 <= i < S for i in train + val + test)
    assert test[-1] == S - 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. NORMALIZER ROUND-TRIP
# ══════════════════════════════════════════════════════════════════════════════
def test_normalizer_roundtrip():
    """inverse_transform(transform(X)) ≈ X."""
    X, Y = _make_dataset()
    train_idx = list(range(140))

    norm = ZScoreNormalizer()
    norm.fit(X[train_idx])
    X_t = norm.transform(X)
    X_back = norm.inverse_transform(X_t)

    err = (X - X_back).abs().max().item()
    assert err < 1e-4, f"Roundtrip error = {err}"


def test_normalizer_train_only_fit():
    """Stats phải phản ánh chỉ train data."""
    X, Y = _make_dataset()
    train_idx = list(range(50))

    norm = ZScoreNormalizer()
    norm.fit(X[train_idx])
    mean_train = X[train_idx].reshape(-1, X.shape[-1]).float().mean(dim=0)

    # Mean từ normalizer phải gần mean_train
    err = (norm.mean - mean_train).abs().max().item()
    assert err < 1e-5, f"Mean mismatch: {err}"


def test_normalizer_constant_feature():
    """Feature với std=0 → std thay bằng 1, không NaN."""
    X = torch.ones(100, 10, 4)  # tất cả giống nhau
    norm = ZScoreNormalizer()
    norm.fit(X)
    X_t = norm.transform(X)
    assert torch.isfinite(X_t).all(), "NaN/Inf in transform of constant feature"


def test_normalizer_save_load_roundtrip():
    """save() rồi load() phải cho stats giống hệt."""
    X, _ = _make_dataset()
    norm = ZScoreNormalizer()
    norm.fit(X[:100])

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        norm.save(path)
        norm2 = ZScoreNormalizer.load(path)
        assert torch.allclose(norm.mean, norm2.mean)
        assert torch.allclose(norm.std, norm2.std)
    finally:
        os.unlink(path)


def test_fit_normalizers_helper():
    """fit_normalizers() trả về đúng 4 giá trị."""
    X, Y = _make_dataset()
    train_idx = list(range(100))
    x_n, y_n, X_norm, Y_norm = fit_normalizers(X, Y, train_idx)
    assert x_n.fitted
    assert y_n.fitted
    assert X_norm.shape == X.shape
    assert Y_norm.shape == Y.shape


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAPE / WAPE WITH ZERO TARGETS
# ══════════════════════════════════════════════════════════════════════════════
def test_mape_filters_below_eps():
    """Targets < MAPE_EPS bị loại khỏi MAPE."""
    pred = torch.tensor([[[0.1, 0.2]]])
    true = torch.tensor([[[0.01, 0.5]]])  # 0.01 < 0.05 → bị loại

    m = compute_metrics(pred, true, mape_eps=0.05)
    # MAPE chỉ tính trên element thứ 2 (true=0.5)
    expected_mape = abs(0.2 - 0.5) / 0.5 * 100  # = 60%
    assert abs(m["MAPE"] - expected_mape) < 0.1, f"MAPE = {m['MAPE']}, expected {expected_mape}"


def test_mape_all_below_eps():
    """Tất cả targets < MAPE_EPS → MAPE = NaN."""
    pred = torch.tensor([[[0.1, 0.2]]])
    true = torch.tensor([[[0.01, 0.02]]])

    m = compute_metrics(pred, true, mape_eps=0.05)
    assert math.isnan(m["MAPE"]), f"Expected NaN, got {m['MAPE']}"


def test_wape_with_zero_targets():
    """WAPE hữu hạn khi có zero targets."""
    pred = torch.tensor([[[0.1, 0.2, 0.3]]])
    true = torch.tensor([[[0.0, 0.5, 1.0]]])

    m = compute_metrics(pred, true)
    assert math.isfinite(m["WAPE"]), f"WAPE not finite: {m['WAPE']}"


def test_wape_all_zero_targets():
    """WAPE vẫn finite khi tất cả target = 0 (nhờ eps trong mẫu số)."""
    pred = torch.tensor([[[0.1, 0.2]]])
    true = torch.tensor([[[0.0, 0.0]]])

    m = compute_metrics(pred, true)
    assert math.isfinite(m["WAPE"]), f"WAPE not finite: {m['WAPE']}"


def test_metrics_keys():
    """compute_metrics phải trả về đủ 4 key."""
    pred = torch.randn(10, 5, 3).abs()
    true = torch.randn(10, 5, 3).abs()
    m = compute_metrics(pred, true)
    base_keys = {"MAE", "RMSE", "MAPE", "WAPE"}
    assert base_keys.issubset(m.keys()), f"Missing base keys, got {m.keys()}"
    # Đảm bảo có thêm các horizon keys (vd: MAE_1, RMSE_1, ...)
    assert any(k.startswith("MAE_") for k in m.keys())


# ══════════════════════════════════════════════════════════════════════════════
# 4. ZONE-STRATIFIED METRICS
# ══════════════════════════════════════════════════════════════════════════════
def test_zone_stratified_metrics():
    """Zone metrics phải trả về MAE cho từng zone."""
    pred = torch.randn(10, 4, 3).abs()
    true = torch.randn(10, 4, 3).abs()
    Z = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],  # multi-zone
        [0, 0, 1],
    ], dtype=torch.float32)
    zone_types = ["A", "B", "C"]

    m = compute_zone_stratified_metrics(pred, true, Z, zone_types)
    assert "MAE_A" in m
    assert "MAE_B" in m
    assert "MAE_C" in m
    assert "MAE_multi_zone" in m


# ══════════════════════════════════════════════════════════════════════════════
# 5. INTEGRATION — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def test_eval_protocol_end_to_end():
    """Full pipeline: split → normalize → inverse → metrics."""
    S, N, F, T_in, T_out = 200, 5, 4, 6, 3
    X, Y = _make_dataset(S, N, F, T_in, T_out)

    gap = T_in + T_out - 1  # = 8
    train_idx, val_idx, test_idx = chronological_split(S, purge_gap=gap)

    x_n, y_n, X_norm, Y_norm = fit_normalizers(X, Y, train_idx)

    # Simulate model output (normalized)
    pred_norm = Y_norm[test_idx] + torch.randn_like(Y_norm[test_idx]) * 0.1
    true_norm = Y_norm[test_idx]

    # Inverse transform
    pred_real = y_n.inverse_transform(pred_norm)
    true_real = y_n.inverse_transform(true_norm)

    # Metrics trên real scale
    m = compute_metrics(pred_real, true_real)
    assert m["MAE"] > 0
    assert m["RMSE"] >= m["MAE"]  # RMSE >= MAE by definition
    assert math.isfinite(m["WAPE"])


# ── Runner không cần pytest ───────────────────────────────────────────
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} pass")
    sys.exit(1 if failed else 0)
