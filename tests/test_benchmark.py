"""
tests/test_benchmark.py
=======================
Unit test cho benchmark/partition_gen.py.

Chay duoc CA HAI kieu:
    pytest tests/test_benchmark.py -v
    python tests/test_benchmark.py          (khong can cai pytest)

Bao phu: 4 co che phan vung, Gini, chong ro ri thoi gian, tinh tai lap,
schema ban ghi, va input masking v2.
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)          # de duong dan tuong doi (data/...) luon dung

from benchmark.partition_gen import (  # noqa: E402
    apply_input_mask, assert_no_leakage, build_record, concept_drift, gini,
    load_meta, mask_hash, overlap_gap, quantity_skew, recover_clock, rush_mask,
    temporal_shift, zone_skew,
)

META = {"N": 17, "S": 637, "T_in": 12, "T_out": 24, "F": 4,
        "zone_types": ["commercial", "residential", "industrial", "school",
                       "university", "hospital", "transport", "park"]}
S, N = META["S"], META["N"]
SEEDS = [42, 43, 44]


def _zone_matrix():
    """Z that neu doc duoc file, khong thi dung ma tran gia tat dinh."""
    path = os.path.join(REPO_ROOT, "data", "raw", "zone_labels.csv")
    if os.path.exists(path):
        import csv
        rows = {r["node"]: r for r in csv.DictReader(open(path))}
        nodes = sorted(rows)
        return np.array([[float(rows[n][z]) for z in META["zone_types"]] for n in nodes])
    rng = np.random.default_rng(0)
    return (rng.random((N, len(META["zone_types"]))) > 0.6).astype(float)


# ── Gini ──────────────────────────────────────────────────────────────
def test_gini_bien():
    assert gini([10, 10, 10]) == 0.0                 # deu tuyet doi
    assert gini([1, 1, 1, 1]) == 0.0
    assert gini([30, 0, 0]) > 0.6                    # mot node om het
    assert gini([0, 0, 0]) == 0.0                    # khong chia cho 0
    assert gini([]) == 0.0                           # rong


def test_gini_don_dieu_theo_alpha():
    """alpha cang lon => phan bo cang deu => Gini cang nho."""
    g = [np.mean([quantity_skew(S, N, a, s)[1]["gini"] for s in SEEDS])
         for a in (0.1, 0.5, 1.0, 5.0)]
    assert g == sorted(g, reverse=True), f"Gini khong giam don dieu: {g}"


# ── Co che 1: quantity_skew ───────────────────────────────────────────
def test_quantity_skew_ton_trong_tran():
    for a in (0.1, 1.0, 5.0):
        for s in SEEDS:
            _, st = quantity_skew(S, N, a, s)
            assert max(st["n_per_node"]) <= S


def test_quantity_skew_giu_ngan_sach():
    """mode fixed_coverage phai giu tong quan sat khong doi theo alpha."""
    B = 0.5 * N * S
    for a in (0.1, 1.0, 5.0):
        for s in SEEDS:
            _, st = quantity_skew(S, N, a, s, mode="fixed_coverage")
            assert abs(st["total_obs"] - B) <= N, f"alpha={a} seed={s}: {st['total_obs']}"


def test_quantity_skew_stats_khop_mask():
    """stats phai duoc dem TU MASK -- day la cai bat duoc loi mask rong."""
    for s in SEEDS:
        m, st = quantity_skew(S, N, 0.5, s)
        assert (m.sum(axis=0) == np.array(st["n_per_node"])).all()
        assert m.sum() == st["total_obs"]


def test_quantity_skew_block_len_khong_chong_lap():
    for bl in (1, 4, 16, 64):
        m, st = quantity_skew(S, N, 1.0, 42, block_len=bl)
        assert (m.sum(axis=0) == np.array(st["n_per_node"])).all(), f"block_len={bl}"


def test_quantity_skew_mode_khong_hop_le():
    try:
        quantity_skew(S, N, 0.5, 42, mode="fixed-coverage")
        raise AssertionError("mode sai le ra phai raise ValueError")
    except ValueError:
        pass


# ── Co che 2: zone_skew ───────────────────────────────────────────────
def test_zone_skew_truc_giao_voi_quantity():
    """So mau moi node phai BANG NHAU => Gini = 0 => khong lan voi quantity_skew."""
    Z = _zone_matrix()
    for k in (2, 3, 4):
        m, st = zone_skew(Z, S, META, n_clusters=k, seed=42)
        assert st["gini"] == 0.0, f"k={k}: gini={st['gini']}"
        assert len(set(st["n_per_node"])) == 1


def test_zone_skew_lech_tang_khi_off_band_giam():
    Z = _zone_matrix()
    j = [zone_skew(Z, S, META, 3, 42, off_band_weight=w)[1]["jsd_between_clusters"]
         for w in (1.0, 0.5, 0.1)]
    assert j == sorted(j), f"JSD khong tang khi off_band_weight giam: {j}"


def test_zone_skew_labels_thu_cong():
    Z = _zone_matrix()
    manual = np.arange(N) % 2
    _, st = zone_skew(Z, S, META, seed=42, labels=manual)
    assert st["cluster_of_node"] == manual.tolist()


# ── Co che 3: temporal_shift ──────────────────────────────────────────
def test_temporal_shift_khong_ro_ri():
    """Bai test quan trong nhat: cua so truot lam train/test chong lap."""
    sp, st = temporal_shift(S, META)
    gap = overlap_gap(META)
    assert gap == META["T_in"] + META["T_out"] - 1 == 35
    assert_no_leakage(sp["train"], sp["test"], gap)
    assert_no_leakage(sp["train"], sp["val"], gap)
    assert_no_leakage(sp["val"], sp["test"], gap)
    assert st["min_gap_actual"] >= st["gap_required"]


def test_temporal_shift_val_lay_tu_phan_phoi_train():
    """Val phai la NGAY THUONG. Val la cuoi tuan = chon hyperparam tren test."""
    _, st = temporal_shift(S, META)
    assert st["dow_train"] == [0, 1, 2, 3]
    assert st["dow_val"] == [4]
    assert st["dow_test"] == [5, 6]


def test_temporal_shift_ba_tap_roi_nhau():
    sp, _ = temporal_shift(S, META)
    tr, va, te = map(set, (sp["train"], sp["val"], sp["test"]))
    assert not (tr & va) and not (va & te) and not (tr & te)


def test_temporal_shift_gap_scale_theo_horizon():
    burned = {}
    for t_out in (3, 12, 24):
        _, st = temporal_shift(S, dict(META, T_out=t_out))
        burned[t_out] = st["n_burned"]
    assert burned[3] < burned[12] < burned[24], burned


def test_normal_to_rush_bi_tu_choi():
    """Kich ban nay BAT KHA THI (0/449 cua so song sot) -- xem docs/khoa/03 §9.2."""
    try:
        temporal_shift(S, META, "normal_to_rush")
        raise AssertionError("normal_to_rush le ra phai raise ValueError")
    except ValueError:
        pass


def test_rush_mask_hop_ly():
    rm = rush_mask(S, META)
    hour, _ = recover_clock(S, META["T_in"])
    assert set(np.unique(hour[rm]).tolist()) == {7, 8, 9, 16, 17, 18, 19}
    assert 0.2 < rm.mean() < 0.4


# ── Co che 4: concept_drift ───────────────────────────────────────────
def test_concept_drift_nam_trong_test():
    sp, _ = temporal_shift(S, META)
    spec, _ = concept_drift(sp["test"], N, seed=42)
    assert spec["window_start"] >= min(sp["test"])
    assert spec["window_end"] <= max(sp["test"])


def test_concept_drift_profile_hinh_thang():
    sp, _ = temporal_shift(S, META)
    spec, _ = concept_drift(sp["test"], N, seed=42, magnitude=1.8, ramp=8, duration=24)
    p = np.array(spec["profile"])
    assert len(p) == 24 + 2 * 8
    assert p.max() == 1.8 and p.min() >= 1.0
    assert (np.diff(p[:8]) > 0).all()        # len dan
    assert (np.diff(p[-8:]) < 0).all()       # xuong dan


# ── Tai lap ───────────────────────────────────────────────────────────
def test_tai_lap_theo_seed():
    Z = _zone_matrix()
    for fn, args, kw in [(quantity_skew, (S, N, 0.5, 42), {}),
                         (quantity_skew, (S, N, 0.5, 42), {"block_len": 8}),
                         (zone_skew, (Z, S, META), {"seed": 42})]:
        assert mask_hash(fn(*args, **kw)[0]) == mask_hash(fn(*args, **kw)[0])


def test_seed_khac_cho_mask_khac():
    assert mask_hash(quantity_skew(S, N, 0.5, 42)[0]) != mask_hash(quantity_skew(S, N, 0.5, 43)[0])


def test_khong_dung_rng_toan_cuc():
    """np.random.seed() khong duoc anh huong ket qua -- train.py co set_seed(42)."""
    np.random.seed(1); a = mask_hash(quantity_skew(S, N, 0.5, 42)[0])
    np.random.seed(999); b = mask_hash(quantity_skew(S, N, 0.5, 42)[0])
    assert a == b, "partition_gen dang doc rng toan cuc"


# ── Schema ban ghi ────────────────────────────────────────────────────
def test_build_record_du_truong():
    m, st = quantity_skew(S, N, 0.5, 42)
    meta = load_meta() if os.path.exists("data/processed/meta.json") else META
    if not os.path.exists("data/processed/graph_dataset.pt"):
        return                          # khong co dataset thi bo qua kiem fingerprint
    rec = build_record("qskew_a0.5_s42", "quantity_skew", {"alpha": 0.5}, 42,
                       meta, mask=m, stats=st)
    for k in ("schema_version", "partition_id", "scenario", "params", "seed",
              "dataset_fingerprint", "meta_snapshot", "overlap_gap",
              "git_commit", "mask_hash", "node_windows", "stats"):
        assert k in rec, f"ban ghi thieu truong {k}"
    assert json.dumps(rec)              # phai serialize duoc


# ── Input masking v2 ──────────────────────────────────────────────────
def _fake_X():
    rng = np.random.default_rng(0)
    return rng.random((S, N, META["T_in"] * META["F"])).astype(np.float32) + 1.0


def test_input_mask_flag_them_kenh():
    X = _fake_X()
    m, _ = quantity_skew(S, N, 0.5, 42)
    Xm, info = apply_input_mask(X, m, META, strategy="flag")
    assert info["F_out"] == META["F"] + 1
    assert Xm.shape == (S, N, META["T_in"] * (META["F"] + 1))
    x = Xm.reshape(S, N, META["T_in"], META["F"] + 1)
    assert (x[..., -1][m] == 1).all()        # node co du lieu -> co 1
    assert (x[..., -1][~m] == 0).all()       # node bi che     -> co 0


def test_input_mask_thuc_su_xoa_dac_trung():
    X = _fake_X()
    m, _ = quantity_skew(S, N, 0.5, 42)
    for strat in ("zero", "flag"):
        Xm, _ = apply_input_mask(X, m, META, strategy=strat)
        x = Xm.reshape(S, N, META["T_in"], -1)[..., :META["F"]]
        assert (x[~m] == 0).all(), f"{strat}: dac trung bi che chua bi xoa"
        assert np.allclose(x[m], X.reshape(S, N, META["T_in"], META["F"])[m])


def test_input_mask_neighbor_khong_de_lai_o_trong():
    X = _fake_X()
    A = np.full((N, N), 1.0 / N)
    m, _ = quantity_skew(S, N, 1.0, 42)
    Xm, _ = apply_input_mask(X, m, META, strategy="neighbor", A=A)
    x = Xm.reshape(S, N, META["T_in"], META["F"])
    assert np.isfinite(x).all()
    assert (x[~m] != 0).any(), "noi suy le ra phai dien gia tri khac 0"


def test_input_mask_giu_shape_khi_khong_flag():
    X = _fake_X()
    m, _ = quantity_skew(S, N, 0.5, 42)
    for strat in ("zero", "last_seen"):
        Xm, info = apply_input_mask(X, m, META, strategy=strat)
        assert Xm.shape == X.shape and info["F_out"] == META["F"]


def test_input_mask_bat_loi_shape():
    X = _fake_X()
    for bad in (np.ones((S, N + 1), bool), np.ones((S - 1, N), bool)):
        try:
            apply_input_mask(X, bad, META)
            raise AssertionError("mask sai shape le ra phai raise")
        except ValueError:
            pass


# ── Runner khong can pytest ───────────────────────────────────────────
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
