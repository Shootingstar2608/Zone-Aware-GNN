"""
benchmark/partition_gen.py
==========================
Bo sinh phan vung Non-IID cho benchmark HCM-Sim.

Thiet ke va ly do: docs/03_benchmark_partition_design.md

NGUYEN TAC BAT DI BAT DICH
    partition = index/mask + config + seed.  KHONG BAO GIO copy tensor.

TRANG THAI
    [x] plumbing: fingerprint, gini, clock, leakage guard, io
    [ ] quantity_skew   (P1)
    [ ] temporal_shift  (P2)
    [ ] zone_skew       (P3)
    [ ] concept_drift   (P4 - lam o tang generator, xem docs/03 §7)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np

SCHEMA_VERSION = "0.1.0"
DATASET_PATH = "data/processed/graph_dataset.pt"
META_PATH = "data/processed/meta.json"
OUT_PATH = "data/partitions/partitions_meta.json"

# Hang so dong ho cua HCM-Sim (xem docs/00 §2 buoc 3)
STEPS_PER_HOUR = 4          # 15 phut / snapshot
SNAPSHOTS_PER_DAY = 96
DAY0_WEEKDAY = 0            # snapshot 0 = 00:00 thu Hai (2026-05-18)


# ══════════════════════════════════════════════════════════════
# PLUMBING — da hien thuc, Nguoi 4 dung duoc ngay
# ══════════════════════════════════════════════════════════════
def dataset_fingerprint(path: str = DATASET_PATH) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_meta(path: str = META_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def gini(counts) -> float:
    """0 = phan bo deu tuyet doi, ->1 = mot node om het. Truc x cua figure Nguoi 4."""
    n = np.asarray(counts, dtype=float)
    total = n.sum()
    if total <= 0 or n.size == 0:
        return 0.0
    diffs = np.abs(n[:, None] - n[None, :]).sum()
    return float(diffs / (2 * n.size * total))


def overlap_gap(meta: dict) -> int:
    """
    Hai cua so i, j chong lan du lieu khi |i - j| < T_in + T_out
    => moi split theo thoi gian phai co purge gap >= T_in + T_out - 1.

    DOC TU meta.json, KHONG hardcode: T_out doi theo lan chay multistep (3..24).
    """
    return meta["T_in"] + meta["T_out"] - 1


def recover_clock(S: int, t_in: int):
    """
    Khoi phuc (hour, dow) cho tung cua so.

    graph_dataset.pt khong luu timestamp, nhung HCM-Sim la luoi 15 phut deu tuyet doi
    bat dau 00:00 thu Hai nen suy nguoc duoc chinh xac (docs/00 §2 buoc 3).

    Cua so s mang nhan cua buoc input CUOI: snapshot j = s + t_in - 1
    (giong build_graph.py: idx = t + t_in - 1).
    """
    j = np.arange(S) + t_in - 1
    hour = (j // STEPS_PER_HOUR) % 24
    dow = ((j // SNAPSHOTS_PER_DAY) + DAY0_WEEKDAY) % 7
    return hour, dow


def assert_no_leakage(train_idx, test_idx, gap: int) -> None:
    """Raise neu train/test giao nhau, hoac gan nhau hon `gap` cua so."""
    tr = np.sort(np.asarray(train_idx))
    te = np.sort(np.asarray(test_idx))
    if tr.size == 0 or te.size == 0:
        return
    both = np.intersect1d(tr, te)
    if both.size:
        raise ValueError(f"train/test giao nhau {both.size} cua so, vd {both[:5]}")
    pos = np.searchsorted(tr, te)
    left = np.where(pos > 0, tr[np.clip(pos - 1, 0, None)], -(10 ** 9))
    right = np.where(pos < tr.size, tr[np.clip(pos, None, tr.size - 1)], 10 ** 9)
    dmin = int(np.minimum(np.abs(te - left), np.abs(te - right)).min())
    if dmin < gap:
        raise ValueError(f"ro ri thoi gian: khoang cach nho nhat {dmin} < gap yeu cau {gap}")


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest()


def check_fingerprint(record: dict, path: str = DATASET_PATH) -> None:
    """Fail TO TIENG neu dataset da doi ke tu luc sinh partition."""
    now = dataset_fingerprint(path)
    if record["dataset_fingerprint"] != now:
        raise RuntimeError(
            f"partition '{record['partition_id']}' sinh tu dataset khac.\n"
            f"  luc sinh : {record['dataset_fingerprint'][:16]}...\n"
            f"  hien tai : {now[:16]}...\n"
            "Index cu da vo nghia. Sinh lai partition truoc khi chay."
        )


# ══════════════════════════════════════════════════════════════
# CO CHE — stub, Nguoi 4 code duoc song song dua vao chu ky
# ══════════════════════════════════════════════════════════════
def quantity_skew(S: int, N: int, alpha: float, seed: int, block_len: int = 1):
    """
    Coverage skew: moi node giu mot phan cua so thoi gian, ti le boc tu Dirichlet.

        p ~ Dir(alpha * 1_N),   n_v = floor(p_v * S)

    Returns
    -------
    mask  : (S, N) bool  -- mask[s, v] = node v CO du lieu tai cua so s
    stats : dict         -- n_per_node, gini, n_zero_nodes, zero_nodes,
                            alpha, seed, block_len

    QUY UOC (docs/03 §4)
      - alpha nho => lech manh; alpha lon => deu
      - CHO PHEP node nhan 0 cua so, KHONG dat san. Node do la thi nghiem chinh:
        du bao mot node chua tung duoc giam sat, chi bang zone label + hang xom.
      - block_len > 1 => chon theo KHOI lien tiep (mo phong sensor chet ca cum)
        block_len = 1 => chon ngau nhien tung cua so
      - BAT BUOC rng cuc bo: rng = np.random.default_rng(seed)
        KHONG dung np.random.seed() -- train.py co set_seed(42) se de len.
    """
    raise NotImplementedError("P1")


def zone_skew(Z: np.ndarray, S: int, n_clusters: int, seed: int):
    """Cum node theo thanh phan zone, moi cum nhan mot phan phoi khac nhau.

    Returns (mask, stats) voi stats co them: cluster_of_node, jsd_between_clusters.
    """
    raise NotImplementedError("P3")


def temporal_shift(S: int, meta: dict, scenario: str, seed: int = 0):
    """
    scenario in {"weekday_to_weekend", "normal_to_rush"}.

    Returns (splits, stats); splits = {"train": [...], "val": [...], "test": [...]}.
    BAT BUOC goi assert_no_leakage(train, test, overlap_gap(meta)) truoc khi tra ve.
    Dung recover_clock(S, meta["T_in"]) de lay hour/dow.

    LUU Y (docs/03 §8): chi co DUNG 1 cuoi tuan trong 7 ngay => test set
    weekday->weekend confounded hoan toan voi "2 ngay cuoi chuoi". Ghi vao Data Card.
    """
    raise NotImplementedError("P2")


def concept_drift(*args, **kwargs):
    """KHONG hack spike vao tensor -- them drift mode vao generate_synthetic_traffic.py.
    Ly do: docs/03 §7. Ham nay chi tra ve spec de generator doc."""
    raise NotImplementedError("P4")


# ══════════════════════════════════════════════════════════════
# IO
# ══════════════════════════════════════════════════════════════
def build_record(partition_id, scenario, params, seed, meta,
                 mask=None, splits=None, stats=None) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "partition_id": partition_id,
        "scenario": scenario,
        "params": params,
        "seed": seed,
        "dataset_fingerprint": dataset_fingerprint(),
        "meta_snapshot": {k: meta[k] for k in ("N", "S", "T_in", "T_out")},
        "overlap_gap": overlap_gap(meta),
        "git_commit": git_commit(),
        "stats": stats or {},
    }
    if mask is not None:
        rec["mask_hash"] = mask_hash(mask)
        rec["node_windows"] = {
            str(v): np.flatnonzero(mask[:, v]).tolist() for v in range(mask.shape[1])
        }
    if splits is not None:
        rec["splits"] = {k: list(map(int, v)) for k, v in splits.items()}
    return rec


def save_partitions(records, path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "partitions": records}, f, indent=2)


def load_partitions(path: str = OUT_PATH) -> list:
    with open(path) as f:
        return json.load(f)["partitions"]