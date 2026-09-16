"""
benchmark/partition_gen.py
==========================
Bo sinh phan vung Non-IID cho benchmark HCM-Sim.

Thiet ke va ly do: docs/khoa/03_benchmark_partition_design.md

NGUYEN TAC BAT DI BAT DICH
    partition = index/mask + config + seed.  KHONG BAO GIO copy tensor.

TRANG THAI
    [x] plumbing: fingerprint, gini, clock, leakage guard, io
    [x] quantity_skew   (P1)  mode mac dinh: fixed_coverage
    [x] temporal_shift  (P2)  chi weekday_to_weekend; normal_to_rush bat kha thi
    [x] zone_skew       (P3)  giu so mau deu nhau => truc giao voi quantity_skew
    [x] concept_drift   (P4)  tra ve SPEC cho generator, khong sua tensor
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

def _water_fill(p, S, B):
    """Chia ngan sach B cho N node theo ti le p, tran moi node la S.

    Clip thuong khien phan vuot tran bi vut di => tong ngan sach boc hoi.
    O day phan vuot duoc CHIA LAI cho cac node chua day, lap den khi
    khong con ai vuot tran.
    """
    N = len(p)
    n = np.zeros(N)
    free = np.ones(N, dtype=bool)          # node chua bi chot o tran
    remaining = float(B)

    while True:
        w = np.where(free, p, 0.0)
        if w.sum() <= 0 or remaining <= 0:
            break
        share = remaining * w / w.sum()     # chia theo ti le TRONG nhom chua day
        over = free & (share > S)
        if not over.any():                  # khong ai vuot -> xong
            n[free] = share[free]
            break
        n[over] = S                         # chot cac node vuot o tran
        remaining -= S * over.sum()
        free &= ~over                       # loai ho ra, vong sau chia lai phan du

    # lam tron kieu largest-remainder de tong khop B (floor thuong hut toi N don vi)
    fl = np.floor(n).astype(int)
    deficit = int(round(B)) - fl.sum()
    for v in np.argsort(-(n - fl)):         # uu tien node co phan thap phan lon nhat
        if deficit <= 0:
            break
        if fl[v] < S:                       # van phai ton trong tran
            fl[v] += 1
            deficit -= 1
    return np.clip(fl, 0, S)

# ══════════════════════════════════════════════════════════════
# CO CHE — stub, Nguoi 4 code duoc song song dua vao chu ky
# ══════════════════════════════════════════════════════════════
def quantity_skew(S: int, N: int, alpha: float, seed: int, block_len: int = 1, 
                  mode="fixed_coverage", c_bar=0.5):
    """
    Coverage skew: moi node giu mot phan cua so thoi gian, ti le boc tu Dirichlet.

        p ~ Dir(alpha * 1_N),   n_v = floor(p_v * S)

    Returns
    -------
    mask  : (S, N) bool  -- mask[s, v] = node v CO du lieu tai cua so s
    stats : dict         -- n_per_node, gini, n_zero_nodes, zero_nodes,
                            alpha, seed, block_len

    QUY UOC (docs/khoa/03 §4)
      - alpha nho => lech manh; alpha lon => deu
      - CHO PHEP node nhan 0 cua so, KHONG dat san. Node do la thi nghiem chinh:
        du bao mot node chua tung duoc giam sat, chi bang zone label + hang xom.
      - block_len > 1 => chon theo KHOI lien tiep (mo phong sensor chet ca cum)
        block_len = 1 => chon ngau nhien tung cua so
      - BAT BUOC rng cuc bo: rng = np.random.default_rng(seed)
        KHONG dung np.random.seed() -- train.py co set_seed(42) se de len.
    """
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(alpha * np.ones(N))

    if mode not in ("fixed_coverage", "budget", "relative"):
        raise ValueError(f"mode khong hop le: {mode!r}")

    if mode == "relative":
        n = np.clip(np.floor(p / p.max() * S).astype(int), 0, S)
    else:
        B = N * S if mode == "budget" else c_bar * N * S
        n = _water_fill(p, S, B)

    mask = np.zeros((S, N), dtype=bool)
    for v in range(N):
        if n[v] == 0:
            continue
        if block_len == 1:
            idx = rng.choice(S, size=n[v], replace=False)
        else:
            # khoi cuoi cung co the ngan hon block_len => tong suc chua van la S
            starts = np.arange(0, S, block_len)
            order = rng.permutation(len(starts))
            picked, total = [], 0
            for si in order:
                s0 = starts[si]
                picked.append(np.arange(s0, min(s0 + block_len, S)))
                total += min(block_len, S - s0)
                if total >= n[v]:
                    break
            idx = np.concatenate(picked)[:n[v]]
        mask[idx, v] = True

    n_per_node = mask.sum(axis=0).tolist()
    stats = {
        "n_per_node": n_per_node,
        "total_obs": int(sum(n_per_node)),
        "coverage": float(sum(n_per_node)) / (S * N),
        "gini": gini(n_per_node),
        "n_zero_nodes": int(sum(1 for x in n_per_node if x == 0)),
        "zero_nodes": [int(v) for v, x in enumerate(n_per_node) if x == 0],
        "alpha": float(alpha),
        "seed": int(seed),
        "block_len": int(block_len),
        "mode": mode,
        "c_bar": float(c_bar),
    }
    return mask, stats



# ── P3: Feature / Zone Skew ────────────────────────────────────────────
def _cluster_zones(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Agglomerative average-linkage tren cosine distance cua vector zone.

    Tu viet thay vi dung sklearn: 17 node nen chi phi khong dang ke, doi lai
    khong phu thuoc phien ban sklearn (API metric/affinity da doi vai lan)
    va tat dinh tuyet doi.
    """
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    D = 1.0 - Zn @ Zn.T
    np.fill_diagonal(D, 0.0)
    clusters = [[i] for i in range(Z.shape[0])]
    while len(clusters) > n_clusters:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = float(D[np.ix_(clusters[a], clusters[b])].mean())
                if best is None or d < best[0]:      # tie -> giu cap dau tien
                    best = (d, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    clusters.sort(key=min)                            # nhan on dinh theo node nho nhat
    lab = np.zeros(Z.shape[0], dtype=int)
    for c, mem in enumerate(clusters):
        lab[mem] = c
    return lab


def _jsd(p, q) -> float:
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(a / np.where(b > 0, b, 1)), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def zone_skew(Z: np.ndarray, S: int, meta: dict, n_clusters: int = 3, seed: int = 42,
              c_bar: float = 0.5, off_band_weight: float = 0.1, labels=None):
    """Cum node theo thanh phan zone; moi cum quan sat mot KHUNG GIO khac nhau.

    Khac quantity_skew o cho: so cua so moi node deu NHU NHAU (= c_bar * S),
    chi khac o CUA SO NAO. Nho vay hai co che truc giao -- quantity_skew doi
    "bao nhieu", zone_skew doi "phan phoi dac trung" -- va Nguoi 4 sweep duoc
    tung cai mot ma khong lan nhau.

    labels: gan cum THU CONG (mang (N,) int). None => tu dong cum theo Z.
    off_band_weight: trong so cua cua so NGOAI khung gio cua cum (1.0 = khong
    lech ti nao, cang nho cang lech).
    """
    N = Z.shape[0]
    rng = np.random.default_rng(seed)
    hour, _ = recover_clock(S, meta["T_in"])
    if labels is None:
        lab = _cluster_zones(Z, n_clusters)
    else:                                  # cho phep gan cum THU CONG theo ngu nghia
        lab = np.asarray(labels, dtype=int)
        if lab.shape != (N,):
            raise ValueError(f"labels phai co shape ({N},), nhan duoc {lab.shape}")
        n_clusters = int(lab.max()) + 1

    edges = np.linspace(0, 24, n_clusters + 1)        # chia ngay thanh n_clusters dai gio
    n_keep = int(round(c_bar * S))
    if off_band_weight <= 0:
        raise ValueError(
            "off_band_weight phai > 0: bang 0 nghia la cum chi duoc lay cua so trong "
            f"khung gio cua no (~{S//n_clusters}), khong du cho n_keep={n_keep}. "
            "Dung 0.01 neu muon lech gan nhu tuyet doi."
        )
    mask = np.zeros((S, N), dtype=bool)
    for v in range(N):
        c = lab[v]
        in_band = (hour >= edges[c]) & (hour < edges[c + 1])
        w = np.where(in_band, 1.0, off_band_weight)
        idx = rng.choice(S, size=n_keep, replace=False, p=w / w.sum())
        mask[idx, v] = True

    hist = []                                          # phan bo gio thuc te cua tung cum
    for c in range(n_clusters):
        h = np.zeros(24)
        for v in np.flatnonzero(lab == c):
            h += np.bincount(hour[mask[:, v]], minlength=24)
        hist.append(h)
    pairs = [_jsd(hist[i], hist[j]) for i in range(n_clusters) for j in range(i + 1, n_clusters)]

    n_per_node = mask.sum(axis=0).tolist()
    stats = {
        "n_per_node": n_per_node,
        "total_obs": int(sum(n_per_node)),
        "coverage": float(sum(n_per_node)) / (S * N),
        "gini": gini(n_per_node),
        "cluster_of_node": lab.tolist(),
        "cluster_sizes": np.bincount(lab, minlength=n_clusters).tolist(),
        "jsd_between_clusters": float(np.mean(pairs)) if pairs else 0.0,
        "jsd_pairs": [float(x) for x in pairs],
        "n_clusters": int(n_clusters), "c_bar": float(c_bar),
        "off_band_weight": float(off_band_weight), "seed": int(seed),
    }
    return mask, stats


# ── P4: Concept Drift ──────────────────────────────────────────────────
def concept_drift(test_idx, N: int, seed: int = 42, n_targets: int = 3,
                  magnitude: float = 1.8, duration: int = 24, ramp: int = 8):
    """Sinh SPEC su co giao thong, KHONG sua tensor.

    Tra ve mo ta de generate_synthetic_traffic.py doc va ap o tang sinh du lieu,
    nho vay traffic_delay_s / travel_time_s / congestion_ratio dich chuyen NHAT
    QUAN VE VAT LY. Sua thang tensor thi phai tu tay giu quan he giua chung,
    va sai mot cai la model hoc duoc quan he phi vat ly (docs/khoa/03 §7).

    Su co dat TRONG test set: drift phai la thu model chua tung thay luc train.
    profile la he so nhan hinh thang: len dan `ramp`, giu `duration`, xuong dan.
    """
    test_idx = np.asarray(test_idx)
    rng = np.random.default_rng(seed)
    if test_idx.size < duration + 2 * ramp:
        raise ValueError(f"test qua ngan: {test_idx.size} < {duration + 2*ramp}")

    targets = np.sort(rng.choice(N, size=min(n_targets, N), replace=False))
    lo = int(rng.integers(0, test_idx.size - (duration + 2 * ramp) + 1))
    span = test_idx[lo: lo + duration + 2 * ramp]

    profile = np.concatenate([
        np.linspace(1.0, magnitude, ramp, endpoint=False),
        np.full(duration, magnitude),
        np.linspace(magnitude, 1.0, ramp, endpoint=False),
    ])
    spec = {
        "kind": "incident",
        "target_nodes": targets.tolist(),
        "window_start": int(span[0]), "window_end": int(span[-1]),
        "magnitude": float(magnitude), "duration": int(duration), "ramp": int(ramp),
        "profile": [float(x) for x in profile],
        "applies_to": "congestion_ratio (generator suy ra delay va travel_time)",
    }
    stats = {
        "n_targets": int(targets.size), "span_len": int(span.size),
        "frac_of_test": float(span.size / test_idx.size),
        "peak": float(magnitude), "seed": int(seed),
    }
    return spec, stats


def rush_mask(S: int, meta: dict) -> np.ndarray:
    """(S,) bool -- cua so nao roi vao gio cao diem.

    Dung cho rush-stratified METRIC (viec cua Nguoi 4), khong phai partition:
    tach rush/normal thanh hai tap roi nhau la BAT KHA THI voi bo du lieu nay,
    xem docs/khoa/03 §5.
    """
    hour, _ = recover_clock(S, meta["T_in"])
    return ((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 20))


def _min_dist(a: np.ndarray, b: np.ndarray) -> int:
    if a.size == 0 or b.size == 0:
        return 10 ** 9
    return int(np.abs(a[:, None] - b[None, :]).min())


def temporal_shift(S: int, meta: dict, scenario: str = "weekday_to_weekend",
                   seed: int = 0, val_size: int = 60):
    """
    Train tren ngay thuong -> test tren cuoi tuan.

    scenario chi con "weekday_to_weekend". "normal_to_rush" da bi bo:
    voi gap = T_in + T_out - 1, khong con cua so normal nao song sot sau purge
    (0/449 o T_out=24). Dung rush_mask() lam metric thay vi partition.

    seed KHONG duoc dung -- split nay hoan toan tat dinh. Giu tham so cho
    dong nhat chu ky voi cac co che khac.

    Returns
    -------
    splits : {"train": [...], "val": [...], "test": [...]}
    stats  : dict
    """
    if scenario != "weekday_to_weekend":
        raise ValueError(
            f"scenario khong ho tro: {scenario!r}. "
            "normal_to_rush bat kha thi, xem docs/khoa/03 §5."
        )

    gap = overlap_gap(meta)
    _, dow = recover_clock(S, meta["T_in"])

    test = np.flatnonzero(dow >= 5)
    weekday = np.flatnonzero(dow <= 4)
    if test.size == 0 or weekday.size == 0:
        raise ValueError("du lieu khong co du ca ngay thuong lan cuoi tuan")

    # Dung LUI tu test: test la tai nguyen khan hiem nhat, co dinh truoc.
    # val lay tu phan phoi TRAIN (ngay thuong) -- neu lay tu cuoi tuan thi
    # early-stopping se nhin thay phan phoi test => leakage kieu khac.
    val_end = test.min() - gap - 1
    cand = weekday[weekday <= val_end]
    if cand.size < val_size:
        raise ValueError(f"khong du cua so ngay thuong cho val: {cand.size} < {val_size}")
    val = cand[-val_size:]

    train_end = val.min() - gap - 1
    train = weekday[weekday <= train_end]
    if train.size == 0:
        raise ValueError("khong con cua so nao cho train sau khi tru 2 khoang gap")

    for a, b, name in [(train, val, "train-val"), (val, test, "val-test"),
                       (train, test, "train-test")]:
        assert_no_leakage(a, b, gap)

    splits = {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}
    stats = {
        "scenario": scenario,
        "gap_required": gap,
        "min_gap_actual": min(_min_dist(train, val), _min_dist(val, test),
                              _min_dist(train, test)),
        "n_train": int(train.size), "n_val": int(val.size), "n_test": int(test.size),
        "n_burned": int(S - train.size - val.size - test.size),
        "dow_train": sorted(int(x) for x in set(dow[train])),
        "dow_val": sorted(int(x) for x in set(dow[val])),
        "dow_test": sorted(int(x) for x in set(dow[test])),
        "rush_frac_test": float(rush_mask(S, meta)[test].mean()),
        "val_size": int(val_size), "seed": int(seed),
    }
    return splits, stats


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

# ══════════════════════════════════════════════════════════════
# INPUT MASKING v2 — che DAC TRUNG dau vao, khong chi che loss
# ══════════════════════════════════════════════════════════════
def apply_input_mask(X, mask, meta: dict, strategy: str = "flag",
                     A=None, fill_value: float = 0.0):
    """Che dac trung dau vao cua nhung (cua so, node) khong co du lieu.

    v1 (loss masking) chi bo cham diem node thieu du lieu -- dac trung cua no
    VAN nam trong input, nen model khong he bi ep phai suy tu hang xom.
    v2 che luon input: do moi la bai test that.

    Tham so
    -------
    X    : (S, N, T_in*F) -- torch.Tensor hoac np.ndarray
    mask : (S, N) bool    -- True = node CO du lieu tai cua so do
    strategy :
        "flag"      Them 1 kenh availability => F -> F+1. Dac trung bi che
                    dat ve fill_value, kenh moi bao 1/0. Model PHAN BIET duoc
                    "khong co du lieu" voi "duong thoang". MAC DINH.
        "zero"      Chi dat ve fill_value, khong co kenh bao. Baseline ngay tho:
                    model khong phan biet duoc thieu du lieu voi gia tri that.
        "neighbor"  Noi suy tu hang xom theo A. XEM CANH BAO BEN DUOI.
        "last_seen" Lay lai gia tri quan sat gan nhat cua chinh node do.

    CANH BAO ve "neighbor"
    ----------------------
    Plan goi y "thay the bang gia tri noi suy de ep model khai thac Zone
    Adjacency". Nhung neu TA noi suy ho thi model KHONG CAN hoc cach dung do
    thi nua -- ta da lam ho no roi. Noi suy lam bai toan DE DI, khong kho len.
    Chi "flag" moi that su ep model phai tu suy tu hang xom.

    => Dung "flag" lam mac dinh (dieu kien thi nghiem chinh), con "neighbor"
       va "last_seen" la BASELINE de so sanh: chung la can tren cua viec noi
       suy thu cong, va model zone-aware phai vuot duoc chung moi co y nghia.

    Tra ve
    ------
    X_masked : (S, N, T_in*F) -- hoac (S, N, T_in*(F+1)) khi strategy="flag"
    info     : dict -- ti le bi che, F moi, ...
    """
    is_torch = hasattr(X, "detach")
    Xn = X.detach().cpu().numpy().copy() if is_torch else np.array(X, copy=True)
    m = np.asarray(mask, dtype=bool)

    S, N, flat = Xn.shape
    T_in, F = meta["T_in"], meta["F"]
    if T_in * F != flat:
        raise ValueError(f"X co {flat} cot nhung T_in*F = {T_in}*{F} = {T_in*F}")
    if m.shape != (S, N):
        raise ValueError(f"mask phai co shape ({S}, {N}), nhan duoc {m.shape}")

    x = Xn.reshape(S, N, T_in, F)          # layout cua build_graph.py
    missing = ~m                            # (S, N)

    if strategy == "neighbor":
        if A is None:
            raise ValueError("strategy='neighbor' can ma tran ke A")
        An = A.detach().cpu().numpy() if hasattr(A, "detach") else np.asarray(A)
        for s in np.flatnonzero(missing.any(axis=1)):
            avail = m[s]                                    # (N,)
            if not avail.any():
                x[s][missing[s]] = fill_value
                continue
            w = An[:, avail]                                # (N, n_avail)
            denom = w.sum(axis=1, keepdims=True)
            src = x[s][avail]                               # (n_avail, T_in, F)
            interp = np.einsum("nk,ktf->ntf", w, src)
            interp = np.divide(interp, denom[:, :, None],
                               out=np.full_like(interp, fill_value),
                               where=denom[:, :, None] > 0)
            x[s][missing[s]] = interp[missing[s]]
    elif strategy == "last_seen":
        for v in range(N):
            last = None
            for s in range(S):
                if m[s, v]:
                    last = x[s, v].copy()
                elif last is not None:
                    x[s, v] = last
                else:
                    x[s, v] = fill_value
    elif strategy in ("flag", "zero"):
        x[missing] = fill_value
    else:
        raise ValueError(f"strategy khong hop le: {strategy!r}")

    if strategy == "flag":
        avail_ch = np.broadcast_to(m[:, :, None, None].astype(x.dtype),
                                   (S, N, T_in, 1))
        x = np.concatenate([x, avail_ch], axis=3)          # F -> F+1
        F_out = F + 1
    else:
        F_out = F

    out = x.reshape(S, N, T_in * F_out)
    if is_torch:
        import torch
        out = torch.tensor(out, dtype=X.dtype)

    info = {
        "strategy": strategy,
        "F_in": int(F), "F_out": int(F_out),
        "in_channels": int(T_in * F_out),
        "frac_masked": float(missing.mean()),
        "n_fully_dark_nodes": int((~m).all(axis=0).sum()),
        "fill_value": float(fill_value),
    }
    return out, info
