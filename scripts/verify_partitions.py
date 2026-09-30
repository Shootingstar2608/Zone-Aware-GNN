"""
scripts/verify_partitions.py
============================
Kiem chung DoD: "moi partition tai tao duoc tu JSON metadata".

Doc partitions_meta.json, sinh lai tung partition tu (scenario, params, seed),
roi so mask_hash / splits / spec voi ban da commit. Lech mot bit la fail.

Chi can numpy. Chay: python scripts/verify_partitions.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from benchmark.partition_gen import (            # noqa: E402
    OUT_PATH, concept_drift, dataset_fingerprint, load_meta, load_partitions,
    load_zone_matrix, mask_hash, quantity_skew, temporal_shift, zone_skew,
)


def main():
    meta = load_meta()
    recs = load_partitions(OUT_PATH)
    Z = load_zone_matrix(meta)
    now_fp = dataset_fingerprint()
    S, N = meta["S"], meta["N"]

    bad, checked = [], {"mask": 0, "splits": 0, "spec": 0}

    for rec in recs:
        pid, p, seed = rec["partition_id"], rec["params"], rec["seed"]

        if rec["dataset_fingerprint"] != now_fp:
            bad.append(f"{pid}: dataset_fingerprint lech "
                       f"({rec['dataset_fingerprint'][:12]} != {now_fp[:12]})")
            continue

        sc = rec["scenario"]

        if sc == "quantity_skew":
            mask, _ = quantity_skew(S, N, alpha=p["alpha"], seed=seed,
                                    block_len=p["block_len"], mode=p["mode"],
                                    c_bar=p["c_bar"])
            checked["mask"] += 1
            if mask_hash(mask) != rec["mask_hash"]:
                bad.append(f"{pid}: mask_hash lech")

        elif sc == "zone_skew":
            mask, _ = zone_skew(Z, S, meta, n_clusters=p["n_clusters"], seed=seed,
                                c_bar=p["c_bar"],
                                off_band_weight=p["off_band_weight"])
            checked["mask"] += 1
            if mask_hash(mask) != rec["mask_hash"]:
                bad.append(f"{pid}: mask_hash lech")

        elif sc == "temporal_shift":
            splits, _ = temporal_shift(S, meta, scenario=p["scenario"])
            checked["splits"] += 1
            for k in ("train", "val", "test"):
                if list(splits[k]) != list(rec["splits"][k]):
                    bad.append(f"{pid}: splits['{k}'] khong khop")
                    break

        elif sc == "concept_drift":
            base, _ = temporal_shift(S, meta)
            spec, _ = concept_drift(base["test"], N, seed=seed,
                                    n_targets=p["n_targets"])
            checked["spec"] += 1
            if spec != rec["perturbation_spec"]:
                bad.append(f"{pid}: perturbation_spec khong khop")

        else:
            bad.append(f"{pid}: scenario la '{sc}' — khong biet kiem the nao")

    print(f"  partition     : {len(recs)}")
    print(f"  kiem mask_hash: {checked['mask']}")
    print(f"  kiem splits   : {checked['splits']}")
    print(f"  kiem spec     : {checked['spec']}")
    print(f"  fingerprint   : {now_fp[:16]}...")

    if bad:
        print(f"\n  ❌ {len(bad)} loi:")
        for b in bad:
            print(f"     - {b}")
        return 1
    print("\n  ✅ Moi partition tai tao duoc bit-for-bit tu metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())