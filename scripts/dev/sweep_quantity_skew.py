"""
sweep_quantity_skew.py
======================
Quet 3 cach dien giai Dirichlet cua quantity_skew de CHON mode mac dinh.

Chay:  python scripts/dev/sweep_quantity_skew.py

Tieu chi chon: mode nao giu `coverage` PHANG khi `gini` bien thien.
Neu alpha nho vua lech hon vua it du lieu hon thi truc x cua figure
"Performance vs Heterogeneity" tron hai bien => khong ket luan duoc gi.

Script nay chi dung de ra quyet dinh thiet ke, khong nam trong pipeline.
Ket qua da ghi vao docs/khoa/03_benchmark_partition_design.md §4.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.partition_gen import load_meta, quantity_skew

MODES = ["budget", "relative", "fixed_coverage"]
ALPHAS = [0.1, 0.5, 1.0, 5.0]
SEEDS = list(range(42, 52))


def main():
    meta = load_meta()
    S, N = meta["S"], meta["N"]
    print(f"S={S}  N={N}  tran tuyet doi N*S={N*S}  seeds={SEEDS[0]}..{SEEDS[-1]}\n")
    print(f"{'mode':<16}{'alpha':>6} {'coverage':>16} {'gini':>16} {'n_zero':>8}")
    print("-" * 66)

    flat = {}
    for mode in MODES:
        for a in ALPHAS:
            cov, gi, nz = [], [], []
            for s in SEEDS:
                _, st = quantity_skew(S, N, a, s, mode=mode)
                cov.append(st["coverage"])
                gi.append(st["gini"])
                nz.append(st["n_zero_nodes"])
            f = lambda x: f"{np.mean(x):.3f}±{np.std(x):.3f}"
            print(f"{mode:<16}{a:>6} {f(cov):>16} {f(gi):>16} {np.mean(nz):>8.1f}")
            flat.setdefault(mode, []).append((np.mean(cov), np.mean(gi)))
        print()

    print("=" * 66)
    for mode, rows in flat.items():
        c = [r[0] for r in rows]
        g = [r[1] for r in rows]
        print(f"  {mode:<16} coverage bien thien {max(c)-min(c):.3f} | "
              f"gini trai rong {max(g)-min(g):.3f}")
    print("\n  Chon mode co coverage bien thien ~0 va gini trai rong nhat.")


if __name__ == "__main__":
    main()
