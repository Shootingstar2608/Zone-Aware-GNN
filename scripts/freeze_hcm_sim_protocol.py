"""Freeze the reproducible chronological protocol for one HCM-Sim dataset.

This creates an immutable-in-practice split manifest.  Training scripts may
read it rather than silently recomputing a split after a dataset changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from utils.eval_protocol import TRAIN_RATIO, VAL_RATIO, chronological_split


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="Freeze chronological HCM-Sim split manifest.")
    p.add_argument("--data-dir", default="data/processed/hcm_sim_v1")
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true", help="Replace an existing manifest deliberately.")
    args = p.parse_args()
    data_dir = (ROOT / args.data_dir).resolve()
    dataset_path, meta_path = data_dir / "graph_dataset.pt", data_dir / "meta.json"
    out = Path(args.out).resolve() if args.out else data_dir / "benchmark_protocol_v1.json"
    if out.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {out}; inspect it or use --force deliberately.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data = torch.load(dataset_path, weights_only=False)
    S = int(data["X"].shape[0])
    if S != int(meta["S"]):
        raise ValueError("meta S does not match graph_dataset X")
    gap = int(meta["T_in"]) + int(meta["T_out"]) - 1
    train, val, test = chronological_split(S, TRAIN_RATIO, VAL_RATIO, gap)
    artifact = {
        "protocol_version": "hcm-sim-chrono-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": sha256(dataset_path),
        "meta_path": str(meta_path.relative_to(ROOT)),
        "meta_sha256": sha256(meta_path),
        "T_in": meta["T_in"], "T_out": meta["T_out"],
        "purge_gap": gap,
        "ratios": {"train": TRAIN_RATIO, "validation": VAL_RATIO, "test": 1 - TRAIN_RATIO - VAL_RATIO},
        "splits": {"train": train, "validation": val, "test": test},
        "invariants": {
            "train_end_before_validation_start": max(train) < min(val),
            "validation_end_before_test_start": max(val) < min(test),
            "train_validation_gap": min(val) - max(train),
            "validation_test_gap": min(test) - max(val),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen split manifest: {out}")
    print(f"train={len(train)}, val={len(val)}, test={len(test)}, purge_gap={gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
