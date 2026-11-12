"""Load and validate frozen chronological benchmark split artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_frozen_splits(protocol_path: str | Path, dataset_path: str | Path, meta_path: str | Path) -> tuple[list[int], list[int], list[int]]:
    """Return train/val/test only when the protocol matches the exact data."""
    artifact = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if artifact.get("dataset_sha256") != sha256(dataset_path):
        raise RuntimeError("Frozen protocol belongs to another graph_dataset.pt; re-freeze split deliberately.")
    if artifact.get("meta_sha256") != sha256(meta_path):
        raise RuntimeError("Frozen protocol belongs to another meta.json; re-freeze split deliberately.")
    splits = artifact.get("splits", {})
    train, val, test = splits.get("train"), splits.get("validation"), splits.get("test")
    if not all(isinstance(x, list) and x for x in (train, val, test)):
        raise ValueError("Frozen protocol has incomplete train/validation/test splits")
    gap = int(artifact["purge_gap"])
    if min(val) - max(train) < gap or min(test) - max(val) < gap:
        raise ValueError("Frozen protocol violates its purge gap")
    return train, val, test
