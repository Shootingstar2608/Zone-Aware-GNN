# Task C runbook (Windows PowerShell)

> Run from the repository root after applying the patch ZIP. These commands assume
> the team-approved HCM-Sim v2 contract is `N=17`, 7 days, 15-minute cadence,
> `T_in=12`, `T_out=24`. The supplied ZIPs did not include the v2 dataset/protocol.

## 0. Check required static inputs

```powershell
Test-Path data/raw/hcm_osrm_dataset.csv
Test-Path data/raw/zone_labels.csv
```

Both must be `True`. Do not substitute TomTom observed traffic for the synthetic input.

## 1. Build the synthetic traffic and graph dataset (only if v2 is not already present)

```powershell
python scripts/dev/generate_synthetic_traffic.py `
  --seed 42 `
  --days 7 `
  --start 2026-05-18 `
  --interval-min 15 `
  --out data/generated/hcm_sim_v2/hcm_sim_traffic.csv

python scripts/build_graph.py `
  --traffic-path data/generated/hcm_sim_v2/hcm_sim_traffic.csv `
  --out-dir data/processed/hcm_sim_v2 `
  --t-in 12 `
  --t-out 24
```

Important: the current `build_graph.py` default is `T_out=3`, so `--t-out 24`
must be explicit for this protocol. The generator currently reports its own
`simulator_version` separately from the dataset folder name; confirm with the
project lead that the `hcm_sim_v2` label is the approved dataset release name.

## 2. Validate and freeze protocol

```powershell
@'
import json, torch
from pathlib import Path
p = Path("data/processed/hcm_sim_v2")
meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
ds = torch.load(p / "graph_dataset.pt", map_location="cpu", weights_only=False)
assert meta["data_source"] == "synthetic", meta
assert meta["N"] == 17, meta
assert meta["T_in"] == 12 and meta["T_out"] == 24, meta
assert meta["S"] == 637, meta
assert ds["X"].shape[0] == 637 and ds["Y"].shape[-1] == 24
print("HCM-Sim v2 dataset checks passed:", meta)
'@ | python -

python scripts/freeze_hcm_sim_protocol.py --data-dir data/processed/hcm_sim_v2
```

The protocol file must be created at
`data/processed/hcm_sim_v2/benchmark_protocol_v1.json`.

## 3. Seasonal-naive causal test

```powershell
pytest -q tests/test_temporal_benchmark.py
```

## 4. Clean benchmark (8 models × 5 seeds = 40 records)

```powershell
New-Item -ItemType Directory -Force data/results/hcm_sim_v2 | Out-Null

python scripts/run_multi_seed.py `
  --data-dir data/processed/hcm_sim_v2 `
  --protocol data/processed/hcm_sim_v2/benchmark_protocol_v1.json `
  --models persistence mean_by_node seasonal_naive lstm gcn_gru stgcn baseline_ahgnn zone_full `
  --split-modes chrono `
  --seeds 1 2 3 4 5 `
  --epochs 100 `
  --save-checkpoints `
  --out data/results/hcm_sim_v2/clean_5seed.csv
```

If interrupted, rerun the same command with `--resume`. The patched runner honors
the exact `--out` path and refuses to silently overwrite an existing CSV.

## 5. Pilot statistics

```powershell
python scripts/stat_analysis.py `
  --runs data/results/hcm_sim_v2/clean_5seed.csv `
  --proposed zone_full `
  --metrics MAE RMSE MAPE WAPE `
  --out-prefix data/results/hcm_sim_v2/clean_5seed_statistics
```

Expected outputs: `_summary.csv`, `_significance.csv`, `_tables.tex`.

## 6. Quantity-skew pilot

```powershell
python scripts/eval_non_iid.py `
  --data-dir data/processed/hcm_sim_v2 `
  --partition-path data/partitions/hcm_sim_v2/partitions_meta.json `
  --models seasonal_naive lstm gcn_gru stgcn baseline_ahgnn zone_full `
  --alphas 0.1 0.3 1.0 `
  --seed 1 `
  --epochs 100 `
  --out-dir data/results/hcm_sim_v2/quantity_skew_seed1
```

Inspect the output CSV and log for all model/alpha combinations; report any
skipped or failed runs rather than presenting the pilot as complete.
