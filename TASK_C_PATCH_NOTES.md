# Task C patch — Zone-Aware-GNN

## Files changed
- `tests/test_temporal_benchmark.py`
- `utils/history.py`
- `scripts/eval_non_iid.py`
- `scripts/run_multi_seed.py`
- `scripts/stat_analysis.py`

## Apply
Extract this ZIP into the repository root and allow these five files to be overwritten.
Review with:
```powershell
git diff --check
git diff --stat
git diff
```

## What changed
1. Added causal seasonal-naive tests: exact previous-day alignment, invariance to changes in current/future targets, and fallback behavior.
2. Made seasonal-naive respect the observation mask in quantity-skew evaluation. It no longer reads a previous-day target for a node-window marked unavailable. Fallback uses persistence when current input is available, otherwise a train-visible per-node mean.
3. `run_multi_seed.py` now honors the exact `--out` path. It refuses to silently overwrite an existing CSV; rerun interrupted work with `--resume`.
4. Statistical defaults now include WAPE and the strongest-baseline search includes seasonal naive and AH-GNN, not only LSTM/GCN-GRU/STGCN.

## Validation performed
- `pytest -q tests/test_temporal_benchmark.py tests/test_benchmark.py tests/test_eval_protocol.py`: **52 passed**
- `python scripts/run_multi_seed.py --help`: passed
- `python scripts/eval_non_iid.py --help`: passed
- `python scripts/stat_analysis.py --help`: passed
- Broader test limitations in supplied ZIP:
  - `pytest -q` collection hits `scripts/quick_test.py`, which requires omitted `data/raw/hcm_osrm_dataset.csv`.
  - `pytest -q tests` has two failures because `data/config/hcm_tomtom_replacement_candidates.v1.json` and `data/config/hcm_tomtom_segments.v1.json` were not included in the uploaded source ZIP.
  - Excluding those data-dependent checks: **86 passed, 1 skipped**.

## Important dataset gap
The supplied ZIPs do **not** contain `data/processed/hcm_sim_v2/graph_dataset.pt`, `meta.json`, or `benchmark_protocol_v1.json`. The feature ZIP contains older root-level processed artifacts, not the required HCM-Sim v2 dataset. Therefore no 40-run benchmark or quantity-skew experiment has been run and no result metrics are claimed.

The current `scripts/build_graph.py` still defaults to `T_out=3`; when creating the planned 24-step HCM-Sim dataset, pass `--t-in 12 --t-out 24` explicitly. Do not silently change the legacy default.
