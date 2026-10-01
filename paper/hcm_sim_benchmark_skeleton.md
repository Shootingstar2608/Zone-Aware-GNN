# HCM-Sim: A Reproducible and Controllable Benchmark for Non-IID Urban Traffic Forecasting

## Abstract — draft outline

Spatiotemporal traffic forecasting is commonly evaluated under stationary,
independently distributed assumptions despite deployment settings involving
uneven observation coverage, functional-zone shifts, temporal distribution
shifts, incidents, and missing inputs. We introduce HCM-Sim, a reproducible
synthetic benchmark built on an OSRM-derived urban road graph and OSM-derived
functional-zone labels. HCM-Sim provides controlled Non-IID scenarios,
provenance manifests, leakage-safe chronological evaluation, and deterministic
partition generation. We benchmark representative temporal and graph-based
forecasting baselines under IID and Non-IID conditions, quantify degradation
with multi-seed statistical reporting, and document limitations of synthetic
traffic generation. HCM-Sim is intended as a controlled robustness benchmark,
not as a replacement for observed traffic datasets.

## 1. Problem

Existing traffic benchmarks seldom isolate why a model fails when node coverage,
zone composition, temporal regime, or sensor availability shifts. Real traffic
datasets are indispensable, but often cannot expose a ground-truth mechanism
for a particular Non-IID failure mode. The paper asks: how can forecasting
methods be evaluated reproducibly under controllable, auditable Non-IID shifts?

## 2. Benchmark Design

HCM-Sim combines an OSRM-derived directed road graph and OSM-derived multi-label
functional zones with rule-based synthetic dynamic traffic. The release includes
traffic features, future congestion-ratio targets, exact time metadata, dataset
hashes, generator parameters, and versioned manifests. The dynamic traffic is
synthetic; the paper makes no claim that it is observed TomTom traffic.

## 3. Non-IID Scenarios

The benchmark contains: (i) quantity skew across nodes, (ii) zone-cluster skew,
(iii) weekday-to-weekend temporal shift, (iv) controllable incident/concept
drift, and (v) input masking with explicit missingness indicators or imputation.
Each scenario records parameters, random seed, split indices, purge gap, and a
hash for exact regeneration.

## 4. Reproducibility and Evaluation Protocol

Headline experiments use chronological train/validation/test splits with a
purge gap of `T_in + T_out - 1`. Normalization is fitted only on training data;
predictions are inverse-transformed before metric computation. Results report
at least ten seeds, Mean ± Std, 95% confidence intervals, effect sizes, and
Holm-corrected significance tests. The repository includes unit tests, a
partition verifier, and continuous integration.

## 5. Experimental Questions

1. How much do temporal, graph-based, and naive forecasters degrade under each
   Non-IID mechanism and severity?
2. Which shift types are most harmful at short and long prediction horizons?
3. Do missing-input strategies improve robustness consistently?
4. Are benchmark conclusions stable over random initialization and partition
   seeds?

## 6. Limitations and Scope

HCM-Sim has synthetic dynamic traffic, currently limited graph scale, and
generator-defined zone effects. It is a controlled robustness benchmark rather
than evidence of real-world deployment performance. A future extension should
add larger graphs, multiple city instances, and external validation on licensed
or public observed traffic datasets.
