---
status: complete
phase: 12-builder-typestate-convention-foundation
source: [12-VERIFICATION.md]
started: 2026-06-23T02:13:16Z
updated: 2026-06-26T00:00:00Z
resolved: 2026-06-26
---

## Current Test

number: 1
name: Live PyO3 estimator integration for PyUMAP and PyHDBSCAN
expected: |
  In an environment with maturin + pyarrow installed, build the wheel
  (`maturin develop --features cpu`) and verify via the real PyO3 capsule boundary:
  - `UMAP().fit(X)` stores AnyUmap::F32/F64; `embedding_` returns an `(rows, n_components)`
    array (real embedding — UMAP graduated from the zeros shell to the full SGD layout in Phase 14).
  - `HDBSCAN().fit(X)` stores AnyHdbscan::F32/F64; `labels_` returns `(rows,)` int32
    (real cluster labels — HDBSCAN graduated from the all-`-1` shell in Phase 15).
  - Both raise NotFittedError / PyValueError when an accessor is called before fit.
  - `build_err_to_py` surfaces `BuildError::InvalidMinDist` / `InvalidMinClusterSize`
    as Python ValueError.
result: passed

## Tests

### 1. Live PyO3 estimator integration for PyUMAP and PyHDBSCAN
expected: |
  Build the wheel with maturin (cpu feature) and, through the real Python interpreter
  + pyarrow capsule FFI path:
  - UMAP().fit(X) stores AnyUmap::F32/F64; embedding_ returns an (rows, n_components) array.
  - HDBSCAN().fit(X) stores AnyHdbscan::F32/F64; labels_ returns (rows,) int32.
  - Accessing an estimator before fit raises NotFittedError / PyValueError.
  - BuildError::InvalidMinDist / InvalidMinClusterSize surface as Python ValueError.
result: passed
evidence: |
  Resolved 2026-06-26. A venv was provisioned (maturin 1.14, pyarrow 24, numpy 2.5,
  sklearn 1.9 — PyPI reachable in this run, the host the gate said was absent), the cpu
  wheel was built (`maturin develop -m crates/mlrs-py/Cargo.toml --features cpu,extension-module`,
  exit 0, mlrs-cpu-0.1.0 installed), and the live UAT script
  (scratchpad/uat_12_live_ffi.py) was run through the real interpreter + pyarrow capsule:
  ALL 22 assertions PASS for BOTH dtype arms (f32 + f64):
  - UMAP fit → embedding_ (75, 2) finite; fit_transform; transform(X_new) (10, 2);
    same random_state byte/value-reproducible.
  - HDBSCAN fit → labels_ (75,) int32 in {-1..k}; probabilities_ ∈ [0, 1].
  - Unfit accessor (UMAP embedding_, HDBSCAN labels_) raises NotFittedError.
  - UMAP(min_dist=2.0, spread=1.0).fit → ValueError "min_dist = 2 is invalid …";
    HDBSCAN(min_cluster_size=1).fit → ValueError "min_cluster_size = 1 … (must be >= 2)".
  NOTE: the original Phase-12 expectation (zeros-shell embedding / all-`-1` labels) reflected
  the pre-algorithm shells; UMAP (Phase 14) and HDBSCAN (Phase 15) now do real work, so the
  live test asserts real embeddings/labels — a strictly stronger result than the shell gate.

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

None — the single live-FFI scenario passed end-to-end (see test 1 evidence).
