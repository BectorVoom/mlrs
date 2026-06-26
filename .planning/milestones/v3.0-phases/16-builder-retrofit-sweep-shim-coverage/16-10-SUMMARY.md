---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 10
subsystem: mlrs-py (PyO3 FFI binding)
tags: [shim, pyo3, umap, hdbscan, transform, fit_predict, SHIM-02, D-08]
requires:
  - "PyUMAP wrap (manifold.rs) — shipped, registered lib.rs:265-266"
  - "PyHDBSCAN wrap (cluster.rs) — shipped, registered lib.rs:265-266"
  - "Umap<F, Fitted>::Transform::transform (umap.rs:568) + Umap<F, Unfit>::fit_transform (umap.rs:215)"
  - "Hdbscan<F, Unfit>::fit_predict (hdbscan.rs:284) + Hdbscan<F, Fitted>::{probabilities,outlier_scores} (hdbscan.rs:966/974)"
provides:
  - "PyUMAP.transform_f32 / PyUMAP.transform_f64 (dtype-split #[pymethods])"
  - "PyUMAP.fit_transform_f32 / PyUMAP.fit_transform_f64 (dtype-split #[pymethods])"
  - "PyHDBSCAN.fit_predict (i32 labels, dtype-agnostic)"
  - "PyHDBSCAN.probabilities_f32 / probabilities_f64 (Option<Vec<F>>)"
  - "PyHDBSCAN.outlier_scores_f32 / outlier_scores_f64 (Option<Vec<F>>)"
affects:
  - "Plan 11 (mlrs.UMAP / mlrs.HDBSCAN pure-Python shim classes — forward to these wraps)"
tech-stack:
  added: []
  patterns:
    - "dtype-suffixed #[pymethods] forwarder (a #[pyclass] method can't be generic over F; split f32/f64 + arm-match) — PyRidge predict_f32/predict_f64 template (linear.rs:291-316)"
    - "ClusterMixin fit_predict = fit-then-return-labels; mutates self into Fitted arm"
    - "Option<Vec<F>> algos accessor -> Python None (probabilities_/outlier_scores_ None until feature-space front-end lands)"
key-files:
  created: []
  modified:
    - "crates/mlrs-py/src/estimators/manifold.rs (PyUMAP transform/fit_transform + unfit_hyperparams helper + Transform import)"
    - "crates/mlrs-py/src/estimators/cluster.rs (PyHDBSCAN fit_predict + probabilities_/outlier_scores_ getters)"
decisions:
  - "fit_predict mutates self into the Fitted arm (calls existing fit then reads labels) rather than building a throwaway — the shim wants the estimator fitted for subsequent labels_/probabilities_ access, and sklearn ClusterMixin.fit_predict has identical semantics"
  - "probabilities_/outlier_scores_ return Option (algos return Option<Vec<F>>) -> surfaces as Python None when unavailable, rather than erroring"
  - "transform returns DeviceArray -> .to_host_metered(&mut pool) (PyRidge predict pattern), unlike the embedding_ getter which calls e.embedding(&pool) directly (returns Vec)"
metrics:
  duration: ~6 min
  completed: 2026-06-24
  tasks: 2
  files: 2
  commits: 2
status: complete
---

# Phase 16 Plan 10: SHIM-02 — PyUMAP/PyHDBSCAN method-gap fill Summary

Filled the verified method gaps on the existing PyUMAP/PyHDBSCAN PyO3 wraps: PyUMAP now exposes `transform`/`fit_transform` and PyHDBSCAN exposes `fit_predict`/`probabilities_`/`outlier_scores_`, all dtype-split forwarders onto the already-shipped Rust `Fitted`/`Unfit` typestate methods — completing the D-08 UMAP/HDBSCAN Python boundary surface so Plan 11's `mlrs.UMAP`/`mlrs.HDBSCAN` shim classes can forward to a complete wrap.

## What Was Built

### Task 1 — PyUMAP transform + fit_transform (manifold.rs)
- `transform_f32` / `transform_f64`: arm-match `AnyUmap::{F32,F64}`, `validated_f32/f64` upload, call `Transform::transform` on the fitted `Umap<F, Fitted>`, return host `Vec<F>` via `.to_host_metered(&mut pool)`. `not_fitted("umap", "transform (…)")` on the `Unfit`/wrong-dtype arm.
- `fit_transform_f32` / `fit_transform_f64`: build via the umap builder (data-independent validation -> `build_err_to_py`), call `Umap::<F, Unfit>::fit_transform` (consumes the estimator, returns the embedding). Does NOT mutate `self` (sklearn `fit_transform` returns the embedding; the fitted estimator is dropped).
- Added `use mlrs_algos::typestate::Transform` and a private `unfit_hyperparams()` helper (extracts the hyperparameter tuple from the `Unfit` arm; `not_fitted` analog on an already-fitted arm).
- Commit: `5b7e18e`

### Task 2 — PyHDBSCAN fit_predict + probabilities_ + outlier_scores_ (cluster.rs)
- `fit_predict`: calls the existing `fit` (build/guard_f64/lock_pool/TypestateFit) then returns the i32 `labels_` (dtype-agnostic arm-match, like `labels_inner`). Mutates `self` into the `Fitted` arm — sklearn `ClusterMixin.fit_predict` semantics.
- `probabilities_f32`/`probabilities_f64` and `outlier_scores_f32`/`outlier_scores_f64`: dtype-suffixed getters forwarding to `Hdbscan<F, Fitted>::{probabilities,outlier_scores}` (both return `Option<Vec<F>>` -> Python `None` when unavailable). `not_fitted("hdbscan", …)` on the `Unfit`/wrong-dtype arm.
- Commit: `98eab15`

## New Method Names (for Plan 11 shim forwarding)

| Wrap | #[pymethods] | Rust target | Return |
|------|--------------|-------------|--------|
| PyUMAP | `transform_f32(x, rows, cols)` / `transform_f64(...)` | `Transform::transform` on `Umap<F, Fitted>` | `Vec<f32>` / `Vec<f64>` |
| PyUMAP | `fit_transform_f32(x, rows, cols)` / `fit_transform_f64(...)` | `Umap::<F, Unfit>::fit_transform` | `Vec<f32>` / `Vec<f64>` |
| PyHDBSCAN | `fit_predict(x, rows, cols)` | `fit` + `labels()` | `Vec<i32>` |
| PyHDBSCAN | `probabilities_f32()` / `probabilities_f64()` | `Hdbscan<F, Fitted>::probabilities` | `Option<Vec<f32>>` / `Option<Vec<f64>>` |
| PyHDBSCAN | `outlier_scores_f32()` / `outlier_scores_f64()` | `Hdbscan<F, Fitted>::outlier_scores` | `Option<Vec<f32>>` / `Option<Vec<f64>>` |

Note: the existing `labels_()`, `embedding_f32/f64()`, `is_fitted()`, `dtype()` getters and the `fit` methods were NOT modified; wrap registration (lib.rs:265-266) needs no change (new `#[pymethods]` on an existing `#[pyclass]`).

## Threat Model Compliance

| Threat ID | Mitigation applied |
|-----------|--------------------|
| T-16-GUARDF64 | `guard_f64()` BEFORE the F64 upload on `transform_f64`/`fit_transform_f64`; on `fit_predict` it runs inside the delegated `fit` F64 arm |
| T-16-V5 | `validated_f32/f64` ingress on the transform paths; the Rust `Transform::transform` runs its own `n_features_in_` geometry guard (WR-02) |
| T-16-POISON | `lock_pool()` (poison-recovering) on every new method — no `.lock().expect()` |
| T-16-NOTFIT | the `Unfit`/wrong-dtype arm returns `not_fitted(...)` on transform/probabilities_/outlier_scores_/fit_predict |

## Deviations from Plan

None — plan executed as written. Both methods of each wrap were implemented as dtype-split forwarders per PATTERNS §5. The plan's grep acceptance counts (>=2 for manifold, >=3 for cluster) were exceeded (4 and 5 respectively) because the dtype-split produces two helpers per logical method, which the plan explicitly permits ("dtype-split helpers acceptable").

## Deferred Issues (out of scope)

- `cargo clippy -p mlrs-py --features cpu` surfaces a pre-existing error in `mlrs-kernels/src/elementwise.rs` (`approximate value of FRAC_PI_2` — a transcendental kernel constant). This is in a different crate, predates this plan, and is unrelated to the FFI binding work. The plan's verification gate is `cargo build` (green); clippy on the new files (manifold.rs/cluster.rs) is clean. Not fixed.

## Verification

- `cargo build -p mlrs-py --features cpu` — GREEN (only pre-existing spectral.rs dead-field warnings).
- `grep -cE 'fn transform|fn fit_transform' manifold.rs` = 4 (>=2 required).
- `grep -cE 'fn fit_predict|fn probabilities|fn outlier_scores' cluster.rs` = 5 (>=3 required).
- `guard_f64` present on both files; `not_fitted` present on every new fitted-arm accessor.

## Self-Check: PASSED
- FOUND: crates/mlrs-py/src/estimators/manifold.rs
- FOUND: crates/mlrs-py/src/estimators/cluster.rs
- FOUND commit: 5b7e18e (PyUMAP transform/fit_transform)
- FOUND commit: 98eab15 (PyHDBSCAN fit_predict/probabilities_/outlier_scores_)
