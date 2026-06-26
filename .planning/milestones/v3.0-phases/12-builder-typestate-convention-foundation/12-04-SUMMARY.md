---
phase: 12-builder-typestate-convention-foundation
plan: 04
subsystem: mlrs-py
tags: [pyo3, typestate, dispatch-macro, umap-shell, hdbscan-shell, bldr-04, convention-foundation]
requires:
  - "crates/mlrs-py/src/dispatch.rs any_estimator! (existing 35-call-site macro, read-only)"
  - "mlrs_algos::manifold::umap::Umap<F, S = Unfit> + UmapBuilder + Fitted embedding accessor (Plan 02)"
  - "mlrs_algos::cluster::hdbscan::Hdbscan<F, S = Unfit> + HdbscanBuilder + Fitted labels accessor (Plan 02)"
  - "mlrs_algos::typestate::Fit (consuming-self) (Plan 01)"
  - "crates/mlrs-py/src/errors.rs {algo_err_to_py, build_err_to_py, not_fitted} (reused unchanged, D-13)"
provides:
  - "crate::any_estimator_typestate! — second additive dispatch macro spelling fitted arms <F, Fitted> (D-04)"
  - "estimators::manifold::PyUMAP (#[pyclass name=\"UMAP\"]) — first PyO3 shell over a v3 typestate estimator"
  - "estimators::cluster::PyHDBSCAN (#[pyclass name=\"HDBSCAN\"]) — labels-only typestate shell"
  - "PyUMAP/PyHDBSCAN registered in mlrs-py/src/lib.rs"
affects:
  - "Phase 14 (UMAP algorithm fills the trivial fit — PyUMAP surface already correct)"
  - "Phase 15 (HDBSCAN algorithm fills the trivial fit)"
  - "Phase 16 (retrofit of all 30 estimators — the same any_estimator_typestate! pattern wraps each retrofitted shell)"
  - "pure-Python mlrs shim (UAT — subclasses sklearn UMAP/HDBSCAN, delegates to these pyclasses)"
tech-stack:
  added: []
  patterns:
    - "Second additive dispatch macro (clone, not edit) to protect the shared 35-call-site any_estimator! (D-04)"
    - "Consuming-fit PyO3 shell: build().map_err(build_err_to_py)? then typestate::Fit::fit returning the Fitted arm"
    - "Fit-name alias (TypestateFit) at a call site that already imports the legacy traits::Fit (Pitfall 1)"
    - "sklearn-named enum STRINGS stored verbatim in the Unfit arm; parsed to typed enums at fit (→ ValueError)"
    - "Runtime not_fitted analog on the Unfit arm = the Python-boundary counterpart of Plan 03's compile-time gate"
key-files:
  created:
    - "crates/mlrs-py/src/estimators/manifold.rs"
    - "crates/mlrs-py/tests/manifold_test.rs"
  modified:
    - "crates/mlrs-py/src/dispatch.rs"
    - "crates/mlrs-py/src/estimators/cluster.rs"
    - "crates/mlrs-py/src/estimators/mod.rs"
    - "crates/mlrs-py/src/lib.rs"
decisions:
  - "Open Question 2 (macro): RESOLVED — a SECOND additive macro any_estimator_typestate! (byte-for-byte clone of any_estimator! with ONLY the two fitted arms spelling <F, Fitted>), NOT an edit to the shared arm. Lowest risk to Success Criterion 3 — the existing macro is untouched, so all 35 no-S call sites keep their exact monomorphization."
  - "Open Question 3 (HDBSCAN location): RESOLVED — PyHDBSCAN lives in estimators/cluster.rs with the cluster family (alongside PyKMeans/PyDBSCAN), so estimators/mod.rs needs no edit beyond the new manifold module."
  - "Fit-name collision: cluster.rs already imports traits::Fit (for PyKMeans/PyDBSCAN); the typestate Fit is imported as `TypestateFit` and called UFCS (`TypestateFit::fit(est, ...)`) so the two Fit names never collide in one file."
  - "min_samples is Option<usize> in the PyO3 Unfit arm (not a usize sentinel) — PyO3 maps Python None directly to Option, and the algos builder already resolves None→min_cluster_size; cleaner than a 0-sentinel."
metrics:
  duration_min: 14
  completed: 2026-06-23
  tasks: 4
  files: 6
---

# Phase 12 Plan 04: PyO3 Typestate Collapse (PyUMAP / PyHDBSCAN) Summary

Collapsed the v3 builder + typestate convention behind the existing PyO3
dtype-dispatch machinery so the Python surface is UNCHANGED (BLDR-04). A second
additive dispatch macro (`any_estimator_typestate!`) plus two hand-written
`#[pyclass]` shells (`PyUMAP`, `PyHDBSCAN`) prove the convention is invisible to
Python — same `Unfit/F32/F64` enum, same error mappers, same `py.detach` +
`lock_pool` + `guard_f64` fit contract — while every one of the 35 existing
`any_estimator!` call sites stays green.

## What Was Built

- **A second additive macro** `any_estimator_typestate!` (`dispatch.rs`): the
  byte-for-byte clone of `any_estimator!` whose two fitted arms spell the state
  marker explicitly — `F32($algo<f32, mlrs_algos::typestate::Fitted>)` /
  `F64($algo<f64, ...::Fitted>)`. Because the v3 estimators default `S = Unfit`,
  a bare `$algo<f32>` would store the WRONG `Unfit` monomorphization (RESEARCH
  Pitfall 2). The existing `any_estimator!` is untouched (grep count == 2;
  `git show` of the macro commit is `+63 / -0`).
- **`PyUMAP`** (new `estimators/manifold.rs`, 295 lines): `#[pyclass name="UMAP"]`
  over `Umap<F, Fitted>`. Consuming `fit` (no `y` — unsupervised) builds the
  `Unfit` estimator, validates the data-independent hyperparameters at `build()`
  BEFORE the upload (`build_err_to_py` → `ValueError`), guards f64 on the F64 arm,
  and stores the `Fitted` sibling returned by the consuming `typestate::Fit::fit`.
  `embedding_f32`/`embedding_f64` accessors return `not_fitted` on the
  `Unfit`/wrong-dtype arm. `unfit_default`/`is_unfit` cross-crate smoke seam.
- **`PyHDBSCAN`** (extends `estimators/cluster.rs`): `#[pyclass name="HDBSCAN"]`
  over `Hdbscan<F, Fitted>`. Labels-only (NO standalone predict, algos D-08); the
  same consuming-`fit`/`not_fitted` contract; `labels_` accessor returns
  `Vec<i32>`.
- **Wiring**: `pub mod manifold;` in `estimators/mod.rs`; `PyUMAP` + `PyHDBSCAN`
  imported and registered in `lib.rs`.
- **Tests** (`tests/manifold_test.rs`): `typestate_shells_construct_unfit`
  (BLDR-04 cross-crate smoke) + `not_fitted_before_fit` (D-13 runtime analog).

## Exact Public Surface

- `crate::any_estimator_typestate!` (macro_export)
- `mlrs_py::estimators::manifold::PyUMAP` — `unfit_default()`, `is_unfit()`,
  `embedding_f32_for_test()`, `labels`-free; `#[pymethods]`: `new`, `fit`,
  `embedding_f32`, `embedding_f64`, `is_fitted`, `dtype`
- `mlrs_py::estimators::cluster::PyHDBSCAN` — `unfit_default()`, `is_unfit()`,
  `labels_for_test()`; `#[pymethods]`: `new`, `fit`, `labels_`, `is_fitted`,
  `dtype`

The `*_for_test` inherent methods are the Rust-callable accessor seam for the
cross-crate not-fitted test (the live PyO3 boundary path runs in UAT, MEMORY).

## Open Question 2 Resolution (second macro, NOT a shared-arm edit)

Chose a SECOND additive macro over editing the shared `any_estimator!` fitted
arm. The shared macro backs 35 existing (no-`S`) call sites; editing its `F32`
arm to `$algo<f32, Fitted>` would break every legacy estimator (whose
`Estimator<f32>` has no `S` parameter). The clone isolates the change to the two
new typestate shells and makes Success Criterion 3 a tautology — the existing
macro body is unchanged, so the 35 call sites cannot regress.

## Open Question 3 Resolution (HDBSCAN in cluster.rs)

`PyHDBSCAN` lives in `estimators/cluster.rs` with `PyKMeans`/`PyDBSCAN` (the
cluster family), not a new file. Only `estimators/mod.rs` edit this plan needs is
the new `pub mod manifold;` (for UMAP).

## The Fit-Name Alias (cluster.rs)

`cluster.rs` already imports `mlrs_algos::traits::{Fit, PredictLabels}` (the
legacy `&mut self` `Fit` used by `PyKMeans`/`PyDBSCAN`). `PyHDBSCAN` needs the
consuming `mlrs_algos::typestate::Fit`. Importing both as `Fit` would collide, so
the typestate one is `use mlrs_algos::typestate::Fit as TypestateFit;` and called
UFCS: `TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))`. This is the
minimal-change resolution (Pitfall 1 at the call-site level). `manifold.rs` has no
such collision — it imports only the typestate `Fit`, so it calls `est.fit(...)`
directly.

## Tasks & Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Second additive macro any_estimator_typestate! | `547b146` | `dispatch.rs` |
| 2 | PyUMAP shell + mod.rs + lib.rs wiring | `618a576` | `manifold.rs`, `mod.rs`, `lib.rs` |
| 3 | PyHDBSCAN shell (extend cluster.rs) + lib.rs wiring | `e342a23` | `cluster.rs`, `lib.rs` |
| 4 | Cross-crate smoke + not-fitted tests | `58eed06` | `manifold_test.rs` |

`lib.rs` is touched in Tasks 2 and 3 (registering each pyclass as it lands) so
every commit is a green build — Task 2 registers only `PyUMAP`, Task 3 adds
`PyHDBSCAN`.

## Verification Results

- `cargo build -p mlrs-py --features cpu` — green after every task (both shells +
  the second macro compile; the existing macro untouched).
- `cargo test -p mlrs-py --features cpu` — ALL test binaries pass: the new
  `manifold_test` (2 passed: `typestate_shells_construct_unfit`,
  `not_fitted_before_fit`) PLUS every pre-existing suite — `allocator_test` (3),
  `ingress_test` (7), `probe_test` (4), `pyclass_smoke_test` (4),
  `sgd_smoke_test` (3), `spectral_smoke_test` (2). Success Criterion 3 / BLDR-04
  confirmed: no existing `any_estimator!` call site regressed.
- `grep -c "macro_rules! any_estimator" dispatch.rs` == 2 (both macros present).
- `git show 547b146` is `+63 / -0` on `dispatch.rs` — pure addition, the existing
  `any_estimator!` arm is byte-for-byte unchanged.

The full `cargo test --features cpu` cross-crate regression (~6 min per MEMORY)
was NOT run inline; this plan's surface is entirely in `mlrs-py` and the full
`mlrs-py --features cpu` suite is green. The rocm f32 GPU gate (f64 skips with
log) is the opportunistic end-of-phase manual check.

## Threat Model Coverage

- T-12-02 (out-of-range hyperparameter reaching the fit allocation) — MITIGATED:
  both shells call `builder()...build::<F>().map_err(build_err_to_py)?` BEFORE the
  device upload; the data-independent `BuildError` (Plan 02) fires first and maps
  to `PyValueError`.
- T-12-07 (f64 on an f64-incapable backend) — MITIGATED:
  `crate::capability::guard_f64()?` on the F64 arm BEFORE upload (`linear.rs`
  contract).
- T-12-08 (panicked fit poisoning the pool mutex) — MITIGATED: both shells use
  `crate::lock_pool()` (poison-recovering, WR-04), never
  `global_pool().lock().expect()`.
- T-12-09 (predict/accessor-before-fit at the Python boundary) — MITIGATED:
  runtime `not_fitted("umap"|"hdbscan", op)` on the `Unfit` arm → `PyValueError`
  the shim re-raises as `NotFittedError` (D-13). Verified by
  `not_fitted_before_fit`.

## Deviations from Plan

None — plan executed exactly as written. Rules 1–4 were no-ops.

Two plan-anticipated implementation choices, both already flagged in the plan
text: (1) the `Fit`-alias (`TypestateFit`) called UFCS in `cluster.rs` (the plan
named the alias as the minimal change); (2) `min_samples` stored as
`Option<usize>` rather than a `usize` 0-sentinel — PyO3 maps Python `None`
directly to `Option`, and the algos builder already resolves `None →
min_cluster_size`, so the `Option` is the cleaner and equivalent form.

## Deferred / UAT

The live PyO3 estimator pytest (interpreter + pyarrow capsule FFI through the
real `fit`/`embedding_`/`labels_` boundary, and the concrete `PyValueError`
class assertion) cannot run in this environment (no maturin/pyarrow per MEMORY
"Python wheel untestable in env" / SHIM-03). Routed to UAT. The Rust-side gates
compensate: the consuming `fit` body, the `build()`/`guard_f64()` chain, and the
`not_fitted` runtime analog are all exercised here without an interpreter, and
the typed error source the boundary relies on is unchanged from Plan 02.

## Known Stubs

The UMAP/HDBSCAN fit bodies are the INTENTIONAL non-algorithmic shells from Plan
02 (zeros embedding / all-`-1` labels), documented in-source as "real UMAP lands
in Phase 14" / "real HDBSCAN lands in Phase 15". This plan adds only the PyO3
surface over those shells — it introduces no new stub. The plan's goal (the
convention invisible to Python, the dispatch enum + error mappers + fit contract
identical to the legacy shells) is met and verified by the passing tests.

## Self-Check: PASSED

- FOUND: crates/mlrs-py/src/estimators/manifold.rs
- FOUND: crates/mlrs-py/tests/manifold_test.rs
- FOUND: any_estimator_typestate! in crates/mlrs-py/src/dispatch.rs
- FOUND: PyHDBSCAN in crates/mlrs-py/src/estimators/cluster.rs
- FOUND: PyUMAP + PyHDBSCAN registered in crates/mlrs-py/src/lib.rs
- FOUND: commit 547b146 (Task 1)
- FOUND: commit 618a576 (Task 2)
- FOUND: commit e342a23 (Task 3)
- FOUND: commit 58eed06 (Task 4)
