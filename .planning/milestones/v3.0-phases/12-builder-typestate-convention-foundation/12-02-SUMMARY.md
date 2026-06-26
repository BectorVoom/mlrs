---
phase: 12-builder-typestate-convention-foundation
plan: 02
subsystem: mlrs-algos
tags: [builder, typestate, umap-shell, hdbscan-shell, build-error, convention-foundation]
requires:
  - "mlrs_algos::typestate::{Fit, Transform, Unfit, Fitted} (Plan 01)"
  - "crates/mlrs-algos/src/error.rs (BuildError enum — extended in place)"
  - "crates/mlrs-algos/src/linear/mbsgd_regressor.rs (builder template, read-only)"
provides:
  - "mlrs_algos::manifold::umap::Umap<F, S = Unfit> + UmapBuilder (new module manifold/)"
  - "mlrs_algos::cluster::hdbscan::Hdbscan<F, S = Unfit> + HdbscanBuilder"
  - "BuildError::InvalidMinDist + BuildError::InvalidMinClusterSize (data-independent)"
  - "manifold::Metric/Init + cluster::hdbscan::{Metric, ClusterSelectionMethod} enums"
affects:
  - "Plan 03 (trybuild compile-fail gate — proves transform/labels-before-fit is E0599 on Unfit)"
  - "Plan 04 (PyO3 collapse — its any_estimator_typestate! Unfit { .. } arm must match these setters/defaults)"
  - "Phase 14 (UMAP algorithm fills the trivial fit body)"
  - "Phase 15 (HDBSCAN algorithm fills the trivial fit body + validates min_samples)"
tech-stack:
  added: []
  patterns:
    - "Per-estimator <F, S = Unfit> typestate (PhantomData<S>) with default Unfit"
    - "new() as single defaults source; builder Default re-derives via new().into_builder() (D-08)"
    - "Consuming-self Fit returning the Fitted-tagged sibling; accessors only on impl T<F, Fitted>"
    - "data-INDEPENDENT validation at build() -> BuildError; data-DEPENDENT geometry guard at fit() -> AlgoError"
    - "test-visible hyperparams_eq() for defaults-equality (DeviceArray is not PartialEq)"
key-files:
  created:
    - "crates/mlrs-algos/src/manifold/mod.rs"
    - "crates/mlrs-algos/src/manifold/umap.rs"
    - "crates/mlrs-algos/src/cluster/hdbscan.rs"
    - "crates/mlrs-algos/tests/umap_test.rs"
    - "crates/mlrs-algos/tests/hdbscan_test.rs"
  modified:
    - "crates/mlrs-algos/src/error.rs"
    - "crates/mlrs-algos/src/lib.rs"
    - "crates/mlrs-algos/src/cluster/mod.rs"
decisions:
  - "Open Question 1 (builder Default form): RESOLVED — `impl Default for {Umap,Hdbscan}Builder` calls `T::<f64, Unfit>::new().into_builder()`. f64 is pinned only to read the F-independent scalar defaults; the builder is non-generic so the pin is irrelevant. No hand-written literal defaults (Pitfall 4)."
  - "Hdbscan carries an extra PhantomData<F> (`_float`) because the shell stores no F-typed buffer (labels_ is i32); the F type param is retained for API uniformity with UMAP and Phase-15 readiness."
  - "min_samples=None is resolved to min_cluster_size at new()/build() (stored as Some(resolved)); min_samples itself is NOT validated in Phase 12 (deferred to Phase 15) and gets NO BuildError variant."
metrics:
  duration_min: 11
  completed: 2026-06-23
  tasks: 3
  files: 8
---

# Phase 12 Plan 02: Builder + Typestate Convention Foundation (Shells) Summary

Authored the two new-estimator SHELLS that demonstrate the Phase-12 convention
end-to-end — `Umap<F, S = Unfit>` (new `manifold/` module) and
`Hdbscan<F, S = Unfit>` (existing `cluster/` module) — each born builder-fronted,
typestate-tagged, with a NON-algorithmic trivial fit and `Fitted`-only accessors,
plus the two data-independent `BuildError` variants their `build()` validation
raises.

## What Was Built

Three source files and two algos test files, wired into `lib.rs`/`cluster/mod.rs`,
proving the v3 convention runs end-to-end (runtime round-trip + defaults equality
+ memory gate) without any algorithm. The real UMAP / HDBSCAN compute lands in
Phases 14 / 15; these shells give those phases a correct surface to fill.

## Exact Public Paths

- `mlrs_algos::manifold::umap::Umap<F, S = Unfit>` (+ re-export `mlrs_algos::manifold::Umap`)
- `mlrs_algos::manifold::umap::{UmapBuilder, Metric, Init}`
- `mlrs_algos::cluster::hdbscan::Hdbscan<F, S = Unfit>` (+ re-export `mlrs_algos::cluster::Hdbscan`)
- `mlrs_algos::cluster::hdbscan::{HdbscanBuilder, Metric, ClusterSelectionMethod}`
- `mlrs_algos::error::BuildError::InvalidMinDist { estimator, min_dist: f64 }`
- `mlrs_algos::error::BuildError::InvalidMinClusterSize { estimator, min_cluster_size: usize }`

## UMAP Builder Setters + new() Defaults (for Plan 04's PyO3 Unfit arm)

| Setter (`UmapBuilder::`) | Type | `Umap::new()` default |
|---|---|---|
| `n_neighbors` | `usize` | `15` |
| `n_components` | `usize` | `2` |
| `min_dist` | `f64` | `0.1` |
| `spread` | `f64` | `1.0` |
| `metric` | `Metric` | `Metric::Euclidean` |
| `n_epochs` | `Option<usize>` | `None` |
| `init` | `Init` | `Init::Spectral` |
| `random_state` | `Option<u64>` | `None` |
| `learning_rate` | `f64` | `1.0` |
| `set_op_mix_ratio` | `f64` | `1.0` |
| `local_connectivity` | `f64` | `1.0` |
| `repulsion_strength` | `f64` | `1.0` |
| `negative_sample_rate` | `usize` | `5` |
| `a` | `Option<f64>` | `None` |
| `b` | `Option<f64>` | `None` |

UMAP `build()` validation: `min_dist` must be finite and `<= spread`, else
`BuildError::InvalidMinDist`. Fitted-only surface: `embedding(&self, pool) -> Vec<F>`,
`n_features_in(&self) -> usize`, and `impl Transform<F> for Umap<F, Fitted>`.

## HDBSCAN Builder Setters + new() Defaults (for Plan 04's PyO3 Unfit arm)

| Setter (`HdbscanBuilder::`) | Type | `Hdbscan::new()` default |
|---|---|---|
| `min_cluster_size` | `usize` | `5` |
| `min_samples` | `Option<usize>` | `Some(5)` (None → `min_cluster_size`) |
| `cluster_selection_epsilon` | `f64` | `0.0` |
| `cluster_selection_method` | `ClusterSelectionMethod` | `ClusterSelectionMethod::Eom` |
| `metric` | `Metric` | `Metric::Euclidean` |
| `alpha` | `f64` | `1.0` |
| `max_cluster_size` | `usize` | `0` |

HDBSCAN `build()` validation: `min_cluster_size >= 2`, else
`BuildError::InvalidMinClusterSize`. `min_samples` is stored verbatim and NOT
validated this phase (deferred to Phase 15). Fitted-only surface:
`labels(&self, pool) -> Vec<i32>` (all `-1` noise sentinel), `n_features_in(&self) -> usize`.
HDBSCAN is labels-only — NO `Predict`/`Transform` impl.

## Open Question 1 Resolution (builder `Default` form)

`impl Default for UmapBuilder` / `HdbscanBuilder` both call
`T::<f64, Unfit>::new().into_builder()`. This makes `new()` the single source of
default truth (D-08); the `f64` pin only reads the F-independent scalar defaults
and is irrelevant since the builder struct is non-generic. No literal defaults are
hand-written in `Default` (Pitfall 4). Verified by the `defaults_equal` tests:
`T::new().hyperparams_eq(&T::builder().build::<F>()?)` holds for both shells.

## Tasks & Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add the two BuildError variants | `3486a29` | `error.rs` |
| 2 | UMAP shell + module wiring + umap_test | `ebc9018` | `manifold/mod.rs`, `manifold/umap.rs`, `lib.rs`, `umap_test.rs` |
| 3 | HDBSCAN shell + module wiring + hdbscan_test | `b17e26f` | `cluster/hdbscan.rs`, `cluster/mod.rs`, `hdbscan_test.rs` |

Task 2 commits `manifold/umap.rs` together with the `lib.rs` `pub mod manifold;`
registration because the module only compiles once registered (inseparable for a
green build); same rationale binds `cluster/hdbscan.rs` to the `cluster/mod.rs`
edit in Task 3.

## Verification Results

- `cargo test -p mlrs-algos --features cpu --test umap_test --test hdbscan_test` —
  8 passed (4 + 4): `defaults_equal`, `build_rejects_bad_min_dist` /
  `build_rejects_bad_min_cluster_size`, `fit_roundtrip`, `fit_no_leak` per shell.
- `cargo build -p mlrs-algos --features cpu` — green (full crate compiles; the new
  `BuildError` variants + module wiring break no existing estimator).
- `git diff --stat crates/mlrs-algos/src/traits.rs` — empty (frozen-trait
  invariant held; the new surface lives in `typestate.rs`, untouched here).
- `crates/mlrs-py/src/errors.rs` `build_err_to_py` maps via `err.to_string()`
  (no exhaustive match), so the two new variants need NO PyO3 mapper change —
  Plan 04 reuses it unchanged.

The full `cargo test -p mlrs-algos --features cpu` regression (~6 min per MEMORY,
reduce_test/svd_test dominate) was NOT run inline; the targeted 8-test gate +
green full-crate build cover this plan's surface. The rocm f32 GPU gate is the
opportunistic end-of-phase manual check (`umap_test fit_roundtrip` — f64 skips
with log, f32 round-trips).

## Threat Model Coverage

- T-12-02 (out-of-range hyperparameter reaching fit allocation) — MITIGATED:
  validated at `build()` before data → `InvalidMinDist` / `InvalidMinClusterSize`;
  the only fit allocation (`n * n_components` zeros / `n` labels) is bounded by
  validated/`shape` values. Verified by the `build_rejects_*` tests.
- T-12-03 (malformed geometry at fit) — MITIGATED: geometry guard
  (`n==0 || p==0 || x.len() != n*p`) → `AlgoError::Prim(PrimError::ShapeMismatch)`
  before the allocation in both fit bodies.
- T-12-04 (hyperparameter check leaking into fit) — MITIGATED: all hyperparameter
  validation lives in `build()`; only the data-dependent geometry guard lives in
  `fit()`. The `build_rejects_*` tests assert the error surfaces at `build()`.

## Deviations from Plan

None — plan executed exactly as written. All Rules 1–4 were no-ops (pure
convention shells, no architectural change).

One minor, plan-anticipated implementation choice: `Hdbscan` carries a
`PhantomData<F>` (`_float`) field because its only fitted buffer is `i32` labels
(no `F`-typed storage in the shell). The plan's struct sketch named the
`F`-generic `labels_: Option<DeviceArray<ActiveRuntime, i32>>` but did not call
out that `F` then needs a phantom to remain "used"; the `_float` marker is the
zero-cost mechanism. Documented inline.

## Known Stubs

The fit/transform bodies are INTENTIONAL non-algorithmic shells (zeros embedding /
all-`-1` labels), documented in-source as "real UMAP lands in Phase 14" /
"real HDBSCAN lands in Phase 15". This is the explicit goal of Plan 02 (a
convention-demonstration shell, D-10), not a blocking stub: the plan's success
criterion is the convention running end-to-end at runtime, which the 8 passing
tests prove. The named future phases (14 / 15) own filling the real algorithm and
the deferred `min_samples` semantic validation.

## Self-Check: PASSED

- FOUND: crates/mlrs-algos/src/manifold/mod.rs
- FOUND: crates/mlrs-algos/src/manifold/umap.rs
- FOUND: crates/mlrs-algos/src/cluster/hdbscan.rs
- FOUND: crates/mlrs-algos/tests/umap_test.rs
- FOUND: crates/mlrs-algos/tests/hdbscan_test.rs
- FOUND: commit 3486a29 (Task 1)
- FOUND: commit ebc9018 (Task 2)
- FOUND: commit b17e26f (Task 3)
