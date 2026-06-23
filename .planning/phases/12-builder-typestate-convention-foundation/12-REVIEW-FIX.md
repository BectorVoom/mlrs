---
phase: 12-builder-typestate-convention-foundation
fixed_at: 2026-06-23T00:00:00Z
review_path: .planning/phases/12-builder-typestate-convention-foundation/12-REVIEW.md
iteration: 1
findings_in_scope: 10
fixed: 9
skipped: 1
status: partial
---

# Phase 12: Code Review Fix Report

**Fixed at:** 2026-06-23T00:00:00Z
**Source review:** .planning/phases/12-builder-typestate-convention-foundation/12-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 10 (5 Warning + 5 Info; fix_scope = all)
- Fixed: 9
- Skipped: 1

All fixes were verified by compiling `mlrs-algos` and `mlrs-py` with
`--features cpu` and running the affected test crates (`compile_fail`,
`umap_test`, `hdbscan_test`, `typestate_test`) — all green.

## Fixed Issues

### WR-01: `n_components == 0` accepted, producing a silently-empty embedding

**Files modified:** `crates/mlrs-algos/src/error.rs`, `crates/mlrs-algos/src/manifold/umap.rs`
**Commit:** 55dd87a
**Applied fix:** Added a new `BuildError::InvalidNComponents { estimator, param, value }`
variant and guarded `n_components >= 1` and `n_neighbors >= 1` in
`UmapBuilder::build` before the `Ok(...)`. `build_err_to_py` maps via
`err.to_string()` so the new variant needs no PyO3-side match update. Verified:
crate compiles; `umap_test` (build_rejects_bad_min_dist, fit_roundtrip, etc.)
passes.

### WR-02: `Transform::transform` for `Umap<F, Fitted>` ignored fitted `n_features_in_`

**Files modified:** `crates/mlrs-algos/src/manifold/umap.rs`, `crates/mlrs-algos/src/typestate.rs`, `crates/mlrs-algos/src/cluster/hdbscan.rs`
**Commit:** 691e5b6
**Applied fix:** Added the `p != self.n_features_in_` check (returns the documented
`PrimError::ShapeMismatch`) to `Umap::transform`, fulfilling the `Transform` trait
doc contract. Committed together with IN-03 because the transform feature-check
and the shared-helper extraction touch the same `transform` hunk. **Requires
human verification:** this changes runtime behavior (a previously-silent
wrong-shape path now errors); confirm the equality semantics match the intended
fit-vs-transform feature contract.

### WR-03: `cluster.rs` mixed the sanctioned `lock_pool` with the panicking lock form

**Files modified:** `crates/mlrs-py/src/estimators/cluster.rs`
**Commit:** b769507
**Applied fix:** Converted all eight `global_pool().lock().expect("pool mutex")`
sites in `PyKMeans`/`PyDBSCAN` to `crate::lock_pool()`, so the whole file now
participates in the poison-recovery path (`PyHDBSCAN` already did). Verified:
`mlrs-py` compiles with `--features cpu` (the 2 remaining warnings are
pre-existing dead-code warnings in unrelated estimators, not introduced here).

### WR-04: HDBSCAN left `min_samples`/`max_cluster_size` unvalidated with no follow-up

**Files modified:** `crates/mlrs-algos/src/cluster/hdbscan.rs`
**Commit:** ca026d4
**Applied fix:** Added a `// TODO(phase-15)` at the `HdbscanBuilder::build` guard
documenting the deferred `min_samples >= 1` (when `Some`) and `max_cluster_size`
checks, with the `AlgoError::InvalidMinSamples` precedent noted. Documentation/
tracking fix per the review (the review explicitly defers the actual validation
to Phase 15).

### WR-05: `predict_before_fit.stderr` golden brittle and elided `Unfit`

**Files modified:** `crates/mlrs-algos/tests/ui/predict_before_fit.rs`, `crates/mlrs-algos/tests/ui/predict_before_fit.stderr`
**Commit:** 2175c5d
**Applied fix:** Rewrote the fixture to the `E0277` trait-bound assertion style
(asserting `Umap<f32, Unfit>: Transform<f32>`, mirroring `transform_before_fit.rs`)
so the diagnostic prints `Umap<f32, Unfit>` verbatim on its primary help line.
Regenerated the golden with `TRYBUILD=overwrite` under the pinned rustc 1.96.0,
then re-ran `compile_fail` without overwrite — both ui fixtures pass and the
golden now names `Unfit` robustly.

### IN-01: UMAP and HDBSCAN shells diverged in method order and doc text

**Files modified:** `crates/mlrs-algos/src/cluster/hdbscan.rs`
**Commit:** 74e5dd7
**Applied fix:** Reordered the HDBSCAN shell so `hyperparams_eq` precedes
`into_builder` (matching UMAP) and restored UMAP's "available to callers who want
to tweak a constructed estimator before fitting" sentence to `into_builder`'s doc.

### IN-03: Duplicated geometry-guard block across `fit` and `transform`

**Files modified:** `crates/mlrs-algos/src/typestate.rs`, `crates/mlrs-algos/src/manifold/umap.rs`, `crates/mlrs-algos/src/cluster/hdbscan.rs`
**Commit:** 691e5b6 (with WR-02)
**Applied fix:** Extracted a crate-private `typestate::validate_geometry(x, shape)`
helper and called it from the three duplicated `fit`/`transform` guard sites
(UMAP fit, UMAP transform, HDBSCAN fit), removing the now-unused
`mlrs_core::PrimError` import from `hdbscan.rs`.

### IN-04: `_state_phantom` helper exported (doc-hidden) but unused

**Files modified:** `crates/mlrs-algos/tests/typestate_test.rs`
**Commit:** ed7389e
**Applied fix:** Added a `typestate_test` case that invokes `_state_phantom` for
both `Unfit` and `Fitted`, keeping the exported helper compiled-exercised (the
review's preferred option over dropping the surface). Test passes.

### IN-05: Two near-identical dispatch macros kept in lockstep only by prose

**Files modified:** `crates/mlrs-py/src/dispatch.rs`
**Commit:** f683310
**Applied fix:** Added cross-link comments at both `any_estimator!` and
`any_estimator_typestate!` noting they are byte-for-byte clones (except the
fitted-arm `Fitted`-state spelling) that must be edited in tandem.

## Skipped Issues

### IN-02: Single-variant `Metric` enum with a write-only stored field

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:49-53`, `crates/mlrs-algos/src/manifold/umap.rs:44-48`
**Reason:** skipped — the review's Fix explicitly states "No action for the shell;
confirm Phase 14/15 consumes `metric`." This is intentional shell scaffolding (the
trivial fit ignores `metric`); the field becomes load-bearing when the real
algorithms land in Phases 14/15. No code change is warranted in Phase 12, so this
is left as a forward-looking note rather than a fix.
**Original issue:** `Metric` has exactly one variant (`Euclidean`) and the stored
`metric` field is verbatim-stored but ignored by the trivial fit, so it is
currently write-only and a linter may flag it.

---

_Fixed: 2026-06-23T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
