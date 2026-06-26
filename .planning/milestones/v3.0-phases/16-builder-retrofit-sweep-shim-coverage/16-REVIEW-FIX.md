---
phase: 16-builder-retrofit-sweep-shim-coverage
fixed_at: 2026-06-25T00:00:00Z
review_path: .planning/phases/16-builder-retrofit-sweep-shim-coverage/16-REVIEW.md
iteration: 1
findings_in_scope: 10
fixed: 9
skipped: 1
status: partial
---

# Phase 16: Code Review Fix Report

**Fixed at:** 2026-06-25
**Source review:** .planning/phases/16-builder-retrofit-sweep-shim-coverage/16-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 10 (all WR-* and IN-* — `fix_scope: all`)
- Fixed: 9
- Skipped: 1

All fixes were verified by `cargo check -p mlrs-algos -p mlrs-py --features cpu --lib`
(both crates compile cleanly; only pre-existing dead-code warnings in `spectral.rs`
remain) and by `python ast.parse` on the edited shims. Commits are grouped so that each
source file appears in exactly one commit (several findings share a file — e.g.
`classifier.rs` carries WR-01, WR-02, and IN-02 — and the gsd commit tool stages whole
files, so co-located findings are committed together and noted below).

## Fixed Issues

### WR-01: Python `classes_` discards the core's real (non-contiguous) labels

**Files modified:** `crates/mlrs-algos/src/linear/logistic.rs`, `crates/mlrs-algos/src/neighbors/classifier.rs`, `crates/mlrs-py/src/estimators/linear.rs`, `crates/mlrs-py/src/estimators/neighbors.rs`, `crates/mlrs-py/python/mlrs/linear.py`, `crates/mlrs-py/python/mlrs/neighbors.py`
**Commits:** c74aa8b (logistic core accessor), 110a110 (knn core accessor), e47a7a0 (PyO3 + shims)
**Applied fix:** Added a public `classes()` accessor to the `LogisticRegression<F, Fitted>`
and `KNeighborsClassifier<F, Fitted>` cores (returning the DISTINCT sorted training
labels). Exposed a `classes_()` `#[pymethods]` getter on `PyLogisticRegression` and
`PyKNeighborsClassifier` (the KNN one widens the core's `i32` labels to `i64` to match the
SGD/SVM sibling signature). Changed `LogisticRegression.fit` and `KNeighborsClassifier.fit`
shims from `np.arange(obj.n_classes())` to `np.asarray(obj.classes_(), dtype=np.int32)`, so
a non-contiguous target (e.g. `{0, 2}`) now round-trips through `predict`.

### WR-02: `KNeighborsClassifier::fit` skips integer/i32-range label validation

**Files modified:** `crates/mlrs-algos/src/neighbors/classifier.rs`
**Commit:** 110a110
**Applied fix:** Replaced the unguarded `host_to_f64(v).round() as i32` map with a validated
loop that rejects non-finite, non-integer, or out-of-`i32`-range labels via
`AlgoError::InvalidLabels { estimator: "knn_classifier", ... }`, mirroring the
`decode_classes` / `mbsgd_classifier` guards. Prevents a NaN target silently becoming `0`
and an out-of-range label saturating to a spurious class.

### WR-03: Neighbors PyO3 wrappers used the panic-on-poison lock path

**Files modified:** `crates/mlrs-py/src/estimators/neighbors.rs`
**Commit:** e47a7a0
**Applied fix:** Replaced all 10 occurrences of
`crate::global_pool().lock().expect("pool mutex")` with the sanctioned `crate::lock_pool()`
(same guard type, recovers a poisoned mutex and re-baselines pool accounting). The three
neighbor estimators are now inside the brick-prevention guarantee like the cluster/linear
wrappers.

### WR-04: Random-projection / IncrementalPCA transforms omit the `n_samples == 0` guard

**Files modified:** `crates/mlrs-algos/src/projection/gaussian.rs`, `crates/mlrs-algos/src/decomposition/incremental_pca.rs`
**Commit:** 47e34fc
**Applied fix:** Added `n_samples == 0 ||` to the geometry guard in `gaussian.rs::project`
and in `incremental_pca.rs`'s `transform` and `inverse_transform`, so a degenerate empty
query returns a typed `ShapeMismatch` instead of launching a zero-row GEMM — matching the
rest of the crate.

### WR-05: `KernelRidge::fit` builds `n_targets` without checking divisibility

**Files modified:** `crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs`
**Commit:** 315a5c2
**Applied fix:** Made the divisibility intent explicit: reject `n_samples == 0`,
`y.len() == 0`, or `y.len() % n_samples != 0` up front with a typed `ShapeMismatch` before
computing `n_targets = y.len() / n_samples`, so the contract no longer relies on the
post-hoc equality re-check. (The downstream equality guard is left in place as harmless
defense-in-depth.)

### WR-06: `KMeans.fit` shim passes a raw numpy `random_state` without int-coercion

**Files modified:** `crates/mlrs-py/python/mlrs/cluster.py`
**Commit:** b5c6b6f
**Applied fix:** Normalized at the boundary — `seed = None if self.random_state is None else
int(self.random_state)` — mirroring `SpectralClustering.fit`, so a numpy integer scalar
coerces cleanly and a negative value fails with a clear `int(...)`/Python error rather than
an opaque PyO3 `OverflowError`. `None` is forwarded unchanged (PyKMeans maps it to a default
seed).

### IN-01: `LogisticRegression::fit` used `ShapeMismatch` for a label-VALIDITY failure

**Files modified:** `crates/mlrs-algos/src/linear/logistic.rs`
**Commit:** c74aa8b
**Applied fix:** Both the non-integer/negative-label branch and the `< 2 classes` branch now
return `AlgoError::InvalidLabels { estimator: "logistic_regression", reason: ... }` with an
honest reason string, matching the sibling classifiers, instead of a fabricated
`PrimError::ShapeMismatch`.

### IN-02: KNN builders mis-labeled the `n_neighbors == 0` rejection as `InvalidNComponents`

**Files modified:** `crates/mlrs-algos/src/error.rs`, `crates/mlrs-algos/src/neighbors/classifier.rs`, `crates/mlrs-algos/src/neighbors/regressor.rs`, `crates/mlrs-algos/src/neighbors/nearest.rs`
**Commit:** 110a110
**Applied fix:** Added a neighbor-honest `BuildError::InvalidNNeighbors { estimator,
n_neighbors }` variant and switched all three builders (`knn_classifier` / `knn_regressor` /
`nearest_neighbors`) plus their doc references to it. (Note: the pre-existing
`InvalidNComponents` variant the code used already carried `param: "n_neighbors"`, so the
user-facing message was already correct; this change makes the variant NAME match the
hyperparameter too.) `build_err_to_py` uses `err.to_string()` so no exhaustive-match update
was needed.

### IN-03: `LedoitWolf` memoizes host attrs but ignores `pool` after the first call

**Files modified:** `crates/mlrs-algos/src/covariance/ledoit_wolf.rs`
**Commit:** 0d4cc86
**Applied fix:** Documentation-only (the finding explicitly required no behavior change):
added a prominent note to `covariance_` and `location_` stating that `pool` is consulted
ONLY on the first (memoized) call and ignored thereafter, and why that is sound (the device
buffer is immutable post-fit).

## Skipped Issues

### IN-04: `expect("shared path is never plane-gated to None")` couples estimators to a reduce-prim invariant

**File:** `crates/mlrs-algos/src/linear/linear_regression.rs:293`, `ridge.rs:315`, `covariance/ledoit_wolf.rs:257`, `density/kernel_density.rs:448`
**Reason:** skipped — no semantically appropriate typed-error target exists, and forcing a
wrong-fit variant would degrade the diagnostic. `PrimError` (defined in `mlrs-core`) has only
`ShapeMismatch`, `DimMismatch`, `NotSquare`, `NotConverged`, `NotPositiveDefinite`, and
`Overflow`; none represents an "internal invariant violated / prim returned unexpected
`None`" condition. The reviewer's suggested fix `.ok_or(AlgoError::Prim(PrimError::...))?`
does not name a concrete variant precisely because none fits. Inspecting `column_reduce`
(`crates/mlrs-backend/src/prims/reduce.rs:225`) confirms it returns `None` ONLY on the
`Plane` path when the adapter lacks subgroup support; on the `Shared` path used by these four
call sites it ALWAYS returns `Some`, so the `expect` is a genuinely-unreachable documented
invariant, and the finding itself rates it "Low risk." Adding a new `PrimError` variant in
`mlrs-core` for this would be a cross-crate public-API change well beyond the finding's scope
and risks breaking exhaustive matches across the backend crate. Mapping to `NotConverged`
(the nearest existing variant) would emit a misleading "did not converge within N sweeps"
message for a reduce that never converges — strictly worse than the current honest
`expect` message. The safer action is to leave the documented-invariant `expect` in place.

**Original issue:** `column_reduce(.., ReducePath::Shared, ..)?.expect(...)` panics (not a
typed error) if the reduce prim ever returns `None` on the Shared path; a future change to
the prim's plane-gating would turn these into process panics across the PyO3 boundary rather
than a propagated `AlgoError`.

---

_Fixed: 2026-06-25_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
