---
phase: 05-distance-based-iterative-solver-estimators
fixed_at: 2026-06-13T16:40:00Z
review_path: .planning/phases/05-distance-based-iterative-solver-estimators/05-REVIEW.md
iteration: 1
findings_in_scope: 14
fixed: 12
skipped: 2
status: partial
---

# Phase 5: Code Review Fix Report

**Fixed at:** 2026-06-13T16:40:00Z
**Source review:** .planning/phases/05-distance-based-iterative-solver-estimators/05-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 14 (fix_scope = all: 3 critical + 7 warning + 4 info)
- Fixed: 12
- Skipped: 2

All fixes were verified Tier 1 (re-read) + Tier 2 (`cargo check`/`cargo build --tests`
with `--features cpu`). The WR-03 regression test was additionally run and passes.
Three findings are correctness/logic changes flagged for human verification (see
notes per finding).

## Fixed Issues

### CR-01: KMeans empty-cluster relocation can drive a donor count negative

**Files modified:** `crates/mlrs-backend/src/prims/kmeans.rs`
**Commit:** 5827b86
**Status:** fixed — requires human verification (logic change to the relocation algorithm)
**Applied fix:** Rewrote `lloyd_update`'s relocation loop to mirror sklearn's
`_relocate_empty_clusters_dense`: walk the farthest-first ranking and, for each
empty cluster, pick the next candidate whose donor still has `count >= 2` and that
was not already relocated (tracked via a `relocated: Vec<bool>`). If no valid donor
remains, return a typed `PrimError::ShapeMismatch` instead of silently leaving a
centroid at the origin. The finalize loop now carries a `debug_assert!(count > 0)`
invariant.

### CR-02: LogisticRegression infers n_classes from max(label)+1

**Files modified:** `crates/mlrs-algos/src/linear/logistic.rs`
**Commit:** 6bfa48b
**Status:** fixed — requires human verification (label-remap logic + new error path)
**Applied fix:** Added a `classes_: Vec<i64>` field holding the distinct sorted
training labels. `fit` now collects distinct labels, rejects `< 2` classes with a
typed error, remaps `y` to a dense `[0, n_classes)` device buffer that the softmax
kernel consumes, and `predict_labels` maps each argmax column back through
`classes_` to recover the original id. The remapped device buffer is released after
the solve.

### CR-03: KNeighborsClassifier n_classes uses train max+1

**Files modified:** `crates/mlrs-algos/src/neighbors/classifier.rs`
**Commit:** bf73db4
**Status:** fixed — requires human verification (label-remap logic)
**Applied fix:** Added a `classes_: Vec<i32>` field of distinct sorted training
labels. `fit` stores each sample's DENSE class index (position in `classes_`);
`predict_proba` indexes proba columns by that dense position; `predict_labels` maps
the argmax column back through `classes_`. The existing WR-02 bounds guard is
retained as defensive. Module doc updated.

### WR-01: lbfgs_minimize reports a line-search breakdown as success

**Files modified:** `crates/mlrs-backend/src/prims/lbfgs.rs`, `crates/mlrs-algos/src/linear/logistic.rs`
**Commit:** b880b4f
**Status:** fixed
**Applied fix:** Added a `LbfgsStopReason` enum (Converged / FtolStall /
LineSearchFailed / MaxIter) and a `stop_reason` field to `LbfgsResult`, set at each
loop exit. `LogisticRegression::fit` now surfaces `AlgoError::NotConverged` whenever
`stop_reason == LineSearchFailed`, regardless of iteration count.

### WR-02: LogReg objective closure leaks pool buffers on a panic

**Files modified:** `crates/mlrs-algos/src/linear/logistic.rs`
**Commit:** 6e97a37
**Status:** fixed
**Applied fix:** Introduced a `ScratchGuard<'a, F>` RAII guard that owns the
per-iteration `w_d`/`b_d` scratch buffers plus a mutable borrow of the pool and
returns both handles to the free-list in its `Drop`. The softmax launch borrows the
pool and buffers through `guard.parts()`, so a panic during the launch unwinds the
guard and releases the handles.

### WR-03: KMeans never surfaces non-convergence + tol_scaled can be zero

**Files modified:** `crates/mlrs-algos/src/cluster/kmeans.rs`, `crates/mlrs-algos/tests/kmeans_test.rs`
**Commit:** 1c0b2af
**Status:** fixed
**Applied fix:** Documented KMeans's non-convergence contract (matches sklearn:
never errors, returns best-effort; `tol_scaled == 0` for a constant-feature design
is intentional). Added the `wr03_constant_feature_design_does_not_error` regression
test asserting a constant-feature fit succeeds with valid in-range labels. Test was
run and passes.

### WR-04: DBSCAN adjacency held as both u32 and bool

**Files modified:** `crates/mlrs-backend/src/prims/dbscan.rs`, `crates/mlrs-backend/tests/dbscan_mask_test.rs`
**Commit:** 1196185
**Status:** fixed
**Applied fix:** Changed `EpsCoreMask::adjacency` from `Vec<bool>` to the kernel's
native `Vec<u32>` (single representation); `neighbors()` now tests `!= 0`. Removed
the parallel `Vec<bool>` copy that was held live alongside the u32 buffer. Updated
the one direct-index test assertion to `!= 0`.

### WR-05: kmeanspp_sample degenerate fallback expects rather than errors

**Files modified:** `crates/mlrs-backend/src/prims/kmeans.rs`
**Commit:** e99cca3
**Status:** fixed
**Applied fix:** Track chosen-index membership in an `is_chosen: Vec<bool>` (O(1)
lookup, replacing the O(k·n) `chosen.contains`) and return a typed
`PrimError::ShapeMismatch` when no unused index remains instead of `expect`.

### WR-06: inertia / inertia_rows_host omit the k >= 1 guard

**Files modified:** `crates/mlrs-backend/src/prims/kmeans.rs`
**Commit:** aaea715
**Status:** fixed
**Applied fix:** Added an explicit `if k == 0 { return Err(ShapeMismatch{...}) }`
rejection to both `inertia` and `inertia_rows_host` after deriving
`k = centers.len() / d`, for parity with `validate_geometry`'s `1 <= k`.

### WR-07: cd_solve reads y back to host twice

**Files modified:** `crates/mlrs-backend/src/prims/coordinate_descent.rs`
**Commit:** d51b938
**Status:** fixed
**Applied fix:** Capture the single `y.to_host()` read-back as `y_raw: Vec<F>`, use
it both to build the f64 host state and to seed `r_dev`, removing the duplicate
device→host transfer at the residual seed.

### IN-03: KMeans exposes no max_iter / tol override

**Files modified:** `crates/mlrs-algos/src/cluster/kmeans.rs`
**Commit:** e4e23ae
**Status:** fixed
**Applied fix:** Added `KMeans::with_opts(n_clusters, seed, max_iter, tol)` for
parity with `Lasso::with_opts` / `LogisticRegression::with_opts`.

### IN-04: Dead SEED constant kept alive by a let _ = SEED;

**Files modified:** `crates/mlrs-algos/tests/kmeans_test.rs`
**Commit:** 5bb7130
**Status:** fixed
**Applied fix:** Removed the unused `const SEED: u64 = 42;` (the `let _ = SEED;`
suppression was already removed by the WR-03 edit). Test build is warning-clean.

## Skipped Issues

### IN-01: fit_predict round-trips labels host→device→host→device

**File:** `crates/mlrs-algos/src/cluster/dbscan.rs:131-140`; `crates/mlrs-algos/src/cluster/kmeans.rs:415-424`
**Reason:** skipped — no safe in-scope fix available. The only device-clone path
without a host round-trip is `DeviceArray::from_raw(handle.clone(), len)`, but the
CubeCL `Handle` is ref-counted, so a cloned handle would ALIAS the same underlying
buffer as the stored `labels_`; a later `release_into` on either would free the
other's storage (use-after-free hazard). A correct fix needs a new device-side
buffer-copy primitive in `mlrs-backend`, which is a backend feature addition out of
scope for an Info-tier review fix and untestable without new backend coverage. The
review itself rates this "Minor — labels are small."
**Original issue:** `fit_predict` calls `self.labels(pool)` (device→host) then
`DeviceArray::from_host(pool, &labels)` (host→device) although `labels_` is already
device-resident.

### IN-02: host_to_f64 / f64_to_host bit-cast helpers duplicated across modules

**File:** `crates/mlrs-algos/src/cluster/kmeans.rs:429`; `linear/coordinate_descent.rs:232`; `linear/lasso.rs:179`; `linear/logistic.rs:485`; `neighbors/classifier.rs:267`; `neighbors/regressor.rs:182`
**Reason:** skipped — cross-cutting refactor with blast radius far beyond the cited
scope and the per-finding atomic-commit model. Hoisting a shared helper into
`mlrs_core` requires adding a `bytemuck` dependency to a deliberately backend-free
core crate, and the duplicated pair actually lives in ~25 call sites across
`mlrs-core`, `mlrs-backend`, and `mlrs-algos` — including many Phase 2-4 modules
(`linear_regression`, `ridge`, `pca`, `truncated_svd`, `svd`, `eig`, `cholesky`,
`reduce`) that are NOT in this review's scope. Editing those out-of-scope modules to
deduplicate an Info-tier style issue risks regressing unrelated code without
full-workspace test coverage. Recommended as a dedicated refactor task.
**Original issue:** The identical `match size_of::<F>()` cast pair is copy-pasted
into at least six estimator modules with drifting `unreachable!` messages.

---

_Fixed: 2026-06-13T16:40:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
