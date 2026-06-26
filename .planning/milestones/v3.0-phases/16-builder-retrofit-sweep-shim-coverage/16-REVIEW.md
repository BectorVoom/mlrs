---
phase: 16-builder-retrofit-sweep-shim-coverage
reviewed: 2026-06-25T00:00:00Z
depth: standard
files_reviewed: 91
files_reviewed_list:
  - crates/mlrs-algos/src/cluster/dbscan.rs
  - crates/mlrs-algos/src/cluster/kmeans.rs
  - crates/mlrs-algos/src/cluster/spectral_clustering.rs
  - crates/mlrs-algos/src/covariance/ledoit_wolf.rs
  - crates/mlrs-algos/src/decomposition/incremental_pca.rs
  - crates/mlrs-algos/src/density/kernel_density.rs
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs
  - crates/mlrs-algos/src/lib.rs
  - crates/mlrs-algos/src/linear/elastic_net.rs
  - crates/mlrs-algos/src/linear/lasso.rs
  - crates/mlrs-algos/src/linear/linear_regression.rs
  - crates/mlrs-algos/src/linear/logistic.rs
  - crates/mlrs-algos/src/linear/mbsgd_classifier.rs
  - crates/mlrs-algos/src/linear/mbsgd_regressor.rs
  - crates/mlrs-algos/src/linear/ridge.rs
  - crates/mlrs-algos/src/naive_bayes/categorical_nb.rs
  - crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs
  - crates/mlrs-algos/src/neighbors/classifier.rs
  - crates/mlrs-algos/src/neighbors/nearest.rs
  - crates/mlrs-algos/src/neighbors/regressor.rs
  - crates/mlrs-algos/src/projection/gaussian.rs
  - crates/mlrs-algos/src/projection/sparse.rs
  - crates/mlrs-algos/src/typestate.rs
  - crates/mlrs-py/python/mlrs/__init__.py
  - crates/mlrs-py/python/mlrs/cluster.py
  - crates/mlrs-py/python/mlrs/linear.py
  - crates/mlrs-py/python/mlrs/naive_bayes.py
  - crates/mlrs-py/python/mlrs/neighbors.py
  - crates/mlrs-py/python/tests/test_estimator_checks.py
  - crates/mlrs-py/python/tests/test_params.py
  - crates/mlrs-py/python/tests/test_shims.py
  - crates/mlrs-py/src/estimators/cluster.rs
  - crates/mlrs-py/src/estimators/linear.rs
  - crates/mlrs-py/src/estimators/neighbors.rs
findings:
  critical: 0
  warning: 6
  info: 4
  total: 10
status: issues_found
---

# Phase 16: Code Review Report

**Reviewed:** 2026-06-25
**Depth:** standard
**Files Reviewed:** 91 (representative sample of the 91-file scope read in full / cross-referenced)
**Status:** issues_found

## Summary

Phase 16 retrofits every estimator onto the single `typestate` builder surface and
adds 15 new Python shims. The Rust estimator bodies are high quality: validate-before-launch
guards are consistent, the consuming-`self` typestate transition is applied uniformly,
device-buffer release / double-release hazards (Ridge/KernelRidge `out`-threading,
SpectralClustering inner-KMeans release, LogReg `ScratchGuard`) are carefully handled,
and hyperparameter validation is correctly split between `build()` (data-independent)
and `fit()` (data-dependent). No security vulnerabilities, no memory-safety defects,
and no crash/data-loss BLOCKERs were found.

The findings are correctness/consistency traps that emerge from the breadth of the
retrofit: a **Rust↔Python `classes_` divergence** for two classifiers (the cores
support non-contiguous labels, the shims fabricate `np.arange`), a **missing label
validation** in `KNeighborsClassifier::fit` that every sibling classifier performs,
an **inconsistent pool-lock path** in the neighbors PyO3 wrappers (panics on poison
where siblings recover — pre-existing/tracked), and a few **missing `n_samples == 0`
geometry guards** in the random-projection / IncrementalPCA transform paths that other
estimators reject. All are fixable without touching the device math.

## Warnings

### WR-01: `LogisticRegression` / `KNeighborsClassifier` Python `classes_` discards the core's real (non-contiguous) labels

**File:** `crates/mlrs-py/python/mlrs/linear.py:218`, `crates/mlrs-py/python/mlrs/neighbors.py:65`
**Issue:** Both shims set `self.classes_ = np.arange(obj.n_classes(), dtype=np.int32)`
— a contiguous `0..n_classes`. But the Rust cores deliberately store the DISTINCT
sorted training labels (`logistic.rs` CR-02 `classes_`, `classifier.rs` CR-03 `classes_`)
and `predict_labels` maps the argmax column back through them, so for a non-contiguous
target (e.g. `{0, 2}`) the Rust `predict` returns the original `2`. The Python
`classes_` then reports `[0, 1]`, so `predict` can return a value NOT present in
`classes_` — a violation of sklearn's `classes_`/predict consistency contract.
Note the asymmetry: `MBSGDClassifier` / `LinearSVC` expose `classes_()` on their PyO3
wrappers and their shims use `obj.classes_()`; `PyLogisticRegression` and
`PyKNeighborsClassifier` expose only `n_classes()`, forcing the shim to fabricate the
range. (The v1 contract restricts users to contiguous labels — `check_classifiers_classes`
is xfailed in `test_estimator_checks.py` — so this is latent rather than test-visible,
but the cores' own CR-02/CR-03 effort is silently thrown away at the Python boundary.)
**Fix:** Expose the real labels on both PyO3 wrappers and have the shims use them, exactly
like the SGD/SVM classifiers do:
```rust
// in PyLogisticRegression / PyKNeighborsClassifier #[pymethods]
fn classes_(&self) -> PyResult<Vec<i64>> {
    match &self.inner {
        Any...::F32(e) => Ok(e.classes().to_vec()),
        Any...::F64(e) => Ok(e.classes().to_vec()),
        _ => Err(not_fitted("...", "classes_")),
    }
}
```
```python
# linear.py LogisticRegression.fit / neighbors.py KNeighborsClassifier.fit
self.classes_ = np.asarray(obj.classes_(), dtype=np.int32)
```
(LogisticRegression core already has a `classes()`-style field; KNeighborsClassifier
stores `classes_: Vec<i32>` but has no public accessor — add one.)

### WR-02: `KNeighborsClassifier::fit` skips the integer/i32-range label validation every sibling classifier performs

**File:** `crates/mlrs-algos/src/neighbors/classifier.rs:253-257`
**Issue:** Labels are read as `host_to_f64(v).round() as i32` with NO check that the
value is a finite, non-negative integer in `i32` range. Every other classifier in the
crate guards this: `logistic.rs:362` rejects non-integers, `mbsgd_classifier.rs:341-376`
rejects non-integers AND out-of-`i32` labels, and the shared `decode_classes`
(`multinomial_nb.rs:484-504`, used by all five NB variants) does both. A NaN target
silently becomes `0` (`f64::NAN.round() as i32 == 0` in Rust's saturating cast), and a
label beyond `i32::MAX` saturates — producing a spurious/wrong class with no error. This
is exactly the WR-02 "silent wrong label" class the codebase explicitly guards elsewhere.
**Fix:** Mirror the `decode_classes` validation before building `classes_`:
```rust
let mut raw_class: Vec<i32> = Vec::with_capacity(n_train);
for &v in y_host.iter() {
    let lf = host_to_f64(v);
    let lr = lf.round();
    if !lr.is_finite() || (lr - lf).abs() > 1e-6 || i32::try_from(lr as i64).is_err() {
        return Err(AlgoError::InvalidLabels {
            estimator: "knn_classifier",
            reason: format!("labels must be i32-range integers (got {lf})"),
        });
    }
    raw_class.push(lr as i32);
}
```

### WR-03: Neighbors PyO3 wrappers use the panic-on-poison lock path instead of the sanctioned `lock_pool`

**File:** `crates/mlrs-py/src/estimators/neighbors.rs:77,111,126,206,241,260,273,360,394,407` (every `fit`/`predict`/`kneighbors` body)
**Issue:** These wrappers call `crate::global_pool().lock().expect("pool mutex")`, which
PANICS if the mutex is poisoned (a prior device fault / OOM / unsupported-op panic inside
another `py.detach`). The sanctioned path `crate::lock_pool()` (documented in `lib.rs:96-157`
as the single authoritative lock path) recovers the poisoned guard and re-baselines the
pool accounting, so one recoverable device error does not permanently brick the interpreter.
`cluster.rs` / `linear.rs` already use `lock_pool()`. `lib.rs:115-118` explicitly flags the
`neighbors` wrappers as carrying "the legacy panicking form — a pre-existing, tracked
migration," so this is acknowledged tech debt rather than newly introduced — but it leaves
the three neighbor estimators outside the brick-prevention guarantee.
**Fix:** Replace every `crate::global_pool().lock().expect("pool mutex")` in this file with
`crate::lock_pool()` (the rest of the body is unchanged).

### WR-04: Random-projection `project` and `IncrementalPCA` transform/inverse_transform omit the `n_samples == 0` geometry guard

**File:** `crates/mlrs-algos/src/projection/gaussian.rs:374`, `crates/mlrs-algos/src/decomposition/incremental_pca.rs:527,589`
**Issue:** The geometry guards here check `n_features != fitted_n_features || x.len() != n_samples * n_features`
but NOT `n_samples == 0`. With `n_samples == 0`, `x.len() == 0 == 0 * n_features` passes
and a zero-row GEMM is launched. Every other transform/predict path in the crate explicitly
rejects `n_samples == 0` first (e.g. `linear_regression.rs:412`, `ridge.rs:436`,
`kernel_density.rs:393`, `pca`/`logistic`). The inconsistency means a degenerate empty
query reaches the device on these two paths instead of returning a typed `ShapeMismatch`.
**Fix:** Add the zero-row guard to match the rest of the crate:
```rust
// projection/gaussian.rs::project
if n_samples == 0 || n_features != fitted_n_features || x.len() != n_samples * n_features {
// incremental_pca.rs::transform / inverse_transform
if n_samples == 0 || n_features != self.n_features || x.len() != n_samples * n_features {
```

### WR-05: `KernelRidge::fit` builds `n_targets` from `y.len() / n_samples` without checking divisibility

**File:** `crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs:355-359,389`
**Issue:** `n_targets = y.len() / n_samples` (integer division), then the guard is
`if n_targets == 0 || y.len() != n_samples * n_targets`. A `y` whose length is NOT a
multiple of `n_samples` (e.g. `n_samples = 4`, `y.len() = 10` → `n_targets = 2`,
`n_samples * n_targets = 8 != 10`) IS correctly rejected by the second clause — so this
is not exploitable. However, the recovered `n_targets` is silently truncating; the intent
("y.len() must be a positive multiple of n_samples") is enforced only as a side effect of
the post-hoc equality. This is fragile: any future refactor that reorders or relaxes the
`y.len() != n_samples * n_targets` clause would let a non-multiple `y` through with a
truncated target count. The reliance on the equality re-check should be made explicit.
**Fix:** Make the divisibility intent explicit rather than implicit:
```rust
if n_samples == 0 || y.len() == 0 || y.len() % n_samples != 0 {
    return Err(AlgoError::Prim(PrimError::ShapeMismatch {
        operand: "y", rows: n_samples, cols: 0, len: y.len(),
    }));
}
let n_targets = y.len() / n_samples;
```

### WR-06: `KMeans.fit` shim passes a raw numpy `random_state` to the `Option<u64>` ctor without int-coercion

**File:** `crates/mlrs-py/python/mlrs/cluster.py:41-43`
**Issue:** `self._ext().KMeans(self.n_clusters, self.max_iter, self.tol, self.random_state)`
forwards `random_state` unchanged. `PyKMeans::new` accepts `Option<u64>`, so `None` works
and a Python `int` works, but a numpy integer scalar (e.g. `np.int64(7)`, common when users
pass `random_state` derived from numpy) is not guaranteed to coerce to PyO3's `u64`
extractor, and a negative `random_state` (sklearn permits any int / `RandomState`) would
fail `u64` extraction with an opaque `OverflowError`. The sibling `SpectralClustering.fit`
shim (cluster.py:130) correctly normalizes first: `seed = 0 if self.random_state is None
else int(self.random_state)`. KMeans should do the same for consistency and to avoid a
confusing extraction error.
**Fix:** Normalize before the boundary, mirroring SpectralClustering:
```python
seed = None if self.random_state is None else int(self.random_state)
obj = self._ext().KMeans(self.n_clusters, self.max_iter, self.tol, seed)
```

## Info

### IN-01: `LogisticRegression::fit` uses `PrimError::ShapeMismatch` to report a label-VALIDITY failure

**File:** `crates/mlrs-algos/src/linear/logistic.rs:362-369,379-386`
**Issue:** When labels are non-integer/negative or there are `< 2` classes, `fit` returns
`PrimError::ShapeMismatch { operand: "logistic.y (labels must be integers ...)", ... }`.
The shape IS valid; the CONTENT is invalid. `AlgoError::InvalidLabels` (used by
`mbsgd_classifier.rs` and the NB family for the identical condition) is the honest variant
and its own doc (`error.rs:323-339`) calls out this exact "fabricated row/col/len shape
error" anti-pattern. Functionally it still errors, but the diagnostic is misleading.
**Fix:** Return `AlgoError::InvalidLabels { estimator: "logistic_regression", reason: ... }`
for both the non-integer and `< 2 classes` cases, matching the sibling classifiers.

### IN-02: `KNeighborsClassifier`/`Regressor` builder mis-labels the `n_neighbors == 0` rejection as `InvalidNComponents`

**File:** `crates/mlrs-algos/src/neighbors/classifier.rs:193-198`, `crates/mlrs-algos/src/neighbors/regressor.rs:162-167`, `crates/mlrs-algos/src/neighbors/nearest.rs:189-194`
**Issue:** A zero neighbor count is reported via `BuildError::InvalidNComponents { param:
"n_neighbors", ... }`. `n_neighbors` is not an `n_components`; `AlgoError::InvalidK` is the
semantically correct family (and is what the data-dependent `k > n_train` half uses in the
`kneighbors` core). Reusing `InvalidNComponents` produces a user-facing message
("n_components = 0 is out of range") that names the wrong hyperparameter.
**Fix:** Add a `BuildError::InvalidK`/`InvalidNNeighbors` variant (or reuse a neighbor-named
one) so the construction-time error names `n_neighbors`.

### IN-03: `LedoitWolf` memoizes host attrs in `OnceLock` but ignores the `pool` after the first call

**File:** `crates/mlrs-algos/src/covariance/ledoit_wolf.rs:185-207`
**Issue:** `covariance_(&self, pool)` / `location_(&self, pool)` memoize via
`get_or_init(|| ... to_host(pool))`. After the first call the cached `Vec<F>` is returned
and the passed `pool` is ignored. This is correct (the device buffer is immutable post-fit),
but the API invites a caller to assume a fresh download per call. A doc note already exists;
consider taking `pool` by value-less signature or documenting more prominently that the
argument is consulted only on first access. Cosmetic.
**Fix:** No behavior change required; tighten the doc or drop the unused-after-first `pool`
dependency.

### IN-04: Several `expect("shared path is never plane-gated to None")` unwraps couple estimators to a reduce-prim invariant

**File:** `crates/mlrs-algos/src/linear/linear_regression.rs:293`, `ridge.rs:315`, `covariance/ledoit_wolf.rs:257`, `density/kernel_density.rs:448`
**Issue:** `column_reduce(.., ReducePath::Shared, ..)?.expect("shared path is never plane-gated to None")`
panics (not a typed error) if the reduce prim ever returns `None` on the Shared path. The
comment asserts the invariant holds today, but it is a cross-crate coupling: a future change
to the reduce prim's plane-gating would turn these into process panics across the (PyO3)
boundary rather than a propagated `AlgoError`. Low risk given the documented invariant.
**Fix:** Prefer mapping `None` to a typed error
(`.ok_or(AlgoError::Prim(PrimError::...))?`) so the boundary stays panic-free even if the
prim contract drifts.

---

_Reviewed: 2026-06-25_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
