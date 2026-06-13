---
phase: 06-python-surface-pyo3-estimators-per-backend-wheels
reviewed: 2026-06-14T00:00:00Z
depth: standard
files_reviewed: 23
files_reviewed_list:
  - crates/mlrs-py/src/lib.rs
  - crates/mlrs-py/src/ingress.rs
  - crates/mlrs-py/src/egress.rs
  - crates/mlrs-py/src/capability.rs
  - crates/mlrs-py/src/dispatch.rs
  - crates/mlrs-py/src/errors.rs
  - crates/mlrs-py/src/arrow_symbol_probe.rs
  - crates/mlrs-py/src/estimators/mod.rs
  - crates/mlrs-py/src/estimators/linear.rs
  - crates/mlrs-py/src/estimators/cluster.rs
  - crates/mlrs-py/src/estimators/decomposition.rs
  - crates/mlrs-py/src/estimators/neighbors.rs
  - crates/mlrs-py/python/mlrs/__init__.py
  - crates/mlrs-py/python/mlrs/base.py
  - crates/mlrs-py/python/mlrs/_io.py
  - crates/mlrs-py/python/mlrs/linear.py
  - crates/mlrs-py/python/mlrs/cluster.py
  - crates/mlrs-py/python/mlrs/decomposition.py
  - crates/mlrs-py/python/mlrs/neighbors.py
  - crates/mlrs-py/python/tests/conftest.py
  - crates/mlrs-py/python/tests/_wheel_build.py
  - crates/mlrs-py/python/tests/test_estimator_checks.py
  - crates/mlrs-py/python/tests/test_import_probe.py
findings:
  critical: 1
  warning: 8
  info: 6
  total: 15
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-06-14T00:00:00Z
**Depth:** standard
**Files Reviewed:** 23
**Status:** issues_found

## Summary

Reviewed the PyO3 binding layer (`mlrs-py` cdylib) and the pure-Python sklearn
shim for the per-backend Python surface. The FFI ingress contract (owned Arrow
capsule import, offset hard-reject, single metered upload) is sound, the
`catch_unwind` import probe is correct, and the f64-on-incapable-backend guard is
placed before every f64 upload. The error-class mapping is consistent.

However the review surfaced one **BLOCKER**: the `predict_proba` / `transform` /
`kneighbors` egress silently flattens 2-D results to 1-D on the `pyarrow`
output-type path, corrupting matrix-shaped results. Several WARNINGs concern
correctness gaps that the test triage acknowledges but does not actually
mitigate (NaN-in-`y` bypassing finiteness validation; the LogisticRegression
xfail rationale being factually wrong), an unvalidated/ignored `init`
hyperparameter on KMeans, a fitted-but-wrong-dtype error masquerading as
`NotFitted`, and a conservative-`True` f64 fallback that can route f64 data into
the device on an incapable backend in a narrow path. The remaining items are
maintainability/consistency defects.

No structural findings block was provided.

## Critical Issues

### CR-01: `pyarrow` output-type silently flattens 2-D results to 1-D

**File:** `crates/mlrs-py/python/mlrs/_io.py:134-147`
**Issue:** `to_output` ignores `shape` entirely on the `pyarrow` branch and
returns `pa.array(flat.ravel(order="C"))` — a flat 1-D array — for every result,
including the genuinely 2-D ones (`predict_proba` → `(rows, n_classes)`,
`PCA/TruncatedSVD.transform` → `(rows, n_components)`, `kneighbors` distances /
indices → `(rows, k)`, `cluster_centers_` → `(n_clusters, n_features)`,
`components_` → `(n_components, n_features)`). When `output_type="input"` and the
caller passed a pyarrow array (which `resolve_output_type` maps to `"pyarrow"`),
the returned matrix loses its geometry: a `rows × n_classes` probability matrix
comes back as a length `rows*n_classes` 1-D arrow array with no way to recover the
shape. This is silent data-shape corruption, not just a formatting nuisance — a
downstream consumer reshaping by `rows` (the only dimension it knows) gets
transposed/garbage values. The egress contract (`egress.rs` `FloatResult` carries
`(rows, cols)` precisely so geometry is never re-derived) is discarded here.
**Fix:** Either raise for 2-D pyarrow egress in v1, or carry the shape. Minimal
correctness fix — refuse to flatten a matrix:
```python
def to_output(buf, shape, output_type, dtype):
    np_dtype = np.dtype(dtype)
    flat = np.asarray(buf, dtype=np_dtype)
    arr = flat.reshape(shape)
    if output_type == _PYARROW:
        if arr.ndim > 1 and arr.shape[1] != 1:
            # pyarrow Array is 1-D only; do not silently flatten a matrix.
            raise ValueError(
                "mlrs: pyarrow output_type is unsupported for 2-D results "
                f"(shape {arr.shape}); request output_type='numpy'."
            )
        return pa.array(arr.ravel(order="C"))
    return arr
```

## Warnings

### WR-01: NaN in `y` bypasses finiteness validation and reaches the device

**File:** `crates/mlrs-py/python/mlrs/_io.py:106-115`
**Issue:** `normalize_y` does NOT run `check_array` / any finiteness check — it
goes straight to `np.ascontiguousarray(...).ravel()` → `pa.array(...)`. The Rust
bridge (`bridge.rs:112-114`) rejects only Arrow *nulls* (`null_count`), and a
NaN/Inf float is a valid bit pattern, NOT an Arrow null (confirmed:
`mlrs-core/src/error.rs` only has `HasNulls`, no NaN check). So a supervised
`fit(X, y)` with `NaN`/`Inf` in `y` uploads poisoned targets to the device
silently — violating the project's correctness-first invariant. `X` is protected
(`normalize_X` runs `check_array(ensure_all_finite=True)`); `y` is not.
**Fix:** Validate `y` finiteness symmetrically, e.g. via
`sklearn.utils.check_array(y, ensure_2d=False, ensure_all_finite=True, dtype=dtype)`
(or `check_X_y`) before building the pyarrow array.

### WR-02: LogisticRegression xfail rationale is factually wrong — y NaN check is not performed

**File:** `crates/mlrs-py/python/tests/test_estimator_checks.py:144-158`
**Issue:** The triage claims LogisticRegression's `y` "goes through `check_array`,
so `check_supervised_y_no_nan` PASSES" and therefore removes it from LogReg's
xfail map. But `LogisticRegression.fit` (`linear.py:208-219`) calls the SAME
`self._normalize_y(...)` as every other supervised estimator, which (per WR-01)
performs no finiteness check. The stated reason is false; if the check truly
relies on NaN-y rejection it will fail (not xpass) for LogReg, breaking the suite.
This is a test-correctness defect that masks the WR-01 gap behind an incorrect
justification.
**Fix:** Fix WR-01 (validate `y`) so the claim becomes true, OR keep
`check_supervised_y_no_nan` in LogReg's xfail map with an accurate reason.

### WR-03: f64 capability fallback returns `True`, can route f64 into an incapable backend

**File:** `crates/mlrs-py/python/mlrs/_io.py:36-49`
**Issue:** `_backend_supports_f64` returns `True` (f64-capable) on ANY exception
importing `_mlrs`. `pick_dtype` (line 65) then defaults non-float input to
`float64`. The Rust `guard_f64` is the real safety net for the *fit* path — but
only the supervised/unsupervised `fit` arms call `guard_f64`. A user who calls a
shim method that uploads f64 without hitting the guarded fit arm, on a backend
whose `_mlrs` import partially succeeded but whose capability query throws, would
get f64 data with no guard. More practically: defaulting to `True` on failure is
the unsafe direction for a "correctness trumps convenience" project — a transient
capability-query error silently picks the dtype most likely to be rejected or,
worse, downcast. Fail safe is `float32`.
**Fix:** Default to the conservative capable-set on error:
```python
    except Exception:
        return False  # unknown capability -> assume f64-incapable (safe default)
```
(and let an explicit float64 input still hit the Rust `guard_f64` for the hard
error). At minimum, document why `True` is intentional if it must stay.

### WR-04: KMeans `init` hyperparameter is accepted, stored, then silently ignored

**File:** `crates/mlrs-py/python/mlrs/cluster.py:23-47`
**Issue:** `KMeans.__init__` accepts `init="k-means++"` and stores it, but `fit`
never passes it to `_mlrs.KMeans(...)` and never validates it. A caller passing
`init="random"` (a valid sklearn value) gets k-means++ behavior with no error or
warning — a silent contract violation. The docstring says k-means++ is "the only
supported value in v1", but nothing enforces it.
**Fix:** Validate in `fit` (or `__init__`-adjacent, keeping `__init__` pure per
the project rule — validate in `fit`):
```python
def fit(self, X, y=None):
    if self.init != "k-means++":
        raise ValueError(
            f"mlrs KMeans supports init='k-means++' only (got {self.init!r})."
        )
    ...
```

### WR-05: Fitted-but-wrong-dtype predict reports `NotFitted` instead of a dtype error

**File:** `crates/mlrs-py/src/estimators/linear.rs:106-134` (and the analogous
`_ => not_fitted(...)` arms in `cluster.rs`, `decomposition.rs`, `neighbors.rs`)
**Issue:** `predict_f32` matches only `AnyLinearRegression::F32` and returns
`not_fitted(...)` for BOTH the `Unfit` arm and the `F64` (fitted-but-other-dtype)
arm. A fitted f64 estimator called via the f32 accessor reports "is not fitted
yet" — a misleading error that, surfaced through the shim, becomes a spurious
`NotFittedError` for a fully-fitted model. The shim's `_suffixed` currently always
picks the matching arm so the live path is safe, but any direct `_mlrs` consumer
(and the Rust smoke tests) get a wrong diagnosis. This is a latent
mis-classification of a real state.
**Fix:** Distinguish the two states, e.g. return a clear dtype-mismatch error for
the wrong fitted arm:
```rust
AnyLinearRegression::F64(_) => Err(PyTypeError::new_err(
    "estimator was fitted as f64; call the f64 accessor")),
AnyLinearRegression::Unfit { .. } => Err(not_fitted("linear_regression", "predict")),
```

### WR-06: `_io.to_output` reshape ignores ravel order for row-major device buffers

**File:** `crates/mlrs-py/python/mlrs/_io.py:143-147`
**Issue:** `flat.reshape(shape)` uses numpy's default C order, which is correct
*iff* the device buffer is row-major. The egress doc (`egress.rs`) asserts
row-major, so this is currently consistent — but `reshape` does not assert/verify
`flat.size == prod(shape)` for the non-`-1` dims, and a shape mismatch (e.g. a
backend returning a transposed `components_`) would raise an opaque numpy error
rather than a domain message. Combined with the `-1` inference used by callers
(`cluster_centers_`, `components_`), a wrong total length silently produces a
mis-shaped matrix when one axis is `-1`. Low likelihood but unguarded.
**Fix:** Validate the buffer length against the known fixed dims before reshape,
or assert `flat.size == rows * cols` for the fully-known shapes.

### WR-07: Linear/Ridge/Lasso/ElasticNet `intercept_` bypasses output-type mirroring

**File:** `crates/mlrs-py/python/mlrs/linear.py:47-50, 85-88, 130-133, 181-184`
**Issue:** The regressor `intercept_` properties return the raw Rust scalar
(`getattr(self._mlrs_obj, "intercept" + self._suffix())()`) without going through
`_to_output`, while `coef_` and LogisticRegression's `intercept_` DO route through
`_to_output`. For a scalar this is mostly harmless, but it is an inconsistent
egress contract: under `output_type="pyarrow"` the `coef_` is a pyarrow array yet
`intercept_` is a bare Python float, and the value's dtype (f32 vs f64) is the
Rust-returned type rather than the mirrored `_np_float()`. sklearn returns
`intercept_` as a float/ndarray scalar matching `coef_`'s dtype.
**Fix:** Route through the egress helper for dtype/type consistency, e.g. wrap the
scalar as a 0-d / length-1 result via `_to_output`, or document the scalar
exception explicitly.

### WR-08: `kneighbors` ignores its own clamp/validation of `n_neighbors` against `n_samples_fit`

**File:** `crates/mlrs-py/python/mlrs/neighbors.py:32-47`
**Issue:** `kneighbors` takes `n_neighbors` (or falls back to `self.n_neighbors`)
and forwards `k` straight to the Rust `kneighbors_f32/f64` with no check that
`k <= n_samples` seen at fit. sklearn raises a clear `ValueError` ("Expected
n_neighbors <= n_samples_fit"). Here, an out-of-range `k` depends entirely on the
algos layer to error; if the algos layer under-checks, this returns garbage
indices. No `n_samples_fit_` is recorded at fit to validate against.
**Fix:** Record `self.n_samples_fit_ = rows` in `fit` and validate
`k <= self.n_samples_fit_` in `kneighbors`, raising sklearn's message form.

## Info

### IN-01: `egress.rs` helpers (`vec_f_to_py`, `labels_to_py`, `vec_i32_to_py`, `FloatResult`, `LabelResult`) appear unused

**File:** `crates/mlrs-py/src/egress.rs:39-68`
**Issue:** The estimator wrappers call `out.to_host_metered(&mut pool)` directly
and return bare `Vec<F>` / `Vec<i32>` (no `(values, shape)` pairing). None of the
`egress.rs` helpers or the `FloatResult`/`LabelResult` shape-carrying types are
referenced by the reviewed estimator files. If intended for Plan 03+ consumption
they are dead for this phase; otherwise the shape-pairing contract they document
is not actually enforced anywhere (shape lives only shim-side).
**Fix:** Either wire the wrappers through these helpers (so the metered-read +
shape contract is centralized) or mark/remove the unused surface.

### IN-02: `arrow_symbol_probe.rs` is a non-compiled doc file shipped as `.rs`

**File:** `crates/mlrs-py/src/arrow_symbol_probe.rs:1-43`
**Issue:** The file is documentation-only and intentionally excluded from `lib.rs`
`mod` list. Carrying prose as a `.rs` under `src/` invites confusion (it looks
compiled) and could trip tooling that globs `src/**/*.rs`.
**Fix:** Move to `docs/` or a `//!`-only module that is `mod`-declared, or rename
to `.md`.

### IN-03: `not_fitted(estimator, operation)` — `operation` only formats, `estimator` naming is stringly-typed and duplicated

**File:** `crates/mlrs-py/src/errors.rs:68-72` (call sites throughout `estimators/`)
**Issue:** Every accessor hardcodes the estimator name as a string literal
(`"linear_regression"`, `"ridge"`, ...) at dozens of call sites. A rename drifts
silently; a typo produces a wrong-named error. Low risk, high duplication.
**Fix:** Derive the name from a const/associated string per `#[pyclass]`, or a
macro, so it is defined once.

### IN-04: `_post_fit` vs direct `n_features_in_` assignment is inconsistent across families

**File:** `crates/mlrs-py/python/mlrs/neighbors.py:29, 63, 96`
**Issue:** Linear/cluster/decomposition shims call `self._post_fit(cols)`; the
neighbors shims set `self.n_features_in_ = cols` directly (skipping the
`int(...)` normalization and the documented single entry point). Functionally
equivalent today but a maintenance trap if `_post_fit` later does more.
**Fix:** Use `self._post_fit(cols)` uniformly in the neighbors shims.

### IN-05: `proba_allclose` re-normalization can mask an un-normalized-output bug

**File:** `crates/mlrs-py/python/tests/conftest.py:134-150`
**Issue:** The helper re-normalizes each row before the `allclose`, which by
design hides a backend that returns un-normalized probabilities (the very defect
a proba oracle should catch). The docstring frames this as "guarding against" such
a backend, but re-normalizing means the test would PASS on softmax outputs that
don't sum to 1. For a gauge-invariant gate this weakens the oracle.
**Fix:** Assert row-sums ≈ 1 separately (un-normalized), then compare; do not
silently normalize away the defect.

### IN-06: `pick_dtype` does `np.asarray(X)` a second time (after `check_array` will re-copy)

**File:** `crates/mlrs-py/python/mlrs/_io.py:60, 80-103`
**Issue:** `normalize_X` calls `pick_dtype(X)` (which does `np.asarray(X)`) and
then `check_array(...)` re-materializes `X` again, then `np.ascontiguousarray`
again — up to three passes over the input. Out of v1 perf scope, but for large
inputs this is extra host copies in a "memory-efficiency is first-class" project.
**Fix:** Resolve dtype from the already-validated `check_array` output where
possible, or thread the asarray result through.

---

_Reviewed: 2026-06-14T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
