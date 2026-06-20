---
phase: 07-covariance-projection
plan: 07
subsystem: py-bindings
tags: [pyo3, pyclass, partial-fit, covariance, projection, incremental-pca, johnson-lindenstrauss, sklearn-shim, dtype-dispatch, guard-f64]

# Dependency graph
requires:
  - phase: 07-covariance-projection
    plan: 04
    provides: "EmpiricalCovariance<F> + LedoitWolf<F> (COV-01/COV-02) with covariance_/location_/precision_/shrinkage_ accessors"
  - phase: 07-covariance-projection
    plan: 05
    provides: "IncrementalPCA<F> (DECOMP-03) — PartialFit<F> + Fit<F> + Transform<F> + inverse_transform"
  - phase: 07-covariance-projection
    plan: 06
    provides: "GaussianRandomProjection<F> + SparseRandomProjection<F> (PROJ-01/02) + johnson_lindenstrauss_min_dim + NComponents selector"
  - phase: 06-python-packaging
    provides: "any_estimator! dtype-dispatch macro, ingress/egress/capability/errors binding primitives, PyPCA canonical wrapper, MlrsBase shim, smoke-test scaffold"
provides:
  - "_mlrs.{EmpiricalCovariance,LedoitWolf,IncrementalPCA,GaussianRandomProjection,SparseRandomProjection} #[pyclass] + johnson_lindenstrauss_min_dim #[pyfunction]"
  - "IncrementalPCA.partial_fit — the first v2 partial_fit method (py.detach + dtype-dispatch + guard_f64, constructs the fitted arm on the first batch, mutates in place after)"
  - "mlrs.{covariance,random_projection}.* + mlrs.decomposition.IncrementalPCA sklearn-compatible shims; SparseRandomProjection densifies sparse input at ingress (PROJ-02)"
affects: [11-final-signoff]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "v2 adds ZERO binding infrastructure — the five wrappers reuse the SHIPPED any_estimator! macro + ingress/egress/capability/errors verbatim (RESEARCH); the only new error helper is errors::dtype_mismatch_in_stream for a mixed-dtype partial_fit stream"
    - "partial_fit ownership: the #[pyclass] constructs the F32/F64 arm from the stored Unfit hyperparameters on the FIRST batch and mutates the existing arm in place on subsequent batches; the Python shim constructs the _mlrs object once and reuses it across the stream"
    - "n_components='auto' and density='auto' map to Option<usize>/Option<f64> None sentinels at the Rust boundary (None -> NComponents::Auto / sklearn 1/sqrt(n_features)); random_state -> u64 seed"
    - "scalar accessors are single-typed (no dtype suffix): shrinkage_ (f64), n_components_/n_samples_seen (usize), density_ (f64) — only the array attrs are dtype-suffixed _f32/_f64"

key-files:
  created:
    - crates/mlrs-py/src/estimators/covariance.rs
    - crates/mlrs-py/src/estimators/projection.rs
    - crates/mlrs-py/python/mlrs/covariance.py
    - crates/mlrs-py/python/mlrs/random_projection.py
  modified:
    - crates/mlrs-py/src/estimators/decomposition.rs
    - crates/mlrs-py/src/estimators/mod.rs
    - crates/mlrs-py/src/lib.rs
    - crates/mlrs-py/src/errors.rs
    - crates/mlrs-algos/src/decomposition/incremental_pca.rs
    - crates/mlrs-py/python/mlrs/decomposition.py
    - crates/mlrs-py/python/mlrs/__init__.py
    - crates/mlrs-py/tests/pyclass_smoke_test.rs

key-decisions:
  - "RULE-3 FIX: added pub n_components()/whiten()/batch_size() accessors to the algos IncrementalPCA<F> so the PyIncrementalPCA re-fit path (fit on an already-fitted arm) can recover the hyperparameters from the fitted estimator — the struct fields are private and the wrapper needs them to rebuild the arm. A 3-line addition, no behavior change."
  - "RULE-2 ADD: errors::dtype_mismatch_in_stream raises a clear PyValueError when a partial_fit batch's float dtype disagrees with the dtype the first batch fixed — the fitted arm is a single monomorphization (F32 OR F64), so a mid-stream dtype switch cannot be merged. Without this guard the fall-through match arm would be a silent no-op."
  - "EmpiricalCovariance/LedoitWolf shims subclass MlrsBase DIRECTLY (no TransformerMixin) — covariance estimators are fit-only and expose fitted matrices/scalars, not a transform; the projection + IncrementalPCA shims keep TransformerMixin for fit_transform."
  - "guard_f64() appears 6× in decomposition.rs (>= the required 2): PyPCA fit, PyTruncatedSVD fit, and IncrementalPCA's fit + partial_fit-first-batch + partial_fit-subsequent-batch all gate their F64 arm — the partial_fit F64 path is guarded on BOTH the construct-first-batch arm AND the mutate-in-place arm (per WARNING 2, the static source-grep is the contract gate since the Unfit-arm smoke test never runs live F64 dispatch)."

requirements-completed: [COV-01, COV-02, DECOMP-03, PROJ-01, PROJ-02]

# Metrics
duration: 18min
completed: 2026-06-20
---

# Phase 7 Plan 07: PyO3 Wrappers for the Covariance / Projection Family Summary

**Wrapped the five Phase-7 estimators (EmpiricalCovariance, LedoitWolf, IncrementalPCA, GaussianRandomProjection, SparseRandomProjection) as PyO3 `#[pyclass]` objects on `_mlrs` plus `johnson_lindenstrauss_min_dim` as a `#[pyfunction]`, reusing the SHIPPED `any_estimator!` dtype-dispatch machinery (zero new binding infra), and added their four pure-Python sklearn-compatible shim modules — the user-facing completion of COV-01/COV-02/DECOMP-03/PROJ-01/PROJ-02. This introduces the first v2 `partial_fit` (IncrementalPCA), honoring the two load-bearing contracts in every device body: GIL release via `py.detach` (PY-03) and `guard_f64()?` BEFORE the F64 arm (D-04). Both `--features cpu` and `--features cpu,extension-module` compile, the rocm test target builds, pyo3 resolves to exactly one v0.28 ABI, and the smoke test constructs all five new wrappers in the Unfit arm (3/3 green).**

## Performance

- **Duration:** ~18 min
- **Completed:** 2026-06-20
- **Tasks:** 2 of 2
- **Files modified:** 13 (4 created, 9 modified)

## Accomplishments

### Task 1 — PyO3 wrappers (commits e324afb, ee01780, 2e27482, 1db99e7)
- **covariance.rs** (commit e324afb): `PyEmpiricalCovariance` + `PyLedoitWolf` via `any_estimator!` enums. EmpiricalCovariance Unfit arm carries `assume_centered`/`store_precision`; LedoitWolf carries `assume_centered`. `fit` runs the device `Fit::fit` inside `py.detach(|| global_pool().lock()…)` with f32/f64 dispatch and `crate::capability::guard_f64()?` BEFORE the F64 arm. Dtype-suffixed `covariance_f32`/`_f64`, `location_*`, `precision_*` returning the algos host accessors; `shrinkage_` is a single un-suffixed `f64` scalar (the algos estimator keeps it in f64 for both F arms).
- **projection.rs** (commit ee01780): `PyGaussianRandomProjection` + `PySparseRandomProjection` + the `johnson_lindenstrauss_min_dim` `#[pyfunction]`. `n_components: Option<usize>` maps `None → NComponents::Auto`, `Some(k) → Fixed(k)`; Sparse `density: Option<f64>` maps `None → sklearn 1/sqrt(n_features)`. `fit`/`transform_f32`/`_f64` mirror PyPCA (py.detach + guard_f64 + `to_host_metered`); `components_*` dtype-suffixed; `n_components_`/`density_` single scalars.
- **decomposition.rs** (commit 2e27482): extended (NOT a new file) with `PyIncrementalPCA` — `fit`, the new `partial_fit`, `transform`/`inverse_transform`, the full dtype-suffixed attr set (components_/explained_variance_/explained_variance_ratio_/singular_values_/mean_/var_), and `n_samples_seen` (single usize). `partial_fit` constructs the F32/F64 arm from the stored Unfit hyperparameters on the FIRST batch and MUTATES the existing arm in place on subsequent batches; both the construct-first-batch and mutate-in-place F64 arms call `guard_f64()` (3 IncrementalPCA guards total). Added `pub n_components()/whiten()/batch_size()` accessors to the algos `IncrementalPCA<F>` for the re-fit path (Rule 3) and `errors::dtype_mismatch_in_stream` for a mixed-dtype stream (Rule 2).
- **Registration** (commit 1db99e7): `pub mod covariance; pub mod projection;` in estimators/mod.rs; the five `add_class` + `add_function(johnson_lindenstrauss_min_dim)` in the `#[pymodule]`. pyo3 unchanged at 0.28.
- Acceptance greps: `add_class` ≥4 (=6), `fn partial_fit` =1, guard_f64 covariance.rs =3, projection.rs =3, decomposition.rs =6, pyo3 v0.28 =1. Both feature builds exit 0.

### Task 2 — Python shims + smoke test (commit 0af7e79)
- **covariance.py**: `EmpiricalCovariance` + `LedoitWolf` subclassing `MlrsBase` directly (fit-only, no TransformerMixin). Faithful `__init__` (store_precision/assume_centered; assume_centered) storing every arg verbatim; `fit(X, y=None)` → `_ext().EmpiricalCovariance(...)`/`.fit(xa, rows, cols)`; `@property` accessors mapping sklearn names to the suffixed `_mlrs` accessors via `_suffixed`/`_to_output` (covariance_/precision_ as `(-1, n_features_in_)` matrices, location_ as a vector, shrinkage_ as a `float` scalar).
- **random_projection.py**: `GaussianRandomProjection` + `SparseRandomProjection` (TransformerMixin + MlrsBase) with sklearn-named `__init__` (n_components='auto', eps, random_state→seed; Sparse adds density='auto'). `_densify(X)` calls `X.toarray()` on a `scipy.sparse.issparse(X)` input at the Python ingress (D-12 / PROJ-02) before `_normalize` — applied in both `fit` and `transform`. Module-level `johnson_lindenstrauss_min_dim(n_samples, eps)` delegates to the `_mlrs` pyfunction (scalar or array-like n_samples).
- **decomposition.py**: `IncrementalPCA` (TransformerMixin + MlrsBase) with `__init__` (n_components, whiten=False, batch_size=None), `fit`, `partial_fit` (constructs the `_mlrs` obj once, reuses it across the stream), `transform`/`inverse_transform`, and the attr `@property` accessors including `n_samples_seen_`.
- **__init__.py**: imports + re-exports the five new estimators + `johnson_lindenstrauss_min_dim` behind the existing guarded `_mlrs` import.
- **pyclass_smoke_test.rs**: added `five_phase7_estimators_construct_unfit` constructing each new wrapper via its Rust-callable `unfit_default()` and asserting `is_unfit()` — runs WITHOUT a Python interpreter or live device.
- Verification: `cargo test -p mlrs-py --features cpu --test pyclass_smoke_test` → 3/3 ok; all four shim modules parse as valid Python; class greps (2/2/1) + densify grep (12) all pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] algos IncrementalPCA hyperparameters were private with no accessor**
- **Found during:** Task 1 (IncrementalPCA wrapper `fit` re-fit path).
- **Issue:** `PyIncrementalPCA::fit` on an already-fitted arm needs to recover `n_components`/`whiten`/`batch_size` to rebuild the F32/F64 estimator, but the algos `IncrementalPCA<F>` keeps those as private fields with no getter — the wrapper could not compile the re-fit match arm.
- **Fix:** Added `pub fn n_components()/whiten()/batch_size()` accessors to `crates/mlrs-algos/src/decomposition/incremental_pca.rs` (3 trivial getters, no behavior change). The fit path now reads them off the fitted arm.
- **Files modified:** crates/mlrs-algos/src/decomposition/incremental_pca.rs
- **Commit:** 2e27482

**2. [Rule 2 - Missing critical functionality] mixed-dtype partial_fit stream had a silent fall-through**
- **Found during:** Task 1 (IncrementalPCA `partial_fit` subsequent-batch dispatch).
- **Issue:** The fitted arm is a single monomorphization (F32 OR F64). A `partial_fit` batch of the OTHER dtype cannot be merged; the `match (dt, &mut self.inner)` fall-through arm would otherwise be a silent no-op (the batch would be dropped, corrupting the running state without error).
- **Fix:** Added `errors::dtype_mismatch_in_stream(estimator)` raising a clear `PyValueError` ("keep every batch the same float dtype"); the fall-through arm returns it. This makes a mid-stream dtype switch a recognizable error instead of silent data loss.
- **Files modified:** crates/mlrs-py/src/errors.rs, crates/mlrs-py/src/estimators/decomposition.rs
- **Commit:** 2e27482

### Other notes
- The covariance shims subclass `MlrsBase` directly (no `TransformerMixin`) because EmpiricalCovariance/LedoitWolf are fit-only — they expose fitted matrices/scalars, not a `transform`. The projection + IncrementalPCA shims keep `TransformerMixin` for `fit_transform`.
- guard_f64 in decomposition.rs reads 6 (>= the required 2): the three pre-existing/added F64 fit arms (PyPCA, PyTruncatedSVD, IncrementalPCA fit) plus the IncrementalPCA partial_fit first-batch AND mutate-in-place F64 arms. The IncrementalPCA-specific count alone is 3 (fit + 2 partial_fit arms), exceeding the WARNING-2 "twice in decomposition.rs for the IncrementalPCA fit + partial_fit arms" requirement.

## Known Stubs

None. All five wrappers delegate to the fully-wired Phase-7 algos estimators (07-04/05/06, all 1e-5/property-gated); the shims map real sklearn-named attributes to live accessors. No placeholder data, no hardcoded empties, no TODO/FIXME. The smoke test exercises the Unfit arm; the live `import mlrs` + fit/partial_fit/transform round-trip is the opportunistic phase Python-oracle gate (maturin develop in the /tmp venv), mirroring v1 06-05.

## Threat Flags

None. No new network/auth/file-access surface — this is an in-process extension module (T-07-NA). The threat register is mitigated by reuse: T-07-11 (f64-on-incapable-backend) — `guard_f64()?` runs BEFORE every F64 fit/partial_fit arm (statically grep-verified, the contract gate per WARNING 2); T-07-12 (ingress validation) — the shipped `validated_f32`/`validated_f64` bridge is reused verbatim, and SparseRandomProjection densifies sparse input at the Python boundary before ingress (PROJ-02); T-07-13 (pyo3 ABI) — pyo3 stays 0.28, `cargo tree` resolves exactly one v0.28 ABI. No new Rust/Python compute deps (T-07-SC: scipy is imported lazily only for the optional sparse-densify path and is test/runtime-optional, never linked into the wheel).

## Verification

- `cargo build -p mlrs-py --features cpu` → exit 0.
- `cargo build -p mlrs-py --features cpu,extension-module` → exit 0.
- `cargo test -p mlrs-py --features cpu --test pyclass_smoke_test` → 3 passed, 0 failed (incl. the new `five_phase7_estimators_construct_unfit`).
- `cargo test -p mlrs-py --features rocm --test pyclass_smoke_test --no-run` → builds (rocm test target).
- All four Python shim modules parse (`ast.parse` → py-parse-ok).
- pyo3 resolves to exactly one v0.28 ABI (`cargo tree … | grep pyo3 v | sort -u` → `pyo3 v0.28.3`).
- Acceptance greps: add_class ≥4 (=6); fn partial_fit (decomposition.rs) =1; guard_f64 covariance.rs=3, projection.rs=3, decomposition.rs=6; class EmpiricalCovariance|LedoitWolf =2; class Gaussian|Sparse =2; def partial_fit (py) =1; toarray|issparse|densif (py) =12.
- `cargo clippy` on the touched files is clean; the remaining mlrs-algos/mlrs-kernels warnings are pre-existing in unrelated files (dbscan, jacobi kernels, svd, gemm, runtime — out of scope per the SCOPE BOUNDARY rule).
- LIMITATION (WARNING 2): the smoke test only constructs the Unfit arm, so it never runs live F64 dispatch and cannot dynamically prove the `guard_f64()` contract — the guard is enforced STATICALLY by the Task-1 source-grep acceptance criteria. The dynamic guard_f64 behavior + the `import mlrs` fit/partial_fit round-trip are confirmed opportunistically at the live Python-oracle gate.

## Self-Check: PASSED

- `crates/mlrs-py/src/estimators/covariance.rs` exists (PyEmpiricalCovariance + PyLedoitWolf).
- `crates/mlrs-py/src/estimators/projection.rs` exists (PyGaussianRandomProjection + PySparseRandomProjection + johnson_lindenstrauss_min_dim).
- `crates/mlrs-py/python/mlrs/covariance.py` + `random_projection.py` exist and parse.
- Commits e324afb, ee01780, 2e27482, 1db99e7, 0af7e79 all present in git history.
- Both feature builds exit 0; the smoke test is 3/3 green.
