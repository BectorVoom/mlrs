---
phase: 08-kernel-family
plan: 05
subsystem: python-bindings
tags: [pyo3, kernel-ridge, kernel-density, score-samples, any-estimator, dtype-dispatch, guard-f64, gil-release, KERNEL-01, KERNEL-02, PY-06]

# Dependency graph
requires:
  - phase: 08-kernel-family
    provides: "08-03 KernelRidge<F> estimator (inherent fit(X,y,shape,n_targets)/predict(X,shape) + dual_coef accessor) — KERNEL-01, oracle-validated"
  - phase: 08-kernel-family
    provides: "08-04 KernelDensity<F> estimator (fit(X,shape) + ScoreSamples::score_samples(Q,shape) + bandwidth() accessor) — KERNEL-02, oracle-validated"
  - phase: 06-python-bindings
    provides: "any_estimator! Unfit/F32/F64 dtype-dispatch macro, ingress (Arrow PyCapsule → DeviceArray), egress (to_host_metered), capability::guard_f64, errors (algo_err_to_py/not_fitted), global_pool + py.detach GIL-release contract"
provides:
  - "PyKernelRidge #[pyclass] (KERNEL-01): fit(X,y,rows,cols,n_targets)/predict_f32/_f64 + dual_coef_f32/_f64 accessors; f32/f64 dispatch, GIL release, guard_f64 on the F64 arm"
  - "PyKernelDensity #[pyclass] (KERNEL-02): fit(X,rows,cols)/score_samples_f32/_f64 (the one new exposed method) + log_density_f32/_f64 + single-typed bandwidth_ scalar"
  - "Both pyclasses registered in the _mlrs pymodule; a green Python smoke test exercising the real FFI path across f32/f64"
affects: [phase-11-py06-final-signoff]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "v2 kernel-family Python wrap = ZERO new binding infra: reuse the shipped any_estimator! macro + ingress/egress/capability/errors verbatim (the 07-07 incremental-wrap precedent), dtype-suffixed array accessors, single-typed scalar attrs, guard_f64 on every F64 arm"
    - "Unfit stores the kernel NAME (String tag) + raw scalar hyperparameters; the precision-typed Kernel<F>/KdKernel/BandwidthSpec is built at fit, where the algos estimator resolves gamma=None→1/n_features and scott/silverman from n_features (Open Q3 / D-05 / D-09)"

key-files:
  created:
    - crates/mlrs-py/src/estimators/kernel.rs
    - crates/mlrs-py/tests/test_kernel.py
    - .planning/phases/08-kernel-family/08-05-SUMMARY.md
  modified:
    - crates/mlrs-py/src/estimators/mod.rs
    - crates/mlrs-py/src/lib.rs

key-decisions:
  - "The Python smoke test drives the LOW-LEVEL mlrs._mlrs.KernelRidge/KernelDensity classes directly via pyarrow capsules, NOT a pure-Python sklearn shim. The mlrs/ shim has no kernel-family wrapper module (kernel_ridge.py/kernel_density.py) — that shim work is Plan-04/Phase-11 (PY-06 final) scope, explicitly out of this incremental wrapper share. Driving _mlrs directly proves the FFI path + dtype dispatch + score_samples exposure this plan delivers, which is the plan's stated smoke-test goal."
  - "KernelRidge.predict and KernelDensity.score_samples are dtype-suffixed methods (predict_f32/_f64, score_samples_f32/_f64) rather than a single dispatching method, mirroring linear.rs predict_f32/_f64 — the host return type (Vec<f32> vs Vec<f64>) is monomorphized per arm and the pure-Python shim (future) routes by the fitted dtype() the same way it does for the v1 estimators."
  - "log_density_f32/_f64 are thin delegations to score_samples_f32/_f64 (KernelDensity stores no fitted log-density array — the log-density is produced per score_samples call). They exist for accessor-name symmetry with the v2 dtype-suffixed-array precedent and satisfy the must_haves dtype-suffixed-accessor truth without duplicating compute."

patterns-established:
  - "Pattern: a kernel-family pyclass parses the sklearn kernel-name String → typed enum (parse_kernel_kind / parse_kd_kernel) at the FFI boundary (a clean PyValueError on an unknown name), then the algos estimator re-validates at fit — the validate-at-boundary + validate-before-launch double guard."

requirements-completed: [KERNEL-01, KERNEL-02]

# Metrics
duration: 9min
completed: 2026-06-21
---

# Phase 8 Plan 05: PyKernelRidge + PyKernelDensity Python Wrappers Summary

**Both Phase-8 estimators are exposed to the sklearn-compatible Python surface — `PyKernelRidge` (fit/predict) and `PyKernelDensity` (fit/`score_samples`, the one new exposed method) — reusing the shipped `any_estimator!` Unfit/F32/F64 machinery with ZERO new binding infrastructure: f32/f64 runtime dispatch, `py.detach` GIL release, and `guard_f64()` before every F64 arm, with the kernel name + raw hyperparameters stored in `Unfit` and the typed `Kernel<F>`/`BandwidthSpec` built at `fit` once `n_features` is known. A green Python smoke test drives both estimators end-to-end through the real Arrow-PyCapsule FFI path across both dtypes (4/4 passing on cpu).**

## Performance

- **Duration:** ~9 min
- **Tasks:** 2
- **Files modified:** 5 (3 created — kernel.rs + test_kernel.py + this SUMMARY; 2 modified — mod.rs + lib.rs)

## Accomplishments

- Created `crates/mlrs-py/src/estimators/kernel.rs` with two `crate::any_estimator!` invocations — `AnyKernelRidge` over `mlrs_algos::kernel_ridge::kernel_ridge::KernelRidge` and `AnyKernelDensity` over `mlrs_algos::density::kernel_density::KernelDensity` — adding NO new binding infrastructure (the 07-07 incremental-wrap precedent).
- `PyKernelRidge`: `#[new](kernel, alpha, gamma, degree, coef0)` storing the kernel NAME + raw hyperparameters in `Unfit`; `fit(X, y, rows, cols, n_targets=1)` parses the name → `KernelKind`, dispatches on the X float dtype, releases the GIL via `py.detach`, and calls `guard_f64()` BEFORE the F64-arm upload; `predict_f32`/`predict_f64` return the row-major `(rows × n_targets)` host vector; `dual_coef_f32`/`dual_coef_f64` are the dtype-suffixed fitted accessors; `is_fitted`/`dtype` helpers.
- `PyKernelDensity`: `#[new](kernel, bandwidth, bandwidth_rule)` storing the kernel NAME + bandwidth spec; `fit(X, rows, cols)` parses → `KdKernel` + `BandwidthSpec` (numeric/scott/silverman), dispatches + GIL-releases + `guard_f64`s the F64 arm; `score_samples_f32`/`score_samples_f64` (the ONE new exposed method, D-12) return the length-`rows` log-density vector; `log_density_f32`/`_f64` dtype-suffixed accessors (delegating to score_samples); single-typed `bandwidth_()` `f64` scalar; `is_fitted`/`dtype`.
- Registered both pyclasses in the `_mlrs` pymodule (`lib.rs`: `use estimators::kernel::{PyKernelDensity, PyKernelRidge};` + two `m.add_class::<…>()?;`) and added `pub mod kernel;` to `estimators/mod.rs`.
- Added `crates/mlrs-py/tests/test_kernel.py`: a Python smoke test that fits `KernelRidge` (rbf) and asserts `predict` returns the right shape + tracks a sklearn `KernelRidge` reference within a loose smoke band, and fits `KernelDensity` (gaussian) and asserts `score_samples(Q)` returns a length-`rows(Q)` vector of FINITE log-densities tracking sklearn `KernelDensity` — both exercising f32 AND f64 dispatch, with the f64 cases gated behind `mlrs._mlrs.backend_supports_f64()`.

## Task Commits

1. **Task 1: PyKernelRidge + PyKernelDensity (any_estimator! + dispatch + GIL + guard_f64)** — `92def02` (feat)
2. **Task 2: Python smoke test (fit/predict/score_samples, f32/f64 dispatch)** — `c4a5334` (test)

**Plan metadata:** _(final docs commit follows this summary)_

## Files Created/Modified

- `crates/mlrs-py/src/estimators/kernel.rs` — NEW: `PyKernelRidge` + `PyKernelDensity` (two `any_estimator!` invocations + `#[pymethods]`); `parse_kernel_kind`/`parse_kd_kernel`/`parse_bandwidth` boundary parsers.
- `crates/mlrs-py/src/estimators/mod.rs` — added `pub mod kernel;`.
- `crates/mlrs-py/src/lib.rs` — `use estimators::kernel::{PyKernelDensity, PyKernelRidge};` + two `m.add_class::<…>()?;` in the `_mlrs` pymodule.
- `crates/mlrs-py/tests/test_kernel.py` — NEW: the f32/f64 fit/predict/score_samples smoke test against the low-level `mlrs._mlrs` classes.

## Decisions Made

- **Smoke test drives `mlrs._mlrs` directly, not a pure-Python shim.** The `mlrs/` package has no `kernel_ridge.py`/`kernel_density.py` family-module wrapper — building those sklearn-subclassing shims is Plan-04/Phase-11 (PY-06 final) scope, explicitly out of this incremental wrapper share. The smoke test therefore imports the low-level `_mlrs.KernelRidge`/`KernelDensity` extension classes (the surface THIS plan ships) and drives them with fresh-contiguous pyarrow capsules (mirroring `mlrs._io.normalize_X`), proving the FFI path + dtype dispatch + `score_samples` exposure end to end — exactly the plan's stated smoke-test goal.
- **Dtype-suffixed output methods (`predict_f32/_f64`, `score_samples_f32/_f64`).** Each returns a monomorphized host `Vec<f32>`/`Vec<f64>`, mirroring `linear.rs::predict_f32/_f64`. A future pure-Python shim routes by the fitted `dtype()` exactly as it does for the v1 estimators — no new dispatch shape.
- **`log_density_f32/_f64` delegate to `score_samples`.** KernelDensity stores no fitted log-density array (the log-density is produced per `score_samples` call), so the dtype-suffixed `log_density_*` accessors thinly delegate to `score_samples_*`. They satisfy the must_haves dtype-suffixed-accessor truth and keep accessor-name symmetry with the v2 array-accessor precedent without duplicating compute.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Smoke-test harness location + driver: the pure-Python shim has no kernel module**
- **Found during:** Task 2.
- **Issue:** The plan's Task 2 read_first points at the existing py harness "`maturin develop` / `import mlrs`" flow, but the pure-Python `mlrs` package exposes no `KernelRidge`/`KernelDensity` (the shim family-module wrappers are Plan-04/Phase-11 scope). Driving the test through `import mlrs` would require shipping shim code outside this plan's file set.
- **Fix:** Wrote `test_kernel.py` against the low-level `mlrs._mlrs.KernelRidge`/`KernelDensity` extension classes (the FFI surface this plan delivers) using `pytest.importorskip` guards so collection never errors without the built extension. This is the in-scope FFI-path proof the plan asked for; no shim code was added.
- **Files modified:** crates/mlrs-py/tests/test_kernel.py
- **Committed in:** `c4a5334` (Task 2)

**Total deviations:** 1 auto-fixed (blocking — harness wiring). No behavior deviation: the wrappers, dispatch contracts, and `score_samples` exposure are exactly as the plan specified; the plan's `files_modified` set and all acceptance criteria are met.

## Verification Evidence

- `cargo build -p mlrs-py --features cpu` → clean (no warnings).
- Grep gates (Task 1): `kernel.rs` has exactly 2 `crate::any_estimator! {` invocations (lines 100, 297), `score_samples` present, `guard_f64` ×3 (2 fit F64 arms + 1 macro-doc mention — both runtime F64 arms guarded BEFORE upload, statically grep-verifiable per STATE.md [07-07] WARNING 2); `lib.rs` has `add_class::<PyKernelRidge>` + `add_class::<PyKernelDensity>`. `PY_KERNEL_OK` emitted.
- Grep gates (Task 2): `crates/mlrs-py/tests/test_kernel.py` exists, contains `score_samples`, references `float32`/`float64`. `PY_SMOKE_PRESENT` emitted.
- Python smoke test: built the cpu extension via `maturin develop --release` (temp-root-pyproject dance, cpu.pyproject.toml) into a /tmp venv (numpy/pyarrow/scikit-learn/pytest, PEP 668), then `pytest crates/mlrs-py/tests/test_kernel.py -v` → **4 passed** (`test_kernel_ridge_predict[f32]`, `[f64]`, `test_kernel_density_score_samples[f32]`, `[f64]`). cpu supports f64 so the f64 arms RAN (not skipped); predict + score_samples matched the sklearn smoke band; shapes + finiteness asserted.
- rocm f32 opportunistic gate (build/run on gfx1100) documented as manual — not run in this cpu execution; the f64 smoke cases are `mlrs._mlrs.backend_supports_f64()`-gated so they skip-with-reason on the f64-incapable rocm wheel (mirrors the v1 skip precedent).

## Next Phase Readiness

- KERNEL-01 + KERNEL-02 are now exposed end to end (algos estimator → `_mlrs` pyclass → Python), completing Phase 8's user-facing deliverable. The kernel-family wrappers reuse the v1 binding infra verbatim, so the Phase-11 PY-06 final cross-cutting sign-off (sklearn-shim family modules, `estimator_checks`, oracle replay through the shim) can wrap these the same way it wraps the v1 estimators.
- Phase 8 plan 5/5 complete — phase ready for verification / close.

---
*Phase: 08-kernel-family*
*Completed: 2026-06-21*

## Self-Check: PASSED

All 3 created files (`kernel.rs`, `test_kernel.py`, this SUMMARY) verified present on disk; both task commits (`92def02`, `c4a5334`) verified in git history; `cargo build -p mlrs-py --features cpu` clean; the Python smoke test ran green 4/4 (f32 + f64 × KernelRidge.predict + KernelDensity.score_samples) through the real FFI path.
