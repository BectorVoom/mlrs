---
phase: 08-kernel-family
plan: 03
subsystem: estimators
tags: [kernel-ridge, KERNEL-01, kernel-matrix, cholesky, dual-solve, multi-rhs, gamma, oracle]

# Dependency graph
requires:
  - phase: 08-kernel-family
    provides: "08-02 kernel_matrix prim (PRIM-08): K(X,Y) for linear/rbf/poly/sigmoid, oracle-validated 1e-15 f64 / 1e-7 f32"
  - phase: 08-kernel-family
    provides: "08-01 Wave-0 scaffold: Kernel<F> enum (D-01), AlgoError InvalidAlpha/InvalidDegree/InvalidKernel guards, kernel_ridge/ module home, kernel_ridge_test #[ignore] scaffold, committed kernel_ridge oracle fixtures (f32+f64, seed42)"
  - phase: 04-closed-form
    provides: "cholesky_solve (multi-RHS A=L·Lᵀ + fwd/back subst, n≤MAX_DIM=64) + ridge.rs diagonal-α host-pass idiom"
  - phase: 02-primitives
    provides: "gemm (K_test·dual_coef_ matmul)"
provides:
  - "KernelRidge<F> estimator (KERNEL-01): fit (kernel_matrix → diagonal-α → multi-RHS Cholesky dual solve) + predict (K(X_test,X_fit_)·dual_coef_), NO centering / NO intercept (D-06)"
  - "KernelKind selector enum (Linear/Rbf/Poly/Sigmoid) + resolved typed Kernel<F> reused verbatim by predict (Pitfall 5)"
  - "KERNEL-01 oracle tests: 4 kernels + 2-target multi-RHS (D-04) + gamma=None and explicit (D-05), f64 strict 1e-5 / f32 documented band"
affects: [08-05-py-wrappers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dual-coefficient kernel solve = ridge.rs Cholesky path MINUS centering: the normal matrix is K (kernel_matrix(X,X)) not XᵀX, α on the K diagonal, no intercept, multi-target in one multi-RHS cholesky_solve (rhs=t, D-04)"
    - "Resolved gamma baked into the typed Kernel<F> at fit so predict reuses the IDENTICAL kernel (fit-time == predict-time gamma, Pitfall 5)"

key-files:
  created:
    - crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs
  modified:
    - crates/mlrs-algos/src/kernel_ridge/mod.rs
    - crates/mlrs-algos/tests/kernel_ridge_test.rs

key-decisions:
  - "KernelRidge uses an INHERENT fit(x, y, shape, n_targets) / predict(x, shape) pair, NOT the Fit/Predict traits: the Fit trait's y:Option<&DeviceArray>+shape signature cannot carry the multi-target count n_targets the multi-RHS dual solve (D-04) requires. Inherent methods keep the multi-target surface the plan's behavior spec mandates (fit(X n×d, y n×t))."
  - "Diagonal-α host pass copied verbatim from ridge.rs:248-254 but over K (n_samples×n_samples) not XᵀX (n_features²): materialize K → k_host[i*n+i]+=alpha → release K → re-stage regularized K (recycles the released n² buffer), then thread it through cholesky_solve's `out` for in-place factorization (no parallel n² live)."
  - "f32 documented band KR_F32_BAND = (1e-4, 1e-4); the f32 path actually clears the STRICT 1e-5 absolute arm (observed 3.6e-7) so the band is conservative headroom, not a needed relaxation."

patterns-established:
  - "Pattern: a kernel estimator stores X_fit_ (a fresh device copy of the training matrix) so predict can rebuild the cross kernel K(X_test, X_fit_) against it with the resolved fit-time Kernel<F>."

requirements-completed: [KERNEL-01]

# Metrics
duration: 4min
completed: 2026-06-20
---

# Phase 8 Plan 03: KernelRidge (KERNEL-01) Summary

**KernelRidge fits the dual coefficients of `(K + αI)·dual_coef_ = y` over the validated Phase-8 `kernel_matrix` keystone — α on the K diagonal only, multi-target in one multi-RHS `cholesky_solve` (D-04), NO centering / NO intercept (D-06) — and predicts `K(X_test, X_fit_)·dual_coef_`, matching scikit-learn `KernelRidge` to ~5.6e-16 (f64) / ~3.6e-7 (f32) across linear/rbf/poly/sigmoid, multi-target, and both gamma paths.**

## Performance
- **Duration:** ~4 min
- **Tasks:** 2
- **Files modified:** 3 (1 created — the estimator; mod.rs + test re-wired)

## Accomplishments
- Built `KernelRidge<F>` + `KernelKind` under the Plan-01 module home (`kernel_ridge/mod.rs` adds its own `pub mod` / `pub use`; `lib.rs` untouched, owned by the Wave-0 scaffold).
- `fit(X, y, shape, n_targets)`: validates `alpha>=0` (InvalidAlpha), `degree>=1` for poly (InvalidDegree), the kernel name (InvalidKernel) and geometry BEFORE any launch (T-08-03-01); resolves `gamma=None → 1/n_features` and bakes it into the typed `Kernel<F>` (D-05, Pitfall 5); builds `K = kernel_matrix(X, X, kernel)` (n×n, D-02); adds α to the K diagonal only (D-06, the ridge.rs:248-254 host pass over K); solves `dual_coef_ = cholesky_solve(K+αI, y, n, rhs=n_targets)` in ONE multi-RHS call (D-04).
- `predict(X_test)`: `K_test = kernel_matrix(X_test, X_fit_, kernel)` (m×n, resolved fit-time kernel) then `y_pred = K_test · dual_coef_` via gemm (m×t); NO intercept broadcast (D-06).
- Wired the oracle test (removed the Wave-0 `#[ignore]`s): one case per kernel (linear/rbf/poly/sigmoid, single target), a 2-target multi-RHS rbf case (`y_multi`, D-04), gamma=None default (`y_rbf`) AND explicit gamma=0.5 (`y_rbf_gamma`, D-05), with degree=3/coef0=1 defaults exercised by the poly/sigmoid cases. f64 strict `F64_TOL` behind `skip_f64_with_log`; f32 the documented `KR_F32_BAND`.

## Task Commits
1. **Task 1: KernelRidge struct + fit (kernel_matrix → diagonal-α → multi-RHS Cholesky)** — `0258d82` (feat)
2. **Task 2: KernelRidge predict + ≤1e-5 oracle (4 kernels, multi-target, gamma both paths)** — `4d522ca` (feat)

**Plan metadata:** _(final docs commit follows this summary)_

## Files Created/Modified
- `crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs` — KernelRidge<F> + KernelKind: fit (dual solve) + predict (cross-kernel gemm) + host_to_f64/f64_to_host helpers
- `crates/mlrs-algos/src/kernel_ridge/mod.rs` — added `pub mod kernel_ridge; pub use kernel_ridge::{KernelKind, KernelRidge};`
- `crates/mlrs-algos/tests/kernel_ridge_test.rs` — flipped Wave-0 `#[ignore]`s to live oracle cases (4 kernels + multi-target + both gamma paths, f32 + f64)

## Decisions Made
- **Inherent fit/predict over the Fit/Predict traits (multi-target):** The `Fit` trait's `y: Option<&DeviceArray>` + `shape` signature has no slot for the target count `t` the multi-RHS dual solve (D-04) needs. KernelRidge exposes inherent `fit(x, y, shape, n_targets)` / `predict(x, shape)` so a multi-target `y` (n×t) is first-class — the behavior the plan's spec mandates (`fit(X (n×d), y (n×t))`). The trait surface stays unimplemented for this estimator (its `predict` returns n×t, not the trait's length-n contract).
- **Diagonal-α over K, in-place factorization:** copied the ridge.rs:248-254 diagonal-stride `+= alpha` host pass verbatim but applied it to `K` (n_samples²) instead of `XᵀX` (n_features²); the regularized K buffer is threaded through `cholesky_solve`'s `out` for in-place factorization (no parallel n² live), exactly the ridge.rs:276-280 idiom.
- **f32 band is conservative:** `KR_F32_BAND = (1e-4, 1e-4)` per the v1 documented-band precedent, but the f32 path measured max abs error 3.6e-7 — it actually clears the strict 1e-5 absolute arm. The band is documented headroom, not a needed loosening.

## Deviations from Plan
None — plan executed as written. The inherent-method choice (vs the Fit/Predict traits) is a signature realization of the plan's own multi-target `fit(X (n×d), y (n×t))` behavior spec, not a behavior deviation; the plan's `files_modified` and all acceptance criteria are met exactly.

## Verification Evidence
- `cargo build -p mlrs-algos --features cpu` → clean (no warnings).
- `cargo test --features cpu -p mlrs-algos --test kernel_ridge_test` → **4 passed; 0 failed**
  (`kernel_ridge_all_kernels_f64`, `kernel_ridge_all_kernels_f32`, `kernel_ridge_multi_target_f64`, `kernel_ridge_multi_target_f32`).
- Observed max abs errors (cpu): f64 all-kernels **5.55e-16**, f64 multi-target **4.44e-16** (strict 1e-5); f32 all-kernels **3.58e-7**, f32 multi-target **3.58e-7** (band 1e-4, also inside strict 1e-5).
- Grep gates: `KR_FIT_OK` (struct KernelRidge + kernel_matrix + cholesky_solve present; no x_mean/y_mean/intercept_), `KR_PREDICT_GATES_OK` (kernel_matrix + cholesky_solve + no centering tokens).
- rocm f32 opportunistic gate (`cargo test --features rocm kernel_ridge`) documented as manual/gfx1100; not run in this cpu execution (f64 cases behind `skip_f64_with_log` for the rocm f64 gap).

## Next Phase Readiness
- KERNEL-01 estimator is live and oracle-validated on the keystone prim; the kernel_matrix seam is exercised end-to-end (fit Gram K(X,X) + predict cross kernel K(X_test, X_fit_)).
- Plan 08-05 (Python wrappers) can wrap `KernelRidge` against the inherent `new`/`fit`/`predict`/`dual_coef` surface.
- File-disjoint from Plan 04 (KernelDensity); both depend only on Plan 02 and ran in the same wave.

---
*Phase: 08-kernel-family*
*Completed: 2026-06-20*

## Self-Check: PASSED

All 4 created/modified files verified present on disk; both task commits (`0258d82`, `4d522ca`) verified in git history.
