---
phase: 04-closed-form-estimators
plan: 04
subsystem: decomposition
tags: [pca, truncated-svd, svd, svd-flip, explained-variance, sklearn, oracle, device-resident]

# Dependency graph
requires:
  - phase: 04-closed-form-estimators
    plan: 01
    provides: Fit/Transform traits (+ inverse_transform default), AlgoError::InvalidNComponents, decomposition/mod.rs stub, pca tall/wide + tsvd arpack fixtures
  - phase: 03-svd-eigendecomposition-primitive-hard-gate
    provides: thin SVD primitive (U/S/Vᵀ descending), sign_flip::align_rows, skip_f64_with_log gate, oracle load_npz/OracleCase
  - phase: 02-core-compute-primitives
    provides: gemm (transpose flags, D-06), column_reduce (ScalarOp::Mean), DeviceArray/BufferPool
provides:
  - "Pca<F> (DECOMP-01): Fit + Transform + inverse_transform; centered-X SVD; explained_variance_=S²/(n−1); ratio over FULL spectrum; svd_flip via align_rows; device-resident fitted state"
  - "TruncatedSvd<F> (DECOMP-02): Fit + Transform; UNCENTERED-X SVD; explained_variance_=var(transform cols, ddof=0); ratio denom = total per-feature var of X; inverse_transform Unsupported"
  - "PCA matches sklearn svd_solver='full' and TruncatedSVD matches arpack within 1e-5 on cpu(f64)+rocm(f32), incl. PCA wide (n_features>n_samples) Aᵀ-swap case"
  - "gen_oracle.py c() forces C-contiguous arrays (fixes Fortran-order components_ ravel)"
affects: [06-pyo3-bindings]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Decomposition skeleton: column_reduce(Mean)/center → svd → align_rows(Vᵀ) → truncate; PCA and TruncatedSVD share it with three documented per-estimator differences"
    - "svd_flip applied estimator-side via align_rows on Vᵀ rows; the SVD primitive stays raw (D-01/D-03)"
    - "explained_variance_ formulas kept DISTINCT: PCA S²/(n−1) (ddof=1); TruncatedSVD var(transform cols) (ddof=0) — RESEARCH Pitfall 2"
    - "transform = (X[−mean])·componentsᵀ via gemm transb (components_ is nc×n_features row-major, read transposed — no transpose buffer, D-06)"

key-files:
  created:
    - crates/mlrs-algos/src/decomposition/pca.rs
    - crates/mlrs-algos/src/decomposition/truncated_svd.rs
  modified:
    - crates/mlrs-algos/src/decomposition/mod.rs
    - crates/mlrs-algos/tests/pca_test.rs
    - crates/mlrs-algos/tests/truncated_svd_test.rs
    - scripts/gen_oracle.py
    - tests/fixtures/pca_f32_seed42.npz
    - tests/fixtures/pca_f64_seed42.npz
    - tests/fixtures/pca_tall_f32_seed42.npz
    - tests/fixtures/pca_tall_f64_seed42.npz
    - tests/fixtures/pca_wide_f32_seed42.npz
    - tests/fixtures/pca_wide_f64_seed42.npz

key-decisions:
  - "explained_variance_ host-computed: the reduce prim has no Variance ScalarOp, and the ratio/truncation already need a host pass over the tiny length-k S vector — so S²/(n−1) (PCA) and var(transform cols) (TSVD) are formed host-side; the heavy svd/gemm stay on-device (mirrors LinearRegression's host σ⁺ scaling)"
  - "PCA explained_variance_ratio_ denominator is the sum over ALL S (full spectrum) BEFORE truncation (RESEARCH Pitfall 6); a zero total-variance denominator is guarded to 1.0 (T-04-04-03)"
  - "Rule-1 fixture fix: gen_oracle.py c() now wraps np.ascontiguousarray — sklearn PCA components_ is Fortran-contiguous, so the committed npz stored a column-major ravel that load_npz read transposed; regenerated the 6 PCA fixtures C-contiguous so the documented row-major contract holds"

requirements-completed: [DECOMP-01, DECOMP-02]

# Metrics
duration: 11min
completed: 2026-06-12
---

# Phase 4 Plan 04: PCA + TruncatedSVD Decomposition Estimators Summary

**`Pca<F>` (DECOMP-01, centered-X SVD with `explained_variance_=S²/(n−1)`, `svd_flip` via `align_rows`, `transform`/`inverse_transform`) and `TruncatedSvd<F>` (DECOMP-02, UNCENTERED-X arpack SVD with `explained_variance_=var(transform cols)`) — both built on the SAME Phase-3 thin SVD + `align_rows` skeleton with three documented per-estimator differences, matching scikit-learn within 1e-5 on cpu(f64)+rocm(f32) including the PCA wide (n_features>n_samples) case.**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-06-12T07:23Z
- **Completed:** 2026-06-12T07:34Z
- **Tasks:** 2
- **Files modified:** 11 (2 created, 3 modified source/test + gen_oracle.py + 6 regenerated PCA fixtures)

## Accomplishments

- `Pca<F>` (`Fit` + `Transform` + `inverse_transform`) composes the validated Phase-3 thin SVD + Phase-2 `column_reduce(Mean)` + `gemm` — NO eig-of-covariance, NO bespoke matmul. `fit` centers X by column means, runs SVD of the CENTERED design, computes `explained_variance_=S²/(n−1)` over ALL S, `explained_variance_ratio_` over the FULL spectrum BEFORE truncation (RESEARCH Pitfall 6), applies `align_rows` (= sklearn `svd_flip(u_based_decision=False)`) to the Vᵀ rows, and truncates to `n_components`.
- `TruncatedSvd<F>` (`Fit` + `Transform`) reuses the SAME skeleton with the three RESEARCH-Pitfall-2 differences: (1) NO centering — SVD of UNCENTERED X; (2) `explained_variance_ = var(transform cols, ddof=0)` (population variance of U·S columns), NOT `S²/(n−1)`; (3) `explained_variance_ratio_` denominator = total per-feature variance of the ORIGINAL X. `inverse_transform` stays the trait default (`Unsupported`, D-01).
- `svd_flip` is applied BY THE ESTIMATOR via `align_rows` on the Vᵀ rows; the SVD primitive stays raw (D-01/D-03). `transform = (X[−mean])·componentsᵀ` is a single `gemm` with `transb` reading the `(nc×n_features)` `components_` as its transpose — no transpose buffer (D-06).
- `n_components > min(n_samples, n_features)` (and `==0`) is rejected with `AlgoError::InvalidNComponents` BEFORE any prim launch (T-04-04-01); `n_samples ≤ 1` (undefined variance) and geometry mismatch are rejected too (T-04-04-02/03).
- Fitted state (`components_`/`explained_variance_`/`explained_variance_ratio_`/`singular_values_`/`mean_`) is device-resident (D-03); host materialization only at the Rust accessor / oracle boundary.
- Activated 16 oracle tests: PCA (10 — tall + wide, f32 + f64, asserting components_/mean_/singular_values_/explained_variance_/ratio/transform/inverse) and TruncatedSVD (6 — f32 + f64, asserting components_/singular_values_/explained_variance_/transform). All within the strict 1e-5 abs+rel contract after `align_rows`; f64 runs on cpu and skips-with-log on rocm.

## Task Commits

1. **Task 1: PCA estimator (centered SVD, svd_flip, S²/(n−1)) + activate pca_test** — `c8ccaa9` (feat, carries the Rule-1 fixture fix)
2. **Task 2: TruncatedSVD estimator (uncentered SVD, var(transform)) + activate test** — `35e997b` (feat)

**Plan metadata:** _(final docs commit — this SUMMARY + STATE + ROADMAP + REQUIREMENTS)_

## Test Results (reported honestly)

**cpu (f64) gate** — `cargo test -p mlrs-algos --features cpu --test pca_test --test truncated_svd_test`:
```
running 10 tests   (pca_test)
test result: ok. 10 passed; 0 failed; 0 ignored
running 6 tests    (truncated_svd_test)
test result: ok. 6 passed; 0 failed; 0 ignored
```

**rocm (f32) gate** — `cargo test -p mlrs-algos --features rocm --test pca_test --test truncated_svd_test`:
```
running 10 tests   (pca_test)
test result: ok. 10 passed; 0 failed; 0 ignored
running 6 tests    (truncated_svd_test)
test result: ok. 6 passed; 0 failed; 0 ignored
```
The f64 functions print `... SKIPPED (no f64 support on this adapter)` and return early (verified via `--nocapture`: `pca f64 backend=rocm: SKIPPED (no f64 support on this adapter)`); the f32 functions run on the real ROCm GPU. All within the strict 1e-5 abs+rel contract.

## Files Created/Modified

- `crates/mlrs-algos/src/decomposition/pca.rs` (created) — `struct Pca<F>` + `Fit`/`Transform` (+ `inverse_transform`); centered-X SVD, host `explained_variance_`/ratio, `align_rows` svd_flip, device-resident accessors.
- `crates/mlrs-algos/src/decomposition/truncated_svd.rs` (created) — `struct TruncatedSvd<F>` + `Fit`/`Transform`; uncentered-X SVD, `var(transform cols)` explained variance, three documented differences from PCA.
- `crates/mlrs-algos/src/decomposition/mod.rs` (modified) — uncommented `pub mod pca;` and `pub mod truncated_svd;` (the only changes; `lib.rs` untouched, owned by 04-01).
- `crates/mlrs-algos/tests/pca_test.rs` (modified) — removed `#[ignore]`; wired 10 sklearn `svd_solver='full'` oracle assertions (tall + wide) after `align_rows`, keeping `skip_f64_with_log`.
- `crates/mlrs-algos/tests/truncated_svd_test.rs` (modified) — removed `#[ignore]`; wired 6 sklearn arpack oracle assertions after `align_rows`, keeping `skip_f64_with_log`.
- `scripts/gen_oracle.py` (modified) — `gen_pca`'s `c()` now forces `np.ascontiguousarray` (Rule-1 fix below).
- `tests/fixtures/pca_{,tall_,wide_}{f32,f64}_seed42.npz` (regenerated) — 6 PCA fixtures rewritten C-contiguous; byte-identical numerics, only the `components_` memory order changed.

## Decisions Made

- **Host-side `explained_variance_`.** The Phase-2 `reduce` prim exposes no Variance `ScalarOp`, and the ratio computation + `n_components` truncation already require a host pass over the tiny length-k `S` vector. So both `S²/(n−1)` (PCA) and the `var(transform cols)` (TruncatedSVD) are computed host-side; the heavy `svd`/`gemm` products stay on-device. This mirrors LinearRegression's host σ⁺ scaling precedent (04-03).
- **Ratio denominator over the FULL spectrum (Pitfall 6).** PCA's `explained_variance_ratio_` divides by the sum over ALL `S²/(n−1)` BEFORE truncation, so the ratios match sklearn and sum to ≤ 1. A degenerate zero total-variance denominator is guarded to `1.0` (T-04-04-03).
- **`column_reduce(ScalarOp::Mean)` as the PCA `mean_` key-link.** PCA's `mean_` is produced by the Phase-2 column-mean reduction (the documented key-link prim call); the centering itself is then a host pass (the per-column means are tiny).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] PCA fixtures stored `components_` in Fortran (column-major) order, breaking the row-major contract**
- **Found during:** Task 1 (the RED run — `pca_components_mean_singular_values_{f32,f64}` and both `pca_wide_*` FAILED with `components_` mismatches like `got=0.6227 expected=0.6045`, while `mean_`/`singular_values_`/`explained_variance_`/`transform` all PASSED).
- **Issue:** `sklearn.decomposition.PCA.components_` is FORTRAN-contiguous (it comes from scipy's column-major `Vt`). `gen_oracle.py`'s `c()` did `np.asarray(arr).astype(dtype)`, which preserves that F-order, and `np.savez` stores the raw buffer. `npyz`'s `into_vec()` (used by `load_npz`) returns the stored buffer WITHOUT reordering to C-order, so `case.expect_f64("components_")` yielded the column-major ravel — a transposed flat buffer — silently breaking the row-major `n_components × n_features` contract every Rust consumer assumes. The 04-01 scaffold only asserted array LENGTH, so the defect was latent until a real value comparison. `transform`/`X` were already C-contiguous (hence those assertions passed), and TruncatedSVD's `components_` is C-contiguous (arpack path), so ONLY the PCA fixtures were affected.
- **Fix:** `gen_oracle.py`'s `c()` now wraps `np.ascontiguousarray(...)` so every committed array is row-major. Regenerated the 6 PCA fixtures via the project oracle venv (numpy/scipy/scikit-learn, seed 42 — numerics byte-identical, only the `components_` storage order changed). After the fix all 10 PCA tests pass on both gates.
- **Files modified:** `scripts/gen_oracle.py`, `tests/fixtures/pca_{,tall_,wide_}{f32,f64}_seed42.npz`
- **Commit:** `c8ccaa9`
- **Scope note:** `gen_oracle.py` is nominally owned by 04-01, but the fix is a root-cause correctness fix for a fixture-convention bug that directly blocks DECOMP-01 verification (Rule 1 / Rule 3). It is additive (`np.ascontiguousarray` is a no-op for already-C arrays) and does not alter any other estimator's fixtures.

_The TruncatedSVD task (Task 2) passed on the FIRST RED run — the C-contiguous arpack fixture and the distinct `var(transform)` formula were correct from the initial implementation, confirming the shared skeleton + three-differences design._

## Authentication Gates

None — no auth or network was required. The PCA fixtures were regenerated from the local oracle venv (numpy/scipy/scikit-learn); the committed blobs are consumed at test time.

## TDD Gate Compliance

Both tasks carry `tdd="true"`. Structure per task: the estimator was added first (the activated test references the new `Pca`/`TruncatedSvd` symbol, so the test crate could not compile before it existed), then `#[ignore]` was removed to fire the oracle assertions — the RED gate. Task 1's RED run failed on the PCA `components_` order (documented above), driving the Rule-1 fixture fix → GREEN. Task 2's RED run passed immediately (correct from implementation). Each task is a single atomic `feat(...)` commit carrying both the implementation and the activated RED→GREEN test (mirrors 04-03's structure where the activated test is the RED gate within the same task).

## Known Stubs

None. `inverse_transform` on TruncatedSVD is the deliberate trait-default `Unsupported` (D-01, only PCA reconstructs in v1) — a real, documented decision, not a stub. All fitted state is wired to real device buffers; no placeholder/empty values flow to the oracle.

## Threat Flags

None. The two estimators introduce only the host→estimator `n_components`/geometry boundary already enumerated in the plan's `<threat_model>` (T-04-04-01/02/03), all mitigated (n_components + shape + n_samples≤1 rejected before launch). No new network/auth/file surface.

## Self-Check: PASSED

- `crates/mlrs-algos/src/decomposition/pca.rs` — FOUND
- `crates/mlrs-algos/src/decomposition/truncated_svd.rs` — FOUND
- `crates/mlrs-algos/src/decomposition/mod.rs` — `pub mod pca;` + `pub mod truncated_svd;` present
- `crates/mlrs-algos/tests/{pca,truncated_svd}_test.rs` — FOUND (activated, no `#[ignore]`)
- Commits `c8ccaa9`, `35e997b` — present in git log
- `cargo test -p mlrs-algos --features cpu --test pca_test` — 10 passed, 0 failed
- `cargo test -p mlrs-algos --features cpu --test truncated_svd_test` — 6 passed, 0 failed
- `cargo test -p mlrs-algos --features rocm` equivalents — 10 + 6 passed (f64 skips-with-log)
- No deletions introduced by either task commit

## Next Phase Readiness

- **DECOMP-01 + DECOMP-02 satisfied.** The full Phase-4 decomposition surface (`Pca`, `TruncatedSvd`) is green on both gates.
- **06 (PyO3 bindings)** ready: `Pca`/`TruncatedSvd` implement the uniform `Fit`/`Transform`(+inverse) trait surface the bindings wrap generically; fitted attributes have host accessors.
- **04-05 (Ridge)** unaffected: this plan touched only `decomposition/` + its tests + `gen_oracle.py`/PCA fixtures; `linear/` is untouched. The `gen_oracle.py` C-contiguous fix is additive and does not change Ridge/LinearRegression fixtures (those arrays were already C-contiguous).

---
*Phase: 04-closed-form-estimators*
*Completed: 2026-06-12*
