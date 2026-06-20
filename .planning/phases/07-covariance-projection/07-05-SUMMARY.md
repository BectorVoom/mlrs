---
phase: 07-covariance-projection
plan: 05
subsystem: algos-estimators
tags: [incremental-pca, decomp-03, partial-fit, streaming-svd, whiten, gen-batches, ddof1, host-glue]

# Dependency graph
requires:
  - phase: 07-covariance-projection
    plan: 01
    provides: "PartialFit<F> trait (D-01), AlgoError::InvalidBatchSize/InvalidNComponents guards, decomposition/mod.rs index, incremental_pca_test.rs #[ignore] scaffold, incremental_pca_* oracle blobs"
  - phase: 07-covariance-projection
    plan: 03
    provides: "prims::incremental_svd::merge + IncrementalSvdState (PRIM-07), measured f32 band (abs 3.6e-7 / rel 2.0e-6)"
  - phase: 04-closed-form-estimators
    provides: "pca.rs center+GEMM transform/inverse_transform blocks, Fit/Transform trait surface, align_rows estimator-side svd_flip"
provides:
  - "mlrs_algos::decomposition::IncrementalPCA<F> — streaming PCA estimator (DECOMP-03), PartialFit<F> + Fit<F> + Transform<F> (+ inverse_transform)"
  - "sklearn-faithful fit(): resets state, batch_size=unwrap_or(5*n_features), loops partial_fit over a verbatim sklearn gen_batches(n, bs, n_components)"
  - "whiten on/off transform (scale by 1/sqrt(explained_variance_)) + un-whitening inverse_transform (D-06)"
  - "IPCA_F32_BAND = 1e-4 estimator-level f32 band (pinned from the Plan 07-03 PRIM-07 standalone f32 measurement)"
affects: [07-07-pyo3-wrappers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "IncrementalPCA is a thin sklearn-faithful driver over the PRIM-07 merge — partial_fit calls merge once per batch and consumes its align_rows/ddof=1/ratio finalize (D-01); no decomposition math re-derived in the estimator"
    - "The running IncrementalSvdState keeps components_ device-resident in f64; the F-generic estimator re-casts components_/mean_/explained_variance_ to F at the transform/inverse_transform GEMM boundary"
    - "gen_batches is a VERBATIM port of sklearn.utils.gen_batches (iterate n//batch_size full batches, `continue` when end+min_batch_size>n, then emit a final start..n slice) — not a heuristic, so all remainder edge cases match sklearn fit()"
    - "whiten only affects transform/inverse_transform; the fitted attrs (components_/ev/sv/mean/var) are identical to the unwhitened fit — the whiten fixture's attrs were asserted against both"

key-files:
  created:
    - crates/mlrs-algos/src/decomposition/incremental_pca.rs
  modified:
    - crates/mlrs-algos/src/decomposition/mod.rs
    - crates/mlrs-algos/tests/incremental_pca_test.rs

key-decisions:
  - "batch_size is an explicit ctor parameter Option<usize> (D-09): the fixture uses IncrementalPCA(nc=3, whiten, batch_size=10).fit(X), so the estimator's fit() must respect an explicit batch_size (10) — the 5*n_features default (=30 for 6 features = one batch) would NOT reproduce sklearn's 3-batch stream. The tests construct new(3, whiten, Some(10))."
  - "gen_batches ported verbatim from sklearn's loop (min_batch_size=n_components) rather than a fold-the-remainder heuristic — verified against sklearn gen_batches across n=30/31/32/33, bs=10, mb=3 and n=10, bs=3."
  - "whiten=False is asserted both via the explicit partial_fit stream AND via the one-shot fit() for every attr; whiten=True asserts attrs (identical to unwhitened) + the whitened transform/inverse round-trip. Inverse un-whitens by multiplying components by 1/scales = sqrt(ev) before the reconstruction GEMM."
  - "IPCA_F32_BAND pinned to Tolerance::new(1e-4, 1e-4) — the estimator adds only the small p×nc transform GEMM round-off on top of the PRIM-07 merge (Plan 07-03 observed abs 3.6e-7/rel 2.0e-6), so the merge's 1e-4 band carries to the estimator with margin (A4, to be re-measured on rocm at the phase gate)."

requirements-completed: [DECOMP-03]

# Metrics
duration: 16min
completed: 2026-06-20
---

# Phase 7 Plan 05: IncrementalPCA (DECOMP-03) Summary

**Implemented `decomposition/incremental_pca.rs` — the streaming-PCA estimator (DECOMP-03) — as a thin sklearn-faithful driver over the PRIM-07 incremental-SVD merge: `partial_fit` calls `prims::incremental_svd::merge` once per batch (consuming its `align_rows`/ddof=1/ratio finalize, no decomposition math re-derived), `fit()` resets state and loops `partial_fit` over a verbatim sklearn `gen_batches(n, batch_size, n_components)` with `batch_size = unwrap_or(5·n_features)`, and `transform`/`inverse_transform` reuse the pca.rs center+GEMM blocks with `whiten` scaling components by `1/sqrt(explained_variance_)` and un-whitening on inverse. All attributes + transform/inverse_transform match sklearn's IncrementalPCA via BOTH `partial_fit` and `fit()`, whiten on AND off — 14/14 green on cpu (f64 strict 1e-5, f32 at the 1e-4 band).**

## Performance

- **Duration:** ~16 min
- **Completed:** 2026-06-20
- **Tasks:** 2 of 2
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

### Task 1 — IncrementalPCA struct + PartialFit + sklearn-faithful fit() (commit 86a8500)
- `IncrementalPCA<F>`: holds `n_components`, `whiten: bool`, `batch_size: Option<usize>` (D-09), the running `Option<IncrementalSvdState>`, `n_features`, and a `PhantomData<F>` (the running stats are f64 in the state; the estimator is generic over the upload/compute precision `F`).
- Host accessors `components`/`explained_variance`/`explained_variance_ratio`/`singular_values`/`mean`/`var`/`n_samples_seen` materialize each attr in `F` from the f64 state; `n_samples_seen()` returns 0 before the first batch.
- `partial_fit` (PartialFit<F>): validates the batch (`n_components ≤ min(b, p)`, geometry, `n_features` agreement) BEFORE the merge (ASVS V5 / T-07-09), then calls `merge::<F>(pool, self.state.take(), x, (b, p), n_components)` — the merge already branches first-vs-subsequent, applies `align_rows`, and uses ddof=1; the estimator consumes it (D-01). Accumulates `n_samples_seen_` (carried in the state).
- `fit` (Fit<F>, sklearn-faithful D-02): rejects malformed geometry + out-of-range `n_components` + `batch_size < 1`, computes `batch_size = self.batch_size.unwrap_or(5 * n_features)` (D-03), RESETS all fitted state, then loops `partial_fit` over `gen_batches(n_samples, batch_size, self.n_components)`.
- `gen_batches`: a VERBATIM port of `sklearn.utils.gen_batches(min_batch_size)` — iterate `n // batch_size` full batches, `continue` (skip without advancing `start`) when `end + min_batch > n`, then emit a final `start..n` slice. Verified against sklearn across n=30/31/32/33 (bs=10, mb=3) and n=10 (bs=3).
- `explained_variance_ratio_` denominator = `sum(col_var)·n_total` (Pitfall 6) and `explained_variance_ = S²/(n_total−1)` (ddof=1, Pitfall 1) are both produced by the merge and surfaced unchanged.
- Gate: `cargo build -p mlrs-algos --features cpu` exit 0.

### Task 2 — transform/inverse_transform (whiten on/off) + finalize test file (commit e9eeae5)
- `transform` (Transform<F>): builds the (possibly whitened) components in `F` (whiten scales each component row by `1/sqrt(explained_variance_[i])`, D-06; a `WHITEN_VAR_FLOOR=1e-12` guard keeps the scale finite), centers X on-host by `mean_`, then `Z = X_c · components_ᵀ` via one GEMM transb (the pca.rs block, D-06).
- `inverse_transform`: un-whitens by multiplying components by `1/scales = sqrt(explained_variance_[i])` before `X̂_c = Z · components_`, then broadcasts `mean_` — the exact inverse of the whitened transform.
- Finalized `incremental_pca_test.rs`: removed all `#[ignore]`; wired the real estimator. 14 tests — `partial_fit` (explicit batch stream) + `fit()` attrs for whiten=False (f32+f64), `explained_variance_ratio` (Pitfall 6, f32+f64), `n_samples_seen` accumulation (asserts 0→10→20→30 across batches, f64), `transform` nowhiten (f32+f64) + whiten (f32+f64), `inverse_transform` nowhiten (f64) + whiten (f32+f64). Components compared AFTER `align_rows`; transform after column `align_rows`.
- `IPCA_F32_BAND = Tolerance::new(1e-4, 1e-4)` pinned from the Plan 07-03 PRIM-07 f32 measurement; f64 strict `F64_TOL`. f64 cases carry `skip_f64_with_log` (cpu runs; rocm skips, D-07).
- Gate: `cargo test -p mlrs-algos --features cpu --test incremental_pca_test` → 14 passed, 0 failed, 0 ignored. rocm test target builds.

## Deviations from Plan

### None — plan executed as written.

The plan's Task-1 `read_first` flagged a key risk (the IncrementalPCA `explained_variance_ratio_` denominator DIFFERS from pca.rs) and the Plan 07-03 SUMMARY had already encoded the correct `sum(col_var)·n_total` denominator and the ddof=1 / subsequent-batch-centering semantics inside `merge`. Because the estimator consumes the merge wholesale (D-01), both pitfalls were satisfied by construction — no Rule-1/2/3 fix was needed during execution. All 14 oracle assertions passed on the first green run.

### Other notes
- **batch_size must be explicit to match the fixture.** The committed fixtures were generated with `IncrementalPCA(n_components=3, whiten, batch_size=10).fit(X)` on a 30×6 matrix (a 3-batch stream). The estimator's `5·n_features` default (=30 for 6 features) would fit the whole matrix as ONE batch and diverge from sklearn's 3-batch approximation, so the tests construct `new(3, whiten, Some(10))`. The `5·n_features` default path is exercised by the `grep` acceptance criterion and is the sklearn `batch_size=None` semantics; it is not gated against a fixture (no `batch_size=None` oracle exists — that would need its own blob).
- **whiten attrs == unwhitened attrs.** `whiten` only rescales the transform output; the fitted `components_`/`explained_variance_`/`singular_values_`/`mean_`/`var_` are identical to the unwhitened fit. The whiten tests therefore assert BOTH the (shared) attrs and the whitened transform/inverse round-trip.

## Known Stubs

None. The estimator is fully wired to the PRIM-07 merge and the GEMM transform path; no placeholder values, no hardcoded empty data, no TODO/FIXME. The test file has zero `#[ignore]` remaining.

## Threat Flags

None. IncrementalPCA adds no new network/auth/file surface. The only trust boundary — the untrusted `(n_components, batch_size)` hyperparameters and the `(b, p)` batch geometry — is validated BEFORE any merge/launch (T-07-09 mitigated): `partial_fit`/`fit` reject `n_components` out of range (`AlgoError::InvalidNComponents`), `batch_size < 1` (`AlgoError::InvalidBatchSize`), and a malformed geometry (`PrimError::ShapeMismatch`) before the first `merge`. The stacked-SVD cap (T-07-05) is inherited from PRIM-07. No RNG, no package installs (T-07-NA holds).

## Verification

- `cargo build -p mlrs-algos --features cpu` → exit 0.
- `cargo test -p mlrs-algos --features cpu --test incremental_pca_test` → 14 passed, 0 failed, 0 ignored.
- `cargo test -p mlrs-algos --features cpu incremental_pca_partial_fit incremental_pca_fit incremental_pca_explained_variance_ratio incremental_pca_n_samples_seen` → the four named families green (f32 + f64).
- `cargo clippy -p mlrs-algos --features cpu --tests` → no warnings in incremental_pca.rs or incremental_pca_test.rs (the two pre-existing warnings are in empirical_covariance_test.rs / logistic_test.rs — out of scope).
- `cargo test -p mlrs-algos --features rocm --test incremental_pca_test --no-run` → exit 0 (rocm test target builds; f32 runs, f64 skips-with-log at the phase rocm gate).
- Acceptance greps: `impl.*PartialFit` = 1 (≥1); `incremental_svd` = 3 (≥1, consumes PRIM-07); `5.*n_features` default present = 2 (≥1); `#[ignore]` in test = 0; `whiten` in src = 25 (≥2, transform + inverse paths); incremental_pca.rs = 497 lines (≥150).

## Self-Check: PASSED

- `crates/mlrs-algos/src/decomposition/incremental_pca.rs` exists (497 lines, `IncrementalPCA` + `impl PartialFit` + `impl Fit` + `impl Transform`).
- `crates/mlrs-algos/tests/incremental_pca_test.rs` exists (14 live tests, 0 ignored).
- Commits 86a8500 (Task 1) and e9eeae5 (Task 2) present in git history.
