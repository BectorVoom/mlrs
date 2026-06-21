---
phase: 11-naive-bayes
plan: 02
subsystem: algos
tags: [naive-bayes, gaussian-nb, gather, log-sum-exp, oracle, sklearn, NB-01]

# Dependency graph
requires:
  - phase: 11-naive-bayes
    plan: 01
    provides: "nb_common GATHER (class_grouped_sum/sumsq) + log_sum_exp_normalize + empirical_class_log_prior + argmax_decode; PredictLogProba trait; GaussianNB Wave-0 stub (builder + build() validation + geometry guard); committed gaussian_nb_{f32,f64}_seed42 fixtures"
  - phase: 03-reductions
    provides: "v1 column_reduce + ScalarOp::Sum/SumSq (the only prim NB touches)"
provides:
  - "GaussianNB<F>: Fit + PredictLabels + PredictProba + PredictLogProba (NB-01) matching sklearn within 1e-5 (f64) / 1e-3 (f32 proba)"
  - "Global epsilon_ = var_smoothing · max_j Var(X[:,j]) (ddof=0) — the Pitfall-3 correct floor"
  - "Host-f64 Gaussian joint-LL evaluator shared by the three predict surfaces"
  - "Accessors: theta()/var()/class_count()/epsilon()/class_log_prior()"
affects: [11-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Host-f64 Gaussian LL: class_log_prior − 0.5·Σ_j[ln(2π·var) + (x−θ)²/var], computed on host from device-resident theta_/var_ materialized at predict (no new kernel)"
    - "Sufficient stats via the two validated GATHERs (class_grouped_sum + class_grouped_sumsq): var_ = sumsq/n − mean² (population, ddof=0), clamped ≥0 before the epsilon_ floor"
    - "GLOBAL epsilon_ computed ONCE over the full X column variances (not per-class) — Pitfall 3"

key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs
    - crates/mlrs-algos/tests/gaussian_nb_test.rs

key-decisions:
  - "theta_/var_ kept device-resident (per the Wave-0 stub field types) and materialized to host inside the shared joint_log_likelihood evaluator; the WR-07 release path needs DeviceArray buffers to release_into the pool on re-fit."
  - "var_ population variance clamped to ≥0 before the epsilon_ floor: f64 round-off on a near-constant feature can make sumsq/n − mean² dip slightly negative; the clamp keeps var_ ≥ epsilon_ > 0 (no div-by-zero / NaN in the LL)."
  - "default_matches_sklearn asserts the bare-builder fit reproduces the default fixture's predict (exact) + predict_proba (band) — the fixture carries no theta_/var_ blobs, so the litmus is the prediction surface, which is the sklearn-parity contract that matters."
  - "f32 proba band pinned at 1e-3 (measured max residual <5e-4 by bisection); f64 at the 1e-5 oracle gate. A4's 'widest f32 band' holds: GaussianNB's quadratic (x−θ)²/var amplifies f32 round-off before the log-sum-exp."

requirements-completed: [NB-01]

# Metrics
duration: 5min
completed: 2026-06-21
---

# Phase 11 Plan 02: GaussianNB (NB-01) Summary

**GaussianNB fit/predict filled on the Wave-0 seam — per-class θ/var from the validated GATHERs, the global var_smoothing epsilon_ (Pitfall 3), host-f64 Gaussian joint-LL through log_sum_exp_normalize — passing the exact-labels hard gate (f32+f64) and the proba band with rows summing to 1, leak-free across re-fit.**

## Performance

- **Duration:** ~5 min
- **Tasks:** 2
- **Files modified:** 2 (gaussian_nb.rs source + gaussian_nb_test.rs)

## Accomplishments
- Filled `GaussianNB::fit`: host distinct-sorted `classes_` (multiclass, integer-label + i32-range guards), dense `class_of_row` index, per-class `theta_`/`var_` from `nb_common::class_grouped_sum`/`class_grouped_sumsq` (`var_ = sumsq/n − mean²`, population ddof=0), the GLOBAL `epsilon_ = var_smoothing · max_j Var(X[:,j])` computed ONCE over the full X (Pitfall 3) added to every `var_` cell, and the empirical-or-supplied `class_log_prior_` (length-`n_classes` data-dependent guard → `InvalidLabels`, D-05).
- Implemented `PredictLabels` / `PredictProba` / `PredictLogProba` over a shared host-f64 `joint_log_likelihood` evaluator (`class_log_prior − 0.5·Σ_j[ln(2π·var) + (x−θ)²/var]`), normalized per-row by `nb_common::log_sum_exp_normalize`, with a geometry guard before any work and `argmax_decode` for labels.
- WR-07 re-fit release: the prior `theta_`/`var_` device buffers are `release_into(pool)`'d before storing the new ones; `refit_releases_buffers` witnesses non-increasing `live_bytes` across 4 re-fits.
- Un-ignored and filled all 7 oracle cases: `exact_labels`(+f32) HARD gate (integer equality, no band), `proba_band`(+f32) (band + every row sums to 1.0±1e-6), `default_matches_sklearn` (D-02 litmus), `build_rejects_bad_var_smoothing`, `refit_releases_buffers`. 7/7 green on cpu, 0 ignored.

## Task Commits

1. **Task 1: GaussianNB fit (global epsilon_, per-class mean/var via GATHER) + predict surface** — `1eb7aa9` (feat)
2. **Task 2: GaussianNB oracle tests — exact labels (hard), proba band + rows-sum-to-1, default-matches-sklearn, build-rejects, refit no-leak** — `96bcfaf` (test)

## Files Created/Modified
- `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs` — Filled the `todo!()` fit body; added `class_count_`/`epsilon_` fields + accessors (`theta`/`var`/`class_count`/`epsilon`); added the shared `joint_log_likelihood` evaluator and the three predict-trait impls. Imports `nb_common::{class_grouped_sum, class_grouped_sumsq, empirical_class_log_prior, log_sum_exp_normalize, argmax_decode}` and `mlrs_core::{f64_to_host, host_to_f64}`; `LN_2PI` constant for the LL factoring.
- `crates/mlrs-algos/tests/gaussian_nb_test.rs` — Un-ignored + filled all 7 cases; added `fit_gaussian` helper, `assert_band`, `assert_rows_sum_to_one`; pinned `PROBA_BAND_F32=1e-3` / `PROBA_BAND_F64=1e-5` with the A4 documentation comment.

## Decisions Made
- **theta_/var_ stay device-resident** (the Wave-0 stub's field types) and are materialized to host inside the shared `joint_log_likelihood`; the WR-07 release path needs `DeviceArray` buffers to `release_into` the pool.
- **var_ clamped ≥0 before the epsilon_ floor** — f64 round-off on a near-constant feature can make `sumsq/n − mean²` dip slightly negative; the clamp keeps `var_ ≥ epsilon_ > 0`.
- **default_matches_sklearn** asserts the bare-builder fit reproduces the default fixture's `predict` (exact) + `predict_proba` (band); the fixture carries no `theta_`/`var_` blobs, so the prediction surface is the litmus.
- **f32 proba band = 1e-3** pinned from a bisection (5e-4 still passes; actual residual <5e-4); f64 = the 1e-5 oracle gate.

## Deviations from Plan

None — plan executed exactly as written. Both task bodies, the global-epsilon_ math, the GATHER composition, the trait surface, and the test set match the plan's `<action>`/`<acceptance_criteria>`. No Rule 1–4 deviations were triggered (the var_≥0 clamp is the plan's own "var_ floored by epsilon_" correctness requirement, T-11-02-03, not an out-of-scope fix).

## Verification Evidence
- `cargo build --features cpu -p mlrs-algos` — exits 0.
- `cargo test --features cpu -p mlrs-algos --test gaussian_nb_test` — **7 passed, 0 failed, 0 ignored** (exact_labels, exact_labels_f32, proba_band, proba_band_f32, default_matches_sklearn, build_rejects_bad_var_smoothing, refit_releases_buffers).
- HARD gate: `predict_labels == sklearn predict` EXACTLY on both f32 and f64 (integer equality, no tolerance).
- `predict_proba` within band (f64 1e-5, f32 1e-3) AND every row sums to 1.0 ± 1e-6.
- `grep -n "fn predict_log_proba" gaussian_nb.rs` → match (line 452).
- Non-comment `grep -c "SharedMemory\|F::INFINITY\|Atomic"` == 0; no new `#[cube]` kernel (grep == 0).
- `epsilon_` derived from the full-X column variances (max over features), not per-class (Pitfall 3) — verified in the fit body and witnessed by the passing default/proba oracle.

## Threat Flags

None — no new network/auth/file/schema surface. The plan's threat register (T-11-02-01..04) is fully mitigated: build() rejects `var_smoothing<0` (witnessed by `build_rejects_bad_var_smoothing`), fit rejects mismatched-length priors (→`InvalidLabels`), `var_` is epsilon_-floored (no div-by-zero, host single-terminal-log), and the predict geometry guard fires before any host work.

## Self-Check: PASSED

Both modified files verified on disk (`gaussian_nb.rs`, `gaussian_nb_test.rs`); both task commits present in git history (`1eb7aa9`, `96bcfaf`).

---
*Phase: 11-naive-bayes*
*Completed: 2026-06-21*
