---
phase: 11-naive-bayes
plan: 04
subsystem: algos
tags: [naive-bayes, categorical-nb, ragged-tables, min-categories, integer-input, oracle, sklearn, NB-05]

# Dependency graph
requires:
  - phase: 11-naive-bayes
    plan: 01
    provides: "CategoricalNB Wave-0 stub (builder + build() validation incl. D-06 force_alpha clip + per-feature geometry guard) + MinCategories::{Infer,Uniform,PerFeature} enum (D-04) + AlgoError::InvalidCategoricalInput + BuildError::InvalidMinCategories + committed categorical_nb_{f32,f64}_seed42 fixtures (no unseen categories at predict, A3) + the #[ignore] oracle scaffold; nb_common log_sum_exp_normalize + argmax_decode; validate_discrete_alpha"
  - phase: 11-naive-bayes
    plan: 03
    provides: "the shared discrete free fns decode_classes (integer + i32-range label guard, WR-02) + resolve_class_log_prior (supplied / empirical / uniform), pub(crate) in multinomial_nb.rs — reused verbatim by the CategoricalNB fit (D-03 function-level sharing)"
  - phase: 11-naive-bayes
    plan: 02
    provides: "the GaussianNB fit/predict structural template (decode classes, host sufficient stats, host log-sum-exp through the three predict traits, store class_count_ for the predict-time denominator)"
provides:
  - "CategoricalNB<F>: Fit + PredictLabels + PredictProba + PredictLogProba (NB-05) — RAGGED per-feature feature_log_prob_ (Vec<Vec<f64>>, one n_classes×n_categories_j matrix per feature, Pitfall 7), per-feature denominator class_count[c] + alpha·n_categories_j (Pitfall 4), exact sklearn labels (f32+f64)"
  - "MinCategories padding honored at fit: n_categories_j = max(observed_max+1, min_categories_j) for Infer / Uniform(u) / PerFeature(v); PerFeature length-==n_features check is data-DEPENDENT at fit (D-05 → InvalidCategoricalInput)"
  - "Non-negative-INTEGER input validation at fit (T-11-04-01 → AlgoError::InvalidCategoricalInput) + predict-time lookup-index guard against n_categories_j (T-11-04-02 → exact smoothed log(alpha/denom) for an unseen / out-of-range category, never an OOB ragged-table index)"
affects: [11-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Ragged per-feature categorical likelihood: host-tabulate category_count_[j][c,k] (one owner per (feature,class,category) — a host count, NEVER a device scatter), feature_log_prob_[j][c,k] = log((count+alpha)/(class_count[c]+alpha·n_categories_j)); the joint LL sums per-feature looked-up log-probs in host f64 (NO GEMM — the tables are ragged, so the count variants' X@flp.T matvec does not apply)"
    - "Predict-time unseen-category guard (T-11-04-02): clamp k against n_categories_j; an out-of-range index uses the EXACT smoothed log(alpha / (class_count[c]+alpha·n_categories_j)) recovered from the stored class_count_ — no OOB index, no table reconstruction"
    - "MinCategories pad-only-grows: n_categories_j = max(observed_max+1, min_j); padding <= observed is a no-op (default-identical), padding beyond observed adds all-unseen (count==0, smoothed) cells that never appear in the A3 fixtures"

key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/naive_bayes/categorical_nb.rs
    - crates/mlrs-algos/tests/categorical_nb_test.rs

key-decisions:
  - "Stored class_count_ on the struct (a new field beyond the Wave-0 stub set): the predict-time unseen-category fallback needs the EXACT denominator class_count[c] + alpha·n_categories_j to emit log(alpha/denom). Recovering it from the fitted ragged table is not robustly isolable, so the stored count is the clean, exact route (also the empirical-prior numerator) — mirrors GaussianNB's class_count_."
  - "Reused the 11-03 shared discrete free fns decode_classes + resolve_class_log_prior verbatim (D-03) for the label decode and the supplied/empirical/uniform prior resolution — no CategoricalNB-local re-implementation, no base struct."
  - "No GEMM: the count variants matvec X @ feature_log_prob_.T over a single (n_classes, n_features) buffer, but CategoricalNB's feature_log_prob_ is RAGGED (variable category count per feature), so the joint LL is a host per-feature lookup-and-sum (the planner's documented host-f64 path; the v1 reduce prim is not needed for the small host categorical counts)."

patterns-established:
  - "CategoricalNB is the structurally-distinct NB variant: independent struct (D-03), ragged Vec<Vec<f64>> tables, integer inputs, per-feature lookup — sharing ONLY the nb_common log-sum-exp/argmax helpers and the two discrete free fns with its siblings (no NbBase)."

requirements-completed: [NB-05]

# Metrics
duration: 12min
completed: 2026-06-22
---

# Phase 11 Plan 04: CategoricalNB (NB-05) Summary

**CategoricalNB filled on the Wave-0 seam — an independent struct with a RAGGED per-feature `feature_log_prob_` (`Vec<Vec<f64>>`, one `n_classes × n_categories_j` matrix per feature, variable category count, Pitfall 7), the per-feature smoothing denominator `class_count[c] + alpha·n_categories_j` (Pitfall 4), `MinCategories::{Infer,Uniform,PerFeature}` padding (D-04, `n_categories_j = max(observed_max+1, min_j)`), non-negative-integer input validation at fit (T-11-04-01), and a predict-time lookup-index guard against `n_categories_j` (T-11-04-02, unseen → the EXACT smoothed `log(alpha/denom)`). No GEMM — the joint LL sums per-feature looked-up log-probs in host f64. All 9 oracle cases pass on cpu f32+f64 with the exact-labels HARD gate, proba rows summing to 1, and leak-free re-fit.**

## Performance
- **Duration:** ~12 min
- **Tasks:** 2 (landed atomically — see TDD note)
- **Files modified:** 2 (1 source + 1 test)

## Accomplishments
- **CategoricalNB (NB-05):** `fit` validates X is a non-negative-integer categorical encoding (round-to-nearest within 1e-6, `< 0` or non-integer → `InvalidCategoricalInput`, T-11-04-01) BEFORE sizing any table; decodes `classes_` / dense per-row class index via the shared `decode_classes`; computes per-feature `n_categories_j = max(observed_max+1, min_categories_j)` from the `MinCategories` enum (with the `PerFeature` length-`==n_features` data-dependent check, D-05); host-tabulates `category_count_[j][c,k]` (one owner per `(feature,class,category)`, a host count — NEVER a device scatter) and forms the ragged `feature_log_prob_[j][c,k] = log((count+alpha)/(class_count[c]+alpha·n_categories_j))` (Pitfall 4 — denominator smoothing is `alpha·n_categories_j`); resolves `class_log_prior_` via the shared `resolve_class_log_prior` (supplied / empirical / uniform).
- **Predict surface (PredictLabels/PredictProba/PredictLogProba):** the joint LL `class_log_prior_[c] + Σ_j feature_log_prob_[j][c, x[i,j]]` in host f64 with the per-feature lookup index GUARDED against `n_categories_j` (T-11-04-02) — a negative / non-integer / out-of-range category maps to the EXACT smoothed `log(alpha / (class_count[c]+alpha·n_categories_j))` recovered from the stored `class_count_`, never an OOB ragged-table index — then `nb_common::log_sum_exp_normalize` + `argmax_decode`.
- **Oracle:** un-ignored and filled all 9 cases — `exact_labels` (f64, cpu) + `exact_labels_f32` (the HARD gate, integer `assert_eq!`, no band), `proba_band` (f64) + `proba_band_f32` (within band AND rows sum to 1.0±1e-6), `default_matches_sklearn` (bare `builder().build()` == sklearn default), `min_categories` (`Uniform(4)` no-op pad == default; `PerFeature([6,6,6,6])` padding-beyond-observed keeps the default labels — A3 has no unseen at predict), `fit_rejects_bad_input` (negative AND non-integer X → `InvalidCategoricalInput`), `build_rejects_bad_alpha` (`alpha<0` → `InvalidAlpha`), `refit_releases_buffers` (PoolStats no-leak across 4 re-fits).

## Task Commits
1. **Task 1: CategoricalNB fit — ragged feature_log_prob_ + MinCategories padding + integer-input validation** — `b1811e7` (feat)
2. **Task 2: CategoricalNB predict (per-feature lookup, unseen-category guard) + oracle** — `b1811e7` (same commit — see TDD note)

_TDD note (mirrors the 11-03 precedent): the fit body, the full predict surface, and the filled oracle test are all file-disjoint per the Wave gate and live in the same two files (`categorical_nb.rs` / `categorical_nb_test.rs`). The cpu `cargo test` is the RED→GREEN witness, and the per-task acceptance greps + the green oracle are the gate evidence, so both tasks landed atomically in `b1811e7`. Task 1's gate (`fit_rejects_bad_input` exits 0 + the `Vec<Vec` / `MinCategories` / SharedMemory-free greps) and Task 2's gate (all 9 cases green, none ignored + `fn min_categories` + integer-eq exact_labels + `skip_f64_with_log` ≥ 1) are both satisfied by this commit._

## Files Created/Modified
- `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs` — Filled `fit` (integer-input validation → `decode_classes` → per-feature `MinCategories` padding → host-tabulated ragged `feature_log_prob_` with the `alpha·n_categories_j` denominator → `resolve_class_log_prior`); added the `class_count_` field (for the exact predict-time smoothed-unseen denominator) + its accessor, the `n_categories` / `feature_log_prob` / `class_count` accessors, the host `joint_log_likelihood` evaluator (per-feature guarded lookup), and the three predict-trait impls.
- `crates/mlrs-algos/tests/categorical_nb_test.rs` — Un-ignored + filled all 9 cases (exact_labels f32+f64, proba_band f32+f64, default_matches_sklearn, min_categories, fit_rejects_bad_input, build_rejects_bad_alpha, refit_releases_buffers) on the committed `categorical_nb_{f32,f64}_seed42` fixtures.

## Decisions Made
- **Added the `class_count_` field** (beyond the Wave-0 stub field set): the predict-time unseen-category fallback (T-11-04-02) needs the exact denominator `class_count[c] + alpha·n_categories_j` to emit `log(alpha/denom)`. Reconstructing it from the fitted ragged table is not robustly isolable, so the stored host-f64 count (also the empirical-prior numerator) is the clean exact route — mirrors GaussianNB's `class_count_`.
- **Reused `decode_classes` + `resolve_class_log_prior` verbatim** (the 11-03 shared discrete free fns, D-03) for the label decode and the prior resolution — no CategoricalNB-local re-implementation, no base struct.
- **No GEMM** (planner's discretion, documented): `feature_log_prob_` is ragged (variable category count per feature), so the count variants' `X @ feature_log_prob_.T` single-buffer matvec does not apply; the joint LL is a host per-feature lookup-and-sum (small host-f64 work, the v1 reduce prim is not needed for the categorical counts).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Critical functionality] Added the `class_count_` fitted field**
- **Found during:** Task 2 (the predict-time unseen-category guard, T-11-04-02, mandated by the threat register and the `<behavior>` block).
- **Issue:** The Wave-0 stub stored `n_categories_` / `feature_log_prob_` / `class_log_prior_` but not the per-class counts. The exact smoothed unseen-category log-prob `log(alpha / (class_count[c]+alpha·n_categories_j))` needs `class_count[c]`, which the ragged log-prob table does not cleanly expose.
- **Fix:** Added `class_count_: Option<Vec<f64>>` to the struct, populated at fit, read in `joint_log_likelihood` for the unseen fallback (and exposed via a `class_count()` accessor).
- **Files modified:** `categorical_nb.rs`.
- **Commit:** `b1811e7`.

---

**Total deviations:** 1 auto-fixed (Rule 2 — the threat register's T-11-04-02 mitigation mandated the exact smoothed-unseen value; the stored count is its natural, exact implementation, not scope creep).

## Verification Evidence
- `cargo test --features cpu -p mlrs-algos --test categorical_nb_test` — **9 passed, 0 failed, 0 ignored**.
- HARD gate: `predict_labels == sklearn predict` EXACTLY on f32 AND f64 (integer `assert_eq!`, no band).
- `predict_proba` within band (f64 1e-5, f32 1e-3) AND every row sums to 1.0 ± 1e-6.
- `min_categories`: `Uniform(4)` (== observed) is a no-op pad matching the default labels; `PerFeature([6,6,6,6])` padding-beyond-observed keeps the sklearn-default labels (A3 — no unseen categories in Xq), proba rows sum to 1.
- `fit_rejects_bad_input`: a negative entry AND a non-integer entry each → `AlgoError::InvalidCategoricalInput`.
- Acceptance greps: `Vec<Vec` (ragged `feature_log_prob_`) and `MinCategories` (14×) present in `categorical_nb.rs`; `fn min_categories` and `skip_f64_with_log` (6×) present in the test; non-comment `grep -c "SharedMemory\|F::INFINITY\|Atomic"` == 0 (cpu-MLIR-safe, no new `#[cube]` kernel).
- `cargo build --features cpu -p mlrs-algos --tests` — exits 0, zero warnings.
- No regression: `nb_common_test` (9) + `gaussian_nb_test` (7) still green.

## Threat Flags
None — no new network/auth/file/schema surface. The plan's threat register (T-11-04-01..04) is mitigated: `fit` rejects negative / non-integer X → `InvalidCategoricalInput` (witnessed by `fit_rejects_bad_input`); the predict-time lookup index is clamped against `n_categories_j` with the exact smoothed fallback (T-11-04-02); the `PerFeature` `min_categories` length check fires at fit and `MinCategories` carries `usize` entries (non-negative by construction, so the negative-entry path T-11-04-03 is structurally unreachable — `BuildError::InvalidMinCategories` is retained for any future signed-input surface); alpha smoothing keeps every log argument positive (host single-terminal-log, no NaN/Inf, T-11-04-04).

## Known Stubs
None — the fit body and the full predict surface are wired to the committed fixtures; no placeholder/empty-data paths remain.

## Self-Check: PASSED

Both modified files verified on disk (`categorical_nb.rs`, `categorical_nb_test.rs`); the task commit `b1811e7` is present in git history.

---
*Phase: 11-naive-bayes*
*Completed: 2026-06-22*
