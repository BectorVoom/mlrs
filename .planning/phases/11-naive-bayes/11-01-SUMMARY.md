---
phase: 11-naive-bayes
plan: 01
subsystem: testing
tags: [naive-bayes, reduce, gather, log-sum-exp, builder, oracle, scaffold, sklearn]

# Dependency graph
requires:
  - phase: 10-sgd-linear-svm
    provides: "builder pattern standard (D-01/D-02), BuildError two-tier validation, mbsgd_classifier classifier-with-builder analog, gen_oracle.py per-estimator generator shape"
  - phase: 03-reductions
    provides: "v1 reduce prim (column_reduce + ScalarOp::Sum/SumSq) — the only primitive NB needs"
provides:
  - "PredictLogProba<F> trait on the shared estimator surface (D-07)"
  - "nb_common free functions: log_sum_exp_normalize, empirical_class_log_prior, argmax/argmin_decode, accuracy_score, class_grouped_sum/sumsq GATHER (D-03 — no NbBase)"
  - "Five compiling estimator stubs (Gaussian/Multinomial/Bernoulli/Complement/Categorical NB) with sklearn-default builders, build() data-independent validation, real geometry guards, todo!() fit bodies"
  - "MinCategories::{Infer,Uniform,PerFeature} enum (D-04)"
  - "BuildError::{InvalidVarSmoothing,InvalidClassPrior,InvalidMinCategories} + AlgoError::InvalidCategoricalInput"
  - "Ten committed .npz fixtures (5 variants x f32/f64) + five #[ignore] oracle test scaffolds"
  - "File-disjoint Wave gate: Waves 1/2 estimator plans never touch the same file"
affects: [11-02, 11-03, 11-04, 11-05]

# Tech tracking
tech-stack:
  added: ["log (workspace dep added to mlrs-algos for the D-06 force_alpha clip warning)"]
  patterns:
    - "GATHER via reduce-prim composition (class_grouped_sum/sumsq) — one owner per (class,feature), NO new #[cube] kernel, NO SharedMemory/atomics/F::INFINITY"
    - "Host f64 single-terminal-log log-sum-exp (Pattern 3, cpu-MLIR-safe)"
    - "Shared validate_discrete_alpha free fn (D-06 force_alpha clip) reused by the four discrete builders WITHOUT a base struct (D-03)"

key-files:
  created:
    - crates/mlrs-algos/src/naive_bayes/mod.rs
    - crates/mlrs-algos/src/naive_bayes/nb_common.rs
    - crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs
    - crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs
    - crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs
    - crates/mlrs-algos/src/naive_bayes/complement_nb.rs
    - crates/mlrs-algos/src/naive_bayes/categorical_nb.rs
    - crates/mlrs-algos/tests/nb_common_test.rs
    - "crates/mlrs-algos/tests/{gaussian,multinomial,bernoulli,complement,categorical}_nb_test.rs"
    - "tests/fixtures/{gaussian,multinomial,bernoulli,complement,categorical}_nb_{f32,f64}_seed42.npz (10)"
  modified:
    - crates/mlrs-algos/src/traits.rs
    - crates/mlrs-algos/src/error.rs
    - crates/mlrs-algos/src/lib.rs
    - crates/mlrs-algos/Cargo.toml
    - scripts/gen_oracle.py

key-decisions:
  - "A5 RESOLVED: per-axis ScalarOp::SumSq IS exposed by the reduce prim (reduce.rs:260-301), so class_grouped_sumsq composes column_reduce(SumSq) directly — no squared-host-copy needed for GaussianNB var_."
  - "log added as a direct mlrs-algos dependency (Rule 3): the D-06 force_alpha clip warning needs log::warn!, but mlrs-algos did not depend on log (only mlrs-backend did)."
  - "validate_discrete_alpha lives in multinomial_nb.rs as pub(crate) and is reused by the three sibling discrete builders — function-level sharing (D-03), no shared base struct."
  - "NB geometry: 3 classes, N_SAMPLES=39 (40//3*3), N_QUERY=6 (8//3*3) — integer-divisible per-class blocks."

patterns-established:
  - "GATHER substrate: class_grouped_sum/sumsq compose the v1 reduce prim over host-grouped per-class row blocks; release_into each scratch (WR-07). Wave 1 estimators call these — never a scatter-add."
  - "Wave-0 scaffold precedent (mirrors 10-01/09-01/08-01): front-load all shared-file edits + compiling stubs + fixtures + #[ignore] scaffolds so later waves are file-disjoint."

requirements-completed: [NB-01, NB-02, NB-03, NB-04, NB-05]

# Metrics
duration: 25min
completed: 2026-06-21
---

# Phase 11 Plan 01: Naive Bayes Wave-0 Shared Seam Summary

**PredictLogProba trait + nb_common GATHER/log-sum-exp free functions + five sklearn-default-builder estimator stubs + ten committed oracle fixtures, landing the file-disjoint Wave gate for the five Naive Bayes plans with zero new device kernels.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-06-21T21:30Z (approx)
- **Completed:** 2026-06-21T21:55Z
- **Tasks:** 3
- **Files modified:** 22 (5 modified + 7 source created + 6 test files + 10 fixtures, with overlap)

## Accomplishments
- Landed the shared seam: `PredictLogProba<F>` trait (D-07), three `BuildError` variants + `AlgoError::InvalidCategoricalInput`, the `naive_bayes` module index (D-03 free-function sharing, NO `NbBase`), and `MinCategories` enum (D-04).
- Stood up `nb_common` once and standalone-validated it on cpu: the per-row `log_sum_exp_normalize` (proba sums to 1, no overflow on large LLs), `empirical_class_log_prior`, `argmax/argmin_decode` (lowest-index tie-break), `accuracy_score`, and the `class_grouped_sum`/`class_grouped_sumsq` GATHER helpers (reduce-prim composition, cpu launch witnessed). 9/9 nb_common tests green.
- Five estimator stubs compile with sklearn-default builders, `build()` data-independent validation (incl. the D-06 `force_alpha` clip+warn), real geometry guards, and `todo!()` fit bodies — Waves 1/2 are now file-disjoint.
- Ten `.npz` fixtures (5 variants x f32/f64) regenerated from default-constructor sklearn fits and committed; five `#[ignore]` oracle test scaffolds collect (6/6/7/7/8).

## Task Commits

1. **Task 1: Shared-file edits — PredictLogProba, NB error variants, MinCategories, module index, lib.rs wiring** - `26c2b5e` (feat)
2. **Task 2: nb_common free functions (incl. class_grouped_sum GATHER) standalone-validated + five compiling stubs** - `7b44257` (test; the stubs landed in the Task-1 commit since mod.rs requires them to compile)
3. **Task 3: gen_oracle.py NB generators + ten committed fixtures + five #[ignore] oracle scaffolds** - `54cf1c7` (test)

_Note: the five estimator stubs + nb_common.rs were committed in `26c2b5e` (Task 1) because `mod.rs`'s `pub mod`/re-exports require the type definitions to exist for the crate to compile; Task 2's commit `7b44257` is the nb_common standalone validation test._

## Files Created/Modified
- `crates/mlrs-algos/src/traits.rs` - Added `PredictLogProba<F>` trait (joint_ll − logsumexp; sibling of PredictProba).
- `crates/mlrs-algos/src/error.rs` - Added `BuildError::{InvalidVarSmoothing,InvalidClassPrior,InvalidMinCategories}` + `AlgoError::InvalidCategoricalInput`.
- `crates/mlrs-algos/src/lib.rs` - Wired `pub mod naive_bayes`; re-exported `PredictLogProba`.
- `crates/mlrs-algos/Cargo.toml` - Added `log` workspace dep (D-06 clip warning).
- `crates/mlrs-algos/src/naive_bayes/mod.rs` - Module index (D-03 doc, re-exports five structs + MinCategories).
- `crates/mlrs-algos/src/naive_bayes/nb_common.rs` - The shared free functions + GATHER helpers.
- `crates/mlrs-algos/src/naive_bayes/{gaussian,multinomial,bernoulli,complement,categorical}_nb.rs` - Five estimator stubs.
- `crates/mlrs-algos/tests/nb_common_test.rs` - 9 standalone + GATHER-launch-witness tests.
- `crates/mlrs-algos/tests/{gaussian,multinomial,bernoulli,complement,categorical}_nb_test.rs` - Five #[ignore] oracle scaffolds.
- `scripts/gen_oracle.py` - Five NB generators + `_nb_count_blobs`/`_nb_categorical_blobs`/`_save_nb` helpers + main() registration.
- `tests/fixtures/*_nb_{f32,f64}_seed42.npz` - Ten committed oracle blobs.

## Decisions Made
- **A5 resolved:** the reduce prim already exposes per-axis `ScalarOp::SumSq` (reduce.rs:260-301), so `class_grouped_sumsq` composes `column_reduce(SumSq)` directly — no squared-host-copy path needed for GaussianNB variance.
- **`log` added to mlrs-algos** (Rule 3 deviation): the D-06 `force_alpha` clip warning requires `log::warn!`, which mlrs-algos did not depend on (only mlrs-backend did). Added it as a workspace dep.
- **`validate_discrete_alpha` shared free fn:** lives `pub(crate)` in multinomial_nb.rs and is reused by the bernoulli/complement/categorical builders — function-level sharing per D-03, no base struct.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `log` dependency to mlrs-algos**
- **Found during:** Task 1 (the D-06 force_alpha clip+warn in the discrete builders)
- **Issue:** The plan/patterns specify a `log::warn!` for the force_alpha clip, but `mlrs-algos` did not list `log` as a dependency (only `mlrs-backend` did) → `E0433: cannot find module log`.
- **Fix:** Added `log = { workspace = true }` to `crates/mlrs-algos/Cargo.toml` (the workspace already pins log 0.4; the reduce prim uses the same convention).
- **Files modified:** crates/mlrs-algos/Cargo.toml, Cargo.lock
- **Verification:** `cargo build --features cpu -p mlrs-algos` exits 0.
- **Committed in:** `26c2b5e` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The `log` dep is required to emit the sklearn-parity force_alpha warning per D-06; parity depends only on the clipped numeric (A2), not the text. No scope creep.

## Issues Encountered
- The plan's Task-1/Task-2 boundary is logically interleaved: `mod.rs` (Task 1) declares `pub mod gaussian_nb; …` and re-exports the structs, so the estimator stub files and `nb_common.rs` (nominally Task 2 content) must exist for Task 1 to compile. Resolved by landing the full stubs + nb_common in the Task-1 commit and committing the nb_common standalone-validation TEST as the Task-2 commit. All acceptance criteria for both tasks are satisfied.
- Pre-existing `clippy::approx_constant` error in `mlrs-kernels/src/elementwise.rs:282` (FRAC_PI_2) surfaces when running `cargo clippy -p mlrs-algos` (mlrs-kernels is a build dep). Out of scope — already logged in `.planning/phases/10-sgd-linear-svm/deferred-items.md`. No clippy warnings point into the new `naive_bayes` source.

## Verification Evidence
- `cargo build --features cpu -p mlrs-algos` — exits 0 (shared seam + five stubs compile).
- `cargo test --features cpu -p mlrs-algos --test nb_common_test` — 9 passed, 0 failed (free functions + class_grouped_sum/sumsq GATHER cpu-launch witness + empty-class case).
- `cargo build --features cpu -p mlrs-algos --tests` — exits 0 (all five oracle test files compile).
- `--test {variant}_nb_test -- --list` — collects 6/6/7/7/8 tests across the five files.
- `ls tests/fixtures/*_nb_{f32,f64}_seed42.npz | wc -l` == 10.
- `grep "def gen_*_nb" scripts/gen_oracle.py` == 5; `skip_f64_with_log` present in all five test files.
- No new `#[cube]` kernel in naive_bayes (grep matches are doc comments only); nb_common grep clean of SharedMemory/F::INFINITY/Atomic (non-comment count == 0).

## Next Phase Readiness
- Wave gate satisfied: 11-02 (GaussianNB), 11-03 (Multinomial/Bernoulli/Complement), 11-04 (Categorical), 11-05 (PyO3 PY-06) each edit only their own files.
- Wave-1 estimator plans inherit: the validated `nb_common` GATHER + log-sum-exp helpers, the sklearn-default builders + build() validation, the `PredictLogProba`/`PredictProba`/`PredictLabels` trait surface, the committed fixtures, and the `#[ignore]` scaffolds to un-ignore and fill.

## Self-Check: PASSED

All created files verified on disk (mod.rs, nb_common.rs, gaussian_nb.rs, nb_common_test.rs, categorical_nb_f64_seed42.npz, 11-01-SUMMARY.md) and all three task commits present in git history (26c2b5e, 7b44257, 54cf1c7).

---
*Phase: 11-naive-bayes*
*Completed: 2026-06-21*
