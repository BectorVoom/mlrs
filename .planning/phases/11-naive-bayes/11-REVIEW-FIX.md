---
phase: 11-naive-bayes
fixed_at: 2026-06-22T00:00:00Z
review_path: .planning/phases/11-naive-bayes/11-REVIEW.md
iteration: 1
findings_in_scope: 15
fixed: 13
skipped: 2
status: partial
---

# Phase 11: Code Review Fix Report

**Fixed at:** 2026-06-22
**Source review:** .planning/phases/11-naive-bayes/11-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 15 (fix_scope = all — CR + WR + IN)
- Fixed: 13
- Skipped: 2 (IN-04 deliberate, IN-05 no-op)

**Verification:** All edits type-checked via `cargo check -p mlrs-algos --features cpu`
and `cargo check -p mlrs-py --features cpu`. The full NB algos test set
(gaussian / multinomial / bernoulli / complement / categorical / nb_common —
49 tests) passes after the fixes, confirming the existing oracle band tests
still hold for valid inputs. The mlrs-py Python tests (CR-02) cannot run live in
this environment (no maturin/pyarrow per the project memory note) and are routed
to UAT; the Rust-side guard they exercise is gated by the algos tests + the
`fit_rejects_bad_input` categorical test.

## Fixed Issues

### CR-01: Discrete NB variants accept negative / NaN feature values → silent NaN predictions

**Files modified:** `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs`, `crates/mlrs-algos/src/naive_bayes/complement_nb.rs`, `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs`
**Commit:** b4f4f98
**Applied fix:** Added a shared `validate_non_negative_counts(estimator, x_host)`
helper (in `multinomial_nb.rs`, `pub(crate)`) returning
`AlgoError::InvalidLabels` for any non-finite or negative value, mirroring
sklearn's `check_non_negative`. Wired it into MultinomialNB / ComplementNB /
BernoulliNB at BOTH `fit` (before the GATHER reaches the smoothed-log formulas)
and `joint_log_likelihood` (before the predict GEMM). For BernoulliNB the raw
input is validated before binarization.

### CR-02: PyO3 ingress for the count-based NB variants does not reject negative input

**Files modified:** `crates/mlrs-py/tests/test_naive_bayes.py`
**Commit:** 0cd63c9
**Applied fix:** No PyO3 source change was required — `algo_err_to_py` already
maps `AlgoError` → `PyValueError`, so the CR-01 algos-layer guard surfaces as a
Python `ValueError` automatically. Added two parametrized FFI tests asserting
MultinomialNB / BernoulliNB / ComplementNB raise `ValueError` on a negative
matrix at `fit` and on a negative query row at `predict`.

### WR-01: `class_prior` / `priors` sum-to-one not validated

**Files modified:** `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs`, `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs`
**Commit:** 9e1d373 (GaussianNB), b4f4f98 (discrete `resolve_class_log_prior`)
**Applied fix:** After the existing length check, both the GaussianNB explicit-
priors path and the shared discrete `resolve_class_log_prior` now reject a prior
vector whose sum deviates from 1 by more than 1e-6, returning
`AlgoError::InvalidLabels` with sklearn's "the sum of the priors should be 1"
message. The discrete resolver fix is shared by Multinomial / Bernoulli /
Complement / Categorical.

### WR-02: Device-buffer leak on the error path of `grouped_reduce`

**Files modified:** `crates/mlrs-algos/src/naive_bayes/nb_common.rs`
**Commit:** 0b77034
**Applied fix:** Replaced `column_reduce(...)?.expect(...)` with an explicit
`match` that calls `block_dev.release_into(pool)` on every arm (Ok / None / Err)
before propagating, so a reduce failure no longer leaks the per-class scratch.

### WR-03: Device-buffer leak on error path in BernoulliNB fit

**Files modified:** `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs`
**Commit:** b4f4f98
**Applied fix:** Matched on the `class_grouped_sum` result and released
`x_bin_dev` on both the Ok and Err arms before propagating the error.

### WR-04: Python dtype mismatch reports misleading "not fitted"

**Files modified:** `crates/mlrs-py/src/errors.rs`, `crates/mlrs-py/src/estimators/naive_bayes.rs`
**Commit:** c7c0f17
**Applied fix:** Added `errors::dtype_mismatch(estimator, requested, fitted)`
(a `PyValueError` naming the actual fitted dtype) and split the catch-all `_`
arms in the f32/f64 predict_proba / predict_log_proba surface so the wrong-dtype
arm returns the dtype-mismatch error while the genuinely-unfit `Unfit` arm keeps
`not_fitted`.

### WR-05: GaussianNB epsilon_ can be 0 → divide-by-zero

**Files modified:** `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs`
**Commit:** 9e1d373
**Applied fix:** `epsilon_ = (var_smoothing * max_col_var).max(f64::MIN_POSITIVE)`
floors the global variance epsilon to a tiny positive minimum so an all-constant
feature matrix cannot leave a `var_` cell at 0.

### WR-06: CategoricalNB silently treats out-of-range predict categories as unseen

**Files modified:** `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs`
**Commit:** 3a9d019
**Applied fix:** Chose option (a) from the review — a predict-time category index
that is negative, non-integer, or `>= n_categories_j` now returns
`AlgoError::InvalidCategoricalInput` (matching sklearn and the error variant's
documented purpose) instead of the silent smoothed-fallback. The now-unused
`class_count`/`alpha` bindings were cleaned up (`class_count_` kept as a `_`-bound
fitted-state guard).
**NOTE — requires human verification:** This is a behavioral/logic change
(silent fallback → hard error). The existing `categorical_nb_test.rs` suite
(including `fit_rejects_bad_input`) passes, but the developer should confirm no
intended-divergence consumer relied on the old silent fallback.

### WR-07: `accuracy_score` returns `0.0` for empty input

**Files modified:** `crates/mlrs-algos/src/naive_bayes/nb_common.rs`
**Commit:** 0b77034
**Applied fix:** Returns `f64::NAN` (sklearn-like "undefined") for an empty
prediction vector instead of `0.0`; doc updated.

### WR-08: `force_alpha` field dead across four variants

**Files modified:** `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs`, `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs`, `crates/mlrs-algos/src/naive_bayes/complement_nb.rs`, `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs`
**Commit:** b4f4f98 (Multinomial/Bernoulli/Complement), 3a9d019 (Categorical)
**Applied fix:** Added a `pub fn force_alpha(&self) -> bool` accessor to each of
the four variants and removed the `#[allow(dead_code)]` attribute (and the
Categorical `let _ = self.force_alpha;` suppression), so the compiler's dead-code
check is genuinely active and a future re-fit-clip omission would be flagged.

### IN-01: `log_sum_exp_normalize` doc overstates underflow protection

**Files modified:** `crates/mlrs-algos/src/naive_bayes/nb_common.rs`
**Commit:** 0b77034
**Applied fix:** Clarified the doc that only `log_proba` is underflow-safe;
`proba = exp(log_proba)` may still flush small values to 0 (expected behavior).

### IN-02: Magic tolerance `1e-6` for integer-label rounding duplicated

**Files modified:** `crates/mlrs-algos/src/naive_bayes/nb_common.rs`, `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs`, `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs`, `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs`
**Commit:** 0b77034 (const), b4f4f98 / 9e1d373 / 3a9d019 (usages)
**Applied fix:** Added `pub const NB_LABEL_INT_TOL: f64 = 1e-6;` in `nb_common`
and replaced the four hardcoded `1e-6` integer-round-trip literals (gaussian_nb
label decode, discrete `decode_classes`, categorical fit + predict category
checks) with it.

### IN-03: `decode_classes` error `estimator` field is the literal `"discrete_nb"`

**Files modified:** `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs`, `crates/mlrs-algos/src/naive_bayes/complement_nb.rs`, `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs`, `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs`
**Commit:** b4f4f98 (helper + Multinomial/Complement/Bernoulli), 3a9d019 (Categorical)
**Applied fix:** `decode_classes` now takes an `estimator: &'static str` first
parameter and uses it in its `InvalidLabels` errors; all four call sites pass
their concrete estimator name (`"multinomial_nb"` / `"complement_nb"` /
`"bernoulli_nb"` / `"categorical_nb"`).

## Skipped Issues

### IN-04: Duplicated `host_to_f64` / `f64_to` helpers in every test file

**File:** `crates/mlrs-algos/tests/gaussian_nb_test.rs:63-77` (and siblings)
**Reason:** skipped — deliberate. These helpers are duplicated across the ENTIRE
algos test suite (30+ `tests/*.rs` files), not just the NB files; the
per-test-file self-containment is the established project convention. Extracting
a shared test-support `mod` would be a project-wide cross-cutting refactor
(shared file + `mod` includes across every integration test) that exceeds the
scope of a targeted review fix and risks the established test gate. The review
itself flags this as "Test-only, low priority." Left to a dedicated test-support
refactor task.
**Original issue:** Each NB test file re-defines identical helper functions; a
shared module would reduce drift risk.

### IN-05: ComplementNB `class_log_prior_` computed but only used in single-class edge case

**File:** `crates/mlrs-algos/src/naive_bayes/complement_nb.rs:250-256,322-332`
**Reason:** skipped — no action required. The review explicitly states "No change
required; documented for reviewer awareness" and notes the behavior matches
sklearn (the prior is consumed by the accessor and the single-class branch).
**Original issue:** `class_log_prior_` is resolved unconditionally at fit but
consumed only in the `n_classes == 1` predict branch — matches sklearn, noted for
awareness only.

---

_Fixed: 2026-06-22_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
