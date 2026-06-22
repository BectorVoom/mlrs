---
phase: 11-naive-bayes
reviewed: 2026-06-22T00:00:00Z
depth: standard
files_reviewed: 23
files_reviewed_list:
  - crates/mlrs-algos/Cargo.toml
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/lib.rs
  - crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs
  - crates/mlrs-algos/src/naive_bayes/categorical_nb.rs
  - crates/mlrs-algos/src/naive_bayes/complement_nb.rs
  - crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs
  - crates/mlrs-algos/src/naive_bayes/mod.rs
  - crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs
  - crates/mlrs-algos/src/naive_bayes/nb_common.rs
  - crates/mlrs-algos/src/traits.rs
  - crates/mlrs-algos/tests/bernoulli_nb_test.rs
  - crates/mlrs-algos/tests/categorical_nb_test.rs
  - crates/mlrs-algos/tests/complement_nb_test.rs
  - crates/mlrs-algos/tests/gaussian_nb_test.rs
  - crates/mlrs-algos/tests/multinomial_nb_test.rs
  - crates/mlrs-algos/tests/nb_common_test.rs
  - crates/mlrs-py/src/estimators/mod.rs
  - crates/mlrs-py/src/estimators/naive_bayes.rs
  - crates/mlrs-py/src/lib.rs
  - crates/mlrs-py/tests/pyclass_smoke_test.rs
  - crates/mlrs-py/tests/test_naive_bayes.py
  - scripts/gen_oracle.py
findings:
  critical: 2
  warning: 8
  info: 5
  total: 15
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-06-22
**Depth:** standard
**Files Reviewed:** 23
**Status:** issues_found

## Summary

Reviewed the five Phase-11 Naive Bayes estimators (`GaussianNB` / `MultinomialNB`
/ `BernoulliNB` / `ComplementNB` / `CategoricalNB`), the shared `nb_common` free
functions, the `AlgoError` / `BuildError` surface, the PyO3 wrappers, and the
oracle/smoke tests.

The numerical formulas largely follow sklearn and are gated by oracle tests with
a tight band, which is a real strength. However, the adversarial pass surfaced
**two correctness/robustness BLOCKERs** rooted in *missing input validation on
the discrete count-based variants*: negative / NaN inputs (which sklearn rejects
with `check_non_negative`) flow unguarded into `ln()` and produce silent `NaN`
joint-log-likelihoods, corrupting `predict`/`predict_proba` without any error.
Both the algos layer and the PyO3 ingress accept these. Several WARNING-level
issues concern error-path device-buffer leaks (contradicting the stated WR-07
no-leak contract), a `class_prior` sum-to-one check that sklearn performs but
this code omits, and a Python dtype-dispatch surface that reports a misleading
"not fitted" error on a dtype mismatch.

## Critical Issues

### CR-01: Discrete NB variants accept negative / NaN feature values → silent NaN predictions

**File:** `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs:199-218`,
`crates/mlrs-algos/src/naive_bayes/complement_nb.rs:204-246`,
`crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs:235-260`

**Issue:** `MultinomialNB`, `ComplementNB`, and `BernoulliNB` never validate that
the input matrix `X` is non-negative (a count matrix). sklearn explicitly rejects
this with `check_non_negative(X, "MultinomialNB (input X)")` and raises
`ValueError: Negative values in data passed to ...`. Here a negative count flows
straight through the GATHER into the smoothed-log formulas:

- MultinomialNB (`multinomial_nb.rs:216`):
  `((feature_count[c][j] + alpha) / denom).ln()` — if `feature_count[c][j] + alpha < 0`,
  this is `ln(negative) = NaN`.
- ComplementNB (`complement_nb.rs:233`): `comp_sum` can become 0 or negative, and
  `(cc / comp_sum).ln()` of a non-positive argument is `NaN`/`-inf`.

A `NaN` in `feature_log_prob_` propagates through the predict GEMM into every
joint-LL row; `log_sum_exp_normalize` then returns `NaN` probabilities **with no
error surfaced** — the estimator silently returns garbage. This is a correctness
and data-integrity defect (untrusted host input per the project threat model,
T-04-01-01 / ASVS V5), and it diverges from documented sklearn parity.

NaN inputs are equally unguarded: `host_to_f64(NaN)` flows into the sum and `ln`.

**Fix:** Validate non-negativity (and finiteness) of `X` at `fit` for the three
count-based variants, mirroring sklearn:
```rust
// In each discrete fit(), after the host read of x:
let x_host = x.to_host(pool);
for &xv in x_host.iter() {
    let v = host_to_f64(xv);
    if !v.is_finite() || v < 0.0 {
        return Err(AlgoError::InvalidLabels {  // or a new InvalidCountInput variant
            estimator: "multinomial_nb",
            reason: format!("input X must be finite and non-negative (got {v})"),
        });
    }
}
```
The same guard is needed at `predict`/`joint_log_likelihood` time for the GEMM
input (negative query rows are equally invalid for the count model).

---

### CR-02: PyO3 ingress for the count-based NB variants does not reject negative input either

**File:** `crates/mlrs-py/src/estimators/naive_bayes.rs:512-545` (Multinomial),
`655-690` (Bernoulli), `800-835` (Complement)

**Issue:** The Python `fit` paths call `validated_f32`/`validated_f64` (which, per
the ingress contract, validate finiteness/contiguity but not domain) and then the
algos `fit`, which — per CR-01 — does not reject negatives. So a Python caller
doing `MultinomialNB().fit(X_with_negatives, y)` gets silent `NaN` predictions
instead of the `ValueError` sklearn raises. This is the same defect surfaced at
the FFI boundary (the load-bearing sklearn-parity contract, D-09). The smoke test
`test_naive_bayes.py` only exercises clean non-negative inputs, so it does not
catch this.

**Fix:** Once CR-01 adds the algos-layer guard returning `AlgoError::InvalidLabels`
(or a dedicated variant), the existing `.map_err(algo_err_to_py)?` will surface it
as a Python error automatically — confirm the mapped error is a `ValueError` (not
a generic exception) to match sklearn, and add a negative-input rejection test to
`test_naive_bayes.py`.

---

## Warnings

### WR-01: `class_prior` / `priors` sum-to-one not validated (sklearn rejects)

**File:** `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs:297-311`,
`crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs:453-475`

**Issue:** Both `empirical`/explicit prior resolvers only check the prior *length*
== `n_classes` (data-dependent) and per-entry finiteness/non-negativity (at
build). sklearn additionally requires the priors to sum to 1: GaussianNB raises
`ValueError("The sum of the priors should be 1.")` and the discrete variants
require a normalized `class_prior`. Here a non-normalized prior (e.g. `[0.3, 0.3]`)
is silently accepted and `.ln()`-mapped, yielding joint-LLs that no longer match
sklearn — `predict_proba` rows will still renormalize to 1 via log-sum-exp, so the
rows-sum-to-1 tests pass, masking the divergence from the sklearn oracle.

**Fix:** After the length check, validate `(p.iter().sum::<f64>() - 1.0).abs() <= 1e-6`
and return `AlgoError::InvalidLabels`/a dedicated variant otherwise, matching
sklearn's contract. (The doc comment at `error.rs:522-533` even promises this
"sum-to-one check stays at fit" — but it is not implemented.)

### WR-02: Device-buffer leak on the error path of `grouped_reduce`

**File:** `crates/mlrs-algos/src/naive_bayes/nb_common.rs:265-271`

**Issue:** `block_dev` is uploaded (line 265), then `column_reduce::<F>(...)?` is
called (line 270). If `column_reduce` returns `Err`, the `?` propagates and
`block_dev.release_into(pool)` (line 280) is never reached — the scratch buffer
leaks. This contradicts the WR-07 "conserve live_bytes" contract the module
documents. The `.expect(...)` on the `Option` (line 271) would also panic-leak.

**Fix:** Release `block_dev` before the `?` propagation, e.g. capture the result,
release, then `?`:
```rust
let reduced_res = column_reduce::<F>(pool, &block_dev, n_c, n_features, op, ReducePath::Shared);
let reduced = match reduced_res {
    Ok(Some(r)) => r,
    Ok(None) => { block_dev.release_into(pool); /* unreachable */ unreachable!("shared path always available") }
    Err(e) => { block_dev.release_into(pool); return Err(e); }
};
```

### WR-03: Device-buffer leak on error path in BernoulliNB fit

**File:** `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs:229-237`

**Issue:** `x_bin_dev` is uploaded (line 229), then `class_grouped_sum::<F>(...)?`
(line 236) can return `Err`, propagating before `x_bin_dev.release_into(pool)`
(line 237). The binarized query buffer leaks on a GATHER failure. Same WR-07
violation as WR-02.

**Fix:** Match on the `class_grouped_sum` result, release `x_bin_dev` on both the
Ok and Err arms before propagating.

### WR-04: Python dtype mismatch reports misleading "not fitted" instead of a dtype error

**File:** `crates/mlrs-py/src/estimators/naive_bayes.rs:174,196,219,241`

**Issue:** `predict_proba_f32` on an estimator fitted as f64 (or vice-versa) falls
into the `_ => Err(not_fitted($name, "predict_proba (f32 path)"))` arm. The
estimator *is* fitted — it is simply the wrong dtype. Surfacing a "called before
fit (no fitted state)" error is misleading and will confuse a Python user who
fitted in f64 and called the f32 accessor. The `predict_labels` path (line 151)
has the same issue but is less likely (it dispatches on both arms).

**Fix:** Return a dtype-mismatch error (`PyTypeError`/`PyValueError`) naming the
actual fitted dtype, e.g. "estimator fitted as f64; call predict_proba_f64",
rather than reusing `not_fitted`.

### WR-05: GaussianNB negative-variance clamp can still divide by a tiny epsilon → f32 overflow

**File:** `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs:265,289-292,397`

**Issue:** When `var_smoothing` is the default `1e-9` and a feature column has
~zero variance, `epsilon_ = 1e-9 * max_col_var` can be extremely small (or `0` if
*all* feature variance is ~0). `var[cell] += epsilon_` then leaves `var_` at or
near `0`, and the predict quadratic `(d*d)/v` (line 397) divides by it. In f32
this drives the joint-LL to `+inf`/`NaN`. The `raw_var.max(0.0)` clamp (line 265)
guards negatives but not the "all columns constant → epsilon_ == 0" degenerate.
sklearn guarantees `epsilon_ > 0` only when at least one column has variance;
mlrs inherits the same edge but with no test for an all-constant feature matrix.

**Fix:** Floor `epsilon_` to a small positive minimum (or document/guard the
all-constant-feature case), e.g. `let epsilon_ = (self.var_smoothing * max_col_var).max(f64::MIN_POSITIVE);`
and add a test fitting on a constant-feature matrix.

### WR-06: CategoricalNB silently treats out-of-range predict categories as unseen instead of erroring

**File:** `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs:416-431`

**Issue:** The doc on `AlgoError::InvalidCategoricalInput` (`error.rs:341-355`)
and the module docstring describe a "predict-time category index that exceeds the
per-feature category count" as an *error condition*. But the implementation
instead **silently** maps any out-of-range / negative / non-integer category to
the smoothed `log(alpha/denom)` fallback (lines 421-431). sklearn's CategoricalNB
*raises* `IndexError`/`ValueError` on a category index `>= n_categories_[j]` at
predict. The chosen fallback is a deliberate divergence, but it is undocumented
as a behavior change vs sklearn and contradicts the error variant's own doc.

**Fix:** Either (a) return `AlgoError::InvalidCategoricalInput` for an out-of-range
category at predict (matching sklearn and the error variant's documented purpose),
or (b) explicitly document the fallback as an intentional mlrs divergence and
update the `error.rs` doc. As written, code and documentation disagree.

### WR-07: `accuracy_score` returns `0.0` for empty input — ambiguous with "0% correct"

**File:** `crates/mlrs-algos/src/naive_bayes/nb_common.rs:157-159`

**Issue:** On an empty prediction vector `accuracy_score` returns `0.0`, which is
indistinguishable from "all predictions wrong". sklearn raises on an empty input.
For `score(X_empty, y_empty)` over the FFI this yields a silent `0.0` rather than
an error. Low impact (empty predict is already rejected upstream by the geometry
guard `n_query == 0`), but the function is `pub` and independently callable.

**Fix:** Document the empty-input semantics explicitly, or return `f64::NAN`
(sklearn-like "undefined") instead of `0.0` to avoid the false "0% accuracy"
reading.

### WR-08: `force_alpha` field retained but dead across four variants; clip-vs-fit provenance unverifiable

**File:** `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs:38-39`,
`bernoulli_nb.rs:43-44`, `complement_nb.rs:40-41`, `categorical_nb.rs:60`

**Issue:** `force_alpha` is stored on each struct with `#[allow(dead_code)]`
(Categorical reads it only via `let _ = self.force_alpha;` at `categorical_nb.rs:248`).
The clip already happened at `build()`, so the stored field is pure provenance
and never affects behavior. This is acceptable as a deliberate choice, but the
`#[allow(dead_code)]` suppresses the compiler's own check that the field is truly
unused — if a future edit *should* consult it (e.g. a re-fit path that re-runs the
clip), the lint will not flag the omission. Marginally a maintainability risk.

**Fix:** Either drop the field (provenance can be reconstructed from the clipped
`alpha`) or expose it via an accessor so it is genuinely used, removing the
`#[allow(dead_code)]`.

---

## Info

### IN-01: `log_sum_exp_normalize` doc overstates underflow protection

**File:** `crates/mlrs-algos/src/naive_bayes/nb_common.rs:56-58`

**Issue:** The doc claims the single terminal log "keeps the small probabilities
from underflowing to 0". But `proba_c = exp(log_proba_c)` (line 79) *does*
underflow tiny `log_proba` to `0.0`. Only the `log_proba` output is underflow-safe;
the `proba` output is not. Misleading comment.

**Fix:** Clarify that only `log_proba` avoids underflow; `proba` may still flush
small values to 0 (which is the expected, correct behavior).

### IN-02: Magic tolerance `1e-6` for integer-label rounding duplicated across files

**File:** `gaussian_nb.rs:210`, `multinomial_nb.rs:422`, `categorical_nb.rs:259,421`

**Issue:** The `1e-6` integer round-trip tolerance is hardcoded in four places.
Extract a shared `const NB_LABEL_INT_TOL: f64 = 1e-6;` in `nb_common` so all
variants stay consistent if the tolerance is ever tuned.

### IN-03: `decode_classes` error `estimator` field is the literal `"discrete_nb"`

**File:** `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs:424,436`

**Issue:** The shared `decode_classes` hardcodes `estimator: "discrete_nb"` in its
`InvalidLabels` errors, so a Python user fitting a `ComplementNB` with bad labels
sees `estimator 'discrete_nb'` rather than `'complement_nb'`. Cosmetic, but it
leaks an internal helper name into a user-facing error. Pass the estimator name in
as a parameter (as `resolve_class_log_prior` already does).

### IN-04: Duplicated `host_to_f64` / `f64_to` helpers in every test file

**File:** `gaussian_nb_test.rs:63-77` (and the four sibling test files)

**Issue:** Each NB test file re-defines identical `host_to_f64` / `f64_to` /
`assert_band` / `fixture` helpers. Mild duplication; a shared test-support module
would reduce drift risk. (Test-only, low priority.)

### IN-05: ComplementNB `class_log_prior_` computed but only used in the single-class edge case

**File:** `crates/mlrs-algos/src/naive_bayes/complement_nb.rs:250-256,322-332`

**Issue:** `class_log_prior_` is always resolved at fit (incurring the
empirical/prior computation) but consumed only when `n_classes == 1` at predict.
This matches sklearn's behavior and feeds the accessor, so it is correct — noted
only because the cost is paid unconditionally for a rarely-taken branch. No change
required; documented for reviewer awareness.

---

_Reviewed: 2026-06-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
