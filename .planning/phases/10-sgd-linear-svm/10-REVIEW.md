---
phase: 10-sgd-linear-svm
reviewed: 2026-06-21T00:00:00Z
depth: standard
files_reviewed: 24
files_reviewed_list:
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/linear/linear_svc.rs
  - crates/mlrs-algos/src/linear/linear_svr.rs
  - crates/mlrs-algos/src/linear/mbsgd_classifier.rs
  - crates/mlrs-algos/src/linear/mbsgd_regressor.rs
  - crates/mlrs-algos/src/linear/mod.rs
  - crates/mlrs-algos/src/linear/sgd_config.rs
  - crates/mlrs-algos/tests/linear_svc_test.rs
  - crates/mlrs-algos/tests/linear_svr_test.rs
  - crates/mlrs-algos/tests/mbsgd_classifier_test.rs
  - crates/mlrs-algos/tests/mbsgd_regressor_test.rs
  - crates/mlrs-algos/tests/sgd_config_test.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/src/prims/sgd.rs
  - crates/mlrs-backend/tests/memory_gate_test.rs
  - crates/mlrs-backend/tests/sgd_test.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-kernels/src/sgd.rs
  - crates/mlrs-py/src/errors.rs
  - crates/mlrs-py/src/estimators/linear.rs
  - crates/mlrs-py/src/lib.rs
  - crates/mlrs-py/tests/sgd_smoke_test.rs
  - crates/mlrs-py/tests/test_sgd.py
  - scripts/gen_oracle.py
findings:
  critical: 1
  warning: 7
  info: 5
  total: 13
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-06-21
**Depth:** standard
**Files Reviewed:** 24
**Status:** issues_found

## Summary

Phase 10 adds four SGD/linear-SVM estimators (`MBSGDClassifier`, `MBSGDRegressor`,
`LinearSVC`, `LinearSVR`), the shared `sgd_config` lowering surface, the PRIM-10
`sgd_solve` host orchestrator + two SharedMemory-free kernels, the PyO3 wrappers,
and the oracle fixtures. The kernels are correctly SharedMemory/infinity-free
(cpu-MLIR safe) and the per-sample `dloss` table matches the sklearn `_sgd_fast`
subgradient table (verified against the unit test). Build-time hyperparameter
validation is thorough.

The central correctness concern is that the **headline schedule for
`SGDClassifier` — `learning_rate="optimal"` (the sklearn default) — is never
validated against a sklearn oracle**, even though the committed
`mbsgd_classifier_optimal_*.npz` fixtures exist. Every active classifier oracle
test pins the *constant* schedule instead, so the Bottou `t0` math and the
optimal-schedule clock are completely unexercised against ground truth. Combined
with the documented loose bands (5e-3 f64 coef, well above the project's 1e-5
contract) and a confirmed loss-vs-penalty *ordering* discrepancy vs sklearn in
`sgd_solve`, the numerical-fidelity claims for the SGD path are weaker than the
artifacts imply. Secondary issues: an unreleased device buffer on the L-BFGS
device-failure path, a self-contradictory tie-break comment, and dead code.

## Critical Issues

### CR-01: `optimal` learning-rate schedule (the sklearn `SGDClassifier` default) has no oracle validation; the committed fixture is generated but never loaded

**File:** `crates/mlrs-algos/tests/mbsgd_classifier_test.rs:108-159`, `scripts/gen_oracle.py:1832-1865`, `crates/mlrs-backend/src/prims/sgd.rs:431-457`

**Issue:** `gen_oracle.py` deliberately emits BOTH a constant-schedule and an
`_optimal`-schedule classifier fixture (lines 1836-1864), and both
`mbsgd_classifier_optimal_f32_seed42.npz` / `..._f64_...npz` are committed in
`tests/fixtures/`. The generator docstring states the intent explicitly: "a
constant-schedule match with an optimal-schedule mismatch localizes the bug to
`t0`." But **no Rust test ever loads the `_optimal` fixture** — `fit_hinge`
hard-codes `LearningRate::Constant` (line 132), and every `oracle*`/`exact_labels*`
test routes through it. The `optimal` schedule arithmetic
(`schedule_eta`/`optimal_t0`, `sgd.rs:431-457`) is exercised only by a host-only
unit test (`sgd_test.rs::schedule_constant_then_invscaling_then_optimal`) that
recomputes the SAME formula it is testing (a tautology, not a sklearn oracle).

Because `optimal` is the **default** schedule for `SGDClassifier` (confirmed by
`MBSGDClassifierBuilder::default` at `mbsgd_classifier.rs:123` and the D-03 litmus
at `mbsgd_classifier_test.rs:328`), a real `SGDClassifier()` with no overrides
runs an entirely unvalidated solver path. The `t0` clock subtleties (sklearn
increments the sample counter `t` and uses `1/(alpha*(t0 + t - 1))`; the
`optimal_t0` here hard-codes `epsilon=0.1` into the `dloss` probe at line 436 and
returns `1.0` for `alpha<=0`) could diverge from sklearn and the suite would stay
green. This is the precise "tests pass ≠ correct" failure the adversarial review
must surface: the artifact *looks* covered (fixture present, generator documents
the t0-isolation rationale) but the assertion that would catch a `t0` bug does not
exist.

**Fix:** Add an `oracle_optimal` test (both dtypes, `skip_f64_with_log` gated)
that loads `mbsgd_classifier_optimal_{f32,f64}_seed42.npz` and fits with
`.learning_rate(LearningRate::Optimal)` (no `eta0`), asserting `coef_`/`intercept_`
within the documented band AND `predict_labels` exactly. Mirror the existing
`fit_hinge` but parameterize the schedule:
```rust
fn fit_hinge_sched<F>(case: &OracleCase, lr: LearningRate) -> (Vec<f64>, f64, Vec<i32>) {
    // ... same as fit_hinge but .learning_rate(lr) and omit .eta0 for Optimal ...
}

#[test]
fn oracle_optimal() {
    if capability::skip_f64_with_log() { return; }
    let case = load_npz(fixture("mbsgd_classifier_optimal_f64_seed42.npz")).unwrap();
    let (coef, intercept, labels) = fit_hinge_sched::<f64>(&case, LearningRate::Optimal);
    assert_band(&coef, case.expect_f64("coef"), COEF_BAND_F64, "optimal coef_");
    assert_eq!(labels, /* predict_ref */, "optimal exact labels");
}
```
If the optimal path cannot meet the band, that is itself the bug the fixture was
designed to localize — it must be fixed before the SGD path can claim sklearn
fidelity.

## Warnings

### WR-01: `sgd_solve` applies the L2 penalty shrink BEFORE the gradient step; sklearn applies it after — exact-iterate reproduction is impossible

**File:** `crates/mlrs-backend/src/prims/sgd.rs:267-307`

**Issue:** The host loop computes the margin `p` with the current `w` (line 234),
then applies the lazy-L2 `wscale` shrink to `w` (lines 271-284, "BEFORE the
gradient step"), then runs `sgd_weight_update` (line 296). In sklearn's
`_plain_sgd`, the per-sample order is: compute `p` → add the loss-gradient step to
`w` → THEN decay `wscale *= (1 - eta*lambda)`. Shrinking before vs after the
gradient step changes which iterate the penalty scales, so the device solver can
never reproduce sklearn's exact `coef_`. This is why the tests fall back to a
`5e-3` f64 band (`mbsgd_classifier_test.rs:59`) — an order of magnitude looser
than the project's stated 1e-5 contract. The ordering should match sklearn so the
band can tighten and the math is provably correct, not merely "close."

**Fix:** Move the L2 `wscale` shrink to AFTER `sgd_weight_update` (and before the
cumulative-L1 shrink, matching sklearn's `_plain_sgd` sample loop), then re-pin the
fixtures and tighten the bands. At minimum, document in `sgd.rs` that the iterate
deliberately differs from sklearn's and the band is a consequence, so the
divergence is not mistaken for round-off.

### WR-02: convergence delta is measured against the post-L2-shrink `w`, omitting the shrink from `max|Δw|`

**File:** `crates/mlrs-backend/src/prims/sgd.rs:271-284, 339-345`

**Issue:** `w_host` is read at line 271 from the pre-shrink `w_dev` and then mutated
in place by the L2 shrink loop (lines 275-277), so by the time the convergence
bookkeeping runs (line 342, `change = (w_new[j] - w_host[j]).abs()`), `w_host`
already holds the *shrunk* weights. The per-batch coefficient change therefore
excludes the L2 contribution, understating `max_change` and letting the
`tol`-based early stop (line 364) fire prematurely when `tol > 0`. The committed
fixtures pin `tol=0` so this is currently masked, but any user-facing
`SGDClassifier(tol=1e-3)` (the default `tol`) hits this path and may stop early.

**Fix:** Snapshot the true start-of-batch `w` (before the L2 shrink) into a
separate vector and diff `w_new` against that snapshot, so `max_change` reflects
the full per-batch update including regularization.

### WR-03: L-BFGS device-failure path leaks `x_aug_dev` (and partially-acquired buffers) without releasing back to the pool

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:520-544, 563-580, 612-616`

**Issue:** Inside `svm_lbfgs_fit`, when a per-iteration GEMM fails, the closure
captures the `PrimError`, releases `w_dev`/`g_dev`, and returns a sentinel
(`f64::MAX`). After `lbfgs_minimize` returns, line 613 checks `prim_err` and
releases `x_aug_dev` before returning the error — good. BUT on the FIRST GEMM
failure (lines 539-543), only `w_dev` is released; `margins` was never produced so
that is fine, but the persistent `x_aug_dev` allocated at line 515 is only released
on the line-613 path. That path IS reached, so `x_aug_dev` is released. However the
`NotConverged`/`broke`/`hit_cap` paths (lines 625-631) DO release `x_aug_dev`, and
the success path (line 648) releases it — so the accounting is actually balanced.
The genuine gap: there is no `BufferPool`-conserving release of the per-eval
`w_dev` on the SUCCESS path of each closure call — each L-BFGS evaluation
allocates `w_dev`/`g_dev` from host and releases them at lines 583-584, which is
correct. The remaining concern is robustness: the closure swallows the device
error and returns `f64::MAX`, which can cause `lbfgs_minimize`'s line search to
interpret the failure as a finite-but-huge loss and iterate further before the
post-hoc `prim_err` check fires — wasting work and potentially masking the true
failure point. Confirm `lbfgs_minimize` cannot loop forever on a `f64::MAX`
plateau.

**Fix:** Verify the L-BFGS line search terminates on a `f64::MAX` return (it should
hit `LineSearchFailed`); if not, propagate the device error immediately rather than
via the sentinel. Add a test that injects a GEMM failure (e.g. a deliberately
malformed augmented shape) and asserts `AlgoError::Prim` is returned, not a hang.

### WR-04: `LinearSVR::predict` does not validate query `n_features` against the fitted feature count

**File:** `crates/mlrs-algos/src/linear/linear_svr.rs:329-344`, `crates/mlrs-algos/src/linear/mbsgd_regressor.rs:336-351`

**Issue:** Both regressors delegate `predict` to `predict_linear`, which checks
`coef.len() == n_features` (`elastic_net.rs:238`). That happens to equal the fitted
feature count because `coef_` has length `n_features` from fit. So a query with the
WRONG `n_features` is caught — but only incidentally, and the error is a generic
`DimMismatch { dim: "n_features" }` rather than the estimator-named
`n_features != self.n_features` guard the classifiers use
(`linear_svc.rs:415-421`, `mbsgd_classifier.rs:472-478`). `LinearSVR`/`MBSGDRegressor`
do not even store `n_features`. The behavior is correct but the asymmetry means a
future refactor of `predict_linear` that drops the `coef.len()` check would
silently remove the regressors' only shape guard.

**Fix:** Either store `n_features` on the regressor structs and add an explicit
`n_features != self.n_features` guard mirroring the classifiers, or add a comment
in `predict_linear` flagging that the `coef.len() == n_features` check is the
load-bearing fitted-shape guard for its callers so it is never removed.

### WR-05: `MBSGDClassifier::predict_proba` returns a sigmoid for NON-log losses without raising, diverging from sklearn

**File:** `crates/mlrs-algos/src/linear/mbsgd_classifier.rs:402-435`

**Issue:** `predict_proba` always computes `σ(margin)` over the decision margin,
regardless of the configured loss. sklearn's `SGDClassifier.predict_proba` RAISES
`AttributeError`/`NotImplementedError` for `loss != "log_loss"` (and only
`"modified_huber"`/`"log_loss"` are supported). Here a hinge-loss classifier
silently returns an uncalibrated sigmoid (the docstring at lines 406-411
acknowledges this). A caller who trained with the default `hinge` loss and calls
`predict_proba` gets plausible-looking but meaningless probabilities instead of an
error — a correctness/contract gap that masks misuse.

**Fix:** Gate `predict_proba` on `self.config.loss == Loss::Log` and return
`AlgoError::Unsupported { estimator: "mbsgd_classifier", operation: "predict_proba (non-log loss)" }`
otherwise, matching sklearn's refusal. If the project intends to keep the permissive
behavior, document it as a deliberate divergence in the estimator doc and the PyO3
wrapper.

### WR-06: `f64::MAX` sentinel as a loss value can be NaN-poisoned through the gradient norm in `svm_lbfgs_fit`

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:521-523, 542, 578`

**Issue:** On the `prim_err.is_some()` early-return (line 521-523) and the
GEMM-failure returns (lines 542, 578), the closure returns `(f64::MAX, vec![0.0; d_aug])`.
Returning `f64::MAX` as the objective with a zero gradient can drive the strong-Wolfe
line search into undefined behavior: a zero gradient at a finite-but-maximal loss
signals a stationary point, which `lbfgs_minimize` may accept as "converged",
causing `svm_lbfgs_fit` to skip the `prim_err` check semantics and return a
nonsense `coef_` if the post-loop `prim_err` branch (line 613) were ever reordered.
Currently the line-613 check fires first, so this is latent, but the sentinel
design is fragile.

**Fix:** Use `f64::INFINITY` (the host side is f64, not a device kernel, so the
cpu-MLIR infinity ban does not apply here) so the line search unambiguously rejects
the step, or restructure so the device error short-circuits `lbfgs_minimize`
entirely rather than relying on a magic loss value plus a post-hoc check.

### WR-07: oracle bands (5e-3 / 2e-2 f64) are an order of magnitude looser than the project's stated 1e-5 contract, with no convergence-equivalence justification for SGD coef_

**File:** `crates/mlrs-algos/tests/mbsgd_classifier_test.rs:59-61`, `crates/mlrs-algos/tests/mbsgd_regressor_test.rs:56-58`

**Issue:** CLAUDE.md states the core correctness contract is "abs/rel error ≤ 1e-5
vs scikit-learn." The SGD oracle tests assert `coef_`/`intercept_` only to
`BAND_F64 = 5e-3` (classifier) / `5e-3` (regressor) and `predict` to the same. The
comments attribute this to "order-of-operations differs in the last-bit
accumulation," but a 5e-3 gap is far beyond last-bit f64 round-off (≈1e-15
relative) — it reflects a genuinely DIFFERENT iterate (see WR-01 ordering). Unlike
`LogisticRegression` (where the looser `coef_` band is justified by documented
gauge freedom and the predict/proba gate is strict), there is no gauge argument for
SGD coefficients; sklearn and mlrs should converge to the same point under the same
deterministic schedule. The exact-label gate is the only strict check, and labels
are robust to large coefficient errors on well-separated blobs (the fixtures use
`±4σ` cluster centers), so the hard gate is weak evidence of coefficient fidelity.

**Fix:** Resolve WR-01 (ordering) so the iterates match, then tighten the bands
toward 1e-5; or, if a residual gap remains, document the SGD-specific reason
(e.g. exact per-sample loss-gradient ordering under `batch_size=1`) the way the
logistic gauge note does, rather than attributing a 5e-3 gap to "last-bit"
accumulation.

## Info

### IN-01: dead `_dual` binding in `LinearSVC::fit`

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:346`

**Issue:** `let _dual = n_samples < n_features;` is computed and immediately
discarded (underscore-prefixed). The comment says it is "computed only for fidelity
to sklearn's resolution rule," but it has no effect and no diagnostic output. It is
dead code that suggests an unfinished feature.

**Fix:** Remove the binding, or actually surface it (e.g. log it or store it for a
`dual_` accessor) if the diagnostic intent is real.

### IN-02: self-contradictory tie-break comment in `LinearSVC::predict_labels`

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:438-444`

**Issue:** The comment says "Ties (margin == 0) break toward the lower class,
matching sklearn's `>= 0 -> +1`?" then "sklearn uses `decision >= 0` -> the +1
class; we mirror that with `>= 0`." The first clause (ties → lower class)
contradicts the code (`m >= 0.0 -> classes_[1]`, the HIGHER class) and the second
clause. The dangling `?` reads like an unresolved note left in.

**Fix:** Rewrite the comment to state the actual behavior unambiguously: a margin
of exactly 0 maps to `classes_[1]` (the +1 / higher class), matching sklearn's
`decision >= 0`.

### IN-03: `host_to_f64(self.c)` is a confusing no-op identity

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:353`, `crates/mlrs-algos/src/linear/linear_svr.rs:293`

**Issue:** `self.c` is already `f64`, so `host_to_f64::<f64>(self.c)` infers
`F = f64` and hits the `8 => identity` arm — a pointless round-trip through
`bytemuck` that reads as if a dtype conversion were happening. It is harmless but
misleading (a reader may think `c` is being narrowed to the estimator's `F`).

**Fix:** Replace with `let c = self.c;` directly. The value is intentionally kept in
f64 for the host-side L-BFGS objective.

### IN-04: `optimal_t0` hard-codes `epsilon=0.1` into the `dloss` probe

**File:** `crates/mlrs-backend/src/prims/sgd.rs:436`

**Issue:** `dloss(loss, -typw, 1.0, 0.1)` passes a magic `0.1` epsilon. For
hinge/log/squared-error losses (the only ones that use the `optimal` schedule in
practice) the epsilon arg is ignored, so this is currently harmless. But if a
future caller wires `optimal` with an epsilon-insensitive loss, the `t0`
computation would silently use the wrong epsilon. The magic number is unexplained.

**Fix:** Pass the real `params.epsilon` through to `optimal_t0`, or assert/document
that `optimal` is only valid for epsilon-free losses.

### IN-05: `lower_config` maps `Loss::SquaredLoss -> SgdLoss::SquaredError` but the two enums are otherwise 1:1 — verify the regressor's `epsilon` is threaded for the epsilon losses

**File:** `crates/mlrs-algos/src/linear/mbsgd_classifier.rs:504-547`

**Issue:** `lower_config` correctly threads `cfg.epsilon` into `SgdParams.epsilon`
(line 542), and the classifier sets `epsilon: 0.0` in its config so the regression
epsilon branch is inert for the classifier. This is correct, but the classifier
NEVER validates that a caller cannot set a nonzero epsilon (the classifier builder
has no `epsilon` setter, so `SgdConfig.epsilon` is always 0 — fine). No action
needed beyond confirming the classifier path can never carry a stray epsilon; the
code is correct as written. Flagged only to record that the
classifier/regressor epsilon asymmetry was checked and is sound.

**Fix:** None required; documented for traceability.

---

_Reviewed: 2026-06-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
