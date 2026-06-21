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
  critical: 2
  warning: 7
  info: 5
  total: 14
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-06-21
**Depth:** standard
**Files Reviewed:** 24
**Status:** issues_found

## Summary

Phase 10 adds four SGD / linear-SVM estimators (`MBSGDClassifier`, `MBSGDRegressor`,
`LinearSVC`, `LinearSVR`), a PRIM-10 `sgd_solve` host orchestration over two
SharedMemory-free kernels, and the PyO3 wrapper layer. The structure is clean and the
oracle harness is thorough, but adversarial tracing surfaced two correctness defects
that are masked because every committed fixture pins `batch_size=1`: the minibatch L2
penalty and gradient averaging diverge from sklearn for `batch_size > 1`, which is a
publicly settable knob. Several robustness / API-contract defects (ignored `tol`,
unvalidated `power_t`, integer-label-overflow via `as i32` / `as i64` casts, and a known
mutex-poison accounting / lock-path mismatch carried by the new linear wrappers) round
out the findings.

The dominant theme: **the test suite only exercises `batch_size=1` and never asserts the
documented sklearn semantics for the minibatch knob the API exposes.** "Tests pass" here
does not establish that the minibatch path is correct.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: L2 `wscale` penalty applied once per BATCH, not per SAMPLE — wrong for `batch_size > 1`

**File:** `crates/mlrs-backend/src/prims/sgd.rs:299-317`
**Issue:** sklearn's `_plain_sgd` decays the weight scale **once per training sample**
(`wscale *= (1 - eta * alpha * (1 - l1_ratio))` inside the per-sample loop). This prim
applies the L2 shrink factor exactly **once per minibatch**, after a single averaged
gradient step:

```rust
let l2_factor = (1.0 - (1.0 - params.l1_ratio) * eta * params.alpha).max(0.0);
if params.alpha > 0.0 && l2_factor != 1.0 {
    // ...applied once over the whole batch...
}
```

For `batch_size = b`, sklearn shrinks by `(1 - eta·α·(1−l1_ratio))^b` over the batch (and
re-reads the margin between samples); this code shrinks by `(1 - eta·α·(1−l1_ratio))^1`.
The two agree **only** when `b == 1`. Every committed fixture pins `batch_size=1`
(`gen_oracle.py` SGD section; all oracle tests call `.batch_size(1)`), so the divergence is
invisible to CI. `batch_size` is a public builder/PyO3 knob, so a user calling
`MBSGDClassifier(batch_size=8)` silently gets a non-sklearn result with no error.

**Fix:** Either (a) loop the gradient + penalty update per sample within the batch (true
sklearn `_plain_sgd` semantics), or (b) raise `l2_factor` to the `bsz` power and document
that the margin is not re-read mid-batch:

```rust
// (b) minimal correctness patch for the averaged-batch model:
let l2_factor =
    (1.0 - (1.0 - params.l1_ratio) * eta * params.alpha).max(0.0).powi(bsz as i32);
```

Add an oracle fixture with `batch_size > 1` so the path is actually gated.

### CR-02: cumulative-L1 soft-shrink `u` / `q` budget advanced once per BATCH, not per SAMPLE

**File:** `crates/mlrs-backend/src/prims/sgd.rs:319-338`
**Issue:** sklearn accumulates the L1 penalty budget `u += eta * alpha * l1_ratio` **per
sample** and applies `l1penalty` per sample. This prim advances `u_l1` and runs the
soft-shrink **once per batch**:

```rust
if params.apply_l1 && params.l1_ratio > 0.0 && params.alpha > 0.0 {
    u_l1 += params.l1_ratio * eta * params.alpha;   // once per batch
    // ...single soft-shrink over all coords...
}
```

For `batch_size > 1` the cumulative budget `u` grows too slowly relative to the number of
samples consumed (`t` is incremented by `bsz`, but `u` only once), so the L1 path produces
a different sparsity pattern than sklearn. As with CR-01 this is masked by the
`batch_size=1` fixtures and by the fact that no L1/ElasticNet oracle fixture exists at all
(the committed SGD fixtures are all `penalty="l2"`). The L1 lowering
(`mbsgd_classifier.rs:519-532` `apply_l1`) is therefore completely **untested** end to end.

**Fix:** Advance `u_l1` and apply `l1penalty` per sample inside the batch loop (matching
`_plain_sgd`), and add an L1/ElasticNet oracle fixture to gate the path. Until then the
penalty/`apply_l1` lowering should be considered unverified.

## Warnings

### WR-01: `LinearSVC` / `LinearSVR` builder `tol` is silently ignored by the solver

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:612`
**Issue:** `svm_lbfgs_fit` hard-codes the L-BFGS gradient tolerance to `1e-9` and never
threads the user's `tol` (stored in `SgdConfig.tol`, settable via `.tol()` and the PyO3
`tol=` kwarg) into `lbfgs_minimize`:

```rust
let result = lbfgs_minimize(x0, closure, 1e-9, LBFGS_FTOL, LBFGS_MAXLS, max_iter)?;
```

`config.tol` is plumbed all the way from Python (`PyLinearSVC::new(tol=...)`) into the
`SgdConfig`, then dropped. A caller tightening or loosening `tol` gets identical behavior,
silently violating the sklearn-compatible API contract this phase advertises.
**Fix:** pass the configured tolerance through, e.g.
`lbfgs_minimize(x0, closure, self.config.tol.max(1e-12), ...)`, or document that
`LinearSVC`/`LinearSVR` deliberately fix `gtol` and remove the `tol` setter to avoid a
dead knob.

### WR-02: integer label overflow via `as i64` / `as i32` truncation casts

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:322, 445-449`; `crates/mlrs-algos/src/linear/mbsgd_classifier.rs:331, 392-396`
**Issue:** Labels are rounded into `i64` via `li as i64` from an `f64`
(`linear_svc.rs:322`, `mbsgd_classifier.rs:331`) and then emitted as `i32` in
`predict_labels` via `self.classes_[k] as i32`. A label that fits in the `f64` mantissa
but exceeds `i32::MAX` (e.g. a class id like `3_000_000_000`) is **silently truncated**
(`as i32` wraps), producing a wrong predicted label rather than an error. The integrality
check (`(li - lf).abs() > 1e-6`) accepts such values. `predict_labels` returns
`DeviceArray<_, i32>`, so the API itself caps labels at i32 — but nothing validates the
incoming labels against that range.
**Fix:** validate that each distinct class label fits in `i32` at fit (return a typed
`AlgoError`), or widen the predicted-label dtype. At minimum, replace the silent `as i32`
with a checked conversion.

### WR-03: minibatch gradient averaging by `inv_b` diverges from sklearn at `batch_size > 1`

**File:** `crates/mlrs-backend/src/prims/sgd.rs:275-296, 340-345`
**Issue:** The weight and intercept steps scale the summed gradient by `binv = 1/bsz`
(`sgd_weight_update` multiplies by `inv_b`; intercept `bias -= eta·binv·g_sum`). sklearn's
`SGDClassifier`/`SGDRegressor` are **per-sample** (effective `inv_b = 1`), not averaged
minibatch. Combined with CR-01/CR-02 this means the entire `batch_size > 1` regime is a
different algorithm from sklearn, yet `batch_size` is presented as a sklearn-compatible
knob. The kernel comment even calls the averaging "the host's choice" (`A2`), which
acknowledges the divergence without surfacing it to the user.
**Fix:** document explicitly that `batch_size > 1` is NOT sklearn-equivalent, or implement
true per-sample SGD; add a gating fixture either way.

### WR-04: `power_t` is never validated

**File:** `crates/mlrs-algos/src/linear/mbsgd_classifier.rs:217-251`; `crates/mlrs-algos/src/linear/mbsgd_regressor.rs:212-254`
**Issue:** Every other schedule scalar is validated at `build()` (`alpha >= 0`,
`eta0 > 0`, `l1_ratio ∈ [0,1]`, `epsilon >= 0`), but `power_t` is accepted unchecked and
flows into `schedule_eta`'s `eta0 / t.powf(power_t)` (`sgd.rs:466`). A negative `power_t`
makes the step rate GROW with `t` (divergence); `power_t = 0` silently degenerates
invscaling to constant. This is the same untrusted-hyperparameter class the phase's threat
model (T-10-03-01) covers for the sibling scalars.
**Fix:** add a `BuildError` guard (or at least reject non-finite `power_t` and document the
divergence behavior for negatives).

### WR-05: new linear wrappers mix the poisoning-prone lock path

**File:** `crates/mlrs-py/src/estimators/linear.rs:87, 114, 143` (and the other legacy `global_pool().lock().expect(...)` sites in the file)
**Issue:** `lib.rs:108-118` documents that `lock_pool()` is the SANCTIONED lock path and
that "one surviving `global_pool().lock().expect("pool mutex")` re-panics on a poisoned
mutex and re-bricks the interpreter." The Phase-10 SGD/SVM wrappers correctly use
`lock_pool()`, but they live in the SAME `linear.rs` file as `LinearRegression`/`Ridge`/
`Lasso`/`ElasticNet`/`LogisticRegression`, which still use the panicking
`.lock().expect("pool mutex")` form. Because the pool is process-global, a poison created
by ANY estimator (including a Phase-10 device fault) will brick the legacy linear
wrappers, defeating the recovery the new code relies on. This is acknowledged as a
"tracked migration" in `lib.rs`, but Phase-10 added more code to the very file that mixes
the two helpers.
**Fix:** convert the remaining `global_pool().lock().expect(...)` sites in `linear.rs` to
`lock_pool()`; the change is mechanical and removes the partial-recovery footgun for
estimators shipped together.

### WR-06: mutex-poison recovery permanently corrupts the FOUND-05 memory-conservation gate

**File:** `crates/mlrs-py/src/lib.rs:120-144` (interaction with `crates/mlrs-backend/tests/memory_gate_test.rs`)
**Issue:** `lock_pool()`'s own doc (the "ACCOUNTING CAVEAT", `lib.rs:120-136`) states that
after a recovered poison, `live_bytes`/`peak_bytes` are "monotonically INFLATED for the
rest of the process" and the conservation property is "VOID." The new SGD path does many
incremental `from_host`/`release_into` allocations inside `py.detach` while holding the
guard (`sgd.rs` epoch loop, `svm_lbfgs_fit` per-eval GEMMs), so a device fault mid-`fit`
is a realistic poison trigger. The memory gate (`memory_gate_test.rs`) asserts
conservation but cannot detect this post-poison inflation, so a real leak-class regression
after a recovered fault would pass the gate.
**Fix:** on poison recovery, reset/reconstruct the pool counters (or replace the pool) so
conservation accounting stays meaningful, rather than only documenting that it is void.

### WR-07: misleading error type/message for non-binary or non-integer labels

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:314-334`; `crates/mlrs-algos/src/linear/mbsgd_classifier.rs:323-343`
**Issue:** Invalid-label conditions (non-integer label, not-exactly-2 classes) are reported
as `PrimError::ShapeMismatch` with `operand: "linear_svc.y (labels must be integers)"`.
This is a data-validity error shoehorned into a geometry error type, and at the PyO3
boundary `algo_err_to_py` maps everything to `PyValueError`, so the Python caller sees a
"shape mismatch"-flavored message for what is actually a label-content problem. `AlgoError`
has no dedicated label variant, but reusing `ShapeMismatch` with fabricated `rows/cols/len`
is a category error that will mislead debugging.
**Fix:** add a typed label-validity `AlgoError` variant (e.g. `InvalidLabels`) or at least
use an honest message that does not claim a shape mismatch.

## Info

### IN-01: dead `inv_b` binding and dead `.max(1)` on a constant

**File:** `crates/mlrs-backend/src/prims/sgd.rs:179, 369, 224-227, 280-283`
**Issue:** `let inv_b = 1.0 / batch as f64;` (line 179) is shadowed by per-batch `binv` and
only "used" via `let _ = inv_b;` (line 369) — dead code. Separately, `cube_block.max(1)`
appears in both `CubeCount::Static` computations, but `cube_block` is the constant `256u32`
so `.max(1)` can never change it.
**Fix:** delete the unused `inv_b` and the `let _ = inv_b;` line; drop the redundant
`.max(1)`.

### IN-02: redundant per-crate duplication of `host_to_f64` / `f64_to_host` / `narrow`

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:654-679`, `linear_svr.rs:348-354`, `mbsgd_classifier.rs:551-565`, `crates/mlrs-backend/src/prims/sgd.rs:473-488`
**Issue:** The same `size_of::<F>()` bit-reinterpret helper is re-implemented in at least
six places with identical bodies and `unreachable!` arms. Acknowledged in comments
("mirrors the logistic.rs helper") but remains copy-paste that will drift.
**Fix:** hoist a single `pub(crate)` helper (e.g. a `float_cast` module) and reuse it.

### IN-03: `host_to_f64(self.c)` is an identity no-op on an already-`f64` field

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:353`; `crates/mlrs-algos/src/linear/linear_svr.rs:293`
**Issue:** `self.c` is typed `f64`; `host_to_f64(self.c)` instantiates the helper with
`F = f64`, a size-8 identity bit-cast. Harmless but reads as if a narrowing conversion is
happening when none is.
**Fix:** use `self.c` directly.

### IN-04: stale Wave-0 scaffold comments contradict the shipped implementation

**File:** `crates/mlrs-algos/src/linear/mbsgd_classifier.rs:6-10`, `mbsgd_regressor.rs:5-8`, `crates/mlrs-algos/src/linear/mod.rs:49`, `crates/mlrs-backend/src/prims/mod.rs:42-47`
**Issue:** Multiple docs still describe the code as a "Wave-0 scaffold" with "fit/predict
bodies land in Wave-1/Wave-3", and `mod.rs:49` claims "`LinearSVC`/`LinearSVR` reuse the v1
coordinate-descent solver (D-07 — liblinear CD)" — but the shipped solver is L-BFGS
(`linear_svc.rs` Open-Q1 resolution), not CD. `prims/mod.rs:42-47` says `sgd_solve`'s
"compute path `todo!()` until Wave-1" though it is fully implemented. These stale comments
mislead future readers about the actual solver.
**Fix:** update the module docs to match the shipped L-BFGS / `sgd_solve` reality.

### IN-05: `optimal_t0` hard-codes `epsilon = 0.1` in its `dloss` probe

**File:** `crates/mlrs-backend/src/prims/sgd.rs:448`
**Issue:** `dloss(loss, -typw, 1.0, 0.1)` passes a magic `0.1` epsilon. It is inert for the
classifier losses that actually use the optimal schedule (and sklearn's `optimal_init`
likewise ignores epsilon there), so this is not a correctness bug — but a magic literal in
a schedule-init path invites a future mistake if `optimal` is ever paired with an
epsilon-insensitive loss.
**Fix:** pass the configured `params.epsilon` (or a named constant with a comment) instead
of a bare `0.1`.

---

_Reviewed: 2026-06-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
