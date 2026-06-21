---
phase: 10-sgd-linear-svm
fixed_at: 2026-06-21T00:00:00Z
review_path: .planning/phases/10-sgd-linear-svm/10-REVIEW.md
iteration: 1
findings_in_scope: 14
fixed: 13
skipped: 1
status: partial
---

# Phase 10: Code Review Fix Report

**Fixed at:** 2026-06-21
**Source review:** .planning/phases/10-sgd-linear-svm/10-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 14 (fix_scope = all — CR + WR + IN)
- Fixed: 13
- Skipped: 1

All Phase-10 oracle/build tests were re-run after the fixes and pass:
`mlrs-backend` `sgd_test` (6), `mlrs-algos` `mbsgd_classifier_test` (10),
`mbsgd_regressor_test` (5), `linear_svc_test` (5), `linear_svr_test` (5),
`sgd_config_test` (5). Each touched crate (`mlrs-backend`, `mlrs-algos`,
`mlrs-py`) was `cargo check`ed under `--features cpu` after its edits.

## Fixed Issues

### CR-01: L2 `wscale` penalty applied once per BATCH, not per SAMPLE

**Files modified:** `crates/mlrs-backend/src/prims/sgd.rs`
**Commit:** 2dfd00f
**Applied fix:** Adopted option (b) from the review — the L2 shrink factor is now
raised to the `bsz` power (`(1 - eta·α·(1−l1_ratio))^bsz`) so the averaged-batch
model compounds the per-sample L2 decay over the batch, matching sklearn's
per-sample compounding. `bsz == 1` is unchanged. A comment documents that the
margin is still not re-read mid-batch (the WR-03 divergence).
**Requires human verification:** this is a numerical-logic change; the existing
`batch_size=1` oracle fixtures cannot gate the `batch_size>1` path. A
`batch_size>1` oracle fixture should be added to truly gate it (called out in the
review's CR-01 fix note).

### CR-02: cumulative-L1 soft-shrink `u`/`q` budget advanced once per BATCH

**Files modified:** `crates/mlrs-backend/src/prims/sgd.rs`
**Commit:** e845979
**Applied fix:** The L1 budget advance (`u_l1 += l1_ratio·eta·alpha`) and the
per-coordinate soft-shrink now loop `bsz` times within the batch, so the
cumulative budget tracks the number of samples consumed (matching `_plain_sgd`).
`bsz == 1` is unchanged.
**Requires human verification:** numerical-logic change; no L1/ElasticNet oracle
fixture exists yet, so the L1 path remains end-to-end ungated. The review asks for
an L1/ElasticNet fixture — recommended follow-up.

### WR-01: `LinearSVC`/`LinearSVR` builder `tol` silently ignored

**Files modified:** `crates/mlrs-algos/src/linear/linear_svc.rs`,
`crates/mlrs-algos/src/linear/linear_svr.rs`
**Commit:** 1d3f303
**Applied fix:** Added a `gtol` parameter to the shared `svm_lbfgs_fit`, threaded
`self.config.tol` from both `LinearSVC` and `LinearSVR` call sites, and clamp it to
`>= 1e-12` so a `tol = 0` deterministic-epochs override still requests a deep
converged solve. The hard-coded `1e-9` is gone.

### WR-02: integer label overflow via `as i32`/`as i64` truncation

**Files modified:** `crates/mlrs-algos/src/linear/linear_svc.rs`,
`crates/mlrs-algos/src/linear/mbsgd_classifier.rs`
**Commit:** 7255052
**Applied fix:** After deriving `classes_`, each distinct class id is validated
against `i32` range with `i32::try_from(...)` at fit; an out-of-range label now
returns a typed `AlgoError::InvalidLabels` rather than being silently truncated by
the `as i32` cast in `predict_labels`.

### WR-03: minibatch gradient averaging diverges from sklearn at `batch_size > 1`

**Files modified:** `crates/mlrs-backend/src/prims/sgd.rs`
**Commit:** 980d431
**Applied fix:** Documented (doc-only) on the `SgdParams::batch_size` field and at
the solve site that `batch_size > 1` is NOT sklearn-equivalent (averaged gradient,
no mid-batch re-margin); only `batch_size == 1` reproduces sklearn to tolerance.

### WR-04: `power_t` never validated

**Files modified:** `crates/mlrs-algos/src/error.rs`,
`crates/mlrs-algos/src/linear/mbsgd_classifier.rs`,
`crates/mlrs-algos/src/linear/mbsgd_regressor.rs`
**Commit:** b8dd332
**Applied fix:** Added `BuildError::InvalidPowerT` and a `!power_t.is_finite()`
guard in both SGD builders' `build()`. Non-finite `power_t` (NaN/±inf) is rejected;
negative finite values are accepted and documented as a sklearn-divergent but
well-defined behavior (rate grows with t).

### WR-05: new linear wrappers mix the poisoning-prone lock path

**Files modified:** `crates/mlrs-py/src/estimators/linear.rs`
**Commit:** a882559
**Applied fix:** Converted all 36 legacy
`global_pool().lock().expect("pool mutex")` sites in `linear.rs` (the
`LinearRegression`/`Ridge`/`Lasso`/`ElasticNet`/`LogisticRegression` wrappers) to
the sanctioned `crate::lock_pool()` path, so a process-global poison no longer
re-bricks the legacy linear wrappers shipped alongside the Phase-10 estimators.

### WR-06: mutex-poison recovery permanently corrupts the FOUND-05 memory gate

**Files modified:** `crates/mlrs-backend/src/pool.rs`,
`crates/mlrs-py/src/lib.rs`
**Commit:** 8504298
**Applied fix:** Added `BufferPool::reset_after_poison()` (clears the stale
free-list and resets counters) and call it from `lock_pool()`'s poison-recovery
branch, so the conservation accounting is re-baselined to a meaningful state after
a recovered poison instead of being permanently inflation-blinded. The `lock_pool`
doc was updated from "conservation VOID" to the re-baseline behavior.
**Requires human verification:** the re-baseline semantics (dropping the pool's own
free-list references) are memory-safe by the same ref-count argument the existing
recovery relies on, but the accounting-recovery policy is a judgment call worth a
human confirm.

### WR-07: misleading error type/message for non-binary or non-integer labels

**Files modified:** `crates/mlrs-algos/src/error.rs`,
`crates/mlrs-algos/src/linear/linear_svc.rs`,
`crates/mlrs-algos/src/linear/mbsgd_classifier.rs`
**Commit:** 17244ab
**Applied fix:** Added `AlgoError::InvalidLabels { estimator, reason }` and replaced
the fabricated `PrimError::ShapeMismatch` (geometry) errors used for non-integer
and non-binary labels with it, carrying an honest reason string. `algo_err_to_py`
already maps any `AlgoError` to `PyValueError` via `Display`, so the Python message
is now accurate.

### IN-01: dead `inv_b` binding and dead `.max(1)` on a constant

**Files modified:** `crates/mlrs-backend/src/prims/sgd.rs`
**Commit:** 8b0d54b
**Applied fix:** Deleted the unused `let inv_b = ...;` and `let _ = inv_b;` lines
and dropped the redundant `.max(1)` from both `cube_block` `CubeCount::Static`
divisors (`cube_block` is the constant `256u32`).

### IN-03: `host_to_f64(self.c)` identity no-op on an already-`f64` field

**Files modified:** `crates/mlrs-algos/src/linear/linear_svc.rs`,
`crates/mlrs-algos/src/linear/linear_svr.rs`
**Commit:** 257eb0e
**Applied fix:** Replaced `host_to_f64(self.c)` with `self.c` directly (it is
typed `f64`). `host_to_f64` remains in use elsewhere in both files.

### IN-04: stale Wave-0 scaffold comments contradict the shipped implementation

**Files modified:** `crates/mlrs-algos/src/linear/mbsgd_classifier.rs`,
`crates/mlrs-algos/src/linear/mbsgd_regressor.rs`,
`crates/mlrs-algos/src/linear/mod.rs`,
`crates/mlrs-backend/src/prims/mod.rs`
**Commit:** e5584ac
**Applied fix:** Rewrote the stale "Wave-0 scaffold / bodies land in Wave-1/Wave-3
/ `todo!()`" docs to describe the shipped reality: `fit`/`predict` are implemented;
`LinearSVC`/`LinearSVR` use the L-BFGS primal solver (NOT coordinate descent);
`sgd_solve` is fully implemented.

### IN-05: `optimal_t0` hard-codes `epsilon = 0.1` in its `dloss` probe

**Files modified:** `crates/mlrs-backend/src/prims/sgd.rs`
**Commit:** 437752c
**Applied fix:** Replaced the bare magic `0.1` with a named
`const OPTIMAL_INIT_EPSILON: f64 = 0.0` and a comment explaining the probe epsilon
is inert for the classifier losses the optimal schedule is paired with (matching
sklearn's `optimal_init`).

## Skipped Issues

### IN-02: redundant per-crate duplication of `host_to_f64` / `f64_to_host` / `narrow`

**File:** `crates/mlrs-algos/src/linear/linear_svc.rs:654-679`,
`linear_svr.rs:348-354`, `mbsgd_classifier.rs:551-565`,
`crates/mlrs-backend/src/prims/sgd.rs:473-488`
**Reason:** skipped — the dedup is module-wide and crosses a crate boundary, so a
faithful "single `pub(crate)` helper" fix is out of the narrow per-finding scope.
The identical helper bodies exist in NINE `mlrs-algos/src/linear/` files
(`linear_regression`, `ridge`, `lasso`, `elastic_net`, `coordinate_descent`,
`logistic`, plus the four Phase-10 files) AND independently in `mlrs-backend`'s
`prims/sgd.rs`. A single `pub(crate)` helper cannot span the two crates (it would
have to move to `mlrs-core`), and consolidating only the Phase-10 files would not
achieve the finding's "single helper" goal while still editing 5+ non-Phase-10
estimator files unrelated to this phase — violating the narrow-scope rule and
risking regressions in estimators outside this review. This is an Info-level
cosmetic finding already acknowledged in-code as accepted copy-paste; the proper
fix is a dedicated module-wide refactor task.
**Original issue:** The same `size_of::<F>()` bit-reinterpret helper is
re-implemented in at least six places with identical bodies and `unreachable!`
arms; it remains copy-paste that will drift.

---

_Fixed: 2026-06-21_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
