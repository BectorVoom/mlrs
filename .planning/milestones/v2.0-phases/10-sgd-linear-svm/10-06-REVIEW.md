---
phase: 10-sgd-linear-svm
plan: 06
reviewed: 2026-06-21T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - crates/mlrs-backend/src/prims/sgd.rs
  - crates/mlrs-algos/tests/mbsgd_classifier_test.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 10 Plan 06: Code Review Report

**Reviewed:** 2026-06-21
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

This gap-closure plan correctly addresses CR-01 (missing optimal-schedule oracle tests) and
WR-01/WR-02 (L2 wscale shrink order and convergence-delta snapshot). The implementation is
functionally sound: the ordering change is empirically validated by the ~3e-7 constant-schedule
error and the exact-label hard gate on both optimal-schedule tests.

Three warnings remain:
- The module-level and function-level doc comments in `sgd.rs` still describe the old (pre-fix)
  execution order, leaving the documentation actively misleading to the next reader.
- The `assert_band` tolerance formula (`band + band * |expected|`) uses the *fixture* value `e`
  as the reference magnitude, which inverts the typical relative-error convention and
  over-inflates the allowed error when fixture values are large, but under-inflates it when they
  are small (including zero) — a latent test-rigour defect.
- `oracle_optimal_f32` omits the fixture-length sanity checks (`coef_ref.len() == N_FEATURES`,
  `intercept_ref.len() == 1`) that `oracle_optimal` (f64) includes, creating an asymmetric
  test harness.

Two informational items: a pre-existing float-equality guard (`l2_factor != 1.0`) that carries
into the new code path unchanged, and the RESEARCH.md per-sample sequence that contradicts the
implementation order (a documentation artifact acknowledged in the SUMMARY).

---

## Warnings

### WR-01: Module-level and function-level doc comments still describe the pre-fix order

**File:** `crates/mlrs-backend/src/prims/sgd.rs:23-25` and `:130-132`

**Issue:** The module-level doc comment (lines 23-25) reads:

```
[`sgd_solve`] drives the two-pass GATHER kernels per minibatch:
  `sgd_margin` → g[] → eta → host lazy-L2 / cumulative-L1 penalty shrink →
  `sgd_weight_update` (pass 2) → host intercept step
```

The `sgd_solve` function-level doc comment (lines 130-132) reads identically:

```
`sgd_margin::launch` → … → `eta = schedule_eta(t)` →
  host lazy-L2 / cumulative-L1 penalty shrink →
  `sgd_weight_update::launch` → host intercept …
```

Both state the penalty shrink runs BEFORE `sgd_weight_update`. The implementation now does the
opposite — gradient step first, then shrink — which is exactly what WR-01 was supposed to fix.
A reader following these doc comments to understand the algorithm or to port it will reproduce
the pre-fix (wrong) order.

**Fix:** Update both doc blocks to reflect the post-fix order:

```rust
//! [`sgd_solve`] drives the two-pass GATHER kernels per minibatch:
//! `sgd_margin::launch` → host `g[i] = dloss(p_i, y_i)` → `eta = schedule_eta(t)`
//! → `sgd_weight_update::launch` (gradient step) → host lazy-L2 `wscale` shrink →
//! host cumulative-L1 soft-shrink → host intercept step.
//! Order matches sklearn `_plain_sgd` / RESEARCH §Per-sample update sequence line 231.
```

Apply the same correction to the `sgd_solve` function-level `## Compute` section at lines
128-135.

---

### WR-02: `assert_band` tolerance formula uses fixture magnitude, not `got` magnitude — test bands are vacuous for large-coef fixtures

**File:** `crates/mlrs-algos/tests/mbsgd_classifier_test.rs:112-114`

**Issue:** The band check is:

```rust
let abs_err = (g - e).abs();
assert!(abs_err <= band + band * e.abs(), …);
```

The allowed tolerance is `band * (1 + |e|)` where `e` is the *fixture* (expected) value. This
is a one-sided relative-error bound anchored on the oracle rather than on the computed result.
Two concrete problems:

1. **Vacuous for large expected values.** With `coef_` magnitudes ~28 (the optimal-schedule
   case) and `COEF_BAND_F32 = 1e-3`, the allowed absolute error per coefficient is up to
   `1e-3 + 1e-3 * 28 = 0.029`. This is wider than the 1e-5 project contract and wider than the
   ~2.8e-2 observed residual, giving the test essentially zero margin before it would start
   catching real regressions in the coef magnitude range.

2. **Zero `expected` corner.** If any fixture coefficient is exactly 0, the bound collapses to
   `band` (1e-3), which is fine numerically but inconsistent with the stated ~1e-5 project
   contract.

The COEF_BAND constants are documented as "relative-scaled: `band + band·|expected|`", so this
is intentional, but it means the actual precision gate on the coef is ~1e-3 relative at the
binding magnitude — correct but on the edge. A regression that shifts `coef_[j]` by 3e-2 at
magnitude 28 would still pass. The exact-label hard gate is the only strict correctness witness,
as the doc comment correctly states.

**Fix:** This is a design trade-off, not a simple line-fix. The minimum action is to add a
comment at `assert_band` explicitly stating that the tolerance is anchor-relative on `expected`
(not on `got`) and that large-magnitude fixtures dilate the allowed absolute error to
`band * (1 + |e_max|)`, so callers must account for this when choosing band values. If stricter
precision is needed, consider bounding on `max(|g|, |e|)` or the relative error directly.

```rust
// Tolerance: abs_err <= band * (1 + |expected|). Note: the absolute tolerance
// grows with |expected|; for coef magnitudes ~28 the effective abs tol is ~0.029.
// The exact-label hard gate is the strict correctness witness for large-step schedules.
```

---

### WR-03: `oracle_optimal_f32` omits fixture-length sanity assertions present in `oracle_optimal`

**File:** `crates/mlrs-algos/tests/mbsgd_classifier_test.rs:285-309`

**Issue:** The f64 test `oracle_optimal` (lines 316-346) includes:

```rust
assert_eq!(coef_ref.len(), N_FEATURES, "fixture coef length");
assert_eq!(intercept_ref.len(), 1, "fixture intercept length");
```

The f32 test `oracle_optimal_f32` (lines 285-309) does not. If the `_f32_` fixture is ever
regenerated with a different geometry (e.g., accidentally a scalar coef), `oracle_optimal_f32`
would silently pass the `assert_band` check via `assert_eq!(got.len(), expected.len())` in
`assert_band` — which would catch a length mismatch — but the test would produce no immediate
diagnostic at the fixture-load boundary, making debugging harder.

This is the same asymmetry present in the pre-existing `oracle_f32` vs `oracle` pair (which
also lacks the length asserts in the f32 case). The gap-closure plan introduced the new optimal
tests mirroring that asymmetry rather than correcting it.

**Fix:** Add the two fixture-length assertions to `oracle_optimal_f32` immediately after the
fixture loads:

```rust
let case = load_npz(fixture("mbsgd_classifier_optimal_f32_seed42.npz"))
    .expect("load mbsgd_classifier_optimal_f32");
let coef_ref = case.expect_f64("coef");
let intercept_ref = case.expect_f64("intercept");
assert_eq!(coef_ref.len(), N_FEATURES, "f32 optimal fixture coef length");   // ADD
assert_eq!(intercept_ref.len(), 1, "f32 optimal fixture intercept length");  // ADD
```

---

## Info

### IN-01: `l2_factor != 1.0` float-equality guard is imprecise (pre-existing, now in the guarded branch)

**File:** `crates/mlrs-backend/src/prims/sgd.rs:306`

**Issue:** The guard condition `l2_factor != 1.0` uses exact float equality. `l2_factor` is
computed as `(1.0 - (1.0 - l1_ratio) * eta * alpha).max(0.0)`. For a pure L2 penalty
(`l1_ratio = 0`) with `eta * alpha` very small but non-zero, `l2_factor` may equal `1.0`
exactly due to floating-point cancellation (the subtraction underflows to 0.0 in the mantissa),
causing the shrink to be silently skipped for that batch rather than applied at nearly-one.
Conversely, with exact arithmetic, `l2_factor == 1.0` implies a no-op shrink, so the guard is
correct in the limit. This is a pre-existing condition (unchanged from the original code) and
harmless in practice for the tested alpha=1e-4 range, but the guard's semantics are subtle.

**Fix:** Add a comment clarifying the intent:

```rust
// Guard: l2_factor == 1.0 (exact float) means the shrink is a no-op for this step.
// For small alpha/eta the subtraction may underflow to exactly 1.0, which is safe
// (skipping a near-1 multiply). For eta*alpha > 0 the factor is strictly < 1.
if params.alpha > 0.0 && l2_factor != 1.0 {
```

---

### IN-02: RESEARCH.md per-sample update sequence contradicts the implemented order (documentation artifact)

**File:** `.planning/phases/10-sgd-linear-svm/10-RESEARCH.md:346-347` (not a source file — info only)

**Issue:** RESEARCH.md lines 346-347 show:

```
w = w * max(0, 1 - (1-l1_ratio)*eta*alpha)   # L2 LAZY shrink — BEFORE gradient
w = w + update * x_i                          # gradient step
```

The implementation — and RESEARCH line 231 — both mandate the opposite order (gradient first,
then shrink). The SUMMARY acknowledges this as "an apparent internal tension" and confirms that
RESEARCH line 231 / the plan body are the controlling documents. The ~3e-7 abs error on the
constant schedule empirically validates the chosen order.

The risk is that a future developer reading the per-sample sequence table in RESEARCH.md will
re-implement the pre-fix order. This is a documentation-only issue; no source fix is needed
from this review, but the RESEARCH.md should be corrected in a future documentation pass.

**Fix (documentation, not source):** Update RESEARCH.md lines 346-347 to show the gradient
step before the L2 shrink, and annotate with a note citing RESEARCH line 231 as the authoritative
statement. Mark this as a future documentation cleanup task (out of scope for this gap-closure plan
per the plan's explicit deferral).

---

_Reviewed: 2026-06-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
