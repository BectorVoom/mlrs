---
phase: 08-kernel-family
reviewed: 2026-06-21T00:00:00Z
depth: standard
files_reviewed: 20
files_reviewed_list:
  - crates/mlrs-algos/Cargo.toml
  - crates/mlrs-algos/src/density/kernel_density.rs
  - crates/mlrs-algos/src/density/mod.rs
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs
  - crates/mlrs-algos/src/kernel_ridge/mod.rs
  - crates/mlrs-algos/src/lib.rs
  - crates/mlrs-algos/src/traits.rs
  - crates/mlrs-algos/tests/kernel_density_test.rs
  - crates/mlrs-algos/tests/kernel_ridge_test.rs
  - crates/mlrs-backend/src/prims/kernel_matrix.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/tests/kernel_matrix_test.rs
  - crates/mlrs-kernels/src/elementwise.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-py/src/estimators/kernel.rs
  - crates/mlrs-py/src/estimators/mod.rs
  - crates/mlrs-py/src/lib.rs
  - crates/mlrs-py/tests/test_kernel.py
  - scripts/gen_oracle.py
findings:
  critical: 1
  warning: 7
  info: 4
  total: 12
status: issues_found
---

# Phase 8: Code Review Report

**Reviewed:** 2026-06-21
**Depth:** standard
**Files Reviewed:** 20
**Status:** issues_found

## Summary

Phase 8 adds the kernel family: the `kernel_matrix` backend primitive (PRIM-08), the
`KernelRidge` (KERNEL-01) and `KernelDensity` (KERNEL-02) estimators, the six KDE
density-value maps plus the three kernel-matrix maps in `mlrs-kernels`, the PyO3
wrappers, and the oracle generators/fixtures.

The numerical core is largely sound. I independently re-derived sklearn's KDE
`log_norm` formulas (`logVn`/`logSn`, gaussian/tophat/epanechnikov/exponential/linear
factors, and the cosine chain-rule series) against `_binary_tree.pxi` and they match
to f64 round-off at the tested dimension (d=3). The scott/silverman bandwidth closed
forms match sklearn `_kde.py`. The kernel-matrix base-op→in-place-map composition and
the validate-before-launch geometry guards are correct.

The one BLOCKER is a real correctness gap: `KernelRidge::fit` validates `degree >= 1`
**only** for the poly kernel, but the documented invariant and `AlgoError::InvalidDegree`
exist to reject untrusted hyperparameters before launch — and more importantly the
`alpha`/`degree` guards are bypassable in a way that lets a non-SPD or NaN-producing
configuration reach the device. The principal concrete defect is in the validation
ordering / completeness for KernelRidge described in CR-01. The remaining findings are
robustness, dead-code, and test-rigor warnings.

## Critical Issues

### CR-01: `KernelRidge::fit` skips the degree guard for non-poly kernels, but the degree is still consumed by `Kernel::Poly` construction only — and NaN poly predictions are not surfaced

**File:** `crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs:189-195`, `:233-245`, `:374-385`

**Issue:** Two related correctness problems converge here.

1. The poly kernel computes `F::powf(gamma * g + coef0, degree)` on device
   (`elementwise.rs:140`). When the GEMM Gram entry makes `gamma*g + coef0` negative and
   `degree` is a non-integer real (the type is `F`, sklearn `Interval(Real, 1, None)`),
   `powf` returns **NaN**. Those NaNs flow into `K`, then into the Cholesky factorization.
   A NaN on the diagonal does **not** reliably trip `PrimError::NotPositiveDefinite`
   (the pivot test is typically `pivot <= 0`, and `NaN <= 0` is `false`), so the solve
   can silently produce a NaN `dual_coef_` and NaN predictions instead of the documented
   typed error. The module doc (lines 35-39) claims "never a NaN `dual_coef_`," but the
   degree guard only rejects `degree < 1`, not the negative-base case, and there is no
   post-solve finiteness check.

2. The validate-before-launch contract (lines 177-208) is asymmetric: `alpha` is checked,
   `degree` is checked only for `Poly`, but the resolved `gamma` (which can be supplied by
   the user as any `F`, including `0`, negative, or non-finite via the Python `gamma`
   arg) is **never validated**. A user-supplied non-finite or pathological gamma reaches
   the device kernels unguarded — the same untrusted-hyperparameter class the
   T-08-03-01 / ASVS V5 guard block claims to cover.

**Fix:** Validate the full hyperparameter set before launch and add a finiteness guard on
the produced duals (cheap, n×t host read already happens for the accessor path):

```rust
// after resolving gamma, before building Kernel:
let gamma64 = host_to_f64(gamma);
if !gamma64.is_finite() {
    return Err(AlgoError::InvalidGamma { estimator: "kernel_ridge", gamma: gamma64 });
}
// ... after cholesky_solve, before storing:
let duals_host = dual_coef.to_host(pool);
if duals_host.iter().any(|&v| !host_to_f64(v).is_finite()) {
    return Err(AlgoError::Prim(PrimError::NotPositiveDefinite { /* ... */ }));
}
```

(If adding `InvalidGamma` is out of scope, at minimum add the post-solve finiteness check
so the doc's "never a NaN dual_coef_" guarantee actually holds.)

## Warnings

### WR-01: `score_samples` forces `ReducePath::Shared` and `.expect()`s the result — a brittle panic surface on a non-subgroup adapter

**File:** `crates/mlrs-algos/src/density/kernel_density.rs:321-329`

**Issue:** `row_reduce(.., ReducePath::Shared)?.expect("shared path is never plane-gated to None")`.
The invariant holds today (verified in `reduce.rs:192-218` — only the `Plane` path returns
`None`), so the `.expect()` is not currently reachable. But this couples the estimator to a
private contract of the reduce prim: if a future change makes the Shared path capability-gated
(e.g. the documented cpu-MLIR SharedMemory limitation `[cubecl-cpu-no-shared-memory]` is
enforced as a `None` return), this becomes a hard panic in library code instead of a typed
error. The reduce kernels here use `SharedMemory::<F>::new(256)` (`reduce.rs:96`), exactly the
construct the project memory warns can panic at launch on the cpu backend.

**Fix:** Convert the `None` case to a typed error rather than panicking:

```rust
let row_sum = row_reduce::<F>(pool, &dmat, n_query, n_samples, ScalarOp::Sum, ReducePath::Shared)?
    .ok_or(AlgoError::Prim(PrimError::Unsupported { /* reduce path unavailable */ }))?;
```

### WR-02: `n as u32` launch-dim cast can silently truncate for large element counts

**File:** `crates/mlrs-algos/src/density/kernel_density.rs:402-403`; `crates/mlrs-backend/src/prims/kernel_matrix.rs:232-233`

**Issue:** `let cubes = ((n as u32) + block - 1) / block;` casts `usize` → `u32` unchecked. For
`n > u32::MAX` (a query×sample product over ~4.29e9 elements) the cube count silently wraps,
under-provisioning threads so the trailing elements are never mapped — a silent wrong-result,
not a crash. The KDE/kernel-matrix problem sizes are small today, but this is an unguarded
truncation in a launch-config helper shared by two prims.

**Fix:** Guard the cast (or compute in `u64` and saturate), e.g.:

```rust
let cubes = u32::try_from((n + block as usize - 1) / block as usize)
    .expect("element count exceeds u32 launch-grid limit");
```

### WR-03: `KernelDensity::fit` does not reject non-finite or NaN bandwidth from the `Numeric` spec, and `score_samples` divides by `h`/`h*h` unguarded on device

**File:** `crates/mlrs-algos/src/density/kernel_density.rs:217-229`

**Issue:** The guard is `if !(bandwidth > 0.0)`, which correctly rejects `0`, negatives, and
`NaN` (since `NaN > 0.0` is `false`) — good. But `BandwidthSpec::Numeric(f64::INFINITY)`
passes (`inf > 0.0` is `true`). An infinite bandwidth then drives `-d * h.ln()` →
`-inf` in `kde_log_norm` and `exp(-0.5*sqdist/inf²)=exp(0)=1` on device, producing a
finite-but-meaningless log-density rather than a typed rejection. sklearn's
`Interval(Real, 0, None, closed='neither')` accepts any positive finite value but the
estimator should not silently accept `inf`.

**Fix:** Tighten the guard to require a finite positive bandwidth:

```rust
if !(bandwidth > 0.0 && bandwidth.is_finite()) {
    return Err(AlgoError::InvalidBandwidth { estimator: "kernel_density", bandwidth });
}
```

### WR-04: `div_by_row` kernel is exported but never launched — dead code

**File:** `crates/mlrs-kernels/src/elementwise.rs:301-313`; re-exported `crates/mlrs-kernels/src/lib.rs:26`

**Issue:** `div_by_row` (the log-sum-exp reduce-max rescale helper) is implemented, documented,
and publicly re-exported, but `grep` shows it is referenced only in comments
(`kernel_density.rs:317` says the rescale is "NOT needed"). It is genuine dead code added "for
the optional rescale step" that the implementation deliberately does not take. Dead public API
in the feature-free kernels crate adds maintenance surface and a misleading "this is wired up"
signal.

**Fix:** Either remove `div_by_row` (and its re-export) until the rescale path is actually
needed, or add a `#[doc(hidden)]` + an explicit comment that it is intentionally unwired and
unused by any current caller.

### WR-05: KernelRidge `assert_close` (test helper) does not handle non-finite values — a NaN prediction would surface as a confusing `allclose` failure, not a clear NaN report

**File:** `crates/mlrs-algos/tests/kernel_ridge_test.rs:78-99`

**Issue:** Unlike the KDE test's `assert_close` (which explicitly handles `±inf`/`NaN`,
`kernel_density_test.rs:122-128`), the KernelRidge test compares with a plain
`abs_err <= tol.abs + tol.rel*e.abs()`. If CR-01's NaN-poly path triggers, `(NaN - e).abs()`
is `NaN`, `NaN <= ...` is `false`, and the test fails with an opaque "allclose failed" message
rather than diagnosing the NaN. Given that the poly kernel is exercised here (`run_all_kernels`
includes `KernelKind::Poly`), the test should fail-loud on non-finite output.

**Fix:** Add a non-finite check before the tolerance compare that asserts both are finite (or
both equal), mirroring the KDE helper, so a regression in CR-01 is diagnosed precisely.

### WR-06: Python `predict_f32`/`predict_f64` and `score_samples_*` re-validate and re-upload `x` inside the GIL-released closure but read `self.inner` across the `py.detach` boundary without re-borrow safety notes — and the accessor `dual_coef_f32` locks the pool but never releases the read buffer

**File:** `crates/mlrs-py/src/estimators/kernel.rs:266-280`

**Issue:** `dual_coef_f32`/`_f64` lock the global pool and call `e.dual_coef(&pool)`, which does
`c.to_host(pool)` — a host materialization. This is fine functionally, but it is the only path
that does **not** run under `py.detach` (no GIL release) despite doing a device→host copy. A
large `dual_coef_` read blocks the GIL for the duration of the transfer. The PY-03 contract
("the device-compute body runs inside `py.detach`") is documented as load-bearing on
`crate::dispatch`; this accessor silently violates it.

**Fix:** Wrap the `to_host` in `py.detach` like the other device-touching methods, or document
explicitly why this accessor is exempt (small fixed-size read).

### WR-07: `KernelDensity::fit` / `KernelRidge::fit` round-trip the input `x` through host to make a private device copy, discarding the previous fitted `x_fit_` without `release_into`

**File:** `crates/mlrs-algos/src/density/kernel_density.rs:231-235`; `crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs:291-300`

**Issue:** Re-`fit`ting an already-fitted estimator overwrites `self.x_fit_ = Some(x_fit)`
(and `dual_coef_`) without first `release_into(pool)`-ing the prior device buffers. The old
allocations are dropped (returned to the allocator) instead of the pool free-list, so the
buffer-reuse memory contract the phase emphasizes is broken on the re-fit path. The per-prim
memory gate (`kernel_matrix_memory_gate`) only exercises the single-fit prim loop, not the
estimator re-fit path, so this is untested. (Pure memory-efficiency, but the project elevates
memory to a first-class correctness concern verified per phase.)

**Fix:** Before reassigning, `if let Some(old) = self.x_fit_.take() { old.release_into(pool); }`
(and likewise for `dual_coef_`). Add a re-fit memory-conservation assertion to the estimator
tests.

## Info

### IN-01: Stale "Wave-0 stub / todo!()" documentation left in shipped `kernel_matrix.rs`

**File:** `crates/mlrs-backend/src/prims/kernel_matrix.rs:31-37`, `:118-119`

**Issue:** The module doc still describes the file as "the 08-01 Wave-0 COMPILING STUB" with
"the compute path left as `todo!()`," and the `kernel_matrix` fn doc repeats "**Wave-0 stub:**
... the compute path is `todo!()`." The compute path is fully implemented (lines 154-193).
The misleading docs will confuse future readers into thinking the prim is unfinished.

**Fix:** Update the doc comments to reflect the implemented state.

### IN-02: Stale "`#[ignore]` Nyquist scaffold" documentation in active test files

**File:** `crates/mlrs-backend/tests/kernel_matrix_test.rs:2-13`, `:169-170`, `:185-186`, `:199-204`

**Issue:** The module/test docs say every function "is `#[ignore]`d and asserts ONLY that its
fixture loads," and that "Wave-1 (08-02) removes `#[ignore]`." The `#[ignore]` attributes are
in fact gone and the tests run the real `kernel_matrix` + value asserts. Same stale-scaffold
language appears in `kernel_density_test.rs:3-4` and `kernel_ridge_test.rs:2-3`.

**Fix:** Remove the "Wave-0 / `#[ignore]` scaffold" framing now that the tests are live.

### IN-03: `parse_kernel_kind` accepts `"polynomial"` as an alias, but no other layer (algos enum, gen_oracle, KdKernel) recognizes it — silent inconsistency

**File:** `crates/mlrs-py/src/estimators/kernel.rs:54`

**Issue:** `"poly" | "polynomial" => Ok(KernelKind::Poly)` adds an alias the rest of the stack
(and sklearn's `KernelRidge`, which uses `"polynomial"` for `pairwise_kernels` but `"poly"` is
the common form) does not document. Harmless, but the alias is undocumented in the estimator
docstring and untested, so its presence is accidental-looking.

**Fix:** Either document/test the alias or drop it for surface consistency with `parse_kd_kernel`
(which has no aliases).

### IN-04: `log_density_f32`/`log_density_f64` are pure aliases of `score_samples_*`, kept only "for accessor-name symmetry"

**File:** `crates/mlrs-py/src/estimators/kernel.rs:452-471`

**Issue:** These two methods exist solely to delegate to `score_samples_*` "for accessor-name
symmetry with the v2 dtype-suffixed precedent" (their own doc). They are duplicate public API
that recompute the full density on every call (no stored attribute), which is misleading — a
caller may reasonably expect a fitted-attribute accessor to be cheap. Code duplication with a
misleading name.

**Fix:** Remove the aliases (callers should use `score_samples_*`), or rename to make the
recompute-on-call cost explicit.

---

_Reviewed: 2026-06-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
