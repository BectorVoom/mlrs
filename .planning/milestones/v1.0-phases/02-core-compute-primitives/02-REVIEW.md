---
phase: 02-core-compute-primitives
reviewed: 2026-06-12T00:00:00Z
depth: standard
files_reviewed: 22
files_reviewed_list:
  - crates/mlrs-backend/Cargo.toml
  - crates/mlrs-backend/src/capability.rs
  - crates/mlrs-backend/src/device_array.rs
  - crates/mlrs-backend/src/lib.rs
  - crates/mlrs-backend/src/pool.rs
  - crates/mlrs-backend/src/prims/covariance.rs
  - crates/mlrs-backend/src/prims/distance.rs
  - crates/mlrs-backend/src/prims/gemm.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/src/prims/reduce.rs
  - crates/mlrs-backend/tests/covariance_test.rs
  - crates/mlrs-backend/tests/distance_test.rs
  - crates/mlrs-backend/tests/gemm_test.rs
  - crates/mlrs-backend/tests/memory_gate_test.rs
  - crates/mlrs-backend/tests/reduce_test.rs
  - crates/mlrs-backend/tests/spike_test.rs
  - crates/mlrs-core/src/error.rs
  - crates/mlrs-core/src/lib.rs
  - crates/mlrs-kernels/src/elementwise.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-kernels/src/reduce.rs
  - scripts/gen_oracle.py
findings:
  critical: 3
  warning: 7
  info: 4
  total: 14
status: resolved
resolved:
  - CR-01
  - CR-02
  - CR-03
  - WR-01
  - WR-07
  - IN-02
resolved_count: 6
gap_closure: 2026-06-12
gap_closure_commits:
  - "ac93f8b fix(02): distance/covariance force internal Shared path; guard covariance divisor (CR-01/WR-01)"
  - "dde3f9e fix(02): reject empty-input reductions at boundary; op-correct empty identity (CR-03)"
  - "689cf5d fix(02): release transient scratch with true byte size; honest reuse gate (CR-02/WR-07/IN-02)"
---

# Phase 2: Code Review Report

**Reviewed:** 2026-06-12
**Depth:** standard
**Files Reviewed:** 22
**Status:** resolved (gap-closure 2026-06-12 — all 3 Critical + WR-01/WR-07 + IN-02 fixed; see per-finding RESOLVED notes and `02-REVIEW-FIX-SUMMARY.md`)

## Summary

Phase 2 implements the CubeCL compute primitives (GEMM wrap, dual-path reductions, GEMM-expansion distance, covariance) over a `BufferPool` + `DeviceArray` data plane. The shape-validation discipline (D-04) is consistently applied via `validate_geometry` / `validate_matrix` with `checked_mul`, and the kernels are uniformly bounds-checked (`tid < input.len()` / `i < rows && j < cols`), so the tampering surface (T-*-01) is well covered for the tested paths. The clamp-as-statement pattern (D-07) and the lowest-index argmin tie-break are correctly implemented.

However, adversarial tracing surfaced three correctness defects that the test suite does not exercise because every test passes `ReducePath::Shared` to the composite primitives and never drives `distance`/`covariance` on the plane path:

1. `distance` and `covariance` **panic** (not skip-with-log) when invoked with `ReducePath::Plane` on an adapter lacking subgroup support — the documented skip contract (D-03) is violated at the composite layer.
2. The `BufferPool` accounting is effectively dead for real scratch/output buffers: almost no `acquire` is ever paired with a `release`, so `live_bytes`/`peak_bytes` are meaningless and the free-list reuse the D-10 gate "proves" is actually driven by `from_host` metering churn, not scratch reuse.
3. Empty-input `min`/`max`/`l2_norm` silently return `0` instead of a correct identity, and `covariance` with `n_samples == ddof` produces `inf` with no validation.

The numerical-tolerance handling in tests is honest (abs bound never loosened; near-zero floor only relaxes the relative term), and no silent tolerance loosening was found.

## Critical Issues

### CR-01: `distance` / `covariance` panic on `ReducePath::Plane` without subgroup support

**Status:** RESOLVED (commit ac93f8b) — distance/covariance now force the always-portable `ReducePath::Shared` for their INTERNAL norm/mean term and removed the now-dead `path` parameter from both public signatures; regression tests (`distance_internal_norm_portable_no_plane_panic`, `covariance_internal_mean_portable_no_plane_panic`) prove correct results + no panic on a plane-less adapter.

**File:** `crates/mlrs-backend/src/prims/distance.rs:105-108`, `crates/mlrs-backend/src/prims/covariance.rs:96-97`
**Issue:** `row_reduce` and `column_reduce` honor the D-03 skip contract: when `path == Plane` and `capability::plane_supported()` is `false` they log a warning and `return Ok(None)` (reduce.rs:187-190 and 224-227). But the composite primitives unwrap that `None` with `.expect(...)`:

```rust
// distance.rs
let xnorm = row_reduce::<F>(pool, x, rows_x, cols, ScalarOp::SumSq, path)?
    .expect("row SumSq reduction is shared-path-backed (never plane-gated to None)");
// covariance.rs
let means_dev = column_reduce::<F>(pool, a, n_samples, n_features, ScalarOp::Mean, path)?
    .expect("column-mean reduction is shared-path-backed (never plane-gated to None)");
```

The `expect` comment asserts the reduction is "never plane-gated to None," but that is false: `row_reduce`/`column_reduce` gate on the *caller-supplied* `path`, and `distance`/`covariance` forward the user's `path` directly. On any adapter without plane support (e.g. the `cpu` backend, which `spike_subgroup_query_reports_support` confirms typically lacks plane ops), calling `distance(.., ReducePath::Plane, ..)` or `covariance(.., ReducePath::Plane)` will panic. This is undetected because every test (`distance_test.rs`, `covariance_test.rs`, `memory_gate_test.rs`) hard-codes `ReducePath::Shared`. The public API exposes `path` as a parameter, so a downstream caller (Plan 03 KMeans/DBSCAN) selecting `Plane` for the norm term will crash on cpu.

**Fix:** Either (a) force the always-portable shared path internally for the norm/mean term regardless of the caller's `path` (the reduction is an internal implementation detail of distance/covariance, so the public `path` choice is arguably leaking), or (b) propagate the skip honestly. Option (a):

```rust
// distance.rs — the norm term must always succeed; use Shared internally.
let xnorm = row_reduce::<F>(pool, x, rows_x, cols, ScalarOp::SumSq, ReducePath::Shared)?
    .expect("shared path is never plane-gated to None");
let ynorm = row_reduce::<F>(pool, y, rows_y, cols, ScalarOp::SumSq, ReducePath::Shared)?
    .expect("shared path is never plane-gated to None");
```

```rust
// covariance.rs
let means_dev = column_reduce::<F>(pool, a, n_samples, n_features, ScalarOp::Mean, ReducePath::Shared)?
    .expect("shared path is never plane-gated to None");
```

If the `path` parameter is meant to be live, add a test that drives `distance`/`covariance` with `ReducePath::Plane` on cpu and asserts a graceful skip (not a panic).

### CR-02: Acquired scratch/output buffers are never released — pool accounting and reuse are broken

**Status:** RESOLVED (commit 689cf5d) — added `DeviceArray::release_into(pool)` (files the handle under its own true `byte_size()`, consumes `self`); transient scratch is now released once its consuming kernel is launched in covariance (means + centred copy), distance (XYᵀ + both norms), `reduce_segment` (each inter-pass partial), row/column_reduce (per-axis segment + result), and argreduce (per-cube value/index) — never the returned output. Memory gate 1 was rewritten to assert `live_bytes` conservation + `peak_bytes` plateau + growing scratch-reuse, and was verified to go RED when the releases are removed.

**File:** `crates/mlrs-backend/src/prims/reduce.rs:432`, `crates/mlrs-backend/src/prims/gemm.rs:99`, `crates/mlrs-backend/src/prims/distance.rs:118`, `crates/mlrs-backend/src/prims/covariance.rs:106`
**Issue:** Every real buffer obtained via `pool.acquire(...)` for kernel scratch or output is wrapped in a `DeviceArray` (or kept as a loop handle) and **never returned via `pool.release(...)`**. `DeviceArray` has no `Drop` impl (device_array.rs has no `impl Drop`), so dropping a `DeviceArray` neither decrements `live_bytes` nor pushes the handle back onto the free-list. Concretely:

- `reduce_segment` (reduce.rs:432) acquires `out_handle` each pass; it becomes `cur_handle` next iteration or is returned — never released.
- `gemm` (gemm.rs:99) acquires the output buffer (None path) and returns it — never released.
- `distance` (distance.rs:118) and `covariance` (covariance.rs:106, the `centred_handle`) acquire scratch — never released.

Consequences:
1. `PoolStats.live_bytes` / `peak_bytes` grow monotonically and never reflect freed memory, so the D-05 memory-accounting they exist to surface is meaningless.
2. The free-list is fed *only* by `DeviceArray::from_host`'s metering `acquire`+`release` pair (device_array.rs:67-68). The "reuse" the D-10 gate 1 (`memory_gate_reuse_bounded`) claims to prove is therefore reuse of `from_host` metering handles, not reuse of actual scratch/output buffers — the gate is green for the wrong reason and would not catch a genuine scratch-reuse regression.

**Fix:** Give `DeviceArray` an explicit lifecycle tied to the pool, or release transient scratch before returning. Minimal correctness fix — release covariance's `centred_handle` and reduce's intermediate partials once consumed:

```rust
// reduce_segment: after a pass produces the next partials, the *input* of the
// next pass is the previous out; release the consumed cur_handle when it is a
// pool buffer (not the caller's input). Track ownership and release accordingly.
```

At minimum, document that `live_bytes`/`peak_bytes` are non-conserving and rewrite gate 1 to assert on a counter that actually measures scratch reuse, so the gate cannot pass on `from_host` churn alone.

### CR-03: Empty-input reductions return wrong identity; `min`/`max`/`l2_norm([])` yield `0`

**Status:** RESOLVED (commit dde3f9e + covariance arm in ac93f8b) — empty geometry is now rejected at the public boundary: `validate_nonempty` guards the full-array reductions/argreduce, `validate_matrix` rejects `rows==0 || cols==0` for axis-wise reductions, and covariance's `validate_geometry` rejects empty `a`. As defense-in-depth the now-unreachable `reduce_segment` `len==0` branch returns the OP-CORRECT identity (Sum/SumSq→0, Min→+inf, Max→-inf) instead of a blanket 0. Pinned by `empty_reductions_rejected_at_boundary`.

**File:** `crates/mlrs-backend/src/prims/reduce.rs:385-388`
**Issue:** `reduce_segment` short-circuits *all* ops on `len == 0` to `F::from_int(0)`:

```rust
if len == 0 {
    let zero = vec![F::from_int(0i64)];
    return Ok(Some(DeviceArray::from_host(pool, &zero)));
}
```

This is correct for `Sum`/`SumSq` (identity 0) but **wrong for `Min` and `Max`**: an empty minimum should be `+inf` (or an error), an empty maximum `-inf`. Returning `0` silently produces an incorrect result that will then propagate (e.g. an empty-segment row reduction feeding distance/covariance). Combined with the lack of a `rows > 0` / `cols > 0` precondition, a `0 × cols` or `rows × 0` matrix is accepted by `validate_matrix` (since `0 * cols == 0 == len`) and silently yields zeros.

**Fix:** Branch on the op for the empty case, or reject empty reductions:

```rust
if len == 0 {
    let identity = match op {
        Op::Sum | Op::SumSq => F::from_int(0i64),
        Op::Min => F::from_int(i64::MAX),   // or: return Err(PrimError::ShapeMismatch{..})
        Op::Max => F::from_int(i64::MIN),
    };
    return Ok(Some(DeviceArray::from_host(pool, &vec![identity])));
}
```

Prefer rejecting empty geometry at the public boundary (`validate_matrix` / the full-array entry points) so the ambiguity never reaches the kernel driver.

## Warnings

### WR-01: `covariance` with `n_samples == ddof` produces `inf` with no validation

**Status:** RESOLVED (commit ac93f8b) — `covariance::validate_geometry` now takes `ddof` and rejects `(n_samples as i64) - (ddof as i64) <= 0` with `PrimError::DimMismatch { dim: "n_samples-ddof", .. }` before any launch. Kept consistent with CR-03 (empty geometry rejected in the same validator). Pinned by `covariance_rejects_zero_divisor_and_empty_geometry`.

**File:** `crates/mlrs-backend/src/prims/covariance.rs:155-156`
**Issue:** `denom = (n_samples as i64) - (ddof as i64)` and `factor = recip(denom)`. When `n_samples == ddof` (e.g. a single-sample matrix with `ddof = 1`), `denom == 0` and `recip` computes `1.0 / 0.0 == inf`, scaling the entire Gram to `inf`/`NaN` with no error. The docstring claims "`> 0` for any valid covariance" but nothing enforces it.
**Fix:** Validate in `validate_geometry` (or before the scale): `if (n_samples as i64) - (ddof as i64) <= 0 { return Err(PrimError::DimMismatch { dim: "n_samples-ddof", lhs: n_samples, rhs: ddof as usize }); }`.

### WR-02: `gemm` computes `out_len = m * n` without overflow check

**File:** `crates/mlrs-backend/src/prims/gemm.rs:70`
**Issue:** `validate_geometry` uses `checked_mul` for `m*k` and `k2*n`, but the output length `let out_len = m * n;` (and the analogous `rows_x * rows_y` in distance.rs:114, `n_features * n_features` in covariance.rs:158) is an unchecked multiply. For pathological large dims this overflows in release builds (wrapping) and feeds a too-small allocation to `pool.acquire`, then an out-of-bounds device write. Inputs are validated, but `m*n` is a *new* product not covered by the input checks.
**Fix:** Use `checked_mul` for the output length and return `PrimError::ShapeMismatch` on overflow, consistent with the input-geometry checks.

### WR-03: `to_host` slices `view[..self.len]` without verifying read-back length

**File:** `crates/mlrs-backend/src/device_array.rs:120-121`
**Issue:** `let view: &[F] = bytemuck::cast_slice(&bytes); view[..self.len].to_vec()`. If the runtime ever returns fewer than `len` elements (a short read), `view[..self.len]` panics with an opaque slice-index panic rather than a diagnostic. The comment says this guards trailing padding, but it does not guard the *under*-read case.
**Fix:** `assert!(view.len() >= self.len, "device read-back returned {} elems, expected >= {}", view.len(), self.len);` before slicing, or return a `Result`.

### WR-04: `DeviceArray::from_raw` is `pub` and trusts an unchecked `len`

**File:** `crates/mlrs-backend/src/device_array.rs:93-100`
**Issue:** `from_raw(handle, len)` is public and the docstring states "callers MUST pass the true element count." A wrong `len` larger than the backing buffer makes a later `to_host` / `ArrayArg::from_raw_parts` read out of bounds (the exact T-04-01 hazard this type was designed to prevent). The tests (`memory_gate_test.rs:86`, etc.) rely on the caller getting this right. Since the buffer size is not carried by the CubeCL `Handle`, this cannot be checked, but the unchecked `pub` constructor widens the unsafe surface beyond the crate.
**Fix:** Mark `from_raw` `pub(crate)` (all current callers are in-crate prims/tests within the same crate) or gate it behind a feature, so external consumers cannot construct a length-lying array.

### WR-05: `reduce::min`/`max` propagate `min/max` OOB seeding assumption that the host pads with real data

**File:** `crates/mlrs-kernels/src/reduce.rs:239-243`, `198-202`
**Issue:** The min/max shared and plane kernels seed OOB lanes with `input[0]`. This is correct only because the host always launches with at least one valid lane and `input[0]` is in-bounds. But `reduce_segment` calls the kernel with `cur_len` that can be `0` only when guarded earlier — verified safe today. The fragility: any future caller that launches a min/max kernel over an empty array reads `input[0]` out of bounds with no bounds check on that specific access (it is unconditional when `ABSOLUTE_POS >= len`). The `len == 0` guard lives in the host driver, not the kernel.
**Fix:** Add a defensive `if input.len() > 0u32` around the `input[0usize]` seed, or document the kernel precondition `len >= 1` prominently and ensure every launch site enforces it.

### WR-06: `mean`/`l2_norm` index `s_host[0]` assuming a non-empty result

**File:** `crates/mlrs-backend/src/prims/reduce.rs:117-118`, `161-163`
**Issue:** `let s_host = summed.to_host(pool); let scaled = vec![s_host[0] ...]`. `summed` is the result of `reduce_segment`, which for `len == 0` returns a length-1 `[0]` array, so `s_host[0]` is safe today. But this is an invariant coupling between two functions; if CR-03 is fixed by returning an empty array for the degenerate case, `s_host[0]` panics. Flagging so the two fixes stay consistent.
**Fix:** Guard `s_host.first()` or fix in tandem with CR-03.

### WR-07: `BufferPool::release` accepts a size that may not match the handle's real size

**Status:** RESOLVED (commit 689cf5d) — every new release routes through `DeviceArray::release_into`, which releases under the array's own `byte_size()` = `len * size_of::<F>()` (the true acquisition size carried by the array), so a mismatched-size release is impossible by construction. The one raw `pool.release` in `reduce_segment` likewise uses the partial's true `cur_len * elem`.

**File:** `crates/mlrs-backend/src/pool.rs:130-133`
**Issue:** `release(handle, size_bytes)` files the handle under `size_bytes` with no verification that the handle was acquired at that size. A caller that releases a handle under the wrong key pollutes the free-list: a later `acquire(size_bytes)` hands back a buffer of a *different* real size, causing an over/under-read. The `saturating_sub` on `live_bytes` masks the accounting error but not the buffer-size mismatch. Given CR-02 (release is barely used), this is latent, but it becomes load-bearing once releases are added.
**Fix:** Carry the byte size in `DeviceArray` (it already knows `len * size_of::<F>()`) and release with the array's own size, or key the free-list off a handle that records its size, so a mismatched release is impossible.

## Info

### IN-01: `min`/`max` of empty also affects `l2_norm` semantics on degenerate rows

**File:** `crates/mlrs-backend/src/prims/reduce.rs:149-165`
**Issue:** Tied to CR-03: a row/column of length 0 produces `sqrt(0) = 0` for L2, which is arguably defensible, but the silent acceptance of zero-length axes should be a documented decision, not an accident of the `len == 0` short-circuit.
**Fix:** Document the zero-length-axis behavior or reject it at `validate_matrix`.

### IN-02: `feature_enabled` constructs a fresh client on every call

**Status:** RESOLVED (commit 689cf5d, loop-hoist arm only) — the loop-invariant `active_plane_width()` query is now computed ONCE before `reduce_segment`'s multi-pass loop (as a power-of-two cube floor) instead of every pass. The broader "fresh client per call" smell in `capability.rs` is left as documented v1-perf-scope (the facade contract is unchanged).

**File:** `crates/mlrs-backend/src/capability.rs:51-54`, `81-83`, `94-99`
**Issue:** `feature_enabled`, `plane_supported`, and `active_plane_width` each call `crate::runtime::active_client()`, building/cloning a client per invocation. `reduce_segment`'s plane path calls `active_plane_width()` inside the multi-pass loop (reduce.rs:412), repeating the query every pass. Out of v1 perf scope, but it is also a correctness smell if `active_client()` is not idempotent across calls.
**Fix:** Hoist the plane-width query out of the loop in `reduce_segment` (it is invariant across passes).

### IN-03: `host_to_f64` / `f64_to_host` / `recip` duplicate the same f32/f64 bytemuck dispatch

**File:** `crates/mlrs-backend/src/prims/reduce.rs:568-583`, `crates/mlrs-backend/src/prims/covariance.rs:223-230`, and the identical `f64_to_f`/`f_to_f64` in three test files
**Issue:** The 4-/8-byte `bytemuck::from_bytes` dispatch is copy-pasted across `reduce.rs`, `covariance.rs`, and `distance_test.rs`/`gemm_test.rs`/`reduce_test.rs`. A change to the policy (e.g. supporting f16) must be made in 6+ places.
**Fix:** Hoist a single `f_to_f64<F: Pod>` / `f64_to_f<F: Pod>` pair into `mlrs-core` and reuse it.

### IN-04: `gen_oracle.py` `gen_argmin_tie` ignores the seeded RNG it constructs

**File:** `scripts/gen_oracle.py:198-210`
**Issue:** `rng = np.random.default_rng(seed)` then `x = rng.integers(...)` — but every row is immediately overwritten with a hard-coded literal array (`x[0,:] = [...]`, ..., `x[3,:] = [...]`), so the seeded RNG draw is dead. The `seed` parameter has no effect on this fixture.
**Fix:** Remove the unused `rng.integers` draw (or genuinely plant ties into RNG-generated data) so the fixture's provenance is honest; the file name `..._seed42` implies seed dependence that does not exist.

---

_Reviewed: 2026-06-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
