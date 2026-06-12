---
phase: 02-core-compute-primitives
kind: review-gap-closure
date: 2026-06-12
review: 02-REVIEW.md
resolved: [CR-01, CR-02, CR-03, WR-01, WR-07, IN-02]
status: complete
backends_verified: [cpu, wgpu]
tolerance_changed: false
commits:
  - "ac93f8b fix(02): distance/covariance force internal Shared path; guard covariance divisor (CR-01/WR-01)"
  - "dde3f9e fix(02): reject empty-input reductions at boundary; op-correct empty identity (CR-03)"
  - "689cf5d fix(02): release transient scratch with true byte size; honest reuse gate (CR-02/WR-07/IN-02)"
---

# Phase 02 Code-Review Gap-Closure Summary

Closes the 3 Critical findings (plus the two entangled Warnings WR-01/WR-07 and
the cheap IN-02 loop-hoist) from `02-REVIEW.md`. Every fix keeps the full
cpu + wgpu test suite green and loosens NO numerical tolerance (`F32_TOL` /
`F64_TOL` untouched; the f64 paths remain `skip_f64_with_log`-gated).

## CR-01 — distance/covariance no longer panic on `ReducePath::Plane`

**Problem:** `distance`/`covariance` forwarded the caller's `ReducePath` into
their INTERNAL squared-norm / column-mean reductions. On a non-subgroup adapter
(e.g. cpu) the plane path returns `Ok(None)` (the D-03 skip contract), which the
composite `.expect(..)` unwrapped into a **panic**. Undetected because every test
hard-coded `ReducePath::Shared`.

**Fix:** The internal reduction is an implementation detail, so it now runs
unconditionally on the always-portable `ReducePath::Shared`. Because `path` then
became genuinely unused in both primitives, it was **removed from both public
signatures** (option (a) — no misleading parameter wired to a dead/panic path)
and all 5 call sites (all in tests) were updated.

**Regression tests added:**
- `distance_internal_norm_portable_no_plane_panic` (distance_test.rs)
- `covariance_internal_mean_portable_no_plane_panic` (covariance_test.rs)

Both run on the active backend, log `plane_supported`, and assert CORRECT
distances/covariance (vs the independent f64 host reference) without panicking —
which on the cpu interpreter (no subgroup support) exercises exactly the path
that previously crashed.

## CR-03 — empty-input reductions rejected at the boundary (no silent 0 / ±inf)

**Problem:** `reduce_segment` short-circuited ALL ops to `0` on `len == 0`, so
`min([])`/`max([])`/`l2_norm([])` returned a WRONG `0` (an empty min is `+inf`, an
empty max `-inf`), and `validate_matrix` accepted a `0 × cols` / `rows × 0`
matrix (`0 * cols == 0 == len`).

**Fix (PREFERRED — reject at boundary):**
- `validate_nonempty(len)` guards the full-array `sum`/`mean`/`min`/`max`/
  `l2_norm` and `argmin`/`argmax` entry points.
- `validate_matrix` now also rejects `rows == 0 || cols == 0` (axis-wise + the
  argmin/argmax_rows paths).
- covariance's `validate_geometry` rejects empty `a`.
All return `PrimError::ShapeMismatch`.

**Defense-in-depth:** the now-unreachable `reduce_segment` `len == 0` branch
returns the OP-CORRECT identity (`Sum`/`SumSq` → 0, `Min` → +inf, `Max` → -inf
via `f64_to_host`) rather than a blanket 0. This also keeps the WR-06 `s_host[0]`
coupling safe (the boundary rejects empty before `l2_norm`/`mean` index `[0]`).

**Regression test added:** `empty_reductions_rejected_at_boundary` (reduce_test.rs).

## WR-01 — covariance `n_samples - ddof <= 0` rejected (no silent inf)

`covariance::validate_geometry` now takes `ddof` and rejects
`(n_samples as i64) - (ddof as i64) <= 0` with
`PrimError::DimMismatch { dim: "n_samples-ddof", .. }` before any launch — so a
single-sample matrix with `ddof = 1` (divisor 0) or `ddof > n_samples` (negative
divisor) is a typed error instead of `1/0 = inf` scaling the whole Gram. Kept
consistent with the CR-03 empty-geometry rejection in the same validator.
**Regression test:** `covariance_rejects_zero_divisor_and_empty_geometry`.

## CR-02 — transient scratch released with TRUE byte size; honest reuse gate

**Problem:** every `pool.acquire(..)` for scratch/output was wrapped in a
`DeviceArray` (no `Drop`) and never released, so `PoolStats.live_bytes` /
`peak_bytes` grew monotonically and the only thing feeding the free-list was
`DeviceArray::from_host`'s `acquire`+`release` metering churn. The D-10 reuse
gate was therefore green for the WRONG reason.

**Mechanism added:** `DeviceArray::release_into(pool)` — files the handle under
the array's OWN `byte_size()` (`len * size_of::<F>()`, the true acquisition size)
and **consumes `self`**, so a released buffer cannot be read again (the type
system prevents read-after-release aliasing).

### Buffers now released (and why each is safe)

| Site | Buffer released | Last use before release | Why safe (not live / not returned) |
|------|-----------------|-------------------------|-------------------------------------|
| `covariance` | column means (`means_dev`, `n_features` elems) | consumed by the `center_columns` launch | never read again; not returned |
| `covariance` | centred copy (`centred_dev`, `n_samples*n_features`) | consumed by the GEMM (lhs+rhs) | GEMM output (`gram`) is a SEPARATE handle; not returned |
| `distance` | XYᵀ cross term (`xy`, `rows_x*rows_y`) | consumed by `dist_combine_clamp` | output buffer is a distinct handle; sqrt pass reads only the output |
| `distance` | both squared-norm vectors (`xnorm`, `ynorm`) | consumed by `dist_combine_clamp` | never read again; not returned |
| `reduce_segment` | each inter-pass partial (`cur_len * elem`) | read into the next pass's `out_handle` | only released when it is POOL scratch (a previous pass's output), NEVER the caller's `input_handle` nor the returned final partial |
| `row_reduce` / `column_reduce` | per-axis segment (`seg_dev`) + per-axis result (`reduced`) | seg consumed by `reduce_segment`; result read to host | both per-iteration transient; not returned |
| `argreduce` | per-cube value + index scratch | fully read back to host (`vals`/`idxs`) | argreduce returns a host `u32`, not a device buffer |

**NOT released (correctly):** the GEMM output buffer (`gemm.rs`) — it is the
caller's returned result (the caller owns it). Releasing it would hand the live
result to a later `acquire` → aliasing/corruption.

**Aliasing safety:** all releases happen AFTER the consuming kernel is launched.
mlrs uses one client/stream per pool, so any later kernel that reuses a released
buffer is submission-ordered after the kernel that consumed it (no async
read-after-write hazard). No released handle is ever the returned output or
otherwise referenced by a `DeviceArray` that outlives the call.

### WR-07 — release always uses the TRUE size

Every release routes through `release_into` (the array's own `byte_size()`) or,
in `reduce_segment`, the partial's true `cur_len * elem`. A guessed/wrong-size
release that would pollute the free-list is impossible by construction.

### Honest memory gate (gate 1 rewrite)

`memory_gate_reuse_bounded` now asserts, across `N` same-shape `distance` calls
threading one pool:

1. **`live_bytes` CONSERVES** — identical every steady-state iteration
   (`live_after[iter] == live_after[1]`). The new load-bearing honesty signal:
   removing the scratch releases makes `live_bytes` climb monotonically and this
   `assert_eq!` fails.
2. **`peak_bytes` PLATEAUS** — never rises after the warmup iteration.
3. **scratch reuse GROWS** — the per-iteration `reuses` delta is `> 0`, counting
   the released-then-reacquired scratch (not the one-off `from_host` input
   uploads, which happen once before the loop).

**Verified RED-if-removed:** temporarily disabling the three `distance` releases
made gate 1a fail with `iter 2 live_bytes=0 != baseline=8 — transient scratch is
NOT being released`. Restored and re-confirmed green. The gate can no longer pass
on `from_host` churn alone.

## IN-02 — loop-invariant plane-width query hoisted

`reduce_segment` previously called `capability::active_plane_width()` (which
builds/clones a client) inside its multi-pass loop. It is loop-invariant, so it
is now computed ONCE before the loop as a power-of-two cube floor
(`plane_cube_floor`). The broader "fresh client per call" smell in
`capability.rs` is left as documented v1-perf scope.

## Verification

All green, no tolerance loosened:

- `cargo build -p mlrs-kernels` (feature-free, D-13) — clean
- `cargo build -p mlrs-backend --features cpu` / `--features wgpu` — clean
- `cargo test -p mlrs-backend --features cpu` — distance (6), covariance (4),
  memory_gate (3), reduce (5, ~267s), bridge (9), capability (3), gemm (4),
  pipeline (3), pool (5), spike (6) — all pass
- `cargo test -p mlrs-backend --features wgpu` — all 11 test files pass
- `cargo test -p mlrs-core -p mlrs-kernels` — 11 core + kernels/doc tests pass

## Files touched

- `crates/mlrs-backend/src/device_array.rs` — `byte_size()`, `release_into()`
- `crates/mlrs-backend/src/prims/distance.rs` — CR-01 (path removal), CR-02 releases
- `crates/mlrs-backend/src/prims/covariance.rs` — CR-01, WR-01 + CR-03 guards, CR-02 releases
- `crates/mlrs-backend/src/prims/reduce.rs` — CR-03 boundary rejection + op-correct identity, CR-02 releases, IN-02 hoist
- `crates/mlrs-backend/tests/distance_test.rs` — call-site update + CR-01 regression
- `crates/mlrs-backend/tests/covariance_test.rs` — call-site update + CR-01/WR-01/CR-03 regressions
- `crates/mlrs-backend/tests/reduce_test.rs` — CR-03 regression
- `crates/mlrs-backend/tests/memory_gate_test.rs` — call-site updates + honest gate-1 rewrite
