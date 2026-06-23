---
phase: 13-knn-graph-primitive-feasibility-keystone
reviewed: 2026-06-23T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - crates/mlrs-kernels/src/distance.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-backend/src/prims/knn_graph.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/tests/knn_graph_test.rs
  - crates/mlrs-backend/tests/self_drop_gather_test.rs
  - scripts/gen_oracle.py
findings:
  critical: 0
  warning: 5
  info: 4
  total: 9
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-06-23
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed the KNN-graph primitive (PRIM-11): the three direct pairwise distance
kernels (`manhattan_dist` / `chebyshev_dist` / `minkowski_dist`) and the
`self_drop_gather` index-identity GATHER kernel in `mlrs-kernels`, the
`knn_graph` host orchestrator in `mlrs-backend`, the two test harnesses, and the
Phase-13 additions to the oracle generator.

The numerics and the cpu-MLIR authoring contract are followed carefully, and the
self-drop index-identity logic (the load-bearing 002-B catch) is correct under
trace. No BLOCKER-class correctness or security defect was found this pass:
geometry is validated host-side before every `unsafe` launch, all kernels
bounds-check their indices, the Euclidean/Cosine GEMM-vs-direct routing is
internally consistent (squared-select + boundary sqrt for L2, halved squared for
Cosine), and the over-fetch global-lexsort oracle rule is a sound independent
reference that resolves boundary-membership ties to the documented lowest-index
convention.

The findings below are robustness / API-safety / maintainability defects. The
most material is **WR-01**: the Minkowski exponent is carried in BOTH the
`Metric::Minkowski { p }` enum AND a separate `p: f64` parameter, but only the
separate parameter drives the kernel while only the enum's copy is partly
validated — the two can silently disagree and produce wrong-`p` distances for a
real caller (the tests never exercise the divergence).

## Warnings

### WR-01: Minkowski exponent is duplicated across two parameters; only one drives compute, allowing silent disagreement

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:295-299, 420-429`
**Issue:** `knn_graph` accepts the exponent twice: once inside the metric enum
(`Metric::Minkowski { p }`) and once as the standalone `p: f64` argument. The
compute path uses ONLY the standalone argument:
```rust
Metric::Minkowski { .. } => minkowski_dist::launch::<F, ActiveRuntime>(
    ..., f64_to_host::<F>(p), // <-- standalone arg, NOT the enum's p
),
```
while `validate_geometry` checks BOTH but never reconciles them:
```rust
if let Metric::Minkowski { p: mp } = metric {
    if !(mp >= 1.0) || !(p >= 1.0) { return Err(...); }
}
```
A caller passing `Metric::Minkowski { p: 3.0 }` with the standalone arg `p = 2.0`
passes validation (both ≥ 1) and silently computes L2 distances while the enum
(and any logging/serialization keyed on it) says L3. The test suite cannot catch
this because `metric_p()` extracts the enum's `p` and feeds it as the arg, so they
always coincide in tests — the divergence is unreachable by the tests but fully
reachable by callers.
**Fix:** Make the exponent single-source. Read the exponent from the enum inside
`compute_tile_distance` and drop the standalone `p` argument:
```rust
Metric::Minkowski { p } => minkowski_dist::launch::<F, ActiveRuntime>(
    ..., f64_to_host::<F>(p),
),
```
If the standalone parameter must stay for signature stability, reject mismatches
in `validate_geometry` (`(mp - p).abs() > 0.0` ⇒ `ShapeMismatch{operand:"p"}`).

### WR-02: `minkowski_dist` divides by `p` with no positive-`p` guard inside the public kernel

**File:** `crates/mlrs-kernels/src/distance.rs:160`
**Issue:** `let inv_p = F::new(1.0) / p;` assumes `p != 0`. The host validates
`p >= 1` in `knn_graph::validate_geometry`, but `minkowski_dist` is a public
`#[cube(launch)]` export (re-exported from `lib.rs:39`) and relies entirely on an
external invariant. A `p = 0` launch yields a division producing inf, then
`F::powf(acc, inv_p)` silently yields inf/NaN distances rather than a typed
error. Defense-in-depth: numerical safety should not depend on a caller in
another crate replicating the host guard.
**Fix:** Add an explicit precondition doc-comment (`p >= 1` is a caller
obligation) on `minkowski_dist`, and ensure every host launch path funnels
through the validated `knn_graph` entry so no caller can launch with unchecked
`p`.

### WR-03: `self_drop_gather` shift can read one slot past the row if the self index appears more than once

**File:** `crates/mlrs-kernels/src/distance.rs:207-220`; `crates/mlrs-backend/src/prims/knn_graph.rs:340-343`
**Issue:** The kernel computes `bump = count of in_idx[ibase..=ibase+s] == row`,
then `src = s + bump`, and reads `in_idx[(ibase + src)]`. The scheme is correct
ONLY when the self index occurs at most once in the `(k+1)`-wide window. For
X-vs-X self-query that holds, but the prim never asserts uniqueness of the self
index in the assembled top-k result. If self appeared twice (a future non-X-vs-X
caller, or a top-k miscompile feeding two equal indices), `bump` could reach 2
and `src = s + 2` reads `in_idx[ibase + k + 1]` — past the `k1 = k+1`-wide row,
and for the LAST row past the buffer end (`self_drop_full` sizes the `ArrayArg`
at exactly `n * k1`), an OOB device read.
**Fix:** Clamp the source index in-kernel so it can never leave the row:
```rust
let src = s + bump;
let src = if src < k1 { src } else { k1 - 1u32 };
```
and/or document the single-self-occurrence precondition at the prim boundary.

### WR-04: Overflow guard set does not cover the launch dims actually cast to `u32`

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:430-441, 448-457`
**Issue:** `validate_geometry` guards `n`, `d`, and `k+1` against `u32::MAX`, but
the launched values include the ceiling-division `(rows as u32) + bx - 1` in
`launch_dims_2d` and the `out_len = tile * n` element count. With `n` guarded and
`tile ≤ QUERY_TILE = 8` these never trigger for supported sizes, so this is
latent rather than reachable — but the validated set does not provably dominate
every later `as u32` cast, so a future size increase could wrap silently.
**Fix:** Either add a comment that the `n`-guard plus `tile ≤ 8` bounds all
derived launch dims, or extend the guard to the largest derived launch product so
validation provably dominates every cast.

### WR-05: Memory-gate test asserts EXACT byte equality across iterations — brittle, not a true leak bound

**File:** `crates/mlrs-backend/tests/knn_graph_test.rs:499-521`
**Issue:** The gate uses `assert_eq!(live_after[iter], live_baseline)` and
`assert_eq!(peak_after[iter], peak_baseline)`. Exact-equality on allocator byte
counts is correct ONLY if the pool is byte-for-byte deterministic per call. Any
benign future change (allocation rounding, a one-shot cached scratch on the 2nd
call, interleaved same-pool allocation) flips this to a hard red that does NOT
indicate a real leak. The file's own docstring (lines 448-450) says "Threshold
tuning is deferred to plan 13-03," yet the committed assertion is exact, not a
bound.
**Fix:** Assert conservation as a non-growth bound, which still catches leaks:
```rust
assert!(live_after[iter] <= live_baseline, "live_bytes grew → leak");
assert!(peak_after[iter] <= peak_baseline, "peak grew → scratch stacking");
```

## Info

### IN-01: Redundant re-read of `dup_row_a` / `dup_row_b` inside the per-row loop

**File:** `crates/mlrs-backend/tests/knn_graph_test.rs:413-414`
**Issue:** `dup_a` / `dup_b` are decoded from the npz case on every iteration of
`for row in 0..N`; they are loop-invariant.
**Fix:** Hoist the two `expect_f64(...).round() as usize` reads above the loop.

### IN-02: `include_self=true` test over-relaxes the col-0 assertion for `dup_a`

**File:** `crates/mlrs-backend/tests/knn_graph_test.rs:409-423`
**Issue:** The test treats both dup rows as "either self or its duplicate at
col 0." Under the lowest-index tie-break, dup_a (self idx 0) always wins col 0
over dup_b (idx 4) since 0 < 4, so for dup_a the stronger `self@col0` assertion
holds and is being skipped.
**Fix:** Optional — assert `self@col0` strictly for dup_a; keep the relaxed check
only for dup_b (self idx 4 vs dup idx 0 genuinely ties to the lower index).

### IN-03: 2D cube dims duplicated between kernel doc-comment and host constant

**File:** `crates/mlrs-kernels/src/distance.rs:27-29`; `crates/mlrs-backend/src/prims/knn_graph.rs:448-456`
**Issue:** The `16×16` cube dims appear in the kernel docs and as hard-coded
`bx = 16u32; by = 16u32;` in `launch_dims_2d`. The "these must match" contract
lives only in prose (the kernel guards bounds regardless, so they are safe to
differ — the doc just implies coupling).
**Fix:** A named `const CUBE_DIM_2D: u32 = 16;` referenced by both axes documents
the single source; no behavior change.

### IN-04: `QUERY_TILE = 8` magic constant with no in-code bound rationale

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:84`
**Issue:** The tile size is fixed at 8 with a long doc-comment but no in-code note
that `QUERY_TILE.min(n - r0)` (line 173) handles `n < 8` correctly, so a reader
must trace the clamp to confirm small-`n` safety.
**Fix:** None required for correctness; a one-line note that `tile` is clamped to
the remaining rows suffices.

---

_Reviewed: 2026-06-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
