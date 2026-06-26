---
phase: 13-knn-graph-primitive-feasibility-keystone
fixed_at: 2026-06-23T00:00:00Z
review_path: .planning/phases/13-knn-graph-primitive-feasibility-keystone/13-REVIEW.md
iteration: 1
findings_in_scope: 9
fixed: 9
skipped: 0
status: all_fixed
---

# Phase 13: Code Review Fix Report

**Fixed at:** 2026-06-23
**Source review:** .planning/phases/13-knn-graph-primitive-feasibility-keystone/13-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 9 (fix_scope = all — Warnings + Info)
- Fixed: 9
- Skipped: 0

All fixes were applied inside an isolated git worktree, verified by
`cargo check`/`cargo test` under the `cpu` feature (the cpu-MLIR f64 correctness
gate), and committed atomically per finding. The full KNN test suite
(`knn_graph_test` 14 tests + `self_drop_gather_test` 2 tests) is green after the
fixes — all metric oracles within 1e-5, the duplicate-point value gate (002-B),
and the R-6 memory gate.

## Fixed Issues

### WR-01: Minkowski exponent duplicated across two parameters; only one drove compute

**Files modified:** `crates/mlrs-backend/src/prims/knn_graph.rs`
**Commit:** 9d2eeee
**Applied fix:** Made the exponent single-source. The compute path
(`compute_tile_distance`) now reads `p` from the `Metric::Minkowski { p }` enum
(the single source of truth) instead of the standalone `p` argument, and the
now-unused `p` parameter was removed from `compute_tile_distance`. The standalone
`p` argument was kept on the public `knn_graph` signature for stability, and
`validate_geometry` was extended to reject any divergence between the enum-carried
`p` and the standalone `p` (`(mp - p).abs() > 0.0` ⇒ `ShapeMismatch{operand:"p"}`)
so a caller can no longer silently compute one exponent while logging/serializing
keyed on another. Existing tests pass the enum `p` as the standalone arg, so they
remain valid.

### WR-02: `minkowski_dist` divides by `p` with no positive-`p` guard inside the public kernel

**Files modified:** `crates/mlrs-kernels/src/distance.rs`
**Commit:** 0ee71f4
**Applied fix:** Added an explicit `# Precondition (caller obligation)` doc-comment
to `minkowski_dist` stating that `p >= 1` is a hard caller precondition validated
host-side (an in-kernel guard was deliberately NOT added — a branch there risks a
cpu-MLIR mis-lower, and the host already validates `p` typed), and that the only
supported launch path is the validated `knn_graph` entry. Documentation-only; no
behavior change.

### WR-03: `self_drop_gather` shift can read one slot past the row if the self index appears more than once

**Files modified:** `crates/mlrs-kernels/src/distance.rs`
**Commit:** 83db0b8
**Applied fix:** Clamped the source column in-kernel
(`let mut src = s + bump; if src >= k1 { src = k1 - 1u32; }`) so a self index that
unexpectedly appears more than once cannot push `src` past the `(k+1)`-wide row (an
OOB device read on the last row). Used the STATEMENT-form mutable-`if` guard idiom
(cpu-MLIR-safe, matching the existing chebyshev running-max pattern). Inert for the
single-self-occurrence X-vs-X invariant the tests exercise — defense-in-depth, not
a behavior change. `self_drop_gather` f32/f64 tests pass under cpu-MLIR.

### WR-04: Overflow guard set does not cover the launch dims actually cast to `u32`

**Files modified:** `crates/mlrs-backend/src/prims/knn_graph.rs`
**Commit:** 96968d6
**Applied fix:** Chose the documentation option (no behavior change). Added a
domination argument explaining that the guarded `n`/`d`/`k+1` dims provably bound
every later `as u32` cast for supported sizes (ceiling-div launch dims shrink rows,
`tile <= QUERY_TILE = 8`, element counts bounded by `n` and `k+1`), with a note to
re-derive if `QUERY_TILE` ever grows large. Also fixed a stale `WR-03` reference in
the existing guard comment (it concerns the overflow guard, i.e. WR-04).

### WR-05: Memory-gate test asserts EXACT byte equality across iterations

**Files modified:** `crates/mlrs-backend/tests/knn_graph_test.rs`
**Commit:** cd28c30
**Applied fix:** Replaced the two `assert_eq!` exact-byte-equality checks on
`live_bytes`/`peak_bytes` across iterations with `<=` non-growth bounds
(`live_after[iter] <= live_baseline`, `peak_after[iter] <= peak_baseline`). A real
leak still trips the bound (bytes climbing each call), while benign
allocator-rounding / one-shot-cached-scratch changes no longer false-red. The
memory gate test passes.

### IN-01: Redundant re-read of `dup_row_a` / `dup_row_b` inside the per-row loop

**Files modified:** `crates/mlrs-backend/tests/knn_graph_test.rs`
**Commit:** 3430dfe
**Applied fix:** Hoisted the two `expect_f64(...).round() as usize` reads above the
`for row in 0..N` loop (they are loop-invariant). Applied together with IN-02 since
both touch the same code region.

### IN-02: `include_self=true` test over-relaxes the col-0 assertion for `dup_a`

**Files modified:** `crates/mlrs-backend/tests/knn_graph_test.rs`
**Commit:** 3430dfe
**Applied fix:** Strengthened the duplicate-row col-0 assertion from "either self or
its duplicate is acceptable" (distance-only) to a strict index assertion. Under the
lowest-index tie-break, both members of the distance-0 duplicate pair pin the SAME
col-0 index — the lower of the pair (`dup_a.min(dup_b)`) — so the test now asserts
`got_idx[row*K] == dup_col0` for both duplicate rows (a fixture-driven, fully
general strengthening, verified green against the fixture). Committed with IN-01.

### IN-03: 2D cube dims duplicated between kernel doc-comment and host constant

**Files modified:** `crates/mlrs-backend/src/prims/knn_graph.rs`
**Commit:** 54af871
**Applied fix:** Introduced a named `const CUBE_DIM_2D: u32 = 16;` and referenced it
for both launch axes (`bx`/`by`) in `launch_dims_2d`, removing the twice-hard-coded
magic number. No behavior change; the kernel still bounds-checks regardless, so it
stays decoupled from the distance.rs doc-comment.

### IN-04: `QUERY_TILE = 8` magic constant with no in-code bound rationale

**Files modified:** `crates/mlrs-backend/src/prims/knn_graph.rs`
**Commit:** 53c3b9c
**Applied fix:** Added a one-line note to the `QUERY_TILE` doc-comment that
`QUERY_TILE.min(n - r0)` at the loop use site clamps the per-tile row count to the
remaining rows, so `n < QUERY_TILE` yields a single short tile and never an
over-read. Documentation-only.

## Skipped Issues

None — all 9 in-scope findings were fixed.

---

_Fixed: 2026-06-23_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
