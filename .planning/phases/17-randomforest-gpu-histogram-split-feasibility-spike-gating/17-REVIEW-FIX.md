---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
fixed_at: 2026-06-27T06:30:00Z
review_path: .planning/phases/17-randomforest-gpu-histogram-split-feasibility-spike-gating/17-REVIEW.md
iteration: 2
findings_in_scope: 8
fixed: 6
skipped: 2
status: partial
---

# Phase 17: Code Review Fix Report (Round 2)

**Fixed at:** 2026-06-27T06:30:00Z
**Source review:** .planning/phases/17-randomforest-gpu-histogram-split-feasibility-spike-gating/17-REVIEW.md
**Iteration:** 2

**Summary:**
- Findings in scope: 8 (fix_scope = all → 4 Warning + 4 Info)
- Fixed: 6 (WR-03, WR-04, IN-01, IN-02, IN-03, IN-04)
- Skipped: 2 (WR-01, WR-02 — both already resolved in round 1)

This is a round-2 pass over a re-issued 8-finding review. Two findings were
already substantially resolved by round-1 commits and are confirmed against the
current source, then skipped (not re-applied). The six net-new findings were
fixed and committed atomically.

All Rust fixes were verified by recompiling the affected test targets with the
`cpu` feature AND running the full Phase-17 cpu test suite after each
source-changing fix: 8 `tree_witness` + 8 `tree_spike_probes` + 1 `tree_bench`
(depth-8, which exercises the WR-04 cleanup path) — all green. The Python
generator change was syntax-checked and the affected adversarial fixtures were
regenerated in numpy 2.4.6 / sklearn 1.9.0; the committed `.npz` blobs are
byte-identical (no fixture drift).

## Fixed Issues

### WR-03: Adversarial oracle generators omit `max_depth`, diverging from the standard branch

**Files modified:** `scripts/gen_oracle.py`
**Commit:** de3b878
**Applied fix:** Added `max_depth=DT_MAX_DEPTH` to both the adversarial
`DecisionTreeClassifier(criterion="gini", ...)` and
`DecisionTreeRegressor(criterion="squared_error", ...)` constructors, matching
the standard branch and the witness builder (`tree_witness.rs` `MAX_DEPTH=4`),
with a comment noting the adversarial design is depth-1 by construction so the
cap is never reached. Reconciled with round-1 commit 57c9665 (which added the
load-bearing `tree_.feature[0] == 0` tie-break guards) — those guards are
preserved; only the missing depth cap was added.
**Fixture impact:** Regenerated `tree_dt_clf_adv_{f32,f64}_seed42.npz` and
`tree_dt_reg_adv_{f32,f64}_seed42.npz`. All four blobs are byte-identical to the
committed versions (`git status` clean) — the depth-1 adversarial tree is
unaffected by the cap, confirming the documented coupling. No fixture committed.

### WR-04: `build_tree` relaunched a full cumulative-sized histogram for the max-depth leaf cleanup

**Files modified:** `crates/mlrs-backend/tests/tree_spike/mod.rs`
**Commit:** 221bd07
**Applied fix:** In `build_tree_with` (the shared driver, post-ea659c8 refactor
— the `tree_witness.rs` mirror the review cites no longer exists), each splitting
level now derives its two children's `(tot, sum_y)` directly from that level's
histogram by summing the parent's per-bin cells over the same partition the
relabel applies (left = bins `0..=b`, right = remainder). These carried totals
flow into a `frontier_totals: Option<Vec<(f64,f64)>>` and are reused by the
max-depth cleanup, eliminating the redundant `launch_histogram` in the build's
hot tail. The degenerate `max_depth == 0` path (no level ran, root still the
frontier) falls back to a single histogram. A `debug_assert_eq!` guards the
carried-totals/frontier length invariant.
**Verification note (logic change):** This changes HOW the deepest-leaf totals
are computed (parent-histogram partition vs a relabel-then-rehistogram). The
derivation is provably the same partition, and the full sklearn witness
(`<=1e-5` leaf values, exact node/leaf counts, induced-partition equivalence)
plus the depth-8 bench pass unchanged — strong semantic validation. Recommend a
human glance at the bin-range derivation (`0..=b` left / remainder right) to
confirm it matches the relabel rule `bv > thr=b → right`.

### IN-01: `peak_cells` memory note hardcoded `128` instead of the active bin count

**Files modified:** `crates/mlrs-backend/tests/tree_bench.rs`
**Commit:** f829b4e (combined with IN-02 — same file)
**Applied fix:** Introduced `const HEADLINE_BINS: usize = 128;` in `run_bench`
and used it for the headline/sweep `gen_dataset`/`timed_build` calls and the
`peak_cells` computation + memory-note `println!`. The 64-bin delta calls remain
literal (they are deliberately the contrasting bin count). Bench output confirms
the note prints `128 bins` from the const.

### IN-02: `bin_edges` preallocation capacity discarded by the `vec![..; n_feat]` clone

**Files modified:** `crates/mlrs-backend/tests/tree_bench.rs`
**Commit:** f829b4e (combined with IN-01 — same file)
**Applied fix:** Replaced `vec![Vec::with_capacity(n_bins - 1); n_feat]` (which
clones the prototype, dropping the reservation since `Vec::clone` copies only
len) with `(0..n_feat).map(|_| Vec::with_capacity(n_bins - 1)).collect()`, so
each inner `Vec` keeps its reservation.

### IN-03: `launch_dims_2d` ceiling-div could overflow `u32` for `nx` near `u32::MAX`

**Files modified:** `crates/mlrs-backend/tests/tree_spike/mod.rs`
**Commit:** 6b502e5
**Applied fix:** Replaced `((nx as u32) + bx - 1) / bx` (and the `ny` twin) with
`nx.div_ceil(bx as usize) as u32`, computing the ceiling in `usize` so the
`+ (bx-1)` intermediate cannot wrap the u32 add when `nx = n_nodes*n_feat` is
within `bx-1` of `u32::MAX`.

### IN-04: f32 oracle comparisons used `F64_TOL` rather than `F32_TOL`

**Files modified:** `crates/mlrs-backend/tests/tree_witness.rs`
**Commit:** a01cbdb
**Applied fix:** Added a `tol::<F>()` helper that returns `&F32_TOL` for the f32
build and `&F64_TOL` for f64 (by `size_of::<F>()`), and routed all three
`assert_slice_close` call sites (`compare_rec`, `assert_function_equiv`,
`check_adversarial`) through it. Imported `Tolerance` and `F32_TOL`. The two
tolerances are identical today, so behaviour is unchanged, but a future
tightening of `F64_TOL` can no longer silently impose the stricter bound on the
f32 companion path.

## Skipped Issues

### WR-01: `build_tree` and `build_tree_variance` are ~80% duplicated — divergence hazard

**File:** `crates/mlrs-backend/tests/tree_spike/mod.rs` and `crates/mlrs-backend/tests/tree_witness.rs`
**Reason:** already-fixed in round 1 (commit ea659c8). Verified against current
source: the shared `build_tree_with::<F, G, L>(.., level_gain, leaf_value)` driver
exists (`tree_spike/mod.rs:583`) and holds the single copy of the
histogram → split-find → relabel skeleton (frontier init, adjacency D-02, leaf
sentinel D-03/D-04, `max_depth`/`min_samples` termination, relabel). Both
`build_tree` (gini, `tree_spike/mod.rs:562`) and `build_tree_variance` (variance,
`tree_witness.rs:322`) now drive it through gain/leaf closures — the duplicated
per-level loop the review flagged no longer exists. No re-application needed.

### WR-02: Witness routes the decision-equivalence partition by an f32-quantized sklearn threshold

**File:** `crates/mlrs-backend/tests/tree_witness.rs`
**Reason:** already-resolved/documented in round 1 (commit 197beac), matching the
review's own accepted remedy (document the f32 path as non-bit-exact). The
witness module doc (`tree_witness.rs:39-51`, "## The f32 path is a COMPANION
smoke check, NOT a bit-exact sklearn match") explicitly states the f32 run
reconstructs from f32-rounded `X`/`threshold` and could in principle flip a
sample sitting within ~1e-7 of a threshold, so it is a companion SMOKE check of
the kernel plumbing — not an independent bit-exact reproduction. No concrete
defect found in the current fixtures (the well-separated random-normal data
avoids adjacent-f32 boundaries), so per the round-2 instruction the comparison is
left unchanged. Documentation judged sufficient.

---

_Fixed: 2026-06-27T06:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
