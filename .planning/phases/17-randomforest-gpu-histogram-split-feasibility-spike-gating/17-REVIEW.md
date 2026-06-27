---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
reviewed: 2026-06-27T05:25:54Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - scripts/gen_oracle.py
  - crates/mlrs-backend/tests/tree_spike/mod.rs
  - crates/mlrs-backend/tests/tree_spike_probes.rs
  - crates/mlrs-backend/tests/tree_witness.rs
  - crates/mlrs-backend/tests/tree_bench.rs
findings:
  critical: 0
  warning: 4
  info: 4
  total: 8
status: issues_found
---

# Phase 17: Code Review Report

**Reviewed:** 2026-06-27T05:25:54Z
**Depth:** standard
**Files Reviewed:** 5 (gen_oracle.py reviewed for the Phase-17 decision-tree additions only; the pre-existing fixture generators are out of phase scope)
**Status:** issues_found

## Summary

The phase delivers a self-contained RandomForest histogram/split feasibility spike: three cpu-MLIR-safe CubeCL kernels (`tree_gather_histogram`, `tree_split_find`, `tree_relabel_partition`), their host launch wrappers, a host per-level `build_tree` loop, value-asserting probes, a Tier-1 sklearn correctness witness, a wall-clock cost benchmark, and the matching numpy/sklearn oracle generators.

I traced the kernel math, the host build loop, the Gini/variance gain formulas, the tie-break logic, the launch-geometry guards, and the witness comparison logic end-to-end. **The core correctness path is sound** — the single-owner GATHER histogram writes every output cell exactly once (no uninitialized read-back), the seed-from-candidate-0 argmax implements the documented lowest-(feature,bin) tie-break correctly given the ascending candidate order, the relabel `bv > thr → right` is consistent with the host's `bins 0..=b → left` accumulation, every pushed node is finalized as either internal or leaf (no surviving placeholder), and empty-child splits collapse to zero gain so they are never chosen. The probes and witness use exact-value assertions (not bare non-panics) and correctly guard the documented 002-A all-zeros and 002-B silent-miscompile failure modes. No BLOCKER-class bug, security vulnerability, or data-loss path was found.

The findings below are maintainability and robustness concerns: a large duplicated build loop that creates a divergence hazard, an f32-quantized threshold used for exact partition routing in the witness, an undocumented hyperparameter inconsistency in the adversarial oracle generators, and a redundant histogram relaunch in the build tail.

## Warnings

### WR-01: `build_tree` and `build_tree_variance` are ~80% duplicated — divergence hazard

**File:** `crates/mlrs-backend/tests/tree_spike/mod.rs:494-666` and `crates/mlrs-backend/tests/tree_witness.rs:204-381`
**Issue:** The regression witness reimplements the entire per-level frontier loop (frontier init, candidate `col_of`/`bin_of` construction, leaf-placeholder seeding, per-node split/leaf decision, `next_frontier` adjacency push, the relabel call, and the post-loop "remaining frontier → leaves" cleanup) as a near-verbatim copy of `build_tree`, differing only in the gain formula (variance vs Gini) and a second histogram launch on `y^2`. This is intentional per the `build_tree_variance` doc comment (to avoid changing the shared signature Plan 04 depends on), but it is a real maintainability defect: any future correctness fix to the shared loop — adding `min_samples_leaf`, changing pure-node detection, or fixing the cumulative-vs-frontier histogram sizing flagged in `tree_bench.rs:228-241` — must be applied in two places or the classifier and regressor paths will silently diverge. The copies already differ in their pure-node test (`pos == 0.0 || pos == tot` vs `var(...) <= 1e-12`) and empty-child guard (`if tot > 0.0` vs `if lc > 0.0 && rc > 0.0`), so drift has already started.
**Fix:** Extract the per-level orchestration into one generic function parameterized by a gain closure and a leaf-value closure:
```rust
fn build_tree_generic<F, G, L>(/* shared args */, gain_fn: G, leaf_fn: L) -> (Vec<SparseTreeNode<F>>, Vec<f64>)
// build_tree          = build_tree_generic(.., gini_gain,     prob_leaf)
// build_tree_variance = build_tree_generic(.., variance_gain, mean_leaf)
```
so the histogram → split-find → relabel composition exists once.

### WR-02: Witness routes the decision-equivalence partition by an f32-quantized sklearn threshold

**File:** `crates/mlrs-backend/tests/tree_witness.rs:455-478` (and `sk_leaf` at 524-535)
**Issue:** For the f32 fixtures, `threshold` is stored in the `.npz` as f32 (`gen_decision_tree_clf`/`reg` cast every array through `c()` to the fixture dtype). The witness loads it via `case.expect_f64("threshold")` (an f32→f64 upcast that preserves the f32 quantization) and uses it for an EXACT partition comparison: `if v <= sk_thr { sk_l.push(r) }`, then `assert_eq!(my_l, sk_l)`. sklearn computes a threshold as the f64 midpoint of two adjacent feature values; quantizing that midpoint to f32 can round it onto one of the bracketing values, so a feature value `v` at such a collapsed boundary can flip `v <= sk_thr` relative to `v <= my_thr` (the f64 bin midpoint mlrs uses). That produces a spurious `decision-equivalence` failure that is an artifact of fixture quantization, not a kernel bug. The current fixtures happen to avoid adjacent-f32 boundaries so the test passes, but the gate is not robust by construction.
**Fix:** Store `threshold` (and ideally `X`) at full f64 in every fixture regardless of compute dtype — the decision-equivalence routing is a host-side f64 comparison and gains nothing from f32 truncation. Alternatively route `sk_l`/`sk_r` through the same binned representation mlrs uses so both sides share one quantization.

### WR-03: Adversarial oracle generators omit `max_depth`, diverging from the standard branch

**File:** `scripts/gen_oracle.py:3095` and `scripts/gen_oracle.py:3169-3171`
**Issue:** The `structure == "adversarial"` branches build `DecisionTreeClassifier(criterion="gini", random_state=seed)` and `DecisionTreeRegressor(criterion="squared_error", random_state=seed)` WITHOUT `max_depth=DT_MAX_DEPTH`, whereas the standard branch passes `max_depth=DT_MAX_DEPTH` (=4). The mlrs witness builds BOTH standard and adversarial trees with the same `MAX_DEPTH = 4` (`tree_witness.rs:84`, used for every `run_witness` call including `adversarial=true`). This works only because the adversarial design (two identical columns, perfectly separable target) yields a depth-1 tree that never reaches either cap — but the asymmetry is undocumented and silently couples the test's `assert_eq!(nodes.len(), sk_nodes)` gate to that data property. A deeper adversarial fixture would diverge (unbounded sklearn depth vs mlrs cap 4) for a reason unrelated to the kernels.
**Fix:** Pass `max_depth=DT_MAX_DEPTH` in the adversarial branch too (matching the standard branch and the witness builder), or add an explicit comment asserting the adversarial tree is depth-1 by construction and therefore depth-cap-independent.

### WR-04: `build_tree` relaunches a full cumulative-sized histogram for the max-depth leaf cleanup

**File:** `crates/mlrs-backend/tests/tree_spike/mod.rs:648-663` (mirrored at `tree_witness.rs:365-378`)
**Issue:** When the loop exits on `depth == max_depth` with a non-empty frontier, the cleanup launches `launch_histogram` again sized by `nodes.len()` (cumulative node count) only to recompute `tot`/`pos` for the few remaining frontier nodes from feature 0. The final in-loop level already computed those values and discarded them. Beyond the wasted launch, this reuses the cumulative-sizing pattern the bench (`tree_bench.rs:228-241`) identifies as the dominant scratch cost, so the cleanup amplifies the worst-case allocation at exactly the deepest level. Correctness is unaffected (the recomputed `tot`/`pos` are identical), but it is an avoidable redundant device launch in the build's hot tail.
**Fix:** Carry the final level's `counts`/`vsums` (or just per-frontier `tot`/`pos`) out of the loop and reuse them in the cleanup, or convert max-depth nodes to leaves inside the last loop iteration.

## Info

### IN-01: `peak_cells` memory note hardcodes `128` instead of the active bin count

**File:** `crates/mlrs-backend/tests/tree_bench.rs:232-235`
**Issue:** `let peak_cells = nodes_128 * n_feat * 128;` hardcodes the literal `128` in the printed frontier-memory diagnostic rather than a named constant. Correct today (the headline uses 128 bins) but brittle — if the headline bin count changes, the memory note silently misreports.
**Fix:** Bind `const HEADLINE_BINS: usize = 128;` and use it for both the `gen_dataset`/`timed_build` calls and the `peak_cells` computation.

### IN-02: `bin_edges` preallocation capacity is discarded by the `vec![..; n_feat]` clone

**File:** `crates/mlrs-backend/tests/tree_bench.rs:89`
**Issue:** `vec![Vec::with_capacity(n_bins - 1); n_feat]` clones the prototype `Vec` `n_feat` times; `Vec::clone` allocates only `len` (0) capacity, so the intended reservation is lost for every element and the later `push` loop reallocates from zero. Harmless (bench-only) but the optimization does not do what it reads as.
**Fix:** `let mut bin_edges: Vec<Vec<f64>> = (0..n_feat).map(|_| Vec::with_capacity(n_bins - 1)).collect();`

### IN-03: `launch_dims_2d` ceiling-div can overflow `u32` for `nx` near `u32::MAX`

**File:** `crates/mlrs-backend/tests/tree_spike/mod.rs:250-259`
**Issue:** `((nx as u32) + bx - 1) / bx` adds `bx - 1 (=15)` before dividing. `checked_mul` guarantees the cell PRODUCT `≤ u32::MAX`, but `nx` itself (= `n_nodes*n_feat`) is only bounded by `u32::MAX`, so `nx as u32 + 15` wraps if `nx > u32::MAX - 15`. Purely theoretical at this spike's scale, but the ceiling-div is not overflow-safe the way the host `checked_mul` guard implies.
**Fix:** Compute the ceiling in `usize` — `nx.div_ceil(bx as usize) as u32` — to avoid the u32 intermediate.

### IN-04: f32 oracle comparisons use `F64_TOL` rather than `F32_TOL`

**File:** `crates/mlrs-backend/tests/tree_witness.rs:434, 585, 804`
**Issue:** The witness compares both f32 and f64 builds against `&F64_TOL`. The two constants are identical today (`abs = rel = 1e-5`, `tolerance.rs:26-38`), so this is benign, but `F64_TOL`'s own doc comment states it is "kept as a separate constant so an f64 path can be tightened independently later." Tightening `F64_TOL` would silently impose the stricter (wrong) bound on the f32 path.
**Fix:** Select the tolerance by dtype (`if size_of::<F>() == 4 { &F32_TOL } else { &F64_TOL }`), matching the dtype tagging the file already does.

---

_Reviewed: 2026-06-27T05:25:54Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
