---
spike: 006
name: tier1-decisiontree-witness
type: standard
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
requirement: TREE-01
validates: "Given injected fixed bootstrap/feature indices (RNG removed, D-07), when a single tree is built by composing the three cpu-MLIR kernels through the host build loop, then it reproduces sklearn.tree.DecisionTreeClassifier(gini) AND DecisionTreeRegressor(squared_error) EXACTLY — exact split structure + leaf values <=1e-5 (f64) — plus an adversarial pure-leaf+gain-tie backstop; answering abort signal A5. Also carries the A3 per-tree cost benchmark."
verdict: VALIDATED
abort_signal: A5
also_carries: A3
related: [003, 004, 005]
tags: [tree, randomforest, sklearn, witness, oracle, decision-equivalence, A5, A3, TREE-01]
---

# Spike 006: Tier-1 sklearn Correctness Witness (A5) + A3 Cost Benchmark

## What This Validates

**Given** injected fixed bootstrap + feature indices (RNG removed — D-07), **when** a single tree is
built by composing the three Plan-02 cpu-MLIR kernels (histogram / split-find / relabel) through the
host `build_tree` loop, **then** it reproduces **sklearn EXACTLY**: `DecisionTreeClassifier(gini)`
per-node (exact split feature + decision-equivalent routing + leaf values ≤1e-5) and
`DecisionTreeRegressor(squared_error)` as a function (identical induced partition + regression-mean
predictions ≤1e-5), with an **adversarial** forced-pure-leaf + gain-tie backstop for both. This is
**abort signal A5** — the real correctness gate (proves the histogram/gain/partition MATH is right, not
just that kernels launch).

This dir **also carries the A3 per-tree cost benchmark** (`bench.rs`), the one genuinely-unknown
quantity: a `std::time::Instant` probe timing the full depth-8 build on ≈1000×20 at 64 AND 128 bins.

## Result

**VERDICT: VALIDATED ✓ — A5 PASS, A3 PASS (tractable).**

### A5 (correctness)

| Witness | Backend / dtype | Result |
|---------|-----------------|--------|
| clf(gini), 9 nodes / 5 leaves | cpu f64 + f32 | GREEN — exact per-node feature, decision-equivalent, leaf values ≤1e-5 |
| reg(squared_error), 25 nodes / 13 leaves | cpu f64 + f32 | GREEN — counts exact, induced partition identical, predictions ≤1e-5 |
| clf adversarial (pure-leaf + tie) | cpu f64 + f32 | GREEN — 002-B backstop, tie → feature 0 (independent rule) |
| reg adversarial (pure-leaf + tie) | cpu f64 + f32 | GREEN — 002-B backstop, tie → feature 0 (independent rule) |

**A5 = NO ABORT.** With RNG removed (injected indices), a single tree reproduces sklearn. No silent
cpu-MLIR miscompile — the adversarial boundary fixture would have caught one and did not.

**Two resolved subtleties (carried to the VERDICT caveats):**
1. **Threshold = decision-equivalence, not raw float** (Open Question 1 / A2). Global-unique host
   midpoints route a node's samples identically to sklearn's node-local midpoints; gate the decision
   boundary, not the raw `threshold`.
2. **Regressor split-feature ties are sklearn-splitter-RNG** (Pitfall 4). At 2-sample regression nodes
   every feature ties on variance reduction; sklearn's `BestSplitter` shuffle breaks it. Gated on
   function-equivalence (partition + predictions), never on sklearn's RNG pick (that would be circular).
   The classifier had no such ties and passes the strict per-node feature lockstep.

### A3 (cost — `bench.rs`)

Per-tree build on ≈1000×20×depth-8, ~1s total (targeted; no full-suite/disk hazard):

| Build | f32 wall-clock | f64 wall-clock |
|-------|----------------|----------------|
| 128 bins (headline) | 463.8 ms (121 nodes) | 195.5 ms (121 nodes) |
| 64 bins (headline)  | 69.4 ms (123 nodes) | 75.9 ms (123 nodes) |

- **64-vs-128 delta:** 6.68× (f32) / 2.57× (f64) — the D-06 "fewer bins" lever is data-backed.
- **Samples sweep @128 bins:** 250→39.5 ms, 500→62.3 ms (×1.58 sub-quadratic), 1000→463.8 ms (×7.45
  super-quadratic at top end — driven by cumulative-node histogram scratch, the unrealized
  "frontier-only" lever).
- **Tractable:** sub-second per tree with an un-optimized scratch layout → A3 PASS per D-05. The
  super-quadratic top-end step justifies the D-06 levers as headroom, not an abort.

## cpu-MLIR / build Findings (for the Phase 18 prim re-author)

- **Regression variance = a SECOND histogram on `y²`.** `build_tree_variance` (witness-local) launches
  the SAME histogram kernel a second time on `y²` for per-cell sum-of-squares, then a host variance
  formula — the kernels under test are identical to the classifier path. Don't fork a regression kernel.
- **Frontier-memory lever is real:** `build_tree` sizes the histogram by cumulative node count
  (≈309,760-cell peak scratch at 128 bins). Phase 18 `tree_hist` should size by the active frontier.
- **Lockstep / function-equivalence, NOT array-index `assert_eq!`:** sklearn lays nodes depth-first (a
  parent's right child is not `left+1`) while mlrs lays children adjacent (D-02), so structural
  correspondence is gated by traversal (clf) / induced-partition equality (reg), per the research's D-02
  validation method.

## Source (verbatim — durable evidence, D-01)

- `witness.rs` — byte-identical copy of `crates/mlrs-backend/tests/tree_witness.rs` (the clf + reg +
  adversarial Tier-1 witness, incl. the witness-local `build_tree_variance`).
- `kernels_and_harness.rs` — byte-identical copy of `crates/mlrs-backend/tests/tree_spike/mod.rs` (the
  three kernels + the `build_tree` host loop the witness composes).
- `bench.rs` — byte-identical copy of `crates/mlrs-backend/tests/tree_bench.rs` (the A3 cost benchmark).

## How to Run (live gates)

```bash
cargo test -p mlrs-backend --features cpu --test tree_witness -- --nocapture
cargo test -p mlrs-backend --features cpu --test tree_bench -- --nocapture
```

The live test files remain the runnable gate; these copies are the durable artifact (D-01). The Tier-1
witness recipe (injected fixed indices → exact match) is the **milestone-wide D-07 standard** every tree
phase's correctness gate follows.
