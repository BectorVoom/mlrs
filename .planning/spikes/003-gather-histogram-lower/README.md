---
spike: 003
name: gather-histogram-lower
type: standard
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
requirement: TREE-01
validates: "Given samples with node labels + per-feature bins and a target y, when a single-owner GATHER histogram kernel (one unit per (node,feature,bin) cell, same-iteration count + value-sum) is launched under --features cpu, then it lowers under cpu-MLIR and value-correctly reads back per-cell counts/value-sums vs a host oracle (f64 + f32) — answering abort signal A1"
verdict: VALIDATED
abort_signal: A1
related: [004, 005, 006]
tags: [tree, randomforest, cpu-mlir, histogram, gather, TREE-01, A1]
---

# Spike 003: Single-Owner GATHER Histogram Lowering (A1)

## What This Validates

**Given** `n_samples` rows each with a current node label, a per-feature bin index, and a target
`y`, **when** the histogram kernel — one unit per `(node, feature, bin)` cell, accumulating a count
and a `y` value-sum **in the SAME loop iteration** — is launched under `--features cpu` (cpu-MLIR,
the f64 correctness gate), **then** it lowers and reads back the **value-correct** per-cell count +
value-sum vs an in-test host oracle (f64 AND f32). This is **abort signal A1** of TREE-01.

This is the named cpu-MLIR keystone unknown for the tree chain: the GATHER histogram is the
innermost device step of every tree build, and a histogram is the canonical place a naive
implementation reaches for SharedMemory / atomic scatter-add (both banned, both panic at launch).
If the single-owner GATHER form does not lower, the whole RandomForest GPU path is infeasible as
specified.

## Result

**VERDICT: VALIDATED ✓ — A1 PASS.**

- The histogram lowered **cleanly on the first attempt** as the 2D guarded `ABSOLUTE_POS_X/Y` shape
  (X = `node*n_feat + feature`, Y = `bin`) — the proven `mlrs_kernels::manhattan_dist` geometry,
  16×16 `CubeDim`, ceiling-div counts. **Open Question 2 (histogram cube-indexing shape) RESOLVED:**
  the linearized per-cell `CUBE_POS_X` fallback was NOT needed.
- The probe asserts a **non-zero** read-back first (positive 002-A all-zeros guard — a kernel that
  never launched reads back zeros), then VALUE-asserts every cell's count and value-sum against an
  independent host recompute. f64 (correctness gate) and f32 both green.
- Single-owner = one writer per cell, so **no contention, no scatter-add, no atomics** — and both the
  node-label read and the feature-bin read happen in the SAME iteration, so **no 002-B cross-sibling
  loop**.

## cpu-MLIR Findings (for the Phase 18 prim re-author)

- **2D `ABSOLUTE_POS_X/Y` with two guards lowers** (`if nf < n_nodes*n_feat { if bin < n_bins { … } }`).
  No "operation with block successors" pass failure (the 002-A symptom is specific to bare 1D
  `ABSOLUTE_POS` per-row launches, not this 2D cell launch).
- A bounded `while s < n_samples { … s += 1 }` accumulator with `F`/`u32` accumulators and nested
  `if` guards lowers — consistent with Spike 001's feature-loop finding.
- **Histogram landmine avoided by design:** the single-owner GATHER never needs SharedMemory or
  atomics. Keep this form in `tree_hist`; do NOT introduce a shared-memory bin reduction.

## Source (verbatim — durable evidence, D-01)

- `kernels_and_harness.rs` — byte-identical copy of `crates/mlrs-backend/tests/tree_spike/mod.rs`
  (the shared run-vehicle module holding all three kernels + host launch wrappers + the
  `build_tree` loop). The histogram kernel is `tree_gather_histogram<F>` + `launch_histogram<F>`.
- `probes.rs` — byte-identical copy of `crates/mlrs-backend/tests/tree_spike_probes.rs`. The A1
  evidence is `check_histogram<F>` (+ the `tree_histogram_{f32,f64}_value_correct` entry points).

## How to Run (live gate)

```bash
cargo test -p mlrs-backend --features cpu --test tree_spike_probes -- --nocapture
```

The live `tree_spike/` + `tree_spike_probes.rs` files remain the runnable gate (NOT deleted — the
tree chain continues to use them); these `.planning/spikes/003-*/` copies are the durable artifact.
Phase 18 re-authors the production `tree_hist` prim from these findings + `spike-findings-mlrs`.
