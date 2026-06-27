---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
plan: 04
subsystem: tree-cost-benchmark
tags: [tree, randomforest, cpu-mlir, benchmark, cost, A3, TREE-01, spike, nyquist-wave-2]
requires:
  - crates/mlrs-backend/tests/tree_spike/mod.rs (build_tree + the three Plan-02 kernel launch wrappers + n_bins parameter)
  - crates/mlrs-backend/src/capability.rs (skip_f64_with_log + active_backend_name)
provides:
  - A3 per-tree COST evidence — wall-clock per depth-8 build on ~1000x20 at n_bins 64 AND 128
  - the 64-vs-128 wall-clock delta (D-06 "fewer bins" lever, data-backed)
  - a 250/500/1000 samples-scaling sweep printing the cost SHAPE (D-05)
  - a frontier-memory observation (histogram sized by cumulative node count, not active frontier — Pitfall 6 / D-06 "frontier-only" lever)
affects:
  - Plan 05 VERDICT.md (cites THESE numbers for the A3 abort-signal judgement)
tech-stack:
  added: []
  patterns:
    - plain std::time::Instant wall-clock probe (NOT Criterion) run targeted with --nocapture
    - deterministic splitmix64 host RNG (zero new packages)
    - host-precomputed per-feature quantile bin edges (D-10) + diagonal-boundary label for a deep staircase tree
key-files:
  created:
    - crates/mlrs-backend/tests/tree_bench.rs
  modified: []
decisions:
  - "Sweep reuses the headline 1000@128 timing instead of a redundant heavy build; f64 runs headline-only (64+128) to bound wall-clock since the cost SHAPE is dtype-independent"
  - "Label is a diagonal boundary over features 0/1/2 so the axis-aligned binned tree must staircase-approximate it and grows to a non-trivial depth (the timing is of a real build, guarded by a >3-node sanity assert)"
metrics:
  tasks: 1
  files: 1
  commits: 1
  duration_min: 14
status: complete
---

# Phase 17 Plan 04: A3 Per-Tree Cost Benchmark Summary

A `std::time::Instant` wall-clock probe (`tree_bench.rs`) drives the Plan-02 host
`build_tree` loop on the representative ≈1000×20×depth-8 load at `n_bins` 64 AND
128, printing per-build wall-clock, the 64-vs-128 delta, a samples-scaling sweep,
and a frontier-memory note — the measured A3 evidence the Plan-05 verdict cites
instead of "it compiled" (Pitfall 5).

## What Was Built

- **`crates/mlrs-backend/tests/tree_bench.rs`** (~290 lines, one `#[test]`):
  - `mod tree_spike;` reuses the Plan-02 `build_tree` + the three kernel launch
    wrappers (drives `n_bins` ∈ {64, 128}).
  - Deterministic `splitmix64` host RNG (no `rand` dep — zero new packages,
    T-17-SC) generates a binary-classification dataset whose label is a diagonal
    boundary over features 0/1/2, forcing the axis-aligned binned tree to
    staircase-approximate it (so it GROWS to depth — a real, non-trivial build).
  - Host-precomputed per-feature quantile bin edges (D-10, no on-device sort);
    raw values digitized into `0..n_bins`.
  - `timed_build` wraps `build_tree` in `Instant`, printing wall-clock + node /
    internal / leaf counts.
  - `run_bench<F>` records the headline 1000×20×depth-8 build at 128 and 64 bins,
    prints the 64-vs-128 delta, and (f32) runs a 250/500/1000 samples-scaling
    sweep at 128 bins printing time-ratio vs sample-ratio against the
    sub-quadratic threshold (samples-ratio²).
  - f32 always runs (full sweep); f64 is gated by `capability::skip_f64_with_log()`
    and runs headline-only (64+128) to bound wall-clock.
  - A non-no-op sanity assert (`nodes > 3`) confirms the timed build produced a
    real grown tree.

## A3 Measured Results (the raw VERDICT inputs)

Run: `cargo test -p mlrs-backend --features cpu --test tree_bench -- --nocapture`
(cpu-MLIR backend, f64 supported on this adapter so BOTH dtypes ran). Completed
in **~1s total** — comfortably targeted, no full-suite/disk hazard (T-17-08).

**f32 (always-on path):**

| Build (1000×20, depth 8) | Wall-clock | Nodes |
|--------------------------|-----------|-------|
| 128 bins (headline)      | 463.8 ms  | 121   |
| 64 bins (headline)       | 69.4 ms   | 123   |

- **64-vs-128 delta @ n=1000:** Δ(128−64) = **394.4 ms**, ratio **6.68×** —
  the D-06 "fewer bins" lever is strongly data-backed: halving bins cut wall-clock
  by ~6.7× on f32 here.
- **Samples-scaling sweep @ 128 bins (SHAPE):**
  - 250 → 39.5 ms (51 nodes)
  - 500 → 62.3 ms (69 nodes): samples ×2.0, **time ×1.58** (sub-quadratic; ≤ 4.0)
  - 1000 → 463.8 ms (121 nodes): samples ×2.0, **time ×7.45** (super-quadratic;
    exceeds the 4.0 threshold at the top of the range).

**f64 (cpu correctness gate, headline-only):**

| Build (1000×20, depth 8) | Wall-clock | Nodes |
|--------------------------|-----------|-------|
| 128 bins                 | 195.5 ms  | 121   |
| 64 bins                  | 75.9 ms   | 123   |

- **64-vs-128 delta @ n=1000:** Δ = **119.6 ms**, ratio **2.57×**.

### Cost-shape reading (for the VERDICT, not asserted as a gate)

The shape is **broadly tractable but NOT cleanly sub-quadratic at the top end**:
the 250→500 step is sub-quadratic (×1.58), but the 500→1000 step jumps ×7.45 at
128 bins. Two compounding causes, both expected and both pointing at the SAME
D-06 lever:
1. The tree grew more nodes at n=1000 (121 vs 69), and `build_tree` sizes the
   histogram by **cumulative node count**, so per-level cost scales with
   `nodes × features × bins × samples` — node growth multiplies the sample term.
2. 128 bins doubles the per-cell candidate work vs 64 (the 6.68× f32 delta).

Crucially this is sub-second per tree at the representative load on a CPU-MLIR
backend with the *un-optimized* cumulative-node scratch — i.e. A3 is **tractable
today and has obvious headroom**, not a pathological blow-up. The super-quadratic
top-end step is the data that justifies the D-06 ADJUST levers ("fewer bins" +
"frontier-only"), not an abort signal.

### Frontier-memory observation (Pitfall 6 → D-06 "frontier-only" lever)

`build_tree` launches the histogram sized by `nodes.len()` (the **cumulative**
node count so far), NOT the active frontier. At the headline 128-bin build the
peak scratch was ≈ **309,760 cells × 2 buffers** (count + vsum) = 121 nodes × 20
feat × 128 bins. So the D-06 "frontier-only" lever is a **genuine, unrealized**
future optimization: bounding the histogram to the active frontier (rather than
all nodes including finalized leaves) would cut both memory and the node-count
multiplier seen in the 500→1000 step.

## Deviations from Plan

None (Rules 1-4). The plan executed exactly as written. One in-scope authoring
choice (recorded as a decision, not a deviation): the f64 path runs headline-only
and the sweep reuses the headline 1000@128 timing, to keep total wall-clock ~1s
while still printing every required datum (the cost SHAPE is dtype-independent).

## Verification

- `cargo build -p mlrs-backend --features cpu --test tree_bench` → exit 0, no warnings.
- `cargo test -p mlrs-backend --features cpu --test tree_bench -- --nocapture` →
  `test result: ok. 1 passed; 0 failed`; prints 64-bin + 128-bin wall-clock +
  64-vs-128 delta + the samples-scaling sweep + the frontier-memory note for both
  f32 and f64. Completed in ~1s (targeted only — no full-suite/disk hazard).
- Non-no-op sanity assert (`nodes > 3`) passed: 121 / 123-node trees were timed.

## Known Stubs

None. The benchmark live-drives the real `build_tree` (the three cpu-MLIR kernels)
and the timing is guarded by a non-trivial-tree assert. This is a feasibility
spike (D-01) — no production `src/` prim is written this phase, which is the
intended scope, not a stub.

## Self-Check: PASSED

- `crates/mlrs-backend/tests/tree_bench.rs` — FOUND
- Commit `a95e40f` (Task 1) — FOUND
