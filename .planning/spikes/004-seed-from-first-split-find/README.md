---
spike: 004
name: seed-from-first-split-find
type: standard
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
requirement: TREE-01
validates: "Given per-(feature,bin) gains for a node, when a seed-from-candidate-0 argmax kernel (no F::INFINITY init, u32 admit/better tie flags, no mutable bool) is launched under --features cpu, then it value-correctly emits the best (col,bin,gain) and resolves a deliberate gain TIE to the lowest feature index then lowest bin (f64 + f32) — answering abort signal A4"
verdict: VALIDATED
abort_signal: A4
related: [003, 005, 006]
tags: [tree, randomforest, cpu-mlir, split-find, argmax, tie-break, TREE-01, A4]
---

# Spike 004: Seed-From-First Split-Find Argmax (A4)

## What This Validates

**Given** a node's `n_candidates` per-`(feature, bin)` gains, **when** the split-find kernel — running
best **seeded from candidate 0** (no floating-infinity init), updated with a statement-form `if`, ties
resolved via `u32` admit/better flags — is launched under `--features cpu`, **then** it value-correctly
emits the maximum-gain split and **resolves a deliberate gain TIE to the lowest feature index, then the
lowest bin** (f64 AND f32). This is **abort signal A4** of TREE-01.

The cpu-MLIR risk here is the argmax pattern: the obvious implementation seeds the running best with
`-F::INFINITY` (banned, panics at launch) and/or carries a mutable `bool` "found better" flag across a
sibling loop (002-B silent miscompile). A4 asks whether the gain argmax can be expressed **without**
either — modelled on the shipped `mlrs_kernels::select_k` running-best.

## Result

**VERDICT: VALIDATED ✓ — A4 PASS.**

- The argmax **seeds its running best from candidate 0** — no `F::INFINITY` init — and updates with a
  statement-form `if`. The tie-break is encoded with `u32` admit/better flags (`better = 1u32` when
  `g > best_gain`, or equal-gain with a lower feature index, or equal-gain-and-feature with a lower
  bin), never a mutable `bool`.
- The probe sets up a **deliberate gain TIE**: gain 0.5 at both (col0,bin1) and (col1,bin0). The kernel
  VALUE-asserts the winner is **(col 0, bin 1)** — lowest feature index, then lowest bin — exactly, on
  f64 AND f32, with a non-zero-gain 002-A guard. This is the same independent tie-break rule sklearn
  uses (lowest feature, then lowest threshold).

## cpu-MLIR Findings (for the Phase 18 prim re-author)

- **Seed-from-candidate-0 replaces the `-INF` sentinel** cleanly — the proven `select_k` idiom carries
  straight over to gain argmax.
- **`u32` admit/better flags replace a mutable `bool` scan.** Nested statement-form `if`/`else if`
  ladders lower; an `if`-expression in value position or a `bool` accumulator is the landmine.
- **VALUE-assert the tie, not just non-panic.** The tie case is exactly where a silent miscompile would
  pick the wrong (col,bin) while still returning a plausible gain; gate it with a known-answer tie
  fixture in the `best_split` prim oracle.

## Source (verbatim — durable evidence, D-01)

- `kernels_and_harness.rs` — byte-identical copy of `crates/mlrs-backend/tests/tree_spike/mod.rs`. The
  split-find kernel is `tree_split_find<F>` + `launch_split_find<F>`.
- `probes.rs` — byte-identical copy of `crates/mlrs-backend/tests/tree_spike_probes.rs`. The A4 evidence
  is `check_split_find<F>` (+ the `tree_splitfind_{f32,f64}_argmax_and_tie` entry points).

## How to Run (live gate)

```bash
cargo test -p mlrs-backend --features cpu --test tree_spike_probes -- --nocapture
```

The live test files remain the runnable gate; these copies are the durable artifact (D-01). Phase 18
re-authors the production `best_split` prim from these findings.
