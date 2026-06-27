---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
plan: 02
subsystem: tree-kernels-spike
tags: [tree, randomforest, cpu-mlir, histogram, split-find, relabel, TREE-01, spike, nyquist-wave-1]
requires:
  - crates/mlrs-kernels/src/distance.rs (manhattan_dist accumulator + self_drop_gather per-row shape)
  - crates/mlrs-kernels/src/topk.rs (select_k seed-from-candidate-0 argmax)
  - crates/mlrs-backend/src/capability.rs (skip_f64_with_log + active_backend_name)
  - crates/mlrs-backend/tests/self_drop_gather_test.rs (host byte-cast + 002-A guard idiom)
provides:
  - three cpu-MLIR-safe tree kernels (tree_gather_histogram / tree_split_find / tree_relabel_partition) + host launch wrappers
  - SparseTreeNode { colid:i32, threshold:F, left_child:i32, value:i32 } format contract (D-02/D-03/D-04)
  - host per-level build_tree loop composing the three kernels (Gini, host-precomputed bin edges, n_bins parameter)
  - per-kernel standalone VALUE-asserting probes (SC-1 / A1 / A4 evidence)
affects:
  - Plan 03 Tier-1 correctness witness (reuses build_tree + SparseTreeNode against sklearn fixtures)
  - Plan 04 64-vs-128-bin cost benchmark (reuses build_tree + the launch wrappers, drives n_bins)
tech-stack:
  added: []
  patterns:
    - single-owner GATHER histogram (one unit per (node,feature,bin), same-iteration count+value-sum)
    - seed-from-candidate-0 gain argmax with u32 admit/better tie flags (no F::INFINITY)
    - per-sample CUBE_POS_X/UNIT_POS_X==0 relabel GATHER (right child = left+1)
    - host-side u32-overflow launch-geometry validation before unsafe from_raw_parts
key-files:
  created:
    - crates/mlrs-backend/tests/tree_spike/mod.rs
    - crates/mlrs-backend/tests/tree_spike_probes.rs
  modified: []
decisions:
  - "Histogram 2D guarded ABSOLUTE_POS_X/Y (X=node*feat, Y=bin) lowered cleanly first try (Open Question 2 resolved — no linearized CUBE_POS_X fallback needed)"
  - "tree_relabel_partition is non-generic (all-u32 arrays); a split_active u32 flag array replaces a signed -1 leaf sentinel to avoid in-kernel signed value-casts"
  - "build_tree implements binary-classification Gini gain (computable from count + positive-sum alone), matching the histogram's two output buffers"
metrics:
  tasks: 2
  files: 2
  commits: 2
  duration_min: 18
status: complete
---

# Phase 17 Plan 02: Tree Spike Kernels + Build Loop Summary

GATHER-histogram, seed-from-first split-find, and relabel-partition kernels each standalone-launch
and VALUE-correctly compute under cpu-MLIR (f64 + f32), composed by a host per-level `build_tree`
loop emitting a flat `SparseTreeNode` array + shared leaf buffer — answering SC-1, A1, and A4 with
value evidence.

## What Was Built

- **`crates/mlrs-backend/tests/tree_spike/mod.rs`** (shared test-support module, ~520 lines):
  - `tree_gather_histogram<F>` — one unit per `(node,feature,bin)` cell; 2D guarded
    `ABSOLUTE_POS_X/Y` launch; in the SAME loop iteration reads the sample's node label and feature
    bin and accumulates a count and a `y` value-sum. Modelled on `manhattan_dist`.
  - `tree_split_find<F>` — one cube per node, unit 0; running-best gain argmax SEEDED from candidate
    0 (no `F::INFINITY`); `u32` admit/better flags resolve ties to lowest feature index then lowest
    bin. Modelled on `select_k`.
  - `tree_relabel_partition` (non-generic, all-`u32`) — one cube per sample
    (`CUBE_POS_X`/`UNIT_POS_X==0`); reads its node's split from per-node frontier arrays and
    overwrites its own label with `left_child` (go-left) or `left_child + 1` (go-right, D-02).
    Modelled on `self_drop_gather`.
  - Host launch wrappers (`launch_histogram` / `launch_split_find` / `launch_relabel`) cloning the
    `spike_test.rs` / `self_drop_gather_test.rs` boilerplate, each running `checked_mul` u32-overflow
    geometry validation BEFORE any `unsafe { ArrayArg::from_raw_parts }` (T-17-03 mitigation).
  - `SparseTreeNode<F> { colid:i32, threshold:F, left_child:i32, value:i32 }` with D-02 (right =
    left+1), D-03 (leaf sentinel `colid == -1`, diverging from cuML's `left_child==-1`), D-04
    (`value` = offset into a shared leaf buffer) documented.
  - `build_tree<F>` host per-level loop: host `while` bounded by `max_depth`, each level launches
    histogram → split-find → relabel, appends adjacent children (D-02), and marks leaves when pure /
    max-depth / `< min_samples`. Accepts host-precomputed quantile bin edges (D-10) and `n_bins` as a
    parameter (Plan 04 drives 64 vs 128).

- **`crates/mlrs-backend/tests/tree_spike_probes.rs`** (8 tests, all GREEN on cpu f64+f32):
  histogram (per-cell count/value-sum vs in-test oracle + 002-A all-zeros guard), split-find (gain
  TIE → lowest feature/bin, A4 value assert), relabel (exact left/right labels per D-02), and a
  `build_tree` end-to-end probe validating SparseTreeNode D-02/D-03/D-04. Every f64 probe
  early-returns via `capability::skip_f64_with_log()` with a backend/dtype log line; f32 always runs.

## Wrapper Signatures (Plans 03/04 call these)

```rust
pub fn launch_histogram<F>(node_id:&[u32], binned:&[u32], y:&[F],
    n_samples:usize, n_feat:usize, n_nodes:usize, n_bins:usize) -> (Vec<F>, Vec<F>); // (counts, vsums)

pub fn launch_split_find<F>(gain:&[F], col_of:&[u32], bin_of:&[u32],
    n_nodes:usize, n_candidates:usize) -> (Vec<F>, Vec<u32>, Vec<u32>); // (best_gain, best_col, best_bin)

pub fn launch_relabel(node_id:&[u32], binned:&[u32], split_active:&[u32], split_col:&[u32],
    split_bin:&[u32], left_child:&[u32], n_samples:usize, n_feat:usize) -> Vec<u32>; // relabeled node_id

pub fn build_tree<F>(binned:&[u32], y:&[F], bin_edges:&[Vec<f64>],
    n_samples:usize, n_feat:usize, n_bins:usize, max_depth:usize, min_samples:usize)
    -> (Vec<SparseTreeNode<F>>, Vec<f64>); // (flat nodes, leaf-value buffer)

pub fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64;
pub fn from_f64<F: bytemuck::Pod>(x: f64) -> F;
```

Histogram cell order: `((node*n_feat + feature)*n_bins + bin)`. Candidate index `c` maps
`feature = c/(n_bins-1)`, `bin = c%(n_bins-1)` (split-after-bin).

## Open Question 2 — histogram cube-indexing shape

**RESOLVED: 2D guarded `ABSOLUTE_POS_X/Y` lowered cleanly on the first attempt** (X over
`node*n_feat + feature`, Y over `bin`, 16×16 `CubeDim`, ceiling-div counts — the proven
`manhattan_dist` geometry). The linearized per-cell `CUBE_POS_X` fallback was NOT needed. f64 and f32
histogram probes both return value-correct read-back.

## 002-A / 002-B near-misses

None tripped. By construction:
- **002-A (loud, all-zeros read-back):** avoided by using the proven launch shapes — 2D
  `ABSOLUTE_POS_X/Y` for the histogram, `CUBE_POS_X`/`UNIT_POS_X==0` for split-find and relabel.
  Never a bare 1D `ABSOLUTE_POS`. The histogram and split-find probes each assert a non-zero
  read-back as a positive 002-A guard; all passed.
- **002-B (silent cross-loop miscompile):** avoided — the histogram reads node label and feature bin
  in the SAME loop iteration (no sibling-loop counter), and every probe VALUE-asserts the read-back
  against an in-test host oracle (never a bare non-panic). All value asserts passed.

## Deviations from Plan

None for Rules 1-4. Two in-scope authoring choices (recorded as decisions, not deviations):
`tree_relabel_partition` is non-generic and uses a `split_active` `u32` flag array instead of a
signed `-1` split-column sentinel, to keep the kernel strictly in the all-`u32` op-set and avoid
in-kernel signed value-casts (the authoring contract permits `as usize` only at the index boundary).

## Verification

- `cargo build -p mlrs-backend --features cpu --tests` → exit 0.
- `cargo test -p mlrs-backend --features cpu --test tree_spike_probes -- --nocapture` →
  `test result: ok. 8 passed; 0 failed`.
- f64 + f32 both run on cpu (the f64 correctness gate); f64 probes carry the
  `skip_f64_with_log()` early-return so they SKIP-with-log on a no-f64 adapter (rocm), f32 always runs.

## Known Stubs

None. The three kernels are live-launched and value-asserted; `build_tree` runs end-to-end. This is a
feasibility spike (D-01) — no production `src/` prim is written this phase (Phase 18 re-authors prims
from these findings), which is the intended scope, not a stub.

## Self-Check: PASSED

- `crates/mlrs-backend/tests/tree_spike/mod.rs` — FOUND
- `crates/mlrs-backend/tests/tree_spike_probes.rs` — FOUND
- Commit `da658f7` (Task 1) — FOUND
- Commit `04e4939` (Task 2) — FOUND
