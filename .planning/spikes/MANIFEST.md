# Spike Manifest

## Idea

Phase 13 feasibility keystone (PRIM-11): land a single shared, **multi-metric directed
KNN-graph primitive** in `mlrs-backend` — ascending-ordered k-nearest-neighbor `(indices,
distances)` of shape `(n, k)` — composed cpu-MLIR-safely from the launch-proven
`distance → top_k` GATHER path, with new **direct pairwise feature-loop distance kernels**
for Manhattan (L1), Chebyshev (L∞), and parameterized **Minkowski-p** (Euclidean/Cosine
reuse the v1 GEMM-expansion path). Standalone-validate the prim per metric **before** UMAP
(Phase 14) or HDBSCAN (Phase 15) consume it. The spike's job (per decision D-06) is to prove
the new direct distance kernels — **especially Minkowski-p's in-kernel `pow`** — and the
directed `distance → top_k → (n,k)` composition with index-identity self-drop all **launch
under `--features cpu`** (cpu-MLIR) and match a host/sklearn oracle.

## Requirements

Established design decisions (from Phase 13 CONTEXT.md; non-negotiable for the real build):

- **R-1 — Directed graph only.** The prim emits the directed `(indices, distances)` only;
  symmetrization is each consumer's job (UMAP t-conorm union / HDBSCAN mutual-reachability). (D-04)
- **R-2 — `include_self: bool`, one prim.** `true` → `top_k` of `k` returns self (dist 0) as
  neighbor 0 (HDBSCAN core-distance path); `false` → query `k+1` internally, drop the self
  column, return `k` true neighbors (UMAP path). Bookkeeping hidden in the prim. (D-01)
- **R-3 — Self-drop by INDEX IDENTITY.** When `include_self=false`, drop the neighbor whose
  returned index == the query row index, NOT "the first zero-distance entry" — robust against
  duplicate points at distance 0. Fallback: drop the last column if self isn't present. (D-02)
- **R-4 — Full fixed metric set.** Euclidean, Manhattan (L1), Cosine, Chebyshev (L∞),
  parameterized Minkowski-p — implemented and per-metric oracle-validated this phase. No
  custom/callable metrics. (D-05)
- **R-5 — cpu-MLIR-safe kernels.** No SharedMemory / Atomic / `F::INFINITY` / mutable-bool
  scan / descending-shift loop. GATHER idiom over the feature dim. Generic over `F` (`f32`/`f64`).
- **R-6 — Memory gate.** Query-axis tiled; never a full `n×n` distance matrix resident-and-leaking.
- **R-7 — New standalone prim fn** at `crates/mlrs-backend/src/prims/knn_graph.rs`, composing
  `distance.rs` + `topk.rs`; no estimator wrapper this phase. (D-03)
- **R-8 — Self-drop kernel authoring constraints (emerged from Spike 002).** The
  `include_self=false` self-drop kernel MUST (a) use the `top_k` launch shape (`row =
  CUBE_POS_X` + `UNIT_POS_X == 0` guard), NOT a bare-`ABSOLUTE_POS` 1D launch — the latter
  triggers a cpu-MLIR "operation with block successors" pass failure (FINDING 002-A); and (b)
  carry NO cross-sibling-loop mutable accumulator — a flag written in one `while` and read in a
  separate sibling `while` SILENTLY miscompiles (FINDING 002-B). Compute per-row positional
  values with a self-contained nested accumulate inside the output loop.
- **R-9 — Oracle gate must include a duplicate point + assert VALUES.** FINDING 002-B compiled,
  launched, and returned plausible wrong data; only an end-to-end value assertion against a
  brute-force/sklearn oracle (with rows at distance 0) caught it. The per-metric oracle test
  must include a duplicate-point row and assert returned indices/distances, not just non-panic.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | direct-feature-loop-distance-kernels | standard | Direct pairwise feature-loop kernels (Manhattan/Chebyshev/Minkowski-p incl. in-kernel `pow`) launch under cpu-MLIR and match a host reference ≤1e-6 | **VALIDATED ✓** | knn, cpu-mlir, kernel, minkowski, distance |
| 002 | directed-knn-compose-and-self-drop | standard | Directed `(indices,distances)` composition with `include_self` true/false and index-identity self-drop matches brute-force host KNN incl. duplicate-point/tie cases | **VALIDATED ✓** | knn, composition, self-drop, topk |

---

# Phase 17 — RandomForest GPU Histogram/Split Feasibility Spike (TREE-01)

## Idea

Phase 17 feasibility spike (TREE-01), the gate for the serial tree chain Phases 18→21: prove (or
refute) that a single-owner **GATHER histogram → seed-from-first split-find → relabel-partition**
tree-construction path **lowers and is tractable under cpu-MLIR** — with NO SharedMemory, NO atomic
scatter-add, NO `F::INFINITY` init, NO cross-sibling-loop accumulator — and that a single tree built by
composing those three kernels on **injected fixed bootstrap/feature indices** (RNG removed) reproduces
`sklearn.tree.DecisionTree*` EXACTLY (Tier-1 correctness witness = abort signal A5). The spike's three
kernels are recompositions of already-proven idioms (histogram = `manhattan_dist`'s accumulator;
split-find = `select_k`'s running best; relabel = `self_drop_gather`'s per-row shape), so the genuinely
unknown quantity is **cost (A3)**, not compilability. Deliverable: a `VERDICT.md` (GO/ADJUST/ABORT) with
abort signals A1–A5 evaluated, a finalized `SparseTreeNode` contract, and the two-tier stochastic-gate
convention — all gating whether Phases 18–21 proceed. Spike code is **evidence, not promoted code**
(D-01); Phase 18 re-authors the production prims from these findings.

**Verdict: GO** — A1/A2/A4/A5 PASS, A3 tractable (sub-second/tree, D-06 levers as headroom). See
`.planning/phases/17-randomforest-gpu-histogram-split-feasibility-spike-gating/VERDICT.md`.

## Non-Negotiable Design Decisions (binding for Phases 18–21)

- **SparseTreeNode contract — FINALIZED.** `SparseTreeNode { colid:i32, threshold:F, left_child:i32,
  value:i32 }`. Leaf sentinel **`colid == -1`** (D-03; FIL stops on `colid < 0`); right child implicit
  **`right = left_child + 1`** (D-02); **`value` is an OFFSET into a shared leaf-value buffer**, not a
  scalar (D-04, multiclass-uniform). **cuML divergence:** cuML marks a leaf with `left_child == -1`;
  mlrs uses `colid == -1`. **Phase 20 FIL MUST bind to the mlrs convention.**
- **Two-tier stochastic-gate convention (milestone-wide standard).** Tier-1 (D-07): on injected fixed
  indices a single tree matches sklearn exactly (exact structure + ≤1e-5 leaf values, f64) — abort
  signal A5. Tier-2 (D-08): because SplitMix64 ≠ MT19937, forests are gated on **predictive band only**
  (accuracy / R² within ~0.02–0.05 of `sklearn.ensemble.RandomForest*`), never element-wise — documented
  now, governs Phase 19.
- **cpu-MLIR op-set (locked).** Single-owner GATHER histogram (one writer per cell, same-iteration
  reads); seed-from-candidate-0 argmax with `u32` admit/better flags (no `F::INFINITY`, no mutable
  `bool`); per-sample relabel self-overwrite GATHER (no scan/compaction). Banned: SharedMemory, Atomic,
  `F::INFINITY`, mutable-bool scans, cross-sibling-loop accumulators. Both 002-A (loud all-zeros) and
  002-B (silent cross-loop) avoided by construction and positively guarded.
- **Host quantile binning (D-10).** Bin edges computed on the host once per fit; device kernels consume
  binned `u32` only — NO on-device sort/scan (A2 mitigation). Default 128 bins; benchmark 64 vs 128 so
  the D-06 "fewer bins" lever has data.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 003 | gather-histogram-lower | standard | Single-owner GATHER histogram (one unit per (node,feature,bin), same-iteration count+value-sum) lowers under cpu-MLIR as the 2D `ABSOLUTE_POS_X/Y` shape and value-correctly reads back per-cell counts/value-sums vs a host oracle (f64+f32) — **abort signal A1** | **VALIDATED ✓** | tree, cpu-mlir, histogram, gather, A1 |
| 004 | seed-from-first-split-find | standard | Seed-from-candidate-0 gain argmax (no `F::INFINITY`, `u32` admit/better tie flags, no mutable bool) value-correctly emits best (col,bin,gain) and resolves a deliberate gain TIE → lowest feature index then lowest bin (f64+f32) — **abort signal A4** | **VALIDATED ✓** | tree, cpu-mlir, split-find, argmax, tie-break, A4 |
| 005 | relabel-partition | standard | Per-sample relabel-partition GATHER (one cube per sample, `CUBE_POS_X`/`UNIT_POS_X==0`, self-overwrite, no scan/compaction) moves each sample to left child or `left_child+1` (D-02) value-correctly | **VALIDATED ✓** | tree, cpu-mlir, relabel, partition, gather, D-02 |
| 006 | tier1-decisiontree-witness | standard | A single tree on injected fixed indices (D-07) reproduces `DecisionTreeClassifier(gini)` AND `DecisionTreeRegressor(squared_error)` EXACTLY (exact structure + ≤1e-5 leaf values) + adversarial pure-leaf/gain-tie backstop — **abort signal A5**; also carries the **A3** 64-vs-128 per-tree cost benchmark | **VALIDATED ✓** | tree, sklearn, witness, oracle, A5, A3 |

## Notes (Phase 17)

- **Run vehicle (same precedent as Phase 13):** the kernels launch as self-contained integration tests
  in `crates/mlrs-backend/tests/` (`tree_spike/mod.rs` + `tree_spike_probes.rs`, `tree_witness.rs`,
  `tree_bench.rs`) against `ActiveRuntime` with `--features cpu`. **Unlike Phase 13, these live test
  files are NOT deleted** — they remain the authoritative runnable gate for the tree chain. The
  `.planning/spikes/003–006-*/` directories hold byte-identical verbatim copies as the durable evidence
  artifact (D-01). f64 is the cpu correctness gate; every f64 probe carries `skip_f64_with_log()` (SKIPS
  with a log line on a no-f64 adapter, e.g. rocm); f32 always runs.
- **Source layout in each dir:** `kernels_and_harness.rs` = verbatim `tree_spike/mod.rs` (the three
  kernels + launch wrappers + `build_tree` loop); `probes.rs` = verbatim `tree_spike_probes.rs`
  (003/004/005); `witness.rs` = verbatim `tree_witness.rs` and `bench.rs` = verbatim `tree_bench.rs`
  (006).

---

## Notes (Phase 13)

- **Run vehicle:** CubeCL kernels can't launch from a standalone `.planning/spikes/` cargo
  project without replicating the whole cubecl/runtime/feature setup. Per the repo's
  launch-proven precedent (`crates/mlrs-backend/tests/spike_test.rs`), each spike runs as a
  self-contained temp test in `crates/mlrs-backend/tests/` against `ActiveRuntime` with
  `--features cpu`; the kernel + harness source is copied into `.planning/spikes/NNN-*/` as the
  durable artifact. Verification is stdout/test-assertion (a "does-it-launch / match-oracle"
  fact spike, not a feel spike).
- **Prior art consulted:** `poly_map` (`elementwise.rs:140`) already launches `F::powf`;
  jacobi/`rbf_map`/`kde_*` prove `.abs()`/`.sqrt()`/`F::exp`/running-max statement-form all lower
  under cpu-MLIR. The genuinely-new risk is a feature-dim accumulator loop + those ops inside a
  *single direct pairwise kernel* — which is exactly what these spikes isolate.
