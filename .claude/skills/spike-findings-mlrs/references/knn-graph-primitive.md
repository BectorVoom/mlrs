# KNN-Graph Primitive (PRIM-11, Phase 13)

Implementation blueprint for the shared multi-metric directed KNN-graph primitive, proven by
spikes 001 + 002 under `--features cpu` (cpu-MLIR, f64). Follow this and the prim can be built
without re-spiking.

## Requirements (non-negotiable)

- **R-1** Emit the **directed** `(indices, distances)` `(n, k)` graph ONLY. No symmetrization —
  that is each consumer's job (UMAP t-conorm union / HDBSCAN mutual-reachability).
- **R-2** One prim, `include_self: bool`. `true` → `top_k(k)` returns self (dist 0) as a
  neighbour (HDBSCAN core-distance path). `false` → query `k+1` internally, drop the self
  column, return `k` true neighbours (UMAP path). Bookkeeping hidden in the prim.
- **R-3** Self-drop by **INDEX IDENTITY** (drop the neighbour whose returned index == query row
  index), NOT "first zero-distance" — robust to duplicate points at distance 0. Fallback: drop
  the last column if self isn't present.
- **R-4** Full fixed metric set: Euclidean, Manhattan (L1), Cosine, Chebyshev (L∞),
  parameterized Minkowski-p. No custom/callable metrics.
- **R-5** cpu-MLIR-safe kernels, generic over `F` (`f32`/`f64`), f64-on-rocm skips-with-log.
- **R-6** Query-axis-tiled memory gate — never a full `n×n` distance matrix resident-and-leaking.
- **R-7** New standalone prim fn `crates/mlrs-backend/src/prims/knn_graph.rs` composing
  `distance.rs` + `topk.rs`; no estimator wrapper this phase.
- **R-8 / R-9** See `cpu-mlir-kernel-authoring.md` (self-drop kernel constraints) and the
  duplicate-point oracle-gate requirement below.

## How to Build It

**Metric routing (proven in Spike 001 — one general kernel subsumes L1/L2 exactly, so
special-casing is an optimization, not correctness):**
- **Euclidean** → reuse the v1 GEMM-expansion `distance()` (`prims/distance.rs`), sqrt at the
  `top_k` boundary. Fast path.
- **Cosine** → same GEMM path on L2-normalized rows (`1 − x̂·ŷ`).
- **Manhattan / Chebyshev / Minkowski-p** → NEW direct pairwise feature-loop kernels (below).
  The GEMM-expansion is Euclidean-specific and cannot back these.

**Direct distance kernels** — one `#[cube(launch)]` per metric, one unit per output element
`(i,j)`, a runtime `while kk < cols` loop over the feature dim. Verbatim-proven shapes (from
`sources/001-.../kernels_and_harness.rs`):

```rust
// Manhattan: Σ|Δ|
let mut acc = F::from_int(0i64);
let mut kk = 0u32;
while kk < cols {
    acc += (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
    kk += 1u32;
}
out[(i * rows_y + j) as usize] = acc;

// Chebyshev: max|Δ| (running max via STATEMENT-form if; diffs ≥0 so seed 0 is correct)
let diff = (x[..] - y[..]).abs();
if diff > acc { acc = diff; }

// Minkowski-p: (Σ|Δ|^p)^(1/p) — in-kernel F::powf, proven to lower under cpu-MLIR
acc += F::powf(diff, p);              // per term
let inv_p = F::new(1.0) / p;
out[..] = F::powf(acc, inv_p);        // final root
```

Launch: 2D `CubeDim {x:16, y:16}`, ceiling-div over (rows_x, rows_y); `i = ABSOLUTE_POS_X`,
`j = ABSOLUTE_POS_Y`, guarded `if i < rows_x { if j < rows_y { … } }`. Validate `p ≥ 1`
host-side at the prim boundary (the kernel does not guard it).

**Composition (`include_self=false`, proven in Spike 002):**
1. `distance()` (or direct kernel) → `top_k(k+1, sqrt=true)` → ascending `(val, idx)` `(n, k+1)`.
2. `self_drop_gather` kernel → directed `(n, k)`. **Use the exact shape in
   `cpu-mlir-kernel-authoring.md`** — `CUBE_POS_X`/`UNIT_POS_X==0`, and compute the self-shift
   per output slot with a self-contained nested count (`src = s + #self-cols at cols ≤ s`). NO
   cross-sibling-loop accumulator.

**Composition (`include_self=true`):** just `top_k(k)`. No new kernel — self is naturally
present (HDBSCAN counts it).

## What to Avoid

- **Don't** special-case Euclidean=Minkowski(2)/Manhattan=Minkowski(1) for *correctness* — the
  general kernel reproduces them exactly (Spike 001 depth probe, ≤1e-9). Special-case only if
  profiling shows the GEMM path is worth it (it is, for Euclidean/Cosine).
- **Don't** drop self by "first zero-distance" — it removes the genuine neighbour when a
  duplicate point sits at distance 0 (Spike 002 adversarial case). Index identity only.
- **Don't** materialize the full `n×n` distance block (R-6). The kernels are per-output-element
  and tile trivially over the query (i) axis.

## Constraints

- cpu-MLIR f64 + rocm f32 gate; f64-on-rocm skips-with-log. Tests separated from source.
- Per-metric oracle vs `sklearn.neighbors.NearestNeighbors` with the matching metric: indices
  set-equal up to tie-ordering, distances ≤1e-5 f64, lowest-index tie-break.
- **Oracle gate MUST include a duplicate-point row and assert VALUES** (R-9): a silent
  miscompile in the self-drop (Spike 002-B) compiled, launched, and returned plausible wrong
  data — only an end-to-end value assertion with a distance-0 duplicate caught it.

## Origin

Synthesized from spikes: 001, 002 (both VALIDATED).
Source files: `sources/001-direct-feature-loop-distance-kernels/`,
`sources/002-directed-knn-compose-and-self-drop/`.
