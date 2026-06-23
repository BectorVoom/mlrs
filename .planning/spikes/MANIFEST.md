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

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | direct-feature-loop-distance-kernels | standard | Direct pairwise feature-loop kernels (Manhattan/Chebyshev/Minkowski-p incl. in-kernel `pow`) launch under cpu-MLIR and match a host reference ≤1e-6 | **VALIDATED ✓** | knn, cpu-mlir, kernel, minkowski, distance |
| 002 | directed-knn-compose-and-self-drop | standard | Directed `(indices,distances)` composition with `include_self` true/false and index-identity self-drop matches brute-force host KNN incl. duplicate-point/tie cases | PENDING | knn, composition, self-drop, topk |

## Notes

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
