---
name: spike-findings-mlrs
description: Implementation blueprint from Phase 13 spike experiments. Proven recipes + cpu-MLIR landmines for building the mlrs KNN-graph primitive (PRIM-11) and authoring cubecl-cpu-safe kernels. Auto-loaded during mlrs implementation work involving kernels, distance/KNN, or cpu-MLIR.
---

<context>
## Project: mlrs (cuML in Rust)

mlrs is a ground-up Rust rewrite of RAPIDS cuML, with compute kernels written once in CubeCL
generic over float type (`f32`/`f64`) and runtime (cuda/rocm/wgpu/cpu). Phase 13 is the
feasibility keystone for v3.0: a shared multi-metric directed KNN-graph primitive (PRIM-11)
that UMAP (Phase 14) and HDBSCAN (Phase 15) consume.

These findings come from spiking Phase 13 under `--features cpu` (cpu-MLIR, the f64 gate).

Spike session wrapped: 2026-06-23.
</context>

<requirements>
## Requirements (non-negotiable design decisions from spiking)

- **R-1** KNN-graph prim emits the **directed** `(indices, distances)` `(n,k)` only;
  symmetrization is each consumer's job.
- **R-2** One prim, `include_self: bool` — `true`=self-counted (HDBSCAN core dist),
  `false`=query `k+1` + drop self (UMAP).
- **R-3** Self-drop by **index identity**, not first-zero-distance (duplicate-point robust).
- **R-4** Full metric set: Euclidean, Manhattan, Cosine, Chebyshev, Minkowski-p. No custom metrics.
- **R-5** cpu-MLIR-safe kernels, generic over `F`, f64-on-rocm skips-with-log.
- **R-6** Query-axis-tiled memory gate; never full `n×n` resident.
- **R-7** New standalone prim `crates/mlrs-backend/src/prims/knn_graph.rs`; no estimator wrapper.
- **R-8** Self-drop kernel: `CUBE_POS_X`/`UNIT_POS_X==0` launch shape; NO cross-sibling-loop
  accumulator (both are cpu-MLIR landmines — see reference).
- **R-9** Per-metric oracle gate MUST include a duplicate-point row and assert VALUES, not just
  non-panic (a silent miscompile passes a happy-path check).
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| KNN-graph primitive | references/knn-graph-primitive.md | Multi-metric directed prim is feasible; in-kernel `F::powf` (Minkowski-p) lowers under cpu-MLIR; one general kernel subsumes L1/L2; index-identity self-drop validated on duplicate points |
| cpu-MLIR kernel authoring | references/cpu-mlir-kernel-authoring.md | Proven op-set + two landmines: bare-`ABSOLUTE_POS` 1D launch → MLIR pass failure (002-A); cross-sibling-loop accumulator → SILENT miscompile (002-B) |

## Source Files

Original spike kernels + harnesses preserved in `sources/` (verbatim, runnable shape).
</findings_index>

<metadata>
## Processed Spikes

- 001-direct-feature-loop-distance-kernels (VALIDATED)
- 002-directed-knn-compose-and-self-drop (VALIDATED)
</metadata>
