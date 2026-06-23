# Spike Wrap-Up Summary

**Date:** 2026-06-23
**Spikes processed:** 2
**Feature areas:** KNN-graph primitive; cpu-MLIR kernel authoring
**Skill output:** `./.claude/skills/spike-findings-mlrs/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | direct-feature-loop-distance-kernels | standard | VALIDATED | KNN-graph primitive / cpu-MLIR kernel authoring |
| 002 | directed-knn-compose-and-self-drop | standard | VALIDATED | KNN-graph primitive / cpu-MLIR kernel authoring |

## Key Findings

- **Phase 13 (PRIM-11) is feasible as specified.** Every named D-06 unknown resolved under
  `--features cpu` (cpu-MLIR, f64): the new direct feature-loop distance kernels, **in-kernel
  `F::powf` for Minkowski-p**, and the directed `distance → top_k → (n,k)` composition with
  index-identity self-drop all launch and match a host/sklearn oracle.
- **One general Minkowski-p kernel subsumes L1 and L2 exactly** (≤1e-9) — fast-path
  special-casing is an optimization, not a correctness need. Keep Euclidean/Cosine on the GEMM
  path for speed; route Manhattan/Chebyshev/Minkowski-p through the general direct kernel.
- **Two cpu-MLIR landmines captured for the build** (the cheap-looking self-drop, not the new
  math kernels, produced both):
  - **002-A (loud):** bare-`ABSOLUTE_POS` 1D launch → `"operation with block successors must
    terminate its parent block"` MLIR pass failure. Use the `top_k` `CUBE_POS_X`/`UNIT_POS_X==0`
    shape.
  - **002-B (silent):** a cross-sibling-loop mutable accumulator compiled, launched, and
    returned plausible wrong data. Recompute per-row positional values with a self-contained
    nested accumulate. Only an end-to-end value assertion with a duplicate point caught it.
- **New build requirements** R-8 (self-drop kernel authoring) and R-9 (duplicate-point
  value-asserting oracle gate) added to the MANIFEST.

## Artifacts

- `.planning/spikes/MANIFEST.md` — idea, R-1…R-9, spike table (both VALIDATED).
- `.planning/spikes/CONVENTIONS.md` — cpu-MLIR-safe spike/kernel conventions.
- `.claude/skills/spike-findings-mlrs/` — SKILL.md + 2 reference blueprints + verbatim sources.
- Temp run-vehicle tests removed from `crates/mlrs-backend/tests/`; real kernels land in
  Phase 13 execution.
