---
title: v3 hard-algorithm backlog (deferred Tier-3)
date: 2026-06-14
context: /gsd-explore — Tier-3 algorithms deferred out of v2 breadth sweep
---

# v3 Hard-Algorithm Backlog

Deferred from v2 ([[../seeds/v2-breadth-roadmap]]) because each needs major new
infrastructure and several fight the CubeCL/cpu-MLIR constraints flagged in project memory
(cpu MLIR backend has no SharedMemory / no cross-unit atomics; f64 unsupported on rocm).

## Dependency-ordered

1. **RandomForest (Classifier + Regressor)** — the keystone. GPU decision-tree
   construction is the single biggest lift; histogram/split kernels need atomics or a
   GATHER redesign to satisfy cpu-MLIR. **Unblocks → FIL → TreeSHAP.**
   - FIL (Forest Inference Library): batched tree traversal; depends on a tree format.
   - TreeSHAP (explainer): SHAP values for tree models; depends on FIL/trees.

2. **UMAP** — needs a KNN graph (have NearestNeighbors) → fuzzy simplicial set →
   SGD-based low-dim layout optimization. Reference: umap-learn (CPU oracle).

3. **HDBSCAN** — mutual-reachability distance → MST → condensed cluster tree → stability
   extraction. Reference: hdbscan (CPU oracle). Shares KNN-graph work with UMAP.

4. **ARIMA / AutoARIMA** — Kalman filter + batched L-BFGS over many series; AutoARIMA adds
   order search. cuML uses a batched_lbfgs; mlrs has L-BFGS but not batched.

5. **Kernel SVC / SVR** — SMO (sequential minimal optimization) solver; the hardest solver
   to make GPU-friendly. LinearSVC/LinearSVR (v2 Tier-2) do NOT need SMO.

## Whole subsystems (scope each as its own milestone if pursued)

- **explainer / SHAP** — Kernel, Permutation, Tree explainers (Tree depends on #1).
- **genetic** — symbolic regression.
- **cuml.accel** — transparent sklearn/umap/hdbscan drop-in acceleration layer.
- **Dask multi-GPU** — the `*_mg` distributed variants of every estimator.
- **metrics / preprocessing / feature_extraction / model_selection** — sklearn-utility
  surface; large, mostly non-device or light-device.

## Notes

- RandomForest is the highest-demand Tier-3 item; if v3 goes "flagship" instead of
  breadth, it is the natural single focus (unblocks the most downstream surface).
- Before any tree work, spike GPU histogram/split under cpu-MLIR constraints — this is the
  make-or-break feasibility question.
