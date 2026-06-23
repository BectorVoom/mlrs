# Phase 13: KNN-Graph Primitive (feasibility keystone) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-23
**Phase:** 13-knn-graph-primitive-feasibility-keystone
**Areas discussed:** Self-inclusion API, Prim layer & form, Symmetrization scope, Metric scope

---

## Self-inclusion API

| Option | Description | Selected |
|--------|-------------|----------|
| `include_self: bool` flag | One bool. true → top_k of k returns self (HDBSCAN core-distance); false → query k+1, drop self-column, return k true neighbors (UMAP). k+1/self-drop hidden in prim. | ✓ |
| Always include self, caller drops | Prim always returns raw top_k incl. self; consumers slice themselves. | |
| Two entry points | Separate self-included / self-excluded fns. | |

**User's choice:** `include_self: bool` flag
**Notes:** Single cleanest surface for both consumers.

### Follow-up: self-drop mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Drop by index identity | Drop neighbor whose returned INDEX == query row index; robust vs duplicate points at distance 0; fallback drop last column. | ✓ |
| Drop first zero-distance | Drop first entry with distance 0; WRONG when a duplicate coincides. | |
| Defer to research/spike | Lock intent, let spike confirm GATHER-safe mechanism. | |

**User's choice:** Drop by index identity
**Notes:** Correctness on tie-heavy / duplicate-point data; matches sklearn/umap-learn semantics.

---

## Prim layer & form

| Option | Description | Selected |
|--------|-------------|----------|
| New `mlrs-backend` prim fn | New `prims/knn_graph.rs` over distance+top_k; standalone fn, called from mlrs-algos. | ✓ |
| Extend v1 NearestNeighbors core | Reuse/extend the kneighbors core in mlrs-algos. | |
| New prim fn + thin estimator | Backend prim fn plus a builder-fronted estimator wrapper. | |

**User's choice:** New `mlrs-backend` prim fn
**Notes:** Primitive-first; prim layer is mlrs-backend; no estimator surface needed for a keystone.

---

## Symmetrization scope

| Option | Description | Selected |
|--------|-------------|----------|
| Directed only (no symmetrize) | Emit only directed (indices, distances); symmetrization deferred to UMAP/HDBSCAN consumers. | ✓ |
| Include a symmetrize map | Build a generic symmetrize/transpose-union into the prim now. | |
| Directed now, note as deferred | Directed-only + explicitly record deferral in CONTEXT. | |

**User's choice:** Directed only (no symmetrize)
**Notes:** Matches PRIM-11 success criteria (only (indices, distances) named). Narrows the spike — the "symmetrize-map" step is removed from Phase-13 spike scope. (Deferral still recorded in CONTEXT for clarity.)

---

## Metric scope

| Option | Description | Selected |
|--------|-------------|----------|
| Euclidean-only | Build over v1 squared-Euclidean prim only. | |
| Thread a metric param now | Add a metric param for future-proofing. | ✓ (then expanded) |

**User's choice (initial):** Thread a metric param now → on clarification, user chose to **implement extra metrics**, not just reserve the param.

### Follow-up: metric set

| Option | Description | Selected |
|--------|-------------|----------|
| Euclidean + Manhattan + Cosine | Three most common; one new direct kernel (Manhattan) + cosine GEMM path. | |
| + Chebyshev | Four metrics; two new direct kernels. | |
| Full set incl. Minkowski(p) | Euclidean, Manhattan, Cosine, Chebyshev, parameterized Minkowski-p (in-kernel pow). | ✓ |

**User's choice:** Full set incl. Minkowski(p) — implemented and per-metric oracle-validated this phase.
**Notes:** Deliberate scope expansion. User directed "change all document" — REQUIREMENTS.md (PRIM-11 + out-of-scope row), ROADMAP.md (Phase 13 goal/criteria/spike-flag/one-liner), and PROJECT.md (target-features + Active bullet) were all updated to reflect multi-metric + the new direct distance kernels + the re-scoped spike (drop symmetrize-map; add new-kernel/Minkowski-p `pow` cpu-MLIR feasibility). Technical caveat surfaced: the v1 GEMM-expansion is Euclidean-specific; Cosine reuses GEMM on L2-normalized rows; Manhattan/Chebyshev/Minkowski-p need new direct pairwise GATHER kernels, each proven cpu-MLIR-safe.

---

## Claude's Discretion

- Exact `Metric` enum shape; whether Euclidean=Minkowski(2)/Manhattan=Minkowski(1) special-case to fast paths or route through the general kernel.
- Query-axis tile size for the PoolStats memory gate.
- Final prim filename and public signature/param order.
- Minkowski-p `p` type (`F` vs `f64`) and its boundary validation (`p ≥ 1`).

## Deferred Ideas

- Graph symmetrization → UMAP/HDBSCAN consumer phases (14/15).
- Public `KNNGraph` estimator wrapper → later phase if ever wanted.
- Supervised/target-metric KNN, approximate/NN-Descent, native sparse path → already out of scope in REQUIREMENTS.md.
