# Phase 14: UMAP - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-23
**Phase:** 14-umap
**Areas discussed:** Metric surface, transform(X_new) fidelity, Property gate + reproducibility, a/b curve fit

---

## Metric surface

| Option | Description | Selected |
|--------|-------------|----------|
| All 5 (full prim set) | Expose euclidean, manhattan, cosine, chebyshev, minkowski-p; per-metric oracle + property-gated layout each | ✓ |
| Euclidean-only for v3 | Ship euclidean only, reject others with typed BuildError; smallest matrix | |
| Euclidean + cosine | The two GEMM-fast-path metrics only; middle-ground matrix | |

**User's choice:** All 5 (full prim set)
**Notes:** Carries through the deliberate Phase-13 prim-scope expansion into UMAP's surface.

| Option | Description | Selected |
|--------|-------------|----------|
| Full value-gate × all 5 | 1e-5 deterministic value-gate + property-gated layout for every metric | ✓ |
| Euclidean full + others KNN-only | Full gate on Euclidean; rely on Phase-13 KNN gate + one property run for the rest | |
| You decide | Planner chooses oracle depth | |

**User's choice:** Full value-gate × all 5
**Notes:** Maximum correctness confidence; accepts the larger umap-learn fixture set.

---

## transform(X_new) fidelity

| Option | Description | Selected |
|--------|-------------|----------|
| Full umap-learn path | KNN(new→train) → fuzzy membership → neighbor-weighted init → frozen-train SGD on new points only, reusing the vertex-owner kernel | ✓ |
| Init-only (no new-point SGD) | Neighbor-weighted average init returned without optimization; simpler, looser gate | |
| You decide | Planner chooses based on kernel constraints | |

**User's choice:** Full umap-learn path
**Notes:** Implies the vertex-owner SGD kernel must support a frozen-subset mode (used by both fit and transform).

---

## Property gate + reproducibility

| Option | Description | Selected |
|--------|-------------|----------|
| Track umap-learn tightly | trustworthiness/kNN-overlap ≥ umap-learn − ε, downstream-ARI within tight band; relative-to-oracle | ✓ |
| Absolute-floor structural | Clear absolute quality floors without requiring closeness to umap-learn | |
| Both (relative + floor) | Require both floors and within-margin-of-umap-learn | |

**User's choice:** Track umap-learn tightly
**Notes:** Strongest "matches umap-learn" claim; margins calibrated on first fixture run; accepts possible iteration.

| Option | Description | Selected |
|--------|-------------|----------|
| fit + transform | Both byte-identical for fixed random_state across full stochastic surface | ✓ |
| fit only | Only fit byte-identical; transform property-gated but not byte-pinned | |
| fit + transform, same backend only | Byte-identical scoped to a single backend explicitly | |

**User's choice:** fit + transform
**Notes:** Captured in CONTEXT D-05 with the necessary clarification that bit-identity is per (backend, dtype) — f32-vs-f64 alone precludes cross-dtype identity. Forces order-deterministic PRNG draws throughout.

---

## a/b curve fit

| Option | Description | Selected |
|--------|-------------|----------|
| Port LM least-squares | Host-side Levenberg–Marquardt replicating scipy curve_fit; value-gated ≤1e-5 vs umap-learn a/b | ✓ |
| Precomputed lookup/closed-form | Closed-form/table approximation; small fixed offset the gate must absorb | |
| Default-curve only + a/b override | Constants for default only; require explicit a=/b= otherwise | |

**User's choice:** Port LM least-squares
**Notes:** Effectively a fifth deterministic value-gated stage; self-contained host routine, no device kernel.

---

## Claude's Discretion

- Spectral-init Jacobi size cap, above-cap random-fallback behavior, and disconnected-graph handling — follow the existing v2 graph-Laplacian + v1 Jacobi-eig convention (not separately discussed).
- `n_epochs=None` auto heuristic — match umap-learn; exact threshold confirmed against oracle.
- Negative-sampling index-draw mechanics (order-deterministic + cpu-MLIR-safe).
- Exact `Metric` enum extension shape / minkowski-p `p` type — follow Phase-13 prim.
- LM solver internals (variant, damping, tolerances) — any choice hitting the ≤1e-5 a/b gate.

## Deferred Ideas

- Spectral-init Jacobi cap / disconnected-graph handling — defaults to v2 convention; raise only if it doesn't transfer.
- PyO3 wrap of `Umap` and the builder-retrofit sweep — Phase 16.
- Supervised/target-metric UMAP, approximate/NN-Descent KNN, native sparse path — already out of scope in REQUIREMENTS.md.
