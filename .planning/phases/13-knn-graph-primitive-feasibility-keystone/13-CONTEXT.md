# Phase 13: KNN-Graph Primitive (feasibility keystone) - Context

**Gathered:** 2026-06-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Land the **single shared, multi-metric KNN-graph primitive** — ascending-ordered k-nearest-neighbor **indices `(n, k)` + distances `(n, k)`** — as a new standalone `mlrs-backend` prim function composed cpu-MLIR-safe from the launch-proven `distance → top_k` GATHER path, and **standalone-validate it (per metric) BEFORE** UMAP (Phase 14) or HDBSCAN (Phase 15) consume it (primitive-first discipline). This is the milestone's feasibility keystone. Requirement: **PRIM-11**.

**In scope:**
- A new `mlrs-backend` prim fn returning directed `(indices, distances)` over a `metric` parameter and a self-inclusion parameter.
- **Multi-metric distance layer (full set, implemented this phase):** Euclidean, Manhattan (L1), Cosine, Chebyshev (L∞), parameterized Minkowski-p.
- New **direct pairwise GATHER distance kernels** for Manhattan/Chebyshev/Minkowski-p (the v1 GEMM-expansion only covers Euclidean; Cosine reuses GEMM on L2-normalized rows).
- Per-metric oracle validation vs `sklearn.neighbors.NearestNeighbors` (indices set-equal up to tie-ordering; distances ≤1e-5 f64; lowest-index tie-break documented).
- Build-failing PoolStats memory gate (query-axis tiled; never full `n×n` resident-and-leaking).
- cpu(f64) + rocm(f32) launch gate, f64-on-rocm skips-with-log; tests separated from source.

**Out of scope (deferred to consumers / later):**
- **Graph symmetrization** — the prim emits the **directed** graph only. UMAP's fuzzy-set union (t-conorm) and HDBSCAN's mutual-reachability are each consumer's job in Phases 14/15.
- Any UMAP/HDBSCAN algorithm or estimator work.
- Custom/callable (Python) metrics; approximate / NN-Descent / tree KNN build; native sparse path (all already out of scope in REQUIREMENTS.md).

</domain>

<decisions>
## Implementation Decisions

### Self-inclusion API (PRIM-11)
- **D-01: `include_self: bool` flag.** One prim, one bool. `include_self=true` → `top_k` of `k` returns self (distance 0) as neighbor 0 — the HDBSCAN self-counted core-distance path. `include_self=false` → the prim internally queries `k+1`, drops the self column per row, returns `k` true neighbors — the UMAP self-excluded path. The `k+1`/self-drop bookkeeping is hidden inside the prim (rejected: "always include self, caller drops"; rejected: two separate entry points).
- **D-02: Self-drop by INDEX IDENTITY.** When `include_self=false`, drop the neighbor whose returned **index == the query row index**, NOT "the first zero-distance entry." This is robust against duplicate points sitting at distance 0 (which would otherwise drop a genuine neighbor and keep self, diverging from the oracle on tie-heavy data). If self is somehow not in the top-`(k+1)` (shouldn't happen for X-vs-X since self-distance 0 is minimal), drop the last column. The exact GATHER-safe mechanism under cpu-MLIR (no mutable-bool scans) is a planner/spike confirmation.

### Prim layer & form (PRIM-11)
- **D-03: New standalone `mlrs-backend` prim fn.** Lives at `crates/mlrs-backend/src/prims/knn_graph.rs` (planner may finalize the filename), alongside `distance.rs`/`topk.rs`, composing them directly: `knn_graph<F>(pool, x, (n, d), k, metric, include_self, …) -> (indices, distances)`. Matches how every v1/v2 prim is structured; UMAP/HDBSCAN call it from `mlrs-algos`. **No estimator wrapper** this phase (rejected: extending the v1 `NearestNeighbors` algos-layer core — wrong altitude for a shared prim; rejected: prim + thin estimator — more surface than a keystone needs).

### Symmetrization scope (PRIM-11)
- **D-04: Directed graph only.** Phase 13 emits ONLY the directed `(indices, distances)` k-NN graph. Symmetrization belongs to each consumer (UMAP fuzzy-set union; HDBSCAN mutual-reachability) — they symmetrize differently, so a shared symmetrize here would serve neither. This matches the PRIM-11 success criteria, which name only `(indices, distances)`. **Consequence for the spike:** the previously-named "symmetrize-map" step is REMOVED from the Phase-13 spike scope — see D-06.

### Metric scope (PRIM-11) — SCOPE EXPANSION (user-directed)
- **D-05: Full metric set, implemented and oracle-validated this phase.** The prim ships a fixed `metric` parameter covering **Euclidean, Manhattan (L1), Cosine, Chebyshev (L∞), and parameterized Minkowski-p**. This is a deliberate scope expansion beyond the original "over the v1 distance prim" (Euclidean-only) wording — user chose actual extra-metric *implementations*, not just a reserved param surface. Mechanics:
  - **Euclidean** — reuse the v1 GEMM-expansion fast path (`‖x‖²+‖y‖²−2·XYᵀ`, sqrt at the `top_k` boundary).
  - **Cosine** — reuse GEMM on L2-normalized rows (`1 − x̂·ŷ`).
  - **Manhattan / Chebyshev / Minkowski-p** — NEW direct pairwise GATHER distance kernels (the GEMM-expansion is Euclidean-specific and cannot back these), each cpu-MLIR-safe (no SharedMemory/atomics).
  - Validation is **per metric** vs `sklearn.neighbors.NearestNeighbors` with the matching `metric`.
- **D-06: Spike scope re-scoped by D-04 + D-05.** ROADMAP "Spike flag" updated: (a) **dropped** the symmetrize-map step (directed-only per D-04); (b) **added** proving the new direct Manhattan/Chebyshev/Minkowski-p GATHER distance kernels launch under cpu-MLIR — **Minkowski-p needs in-kernel `pow`, which is the named cpu-MLIR unknown** for this phase. The spike must confirm both the directed `distance → top_k → [n,k]` composition AND the new distance kernels (incl. Minkowski-p `pow`) launch under `--features cpu`.

### Claude's Discretion
- Exact `Metric` enum shape (e.g., `Metric::{Euclidean, Manhattan, Cosine, Chebyshev, Minkowski { p }}`), and whether Euclidean=Minkowski(2) / Manhattan=Minkowski(1) are special-cased to the fast paths or routed through the general direct kernel.
- Query-axis tile size for the PoolStats memory gate (criterion already locks "query-axis tiled, big distance operand kept global").
- Final prim filename (`knn_graph.rs`) and exact public signature/param order, following existing `distance.rs`/`topk.rs` conventions.
- Whether Minkowski-p `p` is `F` or `f64`; how `p` is validated (e.g. `p ≥ 1`) at the prim boundary.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — **PRIM-11** (updated this session: multi-metric, directed-only, per-metric oracle); the Out-of-Scope table (custom/callable metrics excluded; fixed set fixed)
- `.planning/ROADMAP.md` § "Phase 13" — goal, four Success Criteria, and the **updated Spike flag** (directed-only + new direct-distance-kernel feasibility incl. Minkowski-p `pow`)
- `.planning/PROJECT.md` — milestone target-features + Active list (KNN-graph bullet updated to multi-metric)

### Existing machinery this prim composes (reusable, launch-proven)
- `crates/mlrs-backend/src/prims/distance.rs` — v1 GEMM-expansion squared-Euclidean distance prim (`distance<F>`, validate-before-launch, device-resident, pool/`out` buffer reuse); the Euclidean fast path + the model for the new direct kernels
- `crates/mlrs-backend/src/prims/topk.rs` — `top_k<F>` k-smallest distances + indices per row, **lowest-index tie-break already documented**, `sqrt` at boundary, device-resident
- `crates/mlrs-algos/src/neighbors/nearest.rs` — v1 `NearestNeighbors` `kneighbors` core (reference for k validation / geometry; NOT the home for this prim per D-03)
- `crates/mlrs-backend/src/prims/mod.rs` — prim module registration

### Conventions & feasibility guidance
- `AGENTS.md` — tests separated from source (`crates/*/tests/`); on any CubeCL build error consult the error guideline FIRST
- `.planning/codebase/CONVENTIONS.md` — coding conventions
- CubeCL manuals at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` — `generics`, `INDEX.md`, error guideline (kernels MUST be generic-over-float; cpu-MLIR no-SharedMemory/atomics constraints)
- `.planning/phases/12-builder-typestate-convention-foundation/12-CONTEXT.md` — prior phase; builder/typestate convention (this is a prim, not an estimator, so the builder convention does not directly apply here)

### Project memory (environment landmines)
- cpu-MLIR backend panics on SharedMemory kernels w/ mutable bool / `F::INFINITY` / shift-loops — write SharedMemory-free kernels (applies to all new direct distance kernels)
- rocm is the runnable GPU gate: gfx1100/ROCm 7.1.1 runs f32; f64 UNSUPPORTED on rocm — gate is cpu(f64) + rocm(f32)
- oracle fixture regen needs a `/tmp` venv with numpy (PEP 668); fixtures are committed blobs

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`distance<F>` prim** (`crates/mlrs-backend/src/prims/distance.rs`): launch-proven Euclidean GEMM-expansion; pool/`out` buffer reuse, device-resident, validate-before-launch. Backs the Euclidean (and, on L2-normalized rows, Cosine) metric paths and is the structural template for the new direct distance kernels.
- **`top_k<F>` prim** (`crates/mlrs-backend/src/prims/topk.rs`): k-smallest + indices per query row, lowest-index tie-break documented, `sqrt` applied to returned values only — the second half of every metric path. Returns `(distances f, indices u32)`.

### Established Patterns
- **Prim shape:** `fn prim<F>(pool, operands…, out: Option<…>) -> Result<…, PrimError>`, geometry validated BEFORE any unsafe launch, outputs device-resident with optional caller-supplied buffer reuse (D-11 lineage). The new prim and any new distance kernels follow this exactly.
- **cpu-MLIR safety:** no SharedMemory / Atomic / `F::INFINITY` / mutable-bool / shift-loop — GATHER idiom over the feature dim for the new direct kernels.
- **Generics-over-float + backend gate:** every kernel generic over `F` (`f32`/`f64`); f64-on-rocm skips-with-log.

### Integration Points
- UMAP (Phase 14) calls the prim with `include_self=false` → KNN graph → fuzzy simplicial set; HDBSCAN (Phase 15) calls with `include_self=true` → core distances → mutual-reachability. Both symmetrize on their own side (D-04).

</code_context>

<specifics>
## Specific Ideas

- User explicitly expanded metric scope from Euclidean-only to the **full fixed set incl. Minkowski-p**, and directed "change all document" — REQUIREMENTS.md (PRIM-11 + out-of-scope row), ROADMAP.md (Phase 13 goal/criteria/spike + one-liner), and PROJECT.md (target-features + Active bullet) were all updated this session to match.
- Self-drop correctness must be by index identity, not first-zero-distance (D-02) — user confirmed the duplicate-point edge case matters.

</specifics>

<deferred>
## Deferred Ideas

- **Graph symmetrization** — moved to the UMAP (fuzzy-set union) and HDBSCAN (mutual-reachability) consumer phases (D-04). Not lost; it is each consumer's responsibility.
- **Public `KNNGraph` estimator wrapper** — if a user-facing estimator over the prim is ever wanted, add it in a later phase; the keystone ships the prim fn only (D-03).
- Supervised/target-metric KNN, approximate/NN-Descent build, native sparse path — already out of scope in REQUIREMENTS.md; unchanged.

</deferred>

---

*Phase: 13-knn-graph-primitive-feasibility-keystone*
*Context gathered: 2026-06-23*
