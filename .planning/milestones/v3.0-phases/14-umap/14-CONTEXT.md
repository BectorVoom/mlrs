# Phase 14: UMAP - Context

**Gathered:** 2026-06-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Fill the Phase-12 `Umap` shell with the real algorithm: deliver `Umap::fit` / `fit_transform`
→ `embedding_` `(n, n_components)`, plus `transform(X_new)`, with umap-learn/sklearn-named
hyperparameters. Pipeline: **KNN graph** (reuse the Phase-13 prim, `include_self=false`) →
**fuzzy simplicial set** (smooth-kNN `ρ`/`σ` binary search) → **fuzzy-set union** (t-conorm) →
**init** (spectral default via the v2 graph-Laplacian + v1 Jacobi eig under the size cap;
random fallback above it) → **vertex-owner GATHER SGD layout kernel** with negative sampling.
Deterministic stages value-gated to ≤1e-5 (f64); the stochastic SGD layout property-gated.
Requirements: **UMAP-01, UMAP-02, UMAP-03, UMAP-04**. File-disjoint from HDBSCAN (Phase 15).

**In scope:**
- Real `fit` / `fit_transform` / `transform` bodies replacing the trivial zeros shell.
- All 5 metrics on UMAP's `metric=` param (euclidean, manhattan/L1, cosine, chebyshev/L∞,
  minkowski-p) via the Phase-13 prim.
- Fuzzy simplicial set, t-conorm union, spectral/random init, vertex-owner SGD layout with
  negative sampling, host-side `a`/`b` curve fit.
- `transform(X_new)` via the full umap-learn path (new-point-only frozen-train SGD).
- Per-metric deterministic value-gate (≤1e-5 f64) + per-metric property-gated layout.

**Out of scope (deferred / other phases):**
- Builder/typestate convention work (done in Phase 12 — shell already born builder-fronted).
- KNN-graph prim internals / new distance kernels (Phase 13 — already landed + per-metric gated).
- The PyO3 wrap of `Umap` and the builder-retrofit sweep (Phase 16).
- HDBSCAN (Phase 15).
- Custom/callable metrics, approximate/NN-Descent KNN, native sparse path (REQUIREMENTS out-of-scope).

</domain>

<decisions>
## Implementation Decisions

### Metric surface (UMAP-01 / UMAP-02)
- **D-01: Expose ALL 5 Phase-13 metrics** on UMAP's `metric=` param — euclidean, manhattan (L1),
  cosine, chebyshev (L∞), minkowski-p. Deliberate carry-through of the Phase-13 prim-scope
  expansion and umap-learn's multi-metric surface (rejected: Euclidean-only; rejected:
  Euclidean+cosine middle ground). The shell's `Metric` enum (currently `Euclidean`-only) is
  extended to the full set this phase.
- **D-02: Full deterministic value-gate × all 5 metrics.** Run the ≤1e-5 (f64) deterministic-stage
  value-gate (fuzzy set, union, spectral init, a/b) AND a property-gated layout run for **every**
  metric — not Euclidean-only. Maximum correctness confidence; accept the larger umap-learn fixture
  set. (Consequence: the oracle-fixture regen covers all 5 metrics — needs the `/tmp` numpy venv per
  the project landmine; fixtures are committed blobs.)

### transform(X_new) (UMAP-04)
- **D-03: Full umap-learn transform path.** `transform(X_new)` = KNN(new→train, via the Phase-13
  prim) → fuzzy membership against the fitted graph → neighbor-weighted-average init of each new
  point → reduced-epoch SGD optimizing **only the new points** with the **training embedding
  frozen** (read-only GATHER targets). Reuse the SAME vertex-owner layout kernel — new points are
  the sole "owners"; trained coords are read-only. (Rejected: init-only / no new-point SGD —
  diverges from umap-learn and would force looser property thresholds.) **Kernel design implication
  for planner:** the vertex-owner SGD kernel must support a "frozen-subset" mode from day one (a
  contiguous owner set whose non-owner neighbors are read-only), since both `fit` and `transform`
  drive the same kernel.

### Property gate + reproducibility (UMAP-03)
- **D-04: Track umap-learn 0.5.12 TIGHTLY.** The property gate requires mlrs to score within a
  small margin of umap-learn on the SAME data: trustworthiness ≥ umap-learn − ε, kNN-overlap ≥
  umap-learn − ε, downstream-ARI within a tight band — NOT just absolute floors. Margins (`ε`,
  band) are calibrated empirically on the first oracle-fixture run (per the Spike flag), but kept
  tight. (Rejected: absolute-floor-only structural; rejected: both/floor+relative as the framing —
  user chose relative-to-umap-learn as the primary philosophy.) Risk acknowledged: a borderline run
  may need threshold/algorithm iteration to pass.
- **D-05: `fit` AND `transform` byte-identical for a fixed `random_state`.** The same-`random_state`
  reproducibility contract covers the FULL stochastic surface — init RNG + negative-sampling RNG +
  new-point SGD RNG. Both `fit` and `transform` reproduce byte-identical mlrs embeddings across
  runs. Implication: every PRNG draw (init, negative-sampling, shuffle) must be **order-deterministic**
  in the kernel. **Necessary scope clarification (NOT a re-ask):** byte-identity is per `(backend,
  dtype)` — f32-vs-f64 alone precludes cross-dtype bit-identity, and float reduction order differs
  across runtimes, so the contract is "byte-identical across runs within a fixed backend+dtype,"
  which is the only physically achievable reading of "fit + transform." SplitMix64 PRNG (≠ NumPy MT)
  is why the layout is property-gated, never coordinate value-matched (REQUIREMENTS landmine).

### a/b curve fit (UMAP-01 / UMAP-02)
- **D-06: Port a host-side Levenberg–Marquardt least-squares fit.** When `a`/`b` are not overridden
  via the `a=`/`b=` params, derive them by least-squares fitting `1/(1 + a·d^(2b))` to the smooth
  target curve from `min_dist`/`spread`, replicating scipy's `curve_fit`. Value-gate the derived
  `a`/`b` to ≤1e-5 vs umap-learn — effectively a FIFTH deterministic value-gated stage. Self-contained
  host numeric routine (NO device kernel). (Rejected: precomputed lookup/closed-form — fixed offset
  the tight property gate would have to absorb; rejected: default-curve-only + mandatory override —
  narrows UMAP-01's parameter freedom.)

### Claude's Discretion
- **Spectral-init Jacobi size cap & disconnected-graph handling** — NOT discussed; follow the
  existing v2 graph-Laplacian + v1 Jacobi-eig convention (cap value, above-cap random fallback
  behavior, disconnected-component handling). Planner may finalize using the established convention;
  surface to the user only if the v2 convention doesn't transfer cleanly.
- `n_epochs=None` auto heuristic (umap-learn: 500 for small n, 200 for large) — match umap-learn;
  exact threshold is planner's to confirm against the oracle.
- Negative-sampling index draw mechanics under cpu-MLIR (must be order-deterministic per D-05 and
  GATHER/SharedMemory-free per the spike landmines) — planner/spike detail.
- Exact `Metric` enum extension shape and whether minkowski-p `p` is `F` or `f64` — follow the
  Phase-13 prim's `Metric` shape for consistency.
- LM solver internals (Gauss-Newton vs full LM, damping schedule, convergence tol) — any choice that
  hits the ≤1e-5 a/b value-gate.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — **UMAP-01..04** (full hyperparameter surface + defaults; deterministic
  ≤1e-5 stages; property/structural gate for the stochastic layout; transform sub-gate); the
  Out-of-Scope table (element-wise embedding match excluded; custom metrics excluded; approximate KNN
  excluded; native sparse excluded)
- `.planning/ROADMAP.md` § "Phase 14: UMAP" — goal, four Success Criteria, **Spike flag** (vertex-owner
  `umap_layout_step` GATHER kernel must launch under cpu-MLIR; property-gate thresholds calibrated on
  first fixture run)
- `.planning/PROJECT.md` — v3.0 milestone target-features (UMAP bullet: stochastic layout →
  property/structural gate, not element-wise 1e-5)

### Prior phase context (consume directly)
- `.planning/phases/13-knn-graph-primitive-feasibility-keystone/13-CONTEXT.md` — the KNN-graph prim
  decisions UMAP depends on: D-01/D-02 (`include_self=false` + self-drop by index identity for the
  UMAP path), D-04 (directed-only — **UMAP owns fuzzy-set-union symmetrization**), D-05 (full metric
  set + per-metric oracle)
- `.planning/phases/12-builder-typestate-convention-foundation/12-CONTEXT.md` — builder/typestate
  convention the shell already embodies (born builder-fronted; `fit` consumes `self` → `Fitted`;
  accessors/`Transform` only on `Fitted`)

### Existing code this phase fills / composes
- `crates/mlrs-algos/src/manifold/umap.rs` — the Phase-12 `Umap<F,S>` SHELL: full hyperparameter
  surface + builder + typestate already present; `Metric` enum (extend to 5), trivial-zeros `fit`/
  `transform` (replace with the real algorithm). **Single source of sklearn defaults** is `Umap::new`.
- `crates/mlrs-backend/src/prims/knn_graph.rs` — the Phase-13 KNN-graph prim UMAP calls (`include_self=false`)
- `crates/mlrs-algos/tests/umap_test.rs` — UMAP test home (tests separated from source, AGENTS.md §2)
- v2 graph-Laplacian + v1 eig (Jacobi) stack — spectral-init reuse target (planner to locate exact
  prim/algo paths in `crates/mlrs-backend/src/prims/` and the v1 decomposition algos)

### Conventions & feasibility guidance
- `.claude/skills/spike-findings-mlrs/SKILL.md` + `references/` — cpu-MLIR kernel-authoring landmines
  (no SharedMemory/atomics/`F::INFINITY`/mutable-bool/shift-loop; bare-`ABSOLUTE_POS` 1D launch fails;
  cross-sibling-loop accumulator SILENTLY miscompiles); the vertex-owner GATHER SGD kernel MUST obey these
- `AGENTS.md` — tests separated from source; on any CubeCL build error consult the error guideline FIRST
- `.planning/codebase/CONVENTIONS.md` — coding conventions
- CubeCL manuals at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` — generics, error guideline

### Project memory (environment landmines)
- cpu-MLIR backend panics on SharedMemory kernels w/ mutable bool / `F::INFINITY` / shift-loops — the
  layout kernel must be SharedMemory-free (vertex-owner GATHER idiom)
- rocm is the runnable GPU gate: gfx1100/ROCm 7.1.1 runs f32; f64 UNSUPPORTED on rocm → gate is
  cpu(f64) + rocm(f32), f64-on-rocm skips-with-log
- oracle fixture regen needs a `/tmp` venv with numpy (PEP 668); fixtures are committed blobs (now
  also umap-learn fixtures — pin umap-learn 0.5.12)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`Umap<F,S>` shell** (`crates/mlrs-algos/src/manifold/umap.rs`): full umap-learn hyperparameter
  surface, builder, typestate, build-time validation (`min_dist ≤ spread`, `n_components ≥ 1`,
  `n_neighbors ≥ 1`) already shipped. Phase 14 replaces only the trivial-zeros `fit`/`transform`
  bodies and extends `Metric`.
- **Phase-13 KNN-graph prim** (`crates/mlrs-backend/src/prims/knn_graph.rs`): directed `(indices,
  distances)` `(n,k)`, `include_self=false` for UMAP, per-metric oracle-validated. The KNN entry point.
- **v2 graph-Laplacian + v1 Jacobi eig**: spectral-init building blocks (reuse; do not re-derive).

### Established Patterns
- **Prim shape**: `fn prim<F>(pool, operands…, out: Option<…>)`, geometry validated before launch,
  device-resident outputs with buffer reuse — the new layout kernel and a/b host routine fit alongside.
- **cpu-MLIR safety**: no SharedMemory/atomics/`F::INFINITY`/mutable-bool/shift-loop; GATHER idiom.
  Generic-over-`F`; f64-on-rocm skips-with-log.
- **Single-source defaults**: `Umap::new` is the one place sklearn defaults live; builder re-derives.

### Integration Points
- UMAP owns its symmetrization (fuzzy-set t-conorm union) on top of the directed Phase-13 graph (D-04).
- `transform` and `fit` drive the SAME vertex-owner SGD kernel (D-03) — frozen-subset mode required.
- Phase 16 later PyO3-wraps `Umap` and retrofits nothing here (file-disjoint from HDBSCAN/Phase 15).

</code_context>

<specifics>
## Specific Ideas

- User carried the deliberate Phase-13 multi-metric scope expansion straight into UMAP's surface
  (all 5 metrics) AND demanded the strongest oracle depth (full value-gate × every metric) — consistent
  with the project's "correctness first" core value and the user's broad-API-scope preference.
- Property gate must **track umap-learn tightly** (relative-to-oracle), not just clear absolute floors —
  the strongest "matches umap-learn" claim, accepting possible threshold iteration.
- Reproducibility is maximal: byte-identical `fit` AND `transform` for a fixed `random_state` (within a
  fixed backend+dtype), forcing order-deterministic PRNG draws throughout.
- a/b is treated as a real numeric stage (ported LM least-squares, ≤1e-5 vs umap-learn), not approximated.

</specifics>

<deferred>
## Deferred Ideas

- **Spectral-init Jacobi cap / disconnected-graph handling** — not separately discussed; defaults to the
  existing v2 convention (Claude's discretion above). Raise to the user only if the v2 convention doesn't
  transfer.
- **PyO3 wrap of `Umap`** and the **builder-retrofit sweep** — Phase 16.
- Supervised/target-metric UMAP, approximate/NN-Descent KNN build, native sparse path — already out of
  scope in REQUIREMENTS.md; unchanged.

None — discussion otherwise stayed within phase scope.

</deferred>

---

*Phase: 14-umap*
*Context gathered: 2026-06-23*
