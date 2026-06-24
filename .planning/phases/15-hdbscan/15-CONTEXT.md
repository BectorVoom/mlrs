# Phase 15: HDBSCAN - Context

**Gathered:** 2026-06-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Fill the Phase-12 `Hdbscan<F,S>` shell with the real algorithm: deliver
`Hdbscan::fit` / `fit_predict` → `labels_` (`-1` = noise) + `probabilities_`
`∈[0,1]`, plus GLOSH `outlier_scores_` (HDBS-03) and `store_centers` →
`centroids_`/`medoids_` (HDBS-04), with sklearn-named hyperparameters.
Pipeline: **device front-end** (core distances + mutual-reachability via the
Phase-13 KNN prim, `include_self=true` for self-counted core distances) →
**host back-end** (Prim's MST → single-linkage → condensed cluster tree →
EoM/leaf stability extraction). The device/host split deliberately dodges the
GPU-tree-atomics wall. **Exact-label hard gate** (up to permutation, `-1`
pinned). Requirements: **HDBS-01, HDBS-02, HDBS-03, HDBS-04** (+ the algorithm
side of SHIM-02 is consumed in Phase 16). File-disjoint from UMAP (Phase 14).

**In scope:**
- Real `fit` / `fit_predict` bodies replacing the trivial all-`-1` shell.
- All 5 feature-space metrics on `metric=` (euclidean, manhattan/L1, cosine,
  chebyshev/L∞, minkowski-p) via the Phase-13 prim, PLUS a new
  `Metric::Precomputed` variant (X interpreted as a square n×n distance matrix,
  bypassing the device distance front-end).
- Core distances → mutual-reachability → host MST → single-linkage → condensed
  tree → EoM **and** leaf stability extraction.
- `probabilities_`, GLOSH `outlier_scores_`, `store_centers`
  ('centroid'+'medoid' → `centroids_`/`medoids_`).
- The full selection-knob surface: `cluster_selection_method` (eom+leaf),
  `cluster_selection_epsilon`, `max_cluster_size`, `alpha`, `min_samples`.
- Per-metric exact-label gate, score value-gates, center value-gates.
- Extend `mlrs_core::label_perm` to pin `-1→-1` in the permutation match.

**Out of scope (deferred / other phases):**
- Builder/typestate convention work (done in Phase 12 — shell already born
  builder-fronted; `fit` consumes `self` → `Fitted`, accessors `Fitted`-only).
- KNN-graph prim internals / distance kernels (Phase 13 — landed + per-metric gated).
- The PyO3 wrap of `Hdbscan` + the builder-retrofit sweep + Python shim (Phase 16).
- UMAP (Phase 14).
- `approximate_predict` / `membership_vector` (new-point predict), condensed-tree /
  dendrogram plot objects, approximate/NN-Descent/tree KNN build, custom/callable
  metrics, native sparse path — all out of scope in REQUIREMENTS.md.

</domain>

<decisions>
## Implementation Decisions

### Metric surface (HDBS-01 / HDBS-02)
- **D-01: Expose ALL 5 feature-space metrics + `Precomputed`.** `metric=` covers
  euclidean, manhattan (L1), cosine, chebyshev (L∞), minkowski-p (all via the
  Phase-13 prim, `include_self=true`) PLUS a new `Metric::Precomputed` variant.
  Deliberate carry-through of the Phase-13/UMAP multi-metric scope expansion and
  the user's broad-API-scope preference. (Rejected: euclidean+precomputed only;
  rejected: all-5-but-no-precomputed — drops the cleanest exact anchor.) The
  shell's `Metric` enum (currently `Euclidean`-only) is extended to all 6 this phase.
- **D-02: `precomputed` via a `Metric::Precomputed` enum variant.** When set, `fit`
  interprets `X` as a square **n×n distance matrix** (shape `(n,n)`) instead of
  feature rows, skipping the KNN/distance device front-end and feeding distances
  straight into core-distance + mutual-reachability. Mirrors sklearn's
  `metric='precomputed'` single-`metric=` surface; one `Fit` impl. (Rejected: a
  separate `fit_precomputed` entry point — diverges from sklearn and the shell's
  single `Fit`.) Planner: validate squareness (and document symmetry expectation)
  before the back-end runs.

### Exact-label gate anchoring (HDBS-02) — MAXIMAL CORRECTNESS
- **D-03: Exact-up-to-permutation on EVERY metric** (not just precomputed f64).
  `labels_` match `sklearn.cluster.HDBSCAN` exactly up to permutation with `-1`
  pinned, for precomputed AND all 5 feature-space metrics. (Rejected:
  precomputed-only exact / euclidean+precomputed exact with others band-gated —
  user chose the strongest correctness claim.) **⚠ RISK (flag for spike+planner):**
  for non-euclidean brute-KNN metrics, distance ties can be ordered differently
  than the oracle internally, flipping an MST edge and cascading into a label
  difference — all-metric exactness is physically fragile and hinges entirely on
  D-04.
- **D-04: Match the ORACLE's internal MST tie-break exactly.** Reverse-engineer
  and replicate `sklearn.cluster.HDBSCAN` / `hdbscan` 0.8.44 internal MST
  edge-tie ordering so even tied edges align — not merely the mlrs lowest-index
  convention. (Rejected: stable-sort + lowest-index mlrs convention — reproducible
  within mlrs but won't guarantee oracle-exact ties across all metrics.) Couples
  the host MST to oracle internals; the **pre-planning spike** validates this on a
  deliberately tie-heavy fixture and locks the documented deterministic rule.
- **D-05: HOLD THE EXACT LINE — non-negotiable.** If the spike finds a metric
  cannot hit exact-up-to-perm even with the oracle-matched tie-break, iterate the
  algorithm / tie-break until it passes — do NOT auto-demote to a band gate, do
  NOT silently drop the metric. (Rejected: demote-to-band fallback; rejected:
  escalate-to-user fallback — user chose exact-on-all as a hard requirement.)
  **⚠ Consequence (flag for planner):** the pre-planning exactness spike is a TRUE
  gate — a metric that proves un-exactable is a phase blocker, not a documented
  caveat. Surface early.

### probabilities_ + GLOSH outlier_scores_ band (HDBS-01 / HDBS-03) — MAXIMAL
- **D-06: Treat the scores as ≤1e-5 value gates, escalate if divergent.** Gate
  `probabilities_` and GLOSH `outlier_scores_` to ≤1e-5 (abs+rel) rather than a
  loose "documented band"; if the spike/first-fixture run shows the scores diverge
  beyond that for genuine algorithmic/float-order reasons, escalate to the user
  rather than silently widening. (Rejected: tight relative-to-oracle band à la UMAP
  D-04; rejected: fixed absolute tolerance — user chose the most aggressive
  correctness target for the scores.)
- **D-07: Per-score oracle hierarchy.** `probabilities_` value-gated vs
  `sklearn.cluster.HDBSCAN` (primary, zero new dep) with `hdbscan` 0.8.44
  cross-check; GLOSH `outlier_scores_` (HDBS-03, a differentiator sklearn lacks)
  value-gated vs the `hdbscan` 0.8.44 library. Matches the REQUIREMENTS oracle
  hierarchy. (Rejected: both-vs-hdbscan-lib single oracle — leans on the cross-check
  lib as primary for probabilities_ instead of zero-dep sklearn.)

### store_centers + selection knobs (HDBS-04 / HDBS-01) — FULL PARITY
- **D-08: `store_centers` centroid AND medoid, value-gated vs sklearn.** Support
  `store_centers='centroid'` → `centroids_` (weighted mean per cluster) AND
  `'medoid'` → `medoids_` (min-total-distance cluster member), both value-gated
  ≤1e-5 vs `sklearn.cluster.HDBSCAN`. Full sklearn parity. (Rejected:
  presence/shape-only structural gate; rejected: centroid-now-medoid-deferred —
  breaks parity, against broad-scope preference.)
- **D-09: Full selection-knob surface, ALL under the exact gate.** Implement
  `cluster_selection_method` 'eom' AND 'leaf', `cluster_selection_epsilon` (>0
  merge logic), `max_cluster_size` bound, and `alpha` scaling — with oracle
  fixtures exercising **non-default** values of each, all held to the exact-label
  gate. Also resolves the shell's deferred validation TODO (min_samples >= 1 when
  Some; max_cluster_size 0=unbounded else >= min_cluster_size). (Rejected:
  defaults-only exact gate with lighter structural spot-checks for non-defaults;
  rejected: planner-decides fixture depth — user chose full exact coverage.)

### Claude's Discretion
- **Host MST algorithm internals** — Prim's is named in REQUIREMENTS/ROADMAP; the
  exact data structures (priority queue vs dense scan, single-linkage union-find
  shape) are the planner's, provided D-04's oracle-matched tie-break holds.
- **Memory / PoolStats gate for the n×n mutual-reachability** — follow the
  established per-phase build-failing PoolStats gate convention (query-axis-tiled,
  never full n×n device-resident where avoidable); planner sets the exact assertion.
- **Condensed-tree / stability-extraction data structures** (cluster hierarchy
  representation, EoM vs leaf selection traversal) — planner's choice that hits
  the exact-label gate.
- **Edge cases** (all-noise result, single point, single cluster, fewer than
  `min_cluster_size` points) — match sklearn behavior; planner confirms against
  the oracle, surfaces only if sklearn's behavior is ambiguous.
- **`min_samples=None → min_cluster_size`** default resolution is already in the
  shell's `new()`/`build()`; keep it.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — **HDBS-01..04** (full hyperparameter surface +
  defaults; device front-end / host back-end split; exact-labels-up-to-perm gate
  with `-1` pinned; `probabilities_` band; GLOSH; `store_centers`); the oracle note
  (exact on `metric='precomputed'` f64; sklearn primary + hdbscan 0.8.44 cross-check);
  the Out-of-Scope table (approximate_predict/membership_vector excluded;
  condensed-tree/dendrogram plot objects excluded; approximate/NN-Descent/tree KNN
  excluded; custom metrics excluded; native sparse excluded)
- `.planning/ROADMAP.md` § "Phase 15: HDBSCAN" — goal, Success Criteria, and the
  **Spike flag**: SPIKE BEFORE PLANNING — confirm host MST (Prim's) + condensed-tree
  exactness vs the reference on a tie-heavy fixture; lock the tie-break convention
  (now D-04: match the ORACLE's internal tie-break) BEFORE the exact-label gate commits
- `.planning/PROJECT.md` — v3.0 milestone target-features (HDBSCAN bullet:
  mutual-reachability → MST → condensed tree → stability; exact labels up to
  permutation as the hard gate)

### Prior phase context (consume directly)
- `.planning/phases/13-knn-graph-primitive-feasibility-keystone/13-CONTEXT.md` — the
  KNN-graph prim decisions HDBSCAN depends on: `include_self=true` (HDBSCAN
  self-counted core distances), directed-only output (**HDBSCAN owns
  mutual-reachability symmetrization**), full metric set + per-metric oracle,
  index-identity self handling, query-axis-tiled memory gate
- `.planning/phases/14-umap/14-CONTEXT.md` — the sibling estimator's pattern:
  multi-metric carry-through (D-01/D-02), tight-tracking gate philosophy (D-04),
  full-value-gate × every-metric oracle depth — HDBSCAN mirrors the maximal-correctness
  posture
- `.planning/phases/12-builder-typestate-convention-foundation/12-CONTEXT.md` —
  builder/typestate convention the `Hdbscan` shell already embodies (born
  builder-fronted; `fit` consumes `self` → `Fitted`; `labels`/`n_features_in`
  accessors only on `Fitted`; `Hdbscan::new` is the single source of sklearn defaults)

### Existing code this phase fills / composes
- `crates/mlrs-algos/src/cluster/hdbscan.rs` — the Phase-12 `Hdbscan<F,S>` SHELL:
  full hyperparameter surface + builder + typestate present; `Metric` enum
  (currently `Euclidean`-only — extend to all 5 + `Precomputed`),
  `ClusterSelectionMethod` (Eom/Leaf), trivial all-`-1` `fit` (replace with the real
  algorithm). **Single source of sklearn defaults** is `Hdbscan::new`. Contains the
  deferred validation TODO (min_samples / max_cluster_size) to resolve (D-09). NO
  `probabilities_`/`outlier_scores_`/`centroids_`/`medoids_`/`store_centers` fields yet —
  add them this phase.
- `crates/mlrs-backend/src/prims/knn_graph.rs` — the Phase-13 KNN-graph prim HDBSCAN
  calls (`include_self=true`) for core distances + the mutual-reachability neighbor set
- `crates/mlrs-core/src/label_perm.rs` — the label-permutation helper
  (`best_match_accuracy`) used by the kmeans exact-label test (D-09 precedent in
  `kmeans_test.rs`); **extend it to pin `-1→-1`** for the HDBSCAN noise sentinel (HDBS-02)
- `crates/mlrs-algos/tests/hdbscan_test.rs` — HDBSCAN test home (tests separated from
  source, AGENTS.md §2)
- `crates/mlrs-algos/src/cluster/dbscan.rs` + `cluster/mod.rs` — the closest labels-only
  estimator analog (`-1` noise sentinel contract); `crates/mlrs-py/src/estimators/cluster.rs`
  is where the Phase-16 PyO3 wrap will live (NOT this phase)

### Conventions & feasibility guidance
- `.claude/skills/spike-findings-mlrs/SKILL.md` + `references/` — cpu-MLIR
  kernel-authoring landmines (no SharedMemory/atomics/`F::INFINITY`/mutable-bool/
  shift-loop; bare-`ABSOLUTE_POS` 1D launch fails; cross-sibling-loop accumulator
  SILENTLY miscompiles); the mutual-reachability GATHER kernel MUST obey these. R-2
  (`include_self=true` for HDBSCAN), R-9 (per-metric oracle MUST assert VALUES incl. a
  duplicate-point row, not just non-panic)
- `AGENTS.md` — tests separated from source; on any CubeCL build error consult the
  error guideline FIRST
- `.planning/codebase/CONVENTIONS.md` — coding conventions
- CubeCL manuals at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/`

### Project memory (environment landmines)
- cpu-MLIR backend panics on SharedMemory kernels w/ mutable bool / `F::INFINITY` /
  shift-loops — the mutual-reachability device kernel must be SharedMemory-free (GATHER idiom)
- rocm is the runnable GPU gate: gfx1100/ROCm 7.1.1 runs f32; f64 UNSUPPORTED on rocm →
  gate is cpu(f64) + rocm(f32), f64-on-rocm skips-with-log
- oracle fixture regen needs a `/tmp` venv with numpy (PEP 668); fixtures are committed
  blobs (now also sklearn.cluster.HDBSCAN + hdbscan 0.8.44 fixtures — pin hdbscan 0.8.44)
- kNN lowest-index tie-break is the mlrs convention BUT HDBSCAN's MST tie-break follows
  the ORACLE (D-04), which may differ — do not conflate the two
- full `cargo test --features cpu` exhausts disk / is slow — run targeted tests; the
  backend cpu suite is ~6min

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`Hdbscan<F,S>` shell** (`crates/mlrs-algos/src/cluster/hdbscan.rs`): full
  hyperparameter surface, builder, typestate, build-time `min_cluster_size >= 2`
  validation already shipped. Phase 15 replaces the trivial all-`-1` `fit`, extends
  `Metric`, adds the score/center fitted fields + accessors, and resolves the deferred
  min_samples/max_cluster_size validation TODO.
- **Phase-13 KNN-graph prim** (`crates/mlrs-backend/src/prims/knn_graph.rs`): directed
  `(indices, distances)` `(n,k)`, `include_self=true` for HDBSCAN core distances,
  per-metric oracle-validated. The neighbor entry point for the device front-end.
- **`mlrs_core::label_perm`** (`crates/mlrs-core/src/label_perm.rs`): `best_match_accuracy`
  label-permutation matcher — extend with `-1→-1` pinning for HDBS-02.
- **DBSCAN** (`crates/mlrs-algos/src/cluster/dbscan.rs`): closest labels-only analog,
  `-1` noise sentinel contract shared via `cluster/mod.rs`.

### Established Patterns
- **Device front-end / host back-end split**: the deliberate architecture (REQUIREMENTS)
  to dodge the GPU-tree-atomics wall — device GATHER kernels for distances/mutual-reach,
  host CPU for MST/condensed-tree/stability.
- **Prim shape**: `fn prim<F>(pool, operands…, out: Option<…>)`, geometry validated
  before launch, device-resident outputs with buffer reuse; query-axis-tiled memory
  with a build-failing PoolStats gate.
- **cpu-MLIR safety**: no SharedMemory/atomics/`F::INFINITY`/mutable-bool/shift-loop;
  GATHER idiom. Generic-over-`F`; f64-on-rocm skips-with-log.
- **Exact-label test pattern**: `best_match_accuracy == 1.0` up to permutation
  (`kmeans_test.rs` D-09 precedent), here extended with `-1` pinning.
- **Single-source defaults**: `Hdbscan::new` is the one place sklearn defaults live;
  builder re-derives.

### Integration Points
- HDBSCAN owns its symmetrization (**mutual-reachability**) on top of the directed
  Phase-13 graph — sibling to UMAP's fuzzy-set-union (KNN prim D-04).
- `Metric::Precomputed` short-circuits the device front-end: X is the distance matrix
  fed straight to core-distance + mutual-reachability.
- Phase 16 later PyO3-wraps `Hdbscan` (`fit_predict`/`labels_`) and retrofits the builder
  sweep — nothing here is file-shared with that (file-disjoint from UMAP/Phase 14).

</code_context>

<specifics>
## Specific Ideas

- The user pushed for **maximal correctness on every axis**, consistently: all 5
  metrics + precomputed (D-01); exact-up-to-perm on EVERY metric (D-03) with the
  ORACLE's internal MST tie-break matched (D-04); HOLD THE EXACT LINE as a
  non-negotiable hard gate, not a band-fallback (D-05); scores as ≤1e-5 value gates
  with escalation (D-06); full centroid+medoid + full selection-knob surface all
  under the exact gate (D-08/D-09). This is the strongest possible "matches sklearn
  HDBSCAN" claim — consistent with the project's correctness-first core value and the
  user's documented broad-API-scope / tight-tracking preference (UMAP D-04).
- **Two explicit risk flags for the spike/planner** (do not lose): (1) all-metric
  exactness is physically fragile for non-euclidean brute-KNN ties (D-03); (2) because
  the exact line is held non-negotiably (D-05), the pre-planning MST/tie-break exactness
  spike is a TRUE gate — an un-exactable metric is a phase blocker, surface EARLY.

</specifics>

<deferred>
## Deferred Ideas

- **PyO3 wrap of `Hdbscan`** (`#[pyclass]`, `fit_predict`/`labels_`) and the
  **builder-retrofit sweep** + **Python sklearn shim** (SHIM-01/02/03) — Phase 16.
- `approximate_predict` / `membership_vector` (new-point predict), condensed-tree /
  dendrogram plot objects — already out of scope in REQUIREMENTS.md (need persisted
  prediction-data structures / no algorithmic value); unchanged.
- Approximate / NN-Descent / tree KNN-graph build, custom/callable metrics, native
  sparse path — already out of scope in REQUIREMENTS.md; unchanged.

None new — discussion otherwise stayed within phase scope.

</deferred>

---

*Phase: 15-hdbscan*
*Context gathered: 2026-06-24*
