# Roadmap: mlrs — cuML in Rust

## Milestones

- ✅ **v1.0 Core ML Library** — Phases 1–6 (shipped 2026-06-14) → [archive](milestones/v1.0-ROADMAP.md)
- ✅ **v2.0 Breadth Sweep** — Phases 7–11 (shipped 2026-06-22) → [archive](milestones/v2.0-ROADMAP.md)
- 🚧 **v3.0 Manifold Algorithms & Rust-Native API** — Phases 12–16 (in progress)

## Overview

v3.0 adds the UMAP + HDBSCAN manifold/clustering pair on a single shared, feasibility-critical KNN-graph primitive, and establishes a Rust-native builder/typestate API convention that is retrofitted across the whole 30-estimator surface. The journey is primitive-first and dependency-ordered: establish the builder *convention* so the new estimators are born idiomatic (Phase 12) → land + standalone-gate the shared KNN-graph prim before any consumer touches it (Phase 13) → build UMAP and HDBSCAN as file-disjoint, parallel-buildable estimator phases on top of that prim (Phases 14–15) → isolate the broad, parallel-unsafe builder-retrofit sweep + full Python-shim coverage to the last phase (Phase 16). Same backend gate as v1/v2 (cpu f64 + rocm f32, f64-on-rocm skips-with-log), zero new compute dependencies, per-phase build-failing PoolStats memory gates, tests separated from source. The oracle broadens for UMAP (umap-learn) and the 1e-5 value gate relaxes to a property/structural gate for UMAP's stochastic layout only; HDBSCAN keeps an exact-label hard gate.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Phase numbering continues from v2.0 (which ended at Phase 11); v3.0 starts at Phase 12.

<details>
<summary>✅ v1.0 Core ML Library (Phases 1–6) — SHIPPED 2026-06-14 — 38 plans, 12 estimators</summary>

- [x] Phase 1: Foundation — Oracle, Backend Abstraction, Arrow Bridge (5/5 plans) — completed 2026-06-11
- [x] Phase 2: Core Compute Primitives (5/5 plans) — completed 2026-06-12
- [x] Phase 3: SVD / Eigendecomposition Primitive — Hard Gate (5/5 plans) — completed 2026-06-12
- [x] Phase 4: Closed-Form Estimators (5/5 plans) — completed 2026-06-12
- [x] Phase 5: Distance-Based & Iterative-Solver Estimators (11/11 plans) — completed 2026-06-13
- [x] Phase 6: Python Surface — PyO3 Estimators & Per-Backend Wheels (6/6 plans) — completed 2026-06-14

Full phase detail, plans, and per-plan notes: [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)

</details>

<details>
<summary>✅ v2.0 Breadth Sweep (Phases 7–11) — SHIPPED 2026-06-22 — 27 plans, 18 estimators</summary>

~18 sklearn-compatible estimators across five families, built as assembly on v1's validated primitive base plus five new feature-free CubeCL primitives (RNG-matrix, incremental-SVD, kernel-matrix, graph-Laplacian, SGD solver). No new compute dependency. Oracle = scikit-learn ≤ 1e-5; gate = cpu(f64) + rocm(f32), f64-on-rocm skips-with-log.

- [x] Phase 7: Covariance & Projection — RNG-matrix + incremental-SVD prims, PartialFit trait; EmpiricalCovariance, LedoitWolf, IncrementalPCA, Gaussian/SparseRandomProjection (7/7 plans) — completed 2026-06-20
- [x] Phase 8: Kernel Family — kernel-matrix prim (linear/RBF/poly/sigmoid), ScoreSamples trait; KernelRidge, KernelDensity (5/5 plans) — completed 2026-06-21
- [x] Phase 9: Spectral Family — graph-Laplacian prim (hard dep on Phase 8); SpectralEmbedding, SpectralClustering (4/4 plans) — completed 2026-06-21
- [x] Phase 10: SGD / Linear-SVM — SGD solver prim (two-pass GATHER, cpu-MLIR-safe); MBSGDClassifier, MBSGDRegressor, LinearSVC, LinearSVR (6/6 plans) — completed 2026-06-21
- [x] Phase 11: Naive Bayes — reductions-only closing bookend; GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB + PY-06 final PyO3 sign-off (5/5 plans) — completed 2026-06-22

Full phase detail, plans, and per-plan notes: [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md)

</details>

### 🚧 v3.0 Manifold Algorithms & Rust-Native API (In Progress)

**Milestone Goal:** Add UMAP + HDBSCAN on a shared KNN-graph primitive, and establish + retrofit a Rust-native builder/typestate API across the full estimator surface, plus extend the pure-Python sklearn shim.

- [x] **Phase 12: Builder + Typestate Convention Foundation** — Establish the shared idiomatic Rust builder + fit/unfit typestate convention (born-with-it for the new estimators; no retrofit yet) (4/4 plans) — completed 2026-06-23
- [x] **Phase 13: KNN-Graph Primitive (feasibility keystone)** — Land + standalone-gate the shared multi-metric `(indices, distances)` KNN-graph prim (euclidean/manhattan/cosine/chebyshev/minkowski-p) before any consumer touches it ✅ 2026-06-23
- [x] **Phase 14: UMAP** — Fuzzy simplicial set → spectral/random init → vertex-owner SGD layout; deterministic stages value-gated, stochastic layout property-gated (7/7 plans executed incl. gap closure 14-06/14-07; verification PASSED 4/4 — CR-01 cross-cube write race + CR-03 force double-count fixed via owner-only move_other=0, CR-02 n_components<n guard added) (completed 2026-06-24)
- [x] **Phase 15: HDBSCAN** — Device front-end (core/mutual-reach) + host tree back-end (MST → condensed tree → stability); exact-label hard gate (completed 2026-06-24)
- [ ] **Phase 16: Builder Retrofit Sweep + Shim Coverage** — Retrofit the convention across all existing estimators (additive) and complete the pure-Python sklearn shim coverage

## Phase Details

### Phase 12: Builder + Typestate Convention Foundation

**Goal**: Establish the idiomatic Rust-native estimator-construction convention — a shared owned-builder + fit/unfit typestate + typed validation error surface — so the v3 estimators (UMAP/HDBSCAN) are born builder-fronted and the later retrofit has a single target shape. Pure API foundation; no algorithm, no device work; lowest risk; unblocks everything downstream.
**Depends on**: Phase 11 (v2.0 estimator surface + shipped `BuildError` / `any_estimator!` machinery)
**Requirements**: BLDR-01, BLDR-02, BLDR-04
**Success Criteria** (what must be TRUE):

  1. A developer can construct an estimator via `T::builder().param(..)…build() -> Result<T<Unfit>, BuildError>` with owned chained setters and typed `thiserror` validation variants, where `T::builder().build()? == T::new()` == the sklearn default (single-source defaults).
  2. The fit/unfit distinction is modeled at compile time (`T<Unfit>` → `T<Fitted>`) such that `predict`/`transform`/fitted-attr accessors exist only on the fitted type — predict-before-fit fails to compile.
  3. The PyO3 surface is unchanged: the Rust typestate collapses behind the existing `any_estimator!` `Unfit/F32/F64` enum, with a runtime `NotFittedError` analog at the Python boundary, and every existing `any_estimator!` call site still compiles and passes its suite.
  4. The convention is demonstrated end-to-end on the two new-estimator shells (UMAP/HDBSCAN homes) so Phases 14–15 inherit it from birth.

**Plans**: 4 plansPlans:
**Wave 1**

- [x] 12-01-PLAN.md — typestate foundation: sealed `State` + `Unfit`/`Fitted` markers + consuming `Fit`/`Predict`/`Transform`/`PartialFit` traits (new module, `traits.rs` frozen) + trybuild dev-dep — completed 2026-06-23 (3 commits, cpu-green; traits.rs byte-for-byte unchanged)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 12-02-PLAN.md — UMAP + HDBSCAN shells: full param surface, owned builder, single-source defaults, non-algorithmic trivial fit, Fitted-only accessors, 2 new `BuildError` variants — completed 2026-06-23 (3 commits, cpu-green 8/8 tests; closes BLDR-01 + Rust-side of BLDR-02, the compile-fail proof is 12-03)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 12-03-PLAN.md — trybuild compile-fail gate proving predict/transform-before-fit won't compile (BLDR-02 structural proof) — completed 2026-06-23 (1 commit dd6c99f, cpu-green; compile_fail gate + 2 ui fixtures w/ Unfit-referencing E0277/E0308 goldens; additive regression 11/11)
- [x] 12-04-PLAN.md — PyO3 collapse: additive `any_estimator_typestate!` macro + `PyUMAP`/`PyHDBSCAN` shells + runtime `NotFittedError` analog; existing 35 call sites stay green — completed 2026-06-23 (4 commits 547b146/618a576/e342a23/58eed06, cpu-green; closes BLDR-04; full mlrs-py suite green incl. new manifold_test 2/2; live PyO3 pytest routed to UAT)

**UI hint**: no

### Phase 13: KNN-Graph Primitive (feasibility keystone)

**Goal**: Land the single shared KNN-graph primitive — ascending-ordered k-nearest-neighbor indices `(n, k)` + distances `(n, k)` over a **multi-metric** distance layer (Euclidean, Manhattan/L1, Cosine, Chebyshev/L∞, Minkowski-p), with a self-inclusion parameter — exposed as a new standalone `mlrs-backend` prim fn composed cpu-MLIR-safe from the launch-proven distance → top-k GATHER path (no SharedMemory/atomics/heap kernel), and standalone-validate it (per metric) BEFORE UMAP or HDBSCAN consume it (primitive-first discipline). Euclidean/Cosine reuse the v1 GEMM-expansion (Cosine on L2-normalized rows); Manhattan/Chebyshev/Minkowski-p add new direct pairwise GATHER distance kernels. Emits the **directed** `(indices, distances)` graph only (symmetrization deferred to the consumers). This is the milestone's feasibility keystone.
**Depends on**: Phase 12 (born builder-fronted convention is established; the prim itself reuses v1 distance + top-k, plus new direct-distance kernels)
**Requirements**: PRIM-11
**Success Criteria** (what must be TRUE):

  1. The KNN-graph prim (a new `mlrs-backend` prim fn) returns ascending-ordered neighbor indices `(n, k)` and distances `(n, k)`, with a `metric` parameter (Euclidean, Manhattan, Cosine, Chebyshev, Minkowski-p) and a self-inclusion parameter (UMAP self-excluded via k+1/self-drop-by-index-identity; HDBSCAN self-counted core distances), composed from distance → top-k GATHER with no new heap kernel. Output is the **directed** graph only; symmetrization is each consumer's job.
  2. The prim launches under `--features cpu` (verified at launch, not just compile) with no `Atomic`/`SharedMemory`/`F::INFINITY`/mutable-bool/shift-loop — for **every** metric, including the new direct Manhattan/Chebyshev/Minkowski-p kernels — and on rocm f32.
  3. For **each** metric, indices are set-equal to `sklearn.neighbors.NearestNeighbors` (with the matching `metric`) up to tie-ordering and distances match to ≤1e-5 (f64), with the lowest-index tie-break documented as the mlrs convention.
  4. A build-failing PoolStats memory gate passes at fixture sizes (big distance operand kept global / query-axis tiled; never the full `n×n` resident-and-leaking).

**Plans**: 3 plansPlans:
**Wave 1**

- [x] 13-01-PLAN.md — Nyquist Wave 0: per-metric sklearn oracle fixtures (incl. duplicate-point design) + knn_graph_test.rs harness (set-equal index, dup-point VALUE assert, geometry-rejection, query-axis memory gate) + kernel/prim module scaffolds ✅ 2026-06-23 (RED-by-design pending 13-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 13-02-PLAN.md — new cpu-MLIR-safe device kernels: manhattan/chebyshev/minkowski direct pairwise distance (STATIC F::powf) + self_drop_gather (index-identity, CUBE_POS_X shape) + launch smoke test ✅ 2026-06-23 (launch-proven cpu f32+f64 / rocm f32)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 13-03-PLAN.md — knn_graph<F> prim + Metric enum: validate-before-launch host orchestrator (metric routing, query-axis-tiled distance->top_k + single self_drop_gather, directed-only) — turns the per-metric oracle + memory gate GREEN (PRIM-11) ✅ 2026-06-23 (all 5 metrics ≤1e-5 cpu f64+f32 / rocm f32; R-9 dup-point VALUE + query-axis memory gate GREEN)

**UI hint**: no
**Spike status**: VALIDATED (spikes 001+002, 2026-06-23) — both feasibility unknowns confirmed under --features cpu; planning built on the proven kernel shapes.
**Spike flag (historical)**: SPIKE BEFORE PLANNING — confirm (a) the composed distance → top_k → dense `[n,k]` path launches under `--features cpu` (directed-only; the symmetrize-map step is removed — symmetrization moved to the UMAP/HDBSCAN consumers), and (b) the **new direct pairwise GATHER distance kernels** for Manhattan/Chebyshev/Minkowski-p launch under cpu-MLIR with no SharedMemory/atomics (Minkowski-p needs in-kernel `pow` — the named cpu-MLIR unknown). Precedent (v2 top_k + GATHER kernels on cpu-MLIR) is favorable but unverified for these new distance kernels.

### Phase 14: UMAP

**Goal**: Deliver UMAP `fit`/`fit_transform` → `embedding_` `(n, n_components)` with umap-learn/sklearn-named hyperparameters: KNN graph (reusing Phase 13) → fuzzy simplicial set (smooth-kNN ρ/σ binary search + t-conorm union) → init (random default; spectral via the v2 graph-Laplacian + v1 eig stack under the Jacobi size cap) → a new vertex-owner GATHER SGD layout kernel with negative sampling. Value-gate the deterministic stages 1–4; property-gate the stochastic layout. File-disjoint from HDBSCAN.
**Depends on**: Phase 12 (builder convention), Phase 13 (KNN-graph prim)
**Requirements**: UMAP-01, UMAP-02, UMAP-03, UMAP-04
**Success Criteria** (what must be TRUE):

  1. A user can `fit`/`fit_transform` UMAP to produce `embedding_` `(n, n_components)` with the umap-learn-named hyperparameters and defaults (`n_neighbors=15`, `n_components=2`, `min_dist=0.1`, `init='spectral'`, `random_state`, …), with `min_dist ≤ spread` validated at build.
  2. UMAP's deterministic stages — KNN graph, fuzzy simplicial set, fuzzy-set union, spectral init (reusing the v2 graph-Laplacian + v1 eig; random-init fallback above the Jacobi cap) — value-match umap-learn intermediates to ≤1e-5 (f64).
  3. UMAP's stochastic SGD layout passes a property/structural gate vs umap-learn 0.5.12 — trustworthiness / kNN-overlap ≥ umap-learn − margin and downstream-ARI within band — NOT coordinate value-match, and the same `random_state` reproduces a byte-identical mlrs embedding across runs.
  4. A user can embed new data via `transform(X_new)` against the fitted fuzzy graph, gated by a property sub-gate on the new points.

**Plans**: 7/7 plans complete
Plans:
**Wave 1**

- [x] 14-01-PLAN.md — Nyquist Wave 0: umap-learn 0.5.12 oracle fixtures (5 metrics × every stage) + RED value/property/reproducibility/transform harness + Metric→5 variants + umap_internals/umap_init module stubs

**Wave 2** *(blocked on Wave 1; 02 & 03 file-disjoint, parallel)*

- [x] 14-02-PLAN.md — deterministic stages 1–3 (host f64): smooth-kNN ρ/σ binary search + membership strengths + t-conorm fuzzy union (UMAP's symmetrization); value-gated ≤1e-5 × 5 metrics
- [x] 14-03-PLAN.md — deterministic stages: host LM a/b curve fit + spectral init (reuse laplacian+eig+recover, random fallback above n=64) ; value-gated ≤1e-5 × 5 metrics (up-to-sign spectral)

**Wave 3** *(blocked on Wave 2; SPIKE-GATED)*

- [x] 14-04-PLAN.md — NEW umap_layout_step<F> vertex-owner GATHER SGD kernel (cpu-MLIR-safe, frozen-subset) + host epoch driver + real fit/fit_transform + property-gate calibration; property-gated + byte-identical reproducibility

**Wave 4** *(blocked on Wave 3)*

- [x] 14-05-PLAN.md — transform(X_new) frozen-subset path (same kernel, move_other=false) + property sub-gate + replace stale zeros shell tests

**Gap closure** *(from verification gaps_found — CR-01/CR-02/CR-03)*

- [x] 14-06-PLAN.md — GAP 2 (CR-02): n_components < n guard in Umap::fit (mirror SpectralEmbedding) + typed-error test (UMAP-01, wave 1)
- [x] 14-07-PLAN.md — GAP 1 (CR-01+CR-03): owner-only move_other=0 fit launch (kills cross-cube race + edge double-count) + scheduling-order determinism test + property-gate recalibration (UMAP-03, wave 2, depends 14-06)

**UI hint**: no
**Spike flag**: SPIKE BEFORE PLANNING — (1) confirm the vertex-owner `umap_layout_step` single-owner GATHER kernel launches under cpu-MLIR (the named cpu-MLIR unknown; precedent: v2 two-pass SGD solver launched first try); (2) calibrate the property-gate thresholds (trustworthiness / kNN-overlap floors relative to umap-learn) empirically on the first oracle fixture run.

### Phase 15: HDBSCAN

**Goal**: Deliver HDBSCAN `fit`/`fit_predict` → `labels_` (`-1` = noise) + `probabilities_` with sklearn-named hyperparameters, as a device front-end (core distances + mutual-reachability via GATHER, reusing Phase 13) plus a host tree back-end (MST → single-linkage → condensed tree → EoM/leaf stability extraction), deliberately dodging the GPU-tree-atomics wall. Plus the GLOSH `outlier_scores_` differentiator and `store_centers`. Exact-label hard gate. File-disjoint from UMAP.
**Depends on**: Phase 12 (builder convention), Phase 13 (KNN-graph prim); feature-disjoint from Phase 14 (parallel-buildable after Phase 13)
**Requirements**: HDBS-01, HDBS-02, HDBS-03, HDBS-04
**Success Criteria** (what must be TRUE):

  1. A user can `fit`/`fit_predict` HDBSCAN to produce `labels_` (`-1`=noise) and `probabilities_` ∈[0,1] with sklearn-named defaults (`min_cluster_size=5`, `min_samples=None→min_cluster_size`, `cluster_selection_method='eom'` and `'leaf'`, …), with the device front-end / host tree back-end split holding under the cpu gate.
  2. `labels_` match `sklearn.cluster.HDBSCAN` exactly up to permutation with `-1` pinned (exact on `metric='precomputed'` f64; label-perm helper extended to fix `-1→-1`); MST edge tie-breaking is stable-sorted with a documented deterministic rule; `probabilities_` agree within a documented band.
  3. A user can read per-point `outlier_scores_` (GLOSH) from a fitted HDBSCAN, gated within band vs the `hdbscan` library.
  4. A user can request cluster centers via `store_centers` (`'centroid'`/`'medoid'`) producing `centroids_`/`medoids_`.

**Plans**: 7/7 plans complete
**Wave 1**

- [x] 15-01-PLAN.md — label_perm `-1→-1` pinned matcher + unit test (HDBS-02)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 15-02-PLAN.md — gen_hdbscan_* fixtures + committed .npz blobs + oracle gate suite (Wave 0)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 15-03-PLAN.md — Metric enum + build validation + both oracle MST variants + single-linkage; D-04/D-05 tie-break TRUE GATE (precomputed anchor)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 15-04-PLAN.md — condense + stability + eom/leaf/ε/max selection + labelling + probabilities (precomputed exact labels + ≤1e-5 probs)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 15-05-PLAN.md — mutual_reachability GATHER kernel + feature-metric device front-end (all 5 metrics exact) + memory gate

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 15-06-PLAN.md — GLOSH outlier_scores_ (vs hdbscan 0.8.44) + store_centers centroid/medoid (vs sklearn)

**Gap closure** *(standalone — Wave 1, no deps on 15-01…15-06; closes HDBS-01 BLOCKED gap from 15-VERIFICATION.md)*

- [x] 15-07-PLAN.md — `Hdbscan::fit_predict` convenience method (typestate-correct: consumes `self`) + behavioral-equivalence test (HDBS-01)

**UI hint**: no
**Spike flag**: SPIKE BEFORE PLANNING — RESOLVED IN PLANS: the D-04/D-05 host-MST tie-break exactness is sequenced as the Wave-3 TRUE GATE (15-03 `tie_break_exact` on the tie-heavy + duplicate-point fixture) BEFORE the device front-end (15-05) commits. The oracle-matched tie-break (sklearn `np.argsort` quicksort + the two Prim variants — NOT the mlrs lowest-index convention) is replicated; gate fixtures use distinct MST edge weights (RESEARCH Pitfall 1 option 2). An un-exactable metric is surfaced as a phase blocker per D-05, never band-demoted.

### Phase 16: Builder Retrofit Sweep + Shim Coverage

**Goal**: Retrofit the Phase-12 builder + typestate convention **additively** across all existing estimators (builder constructs the existing config struct; fit path untouched), piloted on 1–2 estimators under the green suite before the full sweep, preserving every shipped 1e-5 / exact-label gate; and complete the pure-Python sklearn shim (get_params/set_params/clone round-trip extended from the v1 12 to the v2 18 + UMAP/HDBSCAN, the two new PyO3 wraps, static Python check). This is the one broad-edit, parallel-unsafe phase — isolated last to protect file-disjoint discipline and the shipped gates.
**Depends on**: Phase 12 (convention), Phase 14 (UMAP estimator + PyO3 wrap), Phase 15 (HDBSCAN estimator + PyO3 wrap)
**Requirements**: BLDR-03, SHIM-01, SHIM-02, SHIM-03
**Success Criteria** (what must be TRUE):

  1. The builder + typestate convention is retrofitted additively across all existing estimators (`new()` kept as a thin wrapper; fit path untouched), piloted on 1–2 estimators with the green suite first, and every shipped 1e-5 / exact-label gate still passes.
  2. Every estimator's pure-Python class stores each constructor arg unchanged in `__init__` (no validation/computation) and exposes `get_params(deep=True)` / `set_params(**kw)` that round-trip exactly and are `clone()`-compatible (coverage extended from the v1 12 to the v2 18 + the two new).
  3. UMAP and HDBSCAN are PyO3-wrapped (`#[pyclass]` on `any_estimator!`, GIL release, `guard_f64` before F64) with sklearn-named params, trailing-underscore fitted attrs, `n_features_in_` set/enforced, `fit` returns `self`, and the correct surface (UMAP `transform`/`fit_transform`; HDBSCAN `fit_predict`/`labels_`).
  4. The shim is verified by Rust-side unit tests plus a static Python check; the live `estimator_checks`/`check_estimator` run stays deferred (needs a maturin+pyarrow host this environment lacks).

**Plans**: 13 plans (sequential, parallel-unsafe — shared traits.rs/typestate.rs/Any* enums; worktrees off)
- [ ] 16-00-PLAN.md — Wave 0 (BLOCKING): add 5 missing typestate traits + Transform::inverse_transform default; lock builder-setter convention; add AST __init__-purity test
- [ ] 16-01-PLAN.md — Pilots: Ridge (shape A full build-out) + MBSGDRegressor (shape B trait-swap)
- [ ] 16-02-PLAN.md — linear/ pt1: LinearRegression, Lasso, ElasticNet
- [ ] 16-03-PLAN.md — linear/ pt2: LogisticRegression, LinearSVC, LinearSVR, MBSGDClassifier
- [ ] 16-04-PLAN.md — decomposition/: PCA, TruncatedSvd, IncrementalPCA (PartialFit multi-transition)
- [ ] 16-05-PLAN.md — cluster/ (non-KMeans): DBSCAN, SpectralClustering, SpectralEmbedding (adopt traits)
- [ ] 16-06-PLAN.md — KMeans (late multi-constructor) + covariance: EmpiricalCovariance, LedoitWolf
- [ ] 16-07-PLAN.md — projection/: Gaussian, Sparse + density/: KernelDensity (adopt traits)
- [ ] 16-08-PLAN.md — neighbors/: NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
- [ ] 16-09-PLAN.md — kernel_ridge/: KernelRidge (adopt traits) + naive_bayes/: 5 NB (sweep complete, 29/29)
- [ ] 16-10-PLAN.md — SHIM-02: PyUMAP transform/fit_transform + PyHDBSCAN fit_predict/probabilities_/outlier_scores_
- [ ] 16-11-PLAN.md — SHIM-01/03: 15 pure-Python shim classes + full static test matrix + AST purity
- [ ] 16-12-PLAN.md — Final: delete traits.rs (grep-gated) + phase-end gate (compile_fail + oracle + Python static)
**UI hint**: no

## Progress

**Execution Order:**
Phases execute in numeric order: 12 → 13 → 14 → 15 → 16 (14 and 15 are file-disjoint and may be planned/built in parallel after 13).

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation — Oracle, Backend Abstraction, Arrow Bridge | v1.0 | 5/5 | Complete | 2026-06-11 |
| 2. Core Compute Primitives | v1.0 | 5/5 | Complete | 2026-06-12 |
| 3. SVD / Eigendecomposition Primitive (Hard Gate) | v1.0 | 5/5 | Complete | 2026-06-12 |
| 4. Closed-Form Estimators | v1.0 | 5/5 | Complete | 2026-06-12 |
| 5. Distance-Based & Iterative-Solver Estimators | v1.0 | 11/11 | Complete | 2026-06-13 |
| 6. Python Surface — PyO3 Estimators & Per-Backend Wheels | v1.0 | 6/6 | Complete | 2026-06-14 |
| 7. Covariance & Projection | v2.0 | 7/7 | Complete | 2026-06-20 |
| 8. Kernel Family | v2.0 | 5/5 | Complete | 2026-06-21 |
| 9. Spectral Family | v2.0 | 4/4 | Complete | 2026-06-21 |
| 10. SGD / Linear-SVM | v2.0 | 6/6 | Complete | 2026-06-21 |
| 11. Naive Bayes | v2.0 | 5/5 | Complete | 2026-06-22 |
| 12. Builder + Typestate Convention Foundation | v3.0 | 0/TBD | Not started | - |
| 13. KNN-Graph Primitive (feasibility keystone) | v3.0 | 3/3 | Complete    | 2026-06-23 |
| 14. UMAP | v3.0 | 7/7 | Complete    | 2026-06-23 |
| 15. HDBSCAN | v3.0 | 7/7 | Complete    | 2026-06-24 |
| 16. Builder Retrofit Sweep + Shim Coverage | v3.0 | 0/13 | Not started | - |
