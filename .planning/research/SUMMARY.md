# Project Research Summary

**Project:** mlrs v3.0 — Manifold Algorithms & Rust-Native API
**Domain:** UMAP + HDBSCAN on a shared KNN-graph primitive; Rust builder/typestate API retrofit; pure-Python sklearn shim
**Researched:** 2026-06-22
**Confidence:** HIGH overall (grounded in shipped codebase, project memory, and stable documented APIs); MEDIUM on two specific execution unknowns (see Gaps)

## Executive Summary

v3.0 adds two high-value algorithms — UMAP (nonlinear manifold embedding) and HDBSCAN (density-based variable-cluster clustering) — on a shared KNN-graph primitive built from the already-validated v1 NearestNeighbors (top-k) prim, and completes the Rust-native builder-pattern API retrofit across all 30+ estimators plus a pure-Python sklearn shim. The critical architectural insight from all four research streams is that the KNN-graph primitive is the feasibility keystone for the entire milestone: it is cpu-MLIR-safe by COMPOSITION of already-shipped prims (distance + top-k + a single-owner GATHER symmetrize map, no new heap/scatter kernel), it must be built and standalone-gated before either estimator consumes it, and the GATHER idiom that makes it viable directly constrains how UMAP's SGD layout step and HDBSCAN's mutual-reachability are structured. Zero new runtime crate dependencies are needed; the five-crate workspace structure, pyo3 0.28/arrow-59 ABI pin, cubecl 0.10, and cubek-{matmul,reduce} 0.2.0 stay entirely unchanged.

The three correctness-gate regimes are distinct and must not be conflated. The KNN-graph prim and UMAP's deterministic stages 1-4 (fuzzy simplicial set through spectral init) use the standard value gate (<=1e-5 vs sklearn NearestNeighbors / umap-learn intermediate arrays). UMAP's SGD layout stage 5 uses a property/structural gate (trustworthiness + kNN-overlap relative to umap-learn, plus seed-reproducibility within mlrs) — attempting a value oracle here is not merely unhelpful, it will report total failure on a correct implementation, exactly the trap v2 hit with RandomProjection D-12. HDBSCAN uses an exact-labels-up-to-permutation hard gate (the v2 classifier exact-label rule), with -1 noise pinned in the permutation search, against sklearn.cluster.HDBSCAN as the primary oracle; hdbscan 0.8.44 as a secondary cross-check.

The two cross-cutting surface features — the Rust builder API and the Python shim — are extensions of already-shipped infrastructure, not greenfield work. Nine v2 estimators already carry hand-written T::builder() -> TBuilder -> build() -> Result (LinearSVR, LinearSVC, MBSGDClassifier, MBSGDRegressor, all five Naive Bayes variants), and the MlrsBase(BaseEstimator) shim already ships for the v1 12 estimators in mlrs-py/python/mlrs/base.py. v3 extends and retrofits both; derive-macro builder crates are explicitly rejected as they add proc-macro dependencies for a pattern already hand-written and proven, and they fight the hand-written any_estimator! PyO3 macro machinery.

## Key Findings

### Recommended Stack

Zero new runtime crate dependencies. The entire v3.0 surface — UMAP, HDBSCAN, KNN-graph, builders, shim — is implemented as new modules in the existing five-crate workspace. The workspace Cargo.toml [workspace.dependencies] block is unchanged. New oracle-regen dev-venv deps (umap-learn 0.5.12, hdbscan 0.8.44) are confined to the /tmp/oracle-venv pattern established in v1/v2 and never enter the shipped wheel.

**Core technologies (all pinned, all unchanged):**
- `cubecl` 0.10 (`default-features=false`): device-kernel layer; new v3 kernels (knn_graph symmetrize, umap_fuzzy_map, umap_layout_step, mutual_reach_map) follow the feature-free GATHER idiom — SharedMemory-free, atomic-free, no mutable-bool/F::INFINITY/shift-loops
- `cubek-reduce` 0.2.0: per-point sigma/rho row reductions (UMAP fuzzy set) and core-distance (HDBSCAN k-th NN distance) already wired
- `pyo3` 0.28 / `arrow` 59 (pyarrow): HARD pin — arrow-59's pyarrow transitively pins pyo3 0.28; mixing ABIs crashes wheel at import (D-09/PY-05); UMAP/HDBSCAN add #[pyclass] wrappers only
- `mimalloc` 0.1, `bytemuck` 1, `thiserror` 2, `anyhow` 1, `log`/`env_logger` 0.4/0.11: all unchanged

**Builder strategy: hand-written builders + runtime fitted-flag. Reject all derive-macro crates.**
The hand-rolled convention is already shipped on 9 estimators. v3 retrofits the same shape across the remaining ~24 estimators plus the 2 new ones, and adds Option<DeviceArray> fitted-flag runtime state (not per-param PhantomData typestate, which would explode the generic surface and break the any_estimator! enum). bon, typed-builder, and derive_builder are all rejected: proc-macro dependencies for a pattern already proven, fighting the hand-written trait-impl machinery.

**Oracle libraries (dev-venv only, never shipped):**
- `umap-learn` 0.5.12: UMAP property gate only; pulls numba transitively (oracle-regen watch-item — numba may lag numpy 2.5.0; mitigation: pin compatible numba at regen time; CI uses committed .npz blobs and is fully decoupled)
- `hdbscan` 0.8.44: HDBSCAN cross-check oracle; no numba dep — clean against existing pins
- `sklearn.cluster.HDBSCAN` (scikit-learn >= 1.6, already pinned): PRIMARY HDBSCAN gate oracle; zero new dependency

### Expected Features

**Must have (v3.0 table stakes):**
- **KNN-graph primitive** — (n x k) neighbor indices + distances; self-inclusion control (k+1/self-drop); directed output; standalone value-gated (<=1e-5 vs sklearn.neighbors.NearestNeighbors); consumed by both UMAP and HDBSCAN
- **UMAP** fit/fit_transform -> embedding_ (n, n_components); table-stakes params (n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', init='spectral'/'random', random_state, n_epochs); stages 1-4 value-gated at <=1e-5, stage 5 property-gated
- **HDBSCAN** fit/fit_predict -> labels_ (-1 = noise), probabilities_; table-stakes params (min_cluster_size=5, min_samples=None->min_cluster_size, cluster_selection_method='eom'/'leaf'); exact-label gate via metric='precomputed'
- **Rust builder + typestate convention** — T::builder().setter().build()? -> T (runtime fitted-flag; build() validates params; new() stays as thin wrapper for any_estimator! compatibility); retrofit across all 30 existing + 2 new estimators
- **Pure-Python sklearn shim** — extend MlrsBase subclassing to v2 18 + UMAP/HDBSCAN; get_params/set_params/clone semantics; PyO3-wrap UMAP/HDBSCAN; FFI-free invariant unit tests in CI

**Should have (differentiators):**
- f32 + f64 device path for both algorithms (cuML is GPU f32-only for both — differentiator)
- Single shared KNN-graph prim feeding both UMAP and HDBSCAN (primitive-first; reusable by future SpectralEmbedding affinity)
- outlier_scores_ (GLOSH) on HDBSCAN — differentiator vs sklearn.cluster.HDBSCAN (which omits it)

**Defer to v3.x / future:**
- UMAP transform (new-data embedding) — fast-follow once fit property gate is stable
- HDBSCAN approximate_predict/membership_vector, store_centers, condensed-tree plot objects
- Supervised/semi-supervised UMAP, inverse_transform
- Approximate/NN-Descent KNN graph build (exact brute-force is correct for v3 oracle sizes)
- Custom callable metrics (no numba on CubeCL)
- Live FFI check_estimator re-triage (needs maturin+pyarrow host; routed to UAT)

### Architecture Approach

v3.0 extends the fixed five-crate workspace (mlrs-kernels -> mlrs-backend -> mlrs-algos -> mlrs-py -> scripts/fixtures) with new files only. The dependency graph is acyclic and unchanged. HDBSCAN uses the "device front-end, host tree back-end" split: the embarrassingly-parallel stages (distance, top-k, mutual-reachability map) run on device via GATHER; the MST (Prim's), condensed-tree, stability extraction, and cluster selection run in plain host Rust — deliberately avoiding the GPU-MST atomics wall that blocks RandomForest. UMAP uses a new umap_layout_step GATHER kernel (one thread per embedding vertex i, owning y[i], scanning incident edges + negative samples into private registers, single write) plus reuse of the v2 graph-Laplacian + eig prims for spectral init (capped at MAX_DIM=64 — init='random' is the default correctness path for realistic sizes).

**Major components (v3 deltas only):**
1. `mlrs-kernels`: knn_graph.rs (symmetrize map), umap_layout.rs (attract/repel GATHER), mutual_reach.rs (elementwise max), elementwise.rs += umap_fuzzy_map
2. `mlrs-backend/prims`: knn_graph.rs (distance+topk+symmetrize — the shared prim), umap_layout.rs (host epoch/neg-sample loop + layout kernel), mutual_reach.rs (device front-end), mst.rs (HOST Prim's), condense.rs (HOST condensed-tree + stability)
3. `mlrs-algos`: builder.rs (shared builder convention), manifold/umap.rs, cluster/hdbscan.rs
4. `mlrs-py`: estimators/manifold.rs (PyUMAP), extend cluster.rs + lib.rs, python/mlrs/manifold.py, extend cluster.py

### Critical Pitfalls

1. **KNN-graph reaches for atomics/SharedMemory (cpu-MLIR panic)** — Compose from the launch-proven top_k prim (distance -> top_k per row -> dense [n,k] pair); no new heap kernel; materialise the graph as [n,k] index+distance arrays (single-owner per row), never as scatter-built CSR. Guard: no Atomic/SharedMemory/F::INFINITY/mutable-bool/shift-loop imports in any KNN kernel. Spike the composed path under --features cpu before UMAP/HDBSCAN consume it.

2. **UMAP layout parallelised over edges (cpu-MLIR panic + nondeterminism)** — Invert to one thread per embedding vertex i (single owner of y[i]), scanning its incident edges + negative samples into private registers, writing y[i] once. This is the vertex-owner GATHER analog of the v2 SGD two-pass solver. Do NOT reuse prims/sgd.rs (wrong gradient/layout); build prims/umap_layout.rs with a new umap_layout_step kernel.

3. **UMAP value oracle instead of property gate** — Forcing <=1e-5 against umap-learn coordinates reports total failure on a correct implementation (SplitMix64 != NumPy RNG). Gate: (a) value-gate stages 1-4 (fuzzy set, spectral init) at <=1e-5 against umap-learn intermediate arrays; (b) property-gate stage 5 on trustworthiness + kNN-overlap relative to umap-learn's own scores + seed-reproducibility. Never a coordinate value oracle for the layout.

4. **HDBSCAN labels diverge from oracle (MST tie-breaking, noise label, min_samples semantics)** — Use stable-sort MST edge ordering (matching np.argsort reference); extend label_perm to pin -1->-1 (noise is not permutable); resolve min_samples=None->min_cluster_size exactly matching hdbscan semantics; gate against sklearn.cluster.HDBSCAN (not cuML — cuML's epsilon/outlier paths are incomplete). Use jittered AND explicitly-tied fixtures.

5. **Builder retrofit breaks any_estimator! PyO3 machinery** — Keep new() as a thin wrapper over builder().build().expect("defaults valid"); builder is additive. Establish the convention + pilot on 1-2 estimators with the PyO3 suite green before the 28-estimator mechanical sweep. No per-param PhantomData (typestate explosion). Audit every any_estimator! call site when touching any constructor.

## Implications for Roadmap

Based on all four research streams, the milestone requires five phases (continuing from v2's Phase 11 -> Phase 12+). Two sequencing variants were proposed across the research files; both are presented here so the roadmapper can decide.

---

### Variant A: Builder convention leads (ARCHITECTURE.md preferred ordering)

**Phase 12 — Builder Convention + Typestate Foundation**
**Rationale:** Establish the shared builder convention FIRST so UMAP/HDBSCAN (Phases 14-15) are born builder-fronted, avoiding a retroactive re-touch. Pure API work, no algorithm, lowest risk, unblocks everything downstream. Does NOT retrofit existing estimators yet (that is Phase 16).
**Delivers:** mlrs-algos/src/builder.rs (shared builder macro/convention; runtime fitted-flag via Option<DeviceArray>; reuse existing BuildError); convention applied to the 2 new estimator shells.
**Addresses:** Builder DX feature; foundation for the 30-estimator retrofit.
**Avoids:** Pitfall 9 (typestate explosion / any_estimator! break) by establishing the safe convention before scale.
**Research flag:** STANDARD PATTERN — the convention is already proven on 9 estimators; no additional phase research needed.

**Phase 13 — KNN-Graph Primitive (the feasibility keystone)**
**Rationale:** The shared substrate both UMAP and HDBSCAN depend on; must be standalone-validated before either consumes it (primitive-first discipline). The cpu-MLIR feasibility answer lives here.
**Delivers:** prims/knn_graph.rs (distance+topk+symmetrize); knn_graph_test.rs (exact gate vs sklearn.neighbors.NearestNeighbors; PoolStats memory gate); self-inclusion parameter; directed output contract.
**Addresses:** KNN-graph primitive feature; UMAP + HDBSCAN prerequisite.
**Avoids:** Pitfalls 1, 3, 4 (atomics/SharedMemory; self-neighbour/symmetrisation; n x n memory overflow).
**Research flag:** SPIKE NEEDED before planning — confirm the composed-from-top_k path launches on --features cpu (MEDIUM confidence; precedent is favorable but not verified for this exact composition).

**Phase 14 — UMAP**
**Rationale:** Hard dep on Phase 12 (builder) and Phase 13 (KNN-graph). The stochastic layout introduces a new GATHER kernel (umap_layout_step) that requires a cpu-MLIR spike. UMAP is file-disjoint from HDBSCAN.
**Delivers:** prims/umap_layout.rs + umap_layout_step kernel; umap_fuzzy_map kernel; manifold/umap.rs estimator; estimators/manifold.rs PyUMAP; python/mlrs/manifold.py shim; property-gate test suite (trustworthiness + kNN-overlap + seed-reproducibility vs umap-learn 0.5.12) + value gate for stages 1-4.
**Addresses:** UMAP feature; spectral init reuses v2 graph-Laplacian + eig prims (default init='random' for realistic sizes).
**Avoids:** Pitfalls 2 and 5 (edge-scatter SGD; wrong oracle type).
**Research flag:** SPIKE NEEDED before planning — vertex-owner GATHER umap_layout_step under cpu-MLIR (MEDIUM confidence); also exact property-gate thresholds need measurement on a first fixture run.

**Phase 15 — HDBSCAN**
**Rationale:** Hard dep on Phase 12 (builder) and Phase 13 (KNN-graph); feature-disjoint from Phase 14 (can be built in parallel after Phase 13 if resources allow). Device front-end + host tree back-end split is clear.
**Delivers:** prims/mutual_reach.rs + mutual_reach_map kernel; prims/mst.rs (HOST Prim's); prims/condense.rs (HOST condensed-tree + stability); cluster/hdbscan.rs estimator; PyHDBSCAN; cluster.py shim extension; exact-label test suite (label_perm with -1-pinned; jittered + tied fixtures; precomputed-distance oracle path).
**Addresses:** HDBSCAN feature; device/host split dodges GPU-tree-atomics wall.
**Avoids:** Pitfalls 6 and 7 (label divergence; MST/tree atomics + float reorder).
**Research flag:** SPIKE NEEDED before planning — confirm host MST + condensed-tree matches hdbscan reference exactly on tie-heavy fixtures.

**Phase 16 — Builder Retrofit Sweep + Shim Coverage**
**Rationale:** The one broad-edit, parallel-unsafe phase. Touching ~24 existing estimator files must be isolated so it never contends with new-estimator phases. Placed last to minimize blast radius.
**Delivers:** Builder convention retrofitted to the ~24 pre-Phase-10 estimators (new() kept as thin wrappers); test_params.py/test_estimator_checks.py extended to UMAP/HDBSCAN + retrofitted set; all FFI-free invariant unit tests (get_params/set_params/clone/double-fit reset/pre-fit-raises) green in CI.
**Addresses:** 30-estimator builder retrofit; full Python shim coverage.
**Avoids:** Pitfalls 9, 10, 11 (typestate explosion; default drift; check_estimator failures).
**Research flag:** STANDARD PATTERN — the retrofit is mechanical; the any_estimator! compatibility invariants are well-understood from v2. Pilot on 1-2 estimators under the PyO3 suite before the 28-estimator sweep.

---

### Variant B: Retrofit trails (alternative — PITFALLS.md and STACK.md implicit ordering)

The PITFALLS.md phase numbering assumed: Phase 12 = KNN-graph prim, Phase 13 = UMAP, Phase 14 = HDBSCAN, Phase 15 = builder convention + retrofit, Phase 16 = sklearn shim. Under this variant, UMAP and HDBSCAN are born with new(...) constructors and the builder is added in a cleanup sweep afterward.

**Trade-off:** Under Variant B, UMAP and HDBSCAN must be lightly re-touched in Phase 15 to add the builder front door. Under Variant A, the builder is established first (one extra phase before any algorithm work) and the algorithm phases carry it from birth. Both variants agree the retrofit sweep is the higher blast-radius work and must be isolated as its own phase. The disagreement is only whether the convention-foundation phase belongs first (A) or after the algorithms are working (B).

**Recommendation for roadmapper:** Variant A is preferred. The cost of a dedicated builder-convention phase (Phase 12) is low (pure API, no device work, no algorithm risk); the benefit is that UMAP and HDBSCAN are born idiomatic and never need a re-touch. Variant B is acceptable if the roadmapper wants algorithm value (KNN graph, UMAP, HDBSCAN) to land as early as possible and is comfortable with a minor re-touch to UMAP/HDBSCAN in the retrofit sweep.

---

### Phase Ordering Rationale

- KNN-graph before UMAP/HDBSCAN is mandatory in both variants: it is the shared substrate; primitive-first discipline requires standalone validation before consumers exist.
- UMAP and HDBSCAN are file-disjoint (manifold/ vs cluster/, separate prim sets) and can be planned/built in parallel after the KNN-graph prim lands — same pattern as v2's parallel five-family wave after PRIM-06 through PRIM-10.
- Retrofit sweep last in both variants: the ~24-file broad edit is the one parallel-unsafe work item in the milestone; isolating it prevents merge contention with algorithm phases.
- Builder convention leads in Variant A because UMAP/HDBSCAN adopting it from birth is cheaper than a retroactive re-touch; in Variant B the algorithms land first and the convention arrives as cleanup.

### Research Flags

Phases needing spikes before planning (MEDIUM confidence unknowns):

- **Phase 13 (Variant A) / Phase 12 (Variant B) — KNN-graph prim:** Spike the composed-from-top_k GATHER path (distance -> top_k -> dense [n,k] -> symmetrize map) under --features cpu. Precedent is strongly favorable (v2 top_k launches on cpu-MLIR first try; same idiom), but the symmetrize-map step is new and must be confirmed.
- **Phase 14 (Variant A) / Phase 13 (Variant B) — UMAP:** Spike the vertex-owner umap_layout_step GATHER kernel on cpu-MLIR before planning the estimator. Also determine the exact property-gate thresholds (trustworthiness floor relative to umap-learn) by running a small prototype.
- **Phase 15 (both variants) — HDBSCAN:** Spike host MST (Prim's) + condensed-tree exactness vs the hdbscan reference on a tied-distance fixture. MST tie-breaking is the main label-divergence risk.

Phases with standard patterns (no phase-level research needed):

- **Phase 12 (Variant A) — Builder convention:** The pattern is proven on 9 shipped estimators; BuildError already exists; the runtime fitted-flag via Option<DeviceArray> matches gaussian_nb.rs's shipped state model. Plan directly.
- **Phase 16 (Variant A) / Phase 15-16 (Variant B) — Builder retrofit + shim coverage:** Mechanical retrofit following the established convention. The any_estimator! compatibility rules are well-documented from v2. Pilot on 1-2 estimators before sweeping; no dedicated research phase.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies; all pins verified from shipped workspace Cargo.toml and pyproject.tomls; oracle-venv strategy proven from v1/v2. One watch-item: numba/numpy-2.5 compat in the oracle-regen venv — fully mitigated by committed .npz fixture blobs (CI is decoupled). |
| Features | HIGH | Algorithm parameter surfaces verified against cuML v26.08 source (local), umap-learn and hdbscan stable documented APIs, and sklearn estimator contract. MEDIUM item: exact f32-on-rocm band magnitudes for UMAP/HDBSCAN continuous outputs — measured empirically per family during Phase 14/15 validation, same posture as every prior phase. |
| Architecture | HIGH (placement/boundaries/device-host split) / MEDIUM (two execution unknowns) | File placement, trait deltas, data-flow seams, and the device-host split are grounded in shipped code reads. Two MEDIUM items: (1) umap_layout_step single-owner GATHER on cpu-MLIR (precedent strongly favorable but unconfirmed); (2) host MST/condensed-tree exactness on tie-heavy fixtures. Both are SPIKE NEEDED before their respective phase plans. |
| Pitfalls | HIGH | Every pitfall is grounded in v1/v2 codebase idioms, project memory, CONCERNS.md cuML reference-behavior anti-patterns, and the hdbscan/umap-learn source and validation literature. The cuML UMAP vertex-parallel nondeterminism bug and the cuML HDBSCAN epsilon-path incompleteness are HIGH-confidence failure modes to avoid. |

**Overall confidence:** HIGH for the approach; MEDIUM for two specific execution spikes that must precede Phase 14 and Phase 15 planning.

### Gaps to Address

- **UMAP layout-step cpu-MLIR feasibility (spike before Phase 14):** Run a minimal umap_layout_step kernel (one thread per vertex, scan neighbors, write embedding row) under --features cpu before committing to the phase plan. If it panics, the ARCHITECTURE.md host-loop fallback for the inner step is the immediate mitigation. Resolution: spike takes < 1 day.
- **Host MST tie-breaking exactness (spike before Phase 15):** Run Prim's MST + condensed-tree on a small fixture with deliberate tied mutual-reachability edges against hdbscan Python reference. Lock the tie-breaking convention (stable-sort on weight, lowest-index tiebreak for equal weights) before the phase plan commits to the exact-label gate. Resolution: spike takes < 1 day.
- **UMAP property-gate thresholds (measure during Phase 14 fixture generation):** The trustworthiness and kNN-overlap floors are relative-to-umap-learn (not hard absolute values) and must be calibrated on the first oracle fixture run. Design the gate as mlrs_trustworthiness >= umap_learn_trustworthiness - delta and measure delta empirically.

## Sources

### Primary (HIGH confidence)
- mlrs codebase (shipped, direct reads): crates/mlrs-algos/src/{traits,error}.rs, crates/mlrs-algos/src/{naive_bayes,linear,cluster}/*, crates/mlrs-backend/src/prims/{topk,distance,eig,laplacian,sgd}.rs, crates/mlrs-py/src/{dispatch,lib}.rs, crates/mlrs-py/python/mlrs/base.py — confirm builder/shim partial existence, top_k GATHER idiom, any_estimator! structure, BuildError, MlrsBase
- .planning/PROJECT.md (v3 scope, gate regime, primitive-first discipline, phase numbering, deferred items)
- .planning/milestones/v2.0-research/{STACK,FEATURES,ARCHITECTURE,PITFALLS,SUMMARY}.md (GATHER idiom validation, RandomProjection D-12 property-gate precedent, exact-label D-08 rule, pyo3 0.28/arrow-59 ABI pin)
- Project memory: cubecl-cpu-no-shared-memory.md, rocm-is-runnable-gpu-gate.md, oracle-fixture-regen-needs-venv.md, cubecl-algo-crates-moved-to-cubek.md
- cuML v26.08 source (local, read-only): cuml-main/python/cuml/cuml/manifold/umap/umap.pyx, cuml-main/python/cuml/cuml/cluster/hdbscan/hdbscan.pyx — algorithm param surfaces and defaults; CONCERNS.md — vertex-parallel nondeterminism bug, epsilon-path incompleteness warnings
- PyPI JSON API (2026-06-22): umap-learn 0.5.12, hdbscan 0.8.44, scikit-learn 1.9.0 version/dep metadata
- crates.io API (2026-06-22): bon 3.9.3, typed-builder 0.23.2, derive_builder 0.20.2 — rationale for rejection
- scikit-learn estimator development guide: check_estimator invariants, get_params/set_params/clone contract, trailing-_ fitted-attr convention
- hdbscan reference source (scikit-learn-contrib/hdbscan/hdbscan_.py): stable-sort MST edge ordering, noise -1, probabilities_ formula, min_samples semantics

### Secondary (MEDIUM confidence)
- UMAP validation practice literature: trustworthiness/continuity/kNN-preservation as structure-preservation gate; 5-15% coordinate variation across implementations confirms value-matching is infeasible
- GitHub lmcinnes/umap: numpy > 2.0 / numba compatibility discussion — corroborates the oracle-venv watch-item
- cubecl 0.10 cpu-MLIR lowering behavior: SharedMemory/atomic failure modes documented in project memory (empirical from v1/v2 codebase)

---
*Research completed: 2026-06-22*
*Ready for roadmap: yes*
