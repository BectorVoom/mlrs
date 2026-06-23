# Requirements: mlrs — v3.0 Manifold Algorithms & Rust-Native API

**Defined:** 2026-06-22
**Core Value:** Correct, memory-efficient ML algorithms that match scikit-learn within 1e-5, running on any CubeCL backend from a single generic codebase.

> **Oracle note for v3.** Two of these algorithms break the uniform "≤1e-5 vs scikit-learn" relationship and therefore carry explicit per-feature gate types:
> - **Value gate (≤1e-5):** KNN-graph prim and UMAP's deterministic stages 1–4 — vs `sklearn.neighbors.NearestNeighbors` / `umap-learn` intermediates.
> - **Property/structural gate (D-12 precedent):** UMAP's stochastic SGD layout — vs `umap-learn` 0.5.12 (SplitMix64 ≠ NumPy RNG → no coordinate match).
> - **Exact-labels-up-to-permutation gate (+ pinned −1 noise):** HDBSCAN — vs `sklearn.cluster.HDBSCAN` (primary, zero new dep) with `hdbscan` 0.8.44 cross-check; exact on `metric='precomputed'` f64.
>
> Backend gate unchanged: **cpu(f64) + rocm(f32)**; f64-on-rocm SKIPS-with-log. Zero new compute dependencies. Primitive-first: the KNN-graph prim lands + is standalone-gated before either estimator consumes it.

## v1 Requirements

Requirements for the v3.0 milestone. Each maps to a roadmap phase.

### KNN-Graph Primitive (shared substrate — build & gate FIRST)

- [ ] **PRIM-11**: A shared KNN-graph primitive returns ascending-ordered k-nearest-neighbor indices `(n, k)` and distances `(n, k)` over the v1 distance prim, with a self-inclusion parameter (UMAP self-excluded / HDBSCAN self-counted core distances), built cpu-MLIR-safe by composition (distance → top-k GATHER, no SharedMemory/atomics/heap kernel), standalone-validated exact vs `sklearn.neighbors.NearestNeighbors` (indices set-equal up to tie-ordering; distances ≤1e-5 f64) with a build-failing PoolStats memory gate.

### UMAP

- [ ] **UMAP-01**: User can fit UMAP (`fit` / `fit_transform`) to produce `embedding_` `(n, n_components)` with umap-learn/sklearn-named hyperparameters and defaults (`n_neighbors=15`, `n_components=2`, `metric='euclidean'`, `min_dist=0.1`, `spread=1.0`, `n_epochs=None`, `init='spectral'`, `random_state`, `learning_rate=1.0`, `set_op_mix_ratio=1.0`, `local_connectivity=1.0`, `repulsion_strength=1.0`, `negative_sample_rate=5`, `a`/`b` override), `min_dist ≤ spread` validated.
- [ ] **UMAP-02**: UMAP's deterministic stages — KNN graph, fuzzy simplicial set (smooth-kNN `ρ`/`σ` binary search), fuzzy-set union, and spectral init (reusing the v2 graph-Laplacian + v1 eig stack; random-init fallback above the Jacobi size cap) — value-match `umap-learn` intermediates to ≤1e-5 (f64).
- [ ] **UMAP-03**: UMAP's stochastic SGD layout (negative-sampling, new vertex-owner GATHER layout kernel) passes a property/structural gate vs `umap-learn` 0.5.12 — trustworthiness / kNN-overlap ≥ umap-learn − margin, downstream-ARI within band, and same-`random_state` reproducibility within mlrs — NOT coordinate value-match.
- [ ] **UMAP-04**: User can embed new data via `transform(X_new)` against the fitted fuzzy graph, gated by a property sub-gate on the new points.

### HDBSCAN

- [ ] **HDBS-01**: User can fit HDBSCAN (`fit` / `fit_predict`) to produce `labels_` (`-1` = noise) and `probabilities_` `∈[0,1]` with sklearn-named hyperparameters and defaults (`min_cluster_size=5`, `min_samples=None→min_cluster_size`, `cluster_selection_epsilon=0.0`, `cluster_selection_method='eom'` and `'leaf'`, `metric='euclidean'`, `alpha=1.0`, `max_cluster_size=0`), implemented as a device front-end (core distances + mutual-reachability) plus a host back-end (MST → single-linkage → condensed tree → EoM/leaf stability extraction), dodging the tree-atomics wall.
- [ ] **HDBS-02**: HDBSCAN `labels_` match `sklearn.cluster.HDBSCAN` (cross-checked vs `hdbscan` 0.8.44) exactly up to permutation with `-1` pinned (exact on `metric='precomputed'` f64; the label-perm helper extended to fix `-1→-1`), and `probabilities_` agree within a documented band; MST edge tie-breaking is stable-sorted with a documented deterministic rule.
- [ ] **HDBS-03**: User can read per-point `outlier_scores_` (GLOSH) from a fitted HDBSCAN — a differentiator vs `sklearn.cluster.HDBSCAN`, gated within band vs the `hdbscan` library.
- [ ] **HDBS-04**: User can request cluster centers via `store_centers` (`'centroid'`/`'medoid'`) producing `centroids_`/`medoids_` attributes (sklearn parity).

### Rust-Native Builder API

- [ ] **BLDR-01**: User can construct any estimator via an idiomatic Rust builder — `T::builder().param(..).…build() -> Result<T<Unfit>, BuildError>` — with owned chained setters, sklearn-equal defaults, and typed `thiserror` validation variants (single-source defaults so `T::builder().build()? == T::new()` == sklearn default).
- [x] **BLDR-02**: The fit/unfit distinction is modeled as compile-time typestate (`T<Unfit>` → `T<Fitted>`); `predict` / `transform` / fitted-attr accessors exist only on `T<Fitted>`, preventing predict-before-fit at compile time (the hybrid Rust-surface design).
- [ ] **BLDR-03**: The builder + typestate convention is retrofitted across all existing estimators **additively** (builder constructs the existing config struct; fit path untouched), piloted on 1–2 estimators under the green suite before the full sweep, preserving every shipped 1e-5 / exact-label gate.
- [x] **BLDR-04**: The PyO3 surface is unchanged — the Rust typestate collapses behind the existing `any_estimator!` `Unfit/F32/F64` enum, with a runtime `NotFittedError` analog at the Python boundary.

### Python sklearn Shim

- [ ] **SHIM-01**: Every estimator's pure-Python class stores each constructor arg unchanged in `__init__` (no validation/computation) and exposes `get_params(deep=True)` / `set_params(**kw)` that round-trip exactly and are `clone()`-compatible (extends the existing `MlrsBase` shim from the v1 12 to the v2 18 + the two new).
- [ ] **SHIM-02**: UMAP and HDBSCAN are PyO3-wrapped (`#[pyclass]` on the existing `any_estimator!` machinery, GIL release, `guard_f64` before F64) with sklearn-named params, trailing-underscore fitted attrs, `n_features_in_` set/enforced, `fit` returns `self`, and the correct surface (UMAP `transform`/`fit_transform`; HDBSCAN `fit_predict`/`labels_`).
- [ ] **SHIM-03**: The shim is verified by Rust-side unit tests plus a static Python check; the live `estimator_checks` / `check_estimator` run stays deferred (needs a maturin+pyarrow host this environment lacks).

## v2 Requirements

Deferred to a future milestone. Tracked but not in the v3.0 roadmap.

### Tier-3 hard algorithms (later milestone)

- **RandomForest → FIL → TreeSHAP** — the keystone tree stack; GPU tree construction needs atomics/histogram-split that fight cpu-MLIR; requires a make-or-break feasibility spike first.
- **ARIMA / AutoARIMA** — batched Kalman filter + batched L-BFGS + order search.
- **Kernel SVC / SVR (SMO)** — the SMO solver; linear SVM (LinearSVC/SVR) already shipped in v2.

## Out of Scope

Explicitly excluded from v3.0. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Element-wise 1e-5 match of UMAP `embedding_` | SGD + negative sampling + per-PRNG shuffle; SplitMix64 ≠ NumPy MT → coordinates can't match. Use the property gate (UMAP-03). |
| Supervised / semi-supervised UMAP (`target_metric`) | Doubles the fuzzy-graph machinery for a niche use; no clean property gate. |
| UMAP `inverse_transform` (embedding → original) | Needs Qhull/Delaunay; host-only, large surface, no device value. |
| HDBSCAN `approximate_predict` / `membership_vector` (new-point predict) | Needs persisted prediction-data structures; large surface — defer to v3.x. |
| HDBSCAN condensed-tree / dendrogram plot objects | Pure-Python inspection surface, no algorithmic value, no oracle. |
| Approximate / NN-Descent / tree KNN-graph build | Fights cpu-MLIR (no SharedMemory) and the approximation breaks the exact-label HDBSCAN gate. Brute-force exact KNN only. |
| Custom / callable metrics | No numba on CubeCL; unbounded surface, no oracle. Fixed string metrics (euclidean; optionally manhattan/cosine). |
| Native sparse KNN-graph path | Densify at ingress for v3. |
| Live FFI `estimator_checks` re-triage | Needs a maturin+pyarrow host this environment lacks (SHIM-03 covers the static path). |
| Builder retrofit that rewrites estimator fit bodies | Touching 30 fit paths risks regressing shipped gates; retrofit is an additive front door (BLDR-03). |
| RandomForest/trees, ARIMA, kernel-SVM/SMO, SHAP/explainers, genetic, cuml.accel, Dask multi-GPU | Deferred past v3 (see v2 Requirements above + `notes/v3-hard-algorithm-backlog.md`). |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PRIM-11 | Phase 13 | Pending |
| UMAP-01 | Phase 14 | Pending |
| UMAP-02 | Phase 14 | Pending |
| UMAP-03 | Phase 14 | Pending |
| UMAP-04 | Phase 14 | Pending |
| HDBS-01 | Phase 15 | Pending |
| HDBS-02 | Phase 15 | Pending |
| HDBS-03 | Phase 15 | Pending |
| HDBS-04 | Phase 15 | Pending |
| BLDR-01 | Phase 12 | Pending |
| BLDR-02 | Phase 12 | Complete |
| BLDR-03 | Phase 16 | Pending |
| BLDR-04 | Phase 12 | Complete |
| SHIM-01 | Phase 16 | Pending |
| SHIM-02 | Phase 16 | Pending |
| SHIM-03 | Phase 16 | Pending |

**Coverage:**
- v3.0 requirements: 16 total
- Mapped to phases: 16 ✓
- Unmapped: 0 ✓

**Phase distribution:**
- Phase 12 (Builder + Typestate Convention): BLDR-01, BLDR-02, BLDR-04 (3)
- Phase 13 (KNN-Graph Primitive): PRIM-11 (1)
- Phase 14 (UMAP): UMAP-01, UMAP-02, UMAP-03, UMAP-04 (4)
- Phase 15 (HDBSCAN): HDBS-01, HDBS-02, HDBS-03, HDBS-04 (4)
- Phase 16 (Builder Retrofit Sweep + Shim Coverage): BLDR-03, SHIM-01, SHIM-02, SHIM-03 (4)

> **Note on BLDR split:** the builder *convention* (BLDR-01/02/04) leads in Phase 12 so UMAP/HDBSCAN are born builder-fronted; the broad, parallel-unsafe 30-estimator *retrofit sweep* (BLDR-03) is isolated to Phase 16 to preserve file-disjoint discipline and protect the shipped 1e-5/exact gates (Variant A, per all four research streams).

---
*Requirements defined: 2026-06-22*
*Last updated: 2026-06-22 after roadmap creation — all 16 v3.0 requirements mapped to Phases 12–16*
