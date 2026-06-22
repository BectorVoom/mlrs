# Feature Research

**Domain:** Manifold learning (UMAP) + density clustering (HDBSCAN) + shared KNN-graph primitive + a Rust-native builder/typestate API + a pure-Python sklearn shim — added to a Rust/CubeCL rewrite of RAPIDS cuML.
**Researched:** 2026-06-22
**Confidence:** HIGH for sklearn/umap-learn/hdbscan algorithm semantics, parameter surfaces, and Rust builder/typestate idioms (stable, documented, cross-checked against the local cuML v26.08 `manifold/umap/umap.pyx` + `cluster/hdbscan/hdbscan.pyx` source). MEDIUM for the exact correctness-gate band magnitudes (must be measured on hardware) and for the precise MST/condensed-tree tie-breaking that drives HDBSCAN label exactness.

> **Framing.** v3 adds two *stochastic / structural* algorithms whose oracle relationship differs from v1/v2. UMAP's layout is SGD with negative sampling → **no element-wise 1e-5**; oracle is `umap-learn` (CPU) and the gate is a **property/structural** one (à la v2 RandomProjection D-12). HDBSCAN is deterministic given a fixed KNN graph → oracle is `hdbscan` / `sklearn.cluster.HDBSCAN` and the gate is **exact labels up to permutation + noise label** (à la v2 classifiers). Both consume the same new **KNN-graph primitive**, which is the feasibility-critical, primitive-first deliverable (land + validate standalone before either estimator). The Rust builder API and Python shim are *surface* features (no new kernels), retrofitted across the existing 30 estimators.

---

## Feature Landscape

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **KNN-graph primitive**: k-NN indices `(n, k)` + distances `(n, k)` over the existing distance prim | Both UMAP and HDBSCAN are *defined on* a k-NN graph; it is the shared substrate and the v3 feasibility keystone | MEDIUM | Built on v1 `NearestNeighbors` (brute-force top-k). cpu-MLIR GATHER idiom (no SharedMemory/atomics). Must expose **self-inclusion control** (UMAP excludes self; HDBSCAN's `min_samples` core-distance needs self-or-not handled consistently) and **directed (raw) output** (symmetrization is the consumer's job). |
| **UMAP**: `fit` / `transform` / `fit_transform` → `embedding_` `(n, n_components)` | The canonical nonlinear dim-reduction / visualization estimator; the headline v3 feature | HIGH | Stages: KNN graph → fuzzy simplicial set (smooth-kNN `ρ`/`σ` per point) → fuzzy set union (symmetrize via `set_op_mix_ratio`) → spectral init → SGD layout with negative sampling. `transform` on new data uses the *fitted* graph/embedding. Stochastic → property gate. |
| UMAP table-stakes hyperparameters | umap-learn users expect exact param names/defaults | — | `n_neighbors=15`, `n_components=2`, `metric='euclidean'`, `min_dist=0.1`, `n_epochs=None`(→200 large / 500 small N), `init='spectral'`, `random_state=None`. These are the "everyone touches them" knobs. |
| **HDBSCAN**: `fit` / `fit_predict` → `labels_`, `probabilities_` | The canonical density clusterer that handles variable-density clusters + noise without `eps`; headline v3 feature | HIGH | Stages: core distances (`min_samples`-th NN) → mutual-reachability distance → MST (over MR-distance) → single-linkage hierarchy → condensed tree (`min_cluster_size`) → stability-based cluster extraction. Deterministic given the KNN graph. `labels_` uses **-1 for noise**. |
| HDBSCAN table-stakes hyperparameters | hdbscan / sklearn.cluster.HDBSCAN users expect exact names/defaults | — | `min_cluster_size=5`, `min_samples=None`(→`min_cluster_size`), `cluster_selection_epsilon=0.0`, `cluster_selection_method='eom'`, `metric='euclidean'`, `alpha=1.0`. |
| HDBSCAN `probabilities_` (∈[0,1], 0 for noise) | Soft membership strength is a defining HDBSCAN output; users index/plot on it | MEDIUM | Per-point = scaled position within its cluster's λ (death) range; exact formula must match the reference. |
| **sklearn-named constructor params + trailing-underscore fitted attrs + `n_features_in_`** | Every mlrs estimator already honors this; v3 must not regress the convention | LOW | UMAP: `embedding_`, `n_features_in_`. HDBSCAN: `labels_`, `probabilities_`, `n_features_in_`. |
| **f32 + f64 generic device path** for both new algorithms | Project core value; v1/v2 set the precedent | — | Gate = cpu(f64) + rocm(f32); f64-on-rocm SKIPS-with-log (cubecl-cpp 0.10 F64 unregistered for HIP). Per-file skip-guard line as in v2. |
| **Rust-native builder + `.fit()`** for the two new estimators *and* a uniform convention retrofit across the 30 existing | "today's surface is sklearn-mirror, consumed mainly via PyO3"; v3's stated goal is an idiomatic Rust caller surface | MEDIUM (convention) / MEDIUM-HIGH (30-estimator retrofit churn) | `Umap::builder().n_neighbors(15).min_dist(0.1).build()?` → `.fit(&x)?`. `build()` returns `Result` (validates params). `fit` returns `Result` (validates data). |
| **Pure-Python sklearn shim**: `get_params` / `set_params` / passes `check_estimator` | Carried-forward v2 debt; required for the estimators to be drop-in sklearn objects (pipelines, grid-search) | MEDIUM | `get_params(deep=True)` returns the constructor kwargs verbatim; `set_params(**kw)` mutates + revalidates; clone-ability (no fitted state copied) and the `__init__`-stores-args-unchanged rule are the load-bearing `check_estimator` requirements. |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Single shared KNN-graph prim** feeding both UMAP and HDBSCAN (+ reusable by future SpectralEmbedding `nearest_neighbors` affinity) | One validated, memory-efficient device prim instead of two ad-hoc graphs; embodies the project's primitive-first discipline | MEDIUM | Lands + gated standalone before consumers; mirrors v2's kernel-matrix/Laplacian/SGD prim pattern. |
| **f64 device UMAP/HDBSCAN** | umap-learn / hdbscan are CPU-f64; cuML is GPU-**f32-only** for both. mlrs offers GPU **and** f64. | — | f64 makes the HDBSCAN MST/MR-distance ties and UMAP smooth-kNN root-find numerically comfortable; differentiator vs both references. |
| **Typestate fit/unfit Rust API** (`Umap<Unfit>` → `Umap<Fitted>`; `predict`/`transform` only exist on `Fitted`) | Compile-time prevention of "predict before fit" — a class of bug sklearn/cuML catch only at runtime (`NotFittedError`) | MEDIUM | Idiomatic Rust; the PyO3 layer collapses the typestate behind the existing `Unfit/F32/F64` enum. Nice differentiator over a naive port. |
| **`outlier_scores_` (GLOSH)** on HDBSCAN | The `hdbscan` library exposes per-point outlier scores; valued for anomaly detection | MEDIUM-HIGH | Differentiator *only vs sklearn.cluster.HDBSCAN* (which omits it). Optional — can ship as a v3.x follow-on. |
| **Builder param-validation at `build()`** with typed errors (`thiserror`) | Caller gets a structured error (`InvalidNNeighbors`, `MinDistGtSpread`) before any device work | LOW | Matches the existing error convention (thiserror in libs, anyhow at boundaries). |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Element-wise 1e-5 match of UMAP `embedding_` vs umap-learn** | "match the oracle like every other estimator" | UMAP layout is SGD + negative sampling + per-PRNG shuffle; even umap-learn isn't reproducible across BLAS/threads. mlrs SplitMix64 ≠ NumPy MT → coordinates can't match | **Property/structural gate** (D-12 style): kNN-recall / trustworthiness of the embedding, local-neighborhood preservation, cluster separability (silhouette/ARI of a downstream KMeans on the embedding vs on umap-learn's), seed-reproducibility within mlrs. NOT coordinate value-match. |
| **`tree`/`ball_tree` / NN-Descent / approximate-KNN graph build** | cuML uses `nn_descent`; umap/hdbscan support trees for speed | Tree/NN-descent builds fight cpu-MLIR (no SharedMemory) and add an approximation that breaks the exact-label HDBSCAN gate | **Brute-force exact KNN** (v1 top-k). Exact graph → deterministic HDBSCAN labels → clean gate. Document the O(n²) size ceiling. |
| **Custom/callable metrics** (UMAP & HDBSCAN) | Both references accept arbitrary callables/`metric_params` | No numba on CubeCL; unbounded surface, no oracle | **Fixed string metrics** reusing the v1 distance prim: `euclidean` (+`manhattan`/`cosine` if cheap). Raise on unsupported. |
| **UMAP supervised / `target_metric` / semi-supervised** (`fit(X, y)` to steer layout) | umap-learn & cuML support it | Doubles the fuzzy-graph machinery (target graph blend) for a niche use; no clean property gate | Unsupervised `fit(X)` only for v3; raise/ignore `y`. Defer supervised UMAP. |
| **UMAP `inverse_transform`** (embedding → original space) | cuML/umap-learn expose it (Delaunay-based in cuML) | Needs Qhull/Delaunay (cuML does it on host via scipy) — large surface, host-only, no device value | Omit; `transform` (new data → embedding) is the table-stakes direction. |
| **HDBSCAN `approximate_predict` / `membership_vector` / soft clustering** | hdbscan library prediction API for new points | Needs the persisted prediction-data structures (condensed-tree internals); large surface | Ship `fit_predict` (the 95% use). Defer new-point prediction to v3.x. |
| **HDBSCAN `cluster_selection_method='leaf'` + the full `gen_min_span_tree`/dendrogram plotting attrs** | hdbscan library exposes `condensed_tree_`, `single_linkage_tree_`, `minimum_spanning_tree_` plot objects | Plotting/inspection objects are pure-Python surface with no algorithmic value and no oracle | Support both `'eom'` (default) and `'leaf'` selection (cheap, same condensed tree); skip the plot wrapper objects (or expose raw arrays only). |
| **Builder retrofit that rewrites estimator internals** | "make it idiomatic everywhere at once" | Touching 30 estimator bodies risks regressing the shipped 1e-5 gates | **Additive builder layer**: builder constructs the *existing* config struct; `fit` path unchanged. Retrofit = new front door, not surgery. |
| **`check_estimator` via live FFI `estimator_checks`** | "prove sklearn compliance end-to-end" | Needs maturin+pyarrow host this env lacks (explicitly deferred in PROJECT) | **Pure-Python shim** implementing get_params/set_params/clone semantics; live `estimator_checks` re-triage stays deferred. |

---

## Per-Feature Detail

### Feature 1 — KNN-graph primitive (the shared substrate; build FIRST)

**New prim.** `prims/knn_graph.rs` over the v1 brute-force distance + top-k. Feature-free, GATHER idiom, standalone PoolStats-gated.

- **Contract (what consumers need):** for each of `n` rows, the `k` nearest neighbor **indices** `(n, k)` and **distances** `(n, k)`, in ascending-distance order.
- **Self-inclusion:** must be a parameter.
  - **UMAP** wants the `k` nearest *excluding self* (umap-learn computes `n_neighbors` neighbors; the smooth-kNN `ρ` is the distance to the nearest *non-zero* neighbor). Practically: request `k+1`, drop the self column (distance 0), keep `k`.
  - **HDBSCAN** core distance is the distance to the `min_samples`-th neighbor; the reference convention **includes the point itself** as the 0th neighbor (so `core_dist = dist to the min_samples-th neighbor counting self`). Pin this counting convention against the oracle — off-by-one here shifts every core distance and breaks the exact-label gate.
- **Directed vs symmetric:** the prim returns the **directed/raw** k-NN (row i's neighbors). Symmetrization is consumer-specific:
  - UMAP: fuzzy-set **union** with `set_op_mix_ratio` (a *weighted* symmetrization, `A + Aᵀ − mix·A∘Aᵀ`), not a plain `max`/`0.5(W+Wᵀ)`.
  - HDBSCAN: builds an MST over mutual-reachability distance, which is symmetric by construction (`mreach(a,b)=max(core_a, core_b, d(a,b))`); it does **not** need the kNN graph symmetrized, only the core distances + pairwise (or kNN-restricted) MR distances.
- **Complexity:** O(n²·d) brute-force distance + O(n·k) top-k (reuses v1). Size-ceiling documented (same as v2 dense-Gram ceiling).
- **Gate:** exact vs a NumPy/sklearn `NearestNeighbors` oracle (indices set-equal up to tie-ordering; distances 1e-5). This is a *value* gate — the prim itself is deterministic.
- **cpu-MLIR:** one-thread-per-(row) GATHER scan for top-k; no SharedMemory, no atomics (proven idiom from v1 top-k).

### Feature 2 — UMAP

**Oracle:** `umap-learn` (CPU, f64). cuML's `umap.pyx` is API-shape reference (confirmed: it exposes exactly the umap-learn param surface). **Gate:** property/structural, NOT 1e-5.

**Algorithm stages (each is a discrete, testable sub-step):**
1. **KNN graph** (Feature 1) — `n_neighbors` nearest, self-excluded.
2. **Fuzzy simplicial set** — per point find `ρ_i` (distance to nearest neighbor, modulo `local_connectivity`) and solve for `σ_i` by **binary search** so that `Σ_j exp(−max(0, d_ij − ρ_i)/σ_i) = log2(n_neighbors)`. This is the "smooth kNN distance". Edge weight `w_ij = exp(−max(0,d_ij−ρ_i)/σ_i)`.
3. **Fuzzy set union (symmetrize)** — `B = A + Aᵀ − set_op_mix_ratio · (A∘Aᵀ)` (probabilistic t-conorm; `set_op_mix_ratio=1.0` → pure union, `0.0` → intersection).
4. **Spectral init** (`init='spectral'`) — eigenvectors of the normalized Laplacian of `B` (reuse the v2 graph-Laplacian prim + v1 eig — smallest nontrivial, exactly the v2 SpectralEmbedding machinery). `init='random'` is the fallback. **This is a major prim-reuse win.**
5. **SGD layout optimization** — `n_epochs` of edge-sampling SGD with **negative sampling**: attract along graph edges, repel `negative_sample_rate` random non-edges, using the `a`/`b` curve from `find_ab_params(spread, min_dist)`. PRNG-driven → the stochastic core.

**Table-stakes hyperparameters (umap-learn defaults; confirmed against cuML source):**
| Param | Default | Role |
|-------|---------|------|
| `n_neighbors` | 15 | local vs global structure; KNN graph size |
| `n_components` | 2 | embedding dimensionality |
| `metric` | `'euclidean'` | distance for KNN graph |
| `min_dist` | 0.1 | min spacing in embedding (drives `a`,`b`); **must be ≤ `spread`** |
| `n_epochs` | None → 500 (n<10k) / 200 (n≥10k) | SGD iterations |
| `init` | `'spectral'` | `'spectral'` or `'random'` |
| `random_state` | None | seed; when set, forces deterministic single-thread path |
| `learning_rate` | 1.0 | SGD initial α |

**Differentiator/advanced hyperparameters (expose, sensible defaults, lower-priority to tune):** `spread=1.0`, `set_op_mix_ratio=1.0`, `local_connectivity=1.0`, `repulsion_strength=1.0`, `negative_sample_rate=5`, `a=None`/`b=None` (override the curve fit), `transform_queue_size=4.0`.

**Outputs:** `fit(X)` → sets `embedding_` `(n, n_components)`; `fit_transform(X)` returns it; `transform(X_new)` embeds new points against the fitted fuzzy graph (umap-learn does a per-new-point kNN against training data + a short SGD). MVP may ship `fit`/`fit_transform` first and add `transform` as a fast-follow (transform reuses the same SGD but is fiddlier to gate).

**Gate (property/structural — the D-12 analog):**
- **kNN preservation / trustworthiness**: fraction of each point's k embedding-neighbors that were k input-neighbors ≥ threshold, compared *against umap-learn's own* trustworthiness (mlrs ≥ umap-learn − margin), not against absolute coordinates.
- **Downstream separability**: on a labeled synthetic blob/cluster dataset, KMeans/ARI on mlrs embedding ≈ ARI on umap-learn embedding (within band).
- **Seed-reproducibility**: same `random_state` → bit-identical mlrs embedding across runs.
- **Determinism of stages 1–4** (graph, fuzzy set, spectral init) — these *are* value-gateable at 1e-5 vs umap-learn's intermediate arrays; gate them directly. Only stage 5 (SGD) is property-only.

### Feature 3 — HDBSCAN

**Oracle:** `hdbscan` library and/or `sklearn.cluster.HDBSCAN` (CPU). **Gate:** exact `labels_` up to permutation + noise (`-1`), `probabilities_` within band. Deterministic given the KNN graph → a *real* exact gate (unlike UMAP).

**Algorithm stages:**
1. **Core distances** — `core_dist_i = ` distance to the `min_samples`-th neighbor (counting convention pinned in Feature 1).
2. **Mutual-reachability distance** — `mreach(a,b) = max(core_a, core_b, alpha·d(a,b))` (`alpha=1.0`).
3. **MST** — minimum spanning tree over MR distance (Prim's/Borůvka). cpu-MLIR-safe: Prim's with a GATHER per-iteration min-scan, no atomics. Tie-breaking must match the oracle for exact labels (document the rule).
4. **Single-linkage hierarchy** — sort MST edges ascending; union-find to build the dendrogram.
5. **Condensed tree** — walk the hierarchy top-down; a split is a "real" split only if **both** sides have ≥ `min_cluster_size` points; otherwise the smaller side "falls out of" the parent (points become noise at that λ). Tracks birth/death λ = 1/distance.
6. **Cluster extraction** — `cluster_selection_method='eom'` (Excess of Mass): select clusters maximizing total stability `Σ (λ_p − λ_birth)` subject to no ancestor/descendant both selected; `'leaf'` selects all leaf clusters. `cluster_selection_epsilon>0` merges clusters below a distance threshold (DBSCAN-like floor).

**Table-stakes hyperparameters (defaults — note sklearn vs hdbscan-library differences):**
| Param | hdbscan lib default | sklearn.cluster.HDBSCAN default | Role |
|-------|--------------------|-------------------------------|------|
| `min_cluster_size` | 5 | 5 | smallest accepted cluster |
| `min_samples` | None→`min_cluster_size` | None→`min_cluster_size` | core-distance k; conservatism/noise |
| `cluster_selection_epsilon` | 0.0 | 0.0 | distance floor to merge micro-clusters |
| `cluster_selection_method` | `'eom'` | `'eom'` | `'eom'` or `'leaf'` |
| `metric` | `'euclidean'` | `'euclidean'` | distance |
| `alpha` | 1.0 | 1.0 | MR-distance scaling (robust-single-linkage) |
| `max_cluster_size` | 0 (off) | 0 (off) | upper cap (eom only) |

**sklearn vs hdbscan-library API differences (pick a target, note both):**
- **Namespace/import:** `sklearn.cluster.HDBSCAN` vs `import hdbscan; hdbscan.HDBSCAN`.
- **`store_centers`:** sklearn-only param (`'centroid'`/`'medoid'`) → `centroids_`/`medoids_` attrs; not in hdbscan lib.
- **`outlier_scores_` (GLOSH):** hdbscan lib **has it**; sklearn **omits it**. (mlrs differentiator if shipped.)
- **Prediction/inspection attrs:** hdbscan lib exposes `condensed_tree_`, `single_linkage_tree_`, `minimum_spanning_tree_`, `approximate_predict`; sklearn exposes none of these.
- **`metric='precomputed'`:** both accept a precomputed distance matrix (useful for a clean oracle harness — feed identical distances to both, isolating the clustering logic from KNN ties).
- **Recommendation:** target **sklearn.cluster.HDBSCAN's surface** as the table-stakes contract (`labels_`, `probabilities_`, the 7 params above), and gate against **both** oracles with `metric='precomputed'` to neutralize KNN tie ambiguity; add `outlier_scores_` as an opt-in differentiator.

**Outputs:** `labels_` `(n,)` int (`-1`=noise), `probabilities_` `(n,)` ∈[0,1] (0 for noise). Optional: `outlier_scores_` `(n,)`. `fit_predict(X)` returns `labels_`.

**Gate:** `labels_` exact up to permutation + noise (v2 label-perm helper handles permutation; add noise-label alignment). `probabilities_` within a documented band (the λ-scaling can differ slightly in f32). With `metric='precomputed'` the gate should be *exact* on f64.

### Feature 4 — Rust-native builder + typestate API

**No new kernels.** A caller-surface convention + a 30-estimator retrofit. Idiomatic target:

```rust
let umap = Umap::builder()           // -> UmapBuilder (all params Optional, defaulted)
    .n_neighbors(15)
    .min_dist(0.1)
    .n_components(2)
    .random_state(42)
    .build()?;                       // -> Result<Umap<Unfit>, BuildError>  (validates params)

let fitted = umap.fit(&x)?;          // -> Result<Umap<Fitted>, FitError>   (validates data)
let emb = fitted.embedding();        // accessor only exists on Fitted
let y    = fitted.transform(&x2)?;   // transform/predict only on Fitted typestate
```

- **Table-stakes ergonomics:**
  - **Owned builder, chained setters** returning `Self` (move-based; the dominant Rust builder style, no lifetime friction).
  - **`build()` returns `Result`** and validates params (e.g. `min_dist ≤ spread`, `n_neighbors ≥ 1`) with `thiserror` variants.
  - **`fit` returns `Result`**; on success yields the fitted estimator (or `&mut self` set-state — pick one and apply uniformly across all 30).
  - **Sensible defaults** = sklearn defaults, so `Umap::builder().build()?` reproduces sklearn-default behavior.
  - **Doc-comments on every setter** stating the sklearn param it mirrors.
- **Nice-to-haves (differentiators):**
  - **Typestate `Unfit`/`Fitted`** so `transform`/`predict`/accessors are compile-time-gated (can't call before fit). The PyO3 layer hides this behind the existing `any_estimator!` `Unfit/F32/F64` enum.
  - **Derive-macro builder** (e.g. `derive_builder` / `bon` / `typed-builder`) vs hand-rolled — evaluate in STACK research; hand-rolled keeps zero new deps and full control over `Result`-returning `build()`.
  - **`From`/`TryFrom` between builder and the existing config struct** so the retrofit is *additive* (builder → existing config → existing fit path) — protects the shipped 1e-5 gates.
- **Anti-feature:** rewriting estimator fit bodies for the retrofit. Keep `fit` logic untouched; the builder is a new front door.

### Feature 5 — Pure-Python sklearn shim

**No new kernels.** Carried-forward v2 debt. What `check_estimator` / `get_params` / `set_params` actually require:

- **`__init__` stores every constructor arg unchanged** as a same-named attribute, with **no validation and no computation in `__init__`** (sklearn rule: `__init__` must not mutate args). Validation happens in `fit`. The existing PyO3 estimators must surface their params this way at the Python layer.
- **`get_params(deep=True)`** returns `{param: value}` for every `__init__` arg (discovered via the signature). For nested estimators, `deep=True` recurses with `param__subparam` keys (not needed for UMAP/HDBSCAN — flat params).
- **`set_params(**params)`** sets the corresponding attrs, supports `param__subparam`, **revalidates nothing** (just stores), and returns `self`.
- **`clone()` compatibility**: `clone(est)` calls `get_params` then `__init__(**params)` and asserts the new estimator's `get_params` equals the original's — so params must round-trip exactly (no coercion in `__init__`).
- **Tags / mixin surface** that `check_estimator` exercises: `fit` returns `self`; fitted attrs end in `_` and only appear after `fit`; `NotFittedError` (or analog) before fit; `n_features_in_` set in `fit`; consistent `n_features_in_` enforced at `predict`/`transform`. For clusterers (HDBSCAN): `fit_predict` present, `labels_` set. For transformers (UMAP): `transform`/`fit_transform`, output shape `(n, n_components)`.
- **Scope note:** the *pure-Python* shim implements get_params/set_params/clone semantics over the existing `MlrsBase`; the **live `estimator_checks` run is explicitly deferred** (needs maturin+pyarrow host). The shim is verified by Rust-side unit tests + a static Python check, not a live sklearn `check_estimator` invocation, in this environment.

---

## Feature Dependencies

```
[KNN-graph prim (Feature 1)]                       <- build & gate FIRST (primitive-first)
    ├──required──> [UMAP (Feature 2)]   (self-excluded k-NN)
    └──required──> [HDBSCAN (Feature 3)] (self-inclusive core distances)

[v2 graph-Laplacian prim] ──required──> [UMAP spectral init]   (REUSE, not new)
[v1 symmetric eig]        ──required──> [UMAP spectral init]   (REUSE — smallest nontrivial)
[v1 distance prim + top-k]──required──> [KNN-graph prim]       (REUSE)
[host SplitMix64 RNG (v2 prims/rng.rs)] ──required──> [UMAP SGD negative sampling]  (REUSE)
[v2 SGD/edge-sampling idiom] ──informs──> [UMAP SGD layout]    (pattern reuse, not the solver)

[union-find + MST] ──new, internal to──> [HDBSCAN]             (host-side or GATHER min-scan)

[Rust builder/typestate convention (Feature 4)]
    ├──wraps──> [UMAP], [HDBSCAN], and all 30 existing estimators
    └──underlies──> [PyO3 surface]  (typestate collapses into any_estimator! enum)

[Pure-Python sklearn shim (Feature 5)]
    └──depends on──> [PyO3 estimators exposing get_params-able args]  (UMAP/HDBSCAN PyO3-wrapped)
```

### Dependency Notes
- **KNN-graph prim before BOTH estimators (mandatory, primitive-first):** it is the feasibility-critical shared substrate; land + gate it standalone (exact vs NearestNeighbors oracle) before UMAP or HDBSCAN consume it — exactly the v1/v2 "land the prim, then assemble" discipline.
- **UMAP reuses the v2 Spectral stack:** spectral init *is* SpectralEmbedding (graph-Laplacian + smallest-eig). This is the single biggest UMAP cost-saver and de-risks the deterministic stages.
- **HDBSCAN's MST/union-find is the one genuinely new piece** with no v1/v2 analog; the MR-distance and condensed-tree are pure host/GATHER logic over the KNN graph + distances.
- **UMAP and HDBSCAN are otherwise independent** — buildable in parallel once the KNN prim lands (file-disjoint, like v2 families).
- **Builder retrofit is independent of the algorithms** but touches every estimator file → schedule as its own phase to contain churn; make it additive (builder→existing config) to protect shipped gates.
- **Python shim depends on the PyO3 wrap of UMAP/HDBSCAN** existing first.

## MVP Definition

### Launch With (v3.0)
- [ ] **KNN-graph prim** — shared, standalone-gated (exact vs NearestNeighbors); self-inclusion param; directed output. *Essential: both algorithms depend on it.*
- [ ] **UMAP** `fit`/`fit_transform` → `embedding_`; table-stakes params; stages 1–4 value-gated, stage-5 property-gated. *Essential: headline feature.*
- [ ] **HDBSCAN** `fit`/`fit_predict` → `labels_`, `probabilities_`; table-stakes params (`'eom'`+`'leaf'`); exact-label gate via `metric='precomputed'`. *Essential: headline feature.*
- [ ] **Rust builder + typestate convention** + retrofit across the 30 existing estimators + the 2 new. *Essential: stated milestone goal.*
- [ ] **Pure-Python sklearn shim** (get_params/set_params/clone) + PyO3-wrap UMAP/HDBSCAN. *Essential: carried-forward debt + makes new estimators drop-in.*

### Add After Validation (v3.x)
- [ ] **UMAP `transform`** (new-data embedding) — fast-follow once `fit` property gate is stable.
- [ ] **HDBSCAN `outlier_scores_` (GLOSH)** — differentiator vs sklearn; opt-in.
- [ ] **HDBSCAN `store_centers` → `centroids_`/`medoids_`** — sklearn parity nicety.

### Future Consideration (v3+ / later milestone)
- [ ] Supervised/semi-supervised UMAP (`target_metric`), UMAP `inverse_transform` — large surface, niche, no clean gate.
- [ ] HDBSCAN `approximate_predict`/`membership_vector` (new-point prediction), condensed-tree plot objects.
- [ ] Approximate/NN-Descent KNN graph build — only if O(n²) ceiling becomes the bottleneck and a cpu-MLIR-safe formulation exists; would force re-gating HDBSCAN to approximate-label agreement.
- [ ] Native sparse KNN-graph path (densified at ingress for v3).

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Correctness gate |
|---------|------------|---------------------|----------|------------------|
| KNN-graph prim | HIGH (substrate) | MEDIUM | P1 | **Value** 1e-5 vs NearestNeighbors (exact, deterministic) |
| HDBSCAN | HIGH | HIGH (MST/condensed tree new) | P1 | **Exact labels** up to perm+noise; probs band; exact on `precomputed`+f64 |
| UMAP | HIGH | HIGH (SGD + 5 stages) | P1 | **Property/structural** (D-12 style) for SGD; 1e-5 for stages 1–4 |
| Rust builder + typestate | MEDIUM (DX) | MEDIUM-HIGH (30-est retrofit churn) | P1 | Compile + behavior-preserving (existing gates unchanged) |
| Pure-Python sklearn shim | MEDIUM | MEDIUM | P1 | get_params/set_params/clone round-trip; static check (live deferred) |
| UMAP `transform` | MEDIUM | MEDIUM | P2 | Property gate on new points |
| HDBSCAN `outlier_scores_` | MEDIUM | MEDIUM-HIGH | P2/P3 | Band vs hdbscan lib |

**Priority key:** P1 = must have for v3.0 launch; P2 = fast-follow; P3 = future.

## Competitor / Reference Feature Analysis

| Feature | umap-learn / hdbscan (CPU) | cuML (GPU) | sklearn.cluster.HDBSCAN | mlrs (our approach) |
|---------|---------------------------|------------|-------------------------|---------------------|
| Precision | f64 | **f32-only** | f64 | **f32 + f64** (differentiator) |
| KNN build | tree / NN-descent / brute | nn_descent | tree/brute | **brute-force exact** (cpu-MLIR-safe, clean gate) |
| UMAP supervised / inverse_transform | yes | yes | n/a | **omit** (anti-feature for v3) |
| HDBSCAN outlier_scores_ (GLOSH) | yes (hdbscan lib) | yes | **no** | **opt-in differentiator** |
| HDBSCAN new-point predict | yes | yes | no | **defer** to v3.x |
| Custom callable metrics | yes | partial | yes | **fixed string metrics** (no numba on CubeCL) |
| Caller surface | Python sklearn API | Python sklearn API | sklearn API | **Rust builder/typestate + PyO3 sklearn shim** (differentiator) |

## Sources

- **cuML v26.08 source (read-only reference, local):** `cuml-main/python/cuml/cuml/manifold/umap/umap.pyx` — confirmed UMAP param surface/defaults (`n_neighbors`, `min_dist`, `spread`, `set_op_mix_ratio`, `local_connectivity`, `repulsion_strength`, `negative_sample_rate`, `target_*`), `find_ab_params(spread, min_dist)`, `min_dist ≤ spread` validation, `init`/`random_state` deterministic-path handling. `cuml-main/python/cuml/cuml/cluster/hdbscan/hdbscan.pyx` — HDBSCAN surface. [HIGH]
- **umap-learn algorithm semantics** (smooth-kNN ρ/σ binary search to `log2(n_neighbors)`, fuzzy set union via `set_op_mix_ratio` t-conorm, spectral init, negative-sampling SGD, `a`/`b` from min_dist/spread): stable documented algorithm, knowledge cutoff Jan 2026. [HIGH]
- **hdbscan library + sklearn.cluster.HDBSCAN** semantics (core distance via min_samples-th NN, mutual-reachability `max(core_a,core_b,d)`, MST→single-linkage→condensed tree with min_cluster_size, EoM stability extraction, `probabilities_`, GLOSH `outlier_scores_`, sklearn `store_centers`/no-outlier-scores difference, `metric='precomputed'`): stable documented APIs. [HIGH]
- **scikit-learn estimator contract** (`__init__` stores args unmodified, get_params/set_params/clone round-trip, fitted-attr `_` convention, `n_features_in_`, `check_estimator` requirements): stable, documented. [HIGH]
- **Rust builder/typestate idioms** (owned chained builder returning Self, `build()->Result`, typestate phantom markers, `derive_builder`/`bon`/`typed-builder`): established community patterns. [HIGH]
- **Project context:** `.planning/PROJECT.md` (v3 scope, gate=cpu f64+rocm f32, property-gate precedent D-12, primitive-first), `.planning/notes/v3-hard-algorithm-backlog.md`, `.planning/milestones/v2.0-research/{FEATURES,SUMMARY}.md` (GATHER idiom, label-perm helper, RandomProjection property gate, graph-Laplacian + eig prims to reuse). [HIGH]
- **Project memory:** cpu-MLIR no-SharedMemory/no-atomics (GATHER idiom); f64-on-rocm skip-with-log; oracle-fixture /tmp-venv regen; Python wheel untestable in env (live estimator_checks deferred). [HIGH]

---
*Feature research for: UMAP + HDBSCAN + KNN-graph prim + Rust builder API + Python sklearn shim (mlrs v3.0)*
*Researched: 2026-06-22*
