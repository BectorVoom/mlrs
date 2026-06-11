# Feature Research

**Domain:** sklearn-compatible GPU ML library (Rust/CubeCL rewrite of RAPIDS cuML), v1 algorithm surface
**Researched:** 2026-06-11
**Confidence:** HIGH (grounded in cuML v26.08 source signatures + scikit-learn public API conventions)

---

## How to read this document

The "product" here is a set of estimators. "Table stakes" means *the estimator is wrong or
unusable without it* — a missing `coef_` or an init that silently differs from sklearn breaks
the 1e-5 oracle contract. "Differentiators" are fidelity-to-cuML or quality-of-life features
that are not required for correctness. "Anti-features" are things we deliberately do NOT build
in v1 (either out of PROJECT scope, or sklearn knobs that add surface area without algorithmic
value at the 1e-5 tolerance).

Every algorithm section enumerates: **methods**, **hyperparameters**, **fitted attributes**,
**solver/variant choices**, then the three feature categories with complexity.

**The single hardest cross-cutting constraint:** the oracle compares against *scikit-learn on
CPU*, not cuML. So for each estimator the table-stakes solver is "whatever variant makes the
result match sklearn's default within 1e-5", which is sometimes NOT cuML's default solver.
This is called out per algorithm below (it is the biggest correctness risk in the whole project).

---

## Cross-Cutting Feature Surface (applies to all estimators)

### sklearn API conventions — TABLE STAKES

| Convention | Requirement | Complexity | Notes |
|------------|-------------|------------|-------|
| `fit(X, y=None)` returns `self` | Mandatory sklearn contract; enables chaining and pipelines | LOW | Trivial in Rust→PyO3 (`return slf`) but must be wired through every estimator |
| `get_params()` / `set_params()` | Required for clone, grid search, pipelines | MEDIUM | Must expose every constructor hyperparameter by name; cuML uses `_get_param_names()` returning the exact list per estimator (see per-algo sections) |
| Constructor stores params unchanged | sklearn rule: `__init__` does no validation/transformation, only assignment | LOW | Validation happens in `fit`, not `__init__` — important for `clone()` round-trip |
| Trailing-underscore fitted attributes | `coef_`, `labels_`, etc. only exist after `fit`; accessing before raises `NotFittedError` | MEDIUM | cuML uses `CumlArrayDescriptor`; mlrs must lazily materialize device→host on attribute access |
| `n_features_in_` set at fit | sklearn ≥1.0 convention; checked by `check_is_fitted` patterns | LOW | Simple integer attribute |
| Mixin tags (`RegressorMixin`/`ClassifierMixin`/`ClusterMixin`) | Determine default `score()` semantics and estimator type checks | LOW | cuML inherits these; mlrs replicates the `score()` defaults (R² for regressors, accuracy for classifiers) |

### dtype handling (f32/f64) — TABLE STAKES

| Feature | Requirement | Complexity | Notes |
|---------|-------------|------------|-------|
| Accept f32 and f64 input, dispatch to matching kernel | PROJECT mandates both validated; kernels generic over float | MEDIUM | CubeCL kernels are generic-over-float; the binding picks `f32`/`f64` codepath by input dtype |
| `convert_dtype=True` behavior | cuML auto-casts mismatched dtypes (e.g. predict input cast to fit dtype) | MEDIUM | Present on nearly every cuML `fit`/`predict`/`transform`; v1 should honor it for sklearn-like leniency |
| Output dtype matches input dtype | f32 in → f32 out; preserves the 1e-5 budget (f64 makes tolerance comfortable, f32 is tighter) | LOW | f32 paths are the *risky* ones for 1e-5 — accumulate reductions in higher precision where needed |
| Reject/handle non-finite (NaN/Inf) | sklearn raises on NaN by default | LOW–MEDIUM | Input validation step; cuML's `check_inputs()` analog |

### Input validation & interchange — TABLE STAKES

| Feature | Requirement | Complexity | Notes |
|---------|-------------|------------|-------|
| Shape/contiguity validation | Reject ragged, wrong-ndim, mismatched `X`/`y` lengths | LOW | At the PyO3 boundary before device upload |
| Arrow zero-copy ingest → device buffer | PROJECT foundation requirement | HIGH | Memory-efficiency priority; the buffer abstraction feeds every algorithm |
| C/F-contiguity awareness | cuML stores several attrs in F-order (`components_`, ElasticNet `coef_`); GEMM/SVD primitives are layout-sensitive | MEDIUM | Layout mistakes are a top correctness/perf pitfall |

### Oracle-test contract — TABLE STAKES

| Feature | Requirement | Complexity | Notes |
|---------|-------------|------------|-------|
| Random-data generation matching sklearn fixtures | `make_blobs`/`make_regression`/random uniform, fixed seed | LOW | Mirror cuML's `testing/datasets.py` approach |
| abs/rel error ≤ 1e-5 vs sklearn | The core correctness gate | — | Per-estimator tolerance realities documented below |
| Sign/permutation invariance helpers | PCA/SVD components have sign ambiguity; KMeans/DBSCAN labels are permutation-invariant | MEDIUM | cuML ships `assert_dbscan_equal` (label-order-independent) and array fuzzy-compare; mlrs needs the same harness or 1e-5 fails spuriously |
| Both f32 and f64 tested | PROJECT requirement | LOW | Parameterize oracle over dtype |

### Shared compute primitives (dependency backbone)

These are not user-facing features but every algorithm depends on them; ordering the roadmap
around them is essential.

| Primitive | Used by | Complexity | Notes |
|-----------|---------|------------|-------|
| GEMM / matmul | LinearRegression, Ridge, PCA, TSVD, all KNN brute distance | HIGH | CubeCL matmul manual; foundational |
| Reductions (sum/mean/L2/argmin) | KMeans (assign), variance/mean centering (PCA), norms | MEDIUM | CubeCL reduce manual |
| Pairwise distance (Euclidean/L2, cosine) | KMeans, DBSCAN, all KNN | HIGH | Shared by 3 of 4 families; build once, well |
| SVD (full + Jacobi) and/or symmetric eigendecomposition | PCA, TSVD, OLS (`svd` algo), Ridge (`svd`/`eig`) | HIGH | Hardest primitive; gates two whole families |
| Covariance / Gram matrix (XᵀX) | PCA (`full` via eig of covariance), Ridge `eig`, OLS `eig` | MEDIUM | Built from GEMM + reductions |
| Top-k selection / sort | KNN neighbor selection, DBSCAN core detection | MEDIUM | CubeCL selection pattern |
| Coordinate descent solver | Lasso, ElasticNet | MEDIUM | Iterative; soft-thresholding |
| Quasi-Newton (L-BFGS / OWL-QN) | LogisticRegression (and cuML's Lasso/ENet `qn`) | HIGH | Required for LogisticRegression; convergence to match sklearn `lbfgs`/`liblinear` is delicate |

---

## Linear Models

### LinearRegression (OLS) — `RegressorMixin`

**Methods:** `fit(X, y)`, `predict(X)`, `score(X, y)` (R²), `get_params`/`set_params`.
**Hyperparameters (sklearn):** `fit_intercept=True`, `copy_X=True`, `positive=False`, `n_jobs`.
cuML adds `algorithm={'auto','eig','svd','qr','svd-qr','svd-jacobi','lsmr'}`.
**Fitted attributes:** `coef_` (shape `(n_features,)` or `(n_targets, n_features)`), `intercept_` (float or array), `n_features_in_`, `rank_`/`singular_` (sklearn has them; low value).
**Solvers:** `eig` (normal equations via XᵀX eigendecomp — cuML default, fast) vs `svd` (numerically robust, matches sklearn's LAPACK `gelsd` more closely).

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | `coef_`, `intercept_`; `fit_intercept`; `predict`; `score` (R²); a solver producing ≤1e-5 vs sklearn (sklearn uses SVD-based lstsq → **`svd` path is the safest match**); multi-target `y` (2D) since sklearn supports it | MEDIUM (GEMM + one of eig/SVD) |
| **Differentiators** | Multiple solver choices (`eig`/`svd`/`qr`) à la cuML; `lsmr` for tall-skinny; sample-weight passthrough | MEDIUM |
| **Anti-features** | `positive=True` (NNLS — out of scope, separate solver); `n_jobs` (meaningless on GPU); sparse-X OLS in v1 | — |

**Dependency:** GEMM + (eig or SVD primitive). The cheapest correct v1 path is `eig` on XᵀX, but
SVD is what makes f32 tolerance comfortable on ill-conditioned data.

### Ridge — `RegressorMixin`

**Methods:** `fit`, `predict`, `score`, params.
**Hyperparameters:** `alpha=1.0` (scalar or per-target array), `fit_intercept=True`, `copy_X`,
`solver={'auto','eig','svd','lsmr'}` (cuML), sklearn adds `'cholesky','lsqr','sag','saga'` + `max_iter`, `tol`.
**Fitted attributes:** `coef_`, `intercept_`, `solver_` (which solver ran), `n_iter_` (for iterative solvers, else `None`).
**Solvers:** `eig`/`svd` closed-form (default) match sklearn `cholesky`/`svd` within 1e-5.

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | `alpha` (scalar minimum), `fit_intercept`, `coef_`/`intercept_`, closed-form solver matching sklearn default; multi-target | MEDIUM |
| **Differentiators** | Per-target `alpha` array; `lsmr` iterative solver + `n_iter_`/`solver_` reporting (cuML fidelity) | MEDIUM |
| **Anti-features** | `sag`/`saga` stochastic solvers; sparse paths; `positive=True` | — |

**Dependency:** XᵀX + regularized eig/SVD (same primitive as OLS).

### Lasso — `RegressorMixin`

**Methods:** `fit`, `predict`, `score`, params.
**Hyperparameters:** `alpha=1.0`, `fit_intercept=True`, `max_iter=1000`, `tol=1e-3`,
`selection={'cyclic','random'}`. cuML exposes `solver={'cd','qn'}` (default `cd`).
**Fitted attributes:** `coef_`, `intercept_`, `n_iter_`, `dual_gap_` (sklearn), `sparse_coef_` (sklearn convenience).
**Solver:** Coordinate descent (`cd`) is the table-stakes solver — it is what sklearn uses, so
it is the one that hits 1e-5. cuML's `qn` is an alternative but converges to a *different*
point at loose tolerance.

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | Coordinate-descent solver (matches sklearn), `alpha`, `fit_intercept`, `max_iter`, `tol`, `coef_`/`intercept_`, `n_iter_` | MEDIUM–HIGH (CD on GPU, soft-thresholding, convergence parity is delicate) |
| **Differentiators** | `selection='random'`; `qn` solver option; `dual_gap_` reporting | MEDIUM |
| **Anti-features** | `sparse_coef_` materialization; multi-target Lasso (`MultiTaskLasso`); `precompute`/`warm_start`; positive constraint | — |

**Dependency:** Coordinate-descent solver primitive (shared with ElasticNet). CD convergence to
sklearn within 1e-5 is a known correctness risk — flag for deeper research.

### ElasticNet — `RegressorMixin`

**Methods/attributes:** same as Lasso plus `l1_ratio`.
**Hyperparameters:** `alpha=1.0`, `l1_ratio=0.5`, `fit_intercept=True`, `max_iter=1000`,
`tol=1e-4`, `selection={'cyclic','random'}`. cuML stores `coef_` in F-order.
**Solver:** Coordinate descent (Lasso is the `l1_ratio=1` special case; both share the CD kernel).

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | `alpha`, `l1_ratio`, CD solver, `coef_`/`intercept_`, `n_iter_`, `fit_intercept` | MEDIUM–HIGH (shares Lasso CD) |
| **Differentiators** | `selection='random'`, `qn` solver, `dual_gap_` | MEDIUM |
| **Anti-features** | Multi-task variant; sparse coef; `precompute`/`warm_start`; positive constraint | — |

**Dependency:** Same CD solver as Lasso — build Lasso and ElasticNet together.

### LogisticRegression — `ClassifierMixin`

**Methods:** `fit`, `predict`, `predict_proba`, `predict_log_proba`, `decision_function`, `score` (accuracy), params.
**Hyperparameters (sklearn):** `penalty={'l1','l2','elasticnet',None}` (default `'l2'`), `C=1.0`,
`fit_intercept=True`, `class_weight`, `max_iter=1000` (cuML), `tol=1e-4`, `l1_ratio`,
`solver={'lbfgs','liblinear','newton-cg','sag','saga'}` (sklearn) — **cuML supports only `solver='qn'`**.
cuML extra: `linesearch_max_iter`, `penalty_normalized`.
**Fitted attributes:** `coef_` (`(n_classes, n_features)`), `intercept_`, `classes_`, `n_iter_`.
**Solver:** Quasi-Newton (L-BFGS for L2/none, OWL-QN for L1/elasticnet). Multiclass via softmax
(cuML) — note sklearn's default historically was OvR/multinomial depending on version; matching
1e-5 requires care over which multinomial formulation is used.

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | QN solver (L-BFGS), `C`/`penalty` (`l2` minimum), `fit_intercept`, `predict`, `predict_proba`, `coef_`/`intercept_`/`classes_`, binary classification; multinomial/softmax for multiclass | HIGH (QN optimizer + numerically stable logistic loss/grad; convergence parity with sklearn `lbfgs` is the trickiest 1e-5 target in the project) |
| **Differentiators** | `l1`/`elasticnet` penalties (OWL-QN); `class_weight='balanced'`; `decision_function`; `predict_log_proba`; `n_iter_` reporting | HIGH |
| **Anti-features** | `liblinear`/`sag`/`saga`/`newton-cg` solvers (cuML itself drops these — only `qn`); `multi_class` legacy param; `warm_start`; `intercept_scaling`; sample-weight in v1 (optional) | — |

**Dependency:** Quasi-Newton solver primitive + stable softmax/sigmoid + cross-entropy gradient
(GEMM for the linear term). Flag for deeper research — this is the highest correctness risk.

---

## Clustering

### KMeans — `ClusterMixin`

**Methods:** `fit`, `predict`, `transform` (distance to centers), `fit_predict`, `fit_transform`, `score`, params.
**Hyperparameters (sklearn):** `n_clusters=8`, `init={'k-means++','random',array,callable}`,
`n_init='auto'`, `max_iter=300`, `tol=1e-4`, `random_state`, `algorithm={'lloyd','elkan'}`.
cuML defaults `init='scalable-k-means++'` (k-means||), `oversampling_factor=2.0`.
**Fitted attributes:** `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, `n_features_in_`.
**Variants:** Lloyd's iteration (assignment + update) is table-stakes. Init: **k-means++ is the
sklearn default and what the oracle expects** — cuML's k-means|| (scalable) gives statistically
similar but not identical centers, so to hit 1e-5 against sklearn the v1 default init should be
deterministic k-means++ (or seeded `init='random'`/explicit array for the strictest tests).

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | Lloyd iteration; `n_clusters`; `cluster_centers_`/`labels_`/`inertia_`/`n_iter_`; `predict`/`transform`; `random_state` for reproducibility; **k-means++ init** (to match sklearn) OR explicit-array init for oracle determinism; `tol`/`max_iter` convergence | MEDIUM–HIGH (assignment = pairwise distance + argmin; update = scatter-mean) |
| **Differentiators** | k-means\|\| / scalable init (`oversampling_factor`) for cuML fidelity + large-data quality; `n_init` multi-restart; `sample_weight`; Elkan-style triangle-inequality pruning | MEDIUM–HIGH |
| **Anti-features** | `MiniBatchKMeans`; callable init; `copy_x`; bisecting k-means | — |

**Dependency:** Pairwise-distance primitive + argmin reduction + scatter-mean update.
**Oracle note:** labels are permutation-invariant — need label-matching comparison, and centers
compared up to permutation. Init choice is the dominant determinant of 1e-5 agreement.

### DBSCAN — `ClusterMixin`

**Methods:** `fit`, `fit_predict`, params. (No `predict` — DBSCAN cannot label new points; sklearn also omits it.)
**Hyperparameters (sklearn):** `eps=0.5`, `min_samples=5`, `metric={'euclidean','cosine','precomputed',...}`,
`metric_params`, `algorithm`, `leaf_size`, `p`. cuML: `metric={'euclidean','cosine','precomputed'}`,
`algorithm={'brute','rbc'}`, `max_mbytes_per_batch`, `calc_core_sample_indices`.
**Fitted attributes:** `labels_` (−1 = noise), `core_sample_indices_`, `components_` (the core samples themselves).
**Variants:** Brute-force neighborhood (range query within `eps`) is table-stakes.

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | `eps`, `min_samples`, Euclidean metric, brute neighborhood + connected-components labeling, `labels_` with `-1` noise convention, `fit_predict` | HIGH (range query at scale + union-find/BFS expansion on GPU) |
| **Differentiators** | `core_sample_indices_` + `components_` (cuML `calc_core_sample_indices`); cosine metric; `precomputed` distance matrix; batched distance (`max_mbytes_per_batch`) for memory limits; `rbc` (random ball cover) acceleration | MEDIUM–HIGH |
| **Anti-features** | KD-tree/ball-tree `algorithm` variants (CPU-tree concepts); `leaf_size`; arbitrary Minkowski `p`; OPTICS/HDBSCAN | — |

**Dependency:** Pairwise/range-distance primitive (shared with KMeans/KNN) + connected-components.
**Oracle note:** cuML ships `assert_dbscan_equal` because cluster IDs and noise handling differ in
ordering — mlrs must replicate this label-agnostic comparison or oracle tests fail spuriously.

---

## Decomposition

### PCA — transformer

**Methods:** `fit`, `transform`, `inverse_transform`, `fit_transform`, `score`/`score_samples` (sklearn), params.
**Hyperparameters (sklearn):** `n_components` (int/float/'mle'/None), `whiten=False`,
`svd_solver={'auto','full','covariance_eigh','randomized','arpack'}`, `tol`, `iterated_power`, `random_state`, `copy`.
cuML: `n_components`, `whiten`, `svd_solver={'full','jacobi','auto'}`, `tol=1e-7`, `iterated_power=15`, `copy`.
**Fitted attributes:** `components_`, `explained_variance_`, `explained_variance_ratio_`,
`singular_values_`, `mean_`, `noise_variance_`, `n_components_`, `n_features_in_`.
**Variants:** `full` (mean-center → covariance eigendecomp or full SVD; **this is the sklearn-match
path**) vs `jacobi` (iterative, faster, cuML's accelerated option). sklearn's `randomized` is its
own thing — match `full` for the oracle.

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | Mean-centering + `full` SVD/eig; `n_components` (int); `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`; `transform`/`inverse_transform`; `whiten` | HIGH (SVD/eig primitive is the gate) |
| **Differentiators** | `jacobi` solver (cuML); `noise_variance_`; `n_components` as float (variance ratio) or None; `randomized` SVD for tall data | MEDIUM–HIGH |
| **Anti-features** | `n_components='mle'` (cuML explicitly rejects it); `arpack` solver; `IncrementalPCA`; sparse PCA; `score`/`score_samples` log-likelihood (low value for v1) | — |

**Dependency:** SVD (full) and/or symmetric eigendecomposition of covariance + mean reduction + GEMM.
**Oracle note:** component sign ambiguity — sklearn applies `svd_flip` sign convention; mlrs must
match it (deterministic sign) or compare up to sign per component.

### TruncatedSVD — transformer

**Methods:** `fit`, `transform`, `inverse_transform`, `fit_transform`, params.
**Hyperparameters (sklearn):** `n_components=2`, `algorithm={'randomized','arpack'}`, `n_iter=5`,
`n_oversamples`, `power_iteration_normalizer`, `random_state`, `tol`.
cuML: `n_components=1`, `algorithm={'full','jacobi','auto'}`, `n_iter=15`, `tol=1e-7`, `random_state`.
**Fitted attributes:** `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`.
**Difference from PCA:** NO mean-centering (works on raw/sparse matrices, e.g. TF-IDF). Otherwise
shares the SVD primitive.

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | SVD without centering; `n_components`; `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`; `transform`/`inverse_transform` | HIGH (shares PCA's SVD primitive) |
| **Differentiators** | `jacobi` solver; `n_iter` control for iterative path | MEDIUM |
| **Anti-features** | Sparse-matrix input in v1 (TruncatedSVD's main sklearn use case is sparse TF-IDF — defer); `arpack`; `randomized` exact-match to sklearn default | — |

**Dependency:** Same SVD/eig primitive as PCA (build PCA + TSVD together; TSVD ≈ PCA minus centering).
**Oracle note:** sklearn TruncatedSVD's default `algorithm='randomized'` is stochastic — for the
1e-5 oracle, compare against sklearn with `algorithm='arpack'` (deterministic) or use a tolerance/
sign-aware comparison. This solver-default mismatch is a documented gotcha.

---

## Neighbors

### NearestNeighbors — unsupervised base

**Methods:** `fit`, `kneighbors(X, n_neighbors, return_distance)`, `kneighbors_graph`,
`radius_neighbors` (sklearn), params.
**Hyperparameters (sklearn):** `n_neighbors=5`, `radius`, `algorithm={'auto','ball_tree','kd_tree','brute'}`,
`leaf_size`, `metric`, `p`, `metric_params`. cuML: `n_neighbors`, `algorithm={'brute','rbc','ivfflat','ivfpq'}`,
`metric`, `p`, `metric_params`.
**Fitted attributes:** the stored training set (`_fit_X`), `n_features_in_`, `effective_metric_`.
**Variants:** **Brute-force exact is table-stakes and the oracle match** (sklearn returns exact
neighbors regardless of tree algorithm). Approximate (IVF/RBC) changes results → cannot match 1e-5.

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | Brute-force exact kNN; Euclidean metric; `n_neighbors`; `kneighbors` returning sorted (distance, index); `kneighbors_graph` | MEDIUM (pairwise distance + top-k selection) |
| **Differentiators** | Additional metrics (cosine, Minkowski-`p`, Manhattan); `radius_neighbors`; `rbc` random ball cover acceleration | MEDIUM |
| **Anti-features** | `kd_tree`/`ball_tree` (CPU tree structures — cuML doesn't implement them either); approximate `ivfflat`/`ivfpq` (break exact 1e-5 oracle — defer to a later "approximate" milestone); `leaf_size` | — |

**Dependency:** Pairwise-distance primitive (shared with KMeans/DBSCAN) + top-k selection.

### KNeighborsClassifier — `ClassifierMixin`

**Methods:** `fit(X, y)`, `predict`, `predict_proba`, `score` (accuracy), params.
**Hyperparameters:** inherits NearestNeighbors params + `weights={'uniform','distance',callable}`.
**Fitted attributes:** `classes_`, plus the stored training set + labels.
**Logic:** kNN search → weighted majority vote (uniform or inverse-distance weights).

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | Brute kNN + uniform-weight majority vote; `predict`; `predict_proba`; `classes_`; `n_neighbors`; `score` | MEDIUM (vote/tally over neighbor labels) |
| **Differentiators** | `weights='distance'` (inverse-distance vote); multi-output `y`; additional metrics | MEDIUM |
| **Anti-features** | Callable `weights` (cuML rejects for CPU conversion); tree algorithms; approximate search; `RadiusNeighborsClassifier` | — |

**Dependency:** NearestNeighbors brute search + label-gather + weighted mode reduction.

### KNeighborsRegressor — `RegressorMixin`

**Methods:** `fit(X, y)`, `predict`, `score` (R²), params.
**Hyperparameters:** same as classifier (`weights` etc.).
**Fitted attributes:** stored training set + targets.
**Logic:** kNN search → (weighted) mean of neighbor targets.

| Category | Items | Complexity |
|----------|-------|------------|
| **Table stakes** | Brute kNN + uniform-weight mean; `predict`; `score` (R²); `n_neighbors`; multi-target `y` | MEDIUM |
| **Differentiators** | `weights='distance'` (inverse-distance weighted mean); additional metrics | MEDIUM |
| **Anti-features** | Callable weights; tree/approximate algorithms; `RadiusNeighborsRegressor` | — |

**Dependency:** NearestNeighbors brute search + target-gather + weighted-mean reduction.

---

## Feature Dependencies

```
GEMM / matmul (CubeCL)
   ├──> Pairwise distance ──┬──> KMeans (assign+argmin, scatter-mean update)
   │                        ├──> DBSCAN (range query) ──> connected-components
   │                        └──> NearestNeighbors (+ top-k select)
   │                                  ├──> KNeighborsClassifier (weighted vote)
   │                                  └──> KNeighborsRegressor (weighted mean)
   ├──> Covariance / XtX ──┬──> OLS (eig path)
   │                       ├──> Ridge (regularized eig/svd)
   │                       └──> PCA (full = center + eig of covariance)
   └──> SVD (full + jacobi) ──┬──> OLS (svd path, best 1e-5 match)
                              ├──> PCA (full SVD path)
                              └──> TruncatedSVD (SVD, no centering)

Coordinate-descent solver ──┬──> Lasso
                            └──> ElasticNet   (Lasso = l1_ratio==1 special case)

Quasi-Newton (L-BFGS / OWL-QN) ──> LogisticRegression

Reductions (sum/mean/norm/argmin) ──> underpins KMeans, PCA centering, KNN, all norms

Sign-flip / label-permutation oracle helpers ──> PCA, TSVD, KMeans, DBSCAN tests
```

### Dependency Notes

- **SVD primitive gates two families:** PCA and TruncatedSVD are both unbuildable until a working
  SVD (or covariance-eigendecomposition) exists; OLS/Ridge `svd`/`eig` paths reuse it. Build the
  decomposition primitive once, validate it standalone, then both estimators are mostly assembly.
- **Pairwise distance gates three families:** KMeans, DBSCAN, and all KNN share it. It is the
  highest-leverage primitive after GEMM — prioritize it.
- **Lasso and ElasticNet are one feature:** they share the coordinate-descent kernel; Lasso is the
  `l1_ratio=1` case. Sequence them together.
- **PCA depends on Ridge/OLS's eig path conceptually** (covariance eigendecomposition) — if you
  build the eig primitive for linear models first, PCA's `full` solver follows cheaply.
- **Oracle helpers are a prerequisite, not an afterthought:** without sign-flip and
  label-permutation comparison, PCA/SVD/KMeans/DBSCAN tests will fail at 1e-5 for non-bugs.

---

## MVP Definition

### Launch With (v1) — the fixed PROJECT scope

- [ ] **Foundation primitives first:** GEMM, reductions, pairwise distance, SVD/eig, CD solver, QN solver — each validated standalone before the estimator that needs it.
- [ ] **LinearRegression (OLS)** — `svd` path for sklearn match; `coef_`/`intercept_`; multi-target.
- [ ] **Ridge** — closed-form eig/svd; `alpha`, `coef_`/`intercept_`.
- [ ] **Lasso + ElasticNet** — shared CD solver; `alpha`(+`l1_ratio`), `coef_`/`intercept_`, `n_iter_`.
- [ ] **LogisticRegression** — QN solver, `predict`/`predict_proba`, `coef_`/`intercept_`/`classes_` (highest risk; research-flagged).
- [ ] **KMeans** — Lloyd + k-means++ init; `cluster_centers_`/`labels_`/`inertia_`/`n_iter_`; predict/transform.
- [ ] **DBSCAN** — brute range query + components labeling; `labels_` with noise; `core_sample_indices_`/`components_`.
- [ ] **PCA** — full SVD/eig + centering; all five fitted attributes + `mean_`; transform/inverse_transform.
- [ ] **TruncatedSVD** — SVD without centering; four fitted attributes.
- [ ] **NearestNeighbors / KNeighborsClassifier / KNeighborsRegressor** — brute exact + uniform & distance weights.
- [ ] **Cross-cutting:** f32/f64 dispatch, `get/set_params`, `fit`-returns-`self`, Arrow zero-copy ingest, oracle harness with sign/permutation helpers.

### Add After Validation (v1.x)

- [ ] Extra solver variants per estimator (`jacobi` SVD, `lsmr`, `qr`) — once `full`/closed-form paths pass oracle.
- [ ] Additional distance metrics (cosine, Minkowski-p, Manhattan) across KMeans/DBSCAN/KNN.
- [ ] `class_weight='balanced'`, L1/elasticnet penalties for LogisticRegression.
- [ ] Sample-weight support across estimators.
- [ ] `rbc` (random ball cover) acceleration for DBSCAN/KNN (still exact).

### Future Consideration (v2+ / explicitly later milestones)

- [ ] Approximate KNN (`ivfflat`/`ivfpq`) — breaks exact 1e-5 oracle; needs its own approximate-tolerance test design.
- [ ] Sparse-input paths (TruncatedSVD on TF-IDF, sparse Lasso).
- [ ] IncrementalPCA, MiniBatchKMeans, multi-task linear models.
- [ ] f16/bf16 validated precision paths (PROJECT marks infra-allowed, not v1).

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Pairwise-distance primitive | HIGH (3 families) | HIGH | P1 |
| SVD/eig primitive | HIGH (2 families + linear) | HIGH | P1 |
| GEMM + reductions | HIGH (everything) | MEDIUM | P1 |
| Oracle harness w/ sign+permutation helpers | HIGH (gates all tests) | MEDIUM | P1 |
| OLS / Ridge closed-form | HIGH | MEDIUM | P1 |
| Lasso/ElasticNet CD solver | HIGH | MEDIUM-HIGH | P1 |
| LogisticRegression QN | HIGH | HIGH | P1 (research-flag) |
| KMeans (Lloyd + ++) | HIGH | MEDIUM-HIGH | P1 |
| DBSCAN brute + components | HIGH | HIGH | P1 |
| PCA / TruncatedSVD | HIGH | HIGH | P1 |
| KNN brute (NN/clf/reg) | HIGH | MEDIUM | P1 |
| Distance-weighted KNN | MEDIUM | LOW-MEDIUM | P2 |
| Extra solver variants (jacobi/lsmr/qr) | MEDIUM | MEDIUM | P2 |
| Extra metrics (cosine/minkowski) | MEDIUM | MEDIUM | P2 |
| L1/elasticnet LogisticRegression | MEDIUM | HIGH | P2 |
| Approximate KNN (IVF) | LOW (v1) | HIGH | P3 |
| Sparse-input paths | LOW (v1) | HIGH | P3 |

**Priority key:** P1 = must have for v1; P2 = add after core passes oracle; P3 = future/out-of-scope-v1.

---

## Competitor Feature Analysis

| Feature | scikit-learn (oracle) | RAPIDS cuML (reference) | mlrs v1 approach |
|---------|----------------------|--------------------------|------------------|
| OLS solver | SVD-based lstsq | `eig` default (+ svd/qr/lsmr) | `svd` path for 1e-5 match; `eig` as fast option |
| LogisticRegression solver | `lbfgs`/`liblinear`/`saga` | `qn` only | `qn` (L-BFGS) — match sklearn `lbfgs` no-penalty case |
| KMeans init | `k-means++` default | `scalable-k-means++` (k-means\|\|) default | **k-means++ default** to match oracle; k-means\|\| as differentiator |
| PCA solver | `auto`→full/covariance_eigh/randomized | `full`/`jacobi` | `full` for oracle; `jacobi` differentiator; reject `mle` |
| TruncatedSVD default | `randomized` (stochastic) | `full`/`jacobi` | `full`; oracle compares vs sklearn `arpack` (deterministic) |
| KNN algorithm | tree + brute | brute/rbc/ivf | brute exact only in v1 |
| DBSCAN labels comparison | deterministic-ish ordering | label-agnostic helper | replicate `assert_dbscan_equal` |
| Multi-GPU / accel | n/a | Dask + cuml.accel | **out of scope** (PROJECT) |

---

## Sources

- cuML v26.08 estimator sources (signatures, hyperparameters, fitted attributes, `_get_param_names`):
  `cuml-main/python/cuml/cuml/{linear_model,cluster,decomposition,neighbors}/*.pyx,*.py` (HIGH confidence — read directly)
- `.planning/PROJECT.md` (v1 scope, out-of-scope, 1e-5 oracle constraint)
- `.planning/codebase/{ARCHITECTURE,TESTING,STRUCTURE}.md` (cuML Base/CumlArray patterns, oracle/test harness)
- scikit-learn public estimator API conventions (`fit` returns self, `get/set_params`, mixin `score` defaults,
  fitted-attribute naming, solver defaults) — established API, MEDIUM-HIGH confidence from training; the
  *solver-default mismatches* (sklearn OLS=SVD, KMeans=k-means++, TruncatedSVD=randomized) are the
  load-bearing claims and are the documented oracle risks.

---
*Feature research for: sklearn-compatible GPU ML estimators (mlrs v1)*
*Researched: 2026-06-11*
