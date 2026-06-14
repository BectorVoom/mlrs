---
title: cuML → mlrs algorithm gap inventory
date: 2026-06-14
context: /gsd-explore — what cuML implements that mlrs (v1.0) does not
source_of_truth: cuml-main/python/cuml/cuml/*/ (single-GPU .pyx/.py, _mg variants excluded)
---

# cuML → mlrs Gap Inventory

Reference snapshot taken at v1.0 close. mlrs deliberately targets the **scikit-learn core
subset** (sklearn, not cuML, is the oracle), so the gap is large and expected.

## ✅ Implemented in mlrs v1.0 (12 estimators)

LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, PCA, TruncatedSVD,
KMeans, DBSCAN, NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor.

## ❌ In cuML, not in mlrs

Tiered by feasibility against v1's validated primitives (GEMM, reductions, distance,
covariance/Gram, SVD + symmetric eig, Cholesky/triangular-solve, top-k, L-BFGS,
coordinate descent).

### Tier 1 — cheap wins (assembly on existing prims)
| Estimator | cuML location | Reuses |
|---|---|---|
| KernelRidge | kernel_ridge/kernel_ridge.py | Cholesky + kernel matrix |
| EmpiricalCovariance | covariance/empirical_covariance.py | covariance prim |
| LedoitWolf | covariance/ledoit_wolf.py | covariance prim |
| IncrementalPCA | decomposition/incremental_pca.py | batched SVD |
| GaussianRandomProjection | random_projection/random_projection.py | RNG + GEMM |
| SparseRandomProjection | random_projection/random_projection.py | RNG + GEMM |
| Naive Bayes (Gaussian/Multinomial/Bernoulli/Complement/Categorical) | naive_bayes/naive_bayes.py | reductions |
| SpectralEmbedding | manifold/spectral_embedding.pyx | **eig** |
| SpectralClustering | cluster/spectral_clustering.pyx | **eig** + KMeans |

### Tier 2 — moderate (one new solver/kernel each)
| Estimator | cuML location | New work |
|---|---|---|
| LinearSVC | svm/linear_svc.py, svm/linear.pyx | hinge loss over L-BFGS/CD |
| LinearSVR | svm/linear_svr.py | ε-insensitive loss |
| MBSGDClassifier | linear_model/mbsgd_classifier.py | SGD solver |
| MBSGDRegressor | linear_model/mbsgd_regressor.py | SGD solver |
| KernelDensity | neighbors/kernel_density.pyx | kernel matrix + reductions |
| HoltWinters | tsa/holtwinters.pyx | L-BFGS |
| AgglomerativeClustering | cluster/agglomerative.pyx | distance + union-find (host-heavy) |
| TSNE (exact) | manifold/t_sne.pyx | gradient descent |

### Tier 3 — hard (major new infrastructure; fights CubeCL/cpu-MLIR constraints)
| Estimator | cuML location | Why hard |
|---|---|---|
| RandomForestClassifier/Regressor | ensemble/randomforest*.py | GPU tree construction; cpu-MLIR has no SharedMemory/atomics |
| FIL (Forest Inference) | fil/ | depends on RandomForest |
| UMAP | manifold/ (umap) | fuzzy simplicial set + SGD layout on KNN graph |
| HDBSCAN | cluster/ (hdbscan) | MST + condensed tree |
| ARIMA / AutoARIMA | tsa/arima.pyx, auto_arima.pyx | Kalman filter + batched L-BFGS |
| SVC / SVR (kernel) | svm/svc.py, svr.py, svm_base.pyx | SMO solver |

### Whole subsystems not addressed
explainer (SHAP: Kernel/Permutation/Tree), genetic (symbolic regression), metrics,
preprocessing, feature_extraction (TfidfVectorizer etc.), multiclass, model_selection,
`cuml.accel` drop-in layer, Dask multi-GPU (`*_mg` variants), standalone solvers
(SGD/QN/CD as public API).

## Routing

Tier 1 + Tier 2 → [[../seeds/v2-breadth-roadmap]] (v2 milestone).
Tier 3 + subsystems → [[v3-hard-algorithm-backlog]].
