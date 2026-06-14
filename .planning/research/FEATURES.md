# Feature Research — mlrs v2.0 Breadth Sweep

**Domain:** scikit-learn-compatible ML estimators (Rust/CubeCL rewrite of cuML)
**Researched:** 2026-06-14
**Confidence:** HIGH (sklearn algorithm semantics are stable, documented, and confirmed against cuML source + scikit-learn docs; the parity-sensitive defaults — SGD `optimal` schedule, LinearSVC `dual='auto'`/`squared_hinge`, NB smoothing — were verified against the live scikit-learn docs)

> **Framing.** The oracle is **scikit-learn** (abs/rel ≤ 1e-5 on f64), *not* cuML. Where cuML and
> sklearn diverge in algorithm/objective (notably LinearSVC/SVR and the SGD learning-rate schedule),
> **match sklearn**, not cuML. cuML is reference for API shape and kernel structure only. Each
> estimator below pins (a) table-stakes behavior, (b) exact objective/formula/defaults, (c) fitted
> attributes, (d) complexity + primitive reuse vs new kernel, and flags subtle parity risks.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Every v2 estimator must expose the sklearn-named constructor params (defaults below), the
`fit`/`predict`/`transform`/`score` surface appropriate to its mixin, the sklearn fitted attributes
(trailing-underscore), `n_features_in_`, and pass the v1 `estimator_checks` harness. The 1e-5 oracle
gate (f64 strict; f32 documented epsilon band) is the non-negotiable.

| Family | Estimators | Surface | Complexity |
|--------|-----------|---------|------------|
| Covariance | EmpiricalCovariance, LedoitWolf | `fit`, `score`, `get_precision`, `mahalanobis`, `error_norm` | LOW |
| Projection | IncrementalPCA, Gaussian/SparseRandomProjection | `fit`, `partial_fit` (IPCA), `transform`, `inverse_transform` (IPCA) | LOW–MEDIUM |
| Kernel | KernelRidge, KernelDensity | KRR: `fit`/`predict`; KDE: `fit`/`score_samples`/`score`/`sample` | LOW–MEDIUM |
| Spectral | SpectralEmbedding, SpectralClustering | SE: `fit`/`fit_transform`; SC: `fit`/`fit_predict` | MEDIUM |
| SGD / linear-SVM | MBSGD{Classifier,Regressor}, LinearSVC, LinearSVR | `fit`/`predict` (+`decision_function`, `partial_fit`) | MEDIUM–HIGH |
| Naive Bayes | Gaussian/Multinomial/Bernoulli/Complement/Categorical NB | `fit`/`partial_fit`/`predict`/`predict_proba`/`predict_log_proba` | LOW–MEDIUM |

### Differentiators (vs the surrounding ecosystem)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| f64 device path for all 16 | sklearn parity is *comfortable* at 1e-5; most GPU libs are f32-only | — | cpu gates f64; rocm gates f32 |
| Single generic kernel-matrix prim | linear/RBF/poly/sigmoid reused by KRR, KDE (+ future kernel SVM) | MEDIUM | Phase-8 keystone |
| Exact spectral (no approximation) | sklearn parity, not approximate ANN-graph spectral | MEDIUM | full-spectrum eig then take smallest |

### Anti-Features (Avoid)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Bit-exact cuML reproduction | "match the reference" | cuML's LinearSVC=L-BFGS, SGD lacks schedules — diverges from sklearn oracle | Match **sklearn** objective; cuML is API-shape reference only |
| Callable/numba custom kernels (KRR/KDE) | cuML supports device-fn kernels | No numba on CubeCL; huge surface, no oracle | Fixed string kernels only (linear/rbf/poly/sigmoid/laplacian) |
| `kd_tree`/`ball_tree` for KDE | sklearn default `algorithm='auto'`→tree | Tree builds fight cpu-MLIR no-SharedMemory; v1 KNN is brute-force | Brute-force kernel-sum KDE; identical to sklearn within 1e-5 |
| liblinear's exact dual-CD iterate path (LinearSVC) | "match sklearn iteration-for-iteration" | liblinear shrinking/working-set is host-serial, not portable to CubeCL | Match the **converged optimum** of the same regularized objective (1e-5), not the iterate trajectory |
| `crammer_singer` multiclass (LinearSVC) | sklearn option | Rarely used, separate joint QP | Implement `ovr` only (sklearn default); raise on `crammer_singer` |
| Iterate-exact SGD via RNG-shuffle reproduction | sklearn `shuffle=True` default | t0 heuristic + per-epoch NumPy shuffle make iterate-exact parity infeasible across PRNGs | Gate on converged optimum (oracle with `shuffle=False`, fixed `max_iter`, `tol=0`) + f32 band |

---

## Per-Family Detail

### Family 1 — Covariance & Projection (Phase 7)

**New shared primitive:** seeded device RNG-matrix generator (Gaussian + sparse-ternary); incremental-SVD merge step.
**Reuses:** v1 covariance/Gram prim, Jacobi SVD, GEMM, reductions, Cholesky/pinv.

#### EmpiricalCovariance
- **(a) Table stakes:** MLE covariance of centered (or assume-centered) data; `get_precision`, `mahalanobis`, `error_norm`, `score` (Gaussian log-likelihood).
- **(b) Objective/defaults pinned:**
  - `covariance_ = (Xc.T @ Xc) / n_samples` (**ddof=0 / divide by n, not n−1** — sklearn MLE).
  - `location_ = mean(X, axis=0)` unless `assume_centered=True` → zeros.
  - `precision_ = pinv(covariance_)` (sklearn uses `linalg.pinvh`; Moore–Penrose pseudo-inverse of the SPD covariance).
  - `score(X_test)` = mean Gaussian log-likelihood: `-0.5*(n_features*log(2π) − logdet(precision) + mean(mahalanobis))` (matches LedoitWolf.score in cuML source lines 335–339).
  - Defaults: `store_precision=True`, `assume_centered=False`.
- **(c) Fitted attrs:** `covariance_` (d×d), `location_` (d), `precision_` (if stored), `n_features_in_`.
- **(d) Complexity:** O(n·d²). Pure assembly on covariance prim + pinv (eig/SVD-based). No new kernel.

#### LedoitWolf
- **(a) Table stakes:** shrinkage-regularized covariance toward scaled identity; same `score`/`mahalanobis`/`get_precision` surface.
- **(b) Objective/defaults pinned** (sklearn formula, confirmed in cuML `_ledoit_wolf_shrinkage`, lines 18–86):
  - `emp_cov = (Xc.T @ Xc)/n`, `mu = trace(emp_cov)/n_features`.
  - `beta_ = Σ(X²ᵀ X²)`, `delta_ = Σ(XᵀX)² / n²`.
  - `beta = (1/(n_features·n))·(beta_/n − delta_)`; `delta = (delta_ − 2·mu·Σ(diag emp_cov) + n_features·mu²)/n_features`.
  - `beta = min(beta, delta)`; **`shrinkage = 0` if `beta==0` else `beta/delta`** (∈[0,1] since β≤δ).
  - `covariance_ = (1−shrinkage)·emp_cov`, then `diag += shrinkage·mu` in-place (cuML line 273).
  - Defaults: `store_precision=True`, `assume_centered=False`, `block_size=1000` (memory tiling only — must not change result).
  - **Single-feature special case:** `shrinkage=0`, `emp_cov = cov(X, ddof=0)` (cuML lines 43–48).
- **(c) Fitted attrs:** `covariance_`, `location_`, `precision_`, **`shrinkage_`** (float ∈[0,1]), `n_features_in_`.
- **(d) Complexity:** O(n·d²). Assembly on covariance prim + reductions. No new kernel.
- **⚠ Parity risk:** the `beta`/`delta` accumulations sum `Σ(XᵀX)²` over the full Gram — f32 accumulation order matters; accumulate in f32→f32 carefully or document the f32 band. `block_size` tiling must be result-invariant.

#### IncrementalPCA
- **(a) Table stakes:** batched/streaming PCA via `partial_fit`; identical `transform`/`inverse_transform`/`components_` semantics to PCA; supports out-of-core and `n_samples < n_features`.
- **(b) Objective/defaults pinned (sklearn `IncrementalPCA`):**
  - Maintains running `mean_`, `var_`, `n_samples_seen_`. Per batch: update mean/var via **batch merge** (sklearn `_incremental_mean_and_var`).
  - Augment current factors: stack `[ sqrt(n_seen)·diag(S)·Vᵀ ; Xc_batch ; mean-correction row ]`, SVD that small matrix → new `components_ = Vᵀ[:n_components]`.
  - Mean-correction row = `sqrt(n_seen·n_batch / (n_seen+n_batch)) · (batch_mean − running_mean)`.
  - **`svd_flip(U, V, u_based_decision=False)`** sign convention: force sign by the largest-abs entry of each **right** singular vector. Must replicate exactly.
  - `explained_variance_ = S²/(n_total−1)` (**ddof=1**, unlike covariance estimators), `explained_variance_ratio_`, `noise_variance_` = mean of discarded eigenvalues.
  - Defaults: `n_components=None` (→ min(batch, n_features)), `whiten=False`, `batch_size=None` (→ `5·n_features`), `copy=True`.
- **(c) Fitted attrs:** `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`, `var_`, `noise_variance_`, `n_components_`, `n_samples_seen_`, `batch_size_`, `n_features_in_`.
- **(d) Complexity:** O(batch·d·k) per step + small SVD of (k+batch)×d. Reuses Jacobi SVD; **needs incremental-SVD merge prim** (stack-and-resvd). See `research/questions.md [v2-P1]`.
- **⚠ Parity risk (HIGH):** (1) **`svd_flip(u_based_decision=False)`** sign — wrong convention flips `components_` signs (harness has sign-flip helper, but be deliberate). (2) **ddof=1** for explained_variance vs ddof=0 covariance MLE. (3) f32 stability of the stacked-resvd merge on rocm — open research question.

#### GaussianRandomProjection / SparseRandomProjection
- **(a) Table stakes:** `fit` builds a random projection matrix; `transform = X @ components_.T`; auto-dim via Johnson–Lindenstrauss.
- **(b) Objective/defaults pinned (sklearn `random_projection`):**
  - `n_components='auto'` → `johnson_lindenstrauss_min_dim(n_samples, eps) = (4·ln(n_samples))/(eps²/2 − eps³/3)`, ceil'd; **`eps=0.1`** default.
  - **Gaussian:** entries iid `N(0, 1/n_components)`.
  - **Sparse:** density `s`; default `density='auto' = 1/sqrt(n_features)`. Entries from `{−sqrt(1/(s·n_components)), 0, +sqrt(1/(s·n_components))}` with prob `{s/2, 1−s, s/2}` (Achlioptas/Li). sklearn stores sparse `components_`; mlrs may densify (small) or keep CSR.
  - `compute_inverse_components=False` default; `transform` does **not** center.
  - **RNG:** sklearn uses NumPy MT19937 via `check_random_state`. mlrs must use a **seeded reproducible PRNG** (ASVS V6 — no OsRng); exact value-parity with NumPy MT is **not** achievable.
- **(c) Fitted attrs:** `components_` (n_components×n_features), `n_components_`, `density_` (sparse), `n_features_in_`; `inverse_components_` if requested.
- **(d) Complexity:** O(n·d·k) transform GEMM + matrix gen. Reuses GEMM; **needs RNG-matrix prim** (`research/questions.md [v2-P1]`).
- **⚠ Parity risk (HIGH — no value oracle):** because mlrs's PRNG ≠ NumPy MT, the projection matrix can't be value-matched at 1e-5. Gate instead on (i) correct `n_components_` from the JL formula, (ii) JL pairwise-distance distortion bound within `eps` on a test set, (iii) shape/dtype/density. **This is the one v2 family where the elementwise 1e-5 gate does not apply** — requirements must state property-based acceptance.

---

### Family 2 — Kernel (Phase 8)

**New shared primitive:** kernel-matrix prim covering linear/RBF/poly/sigmoid (+laplacian for KDE) over pairwise distance.
**Reuses:** v1 pairwise-distance prim, Cholesky/`posv`, GEMM, reductions, log-sum-exp.

#### KernelRidge
- **(a) Table stakes:** dual-space ridge with kernel trick; multi-output; `fit`/`predict`.
- **(b) Objective/defaults pinned (sklearn `KernelRidge`, confirmed cuML `_solve_cholesky_kernel`):**
  - Solve `(K + α·I)·dual_coef = y`, `K = kernel(X_fit, X_fit)`.
  - Solver: **Cholesky** (`posv`) on `K + αI`; on singularity fall back to least-squares (`lstsq`) with a warning (cuML `_safe_solve`, lines 34–52).
  - Per-target α (array): loop adding/subtracting α on the diagonal per target (cuML lines 81–95).
  - `sample_weight`: `K *= outer(sqrt(sw), sqrt(sw))`, `y *= sqrt(sw)`, un-scale dual after (cuML lines 66–78).
  - `predict(X) = kernel(X, X_fit) @ dual_coef`.
  - Defaults: `alpha=1.0`, `kernel='linear'`, `gamma=None` (→ `1/n_features` for rbf/poly/sigmoid), `degree=3`, `coef0=1`.
  - Kernel formulas: linear `XYᵀ`; rbf `exp(−γ‖x−y‖²)`; poly `(γ·XYᵀ + coef0)^degree`; sigmoid `tanh(γ·XYᵀ + coef0)`.
- **(c) Fitted attrs:** `dual_coef_` (n_samples or n_samples×n_targets), `X_fit_` (training data, needed at predict), `n_features_in_`.
- **(d) Complexity:** O(n²·d) kernel build + O(n³) Cholesky. Reuses distance/Cholesky; needs **kernel-matrix prim** (`research/questions.md [v2-P2]`).
- **⚠ Parity risk:** singular-K fallback (lstsq) — decide whether to replicate warn+lstsq or always regularize; document. f64 Cholesky hits 1e-5 easily; f32 large-n may trigger fallback more often.

#### KernelDensity
- **(a) Table stakes:** `fit` (store training pts); `score_samples` (per-point log-density), `score` (total log-likelihood), `sample` (draw from KDE).
- **(b) Objective/defaults pinned (sklearn `neighbors.KernelDensity`):**
  - `log_density(x) = logsumexp_i( log_kernel(‖x − Xᵢ‖/h) ) − log(n) + log_norm(kernel, h, d)` — normalization is kernel- and dimension-specific.
  - Kernels: **gaussian** `exp(−d²/(2h²))`, tophat, epanechnikov, exponential, linear, cosine. mlrs MVP: gaussian + at least tophat/exponential; document supported set.
  - **Per-kernel log-normalization constants** (sklearn `_kernels`): gaussian `−0.5·d·log(2π) − d·log(h)`; tophat/epanechnikov use the unit d-ball volume via `lgamma`. **Must match sklearn exactly.**
  - Defaults: `bandwidth=1.0` (also `'scott' = n^(−1/(d+4))`, `'silverman' = (n·(d+2)/4)^(−1/(d+4))` since sklearn 1.0), `kernel='gaussian'`, `metric='euclidean'`, `atol=0`, `rtol=0`, `algorithm='auto'` (mlrs brute-force; identical result).
  - `score_samples` returns **log** density.
- **(c) Fitted attrs:** training data (mlrs stores `X` in place of `tree_`), `bandwidth_` (resolved float, sklearn 1.x), `n_features_in_`.
- **(d) Complexity:** O(m·n·d) brute-force. Reuses distance prim + log-sum-exp reduction; needs kernel-matrix (distance variant) + numerically-stable logsumexp.
- **⚠ Parity risk:** (1) **per-kernel log-norm constants** (esp. d-ball volume via `lgamma` for tophat/epanechnikov). (2) logsumexp f32 stability. (3) `sample()` needs the RNG-matrix prim (gaussian only) — may defer `sample`, gate only `score_samples`/`score`.

---

### Family 3 — Spectral (Phase 9)

**New shared primitive:** graph-Laplacian builder (affinity → degree → normalized Laplacian).
**Reuses:** v1 symmetric **eig**, pairwise distance, KMeans (SpectralClustering), GEMM.

#### SpectralEmbedding
- **(a) Table stakes:** `fit_transform` → low-dim embedding from the smallest nontrivial eigenvectors of the graph Laplacian.
- **(b) Objective/defaults pinned (sklearn `manifold.SpectralEmbedding`):**
  - Affinity `W`: default **`affinity='nearest_neighbors'`** → kNN graph (`n_neighbors=max(n_samples//10, 1)`), symmetrized `0.5(W+Wᵀ)`; or `affinity='rbf'` → `exp(−γ‖x−y‖²)`, `gamma=None`→`1/n_features`.
  - Degree `D=diag(rowsum W)`; **normalized Laplacian** `L_sym = I − D^{−1/2} W D^{−1/2}` (`norm_laplacian=True` default).
  - Take eigenvectors of the **`n_components` smallest** eigenvalues, **drop the trivial first**, then **recover** `embedding = eigvec / sqrt(degree)` (`drop_first=True`).
  - Deterministic sign flip (`_deterministic_vector_sign_flip`: sign of the max-abs entry per vector). **Must replicate.**
  - Defaults: `n_components=2`, `affinity='nearest_neighbors'`, `gamma=None`, `n_neighbors=None`, `eigen_solver=None`.
- **(c) Fitted attrs:** `embedding_` (n×n_components), `affinity_matrix_`, `n_features_in_`, `n_neighbors_`.
- **(d) Complexity:** O(n²) affinity + O(n³) dense eig (fine at v2 sizes). Reuses eig + distance; needs **Laplacian prim**.
- **⚠ Parity risk (HIGH):** (1) **smallest** eigenvectors — v1 Jacobi eig returns full spectrum descending; take the tail. Full-spectrum-then-take-smallest acceptable at v2 sizes (`research/questions.md [v2-P3]`); document the size ceiling. (2) **deterministic sign flip** must match sklearn's convention. (3) the `embedding /= sqrt(degree)` recovery step is easy to omit. (4) disconnected-graph warning behavior for nearest_neighbors.

#### SpectralClustering
- **(a) Table stakes:** `fit_predict` → labels via spectral embedding + KMeans on the embedding.
- **(b) Objective/defaults pinned (sklearn `cluster.SpectralClustering`):**
  - Build affinity (default **`affinity='rbf'`**, *unlike* SpectralEmbedding's nearest_neighbors), normalized Laplacian, `n_clusters` smallest eigenvectors → `maps`.
  - **`assign_labels='kmeans'`** default → KMeans (k-means++, `n_init=10`) on the embedding rows. (`'discretize'`/`'cluster_qr'` deferred.)
  - Pin sklearn's `spectral_clustering` exact pipeline (the embedding is passed to KMeans without per-row L2 normalization for the kmeans path).
  - Defaults: `n_clusters=8`, `eigen_solver=None`, `n_components=n_clusters`, **`gamma=1.0`** (rbf gamma default is 1.0 here, **not** 1/n_features), `affinity='rbf'`, `n_neighbors=10`, `assign_labels='kmeans'`, `n_init=10`.
- **(c) Fitted attrs:** `labels_`, `affinity_matrix_`, `n_features_in_`.
- **(d) Complexity:** SpectralEmbedding + KMeans. Reuses Laplacian prim + eig + **KMeans**.
- **⚠ Parity risk (HIGH):** (1) **label permutation** (use harness label-perm helper — KMeans labels arbitrary up to permutation). (2) `gamma` default differs from SpectralEmbedding (1.0 vs 1/n_features). (3) KMeans seed-sensitivity — gate on ARI / clustering agreement, possibly not exact labels under f32.

---

### Family 4 — SGD / Linear-SVM (Phase 10)

**New shared primitive:** minibatch SGD solver (hinge / log / squared_loss / squared_hinge / epsilon_insensitive losses; l1/l2/elasticnet penalties; learning-rate schedules).
**Reuses:** reductions, GEMM/GEMV, v1 coordinate descent (for LinearSVC/SVR converged-optimum path).

> **Critical divergence from cuML.** cuML's `SGD` (source read) supports only
> `{squared_loss, log, hinge}` losses, `{constant, invscaling, adaptive}` schedules, **no
> averaging, no `optimal` schedule, no squared_hinge/epsilon_insensitive**. And cuML's
> **LinearSVC uses L-BFGS/OWL-QN**, not liblinear. The oracle is **sklearn**, so mlrs implements
> the **sklearn** objectives/schedules below (richer than cuML's). cuML `sgd.pyx` = API-shape reference only.

#### MBSGDClassifier / MBSGDRegressor (mlrs-named; sklearn analog = SGDClassifier / SGDRegressor)
- **(a) Table stakes:** minibatch SGD over a linear model; `fit`/`predict` (+`decision_function`, `partial_fit`).
- **(b) Objective/defaults pinned (sklearn SGD*, verified live):**
  - **Losses** — Classifier: `hinge`(default, linear SVM), `log_loss`, `modified_huber`, `squared_hinge`, `perceptron`; Regressor: `squared_error`(default), `huber`, `epsilon_insensitive`, `squared_epsilon_insensitive`. mlrs MVP covers ≥ `hinge`, `log_loss`, `squared_error`, `epsilon_insensitive` (the four named in the seed).
  - **Penalty:** `l2`(default), `l1`, `elasticnet`; `alpha=0.0001`, `l1_ratio=0.15`. Term = `alpha·(l1_ratio·‖w‖₁ + (1−l1_ratio)·0.5·‖w‖₂²)`.
  - **Learning-rate schedules** (`learning_rate` default **`'optimal'`** for Classifier, **`'invscaling'`** for Regressor — *they differ*):
    - `optimal`: `η(t) = 1/(alpha·(t + t0))`, `t0` from Bottou's heuristic (`typw = sqrt(1/sqrt(alpha))`; `eta0_init = 1/(typw·max(1,−dloss(−typw)))`; `t0 = 1/(alpha·eta0_init)`). **Replicate this t0 formula for parity.**
    - `constant`: `η = eta0`. `invscaling`: `η(t) = eta0/t^power_t`. `adaptive`: start `eta0`, ÷5 when no improvement for `n_iter_no_change` epochs.
    - Defaults: `eta0=0.0` (must be >0 for constant/invscaling/adaptive), `power_t=0.5`.
  - **Stopping:** `max_iter=1000`, `tol=1e-3`; stop when `loss > best_loss − tol` for `n_iter_no_change=5` consecutive epochs (vs **training** loss unless `early_stopping=True`).
  - **Averaging:** `average=False` default; if True/int → `coef_` is the running ASGD average of weights. Implement to match when set.
  - **`t_`** (weight-update counter) drives the `optimal` schedule: `t_ = n_iter_·n_samples + 1`.
  - **Intercept** updated at full η; `fit_intercept=True`. **Multiclass:** OvR (one binary SGD per class).
- **(c) Fitted attrs:** `coef_` ((1,d) binary / (n_classes,d) multiclass / (d,) regressor), `intercept_`, `n_iter_`, `t_`, `classes_` (classifier), `n_features_in_`.
- **(d) Complexity:** O(epochs·n·d). New SGD solver prim (`research/questions.md [v2-P4]`); reuses GEMV/reductions.
- **⚠ Parity risk (HIGH — hardest in v2):**
  1. **`optimal` schedule + Bottou `t0`** — exact replication needed or iterate path / fixed-epoch optimum drifts.
  2. **Per-epoch shuffle uses RNG** — NumPy permutation ≠ mlrs PRNG, so the iterate trajectory can't be value-matched. Gate on **converged optimum** (oracle `shuffle=False`, fixed `max_iter`, `tol=0`) OR a generous f32 band. Document the oracle harness setup explicitly.
  3. **Loss derivatives** — hinge subgradient (0 inside margin), modified_huber, squared_hinge must match sklearn `_sgd_fast` piecewise definitions.
  4. **Regressor default schedule `invscaling` vs classifier `optimal`** — easy to conflate.
  5. cpu-MLIR: minibatch update loop must avoid SharedMemory / cross-unit atomics (cf. v1 GATHER idiom).

#### LinearSVC
- **(a) Table stakes:** large-margin linear classifier; `fit`/`predict`/`decision_function`; OvR multiclass.
- **(b) Objective/defaults pinned (sklearn liblinear, verified live):**
  - Objective (default `squared_hinge`, `l2`): minimize `0.5‖w‖² + C·Σ max(0, 1 − yᵢ(wᵀxᵢ+b))²`.
  - Defaults: **`penalty='l2'`, `loss='squared_hinge'` (NOT hinge), `dual='auto'`, `C=1.0`, `tol=1e-4`, `max_iter=1000`, `multi_class='ovr'`, `fit_intercept=True`, `intercept_scaling=1.0`.**
  - `dual='auto'`: choose dual vs primal by n_samples vs n_features + penalty/loss combo (prefers `dual=False` when n_samples>n_features). Combos: `l1`+`hinge` unsupported; `l1`+`squared_hinge` primal-only; `l2`+`hinge` dual-only; `l2`+`squared_hinge` both.
  - **Intercept regularization quirk:** liblinear augments X with a constant `intercept_scaling` column and **regularizes the intercept** (part of `‖w‖²`, scaled by intercept_scaling). Pin this — it's the known sklearn-vs-clean-SVM difference.
  - solver = liblinear CD. **mlrs matches the converged optimum of the objective (1e-5), NOT liblinear's iterate path.**
- **(c) Fitted attrs:** `coef_` ((1,d) binary / (n_classes,d)), `intercept_`, `classes_`, `n_iter_`, `n_features_in_`.
- **(d) Complexity:** convex QP via coordinate descent (reuse v1 CD adapted to hinge/squared_hinge) or dual CD. New: hinge/squared_hinge loss path.
- **⚠ Parity risk (HIGH):** (1) default loss is **squared_hinge** (frequent mistake). (2) **regularized intercept via intercept_scaling** — without it the intercept differs from sklearn. (3) `dual='auto'` resolution. (4) OvR label/sign conventions.

#### LinearSVR
- **(a) Table stakes:** linear ε-insensitive regression; `fit`/`predict`.
- **(b) Objective/defaults pinned (sklearn liblinear, verified live):**
  - Objective (default `squared_epsilon_insensitive`): minimize `0.5‖w‖² + C·Σ max(0, |yᵢ − (wᵀxᵢ+b)| − ε)²`.
  - Defaults: **`epsilon=0.0`, `loss='squared_epsilon_insensitive'` (NOT epsilon_insensitive), `dual=True`, `C=1.0`, `tol=1e-4`, `max_iter=1000`, `fit_intercept=True`, `intercept_scaling=1.0`.**
  - `epsilon_insensitive` (L1) uses the non-squared residual hinge. Same liblinear intercept-reg quirk.
- **(c) Fitted attrs:** `coef_` ((d,)/(1,d)), `intercept_`, `n_iter_`, `n_features_in_`.
- **(d) Complexity:** convex; CD on ε-insensitive objective. New: ε-insensitive loss path (shared with MBSGDRegressor's `epsilon_insensitive`).
- **⚠ Parity risk (MED-HIGH):** (1) default loss **squared_epsilon_insensitive** + default **`epsilon=0.0`** (behaves like squared error + intercept quirk). (2) `dual=True` default (no auto). (3) intercept regularization.

---

### Family 5 — Naive Bayes (Phase 11)

**New shared primitive:** none (reductions only — class-conditional sums/counts).
**Reuses:** reductions, log/exp, segment-by-class grouped reductions, log-sum-exp.

All five share: `fit`/`partial_fit`/`predict`/`predict_proba`/`predict_log_proba`/`score`; predict = argmax of joint log-likelihood `log P(y) + Σ log P(xⱼ|y)`; `predict_proba` = log-sum-exp normalize of the joint LL. `fit_prior=True` default → empirical priors; else uniform. Common attrs: `classes_`, `class_count_`, `class_log_prior_`, `n_features_in_`.

#### GaussianNB
- **(b) Pinned:** per-class `theta_` (mean), `var_` (variance). **`var_smoothing=1e-9`** default: `epsilon_ = var_smoothing · max_j(Var(X[:,j]) over whole dataset)` added to all variances (sklearn: `var_smoothing * X.var(axis=0).max()`). LL = `−0.5·Σ[log(2π·var) + (x−mean)²/var]`. `priors=None`→empirical.
- **(c) Attrs:** `theta_` (n_classes×d), `var_` (n_classes×d), `class_prior_`, `class_count_`, `epsilon_`, `classes_`.
- **⚠ Parity risk:** `epsilon_` from **global** feature variance (max over features), not per-class — common bug. `partial_fit` uses running-variance merge.

#### MultinomialNB
- **(b) Pinned:** `feature_log_prob_[c,j] = log((count[c,j] + alpha)/(Σ_j count[c,j] + alpha·n_features))` (Lidstone/Laplace). **`alpha=1.0`**, `force_alpha=True` default. `fit_prior=True`, `class_prior=None`. Joint LL = `class_log_prior_ + X @ feature_log_prob_.T`.
- **(c) Attrs:** `feature_log_prob_` (n_classes×d), `feature_count_`, `class_log_prior_`, `class_count_`, `classes_`.
- **⚠ Parity risk:** smoothing denominator uses `alpha·n_features`, not `alpha·1`.

#### BernoulliNB
- **(b) Pinned:** **`binarize=0.0`** default → X>binarize → 1 (None skips, assumes binary). `feature_log_prob_[c,j] = log((count[c,j]+alpha)/(class_count[c]+2·alpha))`. **alpha=1.0**. Decision uses explicit non-occurrence term: `LL = class_log_prior + Σ_j[ x_j·log p_cj + (1−x_j)·log(1−p_cj) ]` (`neg_prob`). `fit_prior=True`.
- **(c) Attrs:** `feature_log_prob_`, `feature_count_`, `class_log_prior_`, `class_count_`, `classes_`.
- **⚠ Parity risk:** the `(1−x)·log(1−p)` non-occurrence term is what distinguishes Bernoulli from Multinomial — must include it.

#### ComplementNB
- **(b) Pinned:** complement-class stats: `weights = log((complement_count + alpha)/(complement_count.sum() + alpha·n_features))` (counts from **all classes except c**), then if **`norm=True`** L1-normalize the weights (default `norm=False`). **alpha=1.0**, `fit_prior=True`. Decision: `argmin` of `X @ weights` (CNB picks the class whose *complement* fits worst — note the sign).
- **(c) Attrs:** `feature_log_prob_`, `feature_all_` (=`feature_count_.sum(axis=0)`), `feature_count_`, `class_log_prior_`, `class_count_`, `classes_`.
- **⚠ Parity risk:** the **complement** weighting + optional L1 norm (`norm`) + sign/argmin convention are CNB-specific — do not copy MNB.

#### CategoricalNB
- **(b) Pinned:** each feature categorical with `n_categories_[j]` levels; per-(class, feature, category) counts. `feature_log_prob_[j][c,k] = log((count + alpha)/(class_count[c] + alpha·n_categories_j))`. **alpha=1.0**, `fit_prior=True`, **`min_categories=None`** (infer per feature, or pad). Inputs must be non-negative integer-encoded categories. Joint LL sums per-feature looked-up log-probs.
- **(c) Attrs:** `category_count_` (list per feature), `feature_log_prob_` (list of n_classes×n_categories_j), `class_log_prior_`, `class_count_`, `n_categories_`, `classes_`.
- **⚠ Parity risk:** `feature_log_prob_` is a **ragged list** (one matrix per feature, variable category count) — not a single tensor; unseen categories at predict map to a smoothed prob. `min_categories` padding.

---

## Feature Dependencies

```
[RNG-matrix prim (P7)] ──required──> [Gaussian/SparseRandomProjection]
                       ──enhances──> [KernelDensity.sample], [SGD shuffle]

[incremental-SVD merge (P7)] ──required──> [IncrementalPCA]   (reuses Jacobi SVD)

[covariance prim (v1)] ──required──> [EmpiricalCovariance] ──required──> [LedoitWolf]

[kernel-matrix prim (P8)] ──required──> [KernelRidge], [KernelDensity]
                          ──enables──> [future kernel SVM (v3)]

[graph-Laplacian prim (P9)] ──required──> [SpectralEmbedding] ──required──> [SpectralClustering]
[v1 eig]    ──required──> [SpectralEmbedding/Clustering]
[v1 KMeans] ──required──> [SpectralClustering]

[SGD solver (P10)] ──required──> [MBSGDClassifier], [MBSGDRegressor]
[epsilon-insensitive loss] ──shared by──> [MBSGDRegressor], [LinearSVR]
[hinge/squared_hinge loss] ──shared by──> [MBSGDClassifier], [LinearSVC]
[v1 coordinate descent] ──reused by──> [LinearSVC], [LinearSVR]  (converged-optimum parity)

[reductions (v1)] ──required──> [all 5 Naive Bayes]
```

### Dependency Notes
- **EmpiricalCovariance before LedoitWolf:** LW = EmpiricalCovariance + a shrinkage scalar; validate the empirical path first.
- **SpectralEmbedding before SpectralClustering:** SC = SE-embedding + KMeans; share the Laplacian+eig core.
- **MBSGD before LinearSVC/SVR is optional:** LinearSVC/SVR can target the converged objective via v1 CD instead of SGD; the SGD solver is required only for the MBSGD* pair. The hinge / ε-insensitive *loss definitions* are shared either way.
- **Naive Bayes are mutually independent** — five parallel-buildable estimators sharing only reductions; lowest-risk family.

## MVP Definition (this milestone = the 16 firm estimators)

### Launch With (v2.0)
- [ ] EmpiricalCovariance, LedoitWolf — covariance + pinv + score path
- [ ] IncrementalPCA — incremental-SVD merge prim (flag svd_flip sign / ddof parity)
- [ ] Gaussian/SparseRandomProjection — RNG-matrix prim (property-gated, not 1e-5)
- [ ] KernelRidge, KernelDensity — kernel-matrix prim
- [ ] SpectralEmbedding, SpectralClustering — Laplacian prim (cash in v1 eig + KMeans)
- [ ] MBSGDClassifier, MBSGDRegressor — SGD solver
- [ ] LinearSVC, LinearSVR — hinge/squared_hinge/ε-insensitive losses (sklearn liblinear objective)
- [ ] Gaussian/Multinomial/Bernoulli/Complement/Categorical NB — reductions only

### Future Consideration (v3)
- [ ] `crammer_singer` multiclass, callable kernels, KDE `sample`, spectral `discretize`/`cluster_qr`, tree-based KDE — deferred (see Anti-Features)

## Feature Prioritization Matrix

| Family | User Value | Implementation Cost | Priority | Parity risk |
|--------|------------|---------------------|----------|-------------|
| Naive Bayes (5) | HIGH | LOW | P1 | LOW (per-variant smoothing nuances) |
| Covariance (2) | MEDIUM | LOW | P1 | LOW |
| KernelRidge | MEDIUM | LOW | P1 | LOW–MED (singular fallback) |
| KernelDensity | MEDIUM | MEDIUM | P1 | MED (log-norm constants) |
| SpectralEmbedding/Clustering | MEDIUM | MEDIUM | P1 | HIGH (smallest-eig, sign, label-perm) |
| IncrementalPCA | MEDIUM | MEDIUM | P1 | HIGH (svd_flip sign, ddof, f32 merge) |
| RandomProjection (2) | MEDIUM | LOW | P1 | HIGH (no value oracle — property-gate) |
| MBSGD (2) | MEDIUM | HIGH | P1 | HIGH (optimal schedule, shuffle RNG) |
| LinearSVC/SVR | HIGH | HIGH | P1 | HIGH (squared_hinge default, intercept reg) |

## Sources

- cuML v26.08 source (read-only reference): `covariance/ledoit_wolf.py`, `kernel_ridge/kernel_ridge.py`, `solvers/sgd.pyx` — confirms LW shrinkage formula, KRR Cholesky dual solve, and that cuML's SGD is a *subset* of sklearn's (no optimal/averaging/squared_hinge). [HIGH]
- scikit-learn docs (verified live 2026-06-14): [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) — optimal/invscaling/constant/adaptive schedules, t0 heuristic, tol/n_iter_no_change stopping, `average`. [HIGH]
- scikit-learn docs (verified live 2026-06-14): [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) — `squared_hinge`/`dual='auto'` defaults; LinearSVR `squared_epsilon_insensitive`/`epsilon=0.0`/`dual=True`, liblinear solver. [HIGH]
- scikit-learn algorithm semantics (NB smoothing, IncrementalPCA merge/svd_flip, RandomProjection JL min-dim + sparse density, KDE kernels/bandwidth, spectral normalized-Laplacian + deterministic sign flip): stable documented API, knowledge cutoff Jan 2026, cross-checked against cuML where overlapping. [HIGH]
- Project context: `.planning/PROJECT.md`, `seeds/v2-breadth-roadmap.md`, `notes/cuml-mlrs-gap-inventory.md`, `research/questions.md`. [HIGH]

---
*Feature research for: scikit-learn-compatible ML estimators (mlrs v2.0)*
*Researched: 2026-06-14*
