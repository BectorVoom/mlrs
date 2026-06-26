# Requirements: mlrs — v4.0 Tree Ensembles, Time-Series & Full-Surface Completion

**Defined:** 2026-06-26
**Core Value:** Correct, memory-efficient ML algorithms that match scikit-learn within 1e-5, running on any CubeCL backend from a single generic codebase.

> **Oracle note for v4.** v4.0 is the first milestone to span **all four gate regimes at once** — every requirement below carries an explicit gate type:
> - **Value gate (≤1e-5):** FIL batched inference (exact vs a host reference traversal of the identical node arrays), TreeSHAP (vs `shap.TreeExplainer` on mlrs's *own* tree), metrics, preprocessing scalers, Tfidf weights.
> - **Exact / structural gate:** decision-tree core on injected fixed indices, encoders/splitters/vocabulary parity.
> - **Property/structural + band gate (D-12 / UMAP-03 precedent):** RandomForest ensemble, Kernel/Permutation SHAP, symbolic regression — SplitMix64 ≠ NumPy MT19937 ⇒ no element-wise match. The **exact-predicted-label gate (deterministic SGD/SVM/NB) does NOT apply to RF.** The milestone-wide convention is the **two-tier stochastic gate**: a deterministic injected-fixed-index single-tree/core tier (the real correctness witness) + an ensemble/predictive-quality band tier (RNG-tolerant), established in the Phase-17 spike.
> - **Stats band gate:** ARIMA — oracle is **`statsmodels.tsa`, NOT sklearn** (sklearn has no ARIMA; raw coefficients are multimodal/ungatable). Gate log-likelihood / forecast-band / known-coefficient recovery via the Jones/PACF transform.
>
> Backend gate unchanged: **cpu(f64) + rocm(f32)**; f64-on-rocm SKIPS-with-log. **Zero new compute dependencies** (continuing the v2/v3 record); the only additive deps are three test-only Python oracles — `shap 0.52.0`, `statsmodels 0.14.4`, `gplearn 0.4.3` — plus the already-pinned `scikit-learn ≥1.6`. `pyo3` stays 0.28, `cubecl` stays 0.10.0. Primitive-first: every new compute primitive (GATHER histogram/split/partition, batched Kalman, program-eval) is standalone-gated before any estimator consumes it. **The RandomForest feasibility spike gates the entire tree chain (RF → FIL → TreeSHAP); nothing tree-family is committed until it returns GO.**

## v1 Requirements

Requirements for the v4.0 milestone. Each maps to a roadmap phase.

### Tree Feasibility & Primitives (build & gate FIRST — gates the tree chain)

- [ ] **TREE-01**: A GPU tree-construction **feasibility spike** proves (or refutes) that a single-owner GATHER histogram/split lowers and is tractable under cpu-MLIR — no SharedMemory, no atomics, no `F::INFINITY` init. It delivers GATHER-histogram + relabel-partition + seed-from-first split-find kernels standalone-launching on cpu(f64)+rocm(f32), a VALUE-asserting correctness test vs `sklearn.tree.DecisionTree*` on **injected fixed bootstrap/feature indices**, a per-tree cost benchmark, a finalized `SparseTreeNode { colid, threshold, left_child, value }` (right = left+1) format contract, an established two-tier stochastic-gate convention, and an explicit **GO / ADJUST / ABORT** verdict with abort signals A1–A5 evaluated. *(gate: spike verdict + VALUE on fixed-index tree)*
- [ ] **TREE-02**: The tree primitives (`quantiles`, `tree_hist`, `best_split`, `node_partition`) are standalone-validated and a `DecisionTreeClassifier`/`DecisionTreeRegressor` core (level-wise host loop) is oracle-gated vs `sklearn.tree.DecisionTree*` on injected fixed indices, with a build-failing frontier-only-histogram PoolStats memory gate. *(gate: exact/structural on fixed-index tree; ≤1e-5 leaf values f64)*

### Random Forest (RF)

- [ ] **RF-01**: User can fit `RandomForestClassifier` (`fit`/`predict`/`predict_proba`) with sklearn-named hyperparameters and defaults (`n_estimators=100`, `criterion='gini'|'entropy'|'log_loss'`, `max_depth`, `max_features='sqrt'`, `min_samples_split=2`, `min_samples_leaf=1`, `bootstrap=True`, `max_samples`, `random_state`), passing the two-tier gate: deterministic injected-index single-tree match + accuracy-within-band vs `sklearn.ensemble.RandomForestClassifier`. *(gate: two-tier property+band)*
- [ ] **RF-02**: User can fit `RandomForestRegressor` (`fit`/`predict`) with `criterion='squared_error'|'absolute_error'` and the shared forest hyperparameters, passing the two-tier gate (R²-within-band vs `sklearn.ensemble.RandomForestRegressor`). *(gate: two-tier property+band)*
- [ ] **RF-03**: A fitted RandomForest exposes `feature_importances_` (impurity-based, summing to 1) and `oob_score_` (when `bootstrap=True` and `oob_score=True`), each structurally/band gated vs sklearn. *(gate: structural + band)*

### Forest Inference (FIL)

- [ ] **FIL-01**: User can run batched forest inference (`predict`/`predict_proba`) over the mlrs node store via an iterative (non-recursive) `node_id` device traversal, producing output **exactly equal** to a host reference walk of the identical node arrays, with a row-streaming PoolStats memory gate. *(gate: exact vs host reference traversal)*

### Explainers / SHAP (SHAP)

- [ ] **SHAP-01**: User can compute **path-dependent TreeSHAP** values for a fitted mlrs forest, matching `shap.TreeExplainer` (fed mlrs's *own* tree, NOT sklearn's forest) to ≤1e-5, satisfying the additive-efficiency invariant (Σφ + base = prediction), and cross-checked against a brute-force exact Shapley enumeration on small hand-built trees. *(gate: ≤1e-5 + exact additive-efficiency)*
- [ ] **SHAP-02**: User can compute model-agnostic **Kernel SHAP** values for any fitted mlrs estimator (weighted-lstsq over sampled coalitions, reusing the existing linear solver), satisfying the additive-efficiency invariant exactly, with a convergence-band gate vs `shap.KernelExplainer` and a coalition-block-streaming PoolStats gate. *(gate: exact additive-efficiency + band)*
- [ ] **SHAP-03**: User can compute model-agnostic **Permutation SHAP** values for any fitted mlrs estimator, satisfying additive-efficiency exactly and matching `shap.PermutationExplainer` within a convergence band. *(gate: exact additive-efficiency + band)*

### Time Series (ARIMA)

- [ ] **ARIMA-01**: User can fit `ARIMA(order=(p,d,q))` and `forecast(steps)` via a batched Kalman filter + batched L-BFGS with the Jones/PACF stationarity transform and f64 log-likelihood accumulation (even on the f32 rocm path), gated on log-likelihood band + forecast band + known-coefficient recovery vs `statsmodels.tsa.arima.model.ARIMA`, with per-series convergence flags so one non-converging series cannot NaN-poison the batch. *(gate: stats band on likelihood/forecast)*
- [ ] **ARIMA-02**: User can fit `AutoARIMA` (order search via KPSS/stationarity test + information-criterion stepwise/grid), recovering the correct selected `(p,d,q)` on synthetic series with known structure. *(gate: exact selected-order on synthetic + likelihood band)*
- [ ] **ARIMA-03**: User can fit **seasonal** `ARIMA(order, seasonal_order=(P,D,Q,s))` with optional exogenous regressors (`exog`), gated on likelihood + forecast band vs `statsmodels` SARIMAX. *(gate: stats band)*

### Metrics (METR)

- [ ] **METR-01**: User can call classification metrics — `accuracy_score`, `confusion_matrix`, `precision_score`/`recall_score`/`f1_score` (with `average` modes) — matching sklearn exactly/≤1e-5, with mandatory degenerate fixtures (empty class, single sample, all-one-label). *(gate: exact / ≤1e-5)*
- [ ] **METR-02**: User can call regression metrics — `r2_score`, `mean_squared_error`, `mean_absolute_error` — matching sklearn to ≤1e-5, including constant-target and single-sample degenerate fixtures. *(gate: ≤1e-5)*
- [ ] **METR-03**: User can call probabilistic/ranking metrics — `roc_auc_score`, `log_loss`, `precision_recall_curve` — matching sklearn to ≤1e-5, with zero-division/edge fixtures. *(gate: ≤1e-5)*

### Preprocessing (PREP)

- [ ] **PREP-01**: User can fit/transform scalers — `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `Binarizer` — where column statistics are learned only in `fit` and applied in `transform`, matching sklearn to ≤1e-5, with zero-variance/constant-column degenerate handling. *(gate: ≤1e-5 + fit/transform statefulness)*
- [ ] **PREP-02**: User can fit/transform encoders & imputers — `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `LabelBinarizer`, `SimpleImputer` — matching sklearn's category ordering, sparse/dense output, and `handle_unknown` behavior structurally. *(gate: exact structural parity)*

### Feature Extraction (FEAT)

- [ ] **FEAT-01**: User can fit/transform text vectorizers — `CountVectorizer` and `TfidfVectorizer` — producing a vocabulary and document-term matrix matching sklearn (vocabulary exact; Tfidf weights ≤1e-5 under the same `norm`/`sublinear_tf`/`smooth_idf` settings). *(gate: exact vocabulary + ≤1e-5 weights)*

### Model Selection (MODSEL)

- [ ] **MODSEL-01**: User can split data — `train_test_split`, `KFold`, `StratifiedKFold` — with a recorded MT19937-host-match decision (so `shuffle=True` reproducibility is either bit-for-bit vs sklearn or property-gated, decision documented), gated structurally (fold sizes, stratification balance, no leakage). *(gate: structural + recorded RNG decision)*
- [ ] **MODSEL-02**: `GridSearchCV` / `RandomizedSearchCV` operate correctly over mlrs estimators via sklearn delegation (the cuML passthrough pattern), verified to fit/score/select across an mlrs estimator's parameter grid. *(gate: behavioral / integration)*

### Genetic / Symbolic Regression (GEN)

- [ ] **GEN-01**: User can fit `SymbolicRegressor` (a `program_eval` device prim + host evolutionary loop) with a configurable function set, population, and generations, passing a property gate vs `gplearn.SymbolicRegressor` (R²-within-band + valid program trees + internal same-seed reproducibility — never element-wise expression match). *(gate: property + R² band)*
- [ ] **GEN-02**: User can fit `SymbolicClassifier` (sigmoid-wrapped programs) passing a predictive-quality band + internal seed-reproducibility gate vs `gplearn.SymbolicClassifier`. *(gate: property + band)*
- [ ] **GEN-03**: User can fit `SymbolicTransformer` to generate engineered features, gated structurally (program validity, output shape, internal seed-reproducibility) and on downstream predictive lift. *(gate: structural + property)*

### cuml.accel Drop-in (pure Python — build LAST)

- [ ] **ACCEL-01**: User can `mlrs.accel.install()` to transparently proxy `sklearn`/`umap`/`hdbscan` estimator imports to the mlrs equivalents via a `sys.meta_path` `MetaPathFinder` + `AccelModule.__getattr__`, with a caller-module exclusion list and a detect-and-warn if the target package was already imported; `uninstall()` restores the originals. *(gate: behavioral / integration; zero Rust)*
- [ ] **ACCEL-02**: The accel layer is **fail-closed**: any proxied estimator with an unsupported parameter/config falls back to CPU sklearn (never a silent wrong result), and fitted-attribute names/shapes mirror sklearn exactly — verified by a fallback matrix covering every proxied estimator × an unsupported config. *(gate: fallback-matrix + fitted-attribute parity)*

## Future Requirements

Deferred to a future milestone. Tracked but not in the v4.0 roadmap.

### Tree-stack extensions
- **TREE-FUT-01**: Interventional / feature-perturbation TreeSHAP with background data (path-dependent TreeSHAP ships first in v4.0).
- **FIL-FUT-01**: External tree-model ingestion (Treelite / XGBoost / LightGBM) into FIL — external-model import is its own milestone.

### Kernel SVM
- **SVM-FUT-01**: Kernel `SVC` / `SVR` via the SMO solver — the hardest solver to make GPU-friendly; deferred past v4.0 (linear SVM shipped in v2).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Kernel SVM (SVC/SVR via SMO) | SMO is the hardest solver to make GPU/cpu-MLIR-friendly; deferred past v4.0. Linear SVM (LinearSVC/SVR) already shipped in v2. |
| Multi-GPU / distributed (Dask, NCCL/UCX, `*_mg`) | Single-device first; distribution is a separate milestone. |
| External tree-model import (Treelite/XGBoost/LightGBM → FIL) | mlrs owns its own tree format; foreign-model ingestion is a later milestone. |
| Interventional TreeSHAP with background data | Path-dependent TreeSHAP ships first; interventional is a graded extension. |
| Bit-exact reproduction of cuML internals | Goal is numerical agreement with scikit-learn (≤1e-5) / appropriate gate, not kernel-for-kernel cuML parity. |
| Half-precision (f16/bf16) validated paths | Infrastructure may allow it but not a near-term deliverable. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TREE-01 | Phase 17 | Pending |
| TREE-02 | Phase 18 | Pending |
| RF-01 | Phase 19 | Pending |
| RF-02 | Phase 19 | Pending |
| RF-03 | Phase 19 | Pending |
| FIL-01 | Phase 20 | Pending |
| SHAP-01 | Phase 21 | Pending |
| SHAP-02 | Phase 23 | Pending |
| SHAP-03 | Phase 23 | Pending |
| ARIMA-01 | Phase 22 | Pending |
| ARIMA-02 | Phase 22 | Pending |
| ARIMA-03 | Phase 22 | Pending |
| METR-01 | Phase 24 | Pending |
| METR-02 | Phase 24 | Pending |
| METR-03 | Phase 24 | Pending |
| PREP-01 | Phase 24 | Pending |
| PREP-02 | Phase 24 | Pending |
| FEAT-01 | Phase 24 | Pending |
| MODSEL-01 | Phase 24 | Pending |
| MODSEL-02 | Phase 24 | Pending |
| GEN-01 | Phase 25 | Pending |
| GEN-02 | Phase 25 | Pending |
| GEN-03 | Phase 25 | Pending |
| ACCEL-01 | Phase 26 | Pending |
| ACCEL-02 | Phase 26 | Pending |

**Coverage:**
- v4.0 requirements: 25 total
- Mapped to phases: 25 ✓ (all v1 requirements mapped to exactly one phase; no orphans, no duplicates)
- Unmapped: 0 ✓

**Phase → requirement rollup:**
- Phase 17: TREE-01 · Phase 18: TREE-02 · Phase 19: RF-01/02/03 · Phase 20: FIL-01 · Phase 21: SHAP-01
- Phase 22: ARIMA-01/02/03 · Phase 23: SHAP-02/03 · Phase 24: METR-01/02/03 + PREP-01/02 + FEAT-01 + MODSEL-01/02 · Phase 25: GEN-01/02/03 · Phase 26: ACCEL-01/02

---
*Requirements defined: 2026-06-26 for milestone v4.0*
*Last updated: 2026-06-26 — traceability filled by roadmapper (Phases 17–26, 25/25 mapped)*
