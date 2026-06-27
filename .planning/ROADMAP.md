# Roadmap: mlrs — cuML in Rust

## Milestones

- ✅ **v1.0 Core ML Library** — Phases 1–6 (shipped 2026-06-14) → [archive](milestones/v1.0-ROADMAP.md)
- ✅ **v2.0 Breadth Sweep** — Phases 7–11 (shipped 2026-06-22) → [archive](milestones/v2.0-ROADMAP.md)
- ✅ **v3.0 Manifold Algorithms & Rust-Native API** — Phases 12–16 (shipped 2026-06-26) → [archive](milestones/v3.0-ROADMAP.md)
- 🚧 **v4.0 Tree Ensembles, Time-Series & Full-Surface Completion** — Phases 17–26 (in progress) — RandomForest→FIL→TreeSHAP (spike-gated), ARIMA/AutoARIMA/SARIMA, Kernel/Permutation SHAP, sklearn-utility surface, genetic/symbolic regression, cuml.accel

## Overview

All three shipped milestones grew one sklearn-compatible ML library on a single CubeCL-generic codebase (cpu f64 + rocm f32 gate, scikit-learn ≤1e-5 oracle): v1.0 stood up the foundation + 12 estimators, v2.0 swept 18 more across five families, and v3.0 added the UMAP + HDBSCAN manifold/clustering pair on a shared KNN-graph primitive plus a Rust-native builder/typestate API and pure-Python sklearn shim. Full per-phase detail for each shipped milestone lives in its archive (linked above). **v4.0 closes out the remaining cuML algorithm surface** (everything except kernel SVM/SMO): the tree-ensemble family (RandomForest → FIL → TreeSHAP, gated by a feasibility spike), time-series (ARIMA/AutoARIMA/SARIMA), model-agnostic explainers (Kernel/Permutation SHAP), the genetic/symbolic subsystem, the sklearn-utility surface, and the transparent `cuml.accel` drop-in. Zero new compute dependencies; test-only oracles broaden to `statsmodels`/`shap`/`gplearn`. Same backend gate (cpu f64 + rocm f32, f64-on-rocm skips-with-log).

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)
- Phase numbering is continuous across milestones (never restarts); v4.0 continues from Phase 17.

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

<details>
<summary>✅ v3.0 Manifold Algorithms & Rust-Native API (Phases 12–16) — SHIPPED 2026-06-26 — 34 plans, UMAP + HDBSCAN + builder/typestate retrofit</summary>

UMAP + HDBSCAN on a single shared, multi-metric KNN-graph primitive (primitive-first, standalone-gated before either consumer), plus a Rust-native builder/typestate API additively retrofitted across the full 32-estimator surface and a pure-Python sklearn shim. Zero new compute dependencies. Oracle broadened to umap-learn 0.5.12 (property gate for UMAP's stochastic SGD layout; ≤1e-5 value gates for the deterministic stages); HDBSCAN keeps an exact-label hard gate. Same gate as v1/v2 (cpu f64 + rocm f32, f64-on-rocm skips-with-log).

- [x] Phase 12: Builder + Typestate Convention Foundation (4/4 plans) — completed 2026-06-23
- [x] Phase 13: KNN-Graph Primitive (feasibility keystone) (3/3 plans) — completed 2026-06-23
- [x] Phase 14: UMAP (7/7 plans) — completed 2026-06-24
- [x] Phase 15: HDBSCAN (7/7 plans) — completed 2026-06-24
- [x] Phase 16: Builder Retrofit Sweep + Shim Coverage (13/13 plans) — completed 2026-06-24

Full phase detail, plans, and per-plan notes: [milestones/v3.0-ROADMAP.md](milestones/v3.0-ROADMAP.md)

</details>

### 🚧 v4.0 Tree Ensembles, Time-Series & Full-Surface Completion (Phases 17–26) — In Progress

**Milestone Goal:** Close out the remaining cuML algorithm surface (everything except kernel SVM/SMO). Land the tree-ensemble family (RandomForest → FIL → TreeSHAP, gated by a feasibility spike), time-series (ARIMA/AutoARIMA/SARIMA), model-agnostic explainers (Kernel/Permutation SHAP), the genetic/symbolic subsystem, the sklearn-utility surface, and the transparent `cuml.accel` drop-in.

**Dependency shape (load-bearing — honor the order):**

- **Phase 17 (feasibility spike) GATES the tree chain.** The hard serial chain **17 → 18 → 19 → 20 → 21** cannot be parallelized: spike GO → tree prims + DecisionTree → RandomForest + importances → FIL (needs the node format + a forest) → TreeSHAP (needs FIL's tree store). Nothing tree-family is committed until Phase 17 returns **GO**.
- **Phases 22–25 are spike-INDEPENDENT and parallel-eligible** (no dependency on the tree spike result): ARIMA/AutoARIMA/SARIMA, model-agnostic Kernel+Permutation SHAP, the sklearn-utility surface, and genetic/symbolic regression. Each still obeys primitive-first internally.
- **Phase 26 (cuml.accel) is LAST** — its proxy override table is only complete once the entire estimator surface (v1–v3 + all v4 estimators) exists.

**Gate regimes (v4.0 spans all four):** value ≤1e-5 (FIL exact-vs-host-traversal, TreeSHAP, metrics, scalers, Tfidf weights), exact/structural (DecisionTree core on fixed indices, encoders/splitters/vocabulary), property+band (RandomForest, Kernel/Permutation SHAP, symbolic — SplitMix64 ≠ MT19937), stats-band (ARIMA — oracle is `statsmodels.tsa`, not sklearn). The **two-tier stochastic gate** (deterministic injected-fixed-index core tier + ensemble/predictive-quality band tier) is established in the Phase-17 spike as the milestone-wide convention.

- [ ] **Phase 17: RandomForest GPU Histogram/Split Feasibility Spike (GATING)** — Prove or refute that a single-owner GATHER histogram/split lowers and is tractable under cpu-MLIR; deliver a GO/ADJUST/ABORT verdict that gates the entire tree chain
- [ ] **Phase 18: Tree Primitives + DecisionTree Core** — Promote the spike's kernel probes to standalone-validated `quantiles`/`tree_hist`/`best_split`/`node_partition` prims + an oracle-gated DecisionTree core
- [ ] **Phase 19: RandomForestClassifier + RandomForestRegressor** — GPU-constructed forests with the full sklearn hyperparameter surface, `feature_importances_`, and `oob_score_`, under the two-tier gate
- [ ] **Phase 20: FIL — Batched Forest Inference** — Iterative `node_id` device traversal over the mlrs node store, exactly equal to a host reference walk
- [ ] **Phase 21: TreeSHAP** — Path-dependent TreeSHAP for a fitted mlrs forest, ≤1e-5 vs `shap.TreeExplainer` on mlrs's own tree + exact additive-efficiency
- [ ] **Phase 22: ARIMA / AutoARIMA / Seasonal (SARIMAX)** *(parallel-eligible)* — Batched Kalman + batched L-BFGS; fit/forecast + order search + seasonal/exog, stats-band gated vs `statsmodels`
- [ ] **Phase 23: Kernel + Permutation SHAP** *(parallel-eligible)* — Model-agnostic explainers for any fitted mlrs estimator; exact additive-efficiency + convergence band vs `shap`
- [ ] **Phase 24: sklearn-Utility Surface** *(parallel-eligible)* — metrics / preprocessing / feature_extraction (Count+Tfidf) / model_selection (split + GridSearch passthrough)
- [ ] **Phase 25: Genetic / Symbolic Regression** *(parallel-eligible)* — `program_eval` device prim + host evolutionary loop; SymbolicRegressor/Classifier/Transformer, property+band vs `gplearn`
- [ ] **Phase 26: cuml.accel Drop-in (pure Python — LAST)** — Transparent `sys.meta_path` proxy of sklearn/umap/hdbscan to mlrs, fail-closed with CPU fallback

## Phase Details

### Phase 17: RandomForest GPU Histogram/Split Feasibility Spike (GATING)

**Goal**: Prove (or refute) that GPU tree construction — single-owner GATHER histogram, relabel-partition, seed-from-first split-find — lowers and is tractable under cpu-MLIR, delivering an explicit GO/ADJUST/ABORT verdict that gates the entire tree chain (RF → FIL → TreeSHAP). Models the v3.0 Phase 13 KNN-graph keystone spike.
**Depends on**: Phase 16 (v3.0 complete) — first v4.0 phase; gates Phases 18–21
**Requirements**: TREE-01
**Gate**: spike verdict (A1–A5 evaluated) + VALUE on the fixed-index tree
**Success Criteria** (what must be TRUE):

  1. GATHER-histogram, relabel-partition, and seed-from-first split-find kernels standalone-launch on cpu(f64) + rocm(f32) with no SharedMemory, no atomics, no `F::INFINITY` init
  2. A single decision tree built on injected fixed bootstrap/feature indices VALUE-matches `sklearn.tree.DecisionTree*` (split thresholds + leaf values)
  3. The `SparseTreeNode { colid, threshold, left_child, value }` format contract is finalized (right child = `left_child + 1`)
  4. A per-tree cost benchmark is recorded and abort signals A1–A5 are each evaluated
  5. An explicit GO / ADJUST / ABORT verdict is delivered and the two-tier stochastic-gate convention is documented as the milestone-wide standard

**Plans**: 5 plans
**Wave 1**

- [ ] 17-01-PLAN.md — Wave-0 oracle foundation: gen_decision_tree_clf(gini)/reg(squared_error) generators + committed sklearn .npz fixtures (standard + adversarial, f32+f64)
- [ ] 17-02-PLAN.md — Three cpu-MLIR-safe kernels (GATHER histogram, seed-from-first split-find, relabel-partition) + SparseTreeNode + host build loop + standalone-launch VALUE probes (SC-1, A1, A4)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 17-03-PLAN.md — Tier-1 witness: single tree VALUE-matches sklearn DecisionTree clf+reg + adversarial; SparseTreeNode contract validated (SC-2, SC-3, A5)
- [ ] 17-04-PLAN.md — Per-tree cost benchmark at 64 vs 128 bins + scaling sweep (SC-4, A3)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 17-05-PLAN.md — VERDICT.md (A1–A5 + GO/ADJUST/ABORT) + two-tier convention + spike wrap-up; blocking human gate (SC-4, SC-5)

**Research**: COMPLETE — 17-RESEARCH.md + 17-PATTERNS.md; `Skill("spike-findings-mlrs")` carries the proven GATHER op-set and the 002-A (loud) / 002-B (silent) cpu-MLIR landmines

### Phase 18: Tree Primitives + DecisionTree Core

**Goal**: Promote the spike's kernel probes to production primitives with full prim contracts and deliver an oracle-gated DecisionTree core (level-wise host loop) — primitive-first, before RandomForest assembles N of them.
**Depends on**: Phase 17 (GO verdict)
**Requirements**: TREE-02
**Gate**: exact/structural on fixed-index tree; ≤1e-5 leaf values (f64)
**Success Criteria** (what must be TRUE):

  1. `quantiles`, `tree_hist`, `best_split`, `node_partition` are standalone-validated primitives in mlrs-backend
  2. A `DecisionTreeClassifier` core matches `sklearn.tree.DecisionTreeClassifier` on injected fixed indices (exact structure, ≤1e-5 leaf values f64)
  3. A `DecisionTreeRegressor` core matches `sklearn.tree.DecisionTreeRegressor` on injected fixed indices
  4. A build-failing frontier-only-histogram PoolStats memory gate is green

**Plans**: TBD
**Research**: Standard primitive-first pattern (spike delivers the kernel shapes) — skip research-phase

### Phase 19: RandomForestClassifier + RandomForestRegressor

**Goal**: Users can fit GPU-constructed random forests with the full sklearn hyperparameter surface, importances, and OOB score — the first phase the two-tier property+band gate applies against `sklearn.ensemble.RandomForest*`.
**Depends on**: Phase 18
**Requirements**: RF-01, RF-02, RF-03
**Gate**: two-tier property+band (deterministic injected-index single-tree match + ensemble accuracy/R² band)
**Success Criteria** (what must be TRUE):

  1. User can fit `RandomForestClassifier` (`fit`/`predict`/`predict_proba`) with sklearn-named hyperparameters/defaults, passing the two-tier gate (injected-index single-tree match + accuracy-within-band vs sklearn)
  2. User can fit `RandomForestRegressor` (`fit`/`predict`) with `squared_error`/`absolute_error`, passing the R²-within-band tier vs sklearn
  3. A fitted forest exposes `feature_importances_` (impurity-based, summing to 1) and `oob_score_` (when `bootstrap=True, oob_score=True`), each structurally/band-gated vs sklearn
  4. Both estimators are PyO3-exposed with a pure-Python sklearn shim

**Plans**: TBD
**Research**: Two-tier gate convention established in Phase 17 — skip research-phase

### Phase 20: FIL — Batched Forest Inference

**Goal**: Users can run batched forest inference over the mlrs node store via an iterative (non-recursive) device traversal — the one tree-stack gate that is exact.
**Depends on**: Phase 18 (node format) + Phase 19 (a fitted forest to traverse)
**Requirements**: FIL-01
**Gate**: exact vs host reference traversal of the identical node arrays
**Success Criteria** (what must be TRUE):

  1. User can run batched `predict`/`predict_proba` over the mlrs node store via an iterative `node_id` device traversal (no recursion)
  2. Output is **exactly equal** to a host reference walk of the identical node arrays
  3. A row-streaming PoolStats memory gate is green (output rows streamed, no recursive walk)

**Plans**: TBD
**Research**: Deterministic traversal with a clear oracle — skip research-phase

### Phase 21: TreeSHAP

**Goal**: Users can compute exact path-dependent TreeSHAP explanations for a fitted mlrs forest (deterministic Lundberg algorithm), gated only against `shap.TreeExplainer` fed mlrs's own tree.
**Depends on**: Phase 20 (FIL / tree store)
**Requirements**: SHAP-01
**Gate**: ≤1e-5 + exact additive-efficiency invariant
**Success Criteria** (what must be TRUE):

  1. User can compute path-dependent TreeSHAP values for a fitted mlrs forest
  2. Values match `shap.TreeExplainer` (fed mlrs's *own* tree, NOT sklearn's forest) to ≤1e-5
  3. The additive-efficiency invariant holds exactly (Σφ + base = prediction)
  4. Values cross-check against a brute-force exact Shapley enumeration on small hand-built trees

**Plans**: TBD
**Research**: Deterministic algorithm with a clear oracle — skip research-phase

### Phase 22: ARIMA / AutoARIMA / Seasonal (SARIMAX)

**Goal**: Users can fit and forecast (seasonal) ARIMA models via a batched Kalman filter + batched L-BFGS with the Jones/PACF stationarity transform — spike-independent, parallel-eligible with the tree chain.
**Depends on**: Phase 16 (v3.0 complete) — spike-independent; **parallel-eligible** with Phases 17–21
**Requirements**: ARIMA-01, ARIMA-02, ARIMA-03
**Gate**: stats band (log-likelihood / forecast) + exact selected-order for AutoARIMA — oracle is `statsmodels.tsa`, NOT sklearn
**Success Criteria** (what must be TRUE):

  1. The batched Kalman primitive is standalone-validated (f64 likelihood accumulation even on the f32 rocm path, per-series convergence flags) before the estimators consume it
  2. User can fit `ARIMA(order=(p,d,q))` and `forecast(steps)`, gated on log-likelihood band + forecast band + known-coefficient recovery vs `statsmodels.tsa.arima.model.ARIMA` (Jones/PACF transform)
  3. User can fit `AutoARIMA` recovering the correct selected `(p,d,q)` on synthetic series with known structure (KPSS/IC order search)
  4. User can fit seasonal `ARIMA(order, seasonal_order=(P,D,Q,s))` with optional `exog`, gated on likelihood + forecast band vs `statsmodels` SARIMAX

**Plans**: TBD
**Research**: NEEDS `--research-phase` — Jones/PACF transform, Joseph-form stable Kalman, batched L-BFGS convergence-flag design, and `statsmodels` oracle matching are domain-specific (not covered by existing project patterns)

### Phase 23: Kernel + Permutation SHAP

**Goal**: Users can compute model-agnostic SHAP explanations for any fitted mlrs estimator (reusing the existing linear solver / predict path) — spike-independent, parallel-eligible.
**Depends on**: Phase 16 (needs only a fitted estimator) — spike-independent; **parallel-eligible**
**Requirements**: SHAP-02, SHAP-03
**Gate**: exact additive-efficiency invariant + convergence band vs `shap`
**Success Criteria** (what must be TRUE):

  1. User can compute **Kernel SHAP** values (weighted-lstsq over sampled coalitions) for any fitted mlrs estimator, satisfying additive-efficiency exactly, with a convergence band vs `shap.KernelExplainer` and a coalition-block-streaming PoolStats gate
  2. User can compute **Permutation SHAP** values for any fitted mlrs estimator, satisfying additive-efficiency exactly and matching `shap.PermutationExplainer` within a convergence band
  3. Both cross-check against a brute-force exact Shapley enumeration on small `n`

**Plans**: TBD
**Research**: SHAP axioms and weighted-lstsq are well-documented — skip research-phase

### Phase 24: sklearn-Utility Surface (metrics / preprocessing / feature_extraction / model_selection)

**Goal**: Users have the deterministic sklearn-utility surface — classification/regression/ranking metrics, preprocessing transformers, text vectorizers, and data splitters/search — with mandatory degenerate fixtures. Spike-independent, parallel-eligible.
**Depends on**: Phase 16 (v3.0 complete) — spike-independent; **parallel-eligible**
**Requirements**: METR-01, METR-02, METR-03, PREP-01, PREP-02, FEAT-01, MODSEL-01, MODSEL-02
**Gate**: ≤1e-5 (metrics, scalers, Tfidf weights) + exact structural (encoders/imputers, splitters, vocabulary) + behavioral integration (GridSearch passthrough); MT19937-host-match decision recorded
**Success Criteria** (what must be TRUE):

  1. User can call classification/regression/ranking metrics (`accuracy_score`, `confusion_matrix`, `precision`/`recall`/`f1`, `r2_score`, `mean_squared_error`, `mean_absolute_error`, `roc_auc_score`, `log_loss`, `precision_recall_curve`) matching sklearn exactly/≤1e-5, with mandatory degenerate fixtures (empty class, single sample, constant target, zero-division)
  2. User can fit/transform scalers (`StandardScaler`/`MinMaxScaler`/`MaxAbsScaler`/`RobustScaler`/`Normalizer`/`Binarizer`) and encoders/imputers (`OneHotEncoder`/`OrdinalEncoder`/`LabelEncoder`/`LabelBinarizer`/`SimpleImputer`) matching sklearn (≤1e-5 / structural), with column statistics learned only in `fit` and applied in `transform`
  3. User can fit/transform `CountVectorizer` and `TfidfVectorizer` producing an exact vocabulary + ≤1e-5 Tfidf weights under the same `norm`/`sublinear_tf`/`smooth_idf` settings
  4. User can split data (`train_test_split`, `KFold`, `StratifiedKFold`) gated structurally with a recorded MT19937-host-match decision, and run `GridSearchCV`/`RandomizedSearchCV` over mlrs estimators via the sklearn-delegation passthrough

**Plans**: TBD
**Research**: Deterministic functions with exhaustively-documented sklearn contracts — skip research-phase

### Phase 25: Genetic / Symbolic Regression

**Goal**: Users can fit symbolic regression/classification/transformer estimators via a `program_eval` device primitive + a host evolutionary loop — property-gated vs `gplearn`, never element-wise expression match. Spike-independent, parallel-eligible.
**Depends on**: Phase 16 (v3.0 complete) — spike-independent; **parallel-eligible**
**Requirements**: GEN-01, GEN-02, GEN-03
**Gate**: property + R²/predictive band + structural (program validity, internal same-seed reproducibility)
**Success Criteria** (what must be TRUE):

  1. The `program_eval` device primitive is standalone-validated before the evolutionary host loop consumes it
  2. User can fit `SymbolicRegressor` (configurable function set/population/generations) passing an R²-within-band gate vs `gplearn.SymbolicRegressor` + valid program trees + internal same-seed reproducibility
  3. User can fit `SymbolicClassifier` (sigmoid-wrapped programs) passing a predictive-quality band + seed-reproducibility gate vs `gplearn.SymbolicClassifier`
  4. User can fit `SymbolicTransformer` generating engineered features, gated on program validity, output shape, seed-reproducibility, and downstream predictive lift

**Plans**: TBD
**Research**: `gplearn` API is stable and the host-evolve/device-evaluate pattern is established — skip research-phase

### Phase 26: cuml.accel Drop-in (pure Python — LAST)

**Goal**: Users can transparently accelerate existing sklearn/umap/hdbscan code by proxying imports to the mlrs equivalents, fail-closed — landed last so the proxy table covers the entire estimator surface. Zero Rust changes.
**Depends on**: Phases 17–25 (full v1–v4 estimator surface must exist)
**Requirements**: ACCEL-01, ACCEL-02
**Gate**: behavioral/integration — fallback matrix + fitted-attribute parity (zero Rust)
**Success Criteria** (what must be TRUE):

  1. User can `mlrs.accel.install()` to transparently proxy `sklearn`/`umap`/`hdbscan` estimator imports to the mlrs equivalents via a `sys.meta_path` `MetaPathFinder` + `AccelModule.__getattr__`, with a caller-module exclusion list and a detect-and-warn if the target package was already imported; `uninstall()` restores the originals
  2. The accel layer is **fail-closed**: any proxied estimator with an unsupported parameter/config falls back to CPU sklearn (never a silent wrong result)
  3. Fitted-attribute names/shapes mirror sklearn exactly, verified by a fallback matrix covering every proxied estimator × an unsupported config

**Plans**: TBD
**Research**: `importlib.abc` MetaPathFinder is standard-library Python; cuML's own `accel/` is the direct in-tree reference — skip research-phase

## Progress

**Execution Order:**
Tree chain is serial: 17 → 18 → 19 → 20 → 21. Phases 22–25 are parallel-eligible (spike-independent). Phase 26 is last.

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
| 12. Builder + Typestate Convention Foundation | v3.0 | 4/4 | Complete | 2026-06-23 |
| 13. KNN-Graph Primitive (feasibility keystone) | v3.0 | 3/3 | Complete | 2026-06-23 |
| 14. UMAP | v3.0 | 7/7 | Complete | 2026-06-24 |
| 15. HDBSCAN | v3.0 | 7/7 | Complete | 2026-06-24 |
| 16. Builder Retrofit Sweep + Shim Coverage | v3.0 | 13/13 | Complete | 2026-06-24 |
| 17. RandomForest GPU Histogram/Split Feasibility Spike (GATING) | v4.0 | 0/TBD | Not started | - |
| 18. Tree Primitives + DecisionTree Core | v4.0 | 0/TBD | Not started | - |
| 19. RandomForestClassifier + RandomForestRegressor | v4.0 | 0/TBD | Not started | - |
| 20. FIL — Batched Forest Inference | v4.0 | 0/TBD | Not started | - |
| 21. TreeSHAP | v4.0 | 0/TBD | Not started | - |
| 22. ARIMA / AutoARIMA / Seasonal (SARIMAX) | v4.0 | 0/TBD | Not started | - |
| 23. Kernel + Permutation SHAP | v4.0 | 0/TBD | Not started | - |
| 24. sklearn-Utility Surface | v4.0 | 0/TBD | Not started | - |
| 25. Genetic / Symbolic Regression | v4.0 | 0/TBD | Not started | - |
| 26. cuml.accel Drop-in | v4.0 | 0/TBD | Not started | - |
