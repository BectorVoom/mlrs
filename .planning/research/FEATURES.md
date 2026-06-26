# Feature Research

**Domain:** ML algorithm library (Rust rewrite of RAPIDS cuML) — v4.0 final-surface completion
**Researched:** 2026-06-26
**Confidence:** HIGH (cuML reference source + sklearn/gplearn/shap API are read-only and in-tree; stochastic-gate precedent already set in v2/v3)

> Scope note: this file covers ONLY the v4.0 NEW features. The existing 32 estimators, KNN-graph
> prim, PyO3 wheels, builder/typestate API and sklearn shim are settled and not re-researched.
> Per-feature **oracle + gate** is the load-bearing output (downstream = REQ-IDs grouped by
> category + roadmap ordering). The recurring theme: **RandomForest, all sampling-SHAP, and
> symbolic regression are stochastic and will NOT match element-wise** — each needs a structural/
> property contract, exactly as RandomProjection (D-12) and UMAP-03 established.

---

## Oracle & Gate Summary (the headline table)

| Feature group | Oracle | Gate type | Why this gate |
|---|---|---|---|
| RandomForest (clf/reg) | `sklearn.ensemble.RandomForest*` | **Property/predictive-quality + structural** | mlrs SplitMix64 ≠ NumPy RNG → bootstrap rows & feature subsets differ → trees differ → no element-wise match. Contract = test accuracy/R² within margin of sklearn + structural tree invariants + a fully-deterministic single-tree config checked vs `DecisionTree`. |
| FIL (forest inference) | host reference traversal of the **same** mlrs trees | **Exact (≤1e-5 / bit-exact labels)** | Inference is deterministic given fixed trees → device traversal must equal a CPU reference walk of the identical node arrays. This is the one tree-stack gate that is exact. |
| TreeSHAP | `shap.TreeExplainer` on the same forest | **Exact ≤1e-5 + additive-efficiency invariant** | Lundberg path-dependent TreeSHAP is a deterministic exact algorithm → matches `shap` to ≤1e-5 AND `sum(shap)+base == model_output`. |
| ARIMA / AutoARIMA | `statsmodels.tsa` (SARIMAX) / `pmdarima` (auto) | **Value-band on params + tighter band on forecasts** (NOT 1e-5) | ML estimate via L-BFGS → params agree to an optimizer band; log-likelihood & forecasts agree tighter. AutoARIMA = exact selected order on clean synthetic series. |
| Kernel SHAP | `shap.KernelExplainer` | **Additive-efficiency invariant (exact) + sampling band vs shap** | Weighted-lstsq over sampled coalitions → stochastic value, but `sum(shap)+base == f(x)` holds exactly. |
| Permutation SHAP | `shap.PermutationExplainer` | **Additive-efficiency invariant + band** | Permutation sampling is stochastic; efficiency property is exact. |
| cuml.accel | direct mlrs estimator + sklearn fallback | **Behavioral/equivalence** (no numeric oracle of its own) | Proxy must (a) reproduce the wrapped mlrs estimator's output and (b) fall back to sklearn on unsupported params. |
| metrics | `sklearn.metrics` | **≤1e-5 (float scores) / exact (integer/label scores)** | Deterministic functions. `confusion_matrix`, `accuracy` exact; `r2`, `roc_auc`, `log_loss` ≤1e-5. |
| preprocessing | `sklearn.preprocessing` | **≤1e-5 (transform output)** | Deterministic transforms; fitted stats (mean/var/min/max/categories) match exactly → transformed matrix ≤1e-5. |
| model_selection | `sklearn.model_selection` | **Structural (split proportions exact, reproducible) — NOT index-identical** | Shuffle RNG differs from NumPy → can't reproduce exact indices; gate = correct sizes, no leakage, strat proportions exact, same-seed reproducibility within mlrs. |
| feature_extraction (tfidf) | `sklearn.feature_extraction.text` | **≤1e-5 on the tfidf matrix (vocabulary exact)** | Deterministic given a fixed tokenizer/vocab. |
| Symbolic regression | `gplearn.SymbolicRegressor` | **Property/predictive-quality + structural** | GP is heavily stochastic; gate = test fitness ≥ gplearn − margin, valid program trees, seed-reproducibility within mlrs, parsimony respected. |

---

## Feature Landscape

### Table Stakes (users expect these in a "complete cuML-parity" library)

| Feature | Why Expected | Complexity | Notes |
|---|---|---|---|
| RandomForestClassifier / Regressor | The single highest-demand missing estimator; "no RF" = library feels incomplete | **HIGH** (spike-gated) | GPU histogram/split under cpu-MLIR no-SharedMemory/no-atomics is the make-or-break; quantile binning (`n_bins`), per-tree bootstrap, feature subsampling. Keystone — unblocks FIL→TreeSHAP. |
| FIL — batched tree inference | A forest you can't predict from fast is useless; predict path must traverse trees in bulk | **MEDIUM** | Deterministic; pure GATHER traversal over node arrays. Defines the canonical tree format the rest of the stack reads. |
| ARIMA(p,d,q) + forecast | Table-stakes for any time-series story | **HIGH** | Batched Kalman-filter log-likelihood + **batched** L-BFGS (mlrs has L-BFGS but not batched). Differencing (d), intercept (mu/k), exog. |
| metrics: accuracy / r2 / confusion_matrix / mse / mae | Every classifier/regressor user calls `.score`-adjacent metrics | **LOW** | Reductions only; deterministic; ≤1e-5 / exact. Highest-value first. |
| preprocessing: StandardScaler / MinMaxScaler / MaxAbsScaler / RobustScaler / Normalizer | Universal pipeline front-ends; expected before any estimator | **LOW–MEDIUM** | Fit = column stats (mean/var/min/max/median/IQR); transform = elementwise. Deterministic ≤1e-5. PartialFit precedent exists (v2 IncrementalPCA). |
| preprocessing encoders: OneHotEncoder / OrdinalEncoder / LabelEncoder / LabelBinarizer | Categorical handling expected for real datasets | **MEDIUM** | Host-heavy (category discovery); device-light. Exact category mapping. |
| model_selection: train_test_split / KFold / StratifiedKFold | First call in nearly every ML script | **LOW** | Mostly host/indexing. Structural gate (see oracle table). cuML delegates `train_test_split` semantics to sklearn — mlrs can mirror. |
| Kernel SHAP | The default model-agnostic explainer users reach for | **MEDIUM–HIGH** | Background dataset + coalition sampling + weighted lstsq. Reuses GEMM/lstsq prims. |

### Differentiators (align with Core Value: correctness + single generic backend)

| Feature | Value Proposition | Complexity | Notes |
|---|---|---|---|
| TreeSHAP (exact) | Exact, fast tree explanations matching `shap` to ≤1e-5 — the rare stochastic-domain feature with an EXACT gate | **HIGH** | Path-dependent Lundberg algorithm over FIL trees. Strongest correctness story of the milestone. Depends on FIL/tree format. |
| AutoARIMA order search | "Just fit my series" — automatic (p,d,q)(P,D,Q,s) via IC + stationarity/seasonality tests | **HIGH** | KPSS (d) + seasonal test (D) + grid/stepwise over orders, scored by aic/aicc/bic. Built on ARIMA. |
| cuml.accel transparent drop-in | Zero-code-change acceleration of existing sklearn/umap/hdbscan scripts → proxies to the 32 mlrs estimators | **MEDIUM** (mostly Python plumbing) | Import-hook/module-swap + per-estimator overrides + CPU fallback. No new algorithm; high adoption value. Reverses the prior Out-of-Scope decision. |
| Permutation SHAP | Cheaper model-agnostic explainer; complements Kernel SHAP | **MEDIUM** | Forward/reverse permutation passes; shares SHAPBase plumbing with Kernel SHAP. |
| Symbolic / genetic regression | Interpretable closed-form models; a capability sklearn itself lacks (parity with cuML's `genetic`) | **HIGH** | Population of program trees, tournament selection, crossover/mutation. Heavily stochastic. Oracle = `gplearn`. |
| feature_extraction: TfidfVectorizer / CountVectorizer | Enables text pipelines end-to-end | **MEDIUM** | Host tokenization + sparse counts; device-light. Lower priority than numeric preprocessing. |
| RF OOB score, feature_importances_ | Expected RF attributes that add diagnostic value | **MEDIUM** | `oob_score_`, `feature_importances_` — structural-gated (within band vs sklearn). |

### Anti-Features (commonly requested, deliberately avoid or bound tightly)

| Feature | Why Requested | Why Problematic | Alternative |
|---|---|---|---|
| Element-wise 1e-5 match of RF predictions/trees vs sklearn | "Match sklearn like everything else" | Different RNG → different bootstrap/feature subsets → fundamentally different forests; chasing it is impossible | Predictive-quality band + deterministic single-tree exact check + structural invariants |
| Bit-exact match of Kernel/Permutation SHAP vs `shap` | "SHAP values should be identical" | Sampling-based → stochastic; only the additive-efficiency property is exact | Efficiency invariant (exact) + band vs `shap` library |
| cuml.accel that silently produces wrong results on unsupported params | Maximize "it just works" coverage | Silent divergence destroys trust; cuML's own design fights this with `UnsupportedOnGPU`→CPU fallback | Explicit unsupported-config detection → transparent sklearn CPU fallback, never silent approximation |
| Treelite/XGBoost/LightGBM model ingestion into FIL | "Load my existing boosted models" | Pulls in Treelite + a foreign serialization format; huge surface, no oracle in mlrs's sklearn-only world | FIL consumes ONLY mlrs's own forest format in v4.0; external-model import is a later milestone |
| GridSearchCV / RandomizedSearchCV as native device code | "Full model_selection parity" | Search is orchestration, not compute; cuML itself just `__getattr__`-delegates to sklearn | Delegate to sklearn (passthrough); ship only the data-splitters natively |
| Interventional/feature-perturbation TreeSHAP with background data | "Match shap's newer default" | Needs background-dataset marginalization; doubles the algorithm | Ship path-dependent (tree-stats) TreeSHAP first; interventional is a follow-up |
| ARIMA via sklearn oracle | PROJECT.md lists "sklearn for ARIMA" | **sklearn has no ARIMA** — this is a documentation slip | Use `statsmodels.tsa` (SARIMAX) / `pmdarima` as the ARIMA oracle; flag the PROJECT.md line for correction |
| Seasonal ARIMA + exog + missing-data in the first cut | "Full SARIMAX parity" | Each multiplies the Kalman state-space surface | Non-seasonal ARIMA(p,d,q) first; add seasonal (P,D,Q,s) and exog as graded sub-requirements |

---

## Concrete API Surface (constructor params → fit/predict → fitted attrs)

### (a) RandomForest

**RandomForestClassifier** (cuML defaults): `n_estimators=100`, `split_criterion='gini'` (`gini`|`entropy`),
`bootstrap=True`, `max_samples=1.0`, `max_depth=None`, `max_leaves=-1`, `max_features='sqrt'`
(`'sqrt'`|`'log2'`|`None`|int|float), `n_bins=128`, `n_streams=4`, `min_samples_leaf=1`,
`min_samples_split=2`, `min_impurity_decrease=0.0`, `max_batch_size=4096`, `random_state`.
**RandomForestRegressor**: same, but `split_criterion='mse'` (`mse`|`mae`|`poisson`).
- Criterion enum in cuML: `GINI, ENTROPY, MSE, MAE, POISSON`.
- API: `fit(X, y)` → `predict(X)`, `predict_proba(X)` (clf), `score(X,y)`; attrs `feature_importances_`,
  `oob_score_` (if `bootstrap=True`), `n_features_in_`, `classes_` (clf).
- **What FIL needs from the tree format:** a per-tree node table — for each node: `feature_index`
  (or LEAF sentinel), `threshold` (float, split `x[f] <= threshold` → left), `left_child`,
  `right_child` indices, and `leaf_value` (regression: scalar; classification: class-probability
  vector or class id). Dense (complete-array) or sparse (CSR-of-nodes) layout. cuML routes this via
  Treelite; **mlrs should define its own flat node-array format** (cpu-MLIR-safe GATHER traversal),
  NOT adopt Treelite. Quantile bin edges (`n_bins`) are the candidate split thresholds.
- **Note:** `n_bins` quantile binning means even a "deterministic" mlrs tree won't equal sklearn's
  exhaustive-threshold tree exactly; the deterministic-single-tree exact check should be against a
  *binned* reference, or accept a documented split-threshold band.

### (b) ARIMA / AutoARIMA

**ARIMA**: `order=(p,d,q)` default `(1,1,1)`, `seasonal_order=(P,D,Q,s)` default `(0,0,0,0)`,
`fit_intercept`/`k` (mu term), `exog` (n_exog regressors), `simple_differencing=True`,
`handle`, `convert_dtype`. Internals: `ARIMAOrder{p,d,q,P,D,Q,s,k,n_exog}`,
`ARIMAParams{mu,beta,ar,ma,sar,sma,sigma2}`.
- API: `fit()` (ML via batched L-BFGS over Kalman-filter log-likelihood) → `forecast(nsteps)`,
  `predict(start,end)`; attrs `aic`, `aicc`, `bic`, fitted `mu_/ar_/ma_/sar_/sma_`.
- Batched over many series simultaneously — cuML's whole value prop. mlrs has L-BFGS but **batched
  L-BFGS is new work**; the batched Kalman-filter likelihood is the new device kernel.

**AutoARIMA**: `search(s, d, D, max_p, max_q, max_P, max_Q, start_p, start_q, ic='aicc'`
(`aic`|`aicc`|`bic`)`, test='kpss'` (d-selection)`, seasonal_test='seas'` (D-selection)`, ...)`.
- Picks `d` via stationarity test (KPSS), `D` via seasonal test, then grid/stepwise over remaining
  orders scored by the information criterion. Built directly on ARIMA.

### (c) Kernel SHAP vs Permutation SHAP

**KernelExplainer**: `__init__(model, data` (background)`, nsamples='auto'` =`2*n_features+2048`,
`link='identity'` (`identity`|`logit`)`, ...)` → `shap_values(X)`.
Semantics: sample coalitions of present/absent features, evaluate model on masked rows (absent →
background values), solve a **weighted least-squares** (SHAP kernel weights) per row → local linear
attributions. Model-agnostic, expensive (many model calls), stochastic in sampling.

**PermutationExplainer**: `__init__(model, data, link='identity', ...)` → `shap_values(X)`.
Semantics: iterate feature **permutations**, do forward+reverse masking passes, accumulate marginal
contributions. Cheaper, fewer model evals, also stochastic.
- Both share `SHAPBase` (background data, link function, masking machinery). Both satisfy
  **efficiency exactly**: `base_value + sum(shap_values) == link(model(x))`.

### (d) Highest-value utility surface

| Module | Ship these (highest value) | Defer / passthrough |
|---|---|---|
| **metrics** | `accuracy_score`, `confusion_matrix`, `r2_score`, `mean_squared_error`, `mean_absolute_error`, `log_loss`, `roc_auc_score`, `precision_recall_curve` | `kl_divergence`, `hinge_loss`, cluster metrics (`adjusted_rand_score` already an internal helper), `mean_squared_log_error`, `median_absolute_error` |
| **preprocessing** | `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `Binarizer`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `LabelBinarizer`, `SimpleImputer` | `PolynomialFeatures`, `PowerTransformer`, `QuantileTransformer`, `KBinsDiscretizer`, `FunctionTransformer`, `KernelCenterer`, `TargetEncoder` |
| **model_selection** | `train_test_split`, `KFold`, `StratifiedKFold` | `GridSearchCV`/`RandomizedSearchCV` → delegate to sklearn (cuML does exactly this via `__getattr__`) |
| **feature_extraction** | `CountVectorizer`, `TfidfTransformer`, `TfidfVectorizer` (lower priority; text/host-heavy) | — |

### (e) Symbolic / genetic regression (gplearn oracle)

**SymbolicRegressor** (gplearn defaults): `population_size=1000`, `generations=20`,
`tournament_size=20`, `function_set=('add','sub','mul','div')` (+ optional `sqrt,log,abs,neg,inv,
max,min,sin,cos,tan`), `metric='mean absolute error'`, `parsimony_coefficient=0.001`,
`p_crossover=0.9`, `p_subtree_mutation=0.01`, `p_hoist_mutation=0.01`, `p_point_mutation=0.01`,
`init_depth=(2,6)`, `init_method='half and half'`, `const_range=(-1,1)`, `stopping_criteria=0.0`,
`max_samples=1.0`, `random_state`.
- API: `fit(X,y)` → `predict(X)`; attr `_program` (best program), `program` repr.
- Also `SymbolicClassifier`, `SymbolicTransformer` in gplearn (defer; differentiator/later).
- Heavily stochastic → property gate (fitness band + valid trees + seed-reproducibility).

---

## Feature Dependencies

```
[RF feasibility spike]  (gating, FIRST — make-or-break under cpu-MLIR)
        └──gates──> [RandomForestClassifier + RandomForestRegressor]
                          └──defines tree format──> [FIL (forest inference)]
                                                          └──requires trees──> [TreeSHAP]

[Kernel SHAP] ──shares SHAPBase──> [Permutation SHAP]    (model-agnostic; need any fitted estimator)
[ARIMA] ──requires batched L-BFGS + batched Kalman prim──> [AutoARIMA order search]
[Symbolic regression]   (independent; new GP engine, no device-prim dependency on others)

[metrics] ─┐
[preprocessing] ─┼──independent, light/host; enable everything else's testing & pipelines
[model_selection] ─┘

[cuml.accel] ──proxies to──> ALL 32 existing estimators + the new v4.0 estimators
                              (depends on everything accelerable; build LAST)
```

### Dependency Notes

- **RF spike gates the whole tree stack:** if GPU histogram/split can't be made cpu-MLIR-safe
  (no SharedMemory, no cross-unit atomics — see project memory + spike-findings-mlrs), RF→FIL→
  TreeSHAP scope is renegotiated before any of it is committed. Mirrors Phase 13's KNN-graph
  keystone spike. **Run first.**
- **FIL depends on RF's tree format, TreeSHAP depends on FIL:** strict linear order. FIL's node-array
  layout is the contract both RF (writer) and TreeSHAP (reader) bind to — design it once, in the RF
  phase, validated by FIL.
- **TreeSHAP needs only the tree format, not the trainer:** once the format is frozen, TreeSHAP can
  be validated against `shap.TreeExplainer` on any forest (even a hand-built one) — exact gate.
- **AutoARIMA requires ARIMA:** order search calls ARIMA fit repeatedly; batched L-BFGS + batched
  Kalman likelihood are new prims that ARIMA introduces.
- **Kernel/Permutation SHAP need a fitted model + GEMM/lstsq prims (already exist):** independent of
  the tree stack; can land in parallel. Most compelling demoed on RF, but not blocked by it.
- **cuml.accel proxies to the full estimator surface:** pure Python orchestration (import-hook,
  module-swap, per-estimator override, CPU fallback) and should land **last**, after the estimators
  it accelerates exist, so its proxy table is complete.
- **Utility surface (metrics/preprocessing/model_selection) is independent and foundational:** mostly
  host/reduction work; landing `accuracy/r2/confusion_matrix` + the scalers early also strengthens
  every other feature's test harness.

---

## MVP Definition (per v4.0 — "minimum to claim the surface is complete")

### Launch With (the spine)

- [ ] **RF feasibility spike** — make-or-break, gates the tree stack; FIRST phase.
- [ ] **RandomForestClassifier + RandomForestRegressor** — keystone; property/predictive gate.
- [ ] **FIL** — exact inference over the mlrs tree format (the one exact tree-stack gate).
- [ ] **ARIMA(p,d,q)** non-seasonal + forecast — `statsmodels` band gate.
- [ ] **metrics core** — accuracy, confusion_matrix, r2, mse, mae (exact/≤1e-5).
- [ ] **preprocessing scalers** — Standard/MinMax/MaxAbs/Robust/Normalizer (≤1e-5).
- [ ] **model_selection** — train_test_split, KFold, StratifiedKFold (structural gate).

### Add After the Spine (same milestone, contingent)

- [ ] **TreeSHAP** — exact ≤1e-5 vs `shap`; after FIL.
- [ ] **Kernel SHAP + Permutation SHAP** — efficiency-invariant + band vs `shap`.
- [ ] **AutoARIMA** — order search on top of ARIMA; seasonal (P,D,Q,s) + exog graded.
- [ ] **preprocessing encoders** — OneHot/Ordinal/Label/LabelBinarizer/SimpleImputer.
- [ ] **roc_auc_score / log_loss / precision_recall_curve** — remaining high-value metrics.
- [ ] **Symbolic regression** — gplearn property gate.

### Land Last / Lower Priority

- [ ] **cuml.accel** — pure-Python drop-in; build last when the proxy target set is complete.
- [ ] **feature_extraction (Tfidf/Count)** — text pipelines; host-heavy, lower demand.
- [ ] **GridSearchCV/RandomizedSearchCV** — delegate to sklearn (no native build).

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---|---|---|---|
| RandomForest clf/reg (+ spike) | HIGH | HIGH | P1 |
| FIL | HIGH | MEDIUM | P1 |
| metrics core (accuracy/r2/confusion/mse/mae) | HIGH | LOW | P1 |
| preprocessing scalers | HIGH | LOW–MEDIUM | P1 |
| model_selection splitters | HIGH | LOW | P1 |
| ARIMA(p,d,q) | HIGH | HIGH | P1 |
| TreeSHAP | MEDIUM–HIGH | HIGH | P2 |
| Kernel SHAP | MEDIUM | MEDIUM–HIGH | P2 |
| AutoARIMA | MEDIUM | HIGH | P2 |
| preprocessing encoders | MEDIUM | MEDIUM | P2 |
| roc_auc/log_loss/PR-curve | MEDIUM | LOW | P2 |
| Permutation SHAP | MEDIUM | MEDIUM | P2 |
| cuml.accel | MEDIUM–HIGH | MEDIUM (Python) | P2 (build last) |
| Symbolic regression | MEDIUM | HIGH | P3 |
| feature_extraction (tfidf) | LOW–MEDIUM | MEDIUM | P3 |
| GridSearchCV (passthrough) | LOW | LOW | P3 |

---

## Reference Feature Analysis (cuML → mlrs mapping)

| Feature | cuML reference | mlrs approach |
|---|---|---|
| RandomForest | `ensemble/randomforest{classifier,regressor}.py` + `randomforest_common.pyx` (GINI/ENTROPY/MSE/MAE/POISSON, `n_bins` quantile splits, Treelite export) | Own flat node-array tree format (cpu-MLIR-safe GATHER); sklearn property gate; NO Treelite |
| FIL | `fil/` (Treelite-backed batched traversal) | Native batched GATHER traversal over mlrs node arrays; exact vs host reference |
| TreeSHAP | `explainer/tree_shap.pyx` (Treelite path-info + GPUTreeShap) | Path-dependent Lundberg over mlrs trees; exact vs `shap.TreeExplainer` |
| ARIMA/AutoARIMA | `tsa/arima.pyx`, `auto_arima.pyx`, `batched_lbfgs.py`, `stationarity.pyx`, `seasonality.py` | Batched Kalman likelihood + batched L-BFGS; `statsmodels`/`pmdarima` band gate |
| Kernel/Perm SHAP | `explainer/kernel_shap.pyx`, `permutation_shap.pyx` (`SHAPBase`) | GEMM/lstsq prims; `shap` efficiency-invariant + band |
| cuml.accel | `accel/` (`accelerator.py`, `estimator_proxy.py`, `_overrides/{sklearn,umap,hdbscan}`) | Import-hook/module-swap proxying to the 32 mlrs estimators + CPU fallback |
| metrics | `metrics/` (`_classification`, `regression`, `confusion_matrix`, `_ranking`, …) | Reduction kernels; sklearn ≤1e-5/exact |
| preprocessing | `preprocessing/` + `_thirdparty/sklearn/preprocessing` (scalers, encoders) | Column-stat fit + elementwise transform; sklearn ≤1e-5 |
| model_selection | `model_selection/_split.py` (train_test_split, KFold, StratifiedKFold; GridSearchCV via `__getattr__`→sklearn) | Native splitters (structural gate); search delegated to sklearn |
| feature_extraction | `feature_extraction/text.py` (CountVectorizer, Tfidf{Transformer,Vectorizer}) | Host tokenize + sparse counts; sklearn ≤1e-5 |
| Symbolic regression | cuML `genetic` (gplearn-compatible SymbolicRegressor) | GP engine; `gplearn` property gate |

## Sources

- `cuml-main/python/cuml/cuml/{ensemble,fil,tsa,explainer,accel,metrics,preprocessing,model_selection,feature_extraction}/` — RAPIDS cuML v26.08 reference API/behavior (read-only, in-tree) — **HIGH**
- `.planning/PROJECT.md`, `.planning/notes/v3-hard-algorithm-backlog.md`, `.planning/notes/cuml-mlrs-gap-inventory.md` — milestone scope, dependency ordering, Tier-3 rationale — **HIGH**
- `.planning/milestones/v3.0-REQUIREMENTS.md` — REQ-ID / oracle / gate structuring precedent (value vs property vs exact-label gates; D-12) — **HIGH**
- Project memory: cpu-MLIR no-SharedMemory/no-atomics constraint, f64-on-rocm skip, stochastic-gate precedent (RandomProjection, UMAP layout) — **HIGH**
- gplearn / shap / statsmodels public API (constructor defaults) — well-established, version-stable — **MEDIUM** (verify exact defaults at implementation time via `find-docs`)

---
*Feature research for: mlrs v4.0 — Tree Ensembles, Time-Series & Full-Surface Completion*
*Researched: 2026-06-26*
