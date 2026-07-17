# mlrs Coverage-Gap Research — Next-Feature Recommendation

**Agent:** Research (Spec-TDD workflow). **Date:** 2026-07-16.
**Question answered:** Of the sklearn/cuML features NOT yet in mlrs, which single feature is the highest-value next thing to implement, and what must the Planner know to plan it via TDD?

**Evidence labels:** `[VERIFIED: CODEGRAPH <path:symbol>]` · `[VERIFIED: LOCAL <path>]` · `[VERIFIED: WEB <url>]` · `[INFERRED: …]` · `[UNVERIFIED: …]`

---

## 1. Executive Summary + Recommendation

The codebase has advanced **past** `.planning/ROADMAP.md`. The roadmap still marks Phases 18–26 "Not started" and frames RandomForest/HGB as future Phase 18–19 work, but git `fb0c9c7 Add Random Forest and HistGradientBoosting ensemble estimators (ENSEMBLE-01, GBT-01)` already landed **four fitted ensemble estimators** at the Rust-algos layer with full oracle tests. **Do not trust the roadmap status columns.** `[VERIFIED: LOCAL git log]` `[VERIFIED: LOCAL .planning/ROADMAP.md:78-87]`

The single, decisive coverage gap: **those four ensemble estimators have Rust cores + committed oracle fixtures + Rust oracle tests, but ZERO Python-facing surface** — no `crates/mlrs-py/src/estimators/ensemble.rs`, no registration in `_mlrs`, no pure-Python shim, no `__init__` re-export. They are unreachable from `import mlrs`. `[VERIFIED: LOCAL crates/mlrs-py/src/estimators/ (no ensemble.rs)]` `[VERIFIED: LOCAL crates/mlrs-py/src/lib.rs:200-267 (12 wrapper imports, none ensemble)]` `[VERIFIED: LOCAL crates/mlrs-py/python/mlrs/__init__.py:22-98 (no ensemble import/export)]`

### Recommendation (single): **PY-ENSEMBLE — bind the four already-landed ensemble estimators to Python**

Implement `RandomForestClassifier`, `RandomForestRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` on the existing PyO3 + pure-Python shim surface, gated by a new Python-side oracle harness that replays the already-committed `rf_*`/`hgb_*` fixtures through the full binding path (the exact pattern `test_oracle_neighbors.py` uses).

**Why this over building new algorithms:** it is the smallest correct next step, it ships four user-visible sklearn-compatible estimators, every dependency already exists (Rust cores fitted + device-tested, fixtures committed, binding macros + shim base in place), the oracle is deterministic and already written on the Rust side, and it is low-risk to TDD because there is a byte-identical binding template to mirror (`estimators/naive_bayes.rs` + `python/mlrs/naive_bayes.py` for the classifiers; `estimators/linear.rs`/`neighbors.py` regressor surface for the regressors). It touches **one binding layer**, adds **no compute dependency**, and requires **no new kernels**.

**Confidence:** HIGH that the gap is real and self-contained; MEDIUM on a few scope decisions the Planner must lock (see §6): `predict_log_proba`, `feature_importances_`/`oob_score_` (NOT implemented in the Rust core), and `random_state=None` semantics.

---

## 2. Ground-Truth Estimator Matrix

Columns: **Rust core** (`crates/mlrs-algos/src/`), **Rust oracle test** (`crates/mlrs-algos/tests/`), **PyO3 wrapper** (`crates/mlrs-py/src/estimators/`), **Python shim** (`crates/mlrs-py/python/mlrs/`), **`_mlrs` registered** (`lib.rs`). `[VERIFIED: LOCAL]` for every cell via directory listings + `lib.rs:200-267` + `__init__.py:22-98`.

| Estimator | Rust core | Rust oracle test | PyO3 wrapper | Py shim | Registered |
|---|---|---|---|---|---|
| LinearRegression, Ridge, Lasso, ElasticNet | ✅ | ✅ | ✅ linear.rs | ✅ linear.py | ✅ |
| LogisticRegression | ✅ | ✅ | ✅ linear.rs | ✅ linear.py | ✅ |
| MBSGDClassifier, MBSGDRegressor, LinearSVC, LinearSVR | ✅ | ✅ | ✅ linear.rs | ✅ linear.py | ✅ |
| KMeans, DBSCAN | ✅ | ✅ | ✅ cluster.rs | ✅ cluster.py | ✅ |
| HDBSCAN | ✅ | ✅ | ✅ cluster.rs | ✅ cluster.py | ✅ |
| SpectralClustering, SpectralEmbedding | ✅ | ✅ | ✅ spectral.rs | ✅ cluster.py | ✅ |
| PCA, TruncatedSVD, IncrementalPCA | ✅ | ✅ | ✅ decomposition.rs | ✅ decomposition.py | ✅ |
| UMAP | ✅ | ✅ | ✅ manifold.rs | ✅ manifold.py | ✅ |
| NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor | ✅ | ✅ | ✅ neighbors.rs | ✅ neighbors.py | ✅ |
| GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB | ✅ | ✅ | ✅ naive_bayes.rs | ✅ naive_bayes.py | ✅ |
| EmpiricalCovariance, LedoitWolf | ✅ | ✅ | ✅ covariance.rs | ✅ covariance.py | ✅ |
| GaussianRandomProjection, SparseRandomProjection | ✅ | ✅ | ✅ projection.rs | ✅ random_projection.py | ✅ |
| KernelRidge, KernelDensity | ✅ | ✅ | ✅ kernel.rs | ✅ kernel_ridge.py/density.py | ✅ |
| **RandomForestClassifier** | ✅ | ✅ random_forest_classifier_test.rs | ❌ | ❌ | ❌ |
| **RandomForestRegressor** | ✅ | ✅ random_forest_regressor_test.rs | ❌ | ❌ | ❌ |
| **HistGradientBoostingClassifier** | ✅ | ✅ hist_gradient_boosting_classifier_test.rs | ❌ | ❌ | ❌ |
| **HistGradientBoostingRegressor** | ✅ | ✅ hist_gradient_boosting_regressor_test.rs | ❌ | ❌ | ❌ |

The four ensemble rows are the **only** estimators in the tree with a full Rust+test surface but no Python surface. `grep -rniE "randomforest|histgradient|ensemble" crates/mlrs-py --include=*.py --include=*.rs` returns **nothing**. `[VERIFIED: LOCAL grep]`

**Uncommitted work in progress (git status):** `crates/mlrs-backend/src/prims/hist_gradient_boosting.rs`, `crates/mlrs-kernels/src/gbt.rs`, `scripts/gen_oracle.py`, and the four `hgb_*.npz` fixtures are modified but uncommitted (the sibling-histogram-subtraction + oracle-regen refinement noted in MEMORY). This is **prim/kernel-level** churn beneath the HGB estimators; it does not change their public `fit`/`predict`/`predict_proba` signatures, but the Planner should treat the HGB algos layer as "in flux" and rebase fixtures before pinning Python oracle tolerances (see §6 risk). `[VERIFIED: LOCAL git diff --stat]`

---

## 3. Gap Analysis vs sklearn / cuML

Candidates present in sklearn and/or cuML but absent from mlrs's **Python surface**, drawn from the roadmap's own remaining v4.0 surface (`.planning/ROADMAP.md:78-90`) plus the matrix above. `[VERIFIED: LOCAL .planning/ROADMAP.md]`

| Gap | In sklearn | In cuML | mlrs status | Depends on (already in mlrs?) |
|---|---|---|---|---|
| **Ensemble Python bindings** (RF×2, HGB×2) | ✅ | ✅ | **Rust core + tests DONE; Python surface MISSING** | Binding macros, shim base, fixtures — **all present** ✅ |
| DecisionTree standalone (clf/reg) | ✅ | ✅ | Absent (roadmap Phase 18) | Tree prims exist (used by RF/HGB); no standalone estimator |
| FIL (batched forest inference) | — | ✅ | Absent (roadmap Phase 20) | Needs node-store traversal prim (partial) |
| TreeSHAP | via `shap` | ✅ | Absent (roadmap Phase 21) | Needs FIL tree store |
| ARIMA / AutoARIMA / SARIMAX | — (statsmodels) | ✅ | Absent (roadmap Phase 22) | Needs batched Kalman + L-BFGS prims (none) |
| Kernel SHAP / Permutation SHAP | via `shap` | ✅ | Absent (roadmap Phase 23) | Reuses linear solver / predict path |
| sklearn-utility surface: metrics / preprocessing / feature_extraction / model_selection | ✅ | ✅ | Absent (roadmap Phase 24) | Mostly host-deterministic; **no new prims** |
| Genetic/Symbolic regression | — (gplearn) | ✅ | Absent (roadmap Phase 25) | Needs `program_eval` device prim (none) |
| cuml.accel drop-in | — | ✅ | Absent (roadmap Phase 26) | Needs the whole estimator surface first |
| Kernel SVM / SMO | ✅ | ✅ | **Explicitly out of v4.0 scope** | n/a — roadmap excludes it |

`feature_importances_` and `oob_score_` (roadmap Phase 19 success criteria) are **NOT** present even in the landed RF Rust core — the fitted structs expose only `classes()`, `n_classes()`, `model()`, and the predict traits; there is no importances/OOB accessor. `[VERIFIED: CODEGRAPH grep pub fn on crates/mlrs-algos/src/ensemble/*.rs — only classes/n_classes/model/predict]`

---

## 4. Ranked Shortlist

1. **PY-ENSEMBLE — Python bindings for the four landed ensemble estimators (RECOMMENDED).**
   - *Unblocked:* every dependency exists (Rust cores fitted + device-tested, four `.npz` fixtures committed, `any_estimator_typestate!` macro + `MlrsBase` shim + `ClassifierMixin`/`RegressorMixin` all in use). No new kernels, no new algos, no new compute dependency.
   - *Low-risk TDD:* byte-for-byte template exists (`naive_bayes.rs`/`naive_bayes.py` for classifiers; `linear.rs`/`neighbors.py` for regressors) and a ready oracle harness pattern (`test_oracle_neighbors.py`).
   - *Value:* four user-visible sklearn estimators go from unreachable to `import mlrs`-usable. Highest value-per-unit-risk.

2. **sklearn-utility metrics subset (roadmap Phase 24, metrics only).**
   - *Unblocked:* deterministic, host-side, exhaustively documented oracle, no new prims. But it is a **new module family from scratch** (new algos module + new PyO3 module + new shim module + new fixture family + new oracle generators) — materially larger surface and more design decisions (where do free-function metrics live vs estimator classes) than finishing an existing surface. Better as the *next* phase after PY-ENSEMBLE.

3. **DecisionTree standalone (roadmap Phase 18).**
   - Reuses existing tree prims but requires a **new fitted estimator** (host level-wise loop), a new oracle family, and structural exact-match gates. More algos-layer risk than PY-ENSEMBLE and partially redundant with the already-landed forest cores.

Recommendation: **#1 (PY-ENSEMBLE).** It is the cheapest correct completion and unlocks the most user value with the least new design.

---

## 5. Recommended Feature — Deep Dive (Planner-Ready)

### 5.1 What exists in the Rust core (the surface to wrap)

All four are **typestate** estimators `Struct<F, S = Unfit>` with a consuming `typestate::Fit::fit` returning the `Fitted`-tagged sibling. `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/ensemble/*]`

- **`RandomForestClassifier<F, Fitted>`** — `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs`
  - `Fit::fit(self, pool, x, y: Option<&DeviceArray>, (n,d))` → requires `y` (integer-valued `F` labels); ingests labels via `ingest_labels` → `classes_`, `n_classes_`. `[VERIFIED: CODEGRAPH random_forest_classifier.rs:366-418]`
  - `PredictProba::predict_proba` → `n_query × n_classes` device floats, rows sum to 1. `[VERIFIED: CODEGRAPH :427-439]`
  - `PredictLabels::predict_labels` → i32 argmax mapped through `classes_`.
  - Accessors: `classes() -> &[i32]`, `n_classes() -> usize`, `model()`. **No `feature_importances_`, no `oob_score_`.** `[VERIFIED: CODEGRAPH grep :137,:142,:155]`
  - Builder setters: `n_estimators, max_depth, n_bins, max_features(MaxFeatures), min_samples_split(f64), min_samples_leaf(f64), bootstrap(bool), seed(u64)`. Defaults: `n_estimators=100, max_depth=10, n_bins=32, max_features=Sqrt (clf) / All (reg), min_samples_split=2.0, min_samples_leaf=1.0, bootstrap=true, seed=42`. `max_depth` bounded `1..=16`; `n_bins` `2..=256`. `[VERIFIED: CODEGRAPH random_forest_classifier.rs:184-229; random_forest_regressor.rs:39-44; builder_rejects_invalid_hyperparameters test]`

- **`RandomForestRegressor<F, Fitted>`** — `random_forest_regressor.rs`
  - `Fit::fit(self, pool, x, y, (n,d))` requires `y` (targets); `Predict::predict` → length-`n_query` forest mean floats. `model()` only. `[VERIFIED: CODEGRAPH :296-311]`

- **`HistGradientBoostingClassifier<F, Fitted>`** — `hist_gradient_boosting_classifier.rs`
  - `PredictProba` (sigmoid/softmax link, rows sum to 1) + `PredictLabels` (host argmax over ONE metered proba readback, strict-`>` lowest-index tie-break). `classes()`, `n_classes()`, `model()`. `[VERIFIED: CODEGRAPH :313-369]`
  - Builder setters: `max_iter, learning_rate(f64), max_depth, n_bins, l2_regularization(f64), min_samples_leaf(usize)`. Defaults: `max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64, l2=0.0, min_samples_leaf=20`. `[VERIFIED: CODEGRAPH hist_gradient_boosting_classifier.rs:181-214; hist_gradient_boosting_regressor.rs:45-50]`

- **`HistGradientBoostingRegressor<F, Fitted>`** — `hist_gradient_boosting_regressor.rs`
  - `Predict::predict` → raw ensemble scores (baseline mean + shrunk leaf sums). `model()` only. `[VERIFIED: CODEGRAPH :330-342]`

Traits to import from `mlrs_algos::typestate`: `Fit, Predict, PredictLabels, PredictProba` (NOT `PredictLogProba` — ensemble cores don't implement it). `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/typestate.rs:143,168,265,346,371]`

### 5.2 Files/symbols to CREATE or MODIFY

**CREATE `crates/mlrs-py/src/estimators/ensemble.rs`** — four `#[pyclass]` wrappers.
- Use `crate::any_estimator_typestate!` (the `S=Fitted`-explicit macro; ensemble cores default `S=Unfit`, so the plain `any_estimator!` would resolve the WRONG monomorphization — same trap the doc comment calls out). `[VERIFIED: LOCAL crates/mlrs-py/src/dispatch.rs:120-185]`
- **Classifiers** mirror `estimators/naive_bayes.rs` (the exact template: `any_estimator_typestate!` + `fit(x, y, rows, cols)` taking a `y` capsule + `classes_()` i32 getter + `predict_labels` + `predict_proba_f32`/`predict_proba_f64` + `is_fitted` + `dtype`). Each device body: `capsule_to_array` → `float_dtype` → `py.detach(|| { let mut pool = crate::lock_pool(); … })`, `guard_f64()?` BEFORE the F64 upload, build via `Struct::<F>::builder().<setters>().build::<F>().map_err(build_err_to_py)?`, then `TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows,cols)).map_err(algo_err_to_py)?`. `[VERIFIED: LOCAL crates/mlrs-py/src/estimators/naive_bayes.rs:71-181; cluster.rs:435-493 (typestate consuming-fit shape)]`
- **Regressors** mirror the classifier fit path minus `classes_`/proba, plus a float `predict_f32`/`predict_f64` accessor (compose from the `Predict` trait like the linear regressors). No exact "typestate + fit(x,y) + float predict" template exists in one file today — compose naive_bayes's typestate fit(x,y) with a linear-style float `predict`. `[INFERRED: naive_bayes.rs gives typestate+fit(x,y); linear.rs gives float predict; ensemble regressor = their composition]`
- `MaxFeatures` mapping: parse sklearn `max_features` (`"sqrt"`/`"log2"`/float/int/None) → `mlrs_algos::ensemble::MaxFeatures::{Sqrt,Log2,All,Value(n)}` at construction (`ValueError` on bad string), mirroring `parse_hdbscan_metric` in cluster.rs. `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/ensemble/mod.rs:48-71; LOCAL cluster.rs:326-346 parse pattern]`

**MODIFY `crates/mlrs-py/src/estimators/mod.rs`** — add `pub mod ensemble;` (currently 10 submodules, no ensemble). `[VERIFIED: LOCAL estimators/mod.rs:31-40]`

**MODIFY `crates/mlrs-py/src/lib.rs`** — in `#[pymodule] fn _mlrs`, `use estimators::ensemble::{PyRandomForestClassifier, PyRandomForestRegressor, PyHistGradientBoostingClassifier, PyHistGradientBoostingRegressor};` and four `m.add_class::<…>()?;` (registration 30 → 34). Update the "12 estimator" / "30" prose comments. `[VERIFIED: LOCAL lib.rs:198-268]`

**CREATE `crates/mlrs-py/python/mlrs/ensemble.py`** — four shim classes.
- Classifiers: subclass `ClassifierMixin, MlrsBase` (mirror `_BaseNB` in `naive_bayes.py`): pure `__init__` storing every sklearn ctor arg verbatim under the same name (+ `output_type="input"`); `fit(X, y)` → `_normalize`/`_normalize_y` → build `_mlrs` wrapper → `_store_fit` (sets `_mlrs_obj`, `_post_fit(cols)`, `classes_`); `predict` via `predict_labels`; `predict_proba` via `_suffixed("predict_proba")`. `[VERIFIED: LOCAL crates/mlrs-py/python/mlrs/naive_bayes.py:23-105; base.py:32-167]`
- Regressors: subclass `RegressorMixin, MlrsBase`; `fit(X, y)`; `predict` via `_suffixed("predict")` → `_to_output(..., self._np_float())`. `[VERIFIED: LOCAL base.py; RegressorMixin used in neighbors.py per __init__ imports]`
- Defaults in `__init__` MUST equal the `#[new]` defaults, which MUST equal the Rust builder defaults (single-source rule, D-08/D-02). `[VERIFIED: LOCAL naive_bayes.py:13-14 doc]`

**MODIFY `crates/mlrs-py/python/mlrs/__init__.py`** — `from .ensemble import (…)` + add four names to `__all__`. `[VERIFIED: LOCAL __init__.py:22-98]`

**CREATE `crates/mlrs-py/python/tests/test_oracle_ensemble.py`** — replay the four committed fixtures through the full Python path (SECOND consumer; no regeneration), mirroring `test_oracle_neighbors.py`: deterministic tier exact/≤1e-5 on train, statistical tier accuracy/R² band on held-out, f64 cases behind `@requires_f64`. `[VERIFIED: LOCAL crates/mlrs-py/python/tests/test_oracle_neighbors.py:1-70]`

**MODIFY the estimator-enumerating Python tests** — these gate every shim and will fail-closed for a new estimator until updated:
- `crates/mlrs-py/python/tests/test_params.py` — the AST **purity gate** + per-estimator `get_params`/mutation tables (`_PARAM_*` dicts keyed by class name). Add four entries. `[VERIFIED: LOCAL test_params.py:53,152-157,255,273-274,302]`
- `crates/mlrs-py/python/tests/test_shims.py` — mixin/attribute enumeration (`ClassifierMixin`/`RegressorMixin` checks, fitted-attr lists). `[VERIFIED: LOCAL test_shims.py:74-188]`
- `crates/mlrs-py/python/tests/test_estimator_checks.py` — sklearn `check_estimator` sweep; likely enumerates estimators. Verify + extend. `[VERIFIED: LOCAL grep — file enumerates estimators]`

**Rust-side PyO3 unit tests** live in `crates/mlrs-py/tests/` (e.g. `test_naive_bayes.py`, `test_sgd.py`) per AGENTS.md §2 (never in-source `#[cfg(test)] mod`). Add a not-fitted / dtype-guard test file if mirroring the NB precedent. `[VERIFIED: LOCAL crates/mlrs-py/tests/ listing; estimators/mod.rs:28-29 note]`

### 5.3 Binding pattern to mirror (load-bearing contracts)

Every device method (from `crates/mlrs-py/src/dispatch.rs` doc + naive_bayes.rs): `[VERIFIED: LOCAL dispatch.rs:21-60; naive_bayes.rs:130-181; lib.rs:96-158]`
1. **GIL release (PY-03):** wrap the trait call in `py.detach(|| { … })`; the closure is `Send`, touches no Python objects.
2. **Sanctioned lock (WR-04):** inside the closure use `crate::lock_pool()` (poison-recovering) — NEVER `global_pool().lock().expect(...)`.
3. **f64 guard (D-04):** on the `FloatDtype::F64` arm call `crate::capability::guard_f64()?` BEFORE any upload.
4. **Build-before-upload (T-12-02):** validate data-independent hyperparameters at `.build()` (→ `build_err_to_py` → `ValueError`) before the device upload; fit-time/geometry errors → `algo_err_to_py`.
5. **Egress:** materialize host `Vec<f32>`/`Vec<f64>` (dtype-suffixed accessors) or `Vec<i32>` (labels/classes) via `to_host_metered(&mut pool)`; the shim wraps to `output_type` (D-03/D-06).
6. **dtype dispatch (D-06):** `#[pyclass]` cannot be generic over `F`; the `Any<Name>` enum carries `Unfit{..}`/`F32(_)`/`F64(_)`; the shim reads `dtype()` to pick the `_f32`/`_f64` suffix.

### 5.4 Oracle / fixture convention

- **Fixtures already committed** at `tests/fixtures/{rf,hgb}_{cls,reg}_{f32,f64}_seed42.npz` (8 files). PY-ENSEMBLE is a **second consumer** — no regeneration required. `[VERIFIED: LOCAL tests/fixtures listing]`
- **RF classifier fixture keys** (from the Rust test): `X, y, Xq, yq, det_pred_train, det_proba_train, stat_acc_test`. Geometry: 96 train / 48 test / 5 features / 3 classes. Two tiers: deterministic (`bootstrap=false, max_features=All, depth=12, n_estimators=2`) → EXACT train `predict_labels` + `predict_proba` ≤1e-5; statistical (`n_estimators=64, depth=8`) → held-out accuracy within `ACC_MARGIN=0.05`, proba rows sum to 1. `[VERIFIED: LOCAL crates/mlrs-algos/tests/random_forest_classifier_test.rs:36-192]`
- **HGB deterministic tier** relies on `max_leaf_nodes=None` + depth bound making sklearn's leaf-wise tree equal the mlrs level-wise tree; grid-valued features (16 distinct << `max_bins=255`) make `_BinMapper` midpoints equal the mlrs candidate edges; `early_stopping=False`. sklearn det kwargs: `max_iter=20, learning_rate=0.1, max_depth=6, max_leaf_nodes=None, min_samples_leaf=5, l2=0.0, max_bins=255, early_stopping=False, random_state=0`. To reproduce the deterministic tier in Python the estimator must be constructed with `n_bins=255` (NOT the default 64) — same as the Rust test uses `.n_bins(255)`. `[VERIFIED: LOCAL scripts/gen_oracle.py:3413-3452; ensemble/mod.rs:41-44 doc]`
- **Regen path (only if fixtures must change):** `python3 -m venv /tmp/oracle-venv && /tmp/oracle-venv/bin/pip install numpy scipy scikit-learn`, then run `scripts/gen_oracle.py` (PEP-668 venv). Ensemble fixtures use the general `numpy scipy scikit-learn` venv (the pinned `scikit-learn==1.9.0` note applies to HDBSCAN, not ensemble). Confirm the sklearn version actually used before pinning ≤1e-5 tolerances. `[VERIFIED: LOCAL scripts/gen_oracle.py:15-17, 3253-3600]` `[UNVERIFIED: exact sklearn version used to produce the committed ensemble fixtures — not stamped in-repo]`
- **Capability gate:** Rust tests use `capability::skip_f64_with_log()`; Python tests use the `@requires_f64` marker from `conftest.py` (skips f64 on an f64-incapable backend, e.g. rocm). `[VERIFIED: LOCAL random_forest_classifier_test.rs:202; test_oracle_neighbors.py:20,31]`

### 5.5 Builder / typestate convention

- Ensemble cores are v3 typestate (`Struct<F, S=Unfit>` → consuming `Fit::fit` → `Struct<F, Fitted>`), built via `Struct::<F>::builder().<setter>()….build::<F>()` returning `Result` (data-independent validation at build). Use `any_estimator_typestate!` in the wrapper (NOT `any_estimator!`). `[VERIFIED: CODEGRAPH ensemble/*.rs:71-106; LOCAL dispatch.rs:120-185]`
- The wrapper `Unfit` arm stores sklearn-named hyperparameters verbatim; `fit` reads them, builds, calls `TypestateFit::fit`, stores the `Fitted` sibling in `F32`/`F64`. Template: `cluster.rs` `PyHDBSCAN` (typestate, unsupervised) + `naive_bayes.rs` (typestate, supervised fit(x,y)). `[VERIFIED: LOCAL cluster.rs:435-493; naive_bayes.rs]`

### 5.6 Two-tier stochastic gate

RF is bootstrap-stochastic (SplitMix64 ≠ sklearn MT19937), so it uses the milestone two-tier convention: **deterministic tier** (`bootstrap=false`, all features → exact/≤1e-5 train parity) + **statistical tier** (defaults → held-out accuracy/R² band). HGB has no RNG (no bootstrap, no feature subsampling) so its deterministic tier is exact without a stochastic caveat; its statistical tier is a defaults band. The Python oracle harness must replicate BOTH tiers per estimator, exactly as the Rust tests do. `[VERIFIED: LOCAL random_forest_classifier_test.rs:82-193; ensemble/mod.rs:17-37; gen_oracle.py:3413-3428]`

### 5.7 Validation commands (verified against repo, not invented)

- **Rust algos oracle + perf tests** (device backend chosen by feature): `cargo test -p mlrs-algos --release --features wgpu --test hist_gradient_boosting_perf_test -- …` — the exact `cargo test -p mlrs-algos --release --features wgpu` form appears in the perf-test headers and `scripts/bench_rf.py`/`bench_hgb.py`. cpu is the other primary gate: `cargo test -p mlrs-algos --features cpu`. `[VERIFIED: LOCAL scripts/bench_hgb.py:10; crates/mlrs-algos/tests/{random_forest,hist_gradient_boosting}_perf_test.rs:8; compile_fail.rs:31]`
- **PyO3 Rust integration tests:** `cargo test -p mlrs-py --features cpu` (links libpython via the `pyo3/auto-initialize` dev-dependency; do NOT pass `extension-module` for `cargo test`). `[VERIFIED: LOCAL crates/mlrs-py/Cargo.toml dev-dependencies + comments]`
- **Python shim + oracle tests (needs a built wheel):** build in-tree with `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml`, then `pytest crates/mlrs-py/python/tests/`. `[VERIFIED: LOCAL crates/mlrs-py/python/mlrs/__init__.py:104; crates/mlrs-py/pyproject/{cpu,wgpu,cuda,rocm}.pyproject.toml exist]`
- No `justfile`/`Makefile`/`.github` CI in the repo. `[VERIFIED: LOCAL ls — none found]`

### 5.8 Dependency versions (verified from Cargo.toml / Cargo.lock)

- `cubecl 0.10.0` (default-features=false in kernels). `[VERIFIED: LOCAL Cargo.toml; Cargo.lock 0.10.0]`
- `pyo3 0.28` pinned (`0.28.3` resolved) — do NOT bump to 0.29; arrow-59's `pyarrow` feature transitively pins 0.28 and only one PyO3 ABI may link the cdylib. `abi3-py312`. `[VERIFIED: LOCAL Cargo.toml workspace.dependencies; Cargo.lock pyo3 0.28.3; mlrs-py/Cargo.toml]`
- `arrow 59` (`59.0.0`), `bytemuck 1`, `thiserror 2`, `anyhow 1`, `mimalloc 0.1 (local_dynamic_tls)`, `npyz 0.9`. `[VERIFIED: LOCAL Cargo.toml; Cargo.lock arrow 59.0.0]`
- Rust toolchain: `stable` + rustfmt + clippy. `[VERIFIED: LOCAL rust-toolchain.toml]`
- Python ≥3.12 (abi3-py312). Oracle venv: `numpy scipy scikit-learn` (ensemble); `scikit-learn==1.9.0` pin applies to HDBSCAN family only. `[VERIFIED: LOCAL gen_oracle.py:15-17,931]`

---

## 6. Impact & Risk + Open Questions

**Impact scope: cross-module within a single binding layer, external-public (adds four public estimators).** No changes to `mlrs-algos`, `mlrs-backend`, `mlrs-kernels`, or `mlrs-core`. `[INFERRED: all changes land in mlrs-py + its Python package + gen_oracle regen is optional]`

- **Must change:** `estimators/ensemble.rs` (new), `estimators/mod.rs`, `lib.rs`, `python/mlrs/ensemble.py` (new), `python/mlrs/__init__.py`, `python/tests/test_oracle_ensemble.py` (new), `test_params.py`, `test_shims.py`, `test_estimator_checks.py`.
- **May change:** `crates/mlrs-py/tests/*` (add a Rust-side not-fitted/dtype-guard test mirroring `test_naive_bayes.py`).
- **Verification only:** the four `tests/fixtures/{rf,hgb}_*.npz` (consumed, not regenerated) and the Rust ensemble oracle tests (still the primary numeric gate).
- **Explicitly out of scope:** `feature_importances_`, `oob_score_` (not in the Rust core), FIL, TreeSHAP, DecisionTree standalone, ARIMA, SHAP, metrics surface, cuml.accel.

**Migration/compat:** none — purely additive. The `_mlrs` "12 estimators / 30" prose comments in `lib.rs` are stale and should be corrected while editing. `[VERIFIED: LOCAL lib.rs:198-199 "all 12", :254-260 "30"]`

**Risks (trigger → consequence → prevention → verification):**
1. **Wrong monomorphization** — using `any_estimator!` instead of `any_estimator_typestate!` → `F32` arm resolves to `Struct<f32, Unfit>` not `Fitted`, compile error / wrong state. Prevention: use the typestate macro (dispatch.rs:120-185). Verify: `cargo test -p mlrs-py --features cpu` compiles.
2. **HGB default `n_bins=64` vs oracle `n_bins=255`** — the deterministic HGB tier only matches sklearn at `max_bins=255`; constructing the Python estimator with the default 64 breaks exact parity. Prevention: the deterministic oracle test must pass `n_bins=255` explicitly (as the Rust test does). Verify: deterministic-tier Python test.
3. **HGB algos churn (uncommitted prims/kernels/fixtures)** — the HGB prim+kernel+fixture refinement is mid-flight (git status). Pinning Python oracle tolerances against soon-to-change fixtures risks a rebase break. Prevention: land/rebase the HGB algos change and regenerate fixtures BEFORE finalizing HGB Python tolerances; consider sequencing RF bindings first (RF algos are committed/stable). Verify: `git status` clean on `hgb_*` before pinning.
4. **`predict_proba` link-function tolerance** — HGB proba is sigmoid/softmax of raw scores; f32 accumulation may need `atol=1e-4` for f32 (the `_atol` convention in `test_oracle_neighbors.py`). Prevention: reuse `_atol(fixture)` dtype-branch. Verify: statistical-tier proba-sum + ≤tol.
5. **`random_state=None` mapping** — sklearn `random_state=None` must map to a deterministic default seed at the boundary (KMeans maps `None`→`DEFAULT_SEED=0`; RF core default seed is 42). Decide and document the sentinel. `[VERIFIED: LOCAL cluster.rs:27-29,67-72; random_forest_regressor.rs:44]`
6. **Estimator-check sweep** — sklearn `check_estimator` may exercise behaviors the ensemble cores don't support (sparse, NaN); `MlrsBase.__sklearn_tags__` already disables sparse/array-api/NaN, but confirm the ensemble estimators pass or are appropriately xfail'd. Verify: `test_estimator_checks.py`.

**Open questions (Planner must resolve before/at planning):**
- **Q1 (scope):** Expose `predict_log_proba` on the classifiers? The Rust cores do NOT implement `PredictLogProba` for ensembles; options are (a) omit it, (b) compute `log(predict_proba)` host-side in the shim. Owner: Planner + user. `[VERIFIED: CODEGRAPH typestate traits — no PredictLogProba impl on ensembles]`
- **Q2 (scope):** Do we ship `feature_importances_` / `oob_score_` (roadmap Phase 19 criteria) or defer? They are absent from the Rust core, so shipping them requires algos work — recommend **defer** and note the roadmap deviation. Owner: user.
- **Q3 (fixtures):** What exact sklearn version produced the committed ensemble fixtures? Not stamped in-repo; needed to reproduce deterministically if regen is triggered. Owner: Planner (confirm before regen). `[UNVERIFIED]`
- **Q4 (sequencing):** Bind RF first (algos committed) and HGB second (algos in flux)? Recommend yes. Owner: Planner.
- **Q5 (report path):** This report was written to `.planning/plans/RESEARCH.md` per the mission brief, which overrides the generic `./planning/phase/phase-XX-*/research.md` convention. Confirm the Planner reads from `.planning/plans/`. Owner: workflow.

---

## 7. Traceability (symbols + paths cited)

**CodeGraph:**
- `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs` — `RandomForestClassifier::fit` (:366), `predict_proba` (:427), `classes` (:142), `n_classes` (:137), builder setters (:184-229).
- `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs` — struct/defaults (:39-65), `Predict::predict` (:299).
- `crates/mlrs-algos/src/ensemble/hist_gradient_boosting_classifier.rs` — `predict_proba` (:316), `predict_labels` host argmax (:340), `classes`/`n_classes` (:135-142), builder (:181-214).
- `crates/mlrs-algos/src/ensemble/hist_gradient_boosting_regressor.rs` — defaults (:45-50), `Predict::predict` (:330).
- `crates/mlrs-algos/src/ensemble/mod.rs` — `MaxFeatures` enum (:48-71), deviations doc (:17-37).
- `crates/mlrs-algos/src/typestate.rs` — `Fit/Predict/PredictLabels/PredictProba/PredictLogProba` (:143,168,265,346,371).

**Local files:**
- `crates/mlrs-py/src/lib.rs` (:96-158 lock_pool, :176-268 `_mlrs` registration), `dispatch.rs` (:94-185 macros), `estimators/mod.rs` (:31-40), `estimators/naive_bayes.rs` (:71-181), `estimators/cluster.rs` (:35-572 typestate/consuming-fit), `ingress.rs`/`errors.rs`/`capability.rs` (helpers).
- `crates/mlrs-py/python/mlrs/base.py` (:32-183), `naive_bayes.py` (:23-221), `__init__.py` (:22-143).
- `crates/mlrs-py/python/tests/test_oracle_neighbors.py` (:1-70), `test_params.py`, `test_shims.py`.
- `crates/mlrs-algos/tests/random_forest_classifier_test.rs` (:36-297).
- `scripts/gen_oracle.py` (:15-17, :3253-3600 ensemble generators), `scripts/bench_{rf,hgb}.py`.
- `Cargo.toml`, `Cargo.lock`, `crates/mlrs-py/Cargo.toml`, `rust-toolchain.toml`, `.planning/ROADMAP.md` (:78-90).

**Tools used:** CodeGraph `codegraph_explore` (ensemble symbols/blast-radius), local Read/Grep/Bash. Context7/WebSearch not needed (all evidence is in-repo; the sklearn/cuML surface facts are from the repo's own roadmap + gen_oracle). PageIndex MCP not invoked — the repo's own `.planning/*` docs + source were authoritative and directly readable.

---

## 8. Confidence Assessment

- **HIGH:** The four ensemble estimators have Rust cores + oracle tests but no Python surface (directory listings, `lib.rs`, `__init__.py`, empty grep). The binding template and contracts (naive_bayes.rs, dispatch.rs, base.py). Fixture existence + RF fixture keys/tiers. Dependency versions (Cargo.lock). Builder defaults + `1..=16`/`2..=256` bounds + absence of `feature_importances_`/`oob_score_`.
- **MEDIUM:** HGB algos stability (uncommitted prim/kernel/fixture churn in flight). Exact Python oracle tolerances (dtype-dependent `atol`). Whether the regressor float-`predict` composition needs any accessor not already present.
- **LOW / UNVERIFIED:** Exact sklearn version that produced the committed ensemble fixtures (Q3). Whether `check_estimator` fully passes for the ensembles without new xfails (needs a built wheel to exercise).
