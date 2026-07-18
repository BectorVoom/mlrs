---
title: PY-ENSEMBLE — Python bindings for RandomForest/HistGradientBoosting, plus feature_importances_/oob_score_
status: draft
format: markdown
spec_version: 1
spec_revision: 2
updated_at: 2026-07-18T00:00:00Z
source_requirements:
  - "User request: implement features in cuML/sklearn not yet in mlrs (coverage-gap fill)"
  - ".planning/plans/RESEARCH.md — full gap survey, PY-ENSEMBLE ranked #1 unblocked gap"
  - ".planning/plans/py-ensemble/research.md — 2026-07-17 verification pass, reconfirms gap + flags HGB churn"
locked_decisions:
  - "Target feature: PY-ENSEMBLE (Python bindings for the four already-landed Rust ensemble cores), chosen over other roadmap gaps per RESEARCH.md ranking"
  - "Sequencing: ONE plan/spec covers all four estimators (RF classifier/regressor + HGB classifier/regressor); the binding-layer tasks for all four proceed together, but the task that PINS HGB's Python oracle-test numeric tolerances is explicitly gated on the HGB algos churn (hist_gradient_boosting.rs/gbt.rs/gen_oracle.py/hgb_*.npz) landing as a clean commit first — user-confirmed"
  - "predict_log_proba: OMITTED for both classifiers — the Rust cores implement no PredictLogProba; do not add a shim-only log(predict_proba) workaround — user-confirmed"
  - "feature_importances_ / oob_score_: INCLUDED in this plan's scope, for RandomForestClassifier/RandomForestRegressor ONLY — user explicitly chose to expand scope beyond the original binding-only recommendation, accepting that this requires new mlrs-algos + mlrs-kernels work, not just a binding-layer change — user-confirmed"
  - "HGB oracle-fixture dirty state: the task that finalizes HGB Python oracle-test tolerances REQUIRES a clean `git status` on hist_gradient_boosting.rs/gbt.rs/gen_oracle.py/hgb_*.npz before it may be marked done — proceeding against the currently-dirty fixtures is NOT permitted — user-confirmed"
pageindex_update: "NOT APPLICABLE — no mlrs project PageIndex specification document exists to update (consistent with metrics-surface/SPEC.md precedent); this SPEC.md is the authoritative local draft."
---

# PY-ENSEMBLE — Draft Specification

> Draft. Nothing here is approved/implemented. Feeds the Planner Agent (PLAN.md) and Plan Checker gate.
> Evidence labels: `[VERIFIED: CODEGRAPH …]` `[VERIFIED: LOCAL …]` `[VERIFIED: WEB …]` `[INFERRED: …]` `[UNVERIFIED: …]`.
> Full evidence in companion `research.md` (this folder) and `../RESEARCH.md` (original gap survey).

## 1. Context

Four ensemble estimators — `RandomForestClassifier`, `RandomForestRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` — have complete Rust cores in `crates/mlrs-algos/src/ensemble/` with builder validation, typestate `Fit`/`Predict`/`PredictLabels`/`PredictProba` trait implementations, and full Rust oracle tests against committed `.npz` fixtures (landed in commit `fb0c9c7`), but **zero Python-facing surface**: no `crates/mlrs-py/src/estimators/ensemble.rs`, no `_mlrs` pymodule registration, no Python shim module, no `__init__.py` export. `grep -rniE "randomforest|histgradient" crates/mlrs-py` returns nothing. `[VERIFIED: LOCAL crates/mlrs-py/src/estimators/ listing; research.md §a.4]`

The binding-layer gap is a mechanical, low-risk completion: byte-for-byte binding templates already exist (`estimators/naive_bayes.rs` + `python/mlrs/naive_bayes.py` for the typestate-fit-with-y classifier shape; `linear.py` for the float-`predict` regressor shape), the `any_estimator_typestate!` macro handles the `S=Unfit`-default monomorphization trap, and all eight `.npz` fixtures are already committed. `[VERIFIED: LOCAL research.md §b, §5.2-5.3 of ../RESEARCH.md]`

During spec scoping the user chose to **additionally** add `feature_importances_` and `oob_score_` to `RandomForestClassifier`/`RandomForestRegressor` — sklearn/cuML attributes the Rust core does **not** currently compute (`grep -n "feature_importances\|oob_score" -r crates/mlrs-algos/src/ensemble/` → zero hits `[VERIFIED: LOCAL research.md §b.5.1]`). This is **not** a pure binding-layer change: it requires new `mlrs-backend`/`mlrs-kernels` computation. `HistGradientBoostingClassifier`/`Regressor` are excluded from this part of scope because sklearn's own `HistGradientBoostingClassifier`/`Regressor` do not expose `feature_importances_` or `oob_score_` (boosting is not a bagging/OOB scheme in sklearn's HGB implementation) `[INFERRED: well-established scikit-learn public API shape — HGB estimators have no such attributes, unlike GradientBoostingRegressor]`.

The `mlrs.metrics` module landed today (`0788e17`) is confirmed additive and does not touch anything this spec depends on (`estimators/mod.rs`, `dispatch.rs`, `naive_bayes.rs`, `Cargo.toml`/`Cargo.lock` all show zero diff from that commit). `[VERIFIED: LOCAL research.md §a.1, §a.6-7]`

## 2. Scope and Non-Goals

### In scope

**Unit 1 — Python bindings (binding-layer only, all four estimators):**
- `RandomForestClassifier`, `RandomForestRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` `#[pyclass]` wrappers, Python shim classes, `_mlrs` registration, `__init__.py` exports.
- `fit(X, y)`, `predict` (regressors), `predict_proba`/`predict` via `predict_labels` (classifiers), hyperparameter mapping (sklearn-named constructor args → Rust builder setters), `max_features` string/int/float/None parsing.
- Full oracle test replay of the eight already-committed `.npz` fixtures through the Python path (second consumer, no regeneration for RF; HGB gated — see below).
- Estimator-enumerating gate test updates: `test_params.py`, `test_shims.py`, `test_estimator_checks.py`.
- Stale prose-comment correction in `lib.rs`/`estimators/mod.rs` ("12 estimators" / "30").

**Unit 2 — `feature_importances_` / `oob_score_` (RandomForest only, algos + kernel + binding):**
- Rust-core computation of sklearn-equivalent normalized impurity-based `feature_importances_` (length-`n_features` vector, sums to 1, mean over trees of weighted impurity decrease per feature) for both `RandomForestClassifier` and `RandomForestRegressor`.
- Rust-core computation of `oob_score_` at fit time: a new `oob_score: bool` builder setter (default `false`, sklearn-named); when `true`, requires `bootstrap == true` (else a build-time or fit-time error, sklearn raises `ValueError("Out of bag estimation only available if bootstrap=True")`); when computed, aggregates only the trees where each sample was NOT drawn into that tree's bootstrap sample, then scores via accuracy (classifier) or R² (regressor) against the training labels.
- Python bindings exposing `feature_importances_` (fitted property, both estimators) and `oob_score`/`oob_score_` (constructor arg + fitted property, both estimators).

### Non-goals (explicitly out)

- `predict_log_proba` on either classifier — **user-locked decision**, no Rust core support exists.
- `oob_decision_function_` / `oob_prediction_` (sklearn's per-sample OOB-averaged proba/prediction array) — only the scalar `oob_score_` is in scope; the user's scope-expansion decision named `oob_score_` specifically, not the full OOB attribute family. `[INFERRED: narrowest reading of the locked decision — flag for Planner/user to confirm before implementation if broader OOB surface is wanted]`
- `sample_weight` on `fit()` for any of the four estimators — no estimator `fit()` method anywhere in `crates/mlrs-py/python/mlrs/*.py` currently accepts `sample_weight` (grep across all 30 `fit(...)` definitions confirms zero instances), and `RfParams`/HGB builder have no such field. Follow the established project precedent: **omit the parameter from the signature entirely** (do not accept-then-raise; that pattern is metrics-specific, not an estimator-layer precedent). `[VERIFIED: LOCAL grep -n "def fit" crates/mlrs-py/python/mlrs/*.py — 30 matches, none take sample_weight]`
- `class_weight` — absent from `RfParams`, no Rust core support.
- `feature_importances_` / `oob_score_` for `HistGradientBoostingClassifier`/`Regressor` — sklearn's own HGB classes do not expose these attributes; out of scope by API-shape non-applicability, not by deferral.
- DecisionTree standalone estimator, FIL, TreeSHAP, ARIMA, cuml.accel — unrelated roadmap phases, untouched.
- Device-kernel algorithmic changes to HGB (`hist_gradient_boosting.rs`, `gbt.rs`) beyond what is already in flight — this spec consumes that work once committed; it does not modify it.
- Regenerating the RF `.npz` fixtures — RF algos/fixtures are stable and untouched since `fb0c9c7`; no regen needed.

## 3. Dependencies

**Rust ensemble cores (reuse, do not reimplement fit/predict math):**
- `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs` — `RandomForestClassifier<F,S>`, `fit` (:366-419), `predict_proba` (:427-440), `classes()`/`n_classes()` (:137-144), builder (:165-265). `[VERIFIED: CODEGRAPH]`
- `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs` — `RandomForestRegressor<F,S>`, `fit` (:244-291), `predict` (:299-312), builder (:140-235). `[VERIFIED: CODEGRAPH]`
- `crates/mlrs-algos/src/ensemble/hist_gradient_boosting_classifier.rs` / `hist_gradient_boosting_regressor.rs` — same typestate shape, defaults `max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64, l2=0.0, min_samples_leaf=20`. `[VERIFIED: research.md §b, ../RESEARCH.md §5.1]`
- `crates/mlrs-algos/src/ensemble/mod.rs` — `MaxFeatures` enum (`Sqrt|Log2|All|Value(usize)`, `resolve(n_features)`), shared `validate_forest_hyperparams`, `ingest_labels`. `[VERIFIED: LOCAL]`
- `crates/mlrs-algos/src/typestate.rs` — `Fit`, `Predict`, `PredictLabels`, `PredictProba` traits (NOT `PredictLogProba` for ensembles). `[VERIFIED: LOCAL grep]`

**Backend prim + kernel layer (Unit 2 dependency — feature_importances_/oob_score_ only):**
- `crates/mlrs-backend/src/prims/random_forest.rs` — `rf_fit_impl` (:430-837, the shared launch-only level-loop fit driver), `RfModel<F>` (:89-148, currently: `split_feature`, `threshold`, `is_leaf`, `leaf_dist`, `n_trees`, `max_depth`, `total_nodes`, `n_features`, `n_values` — **no per-node impurity-decrease or sample-count field, no persisted bootstrap mask**), `RfParams` (:59-83), `bootstrap_weights` (:379-397, host `SplitMix64`-seeded with-replacement draw producing an `n_trees × n` per-sample multiplicity array — `weight == 0` for tree `t`/sample `i` means sample `i` is OUT-OF-BAG for tree `t`; this array is currently transient and discarded after upload, never persisted). `[VERIFIED: LOCAL Read random_forest.rs:56-148,379-397,430-837]`
- `crates/mlrs-kernels/src/tree.rs` — the device kernels the level-loop launches: `rf_bin_features`, `rf_hist_class`/`rf_hist_reg`, `rf_hist_cum`, `rf_node_total`, `rf_node_max`, `rf_split_scores_class`/`rf_split_scores_reg` (produces a per-candidate impurity-decrease-like score, consumed by K6), `rf_best_split` (K6 — picks the winning candidate, writes `split_feature`/`threshold`/`is_leaf`/`leaf_dist`; the WINNING score value and the node's weighted sample total are computed inside this kernel but **not currently written to any output array**), `rf_count_left`, `rf_partition`. `[VERIFIED: LOCAL random_forest.rs:558-812 launch call sites enumerate every kernel + its current parameter list; kernel BODIES live in crates/mlrs-kernels/src/tree.rs — file located but not read in this pass, so exact CubeCL kernel signatures/internals are UNVERIFIED and must be confirmed by the Planner before task-level design]`
- Neither `RandomForestClassifier<F,Fitted>` nor `RandomForestRegressor<F,Fitted>` currently retains the training `x`/`y` device arrays after `fit` returns (unlike e.g. `KernelDensity`'s `x_fit_`) — so `oob_score_` **must** be computed inside the `fit` call, before `x`/`y` go out of scope; it cannot be a lazily-computed post-fit accessor. `[VERIFIED: LOCAL — Fitted-state field lists for both structs contain no x_fit_/y_fit_]`

**Binding-layer templates (Unit 1 — byte-for-byte precedent to mirror):**
- `crates/mlrs-py/src/estimators/naive_bayes.rs` (1061 lines) — typestate classifier binding: `any_estimator_typestate!`, `fit(x,y,rows,cols)` with a `y` capsule, `classes_()` getter, `predict_labels`, `predict_proba_f32`/`_f64`, `is_fitted`, `dtype`. `[VERIFIED: LOCAL research.md §b.§5.3]`
- `crates/mlrs-py/src/dispatch.rs` (:90-190) — `any_estimator!` (WRONG for ensembles, resolves `S=Unfit`) vs `any_estimator_typestate!` (correct — explicit `S=Fitted` arm). Doc-comment warns about this exact trap. `[VERIFIED: LOCAL]`
- `crates/mlrs-py/python/mlrs/naive_bayes.py` (220 lines), `crates/mlrs-py/python/mlrs/base.py` (182 lines, `MlrsBase`/`ClassifierMixin`/`RegressorMixin`) — Python shim template. `[VERIFIED: LOCAL]`
- `crates/mlrs-py/src/estimators/cluster.rs` `PyHDBSCAN` (:435-493) — typestate unsupervised-fit consuming pattern, second reference alongside naive_bayes.rs. `[VERIFIED: LOCAL]`
- `crates/mlrs-py/python/tests/test_oracle_neighbors.py` — oracle-replay harness pattern (`_atol` dtype branch, `@requires_f64` marker, fixture load via `np.load`). `[VERIFIED: LOCAL]`
- `crates/mlrs-py/python/mlrs/*.py` — 30 existing `def fit(...)` signatures, confirming the no-`sample_weight` precedent (§2). `[VERIFIED: LOCAL grep]`

**Fixtures:**
- `tests/fixtures/{rf,hgb}_{cls,reg}_{f32,f64}_seed42.npz` (8 files, already committed). RF fixture keys: `X, y, Xq, yq, det_pred_train, det_proba_train, stat_acc_test`; geometry 96 train / 48 test / 5 features / 3 classes; deterministic tier `bootstrap=false, max_features=All, depth=12, n_estimators=2` (exact match ≤1e-5); statistical tier `n_estimators=64, depth=8` (`ACC_MARGIN=0.05` held-out band). `[VERIFIED: LOCAL crates/mlrs-algos/tests/random_forest_classifier_test.rs:36-192, reconfirmed unchanged in research.md §b.§5.4]`
- HGB deterministic tier requires constructing with `n_bins=255` (not the Python default 64) to match sklearn's `max_bins=255`. `[VERIFIED: LOCAL research.md §b.§5.4]`
- **HGB fixture state (blocking precondition for Unit 1's HGB oracle-finalization tasks only):** `crates/mlrs-backend/src/prims/hist_gradient_boosting.rs`, `crates/mlrs-kernels/src/gbt.rs`, `scripts/gen_oracle.py`, and all four `hgb_*.npz` are uncommitted and actively changing (sibling-histogram-subtraction kernel + float-noise tie-margin `rng_offset` tuning still in progress as of 2026-07-17). `[VERIFIED: LOCAL research.md §a.2, git status --short]`

**Dependency versions (unchanged, no new external dependency needed for this feature):** `pyo3 0.28.3` (pinned, do not bump), `arrow 59.0.0`, `cubecl 0.10.0`, `abi3-py312`, Python ≥3.12. `[VERIFIED: LOCAL Cargo.lock; research.md §b.§5.7-5.8]`

## 4. Typed Contracts

### 4.1 Rust algos layer — new/changed surface

```rust
// crates/mlrs-algos/src/ensemble/random_forest_classifier.rs / random_forest_regressor.rs
// Builder additions (both):
impl RandomForestClassifierBuilder /* and RandomForestRegressorBuilder */ {
    pub fn oob_score(mut self, v: bool) -> Self;   // sklearn-named, default false
}
// validate_forest_hyperparams (mod.rs) gains a bootstrap/oob_score cross-check:
//   oob_score == true && bootstrap == false  =>  Err(BuildError::OobRequiresBootstrap { estimator })
//   (new BuildError variant; data-INDEPENDENT check, validated at build() per D-08)

impl<F> RandomForestClassifier<F, Fitted> /* and RandomForestRegressor<F, Fitted> */ {
    pub fn feature_importances(&self) -> &[F];   // length n_features(), sums to 1.0 (within float tolerance)
    pub fn oob_score(&self) -> Option<F>;         // Some(..) iff builder oob_score==true; None otherwise
}
```

- `feature_importances()`: `AlgoError`/panic-free — always populated on any `Fitted` instance regardless of `oob_score` (feature_importances_ has no `bootstrap`/`oob_score` precondition in sklearn — it is always computed). Values are the sklearn-equivalent normalized mean-decrease-in-impurity, matching `sklearn.ensemble._forest.py`'s exact aggregation: for each tree, attribute each non-leaf split node's `weighted_n_node_samples * impurity_decrease` to that node's `split_feature`, then **normalize that tree's vector to sum to 1 individually**; **average** the per-tree normalized vectors over the trees that actually split (`S_t > 0`); then renormalize the mean to sum to 1. This per-tree-then-average scheme is NOT the same as a single global normalization `Σ_t d_{t,f} / Σ_t S_t` whenever the per-tree totals `S_t` differ (as they do under `bootstrap=true`, the default) — line 49's "mean over trees" is authoritative; any earlier "sum across all nodes/trees" phrasing was a defect corrected during implementation (code-review finding, 2026-07-18). Degenerate case: if every tree is a single leaf (no split-bearing tree, `Σ_t S_t == 0`), return an all-zero vector rather than dividing by zero (matching sklearn's zeros return). The all-zero guard is unconditionally correct regardless of whether the tree-growth loop early-stops before `max_depth`.
- `oob_score()`: `None` when `oob_score` builder flag is `false` (the common case — avoids paying the OOB-aggregation cost when unrequested). `Some(score)` computed once at `fit`-time when `true`: classifier score = accuracy of `argmax(mean OOB-tree class distribution)` vs training `y`; regressor score = R² of `mean OOB-tree prediction` vs training `y`. **TBD (Planner + user before implementation):** sklearn's convention when a sample has ZERO out-of-bag trees (possible with small `n_estimators`/large `n`) — sklearn skips that sample from the score and emits a `UserWarning`. mlrs has no Python-visible warning channel established for estimators; Planner must decide whether to (a) skip such samples silently, (b) skip and log via `log::warn!` (host-side, matches existing project logging conventions), or (c) return an `AlgoError` variant. Recommend (b) for parity with sklearn's non-fatal behavior — **owner: Planner + user**.

### 4.2 Backend prim layer — new/changed surface (Unit 2 only; exact kernel signatures TBD, see §3)

```rust
// crates/mlrs-backend/src/prims/random_forest.rs
pub struct RfModel<F> {
    // existing fields unchanged, PLUS (exact placement TBD — see below):
    // per-node winning split-score (impurity decrease) and per-node weighted
    // sample total, OR a pre-reduced length-n_features importances vector
    // computed once inside rf_fit_impl before RfModel is returned.
}
pub struct RfFitOutcome<F> {   // NEW — name TBD; wraps RfModel + the two new fit-time-only outputs
    pub model: RfModel<F>,
    pub feature_importances: Vec<F>,      // length n_features, sums to 1
    pub oob_score: Option<F>,             // Some iff params.oob_score
}
pub struct RfParams {
    // existing fields unchanged, PLUS:
    pub oob_score: bool,   // default false; caller (estimator layer) already validated oob_score=>bootstrap at build()
}
```

This is a **contract sketch for the Planner**, not a final signature — the exact mechanism (extend `RfModel` in place vs. wrap it in a new return type; whether `rf_best_split`'s kernel signature grows two new output arrays vs. a separate reduction pass) requires reading `crates/mlrs-kernels/src/tree.rs` (unread in this research pass) and is delegated to the Planner + implementer, verified via CodeGraph before task-level commitment. `[UNVERIFIED: exact kernel-level mechanism]`

### 4.3 PyO3 + Python shim layer

```rust
// crates/mlrs-py/src/estimators/ensemble.rs (NEW)
#[pyclass] struct PyRandomForestClassifier { /* Any<RandomForestClassifier> state enum, per dispatch.rs pattern */ }
// #[pymethods]: #[new](n_estimators=100, max_depth=10, n_bins=32, max_features="sqrt",
//   min_samples_split=2, min_samples_leaf=1, bootstrap=True, oob_score=False, seed=42)
// fit(x_capsule, y_capsule, rows, cols) -> PyResult<()>
// classes_(&self) -> Vec<i32>; predict_labels(...) -> ...; predict_proba_f32/_f64(...) -> ...
// feature_importances_f32/_f64(&self) -> PyResult<Vec<f32|f64>>
// oob_score_(&self) -> PyResult<Option<f32|f64>>
// is_fitted(&self) -> bool; dtype(&self) -> &str
// (mirror shape for PyRandomForestRegressor minus classes_/predict_proba, plus predict_f32/_f64)
// (PyHistGradientBoostingClassifier / PyHistGradientBoostingRegressor: same shape minus
//  feature_importances_/oob_score_ entirely — not applicable, §2 non-goals)
```

```python
# crates/mlrs-py/python/mlrs/ensemble.py (NEW)
class RandomForestClassifier(ClassifierMixin, MlrsBase):
    def __init__(self, n_estimators=100, max_depth=10, n_bins=32, max_features="sqrt",
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True, oob_score=False,
                 seed=42, output_type="input"): ...
    def fit(self, X, y): ...                       # no sample_weight — §2
    def predict(self, X): ...
    def predict_proba(self, X): ...
    @property
    def feature_importances_(self): ...             # raises AttributeError/NotFittedError pre-fit
    @property
    def oob_score_(self): ...                        # raises ValueError-equivalent if oob_score=False and accessed? TBD — see §5

class RandomForestRegressor(RegressorMixin, MlrsBase): ...   # predict-only, same feature_importances_/oob_score_ shape
class HistGradientBoostingClassifier(ClassifierMixin, MlrsBase): ...   # no feature_importances_/oob_score_
class HistGradientBoostingRegressor(RegressorMixin, MlrsBase): ...     # no feature_importances_/oob_score_
```

Defaults in `__init__` MUST equal the `#[new]` defaults, which MUST equal the Rust builder defaults (single-source rule, D-08/D-02 — same discipline as every existing shim). `[VERIFIED: LOCAL naive_bayes.py:13-14 doc precedent]`

## 5. Failure-Isolated Behavioral Specifications

### PY-ENS-01 — RandomForestClassifier Python binding
- **Status:** draft.
- **Rationale/source:** `../RESEARCH.md` §5, this file §1.
- **Preconditions:** `mlrs._mlrs` extension built (`maturin develop`); RF Rust core unchanged since `fb0c9c7` (confirmed stable).
- **Input:** `X: array-like (n_samples, n_features)`, `y: array-like (n_samples,)` integer-valued labels; constructor hyperparameters per §4.3.
- **Output:** fitted `mlrs.ensemble.RandomForestClassifier`; `predict(X) -> ndarray[int]`, `predict_proba(X) -> ndarray[float] (n_query, n_classes)` rows sum to 1; `classes_ : ndarray[int]`.
- **Dependencies:** `RandomForestClassifier<F,Fitted>` (typed interface: `Fit`, `PredictLabels`, `PredictProba` traits, §3).
- **Given/When/Then:**
  - Given valid `X`/`y` and default hyperparameters, when `.fit(X, y).predict(X)` is called on the deterministic-tier fixture inputs, then predictions match the committed `det_pred_train` fixture values exactly (`bootstrap=False` branch, ≤1e-5 for `predict_proba`).
  - Given the statistical-tier fixture (`n_estimators=64, depth=8`, defaults), when `.fit(X,y).predict(Xq)` is called, then held-out accuracy is within `ACC_MARGIN=0.05` of `stat_acc_test`.
  - Given `max_features="sqrt"|"log2"|<float>|<int>|None`, when constructed, then it maps to `MaxFeatures::{Sqrt,Log2,Value,All}` without error; given an invalid string (e.g. `"bogus"`), then construction/fit raises `ValueError`.
  - Given `.predict()`/`.predict_proba()` called before `.fit()`, then raises the project's standard not-fitted error (mirror `naive_bayes.py`'s pattern).
  - Given `y` containing non-integer-valued floats or out-of-i32-range values, then `.fit()` raises `ValueError` (mirrors `ingest_labels` `AlgoError` → `algo_err_to_py`).
- **Invariants/side effects:** GIL released during the device fit/predict call (`py.detach`); `crate::lock_pool()` sanctioned-lock only; f64 path calls `guard_f64()` before upload.
- **Acceptance tests:** Rust-side not-fitted/dtype-guard test (mirror `test_naive_bayes.py`); Python `test_oracle_ensemble.py::test_random_forest_classifier_deterministic` + `::test_random_forest_classifier_statistical` (f32 + f64, f64 behind `@requires_f64`); `test_params.py`/`test_shims.py` entries.
- **Out of scope:** `predict_log_proba`, `sample_weight`, `class_weight`, `feature_importances_`/`oob_score_` (those are PY-ENS-driven-by-RF-IMP-02/RF-OOB-02, not this spec).
- **Traceability:** `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/ensemble/random_forest_classifier.rs:56-440]` `[VERIFIED: LOCAL crates/mlrs-algos/tests/random_forest_classifier_test.rs:36-192]`
- **Unresolved questions:** None blocking — RF algos/fixtures fully stable.

### PY-ENS-02 — RandomForestRegressor Python binding
- **Status:** draft.
- **Rationale/source:** same as PY-ENS-01, regressor counterpart.
- **Preconditions:** same as PY-ENS-01.
- **Input:** `X`, `y: array-like (n_samples,)` continuous target; same constructor shape as PY-ENS-01 minus classifier-only args.
- **Output:** fitted `mlrs.ensemble.RandomForestRegressor`; `predict(X) -> ndarray[float]` (length `n_query`, forest-mean of reached-leaf means).
- **Dependencies:** `RandomForestRegressor<F,Fitted>` (`Fit`, `Predict` traits).
- **Given/When/Then:**
  - Given the deterministic-tier RF regressor fixture, when `.fit(X,y).predict(X)` is called, then predictions match the committed fixture exactly (≤1e-5).
  - Given the statistical-tier fixture, when `.fit(X,y).predict(Xq)` is called, then held-out R²/error is within the fixture's documented statistical band.
  - Given the same `max_features`/invalid-input/not-fitted cases as PY-ENS-01 (regressor default `max_features="all"` not `"sqrt"`), same error behavior.
- **Invariants/side effects:** same GIL/lock/f64-guard contract as PY-ENS-01.
- **Acceptance tests:** mirrors PY-ENS-01's test shape for the regressor fixture.
- **Out of scope:** same exclusions as PY-ENS-01.
- **Traceability:** `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/ensemble/random_forest_regressor.rs:48-312]`
- **Unresolved questions:** None blocking.

### PY-ENS-03 — HistGradientBoostingClassifier Python binding
- **Status:** draft.
- **Rationale/source:** `../RESEARCH.md` §5, this file §1.
- **Preconditions:** **BLOCKING for the oracle-tolerance-finalization sub-task only** (not for writing the `#[pyclass]`/shim structure itself, which is mechanically identical to PY-ENS-01): `git status` must be clean on `crates/mlrs-backend/src/prims/hist_gradient_boosting.rs`, `crates/mlrs-kernels/src/gbt.rs`, `scripts/gen_oracle.py`, and all four `tests/fixtures/hgb_*.npz` — i.e., the sibling-histogram-subtraction work is committed and the fixtures are final — **user-locked decision, §Frontmatter**.
- **Input:** `X`, `y` integer labels; constructor hyperparameters `max_iter, learning_rate, max_depth, n_bins, l2_regularization, min_samples_leaf` (defaults `100, 0.1, 6, 64, 0.0, 20`).
- **Output:** fitted `mlrs.ensemble.HistGradientBoostingClassifier`; `predict`/`predict_proba` (sigmoid/softmax link, rows sum to 1, host argmax with documented strict-`>` lowest-index tie-break).
- **Dependencies:** `HistGradientBoostingClassifier<F,Fitted>` (`Fit`, `PredictLabels`, `PredictProba`).
- **Given/When/Then:**
  - Given the deterministic-tier fixture WITH `n_bins=255` explicitly set (not the class default 64), when `.fit(X,y).predict(X)`/`.predict_proba(X)` is called, then it matches the committed fixture exactly, **once the HGB fixture-freshness precondition above is satisfied**.
  - Given the statistical-tier fixture (class defaults), when `.fit(X,y).predict(Xq)` is called, then held-out accuracy/proba-sum-to-1 is within the fixture's documented band.
  - Given constructor/not-fitted/invalid-input error cases, same shape as PY-ENS-01.
- **Invariants/side effects:** same GIL/lock/f64-guard contract.
- **Acceptance tests:** `test_oracle_ensemble.py::test_hgb_classifier_deterministic` (SKIPPED or xfail-with-reason until the fixture-freshness precondition is met — Planner must decide the exact test-suite-green mechanism; do not silently pin against dirty fixtures) + `::test_hgb_classifier_statistical`.
- **Out of scope:** `predict_log_proba`, `feature_importances_`/`oob_score_` (not applicable to HGB), `sample_weight`, `early_stopping` (already absent from the Rust core per `../RESEARCH.md` §5.1).
- **Traceability:** `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/ensemble/hist_gradient_boosting_classifier.rs (via ../RESEARCH.md §5.1, reconfirmed research.md §b)]`
- **Unresolved questions:** Exact sklearn version used to produce the (soon-to-be-regenerated) HGB fixtures is unstamped in-repo — Planner should stamp it when the fixtures are finally regenerated/committed (research.md Q3/Q6).

### PY-ENS-04 — HistGradientBoostingRegressor Python binding
- **Status:** draft.
- Same shape as PY-ENS-03 minus classes/proba, plus float `predict` (raw ensemble scores: baseline mean + shrunk leaf sums), same HGB-fixture-freshness precondition on the oracle-finalization sub-task.
- **Traceability:** `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/ensemble/hist_gradient_boosting_regressor.rs (via ../RESEARCH.md §5.1)]`
- **Unresolved questions:** same as PY-ENS-03.

### PY-ENS-05 — Estimator registration + gate-test updates
- **Status:** draft.
- **Rationale/source:** every existing estimator is enumerated by three cross-cutting Python test files; a new estimator that doesn't update them either fails those tests or is silently excluded from their coverage.
- **Preconditions:** PY-ENS-01..04 `#[pyclass]`/shim code exists (this spec wires it up, does not create new estimator behavior).
- **Input/Output:** N/A — this is a registration/wiring spec, not a runtime-behavior spec.
- **Dependencies:** `crates/mlrs-py/src/estimators/mod.rs` (add `pub mod ensemble;`), `crates/mlrs-py/src/lib.rs` (4 new `use` + `add_class::<Py...>()?` calls, registration count 32→36; correct the stale "12 estimators"/"30" doc comments — `[VERIFIED: LOCAL research.md §a.5-6, grep -c add_class → 32]`), `crates/mlrs-py/python/mlrs/__init__.py` (import + `__all__`).
- **Given/When/Then:** Given the four new estimators are registered, when `test_params.py`'s AST purity gate + per-estimator `get_params`/mutation tables run, then all four pass; when `test_shims.py`'s mixin/attribute enumeration runs, then all four are covered; when `test_estimator_checks.py`'s sklearn `check_estimator` sweep runs, then all four pass or are explicitly xfail'd with a documented reason (e.g. sparse/NaN input rejection, matching `MlrsBase.__sklearn_tags__`).
- **Acceptance tests:** the three gate-test files themselves, extended with four new entries each.
- **Out of scope:** changing the gate-test *mechanism* — only adding entries for the new estimators.
- **Traceability:** `[VERIFIED: LOCAL research.md §a.8, §b.§5.2]`
- **Unresolved questions:** whether `check_estimator` needs new xfails for the ensemble estimators — unknown until the wheel is built and the sweep is actually run; Planner should treat this as a verification task, not assume pass.

### RF-IMP-01 — `feature_importances_` Rust-core computation (algos + prim + kernel)
- **Status:** draft.
- **Rationale/source:** user-locked scope expansion, §Frontmatter; sklearn `RandomForestClassifier.feature_importances_`/`RandomForestRegressor.feature_importances_`.
- **Preconditions:** a fitted `RandomForestClassifier<F,Fitted>` or `RandomForestRegressor<F,Fitted>` (always computed, no opt-in flag — matches sklearn, where `feature_importances_` has no constructor gate).
- **Input:** none beyond the already-fitted model state (no new fit-time input — this is a computation performed FROM data already flowing through the existing `rf_fit_impl` level loop, per §3/§4.2).
- **Output:** `&[F]` (or `Vec<F>`) of length `n_features`, values `>= 0`, summing to `1.0` within float tolerance (all-zero permitted in the degenerate all-leaf-forest case — see §4.1 TBD note).
- **Dependencies:** `crates/mlrs-backend/src/prims/random_forest.rs::rf_fit_impl` (must be extended to retain/reduce the per-node winning-split score + weighted sample count that `rf_best_split` (K6, `crates/mlrs-kernels/src/tree.rs`) already computes internally but currently discards); typed interface TBD pending Planner's read of `tree.rs` (§3).
- **Given/When/Then:**
  - Given a forest fitted on data with an obviously dominant feature (e.g. one feature perfectly separates classes, others are noise), when `feature_importances()` is read, then the dominant feature's importance is materially larger than the noise features' (a qualitative acceptance test, not an exact-match oracle — sklearn's own impurity-importance has no closed-form cross-implementation exact-match guarantee at the 1e-5 level because tie-breaking/split-order can differ even between equivalent trees; **Planner must decide the acceptance tolerance strategy** — recommend a qualitative ranking/ratio assertion plus an exact-match assertion ONLY on the deterministic `bootstrap=false, max_features=All` fixture tier where sklearn and mlrs are already proven to build IDENTICAL trees).
  - Given the deterministic-tier fixture (where mlrs and sklearn trees are already proven structurally identical per the existing oracle tests, in the sense that `predict`/`predict_proba` match exactly), when both `sklearn`'s and mlrs's `feature_importances_` are computed on the same fitted data, then they match within `atol=0.05` (absolute, per-feature) — **NOT** `≤1e-5`. **[RESOLVED at TASK-02 Green time, 2026-07-18 — supersedes the original ≤1e-5 claim above, kept for history]:** `predict`/`predict_proba` exact-match only proves outcome-equivalence, not split-choice-equivalence — sklearn's Cython "best" splitter breaks near-tied candidate splits using internal state independent of the public `random_state`-controlled bootstrap/feature-subsample streams, so sklearn's own two deterministic-tier trees are themselves NOT bit-identical to each other (confirmed empirically: `det.estimators_[0].feature_importances_ != det.estimators_[1].feature_importances_` on sklearn's own output), even though mlrs's two trees ARE bit-identical (zero RNG consumed at `bootstrap=false, max_features=All`). A genuine tied-split divergence was observed at a low-sample deep node (`feature_3<=0.033` mlrs vs. `feature_3<=0.10` sklearn — both valid, equally-scoring splits), producing a ~0.0022 per-feature divergence in `feature_importances_`, ~200x past `1e-5`/`1e-4`. `atol=0.05` (25x the observed divergence) is tight enough to catch a real attribution bug while tolerant of legitimate tie-break disagreement. The qualitative dominant-feature ranking assertion (below) remains the PRIMARY correctness signal for this spec, not a fallback.
- **Invariants/side effects:** no new host readback beyond what `rf_fit_impl` already performs, UNLESS the chosen mechanism (§4.2 TBD) requires one; Planner must confirm read-back-count impact against the FOUND-05/D-10 memory-conservation gate (`crates/mlrs-backend/src/pool.rs` `PoolStats.read_backs`).
- **Acceptance tests:** new Rust oracle test file (e.g. `crates/mlrs-algos/tests/random_forest_feature_importances_test.rs`), generated via `scripts/gen_oracle.py` extension (new fixture keys, e.g. `feature_importances_expected`), gated on the same deterministic/statistical two-tier convention as the rest of RF.
- **Out of scope:** HGB (not applicable), `oob_score_` (RF-OOB-01, separate spec — different mechanism, different failure mode).
- **Traceability:** `[VERIFIED: LOCAL crates/mlrs-backend/src/prims/random_forest.rs:670-751 (K5/K6 launch sites)]` `[UNVERIFIED: crates/mlrs-kernels/src/tree.rs kernel bodies — Planner must read before task-level design]`
- **Unresolved questions:** (1) exact reduction mechanism (kernel-side output arrays vs. host-side re-derivation) — owner Planner, verify via CodeGraph on `tree.rs`; (2) all-zero-importances degenerate case handling — owner Planner; (3) acceptance-tolerance strategy for the non-deterministic-tier fixture — owner Planner + user.

### RF-IMP-02 — `feature_importances_` Python binding
- **Status:** draft.
- **Rationale/source:** exposes RF-IMP-01's Rust-core output through the Python shim.
- **Preconditions:** RF-IMP-01 implemented; PY-ENS-01/02 binding scaffolding exists.
- **Input:** none (property read on a fitted estimator).
- **Output:** `feature_importances_: np.ndarray[float]` (dtype matches the fitted estimator's dtype, shape `(n_features,)`).
- **Dependencies:** RF-IMP-01's Rust accessor; `crate::estimators::ensemble::PyRandomForestClassifier`/`PyRandomForestRegressor` (`feature_importances_f32`/`_f64` per §4.3); `base.py`'s `_suffixed`/`_to_output` dtype-dispatch helpers (same pattern as every other fitted float-vector accessor in the codebase).
- **Given/When/Then:**
  - Given a fitted estimator, when `.feature_importances_` is read, then it returns a length-`n_features` numpy array respecting `output_type`.
  - Given an UNFITTED estimator, when `.feature_importances_` is read, then it raises the project's standard not-fitted error (same as every other fitted-only property).
- **Acceptance tests:** Python `test_oracle_ensemble.py` extension + `test_shims.py` fitted-attribute-list entry for both RF estimators.
- **Out of scope:** HGB (not applicable, §2).
- **Traceability:** depends on RF-IMP-01.
- **Unresolved questions:** none beyond RF-IMP-01's.

### RF-OOB-01 — `oob_score_` Rust-core computation (algos + prim + kernel)
- **Status:** draft.
- **Rationale/source:** user-locked scope expansion; sklearn `RandomForestClassifier(oob_score=True)`/`RandomForestRegressor(oob_score=True)`.
- **Preconditions:** builder `oob_score=true` (default `false`); build()-time validation rejects `oob_score=true, bootstrap=false` (`BuildError::OobRequiresBootstrap`, per §4.1) — mirrors sklearn's `ValueError`.
- **Input:** none beyond the existing `fit(x, y)` call — `oob_score_` MUST be computed inside `fit` (the Fitted state retains no training data — §3), using the same bootstrap-weight information `bootstrap_weights` (`random_forest.rs:379-397`) already computes for tree growth (`weight[t][i] == 0` ⇒ sample `i` out-of-bag for tree `t`).
- **Output:** `Option<F>` — `None` if `oob_score=false` (avoids the extra aggregation cost); `Some(score)` if `true`. Classifier: accuracy of the OOB-tree-averaged class-distribution argmax vs. training `y`. Regressor: R² of the OOB-tree-averaged prediction vs. training `y`.
- **Dependencies:** `bootstrap_weights` (reuse the SAME per-tree-per-sample weight array already produced for tree growth — either persist it past its current `w_dev.release_into(pool)` discard point, or deterministically re-derive it host-side from `SplitMix64::new(seed)` since the RNG consumption order is fixed and documented — `random_forest.rs:466-470` — **Planner must choose; re-deriving avoids the extra `t*n*sizeof(F)` device memory retention but costs a second host RNG pass, which is cheap (host-only, no device sync)**); the forest's own predict-path traversal logic (reused, not reimplemented, to compute per-tree per-OOB-sample leaf predictions).
- **Given/When/Then:**
  - Given `oob_score=false` (default), when `.fit()` completes, then `oob_score()` returns `None` and no extra computation/readback occurs.
  - Given `oob_score=true, bootstrap=true`, when `.fit()` completes, then `oob_score()` returns `Some(score)` where `score` is within a documented tolerance of sklearn's `oob_score_` on the same data/seed-equivalent bootstrap draws (bootstrap is stochastic and mlrs's `SplitMix64` stream ≠ sklearn's `MT19937`, so this is a **statistical-tier-only** assertion, same two-tier convention as the rest of RF — no exact-match tier is possible for a stochastic quantity).
  - Given `oob_score=true, bootstrap=false`, when `.build()` is called, then it returns `Err(BuildError::OobRequiresBootstrap)`.
  - Given a pathologically small forest where some training sample is never OOB for any tree, then that sample is excluded from the `oob_score_` aggregation (sklearn's own documented behavior) — **Planner must decide the mlrs-side signal**: silent skip, `log::warn!`, or a typed error (§4.1 TBD, owner Planner + user).
- **Invariants/side effects:** no impact on `predict`/`predict_proba` output — `oob_score_` is fit-time-only diagnostic state, does not alter the fitted forest structure.
- **Acceptance tests:** new Rust oracle test (extends `scripts/gen_oracle.py` with an `oob_score`-enabled fixture variant), statistical-tier band assertion only; a builder-validation unit test for the `oob_score=true, bootstrap=false` rejection (mirrors the existing `builder_rejects_invalid_hyperparameters` test pattern already used for other RF hyperparameters).
- **Out of scope:** `oob_decision_function_`/`oob_prediction_` (§2 non-goals); HGB (not applicable).
- **Traceability:** `[VERIFIED: LOCAL crates/mlrs-backend/src/prims/random_forest.rs:379-397,466-470 bootstrap_weights + RNG stream]` `[UNVERIFIED: tree.rs kernel bodies for the OOB-restricted predict-aggregation mechanism]`
- **Unresolved questions:** (1) persist-vs-rederive the bootstrap weight array — owner Planner; (2) zero-OOB-sample signal channel — owner Planner + user; (3) statistical tolerance band for the stochastic oracle test — owner Planner, informed by the same `ACC_MARGIN=0.05` precedent already used elsewhere in RF.

### RF-OOB-02 — `oob_score`/`oob_score_` Python binding
- **Status:** draft.
- **Rationale/source:** exposes RF-OOB-01's Rust-core output through the Python shim.
- **Preconditions:** RF-OOB-01 implemented; PY-ENS-01/02 binding scaffolding exists.
- **Input:** `oob_score: bool` constructor argument (default `False`, sklearn-named).
- **Output:** `oob_score_: float` fitted property, present **only if** `oob_score=True` was passed at construction (sklearn's own contract: the attribute does not exist / raises `AttributeError` if `oob_score=False` — mirror this rather than always exposing a `None`-valued property, to match sklearn's `hasattr(model, "oob_score_")` behavior that some downstream code relies on).
- **Dependencies:** RF-OOB-01's Rust accessor; `oob_score_f32`/`_f64` PyO3 methods returning `PyResult<Option<f32|f64>>` (or a Python-side `AttributeError` raise when `None`, per the sklearn-parity note above — **Planner must decide exactly where this None→AttributeError translation happens**: PyO3 layer vs. Python shim `@property`).
- **Given/When/Then:**
  - Given `oob_score=True, bootstrap=True`, when `.fit(X,y)` completes, then `.oob_score_` returns a float.
  - Given `oob_score=False` (default), when `.oob_score_` is accessed, then it raises `AttributeError` (sklearn parity) — NOT a silent `None`.
  - Given `oob_score=True, bootstrap=False`, when the estimator is constructed (or at `.fit()` time, matching wherever RF-OOB-01 places the validation), then it raises `ValueError`.
- **Acceptance tests:** Python `test_oracle_ensemble.py` extension (statistical tier) + `test_params.py` new constructor-arg entry + `test_shims.py` conditional-attribute entry (this is the first mlrs estimator with a conditionally-present fitted attribute — Planner should check whether `test_shims.py`'s enumeration machinery already supports "attribute present only under a certain constructor arg" or needs extending).
- **Out of scope:** `oob_decision_function_` (§2).
- **Traceability:** depends on RF-OOB-01.
- **Unresolved questions:** conditional-attribute test-machinery support (see acceptance tests above) — owner Planner.

## 6. Acceptance Scenarios

1. `import mlrs; mlrs.RandomForestClassifier().fit(X, y).predict(X)` works end-to-end with sklearn-parity results on the committed fixtures (PY-ENS-01).
2. `import mlrs; mlrs.RandomForestRegressor().fit(X, y).predict(X)` works end-to-end (PY-ENS-02).
3. `import mlrs; mlrs.HistGradientBoostingClassifier(n_bins=255).fit(X, y).predict_proba(X)` matches the (freshly-committed, non-dirty) HGB fixture (PY-ENS-03) — blocked until the HGB churn precondition clears.
4. `import mlrs; mlrs.HistGradientBoostingRegressor().fit(X, y).predict(X)` works end-to-end (PY-ENS-04) — same precondition.
5. All four estimators pass `test_params.py`, `test_shims.py`, and either pass or documented-xfail `test_estimator_checks.py` (PY-ENS-05).
6. `mlrs.RandomForestClassifier().fit(X, y).feature_importances_` returns a length-`n_features` array summing to 1, with the dominant-feature qualitative property holding on a synthetic separable dataset (RF-IMP-01/02).
7. `mlrs.RandomForestRegressor(oob_score=True).fit(X, y).oob_score_` returns a float within the statistical tolerance band of sklearn's `oob_score_` on an equivalent statistical-tier dataset; `mlrs.RandomForestRegressor(oob_score=False).fit(X,y)` then accessing `.oob_score_` raises `AttributeError` (RF-OOB-01/02).
8. `mlrs.RandomForestClassifier(oob_score=True, bootstrap=False)` raises `ValueError` at construction or fit (RF-OOB-01/02).

## 7. Impact Scope

| Area | Classification | Files |
|---|---|---|
| PY-ENS-01..05 binding layer | cross-module (mlrs-py only) | `crates/mlrs-py/src/estimators/ensemble.rs` (new), `estimators/mod.rs`, `lib.rs`, `python/mlrs/ensemble.py` (new), `python/mlrs/__init__.py`, `python/tests/test_oracle_ensemble.py` (new), `test_params.py`, `test_shims.py`, `test_estimator_checks.py` |
| RF-IMP-01, RF-OOB-01 | cross-module (mlrs-backend + mlrs-kernels + mlrs-algos) | `crates/mlrs-backend/src/prims/random_forest.rs`, `crates/mlrs-kernels/src/tree.rs`, `crates/mlrs-algos/src/ensemble/{random_forest_classifier,random_forest_regressor,mod}.rs`, new/extended `crates/mlrs-algos/tests/random_forest_*_test.rs` |
| RF-IMP-02, RF-OOB-02 | cross-module (mlrs-py, depends on above) | `crates/mlrs-py/src/estimators/ensemble.rs`, `python/mlrs/ensemble.py`, `python/tests/*` |
| External/public surface | external/public | four new top-level `mlrs.*` estimator classes; two new fitted attributes + one new constructor arg on `RandomForestClassifier`/`RandomForestRegressor` |
| Operational | operational | `scripts/gen_oracle.py` extension (new fixture generators for feature_importances_/oob_score_); HGB fixture regen (pre-existing in-flight work, this spec only consumes it) |

No changes anticipated to `mlrs-core`, Arrow bridge, or any non-ensemble estimator.

## 8. Compatibility and Migration

Purely additive at the Python surface — no existing estimator, function, or signature changes. The stale `lib.rs`/`estimators/mod.rs` "12 estimators"/"30" prose comments should be corrected while editing (cosmetic, not a behavior change). `RfParams` gains one new field (`oob_score: bool`) and `RfModel`/return-type changes (§4.2) are internal to `mlrs-backend`/`mlrs-algos` and not part of any external contract prior to this spec landing (the four ensemble estimators have no external Python consumers yet).

## 9. Risks and Open Questions

1. **Wrong monomorphization** — using `any_estimator!` instead of `any_estimator_typestate!` (PY-ENS-01..04). Prevention: mirror `naive_bayes.rs`. Verify: `cargo test -p mlrs-py --features cpu` compiles.
2. **HGB `n_bins=64` vs oracle `n_bins=255`** — deterministic HGB tier only matches sklearn at `max_bins=255` (PY-ENS-03/04). Verify: deterministic-tier Python test passes only with the explicit override.
3. **HGB algos churn — HARD BLOCKER for oracle finalization only** (PY-ENS-03/04, user-locked). Verify: `git status --short` clean on the four named HGB files before marking those tasks complete.
4. **Kernel-level unknowns for RF-IMP-01/RF-OOB-01** — `crates/mlrs-kernels/src/tree.rs` kernel bodies were not read in this research pass; the Planner MUST read them via CodeGraph before committing to an exact implementation mechanism (extend `rf_best_split`'s output arrays vs. a separate host/device reduction pass). This is the single largest unresolved-evidence risk in this spec.
5. **Feature-importances cross-implementation tolerance** — **[RESOLVED at TASK-02 Green time, 2026-07-18]** sklearn/mlrs impurity-importance values differ even on the deterministic tier, NOT from floating-point accumulation order, but because sklearn's own splitter breaks near-tied splits with internal randomness independent of the public seed — sklearn's own trees aren't bit-identical to each other even when mlrs's are. No tier supports an exact ≤1e-5 assertion for `feature_importances_` (unlike `predict`/`predict_proba`, which are tie-insensitive). Resolution: `atol=0.05` on the deterministic tier (25x the observed ~0.0022 divergence) plus a qualitative dominant-feature-ranking assertion as the primary signal (RF-IMP-01).
6. **OOB stochastic tolerance** — `SplitMix64` ≠ sklearn `MT19937`; `oob_score_` can only be oracle-tested on a statistical band, never exact-match (RF-OOB-01).
7. **Zero-OOB-sample edge case** — no established mlrs warning channel for estimators; Planner + user must pick a signal (RF-OOB-01 unresolved question).
8. **Conditional fitted-attribute test coverage** — `oob_score_`'s presence depends on a constructor arg; unclear if `test_shims.py`'s enumeration machinery already supports this pattern (RF-OOB-02 unresolved question).
9. **`predict_log_proba` scope creep** — explicitly locked OUT; any implementer temptation to add it "for completeness" must be rejected per the user's locked decision.
10. **`oob_decision_function_` scope creep** — explicitly locked OUT (only the scalar `oob_score_`); flagged in §2 as an `[INFERRED]` narrow reading the user should reconfirm if they actually wanted the full OOB array too.

## 10. Traceability and Sources

- `.planning/plans/RESEARCH.md` — original gap survey (2026-07-16), PY-ENSEMBLE deep dive §5.
- `.planning/plans/py-ensemble/research.md` — verification pass (2026-07-17), reconfirms binding-layer claims, sharpens HGB-churn risk to a hard sequencing constraint.
- `[VERIFIED: CODEGRAPH]` — `crates/mlrs-algos/src/ensemble/{random_forest_classifier,random_forest_regressor,hist_gradient_boosting_classifier,hist_gradient_boosting_regressor,mod}.rs`, `crates/mlrs-algos/src/typestate.rs`.
- `[VERIFIED: LOCAL]` — `crates/mlrs-backend/src/prims/random_forest.rs` (full read, :1-870), `crates/mlrs-backend/src/pool.rs`, `crates/mlrs-py/src/estimators/mod.rs`, `crates/mlrs-py/src/dispatch.rs`, `crates/mlrs-py/src/lib.rs`, `crates/mlrs-py/python/mlrs/*.py` (fit-signature grep), `crates/mlrs-algos/tests/random_forest_classifier_test.rs`, `.planning/plans/metrics-surface/SPEC.md` (house-style precedent).
- `[UNVERIFIED]` — `crates/mlrs-kernels/src/tree.rs` kernel bodies (file located, not read — Planner must read before RF-IMP-01/RF-OOB-01 task design); exact sklearn version behind the RF/HGB fixtures (research.md Q3).
- Tools used: `mcp__codegraph__codegraph_explore` (ensemble estimator layer + rf_fit_impl/RfModel/bootstrap_weights), local Bash/Read/Grep. Context7/WebSearch not invoked — no external library API question arose beyond well-established sklearn public-attribute shape (`feature_importances_`/`oob_score_` absence on HGB), which is common knowledge not requiring a fresh docs fetch.
