# sklearn `estimator_checks` triage — the 12 mlrs estimators (criterion 1)

This is the empirical Wave-0 triage required by the phase gate. It was produced
by running `sklearn.utils.estimator_checks.parametrize_with_checks([...12...])`
(sklearn 1.9.0) against the **real compiled cpu/f64 `_mlrs`** extension and
recording, per estimator, which RELEVANT checks pass and which by-design-
unsupported checks are declared as `expected_failed_checks` (sklearn's native
xfail mechanism, >=1.6) with a documented reason.

Criterion 1 asks for the **relevant** subset to pass — NOT "all checks pass".
A check that fails *only* because mlrs intentionally does not support the
behavior is xfailed-with-reason here. Any check that failed for a **real bug**
was FIXED in the shim (see "Bugs fixed during triage"), not masked.

Run: `pytest crates/mlrs-py/python/tests/test_estimator_checks.py`

As of the WR-01 post-review fix, `_io.normalize_y` runs
`check_array(ensure_all_finite=True)` on `y`, so `check_supervised_y_no_nan` is
rejected with sklearn's own message and PASSES for EVERY supervised estimator —
it was removed from the `_SUPERVISED` xfail map entirely (it would otherwise
xpass). See the supervised table below. The pre-fix counts (475 passed / 102
xfailed) are recomputed by re-running the suite after the fix; the invariant
that matters is **0 unexpected failures and 0 xpassed**.

> The xfail map lives in `test_estimator_checks.py` (`_EXPECTED`); this document
> is its human-readable companion. The two MUST stay in sync.

---

## Bugs fixed during triage (NOT masked)

These genuinely failed and were fixed in the Plan-04 shim during this plan
(deviation Rules 1 & 2), so the relevant checks now pass:

1. **`n_features_in_` missing after fit** (Rule 2 — missing-critical sklearn
   attr). Added `MlrsBase._post_fit(cols)`, called by every `fit`, setting the
   standard `n_features_in_` attribute. This also makes the DEFAULT
   `check_is_fitted` scan succeed (`check_fit_check_is_fitted` now passes).
   Fixes `check_n_features_in`, `check_fit_check_is_fitted`.

2. **`predict`/`transform` before fit raised `AttributeError`, not
   `NotFittedError`** (Rule 1 — bug). Every predict path called
   `_normalize(X, dtype=self._np_float())` first, and `_np_float()` touched
   `self._mlrs_obj.dtype()` → `AttributeError` on an unfitted estimator. Added
   `MlrsBase._check_predict_X(X)` which runs `_check_fitted()` FIRST, then
   feature-count validation, used at the top of every
   `predict`/`predict_proba`/`transform`/`kneighbors`. Fixes
   `check_estimators_unfitted`.

3. **`predict`/`transform` did not validate input feature count** (Rule 2).
   `_check_predict_X` raises a sklearn-shaped `ValueError` when `X` has a
   different column count than the fitted `n_features_in_`. Fixes
   `check_n_features_in_after_fitting`.

---

## By-design xfails (documented, NOT bugs)

### Common to every estimator

| Check | Reason |
|-------|--------|
| `check_estimator_sparse_tag` | Dense Arrow ingress only; the sparse input tag is off by design (`__sklearn_tags__.input_tags.sparse = False`). |
| `check_estimator_sparse_array` | Sparse input unsupported by design. |
| `check_estimator_sparse_matrix` | Sparse input unsupported by design. |
| `check_estimators_pickle` | Fitted state is an opaque Rust `_mlrs` `#[pyclass]` device handle, not picklable in v1 (model serialization is out of v1 scope). |
| `check_dtype_object` | object/string-dtype `X` IS rejected, but via numpy's float-cast error whose message does not match sklearn's expected substring; mlrs is dense-float-only by design. |

### Supervised estimators (the 4 linear regressors, LogReg, both KNN supervised)

| Check | Reason |
|-------|--------|
| `check_supervised_y_2d` | No `DataConversionWarning` on a column-vector `y`; 1-D `y` is the v1 contract (no silent 2-D→1-D reshape warn). |
| `check_requires_y_none` | The "y is required" rejection does not match sklearn's expected message verbatim (it still raises on `y=None`). |

> `check_supervised_y_no_nan` is **no longer xfailed** for any estimator. As of
> the WR-01 fix, `_io.normalize_y` runs `check_array(ensure_all_finite=True)` on
> `y`, so a NaN/Inf target is rejected with sklearn's own `ValueError` message
> for every supervised estimator and the check PASSES (verified). It was removed
> from the `_SUPERVISED` map; keeping it would xpass and break the suite.

### Classifiers only (LogisticRegression, KNeighborsClassifier)

| Check | Reason |
|-------|--------|
| `check_classifiers_classes` | v1 classifiers require contiguous int32 labels `0..n_classes-1`; string class labels are out of the v1 label contract. |
| `check_classifiers_regression_target` | Continuous-target rejection is not emitted with sklearn's exact message; v1 expects pre-encoded discrete labels. |

### Iterative-solver estimators (Lasso, ElasticNet, LogReg, KMeans)

| Check | Reason |
|-------|--------|
| `check_non_transformer_estimators_n_iter` | The coordinate-descent / L-BFGS / Lloyd solvers do not surface an `n_iter_` attribute in v1. |

### Small/degenerate-fixture edge cases (LogReg, PCA, TruncatedSVD)

| Check | Reason |
|-------|--------|
| `check_fit2d_1sample` | A 1-sample fit is not special-cased with sklearn's exact "1 sample" message; the solver instead raises / produces a degenerate result. |

---

## Per-estimator summary

| Estimator | Family | Relevant checks pass | By-design xfails |
|-----------|--------|----------------------|------------------|
| LinearRegression | RegressorMixin | ✅ | common + supervised |
| Ridge | RegressorMixin | ✅ | common + supervised |
| Lasso | RegressorMixin | ✅ | common + supervised + n_iter |
| ElasticNet | RegressorMixin | ✅ | common + supervised + n_iter |
| LogisticRegression | ClassifierMixin | ✅ | common + supervised + classifier + n_iter + fit2d_1sample |
| KMeans | ClusterMixin | ✅ (incl. `check_clustering`, predict-based) | common + n_iter |
| DBSCAN | ClusterMixin | ✅ (no `predict` — see below) | common |
| PCA | TransformerMixin | ✅ | common + fit2d_1sample |
| TruncatedSVD | TransformerMixin | ✅ | common + fit2d_1sample |
| NearestNeighbors | (no scoring mixin) | ✅ (fit/`kneighbors`; no `predict` — see below) | common |
| KNeighborsClassifier | ClassifierMixin | ✅ | common + supervised + classifier |
| KNeighborsRegressor | RegressorMixin | ✅ | common + supervised |

### DBSCAN / NearestNeighbors — sklearn-faithful predict-less surface (RESEARCH Open Q3)

`DBSCAN` and `NearestNeighbors` expose **no `predict`** — exactly like their
sklearn counterparts (`sklearn.cluster.DBSCAN` and
`sklearn.neighbors.NearestNeighbors` have no standalone `predict`). Because the
estimators advertise no `predict`, `parametrize_with_checks` does not generate
predict-based checks for them (there is nothing to xfail) — the relevant
fit / `labels_` (DBSCAN) and fit / `kneighbors` (NearestNeighbors) checks run and
pass. This is the documented resolution of RESEARCH Open Question 3.

---

## Provenance

- sklearn 1.9.0 / numpy 2.4.6 / pyarrow 24.0.0, project oracle venv.
- Compiled cpu/f64 `_mlrs` via `maturin develop` (the local_dynamic_tls
  allocator fix from this plan lets it import with no `LD_PRELOAD`).
- The xfail map is `_EXPECTED` in `test_estimator_checks.py`; it dispatches on
  `type(estimator).__name__`. Editing one without the other will desync the
  triage (a stale xfail surfaces as an `xpassed`, which the suite flags).
