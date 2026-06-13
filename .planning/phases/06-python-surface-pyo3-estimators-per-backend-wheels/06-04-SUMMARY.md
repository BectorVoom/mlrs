---
phase: 06-python-surface-pyo3-estimators-per-backend-wheels
plan: 04
subsystem: python-shim
tags: [sklearn, baseestimator, mixins, arrow, output-type, dtype-dispatch, get-params, notfitted]

# Dependency graph
requires:
  - phase: 06-python-surface-pyo3-estimators-per-backend-wheels
    plan: 03
    provides: 12 _mlrs #[pyclass] wrappers (sklearn-named ctors, dtype()/is_fitted(), dtype-suffixed fit/predict/transform/kneighbors accessors)
  - phase: 06-python-surface-pyo3-estimators-per-backend-wheels
    plan: 02
    provides: backend_supports_f64() flag on _mlrs, Arrow PyCapsule ingress contract
  - phase: 06-python-surface-pyo3-estimators-per-backend-wheels
    plan: 01
    provides: pure-Python mlrs package skeleton + pytest scaffold
provides:
  - "12 sklearn-compatible pure-Python estimator shims (BaseEstimator + family mixin) delegating to _mlrs (D-01/PY-01)"
  - "MlrsBase: output_type='input' verbatim, ingress/egress helpers, _check_fitted (NotFittedError), __sklearn_tags__ (sparse/nan/array-api off), dtype-suffix dispatch (_suffix/_suffixed)"
  - "_io: normalize_X (check_array finite/2-D -> fresh-contiguous row-major pyarrow + (rows,cols), D-02/Pitfall 3/T-06-11), pick_dtype (D-05/Pitfall 5), resolve_output_type input-mirror (D-03), to_output (int32 labels, D-06)"
  - "get_params/set_params/clone/__repr__ free from sklearn via faithful __init__ (PY-02; LogReg C not c, KMeans random_state)"
  - "lazy _mlrs import (PEP-562 __getattr__ + base._ext()) so the shim package imports and is unit-testable before maturin develop"
affects: [06-05-oracle-pytest-harness, 06-06-estimator-checks-wheels]

# Tech tracking
tech-stack:
  added: []
  patterns: [sklearn-baseestimator-direct-subclass, faithful-init-purity-for-free-get-params, lazy-extension-import-pep562, dtype-suffix-shim-dispatch, fresh-contiguous-pyarrow-ingress, output-type-input-mirror]

key-files:
  created:
    - crates/mlrs-py/python/tests/test_io.py
    - crates/mlrs-py/python/tests/test_shims.py
  modified:
    - crates/mlrs-py/python/mlrs/base.py
    - crates/mlrs-py/python/mlrs/_io.py
    - crates/mlrs-py/python/mlrs/linear.py
    - crates/mlrs-py/python/mlrs/cluster.py
    - crates/mlrs-py/python/mlrs/decomposition.py
    - crates/mlrs-py/python/mlrs/neighbors.py
    - crates/mlrs-py/python/mlrs/__init__.py
    - crates/mlrs-py/python/tests/test_params.py

key-decisions:
  - "Lazy _mlrs import (PEP-562 module __getattr__ in __init__.py + MlrsBase._ext() at fit-time) instead of the Plan-01 eager `from . import _mlrs`: the pure-Python shim (construction, get_params/set_params/clone, normalize_X, NotFitted, sklearn tags) imports and is fully unit-testable BEFORE maturin develop; only the actual fit/predict/accessor delegation touches the compiled extension. This is what made the strongest non-device verification possible in a no-maturin environment."
  - "output_type='input' is the only ctor param MlrsBase adds; every subclass repeats `self.output_type = output_type` verbatim (no super().__init__ in subclasses) to keep sklearn check_no_attributes_set_in_init / check_parameters_default_constructible happy — __init__ purity is absolute (PY-02)."
  - "_check_fitted keys on the private `_mlrs_obj` handle via check_is_fitted(self, attributes='_mlrs_obj') — NOT the default trailing-underscore scan, because the fitted-attr PROPERTIES (coef_, labels_, ...) would recurse the default scan. _suffixed() calls _suffix() (which runs _check_fitted) FIRST, before touching self._mlrs_obj, so coef_-before-fit raises NotFittedError not AttributeError (Python evaluates getattr's first arg before the second)."
  - "Egress mirrors input container (D-03 narrowed set): pyarrow-in -> pyarrow-out, else numpy-out; integer labels/indices (labels_, predict on KMeans/LogReg/KNNClf, kneighbors indices) materialize int32 (D-06); matrices reshape via the wrapper's known shape (n_components / n_classes / n_clusters)."
  - "LogisticRegression / KNeighborsClassifier set classes_ = arange(n_classes(), int32) at fit (v1 contiguous 0..n labels); NearestNeighbors / KNN* set n_features_in_ at fit (sklearn-standard fitted attr)."

requirements-completed: [PY-01, PY-02, PY-03, PY-05]

# Metrics
duration: 7min
completed: 2026-06-13
---

# Phase 6 Plan 04: Pure-Python sklearn Shim Summary

**The 12 user-facing `mlrs` estimators now exist as pure-Python `sklearn.BaseEstimator` + family-mixin subclasses (`RegressorMixin`/`ClassifierMixin`/`ClusterMixin`/`TransformerMixin`) that delegate compute to the Plan-03 `_mlrs` `#[pyclass]` wrappers (D-01): each `__init__` stores its sklearn-named ctor args verbatim (so `get_params`/`set_params`/`clone`/`__repr__` come free from sklearn — `LogisticRegression` exposes `C`, `KMeans` exposes `random_state`), `fit` normalizes input to a FRESH-contiguous row-major pyarrow float array + `(rows, cols)` (D-02), constructs/calls the matching `_mlrs.Py*` and `return self` (PY-01), and fitted-attr properties (`coef_`/`labels_`/`components_`/...) raise `NotFittedError` before fit and materialize via the dtype-suffixed wrapper accessor with `output_type` mirror routing (D-03/D-05/D-06).**

## Performance

- **Duration:** ~7 min
- **Completed:** 2026-06-13
- **Tasks:** 3 executed (Tasks 1 & 2 TDD: RED test commit -> GREEN impl commit)
- **Files:** 10 (2 created, 8 modified)
- **Tests:** 109 pure-Python unit tests pass (18 io + 53 shims + 38 params), 41 pre-existing Wave-0 stubs still xfail correctly. Run against sklearn 1.9.0 / numpy 2.4.6 / pyarrow 24.0.0.

## Accomplishments

- **Task 1 — base.py + _io.py (RED `1ea0e1a` -> GREEN `79a3134`):**
  - `_io.py`: `normalize_X` runs sklearn `check_array(ensure_all_finite, ensure_2d, dtype)` (so dimension/NaN errors match estimator_checks point 5) then `np.ascontiguousarray(...).ravel(order="C")` -> `pa.array` — a FRESH contiguous buffer (offset 0, no parent aliasing; Pitfall 3 / T-06-11), never a numpy slice. `pick_dtype` preserves an input float dtype; non-float defaults to f64 on an f64-capable backend, f32 on an incapable one via the lazily-imported `_mlrs.backend_supports_f64()` (D-05 / Pitfall 5). `resolve_output_type` maps default `"input"` to `"numpy"`/`"pyarrow"` by the input container (narrowed D-03 set); `to_output` reshapes/int32-casts the host buffer back to the resolved container. `normalize_y` mirrors `normalize_X` for 1-D float targets.
  - `base.py`: `MlrsBase(BaseEstimator)` with `output_type='input'` stored verbatim, `_normalize`/`_normalize_y`/`_to_output` delegating to `_io`, `_check_fitted` (NotFittedError via `check_is_fitted(attributes='_mlrs_obj')`), `__sklearn_tags__` turning off sparse/array-api/NaN, and the D-06 dtype-suffix helpers `_suffix`/`_np_float`/`_suffixed`.
  - `__init__.py`: replaced the eager `from . import _mlrs` with a PEP-562 module `__getattr__` that lazily resolves `_mlrs` / `backend_supports_f64` (clear ImportError + D-08 probe still fire on first real use) — the family modules now import without the compiled extension.
- **Task 2 — the 12 family-module shims (RED `a06b84b` -> GREEN `0bcf04f`):**
  - `linear.py`: LinearRegression/Ridge/Lasso/ElasticNet (`RegressorMixin`), LogisticRegression (`ClassifierMixin`, sklearn `C` -> Rust `c`, `predict` via `predict_labels`, `predict_proba`, `classes_`).
  - `cluster.py`: KMeans (`ClusterMixin`, `predict` via `predict_labels`, `cluster_centers_`/`labels_`/`inertia_`; `random_state` stored verbatim, mapped at the `_mlrs` boundary), DBSCAN (`ClusterMixin`, `labels_`/`core_sample_indices_`, NO `predict` — D-08).
  - `decomposition.py`: PCA/TruncatedSVD (`TransformerMixin`, `transform`/`components_`/...); PCA adds `inverse_transform`.
  - `neighbors.py`: NearestNeighbors (no scoring mixin — `kneighbors`, no `predict`), KNeighborsClassifier (`ClassifierMixin`), KNeighborsRegressor (`RegressorMixin`).
  - Every `fit` normalizes via base, lazily constructs `self._ext().Py*`, stores `self._mlrs_obj`, and `return self`.
- **Task 3 — test_params.py (`1b17e83`):** converted the Wave-0 xfail/importorskip stub into real PY-02 assertions over all 12 estimators: exact sklearn-named `get_params` keys + documented defaults (RESEARCH Hyperparameter table), `set_params` round-trip, `__init__` purity (kwargs stored verbatim same-name), `LogisticRegression` `C`-not-`c`, `KMeans` `random_state`.
- **Style (`4d76fa7`):** ruff-format (line-length 79) on the three reformatted modules + drop the unused `numpy` import in test_shims.py (F401). No behavior change.

## Task Commits

1. **Task 1 RED — failing io/base unit tests** — `1ea0e1a` (test)
2. **Task 1 GREEN — base.py + _io.py** — `79a3134` (feat)
3. **Task 2 RED — failing family-shim structure tests** — `a06b84b` (test)
4. **Task 2 GREEN — 12 estimator shims** — `0bcf04f` (feat)
5. **Task 3 — real PY-02 get_params/set_params assertions** — `1b17e83` (test)
6. **Style — ruff-format + F401 fix** — `4d76fa7` (style)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Lazy `_mlrs` import so the pure-Python shim imports without the compiled extension**
- **Found during:** Task 1 (RED run failed at collection with `ImportError: cannot import name '_mlrs'`).
- **Issue:** The Plan-01 `__init__.py` did an eager `from . import _mlrs`, so `import mlrs` (and any `mlrs._io` / family-module import) hard-failed before `maturin develop`. The whole pure-Python contract the plan asks me to test (get_params/set_params/clone, normalize_X, NotFitted, tags) was un-importable, blocking the task.
- **Fix:** Moved `_mlrs` to a PEP-562 module `__getattr__` in `__init__.py` and a `MlrsBase._ext()` helper imported at fit-time. The clear ImportError + D-08 driver probe still fire on the first real `fit`/`backend_supports_f64` use; the shim structure now imports and unit-tests pre-build. This matches the Plan-01 `__init__.py` docstring intent ("the pure-Python shims are importable WITHOUT the extension; only `_mlrs` itself is guarded").
- **Files modified:** `__init__.py`, `base.py`.
- **Commit:** `79a3134`.

**2. [Rule 1 - Bug] `coef_`-before-fit raised `AttributeError`, not `NotFittedError`**
- **Found during:** Task 2 (RED->GREEN; the fitted-attr-before-fit cases failed with AttributeError).
- **Issue:** `_suffixed` was written `getattr(self._mlrs_obj, base + self._suffix())`; Python evaluates the first `getattr` argument (`self._mlrs_obj`) before the second, so an unfitted estimator hit `AttributeError: no attribute '_mlrs_obj'` before `_suffix()`'s `_check_fitted()` could raise the sklearn `NotFittedError` the contract (and estimator_checks) requires.
- **Fix:** Compute `suffix = self._suffix()` (which runs `_check_fitted()` first) on its own line, before touching `self._mlrs_obj`. All fitted-attr-before-fit paths now raise `NotFittedError`.
- **Files modified:** `base.py`.
- **Commit:** `0bcf04f`.

### Plan-text vs. reality notes (not deviations)

- **`test_io.py` / `test_shims.py` are new files (the plan named only `test_params.py`).** AGENTS.md §2 requires tests in the pytest tree (not in source). Task 1 and Task 2 are `tdd="true"`, so each needed a RED test; the cleanest place is dedicated `python/tests/test_io.py` (Task-1 ingress/egress/base) and `python/tests/test_shims.py` (Task-2 shim structure), alongside the plan's `test_params.py` (Task 3). All three live in the existing pytest tree.
- **Verification ran the pure-Python subset, not `maturin develop`.** This environment has no `maturin` and no compiled `_mlrs` (and the system interpreter lacked pyarrow/sklearn). Per the executor note "do not fabricate test passes", I did NOT run the plan's `maturin develop ...` verify lines. Instead I installed sklearn 1.9.0 / numpy 2.4.6 / pyarrow 24.0.0 into the project oracle venv and ran the strongest non-device subset (the `io-ok` shape check, the `shim-ok` get_params/set_params/clone/`C` smoke from the plan's own verify blocks, plus 109 unit assertions). The fit/predict/accessor DEVICE delegation (which needs the compiled extension) is covered by the live-extension oracle gate in Plan 05/06.

## Threat-Model Outcomes

| Threat ID | Disposition | Evidence |
|-----------|-------------|----------|
| T-06-11 (sliced/non-contiguous numpy view crossing as an aliased pyarrow slice) | mitigated | `normalize_X` does `np.ascontiguousarray(arr, dtype).ravel(order="C")` -> a FRESH `pa.array`; `test_normalize_X_sliced_view_becomes_fresh_contiguous` asserts `arr.offset == 0` and correct values from a non-C-contiguous `base[:, ::2]` view. The Rust bridge `validate_f32/f64` offset-reject is the backstop. |
| T-06-12 (NaN/inf input silently computed on) | mitigated | `normalize_X` runs `check_array(ensure_all_finite=True)` before normalization; `test_normalize_X_rejects_non_finite` asserts the sklearn-standard `ValueError` on NaN input. The bridge null-reject is the backstop. |
| T-06-13 (reading fitted attrs before fit) | mitigated | `_check_fitted` (`check_is_fitted(attributes='_mlrs_obj')`) raises `NotFittedError` before any `_mlrs` accessor; `test_fitted_attr_raises_before_fit` covers coef_/cluster_centers_/labels_/components_ across the families (and the AttributeError-vs-NotFittedError ordering bug above was fixed so the guard actually fires). |

## Known Stubs

None in the shim logic. `fit`/`predict`/`transform`/`kneighbors`/fitted-attr accessors all delegate to the real `_mlrs.Py*` wrappers — there is no hardcoded/mock fitted state. The only intentionally-deferred items are the **device-path verifications** (live `fit`/`predict` against the oracle fixtures), which require the compiled extension and are owned by Plan 05 (oracle harness) and Plan 06 (estimator_checks + wheels), exactly as the phase plan sequences them. The 41 xfailed Wave-0 tests (`test_oracle_*`, `test_dtype`, `test_estimator_checks`) are those deferred gates and remain xfailed by design.

## Verification

- `import mlrs` + all 12 estimators importable and constructible pure-Python (`all-12-ok`).
- `shim-ok` smoke (from the plan's verify block, device-free part): `Ridge(alpha=2.0).get_params()['alpha']==2.0`; `set_params(alpha=3.0)` round-trips; `clone(KMeans(n_clusters=5)).n_clusters==5`; `hasattr(LogisticRegression(), 'C')`.
- `io-ok` shape check: `normalize_X(np.eye(3, float32))` -> `(pyarrow f32 array len 9, 3, 3)`.
- 109 pure-Python tests pass (18 `test_io` + 53 `test_shims` + 38 `test_params`); 41 pre-existing Wave-0 stubs xfail unchanged. sklearn 1.9.0 / numpy 2.4.6 / pyarrow 24.0.0.
- Acceptance greps: `ascontiguousarray|ravel` and `backend_supports_f64` present in `_io.py`; `__sklearn_tags__` in `base.py`; `RegressorMixin|ClassifierMixin|ClusterMixin|TransformerMixin` across all 4 family modules; DBSCAN/NearestNeighbors define no `predict` (`hasattr` False); every `fit` ends `return self`; LogisticRegression `C` not `c`; KMeans `random_state` present.
- `ruff check --select F` clean; byte-compile (`py_compile`) of all shim modules OK. (E501 80-char docstring lines are consistent with the pre-existing committed Wave-0 test files — the project's ruff config does not enforce E501 on these docstrings.)
- **NOT run here (no maturin/compiled `_mlrs` in this environment):** the plan's `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml` build + live device `fit`/`predict`. Deferred to the build environment / Plan 05-06 oracle gate. No test passes were fabricated.

## Next Phase Readiness

- **Plan 05 (oracle pytest harness)** can now `mlrs.<Estimator>(...).fit(X)` and assert fitted attributes within 1e-5 over the committed `tests/fixtures/*.npz` blobs once `maturin develop` builds `_mlrs`. The shim contract it consumes: `fit(X[, y]) -> self`; numpy/pyarrow ingress (auto fresh-contiguous); `coef_`/`intercept_`/`labels_`/`cluster_centers_`/`inertia_`/`components_`/`mean_`/`explained_variance_`/`explained_variance_ratio_`/`singular_values_`/`classes_`/`n_features_in_` fitted attrs; `predict`/`predict_proba`/`transform`/`inverse_transform`/`kneighbors`; `output_type` mirror egress; `mlrs.backend_supports_f64()` for the f64-skip marker.
- **Plan 06 (estimator_checks + wheels)** gets `__sklearn_tags__` (sparse/array-api/NaN off) already wired, so `parametrize_with_checks` will skip the unsupported-by-design checks.
- No blockers. The one environmental constraint (no maturin here) is expected for a build/packaging phase and does not affect the shim correctness this plan delivers.

## Self-Check: PASSED

- Files: `python/mlrs/{base,_io,linear,cluster,decomposition,neighbors,__init__}.py` modified; `python/tests/{test_io,test_shims}.py` created; `python/tests/test_params.py` rewritten — all present on disk.
- Commits: `1ea0e1a`, `79a3134`, `a06b84b`, `0bcf04f`, `1b17e83`, `4d76fa7` all in `git log`.

---
*Phase: 06-python-surface-pyo3-estimators-per-backend-wheels*
*Completed: 2026-06-13*
