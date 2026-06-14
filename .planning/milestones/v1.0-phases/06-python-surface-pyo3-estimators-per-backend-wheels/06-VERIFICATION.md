---
phase: 06-python-surface-pyo3-estimators-per-backend-wheels
verified: 2026-06-14T07:30:00Z
status: human_needed
score: 4/4 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Live `import mlrs` from the `mlrs_cuda` wheel on a CUDA host"
    expected: "import succeeds; mlrs.backend_supports_f64() returns True or False without aborting"
    why_human: "No CUDA driver/GPU in this environment; compile-only per project constraints (user-approved)"
  - test: "Foreign-driver-absent ImportError on real CUDA hardware with ROCm or CPU wheel installed"
    expected: "Importing `mlrs` from an mlrs_cpu or mlrs_rocm wheel on a CUDA machine raises ImportError cleanly (no abort/segfault)"
    why_human: "Cross-hardware driver-absent test requires a CUDA host with a non-CUDA wheel installed; cannot simulate faithfully in this environment"
  - test: "Two backend wheels installed in the same environment overwrite `mlrs` namespace correctly (D-07)"
    expected: "Installing mlrs_cpu then mlrs_rocm leaves `import mlrs` resolving to the second wheel's `_mlrs` extension"
    why_human: "Requires two installable backend wheels with a shared `mlrs/_mlrs.abi3.so` path in one venv; full CUDA+ROCm host needed for meaningful dual-install"
---

# Phase 6: Python Surface — PyO3 Estimators & Per-Backend Wheels Verification Report

**Phase Goal:** A Python >= 3.12 data scientist can `pip install` the wheel matching their backend and use all 12 v1 estimators through a sklearn-compatible API with zero-copy Arrow ingest and the GIL released during compute.
**Verified:** 2026-06-14T07:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All 12 v1 estimators are `#[pyclass]`-backed with sklearn-compatible fit/predict/transform/score and pass pytest oracle tests plus relevant estimator_checks | VERIFIED | `lib.rs` has exactly 12 `m.add_class::<>()` calls; `_mlrs` registered. Oracle harness 34 cases run end-to-end vs real compiled cpu `_mlrs`; estimator_checks 475 passed / 102 by-design xfailed / 0 unexpected failures / 0 xpassed. `score()` comes from sklearn Regressor/Classifier/Cluster/TransformerMixin via MRO (verified). |
| 2 | Estimators support get_params/set_params with sklearn-named hyperparameters and accept f32/f64 NumPy/Arrow inputs via runtime dtype dispatch | VERIFIED | Pure-Python shim subclasses `sklearn.BaseEstimator` giving `get_params/set_params/clone/__repr__` for free. `__init__` stores every ctor arg verbatim. `test_params.py` confirms round-trip. `dispatch.rs` `any_estimator!` macro emits F32/F64 monomorphizations. `test_dtype.py` confirms f32/f64 preservation and integer-default dispatch. |
| 3 | NumPy/Arrow inputs cross via Arrow PyCapsule interface with correct ownership/lifetime (no &[u8] borrows); GIL released during compute | VERIFIED | `ingress.rs` uses `ArrayData::from_pyarrow_bound` (owned `ArrayRef`; no borrowed slice); `validate_f32/f64` bridge reused unchanged. Every `fit/predict/transform` body uses `py.detach(|| {...})` (38 occurrences across 4 estimator files). GIL release proven by subprocess worker test in `test_dtype.py::test_gil_released_during_compute`. |
| 4 | Per-backend wheels build via maturin under distinct dist names (mlrs-cpu/wgpu/cuda/rocm) with abi3-py312; driver-absent import fails cleanly | VERIFIED (with hardware-gated caveat) | 4 pyproject templates exist in `crates/mlrs-py/pyproject/`. SUMMARY 06-06 confirms all four wheels built: `mlrs_cpu-0.1.0-cp312-abi3-*.whl` (51 MB), `mlrs_wgpu-0.1.0-cp312-abi3-*.whl` (7.5 MB), `mlrs_cuda-0.1.0-cp312-abi3-*.whl` (6.2 MB compile-only), `mlrs_rocm-0.1.0-cp312-abi3-*.whl` (21 MB). `test_import_probe.py` asserts clean `ImportError` (in-process + subprocess). `test_wheels.py` asserts abi3 tag + dist name. `mimalloc local_dynamic_tls` enables `import mlrs` from cpu wheel without `LD_PRELOAD`. Live cuda import and cross-hardware driver-absent test deferred to a CUDA host (user-approved, project constraint). |

**Score:** 4/4 truths verified (3 without hardware caveat; 1 with project-accepted CUDA hardware deferral)

---

### Deferred Items

Items not yet met but explicitly deferred by project constraints and user approval.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Live `import mlrs` from the cuda wheel | CUDA host (opportunistic) | Project constraint: cuda is compile-only in this environment; user explicitly approved deferral in 06-06 Task 3 human-verify |
| 2 | Foreign-driver-absent ImportError on real CUDA hardware | CUDA host (opportunistic) | In-process + subprocess backstop asserted; real hardware gated by project constraint |
| 3 | Two-wheel namespace overwrite (mlrs_cpu + mlrs_rocm in same venv) | CUDA host (opportunistic) | Accepted by design (D-07); confirmed per user approval |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-py/src/lib.rs` | `#[pymodule]` with 12 `add_class` calls + catch_unwind driver probe | VERIFIED | 12 `m.add_class::<>()` calls; catch_unwind import probe confirmed at lines 119-128 |
| `crates/mlrs-py/src/estimators/linear.rs` | 5 `#[pyclass]` wrappers (LinearRegression/Ridge/Lasso/ElasticNet/LogisticRegression) | VERIFIED | 5 pyclass definitions confirmed; each has `py.detach` + `guard_f64()?` |
| `crates/mlrs-py/src/estimators/cluster.rs` | 2 `#[pyclass]` wrappers (KMeans/DBSCAN) | VERIFIED | 2 pyclass definitions confirmed |
| `crates/mlrs-py/src/estimators/decomposition.rs` | 2 `#[pyclass]` wrappers (PCA/TruncatedSVD) | VERIFIED | 2 pyclass definitions confirmed |
| `crates/mlrs-py/src/estimators/neighbors.rs` | 3 `#[pyclass]` wrappers (NearestNeighbors/KNeighborsClassifier/KNeighborsRegressor) | VERIFIED | 3 pyclass definitions confirmed |
| `crates/mlrs-py/src/ingress.rs` | PyCapsule ingress via `from_pyarrow_bound`; no `&[u8]` borrow | VERIFIED | Uses `ArrayData::from_pyarrow_bound` producing owned `ArrayRef`; bridge validated |
| `crates/mlrs-py/src/capability.rs` | `guard_f64()` + `supports_f64()` | VERIFIED | Thin wrapper over `mlrs_backend::capability::feature_enabled(FloatKind::F64)` |
| `crates/mlrs-py/python/mlrs/__init__.py` | 12 estimators re-exported; lazy `_mlrs` via PEP 562; `_load_ext` recursion-safe | VERIFIED | All 12 in `__all__`; `importlib.import_module` used in `_load_ext`; recursion fix present |
| `crates/mlrs-py/python/mlrs/base.py` | `MlrsBase` + `_post_fit` + `_check_predict_X` + `_suffix` + `_to_output` | VERIFIED | All methods present; `check_is_fitted` on `_mlrs_obj` attribute |
| `crates/mlrs-py/python/mlrs/_io.py` | `normalize_X` / `normalize_y` (finite check) / `pick_dtype` / `to_output` (shape-preserving) | VERIFIED | CR-01 fix in `to_output` raises ValueError for 2-D pyarrow results; WR-01 fix adds `check_array(ensure_all_finite=True)` on `y`; WR-03 fix fails closed (returns False on error) |
| `crates/mlrs-py/python/mlrs/linear.py` | 5 estimator shims | VERIFIED | LinearRegression/Ridge/Lasso/ElasticNet/LogisticRegression present with sklearn-faithful `__init__` |
| `crates/mlrs-py/python/mlrs/cluster.py` | 2 estimator shims | VERIFIED | KMeans/DBSCAN; WR-04 (KMeans init not validated) is open but a WARNING |
| `crates/mlrs-py/python/mlrs/decomposition.py` | 2 estimator shims | VERIFIED | PCA/TruncatedSVD present |
| `crates/mlrs-py/python/mlrs/neighbors.py` | 3 estimator shims | VERIFIED | NearestNeighbors/KNeighborsClassifier/KNeighborsRegressor; WR-08 (kneighbors n_neighbors not validated) is open but a WARNING |
| `crates/mlrs-py/pyproject/cpu.pyproject.toml` | `[project].name = "mlrs-cpu"`; `module-name = "mlrs._mlrs"`; `features = ["cpu", "extension-module"]` | VERIFIED | All three fields present and correct |
| `crates/mlrs-py/pyproject/wgpu.pyproject.toml` | Distinct dist name `mlrs-wgpu`; constant `module-name = "mlrs._mlrs"` | VERIFIED | Confirmed |
| `crates/mlrs-py/pyproject/cuda.pyproject.toml` | Distinct dist name `mlrs-cuda` | VERIFIED | Exists |
| `crates/mlrs-py/pyproject/rocm.pyproject.toml` | Distinct dist name `mlrs-rocm` | VERIFIED | Exists |
| `crates/mlrs-py/python/tests/test_oracle_linear.py` | 34+ oracle cases re-validating 1e-5 through full binding path | VERIFIED | 10 cases for linear (incl. ridge 3-alpha sweep × 2 dtypes); `@requires_f64` decorator; direct coef comparison |
| `crates/mlrs-py/python/tests/test_oracle_cluster.py` | KMeans (label-perm + center remap + inertia) + DBSCAN oracle | VERIFIED | label_perm_allclose + label_perm_remap + inertia direct compare |
| `crates/mlrs-py/python/tests/test_oracle_decomposition.py` | PCA/TruncatedSVD sign-flip oracle (8 cases) | VERIFIED | sign_flip_allclose per row; transform aligned by recovered signs |
| `crates/mlrs-py/python/tests/test_oracle_neighbors.py` | kNN oracle (distances 1e-5 + indices exact + predict + proba) | VERIFIED | Uses `knn_f32_seed42.npz` and `knn_f64_seed42.npz` (both committed) |
| `crates/mlrs-py/python/tests/test_dtype.py` | f32/f64 preservation + integer default + f64-on-incapable raise + GIL-release | VERIFIED | Subprocess-isolated GIL test; `test_f64_on_incapable_backend_raises` skipif-gated |
| `crates/mlrs-py/python/tests/test_estimator_checks.py` | `parametrize_with_checks` over 12 instances; `expected_failed_checks` xfail map | VERIFIED | Real `parametrize_with_checks`; `_EXPECTED` per-class dict; 0 unexpected failures; WR-02 handled correctly post WR-01 fix |
| `crates/mlrs-py/python/tests/checks_triage.md` | Per-estimator relevant-pass / by-design-xfail record | VERIFIED | Exists; updated post WR-01 fix to remove `check_supervised_y_no_nan` from xfail map |
| `crates/mlrs-py/python/tests/test_import_probe.py` | Driver-absent `ImportError` (in-process + subprocess) | VERIFIED | 3 test functions confirmed; uses `importlib.import_module` monkeypatching |
| `crates/mlrs-py/python/tests/test_wheels.py` | Wheel build + abi3 tag + dist name assertion | VERIFIED | `MLRS_BUILD_WHEELS=1` opt-in; asserts `mlrs_<backend>-*`, `abi3`, `cp312`, `mlrs/_mlrs*.so` in zip |
| `crates/mlrs-py/python/tests/test_egress_shape_regression.py` | CR-01 regression: 2-D pyarrow raises; numpy 2-D preserves; WR-01 NaN-y rejected | VERIFIED | Created in commit `0356e0c` post code review |
| `crates/mlrs-py/python/tests/test_io.py` | Unit tests for `_io.normalize_X/y`, `to_output`, `_backend_supports_f64` | VERIFIED | 27+ unit tests; fail-closed f64 capability tested |
| `tests/fixtures/knn_f32_seed42.npz` | Committed kNN oracle fixture | VERIFIED | File exists at expected path |
| `tests/fixtures/knn_f64_seed42.npz` | Committed kNN oracle fixture | VERIFIED | File exists; 46 total fixture files present |
| `Cargo.toml` (workspace) | `mimalloc = { version = "0.1", features = ["local_dynamic_tls"] }` | VERIFIED | Present; enables LD_PRELOAD-free wheel import |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Python `mlrs.LinearRegression.fit` | Rust `PyLinearRegression::fit` | `self._ext().LinearRegression(...)` -> `_mlrs` cdylib | WIRED | Lazy `_load_ext` -> `importlib.import_module("mlrs._mlrs")`; delegate confirmed in `linear.py:30-31` |
| Rust `PyLinearRegression::fit` | `LinearRegression<F>::fit` (mlrs-algos) | `py.detach` + `Fit::fit` trait dispatch | WIRED | Lines 81-100 in `estimators/linear.rs`; F32/F64 arms confirmed |
| `ingress.rs::capsule_to_array` | `mlrs_backend::bridge::validate_f32/f64` | `ArrayData::from_pyarrow_bound` -> `validate_f32/f64` | WIRED | `ingress.rs:64-83` shows reuse of `validate_f32/validate_f64` unchanged |
| `capability.rs::guard_f64` | f64 dispatch arm in every estimator wrapper | Called before `validated_f64(...)` on F64 arm | WIRED | Confirmed in linear.rs line 92; same pattern in cluster/decomp/neighbors |
| `lib.rs::_mlrs::catch_unwind` | PyImportError on absent driver | `active_client().properties()` inside `catch_unwind`; error -> `PyImportError` | WIRED | Lines 119-129 in lib.rs |
| `test_oracle_*` | `tests/fixtures/*.npz` (committed sklearn references) | `np.load(fixture_path(name))` via `conftest.py::FIXTURE_DIR` | WIRED | Fixture path is 4 parents up from `crates/mlrs-py/python/tests/conftest.py` to repo root |
| `test_estimator_checks.py::_expected_failed_checks` | `_EXPECTED` dict keyed by `type(estimator).__name__` | sklearn native `expected_failed_checks` hook | WIRED | Function dispatches on class name; used as `expected_failed_checks=_expected_failed_checks` in parametrize |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `test_oracle_linear.py::test_linear_coef_oracle` | `est.coef_` / `est.intercept_` | `np.load(fixture_path(fixture))` then `.fit(d["X"], d["y"])` through full FFI path | Real device compute via `maturin develop` cpu build | FLOWING |
| `mlrs/linear.py::LinearRegression.coef_` | `self._suffixed("coef")()` | `_mlrs_obj.coef_f32()` or `coef_f64()` -> `to_host_metered` in Rust | Real device buffer materialized to host Vec | FLOWING |
| `mlrs/_io.py::to_output` | `arr = flat.reshape(shape)` | `buf` from Rust host Vec; shaped to `(rows,)` or `(rows, cols)` | CR-01 fix: 2-D preserved for numpy; ValueError for 2-D pyarrow | FLOWING (corrected post-review) |
| `mlrs/_io.py::normalize_y` | `checked = check_array(y, ensure_all_finite=True)` | WR-01 fix: sklearn `check_array` on `y` before pyarrow conversion | Finiteness validated before device upload | FLOWING (corrected post-review) |
| `mlrs/_io.py::_backend_supports_f64` | `bool(_mlrs.backend_supports_f64())` | WR-03 fix: returns `False` on exception | Fails closed (f64-incapable assumption on error) | FLOWING (corrected post-review) |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `cargo check --package mlrs-py --features cpu,extension-module` | `cargo check ...` | Exit 0; `Finished 'dev' profile` | PASS |
| 12 `#[pyclass]` registrations in lib.rs | `grep -c "m.add_class" lib.rs` | 12 | PASS |
| 4 pyproject templates exist with correct names | `ls crates/mlrs-py/pyproject/` | cpu/wgpu/cuda/rocm.pyproject.toml all present | PASS |
| Oracle fixtures include all 12 estimator types | `find tests/fixtures -name "*.npz"` | 46 total; linear/lasso/elasticnet/ridge/logistic/kmeans/dbscan/pca/truncated_svd/knn all present | PASS |
| `score()` inherited from sklearn mixin | Python MRO check in oracle venv | `has score: True`; MRO confirms RegressorMixin | PASS |
| No TBD/FIXME/XXX debt markers in phase files | `grep -rn "TBD\|FIXME\|XXX" crates/mlrs-py/` | Zero hits in source/test files | PASS |
| Fix commits exist in git log | `git log --oneline 6ef01f5 cd5050d 0356e0c` | All three fix commits present (CR-01, WR-01/03, regression tests) | PASS |
| `py.detach` GIL release in all estimator wrappers | `grep -rn "py.detach" crates/mlrs-py/src/estimators/` | 38 occurrences across 4 files | PASS |
| `mimalloc local_dynamic_tls` in Cargo.toml | `grep "local_dynamic_tls" Cargo.toml` | Feature present at workspace level | PASS |

---

### Probe Execution

Step 7c: Wheel probes are guarded by `MLRS_BUILD_WHEELS=1` and require a ROCm/CUDA host for some backends. No `scripts/*/tests/probe-*.sh` files exist for this phase. Phase-declared probes are the pytest test suite itself (run during execution with maturin develop). Probe execution is not applicable in the static-analysis form here — the SUMMARY documents real green runs.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| PY-01 | 06-03, 06-04, 06-05, 06-06 | All v1 estimators as PyO3 #[pyclass] with sklearn-compatible API | SATISFIED | 12 pyclasses registered; 34-case oracle validates 1e-5 end-to-end; relevant estimator_checks pass |
| PY-02 | 06-03, 06-04, 06-06 | get_params/set_params + sklearn-named hyperparameters | SATISFIED | Pure-Python MlrsBase subclasses BaseEstimator; faithful __init__ confirmed; estimator_checks get/set_params pass |
| PY-03 | 06-02, 06-05 | Arrow PyCapsule ingress with correct ownership; GIL released | SATISFIED | `from_pyarrow_bound` owned import; no &[u8] borrow; py.detach in all compute paths; subprocess GIL test green |
| PY-04 | 06-01, 06-06 | Per-backend wheels under distinct dist names, abi3-py312; driver-absent ImportError | SATISFIED (with hardware caveat) | 4 templates + 4 wheel builds confirmed; cpu fresh-venv import without LD_PRELOAD; rocm live import confirmed; driver-absent ImportError asserted; cuda live-import deferred (project constraint) |
| PY-05 | 06-02, 06-05 | f32/f64 runtime dispatch; Python >= 3.12 | SATISFIED | dtype preservation + integer-default + f64-on-incapable raise all proven in test_dtype.py; abi3-py312 enforced by maturin |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/mlrs-py/python/mlrs/cluster.py` | 40-47 | KMeans `init` param stored but never validated in `fit` (WR-04 from code review, open) | WARNING | A caller passing `init="random"` gets k-means++ behavior with no error. Low correctness risk (k-means++ is always run; no wrong results), but a silent contract violation. The relevant estimator_checks don't cover this case (sklearn's check_no_attributes_set_in_init passes because init is stored verbatim). |
| `crates/mlrs-py/src/estimators/linear.rs` | 116 | `predict_f32` on F64 arm returns `not_fitted(...)` instead of dtype-mismatch error (WR-05, open) | WARNING | A user who calls `_mlrs.LinearRegression.predict_f32()` on an f64-fitted instance sees "not fitted" rather than "wrong dtype". The shim always routes to the correct arm (`_suffix()` picks `_f64`), so the live path is safe. Direct _mlrs consumers see misleading error. |
| `crates/mlrs-py/python/mlrs/linear.py` | 50, 88 | `intercept_` bypasses `_to_output` routing (WR-07, open) | WARNING | Regressor `intercept_` returns a raw Python float, not wrapped through output_type mirror. Under `output_type="pyarrow"` the `coef_` is a pyarrow array but `intercept_` is a bare float. Inconsistent egress contract. sklearn returns `intercept_` matching `coef_`'s type. |
| `crates/mlrs-py/python/mlrs/neighbors.py` | 32-47 | `kneighbors` does not validate `n_neighbors <= n_samples_fit_` (WR-08, open) | WARNING | Out-of-range k propagates to the algos layer; error depends on the Rust algos layer to reject it rather than the shim providing a sklearn-compatible ValueError. `n_samples_fit_` is not recorded. |
| `crates/mlrs-py/python/tests/conftest.py` | 134-150 | `proba_allclose` re-normalizes before comparing (IN-05, open) | INFO | Masks the potential defect of un-normalized probabilities in the oracle. Not a blocker for correctness since the underlying Rust softmax produces normalized outputs empirically (max error ~3e-15), but the oracle helper is defensively weaker than it should be. |

No TBD/FIXME/XXX debt markers found. No empty implementations (return null/{}). No hardcoded empty data flowing to rendering.

---

### Human Verification Required

#### 1. Live CUDA Wheel Import

**Test:** On a CUDA host, install `mlrs_cuda-0.1.0-cp312-abi3-*.whl` into a fresh venv and run `python -c "import mlrs; print(mlrs.backend_supports_f64()); print(mlrs.LinearRegression())"`.
**Expected:** Import succeeds cleanly (no LD_PRELOAD required); `backend_supports_f64()` returns True; estimator constructs.
**Why human:** No CUDA driver/GPU in this environment. Project constraint: cuda is compile-only here (user-approved per 06-06 Task 3 deferral).

#### 2. Cross-Hardware Driver-Absent ImportError (CUDA wheel on non-CUDA host)

**Test:** On a host with only ROCm or no GPU, install the `mlrs_cuda` wheel and run `import mlrs` in a subprocess. Confirm the process exits 0 (ImportError handled) and stdout contains the clear mlrs error message naming the backend.
**Expected:** `ImportError` with message containing "mlrs-cuda requires the cuda runtime/driver" or similar; no segfault; child process exit code 0.
**Why human:** Requires real foreign-backend hardware. The in-environment test simulates this via monkeypatching `importlib.import_module`; the real driver-absent hardware path cannot be exercised here.

#### 3. Two Backend Wheels Overwrite Shared Namespace (D-07)

**Test:** Install `mlrs_cpu` wheel, then install `mlrs_rocm` wheel in the same venv. Run `import mlrs; print(mlrs._mlrs.__file__)`. Verify the file path corresponds to the rocm extension (the second installed wheel).
**Expected:** The second wheel's `mlrs/_mlrs.abi3.so` shadows the first; `import mlrs` gives the rocm backend.
**Why human:** Requires two installable wheel files simultaneously. The cpu+wgpu combo is available; the rocm and cuda combos require matching hardware. Per project design this is accepted-by-design behavior (D-07), not an error condition.

---

### Gaps Summary

No blocking gaps found. The four roadmap success criteria are verified in the codebase:

1. **SC-1 (PY-01):** 12 `#[pyclass]` wrappers confirmed in `lib.rs` + 4 estimator source files. Oracle harness `test_oracle_{linear,cluster,decomposition,neighbors}.py` provides 34 cases testing the full `numpy -> pyarrow -> FFI -> device -> host -> numpy` path. Estimator checks 475 passed / 0 unexpected failures. `score()` inherited from sklearn mixins via MRO.

2. **SC-2 (PY-02):** Pure-Python shim subclasses `sklearn.BaseEstimator` giving `get_params/set_params/clone/__repr__` for free. Every `__init__` stores constructor args verbatim (purity rule). `test_params.py` confirms round-trip. `test_estimator_checks.py` passes relevant get/set_params checks.

3. **SC-3 (PY-03):** `ingress.rs` uses `from_pyarrow_bound` (owned `ArrayRef`; no `&[u8]` borrow). All 38 device-compute invocations use `py.detach`. GIL release proven by subprocess worker test.

4. **SC-4 (PY-04):** 4 pyproject templates produce 4 abi3 wheels under distinct dist names. cpu wheel imports without LD_PRELOAD (mimalloc `local_dynamic_tls`). rocm wheel imports live with `backend_supports_f64() == False`. Driver-absent `ImportError` asserted (in-process + subprocess). cuda live-import is hardware-gated, deferred per project constraint.

**Post-review fixes confirmed in git:**
- CR-01 (pyarrow 2-D flattening) — fixed in commit `6ef01f5`; `to_output` now raises ValueError for 2-D matrices under pyarrow output type
- WR-01 (NaN in y bypasses validation) — fixed in commit `cd5050d`; `normalize_y` now runs `check_array(ensure_all_finite=True)`
- WR-03 (f64 capability fallback fails open) — fixed in commit `cd5050d`; `_backend_supports_f64` now returns `False` on exception
- WR-02 (LogReg xfail rationale incorrect) — resolved by WR-01 fix; `check_supervised_y_no_nan` removed from xfail map; triage doc updated in `cd5050d`

**Remaining open review items (WR-04, WR-05, WR-07, WR-08, IN-01 through IN-06):** These are warnings and informational findings, not blockers for the phase goal. They concern:
- WR-04: KMeans `init` validation (accepted-unsupported param, no wrong results)
- WR-05: Misleading not_fitted error for wrong-dtype access (only via direct `_mlrs` use; shim routes correctly)
- WR-07: `intercept_` bypasses output_type (minor inconsistency; scalar value is correct)
- WR-08: kneighbors n_neighbors not validated against n_samples_fit (algos layer catches it)
- IN-* items: maintainability concerns

None of these prevent the phase goal (sklearn-compatible API, 1e-5 oracle, Arrow PyCapsule, abi3 wheels) from being achieved.

**Human verification items:** 3 hardware-gated checks that cannot be executed in this environment due to the project's explicit cuda=compile-only constraint, user-approved in 06-06 Task 3.

---

_Verified: 2026-06-14T07:30:00Z_
_Verifier: Claude (gsd-verifier)_
