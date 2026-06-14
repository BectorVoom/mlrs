# Phase 6: Python Surface — PyO3 Estimators & Per-Backend Wheels - Pattern Map

**Mapped:** 2026-06-13
**Files analyzed:** 22 new/modified (Rust extension + pure-Python shim + packaging + tests)
**Analogs found:** 14 with strong in-repo analogs / 22 ; 8 no-in-repo-analog (point at cuML method ref or RESEARCH)

This is a **wrap-only** binding + packaging phase. The 11 estimators already exist in
`mlrs-algos`; Phase 6 grows the existing `mlrs-py` cdylib (currently just the mimalloc
allocator + a stub) into a `#[pymodule] _mlrs` and adds a pure-Python `mlrs/` shim plus
per-backend maturin packaging. **No source edits were made producing this map** — read-only.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-py/src/lib.rs` (M) | binding/module-init | request-response + import-probe | `crates/mlrs-backend/src/runtime.rs` (`active_client`) + existing `lib.rs` | role-match |
| `crates/mlrs-py/src/allocator.rs` (unchanged) | config | n/a | itself (already wired) | exact — no change |
| `crates/mlrs-py/src/ingress.rs` (N) | binding/FFI | transform (Arrow→device) | `crates/mlrs-backend/src/bridge.rs` (`validate_f32/f64`) + `device_array.rs::from_host` | exact (reuse) |
| `crates/mlrs-py/src/egress.rs` (N) | binding | transform (device→host) | `device_array.rs::to_host` + estimator host accessors (`.coef`, `.labels`) | exact (reuse) |
| `crates/mlrs-py/src/capability.rs` (N) | binding/guard | request-response | `crates/mlrs-backend/src/capability.rs` (`feature_enabled`, `skip_f64_with_log`) | exact (reuse) |
| `crates/mlrs-py/src/dispatch.rs` (N) | binding/utility | event-driven (dtype dispatch) | `mlrs-algos` per-estimator `<F>` constructors (e.g. `KMeans::new`) | role-match (new enum wraps them) |
| `crates/mlrs-py/src/estimators/*.rs` (N, 11) | binding/`#[pyclass]` controller | request-response | `mlrs-algos` estimator impls (`KMeans`, `Ridge`, `NearestNeighbors`, …) | exact (delegate) |
| `crates/mlrs-py/Cargo.toml` (M) | config | n/a | itself + workspace `Cargo.toml` `[workspace.dependencies]` | exact |
| `Cargo.toml` (workspace) (M) | config | n/a | itself (existing `cubecl`/`arrow` pins) | exact |
| `crates/mlrs-py/python/mlrs/base.py` (N) | provider (sklearn Base) | request-response | **cuML** `internals/base.py` `Base` (method ref, NOT in-repo) | method-ref only |
| `crates/mlrs-py/python/mlrs/_io.py` (N) | utility | transform (numpy↔pyarrow, output_type) | **cuML** `base.py::_get_output_type` + `array.py::to_output` (method ref) | method-ref only |
| `crates/mlrs-py/python/mlrs/{linear,cluster,decomposition,neighbors}.py` (N) | component (estimator shim) | request-response | **cuML** mixin pattern + sklearn mixins (method ref) | method-ref only |
| `crates/mlrs-py/python/mlrs/__init__.py` (N) | config/entry | n/a | `mlrs-algos/src/lib.rs` re-export shape (structure ref) | partial |
| `crates/mlrs-py/pyproject/{cpu,wgpu,cuda,rocm}.pyproject.toml` (N, 4) | config/packaging | n/a | **none in-repo** — RESEARCH §Pattern 1 (maturin multi-dist) | research-ref only |
| `crates/mlrs-py/python/tests/conftest.py` (N) | test fixture | file-I/O (load `.npz`) | `scripts/gen_oracle.py` + `mlrs_core::oracle::load_npz` (method ref) | role-match |
| `crates/mlrs-py/python/tests/test_oracle_*.py` (N) | test | request-response + compare | `crates/mlrs-algos/tests/kmeans_test.rs` (oracle-compare structure) | role-match |
| `crates/mlrs-py/python/tests/test_estimator_checks.py` (N) | test | request-response | **none in-repo** — RESEARCH §sklearn estimator_checks | research-ref only |
| `crates/mlrs-py/tests/*.rs` (N, optional Rust int tests) | test | n/a | `crates/mlrs-py/tests/allocator_test.rs` (existing separation pattern) | exact |

(M) = modified, (N) = new.

---

## Pattern Assignments

### `crates/mlrs-py/src/lib.rs` (module-init, import-probe) — MODIFIED

**Current state** (`crates/mlrs-py/src/lib.rs:1-16`): only declares `mod allocator;` and a
`BoundaryResult<T> = anyhow::Result<T>` alias. Phase 6 grows it into `#[pymodule] _mlrs`.

**Analog for the import-time driver probe (D-08):** `crates/mlrs-backend/src/runtime.rs:41-46`
```rust
pub fn active_client() -> Client {
    use cubecl::Runtime as _;
    let device = ActiveDevice::default();
    ActiveRuntime::client(&device)   // RESEARCH: client() .unwrap()s internally → PANICS if driver absent
}
```
The `#[pymodule]` init must wrap this in `std::panic::catch_unwind` and translate a caught
panic into `PyImportError` (RESEARCH Pattern 4). The backend name for the message comes from
`crates/mlrs-backend/src/capability.rs:107` `active_backend_name() -> &'static str`.

**Analog for the global pool/client (Claude's-discretion: process-global behind `Mutex`):**
`crates/mlrs-backend/src/pool.rs:73` `BufferPool::new(client)` + `runtime.rs::active_client()`.
Store as `OnceLock<Mutex<BufferPool<ActiveRuntime>>>` in the module.

**Allocator stays untouched:** `crates/mlrs-py/src/allocator.rs:18-23` already defines the single
`#[global_allocator] static GLOBAL: MiMalloc` — do NOT add a second allocator site (FOUND-09).

---

### `crates/mlrs-py/src/ingress.rs` (Arrow PyCapsule → validated device buffer) — NEW

**Analog 1 — the validation bridge to reuse UNCHANGED (D-02):** `crates/mlrs-backend/src/bridge.rs:40-49`
```rust
pub fn validate_f32(arr: &Float32Array) -> Result<&[f32], BridgeError> {
    validate_primitive::<Float32Type>(arr)
}
pub fn validate_f64(arr: &Float64Array) -> Result<&[f64], BridgeError> {
    validate_primitive::<Float64Type>(arr)
}
```
**CRITICAL constraint to carry into the shim** — `bridge.rs:80-104` `validate_no_offset` HARD-REJECTS
sliced/offset arrays (`byte_offset == 0 && inner.len() == values.len() * elem`). So the Python shim
must hand a **freshly-allocated contiguous** pyarrow array (`pa.array(np.ascontiguousarray(X).ravel())`),
never a zero-copy slice of a larger numpy buffer (RESEARCH Pitfall 3).

**Analog 2 — the existing call site that consumes a validated `&[F]`:** `bridge.rs:142-150` `upload`
and `crates/mlrs-backend/src/device_array.rs:59-83` `DeviceArray::from_host(pool, &validated)` — the
ingress path ends by uploading the validated slice into a pooled `DeviceArray`.

**New glue (no in-repo analog):** consuming the `__arrow_c_array__` capsule. Use
`arrow::pyarrow::FromPyArrow` (`ArrayData::from_pyarrow_bound`) → `make_array` → downcast to
`Float32Array`/`Float64Array` → feed `validate_f32/f64`. See RESEARCH Pattern 2 for the exact shape.

---

### `crates/mlrs-py/src/egress.rs` (device → host buffers) — NEW

**Analog — host materialization the shim consumes (D-03):** `device_array.rs:108-122` `to_host`
returns `Vec<F>`; estimator accessors already wrap it:
- `crates/mlrs-algos/src/cluster/kmeans.rs:168-188` `cluster_centers(pool) -> Vec<F>`, `labels(pool) -> Vec<i32>`
- `crates/mlrs-algos/src/linear/ridge.rs:101-118` `coef(pool) -> Vec<F>`, `intercept(pool) -> F`

Egress returns `(Vec<F>, shape)` or `(Vec<i32>, shape)` to Python; the **shim** wraps to
numpy/pyarrow (D-03). Labels/indices are `i32` everywhere (kmeans.rs:389-390 widens `u32`→`i32`).

---

### `crates/mlrs-py/src/capability.rs` (f64-on-incapable-backend guard, D-04) — NEW

**Analog — the capability layer to surface as a Python flag:** `crates/mlrs-backend/src/capability.rs:50-54`
```rust
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn feature_enabled(kind: FloatKind) -> bool {
    let client = crate::runtime::active_client();
    client.properties().supports_type(kind)
}
```
and `capability.rs:146-154` `skip_f64_with_log()` (the skip philosophy). For D-04 the binding
inverts the skip into a hard **`PyValueError`** when f64 is passed to an f64-incapable backend
(never silently downcast). Surface `feature_enabled(FloatKind::F64)` as e.g. a Python
`mlrs.backend_supports_f64()` so the shim picks the default dtype (D-05) and pytest can
`@pytest.mark.skipif(...)` (RESEARCH §pytest harness).

---

### `crates/mlrs-py/src/dispatch.rs` + `estimators/*.rs` (`#[pyclass]` + dtype enum, D-06) — NEW

**Analog — the generic `<F>` constructors the enum arms wrap.** Each `#[pyclass]` cannot be
generic over `F`, so wrap both monomorphizations (`Estimator<f32>` / `Estimator<f64>`) per D-06.
Concrete constructors + accessors + trait calls to delegate to:

| Estimator | Constructor (verified) | Fit/output trait | Host accessors |
|-----------|------------------------|------------------|----------------|
| `KMeans` | `kmeans.rs:112 new(n_clusters, seed)`, `:152 with_opts(n_clusters, seed, max_iter, tol)` | `Fit` + `PredictLabels` (kmeans.rs:220, :400) | `cluster_centers`, `labels`, `inertia` (kmeans.rs:168-197) |
| `Ridge` | `ridge.rs:90 new(alpha, fit_intercept)` | `Fit` + `Predict` (ridge.rs:59) | `coef`, `intercept` (ridge.rs:101-118) |
| `LinearRegression` | `linear/linear_regression.rs::new(fit_intercept)` | `Fit` + `Predict` | `coef`, `intercept` |
| `Lasso` / `ElasticNet` | `new(..)` / `with_opts(.., max_iter, tol)` | `Fit` + `Predict` | `coef`, `intercept` |
| `LogisticRegression` | `logistic.rs:124 new(c, fit_intercept)`, `:130 with_opts(c, fit_intercept, max_iter, tol)` | `Fit` + `PredictLabels` + `PredictProba` | `n_classes` (logistic.rs:169), `classes_` |
| `PCA` / `TruncatedSVD` | `decomposition/*.rs::new(n_components)` | `Fit` + `Transform` (`inverse_transform` PCA only) | `components_`, `mean_` |
| `DBSCAN` | `cluster/dbscan.rs::new(eps, min_samples)` | `Fit` + `PredictLabels` | `labels_` |
| `NearestNeighbors` | `neighbors/nearest.rs:77 new(n_neighbors)` | `Fit` + `KNeighbors` (nearest.rs:139) | (distances, indices) |
| `KNeighborsClassifier` | `neighbors/classifier.rs::new(n_neighbors)` | `Fit` + `KNeighbors` + `PredictLabels` + `PredictProba` | labels/proba |
| `KNeighborsRegressor` | `neighbors/regressor.rs::new(n_neighbors)` | `Fit` + `KNeighbors` + `Predict` | predictions |

**Trait surface signature reference** (the call shape the GIL-released closure invokes):
`crates/mlrs-algos/src/traits.rs:61-67` —
`fn fit(&mut self, pool: &mut BufferPool<ActiveRuntime>, x: &DeviceArray<…,F>, y: Option<&…>, shape: (usize,usize)) -> Result<&mut Self, AlgoError>`.
Every estimator takes `&mut BufferPool` + explicit `(rows, cols)`; the `DeviceArray` stays flat
1-D (D-08). Wrap the trait call in `py.detach(|| { let mut pool = GLOBAL_POOL.lock()…; … })`
(RESEARCH Pattern 5). `enum AnyKMeans { Unfit{..}, F32(KMeans<f32>), F64(KMeans<f64>) }` per D-06;
drive the 11× boilerplate with a `macro_rules!` (RESEARCH Pattern 3).

---

### `crates/mlrs-py/Cargo.toml` + workspace `Cargo.toml` — MODIFIED

**Analog — existing dep-declaration shape.** Current `crates/mlrs-py/Cargo.toml:1-13`:
```toml
[lib]
crate-type = ["cdylib", "rlib"]
[dependencies]
anyhow = { workspace = true }
mimalloc = { workspace = true }
```
Add `pyo3` (workspace, features `["abi3-py312", "extension-module"]`), `arrow` (workspace,
feature `["pyarrow"]`), `mlrs-algos`/`mlrs-backend` path deps, and the `[features]` cpu/wgpu/cuda/rocm
block forwarding to `mlrs-backend/*` + `mlrs-algos/*` (RESEARCH §Installation).

**Workspace pin pattern** (`Cargo.toml:13-39`): single-source `[workspace.dependencies]` —
`arrow = "59"` already present; add `pyo3 = { version = "0.28", default-features = false }`.
**Document the version exception inline:** the workspace comment says "track latest" (`Cargo.toml:11`),
but PyO3 MUST be 0.28 (not 0.29) to match arrow-59's pyarrow feature + pyo3-arrow 0.18 — a hard
ABI pin (RESEARCH §Project Constraints, Pitfall 1). Mirror the existing inline-comment convention
(e.g. the `cubecl … default-features=false` rationale comment at `Cargo.toml:14-16`).

---

### `crates/mlrs-py/python/mlrs/base.py` (`MlrsBase(BaseEstimator)`) — NEW, cuML method-ref only

**No in-repo analog** — this is the new pure-Python shim. Mirror **cuML** `internals/base.py`
(read-only method reference, NOT to copy verbatim — mlrs subclasses sklearn `BaseEstimator`
directly per D-01, narrower than cuML's `output_type` set):

- `output_type` constructor param + `_input_type` storage: `cuml-main/.../base.py:113-121`
- `get_params`/`set_params` from `_get_param_names`: `base.py:156-191` (mlrs gets these from
  sklearn `BaseEstimator` instead — keep `__init__` faithful, store args verbatim same-name)
- **egress routing to mirror** (D-03): `base.py:193-218` `_set_output_type`/`_get_output_type`
  (default `"input"` → infer from the container the data arrived in). mlrs narrows the resolved
  set to numpy + pyarrow only.
- `__repr__` reflection: `base.py:127-149` (mlrs gets this free from sklearn).

`to_output(output_type)` egress wrapper reference: `cuml-main/.../array.py:503` `CumlArray.to_output`.

**`__init__` purity is mandatory** (RESEARCH Pitfall 4 / estimator_checks): store every ctor arg
verbatim under the same name (`self.C = C`, not `self.c`), validate only in `fit`, `fit` returns `self`.

---

### `crates/mlrs-py/python/mlrs/{linear,cluster,decomposition,neighbors}.py` — NEW, mixin method-ref

**No in-repo analog.** Mixin composition reference: **cuML** `internals/mixins.py` and sklearn:
- `RegressorMixin.score` (R²): `cuml-main/.../mixins.py:206-236` → LinReg/Ridge/Lasso/ElasticNet/KNNReg
- `ClassifierMixin.score` (accuracy): `mixins.py:238-269` → LogReg/KNNClf
- `ClusterMixin.fit_predict`: `mixins.py:271-287` → KMeans/DBSCAN
- `__sklearn_tags__` (sklearn ≥1.6 public tag API): `mixins.py:195-203` — override to set
  `input_tags.sparse=False`, `array_api_support=False`, `input_tags.allow_nan=False`
  (RESEARCH §estimator_checks). Use sklearn's own mixins (not cuML's) per D-01.

Hyperparameter names per estimator: see the table in **RESEARCH §Hyperparameter Mapping**
(verified against the Rust constructors above — note `LogisticRegression` exposes sklearn `C`,
Rust field is `c`; `KMeans` maps sklearn `random_state` → Rust `seed`).

---

### `crates/mlrs-py/pyproject/{cpu,wgpu,cuda,rocm}.pyproject.toml` — NEW, RESEARCH-ref only

**No in-repo analog.** Use the maturin multi-distribution template in **RESEARCH §Pattern 1**:
one file per backend, only `[project].name` (`mlrs-cpu`/`mlrs-wgpu`/`mlrs-cuda`/`mlrs-rocm`) and
`[tool.maturin].features` differ; constant `module-name = "mlrs._mlrs"`,
`python-source = "crates/mlrs-py/python"`, `requires-python = ">=3.12"`, abi3-py312 (D-07/D-09).

---

### `crates/mlrs-py/python/tests/conftest.py` + `test_oracle_*.py` — NEW, oracle-harness analog

**Analog 1 — fixture generator + committed blobs (reuse as-is):** `scripts/gen_oracle.py` writes the
committed `.npz` fixtures under `tests/fixtures/` (e.g. `kmeans_f32_seed42.npz`, `ridge_f64_seed42.npz`,
`pca_*`, `logistic_{binary,multi}_*`, `knn_*`, `dbscan_*`, `truncated_svd_*` — one per estimator × dtype).
The Python harness is a **second consumer** of these blobs: `numpy.load` the same `.npz`, run the input
through `mlrs.<Estimator>(...).fit(X)`, assert the fitted attribute within 1e-5 (RESEARCH §pytest harness).
No new fixtures required for the happy path.

**Analog 2 — oracle-compare structure (Rust → re-express in Python):** `crates/mlrs-algos/tests/kmeans_test.rs`
(label-permutation compare), `pca_test.rs`/`truncated_svd_test.rs` (sign-flip compare). Re-express the
sign-flip (PCA/SVD `components_`) and label-permutation (KMeans/DBSCAN `labels_`) helpers in `conftest.py`.
LogisticRegression compares gauge-fixed `predict_proba`, not raw `coef_` (Phase-5 D-12).

**dtype × backend gating:** mirror `skip_f64_with_log` (capability.rs:146) — skip f64 cases on a rocm
wheel via the surfaced `mlrs.backend_supports_f64()` flag.

**Test-separation rule (AGENTS.md §2, HARD):** Rust int tests live in `crates/mlrs-py/tests/`
(analog `crates/mlrs-py/tests/allocator_test.rs` already there); Python tests in the `python/tests/`
pytest tree — never an in-source `#[cfg(test)] mod tests`.

### `crates/mlrs-py/python/tests/test_estimator_checks.py` — NEW, RESEARCH-ref only

**No in-repo analog.** Use `sklearn.utils.estimator_checks.parametrize_with_checks([...11...])`;
scope the "relevant" subset per family per **RESEARCH §sklearn estimator_checks** (Wave-0 triage task,
record pass/fail, skip-with-reason for sparse/array-api/NaN checks mlrs does not support by design).

---

## Shared Patterns

### Arrow validation (reuse, D-02)
**Source:** `crates/mlrs-backend/src/bridge.rs:40-104` (`validate_f32`/`validate_f64`, `validate_no_offset`)
**Apply to:** every `fit`/`predict`/`transform`/`kneighbors` ingress in `estimators/*.rs`.
Hard-rejects offset/sliced/null arrays → the shim must pass a fresh contiguous pyarrow array.

### Device⇄host materialization (reuse, D-03)
**Source:** `crates/mlrs-backend/src/device_array.rs:59-122` (`from_host` / `to_host`) +
per-estimator host accessors (`kmeans.rs:168-197`, `ridge.rs:101-118`).
**Apply to:** all ingress (`from_host`) and egress (`to_host` → `Vec<F>`/`Vec<i32>` + shape).

### Backend capability + name (reuse, D-04/D-08)
**Source:** `crates/mlrs-backend/src/capability.rs:50-54` (`feature_enabled`), `:107` (`active_backend_name`),
`:146-154` (`skip_f64_with_log`).
**Apply to:** the f64 guard (`capability.rs` binding → `PyValueError`), the import probe message
(`active_backend_name`), and the pytest skip flag.

### Runtime client + pool (reuse, lifecycle)
**Source:** `crates/mlrs-backend/src/runtime.rs:41-46` (`active_client`, panics on missing driver) +
`pool.rs:73` (`BufferPool::new`).
**Apply to:** module-init import probe (catch_unwind) and the process-global `Mutex<BufferPool>`.

### Error handling split (CLAUDE.md / memory)
**Source:** existing `crates/mlrs-py/src/lib.rs:15` `BoundaryResult<T> = anyhow::Result<T>`;
`AlgoError`/`PrimError` (`thiserror`) in libs.
**Apply to:** `thiserror` stays in `mlrs-algos`/`mlrs-backend`; `anyhow` at the binding boundary;
map boundary error → `PyErr` via `From` + `pyo3::exceptions::Py*::new_err` (RESEARCH §Error Mapping).

### Source/test separation (AGENTS.md §2, HARD)
**Source:** `crates/mlrs-py/tests/allocator_test.rs` (existing); every `mlrs-algos/tests/*_test.rs`.
**Apply to:** all new tests — Rust in `crates/mlrs-py/tests/`, Python in `python/tests/`. Never
an in-source `#[cfg(test)] mod tests`.

---

## No Analog Found (use method-ref / RESEARCH, not a forced in-repo match)

| File | Role | Data Flow | Closest reference (NOT in-repo to copy) |
|------|------|-----------|------------------------------------------|
| `python/mlrs/base.py` | sklearn Base shim | request-response | cuML `internals/base.py:113-218` (output_type/get_params/repr) |
| `python/mlrs/_io.py` | numpy↔pyarrow + output_type | transform | cuML `base.py::_get_output_type` + `array.py:503 to_output` |
| `python/mlrs/{linear,cluster,decomposition,neighbors}.py` | estimator shims | request-response | cuML `mixins.py:195-287` + sklearn mixins; hyperparam table in RESEARCH §Hyperparameter Mapping |
| `python/mlrs/__init__.py` | package entry (triggers probe) | n/a | structure ref `mlrs-algos/src/lib.rs:31-44` re-export shape |
| `pyproject/{cpu,wgpu,cuda,rocm}.pyproject.toml` | maturin packaging | n/a | RESEARCH §Pattern 1 (maturin multi-dist template) |
| `python/tests/test_estimator_checks.py` | estimator_checks triage | request-response | RESEARCH §sklearn estimator_checks (per-family subset table) |
| `src/ingress.rs` capsule-consume glue | FFI | transform | RESEARCH §Pattern 2 (`arrow::pyarrow::FromPyArrow`); validation half IS in-repo (bridge.rs) |
| `src/dispatch.rs` `AnyEstimator` enum | dtype dispatch | event-driven | RESEARCH §Pattern 3 (enum + `macro_rules!`); arms wrap in-repo `<F>` constructors |

---

## Metadata

**Analog search scope:** `crates/mlrs-py/`, `crates/mlrs-algos/src/{traits,linear,cluster,decomposition,neighbors}.rs`,
`crates/mlrs-backend/src/{bridge,capability,runtime,device_array,pool}.rs`, workspace `Cargo.toml`,
`scripts/gen_oracle.py`, `tests/fixtures/`, `cuml-main/python/cuml/cuml/internals/{base,array,mixins,input_utils}.py` (method ref).
**Files scanned:** ~24 source + 31 fixture blobs + 4 cuML refs.
**Pattern extraction date:** 2026-06-13
