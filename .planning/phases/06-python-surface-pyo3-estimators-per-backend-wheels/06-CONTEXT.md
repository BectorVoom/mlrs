# Phase 6: Python Surface — PyO3 Estimators & Per-Backend Wheels - Context

**Gathered:** 2026-06-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 6 builds the **Python surface** over the 12 already-implemented `mlrs-algos`
estimators and ships **per-backend wheels**. (The boundary text below says "11" in
places — reconciled 2026-06-13 to **12**: the count earlier folded **ElasticNet**
(LINEAR-04) into the Lasso shared-CD family; the surface wraps all 12 structs —
LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, PCA, TruncatedSVD,
KMeans, DBSCAN, NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor.) No new algorithms — this phase wraps
the existing Rust estimator layer (the `Fit` / `Predict` / `Transform` /
`PredictLabels` / `KNeighbors` / `PredictProba` trait surface, each generic over
`<F: Float + CubeElement + Pod>`) so a Python ≥ 3.12 data scientist can
`pip install` the wheel matching their backend and use a sklearn-compatible API.

Covers requirements **PY-01, PY-02, PY-03, PY-04, PY-05**. The four ROADMAP
success criteria are the gate:
1. All 11 estimators are PyO3-backed objects with sklearn-compatible
   `fit`/`predict`/`transform`/`score` (`fit` returns `self`); pass `pytest`
   oracle tests + relevant `sklearn.utils.estimator_checks`.
2. `get_params`/`set_params` with sklearn-named constructor hyperparameters;
   accept f32 and f64 inputs via runtime dtype dispatch.
3. Inputs cross via the Arrow PyCapsule interface with correct ownership/lifetime
   (no bare `&[u8]` borrows into Python-owned buffers); `Python::allow_threads`
   releases the GIL around device compute.
4. Per-backend wheels build via `maturin build --features <backend>` under
   distinct distribution names (`mlrs-cpu`, `mlrs-wgpu`, `mlrs-cuda`,
   `mlrs-rocm`) with `abi3-py312`; importing a wheel whose driver is absent fails
   with a clear error.

**Scope anchors (carried forward — NOT re-decided):**
- **Wrap-only.** The 11 estimators already exist in `mlrs-algos`. Phase 6 adds the
  PyO3 binding layer + Python package + maturin packaging. No algorithm work.
- **Gate = cpu(f64) + rocm(f32) (D-07 from Phase 3).** f64 validates on cpu, f32
  on rocm; **f64-on-rocm skips-with-log** (cubecl-cpp 0.10 does not register F64
  for the HIP backend). This is the root cause of the dtype × backend conflict
  resolved below (D-08/D-09). cuda compiles only (untestable here); wgpu
  opportunistic.
- **`mlrs-py` owns the `#[global_allocator]` (mimalloc, FOUND-09)** — already
  wired; the cdylib stays the single allocator site.
- **`ActiveRuntime` is feature-selected** (exactly one of cpu/wgpu/cuda/rocm) — so
  one source builds N wheels, one backend per wheel.

**User intent for this phase:** follow **cuML's I/O method** as closely as the
locked PY-03 boundary contract allows (see D-04/D-05/D-06).

</domain>

<decisions>
## Implementation Decisions

### Package topology
- **D-01: Thin Python shim over a compiled core.** A compiled PyO3 extension
  (working name `_mlrs`) exposes the low-level estimator entry points; the
  **importable `mlrs` package is pure Python**, with each estimator a class
  subclassing `sklearn.base.BaseEstimator` + the appropriate mixins
  (Regressor/Classifier/Cluster/Transformer), delegating compute to `_mlrs`.
  `get_params`/`set_params`/`clone`/`__repr__`/`_get_tags` come from sklearn for
  (mostly) free; the numpy↔Arrow glue and `output_type` routing live in Python.
  This mirrors cuML's own architecture (Cython core + Python `Base` estimators)
  and is the cleanest path to passing `sklearn.utils.estimator_checks`.
  - **Consequence:** the wheel ships Python source alongside the abi3 extension.
  - Pure-`#[pyclass]`-only and "pyclass + minimal `__init__.py`" were rejected —
    re-implementing sklearn `BaseEstimator` semantics and `clone()` compatibility
    by hand from a pyclass is fiddly and brittle.

### NumPy / Arrow I/O — follows cuML's method within the PY-03 boundary
- **D-02: Hybrid ingress — cuML-style API surface, Arrow PyCapsule boundary.**
  The Python shim accepts the **cuML-style range of inputs** (numpy / pyarrow /
  Python lists via the array-interface), then **normalizes to a contiguous 1-D
  pyarrow float array** (row-major flatten of `X`) and crosses the Rust boundary
  via the **`__arrow_c_array__` PyCapsule** + an explicit `(rows, cols)` tuple.
  Rust imports the capsule via **arrow-rs FFI** (release-callback ownership — no
  bare `&[u8]` borrow into a Python-owned buffer) and **reuses the existing
  `validate_f32`/`validate_f64` bridge unchanged**, then uploads to device.
  - This honors PY-03 *literally* (Arrow PyCapsule at the boundary) while giving
    the user cuML-like flexibility. **No requirement change.**
  - `X` is 2-D but the Arrow array is 1-D + shape tuple — consistent with the
    existing `fit(pool, x, y, (rows, cols))` flat-buffer signature (P2 D-04).
  - **Adds `pyarrow` as a runtime dependency** of the `mlrs` package (accepted).
  - Full-cuML array-interface/DLPack ingress (would require amending PY-03) and
    strict Arrow-native-only ingress were both rejected.
- **D-03: Egress — adopt cuML's `output_type` routing.** A configurable
  `output_type` constructor param + a global override
  (`cuml.using_output_type`-equivalent), default **`"input"` = mirror the
  container the data arrived in**, returning through a `to_output(output_type)`
  equivalent — exactly cuML's `Base.output_type` / `CumlArray.to_output`
  mechanism. **v1 supported output set is narrower than cuML: numpy + pyarrow**
  (mlrs has no cupy/cuDF/numba integration), so numpy-in→numpy-out,
  arrow-in→arrow-out. `labels_` / neighbor indices materialize as **int32**
  (D-06 from Phase 5). PY-03 does not constrain egress, so this is fully
  compatible; `estimator_checks` still pass because numpy-in→numpy-out under
  mirror. Rust returns host buffers (`Vec<F>` / `Vec<i32>`) + shape; the shim
  wraps to the resolved output container.

### dtype × backend dispatch
- **D-04: f64 on an f64-incapable backend → capability-query + clear error.**
  The extension exposes a capability flag (built on the existing
  `crates/mlrs-backend/src/capability.rs`). Passing **float64** to a backend that
  cannot run it (notably `mlrs-rocm`) raises a **clear Python exception**
  (e.g. *"backend 'rocm' does not support float64 — pass float32 or install
  mlrs-cpu"*). Never silently downcast. This matches the project's
  correctness-first value and the `skip_f64_with_log` gate philosophy (D-07).
  cuML-style warn+downcast and silent-downcast were rejected — silent precision
  loss below the 1e-5 contract is unacceptable here (mlrs's f64-incapability is a
  hard backend limitation, unlike cuML's dtype-conversion convenience).
- **D-05: Preserve input float dtype; non-float defaults to f64 where supported.**
  `f32`-in → compute f32 → f32-out; `f64`-in → compute f64 → f64-out (sklearn-like
  preservation). Integer / list / other inputs default to **float64 on
  f64-capable backends** (sklearn-faithful), **float32 on f64-incapable
  backends** (rocm). cuML-style "default everything to float32" was rejected as
  less sklearn-faithful for the cpu gate.
- **D-06: Internal dtype dispatch via an enum on the Arrow array dtype.** Because
  a `#[pyclass]` cannot be generic over `F`, the extension inspects the incoming
  pyarrow array's float type and dispatches to `Estimator<f32>` vs
  `Estimator<f64>` via an internal enum (e.g. `enum AnyKMeans { F32(KMeans<f32>),
  F64(KMeans<f64>) }`). The Python shim does not expose `fit_f32`/`fit_f64` — it
  passes through; dispatch is a Rust-internal detail. (Exact enum/wrapper shape =
  Claude's discretion.)

### Wheel naming & UX
- **D-07: Constant `import mlrs`, distinct distribution names.** Every backend
  wheel exposes the **same top-level `import mlrs`** (cuML-style: `pip install
  mlrs-cuda` → `import mlrs`), so user code is portable across backends unchanged.
  Distribution names differ: `mlrs-cpu` / `mlrs-wgpu` / `mlrs-cuda` / `mlrs-rocm`.
  - **Consequence:** the wheels share the `mlrs` namespace → a user installs
    **exactly one** backend wheel (two would overwrite each other). Document and,
    where feasible, guard against double-install. PY-04's "install the package
    matching your backend" assumes exactly one.
  - Distinct import names per backend (`mlrs_cpu`, …) were rejected — breaks code
    portability.
- **D-08: Import-time driver probe + clear error.** On `import mlrs`, probe for
  the backend driver / attempt cubecl client init; if the driver is absent, raise
  **`ImportError`** with a clear, actionable message (e.g. *"mlrs-rocm requires
  the ROCm/HIP runtime; none detected — install ROCm or use mlrs-cpu"*). Matches
  criterion 4's literal "importing … fails with a clear error" and fails fast
  before the user writes `fit()`. Lazy-on-first-compute was rejected (defers the
  failure past import, in tension with criterion 4).
- **D-09: `abi3-py312` stable ABI** (locked by criterion 4) — one wheel per
  backend covers Python ≥ 3.12.

### Carried forward from Phases 1–5 (reaffirmed, not re-decided)
- The wrapped surface is the `mlrs-algos` trait set: `Fit` (returns `&mut self`),
  `Predict` (regressors), `Transform`/`inverse_transform` (PCA/TruncatedSVD),
  `PredictLabels` (clustering/classifier, i32 labels), `KNeighbors` (distances +
  i32 indices), `PredictProba` (per-class fractions). i32 everywhere for labels /
  indices (D-06 Phase 5) → numpy `int32` at egress.
- Rust estimators take an explicit `&mut BufferPool<ActiveRuntime>` and `(rows,
  cols)` per call; fitted state is device-resident (P4 D-03), host-materialized at
  accessors (`.coef(pool) -> Vec<F>`). The PyO3 layer owns the pool + client and
  releases the GIL (`Python::allow_threads`) around device compute (PY-03).
- Feature-free `#[cube]` kernels in `mlrs-kernels`; runtime-bound code in
  `mlrs-backend`; estimators in `mlrs-algos`; **`mlrs-py` is the single cdylib +
  global allocator site**. Source/test separation per AGENTS.md (tests in
  `crates/mlrs-py/tests/` + Python `pytest`). `thiserror` in libs / `anyhow` at
  the binding boundary. Deps track latest.

### Claude's Discretion
- Exact PyO3 wrapper/enum shape for dtype dispatch (D-06); module/file layout of
  the `mlrs` Python package and `_mlrs` extension; which sklearn mixins each
  estimator composes.
- BufferPool + cubecl client **ownership/lifecycle across the boundary** (process-
  global vs per-estimator) and thread-safety semantics under
  `Python::allow_threads` / joblib — flagged for research/planning; has
  user-visible concurrency implications but no decision was forced this phase.
- The **subset of `sklearn.utils.estimator_checks`** treated as "relevant"
  (criterion 1) per estimator family — researcher/planner to scope; the shim
  (D-01) is chosen specifically to make the standard checks attainable.
- Exact `get_params`/`set_params` hyperparameter names per estimator — must match
  scikit-learn naming (PY-02); planner pins each from the sklearn reference.
- The **maturin multi-distribution mechanism** (dynamic dist name vs per-backend
  pyproject vs `--features` + name override) — ROADMAP research flag; see Open
  Questions. Constraint: same `import mlrs`, distinct dist name, `abi3-py312`,
  import-time driver probe.
- `score()` metric per family (R² for regressors, accuracy for classifiers) —
  inherit from sklearn mixins where the shim subclasses them (D-01).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning context
- `.planning/PROJECT.md` — core value, constraints, out-of-scope, key decisions
  (note: documents the old cpu+wgpu gate — D-07 supersedes it with cpu+rocm)
- `.planning/REQUIREMENTS.md` — PY-01..PY-05 requirement text + traceability table
- `.planning/ROADMAP.md` §"Phase 6: Python Surface — PyO3 Estimators &
  Per-Backend Wheels" — goal + 4 success criteria (the gate) + the maturin
  multi-distribution research flag
- `.planning/phases/05-distance-based-iterative-solver-estimators/05-CONTEXT.md`
  — the extended trait surface (D-05/D-07: PredictLabels/KNeighbors/PredictProba),
  i32 labels/indices (D-06), device-resident state (D-03), the cpu+rocm gate
  (D-07), and the explicit "Phase-6 wraps this surface generically" intent
- `.planning/phases/04-closed-form-estimators/04-CONTEXT.md` — the base
  Fit/Predict/Transform surface (D-04), device-resident fitted state + lazy host
  materialize (D-03), center-then-solve intercept

### cuML reference for the I/O method (read-only — the method this phase mirrors)
- `cuml-main/python/cuml/cuml/internals/base.py` — `Base.output_type`,
  `_get_output_type`/`_set_output_type`, `_get_param_names` (the shim + egress
  routing reference — D-01/D-03)
- `cuml-main/python/cuml/cuml/internals/array.py` — `CumlArray`, `to_output`,
  `output_type` resolution (egress reference — D-03)
- `cuml-main/python/cuml/cuml/internals/input_utils.py` — `input_to_cuml_array`
  (cuML's ingress; mlrs adapts the accept-anything *surface* but crosses via
  Arrow PyCapsule per D-02, NOT cuML's array-interface boundary)
- `cuml-main/python/cuml/cuml/internals/mixins.py` — sklearn mixin pattern the
  shim composes (Regressor/Classifier/Cluster/Transformer)

### Existing mlrs source this phase wraps / extends
- `crates/mlrs-algos/src/traits.rs` — the wrapped surface (Fit/Predict/Transform/
  PredictLabels/KNeighbors/PredictProba) + their exact signatures
- `crates/mlrs-algos/src/{linear,decomposition,cluster,neighbors}/` — the 11
  estimators and their host accessors (`.coef(pool)`, `.intercept(pool)`,
  `labels_`, `cluster_centers_`, …) and constructor hyperparameters
- `crates/mlrs-backend/src/bridge.rs` — `validate_f32`/`validate_f64` Arrow
  validation bridge **reused at the PyCapsule boundary** (D-02); the
  "validated single-upload" (not literal zero-copy) semantics
- `crates/mlrs-backend/src/capability.rs` — backend capability layer +
  `skip_f64_with_log`; basis for the D-04 capability-query + clear error
- `crates/mlrs-backend/src/runtime.rs` — feature-selected `ActiveRuntime` /
  `ActiveDevice` + client creation (the import-time probe target, D-08)
- `crates/mlrs-backend/src/{device_array.rs, pool.rs}` — `DeviceArray::from_host`/
  `to_host`, `BufferPool` (the pool the PyO3 layer owns across the boundary)
- `crates/mlrs-py/src/{lib.rs, allocator.rs}` — current cdylib + mimalloc global
  allocator (FOUND-09); `crate-type = ["cdylib", "rlib"]`; the crate Phase 6 fills
- `crates/mlrs-py/Cargo.toml` — where the PyO3 + backend-feature wiring lands
- `Cargo.toml` (workspace) — `[workspace.dependencies]` single-source pins; add
  PyO3 / pyo3 build deps here (track latest)
- `scripts/gen_oracle.py` — sklearn oracle fixture generator; extend for the
  Python `pytest` oracle tests (criterion 1)

### Build / packaging protocol (MANDATORY before writing bindings or wheels)
- `AGENTS.md` — source/test separation; CubeCL generics-over-float; build-error
  protocol
- `/home/user/Documents/workspace/optimisor/manual/` — zero-copy Arrow↔CubeCL
  guidance (informs the D-02 PyCapsule import + the validated-single-upload bridge)
- maturin documentation — per-feature distribution naming + `abi3-py312`
  (research-flagged; see Open Questions). Use the `find-docs`/`ctx7` flow to fetch
  current maturin + PyO3 docs during research.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **The `mlrs-algos` trait surface (Fit/Predict/Transform/PredictLabels/
  KNeighbors/PredictProba):** the single, uniform contract the PyO3 layer wraps
  generically — written once per trait family, not per estimator.
- **`bridge.rs` `validate_f32`/`validate_f64`:** the Arrow validation path is
  reused verbatim at the PyCapsule boundary (D-02) — no new validation code.
- **`capability.rs` + `skip_f64_with_log`:** the backend-capability source for the
  D-04 f64-on-incapable-backend error (capability flag already modeled).
- **`runtime.rs` feature-selected `ActiveRuntime` + client creation:** the
  import-time driver probe (D-08) wraps this; per-backend wheels are just this
  feature switch × maturin (D-07).
- **`DeviceArray::from_host`/`to_host` + `BufferPool` + host accessors
  (`.coef(pool)`):** the device⇄host materialization the shim's egress (D-03)
  consumes.
- **`mlrs-py` cdylib + mimalloc allocator (FOUND-09):** the crate is already the
  single cdylib/allocator site — Phase 6 adds PyO3 inside it, no new crate.

### Established Patterns
- Feature-free kernels in `mlrs-kernels`; runtime-bound code in `mlrs-backend`;
  estimators in `mlrs-algos`; bindings + allocator in `mlrs-py`. Phase 6 stays in
  `mlrs-py` (Rust) + a new Python package (the shim, D-01).
- scikit-learn is the API + numerical contract (1e-5); cuML is the *method*
  reference for I/O ergonomics (D-02/D-03) but NOT the numerical oracle.
- Explicit `(rows, cols)` geometry + flat row-major buffers (P2 D-04) — the
  PyCapsule carries a 1-D float array + a shape tuple (D-02).

### Integration Points
- **`mlrs-py` ↔ `mlrs-algos`:** the PyO3 layer calls the trait methods, owning the
  `BufferPool`/client and releasing the GIL around them (PY-03).
- **`mlrs-py` ↔ `mlrs-backend`:** capability query (D-04) + runtime client init /
  import probe (D-08) go through `mlrs-backend`.
- **Python shim ↔ `_mlrs` extension:** shim does numpy/list → pyarrow
  normalization (D-02), `output_type` resolution (D-03), and sklearn
  BaseEstimator/mixin behavior (D-01); `_mlrs` does dtype dispatch (D-06) + device
  compute.
- **maturin ↔ workspace:** builds `mlrs-py` once per backend feature into distinct
  dist names sharing the `mlrs` import (D-07).

</code_context>

<specifics>
## Specific Ideas

- **The whole phase is "follow cuML's method" applied to a Rust core.** Topology
  (Python shim over compiled core, D-01) and egress (`output_type` mirror routing,
  D-03) are taken directly from cuML; ingress (D-02) takes cuML's accept-anything
  *surface* but deliberately crosses the boundary via the Arrow PyCapsule to honor
  the locked PY-03 contract.
- **Correctness over convenience at the dtype boundary (D-04):** unlike cuML's
  silent dtype conversion, mlrs raises a clear error rather than downcast f64→f32,
  because rocm's f64-incapability is a hard limit and silent precision loss would
  break the 1e-5 promise.
- **One import, many dist names, install exactly one (D-07):** code portability
  across backends is the priority; the shared `mlrs` namespace is an accepted
  constraint, not a bug.
- **`pyarrow` is an accepted runtime dependency** of the `mlrs` package (follows
  from the D-02 normalize-to-pyarrow ingress).

</specifics>

<deferred>
## Deferred Ideas

- **cupy / cuDF / numba output_type targets** — cuML mirrors all of these; mlrs v1
  ships numpy + pyarrow only (D-03). Add the others if/when a device-array
  interchange (e.g. `__cuda_array_interface__`/DLPack export) is built.
- **Full cuML array-interface/DLPack ingress** — rejected for v1 (would amend
  PY-03); revisit if a future milestone relaxes the Arrow-PyCapsule boundary
  mandate or adds zero-copy device-array ingest.
- **Multiple backends installable side-by-side** (distinct import names) —
  rejected for v1 (D-07 chose code portability); revisit only if a real need to
  switch backends in one process emerges.
- **`cuml.accel`-style transparent sklearn acceleration** — already out of v1
  scope (PROJECT.md, V2-07); not part of this phase.
- **Multi-GPU / Dask Python surface** — out of v1 scope (V2-06).

### Reviewed Todos (not folded)
None — no pending todos matched this phase.

## Open Questions for Research (run `/gsd-plan-phase --research-phase 6`)
- **Maturin multi-distribution naming** (ROADMAP research flag) — how to produce
  `mlrs-cpu`/`mlrs-wgpu`/`mlrs-cuda`/`mlrs-rocm` distribution names from one source
  with a constant `import mlrs` and `abi3-py312`: dynamic dist name in
  `pyproject.toml` vs per-backend pyproject vs `maturin --features` + name
  override. Undocumented in maturin's first-party docs — confirm the pattern.
- **Arrow PyCapsule import in arrow-rs** (D-02) — exact arrow-rs FFI entry point
  for consuming an `__arrow_c_array__` capsule with correct release-callback
  ownership; confirm it composes with the existing `validate_f32`/`validate_f64`
  bridge and that 2-D `X` as 1-D-array-plus-shape is the right shape contract.
- **PyO3 + abi3-py312 + GIL release** — current PyO3 patterns for `abi3-py312`,
  `Python::allow_threads` around `&mut BufferPool` compute, and exposing
  capability flags; verify thread-safety of the pool/client across the boundary.
- **sklearn `estimator_checks` subset** (D-01) — which checks are "relevant" per
  estimator family, and which require shim adjustments (tags, `_get_tags`,
  input validation) to pass.
- **Import-time driver probe cost/safety** (D-08) — confirm probing the
  cubecl client at import is safe and cheap enough as an import side-effect for
  each backend (esp. rocm/cuda driver detection).
- **pytest oracle harness for Python** (criterion 1) — extend `gen_oracle.py` /
  fixtures so the Python layer re-validates the 1e-5 contract through the full
  numpy→pyarrow→PyCapsule→device path, not just the Rust layer.

</deferred>

---

*Phase: 6-Python Surface — PyO3 Estimators & Per-Backend Wheels*
*Context gathered: 2026-06-13*
