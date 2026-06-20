# Phase 8: Kernel Family - Pattern Map

**Mapped:** 2026-06-21
**Files analyzed:** 12 new/modified
**Analogs found:** 11 / 12 (1 partial — KernelDensity device log-sum-exp has no exact analog)

This phase is **additive** (new prim + 2 estimators + 1 trait + PyO3 wrappers + oracle
generators). Every heavy operation already exists as a validated v1 prim; the only genuinely
new device code is one SharedMemory-free elementwise-map kernel (`kernel_matrix` map) and a
small KD log-sum-exp helper over the v1 `reduce` prim. Pattern reuse is therefore very high.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-backend/src/prims/kernel_matrix.rs` | prim (host orchestration) | transform (base-op → elementwise-map) | `crates/mlrs-backend/src/prims/covariance.rs` | exact (GEMM/distance → in-place map idiom) |
| `crates/mlrs-kernels/src/elementwise.rs` (extend) or new map file | kernel (device `#[cube(launch)]`) | transform (per-element map) | `scale` / `dist_combine_clamp` in `elementwise.rs` | exact |
| `crates/mlrs-backend/src/prims/mod.rs` | config (module registry) | — | existing `pub mod` lines in `prims/mod.rs` | exact |
| `crates/mlrs-algos/src/traits.rs` (add `ScoreSamples<F>`) | trait (estimator surface) | request-response | `PartialFit<F>` / `Predict<F>` in same file | exact |
| `crates/mlrs-algos/src/kernel_ridge/…` (new module group) | estimator (model) | CRUD-ish (fit/predict dual solve) | `crates/mlrs-algos/src/linear/ridge.rs` MINUS centering | role+flow match (with two deletions) |
| `crates/mlrs-algos/src/density/…` (new KD home) | estimator (model) | request-response (score_samples log-density) | `ridge.rs` (estimator skeleton) + `distance.rs` compose | partial (no exact log-density analog) |
| `crates/mlrs-algos/src/error.rs` (extend `AlgoError`) | model (error enum) | — | existing struct variants (`InvalidAlpha` / `InvalidEps` / `InvalidBatchSize`) | exact |
| `crates/mlrs-algos/src/lib.rs` (register modules) | config (module registry) | — | existing `pub mod` + `pub use` lines | exact |
| `crates/mlrs-py/src/estimators/kernel.rs` (NEW) | controller (PyO3 wrapper) | request-response | `crates/mlrs-py/src/estimators/covariance.rs` | exact (`any_estimator!` + dispatch) |
| `crates/mlrs-py/src/lib.rs` + `estimators/mod.rs` (register) | config (pyclass registry) | — | existing `add_class` / `pub mod` lines | exact |
| `crates/mlrs-backend/tests/kernel_matrix_test.rs` | test | — | `incremental_svd_test.rs` (PoolStats gate) | exact (gate shape) |
| `crates/mlrs-algos/tests/kernel_ridge_test.rs` + `kernel_density_test.rs` | test | — | `ridge_test.rs` (oracle harness) | exact |
| `scripts/gen_oracle.py` (extend) | config (oracle generators) | — | `gen_ridge` / `gen_covariance` + `main()` loop | exact |

---

## Pattern Assignments

### `crates/mlrs-backend/src/prims/kernel_matrix.rs` (prim, transform) — KEYSTONE

**Analog:** `crates/mlrs-backend/src/prims/covariance.rs` (GEMM → in-place elementwise-map idiom).
**Base-op signatures (D-03):** `distance.rs:79` and `gemm.rs:54`.

**Imports pattern** (`covariance.rs:42-52`):
```rust
use bytemuck::Pod;
use cubecl::prelude::*;
use mlrs_core::PrimError;
use mlrs_kernels::{center_columns, scale};        // → import the NEW kernel-map fns here
use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gemm::gemm;                      // linear/poly/sigmoid base (D-03)
use crate::prims::reduce::{column_reduce, ReducePath, ScalarOp};  // KernelRidge uses distance for RBF
use crate::runtime::ActiveRuntime;
```
For `kernel_matrix.rs` add `use crate::prims::distance::distance;` (RBF base, `distance.rs:79`).

**Base-op dispatch + in-place map** (the load-bearing idiom — `covariance.rs:151-204`):
```rust
// RBF branch: squared-euclidean (sqrt=false) base, then exp(-gamma·sqdist) in place.
// distance::<F>(pool, x, (rows_x,cols), y, (rows_y,cols), false, out) -> Result<DeviceArray, PrimError>
let base = distance::<F>(pool, x, (rows_x, cols), y, (rows_y, cols), false, out)?;
// linear/poly/sigmoid branch: gemm XYᵀ (transb=true) base.
// gemm::<F>(pool, x, (rows_x,cols), y, (cols,rows_y), false, true, out) -> Result<DeviceArray, PrimError>

// Per-element map IN PLACE over the base buffer (input handle == output handle) —
// copied verbatim from covariance.rs:190-200:
let n = rows_x * rows_y;
let client = pool.client().clone();
let (count, dim) = launch_dims_1d(n);
let in_arg  = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
let out_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
rbf_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, gamma /* scalar F by value */);
Ok(base)   // result IS the base buffer, mapped in place (D-02/D-03 single code path)
```

**Geometry validation BEFORE launch** (mirror `covariance.rs:212-262`; ASVS V5 / T-04-01-01):
validate `rows_x*cols == x.len()`, `rows_y*cols == y.len()`, reject empty geometry, validate
`out` len == `rows_x*rows_y` — return `PrimError::ShapeMismatch`.

**`launch_dims_1d` helper** (copy verbatim — `covariance.rs:266-273`): 256-wide ceiling-div
`CubeCount::Static(cubes.max(1),1,1)`, `CubeDim { x:256, y:1, z:1 }`.

**Kernel-type representation (D-01):** typed `Kernel<F>` enum `{ Linear, Rbf { gamma },
Poly { gamma, degree, coef0 }, Sigmoid { gamma, coef0 } }`; `kernel_matrix` matches on it to
pick base op + map. `linear` is identity (skip the map launch, return the GEMM buffer directly).

---

### `crates/mlrs-kernels/src/elementwise.rs` (kernel, device map) — extend or add a sibling file

**Analog:** `scale` (`elementwise.rs:64-70`) and `dist_combine_clamp` (`elementwise.rs:107-127`).

**Map-kernel shape** (copy `scale`'s exact shape — `elementwise.rs:64-70`):
```rust
#[cube(launch)]
pub fn rbf_map<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, gamma: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        output[tid] = F::exp(-gamma * input[tid]);   // input is squared-euclidean dist
    }
}
// poly:    F::powf(gamma * g + coef0, degree)        // powf = sklearn-faithful (A3); real degree
// sigmoid: F::tanh(gamma * g + coef0)
// linear:  identity — no kernel needed
```
**Transcendental form (Pitfall 7):** ALWAYS `F::exp(x)` / `F::tanh(x)` / `F::powf(x,y)` /
`F::log(x)` / `F::cos(x)` — static associated fns, NOT `x.exp()`. Scalar `F` params passed by
value require the `CubeElement` bound (doc comment `elementwise.rs:18-23`).

**Compact-support guard (D-11, KD kernels) — STATEMENT form** (copy the clamp idiom from
`clamp_nonneg`/`dist_combine_clamp` — `elementwise.rs:38-48` / `120-126`):
```rust
// NEVER an if-expression, NEVER F::INFINITY (cpu-MLIR landmine, Pitfall 3).
let mut val = /* kernel value, e.g. F::new(1.0) - d2 / (h*h) */;
let zero = F::from_int(0i64);
if d >= h { val = zero; }     // out-of-support → exact 0 in the LINEAR domain
output[tid] = val;
```
SharedMemory-free, no atomics, F/u32 accumulators only — the doc comment in
`elementwise.rs:1-27` is the precedent to mirror in the new file's module doc.

---

### `crates/mlrs-algos/src/traits.rs` — add `ScoreSamples<F>` (D-12)

**Analog:** `PartialFit<F>` (`traits.rs:86-104`) — the most recent trait addition; same bound,
same `pool`/`DeviceArray`/explicit-`(rows,cols)` device-resident convention.

**Imports already present** (`traits.rs:36-43`): `bytemuck::Pod`,
`cubecl::prelude::{CubeElement, Float}`, `mlrs_backend::{device_array::DeviceArray,
pool::BufferPool, runtime::ActiveRuntime}`, `crate::error::AlgoError`.

**Trait shape** (mirror `Predict<F>` at `traits.rs:109-122` but returning length-`n`
log-densities — NOT Predict semantics, D-12):
```rust
/// Compute per-sample log-density (length-n), NOT Predict semantics (D-12).
pub trait ScoreSamples<F>
where
    F: Float + CubeElement + Pod,
{
    fn score_samples(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;   // length n_samples log-densities
}
```

**Registration:** add to the `lib.rs` re-export list (`lib.rs:` `pub use traits::{Fit, KNeighbors,
PartialFit, Predict, PredictLabels, PredictProba, Transform, ScoreSamples};`).

---

### `crates/mlrs-algos/src/kernel_ridge/…` (estimator) — mirror `ridge.rs` MINUS centering

**Analog:** `crates/mlrs-algos/src/linear/ridge.rs` (closed-form `(XᵀX+αI)` Cholesky solve).
**Module layout analog:** `covariance/mod.rs` (a `mod.rs` index + per-estimator file; the
estimator plan adds its own `pub mod` line, does NOT edit `lib.rs`).

**Struct + accessors** (`ridge.rs:68-122`): device-resident `Option<DeviceArray>` fitted state,
`new(...)` constructor, host accessor pattern with `NotFitted` error. KernelRidge stores
`dual_coef_` (n×t), the fitted training `X_fit_`, and the resolved `Kernel<F>`.

**Fit: validate → kernel → diagonal-α → multi-RHS solve** (mirror `ridge.rs:128-310`):
```rust
// 1. validate hyperparameters BEFORE launch (ridge.rs:137-166): alpha>=0, degree>=1,
//    geometry. gamma=None → 1/n_features at fit (D-05). NO centering (delete ridge.rs:168-223).
// 2. K = kernel_matrix(X, X, kernel)  — Y=X (D-02), n×n.
// 3. α on the K DIAGONAL only — copy ridge.rs:248-254 host pass verbatim:
let mut k_host = k.to_host(pool);
for i in 0..n { let d = host_to_f64(k_host[i*n+i]) + alpha64; k_host[i*n+i] = f64_to_host::<F>(d); }
k.release_into(pool);
let k_reg = DeviceArray::from_host(pool, &k_host);   // recycles released n² (D-11 gate 2)
// 4. multi-RHS Cholesky solve (D-04 near-free) — mirror ridge.rs:276-280:
//    cholesky_solve::<F>(pool, a, b, n, rhs, out) ; b is n×t, returns n×t.
let k_out = DeviceArray::from_raw(k_reg.handle().clone(), n*n);
let dual = cholesky_solve::<F>(pool, &k_reg, &y, n, n_targets, Some(k_out))?;   // dual_coef_ (n×t)
```
**Delete** the centering / `x_mean` / `y_mean` / `intercept_` block (`ridge.rs:168-223, 282-295`)
— sklearn KernelRidge fits RAW data, NO intercept (D-06 / Pitfall 1).

**Predict** (mirror `ridge.rs:313-373` but kernel-based): `K_test = kernel_matrix(X_test, X_fit_,
kernel)` (m×n), then `y_pred = K_test · dual_coef_` (gemm, m×t). No intercept broadcast.

**Host f64 conversion helpers** (copy verbatim — `ridge.rs:378-393`): `host_to_f64` / `f64_to_host`.

---

### `crates/mlrs-algos/src/density/…` (estimator) — composes `distance` + new log-sum-exp

**Analog:** `ridge.rs` for the estimator skeleton (struct/new/accessors/fit-validate); the
`distance.rs:79` compose for the kernel base; **no exact analog** for the device log-sum-exp.
**Module home:** new `density/` module (RESEARCH Open Q2 recommendation) or `neighbors/`.

**Fit:** store the training `X_fit_`, resolve `bandwidth` (numeric or scott/silverman host
closed-form, D-09 — pinned in RESEARCH §Bandwidth), validate `bandwidth>0` + kernel name.

**`ScoreSamples::score_samples`** (D-08/D-11 — composes v1 prims, NOT `kernel_matrix`):
```rust
// 1. D = distance(Q, X_fit_, sqrt=…)  (m×n). gaussian/epanechnikov use sqrt=false (squared);
//    exponential/linear/cosine/tophat use sqrt=true (raw dist) — Pitfall 4.
// 2. per-element KD map (LINEAR domain, exact 0 out of support — D-11): kernel VALUE, never log.
// 3. per-query (row) log-sum-exp via v1 reduce:
//    reduce::row_reduce(pool, k, m, n, ScalarOp::Max, ReducePath::Shared)  → row_max (optional rescale)
//    reduce::row_reduce(pool, k, m, n, ScalarOp::Sum, ReducePath::Shared)  → row_sum
//    lse_row = log(row_sum) [+ log(row_max) if rescaled]
// 4. log_density = lse_row + log_norm(h,d,kernel) − log(N)   [log_norm host-side f64, RESEARCH §log_norm]
```
**Reduce signatures** (`reduce.rs:180` `row_reduce`, `reduce.rs:138`/`88` full `max`/`sum`;
`ScalarOp` variants `Sum`/`Max` at `reduce.rs:259-273`). `row_reduce` always force
`ReducePath::Shared` (cpu-portable; the plane path returns `None` on non-subgroup adapters —
`reduce.rs:192-195`).
**Anti-pattern:** never `F::INFINITY` / `−∞` / `log` inside the per-element map (Pitfall 3).

---

### `crates/mlrs-algos/src/error.rs` — extend `AlgoError`

**Analog:** existing struct variants `InvalidAlpha` (`error.rs:54-60`), `InvalidEps`
(`error.rs:111-117`), `InvalidBatchSize` (`error.rs:186-191`). Same `#[error("…")]` +
`{ estimator: &'static str, … }` style.

**New variants** (RESEARCH §New AlgoError variants):
```rust
#[error("estimator '{estimator}': bandwidth = {bandwidth} is invalid (must be > 0)")]
InvalidBandwidth { estimator: &'static str, bandwidth: f64 },
#[error("estimator '{estimator}': degree = {degree} is invalid (must be >= 1)")]
InvalidDegree { estimator: &'static str, degree: f64 },
#[error("estimator '{estimator}': kernel '{kernel}' is not supported")]
InvalidKernel { estimator: &'static str, kernel: String },
// alpha>=0 already covered by InvalidAlpha (reuse it). NotFitted/Prim(#[from]) reused as-is.
```

---

### `crates/mlrs-py/src/estimators/kernel.rs` (NEW) — `PyKernelRidge` + `PyKernelDensity`

**Analog:** `crates/mlrs-py/src/estimators/covariance.rs` (`any_estimator!` + dtype dispatch +
`py.detach` GIL release + `guard_f64()`). Macro at `dispatch.rs:85-108`.

**Imports** (`covariance.rs:15-22`):
```rust
use pyo3::prelude::*;
use mlrs_algos::{/* KernelRidge, KernelDensity, */ traits::{Fit, Predict, ScoreSamples}};
use crate::errors::{algo_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};
```

**`any_estimator!` invocation** (`covariance.rs:28-32`): store kernel NAME (u8/string tag) +
raw gamma/degree/coef0/bandwidth in `Unfit` (RESEARCH Open Q3); build the typed `Kernel<F>` at
`fit` once `n_features` is known.
```rust
crate::any_estimator! {
    any:   AnyKernelRidge,
    algo:  mlrs_algos::kernel_ridge::KernelRidge,
    unfit: { kernel: u8, alpha: f64, gamma: f64, degree: f64, coef0: f64 },
}
```

**Fit body — the two load-bearing contracts** (copy verbatim — `covariance.rs:74-104`):
```rust
let fitted = py.detach(|| -> PyResult<AnyKernelRidge> {        // PY-03 GIL release
    let mut pool = crate::global_pool().lock().expect("pool mutex");
    match dt {
        FloatDtype::F32 => { let xd = validated_f32(as_f32(&xa)?, &mut pool)?; /* est.fit(... None? Some(y)) */ }
        FloatDtype::F64 => { crate::capability::guard_f64()?;   // D-04 BEFORE upload
                             let xd = validated_f64(as_f64(&xa)?, &mut pool)?; /* … */ }
    }
})?;
```

**`score_samples` — the new exposed method** (KernelDensity; mirror the fit dispatch shape +
RESEARCH §Code Examples). Dtype-suffixed accessors `dual_coef_f32/_f64`, `log_density_f32/_f64`
(mirror `covariance_f32/_f64` at `covariance.rs:106-147`); `is_fitted`/`dtype` helpers
(`covariance.rs:148-157`).

**Registration:** `estimators/mod.rs` add `pub mod kernel;`; `lib.rs` add
`use estimators::kernel::{PyKernelRidge, PyKernelDensity};` + two `m.add_class::<…>()?;`
(mirror `lib.rs:137,162-163`).

---

### `crates/mlrs-backend/tests/kernel_matrix_test.rs` (PRIM-08 values + PoolStats gate)

**Analog:** `crates/mlrs-backend/tests/incremental_svd_test.rs` (PoolStats memory gate shape).

**Fixture loader + f64 helper** (copy `incremental_svd_test.rs:48-64`): workspace-root
`fixture(name)` (`tests/fixtures/<name>.npz`), `from_f64::<F>` bytemuck cast.

**Value test:** load committed `kernel_matrix_*` oracle, compute K vs host reference per kernel,
`assert_slice_close` (f64 strict `F64_TOL`, f32 documented band `Tolerance::new(1e-4,1e-4)` —
mirror `F32_MERGE_TOL` at `incremental_svd_test.rs:45`).

**f64 capability gate** (copy verbatim — `incremental_svd_test.rs:170-181`):
```rust
let backend = capability::active_backend_name();
capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
if capability::skip_f64_with_log() { println!("… SKIPPED"); return; }
```

**PoolStats memory gate** (copy the structure — `incremental_svd_test.rs:201-269`): drive
`kernel_matrix` N times at fixed shape, collect `pool.stats().live_bytes`/`peak_bytes`, assert
`live_after[w] <= live_after[1]` (no growth after warmup) and `peak_after` plateaus.

---

### `crates/mlrs-algos/tests/kernel_ridge_test.rs` + `kernel_density_test.rs`

**Analog:** `crates/mlrs-algos/tests/ridge_test.rs` (sklearn oracle harness).

**Harness** (copy `ridge_test.rs:23-126`): `fixture(name)`, `host_to_f64`/`f64_to` bytemuck
helpers, `assert_close(got, expected, tol, what)` numpy-allclose form (`ridge_test.rs:71-89`),
per-case fit-and-materialize driver, `load_npz` + `case.expect_f64("…")`.

**KernelRidge oracle cases** (RESEARCH §Oracle cases): one per kernel (linear/rbf/poly/sigmoid),
one 2-target (multi-RHS, D-04), one explicit-gamma + one gamma=None (D-05), at least one
`degree=3, coef0=1` default case. f64 strict `F64_TOL`; f32 documented band.

**KernelDensity oracle cases** (RESEARCH §KD oracle, D-10): per-kernel (all 6), at least one
`bandwidth='scott'` + one `'silverman'`, atol=0/rtol=0 forced-exact sklearn. **Documented KD
tolerance** (not strict 1e-5) per KERNEL-02 wording.

**f64 capability gate** (copy verbatim — `ridge_test.rs:171-179`): `skip_f64_with_log()` →
print SKIPPED + return.

---

### `scripts/gen_oracle.py` — add `gen_kernel_matrix`, `gen_kernel_ridge`, `gen_kernel_density`

**Analog:** `gen_ridge` (`gen_oracle.py:968-1010`) and `gen_covariance`.

**Generator shape** (copy `gen_ridge` — `gen_oracle.py:968-1010`):
```python
def gen_kernel_ridge(seed=SEED, dtype=np.float32) -> str:
    from sklearn.kernel_ridge import KernelRidge
    rng = np.random.default_rng(seed)                       # authoritative byte-repro RNG
    x = rng.standard_normal((N_SAMPLES, N_FEATURES)); y = …
    reg = KernelRidge(kernel=…, alpha=…, gamma=…, degree=3, coef0=1).fit(x, y)
    def c(arr): return np.ascontiguousarray(np.asarray(arr)).astype(dtype)   # row-major (PCA fix)
    out_path = os.path.join(_FIXTURE_DIR, f"kernel_ridge_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, X=c(x), y=c(y), X_test=c(x_test), y_pred=c(reg.predict(x_test)), …)
    return out_path
```
KernelDensity generator: `from sklearn.neighbors import KernelDensity`, fit with
`atol=0, rtol=0` (D-10 forced-exact), store `score_samples(Q)`.

**Registration in `main()`** (`gen_oracle.py:1342-1349` precedent — both dtypes):
```python
for dtype in (np.float32, np.float64):
    print(f"wrote {gen_kernel_matrix(dtype=dtype)}")
for dtype in (np.float32, np.float64):
    print(f"wrote {gen_kernel_ridge(dtype=dtype)}")
for dtype in (np.float32, np.float64):
    print(f"wrote {gen_kernel_density(dtype=dtype)}")
```
Regen requires a `/tmp` venv with numpy+scipy+sklearn (PEP 668); fixtures are committed `.npz`
blobs, CI never runs Python.

---

## Shared Patterns

### Validate-before-launch (ASVS V5 / T-04-01-01)
**Source:** `ridge.rs:137-166`, `covariance.rs:212-262`.
**Apply to:** `kernel_matrix.rs`, `KernelRidge`, `KernelDensity`.
All hyperparameter guards (`alpha>=0`, `bandwidth>0`, `degree>=1`, kernel-name) and geometry
checks run BEFORE any `unsafe` `ArrayArg::from_raw_parts` launch; return a typed
`AlgoError`/`PrimError`, never an out-of-bounds device read.

### f32/f64 symmetry + `skip_f64_with_log` gate
**Source:** `incremental_svd_test.rs:170-181`, `ridge_test.rs:171-179`.
**Apply to:** every test (prim + estimator). f64 cases behind `capability::skip_f64_with_log()`
(cpu runs f64; rocm skips-with-log); f32 runs everywhere with a documented per-family band for
KernelRidge predictions + KernelDensity log-density (Claude's discretion in CONTEXT).

### Host f64 round-trip for tiny serial passes
**Source:** `ridge.rs:378-393` (`host_to_f64`/`f64_to_host`), `ridge.rs:248-254` (diagonal-α).
**Apply to:** KernelRidge diagonal-α injection; KD host-side `log_norm`/bandwidth f64 computation
(lgamma/log in std/`libm` — A1). cubecl 0.10 has no in-place device scalar write, so materialize
the tiny n×n / length-d host vector, mutate, re-upload (recycles the released buffer — D-11).

### GIL release + f64 guard (PyO3)
**Source:** `dispatch.rs:21-54` (doc), `covariance.rs:84-101`.
**Apply to:** every device-touching `#[pymethods]` in `estimators/kernel.rs`. Wrap the trait
call in `py.detach(|| { global_pool().lock()… })`; call `guard_f64()?` BEFORE the F64 arm upload.

### Module-index registration discipline
**Source:** `covariance/mod.rs`, `prims/mod.rs`, `algos lib.rs`.
**Apply to:** new estimator modules edit their own `mod.rs` + add a `pub mod` line; `lib.rs`
re-export and the prim `mod.rs` registration are the only shared-file edits. Keeps plans
file-disjoint and parallel-safe (the Wave-0 scaffold owns `lib.rs`).

### STATEMENT-form conditional + transcendental static-fn form (cpu-MLIR safety)
**Source:** `elementwise.rs:38-48` (clamp statement), doc `elementwise.rs:10-27`.
**Apply to:** every new map kernel. `if d < zero { d = zero; }` statement (NOT if-expression /
`max()`); `F::exp(x)`/`F::tanh(x)`/`F::powf(x,y)`/`F::log(x)`/`F::cos(x)` static assoc fns;
SharedMemory-free, no atomics, F/u32 accumulators only, NEVER `F::INFINITY`/`−∞`.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| KD device log-sum-exp helper (inside `density/…` or a small `mlrs-backend` helper) | prim/estimator-internal | reduce + map | No existing device log-sum-exp; it COMPOSES v1 `reduce::row_reduce` (Max/Sum, `reduce.rs:180`) + an `exp`/`scale`/`log` map (mirror `scale` shape). Linear-domain max-rescale (D-11) is novel but assembled entirely from validated prims — partial analog only. RESEARCH Open Q1: implement plain reduce-sum first, add reduce-max rescale only if the f32 band fails. |

---

## Metadata

**Analog search scope:** `crates/mlrs-backend/src/prims/` (covariance, distance, gemm, cholesky,
reduce), `crates/mlrs-kernels/src/elementwise.rs`, `crates/mlrs-algos/src/{traits.rs, error.rs,
linear/ridge.rs, covariance/mod.rs, lib.rs}`, `crates/mlrs-py/src/{dispatch.rs,
estimators/covariance.rs, estimators/mod.rs, lib.rs}`,
`crates/mlrs-backend/tests/incremental_svd_test.rs`, `crates/mlrs-algos/tests/ridge_test.rs`,
`scripts/gen_oracle.py`.
**Files scanned:** 16 read in full / targeted.
**Pattern extraction date:** 2026-06-21
