# Phase 10: SGD / Linear-SVM - Pattern Map

**Mapped:** 2026-06-21
**Files analyzed:** 13 new/modified (2 kernel/prim, 5 algos, 2 py, 1 oracle generator, +module-index edits)
**Analogs found:** 13 / 13 (every new file has a shipped in-tree analog; no external-pattern fallback needed)

This file maps each Phase-10 new/modified file to its closest shipped analog with
line-referenced excerpts the planner hands to executors. All Phase-10 files are
ASSEMBLY over validated v1/v2 primitives; the only irreducibly-new device code is
the two `sgd_*` GATHER kernels (`crates/mlrs-kernels/src/sgd.rs`).

> Convention note (D-01/D-08): Phase 10 INTRODUCES the builder pattern + split
> validation for its four new estimators. The analogs below are `new()` +
> validate-at-`fit` (Phases 4–9). The planner adapts: copy the analog's STRUCT
> SHAPE / fit-orchestration / device-residency / host-helper code verbatim, but
> replace the `new(positional)` constructor with a `*Builder` + `build() ->
> Result<_, BuildError>` that validates the data-INDEPENDENT params (the
> validation BODY moves out of `fit` into `build`, but the checks themselves are
> the same `if !(alpha >= 0.0)` shape seen in the analogs).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-kernels/src/sgd.rs` (NEW) | kernel (`#[cube]`) | transform / GATHER | `crates/mlrs-kernels/src/coordinate.rs` (`col_dot` + `residual_axpy`) | exact (two-pass single-owner GATHER) |
| `crates/mlrs-backend/src/prims/sgd.rs` (NEW) | prim (host orchestration) | batch / iterative | `crates/mlrs-backend/src/prims/coordinate_descent.rs` (`cd_solve` epoch loop) | exact (validate→loop→launch→scalar-readback) |
| `crates/mlrs-algos/src/linear/sgd_config.rs` (NEW) | config + enums + error | n/a (typing) | `kernel_ridge.rs` `KernelKind` (enum) + `error.rs` `AlgoError` (variants) | role-match |
| `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` (NEW) | estimator (classifier) | request-response (fit/predict) | `crates/mlrs-algos/src/linear/logistic.rs` (`LogisticRegression`) | exact (classifier: classes_ remap + PredictLabels + PredictProba) |
| `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` (NEW) | estimator (regressor) | request-response | `crates/mlrs-algos/src/linear/elastic_net.rs` (`ElasticNet`) | exact (regressor: Fit + Predict, alpha/l1_ratio) |
| `crates/mlrs-algos/src/linear/linear_svc.rs` (NEW) | estimator (classifier) | request-response | `crates/mlrs-algos/src/linear/elastic_net.rs` + `coordinate_descent.rs` (`cd_fit` reuse) | role-match (CD-reuse, intercept_scaling NOT center-then-solve) |
| `crates/mlrs-algos/src/linear/linear_svr.rs` (NEW) | estimator (regressor) | request-response | `crates/mlrs-algos/src/linear/elastic_net.rs` + `coordinate_descent.rs` | role-match (CD-reuse) |
| `crates/mlrs-algos/src/error.rs` (EDIT) | error type | n/a | existing `AlgoError` variants (`InvalidAlpha`/`InvalidL1Ratio`/`InvalidC`) | exact (add `BuildError` enum or new variants) |
| `crates/mlrs-py/src/estimators/linear.rs` (EDIT) | binding (`#[pyclass]`) | request-response (FFI) | `PyLogisticRegression` (same file, lines 632–806) | exact (classifier wrapper w/ predict_labels + predict_proba) |
| `crates/mlrs-py/src/errors.rs` (EDIT) | binding error mapping | n/a | `algo_err_to_py` (lines 55–57) | exact (add `build_err_to_py`) |
| `scripts/gen_oracle.py` (EDIT) | test fixture generator | file-I/O (npz blobs) | `gen_logistic` (lines 782–886) + `gen_elastic_net` | exact |
| `crates/mlrs-kernels/src/lib.rs`, `prims/mod.rs`, `algos/src/linear/mod.rs` (EDIT) | module index | n/a | existing `pub mod coordinate;` / `pub mod elastic_net;` lines | exact (Wave-0 scaffold owns these) |

---

## Pattern Assignments

### `crates/mlrs-kernels/src/sgd.rs` (kernel, two-pass GATHER)

**Analog:** `crates/mlrs-kernels/src/coordinate.rs`

This is the canonical cpu-MLIR-safe GATHER idiom. Copy BOTH the kernel structure
AND the module-doc safety contract verbatim — the planner's two kernels
(`sgd_margin` pass-1-per-sample, `sgd_weight_update` pass-2-per-coordinate) are
direct analogs of `col_dot` (GATHER into one accumulator) and `residual_axpy`
(over-provisioned per-element map).

**cpu-MLIR safety contract to copy** (coordinate.rs lines 19–34) — this exact
prose is the gate (`grep -c SharedMemory == 0` per RESEARCH §Validation):
```
//! The cpu(f64) backend's MLIR lowering rejects `SharedMemory` + mutable `bool`
//! flags + `F::INFINITY` consts + descending-shift loops (plan 05-02 hit this).
//! Both kernels here use ONLY `F`/`u32` accumulators and `if`-guarded forward
//! loops — no `SharedMemory`, no `bool`, no infinity sentinel, no
//! atomics/scatter
```

**Imports + re-export pattern** (coordinate.rs lines 36–40):
```rust
use cubecl::prelude::*;

pub use self::col_dot as cd_col_dot;       // → pub use self::sgd_margin as sgd_margin;
pub use self::residual_axpy as cd_residual_axpy;
```

**Pass-2 (per-coordinate GATHER) kernel — copy `col_dot` shape** (coordinate.rs lines 57–78). Note: scalar args BY VALUE (`rows: u32`, no `ScalarArg`); forward `while` scan; `F::from_int(0i64)` accumulator seed; no `bool`/no `SharedMemory`:
```rust
#[cube(launch)]
pub fn col_dot<F: Float + CubeElement>(
    x: &Array<F>, r: &Array<F>, out: &mut Array<F>,
    rows: u32, cols: u32, j: u32,
) {
    if UNIT_POS == 0 {
        let mut acc = F::from_int(0i64);
        let mut i = 0u32;
        while i < rows {
            let idx = (i * cols + j) as usize;
            acc += x[idx] * r[i as usize];
            i += 1u32;
        }
        out[0] = acc;
    }
}
```
For `sgd_weight_update`: one unit per coordinate `j` (bounds-checked
`if j < d as usize`, the over-provisioned form below), GATHER `grad_j = Σ_i
g[i]·x[i·d+j]`, then `w[j] = w[j] - eta*grad*inv_b`. The RESEARCH §Pattern-1
kernel body (RESEARCH lines 232–256) is the target.

**Pass-1 (per-sample, over-provisioned map) — copy `residual_axpy` bounds-check** (coordinate.rs lines 93–107):
```rust
#[cube(launch)]
pub fn residual_axpy<F: Float + CubeElement>(
    x: &Array<F>, r: &mut Array<F>, factor: F,
    rows: u32, cols: u32, j: u32,
) {
    let i = ABSOLUTE_POS;
    if i < rows as usize {                  // bounds-check — cubes over-provision
        let idx = i * cols as usize + j as usize;
        r[i] += factor * x[idx];
    }
}
```
For `sgd_margin`: `let i = ABSOLUTE_POS; if i < b as usize { ... p[i] = acc + bias; }`
with the forward `while j < d` dot inside (RESEARCH lines 206–226). `bias: F` is
a scalar passed BY VALUE.

---

### `crates/mlrs-backend/src/prims/sgd.rs` (prim, host epoch loop)

**Analog:** `crates/mlrs-backend/src/prims/coordinate_descent.rs` (`cd_solve`)

Copy the `validate-before-launch → host iteration loop → per-iter launch →
scalar/convergence readback → NotConverged-at-cap` shape.

**Convergence-cap constants** (coordinate_descent.rs lines 40, 52–54 / cd_fit lines 50–54):
```rust
pub const CD_DEFAULT_TOL: f64 = 1e-4;
pub const CD_DEFAULT_MAX_ITER: usize = 1000;
```
SGD analog: define `SGD_DEFAULT_MAX_ITER = 1000`, `SGD_DEFAULT_TOL = 1e-3`
(MBSGDClassifier sklearn default — RESEARCH §Defaults table). The pinned oracle
OVERRIDES with `tol=0` + fixed `max_iter` (Pitfall 2/7).

**Host loop + launch shape** (coordinate_descent.rs lines 137–222, abridged):
```rust
let max_iter = if max_iter == 0 { CD_MAX_ITER } else { max_iter };
for n_iter in 0..max_iter {
    // ... cd_col_dot::launch::<F, ActiveRuntime>(...) per coordinate ...
    // ... host scalar update (soft-threshold) ...
    // ... cd_residual_axpy::launch::<F, ActiveRuntime>(...) when coef changed ...
    let last_iter = n_iter + 1 == max_iter;
    if /* gap check cadence */ {
        cd_enet_gap::launch::<F, ActiveRuntime>(...);   // ONE scalar readback
    }
}
```
SGD analog: `for epoch in 0..max_iter { for batch in minibatches(no shuffle) {
sgd_margin::launch(...); host: g[i]=dloss(p_i,y_i), eta=schedule(t);
sgd_weight_update::launch(...); b -= eta*Σg*intercept_decay } }`. The schedule
arithmetic (`optimal` t0, invscaling) and the `dloss` table are HOST f64 — see
RESEARCH §"Code Examples" (`dloss` lines 490–504, `optimal_t0` lines 510–514).

**NotConverged map at cap** (coordinate_descent.rs lines 220–228 via cd_fit's `map_cd_error`):
```rust
fn map_cd_error(e: PrimError, estimator: &'static str, max_iter: usize) -> AlgoError {
    match e {
        PrimError::NotConverged { .. } => AlgoError::NotConverged { estimator, max_iter },
        other => AlgoError::Prim(other),
    }
}
```

**Standalone convex-objective gate (PRIM-10):** the prim must be validated on a
host-reference convex problem BEFORE any estimator wires it (RESEARCH §Validation
Criterion 1). The test home is `crates/mlrs-backend/tests/sgd_test.rs` (the
`cd_test.rs` precedent referenced in coordinate.rs line 33).

---

### `crates/mlrs-algos/src/linear/sgd_config.rs` (enums + TryFrom + SgdConfig + BuildError)

**Analog:** `kernel_ridge.rs` `KernelKind` (D-04 enum precedent) + `error.rs` (variant shape)

**Enum + `name()` shape** (kernel_ridge.rs lines 67–89) — copy for `Loss`/`Penalty`/`LearningRate`:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind { Linear, Rbf, Poly, Sigmoid }
impl KernelKind {
    fn name(self) -> &'static str {
        match self {
            KernelKind::Linear => "linear",
            KernelKind::Rbf => "rbf",
            /* ... */
        }
    }
}
```

**`TryFrom<&str>` (D-05 single source) — NEW, target shape** (RESEARCH lines 520–533, accept legacy aliases per §State-of-the-Art):
```rust
impl TryFrom<&str> for Loss {
    type Error = BuildError;
    fn try_from(s: &str) -> Result<Self, BuildError> {
        match s {
            "hinge" => Ok(Loss::Hinge),
            "log" | "log_loss" => Ok(Loss::Log),                 // 1.1 alias
            "squared_hinge" => Ok(Loss::SquaredHinge),
            "squared_error" | "squared_loss" => Ok(Loss::SquaredLoss), // 1.0 alias
            "epsilon_insensitive" => Ok(Loss::EpsilonInsensitive),
            "squared_epsilon_insensitive" => Ok(Loss::SquaredEpsilonInsensitive),
            other => Err(BuildError::UnknownLoss { value: other.to_string() }),
        }
    }
}
```

**`SgdConfig` field layout** (RESEARCH §Pattern-3 lines 279–294) — the shared lowering D-06:
```rust
pub struct SgdConfig {
    pub loss: Loss, pub penalty: Penalty, pub alpha: f64, pub l1_ratio: f64,
    pub fit_intercept: bool, pub max_iter: usize, pub tol: f64,
    pub learning_rate: LearningRate, pub eta0: f64, pub power_t: f64,
    pub epsilon: f64, pub batch_size: usize, pub shuffle: bool, pub seed: u64,
}
```

**`KernelKind` is `Copy`** and stored as a typed field in the estimator
(kernel_ridge.rs line 103 `kernel_kind: KernelKind`) — `SgdConfig` is the
Phase-10 equivalent stored on each estimator (D-06).

**ANTI-PATTERN to reject** (CONTEXT D-04): do NOT use `affinity: String` like
`spectral_clustering.rs`. Categorical knobs are typed enums.

---

### `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` (estimator, classifier)

**Analog:** `crates/mlrs-algos/src/linear/logistic.rs`

The closest analog by role (classifier) AND surface (`Fit` + `PredictLabels` +
`PredictProba`). Copy: the struct shape, the `classes_` distinct-sorted-label
remap (Pitfall 4 ±1 encoding), the device-residency, the host_to_f64/f64_to_host
helpers, the predict argmax-with-classes_-roundtrip, and the per-iteration
`prim_err` capture + `ScratchGuard` RAII.

**Imports** (logistic.rs lines 48–61):
```rust
use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;
use crate::error::AlgoError;
use crate::traits::{Fit, PredictLabels, PredictProba};
```
For MBSGD add `use mlrs_backend::prims::sgd::sgd_solve;` (the new prim) replacing
the `lbfgs::{lbfgs_minimize, ...}` import.

**Struct shape with device-resident fitted state** (logistic.rs lines 99–125) — the model:
```rust
pub struct LogisticRegression<F> {
    c: F, fit_intercept: bool, max_iter: usize, tol: F,
    n_classes: usize,
    classes_: Vec<i64>,          // DISTINCT sorted labels (CR-02 / Pitfall 4)
    n_features: usize,
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}
```
MBSGD analog: replace the loose hyperparameter fields with `config: SgdConfig`
(D-06); KEEP `classes_`, `n_features`, `coef_`, `intercept_`.

**classes_ remap (Pitfall 4 — ±1 label encoding) — copy verbatim** (logistic.rs lines 239–282): round+validate integer labels, `classes_.sort_unstable(); classes_.dedup();`, reject `< 2` classes, build the dense `y_remapped`. For hinge/log the dense index maps to ±1 for the margin loss.

**predict_labels argmax + classes_ roundtrip** (logistic.rs lines 586–601):
```rust
let mut best = 0usize;
let mut best_v = host_to_f64(proba_host[r * k]);
for c in 1..k {
    let v = host_to_f64(proba_host[r * k + c]);
    if v > best_v { best_v = v; best = c; }    // lowest-index tie via strict `>`
}
labels[r] = self.classes_[best] as i32;        // map dense col → original id
```

**predict_proba via log loss → sigmoid** (logistic.rs lines 481–564) — the stable softmax host pass is the proba terminal; for MBSGD(log) it is the per-sample sigmoid `1/(1+exp(-margin))` (RESEARCH §SGD-Math `Log` row, SGDSVM-01 proba gate).

**WR-01 prim_err capture + ScratchGuard RAII** (logistic.rs lines 314–364, 606–660) — copy if the SGD epoch loop launches a fallible per-batch closure; a device launch failure must surface as typed `AlgoError::Prim`, never panic across the (future PyO3) boundary.

**host_to_f64 / f64_to_host / f_epsilon helpers** (logistic.rs lines 662–690) — copy the trio verbatim (every estimator has its own private copy; this is the established convention, see kernel_ridge.rs lines 444–459, elastic_net.rs lines 272–287).

**Builder adaptation (D-01/D-08):** the `new()`/`with_opts()` (logistic.rs lines 135–153) becomes `MBSGDClassifier::builder()...build()`. The validation that logistic.rs does at `fit` (`if !(c64 > 0.0) { InvalidC }`, lines 201–207) MOVES to `build()` as the data-independent check; the geometry/label checks (lines 208–268) STAY in `fit`.

---

### `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` (estimator, regressor)

**Analog:** `crates/mlrs-algos/src/linear/elastic_net.rs`

Closest by role (regressor: `Fit` + `Predict`) AND by the `alpha`/`l1_ratio`/penalty
semantics (RESEARCH: "ElasticNet's alpha/l1_ratio handling is the nearest analog
for the SGD penalty math").

**Struct shape** (elastic_net.rs lines 57–74) — alpha/l1_ratio/fit_intercept + device-resident coef_/intercept_:
```rust
pub struct ElasticNet<F> {
    alpha: F, l1_ratio: F, fit_intercept: bool, max_iter: usize, tol: f64,
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}
```
MBSGDRegressor: replace scalars with `config: SgdConfig` (adds `epsilon` for the
epsilon-insensitive loss, D-06); KEEP `coef_`/`intercept_`.

**`Predict` impl delegating to shared `predict_linear`** (elastic_net.rs lines 180–268) — the `X·coef_ + intercept_` GEMM-then-broadcast path is REUSABLE; MBSGDRegressor's `predict` can call the same `predict_linear` helper (it is `pub(crate)`, elastic_net.rs line 207):
```rust
fn predict(&self, pool, x, shape) -> Result<DeviceArray<...>, AlgoError> {
    predict_linear(self.coef_.as_ref(), self.intercept_.as_ref(),
                   "mbsgd_regressor", pool, x, shape)
}
```

**`Fit` impl** (elastic_net.rs lines 140–178): unwrap `y` with `NotFitted`, then
call the solver. For MBSGD swap `cd_fit(...)` → `sgd_solve(...)` (the new prim),
passing the `SgdConfig`-lowered params.

**coef/intercept host accessors** (elastic_net.rs lines 117–137) — copy verbatim (the `to_host(pool)` + `NotFitted` pattern).

---

### `crates/mlrs-algos/src/linear/linear_svc.rs` + `linear_svr.rs` (estimators, CD-reuse)

**Analog:** `crates/mlrs-algos/src/linear/elastic_net.rs` (struct + predict) + `coordinate_descent.rs` (`cd_fit`)

**D-07 / RESEARCH §LinearSVC:** these reuse the v1 coordinate-descent solver
(`cd_fit`), NOT the SGD prim (sklearn LinearSVC/SVR are liblinear CD — converged).

**`cd_fit` reuse** (coordinate_descent.rs lines 72–214): LinearSVC/SVR call
`cd_fit` for the penalized-CD solve. BUT note OPEN-QUESTION-Q1 (RESEARCH lines
562–565): `cd_fit` is the Lasso/ElasticNet soft-threshold CD; the SVM
squared-hinge / squared-epsilon-insensitive loss is a DIFFERENT per-coordinate
update. The planner must spike whether `cd_fit` expresses the SVM objective or a
thin SVM-CD update is needed.

**CRITICAL divergence — `intercept_scaling` NOT center-then-solve** (Pitfall 5):
`cd_fit` does center-then-solve intercept recovery (coordinate_descent.rs lines
122–205, `intercept_ = ȳ − x̄·coef_`). LinearSVC/SVR must NOT use this — they
append a synthetic feature column of value `intercept_scaling`, solve, and recover
`intercept_ = intercept_scaling · w_last` (RESEARCH §LinearSVC lines 404,
Pitfall 5 lines 469–472). So copy the elastic_net STRUCT + predict, but the
fit-time intercept handling is the synthetic-feature mechanism, not `cd_fit`'s
centering path (call `cd_fit` with `fit_intercept=false` on the augmented design).

**`C ↔ alpha` mapping** (RESEARCH line 408): LinearSVC uses `C`; map like
LogisticRegression's `l2_reg = 1/(C·n)` precedent (logistic.rs lines 284–286).

**LinearSVC = classifier** (`Fit` + `PredictLabels`, copy logistic.rs classes_
remap); **LinearSVR = regressor** (`Fit` + `Predict`, copy elastic_net predict).

---

### `crates/mlrs-algos/src/error.rs` (EDIT — add BuildError)

**Analog:** existing `AlgoError` variants (`InvalidAlpha` lines 54–60, `InvalidL1Ratio` lines 137–146, `InvalidC` lines 152–158)

**Existing variant shape to copy for `BuildError`** (error.rs lines 54–60):
```rust
#[error("estimator '{estimator}': alpha = {alpha} is invalid (must be >= 0)")]
InvalidAlpha { estimator: &'static str, alpha: f64 },
```

**D-09 / RESEARCH recommendation** (lines 416–423): add a SEPARATE `BuildError`
enum (using `thiserror`, the same `#[derive(Debug, Error)]` + `#[error("...")]`
attribute shape as `AlgoError`) carrying `InvalidAlpha`/`InvalidL1Ratio`/
`InvalidEta0`/`InvalidEpsilon`/`UnknownLoss`/`UnknownPenalty`/
`UnknownLearningRate`/`InvalidLossForEstimator` variants — fold the enum
`TryFrom` failures into `BuildError` so a SINGLE `build_err_to_py` mapper covers
both (mirrors the single-site `algo_err_to_py` rationale). `thiserror` in libs
(error.rs lines 11–14 doc; CLAUDE.md / MEMORY error-handling-convention).

The fit-time data-DEPENDENT checks (geometry/labels) keep using `AlgoError`
(unchanged), so `BuildError` and `AlgoError` are sibling types.

---

### `crates/mlrs-py/src/estimators/linear.rs` (EDIT — four #[pyclass] wrappers)

**Analog:** `PyLogisticRegression` (same file, lines 632–806)

The closest analog: a classifier wrapper exposing `predict_labels` (i32) +
`predict_proba` + dtype-suffixed `coef_`/`intercept_` accessors. The
`any_estimator!` macro needs NO change (RESEARCH line 412 / dispatch.rs lines
90–115 — the macro emits ONLY the enum; `#[pymethods]` are hand-written).

**Macro invocation — Unfit stores sklearn STRINGS + scalars** (linear.rs lines 632–636; kernel.rs line 103 stores `kernel: String`):
```rust
crate::any_estimator! {
    any:   AnyMBSGDClassifier,
    algo:  mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier,
    unfit: { loss: String, penalty: String, alpha: f64, l1_ratio: f64,
             learning_rate: String, eta0: f64, max_iter: usize, tol: f64, /* ... */ },
}
```

**fit body — the builder-chain adaptation** (logistic.rs Py wrapper lines 673–710 is the template; the NEW part is the TryFrom + builder + build_err_to_py):
```rust
let loss = Loss::try_from(loss_str.as_str()).map_err(build_err_to_py)?;   // D-05 → ValueError
let penalty = Penalty::try_from(penalty_str.as_str()).map_err(build_err_to_py)?;
// inside py.detach / match dt arms:
let mut est = MBSGDClassifier::<f32>::builder()
    .loss(loss).penalty(penalty).alpha(alpha as f32)/* ... */
    .build().map_err(build_err_to_py)?;                                   // D-09 → ValueError
est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?; // fit-time AlgoError
```
This REPLACES the analog's `LogisticRegression::<F>::with_opts(...)` (logistic Py
wrapper lines 694/702) call site with the `builder()...build()?` chain.

**GIL release + f64 guard + dtype dispatch — copy verbatim** (logistic Py wrapper lines 688–706): `py.detach(|| { let mut pool = crate::lock_pool(); match dt { F32 => {...} F64 => { crate::capability::guard_f64()?; ... } } })`. NOTE: the existing logistic wrapper still uses `crate::global_pool().lock().expect("pool mutex")` (line 689) — but the SANCTIONED path is `crate::lock_pool()` (poison-recovering, dispatch.rs lines 28–33; kernel.rs line 183 uses it). New Phase-10 wrappers SHOULD use `crate::lock_pool()`.

**predict_labels + predict_proba_f32/_f64 + coef_/intercept_ dtype-suffixed accessors + is_fitted + dtype** — copy the full method set (logistic Py wrapper lines 712–805) verbatim, renaming the estimator string.

---

### `crates/mlrs-py/src/errors.rs` (EDIT — add build_err_to_py)

**Analog:** `algo_err_to_py` (errors.rs lines 55–57)

```rust
pub fn algo_err_to_py(err: AlgoError) -> PyErr {
    PyValueError::new_err(err.to_string())
}
```
**Add (D-09, RESEARCH line 421):**
```rust
pub fn build_err_to_py(err: BuildError) -> PyErr {     // import BuildError from mlrs_algos::error
    PyValueError::new_err(err.to_string())
}
```
Same `PyValueError` class, same `.to_string()` body. Add the `BuildError` row to
the module-doc error→exception table (errors.rs lines 11–19).

---

### `scripts/gen_oracle.py` (EDIT — add four generators)

**Analog:** `gen_logistic` (lines 782–886) + `gen_elastic_net` (lines 667–700)

Add `gen_mbsgd_classifier` / `gen_mbsgd_regressor` / `gen_linear_svc` /
`gen_linear_svr` following the `gen_logistic` shape.

**Generator skeleton to copy** (gen_logistic lines 810–886):
```python
rng = np.random.default_rng(seed)                 # SEED = 42 (line 44)
# ... build well-separated blobs X / Xq / y ...
clf = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4,
                    learning_rate="constant", eta0=..., shuffle=False,  # PINNED (Pitfall 2/7)
                    tol=0, max_iter=K, fit_intercept=True).fit(x, y)    # tol=0 + fixed max_iter
coef = clf.coef_; intercept = clf.intercept_
predict = clf.predict(xq); predict_proba = ...                          # log loss → proba
def c(arr): return np.ascontiguousarray(np.asarray(arr)).astype(dtype)
dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
out_path = os.path.join(_FIXTURE_DIR, f"mbsgd_classifier_{dtype_tag}_seed{seed}.npz")
np.savez(out_path, X=c(x), Xq=c(xq), y=c(y), coef=c(coef),
         intercept=c(intercept), predict=c(predict), predict_proba=c(predict_proba))
```

**Pinned-determinism (Pitfall 2/7):** the oracle fixtures pin `shuffle=False`,
`tol=0`, fixed `max_iter`, fixed `eta0`/schedule (so neither side early-stops).
The Rust oracle test constructs the estimator with EXPLICIT pinned setters (NOT
the bare `builder().build()` default — a SEPARATE D-03 litmus test checks the
default equals sklearn's default).

**MEMORY caveat (oracle-fixture-regen-needs-venv):** `gen_oracle.py` needs numpy
via a `/tmp` venv (PEP 668); fixtures are committed `.npz` blobs. Regen in
ISOLATION (per 08-01/09-01 lesson — do not churn other phases' blobs).

---

### Module-index edits (Wave-0 scaffold)

**Analog:** the existing `pub mod` lines in each index.

- `crates/mlrs-kernels/src/lib.rs` (lines 8–22): add `pub mod sgd;` (+ a `pub use sgd::{...}` if the kernels re-export, lines 24–36). The `coordinate` precedent: each kernel file adds its OWN `pub use` (coordinate.rs lines 38–40), file-disjoint.
- `crates/mlrs-backend/src/prims/mod.rs` (lines 12–43): add `pub mod sgd;`.
- `crates/mlrs-algos/src/linear/mod.rs`: add `pub mod sgd_config; pub mod mbsgd_classifier; pub mod mbsgd_regressor; pub mod linear_svc; pub mod linear_svr;` (the doc-comment notes estimator plans uncomment their own `pub mod` line; they do NOT edit `lib.rs`).

---

## Shared Patterns

### Validate-before-launch (ASVS V5) — but SPLIT for Phase 10 (D-08)
**Source:** `coordinate_descent.rs` `cd_fit` lines 94–120; `logistic.rs` fit lines 198–268
**Apply to:** all four new estimators
The analog checks (`if !(alpha >= 0.0) { InvalidAlpha }`, `if !(0.0..=1.0).contains(&l1_ratio) { InvalidL1Ratio }`, geometry `x.len() != n*d`) are the SAME predicates Phase 10 uses — but DATA-INDEPENDENT ones (`alpha>=0`, `l1_ratio∈[0,1]`, `eta0>0`, `epsilon>=0`, valid enum/loss combo) move to `build() -> Result<_, BuildError>`; DATA-DEPENDENT ones (geometry, label integrality) stay in `fit() -> AlgoError`.

### Device-resident fitted state (D-03) + host materialization only at accessor
**Source:** `elastic_net.rs` lines 69–74 (struct), 117–137 (accessors); `logistic.rs` lines 122–124, 157–177
**Apply to:** all four new estimators
`coef_`/`intercept_` are `Option<DeviceArray<ActiveRuntime, F>>`; host copy only via `to_host(pool)` in a Rust accessor or the oracle boundary.

### host_to_f64 / f64_to_host helper trio (f64 host accumulation)
**Source:** `logistic.rs` lines 662–690; identical copies in `elastic_net.rs` 272–287, `kernel_ridge.rs` 444–459, `coordinate_descent.rs` 232–258 (`narrow_to_f` too)
**Apply to:** all four estimators + `prims/sgd.rs` (schedule/loss scalars in f64)
Each module carries its OWN private copy (the established convention — do not factor into a shared util). `narrow_to_f` (coordinate_descent.rs 252–258) is needed if Phase 10 centers in the working dtype like cd_fit (WR-04).

### Re-fit buffer reuse (WR-07)
**Source:** `kernel_ridge.rs` lines 342–347
**Apply to:** all four estimators (re-`fit` path)
```rust
if let Some(old) = self.coef_.take() { old.release_into(pool); }
if let Some(old) = self.intercept_.take() { old.release_into(pool); }
```
On re-fit, release old device buffers to the pool free-list BEFORE reassigning.

### Trait surface (unchanged — the post-build() estimator implements existing traits)
**Source:** `traits.rs` — `Fit` (lines 53–68), `Predict` (109–122), `PredictLabels` (168–181), `PredictProba` (256–270), `PartialFit` (86–104, y-slot reserved for MBSGD per lines 79–83)
**Apply to:** MBSGDClassifier (Fit+PredictLabels+PredictProba [+PartialFit]); MBSGDRegressor (Fit+Predict [+PartialFit]); LinearSVC (Fit+PredictLabels); LinearSVR (Fit+Predict)
`fit` returns `&mut self` (sklearn chaining); the builder only changes CONSTRUCTION, not the trait surface.

### PyO3 boundary: GIL release + f64 guard + dtype dispatch
**Source:** `dispatch.rs` lines 21–60 (the contract); `kernel.rs` lines 182–210 (the canonical body with `crate::lock_pool()`)
**Apply to:** all four new Py wrappers
`py.detach(|| { let mut pool = crate::lock_pool(); match dt { F32 => {...} F64 => { crate::capability::guard_f64()?; ...} } })`. Use `crate::lock_pool()` (poison-recovering), NOT `global_pool().lock().expect(...)`.

---

## No Analog Found

None. Every Phase-10 file has a shipped in-tree analog. The two genuinely-new
elements are not "no analog" but "exact analog, new math":
- `sgd.rs` kernels: structure = `coordinate.rs` GATHER idiom; the SGD margin/grad
  MATH is new (RESEARCH §SGD-Math), but the kernel SHAPE is copied.
- The `optimal`-schedule `t0` and per-loss `dloss` are new HOST f64 helpers
  (RESEARCH lines 490–514) with no codebase analog — but they are small pure
  functions, fully specified in RESEARCH, validated by the Wave-0 live-sklearn
  oracle (A1 flagged ASSUMED until the oracle confirms).

---

## Metadata

**Analog search scope:** `crates/mlrs-kernels/src/`, `crates/mlrs-backend/src/prims/`, `crates/mlrs-algos/src/linear/`, `crates/mlrs-algos/src/{error.rs,traits.rs}`, `crates/mlrs-algos/src/kernel_ridge/`, `crates/mlrs-py/src/{dispatch.rs,errors.rs,estimators/}`, `scripts/gen_oracle.py`
**Files scanned (read in full or targeted):** 11 source files + 1 oracle generator + module indices
**Pattern extraction date:** 2026-06-21
**cpu-MLIR caveat:** the `sgd.rs` SharedMemory/atomic/INFINITY/bool/shift constraints are copied from `coordinate.rs` (verified) but the cpu-LAUNCH gate (`cargo test --features cpu`) is the authoritative test, not the recalled list (RESEARCH Pitfall 1 / A5 / MEMORY cubecl-cpu-no-shared-memory).
