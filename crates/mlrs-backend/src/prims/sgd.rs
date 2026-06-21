//! `sgd` — minibatch-SGD solver primitive (PRIM-10).
//!
//! Host-side orchestration of the two-pass `sgd_margin` / `sgd_weight_update`
//! kernels into the converged (pinned-epoch) SGD solve that
//! `MBSGDClassifier`/`MBSGDRegressor` consume. Mirrors the
//! [`cd_solve`](crate::prims::coordinate_descent::cd_solve) shape:
//! validate-before-launch → host epoch loop → per-batch launch → scalar/loss
//! readback → `NotConverged` at the `max_iter` cap.
//!
//! ## Flat-scalar contract (NOT the algos `SgdConfig`)
//! `mlrs-backend` does NOT depend on `mlrs-algos` (the dependency runs the other
//! way), so this prim CANNOT take the `mlrs_algos::linear::sgd_config::SgdConfig`
//! type — that would be a circular dependency. Following the `cd_solve`
//! precedent, the estimator LOWERS its `SgdConfig` into the flat scalar / enum
//! arguments here at the call site (the [`SgdLoss`] / [`SgdSchedule`] prim-local
//! enums encode the loss / schedule families the kernels need). The host
//! `dloss` table + `optimal`/`invscaling` schedule arithmetic live in this layer
//! (f64), feeding `g[]` and `eta` to the kernels.
//!
//! ## Status (Wave-0 / plan 10-01 — STUB)
//! The host signature + a REAL geometry guard (`x.len() == n*d`, `y.len() == n`,
//! non-empty) compile today; the compute path is `todo!()`. The Wave-1 plan fills
//! the epoch loop + the `dloss`/schedule helpers + the lazy-L2 / cumulative-L1
//! penalty bookkeeping. The two kernels it drives are already SharedMemory-free
//! by construction (cubecl-cpu MLIR-safe).
//!
//! Tests live in `crates/mlrs-backend/tests/sgd_test.rs` (AGENTS.md §2 — never an
//! in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Default SGD epoch cap (sklearn `SGDClassifier`/`SGDRegressor` `max_iter`).
pub const SGD_DEFAULT_MAX_ITER: usize = 1000;

/// Default SGD stopping tolerance (sklearn `SGDClassifier`/`SGDRegressor`
/// `tol`). The pinned oracle OVERRIDES this with `tol = 0` + a fixed `max_iter`
/// so neither side early-stops (Pitfall 2/7).
pub const SGD_DEFAULT_TOL: f64 = 1e-3;

/// The loss family the host `dloss` table selects on (the estimator lowers its
/// typed `mlrs_algos` `Loss` into this prim-local enum at the call site — the
/// prim layer cannot depend on algos). Wave-1 wires the per-sample gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgdLoss {
    /// Hinge `max(0, 1 - y·p)`.
    Hinge,
    /// Log `log(1 + exp(-y·p))`.
    Log,
    /// Squared hinge `max(0, 1 - y·p)²`.
    SquaredHinge,
    /// Squared error `½(p - y)²`.
    SquaredError,
    /// Epsilon-insensitive `max(0, |y - p| - ε)`.
    EpsilonInsensitive,
    /// Squared epsilon-insensitive `max(0, |y - p| - ε)²`.
    SquaredEpsilonInsensitive,
}

/// The learning-rate schedule the host step arithmetic selects on (lowered from
/// the algos `LearningRate` at the call site).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgdSchedule {
    /// Bottou `optimal` `η(t) = 1/(α·(t₀ + t − 1))`.
    Optimal,
    /// Inverse-scaling `η(t) = eta0 / t^power_t`.
    InvScaling,
    /// Constant `η = eta0`.
    Constant,
    /// Adaptive `η = eta0`, halved on a stalled objective.
    Adaptive,
}

/// Flat-scalar penalty / schedule arguments the estimator lowers its `SgdConfig`
/// into (the prim cannot take the algos `SgdConfig` — circular dependency). All
/// scalars are host f64; the host `dloss`/schedule math runs in f64 and feeds the
/// kernels.
#[derive(Debug, Clone, Copy)]
pub struct SgdParams {
    /// The loss family (prim-local enum, lowered from algos `Loss`).
    pub loss: SgdLoss,
    /// The learning-rate schedule (prim-local enum, lowered from algos
    /// `LearningRate`).
    pub schedule: SgdSchedule,
    /// L2 penalty strength.
    pub alpha: f64,
    /// ElasticNet L1/L2 mixing `∈ [0, 1]`.
    pub l1_ratio: f64,
    /// Whether the host applies the L1 shrink (`l1_ratio > 0` / penalty includes
    /// L1).
    pub apply_l1: bool,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
    /// Initial learning rate `eta0`.
    pub eta0: f64,
    /// Inverse-scaling exponent `power_t`.
    pub power_t: f64,
    /// Epsilon-insensitive margin.
    pub epsilon: f64,
    /// Minibatch size.
    pub batch_size: usize,
    /// Epoch cap.
    pub max_iter: usize,
    /// Stopping tolerance (`0` ⇒ run all `max_iter` epochs).
    pub tol: f64,
}

/// Solve the minibatch-SGD problem on the design `x` (`n × d`, row-major) and
/// target `y` (length `n`), returning the device-resident pair
/// `(coef, intercept)` (`coef` length `d`, `intercept` length 1).
///
/// - Geometry (`n*d == x.len()`, `y.len() == n`, non-empty) is validated BEFORE
///   any unsafe launch (ASVS V5 / T-10-01-02); a wrong shape returns
///   [`PrimError::ShapeMismatch`] so a malformed shape can never reach a device
///   launch even at scaffold stage.
/// - `params` carries the lowered (flat-scalar) penalty / schedule the host
///   gradient + step math reads.
///
/// ## Status (Wave-0 — STUB)
/// The geometry guard below is REAL; the compute body is `todo!()` (the Wave-1
/// plan fills the epoch loop driving `sgd_margin` / `sgd_weight_update`).
pub fn sgd_solve<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    params: &SgdParams,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    PrimError,
>
where
    F: Float + CubeElement + Pod,
{
    let (n, d) = shape;

    // --- Geometry guard (ASVS V5 / T-10-01-02): validate BEFORE any launch so a
    //     malformed shape is a recoverable typed error, not an out-of-bounds
    //     device read. Mirrors cd_solve / laplacian. ---
    validate_geometry(x.len(), y.len(), n, d)?;

    // The flat-scalar params are consumed by the Wave-1 epoch loop; reference one
    // field so the scaffold compiles without an unused-binding warning while the
    // body is a stub.
    let _ = (params.loss, params.schedule, params.max_iter);
    let _ = pool;

    // Wave-1 fills: acquire device w (len d) + bias scalar, then per-epoch
    // per-batch: sgd_margin::launch → host g[i]=dloss(p_i,y_i)+schedule eta →
    // sgd_weight_update::launch → host bias -= eta·Σg (with intercept_decay) →
    // NotConverged at the cap. The kernels are already SharedMemory-free.
    todo!("PRIM-10 sgd_solve compute body — filled in Wave-1 (plan 10-02)")
}

/// Validate the SGD solve geometry (ASVS V5 / T-10-01-02): `x` must be a
/// well-formed `n × d` matrix (`n*d == x.len()`), `y` must be length `n`, and
/// neither dimension may be zero. A malformed shape is rejected at the boundary
/// (no well-defined solve) BEFORE any device launch.
fn validate_geometry(x_len: usize, y_len: usize, n: usize, d: usize) -> Result<(), PrimError> {
    if n == 0 || d == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if n.checked_mul(d).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if y_len != n {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n,
            cols: 1,
            len: y_len,
        });
    }
    Ok(())
}
