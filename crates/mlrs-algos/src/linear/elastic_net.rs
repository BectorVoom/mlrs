//! `ElasticNet` (LINEAR-04) — L1+L2-penalized least squares via the shared
//! coordinate-descent solver (D-03), matching
//! `sklearn.linear_model.ElasticNet`.
//!
//! ## Solver (deliberately coordinate descent — the iterative-solver family)
//! ElasticNet minimizes
//! `½‖y − Xβ‖² + α·l1_ratio·n·‖β‖₁ + ½·α·(1−l1_ratio)·n·‖β‖₂²`
//! by cyclic coordinate descent on the CENTERED design via the validated 05-05
//! [`cd_solve`](mlrs_backend::prims::coordinate_descent::cd_solve) primitive,
//! driven by the shared [`cd_fit`] host helper. It is NOT a Cholesky / SVD
//! normal-equations solve (those are Ridge / LinearRegression) nor the L-BFGS
//! LogReg optimizer (05-10) — the three families use deliberately different
//! solvers and must not be unified (see `linear/mod.rs`).
//!
//! ## Lasso is the `l1_ratio == 1` case (D-03)
//! [`Lasso`](crate::linear::lasso::Lasso) is a thin wrapper that delegates to the
//! same [`cd_fit`] with `l1_ratio = 1.0` (→ `l2_reg = 0`, pure L1). Both
//! estimators share one coordinate-descent implementation.
//!
//! ## Penalty mapping + center-then-solve intercept
//! `cd_fit` maps the user-facing `(alpha, l1_ratio)` to sklearn's un-normalized
//! `(l1_reg = α·l1_ratio·n, l2_reg = α·(1−l1_ratio)·n)` (Pitfall 1), centers
//! `(X, y)` when `fit_intercept` (D-13), and recovers the unpenalized
//! `intercept_ = ȳ − x̄·coef_` — reproducing sklearn's `coef_`/`intercept_` within
//! 1e-5 INCLUDING the exact sparsity (zero) pattern.
//!
//! ## Device residency (D-03)
//! Fitted `coef_` (length `n_features`) and `intercept_` (length 1) are stored as
//! device-resident [`DeviceArray`]s; `predict` runs the `X_test · coef_` GEMM
//! on-device and broadcasts the intercept (the `ridge.rs` predict path),
//! materializing to the host only at a Rust accessor / oracle boundary.
//!
//! Tests live in `crates/mlrs-algos/tests/elastic_net_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::linear_predict::{
    linear_predict, linear_predict_from_host, HostMirror, HostPrediction,
};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::coordinate_descent::{cd_fit, CD_DEFAULT_MAX_ITER, CD_DEFAULT_TOL};
// LINEAR-PERSIST: the safetensors container. ElasticNet is the shared
// dense-linear core plus the coordinate-descent knobs it holds identically with
// `Lasso`, plus the ONE field that distinguishes the two — `l1_ratio`.
use crate::linear::linear_persist::{
    read_linear_core, AlignedBytes, CdScalars, LinearCoreRef, LinearFile, LinearWriter, LoadModel,
    PersistError, SaveModel,
};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// L1+L2-penalized least squares (LINEAR-04) fitted by the shared
/// coordinate-descent solver.
///
/// Construct with the zero-arg [`ElasticNet::new`] (sklearn defaults:
/// `alpha = 1.0`, `l1_ratio = 0.5`, `fit_intercept = true`, `max_iter = 1000`,
/// `tol = 1e-4`) or [`ElasticNet::builder`] (which subsumes the former
/// `new`/`with_opts` constructors — every hyperparameter is a builder setter),
/// then the consuming [`Fit::fit`] (returns the `Fitted`-tagged sibling) and
/// [`Predict::predict`]. Fitted `coef_`/`intercept_` are device-resident (D-03);
/// the host accessors [`coef`](ElasticNet::coef) /
/// [`intercept`](ElasticNet::intercept) materialize them on demand and exist ONLY
/// on `ElasticNet<F, Fitted>` (the compile-time typestate replaces the old
/// runtime `NotFitted` guard, D-03).
pub struct ElasticNet<F, S = Unfit> {
    /// Overall penalty strength (`alpha ≥ 0`; `alpha = 0` degenerates to OLS).
    /// Validated at `build()` → [`BuildError::InvalidAlpha`] (T-05-09-01).
    alpha: F,
    /// L1/L2 mixing parameter (`0 ≤ l1_ratio ≤ 1`; `1` ⇒ Lasso, `0` ⇒ Ridge-like
    /// pure L2). Validated at `build()` → [`BuildError::InvalidL1Ratio`]
    /// (T-05-09-01).
    l1_ratio: F,
    /// Whether to center `X`/`y` and recover a bias term (D-13).
    fit_intercept: bool,
    /// Coordinate-descent iteration cap (sklearn default 1000).
    max_iter: usize,
    /// Coordinate-descent stopping tolerance (sklearn default 1e-4).
    tol: f64,
    /// Fitted coefficients (length `n_features`), device-resident, `None` until
    /// `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (length 1), device-resident, `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Memoized host copy of `(coef_, intercept_)` for the host-ingress
    /// `predict` path (IN-05 `OnceLock` mirror idiom). Empty until the first
    /// `predict_from_host` on the cpu backend, and never filled at all on the
    /// device backends — see
    /// [`HostMirror`](mlrs_backend::prims::linear_predict::HostMirror) for why a
    /// 64-byte read-back is worth caching. Fresh on every `fit`.
    predict_mirror: HostMirror<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> ElasticNet<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an `ElasticNet` with sklearn's defaults (`alpha = 1.0`,
    /// `l1_ratio = 0.5`, `fit_intercept = true`, `max_iter = 1000`, `tol = 1e-4`)
    /// directly in the `Unfit` state. This is the SINGLE source of truth for the
    /// default hyperparameters (D-08): the builder `Default` re-derives from here
    /// via [`ElasticNet::into_builder`]. Defaults are trusted valid, so this
    /// bypasses [`ElasticNetBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            alpha: F::from_int(1),
            l1_ratio: f64_to_host::<F>(0.5),
            fit_intercept: true,
            max_iter: CD_DEFAULT_MAX_ITER,
            tol: CD_DEFAULT_TOL,
            coef_: None,
            intercept_: None,
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        }
    }

    /// Start building an `ElasticNet` from sklearn's defaults (D-08 single source).
    pub fn builder() -> ElasticNetBuilder {
        ElasticNetBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`ElasticNetBuilder::default`] to re-derive the
    /// defaults from [`ElasticNet::new`] (D-08).
    pub fn into_builder(self) -> ElasticNetBuilder {
        ElasticNetBuilder {
            alpha: host_to_f64(self.alpha),
            l1_ratio: host_to_f64(self.l1_ratio),
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `coef_`/`intercept_` fields are excluded — both are `None` in any `Unfit`
    /// value). Used by the defaults-equality test (BLDR-01):
    /// `ElasticNet::new().hyperparams_eq(&ElasticNet::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        host_to_f64(self.alpha) == host_to_f64(other.alpha)
            && host_to_f64(self.l1_ratio) == host_to_f64(other.l1_ratio)
            && self.fit_intercept == other.fit_intercept
            && self.max_iter == other.max_iter
            && self.tol == other.tol
    }
}

impl<F> Default for ElasticNet<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`ElasticNet`] (D-01). It subsumes BOTH the former `new(alpha,
/// l1_ratio, fit_intercept)` AND `with_opts(alpha, l1_ratio, fit_intercept,
/// max_iter, tol)` constructors — every hyperparameter is a setter. Setters are
/// `f64`/`usize` per the A5 convention; `build::<F>()` narrows `alpha`/`l1_ratio`
/// to the target float `F`. `Default` re-derives the sklearn defaults from
/// [`ElasticNet::new`] (D-08 single source) rather than holding literals
/// (Pitfall 1).
#[derive(Debug, Clone, Copy)]
pub struct ElasticNetBuilder {
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
}

impl Default for ElasticNetBuilder {
    /// Re-derive the sklearn defaults from [`ElasticNet::new`] (D-08 single
    /// source).
    fn default() -> Self {
        ElasticNet::<f64, Unfit>::new().into_builder()
    }
}

impl ElasticNetBuilder {
    /// Set the overall penalty strength `alpha` (A5: `f64` setter).
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
        self
    }

    /// Set the L1/L2 mixing parameter `l1_ratio` (A5: `f64` setter).
    pub fn l1_ratio(mut self, v: f64) -> Self {
        self.l1_ratio = v;
        self
    }

    /// Set whether to center `X`/`y` and recover a bias term.
    pub fn fit_intercept(mut self, v: bool) -> Self {
        self.fit_intercept = v;
        self
    }

    /// Set the coordinate-descent iteration cap (sklearn `max_iter`).
    pub fn max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the coordinate-descent stopping tolerance (sklearn `tol`).
    pub fn tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT hyperparameters
    /// BEFORE any data is seen (relocated from the old `cd_fit` fit-body checks,
    /// Pitfall 7; the data-DEPENDENT geometry check stays in [`Fit::fit`]):
    ///
    /// - `alpha >= 0` ([`BuildError::InvalidAlpha`]).
    /// - `0 <= l1_ratio <= 1` ([`BuildError::InvalidL1Ratio`]).
    ///
    /// The stored `f64` `alpha`/`l1_ratio` are narrowed to the target float `F`
    /// (A5).
    pub fn build<F>(self) -> Result<ElasticNet<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if !(self.alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha {
                estimator: "elastic_net",
                alpha: self.alpha,
            });
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(BuildError::InvalidL1Ratio {
                estimator: "elastic_net",
                l1_ratio: self.l1_ratio,
            });
        }
        Ok(ElasticNet {
            alpha: f64_to_host::<F>(self.alpha),
            l1_ratio: f64_to_host::<F>(self.l1_ratio),
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            coef_: None,
            intercept_: None,
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
}

impl<F> ElasticNet<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `coef_` (length `n_features`). `Some` by
    /// construction on the `Fitted` state (D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on ElasticNet<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_` (scalar). `Some` by construction on
    /// the `Fitted` state (D-03).
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> F {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on ElasticNet<F, Fitted>")
            .to_host(pool)[0]
    }

    /// `predict` for a test matrix that is still on the HOST — returns the
    /// length-`n_samples` predictions plus the operand-finiteness verdict.
    ///
    /// The host-ingress twin of [`Predict::predict`]: same result, but it reads
    /// the caller's buffer in place on cpu instead of paying an `m × n` upload
    /// that costs more than the prediction. All four dense linear regressors
    /// share ONE implementation — see [`predict_linear_from_host`] for the
    /// backend routing, the measurements, and the finiteness verdict's meaning.
    pub fn predict_from_host(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<HostPrediction<F>, AlgoError> {
        predict_linear_from_host(
            self.coef_.as_ref(),
            self.intercept_.as_ref(),
            &self.predict_mirror,
            "elastic_net",
            pool,
            x,
            shape,
        )
    }
}

/// The `estimator` discriminator written into every saved file and required by
/// [`LoadModel::load`]. See [`lasso`](crate::linear::lasso)'s counterpart for
/// why it is what stops these two near-identical files cross-loading.
const PERSIST_TAG: &str = "elastic_net";

/// The `__metadata__` key for the one field that distinguishes an `ElasticNet`
/// file from a `Lasso` one. Named as a constant precisely because it is the
/// whole difference — `save` and `load` disagreeing on it would produce a model
/// that silently reverts to the `l1_ratio = 0.5` default.
const KEY_L1_RATIO: &str = "param:l1_ratio";

impl<F> SaveModel for ElasticNet<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted model to `path` as a safetensors file.
    ///
    /// The shared dense-linear core — `coef_` as `[1, n_features]`,
    /// `intercept_` as `[1]`, both at the model's own float width — plus the
    /// three [`CdScalars`] and `param:l1_ratio`, all in `__metadata__`.
    /// ElasticNet is single-target and reports no fitted diagnostics, so that is
    /// the whole file: byte for byte a `Lasso` file with one extra header key.
    ///
    /// `coef_`/`intercept_` are device-resident, so this costs one readback
    /// each — the only copies on the path.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        // Bound BEFORE the writer: `LinearWriter` borrows every payload so it
        // can stream them out without a second copy, which means the host
        // buffers must outlive it.
        let coef = self
            .coef_
            .as_ref()
            .ok_or_else(|| absent("coef_"))?
            .to_host(pool);
        let intercept = self
            .intercept_
            .as_ref()
            .ok_or_else(|| absent("intercept_"))?
            .to_host(pool);

        let mut w = LinearWriter::new(PERSIST_TAG);
        CdScalars {
            alpha: host_to_f64(self.alpha),
            max_iter: self.max_iter,
            tol: self.tol,
        }
        .write_into(&mut w);
        w.scalar_f64(KEY_L1_RATIO, host_to_f64(self.l1_ratio));
        LinearCoreRef {
            n_features: coef.len(),
            coef: &coef,
            intercept: &intercept,
            n_targets: 1,
            fit_intercept: self.fit_intercept,
        }
        .write_into(&mut w)?;
        w.write(path)
    }
}

impl<F> LoadModel for ElasticNet<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read a model back from `path`, re-uploading `coef_`/`intercept_` to
    /// `pool`.
    ///
    /// The result is `Fitted` by construction — a file only ever holds a fitted
    /// model — so the state parameter has to be named at the call site, either
    /// by turbofish or by annotating the binding:
    ///
    /// ```ignore
    /// let est: ElasticNet<f32, Fitted> = ElasticNet::load(&mut pool, path)?;
    /// ```
    ///
    /// `F` need NOT match the dtype the file was written with — see
    /// [`as_floats`](crate::persist::as_floats). `l1_ratio` is REQUIRED rather
    /// than defaulted: a file missing it is corrupt, and quietly substituting
    /// `0.5` would hand back a model with a different penalty than the one that
    /// was fitted.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<ElasticNet<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = LinearFile::parse(&raw, PERSIST_TAG)?;
        let core = read_linear_core::<F>(&file)?;
        let cd = CdScalars::read(&file)?;

        // ElasticNet is single-target — same reasoning as `Lasso::load`.
        if core.n_targets != 1 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "elastic_net is single-target, but 'coef_' declares {} target rows",
                    core.n_targets
                ),
            });
        }

        Ok(ElasticNet {
            alpha: f64_to_host::<F>(cd.alpha),
            l1_ratio: f64_to_host::<F>(file.scalar_f64(KEY_L1_RATIO)?),
            fit_intercept: core.fit_intercept,
            max_iter: cd.max_iter,
            tol: cd.tol,
            coef_: Some(DeviceArray::from_host(pool, &core.coef)),
            intercept_: Some(DeviceArray::from_host(pool, &core.intercept)),
            // The mirror is a `predict_from_host` memo, not model state — a
            // freshly loaded model refills it on first use, exactly as a
            // freshly fitted one does.
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for ElasticNet<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = ElasticNet<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<ElasticNet<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // Data-DEPENDENT geometry guard BEFORE any prim launch (the
        // data-INDEPENDENT `alpha >= 0` / `l1_ratio ∈ [0, 1]` checks were validated
        // at build() — Pitfall 7).
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "elastic_net",
            operation: "fit (requires y)",
        })?;

        // Delegate to the shared CD helper (penalty map + centering + cd_solve +
        // intercept recovery). cd_fit validates alpha/l1_ratio/geometry BEFORE any
        // launch (T-05-09-01).
        let (coef, intercept) = cd_fit::<F>(
            pool,
            x,
            y,
            n_samples,
            n_features,
            host_to_f64(self.alpha),
            host_to_f64(self.l1_ratio),
            self.fit_intercept,
            self.tol,
            self.max_iter,
            "elastic_net",
        )?;

        Ok(ElasticNet {
            alpha: self.alpha,
            l1_ratio: self.l1_ratio,
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            coef_: Some(coef),
            intercept_: Some(intercept),
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
}

impl<F> Predict<F> for ElasticNet<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        predict_linear(
            self.coef_.as_ref(),
            self.intercept_.as_ref(),
            "elastic_net",
            pool,
            x,
            shape,
        )
    }
}

/// Shared `X·coef_ + intercept_` prediction path for the coordinate-descent
/// linear models (the `ridge.rs` GEMM-then-broadcast precedent). Used by both
/// [`ElasticNet`] and [`Lasso`](crate::linear::lasso::Lasso) so the predict
/// surface is implemented once (D-03). Errors with [`AlgoError::NotFitted`] when
/// called before `fit` and [`PrimError::ShapeMismatch`] / [`PrimError::DimMismatch`]
/// on a geometry / `n_features` disagreement (ASVS V5).
pub(crate) fn predict_linear<F>(
    coef_: Option<&DeviceArray<ActiveRuntime, F>>,
    intercept_: Option<&DeviceArray<ActiveRuntime, F>>,
    estimator: &'static str,
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (n_samples, n_features) = shape;

    let coef = coef_.ok_or(AlgoError::NotFitted {
        estimator,
        operation: "predict",
    })?;
    let intercept = intercept_.ok_or(AlgoError::NotFitted {
        estimator,
        operation: "predict",
    })?;

    // --- ASVS V5: geometry + fitted-n_features consistency. ---
    if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "x",
            rows: n_samples,
            cols: n_features,
            len: x.len(),
        }));
    }
    if coef.len() != n_features {
        return Err(AlgoError::Prim(PrimError::DimMismatch {
            dim: "n_features",
            lhs: coef.len(),
            rhs: n_features,
        }));
    }

    // y_pred = X_test · coef + intercept via ONE fused device launch (the
    // LINEAR-01/02 predict perf lever, shared by ElasticNet + Lasso): the
    // `linear_predict` prim's GATHER matvec+bias kernel replaces the prior
    // gemm→`intercept.to_host()`→`raw.to_host()`→host bias-loop→`from_host`
    // round-trips (the `center`/`gram` host-sync pathology, same class of fix).
    // The result stays device-resident; the PyO3 boundary's terminal readback
    // is the only host↔device crossing.
    Ok(linear_predict::<F>(
        pool,
        x,
        coef,
        intercept,
        (n_samples, n_features),
    )?)
}

/// Host-ingress twin of [`predict_linear`] — `predict` for a test matrix that is
/// still in the CALLER'S memory, returning host predictions plus the operand
/// finiteness verdict.
///
/// Shared by all four dense linear regressors ([`ElasticNet`],
/// [`Lasso`](crate::linear::lasso::Lasso),
/// [`Ridge`](crate::linear::ridge::Ridge),
/// [`LinearRegression`](crate::linear::linear_regression::LinearRegression)), so
/// the guards below are written once and every estimator rejects the same shapes
/// with the same typed error — exactly as [`predict_linear`] does for the
/// device-ingress side.
///
/// Same result as [`predict_linear`] followed by a read-back, different ingress.
/// `predict_linear` takes an already-uploaded [`DeviceArray`]; every caller that
/// starts from host memory (i.e. the whole Arrow/PyO3 surface) had to pay a full
/// `m × n` upload to produce one. On the **cpu** backend that upload is a plain
/// memcpy of the operand into memory the kernel then reads once — measured at
/// 13.5 ms for a 64 MiB `f32` matrix, three times sklearn's ENTIRE `predict` for
/// the same shape — so it dominates a prim whose arithmetic is one pass.
/// `linear_predict_from_host` routes cpu to a zero-copy thread-parallel host
/// matvec that reads the caller's buffer in place, and leaves wgpu/cuda/rocm on
/// the upload + fused-kernel path they already win on (see that prim's docs for
/// both measurements).
///
/// The fitted `coef_`/`intercept_` stay device-resident on every backend (D-03);
/// only those two small buffers are read to host, and only on the cpu arm.
/// [`HostPrediction::operand_finite`] carries the sklearn-contract "`X` contains
/// no NaN/inf" verdict, which the cpu arm computes in the same pass as the
/// arithmetic; the caller decides whether a `false` is a rejection (the Python
/// surface's `ValueError`) or not.
pub(crate) fn predict_linear_from_host<F>(
    coef_: Option<&DeviceArray<ActiveRuntime, F>>,
    intercept_: Option<&DeviceArray<ActiveRuntime, F>>,
    mirror: &HostMirror<F>,
    estimator: &'static str,
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[F],
    shape: (usize, usize),
) -> Result<HostPrediction<F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (n_samples, n_features) = shape;

    let coef = coef_.ok_or(AlgoError::NotFitted {
        estimator,
        operation: "predict",
    })?;
    let intercept = intercept_.ok_or(AlgoError::NotFitted {
        estimator,
        operation: "predict",
    })?;

    // --- ASVS V5: geometry + fitted-n_features consistency, identical to
    // `predict_linear`'s. The prim re-validates, but keeping the check here
    // means both ingresses reject the same shapes with the same typed error
    // rather than relying on a downstream layer to agree.
    if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "x",
            rows: n_samples,
            cols: n_features,
            len: x.len(),
        }));
    }
    if coef.len() != n_features {
        return Err(AlgoError::Prim(PrimError::DimMismatch {
            dim: "n_features",
            lhs: coef.len(),
            rhs: n_features,
        }));
    }

    Ok(linear_predict_from_host::<F>(
        pool,
        x,
        coef,
        intercept,
        mirror,
        (n_samples, n_features),
    )?)
}
