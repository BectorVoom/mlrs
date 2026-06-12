//! Shared estimator trait surface — `Fit` / `Predict` / `Transform` (D-04).
//!
//! These three traits are the uniform, sklearn-mixin-style surface every
//! Phase-4 (and Phase-5) estimator implements. They are the contract Phase-6's
//! PyO3 wrapping is written against once, generically: a regressor is `Fit` +
//! `Predict`; a decomposition is `Fit` + `Transform` (PCA also implements the
//! optional `inverse_transform`). This mirrors scikit-learn's
//! `RegressorMixin` / `TransformerMixin` split.
//!
//! ## Conventions (carried from D-08)
//! - Generic over `<F: Float + CubeElement + Pod>` exactly as the Phase-2/3
//!   primitives (`svd`/`eig`/`gemm`/`covariance`), so an estimator composes
//!   prim calls without a second type parameter.
//! - Inputs are flat row-major device buffers (`&DeviceArray<ActiveRuntime, F>`)
//!   with an explicit `(rows, cols)` geometry passed per call — the
//!   `DeviceArray` stays a flat 1-D buffer (D-08); the host never infers shape
//!   from the buffer.
//! - `fit` returns `&mut self` (sklearn convention: `fit` returns the
//!   estimator) so a `clf.fit(..).predict(..)`-style chain is expressible.
//! - Fitted state is device-resident (D-03): `predict` / `transform` return a
//!   fresh `DeviceArray` and run device-side; host materialization happens only
//!   at a Rust accessor or oracle-comparison boundary.
//! - Errors are the estimator-facing [`AlgoError`](crate::error::AlgoError),
//!   which wraps the primitive-level `PrimError` via `#[from]`.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2 — never an in-source
//! `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;

use crate::error::AlgoError;

/// Fit an estimator to training data, returning `&mut self` (sklearn
/// convention) so the call can be chained.
///
/// `y` is `Some` for supervised estimators (LinearRegression / Ridge) and
/// `None` for the unsupervised decompositions (PCA / TruncatedSVD). `shape` is
/// the explicit `(n_samples, n_features)` geometry of `x` (the `DeviceArray`
/// itself is a flat row-major buffer — D-08). Fitted attributes are stored
/// device-resident on `self` (D-03).
pub trait Fit<F>
where
    F: Float + CubeElement + Pod,
{
    /// Fit to `x` (`shape = (n_samples, n_features)`, row-major) and an optional
    /// target `y` (length `n_samples` for the regressors; `None` for the
    /// decompositions). Returns `&mut self` on success or an [`AlgoError`] on an
    /// invalid hyperparameter / geometry / primitive failure.
    fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError>;
}

/// Predict targets for new samples from a fitted regressor (LinearRegression /
/// Ridge). Runs device-side from the device-resident fitted `coef_`/`intercept_`
/// (D-03); the returned buffer is the length-`n_samples` prediction.
pub trait Predict<F>
where
    F: Float + CubeElement + Pod,
{
    /// Predict for `x` (`shape = (n_samples, n_features)`, row-major). Errors if
    /// the estimator is unfitted or the geometry disagrees with the fitted
    /// `n_features`.
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}

/// Project samples into the fitted latent space (PCA / TruncatedSVD) and,
/// optionally, reconstruct them back (PCA only). Runs device-side from the
/// device-resident `components_`/`mean_` (D-03).
pub trait Transform<F>
where
    F: Float + CubeElement + Pod,
{
    /// Project `x` (`shape = (n_samples, n_features)`, row-major) onto the fitted
    /// components, returning the `n_samples × n_components` transformed buffer.
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;

    /// Reconstruct samples from their latent representation `z`
    /// (`shape = (n_samples, n_components)`, row-major) back into the original
    /// feature space (`n_samples × n_features`). Implemented by PCA only;
    /// TruncatedSVD leaves the provided default, which returns
    /// [`AlgoError::Unsupported`].
    fn inverse_transform(
        &self,
        _pool: &mut BufferPool<ActiveRuntime>,
        _z: &DeviceArray<ActiveRuntime, F>,
        _shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        Err(AlgoError::Unsupported {
            estimator: "transform",
            operation: "inverse_transform",
        })
    }
}
