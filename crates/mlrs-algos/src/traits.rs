//! Shared estimator trait surface — `Fit` / `Predict` / `Transform` (D-04) plus
//! the Phase-5 discrete-output traits `PredictLabels` / `KNeighbors` /
//! `PredictProba` (D-05/D-07).
//!
//! These traits are the uniform, sklearn-mixin-style surface every Phase-4 (and
//! Phase-5) estimator implements. They are the contract Phase-6's PyO3 wrapping
//! is written against once, generically: a regressor is `Fit` + `Predict`; a
//! decomposition is `Fit` + `Transform` (PCA also implements the optional
//! `inverse_transform`). Phase-5 adds the distance-based / classification
//! surface: a clustering/classifier estimator is `Fit` + `PredictLabels`
//! (integer labels, D-05/D-06); a nearest-neighbor estimator is `Fit` +
//! `KNeighbors` (distances + indices, D-07); a probabilistic classifier adds
//! `PredictProba` (per-class fractions, D-07). This mirrors scikit-learn's
//! `RegressorMixin` / `ClassifierMixin` / `ClusterMixin` / `TransformerMixin`
//! split.
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

/// Incrementally fit an estimator to a single batch of training data, returning
/// `&mut self` (sklearn `partial_fit` convention) so calls can be chained across
/// a stream of batches. The estimator accumulates running state (e.g.
/// IncrementalPCA's `n_samples_seen_` / `mean_` / `var_` plus the running
/// SVD basis) across successive `partial_fit` calls rather than re-fitting from
/// scratch (D-01).
///
/// `y` mirrors [`Fit`]'s slot: IncrementalPCA — the only Phase-7 consumer —
/// passes `y: None` (it is unsupervised), but the `Option<&DeviceArray>` slot is
/// RETAINED per the D-01 cross-cutting contract so the Phase-10 mini-batch SGD
/// estimators (`MBSGDClassifier` / `MBSGDRegressor`) can reuse this exact trait
/// surface for supervised streaming without a signature change. `shape` is the
/// explicit `(n_batch_samples, n_features)` geometry of THIS batch's `x` (the
/// `DeviceArray` is a flat row-major buffer — D-08; `n_features` must agree with
/// the running state after the first batch). Accumulated attributes are stored
/// device-resident on `self` (D-03).
pub trait PartialFit<F>
where
    F: Float + CubeElement + Pod,
{
    /// Fit to a single batch `x` (`shape = (n_batch_samples, n_features)`,
    /// row-major), merging it into the running fitted state. `y` is
    /// `None` for IncrementalPCA (unsupervised) — the slot is retained per the
    /// D-01 cross-cutting contract for Phase-10 MBSGD reuse. Returns `&mut Self`
    /// on success or an [`AlgoError`] on an invalid hyperparameter / geometry /
    /// primitive failure (e.g. a batch whose `n_features` disagrees with the
    /// running state).
    fn partial_fit(
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

/// Predict INTEGER class/cluster labels for new samples (D-05/D-06). Unlike
/// [`Predict`], which returns the continuous `F` regression target, this returns
/// a length-`n_samples` `i32` label buffer — the discrete-output surface shared
/// by `KMeans.predict` (nearest cluster centroid → cluster id) and
/// `KNeighborsClassifier.predict` (majority neighbor vote → class id). Labels are
/// `i32` so DBSCAN's noise sentinel `-1` is directly representable (D-06).
///
/// Runs device-side from the device-resident fitted state (D-03); the returned
/// buffer is materialized on the host only at a Rust accessor / oracle-comparison
/// boundary, exactly like [`Predict`].
pub trait PredictLabels<F>
where
    F: Float + CubeElement + Pod,
{
    /// Predict the integer label for each sample of `x`
    /// (`shape = (n_samples, n_features)`, row-major). Errors if the estimator is
    /// unfitted or the geometry disagrees with the fitted `n_features`.
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError>;
}

/// Find the `k` nearest neighbors of each query sample (D-07). Returns BOTH the
/// `n_queries × k` neighbor distances (`F`) and the `n_queries × k` neighbor
/// indices (`i32`, indices into the fitted training set) — the sklearn
/// `NearestNeighbors.kneighbors` contract that `KNeighborsClassifier` /
/// `KNeighborsRegressor` build their votes on.
///
/// Runs device-side from the device-resident fitted training matrix (D-03); both
/// returned buffers are device-resident until a host accessor / oracle boundary.
pub trait KNeighbors<F>
where
    F: Float + CubeElement + Pod,
{
    /// For each row of `x` (`shape = (n_queries, n_features)`, row-major) return
    /// the `(distances, indices)` of its `k` nearest fitted-training neighbors,
    /// each a flat `n_queries × k` row-major buffer (distances `F`, indices
    /// `i32`). Errors if the estimator is unfitted, `k` exceeds the fitted sample
    /// count, or the geometry disagrees with the fitted `n_features`.
    fn kneighbors(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
        k: usize,
    ) -> Result<
        (
            DeviceArray<ActiveRuntime, F>,
            DeviceArray<ActiveRuntime, i32>,
        ),
        AlgoError,
    >;
}

/// Compute the per-sample LOG-DENSITY for new samples (D-12). This is the
/// density-estimation surface implemented by `KernelDensity.score_samples`: given
/// an `n_samples × n_features` query matrix it returns a length-`n_samples` buffer
/// of natural-log probability densities `log p(xᵢ)` evaluated under the fitted
/// kernel density model.
///
/// Distinct from [`Predict`] — this is NOT a regression target. The output is a
/// length-`n_samples` log-density vector (one scalar per query row), not an
/// `n_samples × n_features` reconstruction or an `n_samples`-length regression
/// prediction. The semantic difference (a probability density, evaluated in the
/// log domain for numerical stability) is why it is its own trait rather than a
/// reuse of `Predict` (D-12).
///
/// Runs device-side from the device-resident fitted training matrix / bandwidth
/// (D-03); the returned buffer is host-materialized only at a Rust accessor /
/// oracle-comparison boundary, exactly like [`Predict`].
pub trait ScoreSamples<F>
where
    F: Float + CubeElement + Pod,
{
    /// Compute the length-`n_samples` log-density `log p(xᵢ)` for each row of `x`
    /// (`shape = (n_samples, n_features)`, row-major) under the fitted kernel
    /// density model. Errors if the estimator is unfitted or the geometry
    /// disagrees with the fitted `n_features`.
    fn score_samples(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}

/// Predict per-class membership probabilities for new samples (D-07). Returns the
/// `n_samples × n_classes` row-major matrix of class fractions (each row sums to
/// 1) — the `predict_proba` surface implemented by `KNeighborsClassifier`
/// (neighbor-vote fractions) and `LogisticRegression` (softmax probabilities).
/// For LogisticRegression this is the PRIMARY gauge-invariant oracle gate
/// (RESEARCH Pitfall 5).
///
/// Runs device-side from the device-resident fitted state (D-03); the returned
/// buffer is host-materialized only at a Rust accessor / oracle boundary.
pub trait PredictProba<F>
where
    F: Float + CubeElement + Pod,
{
    /// Predict the per-class probability row for each sample of `x`
    /// (`shape = (n_samples, n_features)`, row-major), returning the flat
    /// `n_samples × n_classes` row-major buffer. Errors if the estimator is
    /// unfitted or the geometry disagrees with the fitted `n_features`.
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}
