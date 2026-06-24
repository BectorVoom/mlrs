//! NEW typestate-aware estimator surface (D-03/D-05/D-06/D-07).
//!
//! This module is the canonical Rust-native lifecycle surface for the v3
//! builder-pattern API: a sealed [`State`] marker trait with the two
//! zero-sized markers [`Unfit`] and [`Fitted`], and the four lifecycle traits
//! [`Fit`] / [`Predict`] / [`Transform`] / [`PartialFit`] that re-tag an
//! estimator's state at the type level.
//!
//! ## The single trait surface (Phase 16, D-01)
//! This module is now the SINGLE trait surface for the whole crate. It mirrors
//! ALL 9 legacy `&mut self` traits — the four lifecycle traits
//! (`Fit` / `Predict` / `Transform` / `PartialFit`) plus the five `&self`
//! accessor traits (`PredictLabels` / `KNeighbors` / `ScoreSamples` /
//! `PredictProba` / `PredictLogProba`) — with the consuming-`self` typestate
//! signatures. The legacy `traits.rs` was HARD-DELETED in Phase 16 (D-01):
//! every estimator now consumes `mlrs_algos::typestate::*`, and the old
//! `traits.rs` module and its `pub mod traits;` declaration are gone — this is
//! the only trait surface. Consumers of this surface write
//! `use mlrs_algos::typestate::Fit;` explicitly.
//!
//! The SIGNATURES differ from the legacy surface: the legacy `Fit::fit` takes
//! `&mut self` and returns `&mut Self`, whereas this module's [`Fit::fit`]
//! CONSUMES `self` and returns an associated [`Fit::Fitted`] type — a
//! compile-time typestate transition. The five accessor traits, by contrast,
//! borrow `&self` (they READ fitted state, they do not transition lifecycle) and
//! carry the SAME signatures as their `traits.rs` originals verbatim, so they are
//! impl'd ONLY on the `Fitted`-tagged estimator (exactly like [`Transform`]).
//!
//! ## Builder-setter type convention (Phase 16, A5)
//! Builder setters are `f64`-typed for uniformity with the shipped mbsgd/umap
//! builders; `build::<F>()` narrows the stored `f64` hyperparameters to the
//! target float `F` via cast. Every downstream retrofit in this phase follows
//! this single convention — setters never take `F`, and the `f64 → F` narrowing
//! happens once, inside `build::<F>()`.
//!
//! ## The sealed `State` marker (D-03)
//! [`State`] is sealed via the private [`sealed::Sealed`] supertrait, so the set
//! of lifecycle states is CLOSED at the crate boundary — a downstream crate
//! cannot introduce a rogue `impl State for Evil` (threat T-12-01). The only two
//! inhabitants are [`Unfit`] (freshly built, not yet fitted) and [`Fitted`]
//! (the only state where `predict`/`transform`/accessors are reachable).
//!
//! ## Conventions (carried from D-08, identical to `traits.rs`)
//! - Generic over `<F: Float + CubeElement + Pod>` exactly as the primitives.
//! - Inputs are flat row-major device buffers (`&DeviceArray<ActiveRuntime, F>`)
//!   with an explicit `(rows, cols)` geometry passed per call (D-08).
//! - Errors are the estimator-facing [`AlgoError`](crate::error::AlgoError).
//!
//! No estimator impls live here — the UMAP/HDBSCAN shells (Plan 02) and the
//! Phase-16 retrofit provide those. Tests live in `crates/mlrs-algos/tests/`
//! (AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::PrimError;

use crate::error::AlgoError;

/// Validate the data-DEPENDENT geometry of a flat row-major device buffer `x`
/// against an explicit `(rows, cols)` shape (D-08), shared by the typestate
/// estimator shells so the `fit`/`transform` guards stay in lockstep (IN-03).
///
/// Rejects a degenerate `rows == 0` / `cols == 0` geometry and a `x.len()` that
/// disagrees with `rows * cols`, surfacing the same
/// [`PrimError::ShapeMismatch`] the inline guards previously constructed
/// verbatim at three sites. Returning the typed error keeps an untrusted host
/// geometry from reaching a device read (mirrors `mbsgd_regressor.rs`).
pub(crate) fn validate_geometry<F>(
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<(), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (n, p) = shape;
    if n == 0 || p == 0 || x.len() != n * p {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: p,
            len: x.len(),
        }));
    }
    Ok(())
}

/// Private sealing supertrait. Living in a private module makes [`State`]
/// impossible to implement outside this crate: a downstream type cannot name
/// `sealed::Sealed` to satisfy the `State: sealed::Sealed` bound, so the
/// lifecycle-state set is closed (D-03, threat T-12-01).
mod sealed {
    /// The sealed marker. Implemented ONLY for [`super::Unfit`] and
    /// [`super::Fitted`] within this crate.
    pub trait Sealed {}
}

/// Sealed marker trait for an estimator's compile-time lifecycle state.
///
/// The only inhabitants are [`Unfit`] and [`Fitted`]. Because [`State`] is
/// sealed via the private [`sealed::Sealed`] supertrait, downstream crates
/// cannot add a third state — the lifecycle is a CLOSED two-element set (D-03).
/// Estimators carry their state as a `PhantomData<S: State>` type parameter so a
/// `predict`-before-`fit` is a compile error rather than a runtime
/// [`AlgoError::NotFitted`].
pub trait State: sealed::Sealed {}

/// Zero-sized marker: a freshly BUILT estimator that has not yet been fitted
/// (D-03). An estimator tagged `Unfit` exposes only [`Fit::fit`] /
/// [`PartialFit::partial_fit`]; `predict`/`transform`/fitted-attribute accessors
/// are unreachable until the type transitions to [`Fitted`].
pub struct Unfit;

/// Zero-sized marker: a FITTED estimator (D-03). This is the only state in which
/// [`Predict::predict`], [`Transform::transform`], and the fitted-attribute
/// accessors are reachable — replacing the old runtime
/// [`AlgoError::NotFitted`](crate::error::AlgoError::NotFitted) guard with a
/// compile-time one.
pub struct Fitted;

impl sealed::Sealed for Unfit {}
impl State for Unfit {}

impl sealed::Sealed for Fitted {}
impl State for Fitted {}

/// Fit an estimator to training data, CONSUMING `self` and returning a freshly
/// typed [`Fit::Fitted`] value (D-05). This is the typestate counterpart of the
/// legacy `&mut self` `Fit`, whose `fit` took `&mut self` and returned
/// `&mut Self`; here the move-and-retag makes a `predict`-before-`fit` a
/// compile error.
///
/// `y` is `Some` for supervised estimators and `None` for the unsupervised
/// decompositions / manifold learners. `shape` is the explicit
/// `(n_samples, n_features)` geometry of `x` (the `DeviceArray` is a flat
/// row-major buffer — D-08).
pub trait Fit<F>
where
    F: Float + CubeElement + Pod,
{
    /// The fitted form of this estimator — typically `Self`'s `Fitted`-tagged
    /// sibling (e.g. `Umap<F, Fitted>`). Producing a distinct type is what makes
    /// the `Unfit → Fitted` transition visible to the type system (D-05).
    type Fitted;

    /// Fit to `x` (`shape = (n_samples, n_features)`, row-major) and an optional
    /// target `y`, CONSUMING `self` and returning the [`Fit::Fitted`] value on
    /// success or an [`AlgoError`] on an invalid geometry / primitive failure.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError>;
}

/// Predict targets for new samples from a FITTED estimator (D-05). Implemented
/// only on the `Fitted`-tagged estimator, so calling `predict` before `fit` does
/// not type-check. Borrows `&self`, runs device-side from the device-resident
/// fitted state (D-03), and returns a fresh length-`n_samples` prediction.
pub trait Predict<F>
where
    F: Float + CubeElement + Pod,
{
    /// Predict for `x` (`shape = (n_samples, n_features)`, row-major). Errors if
    /// the geometry disagrees with the fitted `n_features`.
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}

/// Project samples into a FITTED estimator's latent space (D-05). Implemented
/// only on the `Fitted`-tagged estimator. Borrows `&self`, runs device-side from
/// the device-resident `components_`/`mean_` (D-03).
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
    /// feature space (`n_samples × n_features`). Implemented by PCA only
    /// (the reconstruction path); estimators without a reconstruction (e.g.
    /// TruncatedSVD, UMAP) leave the provided default, which returns
    /// [`AlgoError::Unsupported`] — a compile-time-present but runtime-rejected
    /// method, matching the legacy `traits.rs` `Transform::inverse_transform`
    /// default verbatim (D-01).
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

/// Incrementally fit an estimator to a single batch, CONSUMING `self` and
/// returning the next-state [`PartialFit::Fitted`] value (D-06). Unlike [`Fit`],
/// `PartialFit` is a MULTI-TRANSITION trait: it is intended to be implemented on
/// BOTH `Unfit` (first batch: `Unfit → Fitted`) and `Fitted` (subsequent
/// batches: `Fitted → Fitted`), so a stream of `partial_fit` calls accumulates
/// running state across batches.
///
/// This trait is DEFINED-BUT-UNUSED in Phase 12 — no estimator implements it
/// yet. It is the Phase-16 retrofit target for the streaming estimators
/// (`IncrementalPCA` / `MBSGDClassifier` / `MBSGDRegressor`). `y` mirrors
/// [`Fit`]'s slot (retained for supervised streaming); `shape` is THIS batch's
/// `(n_batch_samples, n_features)` geometry (D-08).
pub trait PartialFit<F>
where
    F: Float + CubeElement + Pod,
{
    /// The state after merging this batch — typically the `Fitted`-tagged
    /// sibling, so the same `Fitted` type can implement `PartialFit` again for
    /// the next batch (`Fitted → Fitted`), giving the multi-transition stream.
    type Fitted;

    /// Fit to a single batch `x` (`shape = (n_batch_samples, n_features)`,
    /// row-major), merging it into the running fitted state and CONSUMING `self`.
    /// `y` is `None` for the unsupervised consumers; retained for supervised
    /// streaming. Returns the [`PartialFit::Fitted`] value or an [`AlgoError`].
    fn partial_fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError>;
}

/// Predict INTEGER class/cluster labels for new samples (D-05/D-06). Unlike
/// [`Predict`], which returns the continuous `F` regression target, this returns
/// a length-`n_samples` `i32` label buffer — the discrete-output surface shared
/// by `KMeans.predict` (nearest cluster centroid → cluster id) and
/// `KNeighborsClassifier.predict` (majority neighbor vote → class id). Labels are
/// `i32` so DBSCAN's noise sentinel `-1` is directly representable (D-06).
///
/// A `&self` ACCESSOR trait: it reads device-resident fitted state (D-03) and
/// does NOT transition lifecycle, so it carries no associated `type Fitted` and
/// is impl'd ONLY on the `Fitted`-tagged estimator (mirroring how [`Transform`]
/// is impl'd only on the fitted sibling). Signature ported verbatim from the
/// legacy `traits.rs::PredictLabels` (D-01).
pub trait PredictLabels<F>
where
    F: Float + CubeElement + Pod,
{
    /// Predict the integer label for each sample of `x`
    /// (`shape = (n_samples, n_features)`, row-major). Errors if the geometry
    /// disagrees with the fitted `n_features`.
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
/// A `&self` ACCESSOR trait (no `type Fitted`), impl'd ONLY on the `Fitted`-tagged
/// estimator. Signature ported verbatim from the legacy `traits.rs::KNeighbors`
/// (D-01).
pub trait KNeighbors<F>
where
    F: Float + CubeElement + Pod,
{
    /// For each row of `x` (`shape = (n_queries, n_features)`, row-major) return
    /// the `(distances, indices)` of its `k` nearest fitted-training neighbors,
    /// each a flat `n_queries × k` row-major buffer (distances `F`, indices
    /// `i32`). Errors if `k` exceeds the fitted sample count or the geometry
    /// disagrees with the fitted `n_features`.
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
/// A `&self` ACCESSOR trait (no `type Fitted`), impl'd ONLY on the `Fitted`-tagged
/// estimator. Signature ported verbatim from the legacy `traits.rs::ScoreSamples`
/// (D-01).
pub trait ScoreSamples<F>
where
    F: Float + CubeElement + Pod,
{
    /// Compute the length-`n_samples` log-density `log p(xᵢ)` for each row of `x`
    /// (`shape = (n_samples, n_features)`, row-major) under the fitted kernel
    /// density model. Errors if the geometry disagrees with the fitted
    /// `n_features`.
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
///
/// A `&self` ACCESSOR trait (no `type Fitted`), impl'd ONLY on the `Fitted`-tagged
/// estimator. Signature ported verbatim from the legacy `traits.rs::PredictProba`
/// (D-01).
pub trait PredictProba<F>
where
    F: Float + CubeElement + Pod,
{
    /// Predict the per-class probability row for each sample of `x`
    /// (`shape = (n_samples, n_features)`, row-major), returning the flat
    /// `n_samples × n_classes` row-major buffer. Errors if the geometry disagrees
    /// with the fitted `n_features`.
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}

/// Predict the per-class LOG-probabilities for new samples (D-07, Phase 11). The
/// sibling of [`PredictProba`]: returns the SAME `n_samples × n_classes`
/// row-major matrix but in the log domain — the `predict_log_proba` surface the
/// five Naive Bayes classifiers (`GaussianNB` / `MultinomialNB` / `BernoulliNB`
/// / `ComplementNB` / `CategoricalNB`) implement alongside `predict_proba`.
///
/// A `&self` ACCESSOR trait (no `type Fitted`), impl'd ONLY on the `Fitted`-tagged
/// estimator. Signature ported verbatim from the legacy
/// `traits.rs::PredictLogProba` (D-01).
pub trait PredictLogProba<F>
where
    F: Float + CubeElement + Pod,
{
    /// Predict the per-class log-probability row for each sample of `x`
    /// (`shape = (n_samples, n_features)`, row-major), returning the flat
    /// `n_samples × n_classes` row-major buffer of `joint_ll − logsumexp`.
    /// Errors if the geometry disagrees with the fitted `n_features`.
    fn predict_log_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}

/// Type-level guard that the consuming-`self` typestate traits compose with a
/// `PhantomData<S>` state slot without requiring any `Self: Sized` workaround.
/// This is a zero-cost helper for downstream estimator authors (Plan 02) — it
/// has no runtime effect and produces no code.
#[doc(hidden)]
pub fn _state_phantom<S: State>() -> PhantomData<S> {
    PhantomData
}
