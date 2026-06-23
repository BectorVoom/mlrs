//! NEW typestate-aware estimator surface (D-03/D-05/D-06/D-07).
//!
//! This module is the canonical Rust-native lifecycle surface for the v3
//! builder-pattern API: a sealed [`State`] marker trait with the two
//! zero-sized markers [`Unfit`] and [`Fitted`], and the four lifecycle traits
//! [`Fit`] / [`Predict`] / [`Transform`] / [`PartialFit`] that re-tag an
//! estimator's state at the type level.
//!
//! ## Coexistence with the frozen `traits.rs` (D-07)
//! The trait NAMES here (`Fit` / `Predict` / `Transform` / `PartialFit`)
//! deliberately mirror the legacy [`crate::traits`] surface, but the SIGNATURES
//! differ: the legacy `Fit::fit` takes `&mut self` and returns `&mut Self`,
//! whereas this module's [`Fit::fit`] CONSUMES `self` and returns an associated
//! [`Fit::Fitted`] type — a compile-time typestate transition. The legacy
//! `traits.rs` is FROZEN: all 30 existing estimators continue to compile against
//! it untouched (Pitfall 1). The two surfaces collide ONLY by path; never glob
//! both into the same `use` at one call site. Consumers of the new surface write
//! `use mlrs_algos::typestate::Fit;` explicitly.
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

use crate::error::AlgoError;

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
/// legacy [`crate::traits::Fit`], whose `fit` took `&mut self` and returned
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

/// Type-level guard that the consuming-`self` typestate traits compose with a
/// `PhantomData<S>` state slot without requiring any `Self: Sized` workaround.
/// This is a zero-cost helper for downstream estimator authors (Plan 02) — it
/// has no runtime effect and produces no code.
#[doc(hidden)]
pub fn _state_phantom<S: State>() -> PhantomData<S> {
    PhantomData
}
