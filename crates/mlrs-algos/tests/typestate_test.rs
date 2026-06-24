//! Plan 12-01 Wave-1 — coexistence/importability smoke test for the NEW
//! `mlrs_algos::typestate` surface (D-03/D-07).
//!
//! This is a COMPILE-LEVEL smoke test only: it proves the new module is
//! importable, that the `Unfit`/`Fitted` markers are zero-sized, and that both
//! satisfy the sealed `State` bound. The structural predict-before-fit PROOF
//! (a compile-fail assertion) is the Plan 03 `trybuild` gate and is deliberately
//! NOT duplicated here; the behavior tests live with the Plan 02 shells.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::AlgoError;
use mlrs_algos::typestate::{
    _state_phantom, Fitted, PredictLabels, ScoreSamples, State, Transform, Unfit,
};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Generic helper that compiles only if `S` satisfies the sealed [`State`]
/// bound — invoking it for a type is a static proof of `State` membership.
fn assert_state<S: State>() {}

#[test]
fn markers_are_zero_sized() {
    // Unfit/Fitted are pure type-level tags: they must add zero bytes when
    // carried as a `PhantomData<S>` state slot on an estimator.
    assert_eq!(std::mem::size_of::<Unfit>(), 0);
    assert_eq!(std::mem::size_of::<Fitted>(), 0);
}

#[test]
fn markers_satisfy_sealed_state_bound() {
    // Both markers must satisfy the sealed `State` bound. If either failed to
    // impl `State`, these calls would not compile.
    assert_state::<Unfit>();
    assert_state::<Fitted>();
}

#[test]
fn typestate_module_is_importable() {
    // The markers are constructible from the public path — confirms the module
    // is wired into lib.rs and reachable as `mlrs_algos::typestate::*`.
    let _unfit = Unfit;
    let _fitted = Fitted;
}

#[test]
fn state_phantom_helper_constructs_zero_sized_marker() {
    // Exercise the doc-hidden `_state_phantom` downstream helper (IN-04) so the
    // exported surface stays compiled-exercised rather than dead. It yields a
    // zero-sized `PhantomData<S>` for any sealed `State`.
    let unfit: PhantomData<Unfit> = _state_phantom::<Unfit>();
    let fitted: PhantomData<Fitted> = _state_phantom::<Fitted>();
    assert_eq!(std::mem::size_of_val(&unfit), 0);
    assert_eq!(std::mem::size_of_val(&fitted), 0);
}

// ---------------------------------------------------------------------------
// Plan 16-00 Wave-0 gate — coherence proof for the 5 NEW accessor traits and
// the `Transform::inverse_transform` default. This is a TYPE/TRAIT-surface
// proof (no CubeCL kernels launched): a tiny marker estimator impls a subset of
// the new traits on its `Fitted` variant, proving the traits are well-formed,
// importable from `mlrs_algos::typestate`, and impl'able ONLY on `Fitted`
// (mirroring how `Transform` is impl'd only on the fitted sibling). The
// `inverse_transform` default is exercised to confirm it returns the
// `Unsupported` error variant when not overridden.
// ---------------------------------------------------------------------------

/// A test-only marker estimator carrying a `PhantomData<S>` lifecycle slot, used
/// to prove the new accessor traits + the `Transform` default are coherent. It
/// has no real fitted state — these are trait-surface proofs, not behavior tests.
struct Marker<F, S = Unfit> {
    _f: PhantomData<F>,
    _state: PhantomData<S>,
}

impl<F> Marker<F, Fitted> {
    fn fitted() -> Self {
        Marker {
            _f: PhantomData,
            _state: PhantomData,
        }
    }
}

/// References 1 of the 5 new traits by name (`PredictLabels`), impl'd ONLY on the
/// `Fitted`-tagged marker — proving the trait is importable and `Fitted`-only.
impl<F> PredictLabels<F> for Marker<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        _x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        // Trivial host-built label buffer (one label per row); no device kernel.
        let labels: Vec<i32> = (0..shape.0 as i32).collect();
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

/// References a 2nd of the 5 new traits by name (`ScoreSamples`), again
/// `Fitted`-only — two distinct new traits exercised proves the surface.
impl<F> ScoreSamples<F> for Marker<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn score_samples(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        _x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let scores: Vec<F> = vec![F::from_int(0); shape.0];
        Ok(DeviceArray::from_host(pool, &scores))
    }
}

/// Impl `Transform` WITHOUT overriding `inverse_transform` — so the defaulted
/// method body (returns `Unsupported`) is the one under test.
impl<F> Transform<F> for Marker<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        _x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let out: Vec<F> = vec![F::from_int(0); shape.0 * shape.1];
        Ok(DeviceArray::from_host(pool, &out))
    }
    // inverse_transform intentionally NOT overridden — exercises the default.
}

#[test]
fn new_accessor_traits_resolve_on_fitted_marker() {
    // Proves PredictLabels + ScoreSamples (2 of the 5 new traits) are importable
    // from `mlrs_algos::typestate` and resolve on a `Fitted`-tagged estimator.
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &[0.0_f32; 6]);
    let est = Marker::<f32, Fitted>::fitted();

    let labels = est
        .predict_labels(&mut pool, &x, (3, 2))
        .expect("predict_labels resolves on the Fitted marker");
    assert_eq!(labels.len(), 3);

    let scores = est
        .score_samples(&mut pool, &x, (3, 2))
        .expect("score_samples resolves on the Fitted marker");
    assert_eq!(scores.len(), 3);
}

#[test]
fn transform_inverse_transform_default_returns_unsupported() {
    // The `Transform::inverse_transform` default body must surface the
    // `Unsupported` error when an impl does not override it (PCA overrides it;
    // every other transformer leaves this default).
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let z: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &[0.0_f32; 4]);
    let est = Marker::<f32, Fitted>::fitted();

    // `DeviceArray` is not `Debug`, so unwrap the Result with an explicit match
    // rather than `expect_err` (which requires the Ok type to be `Debug`).
    match est.inverse_transform(&mut pool, &z, (2, 2)) {
        Ok(_) => panic!("the default inverse_transform must return Unsupported, not Ok"),
        Err(AlgoError::Unsupported { operation, .. }) => {
            assert_eq!(operation, "inverse_transform");
        }
        Err(other) => panic!(
            "expected AlgoError::Unsupported {{ operation: \"inverse_transform\" }}, got {other:?}"
        ),
    }
}
