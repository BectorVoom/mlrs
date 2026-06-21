//! Dtype dispatch generator (D-06) — the `any_estimator!` macro contract that
//! Plan 03's 12 `#[pyclass]` wrappers invoke.
//!
//! A `#[pyclass]` cannot be generic over the float type `F`, so each estimator is
//! wrapped in an internal three-state enum:
//!
//! ```text
//! enum AnyEstimator {
//!     Unfit { /* the sklearn-named hyperparameters, stored verbatim */ },
//!     F32(mlrs_algos::…::Estimator<f32>),   // fitted, f32 monomorphization
//!     F64(mlrs_algos::…::Estimator<f64>),   // fitted, f64 monomorphization
//! }
//! ```
//!
//! `fit` inspects the incoming pyarrow float dtype ([`crate::ingress::float_dtype`]),
//! constructs the matching arm from the stored hyperparameters, and runs the
//! `mlrs_algos` trait call. `predict` / `transform` / `labels` match on the
//! fitted arm. A `macro_rules!` generates this per estimator so the 12 wrappers
//! are not hand-written boilerplate.
//!
//! ## The two load-bearing contracts the macro encodes (for Plan 03)
//!
//! 1. **GIL release around device compute (PY-03 / Pitfall 6).** Every method
//!    that touches the device wraps the trait call in `py.detach(|| { … })`. The
//!    closure is `Send` and touches no Python objects: it locks the process-global
//!    pool ([`crate::global_pool`]), runs the `mlrs_algos` call, and returns a
//!    plain-Rust `Result`. The canonical shape is:
//!
//!    The sanctioned lock path is [`crate::lock_pool`], which RECOVERS from mutex
//!    poisoning (WR-02/WR-04) so a single panicked `fit` cannot permanently brick
//!    the interpreter. Do NOT use `global_pool().lock().expect(...)` directly in a
//!    panic-prone `fit`/accessor — that form re-panics on a poisoned mutex and
//!    defeats the recovery `lock_pool` provides.
//!
//!    ```ignore
//!    let out = py.detach(|| {
//!        let mut pool = crate::lock_pool();
//!        let arr = /* the owned ingress::ArrayRef */;
//!        match $crate::ingress::float_dtype(&arr)? {
//!            $crate::ingress::FloatDtype::F32 => {
//!                let dev = $crate::ingress::validated_f32($crate::ingress::as_f32(&arr)?, &mut pool)?;
//!                // self.inner = AnyKMeans::F32(KMeans::<f32>::new(..).fit(&mut pool, dev, (rows, cols))?);
//!            }
//!            $crate::ingress::FloatDtype::F64 => {
//!                $crate::capability::guard_f64()?;            // D-04: BEFORE upload
//!                let dev = $crate::ingress::validated_f64($crate::ingress::as_f64(&arr)?, &mut pool)?;
//!                // self.inner = AnyKMeans::F64(KMeans::<f64>::new(..).fit(&mut pool, dev, (rows, cols))?);
//!            }
//!        }
//!    });
//!    out.map_err(/* errors:: mapping */)
//!    ```
//!
//! 2. **f64 guard before the f64 arm (D-04).** On the `FloatDtype::F64` branch the
//!    macro calls [`crate::capability::guard_f64`]`()?` BEFORE constructing the
//!    `F64` arm or uploading, so f64 on an f64-incapable backend raises the clear
//!    `PyValueError` and never allocates a device buffer / downcasts.
//!
//! Plan 03 fleshes out per-trait method bodies (Predict / Transform /
//! PredictLabels / KNeighbors / PredictProba) on top of this skeleton.

/// Generate the per-estimator dtype-dispatch enum for one `mlrs_algos`
/// estimator (D-06).
///
/// **Status: skeleton.** This emits the `Any<Name>` three-state enum (the
/// `Unfit { .. }` + `F32(Estimator<f32>)` + `F64(Estimator<f64>)` shape). Plan 03
/// invokes it per estimator and adds the `#[pymethods]` (`fit` with the dtype
/// dispatch + f64 guard + `py.detach` device call documented in this module's
/// doc comment, plus the trait-specific accessors).
///
/// Invocation shape Plan 03 will use:
///
/// ```ignore
/// any_estimator! {
///     any:   AnyKMeans,                       // the internal enum name
///     algo:  mlrs_algos::cluster::KMeans,     // the generic Estimator<F>
///     unfit: { n_clusters: usize, seed: u64 },// sklearn-named hyperparameters
/// }
/// ```
///
/// The emitted enum:
///
/// ```ignore
/// enum AnyKMeans {
///     Unfit { n_clusters: usize, seed: u64 },
///     F32(mlrs_algos::cluster::KMeans<f32>),
///     F64(mlrs_algos::cluster::KMeans<f64>),
/// }
/// ```
#[macro_export]
macro_rules! any_estimator {
    (
        any:   $any:ident,
        algo:  $algo:ident $( :: $algo_rest:ident )*,
        unfit: { $( $field:ident : $ty:ty ),* $(,)? } $(,)?
    ) => {
        /// Internal dtype-dispatch enum (D-06): an unfit state holding the
        /// sklearn-named hyperparameters, plus the two fitted monomorphizations.
        enum $any {
            /// Constructed-but-unfit: the verbatim hyperparameters the matching
            /// `Estimator<F>` arm is built from at `fit`.
            Unfit { $( $field : $ty ),* },
            /// Fitted f32 monomorphization.
            F32($algo $( :: $algo_rest )* <f32>),
            /// Fitted f64 monomorphization.
            F64($algo $( :: $algo_rest )* <f64>),
        }
        // NOTE (Plan 03): the `#[pymethods] impl PyEstimator { fn fit(...) {...} }`
        // block — with `float_dtype` dispatch, `guard_f64()?` on the F64 arm, and
        // the `py.detach(|| { crate::lock_pool()... })` device call (the
        // poison-recovering sanctioned lock path, WR-04) — extends
        // this enum. The skeleton fixes the enum shape + the two contracts
        // (GIL release, f64 guard) documented at the module level above.
    };
}
