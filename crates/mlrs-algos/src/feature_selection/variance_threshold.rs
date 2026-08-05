//! `VarianceThreshold` (FSEL-01) — drop every column whose variance is at or
//! below `threshold`, matching
//! `sklearn.feature_selection.VarianceThreshold`.
//!
//! The only UNSUPERVISED selector in the family: `fit(X, y=None)` and `y` is
//! documented as ignored ("This parameter exists only for compatibility with
//! sklearn.pipeline.Pipeline").
//!
//! ## Three sklearn behaviours that are easy to miss and all observable
//!
//! 1. **NaN input is ACCEPTED.** Alone among the selectors, this one validates
//!    with `ensure_all_finite="allow-nan"` and computes `np.nanvar`. The
//!    [`ColMoments`](mlrs_backend::prims::feature_score::ColMoments) sweep is
//!    NaN-skipping for exactly this reason.
//! 2. **`threshold == 0` compares the PEAK-TO-PEAK range, not the variance.**
//!    sklearn takes `nanmin(variance, ptp)` per column, with the comment "Use
//!    peak-to-peak to avoid numeric precision issues for constant features": a
//!    genuinely constant column can have a variance of `1e-17` rather than `0`
//!    from cancellation, and would survive a `> 0` test. Using the range —
//!    exactly `0` for a constant column — closes that. This applies ONLY at
//!    `threshold == 0`, and `variances_` is OVERWRITTEN with the min, so the
//!    fitted attribute a caller reads is the min rather than the variance.
//! 3. **An all-dropped fit RAISES.** If no column exceeds the threshold sklearn
//!    raises `ValueError("No feature in X meets the variance threshold
//!    {:.5f}")` — with a `(X contains only one sample)` suffix when `n == 1` —
//!    rather than returning an empty selector. Reproduced as
//!    [`AlgoError::InvalidFeatureInput`] carrying sklearn's message verbatim, so
//!    a caller matching on the text sees what it expects.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::feature_score::col_moments;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::host_to_f64;

use crate::error::AlgoError;
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

use super::selector::{inverse_transform_selected, transform_selected, Selector};

/// `VarianceThreshold`'s whole `fit`, over a row-major `f64` HOST slice:
/// `(variances_, support_mask)`.
///
/// Public and host-shaped because there are two callers and they must not drift:
/// the typestate [`Fit`] impl below (which reads its device operand to host
/// first — the statistics are host `f64` by design, see
/// [`mlrs_backend::prims::feature_score`]), and the PyO3 wrapper, which already
/// has host data and would otherwise pay two extra copies of the design to route
/// it through a device buffer it never uses. One implementation means the Rust
/// oracle test covers the Python path too.
///
/// The three sklearn behaviours this reproduces — NaN tolerance, the
/// `threshold == 0` peak-to-peak substitution, and the raise on an all-dropped
/// fit — are documented on the module.
pub fn variances_and_support(
    x: &[f64],
    n: usize,
    d: usize,
    threshold: f64,
) -> Result<(Vec<f64>, Vec<bool>), AlgoError> {
    let moments = col_moments(x, n, d)?;
    let mut variances = moments.variance_biased();
    if threshold == 0.0 {
        // sklearn: `variances_ = nanmin([variances_, peak_to_peaks], axis=0)`.
        // `nanmin` prefers the non-NaN of the pair, so an all-NaN column (both
        // NaN) stays NaN and any other column takes the smaller.
        let ptp = moments.peak_to_peak();
        for c in 0..d {
            variances[c] = match (variances[c].is_nan(), ptp[c].is_nan()) {
                (true, true) => f64::NAN,
                (true, false) => ptp[c],
                (false, true) => variances[c],
                (false, false) => variances[c].min(ptp[c]),
            };
        }
    }
    let support: Vec<bool> = variances.iter().map(|&v| v > threshold).collect();

    // sklearn: `if np.all(~np.isfinite(variances_) | (variances_ <= threshold)):
    // raise`. The condition is on NON-FINITE-or-below, not simply on the support
    // being empty — a column with an INFINITE variance survives the mask
    // (`inf > threshold`) but does NOT rescue the fit from this check.
    // Reproduced literally, including that asymmetry.
    if variances.iter().all(|&v| !v.is_finite() || v <= threshold) {
        let mut reason = format!("No feature in X meets the variance threshold {threshold:.5}");
        if n == 1 {
            reason.push_str(" (X contains only one sample)");
        }
        return Err(AlgoError::InvalidFeatureInput {
            estimator: "variance_threshold",
            reason,
        });
    }
    Ok((variances, support))
}

/// `sklearn.feature_selection.VarianceThreshold(threshold=0.0)`.
#[derive(Debug, Clone)]
pub struct VarianceThreshold<F, S = Unfit> {
    threshold: f64,
    /// `variances_` — empty until fitted.
    variances: Vec<f64>,
    support: Vec<bool>,
    _state: PhantomData<(F, S)>,
}

impl<F> VarianceThreshold<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn default: `threshold = 0.0`.
    pub fn new() -> Self {
        Self::with_threshold(0.0)
    }

    /// `VarianceThreshold(threshold=..)`.
    ///
    /// A NEGATIVE threshold is accepted, as sklearn accepts it
    /// (`Interval(Real, 0, None)` is not applied to this parameter — its
    /// constraint is just `[Interval(Real, 0, None, closed="left")]` in recent
    /// versions, but the comparison `variances_ > threshold` is total for any
    /// finite value, so mlrs does not add a rejection sklearn does not have).
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            threshold,
            variances: Vec::new(),
            support: Vec::new(),
            _state: PhantomData,
        }
    }
}

impl<F> Default for VarianceThreshold<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Fit<F> for VarianceThreshold<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = VarianceThreshold<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        validate_geometry(x, shape)?;
        let (n, d) = shape;
        let host: Vec<f64> = x.to_host(pool).into_iter().map(host_to_f64).collect();
        let (variances, support) = variances_and_support(&host, n, d, self.threshold)?;

        Ok(VarianceThreshold {
            threshold: self.threshold,
            variances,
            support,
            _state: PhantomData,
        })
    }
}

impl<F> VarianceThreshold<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// `variances_` — the per-column variance, or at `threshold == 0` the
    /// `min(variance, peak_to_peak)` sklearn overwrites it with (module docs).
    pub fn variances(&self) -> &[f64] {
        &self.variances
    }

    /// The `threshold` this selector was built with.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl<F, S> Selector for VarianceThreshold<F, S> {
    fn support_mask(&self) -> &[bool] {
        &self.support
    }
}

impl<F> Transform<F> for VarianceThreshold<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        transform_selected(self, pool, x, shape, "variance_threshold")
    }

    fn inverse_transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        z: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        inverse_transform_selected(self, pool, z, shape)
    }
}
