//! `Selector` — the Rust analogue of `sklearn.feature_selection.SelectorMixin`
//! (FSEL-01).
//!
//! sklearn's mixin derives EVERYTHING from one abstract method,
//! `_get_support_mask()`: `get_support`, `transform`, `inverse_transform` and
//! `get_feature_names_out` are all consequences of the mask. This trait keeps
//! that shape — one required method, [`Selector::support_mask`], and provided
//! implementations for the rest — so a downstream selector added later gets the
//! whole surface by supplying its mask, exactly as a `SelectorMixin` subclass
//! does.
//!
//! ## Why a trait here and not a variant of `typestate::Transform`
//! Every selector also implements [`crate::typestate::Transform`], because
//! `transform`/`inverse_transform` on device buffers is what a caller composing
//! a pipeline uses. `Selector` is the SMALLER, mask-shaped surface on top: the
//! mask is the fitted model (a `SelectKBest` is `k` booleans and nothing else),
//! it is what the Python layer needs in order to do its own container-native
//! column take, and it is what `get_support(indices=True)` returns. Keeping it
//! separate is what lets the PyO3 wrapper ship the mask across the boundary and
//! let pandas/polars do the gather natively, instead of round-tripping a whole
//! matrix through a device buffer to drop columns from it.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::feature_score::{gather_columns, scatter_columns};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;

/// A fitted feature selector: something that knows which of its input columns
/// survive.
pub trait Selector {
    /// The length-`n_features_in` boolean mask — `true` for a retained column.
    ///
    /// This is `sklearn`'s `_get_support_mask()`, and the single method a new
    /// selector must provide.
    fn support_mask(&self) -> &[bool];

    /// `get_support(indices=False)` — the mask itself.
    fn get_support(&self) -> &[bool] {
        self.support_mask()
    }

    /// `get_support(indices=True)` — the retained column indices, ascending.
    ///
    /// `u32` rather than `usize` because these feed the device gather kernel,
    /// whose index array is `u32`; the count is bounded by `n_features`, so the
    /// narrowing cannot lose information at any width mlrs supports.
    fn support_indices(&self) -> Vec<u32> {
        self.support_mask()
            .iter()
            .enumerate()
            .filter(|(_, &keep)| keep)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// The feature count this selector was fitted on (`n_features_in_`).
    fn n_features_in(&self) -> usize {
        self.support_mask().len()
    }

    /// The retained feature count — the width of `transform`'s output.
    fn n_features_out(&self) -> usize {
        self.support_mask().iter().filter(|&&k| k).count()
    }

    /// Whether any feature survived.
    ///
    /// sklearn WARNS rather than raising when nothing does ("No features were
    /// selected: either the data is too noisy or the selection test too
    /// strict") and returns an `n × 0` matrix. mlrs reproduces the `n × 0`
    /// result; a caller that wants the warning checks this.
    fn selects_any(&self) -> bool {
        self.support_mask().iter().any(|&k| k)
    }
}

/// `SelectorMixin.transform` for a device-resident design: gather the retained
/// columns of `x` (`rows × n_features_in`) into a `rows × n_features_out`
/// buffer.
///
/// Shared by every selector's [`crate::typestate::Transform`] impl rather than
/// duplicated, because there is genuinely one implementation: sklearn's is
/// `_safe_indexing(X, mask, axis=1)` and nothing more.
pub(crate) fn transform_selected<F, S>(
    selector: &S,
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    estimator: &'static str,
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
    S: Selector,
{
    let (rows, cols) = shape;
    if cols != selector.n_features_in() || x.len() != rows * cols {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "x",
            rows,
            cols: selector.n_features_in(),
            len: x.len(),
        }));
    }
    let _ = estimator;
    Ok(gather_columns(pool, x, shape, &selector.support_indices())?)
}

/// `SelectorMixin.inverse_transform` for a device-resident latent matrix:
/// scatter `z` (`rows × n_features_out`) back into a zero-filled
/// `rows × n_features_in` frame.
pub(crate) fn inverse_transform_selected<F, S>(
    selector: &S,
    pool: &mut BufferPool<ActiveRuntime>,
    z: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
    S: Selector,
{
    let (rows, cols) = shape;
    if cols != selector.n_features_out() || z.len() != rows * cols {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "z",
            rows,
            cols: selector.n_features_out(),
            len: z.len(),
        }));
    }
    Ok(scatter_columns(
        pool,
        z,
        rows,
        &selector.support_indices(),
        selector.n_features_in(),
    )?)
}

/// `_clean_nans` — replace every `NaN` score with `f64::MIN` so it sorts LAST.
///
/// sklearn's comment says why it exists ("NaNs can't be properly compared") and
/// why it is not `-inf` ("-inf seems to be unreliable"). Both `SelectKBest` and
/// `SelectPercentile` call it before ranking, and it is the reason a constant
/// column — whose `f_classif` score is `NaN` — is never selected instead of a
/// merely-uninformative one.
///
/// mlrs applies this to a COPY, as sklearn does (`as_float_array(scores,
/// copy=True)`), so the `scores_` attribute a caller reads still holds the `NaN`
/// the score function produced. That distinction is observable: sklearn's
/// `scores_` on a constant column IS `NaN`, and a test comparing attributes
/// would fail if the cleaning were applied in place.
pub(crate) fn clean_nans(scores: &[f64]) -> Vec<f64> {
    scores
        .iter()
        .map(|&s| if s.is_nan() { f64::MIN } else { s })
        .collect()
}

/// `numpy.percentile(a, q)` with the default LINEAR interpolation — the
/// `SelectPercentile` threshold.
///
/// numpy's default `method="linear"` places the `q`-th percentile at the
/// fractional index `(len − 1)·q/100` and interpolates between the two
/// neighbouring order statistics. Reproducing the INTERPOLATION (rather than
/// picking a nearest order statistic) is what makes the resulting `scores >
/// threshold` mask match sklearn's, because the threshold usually falls strictly
/// between two scores and a different convention moves it across one of them.
pub(crate) fn percentile_linear(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = (v.len() - 1) as f64 * (q / 100.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return v[lo];
    }
    let frac = pos - lo as f64;
    v[lo] + (v[hi] - v[lo]) * frac
}
