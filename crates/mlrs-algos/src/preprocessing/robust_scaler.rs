//! `RobustScaler` (PREP-01) — `(x − median) / IQR`, matching
//! `sklearn.preprocessing.RobustScaler`. Median / quantile are order
//! statistics ([`super::common::columns_host_f64`] + a host sort — no useful
//! device parallelism, the ARIMA/BayesianRidge host-arm precedent).
//!
//! Quantile interpolation matches `numpy.percentile(..., method="linear")`
//! (numpy's default, which sklearn's `np.nanpercentile` call also uses):
//! for a length-`n` sorted column and percentile `q ∈ [0, 100]`, the
//! fractional rank is `h = (q/100) · (n − 1)`; the result linearly
//! interpolates between `sorted[floor(h)]` and `sorted[ceil(h)]`.

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use super::common::{affine_columns_host, columns_host_f64, handle_zeros_in_scale, norm_ppf, zeros_eps};
use super::prep_persist::{
    read_columns, read_range, write_columns, write_range, AlignedBytes, LoadModel, PersistError,
    PrepFile, PrepWriter, SaveModel,
};
use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

/// The `estimator` discriminator written into every `RobustScaler` file.
const PERSIST_TAG: &str = "robust_scaler";

/// The two fitted vectors, in the order they are written and read.
const COLUMNS: [&str; 2] = ["center_", "scale_"];

/// The two halves of the `quantile_range` constructor pair.
const QUANTILE_MIN_KEY: &str = "param:quantile_range_min";
/// See [`QUANTILE_MIN_KEY`].
const QUANTILE_MAX_KEY: &str = "param:quantile_range_max";

/// `numpy.percentile(col, q, method="linear")` on an ALREADY-SORTED column.
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (q / 100.0) * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = h - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

pub struct RobustScaler<F, S = Unfit> {
    with_centering: bool,
    with_scaling: bool,
    quantile_range: (f64, f64),
    unit_variance: bool,
    center_: Option<DeviceArray<ActiveRuntime, F>>,
    scale_: Option<DeviceArray<ActiveRuntime, F>>,
    n_features: usize,
    _state: PhantomData<S>,
}

impl<F> RobustScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn defaults: `with_centering = True`, `with_scaling = True`,
    /// `quantile_range = (25.0, 75.0)`, `unit_variance = False`.
    pub fn new() -> Self {
        Self {
            with_centering: true,
            with_scaling: true,
            quantile_range: (25.0, 75.0),
            unit_variance: false,
            center_: None,
            scale_: None,
            n_features: 0,
            _state: PhantomData,
        }
    }

    pub fn builder() -> RobustScalerBuilder {
        RobustScalerBuilder::default()
    }
}

impl<F> Default for RobustScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RobustScalerBuilder {
    with_centering: bool,
    with_scaling: bool,
    quantile_range: (f64, f64),
    unit_variance: bool,
}

impl Default for RobustScalerBuilder {
    fn default() -> Self {
        Self {
            with_centering: true,
            with_scaling: true,
            quantile_range: (25.0, 75.0),
            unit_variance: false,
        }
    }
}

impl RobustScalerBuilder {
    pub fn with_centering(mut self, v: bool) -> Self {
        self.with_centering = v;
        self
    }

    pub fn with_scaling(mut self, v: bool) -> Self {
        self.with_scaling = v;
        self
    }

    pub fn quantile_range(mut self, q_min: f64, q_max: f64) -> Self {
        self.quantile_range = (q_min, q_max);
        self
    }

    pub fn unit_variance(mut self, v: bool) -> Self {
        self.unit_variance = v;
        self
    }

    pub fn build<F>(self) -> Result<RobustScaler<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let (q_min, q_max) = self.quantile_range;
        let valid = q_min.is_finite()
            && q_max.is_finite()
            && (0.0..=100.0).contains(&q_min)
            && (0.0..=100.0).contains(&q_max)
            && q_min < q_max;
        if !valid {
            return Err(BuildError::InvalidRange {
                estimator: "robust_scaler",
                param: "quantile_range",
                min: q_min,
                max: q_max,
            });
        }
        Ok(RobustScaler {
            with_centering: self.with_centering,
            with_scaling: self.with_scaling,
            quantile_range: self.quantile_range,
            unit_variance: self.unit_variance,
            center_: None,
            scale_: None,
            n_features: 0,
            _state: PhantomData,
        })
    }
}

impl<F> RobustScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    pub fn center(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.center_, pool)
    }

    pub fn scale(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.scale_, pool)
    }

    fn attr(&self, slot: &Option<DeviceArray<ActiveRuntime, F>>, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        slot.as_ref()
            .expect("fitted attribute is Some by construction on RobustScaler<F, Fitted>")
            .to_host(pool)
    }
}

impl<F> SaveModel for RobustScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted scaler to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `center_` / `scale_` | `F` (`F32`/`F64`) | `[n_features]` |
    /// | `param:with_centering` / `param:with_scaling` / `param:unit_variance` | `__metadata__` scalar | — |
    /// | `param:quantile_range_min` / `_max` | `__metadata__` scalar | — |
    ///
    /// `quantile_range` is stored even though `scale_` has already absorbed it —
    /// the quantiles were consumed at fit time and the file records what the
    /// scaler was built with, not what `transform` still needs. Without it a
    /// loaded scaler could not report its own configuration, which is the point
    /// of round-tripping a model rather than just its affine map. The same
    /// applies to `unit_variance`, whose [`norm_ppf`] correction is likewise
    /// already folded into `scale_`.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        // Bound BEFORE the writer, which borrows every payload.
        let center = self.center_.as_ref().ok_or_else(|| absent("center_"))?.to_host(pool);
        let scale = self.scale_.as_ref().ok_or_else(|| absent("scale_"))?.to_host(pool);

        let mut w = PrepWriter::new(PERSIST_TAG);
        w.scalar_bool("param:with_centering", self.with_centering);
        w.scalar_bool("param:with_scaling", self.with_scaling);
        w.scalar_bool("param:unit_variance", self.unit_variance);
        write_range(&mut w, QUANTILE_MIN_KEY, QUANTILE_MAX_KEY, self.quantile_range);
        write_columns(
            &mut w,
            &[(COLUMNS[0], center.as_slice()), (COLUMNS[1], scale.as_slice())],
        )?;
        w.write(path)
    }
}

impl<F> LoadModel for RobustScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read a scaler back from `path`, re-uploading both vectors to `pool`.
    ///
    /// `quantile_range` is read back but NOT re-validated against
    /// `RobustScalerBuilder`'s `0 <= q_min < q_max <= 100` rule, for the reason
    /// [`MinMaxScaler::load`](super::min_max_scaler::MinMaxScaler) gives: that
    /// check belongs to the constructor, where a caller supplied the pair.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<RobustScaler<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = PrepFile::parse(&raw, PERSIST_TAG)?;
        let (cols, n_features) = read_columns::<F>(&file, &COLUMNS)?;

        Ok(RobustScaler {
            with_centering: file.scalar_bool("param:with_centering")?,
            with_scaling: file.scalar_bool("param:with_scaling")?,
            quantile_range: read_range(&file, QUANTILE_MIN_KEY, QUANTILE_MAX_KEY)?,
            unit_variance: file.scalar_bool("param:unit_variance")?,
            center_: Some(DeviceArray::from_host(pool, &cols[0])),
            scale_: Some(DeviceArray::from_host(pool, &cols[1])),
            n_features,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for RobustScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = RobustScaler<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        validate_geometry(x, shape)?;
        let (n, d) = shape;
        let (q_min, q_max) = self.quantile_range;

        let mut columns = columns_host_f64::<F>(pool, x, n, d);
        let mut center64 = vec![0.0f64; d];
        let mut scale64 = vec![0.0f64; d];
        for (c, col) in columns.iter_mut().enumerate() {
            col.sort_by(|a, b| a.total_cmp(b));
            center64[c] = percentile_sorted(col, 50.0);
            let lo = percentile_sorted(col, q_min);
            let hi = percentile_sorted(col, q_max);
            scale64[c] = hi - lo;
        }
        handle_zeros_in_scale(&mut scale64, zeros_eps::<F>());
        if self.unit_variance {
            let denom = norm_ppf(q_max / 100.0) - norm_ppf(q_min / 100.0);
            for s in scale64.iter_mut() {
                *s /= denom;
            }
        }

        let mut to_dev = |v: &[f64]| DeviceArray::from_host(pool, &v.iter().map(|&x| f64_to_host::<F>(x)).collect::<Vec<_>>());

        Ok(RobustScaler {
            with_centering: self.with_centering,
            with_scaling: self.with_scaling,
            quantile_range: self.quantile_range,
            unit_variance: self.unit_variance,
            center_: Some(to_dev(&center64)),
            scale_: Some(to_dev(&scale64)),
            n_features: d,
            _state: PhantomData,
        })
    }
}

impl<F> RobustScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn effective_affine(&self, pool: &BufferPool<ActiveRuntime>) -> (Vec<f64>, Vec<f64>) {
        let center: Vec<f64> = self.center_.as_ref().unwrap().to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        let scale: Vec<f64> = self.scale_.as_ref().unwrap().to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        let d = center.len();
        let mut scale_factor = vec![1.0f64; d];
        let mut shift_factor = vec![0.0f64; d];
        for c in 0..d {
            let s = if self.with_scaling { scale[c] } else { 1.0 };
            let m = if self.with_centering { center[c] } else { 0.0 };
            scale_factor[c] = 1.0 / s;
            shift_factor[c] = -m / s;
        }
        (scale_factor, shift_factor)
    }
}

impl<F> Transform<F> for RobustScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n, d) = shape;
        if d != self.n_features || x.len() != n * d {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n,
                cols: d,
                len: x.len(),
            }));
        }
        let (scale_factor, shift_factor) = self.effective_affine(pool);
        Ok(affine_columns_host(pool, x, n, d, &scale_factor, &shift_factor, None))
    }

    fn inverse_transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        z: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n, d) = shape;
        if d != self.n_features || z.len() != n * d {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "z",
                rows: n,
                cols: d,
                len: z.len(),
            }));
        }
        let (scale_factor, shift_factor) = self.effective_affine(pool);
        let inv_scale: Vec<f64> = scale_factor.iter().map(|&s| 1.0 / s).collect();
        let inv_shift: Vec<f64> = shift_factor
            .iter()
            .zip(scale_factor.iter())
            .map(|(&sh, &sc)| -sh / sc)
            .collect();
        Ok(affine_columns_host(pool, z, n, d, &inv_scale, &inv_shift, None))
    }
}
