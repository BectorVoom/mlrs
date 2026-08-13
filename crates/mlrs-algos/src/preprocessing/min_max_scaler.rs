//! `MinMaxScaler` (PREP-01) — linear map of each column into `feature_range`,
//! matching `sklearn.preprocessing.MinMaxScaler`.
//!
//! `scale_ = (range_max − range_min) / (data_max_ − data_min_)`,
//! `min_ = range_min − data_min_ · scale_`; `transform(x) = x·scale_ + min_`
//! (a constant column's zero `data_range_` is replaced with `1` via
//! [`super::common::handle_zeros_in_scale`]).

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use super::common::{affine_columns_host, column_min_max, handle_zeros_in_scale, zeros_eps};
use super::prep_persist::{
    read_columns, read_range, write_columns, write_range, AlignedBytes, LoadModel, PersistError,
    PrepFile, PrepWriter, SaveModel,
};
use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

/// The `estimator` discriminator written into every `MinMaxScaler` file.
const PERSIST_TAG: &str = "min_max_scaler";

/// The four fitted vectors, in the order they are written and read. One list
/// rather than two so the save and load sides cannot drift — `data_min_` and
/// `min_` are the same length and similar magnitude, so a reordering on one side
/// only is exactly the mistake no geometry check could catch.
const COLUMNS: [&str; 4] = ["data_min_", "data_max_", "scale_", "min_"];

/// The two halves of the `feature_range` constructor pair.
const RANGE_MIN_KEY: &str = "param:feature_range_min";
/// See [`RANGE_MIN_KEY`].
const RANGE_MAX_KEY: &str = "param:feature_range_max";

pub struct MinMaxScaler<F, S = Unfit> {
    feature_range: (f64, f64),
    clip: bool,
    data_min_: Option<DeviceArray<ActiveRuntime, F>>,
    data_max_: Option<DeviceArray<ActiveRuntime, F>>,
    scale_: Option<DeviceArray<ActiveRuntime, F>>,
    min_: Option<DeviceArray<ActiveRuntime, F>>,
    n_features: usize,
    _state: PhantomData<S>,
}

impl<F> MinMaxScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn default: `feature_range = (0.0, 1.0)`, `clip = False`.
    pub fn new() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            clip: false,
            data_min_: None,
            data_max_: None,
            scale_: None,
            min_: None,
            n_features: 0,
            _state: PhantomData,
        }
    }

    pub fn builder() -> MinMaxScalerBuilder {
        MinMaxScalerBuilder::default()
    }
}

impl<F> Default for MinMaxScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MinMaxScalerBuilder {
    feature_range: (f64, f64),
    clip: bool,
}

impl Default for MinMaxScalerBuilder {
    fn default() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            clip: false,
        }
    }
}

impl MinMaxScalerBuilder {
    pub fn feature_range(mut self, min: f64, max: f64) -> Self {
        self.feature_range = (min, max);
        self
    }

    pub fn clip(mut self, v: bool) -> Self {
        self.clip = v;
        self
    }

    pub fn build<F>(self) -> Result<MinMaxScaler<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let (lo, hi) = self.feature_range;
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return Err(BuildError::InvalidRange {
                estimator: "min_max_scaler",
                param: "feature_range",
                min: lo,
                max: hi,
            });
        }
        Ok(MinMaxScaler {
            feature_range: self.feature_range,
            clip: self.clip,
            data_min_: None,
            data_max_: None,
            scale_: None,
            min_: None,
            n_features: 0,
            _state: PhantomData,
        })
    }
}

impl<F> MinMaxScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    pub fn data_min(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.data_min_, pool)
    }

    pub fn data_max(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.data_max_, pool)
    }

    pub fn scale(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.scale_, pool)
    }

    pub fn min(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.min_, pool)
    }

    fn attr(&self, slot: &Option<DeviceArray<ActiveRuntime, F>>, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        slot.as_ref()
            .expect("fitted attribute is Some by construction on MinMaxScaler<F, Fitted>")
            .to_host(pool)
    }
}

impl<F> SaveModel for MinMaxScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted scaler to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `data_min_` / `data_max_` / `scale_` / `min_` | `F` (`F32`/`F64`) | `[n_features]` |
    /// | `param:feature_range_min` / `_max`, `param:clip` | `__metadata__` scalar | — |
    ///
    /// All four vectors are stored even though `scale_` and `min_` are exactly
    /// the affine map `transform` applies and `data_min_`/`data_max_` are the
    /// raw extrema it was derived from. The derivation only runs ONE way —
    /// `scale_` folds in `feature_range` and the degenerate-column substitution
    /// [`handle_zeros_in_scale`] makes, so a constant column's `data_min_` and
    /// `data_max_` cannot be recovered from it. Keeping both pairs is what makes
    /// a loaded scaler introspect identically to the saved one; the cost is two
    /// `[n_features]` vectors on a file that is already four of them.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        // Bound BEFORE the writer, which borrows every payload.
        let data_min = self.data_min_.as_ref().ok_or_else(|| absent("data_min_"))?.to_host(pool);
        let data_max = self.data_max_.as_ref().ok_or_else(|| absent("data_max_"))?.to_host(pool);
        let scale = self.scale_.as_ref().ok_or_else(|| absent("scale_"))?.to_host(pool);
        let min = self.min_.as_ref().ok_or_else(|| absent("min_"))?.to_host(pool);

        let mut w = PrepWriter::new(PERSIST_TAG);
        write_range(&mut w, RANGE_MIN_KEY, RANGE_MAX_KEY, self.feature_range);
        w.scalar_bool("param:clip", self.clip);
        write_columns(
            &mut w,
            &[
                (COLUMNS[0], data_min.as_slice()),
                (COLUMNS[1], data_max.as_slice()),
                (COLUMNS[2], scale.as_slice()),
                (COLUMNS[3], min.as_slice()),
            ],
        )?;
        w.write(path)
    }
}

impl<F> LoadModel for MinMaxScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read a scaler back from `path`, re-uploading all four vectors to `pool`.
    ///
    /// `feature_range` is read back but NOT re-validated against
    /// `MinMaxScalerBuilder`'s `min < max` rule. That check belongs to the
    /// constructor, where a caller supplied the pair; here the pair is only a
    /// record of what the saved scaler was built with, and the fitted `scale_` /
    /// `min_` that `transform` actually applies were computed from it at fit
    /// time. Re-running the builder check would reject a file rather than the
    /// input that caused it, and it cannot fire on a file this crate wrote.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<MinMaxScaler<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = PrepFile::parse(&raw, PERSIST_TAG)?;
        let (cols, n_features) = read_columns::<F>(&file, &COLUMNS)?;

        Ok(MinMaxScaler {
            feature_range: read_range(&file, RANGE_MIN_KEY, RANGE_MAX_KEY)?,
            clip: file.scalar_bool("param:clip")?,
            data_min_: Some(DeviceArray::from_host(pool, &cols[0])),
            data_max_: Some(DeviceArray::from_host(pool, &cols[1])),
            scale_: Some(DeviceArray::from_host(pool, &cols[2])),
            min_: Some(DeviceArray::from_host(pool, &cols[3])),
            n_features,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for MinMaxScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = MinMaxScaler<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        validate_geometry(x, shape)?;
        let (n, d) = shape;

        let (data_min, data_max) = column_min_max::<F>(pool, x, n, d)?;
        let (lo, hi) = self.feature_range;
        let mut data_range: Vec<f64> = data_min.iter().zip(data_max.iter()).map(|(&mn, &mx)| mx - mn).collect();
        handle_zeros_in_scale(&mut data_range, zeros_eps::<F>());
        let scale64: Vec<f64> = data_range.iter().map(|&r| (hi - lo) / r).collect();
        let min64: Vec<f64> = data_min.iter().zip(scale64.iter()).map(|(&mn, &s)| lo - mn * s).collect();

        let mut to_dev = |v: &[f64]| DeviceArray::from_host(pool, &v.iter().map(|&x| f64_to_host::<F>(x)).collect::<Vec<_>>());

        Ok(MinMaxScaler {
            feature_range: self.feature_range,
            clip: self.clip,
            data_min_: Some(to_dev(&data_min)),
            data_max_: Some(to_dev(&data_max)),
            scale_: Some(to_dev(&scale64)),
            min_: Some(to_dev(&min64)),
            n_features: d,
            _state: PhantomData,
        })
    }
}

impl<F> Transform<F> for MinMaxScaler<F, Fitted>
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
        let scale: Vec<f64> = self.scale_.as_ref().unwrap().to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        let min: Vec<f64> = self.min_.as_ref().unwrap().to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        // `clip` folds into the affine pass rather than re-reading the buffer
        // it just uploaded — the clamp is free on a value already in a
        // register, and the round trip it replaces was the whole `n × d`
        // result, twice.
        let clamp = self.clip.then_some(self.feature_range);
        Ok(affine_columns_host(pool, x, n, d, &scale, &min, clamp))
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
        let scale: Vec<f64> = self.scale_.as_ref().unwrap().to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        let min: Vec<f64> = self.min_.as_ref().unwrap().to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        let inv_scale: Vec<f64> = scale.iter().map(|&s| 1.0 / s).collect();
        let inv_shift: Vec<f64> = min.iter().zip(scale.iter()).map(|(&m, &s)| -m / s).collect();
        Ok(affine_columns_host(pool, z, n, d, &inv_scale, &inv_shift, None))
    }
}
