//! `MinMaxScaler` (PREP-01) — linear map of each column into `feature_range`,
//! matching `sklearn.preprocessing.MinMaxScaler`.
//!
//! `scale_ = (range_max − range_min) / (data_max_ − data_min_)`,
//! `min_ = range_min − data_min_ · scale_`; `transform(x) = x·scale_ + min_`
//! (a constant column's zero `data_range_` is replaced with `1` via
//! [`super::common::handle_zeros_in_scale`]).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use super::common::{affine_columns_host, column_min_max, handle_zeros_in_scale, zeros_eps};
use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

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
