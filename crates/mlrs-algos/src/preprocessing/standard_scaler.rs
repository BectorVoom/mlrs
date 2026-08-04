//! `StandardScaler` (PREP-01) — `(x − mean) / std`, matching
//! `sklearn.preprocessing.StandardScaler`.
//!
//! `mean_`/`var_`/`scale_` are always computed at `fit` (so they are always
//! introspectable, a superset of sklearn which leaves them `None` when the
//! corresponding `with_mean`/`with_std` is `False`); `with_mean`/`with_std`
//! gate only which terms `transform` actually applies. `var_` is the
//! POPULATION variance (`ddof = 0`, sklearn's convention); a near-zero `var_`
//! (constant column) gets `scale_ = 1` via
//! [`super::common::handle_zeros_in_scale`] (PREP-01's degenerate-column gate).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, PrimError};

use super::common::{affine_columns_host, column_sum_sumsq, handle_zeros_in_scale, zeros_eps};
use crate::error::AlgoError;
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

/// `StandardScaler` (PREP-01): construct with [`StandardScaler::new`] /
/// [`StandardScaler::builder`], then [`Fit::fit`] / [`Transform::transform`] /
/// [`Transform::inverse_transform`].
pub struct StandardScaler<F, S = Unfit> {
    with_mean: bool,
    with_std: bool,
    mean_: Option<DeviceArray<ActiveRuntime, F>>,
    var_: Option<DeviceArray<ActiveRuntime, F>>,
    scale_: Option<DeviceArray<ActiveRuntime, F>>,
    n_features: usize,
    _state: PhantomData<S>,
}

impl<F> StandardScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn defaults: `with_mean = True`, `with_std = True`.
    pub fn new() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            mean_: None,
            var_: None,
            scale_: None,
            n_features: 0,
            _state: PhantomData,
        }
    }

    pub fn builder() -> StandardScalerBuilder {
        StandardScalerBuilder::default()
    }
}

impl<F> Default for StandardScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StandardScalerBuilder {
    with_mean: bool,
    with_std: bool,
}

impl Default for StandardScalerBuilder {
    fn default() -> Self {
        Self {
            with_mean: true,
            with_std: true,
        }
    }
}

impl StandardScalerBuilder {
    pub fn with_mean(mut self, v: bool) -> Self {
        self.with_mean = v;
        self
    }

    pub fn with_std(mut self, v: bool) -> Self {
        self.with_std = v;
        self
    }

    pub fn build<F>(self) -> StandardScaler<F, Unfit>
    where
        F: Float + CubeElement + Pod,
    {
        StandardScaler {
            with_mean: self.with_mean,
            with_std: self.with_std,
            mean_: None,
            var_: None,
            scale_: None,
            n_features: 0,
            _state: PhantomData,
        }
    }
}

impl<F> StandardScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    pub fn mean(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.mean_, pool)
    }

    pub fn var(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.var_, pool)
    }

    pub fn scale(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.scale_, pool)
    }

    fn attr(&self, slot: &Option<DeviceArray<ActiveRuntime, F>>, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        slot.as_ref()
            .expect("fitted attribute is Some by construction on StandardScaler<F, Fitted>")
            .to_host(pool)
    }
}

impl<F> Fit<F> for StandardScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = StandardScaler<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        validate_geometry(x, shape)?;
        let (n, d) = shape;

        let (sum, sumsq) = column_sum_sumsq::<F>(pool, x, n, d)?;
        let n64 = n as f64;
        let mean64: Vec<f64> = sum.iter().map(|&s| s / n64).collect();
        // var = E[x^2] - mean^2, clamped >= 0 (f64 round-off can dip a true-zero
        // variance slightly negative, the same guard `gaussian_nb.rs` uses).
        let mut var64: Vec<f64> = (0..d)
            .map(|c| {
                let v = sumsq[c] / n64 - mean64[c] * mean64[c];
                v.max(0.0)
            })
            .collect();
        let mut scale64: Vec<f64> = var64.iter().map(|&v| v.sqrt()).collect();
        handle_zeros_in_scale(&mut scale64, zeros_eps::<F>());
        // var_ itself is exposed pre-`handle_zeros_in_scale` in sklearn (only
        // `scale_` gets the degenerate-column substitution); keep both.
        var64.iter_mut().for_each(|v| *v = v.max(0.0));

        let mean_dev = DeviceArray::from_host(pool, &mean64.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<_>>());
        let var_dev = DeviceArray::from_host(pool, &var64.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<_>>());
        let scale_dev = DeviceArray::from_host(pool, &scale64.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<_>>());

        Ok(StandardScaler {
            with_mean: self.with_mean,
            with_std: self.with_std,
            mean_: Some(mean_dev),
            var_: Some(var_dev),
            scale_: Some(scale_dev),
            n_features: d,
            _state: PhantomData,
        })
    }
}

impl<F> StandardScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The effective `(scale_factor, shift_factor)` such that
    /// `transform(x) = x * scale_factor + shift_factor`, honoring
    /// `with_mean`/`with_std` (both `True` gives the textbook `(x−mean)/std`).
    fn effective_affine(&self, pool: &BufferPool<ActiveRuntime>) -> (Vec<f64>, Vec<f64>) {
        use mlrs_core::host_to_f64;
        let mean: Vec<f64> = self
            .mean_
            .as_ref()
            .expect("Some by construction")
            .to_host(pool)
            .iter()
            .map(|&v| host_to_f64(v))
            .collect();
        let scale: Vec<f64> = self
            .scale_
            .as_ref()
            .expect("Some by construction")
            .to_host(pool)
            .iter()
            .map(|&v| host_to_f64(v))
            .collect();
        let d = mean.len();
        let mut scale_factor = vec![1.0f64; d];
        let mut shift_factor = vec![0.0f64; d];
        for c in 0..d {
            let s = if self.with_std { scale[c] } else { 1.0 };
            let m = if self.with_mean { mean[c] } else { 0.0 };
            scale_factor[c] = 1.0 / s;
            shift_factor[c] = -m / s;
        }
        (scale_factor, shift_factor)
    }
}

impl<F> Transform<F> for StandardScaler<F, Fitted>
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
        Ok(affine_columns_host(pool, x, n, d, &scale_factor, &shift_factor))
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
        // x = (out - shift_factor) / scale_factor = out * (1/scale_factor) + (-shift_factor/scale_factor)
        let inv_scale: Vec<f64> = scale_factor.iter().map(|&s| 1.0 / s).collect();
        let inv_shift: Vec<f64> = shift_factor
            .iter()
            .zip(scale_factor.iter())
            .map(|(&sh, &sc)| -sh / sc)
            .collect();
        Ok(affine_columns_host(pool, z, n, d, &inv_scale, &inv_shift))
    }
}
