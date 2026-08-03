//! `Binarizer` (PREP-01) — `x > threshold ? 1 : 0`, matching
//! `sklearn.preprocessing.Binarizer`. Like [`super::normalizer::Normalizer`],
//! `fit` learns no statistic: `threshold` is a construction-time
//! hyperparameter, not data-dependent.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::AlgoError;
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

pub struct Binarizer<F, S = Unfit> {
    threshold: f64,
    n_features: usize,
    _state: PhantomData<(F, S)>,
}

impl<F> Binarizer<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn default: `threshold = 0.0`.
    pub fn new() -> Self {
        Self {
            threshold: 0.0,
            n_features: 0,
            _state: PhantomData,
        }
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            threshold,
            n_features: 0,
            _state: PhantomData,
        }
    }
}

impl<F> Default for Binarizer<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Fit<F> for Binarizer<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = Binarizer<F, Fitted>;

    fn fit(
        self,
        _pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        validate_geometry(x, shape)?;
        Ok(Binarizer {
            threshold: self.threshold,
            n_features: shape.1,
            _state: PhantomData,
        })
    }
}

impl<F> Transform<F> for Binarizer<F, Fitted>
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
        let x_host = x.to_host(pool);
        let out: Vec<F> = x_host
            .iter()
            .map(|&v| {
                if host_to_f64(v) > self.threshold {
                    f64_to_host::<F>(1.0)
                } else {
                    f64_to_host::<F>(0.0)
                }
            })
            .collect();
        Ok(DeviceArray::from_host(pool, &out))
    }
}
