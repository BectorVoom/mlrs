//! `Normalizer` (PREP-01) — per-ROW unit-norm rescale, matching
//! `sklearn.preprocessing.Normalizer`.
//!
//! Unlike the column scalers, `fit` learns NO statistic (sklearn's own
//! `Normalizer.fit` is a no-op beyond validating `X`/recording
//! `n_features_in_` — the norm is recomputed fresh on every `transform`
//! call); `Normalizer<F, Unfit>` and `Normalizer<F, Fitted>` differ only in
//! the compile-time typestate tag, not in any stored statistic. A zero-norm
//! row is left UNCHANGED (sklearn's `normalize()` treats a zero norm as `1`,
//! matching [`super::common::handle_zeros_in_scale`]'s degenerate-column
//! convention applied per-row instead of per-column).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use super::common::zeros_eps;
use crate::error::AlgoError;
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

/// Which row norm [`Normalizer`] rescales by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    L1,
    L2,
    Max,
}

pub struct Normalizer<F, S = Unfit> {
    norm: Norm,
    n_features: usize,
    _state: PhantomData<(F, S)>,
}

impl<F> Normalizer<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn default: `norm = 'l2'`.
    pub fn new() -> Self {
        Self {
            norm: Norm::L2,
            n_features: 0,
            _state: PhantomData,
        }
    }

    pub fn with_norm(norm: Norm) -> Self {
        Self {
            norm,
            n_features: 0,
            _state: PhantomData,
        }
    }
}

impl<F> Default for Normalizer<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Fit<F> for Normalizer<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = Normalizer<F, Fitted>;

    fn fit(
        self,
        _pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        validate_geometry(x, shape)?;
        Ok(Normalizer {
            norm: self.norm,
            n_features: shape.1,
            _state: PhantomData,
        })
    }
}

impl<F> Transform<F> for Normalizer<F, Fitted>
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
        let eps = zeros_eps::<F>();
        let x_host = x.to_host(pool);
        let mut out = vec![F::from_int(0i64); n * d];
        for r in 0..n {
            let row = &x_host[r * d..(r + 1) * d];
            let mut norm = match self.norm {
                Norm::L1 => row.iter().map(|&v| host_to_f64(v).abs()).sum::<f64>(),
                Norm::L2 => row.iter().map(|&v| { let f = host_to_f64(v); f * f }).sum::<f64>().sqrt(),
                Norm::Max => row.iter().map(|&v| host_to_f64(v).abs()).fold(0.0f64, f64::max),
            };
            if norm.abs() < eps {
                norm = 1.0;
            }
            for c in 0..d {
                out[r * d + c] = f64_to_host::<F>(host_to_f64(row[c]) / norm);
            }
        }
        Ok(DeviceArray::from_host(pool, &out))
    }
}
