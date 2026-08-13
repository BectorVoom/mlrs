//! `MaxAbsScaler` (PREP-01) — `x / max(|x|)` per column, matching
//! `sklearn.preprocessing.MaxAbsScaler`.
//!
//! `max_abs_[c] = max(|data_min_[c]|, |data_max_[c]|)` — the column extrema
//! from [`super::common::column_min_max`] already bound `|x|`'s max (no
//! separate abs-reduction needed). A zero `max_abs_` (an all-zero column) is
//! replaced with `1` via [`super::common::handle_zeros_in_scale`].

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
    read_columns, write_columns, AlignedBytes, LoadModel, PersistError, PrepFile, PrepWriter,
    SaveModel,
};
use crate::error::AlgoError;
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

/// The `estimator` discriminator written into every `MaxAbsScaler` file.
const PERSIST_TAG: &str = "max_abs_scaler";

/// The two fitted vectors, in the order they are written and read.
const COLUMNS: [&str; 2] = ["max_abs_", "scale_"];

pub struct MaxAbsScaler<F, S = Unfit> {
    max_abs_: Option<DeviceArray<ActiveRuntime, F>>,
    scale_: Option<DeviceArray<ActiveRuntime, F>>,
    n_features: usize,
    _state: PhantomData<S>,
}

impl<F> MaxAbsScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    pub fn new() -> Self {
        Self {
            max_abs_: None,
            scale_: None,
            n_features: 0,
            _state: PhantomData,
        }
    }
}

impl<F> Default for MaxAbsScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> MaxAbsScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    pub fn max_abs(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.max_abs_, pool)
    }

    pub fn scale(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.scale_, pool)
    }

    fn attr(&self, slot: &Option<DeviceArray<ActiveRuntime, F>>, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        slot.as_ref()
            .expect("fitted attribute is Some by construction on MaxAbsScaler<F, Fitted>")
            .to_host(pool)
    }
}

impl<F> SaveModel for MaxAbsScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted scaler to `path` as a safetensors file.
    ///
    /// Two `[n_features]` tensors — `max_abs_` and `scale_` — and no `param:`
    /// scalars at all: `MaxAbsScaler` has no constructor arguments. It is the
    /// smallest file this family produces, and the only estimator anywhere in
    /// mlrs whose `__metadata__` is nothing but the three discriminators.
    ///
    /// The two vectors differ only on an all-zero column, where `scale_` carries
    /// [`handle_zeros_in_scale`]'s substituted `1` and `max_abs_` keeps the true
    /// `0`. That is precisely why both are stored: neither is recoverable from
    /// the other, and `max_abs_` is a fitted attribute sklearn exposes.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        // Bound BEFORE the writer, which borrows every payload.
        let max_abs = self.max_abs_.as_ref().ok_or_else(|| absent("max_abs_"))?.to_host(pool);
        let scale = self.scale_.as_ref().ok_or_else(|| absent("scale_"))?.to_host(pool);

        let mut w = PrepWriter::new(PERSIST_TAG);
        write_columns(
            &mut w,
            &[(COLUMNS[0], max_abs.as_slice()), (COLUMNS[1], scale.as_slice())],
        )?;
        w.write(path)
    }
}

impl<F> LoadModel for MaxAbsScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read a scaler back from `path`, re-uploading both vectors to `pool`.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<MaxAbsScaler<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = PrepFile::parse(&raw, PERSIST_TAG)?;
        let (cols, n_features) = read_columns::<F>(&file, &COLUMNS)?;

        Ok(MaxAbsScaler {
            max_abs_: Some(DeviceArray::from_host(pool, &cols[0])),
            scale_: Some(DeviceArray::from_host(pool, &cols[1])),
            n_features,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for MaxAbsScaler<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = MaxAbsScaler<F, Fitted>;

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
        let max_abs64: Vec<f64> = data_min
            .iter()
            .zip(data_max.iter())
            .map(|(&mn, &mx)| mn.abs().max(mx.abs()))
            .collect();
        let mut scale64 = max_abs64.clone();
        handle_zeros_in_scale(&mut scale64, zeros_eps::<F>());

        let mut to_dev = |v: &[f64]| DeviceArray::from_host(pool, &v.iter().map(|&x| f64_to_host::<F>(x)).collect::<Vec<_>>());

        Ok(MaxAbsScaler {
            max_abs_: Some(to_dev(&max_abs64)),
            scale_: Some(to_dev(&scale64)),
            n_features: d,
            _state: PhantomData,
        })
    }
}

impl<F> Transform<F> for MaxAbsScaler<F, Fitted>
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
        let inv_scale: Vec<f64> = scale.iter().map(|&s| 1.0 / s).collect();
        let zero_shift = vec![0.0f64; d];
        Ok(affine_columns_host(pool, x, n, d, &inv_scale, &zero_shift, None))
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
        let zero_shift = vec![0.0f64; d];
        Ok(affine_columns_host(pool, z, n, d, &scale, &zero_shift, None))
    }
}
