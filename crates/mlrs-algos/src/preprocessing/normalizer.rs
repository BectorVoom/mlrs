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
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use super::common::zeros_eps;
use super::prep_persist::{
    read_n_features, AlignedBytes, LoadModel, PersistError, PrepFile, PrepWriter, SaveModel,
    N_FEATURES_KEY,
};
use crate::error::AlgoError;
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

/// The `estimator` discriminator written into every `Normalizer` file.
const PERSIST_TAG: &str = "normalizer";

/// The `__metadata__` key holding the row norm.
const NORM_KEY: &str = "param:norm";

/// Which row norm [`Normalizer`] rescales by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    L1,
    L2,
    Max,
}

impl Norm {
    /// The sklearn spelling of this variant (`'l1'` / `'l2'` / `'max'`).
    ///
    /// The model file stores the variant as this string rather than as an
    /// integer tag, so `safetensors.numpy.load_file`'s metadata reads the way
    /// the sklearn constructor argument does — and so adding a variant later
    /// cannot silently renumber an existing file's.
    pub fn name(self) -> &'static str {
        match self {
            Norm::L1 => "l1",
            Norm::L2 => "l2",
            Norm::Max => "max",
        }
    }

    /// The inverse of [`Norm::name`]; `None` for an unrecognised string.
    ///
    /// Returns an `Option` rather than a `Result` so each caller frames the
    /// failure in its own terms — the Python builder raises a `ValueError`
    /// naming the argument, while [`Normalizer::load`] raises a
    /// [`PersistError::BadMetadata`] naming the key it came from.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "l1" => Some(Norm::L1),
            "l2" => Some(Norm::L2),
            "max" => Some(Norm::Max),
            _ => None,
        }
    }
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

impl<F> SaveModel for Normalizer<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted transformer to `path` as a safetensors file.
    ///
    /// NO tensors — `Normalizer` learns no statistic, so the file is a header
    /// and nothing else, a few hundred bytes regardless of the data it was
    /// fitted on. Two `__metadata__` entries carry the whole model:
    /// `param:norm` (the constructor argument) and `n_features_in_` (the one
    /// thing `fit` actually learned, and the one thing `transform` validates
    /// its input against).
    ///
    /// `n_features_in_` is written WITHOUT the `param:` prefix because it is a
    /// fitted attribute rather than a constructor input — see
    /// [`prep_persist`](super::prep_persist) for why this family's two
    /// tensorless members store it and its four scalers do not.
    ///
    /// `pool` is unused: there is nothing device-resident to read back. It is
    /// present because [`SaveModel`] is one signature for every estimator, the
    /// same shape [`Fit`] already has.
    fn save(&self, _pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let mut w = PrepWriter::new(PERSIST_TAG);
        w.scalar_str(NORM_KEY, self.norm.name());
        w.scalar_usize(N_FEATURES_KEY, self.n_features);
        w.write(path)
    }
}

impl<F> LoadModel for Normalizer<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the transformer back from `path`.
    ///
    /// The norm is PARSED rather than trusted: an unrecognised string becomes a
    /// [`PersistError::BadMetadata`] naming its key, so a file written by a
    /// future build that grew a fourth variant fails by name here instead of
    /// silently falling back to `'l2'` and rescaling every row differently than
    /// the saved transformer would have.
    fn load(
        _pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<Normalizer<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = PrepFile::parse(&raw, PERSIST_TAG)?;

        let norm = Norm::from_name(file.scalar_str(NORM_KEY)?)
            .ok_or(PersistError::BadMetadata { key: NORM_KEY })?;

        Ok(Normalizer {
            norm,
            n_features: read_n_features(&file)?,
            _state: PhantomData,
        })
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
