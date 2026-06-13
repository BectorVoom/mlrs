//! `KNeighborsRegressor` (NEIGH-03) — brute-force k-NN mean regressor, matching
//! `sklearn.neighbors.KNeighborsRegressor(algorithm='brute', metric='euclidean',
//! weights='uniform')`.
//!
//! ## Predict = mean of the k neighbor targets (D-07)
//! `predict` finds the `k` nearest neighbors of each query (reusing the validated
//! `NearestNeighbors` core, [`neighbor_indices`]), gathers their continuous `F`
//! targets, and returns the arithmetic MEAN (uniform weights — each of the `k`
//! neighbors contributes `1/k`). `weights='distance'` is deferred per CONTEXT.
//!
//! ## Validate-before-launch (T-05-08-01 / ASVS V5)
//! `predict` rejects `k` outside `1 ..= n_train` ([`AlgoError::InvalidK`]) and a
//! mismatched query geometry BEFORE any prim launch (the shared `neighbor_indices`
//! core enforces this, 05-08 Task 1).
//!
//! ## Device residency (D-03)
//! The fitted training matrix is device-resident; the small `n_train` target
//! vector is kept host-side for the per-query gather, and the mean is staged back
//! to a device-resident output.
//!
//! Tests live in `crates/mlrs-algos/tests/knn_regressor_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::neighbors::nearest::neighbor_indices;
use crate::traits::{Fit, Predict};

/// Brute-force k-NN mean regressor (NEIGH-03).
///
/// Construct with [`KNeighborsRegressor::new`] (`n_neighbors`), then [`Fit::fit`]
/// (stores the training matrix + its `F` regression targets) and
/// [`Predict::predict`] (mean of the k neighbor targets). Fitted state is
/// device-resident (D-03).
pub struct KNeighborsRegressor<F> {
    /// Neighbor count `k` (the averaging pool size). Validated against `n_train`
    /// at predict time ([`AlgoError::InvalidK`]).
    n_neighbors: usize,
    /// Device-resident training matrix (`n_train × n_features`, row-major),
    /// `None` until `fit`.
    x_train_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted training geometry `(n_train, n_features)`, `None` until `fit`.
    train_shape_: Option<(usize, usize)>,
    /// Host copy of the continuous regression targets (length `n_train`),
    /// gathered per neighbor for the mean. `None` until `fit`.
    y_reg_: Option<Vec<F>>,
}

impl<F> KNeighborsRegressor<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `KNeighborsRegressor` with neighbor count `n_neighbors`.
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            x_train_: None,
            train_shape_: None,
            y_reg_: None,
        }
    }

    /// The configured neighbor count.
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }
}

impl<F> Fit<F> for KNeighborsRegressor<F>
where
    F: Float + CubeElement + Pod,
{
    /// Store the training matrix `x` and its `F` regression targets `y`. Geometry
    /// is validated before any state is stored (ASVS V5).
    fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (n_train, n_features) = shape;
        if n_train == 0 || n_features == 0 || x.len() != n_train * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_train,
                cols: n_features,
                len: x.len(),
            }));
        }
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "knn_regressor",
            operation: "fit (requires y)",
        })?;
        if y.len() != n_train {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_train,
                cols: 1,
                len: y.len(),
            }));
        }

        let y_reg = y.to_host(pool);
        let x_host = x.to_host(pool);
        let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_host);
        if let Some(old) = self.x_train_.take() {
            old.release_into(pool);
        }
        self.x_train_ = Some(x_dev);
        self.train_shape_ = Some((n_train, n_features));
        self.y_reg_ = Some(y_reg);
        Ok(self)
    }
}

impl<F> Predict<F> for KNeighborsRegressor<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _) = shape;
        let y_reg = self.y_reg_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "knn_regressor",
            operation: "predict",
        })?;

        // Reuse the validated NearestNeighbors core: validates 1<=k<=n_train +
        // query geometry before launch, returns the host u32 neighbor indices.
        let (val_dev, idx_dev, idx_host) = neighbor_indices::<F>(
            pool,
            self.x_train_.as_ref(),
            self.train_shape_,
            x,
            shape,
            self.n_neighbors,
        )?;
        val_dev.release_into(pool);
        idx_dev.release_into(pool);

        // Mean of the k neighbor targets per query (uniform weights — 1/k each).
        let k = self.n_neighbors;
        let inv_k = 1.0f64 / k as f64;
        let mut pred: Vec<F> = vec![F::from_int(0i64); n_query];
        for q in 0..n_query {
            let mut acc = 0.0f64;
            for j in 0..k {
                let train_idx = idx_host[q * k + j] as usize;
                acc += host_to_f64(y_reg[train_idx]);
            }
            pred[q] = f64_to_host::<F>(acc * inv_k);
        }

        Ok(DeviceArray::from_host(pool, &pred))
    }
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("knn regressor is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("knn regressor is f32/f64 only"),
    }
}
