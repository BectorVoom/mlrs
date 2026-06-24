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

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::neighbors::nearest::neighbor_indices;
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// sklearn `KNeighborsRegressor` default neighbor count.
const KNN_REG_DEFAULT_N_NEIGHBORS: usize = 5;

/// Brute-force k-NN mean regressor (NEIGH-03).
///
/// Construct with the zero-arg [`KNeighborsRegressor::new`] (sklearn default
/// `n_neighbors = 5`) or [`KNeighborsRegressor::builder`], then the consuming
/// [`Fit::fit`] (stores the training matrix + its `F` regression targets) and
/// [`Predict::predict`] (mean of the k neighbor targets), which exists ONLY on
/// `KNeighborsRegressor<F, Fitted>` (the compile-time typestate replaces the old
/// runtime `NotFitted` guard, D-03). Fitted state is device-resident (D-03).
pub struct KNeighborsRegressor<F, S = Unfit> {
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
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> KNeighborsRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfit `KNeighborsRegressor` with sklearn's default
    /// `n_neighbors = 5`. This is the SINGLE source of truth for the default
    /// hyperparameter (D-08): the builder `Default` re-derives from here via
    /// [`KNeighborsRegressor::into_builder`], rather than re-listing the literal.
    pub fn new() -> Self {
        Self {
            n_neighbors: KNN_REG_DEFAULT_N_NEIGHBORS,
            x_train_: None,
            train_shape_: None,
            y_reg_: None,
            _state: PhantomData,
        }
    }

    /// Start building a `KNeighborsRegressor` from sklearn's defaults (D-08
    /// single source).
    pub fn builder() -> KNeighborsRegressorBuilder {
        KNeighborsRegressorBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying the
    /// hyperparameter. Used by [`KNeighborsRegressorBuilder::default`] to
    /// re-derive the defaults from [`KNeighborsRegressor::new`] (D-08).
    pub fn into_builder(self) -> KNeighborsRegressorBuilder {
        KNeighborsRegressorBuilder {
            n_neighbors: self.n_neighbors,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators. Used by the
    /// defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_neighbors == other.n_neighbors
    }

    /// The configured neighbor count (read pre-fit).
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }
}

impl<F> Default for KNeighborsRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> KNeighborsRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The configured neighbor count.
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }
}

/// Builder for [`KNeighborsRegressor`] (D-01). `Default` re-derives the sklearn
/// default from [`KNeighborsRegressor::new`] (D-08 single source) rather than
/// holding a literal (Pitfall 1).
#[derive(Debug, Clone, Copy)]
pub struct KNeighborsRegressorBuilder {
    n_neighbors: usize,
}

impl Default for KNeighborsRegressorBuilder {
    /// Re-derive the sklearn default from [`KNeighborsRegressor::new`] (D-08
    /// single source).
    fn default() -> Self {
        KNeighborsRegressor::<f64, Unfit>::new().into_builder()
    }
}

impl KNeighborsRegressorBuilder {
    /// Set the neighbor count `n_neighbors`.
    pub fn n_neighbors(mut self, v: usize) -> Self {
        self.n_neighbors = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameter BEFORE any data is seen (D-08; the data-DEPENDENT
    /// `k <= n_train` check lives in the `kneighbors` core):
    ///
    /// - `n_neighbors >= 1` ([`BuildError::InvalidNComponents`]). The
    ///   data-DEPENDENT `k > n_train` half stays in the predict path (T-16-V5).
    pub fn build<F>(self) -> Result<KNeighborsRegressor<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.n_neighbors == 0 {
            return Err(BuildError::InvalidNComponents {
                estimator: "knn_regressor",
                param: "n_neighbors",
                value: self.n_neighbors,
            });
        }
        Ok(KNeighborsRegressor {
            n_neighbors: self.n_neighbors,
            x_train_: None,
            train_shape_: None,
            y_reg_: None,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for KNeighborsRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = KNeighborsRegressor<F, Fitted>;

    /// Store the training matrix `x` and its `F` regression targets `y`,
    /// CONSUMING `self` and returning the `Fitted`-tagged sibling. Geometry is
    /// validated before any state is stored (ASVS V5).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<KNeighborsRegressor<F, Fitted>, AlgoError> {
        let (n_train, n_features) = shape;
        validate_geometry(x, shape)?;
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
        Ok(KNeighborsRegressor {
            n_neighbors: self.n_neighbors,
            x_train_: Some(x_dev),
            train_shape_: Some((n_train, n_features)),
            y_reg_: Some(y_reg),
            _state: PhantomData,
        })
    }
}

impl<F> Predict<F> for KNeighborsRegressor<F, Fitted>
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
        // `y_reg_` is `Some` by construction on `KNeighborsRegressor<F, Fitted>`
        // (the compile-time typestate replaces the old runtime `NotFitted` guard,
        // D-03).
        let y_reg = self
            .y_reg_
            .as_ref()
            .expect("y_reg_ is Some by construction on KNeighborsRegressor<F, Fitted>");

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
                // WR-02: a corrupted/oversized neighbor index from top_k must be a
                // typed error at the gather site, not an unchecked panic (debug) or
                // a silent wrong read (release).
                if train_idx >= y_reg.len() {
                    return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                        operand: "knn.train_idx",
                        rows: train_idx,
                        cols: 1,
                        len: y_reg.len(),
                    }));
                }
                acc += host_to_f64(y_reg[train_idx]);
            }
            pred[q] = f64_to_host::<F>(acc * inv_k);
        }

        Ok(DeviceArray::from_host(pool, &pred))
    }
}
