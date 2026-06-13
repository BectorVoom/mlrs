//! `KNeighborsClassifier` (NEIGH-02) — brute-force k-NN majority-vote classifier,
//! matching `sklearn.neighbors.KNeighborsClassifier(algorithm='brute',
//! metric='euclidean', weights='uniform')`.
//!
//! ## Predict = argmax of the per-class neighbor fraction (D-07)
//! `predict_proba` finds the `k` nearest neighbors of each query (reusing the
//! validated `NearestNeighbors` core, [`neighbor_indices`]), gathers their
//! integer class labels, and forms the per-class FRACTION (uniform weights — each
//! neighbor contributes `1/k`). `predict_labels` is the argmax of that proba row
//! with the LOWEST-CLASS-INDEX tie-break (the `argmax_rows` convention, 02), so a
//! tie between two equally-voted classes resolves to the lower class id — exactly
//! sklearn's `mode`/`argmax` behavior over the contiguous `[0, n_classes)` label
//! space.
//!
//! ## Contiguous class space (sklearn `classes_`)
//! The fixture targets are the contiguous integer range `[0, n_classes)` (sklearn
//! re-labels to `classes_` indices). v1 stores the raw integer targets and uses
//! `max + 1` as `n_classes`; the per-class fraction columns are therefore indexed
//! directly by class id (D-07).
//!
//! ## weights='uniform' only (CONTEXT Deferred)
//! Each of the `k` neighbors contributes an equal `1/k` vote; `weights='distance'`
//! is deferred per CONTEXT.
//!
//! ## Validate-before-launch (T-05-08-01 / ASVS V5)
//! Both `predict_labels` and `predict_proba` reject `k` outside `1 ..= n_train`
//! ([`AlgoError::InvalidK`]) and a mismatched query geometry BEFORE any prim
//! launch (the shared `neighbor_indices` core enforces this, 05-08 Task 1).
//!
//! Tests live in `crates/mlrs-algos/tests/knn_classifier_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::reduce::argmax_rows;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::neighbors::nearest::neighbor_indices;
use crate::traits::{Fit, PredictLabels, PredictProba};

/// Brute-force k-NN majority-vote classifier (NEIGH-02).
///
/// Construct with [`KNeighborsClassifier::new`] (`n_neighbors`), then
/// [`Fit::fit`] (stores the training matrix + its i32 class targets) and
/// [`PredictLabels::predict_labels`] / [`PredictProba::predict_proba`]. Fitted
/// state is device-resident (D-03).
pub struct KNeighborsClassifier<F> {
    /// Neighbor count `k` (the vote pool size). Validated against `n_train` at
    /// predict time ([`AlgoError::InvalidK`]).
    n_neighbors: usize,
    /// Device-resident training matrix (`n_train × n_features`, row-major),
    /// `None` until `fit`.
    x_train_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted training geometry `(n_train, n_features)`, `None` until `fit`.
    train_shape_: Option<(usize, usize)>,
    /// Host copy of the integer class targets (length `n_train`), gathered per
    /// neighbor during the vote. `None` until `fit`.
    y_class_: Option<Vec<i32>>,
    /// Number of distinct classes `= max(y_class) + 1` (contiguous `[0, n)`).
    n_classes_: usize,
}

impl<F> KNeighborsClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `KNeighborsClassifier` with neighbor count
    /// `n_neighbors`.
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            x_train_: None,
            train_shape_: None,
            y_class_: None,
            n_classes_: 0,
        }
    }

    /// The configured neighbor count.
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// The number of distinct classes inferred at `fit` (`max(y) + 1`). Errors
    /// with [`AlgoError::NotFitted`] before `fit`.
    pub fn n_classes(&self) -> Result<usize, AlgoError> {
        if self.y_class_.is_some() {
            Ok(self.n_classes_)
        } else {
            Err(AlgoError::NotFitted {
                estimator: "knn_classifier",
                operation: "n_classes",
            })
        }
    }
}

impl<F> Fit<F> for KNeighborsClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    /// Store the training matrix `x` and its integer class targets `y` (passed as
    /// `F`-typed device values that are integer-valued; gathered to host i32).
    /// Geometry is validated before any state is stored (ASVS V5).
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
            estimator: "knn_classifier",
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

        // Gather the integer class labels host-side (they are integer-valued `F`
        // in the fixture). n_classes is max + 1 over the contiguous label space.
        let y_host = y.to_host(pool);
        let y_class: Vec<i32> = y_host
            .iter()
            .map(|&v| host_to_f64(v).round() as i32)
            .collect();
        let n_classes = y_class.iter().copied().max().unwrap_or(-1) + 1;
        if n_classes <= 0 {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_train,
                cols: 1,
                len: y.len(),
            }));
        }

        let x_host = x.to_host(pool);
        let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_host);
        if let Some(old) = self.x_train_.take() {
            old.release_into(pool);
        }
        self.x_train_ = Some(x_dev);
        self.train_shape_ = Some((n_train, n_features));
        self.y_class_ = Some(y_class);
        self.n_classes_ = n_classes as usize;
        Ok(self)
    }
}

impl<F> PredictProba<F> for KNeighborsClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _) = shape;
        let y_class = self.y_class_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "knn_classifier",
            operation: "predict_proba",
        })?;
        let n_classes = self.n_classes_;

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

        // Per-class neighbor FRACTION (uniform weights — each neighbor = 1/k).
        let k = self.n_neighbors;
        let inv_k = 1.0f64 / k as f64;
        let mut proba: Vec<F> = vec![F::from_int(0i64); n_query * n_classes];
        for q in 0..n_query {
            for j in 0..k {
                let train_idx = idx_host[q * k + j] as usize;
                // WR-02: a corrupted/oversized neighbor index from top_k (or a
                // k/n_train mismatch slipping past validation) must be a typed
                // error at the gather site, NOT an unchecked panic (debug) or a
                // silent wrong read (release).
                if train_idx >= y_class.len() {
                    return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                        operand: "knn.train_idx",
                        rows: train_idx,
                        cols: 1,
                        len: y_class.len(),
                    }));
                }
                let class = y_class[train_idx];
                // WR-02: an out-of-range class id (test labels exceeding train
                // max+1, or a negative id) must not write out of the proba row.
                if class < 0 || (class as usize) >= n_classes {
                    return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                        operand: "knn.class_id",
                        rows: class.max(0) as usize,
                        cols: 1,
                        len: n_classes,
                    }));
                }
                let slot = q * n_classes + class as usize;
                let cur = host_to_f64(proba[slot]) + inv_k;
                proba[slot] = f64_to_host::<F>(cur);
            }
        }

        Ok(DeviceArray::from_host(pool, &proba))
    }
}

impl<F> PredictLabels<F> for KNeighborsClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let (n_query, _) = shape;
        let n_classes = self.n_classes_;

        // predict = argmax(proba) with the lowest-class-index tie-break. Build the
        // proba matrix (which itself validates k + geometry), then argmax each row.
        let proba = self.predict_proba(pool, x, shape)?;

        // argmax_rows applies the lowest-index tie-break (02 convention) — over the
        // contiguous [0, n_classes) label space the column index IS the class id.
        let labels_u32 = argmax_rows::<F>(pool, &proba, n_query, n_classes)?;
        proba.release_into(pool);

        let labels_i32: Vec<i32> = labels_u32.iter().map(|&u| u as i32).collect();
        Ok(DeviceArray::from_host(pool, &labels_i32))
    }
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("knn classifier is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("knn classifier is f32/f64 only"),
    }
}
