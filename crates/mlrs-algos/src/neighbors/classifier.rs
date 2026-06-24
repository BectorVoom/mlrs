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
//! ## Class space (sklearn `classes_`)
//! `fit` collects the DISTINCT sorted training labels as `classes_` and remaps
//! each sample to its DENSE class index (its position in `classes_`). The
//! per-class fraction columns are indexed by this dense position, and
//! `predict_labels` maps the argmax column back through `classes_` to recover the
//! original id (CR-03) — so a NON-contiguous target (e.g. `{0, 2}`) returns the
//! original `2`, never a phantom never-trained class (D-07).
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

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::reduce::argmax_rows;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::neighbors::nearest::neighbor_indices;
use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictProba, Unfit};

/// sklearn `KNeighborsClassifier` default neighbor count.
const KNN_CLF_DEFAULT_N_NEIGHBORS: usize = 5;

/// Brute-force k-NN majority-vote classifier (NEIGH-02).
///
/// Construct with the zero-arg [`KNeighborsClassifier::new`] (sklearn default
/// `n_neighbors = 5`) or [`KNeighborsClassifier::builder`], then the consuming
/// [`Fit::fit`] (stores the training matrix + its i32 class targets) and
/// [`PredictLabels::predict_labels`] / [`PredictProba::predict_proba`], which
/// exist ONLY on `KNeighborsClassifier<F, Fitted>` (the compile-time typestate
/// replaces the old runtime `NotFitted` guard, D-03). Fitted state is
/// device-resident (D-03).
pub struct KNeighborsClassifier<F, S = Unfit> {
    /// Neighbor count `k` (the vote pool size). Validated against `n_train` at
    /// predict time ([`AlgoError::InvalidK`]).
    n_neighbors: usize,
    /// Device-resident training matrix (`n_train × n_features`, row-major),
    /// `None` until `fit`.
    x_train_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted training geometry `(n_train, n_features)`, `None` until `fit`.
    train_shape_: Option<(usize, usize)>,
    /// Host copy of each training sample's DENSE class index (`0..n_classes_`),
    /// gathered per neighbor during the vote. CR-03: this is the POSITION of the
    /// sample's raw label in `classes_`, NOT the raw label, so a non-contiguous
    /// target indexes the proba columns densely. `None` until `fit`.
    y_class_: Option<Vec<i32>>,
    /// CR-03: the DISTINCT sorted training labels (`classes_`), one per proba
    /// column. `predict_labels` maps each argmax column back through this vector
    /// so a non-contiguous set (e.g. `{0, 2}`) returns the ORIGINAL id (`2`),
    /// never a phantom column-1 class that never existed in training. Empty until
    /// `fit`.
    classes_: Vec<i32>,
    /// Number of distinct classes `= classes_.len()`.
    n_classes_: usize,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> KNeighborsClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfit `KNeighborsClassifier` with sklearn's default
    /// `n_neighbors = 5`. This is the SINGLE source of truth for the default
    /// hyperparameter (D-08): the builder `Default` re-derives from here via
    /// [`KNeighborsClassifier::into_builder`], rather than re-listing the literal.
    pub fn new() -> Self {
        Self {
            n_neighbors: KNN_CLF_DEFAULT_N_NEIGHBORS,
            x_train_: None,
            train_shape_: None,
            y_class_: None,
            classes_: Vec::new(),
            n_classes_: 0,
            _state: PhantomData,
        }
    }

    /// Start building a `KNeighborsClassifier` from sklearn's defaults (D-08
    /// single source).
    pub fn builder() -> KNeighborsClassifierBuilder {
        KNeighborsClassifierBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying the
    /// hyperparameter. Used by [`KNeighborsClassifierBuilder::default`] to
    /// re-derive the defaults from [`KNeighborsClassifier::new`] (D-08).
    pub fn into_builder(self) -> KNeighborsClassifierBuilder {
        KNeighborsClassifierBuilder {
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

impl<F> Default for KNeighborsClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> KNeighborsClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The configured neighbor count.
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// The number of distinct classes inferred at `fit`. `Some` by construction
    /// on the `Fitted` state (D-03).
    pub fn n_classes(&self) -> usize {
        self.n_classes_
    }

    /// The DISTINCT sorted training labels (`classes_`, CR-03). `predict_labels`
    /// maps the argmax column back through these, so callers exposing a public
    /// `classes_` attribute MUST use this (not a fabricated `0..n_classes`
    /// range) to honour the sklearn `classes_`/`predict` consistency contract.
    pub fn classes(&self) -> &[i32] {
        &self.classes_
    }
}

/// Builder for [`KNeighborsClassifier`] (D-01). `Default` re-derives the sklearn
/// default from [`KNeighborsClassifier::new`] (D-08 single source) rather than
/// holding a literal (Pitfall 1).
#[derive(Debug, Clone, Copy)]
pub struct KNeighborsClassifierBuilder {
    n_neighbors: usize,
}

impl Default for KNeighborsClassifierBuilder {
    /// Re-derive the sklearn default from [`KNeighborsClassifier::new`] (D-08
    /// single source).
    fn default() -> Self {
        KNeighborsClassifier::<f64, Unfit>::new().into_builder()
    }
}

impl KNeighborsClassifierBuilder {
    /// Set the neighbor count `n_neighbors`.
    pub fn n_neighbors(mut self, v: usize) -> Self {
        self.n_neighbors = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameter BEFORE any data is seen (D-08; the data-DEPENDENT
    /// `k <= n_train` check lives in the `kneighbors` core):
    ///
    /// - `n_neighbors >= 1` ([`BuildError::InvalidNNeighbors`]). The
    ///   data-DEPENDENT `k > n_train` half stays in the predict path (T-16-V5).
    pub fn build<F>(self) -> Result<KNeighborsClassifier<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.n_neighbors == 0 {
            // IN-02: name the neighbor-honest variant so the construction-time
            // error matches the hyperparameter (`n_neighbors`), not `n_components`.
            return Err(BuildError::InvalidNNeighbors {
                estimator: "knn_classifier",
                n_neighbors: self.n_neighbors,
            });
        }
        Ok(KNeighborsClassifier {
            n_neighbors: self.n_neighbors,
            x_train_: None,
            train_shape_: None,
            y_class_: None,
            classes_: Vec::new(),
            n_classes_: 0,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for KNeighborsClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = KNeighborsClassifier<F, Fitted>;

    /// Store the training matrix `x` and its integer class targets `y` (passed as
    /// `F`-typed device values that are integer-valued; gathered to host i32),
    /// CONSUMING `self` and returning the `Fitted`-tagged sibling. Geometry is
    /// validated before any state is stored (ASVS V5).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<KNeighborsClassifier<F, Fitted>, AlgoError> {
        let (n_train, n_features) = shape;
        validate_geometry(x, shape)?;
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

        // Gather the integer class labels host-side (they are integer-valued `F`).
        // CR-03: build `classes_` as the DISTINCT sorted labels and remap each
        // sample to its DENSE class index (its position in `classes_`), rather
        // than inferring `n_classes = max+1`. A `max+1` width over a
        // non-contiguous target (e.g. `{0, 2}`) creates a structurally-zero
        // column 1 that argmax can still pick, returning a class id that never
        // existed in training; sklearn maps votes through `classes_` and returns
        // the original id. The WR-02 `class >= n_classes` guard cannot catch this
        // GAP, so the fix is the dense remap + inverse map at predict.
        let y_host = y.to_host(pool);
        // WR-02: validate each label is a finite, integer-valued, i32-range value
        // before remapping — every sibling classifier guards this. Without it a
        // NaN target silently becomes 0 (saturating cast) and an out-of-i32 label
        // saturates, producing a spurious/wrong class with no error.
        let mut raw_class: Vec<i32> = Vec::with_capacity(n_train);
        for &v in y_host.iter() {
            let lf = host_to_f64(v);
            let lr = lf.round();
            if !lr.is_finite()
                || (lr - lf).abs() > 1e-6
                || i32::try_from(lr as i64).is_err()
            {
                return Err(AlgoError::InvalidLabels {
                    estimator: "knn_classifier",
                    reason: format!("labels must be i32-range integers (got {lf})"),
                });
            }
            raw_class.push(lr as i32);
        }
        let mut classes_: Vec<i32> = raw_class.clone();
        classes_.sort_unstable();
        classes_.dedup();
        if classes_.is_empty() {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_train,
                cols: 1,
                len: y.len(),
            }));
        }
        let n_classes = classes_.len();
        // Dense class index per training sample = position of its raw label in
        // the sorted `classes_` (binary search, classes_ is sorted+deduped).
        let y_class: Vec<i32> = raw_class
            .iter()
            .map(|&l| {
                classes_
                    .binary_search(&l)
                    .expect("every raw label is in classes_ by construction") as i32
            })
            .collect();

        let x_host = x.to_host(pool);
        let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_host);
        Ok(KNeighborsClassifier {
            n_neighbors: self.n_neighbors,
            x_train_: Some(x_dev),
            train_shape_: Some((n_train, n_features)),
            y_class_: Some(y_class),
            classes_,
            n_classes_: n_classes,
            _state: PhantomData,
        })
    }
}

impl<F> PredictProba<F> for KNeighborsClassifier<F, Fitted>
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
        // `y_class_` is `Some` by construction on `KNeighborsClassifier<F, Fitted>`
        // (the compile-time typestate replaces the old runtime `NotFitted` guard,
        // D-03).
        let y_class = self
            .y_class_
            .as_ref()
            .expect("y_class_ is Some by construction on KNeighborsClassifier<F, Fitted>");
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

impl<F> PredictLabels<F> for KNeighborsClassifier<F, Fitted>
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

        // argmax_rows applies the lowest-index tie-break (02 convention). CR-03:
        // the argmax column is the DENSE class index (`0..n_classes`); map it back
        // through `classes_` to recover the ORIGINAL training label, so a
        // non-contiguous set (e.g. `{0, 2}`) returns `2`, not the phantom `1`.
        let labels_u32 = argmax_rows::<F>(pool, &proba, n_query, n_classes)?;
        proba.release_into(pool);

        let labels_i32: Vec<i32> = labels_u32
            .iter()
            .map(|&u| {
                // The dense index is always < n_classes (argmax_rows over the
                // n_classes-wide proba row); guard defensively regardless.
                let col = u as usize;
                if col < self.classes_.len() {
                    self.classes_[col]
                } else {
                    u as i32
                }
            })
            .collect();
        Ok(DeviceArray::from_host(pool, &labels_i32))
    }
}
