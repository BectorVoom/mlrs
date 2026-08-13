//! `KNeighborsClassifier` (NEIGH-02) — brute-force k-NN vote classifier matching
//! the FULL `sklearn.neighbors.KNeighborsClassifier` hyperparameter surface
//! (KNN-CLF-PARAMS): `n_neighbors`, `weights`, `metric` (+ the Minkowski `p`).
//!
//! ## Predict = argmax of the per-class weight share (D-07)
//! `predict_proba` finds the `k` nearest neighbors of each query under the
//! configured [`Metric`] (reusing the validated `NearestNeighbors` core,
//! [`neighbor_indices_metric`]), gathers their integer class labels, and sums each
//! neighbor's WEIGHT into its class — `1` each under [`Weights::Uniform`], `1/d_j`
//! under [`Weights::Distance`] — then divides the row by its total. That is
//! sklearn's `predict_proba` verbatim, including the normalization order (sum
//! then divide, never `1/k` per neighbor, so the two agree bit for bit where the
//! float arithmetic would otherwise differ).
//!
//! `predict_labels` is the argmax of that proba row with the LOWEST-CLASS-INDEX
//! tie-break, which is what sklearn's `_mode` / `weighted_mode` both do (scipy's
//! `mode` returns the smallest of the tied values; `weighted_mode` argmaxes over
//! the sorted class axis).
//!
//! ## What is NOT a hyperparameter here
//! sklearn's `algorithm`, `leaf_size`, and `n_jobs` select an INDEX STRUCTURE and
//! a thread count. mlrs is brute-force on a device, so each would be a no-op field
//! that `get_params` reports and nothing reads; they are accepted and validated at
//! the PYTHON shim (where sklearn's `get_params`/`clone` round-trip contract
//! lives) and stop there. `weights=<callable>` and `metric=<callable>` stop at the
//! shim for the mirror-image reason: an arbitrary Python function cannot cross
//! into a device kernel, so the shim serves them from `kneighbors` output.
//!
//! ## Class space (sklearn `classes_`)
//! `fit` collects the DISTINCT sorted training labels as `classes_` and remaps
//! each sample to its DENSE class index (its position in `classes_`). The
//! per-class columns are indexed by this dense position, and `predict_labels` maps
//! the argmax column back through `classes_` to recover the original id (CR-03) —
//! so a NON-contiguous target (e.g. `{0, 2}`) returns the original `2`, never a
//! phantom never-trained class (D-07).
//!
//! ## The vote is formed on the HOST, deliberately (KNN-CLF-PARAMS)
//! The neighbor SEARCH is `O(n_query · n_train · d)` and runs through the tuned
//! device/host kernel family; the vote that follows is `O(n_query · k)` — three to
//! five orders of magnitude smaller at every rung this repo benchmarks. Forming it
//! on the host is not a fallback, it is what removes the two round-trips the
//! device spelling required: an upload of the `n_query × n_classes` proba matrix
//! and `reduce::argmax_rows`, which synchronizes ONCE PER ROW (~100 µs/row — it
//! dominated `predict` outright at every query count past a few hundred). The
//! neighbor indices are already host-resident, because
//! [`neighbor_indices_metric`] reads them back to build its public `i32` surface.
//!
//! ## Validate-before-launch (T-05-08-01 / ASVS V5)
//! Both `predict_labels` and `predict_proba` reject `k` outside `1 ..= n_train`
//! ([`AlgoError::InvalidK`]) and a mismatched query geometry BEFORE any prim
//! launch (the shared `neighbor_indices_metric` core enforces this, 05-08 Task 1);
//! the builder rejects `n_neighbors == 0` and a Minkowski `p < 1` before any data
//! is seen (D-08).
//!
//! Tests live in `crates/mlrs-algos/tests/knn_classifier_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::knn::device_copy;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::neighbors::nearest::neighbor_indices_metric;
use crate::neighbors::neighbors_persist::{
    as_i64, expect_k_fits, read_device, read_fit_x, read_metric, read_weights, shape_1d,
    write_device, write_fit_x, write_metric, write_weights, AlignedBytes, LoadModel, NeighborsFile,
    NeighborsWriter, PersistError, SaveModel, TensorRef, CLASSES_NAME, Y_NAME,
};
use crate::neighbors::{Metric, Weights};
use crate::typestate::{
    validate_geometry, Fit, Fitted, KNeighbors, PredictLabels, PredictProba, Unfit,
};

/// sklearn `KNeighborsClassifier` default neighbor count.
const KNN_CLF_DEFAULT_N_NEIGHBORS: usize = 5;

/// sklearn `KNeighborsClassifier` default weighting.
const KNN_CLF_DEFAULT_WEIGHTS: Weights = Weights::Uniform;

/// sklearn `KNeighborsClassifier` default metric — `metric='minkowski', p=2`,
/// which IS Euclidean and is resolved to the dedicated Euclidean kernels rather
/// than to `minkowski_dist` with `p = 2` (see [`Metric`]).
const KNN_CLF_DEFAULT_METRIC: Metric = Metric::Euclidean;

/// Brute-force k-NN vote classifier (NEIGH-02).
///
/// Construct with the zero-arg [`KNeighborsClassifier::new`] (sklearn defaults:
/// `n_neighbors = 5`, `weights = 'uniform'`, `metric = 'minkowski', p = 2`) or
/// [`KNeighborsClassifier::builder`], then the consuming [`Fit::fit`] (stores the
/// training matrix + its i32 class targets) and
/// [`PredictLabels::predict_labels`] / [`PredictProba::predict_proba`], which
/// exist ONLY on `KNeighborsClassifier<F, Fitted>` (the compile-time typestate
/// replaces the old runtime `NotFitted` guard, D-03). Fitted state is
/// device-resident (D-03).
pub struct KNeighborsClassifier<F, S = Unfit> {
    /// Neighbor count `k` (the vote pool size). Validated against `n_train` at
    /// predict time ([`AlgoError::InvalidK`]).
    n_neighbors: usize,
    /// How each neighbor's vote is weighted into the per-class total.
    weights: Weights,
    /// The distance the neighbor search runs under. Carries the Minkowski
    /// exponent in its own payload — there is no separate `p` field.
    metric: Metric,
    /// Where to run the neighbour search (DEVICE-PARAM-01). `Auto` keeps the
    /// existing gate; `Cpu`/`Gpu` override its PERF half only — the capability
    /// half (`k <= n_train`, non-empty operands) is never overridable.
    device: Device,
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
    /// Construct an unfit `KNeighborsClassifier` with sklearn's defaults. This is
    /// the SINGLE source of truth for them (D-08): the builder `Default`
    /// re-derives from here via [`KNeighborsClassifier::into_builder`], rather
    /// than re-listing the literals.
    pub fn new() -> Self {
        Self {
            n_neighbors: KNN_CLF_DEFAULT_N_NEIGHBORS,
            weights: KNN_CLF_DEFAULT_WEIGHTS,
            metric: KNN_CLF_DEFAULT_METRIC,
            device: Device::Auto,
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

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`KNeighborsClassifierBuilder::default`] to
    /// re-derive the defaults from [`KNeighborsClassifier::new`] (D-08).
    pub fn into_builder(self) -> KNeighborsClassifierBuilder {
        KNeighborsClassifierBuilder {
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            metric: self.metric,
            device: self.device,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators. Used by the
    /// defaults-equality test (BLDR-01). Covers EVERY hyperparameter, so a new
    /// one added to the struct without being threaded through
    /// [`KNeighborsClassifier::into_builder`] fails that test rather than
    /// silently defaulting.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_neighbors == other.n_neighbors
            && self.weights == other.weights
            && self.metric == other.metric
            && self.device == other.device
    }

    /// The configured neighbor count (read pre-fit).
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// The configured weighting (read pre-fit).
    pub fn weights(&self) -> Weights {
        self.weights
    }

    /// The configured metric (read pre-fit).
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Fit from an already-device-resident training matrix taken BY VALUE and
    /// ALREADY-PREPARED host labels — the zero-copy sibling of [`Fit::fit`]
    /// (KNN-REG-FIT).
    ///
    /// Two separate wins over the borrowing form, both specific to this
    /// estimator:
    ///
    /// * `x` is adopted rather than duplicated (see
    ///   [`KNeighborsRegressor::fit_owned`](crate::neighbors::regressor::KNeighborsRegressor::fit_owned)
    ///   for why the borrowing form must copy);
    /// * `y` never touches the device AT ALL. The classifier's vote gather
    ///   works on host `i32` class indices, so [`Fit::fit`]'s `DeviceArray` `y`
    ///   is uploaded by the caller and immediately read back by
    ///   `y.to_host(pool)` — a full device round-trip whose only product is a
    ///   host `Vec` the caller already had. Taking [`PreparedLabels`] skips
    ///   both legs.
    ///
    /// On validation failure `x` is released back into `pool` — the caller has
    /// already given it up, so nothing else can.
    pub fn fit_owned(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: DeviceArray<ActiveRuntime, F>,
        labels: PreparedLabels,
        shape: (usize, usize),
    ) -> Result<KNeighborsClassifier<F, Fitted>, AlgoError> {
        let (n_train, _) = shape;
        if let Err(e) = validate_geometry(&x, shape) {
            x.release_into(pool);
            return Err(e);
        }
        if labels.y_class.len() != n_train {
            x.release_into(pool);
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_train,
                cols: 1,
                len: labels.y_class.len(),
            }));
        }
        Ok(KNeighborsClassifier {
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            metric: self.metric,
            device: self.device,
            x_train_: Some(x),
            train_shape_: Some(shape),
            n_classes_: labels.classes.len(),
            classes_: labels.classes,
            y_class_: Some(labels.y_class),
            _state: PhantomData,
        })
    }
}

/// Validated training labels, remapped to DENSE class indices (CR-03).
///
/// The product of [`prepare_labels`], and the whole of what the classifier
/// needs from `y`. Split out from `fit` so a caller can derive it from a host
/// slice — it is `O(n_train)` work over a label vector, not over the
/// `n_train x n_features` matrix, so it is cheap enough to do eagerly even when
/// the matrix upload is deferred (`crates/mlrs-py`'s wrapper does exactly that,
/// because sklearn's `classes_` must be readable the instant `fit` returns).
#[derive(Clone, Default)]
pub struct PreparedLabels {
    /// The DISTINCT sorted raw labels — sklearn's `classes_`.
    classes: Vec<i32>,
    /// Per training sample, the POSITION of its raw label in `classes`.
    y_class: Vec<i32>,
}

impl PreparedLabels {
    /// The DISTINCT sorted training labels (`classes_`).
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    /// The number of distinct classes.
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

/// Validate `y_host` and remap it to dense class indices (CR-03 / WR-02).
///
/// `y` arrives as integer-valued `F` (the shared float ingress carries labels
/// too). Every value must be finite, integer-valued and in `i32` range:
/// without that check a NaN target silently becomes `0` under the saturating
/// cast and an out-of-range label saturates, either way producing a wrong class
/// with no error.
///
/// `classes` is the DISTINCT SORTED set of raw labels and each sample is
/// remapped to its POSITION in it, rather than inferring `n_classes = max + 1`.
/// A `max+1` width over a non-contiguous target (e.g. `{0, 2}`) creates a
/// structurally-zero column 1 that argmax can still pick, returning a class id
/// that never existed in training; sklearn maps votes through `classes_` and
/// returns the original id. The `class >= n_classes` guard at predict cannot
/// catch that GAP, so the dense remap here plus the inverse map at predict is
/// the fix.
pub fn prepare_labels<F>(y_host: &[F], n_train: usize) -> Result<PreparedLabels, AlgoError>
where
    F: Pod,
{
    if y_host.len() != n_train {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "y",
            rows: n_train,
            cols: 1,
            len: y_host.len(),
        }));
    }
    let mut raw_class: Vec<i32> = Vec::with_capacity(n_train);
    for &v in y_host.iter() {
        let lf = host_to_f64(v);
        let lr = lf.round();
        if !lr.is_finite() || (lr - lf).abs() > 1e-6 || i32::try_from(lr as i64).is_err() {
            return Err(AlgoError::InvalidLabels {
                estimator: "knn_classifier",
                reason: format!("labels must be i32-range integers (got {lf})"),
            });
        }
        raw_class.push(lr as i32);
    }
    if raw_class.is_empty() {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "y",
            rows: n_train,
            cols: 1,
            len: y_host.len(),
        }));
    }

    // --- ALREADY-DENSE fast path (KNN-CLF-PARAMS) ---
    //
    // When the labels are exactly `{0, 1, ..., max}` with none missing, the
    // sorted class vector IS `0..=max` and each sample's position in it is its
    // own label — so the sort, the extra `Vec` the clone allocates, and the
    // per-sample binary search below are all recoverable identity work.
    //
    // This is not a hypothetical case: the Python shim reproduces sklearn's
    // `np.unique(..., return_inverse=True)` encoding itself (it must, so a
    // string or boolean target can reach a float-only ingress at all) and
    // therefore hands this function dense codes on EVERY `fit` through the
    // bindings. Measured at the 50 000-row rung it is what closes the gap
    // against `sklearn.neighbors.KNeighborsClassifier.fit`.
    //
    // The `max < n_train` guard bounds the presence bitmap at `O(n_train)`, so
    // a sparse-but-huge label space (e.g. `{0, 1_000_000}`) takes the general
    // path rather than allocating a megabyte to discover it does not qualify.
    if let Some(dense) = dense_classes(&raw_class, n_train) {
        return Ok(PreparedLabels {
            classes: dense,
            y_class: raw_class,
        });
    }

    let mut classes: Vec<i32> = raw_class.clone();
    classes.sort_unstable();
    classes.dedup();
    // Dense class index per training sample = position of its raw label in the
    // sorted `classes` (binary search; `classes` is sorted + deduped).
    let y_class: Vec<i32> = raw_class
        .iter()
        .map(|&l| {
            classes
                .binary_search(&l)
                .expect("every raw label is in classes by construction") as i32
        })
        .collect();
    Ok(PreparedLabels { classes, y_class })
}

/// `Some(0..=max)` when `raw_class` is exactly the contiguous label set
/// `{0, 1, ..., max}` — the case where the dense remap is the identity.
///
/// Returns `None` (so the caller takes the general sort/dedup path) if any label
/// is negative, if `max >= n_train` (the bitmap would not be `O(n_train)`), or if
/// any value in `0..=max` is unused. That last condition is the one that matters
/// for correctness: a GAPPED set such as `{0, 2}` has `max = 2` but sorts to
/// `[0, 2]`, so label `2` sits at position 1 and the identity mapping would be
/// wrong — it is exactly the non-contiguous case CR-03 exists for.
fn dense_classes(raw_class: &[i32], n_train: usize) -> Option<Vec<i32>> {
    let max = *raw_class.iter().max()?;
    if max < 0 || max as usize >= n_train {
        return None;
    }
    let width = max as usize + 1;
    let mut seen = vec![false; width];
    for &l in raw_class {
        if l < 0 {
            return None;
        }
        seen[l as usize] = true;
    }
    if seen.iter().any(|&s| !s) {
        return None;
    }
    Some((0..=max).collect())
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

    /// The configured weighting.
    pub fn weights(&self) -> Weights {
        self.weights
    }

    /// The configured metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The fitted training geometry `(n_train, n_features)` (sklearn's
    /// `n_samples_fit_` and `n_features_in_`). `Some` by construction on the
    /// `Fitted` state (D-03).
    pub fn train_shape(&self) -> (usize, usize) {
        self.train_shape_
            .expect("train_shape_ is Some by construction on KNeighborsClassifier<F, Fitted>")
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
/// defaults from [`KNeighborsClassifier::new`] (D-08 single source) rather than
/// holding literals (Pitfall 1).
#[derive(Debug, Clone, Copy)]
pub struct KNeighborsClassifierBuilder {
    n_neighbors: usize,
    weights: Weights,
    metric: Metric,
    device: Device,
}

impl Default for KNeighborsClassifierBuilder {
    /// Re-derive the sklearn defaults from [`KNeighborsClassifier::new`] (D-08
    /// single source). `f64` is pinned only to read the F-independent scalar
    /// defaults — the builder is non-generic, so the choice of `F` here is
    /// irrelevant.
    fn default() -> Self {
        KNeighborsClassifier::<f64, Unfit>::new().into_builder()
    }
}

impl KNeighborsClassifierBuilder {
    /// Pin the execution arm of the neighbour search (DEVICE-PARAM-01).
    /// [`Device::Auto`] keeps the existing heuristic.
    pub fn device(mut self, v: Device) -> Self {
        self.device = v;
        self
    }

    /// Set the neighbor count `n_neighbors`.
    pub fn n_neighbors(mut self, v: usize) -> Self {
        self.n_neighbors = v;
        self
    }

    /// Set the neighbor weighting (sklearn `weights=`).
    pub fn weights(mut self, v: Weights) -> Self {
        self.weights = v;
        self
    }

    /// Set the distance metric. The Minkowski exponent rides inside
    /// `Metric::Minkowski { p }` — there is no separate `p` setter, so the two
    /// cannot be set inconsistently.
    pub fn metric(mut self, v: Metric) -> Self {
        self.metric = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08; the data-DEPENDENT
    /// `k <= n_train` check lives in the `kneighbors` core):
    ///
    /// - `n_neighbors >= 1` ([`BuildError::InvalidNNeighbors`]). The
    ///   data-DEPENDENT `k > n_train` half stays in the predict path (T-16-V5).
    /// - `Metric::Minkowski { p }` requires `p >= 1`
    ///   ([`BuildError::InvalidMinkowskiP`], the hdbscan/knn_graph precedent).
    ///   `p < 1` is not a metric at all (the triangle inequality fails), and the
    ///   kernel's `F::powf(acc, 1/p)` would silently produce a finite but
    ///   meaningless ordering rather than failing.
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
        if let Metric::Minkowski { p } = self.metric {
            // `!(p >= 1.0)` rather than `p < 1.0`: a NaN exponent fails EVERY
            // ordered comparison, so the `<` spelling would wave it through and
            // the kernel would emit NaN distances that `top_k` orders
            // arbitrarily.
            if !(p >= 1.0) {
                return Err(BuildError::InvalidMinkowskiP {
                    estimator: "knn_classifier",
                    p,
                });
            }
        }
        Ok(KNeighborsClassifier {
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            metric: self.metric,
            device: self.device,
            x_train_: None,
            train_shape_: None,
            y_class_: None,
            classes_: Vec::new(),
            n_classes_: 0,
            _state: PhantomData,
        })
    }
}

/// The `estimator` discriminator written into every `KNeighborsClassifier` file.
/// See [`regressor`](super::regressor)'s tag for why it is load-bearing.
const PERSIST_TAG: &str = "kneighbors_classifier";

impl<F> SaveModel for KNeighborsClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted classifier to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `_fit_X` | `F` (`F32`/`F64`) | `[n_samples, n_features]` |
    /// | `_y` | `I64` | `[n_samples]` — the ENCODED class ids |
    /// | `classes_` | `I64` | `[n_classes]` — the decode table |
    /// | `param:n_neighbors` / `param:weights` / `param:metric` / `param:device` | `__metadata__` scalar | — |
    /// | `param:p` | `__metadata__` scalar, Minkowski ONLY | — |
    ///
    /// Both label tensors are `I64` rather than the model's float width: these
    /// are label ids, and storing them as floats would make a large label
    /// silently unrepresentable and would invite a reader to compare them with a
    /// tolerance. mlrs holds them as `i32` in memory and widens here, because
    /// `i32` is an internal choice while the file has to survive a model whose
    /// labels do not fit one.
    ///
    /// Storing BOTH is what makes the round-trip faithful. `_y` is the dense
    /// `0..K` encoding the gather kernel indexes by (CR-02) and `classes_` is
    /// what turns a prediction back into the label the caller trained with; a
    /// file with only the former would round-trip its own state perfectly and
    /// predict `{0, 1, 2}` where training said `{0, 2, 7}`.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        let (n_samples, n_features) = self.train_shape_.ok_or_else(|| absent("train_shape_"))?;
        // Bound BEFORE the writer, which borrows every payload. The two label
        // vectors are host-side already; widening them to `i64` is the one copy
        // they cost.
        let x_train = self
            .x_train_
            .as_ref()
            .ok_or_else(|| absent("x_train_"))?
            .to_host(pool);
        let y: Vec<i64> = self
            .y_class_
            .as_ref()
            .ok_or_else(|| absent("y_class_"))?
            .iter()
            .map(|&v| i64::from(v))
            .collect();
        let classes: Vec<i64> = self.classes_.iter().map(|&v| i64::from(v)).collect();

        let mut w = NeighborsWriter::new(PERSIST_TAG);
        w.scalar_usize("param:n_neighbors", self.n_neighbors);
        write_weights(&mut w, self.weights);
        write_metric(&mut w, self.metric);
        write_device(&mut w, self.device);
        write_fit_x(&mut w, &x_train, n_samples, n_features)?;
        w.tensor(Y_NAME, TensorRef::i64s(&y, vec![n_samples])?);
        w.tensor(CLASSES_NAME, TensorRef::i64s(&classes, vec![classes.len()])?);
        w.write(path)
    }
}

impl<F> LoadModel for KNeighborsClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the classifier back from `path`, re-uploading `_fit_X` to `pool`.
    ///
    /// The file is untrusted input (T-04-01-01), and this estimator needs the
    /// strictest checks in the family because its two label tensors index each
    /// other: every entry of `_y` is an index into `classes_`, so a header whose
    /// encoding reaches past the decode table would read out of range the first
    /// time a prediction is decoded. Both the LENGTH of `_y` against `_fit_X`
    /// and the RANGE of its values against `classes_` are validated here.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<KNeighborsClassifier<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = NeighborsFile::parse(&raw, PERSIST_TAG)?;
        let (x_train, n_samples, n_features) = read_fit_x::<F>(&file)?;

        let classes_v = file.tensor(CLASSES_NAME)?;
        let n_classes = shape_1d(&classes_v, CLASSES_NAME)?;
        if n_classes < 2 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{CLASSES_NAME}' holds {n_classes} labels; a fitted \
                     classifier has at least 2"
                ),
            });
        }

        let y_v = file.tensor(Y_NAME)?;
        if shape_1d(&y_v, Y_NAME)? != n_samples {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{Y_NAME}' holds {} entries, but '_fit_X' implies {n_samples}",
                    shape_1d(&y_v, Y_NAME)?
                ),
            });
        }

        // Every `_y` entry is an INDEX into `classes_`, so its range is a
        // cross-tensor invariant, not a formality: an out-of-range id would
        // index the class table out of bounds the first time `predict` decodes
        // a label. Narrowing to `i32` is checked at the same time, since that is
        // the width the kernel consumes.
        let y_class: Vec<i32> = as_i64(&y_v, Y_NAME)?
            .iter()
            .map(|&v| {
                let ok = v >= 0 && (v as u64) < n_classes as u64;
                i32::try_from(v).ok().filter(|_| ok).ok_or_else(|| {
                    PersistError::InconsistentGeometry {
                        reason: format!(
                            "tensor '{Y_NAME}' holds the class id {v}, which is not a \
                             valid index into the {n_classes} labels in '{CLASSES_NAME}'"
                        ),
                    }
                })
            })
            .collect::<Result<_, _>>()?;

        let classes: Vec<i32> = as_i64(&classes_v, CLASSES_NAME)?
            .iter()
            .map(|&v| {
                i32::try_from(v).map_err(|_| PersistError::InconsistentGeometry {
                    reason: format!(
                        "tensor '{CLASSES_NAME}' holds the label {v}, which does not fit \
                         the i32 the classifier's kernels consume"
                    ),
                })
            })
            .collect::<Result<_, _>>()?;

        let n_neighbors = file.scalar_usize("param:n_neighbors")?;
        expect_k_fits(n_neighbors, n_samples)?;

        Ok(KNeighborsClassifier {
            n_neighbors,
            weights: read_weights(&file)?,
            metric: read_metric(&file)?,
            device: read_device(&file)?,
            x_train_: Some(DeviceArray::from_host(pool, &x_train)),
            train_shape_: Some((n_samples, n_features)),
            y_class_: Some(y_class),
            classes_: classes,
            n_classes_: n_classes,
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
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "knn_classifier",
            operation: "fit (requires y)",
        })?;

        // The labels are needed on the HOST (the vote gather works on remapped
        // i32 class indices, not on `F` targets), so this path must read them
        // back — see `fit_owned`, which callers holding the labels host-side
        // already should prefer precisely to avoid this round-trip.
        let y_host = y.to_host(pool);
        let labels = prepare_labels::<F>(&y_host, shape.0)?;

        // Take device-resident ownership of the training matrix with a
        // DEVICE-TO-DEVICE copy rather than round-tripping it through the host
        // (KNN-01) — see `nearest.rs::fit` for the full rationale and why a bare
        // handle clone was NOT ownership.
        let x_dev: DeviceArray<ActiveRuntime, F> = device_copy::<F>(pool, x);
        self.fit_owned(pool, x_dev, labels, shape)
    }
}

impl<F> KNeighborsClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The `n_query × n_classes` row-major class-probability matrix, HOST-side.
    ///
    /// The single implementation of the vote: [`PredictProba::predict_proba`]
    /// uploads what this returns, [`PredictLabels::predict_labels`] argmaxes it,
    /// and the PyO3 wrapper reads it directly (its caller wants host floats, so
    /// routing through a `DeviceArray` would upload a buffer only to download it
    /// again on the very next line).
    ///
    /// The rule is sklearn's `predict_proba`, term for term: each neighbour's
    /// weight is summed into its class column and the row is then divided by its
    /// total. Two details are load-bearing:
    ///
    /// * the normalization is SUM-THEN-DIVIDE, not `1/k` per neighbour. For
    ///   `weights='uniform'` those are the same number in exact arithmetic and
    ///   NOT always the same float, and sklearn does the former;
    /// * a zero distance under `weights='distance'` takes sklearn's indicator
    ///   branch for THAT ROW (`_get_weights`): the coincident neighbours get
    ///   weight 1 and every other neighbour in the row gets 0, because `1/0 = inf`
    ///   would normalize to `inf/inf = NaN`. The device regressor kernel
    ///   implements the identical rule, so the two estimators agree on the
    ///   degenerate case.
    ///
    /// Accumulated in `f64` regardless of `F`: the sums are over at most `k`
    /// terms, so the cost is nil, and it keeps an f32 estimator's proba from
    /// drifting off sklearn's float64 answer by more than the ingress already
    /// costs.
    pub fn predict_proba_host(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<F>, AlgoError> {
        let (n_query, _) = shape;
        // `y_class_` is `Some` by construction on `KNeighborsClassifier<F, Fitted>`
        // (the compile-time typestate replaces the old runtime `NotFitted` guard,
        // D-03).
        let y_class = self
            .y_class_
            .as_ref()
            .expect("y_class_ is Some by construction on KNeighborsClassifier<F, Fitted>");
        let n_classes = self.n_classes_;
        let k = self.n_neighbors;

        // Reuse the validated NearestNeighbors core: validates 1<=k<=n_train +
        // query geometry before launch, returns the host u32 neighbor indices.
        let (val_dev, idx_dev, idx_host) = neighbor_indices_metric::<F>(
            pool,
            self.x_train_.as_ref(),
            self.train_shape_,
            x,
            shape,
            k,
            self.metric,
            self.device,
        )?;
        idx_dev.release_into(pool);
        // The distances are needed ONLY for `1/d` weighting. `val_dev` holds the
        // TRUE metric distances (the Euclidean path applied its boundary sqrt
        // inside `top_k`; the metric kernels never deferred a root), which is what
        // `1/d` requires — weighting by the order-preserving SQUARE would silently
        // produce a different, wrong answer that still looks plausible.
        let dist_host: Vec<F> = match self.weights {
            Weights::Distance => val_dev.to_host(pool),
            Weights::Uniform => Vec::new(),
        };
        val_dev.release_into(pool);

        let mut proba: Vec<F> = vec![F::from_int(0i64); n_query * n_classes];
        let mut acc = vec![0.0f64; n_classes];
        let mut w = vec![0.0f64; k];
        for q in 0..n_query {
            // --- 1. the row's per-neighbour weights (sklearn `_get_weights`) ---
            match self.weights {
                Weights::Uniform => w.iter_mut().for_each(|v| *v = 1.0),
                Weights::Distance => {
                    let row = &dist_host[q * k..q * k + k];
                    let coincident = row.iter().any(|&d| host_to_f64(d) == 0.0);
                    for (j, slot) in w.iter_mut().enumerate() {
                        let d = host_to_f64(row[j]);
                        *slot = if !coincident {
                            1.0 / d
                        } else if d == 0.0 {
                            // The indicator branch: only the zero-distance
                            // neighbours vote at all.
                            1.0
                        } else {
                            0.0
                        };
                    }
                }
            }

            // --- 2. sum each neighbour's weight into its class column ---
            acc.iter_mut().for_each(|v| *v = 0.0);
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
                acc[class as usize] += w[j];
            }

            // --- 3. normalize the row (sklearn leaves an all-zero row alone by
            //        dividing it by 1 rather than by 0) ---
            let total: f64 = acc.iter().sum();
            let denom = if total == 0.0 { 1.0 } else { total };
            let base = q * n_classes;
            for (c, &v) in acc.iter().enumerate() {
                proba[base + c] = f64_to_host::<F>(v / denom);
            }
        }

        Ok(proba)
    }

    /// The `n_query` predicted ORIGINAL class labels, HOST-side — the argmax of
    /// [`KNeighborsClassifier::predict_proba_host`] mapped back through
    /// `classes_`.
    ///
    /// The tie-break is LOWEST DENSE CLASS INDEX (`>` rather than `>=` on the
    /// running maximum), which is what both of sklearn's spellings do: scipy's
    /// `mode` returns the smallest of the tied values, and `weighted_mode`
    /// argmaxes over the sorted class axis. CR-03: the argmax column is the DENSE
    /// index (`0..n_classes`), so it is mapped back through `classes_` to recover
    /// the ORIGINAL training label — a non-contiguous set (e.g. `{0, 2}`) returns
    /// `2`, not the phantom `1`.
    pub fn predict_labels_host(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<i32>, AlgoError> {
        let n_classes = self.n_classes_;
        let proba = self.predict_proba_host(pool, x, shape)?;
        let (n_query, _) = shape;

        let mut labels = Vec::with_capacity(n_query);
        for q in 0..n_query {
            let row = &proba[q * n_classes..q * n_classes + n_classes];
            let mut best = 0usize;
            let mut best_v = host_to_f64(row[0]);
            for (c, &v) in row.iter().enumerate().skip(1) {
                let v = host_to_f64(v);
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            // `best < n_classes == classes_.len()` by construction; the guard is
            // defensive only.
            labels.push(self.classes_.get(best).copied().unwrap_or(best as i32));
        }
        Ok(labels)
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
        let proba = self.predict_proba_host(pool, x, shape)?;
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
        let labels = self.predict_labels_host(pool, x, shape)?;
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> KNeighbors<F> for KNeighborsClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The `k` nearest training points of each query under the CONFIGURED
    /// metric, as `(distances, indices)` — sklearn's `KNeighborsMixin.kneighbors`
    /// on the classifier.
    ///
    /// It is the same neighbour set the vote runs over, by construction: both go
    /// through `neighbor_indices_metric` with the same metric and the same `k`
    /// validation. That identity is what lets the Python shim serve
    /// `weights=<callable>` and multi-output targets — it rebuilds the vote from
    /// these distances and must land on the built-in weightings' answer when
    /// handed their formula.
    fn kneighbors(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
        k: usize,
    ) -> Result<
        (
            DeviceArray<ActiveRuntime, F>,
            DeviceArray<ActiveRuntime, i32>,
        ),
        AlgoError,
    > {
        let (distances, indices, _) = neighbor_indices_metric::<F>(
            pool,
            self.x_train_.as_ref(),
            self.train_shape_,
            x,
            shape,
            k,
            self.metric,
            self.device,
        )?;
        Ok((distances, indices))
    }
}
