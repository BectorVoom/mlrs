//! `KNeighborsRegressor` (NEIGH-03) — brute-force k-NN regressor matching the
//! FULL `sklearn.neighbors.KNeighborsRegressor` hyperparameter surface
//! (KNN-REG-PARAMS): `n_neighbors`, `weights`, `metric` (+ the Minkowski `p`),
//! and multi-output targets.
//!
//! ## Predict = weighted mean of the k neighbor targets (D-07)
//! `predict` finds the `k` nearest neighbors of each query under the configured
//! [`Metric`] (reusing the validated `NearestNeighbors` core,
//! [`neighbor_indices_device_metric`]), gathers their continuous `F` targets,
//! and returns the weighted mean — `1/k` each under [`Weights::Uniform`],
//! `1/d_j` normalized under [`Weights::Distance`]. With an `n_train ×
//! n_outputs` target the same average is formed per output column, so the
//! result is `n_query × n_outputs` row-major.
//!
//! ## What is NOT a hyperparameter here
//! sklearn's `algorithm`, `leaf_size`, and `n_jobs` select an INDEX STRUCTURE
//! and a thread count. mlrs is brute-force on a device: every one of them would
//! be a no-op field that `get_params` reports and nothing reads. They are
//! therefore accepted and validated at the PYTHON shim (where sklearn's contract
//! that they round-trip through `get_params`/`clone` lives) and stop there — the
//! core takes only parameters that change what it computes. `weights=<callable>`
//! stops at the shim for the same reason in reverse: an arbitrary Python
//! function cannot cross into a device kernel, so the shim serves it from
//! `kneighbors` output.
//!
//! ## Validate-before-launch (T-05-08-01 / ASVS V5)
//! `predict` rejects `k` outside `1 ..= n_train` ([`AlgoError::InvalidK`]) and a
//! mismatched query geometry BEFORE any prim launch (the shared
//! `neighbor_indices_device_metric` core enforces this, 05-08 Task 1); the
//! builder rejects `n_neighbors == 0` and a Minkowski `p < 1` before any data is
//! seen (D-08).
//!
//! ## Device residency (D-03 / D-05)
//! BOTH fitted operands are device-resident, and so is the whole predict
//! pipeline: `distance` → `top_k` → the fused neighbor-target gather all run on
//! the device, with no host round-trip between them (KNN-01). The single
//! terminal read-back happens at the caller's API boundary.
//!
//! Tests live in `crates/mlrs-algos/tests/knn_regressor_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::knn::{
    device_copy, knn_regress_gather_weighted, knn_regress_mean_gather,
};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::neighbors::nearest::{neighbor_indices_device_metric, neighbor_indices_metric};
use crate::neighbors::neighbors_persist::{
    expect_k_fits, read_device, read_fit_x, read_metric, read_weights, shape_2d, write_device,
    write_fit_x, write_metric, write_weights, AlignedBytes, LoadModel, NeighborsFile,
    NeighborsWriter, PersistError, SaveModel, TensorRef, Y_NAME,
};
use crate::neighbors::{Metric, Weights};
use crate::typestate::{validate_geometry, Fit, Fitted, KNeighbors, Predict, Unfit};

/// The `estimator` discriminator written into every `KNeighborsRegressor` file.
///
/// Load-bearing rather than decorative: the classifier's file holds a `_y` under
/// the SAME name at the same rank. Only the dtype differs — `I64` encoded class
/// ids against `F` regression targets — which the typed reader would catch, but
/// as a `DtypeMismatch` that reads like corruption rather than as "this is a
/// classifier".
const PERSIST_TAG: &str = "kneighbors_regressor";

/// sklearn `KNeighborsRegressor` default neighbor count.
const KNN_REG_DEFAULT_N_NEIGHBORS: usize = 5;

/// sklearn `KNeighborsRegressor` default weighting.
const KNN_REG_DEFAULT_WEIGHTS: Weights = Weights::Uniform;

/// sklearn `KNeighborsRegressor` default metric — `metric='minkowski', p=2`,
/// which IS Euclidean and is resolved to the dedicated Euclidean kernels rather
/// than to `minkowski_dist` with `p = 2` (see [`Metric`]).
const KNN_REG_DEFAULT_METRIC: Metric = Metric::Euclidean;

/// Brute-force k-NN regressor (NEIGH-03).
///
/// Construct with the zero-arg [`KNeighborsRegressor::new`] (sklearn defaults:
/// `n_neighbors = 5`, `weights = 'uniform'`, `metric = 'minkowski', p = 2`) or
/// [`KNeighborsRegressor::builder`], then the consuming [`Fit::fit`] (stores the
/// training matrix + its `F` regression targets) and [`Predict::predict`], which
/// exists ONLY on `KNeighborsRegressor<F, Fitted>` (the compile-time typestate
/// replaces the old runtime `NotFitted` guard, D-03). Fitted state is
/// device-resident (D-03).
pub struct KNeighborsRegressor<F, S = Unfit> {
    /// Neighbor count `k` (the averaging pool size). Validated against `n_train`
    /// at predict time ([`AlgoError::InvalidK`]).
    n_neighbors: usize,
    /// How each neighbor's target is weighted into the prediction.
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
    /// DEVICE-RESIDENT continuous regression targets, row-major `n_train ×
    /// n_outputs`, gathered per neighbor by the fused device gather (KNN-01 —
    /// previously a host `Vec<F>` driving a host gather loop). `None` until
    /// `fit`.
    y_reg_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Target width, INFERRED at `fit` from `y.len() / n_train` (1 for the
    /// ordinary single-output case). `0` until `fit`.
    n_outputs_: usize,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> KNeighborsRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfit `KNeighborsRegressor` with sklearn's defaults. This is
    /// the SINGLE source of truth for them (D-08): the builder `Default`
    /// re-derives from here via [`KNeighborsRegressor::into_builder`], rather
    /// than re-listing the literals.
    pub fn new() -> Self {
        Self {
            n_neighbors: KNN_REG_DEFAULT_N_NEIGHBORS,
            weights: KNN_REG_DEFAULT_WEIGHTS,
            metric: KNN_REG_DEFAULT_METRIC,
            device: Device::Auto,
            x_train_: None,
            train_shape_: None,
            y_reg_: None,
            n_outputs_: 0,
            _state: PhantomData,
        }
    }

    /// Start building a `KNeighborsRegressor` from sklearn's defaults (D-08
    /// single source).
    pub fn builder() -> KNeighborsRegressorBuilder {
        KNeighborsRegressorBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`KNeighborsRegressorBuilder::default`] to
    /// re-derive the defaults from [`KNeighborsRegressor::new`] (D-08).
    pub fn into_builder(self) -> KNeighborsRegressorBuilder {
        KNeighborsRegressorBuilder {
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            metric: self.metric,
            device: self.device,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators. Used by the
    /// defaults-equality test (BLDR-01). Covers EVERY hyperparameter, so a new
    /// one added to the struct without being threaded through
    /// [`KNeighborsRegressor::into_builder`] fails that test rather than
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

    /// Fit by TAKING OWNERSHIP of already-device-resident operands — the
    /// zero-copy sibling of [`Fit::fit`] (KNN-REG-FIT).
    ///
    /// Same validation, same fitted state; the difference is that `x` and `y`
    /// arrive BY VALUE, so the estimator adopts the caller's buffers instead of
    /// duplicating them.
    ///
    /// ## Why the borrowing `fit` cannot just do this
    /// `Fit::fit` takes `&DeviceArray`, so the only way it can end up owning the
    /// data is to copy it — and it must, because the caller keeps its arrays and
    /// `DeviceArray::release_into` is the repo-wide way to hand a buffer back to
    /// the [`BufferPool`] free list. A fitted estimator merely ALIASING a
    /// caller-owned buffer would see the next `pool.acquire` of that size return
    /// its own live training matrix. The copy is what makes the borrowing form
    /// safe, and it is not removable there.
    ///
    /// It IS removable for a caller that uploaded the operands purely to hand
    /// them over and drops its own handles immediately — which is exactly what
    /// the PyO3 `fit` does. Measured at the 200 000 x 32 f32 rung on the cpu
    /// backend, the two `device_copy` calls the borrowing path makes were ~9 ms
    /// of a 15 ms fit: each is a full pass over the operand PLUS a `cubecl`
    /// launch, and on `cubecl-cpu` a launch spawns one OS thread per unit, so
    /// even the ~800 KB `y` copy cost ~4 ms of thread churn on its own. Moving
    /// ownership costs nothing at all.
    ///
    /// On any validation failure both operands are released back into `pool`
    /// (the caller has already given them up, so nothing else can), keeping the
    /// FOUND-05 `live_bytes` conservation the D-10 memory gate asserts on.
    pub fn fit_owned(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: DeviceArray<ActiveRuntime, F>,
        y: DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<KNeighborsRegressor<F, Fitted>, AlgoError> {
        let n_outputs = match validate_fit_operands::<F>(&x, &y, shape) {
            Ok(n) => n,
            Err(e) => {
                x.release_into(pool);
                y.release_into(pool);
                return Err(e);
            }
        };
        let (n_train, n_features) = shape;
        Ok(KNeighborsRegressor {
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            metric: self.metric,
            device: self.device,
            x_train_: Some(x),
            train_shape_: Some((n_train, n_features)),
            y_reg_: Some(y),
            n_outputs_: n_outputs,
            _state: PhantomData,
        })
    }
}

/// Validate the fit operands and INFER the target width `n_outputs`.
///
/// Shared by [`Fit::fit`] and [`KNeighborsRegressor::fit_owned`] so the two
/// entry points cannot drift on what they accept — a divergence would mean the
/// borrowing path rejects a shape the owning path stores, or worse the reverse.
///
/// `y` is row-major `n_train x n_outputs` and the `Fit` trait carries a shape
/// for `x` only, so the width is inferred as `y.len() / n_train`. A `y` whose
/// length is not a whole multiple of `n_train` is rejected rather than rounded:
/// a fractional width would silently mis-stride every neighbour gather.
fn validate_fit_operands<F>(
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<usize, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (n_train, _) = shape;
    validate_geometry(x, shape)?;
    if n_train == 0 || y.len() == 0 || y.len() % n_train != 0 {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "y",
            rows: n_train,
            cols: 1,
            len: y.len(),
        }));
    }
    Ok(y.len() / n_train)
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

    /// The configured weighting.
    pub fn weights(&self) -> Weights {
        self.weights
    }

    /// The configured metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Target width inferred at `fit` (sklearn's `n_outputs_`; `1` for a 1-D
    /// target). `predict` returns `n_query × n_outputs` row-major.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs_
    }

    /// The fitted training geometry `(n_train, n_features)` (sklearn's
    /// `n_samples_fit_` and `n_features_in_`). `Some` by construction on the
    /// `Fitted` state (D-03).
    pub fn train_shape(&self) -> (usize, usize) {
        self.train_shape_
            .expect("train_shape_ is Some by construction on KNeighborsRegressor<F, Fitted>")
    }
}

/// Builder for [`KNeighborsRegressor`] (D-01). `Default` re-derives the sklearn
/// defaults from [`KNeighborsRegressor::new`] (D-08 single source) rather than
/// holding literals (Pitfall 1).
#[derive(Debug, Clone, Copy)]
pub struct KNeighborsRegressorBuilder {
    n_neighbors: usize,
    weights: Weights,
    metric: Metric,
    device: Device,
}

impl Default for KNeighborsRegressorBuilder {
    /// Re-derive the sklearn defaults from [`KNeighborsRegressor::new`] (D-08
    /// single source). `f64` is pinned only to read the F-independent scalar
    /// defaults — the builder is non-generic, so the choice of `F` here is
    /// irrelevant.
    fn default() -> Self {
        KNeighborsRegressor::<f64, Unfit>::new().into_builder()
    }
}

impl KNeighborsRegressorBuilder {
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
    /// `k <= n_train` check lives in the predict path):
    ///
    /// - `n_neighbors >= 1` ([`BuildError::InvalidNNeighbors`]). The
    ///   data-DEPENDENT `k > n_train` half stays in the predict path (T-16-V5).
    /// - `Metric::Minkowski { p }` requires `p >= 1`
    ///   ([`BuildError::InvalidMinkowskiP`], the hdbscan/knn_graph precedent).
    ///   `p < 1` is not a metric at all (the triangle inequality fails), and the
    ///   kernel's `F::powf(acc, 1/p)` would silently produce a finite but
    ///   meaningless ordering rather than failing.
    pub fn build<F>(self) -> Result<KNeighborsRegressor<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.n_neighbors == 0 {
            // IN-02: name the neighbor-honest variant so the construction-time
            // error matches the hyperparameter (`n_neighbors`), not `n_components`.
            return Err(BuildError::InvalidNNeighbors {
                estimator: "knn_regressor",
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
                    estimator: "knn_regressor",
                    p,
                });
            }
        }
        Ok(KNeighborsRegressor {
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            metric: self.metric,
            device: self.device,
            x_train_: None,
            train_shape_: None,
            y_reg_: None,
            n_outputs_: 0,
            _state: PhantomData,
        })
    }
}

impl<F> SaveModel for KNeighborsRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted regressor to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `_fit_X` | `F` (`F32`/`F64`) | `[n_samples, n_features]` |
    /// | `_y` | `F` | `[n_samples, n_outputs]` |
    /// | `param:n_neighbors` / `param:weights` / `param:metric` / `param:device` | `__metadata__` scalar | — |
    /// | `param:p` | `__metadata__` scalar, Minkowski ONLY | — |
    ///
    /// `n_outputs` is recovered from `_y`'s column extent at load, so it is not
    /// stored again — the same rule the rest of the format follows.
    ///
    /// `_y` is written as a rank-2 tensor even for the single-target case, where
    /// it is `[n_samples, 1]`. A rank-1 special case would be two code paths that
    /// can disagree, for the ~8 bytes one shape entry costs, and it would make
    /// `n_outputs` unreadable from the shape exactly when it is 1.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        let (n_samples, n_features) = self.train_shape_.ok_or_else(|| absent("train_shape_"))?;
        // Bound BEFORE the writer, which borrows every payload.
        let x_train = self
            .x_train_
            .as_ref()
            .ok_or_else(|| absent("x_train_"))?
            .to_host(pool);
        let y = self.y_reg_.as_ref().ok_or_else(|| absent("y_reg_"))?.to_host(pool);

        let mut w = NeighborsWriter::new(PERSIST_TAG);
        w.scalar_usize("param:n_neighbors", self.n_neighbors);
        write_weights(&mut w, self.weights);
        write_metric(&mut w, self.metric);
        write_device(&mut w, self.device);
        write_fit_x(&mut w, &x_train, n_samples, n_features)?;
        w.tensor(
            Y_NAME,
            TensorRef::floats(&y, vec![n_samples, self.n_outputs_])?,
        );
        w.write(path)
    }
}

impl<F> LoadModel for KNeighborsRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the regressor back from `path`, re-uploading `_fit_X` and `_y` to
    /// `pool`.
    ///
    /// The file is untrusted input (T-04-01-01), so `_fit_X` defines
    /// `n_samples` and `_y`'s row extent is checked against it before any value
    /// is stored — a mismatch would otherwise index the target table out of
    /// range on the first prediction, since the gather addresses it by neighbor
    /// index.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<KNeighborsRegressor<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = NeighborsFile::parse(&raw, PERSIST_TAG)?;
        let (x_train, n_samples, n_features) = read_fit_x::<F>(&file)?;

        let y_v = file.tensor(Y_NAME)?;
        let (y_rows, n_outputs) = shape_2d(&y_v, Y_NAME)?;
        if y_rows != n_samples || n_outputs == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{Y_NAME}' declares shape [{y_rows}, {n_outputs}], but \
                     '_fit_X' implies {n_samples} rows and at least one output"
                ),
            });
        }
        let y = crate::neighbors::neighbors_persist::as_floats::<F>(&y_v, Y_NAME)?;

        let n_neighbors = file.scalar_usize("param:n_neighbors")?;
        expect_k_fits(n_neighbors, n_samples)?;

        Ok(KNeighborsRegressor {
            n_neighbors,
            weights: read_weights(&file)?,
            metric: read_metric(&file)?,
            device: read_device(&file)?,
            x_train_: Some(DeviceArray::from_host(pool, &x_train)),
            train_shape_: Some((n_samples, n_features)),
            y_reg_: Some(DeviceArray::from_host(pool, &y)),
            n_outputs_: n_outputs,
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
    ///
    /// `y` is row-major `n_train × n_outputs`; the width is INFERRED as
    /// `y.len() / n_train` because the `Fit` trait carries a shape for `x` only.
    /// A `y` whose length is not a whole multiple of `n_train` is rejected as a
    /// `ShapeMismatch` — inferring a fractional width would silently mis-stride
    /// every gather.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<KNeighborsRegressor<F, Fitted>, AlgoError> {
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "knn_regressor",
            operation: "fit (requires y)",
        })?;
        // Validate BEFORE the copies below, so a bad shape allocates nothing
        // (ASVS V5, and it keeps the pool's `live_bytes` untouched on the error
        // path).
        validate_fit_operands::<F>(x, y, shape)?;

        // KNN-01: keep BOTH fitted operands device-resident with a
        // DEVICE-TO-DEVICE copy, never a host round-trip.
        //
        // The original `x.to_host(pool)` → `DeviceArray::from_host(...)` pair was a
        // full device→host→device copy of the ENTIRE training matrix (at
        // `n = 200_000`, `d = 32`, f32 that is 25 MB across the bus TWICE) whose
        // only purpose was to obtain an OWNED array from a borrowed one. KNN-01
        // replaced it with a HANDLE CLONE, which removed the traffic but left the
        // fitted estimator aliasing buffers the caller still owns — and
        // `DeviceArray::release_into` is the repo-wide way to return a buffer to
        // the pool, so a caller that released its fit inputs afterwards would see
        // the next `pool.acquire` hand that live training matrix straight back.
        // `device_copy` keeps the win (no bus traffic, no host allocation) and
        // makes the ownership real — see its docs.
        //
        // `y` likewise stays on the device: `predict` gathers the neighbor targets
        // with a device kernel, so the fit-time `y.to_host(pool)` read-back — and
        // the host `Vec<F>` it produced — are both gone.
        let x_dev: DeviceArray<ActiveRuntime, F> = device_copy::<F>(pool, x);
        let y_dev: DeviceArray<ActiveRuntime, F> = device_copy::<F>(pool, y);
        // The owning path re-runs the (cheap, allocation-free) validation and
        // owns the single definition of how fitted state is assembled — so the
        // two entry points cannot drift on what they store, only on how the
        // buffers got there.
        self.fit_owned(pool, x_dev, y_dev, shape)
    }
}

impl<F> Predict<F> for KNeighborsRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The `n_query × n_outputs` row-major weighted neighbor mean.
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

        // Reuse the validated NearestNeighbors core in its DEVICE-RESIDENT form
        // (KNN-01): same `1<=k<=n_train` + query-geometry validation and the same
        // query-tiled `distance → top_k` pipeline, but the `u32` neighbor indices
        // stay on the device instead of being read back.
        let (val_dev, idx_dev) = neighbor_indices_device_metric::<F>(
            pool,
            self.x_train_.as_ref(),
            self.train_shape_,
            x,
            shape,
            self.n_neighbors,
            self.metric,
            self.device,
        )?;

        let (n_train, _) = self
            .train_shape_
            .expect("train_shape_ is Some by construction on KNeighborsRegressor<F, Fitted>");

        // Weighted mean of the k neighbor targets per query, formed ON the device
        // by the fused gather kernel. This replaces the host `n_query × k` double
        // loop that previously required reading the indices back and uploading the
        // predictions again; the whole predict path is now device-resident end to
        // end (D-05), with the single terminal read-back happening at the caller's
        // API boundary.
        //
        // The kernels' `t < n_train` guard carries the WR-02 defence that the host
        // loop expressed as a typed `ShapeMismatch`: a corrupted index from top_k
        // contributes nothing rather than reading out of bounds, and flags its row
        // so the host still raises.
        //
        // The DEFAULT configuration keeps its dedicated kernel: `knn_regress_mean`
        // has no weight term and no output stride in its inner loop, and it is the
        // shape every KNN perf number in the repo was measured on. Generalizing it
        // would put a branch and a multiply in the hot path of the common case to
        // serve the uncommon ones.
        let pred = if self.weights == Weights::Uniform && self.n_outputs_ == 1 {
            // The regressor needs only the indices here; the distances are scratch.
            val_dev.release_into(pool);
            knn_regress_mean_gather::<F>(pool, y_reg, &idx_dev, n_query, self.n_neighbors, n_train)
                .map_err(AlgoError::Prim)?
        } else {
            // `val_dev` holds the TRUE metric distances (the Euclidean path
            // applied its boundary sqrt inside `top_k`; the metric kernels never
            // deferred a root), which is exactly what `1/d` weighting requires —
            // weighting by the order-preserving SQUARE would silently produce a
            // different, wrong answer that still looks plausible.
            let out = knn_regress_gather_weighted::<F>(
                pool,
                y_reg,
                &idx_dev,
                &val_dev,
                n_query,
                self.n_neighbors,
                n_train,
                self.n_outputs_,
                self.weights == Weights::Distance,
            );
            val_dev.release_into(pool);
            out.map_err(AlgoError::Prim)?
        };
        idx_dev.release_into(pool);

        Ok(pred)
    }
}

impl<F> KNeighbors<F> for KNeighborsRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The `k` nearest training points of each query under the CONFIGURED
    /// metric, as `(distances, indices)` — sklearn's `KNeighborsMixin.kneighbors`
    /// on the regressor.
    ///
    /// It is the same neighbor set `predict` averages, by construction: both go
    /// through `neighbor_indices_metric` with the same metric and the same `k`
    /// validation. That identity is what lets the Python shim serve
    /// `weights=<callable>` — it recomputes the average from these distances and
    /// must land on the built-in weightings' answer when handed their formula.
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
