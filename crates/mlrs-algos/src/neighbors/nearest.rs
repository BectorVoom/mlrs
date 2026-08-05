//! `NearestNeighbors` (NEIGH-01) — brute-force k-nearest-neighbor query,
//! matching `sklearn.neighbors.NearestNeighbors(algorithm='brute',
//! metric='euclidean')`.
//!
//! ## Brute-force Euclidean, weights uninvolved (CONTEXT Deferred)
//! v1 ships the brute-force path only: `kneighbors` computes the full
//! `n_query × n_train` pairwise distance and partial-selects the `k` smallest per
//! query row. There is no spatial index (kd-tree / ball-tree) and no
//! `weights='distance'` — those are deferred per CONTEXT.
//!
//! ## Squared-distance select, sqrt at the boundary (Pitfall 8 / D-08)
//! `kneighbors` calls [`distance`] with `sqrt = false` (the order-preserving
//! SQUARED form) then [`top_k`] with `sqrt = true`, so only the returned
//! `n_query × k` values are sqrt'd — never the whole distance matrix. The
//! returned distances are therefore true sqrt-Euclidean and the neighbor indices
//! are unaffected by the monotone sqrt. The lowest-index tie-break is inherited
//! directly from the validated `top_k` primitive (05-02).
//!
//! ## i32 neighbor indices (D-06)
//! `top_k` returns `u32` column indices into the fitted training set; this
//! estimator re-uploads them as `i32` to satisfy the [`KNeighbors`] trait
//! contract (i32 indices, shared with the discrete-label surface). The cast is
//! host-side over the small `n_query × k` index buffer (D-06 — confirmed to need
//! zero pool/bridge changes in 05-01).
//!
//! ## Validate-before-launch (T-05-08-01 / ASVS V5)
//! `kneighbors` rejects `k` outside `1 ..= n_train` with [`AlgoError::InvalidK`]
//! and a mismatched query geometry with [`PrimError::ShapeMismatch`] BEFORE any
//! prim launch; the `top_k` primitive re-validates its own geometry (05-02).
//!
//! ## Device residency (D-03)
//! The fitted training matrix is stored device-resident; `kneighbors` runs the
//! distance + select on-device and materializes to the host only for the small
//! u32→i32 index cast and at the oracle-comparison boundary.
//!
//! Tests live in `crates/mlrs-algos/tests/nearest_neighbors_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};
use cubecl::server::Handle;

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::{distance_direct, distance_with_ynorm, metric_distance};
use mlrs_backend::prims::knn::{
    cpu_metric_rows_applicable, cpu_metric_rows_topk, cpu_rows_applicable, cpu_rows_topk,
    device_copy, fused_distance_topk, fused_topk_applicable,
};
use mlrs_backend::prims::knn_graph::Metric;
use mlrs_backend::prims::knn_host::{knn_host_applicable, knn_host_topk};
use mlrs_backend::prims::reduce::{row_reduce, ReducePath, ScalarOp};
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, KNeighbors, Unfit};

/// sklearn `NearestNeighbors` default neighbor count.
const NN_DEFAULT_N_NEIGHBORS: usize = 5;

/// Brute-force k-nearest-neighbor query (NEIGH-01).
///
/// Construct with the zero-arg [`NearestNeighbors::new`] (sklearn default
/// `n_neighbors = 5`) or [`NearestNeighbors::builder`], then the consuming
/// [`Fit::fit`] (stores the training matrix; `y` is ignored — this is the
/// unsupervised neighbor index) and [`KNeighbors::kneighbors`], which exists ONLY
/// on `NearestNeighbors<F, Fitted>` (the compile-time typestate replaces the old
/// runtime `NotFitted` guard, D-03). The fitted training matrix is
/// device-resident (D-03).
pub struct NearestNeighbors<F, S = Unfit> {
    /// Default neighbor count used when a caller passes its own `k` to
    /// `kneighbors`; retained for the sklearn-faithful constructor surface.
    n_neighbors: usize,
    /// Device-resident training matrix (`n_train × n_features`, row-major),
    /// `None` until `fit`.
    x_train_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted training geometry `(n_train, n_features)`, `None` until `fit`.
    train_shape_: Option<(usize, usize)>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> NearestNeighbors<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfit `NearestNeighbors` with sklearn's default
    /// `n_neighbors = 5`. This is the SINGLE source of truth for the default
    /// hyperparameter (D-08): the builder `Default` re-derives from here via
    /// [`NearestNeighbors::into_builder`], rather than re-listing the literal.
    /// The per-call `k` passed to [`KNeighbors::kneighbors`] overrides it; both
    /// are validated against the fitted `n_train` at query time
    /// ([`AlgoError::InvalidK`]).
    pub fn new() -> Self {
        Self {
            n_neighbors: NN_DEFAULT_N_NEIGHBORS,
            x_train_: None,
            train_shape_: None,
            _state: PhantomData,
        }
    }

    /// Start building a `NearestNeighbors` from sklearn's defaults (D-08 single
    /// source).
    pub fn builder() -> NearestNeighborsBuilder {
        NearestNeighborsBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying the
    /// hyperparameter. Used by [`NearestNeighborsBuilder::default`] to re-derive
    /// the defaults from [`NearestNeighbors::new`] (D-08).
    pub fn into_builder(self) -> NearestNeighborsBuilder {
        NearestNeighborsBuilder {
            n_neighbors: self.n_neighbors,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators. Used by the
    /// defaults-equality test (BLDR-01):
    /// `NearestNeighbors::new().hyperparams_eq(&NearestNeighbors::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_neighbors == other.n_neighbors
    }

    /// The configured default neighbor count (read pre-fit).
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Fit by TAKING OWNERSHIP of an already-device-resident training matrix —
    /// the zero-copy sibling of [`Fit::fit`] (KNN-REG-FIT).
    ///
    /// Same validation, same fitted state; `x` arrives BY VALUE, so the
    /// estimator adopts the caller's buffer instead of duplicating it. See
    /// [`KNeighborsRegressor::fit_owned`](crate::neighbors::regressor::KNeighborsRegressor::fit_owned)
    /// for why the borrowing form has to copy and why a caller that uploaded
    /// the operand purely to hand it over does not.
    ///
    /// On validation failure `x` is released back into `pool` — the caller has
    /// already given it up, so nothing else can, and leaking it would raise the
    /// pool's `live_bytes` permanently (the FOUND-05 conservation property the
    /// D-10 memory gate asserts on).
    pub fn fit_owned(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<NearestNeighbors<F, Fitted>, AlgoError> {
        if let Err(e) = validate_geometry(&x, shape) {
            x.release_into(pool);
            return Err(e);
        }
        Ok(NearestNeighbors {
            n_neighbors: self.n_neighbors,
            x_train_: Some(x),
            train_shape_: Some(shape),
            _state: PhantomData,
        })
    }
}

impl<F> Default for NearestNeighbors<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> NearestNeighbors<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The configured default neighbor count.
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// The fitted training geometry `(n_train, n_features)`. `Some` by
    /// construction on the `Fitted` state (D-03).
    pub fn train_shape(&self) -> (usize, usize) {
        self.train_shape_
            .expect("train_shape_ is Some by construction on NearestNeighbors<F, Fitted>")
    }
}

/// Builder for [`NearestNeighbors`] (D-01). `Default` re-derives the sklearn
/// default from [`NearestNeighbors::new`] (D-08 single source) rather than
/// holding a literal (Pitfall 1: default-drift breaks the oracle gate silently).
#[derive(Debug, Clone, Copy)]
pub struct NearestNeighborsBuilder {
    n_neighbors: usize,
}

impl Default for NearestNeighborsBuilder {
    /// Re-derive the sklearn default from [`NearestNeighbors::new`] (D-08 single
    /// source). `f64` is pinned only to read the F-independent scalar default —
    /// the builder is non-generic, so the choice of `F` here is irrelevant.
    fn default() -> Self {
        NearestNeighbors::<f64, Unfit>::new().into_builder()
    }
}

impl NearestNeighborsBuilder {
    /// Set the default neighbor count `n_neighbors`.
    pub fn n_neighbors(mut self, v: usize) -> Self {
        self.n_neighbors = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameter BEFORE any data is seen (D-08; the data-DEPENDENT
    /// `k <= n_train` check lives in [`KNeighbors::kneighbors`]):
    ///
    /// - `n_neighbors >= 1` ([`BuildError::InvalidNNeighbors`]) — a zero
    ///   neighbor count is always invalid regardless of the training data. The
    ///   data-DEPENDENT `k > n_train` half stays in the `kneighbors` core
    ///   (T-16-V5; the fit/kneighbors `k` validation is NOT dropped).
    pub fn build<F>(self) -> Result<NearestNeighbors<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.n_neighbors == 0 {
            // IN-02: name the neighbor-honest variant so the construction-time
            // error matches the hyperparameter (`n_neighbors`), not `n_components`.
            return Err(BuildError::InvalidNNeighbors {
                estimator: "nearest_neighbors",
                n_neighbors: self.n_neighbors,
            });
        }
        Ok(NearestNeighbors {
            n_neighbors: self.n_neighbors,
            x_train_: None,
            train_shape_: None,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for NearestNeighbors<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = NearestNeighbors<F, Fitted>;

    /// Store the training matrix `x` (`shape = (n_train, n_features)`),
    /// CONSUMING `self` and returning the `Fitted`-tagged sibling. `y` is
    /// ignored — `NearestNeighbors` is the unsupervised neighbor index. Geometry
    /// is validated before the matrix is staged (ASVS V5).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<NearestNeighbors<F, Fitted>, AlgoError> {
        validate_geometry(x, shape)?;

        // Take device-resident ownership of the training matrix (D-03) with a
        // DEVICE-TO-DEVICE copy (KNN-01, corrected).
        //
        // This used to be `x.to_host(pool)` → `DeviceArray::from_host(...)`: a
        // full device→host→device copy of the entire training matrix, producing a
        // bit-identical buffer, purely to turn a borrowed array into an owned one.
        // KNN-01 replaced it with `DeviceArray::from_raw(x.handle().clone(), ..)`,
        // which removed the bus traffic but made the fitted estimator ALIAS a
        // buffer the caller still owns — and `DeviceArray::release_into` is the
        // repo-wide way to return a buffer to the pool free list, so any caller
        // that uploaded `x` as a temporary and released it after `fit` would have
        // that exact allocation handed back by the next `pool.acquire` of the same
        // size (the first one `predict` makes), silently overwriting the live
        // training matrix. The invariant was prose; nothing enforced it.
        //
        // `device_copy` keeps the win that mattered — no host round-trip, no host
        // allocation, one device pass — while making the fitted state genuinely
        // owned, so releasing the fit input is safe again.
        let x_dev: DeviceArray<ActiveRuntime, F> = device_copy::<F>(pool, x);

        // The owning path owns the single definition of how fitted state is
        // assembled, so the two entry points can differ only in how the buffer
        // got there.
        self.fit_owned(pool, x_dev, shape)
    }
}

impl<F> KNeighbors<F> for NearestNeighbors<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
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
        let (distances, indices, _) = neighbor_indices::<F>(
            pool,
            self.x_train_.as_ref(),
            self.train_shape_,
            x,
            shape,
            k,
        )?;
        Ok((distances, indices))
    }
}

/// Shared kneighbors core (NEIGH-01/02/03): validate `k` + geometry, then
/// `distance(xq, x_train, sqrt=false)` → `top_k(.., k, sqrt=true)`. Returns the
/// `n_query × k` sqrt-Euclidean distances (`F`, device-resident), the same-shape
/// neighbor indices (`i32`, device-resident, host-cast from the `top_k` `u32`,
/// D-06), AND the host `u32` index buffer so the classifier / regressor can gather
/// neighbor targets without a second round-trip.
///
/// Factored here so `KNeighborsClassifier` / `KNeighborsRegressor` build their
/// vote / mean on EXACTLY the `NearestNeighbors` neighbor set (Pitfall 8 — same
/// tie-break, same distances).
#[allow(clippy::type_complexity)]
pub(crate) fn neighbor_indices<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_train: Option<&DeviceArray<ActiveRuntime, F>>,
    train_shape: Option<(usize, usize)>,
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    k: usize,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, i32>,
        Vec<u32>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    neighbor_indices_metric::<F>(pool, x_train, train_shape, xq, shape, k, Metric::Euclidean)
}

/// [`neighbor_indices`] under an arbitrary [`Metric`] (KNN-REG-PARAMS).
///
/// `Metric::Euclidean` is bit-for-bit the pre-existing path — it dispatches
/// through the SAME [`tiled_distance_topk`] and therefore the same
/// cpu-row-scan / fused / register-blocked kernel family. The other metrics take
/// the [`tiled_metric_distance_topk`] composition. Keeping the default a
/// delegation rather than a `match` arm of its own is what guarantees adding a
/// metric cannot perturb the tuned default.
#[allow(clippy::type_complexity)]
pub(crate) fn neighbor_indices_metric<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_train: Option<&DeviceArray<ActiveRuntime, F>>,
    train_shape: Option<(usize, usize)>,
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    k: usize,
    metric: Metric,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, i32>,
        Vec<u32>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    // --- T-05-08-01 / ASVS V5: validate the untrusted k + query geometry BEFORE
    //     any prim launch (shared with the device core so the two cannot drift). ---
    let (x_train, (n_train, n_features)) =
        validate_kneighbors::<F>(x_train, train_shape, xq, shape, k)?;
    let (n_query, _) = shape;

    // --- 1./2. QUERY-TILED distance → select-k, straight into the full output
    //        buffers (see `neighbor_indices_device` for why tiling is required). ---
    let (val_dev, idx_dev_u32) = metric_topk::<F>(
        pool,
        x_train,
        (n_train, n_features),
        xq,
        (n_query, n_features),
        k,
        metric,
    )?;

    // --- 3. u32 → i32 neighbor indices (D-06). Host-cast the small n_query × k
    //        index buffer; the values are training-column indices in
    //        [0, n_train), so the cast is exact. Keep the host u32 copy so the
    //        classifier gathers targets without a second round-trip. ---
    let idx_host: Vec<u32> = idx_dev_u32.to_host(pool);
    idx_dev_u32.release_into(pool);
    let idx_i32: Vec<i32> = idx_host.iter().map(|&u| u as i32).collect();
    let idx_dev_i32: DeviceArray<ActiveRuntime, i32> = DeviceArray::from_host(pool, &idx_i32);

    Ok((val_dev, idx_dev_i32, idx_host))
}

/// Fully DEVICE-RESIDENT kneighbors core under an arbitrary [`Metric`] (KNN-01 /
/// KNN-REG-PARAMS) — the core `KNeighborsRegressor::predict` runs on.
///
/// Same validation and same `distance → top_k` pipeline as
/// [`neighbor_indices_metric`], but it returns the raw `u32` indices ON the
/// device and performs NO host round-trip at all. [`neighbor_indices`] /
/// [`neighbor_indices_metric`] exist for the consumers that genuinely need host
/// indices (`kneighbors`'s public `i32` surface, the classifier's label gather);
/// the regressor pairs this one with the fused device gathers so its predict
/// never leaves the device.
///
/// Returns `(distances, indices)` both device-resident, the distances TRUE
/// (already rooted / already in the metric's own units) so a `weights='distance'`
/// consumer can weight by `1/d` directly. `Metric::Euclidean` delegates to the
/// unchanged tuned path.
#[allow(clippy::type_complexity)]
pub(crate) fn neighbor_indices_device_metric<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_train: Option<&DeviceArray<ActiveRuntime, F>>,
    train_shape: Option<(usize, usize)>,
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    k: usize,
    metric: Metric,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, u32>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    let (x_train, (n_train, n_features)) =
        validate_kneighbors::<F>(x_train, train_shape, xq, shape, k)?;
    metric_topk::<F>(pool, x_train, (n_train, n_features), xq, shape, k, metric)
}

/// The `distance → select-k` DISPATCH, in one place: host arm, tuned Euclidean
/// device family, or the general per-metric device composition.
///
/// Both kneighbors cores route through here so they cannot pick different arms
/// for the same query — `KNeighborsRegressor::predict` uses the device-resident
/// core while its `kneighbors` uses the host-index one, and the Python shim's
/// `weights=<callable>` path REBUILDS a prediction from `kneighbors` distances
/// and must land on `predict`'s answer.
///
/// Order of preference:
///
/// 1. [`tiled_distance_topk`] for `Metric::Euclidean` — bit-for-bit the
///    pre-existing tuned path (fused / register-blocked / cpu row-scan family).
///    On cpu it is preferred over the host arm only inside the RECTANGLE its
///    row-scan kernel actually covers ([`cpu_rows_applicable`]: `k <= 16`,
///    `n_features <= 32`), which is where it is genuinely faster. A/B on a
///    16-core Zen5 at `n_train = 50_000`, `d = 16`, `n_query = 5_000` (best of
///    5, interleaved; `MLRS_KNN_HOST` forcing each arm):
///
///    | `k`  | tuned kernel | host arm | notes                          |
///    |------|--------------|----------|--------------------------------|
///    | 1    | 0.025 s      | 0.037 s  | kernel 1.5× — inside the cap   |
///    | 15   | 0.034 s      | 0.040 s  | kernel 1.2× — inside the cap   |
///    | 20   | 6.59 s       | 0.042 s  | host 157× — PAST the cap       |
///    | 100  | 29.0 s       | 0.100 s  | host 290× — PAST the cap       |
///
///    Past the cap the kernel is not slower, it is ABSENT:
///    [`tiled_distance_topk`] falls through to the GPU-shaped composition, and
///    the last two rows are that fallback, not a kernel. sklearn takes 0.12 s
///    and 0.30 s on the same two configs — so the 16-slot list was the boundary
///    between beating it 4× and losing to it 50-120×.
///
///    Inside the cap the kernel's remaining edge is its COMPILER, not its shape:
///    `cubecl-cpu` JITs for the host's real feature set while this crate is
///    built for the x86-64 baseline. The host arm closed most of that with a
///    runtime-detected AVX2 body (see `knn_host::dispatch_scan_block`); what is
///    left is the ~1.2-1.5× above, which is why the gate stays.
/// 2. [`cpu_metric_rows_topk`] on the cpu backend (KNN-CUBE-METRIC) — the
///    CubeCL row-scan family, one kernel per metric, whose lane axis is a
///    `Vector<F, Const<32>>` of QUERY ROWS. `cubecl-cpu` lowers those to real
///    MLIR `vector<32xf32>` ops AND JITs for the host cpu, so the kernels get
///    the machine's true register width for free. Caps: `k <= 128`,
///    `n_features <= 128` (comptime local-array sizes).
/// 3. [`knn_host_topk`] (KNN-HOST) — the uncapped plain-Rust fallback, for `k`
///    or `n_features` past those. It used to be arm 2; the CubeCL family matches
///    or beats it on every metric (see its module docs for the table).
/// 4. [`tiled_metric_distance_topk`] for the other metrics on a device backend.
///
/// Keeping the Euclidean default a delegation rather than a `match` arm of its
/// own is what guarantees adding a metric cannot perturb the tuned path.
#[allow(clippy::type_complexity)]
fn metric_topk<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_train: &DeviceArray<ActiveRuntime, F>,
    (n_train, n_features): (usize, usize),
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    k: usize,
    metric: Metric,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, u32>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    let (n_query, _) = shape;
    let tuned_euclidean = matches!(metric, Metric::Euclidean)
        && cpu_rows_applicable::<F>(n_query, n_train, n_features, k);
    // The METRIC cpu row-scan family (KNN-CUBE-METRIC) serves everything the
    // tuned Euclidean kernel does not, inside its own four-times-wider caps.
    if !tuned_euclidean && cpu_metric_rows_applicable::<F>(n_query, n_train, n_features, k) {
        return cpu_metric_rows_topk::<F>(
            pool,
            xq,
            (n_query, n_features),
            x_train,
            n_train,
            k,
            metric,
        )
        .map_err(AlgoError::Prim);
    }
    if knn_host_applicable(n_query, n_train, n_features, k, tuned_euclidean) {
        return knn_host_topk::<F>(
            pool,
            xq,
            (n_query, n_features),
            x_train,
            n_train,
            k,
            metric,
        )
        .map_err(AlgoError::Prim);
    }
    if matches!(metric, Metric::Euclidean) {
        tiled_distance_topk::<F>(pool, x_train, (n_train, n_features), xq, shape, k)
    } else {
        tiled_metric_distance_topk::<F>(pool, x_train, (n_train, n_features), xq, shape, k, metric)
    }
}

/// Validate the untrusted `k` + query geometry BEFORE any prim launch
/// (T-05-08-01 / ASVS V5), returning the unwrapped fitted state.
///
/// `1 <= k <= n_train` (cannot request more neighbors than training points); the
/// query feature count must match the fitted one. Factored out so the host and
/// device cores cannot drift on validation.
fn validate_kneighbors<'a, F>(
    x_train: Option<&'a DeviceArray<ActiveRuntime, F>>,
    train_shape: Option<(usize, usize)>,
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    k: usize,
) -> Result<(&'a DeviceArray<ActiveRuntime, F>, (usize, usize)), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let x_train = x_train.ok_or(AlgoError::NotFitted {
        estimator: "knn",
        operation: "kneighbors",
    })?;
    let (n_train, n_features) = train_shape.ok_or(AlgoError::NotFitted {
        estimator: "knn",
        operation: "kneighbors",
    })?;
    let (n_query, q_features) = shape;

    if k < 1 || k > n_train {
        return Err(AlgoError::InvalidK {
            estimator: "knn",
            k,
            n_samples: n_train,
        });
    }
    if n_query == 0 || q_features == 0 || xq.len() != n_query * q_features {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "xq",
            rows: n_query,
            cols: q_features,
            len: xq.len(),
        }));
    }
    if q_features != n_features {
        return Err(AlgoError::Prim(PrimError::DimMismatch {
            dim: "n_features",
            lhs: q_features,
            rhs: n_features,
        }));
    }
    Ok((x_train, (n_train, n_features)))
}

/// Peak bytes the intermediate `tile_rows × n_train` distance matrix may occupy.
///
/// The brute-force pipeline's memory is dominated by that matrix, NOT by the
/// operands: at `n_query = 10_000` against `n_train = 100_000` in f32 it is 4 GB,
/// which simply fails to allocate (the probe hit exactly that on wgpu). sklearn
/// and cuML both chunk the query rows for this reason; 128 MiB keeps the working
/// set comfortably inside any supported device while staying far above the size
/// at which the GEMM stops saturating.
const DIST_TILE_BUDGET_BYTES: usize = 128 * 1024 * 1024;

/// Row granularity every tile boundary must respect (KNN-01).
///
/// Tiles are zero-copy byte-offset VIEWS of the parent buffers, and a device
/// imposes a `min_storage_buffer_offset_alignment` on any such view — an
/// unaligned binding is a hard validation failure, not a slow path (wgpu rejected
/// a 13_420-byte offset against its 32-byte limit during bring-up). The three
/// view offsets are `start * n_features * size_of::<F>()` and (twice)
/// `start * k * size_of::<u32 | F>()`, so making `start` a multiple of 64 makes
/// every one of them a multiple of `64 * 4 = 256` bytes for the smallest (4-byte)
/// element in play — at or above the alignment any supported backend requests,
/// for ANY `n_features` and `k`.
const TILE_ALIGN_ROWS: usize = 64;

/// Is the GEMM-expansion distance path forced via `MLRS_KNN_GEMM_DISTANCE=1`?
///
/// The direct per-element kernel is the default. This escape hatch exists so the
/// two can be A/B'd on the real target device rather than gated by extrapolation
/// from a different backend's numbers.
fn gemm_distance_forced() -> bool {
    mlrs_backend::abflag::is_on("MLRS_KNN_GEMM_DISTANCE")
}

/// `distance → top_k` over the query rows in TILES, writing each tile's results
/// directly into the full `n_query × k` output buffers (KNN-01).
///
/// Geometry is assumed already validated by [`validate_kneighbors`].
///
/// ## Why tiling, and why it is free
/// See [`DIST_TILE_BUDGET_BYTES`]: materializing the whole `n_query × n_train`
/// distance matrix is what bounds the problem size, and it is pure scratch — no
/// result depends on two tiles at once, because top-k is computed independently
/// per query ROW. Tiling therefore changes NOTHING numerically (each row sees the
/// identical full `n_train` candidate set) while making peak memory independent of
/// `n_query`. The per-tile scratch is released back to the [`BufferPool`] each
/// iteration, so every tile after the first reuses the same device buffer (D-11).
///
/// ## Zero-copy tile views
/// Both the query matrix and the two outputs are row-major and contiguous, so a
/// tile is a byte-offset VIEW of the parent buffer (`Handle::offset_start`) rather
/// than a copy — the tile loop introduces no extra device traffic, and `top_k`
/// writes its tile at the correct global row offset through the offset output
/// handles.
fn tiled_distance_topk<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_train: &DeviceArray<ActiveRuntime, F>,
    (n_train, n_features): (usize, usize),
    xq: &DeviceArray<ActiveRuntime, F>,
    (n_query, _): (usize, usize),
    k: usize,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, u32>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    // CPU-SHAPED path (KNN-04), checked FIRST because on the cpu backend the
    // shared-memory fused family below is structurally wrong, not merely
    // slower: `cubecl-cpu` maps one OS THREAD per unit and implements
    // `sync_cube` as a spin barrier across all of them, so the 256-unit fused
    // kernels perform hundreds of thousands of 256-way spin rendezvous on a
    // machine with a dozen cores. `cpu_rows_topk` is barrier-free and its
    // parallelism is query TILES over core-count units, so it also has no
    // minimum-`n_query` occupancy floor to clear — only the `cols`/`k` caps of
    // its local caches. `MLRS_KNN_CPU_ROWS=0` forces the fused family back for
    // on-target A/B.
    if cpu_rows_applicable::<F>(n_query, n_train, n_features, k) {
        return cpu_rows_topk::<F>(
            pool,
            xq,
            (n_query, n_features),
            x_train,
            n_train,
            k,
            true,
        )
        .map_err(AlgoError::Prim);
    }

    // FUSED path (KNN-02): squared distance + selection in ONE kernel, no
    // intermediate matrix, no query tiling. Applicable for KNN-sized k on
    // adapters whose shared-memory limit covers the kernel's ~19 KB budget
    // (a capability gate — wgpu's default 16 KiB limit takes the two-kernel
    // path below); `MLRS_KNN_UNFUSED=1` forces the two-kernel path for A/B on
    // the real target device.
    if fused_topk_applicable::<F>(n_query, n_train, k) {
        return fused_distance_topk::<F>(
            pool,
            xq,
            (n_query, n_features),
            x_train,
            (n_train, n_features),
            k,
            true,
        )
        .map_err(AlgoError::Prim);
    }

    let tile_rows = dist_tile_rows::<F>(n_train, n_query);

    let val_bytes = n_query * k * size_of::<F>();
    let idx_bytes = n_query * k * size_of::<u32>();
    let val_handle = pool.acquire(val_bytes);
    let idx_handle = pool.acquire(idx_bytes);

    // The tile loop is fallible (`distance_*` and `top_k` both propagate
    // `PrimError`), and the two output buffers above are already charged to
    // `BufferPool::live_bytes`. Run it through a helper so EVERY early return can
    // be met with an explicit release here: dropping the handles instead would
    // leave `live_bytes` permanently raised by `n_query * k * 8` bytes, and
    // `pool.rs` documents that counter as the FOUND-05 conservation property the
    // D-10 memory gate asserts on. A failed `kneighbors`/`predict` must leave the
    // pool exactly as it found it.
    if let Err(e) = tiled_distance_topk_tiles::<F>(
        pool,
        x_train,
        (n_train, n_features),
        xq,
        n_query,
        k,
        tile_rows,
        &val_handle,
        &idx_handle,
    ) {
        pool.release(val_handle, val_bytes);
        pool.release(idx_handle, idx_bytes);
        return Err(e);
    }

    Ok((
        DeviceArray::from_raw(val_handle, n_query * k),
        DeviceArray::from_raw(idx_handle, n_query * k),
    ))
}

/// Query rows per tile for the two-kernel `distance → top_k` pipelines.
///
/// Round the budget-derived tile DOWN to the alignment granularity, then clamp
/// to `n_query`. This preserves the invariant the offset views need: either the
/// result is a multiple of [`TILE_ALIGN_ROWS`] (so every `start` is too), or the
/// clamp made it `n_query`, in which case there is a SINGLE tile at `start = 0`
/// — which is trivially aligned.
///
/// `max(TILE_ALIGN_ROWS)` means a very large `n_train` can exceed the byte
/// budget: 64 rows is the smallest tile that can be cut at all, so
/// [`DIST_TILE_BUDGET_BYTES`] is a target rather than a hard cap. It is only
/// reachable past `n_train ≈ 512_000` in f32, where the brute-force method is
/// impractical for other reasons anyway.
///
/// Shared by the Euclidean and arbitrary-metric tile loops so the two cannot
/// drift into different alignment invariants — the offset-view binding failure a
/// drift would cause is a hard device validation error, not a slow path.
fn dist_tile_rows<F>(n_train: usize, n_query: usize) -> usize
where
    F: Float + CubeElement + Pod,
{
    let row_bytes = n_train * size_of::<F>();
    ((DIST_TILE_BUDGET_BYTES / row_bytes.max(1)) / TILE_ALIGN_ROWS * TILE_ALIGN_ROWS)
        .max(TILE_ALIGN_ROWS)
        .min(n_query)
}

/// The fallible tile loop of [`tiled_distance_topk`], split out so its caller
/// owns the two output buffers and can return them to the pool on ANY error
/// path (see the call site). Writes each tile's results directly into
/// `val_handle` / `idx_handle` through zero-copy byte-offset views; returns
/// nothing on success.
#[allow(clippy::too_many_arguments)]
fn tiled_distance_topk_tiles<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_train: &DeviceArray<ActiveRuntime, F>,
    (n_train, n_features): (usize, usize),
    xq: &DeviceArray<ActiveRuntime, F>,
    n_query: usize,
    k: usize,
    tile_rows: usize,
    val_handle: &Handle,
    idx_handle: &Handle,
) -> Result<(), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    // Read the A/B switch ONCE (it cannot change mid-call) rather than on every
    // tile iteration — a `std::env::var` is a locked environ scan plus a String
    // allocation, and this loop is the predict hot path.
    let gemm_distance = gemm_distance_forced();

    // HOIST the training-set squared-norm term out of the tile loop (KNN-01).
    // `distance` recomputes `‖x_train_j‖²` on every call, and that reduction is
    // `O(n_train × n_features)` — INDEPENDENT of the tile height. Left inside the
    // loop it is repeated once per tile over the whole training set, which is what
    // made a tiled run cost MORE per tile than an untiled run of the same total
    // work. Computed once here, every tile reuses it.
    // Only the GEMM-expansion path consumes this term; the direct kernel needs
    // no norms at all.
    let ynorm = if gemm_distance {
        Some(
            row_reduce::<F>(
                pool,
                x_train,
                n_train,
                n_features,
                ScalarOp::SumSq,
                ReducePath::Shared,
            )
            .map_err(AlgoError::Prim)?
            .expect("shared path is never plane-gated to None"),
        )
    } else {
        None
    };

    let mut start = 0usize;
    while start < n_query {
        let rows = tile_rows.min(n_query - start);

        // Zero-copy row-slice views of the query matrix and of both outputs.
        let xq_tile: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(
            xq.handle()
                .clone()
                .offset_start((start * n_features * size_of::<F>()) as u64),
            rows * n_features,
        );
        let out_val: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(
            val_handle
                .clone()
                .offset_start((start * k * size_of::<F>()) as u64),
            rows * k,
        );
        let out_idx: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_raw(
            idx_handle
                .clone()
                .offset_start((start * k * size_of::<u32>()) as u64),
            rows * k,
        );

        // Pairwise SQUARED Euclidean distance for this tile (no sqrt — top-k
        // selects on the order-preserving squared form, Pitfall 8).
        // DIRECT per-element kernel by default (KNN-01): the GEMM-expansion's
        // `cubek-matmul` call is pathologically slow at the tiny contraction
        // dimension a pairwise-distance shape has (`k = n_features`), and after
        // the selection kernel was fixed it was ~99% of KNN predict. Set
        // `MLRS_KNN_GEMM_DISTANCE=1` to A/B against the expansion path on the
        // target device.
        let dist = {
            let attempt = if gemm_distance {
                distance_with_ynorm::<F>(
                    pool,
                    &xq_tile,
                    (rows, n_features),
                    x_train,
                    (n_train, n_features),
                    false,
                    None,
                    ynorm.as_ref(),
                )
            } else {
                distance_direct::<F>(
                    pool,
                    &xq_tile,
                    (rows, n_features),
                    x_train,
                    (n_train, n_features),
                    None,
                )
            };
            match attempt {
                Ok(d) => d,
                Err(e) => {
                    // `ynorm` is scratch owned by THIS function — release it
                    // before propagating so the pool's `live_bytes` is conserved
                    // across the failure.
                    if let Some(yn) = ynorm {
                        yn.release_into(pool);
                    }
                    return Err(AlgoError::Prim(e));
                }
            }
        };

        // Select the k nearest per query row, sqrt at the boundary so the returned
        // distances are true sqrt-Euclidean (lowest-index tie-break inherited from
        // the validated top_k prim, 05-02). The `out_*` views mean the results land
        // in the full buffers with no copy-back (D-11).
        let (val_view, idx_view) = match top_k::<F>(
            pool,
            &dist,
            rows,
            n_train,
            k,
            true,
            Some(out_val),
            Some(out_idx),
        ) {
            Ok(v) => v,
            Err(e) => {
                // Same conservation duty as the distance arm, plus this tile's
                // distance scratch.
                dist.release_into(pool);
                if let Some(yn) = ynorm {
                    yn.release_into(pool);
                }
                return Err(AlgoError::Prim(e));
            }
        };
        // `val_view`/`idx_view` alias the caller-owned full buffers — drop the
        // wrappers WITHOUT releasing, or the pool would hand the live output to a
        // later acquire (WR-07).
        drop(val_view);
        drop(idx_view);

        // Scratch only: return it so the next tile reuses the same buffer (D-11).
        dist.release_into(pool);

        start += rows;
    }
    // The hoisted norm term is scratch owned by THIS function (distance never
    // releases a caller-supplied one) — return it to the pool now that the last
    // tile has consumed it.
    if let Some(yn) = ynorm {
        yn.release_into(pool);
    }

    Ok(())
}

/// `metric_distance → top_k` over the query rows in TILES for a NON-Euclidean
/// [`Metric`] (KNN-REG-PARAMS) — the arbitrary-metric sibling of
/// [`tiled_distance_topk`].
///
/// Geometry is assumed already validated by [`validate_kneighbors`].
///
/// ## Why this is a separate function rather than a branch inside the Euclidean one
/// The Euclidean path dispatches across three kernel FAMILIES chosen by backend
/// and shape (the cpu row-scan, the shared-memory fused family, the
/// register-blocked two-kernel pipeline), and every one of them hard-codes
/// `Σ(x−y)²`. None generalises to L1/L∞/Lp/cosine, so a metric branch inside
/// them would be dead weight at best and a silent wrong-metric dispatch at
/// worst. This composition uses only the portable per-element metric kernels
/// plus the shared `top_k`, so it runs on every backend at every shape.
///
/// ## Tiling is the same, and free, for the same reason
/// Materializing the whole `n_query × n_train` block is what bounds the problem
/// size, and it is pure scratch: top-k is computed independently per query ROW,
/// so no result depends on two tiles at once. Tiling therefore changes NOTHING
/// numerically while making peak memory independent of `n_query`. Both outputs
/// and the query matrix are row-major and contiguous, so each tile is a
/// zero-copy byte-offset VIEW (`Handle::offset_start`) and `top_k` writes
/// straight into the full buffers at the right global row offset.
///
/// The boundary root is whatever [`metric_distance`] reports it deferred: `true`
/// only for Euclidean (which never reaches here), so in practice the metric
/// kernels' TRUE distances pass through `top_k` untouched. The flag is
/// nonetheless forwarded rather than hard-coded `false`, so a future metric that
/// selects on an order-preserving power gets its root applied to the `k`
/// selected values instead of the whole block.
#[allow(clippy::too_many_arguments)]
fn tiled_metric_distance_topk<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_train: &DeviceArray<ActiveRuntime, F>,
    (n_train, n_features): (usize, usize),
    xq: &DeviceArray<ActiveRuntime, F>,
    (n_query, _): (usize, usize),
    k: usize,
    metric: Metric,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, u32>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    let tile_rows = dist_tile_rows::<F>(n_train, n_query);

    let val_bytes = n_query * k * size_of::<F>();
    let idx_bytes = n_query * k * size_of::<u32>();
    let val_handle = pool.acquire(val_bytes);
    let idx_handle = pool.acquire(idx_bytes);

    // Same conservation duty as the Euclidean path: the two output buffers are
    // already charged to `BufferPool::live_bytes`, so EVERY early return inside
    // the loop must be met with an explicit release here. Dropping the handles
    // instead would leave `live_bytes` permanently raised by `n_query * k * 8`
    // bytes, and `pool.rs` documents that counter as the FOUND-05 conservation
    // property the D-10 memory gate asserts on.
    if let Err(e) = tiled_metric_distance_topk_tiles::<F>(
        pool,
        x_train,
        (n_train, n_features),
        xq,
        n_query,
        k,
        tile_rows,
        metric,
        &val_handle,
        &idx_handle,
    ) {
        pool.release(val_handle, val_bytes);
        pool.release(idx_handle, idx_bytes);
        return Err(e);
    }

    Ok((
        DeviceArray::from_raw(val_handle, n_query * k),
        DeviceArray::from_raw(idx_handle, n_query * k),
    ))
}

/// The fallible tile loop of [`tiled_metric_distance_topk`], split out so its
/// caller owns the two output buffers and can return them to the pool on ANY
/// error path (see the call site).
#[allow(clippy::too_many_arguments)]
fn tiled_metric_distance_topk_tiles<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_train: &DeviceArray<ActiveRuntime, F>,
    (n_train, n_features): (usize, usize),
    xq: &DeviceArray<ActiveRuntime, F>,
    n_query: usize,
    k: usize,
    tile_rows: usize,
    metric: Metric,
    val_handle: &Handle,
    idx_handle: &Handle,
) -> Result<(), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let mut start = 0usize;
    while start < n_query {
        let rows = tile_rows.min(n_query - start);

        // Zero-copy row-slice views of the query matrix and of both outputs.
        let xq_tile: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(
            xq.handle()
                .clone()
                .offset_start((start * n_features * size_of::<F>()) as u64),
            rows * n_features,
        );
        let out_val: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(
            val_handle
                .clone()
                .offset_start((start * k * size_of::<F>()) as u64),
            rows * k,
        );
        let out_idx: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_raw(
            idx_handle
                .clone()
                .offset_start((start * k * size_of::<u32>()) as u64),
            rows * k,
        );

        let (dist, needs_sqrt) = metric_distance::<F>(
            pool,
            &xq_tile,
            (rows, n_features),
            x_train,
            (n_train, n_features),
            metric,
            None,
        )
        .map_err(AlgoError::Prim)?;

        let (val_view, idx_view) = match top_k::<F>(
            pool,
            &dist,
            rows,
            n_train,
            k,
            needs_sqrt,
            Some(out_val),
            Some(out_idx),
        ) {
            Ok(v) => v,
            Err(e) => {
                dist.release_into(pool);
                return Err(AlgoError::Prim(e));
            }
        };
        // `val_view`/`idx_view` alias the caller-owned full buffers — drop the
        // wrappers WITHOUT releasing, or the pool would hand the live output to
        // a later acquire (WR-07).
        drop(val_view);
        drop(idx_view);

        // Scratch only: return it so the next tile reuses the same buffer (D-11).
        dist.release_into(pool);

        start += rows;
    }
    Ok(())
}
