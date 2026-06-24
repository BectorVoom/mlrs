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

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
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
        let (n_train, n_features) = shape;
        validate_geometry(x, shape)?;

        // Stage a device-resident copy of the training matrix (D-03). A fresh
        // `from_host` round-trip clones the buffer so the estimator owns its
        // training state independently of the caller's input handle.
        let x_host = x.to_host(pool);
        let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_host);

        Ok(NearestNeighbors {
            n_neighbors: self.n_neighbors,
            x_train_: Some(x_dev),
            train_shape_: Some((n_train, n_features)),
            _state: PhantomData,
        })
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
    let x_train = x_train.ok_or(AlgoError::NotFitted {
        estimator: "knn",
        operation: "kneighbors",
    })?;
    let (n_train, n_features) = train_shape.ok_or(AlgoError::NotFitted {
        estimator: "knn",
        operation: "kneighbors",
    })?;
    let (n_query, q_features) = shape;

    // --- T-05-08-01 / ASVS V5: validate the untrusted k + query geometry BEFORE
    //     any prim launch. 1 <= k <= n_train (cannot request more neighbors than
    //     training points); the query feature count must match the fitted one. ---
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

    // --- 1. Pairwise SQUARED Euclidean distance Xq × X = n_query × n_train (no
    //        sqrt — top-k selects on the order-preserving squared form,
    //        Pitfall 8). ---
    let dist = distance::<F>(
        pool,
        xq,
        (n_query, n_features),
        x_train,
        (n_train, n_features),
        false,
        None,
    )?;

    // --- 2. Select the k nearest per query row, sqrt at the boundary so the
    //        returned distances are true sqrt-Euclidean (lowest-index tie-break
    //        inherited from the validated top_k prim, 05-02). ---
    let (val_dev, idx_dev_u32) =
        top_k::<F>(pool, &dist, n_query, n_train, k, true, None, None)?;
    dist.release_into(pool);

    // --- 3. u32 → i32 neighbor indices (D-06). Host-cast the small n_query × k
    //        index buffer; the values are training-column indices in
    //        [0, n_train), so the cast is exact. Keep the host u32 copy so the
    //        classifier / regressor gather targets without a second round-trip. ---
    let idx_host: Vec<u32> = idx_dev_u32.to_host(pool);
    idx_dev_u32.release_into(pool);
    let idx_i32: Vec<i32> = idx_host.iter().map(|&u| u as i32).collect();
    let idx_dev_i32: DeviceArray<ActiveRuntime, i32> = DeviceArray::from_host(pool, &idx_i32);

    Ok((val_dev, idx_dev_i32, idx_host))
}
