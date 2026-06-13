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

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::traits::{Fit, KNeighbors};

/// Brute-force k-nearest-neighbor query (NEIGH-01).
///
/// Construct with [`NearestNeighbors::new`] (`n_neighbors`), then [`Fit::fit`]
/// (stores the training matrix; `y` is ignored — this is the unsupervised
/// neighbor index) and [`KNeighbors::kneighbors`]. The fitted training matrix is
/// device-resident (D-03).
pub struct NearestNeighbors<F> {
    /// Default neighbor count used when a caller passes its own `k` to
    /// `kneighbors`; retained for the sklearn-faithful constructor surface.
    n_neighbors: usize,
    /// Device-resident training matrix (`n_train × n_features`, row-major),
    /// `None` until `fit`.
    x_train_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted training geometry `(n_train, n_features)`, `None` until `fit`.
    train_shape_: Option<(usize, usize)>,
}

impl<F> NearestNeighbors<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `NearestNeighbors` with the default `n_neighbors`. The
    /// per-call `k` passed to [`KNeighbors::kneighbors`] overrides it; both are
    /// validated against the fitted `n_train` at query time
    /// ([`AlgoError::InvalidK`]).
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            x_train_: None,
            train_shape_: None,
        }
    }

    /// The configured default neighbor count.
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// The fitted training geometry `(n_train, n_features)`. Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn train_shape(&self) -> Result<(usize, usize), AlgoError> {
        self.train_shape_.ok_or(AlgoError::NotFitted {
            estimator: "nearest_neighbors",
            operation: "train_shape",
        })
    }
}

impl<F> Fit<F> for NearestNeighbors<F>
where
    F: Float + CubeElement + Pod,
{
    /// Store the training matrix `x` (`shape = (n_train, n_features)`). `y` is
    /// ignored — `NearestNeighbors` is the unsupervised neighbor index. Geometry
    /// is validated before the matrix is staged (ASVS V5).
    fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
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

        // Stage a device-resident copy of the training matrix (D-03). A fresh
        // `from_host` round-trip clones the buffer so the estimator owns its
        // training state independently of the caller's input handle.
        let x_host = x.to_host(pool);
        let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_host);

        if let Some(old) = self.x_train_.take() {
            old.release_into(pool);
        }
        self.x_train_ = Some(x_dev);
        self.train_shape_ = Some((n_train, n_features));
        Ok(self)
    }
}

impl<F> KNeighbors<F> for NearestNeighbors<F>
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
