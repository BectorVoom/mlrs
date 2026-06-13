//! `KMeans` (CLUSTER-01) — k-means++ initialization + the Lloyd iteration,
//! matching `sklearn.cluster.KMeans` up to a label permutation (D-09).
//!
//! ## Init: k-means++ default, INJECTED for the oracle (D-09)
//! By default `fit` draws the `k` initial centers with the validated
//! [`kmeanspp_sample`] D²-weighted host-seeded sampler (D-09a, `n_init = 1` per
//! D-09b). For the deterministic oracle, the caller may INJECT a fixed
//! `k × d` init array via [`KMeans::with_init`] (D-09) so both mlrs and sklearn
//! run Lloyd from the SAME starting centers and converge to the same partition
//! (up to a label permutation, compared with `best_match_accuracy`).
//!
//! ## The Lloyd loop reproduces sklearn's strict-OR-tol convergence (Pitfall 6)
//! Each iteration:
//!   1. ASSIGN every sample to its nearest center — `distance(X, centers,
//!      sqrt=false)` (the Phase-2 prim, squared Euclidean, no boundary sqrt)
//!      then [`argmin_rows`] (lowest-index tie-break, D-02).
//!   2. UPDATE the centers as the per-label mean via [`lloyd_update`] (the
//!      Phase-5 prim — empty-cluster relocation lives INSIDE the prim, never a
//!      divide-by-zero NaN, T-05-03-02).
//!   3. CONVERGENCE — first the STRICT `array_equal(labels, labels_old)` BREAK
//!      (sklearn breaks the moment the labeling stops changing, BEFORE the tol
//!      check — Pitfall 6); then `center_shift_tot <= tol_scaled` where
//!      `tol_scaled = mean(var(X, axis=0)) · tol` (sklearn scales the raw `tol`
//!      by the mean feature variance; `tol` default `1e-4`). `max_iter = 300`.
//!   4. If the loop ends WITHOUT a strict label-equality break (it hit the tol
//!      or `max_iter`), run ONE FINAL assignment pass so `labels_` reflects the
//!      final centers (sklearn's post-loop `_labels_inertia` pass — Pitfall 6).
//!
//! ## Stored fitted state (device-resident, D-03)
//! `cluster_centers_` (`k × d`, `F`) and `labels_` (`n`, `i32` — D-06 the
//! `u32`→`i32` idiom; KMeans labels are non-negative but the trait surface is
//! `i32` so DBSCAN's `-1` noise shares it) plus the scalar `inertia_` (`F`).
//!
//! ## Discrete-output surface: PredictLabels, NOT Predict<F> (D-08)
//! `KMeans.predict` returns INTEGER cluster ids, so it implements
//! [`PredictLabels`](crate::traits::PredictLabels) (i32 labels), NOT the
//! continuous-target [`Predict`](crate::traits::Predict) (which returns an
//! `F` buffer — that is the regressor surface). A new sample is assigned to its
//! nearest fitted center via the same `distance` + `argmin_rows` path.
//!
//! ## Validate the untrusted hyperparameter BEFORE any launch (ASVS V5)
//! `fit` rejects `n_clusters < 1` or `n_clusters > n_samples` with
//! [`AlgoError::InvalidK`] BEFORE any prim launch (T-05-07-01) — a tampered `k`
//! never becomes an out-of-bounds device gather.
//!
//! Tests live in `crates/mlrs-algos/tests/kmeans_test.rs` (AGENTS.md §2), never
//! an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::kmeans::{inertia, kmeanspp_sample, lloyd_update};
use mlrs_backend::prims::reduce::argmin_rows;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::traits::{Fit, PredictLabels};

/// sklearn's default `max_iter` for `KMeans` (Pitfall 6).
const DEFAULT_MAX_ITER: usize = 300;

/// K-means clustering (CLUSTER-01) fitted by k-means++ init + the Lloyd loop.
///
/// Construct with [`KMeans::new`] (`n_clusters`, `seed`) for the default
/// k-means++ init, or [`KMeans::with_init`] to INJECT a fixed `k × d` init (the
/// deterministic oracle, D-09). Then [`Fit::fit`] and
/// [`PredictLabels::predict_labels`]. Fitted `cluster_centers_` / `labels_` are
/// device-resident (D-03); the host accessors materialize them on demand.
pub struct KMeans<F> {
    /// Number of clusters `k`. Validated `1 <= k <= n_samples` at `fit` time
    /// → [`AlgoError::InvalidK`] BEFORE any launch (T-05-07-01).
    n_clusters: usize,
    /// Maximum Lloyd iterations (sklearn default `300`, Pitfall 6).
    max_iter: usize,
    /// Convergence tolerance `tol` (sklearn default `1e-4`); the effective
    /// threshold is `tol · mean(var(X, axis=0))` (Pitfall 6).
    tol: f64,
    /// Seed for the k-means++ host PRNG (used only when `init` is `None`).
    seed: u64,
    /// OPTIONAL injected `k × d` row-major init centers (D-09). When `Some`,
    /// Lloyd starts from these EXACT centers (the deterministic oracle); when
    /// `None`, [`kmeanspp_sample`] draws the init.
    init: Option<Vec<F>>,
    /// Fitted `k × d` cluster centers, device-resident, `None` until `fit`.
    cluster_centers_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted length-`n` integer labels (`i32`, D-06), device-resident, `None`
    /// until `fit`.
    labels_: Option<DeviceArray<ActiveRuntime, i32>>,
    /// Fitted inertia `Σ ‖X_i − centers[labels_i]‖²` (scalar), `None` until
    /// `fit`.
    inertia_: Option<F>,
    /// Fitted `n_features` (set at `fit`), used to validate `predict_labels`
    /// geometry against the trained centers.
    n_features_: usize,
}

impl<F> KMeans<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `KMeans` with `n_clusters` and the k-means++ PRNG
    /// `seed` (default init). `max_iter = 300`, `tol = 1e-4` (sklearn defaults,
    /// Pitfall 6). A bad `n_clusters` is rejected at `fit` time
    /// ([`AlgoError::InvalidK`]).
    pub fn new(n_clusters: usize, seed: u64) -> Self {
        Self {
            n_clusters,
            max_iter: DEFAULT_MAX_ITER,
            tol: 1e-4,
            seed,
            init: None,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_features_: 0,
        }
    }

    /// Create an unfitted `KMeans` with an INJECTED `k × d` row-major init array
    /// (D-09 — the deterministic oracle: both mlrs and sklearn run Lloyd from
    /// the SAME centers). `init.len()` must equal `n_clusters · n_features`;
    /// this is checked at `fit` time against the data geometry.
    pub fn with_init(n_clusters: usize, init: Vec<F>) -> Self {
        Self {
            n_clusters,
            max_iter: DEFAULT_MAX_ITER,
            tol: 1e-4,
            seed: 0,
            init: Some(init),
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_features_: 0,
        }
    }

    /// Host copy of the fitted `cluster_centers_` (`k × d` row-major). Errors
    /// with [`AlgoError::NotFitted`] before `fit`.
    pub fn cluster_centers(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.cluster_centers_
            .as_ref()
            .map(|c| c.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "kmeans",
                operation: "cluster_centers_",
            })
    }

    /// Host copy of the fitted `labels_` (length `n`, `i32`). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn labels(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<i32>, AlgoError> {
        self.labels_
            .as_ref()
            .map(|l| l.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "kmeans",
                operation: "labels_",
            })
    }

    /// The fitted `inertia_` scalar. Errors with [`AlgoError::NotFitted`] before
    /// `fit`.
    pub fn inertia(&self) -> Result<F, AlgoError> {
        self.inertia_.ok_or(AlgoError::NotFitted {
            estimator: "kmeans",
            operation: "inertia_",
        })
    }

    /// Assign each row of `x` (`n × d`) to its nearest center in `centers`
    /// (`k × d`) via the Phase-2 `distance(sqrt=false)` + per-row `argmin`
    /// (lowest-index tie-break, D-02). Shared by the Lloyd loop and
    /// `predict_labels`. Returns length-`n` `u32` labels (each in `0..k`).
    fn assign(
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        n: usize,
        d: usize,
        centers: &DeviceArray<ActiveRuntime, F>,
        k: usize,
    ) -> Result<Vec<u32>, PrimError> {
        // n × k squared-distance matrix (no boundary sqrt — nearest is
        // sqrt-monotonic), then the lowest-index per-row argmin.
        let dmat = distance::<F>(pool, x, (n, d), centers, (k, d), false, None)?;
        let labels = argmin_rows::<F>(pool, &dmat, n, k)?;
        dmat.release_into(pool);
        Ok(labels)
    }
}

impl<F> Fit<F> for KMeans<F>
where
    F: Float + CubeElement + Pod,
{
    fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (n_samples, n_features) = shape;
        let k = self.n_clusters;

        // --- T-05-07-01 / ASVS V5: validate the untrusted hyperparameter +
        //     geometry BEFORE any prim launch. A tampered k (k < 1 or
        //     k > n_samples) would otherwise drive an out-of-bounds device
        //     gather in lloyd_update / argmin. ---
        if k < 1 || k > n_samples {
            return Err(AlgoError::InvalidK {
                estimator: "kmeans",
                k,
                n_samples,
            });
        }
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }

        // --- Init centers: injected (D-09, the deterministic oracle) or the
        //     k-means++ D²-weighted host sampler (D-09a, n_init=1 D-09b). ---
        let mut centers: DeviceArray<ActiveRuntime, F> = if let Some(init) = &self.init {
            if init.len() != k * n_features {
                return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                    operand: "init",
                    rows: k,
                    cols: n_features,
                    len: init.len(),
                }));
            }
            DeviceArray::from_host(pool, init)
        } else {
            let idx = kmeanspp_sample::<F>(pool, x, n_samples, n_features, k, self.seed)?;
            // Gather the chosen sample rows into a k × d init buffer.
            let x_host = x.to_host(pool);
            let mut init_host: Vec<F> = vec![F::from_int(0i64); k * n_features];
            for (c, &i) in idx.iter().enumerate() {
                init_host[c * n_features..(c + 1) * n_features]
                    .copy_from_slice(&x_host[i * n_features..(i + 1) * n_features]);
            }
            DeviceArray::from_host(pool, &init_host)
        };

        // --- tol_scaled = tol · mean(var(X, axis=0)) (Pitfall 6). sklearn scales
        //     the raw tol by the mean per-feature variance; computed host-side on
        //     the tiny n-vectors (the heavy assign/update stay on-device). ---
        let x_host: Vec<F> = x.to_host(pool);
        let tol_scaled = {
            let inv_n = 1.0_f64 / n_samples as f64;
            let mut mean = vec![0.0f64; n_features];
            for r in 0..n_samples {
                for c in 0..n_features {
                    mean[c] += host_to_f64(x_host[r * n_features + c]);
                }
            }
            for m in mean.iter_mut() {
                *m *= inv_n;
            }
            let mut var_sum = 0.0f64;
            for r in 0..n_samples {
                for c in 0..n_features {
                    let diff = host_to_f64(x_host[r * n_features + c]) - mean[c];
                    var_sum += diff * diff;
                }
            }
            // mean over features of var(X, axis=0) (population variance, ddof=0 —
            // sklearn uses np.mean(np.var(X, axis=0))).
            let mean_var = (var_sum * inv_n) / n_features as f64;
            self.tol * mean_var
        };

        // --- Lloyd loop (Pitfall 6: strict label-equality break BEFORE the tol
        //     check; one final assignment pass if it did NOT strict-converge). ---
        let mut labels = Self::assign(pool, x, n_samples, n_features, &centers, k)?;
        let mut strict_converged = false;
        for _iter in 0..self.max_iter {
            // UPDATE: per-label mean (empty-cluster relocation inside the prim).
            let new_centers =
                lloyd_update::<F>(pool, x, &labels, n_samples, n_features, k)?;

            // ASSIGN to the new centers.
            let new_labels =
                Self::assign(pool, x, n_samples, n_features, &new_centers, k)?;

            // STRICT array_equal break FIRST (Pitfall 6) — the labeling stopped
            // changing, so sklearn breaks before measuring the center shift.
            if new_labels == labels {
                centers.release_into(pool);
                centers = new_centers;
                labels = new_labels;
                strict_converged = true;
                break;
            }

            // center_shift_tot = Σ ‖new_center_c − old_center_c‖² (host pass over
            // the tiny k × d centers). Break when it falls to/under tol_scaled.
            let old_host = centers.to_host(pool);
            let new_host = new_centers.to_host(pool);
            let mut shift = 0.0f64;
            for i in 0..k * n_features {
                let diff = host_to_f64(new_host[i]) - host_to_f64(old_host[i]);
                shift += diff * diff;
            }

            centers.release_into(pool);
            centers = new_centers;
            labels = new_labels;

            if shift <= tol_scaled {
                break;
            }
        }

        // --- One FINAL assignment pass if we did NOT strict-converge (the loop
        //     hit the tol threshold or max_iter): labels_ must reflect the final
        //     centers (sklearn's post-loop _labels_inertia, Pitfall 6). ---
        if !strict_converged {
            labels = Self::assign(pool, x, n_samples, n_features, &centers, k)?;
        }

        // --- inertia_ via the Phase-5 prim (Σ squared dist to assigned center). ---
        let inertia_val = inertia::<F>(pool, x, &centers, &labels, n_samples, n_features)?;

        // --- Store labels as i32 (D-06: the u32 prim labels widen to the i32
        //     trait surface; KMeans labels are non-negative). ---
        let labels_i32: Vec<i32> = labels.iter().map(|&l| l as i32).collect();
        let labels_dev: DeviceArray<ActiveRuntime, i32> = DeviceArray::from_host(pool, &labels_i32);

        self.cluster_centers_ = Some(centers);
        self.labels_ = Some(labels_dev);
        self.inertia_ = Some(inertia_val);
        self.n_features_ = n_features;
        Ok(self)
    }
}

impl<F> PredictLabels<F> for KMeans<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let (n_samples, n_features) = shape;

        let centers = self.cluster_centers_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "kmeans",
            operation: "predict_labels",
        })?;

        // --- ASVS V5: geometry + fitted-n_features consistency BEFORE launch. ---
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        if n_features != self.n_features_ {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: self.n_features_,
            }));
        }

        // Assign new points to the fitted centers (nearest-centroid → i32 label,
        // D-08: KMeans.predict returns INTEGER labels, not an F target).
        let labels = Self::assign(pool, x, n_samples, n_features, centers, self.n_clusters)?;
        let labels_i32: Vec<i32> = labels.iter().map(|&l| l as i32).collect();
        Ok(DeviceArray::from_host(pool, &labels_i32))
    }
}

impl<F> KMeans<F>
where
    F: Float + CubeElement + Pod,
{
    /// Convenience `fit_predict` (sklearn `ClusterMixin`): fit to `x` then return
    /// the fitted `labels_` as a fresh device-resident `i32` buffer. Equivalent
    /// to `fit` followed by reading `labels_`.
    pub fn fit_predict(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        self.fit(pool, x, None, shape)?;
        let labels = self.labels(pool)?;
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors the
/// `ridge.rs` helper).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kmeans is f32/f64 only"),
    }
}
