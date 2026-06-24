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
//!      Phase-5 prim), passing the per-sample distance-to-assigned-center
//!      ([`inertia_rows_host`]) so the prim can run sklearn's EXACT
//!      `_relocate_empty_clusters_dense` (relocate an empty cluster to the
//!      globally-farthest sample, decrementing its donor — never a
//!      divide-by-zero NaN, CR-01 / T-05-03-02).
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
//! [`PredictLabels`](crate::typestate::PredictLabels) (i32 labels), NOT the
//! continuous-target [`Predict`](crate::typestate::Predict) (which returns an
//! `F` buffer — that is the regressor surface). A new sample is assigned to its
//! nearest fitted center via the same `distance` + `argmin_rows` path.
//!
//! ## Builder-fronted construction (Phase 16 retrofit, D-01/D-08)
//! Construct with the zero-arg [`KMeans::new`] (sklearn defaults) or the WIDE
//! [`KMeansBuilder`], which fully folds the THREE legacy constructors
//! (`new(n_clusters, seed)`, `with_init(n_clusters, init)`, and
//! `with_opts(n_clusters, seed, max_iter, tol)`) into setters:
//! `.n_clusters(usize)`/`.seed(u64)`/`.max_iter(usize)`/`.tol(f64)` plus the
//! injected-init `.init(Option<Vec<f64>>)` (the `with_init` replacement). The
//! `init` setter is the wide-builder `Option`-of-data shape: the builder stores
//! the init as `Option<Vec<f64>>` and narrows it to `Vec<F>` in `build::<F>()`
//! (A5 — all setters are `f64`-typed, the `f64 → F` narrowing happens once in
//! `build`). All hyperparameter / init validation is data-DEPENDENT (it depends
//! on `n_samples`/`n_features`), so `build()` is infallible-but-typed (kept for
//! `build_err_to_py` family uniformity); the geometry / `InvalidK` / injected-init
//! dimension checks stay in `fit` (D-03 byte-identical).
//!
//! ## Validate the untrusted hyperparameter BEFORE any launch (ASVS V5)
//! `fit` rejects `n_clusters < 1` or `n_clusters > n_samples` with
//! [`AlgoError::InvalidK`] BEFORE any prim launch (T-05-07-01) — a tampered `k`
//! never becomes an out-of-bounds device gather.
//!
//! Tests live in `crates/mlrs-algos/tests/kmeans_test.rs` (AGENTS.md §2), never
//! an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::kmeans::{inertia, inertia_rows_host, kmeanspp_sample, lloyd_update};
use mlrs_backend::prims::reduce::argmin_rows;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, Unfit};

/// sklearn's default `max_iter` for `KMeans` (Pitfall 6).
const DEFAULT_MAX_ITER: usize = 300;

/// K-means clustering (CLUSTER-01) fitted by k-means++ init + the Lloyd loop.
///
/// Construct with the zero-arg [`KMeans::new`] (sklearn defaults: `n_clusters = 8`,
/// `max_iter = 300`, `tol = 1e-4`, `seed = 0`, default k-means++ init) or the WIDE
/// [`KMeans::builder`] (the three legacy constructors `new`/`with_init`/`with_opts`
/// are fully folded into the `.n_clusters`/`.seed`/`.max_iter`/`.tol`/`.init`
/// setters; `.init(Some(..))` INJECTS a fixed `k × d` init — the deterministic
/// oracle, D-09). Then the consuming [`Fit::fit`] (returns the `Fitted`-tagged
/// sibling) and [`PredictLabels::predict_labels`]. Fitted `cluster_centers_` /
/// `labels_` are device-resident (D-03); the host accessors materialize them on
/// demand and exist ONLY on `KMeans<F, Fitted>` (the compile-time typestate
/// replaces the old runtime `NotFitted` guard, D-03).
pub struct KMeans<F, S = Unfit> {
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
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> KMeans<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfitted `KMeans` with sklearn's `KMeans` defaults
    /// (`n_clusters = 8`, `max_iter = 300`, `tol = 1e-4`, `seed = 0`, default
    /// k-means++ init — `init = None`) directly in the `Unfit` state. SINGLE
    /// source of truth for the defaults (D-08): the builder `Default` re-derives
    /// via [`KMeans::into_builder`]. A bad `n_clusters` (or injected init) is
    /// rejected at `fit` time ([`AlgoError::InvalidK`] / dimension mismatch).
    pub fn new() -> Self {
        Self {
            n_clusters: 8,
            max_iter: DEFAULT_MAX_ITER,
            tol: 1e-4,
            seed: 0,
            init: None,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_features_: 0,
            _state: PhantomData,
        }
    }

    /// Start building a `KMeans` from sklearn's defaults (D-08 single source).
    pub fn builder() -> KMeansBuilder {
        KMeansBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter (the injected `init` is promoted `Vec<F> → Vec<f64>` so the
    /// builder stays non-generic, A5). Used by [`KMeansBuilder::default`] to
    /// re-derive the defaults from [`KMeans::new`] (D-08).
    pub fn into_builder(self) -> KMeansBuilder {
        KMeansBuilder {
            n_clusters: self.n_clusters,
            seed: self.seed,
            max_iter: self.max_iter,
            tol: self.tol,
            init: self
                .init
                .map(|v| v.iter().map(|&e| host_to_f64(e)).collect()),
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `cluster_centers_`/`labels_`/`inertia_` are excluded — `None` in any
    /// `Unfit` value). Used by the defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_clusters == other.n_clusters
            && self.seed == other.seed
            && self.max_iter == other.max_iter
            && self.tol == other.tol
            && match (&self.init, &other.init) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.len() == b.len()
                        && a.iter()
                            .zip(b.iter())
                            .all(|(&x, &y)| host_to_f64(x) == host_to_f64(y))
                }
                _ => false,
            }
    }
}

impl<F> Default for KMeans<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`KMeans`] (D-01) — the WIDE builder that subsumes the three legacy
/// constructors (`new`/`with_init`/`with_opts`). Scalar setters are `f64`-typed
/// (A5); the injected `init` is stored as `Option<Vec<f64>>` and narrowed to
/// `Vec<F>` in [`KMeansBuilder::build`]. `Default` re-derives the sklearn defaults
/// from [`KMeans::new`] (D-08 single source).
#[derive(Debug, Clone)]
pub struct KMeansBuilder {
    n_clusters: usize,
    seed: u64,
    max_iter: usize,
    tol: f64,
    init: Option<Vec<f64>>,
}

impl Default for KMeansBuilder {
    /// Re-derive the sklearn defaults from [`KMeans::new`] (D-08 single source).
    /// `f64` is pinned only to read the F-independent scalar defaults — the
    /// builder is non-generic, so the choice of `F` here is irrelevant.
    fn default() -> Self {
        KMeans::<f64, Unfit>::new().into_builder()
    }
}

impl KMeansBuilder {
    /// Set the number of clusters `k` (sklearn default `8`). Validated
    /// `1 ≤ k ≤ n_samples` at `fit` (data-DEPENDENT → stays in `fit`).
    pub fn n_clusters(mut self, v: usize) -> Self {
        self.n_clusters = v;
        self
    }

    /// Set the k-means++ host-PRNG seed (used only when `init` is `None`).
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    /// Set the maximum Lloyd iteration cap (sklearn default `300`, Pitfall 6).
    pub fn max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the unscaled convergence tolerance `tol` (sklearn default `1e-4`;
    /// scaled by the mean feature variance at `fit`, Pitfall 6).
    pub fn tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }

    /// INJECT a fixed `k × d` row-major init array (D-09 — the deterministic
    /// oracle: both mlrs and sklearn run Lloyd from the SAME centers), or `None`
    /// for the default k-means++ init. The wide-builder `Option`-of-data setter
    /// shape (the `with_init` replacement). Stored as `f64` and narrowed to `F` in
    /// [`build`](Self::build); its `len() == k · n_features` is checked at `fit`
    /// against the data geometry.
    pub fn init(mut self, v: Option<Vec<f64>>) -> Self {
        self.init = v;
        self
    }

    /// Build the (unfit) estimator, narrowing the stored `f64` scalars + the
    /// injected `init` to the target float `F` (A5). KMeans has NO data-INDEPENDENT
    /// hyperparameter to validate at construction: `1 ≤ n_clusters ≤ n_samples`,
    /// the injected-init dimension (`len == k · n_features`), and the geometry are
    /// all data-DEPENDENT and stay in [`Fit::fit`] (D-03 byte-identical). The
    /// `Result` is kept for family uniformity with the other Phase-16 builders so
    /// the `build_err_to_py` PyO3 mapper is shape-identical.
    pub fn build<F>(self) -> Result<KMeans<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        Ok(KMeans {
            n_clusters: self.n_clusters,
            max_iter: self.max_iter,
            tol: self.tol,
            seed: self.seed,
            init: self
                .init
                .map(|v| v.iter().map(|&e| f64_to_host::<F>(e)).collect()),
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_features_: 0,
            _state: PhantomData,
        })
    }
}

impl<F> KMeans<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `cluster_centers_` (`k × d` row-major). `Some` by
    /// construction on the `Fitted` state, so no `NotFitted` branch is needed (the
    /// compile-time typestate replaces the runtime guard, D-03).
    pub fn cluster_centers(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.cluster_centers_
            .as_ref()
            .expect("cluster_centers_ is Some by construction on KMeans<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `labels_` (length `n`, `i32`). `Some` by
    /// construction on the `Fitted` state (D-03).
    pub fn labels(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<i32> {
        self.labels_
            .as_ref()
            .expect("labels_ is Some by construction on KMeans<F, Fitted>")
            .to_host(pool)
    }

    /// The fitted `inertia_` scalar. `Some` by construction on the `Fitted` state
    /// (D-03).
    pub fn inertia(&self) -> F {
        self.inertia_
            .expect("inertia_ is Some by construction on KMeans<F, Fitted>")
    }
}

impl<F, S> KMeans<F, S>
where
    F: Float + CubeElement + Pod,
{
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

impl<F> Fit<F> for KMeans<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = KMeans<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<KMeans<F, Fitted>, AlgoError> {
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
        validate_geometry(x, shape)?;

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

        // --- WR-03: KMeans NON-CONVERGENCE CONTRACT. Unlike Lasso / LogReg (which
        //     surface AlgoError::NotConverged), KMeans matches sklearn's contract:
        //     it NEVER errors on non-convergence — it returns the best-effort fit
        //     after `max_iter` (sklearn only emits a ConvergenceWarning). This is
        //     intentional, not an oversight: KMeans's objective is non-convex and a
        //     `max_iter`-exhausted fit is still a usable clustering. The
        //     `tol_scaled = tol · mean_var` below can be EXACTLY ZERO for a
        //     constant-feature design (mean_var == 0); we deliberately keep that
        //     sklearn `tol == 0` semantics (only the strict label-equality break or
        //     `max_iter` can then stop the loop), and the constant-feature path is
        //     covered by a regression test in `tests/kmeans_test.rs`.
        //
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
            // sklearn empty-cluster relocation (CR-01) needs the per-sample squared
            // distance to the CURRENTLY-assigned center (the same quantity inertia
            // sums) so an empty cluster can take the globally-farthest sample and
            // decrement its donor. Compute it against the CURRENT centers + labels
            // (the assignment that produced the sums lloyd_update is about to form).
            let dist_to_assigned =
                inertia_rows_host::<F>(pool, x, &centers, &labels, n_samples, n_features)?;

            // UPDATE: per-label mean with sklearn-exact empty-cluster relocation
            // (CR-01: relocate to the farthest-from-assigned-center sample, fixing
            // sums + counts + donor — lifted here where labels + centers exist).
            let new_centers = lloyd_update::<F>(
                pool,
                x,
                &labels,
                &dist_to_assigned,
                n_samples,
                n_features,
                k,
            )?;

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

        Ok(KMeans {
            n_clusters: self.n_clusters,
            max_iter: self.max_iter,
            tol: self.tol,
            seed: self.seed,
            init: self.init,
            cluster_centers_: Some(centers),
            labels_: Some(labels_dev),
            inertia_: Some(inertia_val),
            n_features_: n_features,
            _state: PhantomData,
        })
    }
}

impl<F> PredictLabels<F> for KMeans<F, Fitted>
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

        // `Some` by construction on the `Fitted` state (D-03 — the compile-time
        // typestate replaces the old runtime `NotFitted` guard).
        let centers = self
            .cluster_centers_
            .as_ref()
            .expect("cluster_centers_ is Some by construction on KMeans<F, Fitted>");

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

impl<F> KMeans<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// WR-01: Return this KMeans' fitted device buffers (`cluster_centers_` and
    /// `labels_`) to the pool free-list, consuming `self`. `DeviceArray` has no
    /// `Drop` (`device_array.rs`), so a composing estimator that builds a
    /// function-local KMeans (e.g. [`SpectralClustering::fit`]) MUST call this
    /// before the KMeans drops — otherwise the acquired bytes are never returned
    /// and `live_bytes` grows monotonically across re-fits, forfeiting buffer
    /// reuse (the FOUND-05 memory invariant). No-op for buffers still `None`
    /// (an empty fitted value never occurs — `Fitted` always carries both). The
    /// scalar `inertia_` / `n_features_` carry no device memory.
    pub fn release_into(self, pool: &mut BufferPool<ActiveRuntime>) {
        if let Some(centers) = self.cluster_centers_ {
            centers.release_into(pool);
        }
        if let Some(labels) = self.labels_ {
            labels.release_into(pool);
        }
    }
}

impl<F> KMeans<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Convenience `fit_predict` (sklearn `ClusterMixin`): fit to `x` then return
    /// BOTH the `Fitted`-tagged estimator and the fitted `labels_` as a fresh
    /// device-resident `i32` buffer. CONSUMES `self` (the typestate `fit`
    /// transition). Equivalent to `fit` followed by reading `labels_`.
    pub fn fit_predict(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<(KMeans<F, Fitted>, DeviceArray<ActiveRuntime, i32>), AlgoError> {
        let fitted = self.fit(pool, x, None, shape)?;
        let labels = fitted.labels(pool);
        let labels_dev = DeviceArray::from_host(pool, &labels);
        Ok((fitted, labels_dev))
    }
}
