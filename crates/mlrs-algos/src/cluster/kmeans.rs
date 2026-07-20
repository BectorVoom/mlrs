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
//! The loop is fully DEVICE-resident (the "count synchronizations, not FLOPs"
//! treatment): labels never leave the device inside the loop, and each
//! iteration's host traffic is a few KB. Each iteration:
//!   1. UPDATE the centers as the per-label mean via the row-blocked device
//!      gather ([`centroid_sums_dev`]) — small k×d sums + k counts readback,
//!      host f64 divide. An empty cluster (rare) triggers sklearn's EXACT
//!      `_relocate_empty_clusters_dense` ([`relocate_empty_clusters`]) ranked
//!      by the fused assign's per-row distance buffer (CR-01 / T-05-03-02).
//!   2. ASSIGN every sample to its nearest center — the FUSED device
//!      [`assign_min`] prim (direct per-row squared distance + argmin,
//!      lowest-index tie-break D-02; no n×k distance matrix, no
//!      `row_reduce(Shared)` norm term, no per-row argmin launches).
//!   3. CONVERGENCE — first the STRICT `array_equal(labels, labels_old)` BREAK
//!      via the device [`labels_changed`] count (sklearn breaks the moment the
//!      labeling stops changing, BEFORE the tol check — Pitfall 6); then
//!      `center_shift_tot <= tol_scaled` where `tol_scaled =
//!      mean(var(X, axis=0)) · tol` (computed on-device by
//!      [`feature_mean_var`]; `tol` default `1e-4`). `max_iter = 300`.
//!   4. No post-loop assignment pass is needed: every exit path leaves the
//!      labels written against the final adopted centers, so a re-assign
//!      (sklearn's post-loop `_labels_inertia`) would reproduce them exactly.
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
use mlrs_backend::prims::kmeans::{
    assign_min, centroid_sums_dev, feature_mean_var, gather_rows_device, kmeanspp_sample,
    inertia_rows_device, labels_changed, relocate_empty_clusters, row_sqnorms, sum_device,
};
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
    /// (`k × d`) via the FUSED device [`assign_min`] prim (direct per-row
    /// argmin, lowest-index tie-break D-02) into caller-owned DEVICE buffers:
    /// `labels` (`u32`, length `n`) and `dist` (the winning squared distance —
    /// the per-row inertia term). No readback — the Lloyd loop stays
    /// launch-only; `predict_labels` reads the labels back at its boundary.
    #[allow(clippy::too_many_arguments)]
    fn assign_dev(
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        n: usize,
        d: usize,
        centers: &DeviceArray<ActiveRuntime, F>,
        k: usize,
        labels: &DeviceArray<ActiveRuntime, u32>,
        dist: &DeviceArray<ActiveRuntime, F>,
        xnorm: Option<&DeviceArray<ActiveRuntime, F>>,
    ) -> Result<(), PrimError> {
        assign_min::<F>(pool, x, centers, labels, dist, xnorm, n, d, k)
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
        //     k-means++ D²-weighted host sampler (D-09a, n_init=1 D-09b). The
        //     `centers_host` f64 mirror feeds the per-iteration center-shift
        //     check WITHOUT reading the centers back each iteration. ---
        let (mut centers, mut centers_host): (DeviceArray<ActiveRuntime, F>, Vec<f64>) =
            if let Some(init) = &self.init {
                if init.len() != k * n_features {
                    return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                        operand: "init",
                        rows: k,
                        cols: n_features,
                        len: init.len(),
                    }));
                }
                let host = init.iter().map(|&v| host_to_f64(v)).collect();
                (DeviceArray::from_host(pool, init), host)
            } else {
                let idx = kmeanspp_sample::<F>(pool, x, n_samples, n_features, k, self.seed)?;
                // Gather the chosen sample rows into the k × d init buffer ON
                // the device (no full `x` readback); mirror the small result.
                let idx_u32: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
                let dev = gather_rows_device::<F>(pool, x, &idx_u32, n_samples, n_features)?;
                let host = dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
                (dev, host)
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
        //     the raw tol by the mean per-feature variance; computed by the
        //     two-pass blocked DEVICE column reduction (only tiny partials are
        //     read back — never the n × d sample matrix). ---
        let tol_scaled = self.tol * feature_mean_var::<F>(pool, x, n_samples, n_features)?;

        // --- Device work buffers for the launch-only Lloyd loop: u32 labels
        //     (current + previous, swapped each iteration) and the per-row
        //     squared distance to the assigned center (written by every fused
        //     assign — it doubles as the relocation ranking AND the inertia
        //     rows, so neither needs an extra pass). ---
        let elem_u32 = size_of::<u32>();
        let mut labels_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(
            pool.acquire(n_samples * elem_u32),
            n_samples,
        );
        let mut labels_old_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(
            pool.acquire(n_samples * elem_u32),
            n_samples,
        );
        let dist_dev = DeviceArray::<ActiveRuntime, F>::from_raw(
            pool.acquire(n_samples * size_of::<F>()),
            n_samples,
        );

        // ‖x_i‖², computed ONCE per fit for the GEMM assignment path (the prim
        // ignores it on the direct path — a single tiny launch either way).
        let xnorm = row_sqnorms::<F>(pool, x, n_samples, n_features)?;

        // --- Lloyd loop (Pitfall 6: strict label-equality break BEFORE the tol
        //     check), fully DEVICE-resident: per iteration the host sees only
        //     the k × d centroid sums + k counts and the per-block
        //     changed-label counts (a few KB) — never an O(n) buffer. ---
        // KM_PROFILE=1: per-phase wall-clock attribution (laps are delimited by
        // the loop's natural readback sync points, so kernel time lands in the
        // phase whose readback drains it — attribution only, like RF_PROFILE).
        let profile = std::env::var("KM_PROFILE").is_ok();
        let mut t_sums = 0.0_f64;
        let mut t_host = 0.0_f64;
        let mut t_assign = 0.0_f64;
        let mut iters_run = 0usize;

        // Host `x` copy, materialized ONLY if some iteration hits the rare
        // empty-cluster relocation, then reused across later relocations (`x`
        // is immutable — measured 12ms/iteration of repeated O(n·d) readback
        // on a relocation-heavy ladder config without the cache).
        let mut x_host_cache: Option<Vec<F>> = None;

        Self::assign_dev(
            pool, x, n_samples, n_features, &centers, k, &labels_dev, &dist_dev, Some(&xnorm),
        )?;
        for _iter in 0..self.max_iter {
            iters_run += 1;
            let lap0 = std::time::Instant::now();
            // UPDATE: per-centroid sums + counts via the row-blocked device
            // gather (the only per-iteration readback of the update phase).
            let (mut sums_f64, mut counts_i64) =
                centroid_sums_dev::<F>(pool, x, &labels_dev, n_samples, n_features, k)?;
            if profile {
                t_sums += lap0.elapsed().as_secs_f64();
            }
            let lap1 = std::time::Instant::now();

            // RARE path: an empty cluster triggers sklearn's exact relocation
            // (CR-01 / T-05-03-02), which ranks samples by their squared
            // distance to the assigned center — `dist_dev` holds exactly that
            // (written by the assign that produced `labels_dev`). Only this
            // branch ever reads an O(n) buffer back.
            if counts_i64.iter().any(|&c| c == 0) {
                if x_host_cache.is_none() {
                    x_host_cache = Some(x.to_host(pool));
                }
                let labels_host: Vec<u32> = labels_dev.to_host(pool);
                let dist_host: Vec<f64> = dist_dev
                    .to_host(pool)
                    .iter()
                    .map(|&v| host_to_f64(v))
                    .collect();
                relocate_empty_clusters::<F>(
                    &mut sums_f64,
                    &mut counts_i64,
                    x_host_cache.as_ref().expect("cached above"),
                    &labels_host,
                    &dist_host,
                    n_samples,
                    n_features,
                    k,
                )?;
            }

            // Mean divide (f64, matching lloyd_update's finalize) + the center
            // shift against the f64 host mirror of the OLD centers.
            let mut new_centers_host = vec![0.0_f64; k * n_features];
            for c in 0..k {
                // Post-relocation every cluster has count >= 1 (the relocation
                // helper guarantees it or errors).
                debug_assert!(
                    counts_i64[c] > 0,
                    "post-relocation cluster {c} has non-positive count {}",
                    counts_i64[c]
                );
                if counts_i64[c] > 0 {
                    let inv = 1.0_f64 / counts_i64[c] as f64;
                    for j in 0..n_features {
                        new_centers_host[c * n_features + j] = sums_f64[c * n_features + j] * inv;
                    }
                }
            }
            // center_shift_tot = Σ ‖new_center_c − old_center_c‖² (host pass
            // over the tiny k × d mirrors). Consulted AFTER the strict check.
            let mut shift = 0.0_f64;
            for i in 0..k * n_features {
                let diff = new_centers_host[i] - centers_host[i];
                shift += diff * diff;
            }

            let new_f: Vec<F> = new_centers_host
                .iter()
                .map(|&v| f64_to_host::<F>(v))
                .collect();
            centers.release_into(pool);
            centers = DeviceArray::from_host(pool, &new_f);
            centers_host = new_centers_host;

            if profile {
                t_host += lap1.elapsed().as_secs_f64();
            }
            let lap2 = std::time::Instant::now();

            // ASSIGN to the new centers (previous labels kept in the swapped
            // buffer for the strict check).
            std::mem::swap(&mut labels_dev, &mut labels_old_dev);
            Self::assign_dev(
                pool, x, n_samples, n_features, &centers, k, &labels_dev, &dist_dev, Some(&xnorm),
            )?;

            // STRICT array_equal break FIRST (Pitfall 6) — the labeling stopped
            // changing, so sklearn breaks before measuring the center shift.
            let changed = labels_changed(pool, &labels_dev, &labels_old_dev, n_samples)?;
            if profile {
                t_assign += lap2.elapsed().as_secs_f64();
            }
            if changed == 0 {
                break;
            }
            if shift <= tol_scaled {
                break;
            }
        }

        if profile {
            eprintln!(
                "KM_PROFILE n={n_samples} d={n_features} k={k}: iters={iters_run} \
                 sums+readback={t_sums:.4}s host+upload={t_host:.4}s \
                 assign+changed={t_assign:.4}s"
            );
        }

        // NOTE: no post-loop assignment pass is needed (the old code's Pitfall-6
        // re-assign): EVERY exit path above — strict break, tol break, max_iter
        // exhaustion, and the max_iter == 0 degenerate — leaves `labels_dev` /
        // `dist_dev` written by an assign against the FINAL adopted `centers`,
        // and assignment is deterministic, so re-assigning reproduces the same
        // labels sklearn's post-loop `_labels_inertia` would.

        // --- inertia_ = Σ per-row squared distance to the assigned center.
        //     Recompute the rows with the DIRECT gather first (the GEMM staging
        //     distances rank correctly but their f32 cancellation noise exceeds
        //     the 1e-5 oracle tolerance when summed), then the blocked device
        //     sum — still no O(n) readback. ---
        inertia_rows_device::<F>(
            pool, x, &centers, &labels_dev, &dist_dev, n_samples, n_features,
        )?;
        let inertia_val =
            f64_to_host::<F>(sum_device::<F>(pool, &dist_dev, n_samples)?);

        // --- Store labels as i32 (D-06: the u32 prim labels widen to the i32
        //     trait surface; KMeans labels are non-negative). One boundary
        //     readback at the end of fit — never inside the loop. ---
        let labels_u32: Vec<u32> = labels_dev.to_host(pool);
        let labels_i32: Vec<i32> = labels_u32.iter().map(|&l| l as i32).collect();
        let labels_dev_i32: DeviceArray<ActiveRuntime, i32> =
            DeviceArray::from_host(pool, &labels_i32);

        // Return the transient loop buffers to the pool (FOUND-05).
        labels_dev.release_into(pool);
        labels_old_dev.release_into(pool);
        dist_dev.release_into(pool);
        xnorm.release_into(pool);
        let labels_dev = labels_dev_i32;

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
        // D-08: KMeans.predict returns INTEGER labels, not an F target) via the
        // fused device assign; one boundary readback for the u32 → i32 widening.
        let labels_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(
            pool.acquire(n_samples * size_of::<u32>()),
            n_samples,
        );
        let dist_dev = DeviceArray::<ActiveRuntime, F>::from_raw(
            pool.acquire(n_samples * size_of::<F>()),
            n_samples,
        );
        let xnorm = row_sqnorms::<F>(pool, x, n_samples, n_features)?;
        Self::assign_dev(
            pool,
            x,
            n_samples,
            n_features,
            centers,
            self.n_clusters,
            &labels_dev,
            &dist_dev,
            Some(&xnorm),
        )?;
        xnorm.release_into(pool);
        let labels: Vec<u32> = labels_dev.to_host(pool);
        labels_dev.release_into(pool);
        dist_dev.release_into(pool);
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
