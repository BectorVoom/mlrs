//! `KMeans` (CLUSTER-01) — sklearn's FULL `sklearn.cluster.KMeans` parameter
//! surface over the Lloyd / Elkan iteration, matching sklearn up to a label
//! permutation (D-09).
//!
//! ## The parameter surface
//! Every sklearn ctor parameter is implemented:
//!
//! | parameter | mlrs |
//! |---|---|
//! | `n_clusters` | `1 ≤ k ≤ n_samples`, checked at `fit` (data-DEPENDENT) |
//! | `init` | [`KMeansInit`] — `k-means++` / `random` / an explicit `k × d` array |
//! | `n_init` | [`NInit`] — `'auto'` or a count; restart loop keeps the lowest inertia |
//! | `max_iter` | `300` |
//! | `tol` | `1e-4`, scaled by `mean(var(X, axis=0))` (Pitfall 6) |
//! | `verbose` | accepted + stored; a library crate must not print, so INERT |
//! | `random_state` | `Option<u64>`; `None` → [`DEFAULT_SEED`] (see below) |
//! | `copy_x` | accepted + stored; mlrs never writes into the caller's buffer, so INERT |
//! | `algorithm` | [`KMeansAlgorithm`] — `lloyd` / `elkan` |
//!
//! Fitted: `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`.
//!
//! sklearn's `init` also accepts a CALLABLE. That form has no Rust analogue on
//! this surface; the Python shim evaluates it and passes the result as
//! [`KMeansInit::Array`], which is exactly sklearn's own semantics for it.
//!
//! ### `random_state = None` is a fixed seed, not a global RNG
//! sklearn's `None` draws from the global numpy RNG, so a default-constructed
//! `KMeans` is not reproducible. mlrs pins [`DEFAULT_SEED`] instead: it is the
//! only deviation in the table, and it is deliberate — a library whose default
//! fit is irreproducible cannot be oracle-tested at all.
//!
//! ## Init: k-means++, random, or INJECTED for the oracle (D-09)
//! By default `fit` draws the `k` initial centers with the validated
//! [`kmeanspp_sample`] D²-weighted host-seeded sampler (D-09a); `init='random'`
//! draws `k` distinct rows uniformly without replacement ([`random_sample`]).
//! For the deterministic oracle, the caller INJECTS a fixed `k × d` array via
//! `.init(Some(..))` (D-09) so both mlrs and sklearn run the loop from the SAME
//! starting centers and converge to the same partition (compared up to a label
//! permutation with `best_match_accuracy`).
//!
//! Neither STRING init is bit-comparable to sklearn's: mlrs draws from
//! SplitMix64 and sklearn from a numpy `RandomState` (and sklearn's k-means++
//! additionally runs `2 + log(k)` greedy local trials per center). Same
//! distribution, different stream — so the string inits are oracle-tested on
//! designs where every init reaches the same optimum, not by replaying a
//! fixture. See `crates/mlrs-py/python/tests/test_oracle_kmeans_params.py`.
//!
//! ## `n_init` restarts: lowest inertia wins
//! `fit` runs the whole loop `n_init` times from independently seeded inits
//! (restart `i` uses `random_state + i` — SplitMix64 is built for exactly this
//! seed-by-counter use) and keeps the best, applying sklearn's selection rule
//! verbatim: a later run wins only if it is strictly better AND lands on a
//! genuinely DIFFERENT partition (compared up to a permutation), so a
//! float-rounding tie cannot make the winner depend on iteration order.
//! Every `O(n)` and `O(n·k)` buffer is allocated ONCE per fit and reused
//! across restarts (FOUND-05) — `n_init = 10` costs ten runs, not ten
//! allocations.
//!
//! ## `algorithm`: Lloyd and Elkan compute the SAME fit
//! Elkan is an EXACT acceleration, not an approximation. It carries a per-sample
//! upper bound on the distance to the assigned center and `k` lower bounds to
//! every center, plus the `k × k` half center-center distances, and uses the
//! triangle inequality to skip `(sample, center)` distances that provably cannot
//! win — most of them, once the centers settle. Both arms share this module's
//! update / relocation / convergence code and differ only in the ASSIGN step, so
//! they return identical labels and iteration counts from the same init (pinned
//! by `algorithm_elkan_and_lloyd_agree_f64`). The price is an `n × k` bounds
//! matrix, the only `O(n·k)` allocation in the estimator; see
//! `tests/kmeans_params_perf_test.rs` for when it pays.
//!
//! ## The iteration reproduces sklearn's strict-OR-tol convergence (Pitfall 6)
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
//! `i32` so DBSCAN's `-1` noise shares it) plus the scalars `inertia_` (`F`) and
//! `n_iter_` (the WINNING restart's iteration count, not the sum across
//! restarts).
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
//! `with_opts(n_clusters, seed, max_iter, tol)`) into setters, and now carries a
//! setter per sklearn parameter: `.n_clusters` / `.max_iter` / `.tol` /
//! `.random_state` (with `.seed(u64)` kept as the pre-sklearn spelling) /
//! `.init_method` / `.n_init` / `.algorithm` / `.verbose` / `.copy_x`, plus the
//! narrow `.init(Option<Vec<f64>>)` — the `with_init` replacement, which is just
//! `init_method` restricted to "array or default".
//!
//! Scalar setters are `f64`-typed and the injected init is stored as
//! `KMeansInit<f64>`, narrowed to `KMeansInit<F>` once in `build::<F>()` (A5).
//! Exactly ONE rejection is data-INDEPENDENT enough to live in `build()`:
//! `n_init >= 1`. The unrecognised-string rejections for `init` / `n_init` /
//! `algorithm` happen earlier still, in their `TryFrom<&str>` parses, folded
//! into the SAME `BuildError` so one `build_err_to_py` mapper covers them
//! (D-09). The geometry / `InvalidK` / injected-init dimension checks all depend
//! on `n_samples`/`n_features` and stay in `fit` (D-03 byte-identical).
//!
//! ## Validate the untrusted hyperparameter BEFORE any launch (ASVS V5)
//! `fit` rejects `n_clusters < 1` or `n_clusters > n_samples` with
//! [`AlgoError::InvalidK`] BEFORE any prim launch (T-05-07-01) — a tampered `k`
//! never becomes an out-of-bounds device gather.
//!
//! Tests live in `crates/mlrs-algos/tests/kmeans_test.rs` (the core oracle),
//! `kmeans_params_test.rs` (this parameter surface) and
//! `kmeans_params_perf_test.rs` (the cost of `algorithm` / `n_init` / `init`)
//! per AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::kmeans::{
    assign_min, centroid_sums_dev, elkan_assign_device, elkan_init_bounds, elkan_relax_bounds,
    feature_mean_var, gather_rows_device, inertia_rows_device, kmeanspp_sample, labels_changed,
    random_sample, relocate_empty_clusters, row_sqnorms, sum_device,
};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, Unfit};

/// sklearn's default `max_iter` for `KMeans` (Pitfall 6).
const DEFAULT_MAX_ITER: usize = 300;

/// Seed used when sklearn's `random_state` is `None`. sklearn's `None` means
/// "draw from the global numpy RNG"; mlrs pins a fixed seed instead so a
/// default-constructed fit is reproducible (the v1 contract).
pub const DEFAULT_SEED: u64 = 0;

/// sklearn's `default_n_init` for `KMeans` — the restart count `n_init='auto'`
/// resolves to for the inits that are NOT already deterministic-per-seed
/// (`_check_params_vs_input(X, default_n_init=10)`).
const DEFAULT_N_INIT_AUTO: usize = 10;

/// sklearn's `KMeans(init=...)`: the two STRING strategies plus the explicit
/// `k × d` row-major array (D-09 — the deterministic oracle: both mlrs and
/// sklearn run the loop from the SAME starting centers).
///
/// Generic over the array's element type so ONE enum serves both sides of the
/// A5 narrowing: the non-generic builder holds a `KMeansInit<f64>`, and
/// [`KMeansBuilder::build`] maps it to the estimator's `KMeansInit<F>`.
///
/// sklearn's fourth form — a `callable(X, k, random_state)` — has no Rust
/// analogue on this surface; the Python shim evaluates the callable to an
/// array and passes [`KMeansInit::Array`] (which is exactly sklearn's own
/// semantics: a callable init also forces one run per `_init_centroids`).
#[derive(Debug, Clone, PartialEq)]
pub enum KMeansInit<T = f64> {
    /// D²-weighted k-means++ sampling ([`kmeanspp_sample`]) — sklearn's default.
    KMeansPlusPlus,
    /// `k` DISTINCT sample rows drawn uniformly without replacement.
    Random,
    /// An explicit `k × d` row-major init array. Its `len() == k · n_features`
    /// is checked at `fit` against the data geometry (data-DEPENDENT).
    Array(Vec<T>),
}

impl<T> KMeansInit<T> {
    /// The sklearn `init` string, for diagnostics. The array form has no
    /// sklearn string spelling; it renders as `"array"`.
    pub fn name(&self) -> &'static str {
        match self {
            KMeansInit::KMeansPlusPlus => "k-means++",
            KMeansInit::Random => "random",
            KMeansInit::Array(_) => "array",
        }
    }

    /// Is this the explicit-array form? sklearn forces `n_init = 1` for it
    /// (an explicit init makes every restart identical), which
    /// [`NInit::resolve`] reproduces.
    pub fn is_array(&self) -> bool {
        matches!(self, KMeansInit::Array(_))
    }
}

impl<T> TryFrom<&str> for KMeansInit<T> {
    type Error = BuildError;

    /// Parse the STRING form only — `Array` has no string spelling, so an
    /// unrecognised value is rejected exactly as sklearn's `StrOptions` does.
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "k-means++" => Ok(KMeansInit::KMeansPlusPlus),
            "random" => Ok(KMeansInit::Random),
            other => Err(BuildError::UnknownInit {
                value: other.to_string(),
            }),
        }
    }
}

/// sklearn's `KMeans(n_init=...)`: `'auto'` or an explicit restart count.
///
/// The fit runs the whole Lloyd/Elkan loop `n_init` times from independently
/// seeded inits and KEEPS THE LOWEST-INERTIA result (sklearn's
/// `for i in range(self._n_init)` selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NInit {
    /// sklearn's `'auto'` — see [`NInit::resolve`].
    Auto,
    /// An explicit restart count (`>= 1`; `0` is rejected at `build()`).
    Fixed(usize),
}

impl NInit {
    /// Resolve against the chosen `init`, reproducing sklearn's
    /// `_check_params_vs_input` EXACTLY:
    ///
    /// * `'auto'` → `1` for `k-means++` (its D²-weighted draw is already a good
    ///   init, so sklearn does not pay for restarts) and for an explicit array
    ///   (every restart would be identical);
    /// * `'auto'` → `10` (`default_n_init`) for `'random'`;
    /// * an explicit count passes through, EXCEPT that an explicit array forces
    ///   `1` (sklearn warns and overrides — a library crate must not print, so
    ///   mlrs performs the same override silently).
    pub fn resolve<T>(self, init: &KMeansInit<T>) -> usize {
        if init.is_array() {
            return 1;
        }
        match self {
            NInit::Auto => match init {
                KMeansInit::Random => DEFAULT_N_INIT_AUTO,
                _ => 1,
            },
            NInit::Fixed(v) => v,
        }
    }
}

impl TryFrom<&str> for NInit {
    type Error = BuildError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "auto" => Ok(NInit::Auto),
            other => Err(BuildError::UnknownNInit {
                value: other.to_string(),
            }),
        }
    }
}

/// sklearn's `KMeans(algorithm=...)` — WHICH EXACT ALGORITHM computes the
/// assignment step, not which answer it computes. Both arms run the same Lloyd
/// iteration and converge to the same partition from the same init; `Elkan`
/// merely prunes distance computations with the triangle inequality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KMeansAlgorithm {
    /// The classic full assignment — every sample against every center each
    /// iteration. sklearn's default.
    #[default]
    Lloyd,
    /// Elkan's bound-based acceleration: per-sample upper/lower distance bounds
    /// plus the `k × k` center-center distances prune most `(sample, center)`
    /// pairs once the centers stop moving much. Costs an extra `n × k` bound
    /// matrix.
    Elkan,
}

impl KMeansAlgorithm {
    /// The sklearn `algorithm` string.
    pub fn name(self) -> &'static str {
        match self {
            KMeansAlgorithm::Lloyd => "lloyd",
            KMeansAlgorithm::Elkan => "elkan",
        }
    }

    /// sklearn's `_check_params_vs_input` override: `'elkan'` is meaningless
    /// for a single cluster (there is no "other center" to bound against), so
    /// it silently degrades to `'lloyd'`. sklearn warns; a library crate must
    /// not print, so mlrs performs the same override silently.
    pub fn resolve(self, n_clusters: usize) -> KMeansAlgorithm {
        if self == KMeansAlgorithm::Elkan && n_clusters == 1 {
            KMeansAlgorithm::Lloyd
        } else {
            self
        }
    }
}

impl TryFrom<&str> for KMeansAlgorithm {
    type Error = BuildError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "lloyd" => Ok(KMeansAlgorithm::Lloyd),
            "elkan" => Ok(KMeansAlgorithm::Elkan),
            other => Err(BuildError::UnknownAlgorithm {
                value: other.to_string(),
            }),
        }
    }
}

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
    /// Seed for the init host PRNG (sklearn's `random_state`; used only by the
    /// `k-means++` / `random` inits). `None` → [`DEFAULT_SEED`] so the v1
    /// deterministic behavior is preserved.
    random_state: Option<u64>,
    /// sklearn's `init` (`'k-means++'` / `'random'` / an explicit `k × d`
    /// array), narrowed to `F` (A5). The array form's `len() == k · n_features`
    /// is checked at `fit` against the data geometry.
    init: KMeansInit<F>,
    /// sklearn's `n_init` — how many independently seeded restarts to run,
    /// keeping the lowest-inertia result. Resolved against `init` at `fit` by
    /// [`NInit::resolve`].
    n_init: NInit,
    /// sklearn's `algorithm` (`'lloyd'` / `'elkan'`). Selects HOW the
    /// assignment step is computed, not what it computes.
    algorithm: KMeansAlgorithm,
    /// sklearn's `verbose`. Accepted and stored for signature compatibility; a
    /// library crate must not print, so it has no observable effect (the
    /// `SpectralClustering` precedent).
    verbose: bool,
    /// sklearn's `copy_x`. Stored for parity only: sklearn uses it to decide
    /// whether it may mean-center the CALLER's array in place, and mlrs never
    /// writes into the caller's buffer, so this has no observable effect (the
    /// `Ridge::copy_x` precedent).
    copy_x: bool,
    /// Fitted `k × d` cluster centers, device-resident, `None` until `fit`.
    cluster_centers_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted length-`n` integer labels (`i32`, D-06), device-resident, `None`
    /// until `fit`.
    labels_: Option<DeviceArray<ActiveRuntime, i32>>,
    /// Fitted inertia `Σ ‖X_i − centers[labels_i]‖²` (scalar), `None` until
    /// `fit`.
    inertia_: Option<F>,
    /// sklearn's `n_iter_` — the iteration count of the RESTART THAT WON (not
    /// the sum across restarts), `None` until `fit`.
    n_iter_: Option<usize>,
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
            random_state: None,
            init: KMeansInit::KMeansPlusPlus,
            n_init: NInit::Auto,
            algorithm: KMeansAlgorithm::Lloyd,
            verbose: false,
            copy_x: true,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_iter_: None,
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
            random_state: self.random_state,
            max_iter: self.max_iter,
            tol: self.tol,
            init: widen_init(self.init),
            n_init: self.n_init,
            algorithm: self.algorithm,
            verbose: self.verbose,
            copy_x: self.copy_x,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `cluster_centers_`/`labels_`/`inertia_` are excluded — `None` in any
    /// `Unfit` value). Used by the defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_clusters == other.n_clusters
            && self.random_state == other.random_state
            && self.max_iter == other.max_iter
            && self.tol == other.tol
            && self.n_init == other.n_init
            && self.algorithm == other.algorithm
            && self.verbose == other.verbose
            && self.copy_x == other.copy_x
            && match (&self.init, &other.init) {
                (KMeansInit::KMeansPlusPlus, KMeansInit::KMeansPlusPlus) => true,
                (KMeansInit::Random, KMeansInit::Random) => true,
                (KMeansInit::Array(a), KMeansInit::Array(b)) => {
                    a.len() == b.len()
                        && a.iter()
                            .zip(b.iter())
                            .all(|(&x, &y)| host_to_f64(x) == host_to_f64(y))
                }
                _ => false,
            }
    }
}

/// Promote a narrowed `KMeansInit<F>` back to the builder's non-generic
/// `KMeansInit<f64>` (the A5 inverse of [`narrow_init`]).
fn widen_init<F>(init: KMeansInit<F>) -> KMeansInit<f64>
where
    F: Float + CubeElement + Pod,
{
    match init {
        KMeansInit::KMeansPlusPlus => KMeansInit::KMeansPlusPlus,
        KMeansInit::Random => KMeansInit::Random,
        KMeansInit::Array(v) => KMeansInit::Array(v.iter().map(|&e| host_to_f64(e)).collect()),
    }
}

/// Narrow the builder's `KMeansInit<f64>` to the target float `F` (A5 — the
/// `f64 → F` narrowing happens once, in [`KMeansBuilder::build`]).
fn narrow_init<F>(init: KMeansInit<f64>) -> KMeansInit<F>
where
    F: Float + CubeElement + Pod,
{
    match init {
        KMeansInit::KMeansPlusPlus => KMeansInit::KMeansPlusPlus,
        KMeansInit::Random => KMeansInit::Random,
        KMeansInit::Array(v) => KMeansInit::Array(v.iter().map(|&e| f64_to_host::<F>(e)).collect()),
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
    random_state: Option<u64>,
    max_iter: usize,
    tol: f64,
    init: KMeansInit<f64>,
    n_init: NInit,
    algorithm: KMeansAlgorithm,
    verbose: bool,
    copy_x: bool,
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

    /// Set the init host-PRNG seed (used only by the `k-means++` / `random`
    /// inits). Retained spelling of [`Self::random_state`] for callers that
    /// predate the sklearn-named setter; `seed(v)` is exactly
    /// `random_state(Some(v))`.
    pub fn seed(mut self, v: u64) -> Self {
        self.random_state = Some(v);
        self
    }

    /// Set sklearn's `random_state`, which seeds the `k-means++` / `random`
    /// init draws (and, for `n_init > 1`, the per-restart seed derivation).
    /// `None` → [`DEFAULT_SEED`], preserving mlrs's deterministic v1 behavior
    /// (sklearn's `None` means "draw from the global numpy RNG", which no
    /// reproducible-by-default library can honour).
    pub fn random_state(mut self, v: Option<u64>) -> Self {
        self.random_state = v;
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
        self.init = match v {
            Some(a) => KMeansInit::Array(a),
            None => KMeansInit::KMeansPlusPlus,
        };
        self
    }

    /// Set sklearn's `init` in its FULL form — `'k-means++'` / `'random'` /
    /// an explicit `k × d` array ([`KMeansInit`]). The narrow
    /// [`init`](Self::init) setter above is the `Option`-of-array shape kept
    /// for callers that only ever inject the oracle's fixed centers; this one
    /// is the setter the sklearn string parameter maps to
    /// (`KMeansInit::try_from("random")`).
    pub fn init_method(mut self, v: KMeansInit<f64>) -> Self {
        self.init = v;
        self
    }

    /// Set sklearn's `n_init` — the number of independently seeded restarts,
    /// of which the LOWEST-INERTIA one is kept. `NInit::Auto` resolves against
    /// `init` at fit ([`NInit::resolve`]).
    pub fn n_init(mut self, v: NInit) -> Self {
        self.n_init = v;
        self
    }

    /// Set sklearn's `algorithm` (`'lloyd'` / `'elkan'`) — which assignment
    /// implementation runs, not which answer it produces.
    pub fn algorithm(mut self, v: KMeansAlgorithm) -> Self {
        self.algorithm = v;
        self
    }

    /// Set sklearn's `verbose` (accepted and stored; a library crate must not
    /// print, so it has no observable effect).
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Set sklearn's `copy_x` (accepted and stored for parity; mlrs never
    /// writes into the caller's buffer, so it has no observable effect).
    pub fn copy_x(mut self, v: bool) -> Self {
        self.copy_x = v;
        self
    }

    /// Build the (unfit) estimator, narrowing the stored `f64` scalars + the
    /// injected `init` to the target float `F` (A5).
    ///
    /// Only ONE hyperparameter is data-INDEPENDENT enough to validate here:
    /// `n_init >= 1` (sklearn's `Interval(Integral, 1, None, closed="left")`).
    /// `1 ≤ n_clusters ≤ n_samples`, the injected-init dimension
    /// (`len == k · n_features`), and the geometry are all data-DEPENDENT and
    /// stay in [`Fit::fit`] (D-03 byte-identical). The unrecognised-string
    /// rejections for `init` / `n_init` / `algorithm` happen earlier still, in
    /// their `TryFrom<&str>` parses, and are folded into the SAME `BuildError`
    /// so the `build_err_to_py` PyO3 mapper covers all of them (D-09).
    pub fn build<F>(self) -> Result<KMeans<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if let NInit::Fixed(v) = self.n_init {
            if v < 1 {
                return Err(BuildError::InvalidHyperprior {
                    estimator: "kmeans",
                    param: "n_init",
                    value: v as f64,
                    bound: ">= 1",
                });
            }
        }
        Ok(KMeans {
            n_clusters: self.n_clusters,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            init: narrow_init::<F>(self.init),
            n_init: self.n_init,
            algorithm: self.algorithm,
            verbose: self.verbose,
            copy_x: self.copy_x,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_iter_: None,
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

    /// sklearn's `n_iter_` — the iteration count of the restart that WON the
    /// `n_init` selection (not the sum across restarts, which is what a naive
    /// accumulator would report). `Some` by construction on the `Fitted` state.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
            .expect("n_iter_ is Some by construction on KMeans<F, Fitted>")
    }

    /// The `algorithm` that actually RAN, after the `k == 1` degradation
    /// ([`KMeansAlgorithm::resolve`]) — the analogue of `Ridge`'s `solver_`.
    pub fn algorithm_used(&self) -> KMeansAlgorithm {
        self.algorithm.resolve(self.n_clusters)
    }

    /// The number of restarts that actually RAN, after `n_init = 'auto'`
    /// resolution and the explicit-array override ([`NInit::resolve`]).
    pub fn n_init_used(&self) -> usize {
        self.n_init.resolve(&self.init)
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

/// One completed restart, kept on the HOST so the `n_init` selection compares
/// candidates without holding `n_init` sets of device buffers alive.
struct RunOutcome {
    /// `k × d` row-major centers (the f64 mirror the loop already maintains).
    centers_host: Vec<f64>,
    /// Length-`n` labels this run converged to.
    labels_host: Vec<u32>,
    /// `Σ ‖X_i − centers[labels_i]‖²`, recomputed exactly against the final
    /// adopted centers.
    inertia: f64,
    /// Iterations this run took (sklearn's per-run `n_iter_`).
    n_iter: usize,
}

/// Everything a restart needs that is FIXED across restarts (the data, the
/// resolved hyperparameters, and the once-per-fit `‖x_i‖²`).
struct RunEnv<'a, F> {
    x: &'a DeviceArray<ActiveRuntime, F>,
    xnorm: &'a DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    tol_scaled: f64,
    algorithm: KMeansAlgorithm,
    profile: bool,
}

/// The device scratch a restart writes through — allocated ONCE per fit and
/// reused by every restart (FOUND-05: `n_init = 10` must not mean ten `O(n·k)`
/// allocations). `upper` / `lower` are `Some` only on the Elkan arm.
struct RunScratch<'a, F> {
    labels: &'a mut DeviceArray<ActiveRuntime, u32>,
    labels_old: &'a mut DeviceArray<ActiveRuntime, u32>,
    dist: &'a DeviceArray<ActiveRuntime, F>,
    upper: Option<&'a DeviceArray<ActiveRuntime, F>>,
    lower: Option<&'a DeviceArray<ActiveRuntime, F>>,
    x_host_cache: &'a mut Option<Vec<F>>,
    prof: &'a mut Profile,
}

/// `KM_PROFILE=1` per-phase wall-clock accumulators, summed across restarts.
#[derive(Default)]
struct Profile {
    t_sums: f64,
    t_host: f64,
    t_assign: f64,
    iters: usize,
}

/// sklearn's `_is_same_clustering`: are two labelings the same partition up to
/// a label PERMUTATION? Builds the `a → b` map from first occurrence and fails
/// on the first row that contradicts it.
///
/// The `n_init` selection needs this because a later restart can land on the
/// SAME partition with a marginally lower inertia purely from float rounding;
/// sklearn (and mlrs) then keep the earlier one so the winner is stable.
fn is_same_clustering(a: &[u32], b: &[u32], k: usize) -> bool {
    let mut map: Vec<Option<u32>> = vec![None; k];
    for (&la, &lb) in a.iter().zip(b.iter()) {
        match map[la as usize] {
            None => map[la as usize] = Some(lb),
            Some(prev) if prev != lb => return false,
            Some(_) => {}
        }
    }
    true
}

/// Host `k × k` HALF center-center distances `chd[c, j] = ‖c − j‖ / 2` plus
/// `dnc[c] = min_{j != c} chd[c, j]` — Elkan's two pruning tables.
///
/// sklearn computes `dnc` as the SECOND smallest of each column of `chd`
/// (`np.partition(chd, kth=1, axis=0)[1]`); `chd` is symmetric with a zero
/// diagonal, so that is exactly the row minimum excluding the diagonal.
/// Computed on the tiny `k × d` f64 center mirror in f64 (the bounds are
/// compared against `F` distances, so they are narrowed on the way out).
fn center_half_distances<F>(centers: &[f64], k: usize, d: usize) -> (Vec<F>, Vec<F>)
where
    F: Float + CubeElement + Pod,
{
    let mut chd = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in (a + 1)..k {
            let mut acc = 0.0_f64;
            for j in 0..d {
                let diff = centers[a * d + j] - centers[b * d + j];
                acc += diff * diff;
            }
            let half = acc.sqrt() * 0.5;
            chd[a * k + b] = half;
            chd[b * k + a] = half;
        }
    }
    let dnc: Vec<f64> = (0..k)
        .map(|c| {
            let mut best = f64::INFINITY;
            for j in 0..k {
                if j != c && chd[c * k + j] < best {
                    best = chd[c * k + j];
                }
            }
            // k == 1 has no "other" center; `algorithm.resolve(k)` already
            // degrades that case to Lloyd, so this is unreachable defence.
            if best.is_finite() {
                best
            } else {
                f64::MAX
            }
        })
        .collect();
    (
        chd.iter().map(|&v| f64_to_host::<F>(v)).collect(),
        dnc.iter().map(|&v| f64_to_host::<F>(v)).collect(),
    )
}

/// Draw one restart's initial `k × d` centers per sklearn's `_init_centroids`,
/// returning BOTH the device buffer and the f64 host mirror the loop's
/// center-shift check reads (so the centers are never read back per iteration).
///
/// The `k-means++` and `random` draws happen on the HOST PRNG (ASVS V6, D-09c)
/// but gather their chosen ROWS on the device — `x` is never read back.
fn init_centers<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    k: usize,
    init: &KMeansInit<F>,
    seed: u64,
) -> Result<(DeviceArray<ActiveRuntime, F>, Vec<f64>), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let idx = match init {
        KMeansInit::Array(a) => {
            // Dimension already validated against the data geometry in `fit`.
            let host = a.iter().map(|&v| host_to_f64(v)).collect();
            return Ok((DeviceArray::from_host(pool, a), host));
        }
        KMeansInit::KMeansPlusPlus => kmeanspp_sample::<F>(pool, x, n, d, k, seed)?,
        KMeansInit::Random => random_sample(n, k, seed)?,
    };
    let idx_u32: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
    let dev = gather_rows_device::<F>(pool, x, &idx_u32, n, d)?;
    let host = dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    Ok((dev, host))
}

/// Run ONE restart to convergence from `centers` / `centers_host` and return
/// its host-side outcome, releasing the centers buffer on the way out.
///
/// Both algorithm arms share this body — they differ only in how the ASSIGN
/// step is computed (`assign_min`'s fused argmin vs Elkan's bound pruning) and
/// in Elkan's extra post-update bound relaxation. The UPDATE step, the
/// empty-cluster relocation, the strict-then-tol convergence checks, and the
/// final exact inertia are identical, which is what makes the two arms return
/// the same fit (see the `algorithm` oracle test).
fn single_run<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    env: &RunEnv<'_, F>,
    s: &mut RunScratch<'_, F>,
    mut centers: DeviceArray<ActiveRuntime, F>,
    mut centers_host: Vec<f64>,
) -> Result<RunOutcome, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (n, d, k) = (env.n, env.d, env.k);
    let elkan = env.algorithm == KMeansAlgorithm::Elkan;

    // --- Initial assignment. Lloyd writes labels + squared distances; Elkan
    //     additionally seeds its carried upper/lower bounds (and writes the
    //     same labels). ---
    if elkan {
        let (upper, lower) = (
            s.upper.expect("Elkan arm allocates upper"),
            s.lower.expect("Elkan arm allocates lower"),
        );
        elkan_init_bounds::<F>(pool, env.x, &centers, s.labels, upper, lower, n, d, k)?;
    } else {
        assign_min::<F>(
            pool,
            env.x,
            &centers,
            s.labels,
            s.dist,
            Some(env.xnorm),
            n,
            d,
            k,
        )?;
    }

    let mut iters_run = 0usize;
    for _iter in 0..env.max_iter {
        iters_run += 1;
        let lap0 = std::time::Instant::now();
        // UPDATE: per-centroid sums + counts via the row-blocked device gather
        // (the only per-iteration readback of the update phase).
        let (mut sums_f64, mut counts_i64) =
            centroid_sums_dev::<F>(pool, env.x, s.labels, n, d, k)?;
        if env.profile {
            s.prof.t_sums += lap0.elapsed().as_secs_f64();
        }
        let lap1 = std::time::Instant::now();

        // RARE path: an empty cluster triggers sklearn's exact relocation
        // (CR-01 / T-05-03-02), ranked by each sample's squared distance to its
        // assigned center. The Lloyd arm already has that in `dist`; Elkan's
        // `upper` is only a BOUND (exact solely for the rows it tightened), so
        // the Elkan arm recomputes the exact rows first. Only this branch ever
        // reads an O(n) buffer back inside the loop.
        if counts_i64.iter().any(|&c| c == 0) {
            if elkan {
                inertia_rows_device::<F>(pool, env.x, &centers, s.labels, s.dist, n, d)?;
            }
            if s.x_host_cache.is_none() {
                *s.x_host_cache = Some(env.x.to_host(pool));
            }
            let labels_host: Vec<u32> = s.labels.to_host(pool);
            let dist_host: Vec<f64> = s
                .dist
                .to_host(pool)
                .iter()
                .map(|&v| host_to_f64(v))
                .collect();
            relocate_empty_clusters::<F>(
                &mut sums_f64,
                &mut counts_i64,
                s.x_host_cache.as_ref().expect("cached above"),
                &labels_host,
                &dist_host,
                n,
                d,
                k,
            )?;
        }

        // Mean divide (f64, matching lloyd_update's finalize) + the center
        // shift against the f64 host mirror of the OLD centers.
        let mut new_centers_host = vec![0.0_f64; k * d];
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
                for j in 0..d {
                    new_centers_host[c * d + j] = sums_f64[c * d + j] * inv;
                }
            }
        }
        // center_shift_tot = Σ ‖new_center_c − old_center_c‖² (host pass over
        // the tiny k × d mirrors). Consulted AFTER the strict check.
        let mut shift = 0.0_f64;
        for i in 0..k * d {
            let diff = new_centers_host[i] - centers_host[i];
            shift += diff * diff;
        }

        // Elkan: relax the carried bounds by the per-cluster TRUE Euclidean
        // shift BEFORE adopting the new centers — the bounds still describe the
        // current labeling against the OLD centers, which is exactly the state
        // sklearn's post-`elkan_iter` fix-up assumes.
        if elkan {
            let cshift: Vec<F> = (0..k)
                .map(|c| {
                    let mut acc = 0.0_f64;
                    for j in 0..d {
                        let diff = new_centers_host[c * d + j] - centers_host[c * d + j];
                        acc += diff * diff;
                    }
                    f64_to_host::<F>(acc.sqrt())
                })
                .collect();
            elkan_relax_bounds::<F>(
                pool,
                &cshift,
                s.labels,
                s.upper.expect("Elkan arm allocates upper"),
                s.lower.expect("Elkan arm allocates lower"),
                n,
                k,
            )?;
        }

        let new_f: Vec<F> = new_centers_host
            .iter()
            .map(|&v| f64_to_host::<F>(v))
            .collect();
        centers.release_into(pool);
        centers = DeviceArray::from_host(pool, &new_f);
        centers_host = new_centers_host;

        if env.profile {
            s.prof.t_host += lap1.elapsed().as_secs_f64();
        }
        let lap2 = std::time::Instant::now();

        // ASSIGN to the new centers (previous labels kept in the swapped buffer
        // for the strict check). Elkan reads the previous labeling as INPUT (it
        // updates an assignment rather than recomputing one), which is why its
        // kernel takes separate in/out label buffers.
        std::mem::swap(&mut *s.labels, &mut *s.labels_old);
        if elkan {
            let (chd, dnc) = center_half_distances::<F>(&centers_host, k, d);
            elkan_assign_device::<F>(
                pool,
                env.x,
                &centers,
                &chd,
                &dnc,
                s.labels_old,
                s.labels,
                s.upper.expect("Elkan arm allocates upper"),
                s.lower.expect("Elkan arm allocates lower"),
                n,
                d,
                k,
            )?;
        } else {
            assign_min::<F>(
                pool,
                env.x,
                &centers,
                s.labels,
                s.dist,
                Some(env.xnorm),
                n,
                d,
                k,
            )?;
        }

        // STRICT array_equal break FIRST (Pitfall 6) — the labeling stopped
        // changing, so sklearn breaks before measuring the center shift.
        let changed = labels_changed(pool, s.labels, s.labels_old, n)?;
        if env.profile {
            s.prof.t_assign += lap2.elapsed().as_secs_f64();
        }
        if changed == 0 {
            break;
        }
        if shift <= env.tol_scaled {
            break;
        }
    }
    s.prof.iters += iters_run;

    // NOTE: no post-loop assignment pass is needed (the old code's Pitfall-6
    // re-assign): EVERY exit path above — strict break, tol break, max_iter
    // exhaustion, and the max_iter == 0 degenerate — leaves `s.labels` written
    // by an assign against the FINAL adopted `centers`, and assignment is
    // deterministic, so re-assigning reproduces the same labels sklearn's
    // post-loop `_labels_inertia` would.

    // --- inertia_ = Σ per-row squared distance to the assigned center.
    //     Recompute the rows with the DIRECT gather first (the GEMM staging
    //     distances rank correctly but their f32 cancellation noise exceeds the
    //     1e-5 oracle tolerance when summed, and Elkan's `upper` is a bound at
    //     all), then the blocked device sum — still no O(n) readback for the
    //     scalar. ---
    inertia_rows_device::<F>(pool, env.x, &centers, s.labels, s.dist, n, d)?;
    let inertia = sum_device::<F>(pool, s.dist, n)?;

    // ONE O(n) readback per restart (the `n_init` selection compares labelings
    // up to a permutation, which cannot be done on-device).
    let labels_host: Vec<u32> = s.labels.to_host(pool);
    centers.release_into(pool);

    Ok(RunOutcome {
        centers_host,
        labels_host,
        inertia,
        n_iter: iters_run,
    })
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
        //     gather in the assign / update kernels. ---
        if k < 1 || k > n_samples {
            return Err(AlgoError::InvalidK {
                estimator: "kmeans",
                k,
                n_samples,
            });
        }
        validate_geometry(x, shape)?;

        // The injected-init dimension is data-DEPENDENT, so it is checked here
        // (not at `build`) — once, before any restart.
        if let KMeansInit::Array(a) = &self.init {
            if a.len() != k * n_features {
                return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                    operand: "init",
                    rows: k,
                    cols: n_features,
                    len: a.len(),
                }));
            }
        }

        // --- Resolve the three sklearn parameters whose effective value
        //     depends on the OTHERS (sklearn's `_check_params_vs_input`):
        //     `random_state = None` pins mlrs's deterministic default seed,
        //     `n_init = 'auto'` resolves against `init`, and `algorithm =
        //     'elkan'` degrades to `'lloyd'` at k == 1. ---
        let seed = self.random_state.unwrap_or(DEFAULT_SEED);
        let n_init = self.n_init.resolve(&self.init);
        let algorithm = self.algorithm.resolve(k);
        let elkan = algorithm == KMeansAlgorithm::Elkan;

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
        //     read back — never the n × d sample matrix). Data-only, so it is
        //     computed ONCE and shared by every restart. ---
        let tol_scaled = self.tol * feature_mean_var::<F>(pool, x, n_samples, n_features)?;

        // --- Device work buffers for the launch-only loop, allocated ONCE and
        //     reused by every restart: u32 labels (current + previous, swapped
        //     each iteration) and the per-row squared distance to the assigned
        //     center (the relocation ranking AND the inertia rows). ---
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

        // Elkan's carried state: the length-n upper bounds and the n × k lower
        // bounds — the ONLY O(n·k) allocation in the estimator, which is the
        // price `algorithm='elkan'` pays for its pruning. Allocated only on
        // that arm, and only once per fit.
        let (upper_dev, lower_dev) = if elkan {
            (
                Some(DeviceArray::<ActiveRuntime, F>::from_raw(
                    pool.acquire(n_samples * size_of::<F>()),
                    n_samples,
                )),
                Some(DeviceArray::<ActiveRuntime, F>::from_raw(
                    pool.acquire(n_samples * k * size_of::<F>()),
                    n_samples * k,
                )),
            )
        } else {
            (None, None)
        };

        // ‖x_i‖², computed ONCE per fit for the GEMM assignment path (the prim
        // ignores it on the direct path — a single tiny launch either way).
        let xnorm = row_sqnorms::<F>(pool, x, n_samples, n_features)?;

        // KM_PROFILE=1: per-phase wall-clock attribution summed across restarts
        // (laps are delimited by the loop's natural readback sync points, so
        // kernel time lands in the phase whose readback drains it — attribution
        // only, like RF_PROFILE).
        let profile = std::env::var("KM_PROFILE").is_ok();
        let mut prof = Profile::default();

        // Host `x` copy, materialized ONLY if some iteration hits the rare
        // empty-cluster relocation, then reused across later relocations AND
        // across restarts (`x` is immutable — measured 12ms/iteration of
        // repeated O(n·d) readback on a relocation-heavy ladder config).
        let mut x_host_cache: Option<Vec<F>> = None;

        let env = RunEnv {
            x,
            xnorm: &xnorm,
            n: n_samples,
            d: n_features,
            k,
            max_iter: self.max_iter,
            tol_scaled,
            algorithm,
            profile,
        };

        // --- The `n_init` restart loop. sklearn keeps a later run ONLY when it
        //     is strictly better AND lands on a genuinely different partition —
        //     otherwise a float-rounding tie would make the winner depend on
        //     iteration order. `n_init` is >= 1 by construction (`build()`
        //     rejects 0), so `best` is always Some below. ---
        let mut best: Option<RunOutcome> = None;
        for restart in 0..n_init {
            // Distinct, decorrelated stream per restart (SplitMix64 is designed
            // for exactly this "seed by counter" use).
            let restart_seed = seed.wrapping_add(restart as u64);
            let (centers, centers_host) = init_centers::<F>(
                pool,
                x,
                n_samples,
                n_features,
                k,
                &self.init,
                restart_seed,
            )?;
            let mut scratch = RunScratch {
                labels: &mut labels_dev,
                labels_old: &mut labels_old_dev,
                dist: &dist_dev,
                upper: upper_dev.as_ref(),
                lower: lower_dev.as_ref(),
                x_host_cache: &mut x_host_cache,
                prof: &mut prof,
            };
            let outcome = single_run::<F>(pool, &env, &mut scratch, centers, centers_host)?;
            let keep = match &best {
                None => true,
                Some(b) => {
                    outcome.inertia < b.inertia
                        && !is_same_clustering(&outcome.labels_host, &b.labels_host, k)
                }
            };
            if keep {
                best = Some(outcome);
            }
        }
        let best = best.expect("n_init >= 1 (build() rejects 0), so a restart ran");

        if profile {
            eprintln!(
                "KM_PROFILE n={n_samples} d={n_features} k={k} algo={} n_init={n_init}: \
                 iters={} sums+readback={:.4}s host+upload={:.4}s assign+changed={:.4}s",
                algorithm.name(),
                prof.iters,
                prof.t_sums,
                prof.t_host,
                prof.t_assign
            );
        }

        // --- Adopt the winning restart: re-upload its centers, and store the
        //     labels as i32 (D-06: the u32 prim labels widen to the i32 trait
        //     surface; KMeans labels are non-negative). ---
        let centers_f: Vec<F> = best
            .centers_host
            .iter()
            .map(|&v| f64_to_host::<F>(v))
            .collect();
        let centers_dev = DeviceArray::from_host(pool, &centers_f);
        let labels_i32: Vec<i32> = best.labels_host.iter().map(|&l| l as i32).collect();
        let labels_dev_i32: DeviceArray<ActiveRuntime, i32> =
            DeviceArray::from_host(pool, &labels_i32);

        // Return the transient loop buffers to the pool (FOUND-05).
        labels_dev.release_into(pool);
        labels_old_dev.release_into(pool);
        dist_dev.release_into(pool);
        xnorm.release_into(pool);
        if let Some(u) = upper_dev {
            u.release_into(pool);
        }
        if let Some(l) = lower_dev {
            l.release_into(pool);
        }

        Ok(KMeans {
            n_clusters: self.n_clusters,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            init: self.init,
            n_init: self.n_init,
            algorithm: self.algorithm,
            verbose: self.verbose,
            copy_x: self.copy_x,
            cluster_centers_: Some(centers_dev),
            labels_: Some(labels_dev_i32),
            inertia_: Some(f64_to_host::<F>(best.inertia)),
            n_iter_: Some(best.n_iter),
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
