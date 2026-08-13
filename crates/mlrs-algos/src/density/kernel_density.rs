//! `KernelDensity` (KERNEL-02) — kernel density estimation matching
//! `sklearn.neighbors.KernelDensity` forced-exact (`atol=0, rtol=0`).
//!
//! ## Composes the v1 `distance` + a density-value map + a device log-sum-exp (D-08)
//! KernelDensity is a DISTINCT kernel family from the kernel-matrix dot-product
//! kernels: its six kernels are functions of the RAW euclidean distance with a
//! dimension-dependent normalization (D-08). It therefore composes the v1
//! [`distance`](mlrs_backend::prims::distance) prim DIRECTLY (NOT the
//! kernel-matrix prim)
//! + a per-element density-value map (the `mlrs-kernels` `kde_*_map` kernels) + a
//! per-query (row) log-sum-exp over the v1 [`row_reduce`](mlrs_backend::prims::reduce)
//! prim. The final assembly is
//! `log_density(q) = logsumexp_i[log_kernel(dist_i, h)] + log_norm(h, d, kernel) − log(N)`
//! (RESEARCH §"Density assembly"; VERIFIED from sklearn 1.9.0 `_kde.py`).
//!
//! ## Linear-domain log-sum-exp, never `±∞` (D-11 / Pitfall 3)
//! The per-element map computes the kernel VALUE (`exp(log_kernel)`), so the
//! compact-support kernels (tophat/epanechnikov/linear/cosine) yield EXACT `0`
//! out of support — the sum stays a sum of non-negative finites, never poisoned by
//! `−∞`/the infinity constant. The single `log` is applied ONCE at the very end
//! (host-side), after the device row-sum. This is the cpu-MLIR-safe form
//! ([[cubecl-cpu-no-shared-memory]] — the map is shared-memory-free).
//!
//! ## Squared vs raw distance per kernel (Pitfall 4)
//! gaussian/epanechnikov consume `distance(sqrt=false)` (squared `‖q − x‖²`);
//! tophat/exponential/linear/cosine compare the RAW `dist < h`, so they consume
//! `distance(sqrt=true)`.
//!
//! ## Host-side `log_norm` in f64 (A1)
//! The per-kernel log-normalization constant `log_norm(h, d, kernel)` depends only
//! on the bandwidth `h`, the feature dimension `d`, and the kernel — NOT on the
//! data — so it is a per-query CONSTANT computed ONCE on the host in `f64`
//! (`logVn`/`logSn`/`lgamma`), then added to the device-computed `logsumexp`. The
//! `lgamma` is a self-contained Lanczos approximation (matching the C `lgamma`
//! sklearn's Cython uses within the documented KD tolerance, A1) — `lgamma` is
//! NEVER attempted on device.
//!
//! ## Bandwidth resolution (D-09)
//! `bandwidth` is numeric (`> 0`) OR the `'scott'` / `'silverman'` host closed
//! forms (`n^(−1/(d+4))` / `(n·(d+2)/4)^(−1/(d+4))` — the SKLEARN forms, not
//! scipy's). Resolved at `fit` from `n_samples`/`n_features`; `bandwidth_ > 0` is
//! validated (`InvalidBandwidth`) before any launch.
//!
//! ## ScoreSamples (D-12), NOT Predict
//! KernelDensity implements [`ScoreSamples`](crate::typestate::ScoreSamples) — a
//! length-`n` per-query log-density vector — NOT a regression `Predict` / a
//! neighbor surface (it lives in its own `density/` home, RESEARCH Open Q2).
//!
//! Tests live in `crates/mlrs-algos/tests/kernel_density_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::reduce::{row_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};
use mlrs_kernels::{
    kde_cosine_map, kde_epanechnikov_map, kde_exponential_map, kde_gaussian_map, kde_linear_map,
    kde_tophat_map,
};

use crate::error::{AlgoError, BuildError};
// SHAPE A' (RESEARCH Open Q3 / A3): KernelDensity had an INHERENT `fit` plus an
// OLD legacy-`traits`-surface `ScoreSamples` impl (no `Fit` trait). The Phase-16 retrofit
// ADOPTS the typestate `Fit` (its inherent `fit` becomes the consuming-self trait
// impl on `Unfit`) and moves `ScoreSamples` to the typestate version, gated on
// `Fitted` — bringing KernelDensity fully onto the SINGLE trait surface.
use crate::kernel_persist::{
    read_x_fit, write_x_fit, AlignedBytes, KernelFile, KernelWriter, LoadModel, PersistError,
    SaveModel, KERNEL_KEY,
};
use crate::typestate::{validate_geometry, Fit, Fitted, ScoreSamples, Unfit};

/// The `estimator` discriminator written into every `KernelDensity` file. See
/// [`kernel_ridge`](crate::kernel_ridge)'s tag for why it is load-bearing —
/// the two estimators' `param:kernel` vocabularies overlap on `"linear"` while
/// meaning different functions by it.
const PERSIST_TAG: &str = "kernel_density";

/// The `__metadata__` key holding the bandwidth SPECIFICATION — the constructor
/// argument, as its sklearn string.
const BANDWIDTH_SPEC_KEY: &str = "param:bandwidth";

/// The `__metadata__` key holding the RESOLVED numeric bandwidth.
///
/// No `param:` prefix: `bandwidth_` is sklearn's fitted attribute. It is stored
/// alongside the specification rather than re-derived from it, for the reason
/// [`write_resolved_gamma`](crate::kernel_persist::write_resolved_gamma) gives —
/// re-running the `'scott'`/`'silverman'` rules at load would put the same
/// formula in two places with nothing to keep them in step, and a later change
/// to either would silently give every previously-saved model a different
/// bandwidth.
const BANDWIDTH_FITTED_KEY: &str = "bandwidth_";

/// The six sklearn KernelDensity kernels (D-07). Selected at construction; the
/// resolved numeric `bandwidth_` is computed at `fit` (D-09).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdKernel {
    /// Gaussian `exp(−0.5·‖q−x‖²/h²)` — squared distance, no compact support.
    Gaussian,
    /// Tophat `1` if `dist < h` else `0` — raw distance, compact.
    Tophat,
    /// Epanechnikov `1 − ‖q−x‖²/h²` inside, `0` outside — squared distance, compact.
    Epanechnikov,
    /// Exponential `exp(−dist/h)` — raw distance, no compact support.
    Exponential,
    /// Linear `1 − dist/h` inside, `0` outside — raw distance, compact.
    Linear,
    /// Cosine `cos(0.5·π·dist/h)` inside, `0` outside — raw distance, compact.
    Cosine,
}

impl KdKernel {
    /// The sklearn kernel name (for the [`AlgoError::InvalidKernel`] diagnostic,
    /// and for the model file, which stores the variant as this string rather
    /// than as an integer tag so adding a variant later cannot silently renumber
    /// an existing file's).
    pub fn name(self) -> &'static str {
        match self {
            KdKernel::Gaussian => "gaussian",
            KdKernel::Tophat => "tophat",
            KdKernel::Epanechnikov => "epanechnikov",
            KdKernel::Exponential => "exponential",
            KdKernel::Linear => "linear",
            KdKernel::Cosine => "cosine",
        }
    }

    /// The inverse of [`KdKernel::name`]; `None` for an unrecognised string.
    ///
    /// Returns an `Option` rather than a `Result` so each caller frames the
    /// failure in its own terms — a builder raises an `InvalidKernel` naming the
    /// argument, while [`KernelDensity::load`] raises a
    /// [`PersistError::BadMetadata`] naming the key it came from.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "gaussian" => Some(KdKernel::Gaussian),
            "tophat" => Some(KdKernel::Tophat),
            "epanechnikov" => Some(KdKernel::Epanechnikov),
            "exponential" => Some(KdKernel::Exponential),
            "linear" => Some(KdKernel::Linear),
            "cosine" => Some(KdKernel::Cosine),
            _ => None,
        }
    }

    /// Whether this kernel's density map consumes the SQUARED distance
    /// (`distance(sqrt=false)`). gaussian/epanechnikov use squared; the four
    /// raw-distance kernels use `distance(sqrt=true)` (Pitfall 4).
    fn uses_squared_distance(self) -> bool {
        matches!(self, KdKernel::Gaussian | KdKernel::Epanechnikov)
    }
}

/// The bandwidth specification (D-09): a numeric value (`> 0`) used as-is, or one
/// of the two host closed-form auto-bandwidth rules resolved at `fit` from
/// `n_samples`/`n_features` (the SKLEARN forms, not scipy's).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandwidthSpec {
    /// A fixed numeric bandwidth (`> 0`, validated at `fit`).
    Numeric(f64),
    /// `'scott'`: `bandwidth_ = n^(−1/(d+4))`.
    Scott,
    /// `'silverman'`: `bandwidth_ = (n·(d+2)/4)^(−1/(d+4))`.
    Silverman,
}

impl BandwidthSpec {
    /// The sklearn spelling: `'scott'`, `'silverman'`, or the numeric value's
    /// shortest round-tripping decimal.
    ///
    /// One string rather than a number-plus-flag pair, for the reason
    /// [`write_n_components`](crate::projection::proj_persist::write_n_components)
    /// gives for `n_components='auto'`: the two rule variants carry no numeric
    /// value, an optional number would make a dropped key and a deliberate rule
    /// indistinguishable, and a separate flag is two keys that can contradict
    /// each other. This reads exactly the way `bandwidth='scott'` versus
    /// `bandwidth=0.5` does in the sklearn constructor.
    ///
    /// `{:?}` rather than `{}` for the numeric arm: both of Rust's float
    /// formatters emit the shortest decimal that round-trips through
    /// `str::parse`, but `{:?}` picks the exponent form when it is shorter.
    pub fn name(self) -> String {
        match self {
            BandwidthSpec::Numeric(v) => format!("{v:?}"),
            BandwidthSpec::Scott => "scott".to_string(),
            BandwidthSpec::Silverman => "silverman".to_string(),
        }
    }

    /// The inverse of [`BandwidthSpec::name`]; `None` for a string that is
    /// neither rule nor a parsable decimal.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "scott" => Some(BandwidthSpec::Scott),
            "silverman" => Some(BandwidthSpec::Silverman),
            other => other.parse::<f64>().ok().map(BandwidthSpec::Numeric),
        }
    }
}

/// Kernel density estimation (KERNEL-02) over the v1 `distance` prim + a
/// density-value map + a device log-sum-exp (D-08/D-11).
///
/// Construct with the zero-arg [`KernelDensity::new`] (sklearn defaults:
/// `kernel = gaussian`, `bandwidth = 1.0`) or [`KernelDensity::builder`], then
/// the consuming [`Fit::fit`] (returns the `Fitted`-tagged sibling) and
/// [`score_samples`](crate::typestate::ScoreSamples::score_samples). The fitted
/// training matrix `X_fit_` is device-resident; the resolved `bandwidth_` is a
/// host `f64` accessor that exists ONLY on `KernelDensity<F, Fitted>` (the
/// compile-time typestate replaces the old runtime `NotFitted` guard, D-03).
pub struct KernelDensity<F, S = Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Which density kernel to evaluate (D-07).
    kernel: KdKernel,
    /// The bandwidth specification (numeric or scott/silverman, D-09).
    bandwidth_spec: BandwidthSpec,
    /// The fitted training matrix `X_fit_` (`n_samples × n_features`),
    /// device-resident, `None` until `fit`.
    x_fit_: Option<DeviceArray<ActiveRuntime, F>>,
    /// The RESOLVED numeric bandwidth (`> 0`), `None` until `fit` (D-09).
    bandwidth_: Option<f64>,
    /// Fitted `(n_samples, n_features)` geometry, `None` until `fit`.
    fit_shape_: Option<(usize, usize)>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> KernelDensity<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfitted `KernelDensity` with sklearn's defaults
    /// (`kernel = gaussian`, `bandwidth = 1.0`) directly in the `Unfit` state.
    /// SINGLE source of truth for the defaults (D-08): the builder `Default`
    /// re-derives via [`KernelDensity::into_builder`].
    pub fn new() -> Self {
        Self {
            kernel: KdKernel::Gaussian,
            bandwidth_spec: BandwidthSpec::Numeric(1.0),
            x_fit_: None,
            bandwidth_: None,
            fit_shape_: None,
            _state: PhantomData,
        }
    }

    /// Start building a `KernelDensity` from sklearn's defaults (D-08 single
    /// source).
    pub fn builder() -> KernelDensityBuilder {
        KernelDensityBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`KernelDensityBuilder::default`] to re-derive the
    /// defaults from [`KernelDensity::new`] (D-08).
    pub fn into_builder(self) -> KernelDensityBuilder {
        KernelDensityBuilder {
            kernel: self.kernel,
            bandwidth: self.bandwidth_spec,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `x_fit_`/`bandwidth_`/`fit_shape_` are excluded — all `None` in any `Unfit`
    /// value). Used by the defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.kernel == other.kernel && self.bandwidth_spec == other.bandwidth_spec
    }
}

impl<F> Default for KernelDensity<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`KernelDensity`] (D-01). `kernel` takes the [`KdKernel`] enum and
/// `bandwidth` the [`BandwidthSpec`] (numeric value or `'scott'`/`'silverman'`
/// rule) directly — neither is a scalar narrowing (A5). `Default` re-derives the
/// sklearn defaults from [`KernelDensity::new`] (D-08 single source).
#[derive(Debug, Clone, Copy)]
pub struct KernelDensityBuilder {
    kernel: KdKernel,
    bandwidth: BandwidthSpec,
}

impl Default for KernelDensityBuilder {
    /// Re-derive the sklearn defaults from [`KernelDensity::new`] (D-08 single
    /// source). `f64` is pinned only to read the F-independent defaults — the
    /// builder is non-generic.
    fn default() -> Self {
        KernelDensity::<f64, Unfit>::new().into_builder()
    }
}

impl KernelDensityBuilder {
    /// Set the density kernel family (D-07).
    pub fn kernel(mut self, v: KdKernel) -> Self {
        self.kernel = v;
        self
    }

    /// Set the bandwidth specification (numeric value or `'scott'`/`'silverman'`
    /// host rule, D-09).
    pub fn bandwidth(mut self, v: BandwidthSpec) -> Self {
        self.bandwidth = v;
        self
    }

    /// Build the (unfit) estimator. KernelDensity has no purely data-INDEPENDENT
    /// hyperparameter that is unconditionally validated at construction: the
    /// kernel name is a closed enum, and the `bandwidth_ > 0` check is
    /// resolution-path-coupled (the `'scott'`/`'silverman'` rules resolve the
    /// numeric bandwidth at fit against `n_samples`/`n_features`), so it stays in
    /// the fit body (D-03 byte-identical). The `Result` is kept for family
    /// uniformity so the `build_err_to_py` PyO3 mapper is shape-identical across
    /// the Phase-16 builders.
    pub fn build<F>(self) -> Result<KernelDensity<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        Ok(KernelDensity {
            kernel: self.kernel,
            bandwidth_spec: self.bandwidth,
            x_fit_: None,
            bandwidth_: None,
            fit_shape_: None,
            _state: PhantomData,
        })
    }
}

impl<F> KernelDensity<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The resolved numeric `bandwidth_` (`> 0`) after `fit`. `Some` by
    /// construction on the `Fitted` state, so no `NotFitted` branch is needed
    /// (the compile-time typestate replaces the runtime guard, D-03).
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth_
            .expect("bandwidth_ is Some by construction on KernelDensity<F, Fitted>")
    }
}

impl<F> SaveModel for KernelDensity<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted density estimator to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `X_fit_` | `F` (`F32`/`F64`) | `[n_samples, n_features]` |
    /// | `param:kernel` / `param:bandwidth` / `bandwidth_` | `__metadata__` scalar | — |
    ///
    /// ONE tensor and three scalars: `KernelDensity` is the purest case of "the
    /// training set is the model" in mlrs. There is nothing else to store —
    /// `score_samples` evaluates the kernel against every training row, so the
    /// matrix is not an artifact of fitting but the entire fitted state, and the
    /// file is `X_fit_` plus a header.
    ///
    /// Both the bandwidth SPECIFICATION and the RESOLVED value are written: the
    /// `'scott'` and `'silverman'` rules consume `n_samples`/`n_features` at fit
    /// time, so the request and the outcome are two different facts and a
    /// reloaded model reports each.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        let (n_samples, n_features) = self.fit_shape_.ok_or_else(|| absent("fit_shape_"))?;
        let bandwidth = self.bandwidth_.ok_or_else(|| absent("bandwidth_"))?;
        // Bound BEFORE the writer, which borrows the payload.
        let x_fit = self.x_fit_.as_ref().ok_or_else(|| absent("x_fit_"))?.to_host(pool);

        let mut w = KernelWriter::new(PERSIST_TAG);
        w.scalar_str(KERNEL_KEY, self.kernel.name());
        w.scalar_str(BANDWIDTH_SPEC_KEY, &self.bandwidth_spec.name());
        w.scalar_f64(BANDWIDTH_FITTED_KEY, bandwidth);
        write_x_fit(&mut w, &x_fit, n_samples, n_features)?;
        w.write(path)
    }
}

impl<F> LoadModel for KernelDensity<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the density estimator back from `path`, re-uploading `X_fit_` to
    /// `pool`.
    ///
    /// Both enum-shaped scalars are PARSED rather than trusted: an unrecognised
    /// kernel or bandwidth string becomes a [`PersistError::BadMetadata`] naming
    /// its key. That matters here because a silent fallback would be invisible —
    /// every one of the six kernels produces a plausible density, so a model
    /// that loaded `gaussian` where the file said `epanechnikov` would score
    /// every sample differently with nothing to signal it.
    ///
    /// `bandwidth_` is REQUIRED for the same reason. It is the one number
    /// `score_samples` divides by, and no default could stand in for a value the
    /// fit derived from the training geometry.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<KernelDensity<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = KernelFile::parse(&raw, PERSIST_TAG)?;
        let (x_fit, n_samples, n_features) = read_x_fit::<F>(&file)?;

        let kernel = KdKernel::from_name(file.scalar_str(KERNEL_KEY)?)
            .ok_or(PersistError::BadMetadata { key: KERNEL_KEY })?;
        let bandwidth_spec = BandwidthSpec::from_name(file.scalar_str(BANDWIDTH_SPEC_KEY)?)
            .ok_or(PersistError::BadMetadata {
                key: BANDWIDTH_SPEC_KEY,
            })?;
        let bandwidth = file.scalar_f64(BANDWIDTH_FITTED_KEY)?;
        // A non-positive bandwidth divides by zero (or flips the sign of every
        // exponent) inside the density map. The fit rejects it; a hand-edited
        // header must be rejected here too, rather than producing NaN scores on
        // the first query.
        if !(bandwidth > 0.0) || !bandwidth.is_finite() {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "'{BANDWIDTH_FITTED_KEY}' is {bandwidth}; a fitted bandwidth is \
                     finite and strictly positive"
                ),
            });
        }

        Ok(KernelDensity {
            kernel,
            bandwidth_spec,
            x_fit_: Some(DeviceArray::from_host(pool, &x_fit)),
            bandwidth_: Some(bandwidth),
            fit_shape_: Some((n_samples, n_features)),
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for KernelDensity<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = KernelDensity<F, Fitted>;

    /// Fit the density model: store `X_fit_` and resolve `bandwidth_` (D-09),
    /// CONSUMING `self` and returning the `Fitted`-tagged sibling.
    ///
    /// `x` is `(n_samples × n_features)` row-major. Validates the kernel name and
    /// geometry, resolves the bandwidth (numeric or scott/silverman host closed
    /// form), and validates `bandwidth_ > 0` (`InvalidBandwidth`) — all BEFORE any
    /// device launch (T-08-04-01 / ASVS V5).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        // `_y` is unused: the retained `Fit`-trait slot (KernelDensity is an
        // unsupervised density estimator) — not unfinished wiring (IN-02).
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<KernelDensity<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-08-04-01 / ASVS V5: validate the kernel name + geometry BEFORE any
        //     launch. KdKernel is a closed set, but the guard documents the
        //     validate-before-launch contract and surfaces InvalidKernel rather
        //     than fall through (mirrors kernel_ridge.rs). ---
        if !matches!(
            self.kernel,
            KdKernel::Gaussian
                | KdKernel::Tophat
                | KdKernel::Epanechnikov
                | KdKernel::Exponential
                | KdKernel::Linear
                | KdKernel::Cosine
        ) {
            return Err(AlgoError::InvalidKernel {
                estimator: "kernel_density",
                kernel: self.kernel.name().to_string(),
            });
        }
        validate_geometry(x, shape)?;

        // --- Bandwidth resolution (D-09, host f64). scott/silverman are the
        //     SKLEARN closed forms (no per-feature std factor — NOT scipy's). ---
        let n = n_samples as f64;
        let d = n_features as f64;
        let bandwidth = match self.bandwidth_spec {
            BandwidthSpec::Numeric(b) => b,
            BandwidthSpec::Scott => n.powf(-1.0 / (d + 4.0)),
            BandwidthSpec::Silverman => (n * (d + 2.0) / 4.0).powf(-1.0 / (d + 4.0)),
        };
        // Validate-before-launch: a non-positive bandwidth makes the −d·log(h)
        // normalization undefined (T-08-04-01). Require FINITE as well —
        // `inf > 0.0` passes the positivity check but drives `−d·h.ln()` → −inf
        // and `exp(−0.5·sqdist/inf²) = exp(0) = 1` on device, producing a
        // finite-but-meaningless log-density instead of a typed rejection (WR-03).
        if !(bandwidth > 0.0 && bandwidth.is_finite()) {
            return Err(AlgoError::InvalidBandwidth {
                estimator: "kernel_density",
                bandwidth,
            });
        }

        // Store a fresh device copy of X_fit_ (the caller's `x` is borrowed).
        let x_host = x.to_host(pool);
        let x_fit: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_host);

        // The pre-retrofit `&mut self` re-fit path released a prior `x_fit_` buffer
        // before reassigning (WR-07); a freshly-built `Unfit` value carries no
        // fitted state, so that release is a no-op here and is dropped (the typestate
        // transition consumes a fresh `Unfit`; a re-fit constructs a new estimator).
        Ok(KernelDensity {
            kernel: self.kernel,
            bandwidth_spec: self.bandwidth_spec,
            x_fit_: Some(x_fit),
            bandwidth_: Some(bandwidth),
            fit_shape_: Some((n_samples, n_features)),
            _state: PhantomData,
        })
    }
}

impl<F> ScoreSamples<F> for KernelDensity<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Compute the length-`n_query` log-density for each row of `q` (D-12), via
    /// `distance(Q, X_fit_, sqrt=per-kernel)` → per-element density-value map →
    /// per-query (row) log-sum-exp over the v1 `reduce` prim → host assembly
    /// `lse_row + log_norm − log(N)` (D-08/D-11). The fitted `x_fit_`/`bandwidth_`/
    /// `fit_shape_` are `Some` by construction on the `Fitted` state (the
    /// compile-time typestate replaces the old runtime `NotFitted` guard, D-03);
    /// errors only on a geometry / feature-count mismatch.
    fn score_samples(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        q: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, n_features) = shape;

        let x_fit = self
            .x_fit_
            .as_ref()
            .expect("x_fit_ is Some by construction on KernelDensity<F, Fitted>");
        let bandwidth = self
            .bandwidth_
            .expect("bandwidth_ is Some by construction on KernelDensity<F, Fitted>");
        let (n_samples, fit_features) = self
            .fit_shape_
            .expect("fit_shape_ is Some by construction on KernelDensity<F, Fitted>");

        // --- T-08-04-01 / ASVS V5: geometry + fitted-n_features consistency. ---
        if n_query == 0 || n_features == 0 || q.len() != n_query * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "q",
                rows: n_query,
                cols: n_features,
                len: q.len(),
            }));
        }
        if n_features != fit_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: fit_features,
            }));
        }

        // --- 1. D = distance(Q, X_fit_) (m×n). sqrt=false for gaussian/epanechnikov
        //        (squared distance), sqrt=true for the four raw-distance kernels
        //        (Pitfall 4). D-08 — the v1 distance prim DIRECTLY, NOT
        //        the kernel-matrix prim. ---
        let sqrt = !self.kernel.uses_squared_distance();
        let dmat = distance::<F>(
            pool,
            q,
            (n_query, n_features),
            x_fit,
            (n_samples, fit_features),
            sqrt,
            None,
        )?;

        // --- 2. Per-element KD density-value map IN PLACE over the distance buffer
        //        (linear domain — exact 0 out of support, never ±∞, D-11). The map
        //        kernel is shared-memory-free; the m×n operand stays in global
        //        memory (T-08-04-03). input handle == output handle (the
        //        the in-place scale-map idiom). ---
        let n_elems = n_query * n_samples;
        let h = f64_to_host::<F>(bandwidth);
        // The gaussian / exponential / cosine maps evaluate a TRANSCENDENTAL
        // (`exp`, `cos`); on a backend with f64 arithmetic but no f64
        // transcendentals those kernels return garbage (a wgpu f64 gaussian KDE
        // produced `NaN` log-densities and a `+4.45e2` where sklearn has
        // `-5.60`). Route just the map to the host there — the O(n_query ·
        // n_samples · d) distance base above stays on device.
        let dmat = if kde_host_map_applicable::<F>(self.kernel) {
            kde_map_host(pool, dmat, n_elems, self.kernel, bandwidth)
        } else {
            launch_kde_map_in_place(pool, &dmat, n_elems, self.kernel, h);
            dmat
        };

        // --- 3. Per-query (row) log-sum-exp via the v1 reduce prim (D-11). Plain
        //        reduce-SUM in the linear domain: row_sum = Σ_j kernel_value. The
        //        Shared path is forced (cpu-portable; the plane path returns None on
        //        non-subgroup adapters). The reduce-max rescale (div_by_row) is NOT
        //        needed — the kernel values are O(1) bounded (K(0,h)=1), so the
        //        linear sum has no overflow/underflow over the v2 problem sizes
        //        (RESEARCH Open Q1: rescale not needed; the f32 band passes). ---
        let row_sum = row_reduce::<F>(
            pool,
            &dmat,
            n_query,
            n_samples,
            ScalarOp::Sum,
            ReducePath::Shared,
        )?
        .ok_or(AlgoError::Prim(PrimError::InternalNone {
            operand: "column_reduce",
            context: "ReducePath::Shared",
        }))?;
        dmat.release_into(pool);

        // --- 4. Host assembly (the single log applied ONCE at the end, D-11):
        //        log_density = log(row_sum) + log_norm(h, d, kernel) − log(N).
        //        log_norm is the per-kernel host-side f64 scalar (A1 — f64 lgamma,
        //        NEVER device). N = n_training_samples (no sample weights). ---
        let log_norm = kde_log_norm(self.kernel, bandwidth, n_features);
        let log_n = (n_samples as f64).ln();
        let row_sum_host = row_sum.to_host(pool);
        row_sum.release_into(pool);
        let mut out_host: Vec<F> = vec![F::from_int(0i64); n_query];
        for r in 0..n_query {
            let s = host_to_f64(row_sum_host[r]);
            // s is a sum of non-negative kernel values; log(0) → −∞ is the correct
            // log-density for a query with zero density in its support (matches
            // sklearn). It is produced ONLY at this terminal host step, never inside
            // a device map (Pitfall 3), so it cannot poison a device sum.
            let log_density = s.ln() + log_norm - log_n;
            out_host[r] = f64_to_host::<F>(log_density);
        }
        Ok(DeviceArray::from_host(pool, &out_host))
    }
}

/// Does this KD kernel's per-element map need the HOST because the backend
/// cannot evaluate f64 transcendentals?
///
/// Only `Gaussian` (`exp`), `Exponential` (`exp`) and `Cosine` (`cos`) evaluate
/// one; `Tophat` / `Epanechnikov` / `Linear` are pure arithmetic and keep
/// running on device at every precision. See
/// `mlrs_backend::capability::f64_transcendental_supported`.
fn kde_host_map_applicable<F>(kernel: KdKernel) -> bool {
    std::mem::size_of::<F>() == 8
        && !mlrs_backend::capability::f64_transcendental_supported()
        && matches!(
            kernel,
            KdKernel::Gaussian | KdKernel::Exponential | KdKernel::Cosine
        )
}

/// Host twin of [`launch_kde_map_in_place`], in `f64`.
///
/// Applies the SAME per-element formula each `kde_*_map` kernel applies,
/// including the compact-support guards, so the only thing that changes is where
/// the arithmetic runs. `cubecl` 0.10 cannot write in place into an existing
/// handle, so the distance buffer is released and the mapped values re-staged —
/// `from_host` recycles the just-freed bytes off the pool free-list, so no
/// second live `n_query · n_samples` allocation exists.
fn kde_map_host<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    dmat: DeviceArray<ActiveRuntime, F>,
    n: usize,
    kernel: KdKernel,
    bandwidth: f64,
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let h = bandwidth;
    let mut host: Vec<F> = dmat.to_host(pool);
    for v in host.iter_mut().take(n) {
        let x = host_to_f64(*v);
        let mapped = match kernel {
            // `in` is the SQUARED distance for gaussian (Pitfall 4).
            KdKernel::Gaussian => (-0.5 * x / (h * h)).exp(),
            // `in` is the RAW distance for exponential.
            KdKernel::Exponential => (-x / h).exp(),
            // `in` is the RAW distance for cosine; exact 0 outside support.
            KdKernel::Cosine => {
                if x >= h {
                    0.0
                } else {
                    (std::f64::consts::FRAC_PI_2 * x / h).cos()
                }
            }
            // The arithmetic-only kernels never route here (see
            // `kde_host_map_applicable`); leaving the value untouched would be a
            // silent wrong answer, so this is unreachable by construction.
            other => unreachable!("kde_map_host called for a non-transcendental kernel {other:?}"),
        };
        *v = f64_to_host::<F>(mapped);
    }
    dmat.release_into(pool);
    DeviceArray::from_host(pool, &host)
}

/// Launch the per-element KD density-value map IN PLACE over the distance buffer
/// `dmat` (input handle == output handle), the backend prim's
/// scale-in-place idiom. `n` is the element count (`n_query · n_samples`); each
/// `kde_*_map` kernel bounds-checks `tid < input.len()` (T-08-04-01) and is
/// shared-memory-free (the m×n operand stays in global memory, T-08-04-03).
fn launch_kde_map_in_place<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    dmat: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    kernel: KdKernel,
    h: F,
) where
    F: Float + CubeElement + Pod,
{
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n);
    // SAFETY: `n` is the carried distance-prim output element count (n_query ·
    // n_samples, itself derived from the validated geometry); each KD map kernel
    // bounds-checks `tid < input.len()`. input and output are the SAME handle so
    // the map is applied in place over the reused distance buffer (no parallel
    // allocation — T-08-04-03).
    let in_arg = unsafe { ArrayArg::from_raw_parts(dmat.handle().clone(), n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(dmat.handle().clone(), n) };
    match kernel {
        KdKernel::Gaussian => {
            kde_gaussian_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, h)
        }
        KdKernel::Tophat => {
            kde_tophat_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, h)
        }
        KdKernel::Epanechnikov => kde_epanechnikov_map::launch::<F, ActiveRuntime>(
            &client, count, dim, in_arg, out_arg, h,
        ),
        KdKernel::Exponential => kde_exponential_map::launch::<F, ActiveRuntime>(
            &client, count, dim, in_arg, out_arg, h,
        ),
        KdKernel::Linear => {
            kde_linear_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, h)
        }
        KdKernel::Cosine => {
            kde_cosine_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, h)
        }
    }
}

/// Standard ceiling-division 1D launch config for the in-place map pass (the
/// elementwise per-element launch idiom shared with the backend prims).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256usize;
    // Compute the cube count in `usize` and check the `u32` launch-grid cast
    // (WR-02): an unchecked `n as u32` silently wraps for `n > u32::MAX`,
    // under-provisioning threads so trailing elements are never mapped — a silent
    // wrong-result. The KDE problem sizes are small today, but the guard turns the
    // overflow into a loud panic instead.
    let cubes = u32::try_from((n + block - 1) / block)
        .expect("element count exceeds u32 launch-grid limit");
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim {
            x: block as u32,
            y: 1,
            z: 1,
        },
    )
}

/// The per-kernel log-normalization constant `log_norm(h, d, kernel) = −factor −
/// d·log(h)` (RESEARCH §"Per-kernel log-normalization constant" TABLE; VERIFIED
/// from sklearn 1.9.0 `_binary_tree.pxi.tp` lines 438-476). Host-side f64; the
/// `lgamma` is the self-contained Lanczos approximation below (A1 — NEVER device).
fn kde_log_norm(kernel: KdKernel, h: f64, d_features: usize) -> f64 {
    let d = d_features as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    // logVn(n) = 0.5·n·log(π) − lgamma(0.5·n + 1)   (log volume of the unit n-ball)
    let log_vn = |n: f64| 0.5 * n * std::f64::consts::PI.ln() - lgamma(0.5 * n + 1.0);
    // logSn(n) = log(2π) + logVn(n − 1)              (log surface area)
    let log_sn = |n: f64| two_pi.ln() + log_vn(n - 1.0);

    let factor = match kernel {
        KdKernel::Gaussian => 0.5 * d * two_pi.ln(),
        KdKernel::Tophat => log_vn(d),
        KdKernel::Epanechnikov => log_vn(d) + (2.0 / (d + 2.0)).ln(),
        KdKernel::Exponential => log_sn(d - 1.0) + lgamma(d),
        KdKernel::Linear => log_vn(d) - (d + 1.0).ln(),
        KdKernel::Cosine => {
            // Cosine series (chain-rule integration, _binary_tree.pxi.tp 466-473):
            //   factor = 0; tmp = 2/π
            //   for k in 1, 3, 5, …, ≤ d:  factor += tmp;
            //                              tmp *= −(d−k)·(d−k−1)·(2/π)²
            //   factor = log(factor) + logSn(d−1)
            let two_over_pi = 2.0 / std::f64::consts::PI;
            let mut series = 0.0;
            let mut tmp = two_over_pi;
            let mut k = 1.0;
            while k <= d {
                series += tmp;
                tmp *= -(d - k) * (d - k - 1.0) * two_over_pi * two_over_pi;
                k += 2.0;
            }
            series.ln() + log_sn(d - 1.0)
        }
    };
    -factor - d * h.ln()
}

/// Natural log of the gamma function in `f64` via the Lanczos approximation
/// (g = 7, 9 coefficients), valid for `x > 0`. Matches the C `lgamma` sklearn's
/// Cython uses within the documented KD tolerance (A1) — used ONLY host-side for
/// the per-kernel `log_norm`, NEVER on device.
fn lgamma(x: f64) -> f64 {
    // Lanczos g=7 coefficients (Numerical Recipes / standard reference set).
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1−x) = π / sin(πx).
        let pi = std::f64::consts::PI;
        (pi / (pi * x).sin()).ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}
