//! `SparseRandomProjection` (PROJ-02) — Achlioptas sparse random projection,
//! stored DENSE (D-12).
//!
//! An `n_components × n_features` projection matrix whose entries are `0` with
//! probability `1 − density`, `+v` / `−v` each with probability `density/2`,
//! where `v = sqrt((1/density)/n_components)` (RESEARCH Pattern 4). Generated via
//! the Phase-7 [`mlrs_backend::prims::rng::sparse_achlioptas_matrix`] (PRIM-06,
//! host SplitMix64 → ONE upload, seed-reproducible — T-07-02). The matrix is
//! stored DENSE even though it is structurally sparse (D-12 — there is no sparse
//! device kernel in v2; at v2 matrix sizes a dense Achlioptas is acceptable), so
//! `transform == X · components_ᵀ` is the IDENTICAL single GEMM as the Gaussian
//! transformer ([`crate::projection::gaussian::project`]) — RandomProjection does
//! NOT center. Sparse INPUT densification happens at the Python ingress (Plan
//! 07), not here.
//!
//! ## `n_components='auto'` (PROJ-02)
//! [`NComponents::Auto`] resolves via [`johnson_lindenstrauss_min_dim`] exactly
//! like the Gaussian transformer (the ONE value-matched quantity — D-12); the
//! matrix/transform are property-gated, not 1e-5-valued.
//!
//! ## Hyperparameter guards (ASVS V5 / T-07-10)
//! `eps ∉ (0, 1)` → [`AlgoError::InvalidEpsDistortion`]; `density ∉ (0, 1]` →
//! [`AlgoError::InvalidDensity`]; `n_components < 1` →
//! [`AlgoError::InvalidNComponents`] — all rejected BEFORE any RNG matrix
//! allocation. `density = None` resolves to the sklearn default
//! `1/sqrt(n_features)`.
//!
//! Tests live in `crates/mlrs-algos/tests/random_projection_test.rs`
//! (AGENTS.md §2).

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::rng;
use mlrs_backend::runtime::ActiveRuntime;

use super::proj_persist::{
    read_components, read_n_components, write_components, write_n_components, AlignedBytes,
    LoadModel, PersistError, ProjFile, ProjWriter, SaveModel,
};
use crate::error::{AlgoError, BuildError};
use crate::projection::gaussian::{johnson_lindenstrauss_min_dim, project, NComponents};
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

/// The `estimator` discriminator written into every `SparseRandomProjection`
/// file. See [`gaussian`](super::gaussian)'s tag for why it is load-bearing.
const PERSIST_TAG: &str = "sparse_random_projection";

/// The `__metadata__` key holding the FITTED density — the value the Achlioptas
/// draw actually used.
///
/// No [`PARAM_PREFIX`](super::proj_persist::PARAM_PREFIX): `density_` is
/// sklearn's fitted attribute, distinct from the `density` constructor argument
/// which is optional and, when `None`, means "use `1/sqrt(n_features)`". Both
/// are stored, because the request and the outcome are different facts and a
/// reloaded model reports each.
const DENSITY_FITTED_KEY: &str = "density_";

/// Sparse (Achlioptas) random-projection transformer (PROJ-02), `components_`
/// stored DENSE (D-12).
///
/// Construct with the zero-arg [`SparseRandomProjection::new`] (sklearn defaults:
/// `n_components = 'auto'` → [`NComponents::Auto`], `eps = 0.1`, `seed = 0`,
/// `density = None` → `1/sqrt(n_features)`) or
/// [`SparseRandomProjection::builder`], then the consuming [`Fit::fit`] (returns
/// the `Fitted`-tagged sibling) to draw the Achlioptas `components_` and
/// [`Transform::transform`] (`X · components_ᵀ`, one GEMM, NO centering). Fitted
/// state is device-resident (D-03); the host accessors exist ONLY on
/// `SparseRandomProjection<F, Fitted>` (the compile-time typestate replaces the
/// old runtime `NotFitted` guard, D-03).
pub struct SparseRandomProjection<F, S = Unfit> {
    /// Requested embedding dimension (`Auto` → resolved at fit via JL).
    n_components: NComponents,
    /// Documented `u64` seed driving SplitMix64 (T-07-02 — never `OsRng`).
    seed: u64,
    /// JL distortion bound for the `Auto` path (`eps ∈ (0, 1)`).
    eps: f64,
    /// Requested sparsity density (`None` → `1/sqrt(n_features)` at fit;
    /// otherwise must be `∈ (0, 1]`).
    density: Option<f64>,
    /// `components_` (`n_components × n_features`), row-major, device-resident.
    components_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Resolved embedding dimension after fit.
    n_components_: usize,
    /// Resolved density after fit (`None` → `1/sqrt(n_features)`).
    density_: f64,
    /// `n_features` seen at fit, for the `transform` geometry check.
    n_features: usize,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> SparseRandomProjection<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfitted `SparseRandomProjection` with sklearn's defaults
    /// (`n_components = 'auto'` → [`NComponents::Auto`], `eps = 0.1`, `seed = 0`,
    /// `density = None` → `1/sqrt(n_features)` at fit) directly in the `Unfit`
    /// state. SINGLE source of truth for the defaults (D-08): the builder
    /// `Default` re-derives via [`SparseRandomProjection::into_builder`].
    pub fn new() -> Self {
        Self {
            n_components: NComponents::Auto,
            seed: 0,
            eps: 0.1,
            density: None,
            components_: None,
            n_components_: 0,
            density_: 0.0,
            n_features: 0,
            _state: PhantomData,
        }
    }

    /// Start building a `SparseRandomProjection` from sklearn's defaults (D-08
    /// single source).
    pub fn builder() -> SparseRandomProjectionBuilder {
        SparseRandomProjectionBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`SparseRandomProjectionBuilder::default`] to
    /// re-derive the defaults from [`SparseRandomProjection::new`] (D-08).
    pub fn into_builder(self) -> SparseRandomProjectionBuilder {
        SparseRandomProjectionBuilder {
            n_components: self.n_components,
            seed: self.seed,
            eps: self.eps,
            density: self.density,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `components_` is excluded — `None` in any `Unfit` value). Used by the
    /// defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_components == other.n_components
            && self.seed == other.seed
            && self.eps == other.eps
            && self.density == other.density
    }
}

impl<F> Default for SparseRandomProjection<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`SparseRandomProjection`] (D-01). The `n_components` setter takes
/// the [`NComponents`] enum directly; `seed` is `u64`, `eps` is `f64`, `density`
/// is `Option<f64>` (`None` → `1/sqrt(n_features)` at fit). `Default` re-derives
/// the sklearn defaults from [`SparseRandomProjection::new`] (D-08 single source).
#[derive(Debug, Clone, Copy)]
pub struct SparseRandomProjectionBuilder {
    n_components: NComponents,
    seed: u64,
    eps: f64,
    density: Option<f64>,
}

impl Default for SparseRandomProjectionBuilder {
    /// Re-derive the sklearn defaults from [`SparseRandomProjection::new`] (D-08
    /// single source). `f64` is pinned only to read the F-independent scalar
    /// defaults — the builder is non-generic.
    fn default() -> Self {
        SparseRandomProjection::<f64, Unfit>::new().into_builder()
    }
}

impl SparseRandomProjectionBuilder {
    /// Set the requested embedding dimension ([`NComponents::Auto`] JL-sized or
    /// [`NComponents::Fixed`]).
    pub fn n_components(mut self, v: NComponents) -> Self {
        self.n_components = v;
        self
    }

    /// Set the documented `u64` seed driving SplitMix64 (never `OsRng`, T-07-02).
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    /// Set the JL distortion bound for the `Auto` path (`eps ∈ (0, 1)`).
    pub fn eps(mut self, v: f64) -> Self {
        self.eps = v;
        self
    }

    /// Set the sparsity density (`None` → sklearn's `1/sqrt(n_features)` at fit;
    /// otherwise the explicit density `∈ (0, 1]`).
    pub fn density(mut self, v: Option<f64>) -> Self {
        self.density = v;
        self
    }

    /// Build the (unfit) estimator. SparseRandomProjection has no purely
    /// data-INDEPENDENT hyperparameter that is unconditionally validated: the
    /// `eps ∈ (0, 1)` and `density ∈ (0, 1]` checks are resolution-path-coupled
    /// (`density = None` resolves to `1/sqrt(n_features)` against the fit
    /// geometry; the `Auto`/`Fixed` eps paths validate inline) and the
    /// `n_components < 1` check is data-DEPENDENT, so all stay in the fit body
    /// (D-03 byte-identical). The `Result` is kept for family uniformity so the
    /// `build_err_to_py` PyO3 mapper is shape-identical across the Phase-16
    /// builders.
    pub fn build<F>(self) -> Result<SparseRandomProjection<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        Ok(SparseRandomProjection {
            n_components: self.n_components,
            seed: self.seed,
            eps: self.eps,
            density: self.density,
            components_: None,
            n_components_: 0,
            density_: 0.0,
            n_features: 0,
            _state: PhantomData,
        })
    }
}

impl<F> SparseRandomProjection<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of `components_` (`n_components_ × n_features`, row-major,
    /// dense). `Some` by construction on the `Fitted` state, so no `NotFitted`
    /// branch is needed (the compile-time typestate replaces the runtime guard,
    /// D-03).
    pub fn components(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.components_
            .as_ref()
            .expect("components_ is Some by construction on SparseRandomProjection<F, Fitted>")
            .to_host(pool)
    }

    /// The resolved embedding dimension (`Auto` → JL value) after fit.
    pub fn n_components_(&self) -> usize {
        self.n_components_
    }

    /// The resolved density (`None` → `1/sqrt(n_features)`) after fit.
    pub fn density_(&self) -> f64 {
        self.density_
    }
}

impl<F> SaveModel for SparseRandomProjection<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted projection to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `components_` | `F` (`F32`/`F64`) | `[n_components_, n_features]` |
    /// | `param:n_components` / `param:seed` / `param:eps` | `__metadata__` scalar | — |
    /// | `param:density` | `__metadata__` scalar, ABSENT when `None` | — |
    /// | `density_` | `__metadata__` scalar | — |
    ///
    /// The matrix is stored DENSE, mostly-zero as it is —
    /// [`proj_persist`](super::proj_persist) gives the reason: mlrs's projection
    /// is a dense GEMM (D-12), and the file follows the memory layout so a load
    /// is one copy-free hand-off rather than a decode.
    ///
    /// `param:density` is written only when it was given: `None` MEANS "use
    /// `1/sqrt(n_features)`", so key-presence carries the `Option` and costs
    /// zero bytes when absent. `density_` — what that resolved to — is always
    /// written, since it is not recoverable from `components_` without counting
    /// the non-zeros and inferring a probability from a sample.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        // Bound BEFORE the writer, which borrows the payload.
        let components = self
            .components_
            .as_ref()
            .ok_or(PersistError::MissingState {
                estimator: PERSIST_TAG,
                field: "components_",
            })?
            .to_host(pool);

        let mut w = ProjWriter::new(PERSIST_TAG);
        write_n_components(&mut w, self.n_components);
        w.scalar_u64("param:seed", self.seed);
        w.scalar_f64("param:eps", self.eps);
        w.scalar_opt_f64("param:density", self.density);
        w.scalar_f64(DENSITY_FITTED_KEY, self.density_);
        write_components(&mut w, &components, self.n_components_, self.n_features)?;
        w.write(path)
    }
}

impl<F> LoadModel for SparseRandomProjection<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the projection back from `path`, re-uploading `components_` to
    /// `pool`.
    ///
    /// `param:density` is OPTIONAL and `density_` is REQUIRED, which is the
    /// asymmetry the two keys exist to express: absence of the former is a
    /// meaningful `None`, while absence of the latter is a corrupt file — a
    /// model whose fitted density is unknown cannot report what it actually did.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<SparseRandomProjection<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = ProjFile::parse(&raw, PERSIST_TAG)?;
        let (components, n_components_, n_features) = read_components::<F>(&file)?;

        Ok(SparseRandomProjection {
            n_components: read_n_components(&file)?,
            seed: file.scalar_u64("param:seed")?,
            eps: file.scalar_f64("param:eps")?,
            density: file.scalar_opt_f64("param:density")?,
            components_: Some(DeviceArray::from_host(pool, &components)),
            n_components_,
            density_: file.scalar_f64(DENSITY_FITTED_KEY)?,
            n_features,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for SparseRandomProjection<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = SparseRandomProjection<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        // `_y` is unused: the retained `Fit`-trait slot for Phase-10 MBSGD reuse
        // (this estimator is unsupervised; see typestate.rs) — not unfinished
        // wiring (IN-02).
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<SparseRandomProjection<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-07-10 / ASVS V5: geometry consistency BEFORE any RNG launch. ---
        validate_geometry(x, shape)?;

        // --- Resolve density: None → 1/sqrt(n_features) (sklearn default);
        //     validate density ∈ (0, 1] (ASVS V5 / T-07-10) BEFORE generation. ---
        let density = self
            .density
            .unwrap_or_else(|| 1.0 / (n_features as f64).sqrt());
        if !(density.is_finite() && density > 0.0 && density <= 1.0) {
            return Err(AlgoError::InvalidDensity {
                estimator: "sparse_random_projection",
                density,
            });
        }

        // --- Resolve n_components (Auto → JL; Fixed → caller's int). The Auto
        //     path validates eps ∈ (0,1); validate eps on the Fixed path too. ---
        let nc = match self.n_components {
            NComponents::Auto => johnson_lindenstrauss_min_dim(n_samples as f64, self.eps)?,
            NComponents::Fixed(k) => {
                if !(self.eps.is_finite() && self.eps > 0.0 && self.eps < 1.0) {
                    return Err(AlgoError::InvalidEpsDistortion {
                        estimator: "random_projection",
                        eps: self.eps,
                    });
                }
                k
            }
        };
        if nc < 1 {
            return Err(AlgoError::InvalidNComponents {
                estimator: "sparse_random_projection",
                requested: nc,
                max: n_features,
            });
        }

        // --- components_ = Achlioptas v=sqrt((1/density)/n_components), stored
        //     DENSE via PRIM-06 (host SplitMix64 → ONE upload — D-12 / T-07-02). ---
        let components_dev =
            rng::sparse_achlioptas_matrix::<F>(pool, self.seed, nc, n_features, density)?;

        Ok(SparseRandomProjection {
            n_components: self.n_components,
            seed: self.seed,
            eps: self.eps,
            density: self.density,
            components_: Some(components_dev),
            n_components_: nc,
            density_: density,
            n_features,
            _state: PhantomData,
        })
    }
}

impl<F> Transform<F> for SparseRandomProjection<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        // SAME single GEMM as Gaussian (D-12 — dense Achlioptas, no centering).
        project(
            "sparse_random_projection",
            self.components_.as_ref(),
            self.n_components_,
            self.n_features,
            pool,
            x,
            shape,
        )
    }
}
