//! `KernelRidge` (KERNEL-01) — kernel ridge regression via the dual-coefficient
//! Cholesky solve of `(K + αI)`, matching `sklearn.kernel_ridge.KernelRidge`.
//!
//! ## Dual solve over the kernel matrix (NOT XᵀX — D-06)
//! KernelRidge fits the dual coefficients
//! `(K + αI)·dual_coef_ = y`
//! where `K = kernel_matrix(X, X, kernel)` is the `n×n` training Gram (D-02,
//! PRIM-08) — the kernel matrix, NOT the feature-space Gram `XᵀX`. It mirrors the
//! v1 [`crate::linear::ridge::Ridge`] Cholesky path EXCEPT the normal matrix is
//! `K`, `α` goes on the `K` diagonal, and there is NO centering and NO intercept
//! (sklearn KernelRidge — RESEARCH Pitfall 1). Prediction is
//! `y = kernel_matrix(X_test, X_fit_, kernel) · dual_coef_` (no intercept
//! broadcast).
//!
//! ## Multi-target in one multi-RHS solve (D-04)
//! A multi-target `y` (`n×t`) is solved in ONE [`cholesky_solve`] call with
//! `rhs = t` — the multi-RHS dual solve is near-free (the factorization is shared
//! across the `t` right-hand sides), producing `dual_coef_` (`n×t`). That
//! sharing is conditional on ONE penalty; see `alpha` below.
//!
//! ## The full sklearn parameter surface (KERNEL-PARAMS)
//! Every `sklearn.kernel_ridge.KernelRidge` parameter is honoured here:
//!
//! | parameter | how |
//! |---|---|
//! | `kernel` | all nine strings — `linear`, `rbf`, `poly`/`polynomial`, `sigmoid`, `laplacian`, `cosine`, `chi2`, `additive_chi2`, `precomputed` ([`KernelKind`]) |
//! | `alpha` | scalar OR one value per target ([`KernelRidgeBuilder::alphas`]) |
//! | `gamma` | `None` → `1/n_features`, except `chi2`, which has no default |
//! | `degree` | real, `>= 0` (sklearn's interval), evaluated with `powf` |
//! | `coef0` | any finite value; used by `poly` / `sigmoid` |
//! | `sample_weight` | [`KernelRidge::fit_weighted`] |
//!
//! `kernel_params` and a CALLABLE `kernel` have no representation at this layer
//! and cannot: they are a Python object and a Python call. The shim evaluates
//! the callable itself and routes the result through `precomputed`, which is
//! what `precomputed` is for.
//!
//! ## gamma resolution (D-05)
//! `gamma = None` resolves to `1/n_features` at `fit` (computed from
//! `X.shape[1]`); an explicit `gamma` is used as-is. The RESOLVED value is stored
//! inside the typed [`Kernel`] AND in `gamma_` so `predict` reuses the IDENTICAL
//! kernel (RESEARCH Pitfall 5 — the fit-time and predict-time gamma MUST match).
//!
//! `chi2` is the exception: it has NO gamma default, in sklearn or here, and
//! `gamma = None` is an error rather than `1/n_features` — see
//! [`AlgoError::KernelRequiresGamma`] for why matching sklearn's failure is
//! better than being more forgiving than it.
//!
//! ## alpha on the diagonal only (D-06)
//! `α` is added to the `K` DIAGONAL only (`K[i·n+i] += alpha`) — the same
//! diagonal-stride penalty injection as `ridge.rs`, but over `K`, not `XᵀX`.
//! There is no intercept to leave unpenalized (D-06).
//!
//! With ONE `α` (the scalar case, and a per-target vector whose entries are all
//! equal) there is ONE `(K + αI)` and therefore one factorization shared by all
//! `t` targets. With DISTINCT per-target alphas there are `t` different matrices
//! and the sharing is gone — `t` factorizations of the same `n`. sklearn splits
//! on exactly this test and so does `fit`; it is the whole performance story of
//! the array-valued `alpha` and it is measured in the KERNEL-PARAMS bench.
//!
//! ## Cholesky cap (Pitfall 6 / A2)
//! The `n×n` `(K + αI)` is solved by the single-cube Phase-4 Cholesky primitive,
//! which caps `n ≤ MAX_DIM = 64`; oracle fixtures keep `n_samples ≤ 64`.
//!
//! ## Non-SPD guard (T-08-03-02)
//! A non-SPD `(K + αI)` surfaces [`PrimError::NotPositiveDefinite`] from the
//! primitive (propagated as [`AlgoError`] via `#[from]`), never a NaN
//! `dual_coef_`. With `α ≥ 0` on an SPD kernel diagonal the system stays
//! well-conditioned. A NaN reaching the Cholesky diagonal does NOT reliably
//! trip the primitive's `pivot <= 0` test (`NaN <= 0` is `false`) — e.g. a poly
//! kernel with a negative base (`γ·g + coef0 < 0`, non-integer degree) yields a
//! NaN Gram entry — so `fit` ALSO validates the resolved `gamma` is finite
//! before launch ([`AlgoError::InvalidGamma`]) and performs a post-solve
//! finiteness check on the produced duals, returning
//! [`PrimError::NotPositiveDefinite`] rather than storing a NaN `dual_coef_`.
//!
//! Tests live in `crates/mlrs-algos/tests/kernel_ridge_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::cholesky::cholesky_solve;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::kernel_matrix::{kernel_matrix, Kernel};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
// SHAPE A' (RESEARCH Open Q3): KernelRidge had INHERENT `fit`/`predict` methods
// and NO legacy-traits import. The Phase-16 retrofit ADOPTS the typestate `Fit`
// (consuming-self) + `Predict` traits so the estimator joins the SINGLE trait
// surface and the legacy-surface-gone grep (Plan 11) stays clean. The fit/predict
// device math is BYTE-IDENTICAL (D-03); only the signatures, the geometry guard
// call (now `validate_geometry`), and the construction/reconstruction wrapper
// change.
use crate::kernel_persist::{
    expect_len, read_resolved_gamma, read_x_fit, shape_2d, write_resolved_gamma, write_x_fit,
    AlignedBytes, KernelFile, KernelWriter, LoadModel, PersistError, SaveModel, TensorRef,
    KERNEL_KEY,
};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// The `estimator` discriminator written into every `KernelRidge` file.
///
/// Load-bearing rather than decorative: a `KernelDensity` file holds an `X_fit_`
/// of the same shape and dtype and a `param:kernel` under the same key, and the
/// two vocabularies OVERLAP on `"linear"` while meaning entirely different
/// functions by it. The tag is what establishes which vocabulary applies before
/// either is parsed.
const PERSIST_TAG: &str = "kernel_ridge";

/// The tensor holding the dual coefficients, row-major
/// `[n_samples, n_targets]` — sklearn's `dual_coef_`.
const DUAL_COEF_NAME: &str = "dual_coef_";

/// The tensor holding the penalty vector — one entry (sklearn's scalar `alpha`)
/// or one per target (sklearn's array-like `alpha`).
///
/// A tensor rather than another `param:` metadata scalar because `alpha` is the
/// one hyperparameter here whose LENGTH is data-dependent, and the metadata
/// block has no vector-valued spelling.
const ALPHA_NAME: &str = "alpha_";

/// The kernel-family selector accepted at construction (D-01). Mirrors sklearn's
/// `kernel=` string but typed; the hyperparameters (`gamma`/`degree`/`coef0`) are
/// resolved into a precision-typed [`Kernel`] at `fit`. A `gamma = None` resolves
/// to `1/n_features` (D-05).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// Linear kernel `K = X·Yᵀ`.
    Linear,
    /// RBF (Gaussian) kernel `K = exp(-γ·‖xᵢ − yⱼ‖²)`.
    Rbf,
    /// Polynomial kernel `K = (γ·⟨xᵢ, yⱼ⟩ + coef0)^degree`.
    Poly,
    /// Sigmoid kernel `K = tanh(γ·⟨xᵢ, yⱼ⟩ + coef0)`.
    Sigmoid,
    /// Laplacian kernel `K = exp(-γ·‖xᵢ − yⱼ‖₁)`.
    Laplacian,
    /// Cosine kernel `K = ⟨x̂ᵢ, ŷⱼ⟩` (L2-normalised rows). Takes no coefficients.
    Cosine,
    /// Exponential chi-squared kernel `K = exp(γ·additive_chi2)`. Requires a
    /// non-negative input and an EXPLICIT `gamma` (see
    /// [`AlgoError::KernelRequiresGamma`]).
    Chi2,
    /// Additive chi-squared kernel `K = -Σₖ (xᵢₖ − yⱼₖ)²/(xᵢₖ + yⱼₖ)`. Requires a
    /// non-negative input; takes no coefficients.
    AdditiveChi2,
    /// The caller supplies `K` directly: `fit`'s `X` IS the `n×n` training kernel
    /// matrix and `predict`'s `X` is the `n_test×n_fit` cross-kernel. No kernel
    /// is evaluated, so `gamma`/`degree`/`coef0` are all inert.
    Precomputed,
}

impl KernelKind {
    /// The sklearn kernel name (for the [`AlgoError::InvalidKernel`] diagnostic,
    /// and for the model file, which stores the variant as this string rather
    /// than as an integer tag so adding a variant later cannot silently renumber
    /// an existing file's).
    ///
    /// `Poly` names itself `"poly"`, never `"polynomial"`: sklearn accepts both
    /// spellings for the same kernel, and picking one here is what keeps a model
    /// file's kernel key from depending on which spelling the caller happened to
    /// type. [`KernelKind::from_name`] accepts both.
    pub fn name(self) -> &'static str {
        match self {
            KernelKind::Linear => "linear",
            KernelKind::Rbf => "rbf",
            KernelKind::Poly => "poly",
            KernelKind::Sigmoid => "sigmoid",
            KernelKind::Laplacian => "laplacian",
            KernelKind::Cosine => "cosine",
            KernelKind::Chi2 => "chi2",
            KernelKind::AdditiveChi2 => "additive_chi2",
            KernelKind::Precomputed => "precomputed",
        }
    }

    /// The inverse of [`KernelKind::name`]; `None` for an unrecognised string.
    ///
    /// Returns an `Option` rather than a `Result` so each caller frames the
    /// failure in its own terms — a builder raises an `InvalidKernel` naming the
    /// argument, while [`KernelRidge::load`] raises a
    /// [`PersistError::BadMetadata`] naming the key it came from.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "linear" => Some(KernelKind::Linear),
            "rbf" => Some(KernelKind::Rbf),
            // sklearn's `PAIRWISE_KERNEL_FUNCTIONS` carries the polynomial kernel
            // under BOTH spellings and `StrOptions` accepts either, so both are
            // accepted here and both normalise to `Poly`.
            "poly" | "polynomial" => Some(KernelKind::Poly),
            "sigmoid" => Some(KernelKind::Sigmoid),
            "laplacian" => Some(KernelKind::Laplacian),
            "cosine" => Some(KernelKind::Cosine),
            "chi2" => Some(KernelKind::Chi2),
            "additive_chi2" => Some(KernelKind::AdditiveChi2),
            "precomputed" => Some(KernelKind::Precomputed),
            _ => None,
        }
    }

    /// Does this family read `gamma` at all?
    ///
    /// `false` for `linear` / `cosine` / `additive_chi2` / `precomputed`, which
    /// is what lets `fit` skip the `gamma` resolution (and therefore the
    /// `1/n_features` division) for kernels that would only discard it — and,
    /// more importantly, skip REJECTING a `gamma` those kernels never look at,
    /// which sklearn also does not do.
    pub fn uses_gamma(self) -> bool {
        matches!(
            self,
            KernelKind::Rbf
                | KernelKind::Poly
                | KernelKind::Sigmoid
                | KernelKind::Laplacian
                | KernelKind::Chi2
        )
    }

    /// Does `gamma = None` resolve to `1/n_features` for this family?
    ///
    /// True for every γ-taking family EXCEPT `chi2` — see
    /// [`AlgoError::KernelRequiresGamma`] for why that one is the exception.
    pub fn has_gamma_default(self) -> bool {
        self.uses_gamma() && !matches!(self, KernelKind::Chi2)
    }

    /// Does this family require a non-negative input matrix (the chi² pair)?
    pub fn requires_non_negative(self) -> bool {
        matches!(self, KernelKind::Chi2 | KernelKind::AdditiveChi2)
    }
}

/// Kernel ridge regression (KERNEL-01) fitted by the dual-coefficient Cholesky
/// solve of `(K + αI)` over the Phase-8 [`kernel_matrix`] keystone prim.
///
/// Construct with the zero-arg [`KernelRidge::new`] (sklearn defaults:
/// `kernel = linear`, `alpha = 1.0`, `gamma = None`, `degree = 3`, `coef0 = 1`)
/// or [`KernelRidge::builder`], then the consuming [`Fit::fit`] (returns the
/// `Fitted`-tagged sibling) and [`Predict::predict`]. Fitted `dual_coef_` (`n×t`)
/// and `X_fit_` (`n×d`) are device-resident; the host accessor
/// [`dual_coef`](Self::dual_coef) materializes the duals on demand and exists
/// ONLY on `KernelRidge<F, Fitted>` (the compile-time typestate replaces the old
/// runtime `NotFitted` guard, D-03).
pub struct KernelRidge<F, S = Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Which kernel family to build at `fit` (D-01).
    kernel_kind: KernelKind,
    /// L2 penalty strength(s) (`alpha ≥ 0`), added to the `K` diagonal only
    /// (D-06). Length 1 is sklearn's scalar `alpha` and applies to every target;
    /// length `n_targets` is sklearn's array-like `alpha` and penalises each
    /// target column separately. NEVER empty — the builder rejects that.
    alphas: Vec<F>,
    /// Kernel coefficient `γ`; `None` resolves to `1/n_features` at `fit` (D-05).
    gamma: Option<F>,
    /// Polynomial degree (real, `≥ 1`); used by the poly kernel only.
    degree: F,
    /// Independent term `coef0`; used by poly / sigmoid.
    coef0: F,
    /// The resolved precision-typed kernel (gamma resolved, D-05), `None` until
    /// `fit`. Reused VERBATIM by `predict` (Pitfall 5).
    ///
    /// Stays `None` on a FITTED `precomputed` estimator, which is the one kernel
    /// family with no kernel to evaluate. `kernel_kind` is therefore the
    /// authority on which family was selected, and this field only on how to
    /// evaluate it — reading `kernel_.is_none()` as "not fitted" would be wrong.
    kernel_: Option<Kernel<F>>,
    /// The RESOLVED kernel coefficient γ (D-05), meaningful only once fitted; the
    /// families that never read γ resolve it to `0`.
    ///
    /// Redundant with the γ inside `kernel_` for every family that carries one,
    /// and deliberately so: `precomputed` has no `kernel_` to read it back out
    /// of, and the model file's resolved-γ key has to come from somewhere for
    /// EVERY family or the writer needs a per-kernel special case.
    gamma_: F,
    /// Fitted dual coefficients (`n_samples × n_targets`), device-resident,
    /// `None` until `fit`.
    dual_coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// The fitted training matrix `X_fit_` (`n_samples × n_features`),
    /// device-resident, `None` until `fit`. `predict` builds
    /// `K(X_test, X_fit_)` against it.
    x_fit_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted `(n_samples, n_features)` geometry, `None` until `fit`.
    fit_shape_: Option<(usize, usize)>,
    /// Fitted number of targets `t`, `None` until `fit`.
    n_targets_: Option<usize>,
    /// Did the fit fall back from the Cholesky to the host least-squares solve
    /// ([`host_lstsq_solve`])? Reported so the Python shim can raise sklearn's
    /// "Singular matrix in solving dual problem" warning — a caller who gets a
    /// least-squares answer where they asked for an exact one should hear about
    /// it, and sklearn tells them.
    lstsq_fallback_: bool,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> KernelRidge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfitted `KernelRidge` with sklearn's `KernelRidge` defaults
    /// (`kernel = linear`, `alpha = 1.0`, `gamma = None`, `degree = 3`,
    /// `coef0 = 1`) directly in the `Unfit` state. SINGLE source of truth for the
    /// defaults (D-08): the builder `Default` re-derives via
    /// [`KernelRidge::into_builder`]. Defaults are trusted valid, so this bypasses
    /// [`KernelRidgeBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            kernel_kind: KernelKind::Linear,
            alphas: vec![f64_to_host::<F>(1.0)],
            gamma: None,
            degree: f64_to_host::<F>(3.0),
            coef0: f64_to_host::<F>(1.0),
            kernel_: None,
            gamma_: f64_to_host::<F>(0.0),
            dual_coef_: None,
            x_fit_: None,
            fit_shape_: None,
            n_targets_: None,
            lstsq_fallback_: false,
            _state: PhantomData,
        }
    }

    /// Start building a `KernelRidge` from sklearn's defaults (D-08 single
    /// source).
    pub fn builder() -> KernelRidgeBuilder {
        KernelRidgeBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`KernelRidgeBuilder::default`] to re-derive the
    /// defaults from [`KernelRidge::new`] (D-08).
    pub fn into_builder(self) -> KernelRidgeBuilder {
        KernelRidgeBuilder {
            kernel: self.kernel_kind,
            alphas: self.alphas.iter().copied().map(host_to_f64).collect(),
            gamma: self.gamma.map(host_to_f64),
            degree: host_to_f64(self.degree),
            coef0: host_to_f64(self.coef0),
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `dual_coef_`/`x_fit_`/… are excluded — `None` in any `Unfit` value). Used
    /// by the defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        let alphas = |e: &Self| -> Vec<f64> { e.alphas.iter().copied().map(host_to_f64).collect() };
        self.kernel_kind == other.kernel_kind
            && alphas(self) == alphas(other)
            && self.gamma.map(host_to_f64) == other.gamma.map(host_to_f64)
            && host_to_f64(self.degree) == host_to_f64(other.degree)
            && host_to_f64(self.coef0) == host_to_f64(other.coef0)
    }
}

impl<F> Default for KernelRidge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`KernelRidge`] (D-01). `alpha`/`degree`/`coef0` are `f64` (A5:
/// the scalars narrow to `F` at `build::<F>()`); `gamma` is `Option<f64>` and
/// `kernel` takes the [`KernelKind`] enum directly (a non-scalar selector).
/// `Default` re-derives the sklearn defaults from [`KernelRidge::new`] (D-08
/// single source).
#[derive(Debug, Clone)]
pub struct KernelRidgeBuilder {
    kernel: KernelKind,
    alphas: Vec<f64>,
    gamma: Option<f64>,
    degree: f64,
    coef0: f64,
}

impl Default for KernelRidgeBuilder {
    /// Re-derive the sklearn defaults from [`KernelRidge::new`] (D-08 single
    /// source). `f64` is pinned only to read the F-independent scalar defaults.
    fn default() -> Self {
        KernelRidge::<f64, Unfit>::new().into_builder()
    }
}

impl KernelRidgeBuilder {
    /// Set the kernel family (`linear`/`rbf`/`poly`/`sigmoid`). Takes the
    /// [`KernelKind`] enum directly (non-scalar selector).
    pub fn kernel(mut self, v: KernelKind) -> Self {
        self.kernel = v;
        self
    }

    /// Set a SINGLE L2 penalty strength `alpha` (`≥ 0`) applied to every target
    /// — sklearn's scalar `alpha`. The `f64` narrows to `F` at `build::<F>()`
    /// (A5). Equivalent to [`alphas`](Self::alphas) with a one-element vector,
    /// and the two overwrite each other (last call wins).
    pub fn alpha(mut self, v: f64) -> Self {
        self.alphas = vec![v];
        self
    }

    /// Set a PER-TARGET penalty vector — sklearn's array-like `alpha`. Must have
    /// exactly one entry per target column of `y` (checked at `fit`, where the
    /// target count is known), or exactly one entry, which is the scalar case.
    ///
    /// This is not merely a convenience over calling `fit` once per target: the
    /// targets share one kernel matrix, and the per-target path below reuses it
    /// across the `t` solves. What it costs relative to a scalar `alpha` is the
    /// SHARED FACTORISATION — `(K + αI)` differs per target, so the one
    /// multi-RHS Cholesky becomes `t` of them. That is the whole performance
    /// story of this parameter and it is measured in the KERNEL-PARAMS bench.
    pub fn alphas(mut self, v: Vec<f64>) -> Self {
        self.alphas = v;
        self
    }

    /// Set the kernel coefficient `γ` (`None` → `1/n_features` at fit, D-05). The
    /// `Option<f64>` narrows to `Option<F>` at `build::<F>()` (A5).
    pub fn gamma(mut self, v: Option<f64>) -> Self {
        self.gamma = v;
        self
    }

    /// Set the polynomial degree (`≥ 1`; used by the poly kernel only). The `f64`
    /// narrows to `F` at `build::<F>()` (A5).
    pub fn degree(mut self, v: f64) -> Self {
        self.degree = v;
        self
    }

    /// Set the independent term `coef0` (used by poly / sigmoid). The `f64`
    /// narrows to `F` at `build::<F>()` (A5).
    pub fn coef0(mut self, v: f64) -> Self {
        self.coef0 = v;
        self
    }

    /// Build the (unfit) estimator, narrowing the stored `f64` hyperparameters to
    /// the target float `F` (A5). The data-INDEPENDENT `alpha >= 0` check is
    /// relocated here from the old fit body (D-04 / Pitfall 7) →
    /// [`BuildError::InvalidAlpha`]. The `degree >= 1` guard is poly-branch-coupled
    /// (only the poly kernel uses `degree`) and the `gamma` finiteness guard is
    /// resolution-path-coupled (`gamma = None` resolves to `1/n_features` at fit),
    /// so both STAY in the fit body (byte-identical, D-03).
    pub fn build<F>(self) -> Result<KernelRidge<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        // An EMPTY alpha vector is rejected as `alpha = NaN` rather than getting
        // a variant of its own: it is the degenerate spelling of "no penalty was
        // supplied", and every downstream reader of `alphas` assumes at least one
        // entry. Reporting it through the same channel as a negative alpha keeps
        // the caller looking at the same argument.
        if self.alphas.is_empty() {
            return Err(BuildError::InvalidAlpha {
                estimator: "kernel_ridge",
                alpha: f64::NAN,
            });
        }
        if let Some(&bad) = self.alphas.iter().find(|&&a| !(a >= 0.0)) {
            return Err(BuildError::InvalidAlpha {
                estimator: "kernel_ridge",
                alpha: bad,
            });
        }
        // sklearn's `gamma` interval is `[0, inf)`, so a negative gamma is out of
        // domain independently of the data — which puts it at `build()`, beside
        // `alpha`, rather than in the fit body where the None→1/n_features
        // RESOLUTION (and its finiteness check) has to live.
        if let Some(g) = self.gamma {
            if !(g >= 0.0) {
                return Err(BuildError::InvalidGamma {
                    estimator: "kernel_ridge",
                    gamma: g,
                });
            }
        }
        Ok(KernelRidge {
            kernel_kind: self.kernel,
            alphas: self.alphas.iter().copied().map(f64_to_host::<F>).collect(),
            gamma: self.gamma.map(f64_to_host::<F>),
            degree: f64_to_host::<F>(self.degree),
            coef0: f64_to_host::<F>(self.coef0),
            kernel_: None,
            gamma_: f64_to_host::<F>(0.0),
            dual_coef_: None,
            x_fit_: None,
            fit_shape_: None,
            n_targets_: None,
            lstsq_fallback_: false,
            _state: PhantomData,
        })
    }
}

impl<F> KernelRidge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `dual_coef_` (row-major `n_samples × n_targets`).
    /// `Some` by construction on the `Fitted` state, so no `NotFitted` branch is
    /// needed (the compile-time typestate replaces the runtime guard, D-03).
    /// Did this fit reach its duals through the host least-squares fallback
    /// rather than the Cholesky? See [`host_lstsq_solve`] for when that happens.
    pub fn used_lstsq_fallback(&self) -> bool {
        self.lstsq_fallback_
    }

    pub fn dual_coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.dual_coef_
            .as_ref()
            .expect("dual_coef_ is Some by construction on KernelRidge<F, Fitted>")
            .to_host(pool)
    }
}

impl<F> SaveModel for KernelRidge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted regressor to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `X_fit_` | `F` (`F32`/`F64`) | `[n_samples, n_features]` |
    /// | `dual_coef_` | `F` | `[n_samples, n_targets]` |
    /// | `param:kernel` / `param:alpha` / `param:degree` / `param:coef0` | `__metadata__` scalar | — |
    /// | `param:gamma` (optional) / `gamma_` | `__metadata__` scalar | — |
    ///
    /// The training matrix has to be here: a kernel method evaluates against
    /// every training row at predict time, so `X_fit_` is not a fitting artifact
    /// but the model itself — see [`kernel_persist`](crate::kernel_persist) for
    /// why no compressed alternative is offered.
    ///
    /// `n_samples`/`n_features` come off `X_fit_`'s shape at load and
    /// `n_targets` off `dual_coef_`'s, so none is stored again.
    ///
    /// Both the REQUESTED `gamma` and the RESOLVED one are written. Re-running
    /// the `None → 1/n_features` resolution at load instead would put the same
    /// rule in two places with nothing to keep them in step;
    /// [`write_resolved_gamma`] documents the trade.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        let (n_samples, n_features) = self.fit_shape_.ok_or_else(|| absent("fit_shape_"))?;
        let n_targets = self.n_targets_.ok_or_else(|| absent("n_targets_"))?;
        // Bound BEFORE the writer: `KernelWriter` borrows every payload so it
        // can stream them out without a second copy, which means the host
        // buffers must outlive it.
        let x_fit = self.x_fit_.as_ref().ok_or_else(|| absent("x_fit_"))?.to_host(pool);
        let dual_coef = self
            .dual_coef_
            .as_ref()
            .ok_or_else(|| absent("dual_coef_"))?
            .to_host(pool);
        // The RESOLVED coefficient. Read off the dedicated `gamma_` field rather
        // than dug back out of the typed kernel: `precomputed` has no typed
        // kernel to dig into, and a match that has to name every family would
        // need a new arm every time one is added.
        let resolved_gamma = host_to_f64(self.gamma_);
        let alphas: Vec<F> = self.alphas.clone();

        let mut w = KernelWriter::new(PERSIST_TAG);
        w.scalar_str(KERNEL_KEY, self.kernel_kind.name());
        // BOTH spellings of `alpha`. The `ALPHA_NAME` tensor is the real one (it
        // is the only one that can hold a per-target vector); the scalar is the
        // first entry, written so a file produced here still loads in a build
        // that predates the per-target `alpha` and reads only the scalar. A
        // reader that takes the scalar from a genuinely per-target model gets the
        // first target's penalty rather than a parse failure — which is the
        // trade this key exists to make, and the reason the tensor wins when
        // both are present.
        w.scalar_f64("param:alpha", host_to_f64(alphas[0]));
        w.scalar_f64("param:degree", host_to_f64(self.degree));
        w.scalar_f64("param:coef0", host_to_f64(self.coef0));
        write_resolved_gamma(&mut w, self.gamma.map(host_to_f64), resolved_gamma);
        write_x_fit(&mut w, &x_fit, n_samples, n_features)?;
        w.tensor(ALPHA_NAME, TensorRef::floats(&alphas, vec![alphas.len()])?);
        w.tensor(
            DUAL_COEF_NAME,
            TensorRef::floats(&dual_coef, vec![n_samples, n_targets])?,
        );
        w.write(path)
    }
}

impl<F> LoadModel for KernelRidge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the regressor back from `path`, re-uploading `X_fit_` and
    /// `dual_coef_` to `pool`.
    ///
    /// The typed [`Kernel`] `predict` consumes is REBUILT here from the stored
    /// kind and the stored RESOLVED coefficient — not re-resolved from
    /// `param:gamma` and the feature count. That is what makes the file the
    /// authority on what the saved model computed, rather than the build that
    /// happens to read it.
    ///
    /// The file is untrusted input (T-04-01-01), so `X_fit_` defines
    /// `n_samples` and `dual_coef_`'s row extent is checked against it before
    /// any value is stored — a mismatch would otherwise index the training set
    /// out of range on the first prediction.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<KernelRidge<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = KernelFile::parse(&raw, PERSIST_TAG)?;
        let (x_fit, n_samples, n_features) = read_x_fit::<F>(&file)?;

        let dual_v = file.tensor(DUAL_COEF_NAME)?;
        let (dual_rows, n_targets) = shape_2d(&dual_v, DUAL_COEF_NAME)?;
        expect_len(DUAL_COEF_NAME, dual_rows, n_samples, "rows")?;
        if n_targets == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{DUAL_COEF_NAME}' declares 0 targets; a fitted \
                     KernelRidge has at least one"
                ),
            });
        }
        let dual_coef = crate::kernel_persist::as_floats::<F>(&dual_v, DUAL_COEF_NAME)?;

        let kernel_kind = KernelKind::from_name(file.scalar_str(KERNEL_KEY)?)
            .ok_or(PersistError::BadMetadata { key: KERNEL_KEY })?;
        let (gamma_request, gamma_resolved) = read_resolved_gamma(&file)?;
        let degree = f64_to_host::<F>(file.scalar_f64("param:degree")?);
        let coef0 = f64_to_host::<F>(file.scalar_f64("param:coef0")?);
        let gamma = f64_to_host::<F>(gamma_resolved);

        // The penalty vector, preferring the tensor (which can be per-target) and
        // falling back to the legacy scalar for a file written before `alpha`
        // could be one. An EMPTY tensor is rejected rather than silently taken as
        // "unpenalised": every reader downstream indexes `alphas[0]`.
        let alphas: Vec<F> = match file.tensor_opt(ALPHA_NAME) {
            Some(v) => {
                let a = crate::kernel_persist::as_floats::<F>(&v, ALPHA_NAME)?;
                if a.is_empty() {
                    return Err(PersistError::InconsistentGeometry {
                        reason: format!(
                            "tensor '{ALPHA_NAME}' is empty; a fitted KernelRidge \
                             has at least one penalty"
                        ),
                    });
                }
                a.to_vec()
            }
            None => vec![f64_to_host::<F>(file.scalar_f64("param:alpha")?)],
        };
        let n_alphas = alphas.len();
        if n_alphas != 1 && n_alphas != n_targets {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{ALPHA_NAME}' has {n_alphas} entries but \
                     '{DUAL_COEF_NAME}' declares {n_targets} targets"
                ),
            });
        }

        Ok(KernelRidge {
            kernel_kind,
            alphas,
            gamma: gamma_request.map(f64_to_host::<F>),
            degree,
            coef0,
            kernel_: typed_kernel(kernel_kind, gamma, degree, coef0),
            gamma_: gamma,
            dual_coef_: Some(DeviceArray::from_host(pool, &dual_coef)),
            x_fit_: Some(DeviceArray::from_host(pool, &x_fit)),
            fit_shape_: Some((n_samples, n_features)),
            n_targets_: Some(n_targets),
            // NOT persisted: it describes how the ORIGINAL fit reached these
            // duals, and the file carries the duals themselves. A loaded model
            // has not solved anything, so claiming a fallback it did not perform
            // would be the misleading answer.
            lstsq_fallback_: false,
            _state: PhantomData,
        })
    }
}

/// Solve `A·x = b` (multi-RHS, `A` `n×n` row-major, `b` `n×rhs` row-major) on
/// the HOST in `f64` when the Cholesky has refused the matrix — sklearn's
/// `except np.linalg.LinAlgError: dual_coef = linalg.lstsq(K, y)[0]`.
///
/// ## Why this path exists at all
/// `(K + αI)` is only positive definite when `K` is, and two of the kernels
/// sklearn ships are not: `additive_chi2` has a zero diagonal and non-positive
/// entries everywhere else, so at the DEFAULT `alpha = 1` its regularised Gram
/// is plainly indefinite, and `sigmoid` is indefinite for most
/// `(γ, coef0)`. sklearn reaches those configurations by catching LAPACK's
/// refusal and re-solving in the least-squares sense; without the same
/// fallback, `kernel='additive_chi2'` would raise here for every input, which
/// is not "the parameter is supported".
///
/// ## Two solvers, one entry point
/// An indefinite matrix is usually NONSINGULAR — LAPACK's `POSV` declines it
/// for the pivot's sign, not for a rank deficiency — and for a nonsingular `A`
/// the least-squares solution IS `A⁻¹b`. So the fast path is Gaussian
/// elimination with partial pivoting, `O(n³/3)`, which is exact there. Only
/// when elimination finds a negligible pivot — a genuinely rank-deficient
/// system, which needs `lstsq`'s MINIMUM-NORM answer rather than any solution —
/// does this fall through to the symmetric-eigendecomposition pseudo-inverse,
/// which is `O(n³)` per Jacobi sweep and would be the wrong default to pay.
///
/// `f64` throughout regardless of `F`: this is the arm that runs when the
/// matrix is already ill-behaved, and it is not on any hot path.
fn host_lstsq_solve(a: &[f64], b: &[f64], n: usize, rhs: usize) -> Vec<f64> {
    // --- Gaussian elimination with partial pivoting on [A | b]. ---
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    // Rank tolerance, scaled to the matrix: an absolute epsilon would call a
    // well-conditioned matrix of tiny entries singular.
    let scale = a.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let tol = f64::EPSILON * (n as f64) * scale.max(f64::MIN_POSITIVE);

    for k in 0..n {
        let (piv, piv_abs) = (k..n).fold((k, 0.0f64), |(bi, bv), i| {
            let v = m[perm[i] * n + k].abs();
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        });
        if piv_abs <= tol {
            // Rank-deficient: elimination cannot produce the minimum-norm
            // solution, so hand the whole problem to the pseudo-inverse.
            return sym_pinv_solve(a, b, n, rhs);
        }
        perm.swap(k, piv);
        let pk = perm[k];
        let pivot = m[pk * n + k];
        for i in (k + 1)..n {
            let pi = perm[i];
            let f = m[pi * n + k] / pivot;
            if f == 0.0 {
                continue;
            }
            m[pi * n + k] = 0.0;
            for j in (k + 1)..n {
                m[pi * n + j] -= f * m[pk * n + j];
            }
            for t in 0..rhs {
                x[pi * rhs + t] -= f * x[pk * rhs + t];
            }
        }
    }
    // Back substitution, writing the answer in the ORIGINAL row order.
    let mut out = vec![0.0f64; n * rhs];
    for k in (0..n).rev() {
        let pk = perm[k];
        for t in 0..rhs {
            let mut s = x[pk * rhs + t];
            for j in (k + 1)..n {
                s -= m[pk * n + j] * out[j * rhs + t];
            }
            out[k * rhs + t] = s / m[pk * n + k];
        }
    }
    out
}

/// Minimum-norm least-squares solve of a SYMMETRIC `A` via its
/// eigendecomposition — the rank-deficient arm of [`host_lstsq_solve`].
///
/// For symmetric `A = VΛVᵀ` the SVD is `|Λ|` with `U = V·sign(Λ)`, so
/// `lstsq`'s min-norm solution `Σ_{σᵢ>tol} (uᵢᵀb/σᵢ)vᵢ` collapses to
/// `Σ_{|λᵢ|>tol} (vᵢᵀb/λᵢ)vᵢ` — the signs cancel, and no SVD is needed. The
/// cutoff mirrors `scipy.linalg.lstsq`'s default `cond`: singular values below
/// `ε · max σ` are treated as zero.
///
/// Every `A` reaching here is a kernel matrix plus a diagonal, hence symmetric.
/// A caller-supplied `precomputed` `K` that is NOT symmetric would get the
/// symmetric part's answer; sklearn would run a general SVD and could differ.
/// That is a documented limit of this arm, not an oversight — a non-symmetric
/// kernel matrix is not a kernel matrix.
fn sym_pinv_solve(a: &[f64], b: &[f64], n: usize, rhs: usize) -> Vec<f64> {
    // Cyclic Jacobi: rotate away the largest off-diagonal pairs, sweep by sweep,
    // accumulating the rotations into V.
    let mut m = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let frob = |m: &[f64]| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    s += m[i * n + j] * m[i * n + j];
                }
            }
        }
        s.sqrt()
    };
    let scale = a.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
    let thresh = 8.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE) * (n as f64);
    for _ in 0..60 {
        if frob(&m) <= thresh {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() <= thresh / (n as f64) {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                // NOT `theta.signum()`: Rust's `signum` returns 0.0 for 0.0, and
                // `theta == 0` (equal diagonal entries) is the case that needs
                // the LARGEST rotation, `t = 1`, not none at all. A zero `t`
                // there leaves the off-diagonal in place and the sweep spins to
                // its cap without converging.
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = m[k * n + p];
                    let akq = m[k * n + q];
                    m[k * n + p] = c * akp - s * akq;
                    m[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = m[p * n + k];
                    let aqk = m[q * n + k];
                    m[p * n + k] = c * apk - s * aqk;
                    m[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let lambda: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    let max_sv = lambda.iter().fold(0.0f64, |acc, l| acc.max(l.abs()));
    let cut = f64::EPSILON * max_sv;

    let mut out = vec![0.0f64; n * rhs];
    for i in 0..n {
        if lambda[i].abs() <= cut {
            continue;
        }
        for t in 0..rhs {
            // (vᵢᵀ b_t) / λᵢ, then scattered back along vᵢ.
            let mut dot = 0.0;
            for k in 0..n {
                dot += v[k * n + i] * b[k * rhs + t];
            }
            let coef = dot / lambda[i];
            for k in 0..n {
                out[k * rhs + t] += coef * v[k * n + i];
            }
        }
    }
    out
}

/// Build the precision-typed [`Kernel`] the prim consumes from the selected
/// family and the RESOLVED hyperparameters.
///
/// `None` for [`KernelKind::Precomputed`] — that family names the absence of a
/// kernel evaluation, so there is no `Kernel<F>` to build and both `fit` and
/// `predict` branch on the `None` rather than on the kind a second time.
fn typed_kernel<F>(kind: KernelKind, gamma: F, degree: F, coef0: F) -> Option<Kernel<F>>
where
    F: Float + CubeElement + Pod,
{
    Some(match kind {
        KernelKind::Linear => Kernel::Linear,
        KernelKind::Rbf => Kernel::Rbf { gamma },
        KernelKind::Poly => Kernel::Poly {
            gamma,
            degree,
            coef0,
        },
        KernelKind::Sigmoid => Kernel::Sigmoid { gamma, coef0 },
        KernelKind::Laplacian => Kernel::Laplacian { gamma },
        KernelKind::Cosine => Kernel::Cosine,
        KernelKind::Chi2 => Kernel::Chi2 { gamma },
        KernelKind::AdditiveChi2 => Kernel::AdditiveChi2,
        KernelKind::Precomputed => return None,
    })
}

/// Reject a negative entry in a chi²-kernel operand, naming the first one
/// (sklearn's `check_non_negative`, which is what `chi2_kernel` /
/// `additive_chi2_kernel` run before touching the data).
fn check_non_negative<F>(
    values: &[F],
    kind: KernelKind,
    operand: &'static str,
) -> Result<(), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    // `!(v >= 0.0)` rather than `v < 0.0`, so a NaN is caught here too: it would
    // otherwise reach the kernel's `nom > 0` guard, which a NaN fails, silently
    // DROPPING the feature instead of propagating.
    if let Some(idx) = values.iter().position(|&v| !(host_to_f64(v) >= 0.0)) {
        return Err(AlgoError::NegativeKernelInput {
            estimator: "kernel_ridge",
            kernel: kind.name(),
            operand,
            index: idx,
            value: host_to_f64(values[idx]),
        });
    }
    Ok(())
}

/// Least-squares duals for a `(K + αI)` the Cholesky refused — the host arm of
/// [`host_lstsq_solve`], staged back to the device in the layout the caller
/// expects. `k_reg` is the ALREADY-REGULARIZED matrix (α on the diagonal).
fn lstsq_duals<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    k_reg: &[F],
    rhs: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_targets: usize,
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let a: Vec<f64> = k_reg.iter().copied().map(host_to_f64).collect();
    let b: Vec<f64> = rhs.to_host(pool).iter().copied().map(host_to_f64).collect();
    let x = host_lstsq_solve(&a, &b, n_samples, n_targets);
    let x: Vec<F> = x.into_iter().map(f64_to_host::<F>).collect();
    DeviceArray::from_host(pool, &x)
}

/// Solve `(K + αⱼI)·cⱼ = yⱼ` once per target for DISTINCT per-target alphas —
/// sklearn's `_solve_cholesky_kernel` else-branch.
///
/// The kernel matrix is shared (it does not depend on `α`), so what this costs
/// over the one-alpha path is exactly `t` factorisations instead of one. `K` is
/// re-staged per target from the same host copy rather than mutated in place on
/// the device, because the Cholesky consumes its input buffer as the factor's
/// scratch — the host copy is the only thing that survives a solve, which is why
/// it is the loop's source rather than an optimisation to remove.
///
/// Returns the `n_samples × n_targets` row-major duals, matching the layout the
/// one-alpha multi-RHS solve produces, so the caller cannot tell which path ran,
/// plus whether any target took the least-squares fallback.
fn solve_per_target_alphas<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    k_host: &[F],
    rhs: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_targets: usize,
    alphas: &[f64],
) -> Result<(DeviceArray<ActiveRuntime, F>, bool), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let rhs_host = rhs.to_host(pool);
    let mut duals = vec![f64_to_host::<F>(0.0); n_samples * n_targets];
    let mut k_scratch = vec![f64_to_host::<F>(0.0); n_samples * n_samples];
    let mut fell_back = false;

    for (t, &alpha) in alphas.iter().enumerate().take(n_targets) {
        k_scratch.copy_from_slice(k_host);
        for i in 0..n_samples {
            let d = host_to_f64(k_scratch[i * n_samples + i]) + alpha;
            k_scratch[i * n_samples + i] = f64_to_host::<F>(d);
        }
        // This target's column of y, gathered out of the row-major (n × t) block.
        let col: Vec<F> = (0..n_samples).map(|i| rhs_host[i * n_targets + t]).collect();

        let k_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &k_scratch);
        let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &col);
        let k_out = DeviceArray::<ActiveRuntime, F>::from_raw(
            k_dev.handle().clone(),
            n_samples * n_samples,
        );
        let c = cholesky_solve::<F>(pool, &k_dev, &y_dev, n_samples, 1, Some(k_out));
        drop(k_dev);
        let c_host = match c {
            Ok(c) => {
                let h = c.to_host(pool);
                c.release_into(pool);
                h
            }
            Err(PrimError::NotPositiveDefinite { .. }) => {
                // Per-target, like the solve it replaces: one target hitting an
                // indefinite `(K + αⱼI)` must not force the others onto the slow
                // arm, and with distinct alphas it genuinely can be only one.
                fell_back = true;
                let d = lstsq_duals::<F>(pool, &k_scratch, &y_dev, n_samples, 1);
                let h = d.to_host(pool);
                d.release_into(pool);
                h
            }
            Err(e) => {
                y_dev.release_into(pool);
                return Err(AlgoError::Prim(e));
            }
        };
        y_dev.release_into(pool);
        for i in 0..n_samples {
            duals[i * n_targets + t] = c_host[i];
        }
    }
    Ok((DeviceArray::from_host(pool, &duals), fell_back))
}

impl<F> Fit<F> for KernelRidge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = KernelRidge<F, Fitted>;

    /// Fit `(K + αI)·dual_coef_ = y` over the kernel matrix (D-06), CONSUMING
    /// `self`.
    ///
    /// `x` is `(n_samples × n_features)` row-major; `y` is `(n_samples ×
    /// n_targets)` row-major (a single target is `t = 1`). Validates the
    /// hyperparameters and geometry BEFORE any launch (T-08-03-01), resolves
    /// `gamma` (D-05), builds `K = kernel_matrix(X, X, kernel)`, adds `α` to the
    /// `K` diagonal only (D-06), and solves the multi-RHS dual in one
    /// [`cholesky_solve`] (`rhs = t`, D-04). NO centering, NO intercept. `n_targets`
    /// is passed via the `y` geometry: `y.len() == n_samples * n_targets`.
    ///
    /// The [`Fit`] trait's fixed signature carries no `n_targets` slot, so it is
    /// recovered from `y`'s length (`y.len() / n_samples`); a `y` whose length is
    /// not a positive multiple of `n_samples` is rejected as a `ShapeMismatch`
    /// (byte-identical behaviour to the old explicit `n_targets` guard).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<KernelRidge<F, Fitted>, AlgoError> {
        self.fit_weighted(pool, x, y, shape, None)
    }
}

impl<F> KernelRidge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// [`Fit::fit`] with sklearn's third `fit` argument, `sample_weight`.
    ///
    /// The [`Fit`] trait's signature is fixed and carries no weight slot, so the
    /// weighted entry point is inherent and [`Fit::fit`] is the `None` case of
    /// it — one body, so the unweighted path cannot drift from the weighted one.
    ///
    /// ## What a weight does to a DUAL problem
    /// The primal reweighting `Σ wᵢ(yᵢ − f(xᵢ))²` has no `wᵢ` to attach to in the
    /// dual, where the data appear only through `K`. sklearn's
    /// `_solve_cholesky_kernel` gets there by a symmetric similarity transform:
    /// with `s = √w`, solve `(SKS + αI)·c̃ = S·y` and recover `c = S·c̃`, where
    /// `S = diag(s)`. That is reproduced verbatim here — `K *= outer(s, s)`,
    /// `y *= s`, solve, `duals *= s` — on the same host pass that injects `α`,
    /// so weighting costs one extra `n²` multiply and no extra device work.
    ///
    /// A zero weight is legal and drops the sample; ALL-zero is rejected
    /// ([`AlgoError::ZeroSampleWeightSum`]), as is a negative or non-finite
    /// weight ([`AlgoError::InvalidSampleWeight`]) — sklearn would take the
    /// square root of a negative and propagate NaN through the whole solve.
    pub fn fit_weighted(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<KernelRidge<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        let y = y.ok_or(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "y",
            rows: n_samples,
            cols: 0,
            len: 0,
        }))?;
        // Recover n_targets from y's length (the Fit trait carries no n_targets
        // slot). y.len() must be a POSITIVE MULTIPLE of n_samples. WR-05: enforce
        // the divisibility intent explicitly here rather than relying on the
        // post-hoc `y.len() == n_samples * n_targets` equality below, so a future
        // refactor that relaxes that clause cannot let a non-multiple y through
        // with a silently-truncated target count.
        if n_samples == 0 || y.len() == 0 || y.len() % n_samples != 0 {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 0,
                len: y.len(),
            }));
        }
        let n_targets = y.len() / n_samples;

        // --- T-08-03-01 / ASVS V5: validate the untrusted hyperparameters and
        //     geometry BEFORE any prim launch. alpha < 0 is now rejected at
        //     build() → BuildError (data-INDEPENDENT, relocated D-04); degree < 1
        //     is not a valid poly order; a non-finite resolved gamma (checked once
        //     gamma is resolved, below) drives the device kernels to NaN; the
        //     kernel name is fixed by KernelKind (always valid here, but the guard
        //     mirrors the threat register T-08-03-01). ---
        let degree64 = host_to_f64(self.degree);
        // sklearn's degree interval is `[0, inf)` — a FRACTIONAL degree is legal
        // and is why the poly map evaluates `F::powf`. Only a negative (or NaN,
        // which fails the `>= 0` test) order is rejected.
        if self.kernel_kind == KernelKind::Poly && !(degree64 >= 0.0) {
            return Err(AlgoError::InvalidDegree {
                estimator: "kernel_ridge",
                degree: degree64,
            });
        }
        validate_geometry(x, shape)?;
        if n_targets == 0 || y.len() != n_samples * n_targets {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: n_targets,
                len: y.len(),
            }));
        }
        // `alpha` is a scalar (one entry, broadcast) or one entry per target;
        // anything else would leave some target's penalty unspecified.
        if self.alphas.len() != 1 && self.alphas.len() != n_targets {
            return Err(AlgoError::AlphaTargetMismatch {
                estimator: "kernel_ridge",
                n_alphas: self.alphas.len(),
                n_targets,
            });
        }
        let sw = crate::linear::ridge::validate_sample_weight::<F>(
            "kernel_ridge",
            sample_weight,
            n_samples,
        )?;
        // `precomputed` means the caller handed us `K` itself, so `X` must be the
        // square training kernel. Checked here rather than left to the Cholesky's
        // NotSquare, whose message would describe an internal buffer rather than
        // the argument the caller got wrong.
        if self.kernel_kind == KernelKind::Precomputed && n_samples != n_features {
            return Err(AlgoError::PrecomputedNotSquare {
                estimator: "kernel_ridge",
                rows: n_samples,
                cols: n_features,
            });
        }
        // The chi² pair is defined on histogram-like data; sklearn's
        // `check_non_negative` rejects a negative entry before computing anything
        // and so do we (see `AlgoError::NegativeKernelInput` for why the kernel's
        // own zero-denominator guard makes silence the worse alternative).
        let x_host = x.to_host(pool);
        if self.kernel_kind.requires_non_negative() {
            check_non_negative(&x_host, self.kernel_kind, "X")?;
        }

        // --- gamma resolution (D-05): None → 1/n_features computed from the
        //     fitted feature count; explicit gamma as-is. The RESOLVED value is
        //     baked into the typed Kernel<F> so predict reuses it (Pitfall 5).
        //     `chi2` is the one γ-taking family with NO default (sklearn raises
        //     rather than resolving), and the families that never read γ resolve
        //     to a placeholder that is stored but never evaluated. ---
        let gamma = match self.gamma {
            Some(g) => g,
            None if self.kernel_kind.has_gamma_default() => {
                f64_to_host::<F>(1.0 / n_features as f64)
            }
            None if self.kernel_kind == KernelKind::Chi2 => {
                return Err(AlgoError::KernelRequiresGamma {
                    estimator: "kernel_ridge",
                    kernel: "chi2",
                });
            }
            // linear / cosine / additive_chi2 / precomputed: γ is inert. Store a
            // definite 0 rather than leave it unresolved so the model file has one
            // value to round-trip and `predict` cannot pick a different one.
            None => f64_to_host::<F>(0.0),
        };
        // Validate-before-launch (T-08-03-01 / ASVS V5): the resolved gamma is
        // baked into the typed Kernel and consumed on device by
        // rbf/poly/sigmoid/laplacian/chi2 (`exp`/`powf`/`tanh`); a non-finite
        // user-supplied gamma (or a degenerate resolved default — `n_features` is
        // non-zero, but an explicit `gamma=inf` is not) drives those device ops to
        // NaN. Reject it here so the untrusted hyperparameter becomes a typed
        // error, never NaN duals.
        let gamma64 = host_to_f64(gamma);
        if !gamma64.is_finite() {
            return Err(AlgoError::InvalidGamma {
                estimator: "kernel_ridge",
                gamma: gamma64,
            });
        }
        let kernel = typed_kernel(self.kernel_kind, gamma, self.degree, self.coef0);

        // --- K: the n×n training Gram (Y = X, D-02). NO centering — the normal
        //     matrix is K, not XᵀX (D-06). Under `precomputed` there is no kernel
        //     to evaluate: the caller's `X` IS K, and the host copy already read
        //     above is it. ---
        let x_fit_host = x_host;
        let k_host_src: Vec<F> = match kernel {
            None => x_fit_host.clone(),
            Some(kernel) => {
                let k = kernel_matrix::<F>(
                    pool,
                    x,
                    (n_samples, n_features),
                    x,
                    (n_samples, n_features),
                    kernel,
                    None,
                )?;
                let h = k.to_host(pool);
                k.release_into(pool);
                h
            }
        };

        // --- sample_weight, sklearn's symmetric similarity transform: with
        //     `s = √w`, `K ← S·K·S` (i.e. `K[i][j] *= s[i]·s[j]`) and `y ← S·y`,
        //     solved as usual, with `duals ← S·duals` at the end. Folded into the
        //     SAME host pass that injects α — the matrix is already on the host
        //     for that, so weighting adds one multiply per element and no round
        //     trip of its own. ---
        let mut k_host = k_host_src;
        let sqrt_w: Option<Vec<f64>> =
            sw.as_ref().map(|w| w.iter().map(|v| v.sqrt()).collect());
        if let Some(s) = &sqrt_w {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    let v = host_to_f64(k_host[i * n_samples + j]) * s[i] * s[j];
                    k_host[i * n_samples + j] = f64_to_host::<F>(v);
                }
            }
        }
        // The weighted right-hand side. Without weights the caller's `y` device
        // buffer is used directly and nothing is staged.
        let y_scaled: Option<DeviceArray<ActiveRuntime, F>> = sqrt_w.as_ref().map(|s| {
            let mut yh = y.to_host(pool);
            for i in 0..n_samples {
                for t in 0..n_targets {
                    let v = host_to_f64(yh[i * n_targets + t]) * s[i];
                    yh[i * n_targets + t] = f64_to_host::<F>(v);
                }
            }
            DeviceArray::from_host(pool, &yh)
        });
        let rhs: &DeviceArray<ActiveRuntime, F> = y_scaled.as_ref().unwrap_or(y);

        // --- alpha on the K DIAGONAL only (D-06). There is no intercept to leave
        //     unpenalized. cubecl 0.10 has no in-place device scalar write, so the
        //     small n×n K is materialized on the host, α goes on the diagonal, and
        //     the regularized matrix is staged back — `from_host` recycles the
        //     just-freed n² bytes off the pool free-list, so no parallel n² buffer
        //     lives (the ridge.rs diagonal-α host pass). `alpha >= 0` is enforced
        //     at build() (BuildError::InvalidAlpha, relocated D-04).
        //
        //     ONE alpha (the scalar case, and the per-target case where every
        //     entry happens to be equal) means one `(K + αI)` and therefore ONE
        //     multi-RHS Cholesky across all t targets — the factorisation is
        //     shared and the extra targets are nearly free (D-04). DISTINCT
        //     per-target alphas mean t DIFFERENT matrices, so the shared
        //     factorisation is gone and each target pays its own O(n³). sklearn
        //     splits on exactly this test (`(alpha == alpha[0]).all()`), and the
        //     split is the performance story of the array-valued `alpha`. ---
        let alphas64: Vec<f64> = self.alphas.iter().copied().map(host_to_f64).collect();
        let one_alpha = alphas64.iter().all(|&a| a == alphas64[0]);

        let mut lstsq_fallback = false;
        let dual_coef = if one_alpha {
            let alpha64 = alphas64[0];
            for i in 0..n_samples {
                let d = host_to_f64(k_host[i * n_samples + i]) + alpha64;
                k_host[i * n_samples + i] = f64_to_host::<F>(d);
            }
            let k_reg: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &k_host);
            // Thread the regularized K buffer through `out` so the factor reuses
            // it in place — no parallel n² allocation (mirrors ridge.rs). A
            // non-SPD pivot surfaces NotPositiveDefinite → AlgoError
            // (T-08-03-02), never NaN duals.
            let k_out = DeviceArray::<ActiveRuntime, F>::from_raw(
                k_reg.handle().clone(),
                n_samples * n_samples,
            );
            let duals = cholesky_solve::<F>(pool, &k_reg, rhs, n_samples, n_targets, Some(k_out));
            // The K buffer was consumed (its cloned handle threaded through `out`
            // and released by the solve), so it is NOT released again here —
            // dropping the wrapper is the whole cleanup.
            drop(k_reg);
            match duals {
                Ok(d) => d,
                Err(PrimError::NotPositiveDefinite { .. }) => {
                    lstsq_fallback = true;
                    lstsq_duals::<F>(pool, &k_host, rhs, n_samples, n_targets)
                }
                Err(e) => return Err(AlgoError::Prim(e)),
            }
        } else {
            let (d, fell_back) =
                solve_per_target_alphas::<F>(pool, &k_host, rhs, n_samples, n_targets, &alphas64)?;
            lstsq_fallback = fell_back;
            d
        };
        if let Some(ys) = y_scaled {
            ys.release_into(pool);
        }

        // --- Post-solve finiteness guard (CR-01 / T-08-03-02). A non-SPD pivot
        //     normally surfaces NotPositiveDefinite from the primitive, but a NaN
        //     reaching the Cholesky diagonal does NOT reliably trip the `pivot <= 0`
        //     test (`NaN <= 0` is `false`), so a poly kernel with a negative base
        //     (`gamma·g + coef0 < 0`, non-integer degree) can produce NaN duals
        //     silently. Read the small n×t duals back and reject any non-finite
        //     value as NotPositiveDefinite so the module-doc "never a NaN
        //     dual_coef_" guarantee actually holds.
        //
        //     The `duals ← S·duals` un-weighting rides along on the readback the
        //     guard already performs, so weighting costs no extra transfer. ---
        let mut duals_host = dual_coef.to_host(pool);
        if let Some(idx) = duals_host
            .iter()
            .position(|&v| !host_to_f64(v).is_finite())
        {
            dual_coef.release_into(pool);
            return Err(AlgoError::Prim(PrimError::NotPositiveDefinite {
                operand: "kernel_ridge",
                pivot_index: idx,
                pivot_value: host_to_f64(duals_host[idx]),
            }));
        }
        let dual_coef = match &sqrt_w {
            None => dual_coef,
            Some(s) => {
                for i in 0..n_samples {
                    for t in 0..n_targets {
                        let v = host_to_f64(duals_host[i * n_targets + t]) * s[i];
                        duals_host[i * n_targets + t] = f64_to_host::<F>(v);
                    }
                }
                dual_coef.release_into(pool);
                DeviceArray::from_host(pool, &duals_host)
            }
        };

        // --- Store device-resident fitted state. `x_host` was read at the top
        //     (the chi² non-negativity scan and the `precomputed` K both needed
        //     it), so `X_fit_` is staged from that copy rather than re-read.
        //     Under `precomputed` this stores the caller's K, which is what
        //     sklearn's `X_fit_` holds there too. ---
        let x_fit: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_fit_host);

        // --- Reconstruct into the `Fitted`-tagged sibling. The consuming-`self`
        //     transition means there is no prior device-resident fitted state to
        //     release: a freshly-built `Unfit` carries `dual_coef_`/`x_fit_` =
        //     `None` (the old re-fit buffer-release pass is therefore vacuous and
        //     dropped — the IncrementalPCA / KernelDensity reset precedent,
        //     16-04/16-07). ---
        Ok(KernelRidge {
            kernel_kind: self.kernel_kind,
            alphas: self.alphas,
            gamma: self.gamma,
            degree: self.degree,
            coef0: self.coef0,
            kernel_: kernel,
            gamma_: gamma,
            dual_coef_: Some(dual_coef),
            x_fit_: Some(x_fit),
            fit_shape_: Some((n_samples, n_features)),
            n_targets_: Some(n_targets),
            lstsq_fallback_: lstsq_fallback,
            _state: PhantomData,
        })
    }
}

impl<F> Predict<F> for KernelRidge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Predict `y = K(X_test, X_fit_) · dual_coef_` (D-06).
    ///
    /// `x_test` is `(n_test × n_features)` row-major. Builds `K_test =
    /// kernel_matrix(X_test, X_fit_, kernel)` (`m×n`) with the RESOLVED fit-time
    /// kernel (gamma reused, Pitfall 5), then `y_pred = K_test · dual_coef_`
    /// (`m×t`) via [`gemm`]. NO intercept broadcast (D-06). Returns the row-major
    /// `(n_test × n_targets)` predictions; for a single target this is length
    /// `n_test`. The fitted state is `Some` by construction on `KernelRidge<F,
    /// Fitted>` (the compile-time typestate replaces the old runtime `NotFitted`
    /// guard, D-03); errors only on a geometry / feature-count mismatch.
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x_test: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_test, n_features) = shape;

        // `kernel_` is `None` exactly for `precomputed` — see the field's doc.
        let kernel = self.kernel_;
        let dual_coef = self
            .dual_coef_
            .as_ref()
            .expect("dual_coef_ is Some by construction on KernelRidge<F, Fitted>");
        let x_fit = self
            .x_fit_
            .as_ref()
            .expect("x_fit_ is Some by construction on KernelRidge<F, Fitted>");
        let (n_samples, fit_features) = self
            .fit_shape_
            .expect("fit_shape_ is Some by construction on KernelRidge<F, Fitted>");
        let n_targets = self
            .n_targets_
            .expect("n_targets_ is Some by construction on KernelRidge<F, Fitted>");

        // --- T-08-03-01 / ASVS V5: geometry + fitted-n_features consistency. ---
        if n_test == 0 || n_features == 0 || x_test.len() != n_test * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x_test",
                rows: n_test,
                cols: n_features,
                len: x_test.len(),
            }));
        }
        if n_features != fit_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: fit_features,
            }));
        }

        // --- K_test (m×n). For every evaluated family this is
        //     `kernel_matrix(X_test, X_fit_, kernel)` — the cross kernel against
        //     the stored training matrix, reusing the RESOLVED fit-time kernel so
        //     the gamma is identical (Pitfall 5). Under `precomputed` the caller
        //     already supplied it as `x_test`, and the `n_features == fit_features`
        //     guard above is precisely sklearn's `X.shape[1] == X_fit_.shape[0]`
        //     check, since `fit_features == n_samples` for a square fit K. ---
        let (k_test, k_test_is_borrowed) = match kernel {
            Some(kernel) => {
                if self.kernel_kind.requires_non_negative() {
                    let host = x_test.to_host(pool);
                    check_non_negative(&host, self.kernel_kind, "X_test")?;
                }
                let k = kernel_matrix::<F>(
                    pool,
                    x_test,
                    (n_test, n_features),
                    x_fit,
                    (n_samples, fit_features),
                    kernel,
                    None,
                )?;
                (k, false)
            }
            // Borrowed: `x_test` IS K_test, and it belongs to the caller — the
            // release below must not touch it.
            None => (
                DeviceArray::<ActiveRuntime, F>::from_raw(x_test.handle().clone(), x_test.len()),
                true,
            ),
        };

        // --- y_pred = K_test · dual_coef_ (m×t) via gemm. NO intercept broadcast
        //     (D-06 — the normal matrix was K, not XᵀX; there is no bias). ---
        let pred = gemm::<F>(
            pool,
            &k_test,
            (n_test, n_samples),
            dual_coef,
            (n_samples, n_targets),
            false,
            false,
            None,
        );
        if k_test_is_borrowed {
            // A view over the caller's buffer — drop the wrapper, never release
            // the allocation.
            drop(k_test);
        } else {
            k_test.release_into(pool);
        }
        Ok(pred?)
    }
}
