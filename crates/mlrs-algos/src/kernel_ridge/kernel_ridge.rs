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
//! across the `t` right-hand sides), producing `dual_coef_` (`n×t`).
//!
//! ## gamma resolution (D-05)
//! `gamma = None` resolves to `1/n_features` at `fit` (computed from
//! `X.shape[1]`); an explicit `gamma` is used as-is. The RESOLVED value is stored
//! inside the typed [`Kernel`] so `predict` reuses the IDENTICAL kernel
//! (RESEARCH Pitfall 5 — the fit-time and predict-time gamma MUST match).
//!
//! ## alpha on the diagonal only (D-06)
//! `α` is added to the `K` DIAGONAL only (`K[i·n+i] += alpha`) — the same
//! diagonal-stride penalty injection as `ridge.rs`, but over `K`, not `XᵀX`.
//! There is no intercept to leave unpenalized (D-06).
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
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

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
}

impl KernelKind {
    /// The sklearn kernel name (for the [`AlgoError::InvalidKernel`] diagnostic).
    fn name(self) -> &'static str {
        match self {
            KernelKind::Linear => "linear",
            KernelKind::Rbf => "rbf",
            KernelKind::Poly => "poly",
            KernelKind::Sigmoid => "sigmoid",
        }
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
    /// L2 penalty strength (`alpha ≥ 0`); added to the `K` diagonal only (D-06).
    alpha: F,
    /// Kernel coefficient `γ`; `None` resolves to `1/n_features` at `fit` (D-05).
    gamma: Option<F>,
    /// Polynomial degree (real, `≥ 1`); used by the poly kernel only.
    degree: F,
    /// Independent term `coef0`; used by poly / sigmoid.
    coef0: F,
    /// The resolved precision-typed kernel (gamma resolved, D-05), `None` until
    /// `fit`. Reused VERBATIM by `predict` (Pitfall 5).
    kernel_: Option<Kernel<F>>,
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
            alpha: f64_to_host::<F>(1.0),
            gamma: None,
            degree: f64_to_host::<F>(3.0),
            coef0: f64_to_host::<F>(1.0),
            kernel_: None,
            dual_coef_: None,
            x_fit_: None,
            fit_shape_: None,
            n_targets_: None,
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
            alpha: host_to_f64(self.alpha),
            gamma: self.gamma.map(host_to_f64),
            degree: host_to_f64(self.degree),
            coef0: host_to_f64(self.coef0),
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `dual_coef_`/`x_fit_`/… are excluded — `None` in any `Unfit` value). Used
    /// by the defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.kernel_kind == other.kernel_kind
            && host_to_f64(self.alpha) == host_to_f64(other.alpha)
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
#[derive(Debug, Clone, Copy)]
pub struct KernelRidgeBuilder {
    kernel: KernelKind,
    alpha: f64,
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

    /// Set the L2 penalty strength `alpha` (`≥ 0`). The `f64` narrows to `F` at
    /// `build::<F>()` (A5).
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
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
        if self.alpha < 0.0 {
            return Err(BuildError::InvalidAlpha {
                estimator: "kernel_ridge",
                alpha: self.alpha,
            });
        }
        Ok(KernelRidge {
            kernel_kind: self.kernel,
            alpha: f64_to_host::<F>(self.alpha),
            gamma: self.gamma.map(f64_to_host::<F>),
            degree: f64_to_host::<F>(self.degree),
            coef0: f64_to_host::<F>(self.coef0),
            kernel_: None,
            dual_coef_: None,
            x_fit_: None,
            fit_shape_: None,
            n_targets_: None,
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
    pub fn dual_coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.dual_coef_
            .as_ref()
            .expect("dual_coef_ is Some by construction on KernelRidge<F, Fitted>")
            .to_host(pool)
    }
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
        if self.kernel_kind == KernelKind::Poly && degree64 < 1.0 {
            return Err(AlgoError::InvalidDegree {
                estimator: "kernel_ridge",
                degree: degree64,
            });
        }
        // Kernel-name guard (T-08-03-01): KernelKind is a closed set, but a
        // future string→KernelKind parse must surface InvalidKernel rather than
        // fall through. The match below is total, so any KernelKind is supported;
        // the guard documents the validate-before-launch contract.
        if !matches!(
            self.kernel_kind,
            KernelKind::Linear | KernelKind::Rbf | KernelKind::Poly | KernelKind::Sigmoid
        ) {
            return Err(AlgoError::InvalidKernel {
                estimator: "kernel_ridge",
                kernel: self.kernel_kind.name().to_string(),
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

        // --- gamma resolution (D-05): None → 1/n_features computed from the
        //     fitted feature count; explicit gamma as-is. The RESOLVED value is
        //     baked into the typed Kernel<F> so predict reuses it (Pitfall 5). ---
        let gamma = match self.gamma {
            Some(g) => g,
            None => f64_to_host::<F>(1.0 / n_features as f64),
        };
        // Validate-before-launch (T-08-03-01 / ASVS V5): the resolved gamma is
        // baked into the typed Kernel and consumed on device by rbf/poly/sigmoid
        // (`exp`/`powf`/`tanh`); a non-finite user-supplied gamma (or a degenerate
        // resolved default) drives those device ops to NaN. Reject it here so the
        // untrusted hyperparameter becomes a typed error, never NaN duals.
        let gamma64 = host_to_f64(gamma);
        if !gamma64.is_finite() {
            return Err(AlgoError::InvalidGamma {
                estimator: "kernel_ridge",
                gamma: gamma64,
            });
        }
        let kernel = match self.kernel_kind {
            KernelKind::Linear => Kernel::Linear,
            KernelKind::Rbf => Kernel::Rbf { gamma },
            KernelKind::Poly => Kernel::Poly {
                gamma,
                degree: self.degree,
                coef0: self.coef0,
            },
            KernelKind::Sigmoid => Kernel::Sigmoid {
                gamma,
                coef0: self.coef0,
            },
        };

        // --- K = kernel_matrix(X, X, kernel): the n×n training Gram (Y = X,
        //     D-02). NO centering — the normal matrix is K, not XᵀX (D-06). ---
        let k = kernel_matrix::<F>(
            pool,
            x,
            (n_samples, n_features),
            x,
            (n_samples, n_features),
            kernel,
            None,
        )?;

        // --- alpha on the K DIAGONAL only (D-06). Add `alpha` to element
        //     [i·n+i]; there is no intercept to leave unpenalized. cubecl 0.10
        //     has no in-place device scalar write, so we materialize the small
        //     n×n K, add α on the diagonal, RELEASE the K buffer back to the pool
        //     (no parallel n² buffer lives), and re-stage the regularized matrix —
        //     from_host recycles the just-released n² bytes (copies the
        //     ridge.rs:248-254 diagonal-α host pass verbatim). `alpha >= 0` is
        //     enforced at build() (BuildError::InvalidAlpha, relocated D-04); the
        //     value's compute use (diagonal injection) stays here, byte-identical. ---
        let alpha64 = host_to_f64(self.alpha);
        let mut k_host = k.to_host(pool);
        for i in 0..n_samples {
            let d = host_to_f64(k_host[i * n_samples + i]) + alpha64;
            k_host[i * n_samples + i] = f64_to_host::<F>(d);
        }
        k.release_into(pool);
        let k_reg: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &k_host);

        // --- Solve (K + αI)·dual_coef_ = y in ONE multi-RHS Cholesky (rhs = t,
        //     D-04). Thread the regularized K buffer through `out` so the factor
        //     reuses it in place — no parallel n² allocation (mirrors
        //     ridge.rs:276-280). A non-SPD pivot surfaces NotPositiveDefinite →
        //     AlgoError (T-08-03-02), never NaN duals. ---
        let k_out = DeviceArray::<ActiveRuntime, F>::from_raw(
            k_reg.handle().clone(),
            n_samples * n_samples,
        );
        let dual_coef =
            cholesky_solve::<F>(pool, &k_reg, y, n_samples, n_targets, Some(k_out))?;

        // --- Post-solve finiteness guard (CR-01 / T-08-03-02). A non-SPD pivot
        //     normally surfaces NotPositiveDefinite from the primitive, but a NaN
        //     reaching the Cholesky diagonal does NOT reliably trip the `pivot <= 0`
        //     test (`NaN <= 0` is `false`), so a poly kernel with a negative base
        //     (`gamma·g + coef0 < 0`, non-integer degree) can produce NaN duals
        //     silently. Read the small n×t duals back and reject any non-finite
        //     value as NotPositiveDefinite so the module-doc "never a NaN
        //     dual_coef_" guarantee actually holds. ---
        let duals_host = dual_coef.to_host(pool);
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

        // --- Store device-resident fitted state. The K buffer was consumed (its
        //     cloned handle threaded through `out` and released by the Cholesky
        //     solve), so we do NOT release `k_reg` again (avoiding a
        //     double-release of the shared allocation). Store a fresh copy of
        //     X_fit_ for predict (the caller's `x` is borrowed). ---
        drop(k_reg);

        let x_host = x.to_host(pool);
        let x_fit: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_host);

        // --- Reconstruct into the `Fitted`-tagged sibling. The consuming-`self`
        //     transition means there is no prior device-resident fitted state to
        //     release: a freshly-built `Unfit` carries `dual_coef_`/`x_fit_` =
        //     `None` (the old re-fit buffer-release pass is therefore vacuous and
        //     dropped — the IncrementalPCA / KernelDensity reset precedent,
        //     16-04/16-07). ---
        Ok(KernelRidge {
            kernel_kind: self.kernel_kind,
            alpha: self.alpha,
            gamma: self.gamma,
            degree: self.degree,
            coef0: self.coef0,
            kernel_: Some(kernel),
            dual_coef_: Some(dual_coef),
            x_fit_: Some(x_fit),
            fit_shape_: Some((n_samples, n_features)),
            n_targets_: Some(n_targets),
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

        let kernel = self
            .kernel_
            .expect("kernel_ is Some by construction on KernelRidge<F, Fitted>");
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

        // --- K_test = kernel_matrix(X_test, X_fit_, kernel) (m×n): the cross
        //     kernel against the stored training matrix, reusing the resolved
        //     fit-time kernel (identical gamma, Pitfall 5). ---
        let k_test = kernel_matrix::<F>(
            pool,
            x_test,
            (n_test, n_features),
            x_fit,
            (n_samples, fit_features),
            kernel,
            None,
        )?;

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
        )?;
        k_test.release_into(pool);
        Ok(pred)
    }
}
