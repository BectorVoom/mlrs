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

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::cholesky::cholesky_solve;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::kernel_matrix::{kernel_matrix, Kernel};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;

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
/// Construct with [`KernelRidge::new`] (`kernel`, `alpha`, `gamma`, `degree`,
/// `coef0`), then [`fit`](Self::fit) and [`predict`](Self::predict). Fitted
/// `dual_coef_` (`n×t`) and `X_fit_` (`n×d`) are device-resident; the host
/// accessor [`dual_coef`](Self::dual_coef) materializes the duals on demand.
pub struct KernelRidge<F>
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
}

impl<F> KernelRidge<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `KernelRidge`. `kernel` selects the family (D-01),
    /// `alpha` is the L2 penalty (`≥ 0`, validated at `fit`), `gamma` is the
    /// kernel coefficient (`None` → `1/n_features` at `fit`, D-05), `degree` is
    /// the polynomial degree (`≥ 1`, validated at `fit`), and `coef0` the
    /// independent term (poly / sigmoid). Invalid hyperparameters are rejected at
    /// `fit` (`InvalidAlpha` / `InvalidDegree`), not construction.
    pub fn new(kernel: KernelKind, alpha: F, gamma: Option<F>, degree: F, coef0: F) -> Self {
        Self {
            kernel_kind: kernel,
            alpha,
            gamma,
            degree,
            coef0,
            kernel_: None,
            dual_coef_: None,
            x_fit_: None,
            fit_shape_: None,
            n_targets_: None,
        }
    }

    /// Host copy of the fitted `dual_coef_` (row-major `n_samples × n_targets`).
    /// Errors with [`AlgoError::NotFitted`] before `fit`.
    pub fn dual_coef(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.dual_coef_
            .as_ref()
            .map(|c| c.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "kernel_ridge",
                operation: "dual_coef_",
            })
    }

    /// Fit `(K + αI)·dual_coef_ = y` over the kernel matrix (D-06).
    ///
    /// `x` is `(n_samples × n_features)` row-major; `y` is `(n_samples ×
    /// n_targets)` row-major (a single target is `t = 1`). Validates the
    /// hyperparameters and geometry BEFORE any launch (T-08-03-01), resolves
    /// `gamma` (D-05), builds `K = kernel_matrix(X, X, kernel)`, adds `α` to the
    /// `K` diagonal only (D-06), and solves the multi-RHS dual in one
    /// [`cholesky_solve`] (`rhs = t`, D-04). NO centering, NO intercept.
    pub fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
        n_targets: usize,
    ) -> Result<&mut Self, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-08-03-01 / ASVS V5: validate the untrusted hyperparameters and
        //     geometry BEFORE any prim launch. alpha < 0 makes (K + αI)
        //     indefinite (Cholesky undefined); degree < 1 is not a valid poly
        //     order; a non-finite resolved gamma (checked once gamma is resolved,
        //     below) drives the device kernels to NaN; the kernel name is fixed by
        //     KernelKind (always valid here, but the guard mirrors the threat
        //     register T-08-03-01). ---
        let alpha64 = host_to_f64(self.alpha);
        if alpha64 < 0.0 {
            return Err(AlgoError::InvalidAlpha {
                estimator: "kernel_ridge",
                alpha: alpha64,
            });
        }
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
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
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
        //     ridge.rs:248-254 diagonal-α host pass verbatim). ---
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

        self.kernel_ = Some(kernel);
        self.dual_coef_ = Some(dual_coef);
        self.x_fit_ = Some(x_fit);
        self.fit_shape_ = Some((n_samples, n_features));
        self.n_targets_ = Some(n_targets);
        Ok(self)
    }

    /// Predict `y = K(X_test, X_fit_) · dual_coef_` (D-06).
    ///
    /// `x_test` is `(n_test × n_features)` row-major. Builds `K_test =
    /// kernel_matrix(X_test, X_fit_, kernel)` (`m×n`) with the RESOLVED fit-time
    /// kernel (gamma reused, Pitfall 5), then `y_pred = K_test · dual_coef_`
    /// (`m×t`) via [`gemm`]. NO intercept broadcast (D-06). Returns the row-major
    /// `(n_test × n_targets)` predictions; for a single target this is length
    /// `n_test`. Errors with [`AlgoError::NotFitted`] before `fit`, or
    /// [`PrimError`] on a geometry / feature-count mismatch.
    pub fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x_test: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_test, n_features) = shape;

        let kernel = self.kernel_.ok_or(AlgoError::NotFitted {
            estimator: "kernel_ridge",
            operation: "predict",
        })?;
        let dual_coef = self.dual_coef_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "kernel_ridge",
            operation: "predict",
        })?;
        let x_fit = self.x_fit_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "kernel_ridge",
            operation: "predict",
        })?;
        let (n_samples, fit_features) = self.fit_shape_.ok_or(AlgoError::NotFitted {
            estimator: "kernel_ridge",
            operation: "predict",
        })?;
        let n_targets = self.n_targets_.ok_or(AlgoError::NotFitted {
            estimator: "kernel_ridge",
            operation: "predict",
        })?;

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

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors the
/// `ridge.rs` helper).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kernel_ridge is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kernel_ridge is f32/f64 only"),
    }
}
