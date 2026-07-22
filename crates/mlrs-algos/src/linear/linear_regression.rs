//! `LinearRegression` (LINEAR-01) — ordinary least squares via the SVD
//! pseudo-inverse, matching scikit-learn's default `scipy.linalg.lstsq` (gelsd)
//! path (D-02).
//!
//! ## Solver (deliberately NOT Cholesky — that is Ridge, D-02)
//! `coef = V · diag(σ⁺) · Uᵀ · y_centered` where the thin SVD of the (centered)
//! design matrix is `X = U · diag(σ) · Vᵀ` (`U` m×k, `σ` length-k, `Vᵀ` k×n,
//! `k = min(m, n)`), composed from the validated Phase-3 [`svd`] +
//! Phase-2 [`gemm`] / [`column_reduce`] primitives — NO bespoke matmul/solve.
//!
//! The pseudo-inverse uses sklearn's small-singular-value cutoff (RESEARCH
//! Pitfall 1 / Open Q3): `σ⁺_i = 1/σ_i if σ_i > cutoff else 0` with
//! `cutoff = rcond · σ_max`, `rcond = RCOND` (= `1e-6`). This MUST match
//! `sklearn.linear_model.LinearRegression`, which since the `tol` parameter
//! (default `1e-6`) passes that value as scipy's `lstsq(cond=…)` — scipy drops
//! every `σ_i ≤ cond·σ_max`. The looser numpy-lstsq / scipy-gelsd default
//! (`ε_F·max(m,n)`) does NOT match sklearn: on the near-collinear fixture its
//! `σ_min/σ_max ≈ 3e-8` is above that f64 threshold, so numpy reciprocates the
//! ~0 singular value and the coefficients EXPLODE to ~1e4, whereas sklearn (and
//! this estimator) drop it and return the bounded ~0.485 minimum-norm solution
//! (T-04-03-01). A `NEAR_ZERO_FLOOR` fallback keeps the cutoff strictly positive
//! even for an all-zero spectrum.
//!
//! ## Intercept via center-then-solve (D-05)
//! When `fit_intercept`, the column means `x̄` and `ȳ` are removed before the
//! solve and the intercept is recovered as `intercept_ = ȳ − x̄·coef_`. The
//! penalty-free intercept is never part of the SVD system (mirrors sklearn).
//!
//! ## Device residency (D-03)
//! Fitted `coef_` (length n) and `intercept_` (length 1) are stored as
//! device-resident [`DeviceArray`]s; `predict` runs the `X_test · coef_`
//! GEMM on-device and broadcasts the intercept. The host materializes the
//! fitted state only at a Rust accessor / oracle-comparison boundary.
//!
//! Tests live in `crates/mlrs-algos/tests/linear_regression_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::center::center_columns;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::gram::gram_xty;
use mlrs_backend::prims::linear_predict::linear_predict;
use mlrs_backend::prims::svd::svd;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// Dense-eigensolver feature cap for the large-`n_samples` Gram+eig path
/// (mirrors `mlrs_kernels::jacobi_eig::MAX_DIM` — the spectral-family
/// precedent of re-affirming the cap locally rather than importing the
/// kernel crate's constant, `spectral_clustering.rs`/`spectral_embedding.rs`).
const GRAM_EIG_MAX_FEATURES: usize = 64;

/// Row/col cap of the direct-SVD path (mirrors
/// `mlrs_kernels::jacobi_svd::MAX_ROWS` — same local-re-affirm precedent).
/// `LinearRegression::fit` uses the EXACT direct-SVD pseudo-inverse below this
/// size (byte-identical to the pre-dual-path implementation — zero regression
/// risk for the committed oracle fixtures) and the Gram+eig path above it
/// (D-02 dual-path: SVD is the numerically-safer small-problem default: eig
/// squares the condition number of `X`, which is why Ridge/`D-02` forbids
/// unifying the two solvers — but a `MAX_ROWS`-row Jacobi SVD run as a
/// SINGLE cube of `n_features` threads cannot use the GPU's parallelism, and
/// literally cannot RUN past `MAX_ROWS` samples at all, so it is the wrong
/// tool once `n_samples` leaves "small"; Gram+eig's GEMM has no row cap and
/// is embarrassingly parallel in `n_samples`, matching cuML's default
/// `algorithm='eig'`).
const DIRECT_SVD_MAX_ROWS: usize = 256;

/// Near-zero floor for the σ⁺ cutoff (mirrors the `svd.rs` `NEAR_ZERO_FLOOR`
/// precedent — below the 1e-5 tolerance so it never loosens a real check). Keeps
/// the cutoff strictly positive for a degenerate (all-zero) spectrum so a tiny
/// singular value is always zeroed rather than reciprocated.
const NEAR_ZERO_FLOOR: f64 = 1e-8;

/// Relative singular-value cutoff `rcond` for the pseudo-inverse — singular
/// values with `σ_i ≤ rcond·σ_max` are dropped (σ⁺ = 0). Pinned to `1e-6` to
/// match `sklearn.linear_model.LinearRegression`'s default `tol`, which it
/// forwards as `scipy.linalg.lstsq(cond=…)` (D-02 / Open Q3). This is the value
/// that reproduces sklearn on BOTH the full-rank and the near-collinear fixture;
/// the much smaller `ε_F·max(m,n)` numpy default would keep the collinear ~0
/// singular value and explode the coefficients (see module docs).
const RCOND: f64 = 1e-6;

/// Ordinary least squares (LINEAR-01) fitted by the SVD pseudo-inverse.
///
/// Construct with the zero-arg [`LinearRegression::new`] (sklearn default:
/// `fit_intercept = true`) or [`LinearRegression::builder`], then the consuming
/// [`Fit::fit`] (returns the `Fitted`-tagged sibling) and [`Predict::predict`].
/// Fitted `coef_`/`intercept_` are device-resident (D-03); the host accessors
/// [`coef`](LinearRegression::coef) / [`intercept`](LinearRegression::intercept)
/// materialize them on demand and exist ONLY on
/// `LinearRegression<F, Fitted>` (the compile-time typestate replaces the old
/// runtime `NotFitted` guard, D-03).
pub struct LinearRegression<F, S = Unfit> {
    /// Whether to center `X`/`y` and recover a bias term (D-05).
    fit_intercept: bool,
    /// Fitted coefficients (length `n_features`), device-resident, `None` until
    /// `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (length 1), device-resident, `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> LinearRegression<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct a `LinearRegression` with sklearn's default `fit_intercept =
    /// true` directly in the `Unfit` state. This is the SINGLE source of truth for
    /// the default hyperparameter (D-08): the builder `Default` re-derives from
    /// here via [`LinearRegression::into_builder`], rather than re-listing the
    /// literal.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            coef_: None,
            intercept_: None,
            _state: PhantomData,
        }
    }

    /// Start building a `LinearRegression` from sklearn's defaults (D-08 single
    /// source).
    pub fn builder() -> LinearRegressionBuilder {
        LinearRegressionBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`LinearRegressionBuilder::default`] to re-derive
    /// the defaults from [`LinearRegression::new`] (D-08).
    pub fn into_builder(self) -> LinearRegressionBuilder {
        LinearRegressionBuilder {
            fit_intercept: self.fit_intercept,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `coef_`/`intercept_` fields are excluded — both are `None` in any `Unfit`
    /// value). Used by the defaults-equality test (BLDR-01):
    /// `LinearRegression::new().hyperparams_eq(&LinearRegression::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.fit_intercept == other.fit_intercept
    }
}

impl<F> Default for LinearRegression<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`LinearRegression`] (D-01). The lone setter is `.fit_intercept`;
/// `build::<F>()` produces the target-float estimator. `Default` re-derives the
/// sklearn default from [`LinearRegression::new`] (D-08 single source) rather than
/// holding a literal (Pitfall 1). OLS has no data-independent hyperparameter to
/// validate, so `build` is infallible-but-typed (`-> Result<_, BuildError>`) for
/// uniformity with the other linear builders.
#[derive(Debug, Clone, Copy)]
pub struct LinearRegressionBuilder {
    fit_intercept: bool,
}

impl Default for LinearRegressionBuilder {
    /// Re-derive the sklearn default from [`LinearRegression::new`] (D-08 single
    /// source). `f64` is pinned only to read the F-independent default — the
    /// builder is non-generic, so the choice of `F` here is irrelevant.
    fn default() -> Self {
        LinearRegression::<f64, Unfit>::new().into_builder()
    }
}

impl LinearRegressionBuilder {
    /// Set whether to center `X`/`y` and recover a bias term.
    pub fn fit_intercept(mut self, v: bool) -> Self {
        self.fit_intercept = v;
        self
    }

    /// Build the (unfit) estimator. OLS has no data-INDEPENDENT hyperparameter to
    /// validate (the data-DEPENDENT geometry check lives in [`Fit::fit`]), so this
    /// never errors — the `Result` is kept for uniformity with the penalized
    /// linear builders (and so the PyO3 boundary's `build_err_to_py` mapper is
    /// shape-identical across the family).
    pub fn build<F>(self) -> Result<LinearRegression<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        Ok(LinearRegression {
            fit_intercept: self.fit_intercept,
            coef_: None,
            intercept_: None,
            _state: PhantomData,
        })
    }
}

impl<F> LinearRegression<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `coef_` (length `n_features`). `Some` by
    /// construction on the `Fitted` state, so no `NotFitted` branch is needed
    /// (the compile-time typestate replaces the runtime guard, D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on LinearRegression<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_` (scalar). `Some` by construction on
    /// the `Fitted` state (D-03).
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> F {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on LinearRegression<F, Fitted>")
            .to_host(pool)[0]
    }
}

impl<F> Fit<F> for LinearRegression<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = LinearRegression<F, Fitted>;

    /// Dual-path dispatch (D-02): the direct-SVD pseudo-inverse below
    /// [`DIRECT_SVD_MAX_ROWS`] (byte-identical to the pre-dual-path
    /// implementation — the committed oracle fixtures, both full-rank and
    /// near-collinear, exercise ONLY this path and see NO behavioral change),
    /// else the Gram+eig pseudo-inverse (`fit_gram_eig`, unbounded in
    /// `n_samples`, capped at [`GRAM_EIG_MAX_FEATURES`]).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<LinearRegression<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-04-03-02 / ASVS V5: validate geometry BEFORE any prim launch. ---
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "linear_regression",
            operation: "fit (requires y)",
        })?;
        if y.len() != n_samples {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 1,
                len: y.len(),
            }));
        }

        let (coef, intercept_dev) = if n_samples.max(n_features) <= DIRECT_SVD_MAX_ROWS {
            fit_direct_svd::<F>(pool, x, y, n_samples, n_features, self.fit_intercept)?
        } else {
            if n_features > GRAM_EIG_MAX_FEATURES {
                return Err(AlgoError::NFeaturesExceedsMaxDim {
                    estimator: "linear_regression",
                    n_features,
                    max: GRAM_EIG_MAX_FEATURES,
                });
            }
            fit_gram_eig::<F>(pool, x, y, n_samples, n_features, self.fit_intercept)?
        };

        Ok(LinearRegression {
            fit_intercept: self.fit_intercept,
            coef_: Some(coef),
            intercept_: Some(intercept_dev),
            _state: PhantomData,
        })
    }
}

/// Direct-SVD pseudo-inverse solve (`n_samples.max(n_features) <=
/// DIRECT_SVD_MAX_ROWS`) — byte-identical to the original LINEAR-01
/// implementation, extracted verbatim into its own function only so `fit`
/// can dispatch to it (D-02 dual-path). `coef = V · diag(σ⁺) · Uᵀ ·
/// y_centered`, `X_c = U·diag(σ)·Vᵀ` via the Phase-3 [`svd`] primitive.
#[allow(clippy::type_complexity)]
fn fit_direct_svd<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_features: usize,
    fit_intercept: bool,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    // --- 1. Centering (D-05). When fit_intercept, remove the column means x̄
    //        and ȳ; solve on the centered system. Mirrors covariance.rs'
    //        two-pass centring. Done host-side here because the σ⁺ cutoff and
    //        intercept recovery already need a host pass over the tiny k/n
    //        vectors; the heavy products stay on-device via gemm/svd. ---
    let x_host = x.to_host(pool);
    let y_host = y.to_host(pool);

    let mut x_mean = vec![0.0f64; n_features];
    let mut y_mean = 0.0f64;
    if fit_intercept {
        for r in 0..n_samples {
            for c in 0..n_features {
                x_mean[c] += host_to_f64(x_host[r * n_features + c]);
            }
            y_mean += host_to_f64(y_host[r]);
        }
        let inv = 1.0 / n_samples as f64;
        for m in x_mean.iter_mut() {
            *m *= inv;
        }
        y_mean *= inv;
    }

    let mut x_centered: Vec<F> = vec![F::from_int(0i64); n_samples * n_features];
    for r in 0..n_samples {
        for c in 0..n_features {
            let v = host_to_f64(x_host[r * n_features + c]) - x_mean[c];
            x_centered[r * n_features + c] = f64_to_host::<F>(v);
        }
    }
    let mut y_centered: Vec<F> = vec![F::from_int(0i64); n_samples];
    for r in 0..n_samples {
        y_centered[r] = f64_to_host::<F>(host_to_f64(y_host[r]) - y_mean);
    }

    let x_c_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_centered);
    let y_c_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y_centered);

    // --- 2. Thin SVD of the centered design (D-02): X_c = U·diag(σ)·Vᵀ,
    //        U (m×k), σ (k), Vᵀ (k×n), k = min(m, n). ---
    let k = n_samples.min(n_features);
    let (u, s, vt) = svd::<F>(pool, &x_c_dev, (n_samples, n_features))?;

    // --- 3. σ⁺ with sklearn's small-σ cutoff (Pitfall 1 / T-04-03-01 /
    //        Open Q3). cutoff = RCOND · σ_max (RCOND = 1e-6 = sklearn's
    //        default `tol`, forwarded as scipy `lstsq(cond=…)`), floored at
    //        NEAR_ZERO_FLOOR so it is strictly positive even for a degenerate
    //        spectrum. The looser ε_F·max(m,n) numpy default would keep the
    //        collinear ~0 singular value and explode the coefficients. ---
    let s_host = s.to_host(pool);
    let s64: Vec<f64> = s_host.iter().map(|&v| host_to_f64(v)).collect();
    let sigma_max = s64.iter().cloned().fold(0.0f64, f64::max);
    let cutoff = (RCOND * sigma_max).max(NEAR_ZERO_FLOOR);

    // --- 4. coef = V · diag(σ⁺) · (Uᵀ · y_c). Compose with gemm; the only
    //        host arithmetic is the length-k σ⁺ scaling (the cutoff guard). ---
    // t1 = Uᵀ · y_c  (k×1). U is (m×k) row-major; transa reads it as Uᵀ
    // (k×m) — no transpose buffer (D-06).
    let t1 = gemm::<F>(
        pool,
        &u,
        (k, n_samples), // logical Uᵀ is (k × m)
        &y_c_dev,
        (n_samples, 1),
        true, // u buffer is U (m×k) row-major; transa reads it as Uᵀ.
        false,
        None,
    )?;
    let t1_host = t1.to_host(pool);

    // t2 = diag(σ⁺) · t1  (k×1) — the small-σ cutoff lives here.
    let mut t2_host: Vec<F> = vec![F::from_int(0i64); k];
    for i in 0..k {
        let sigma = s64[i];
        let scaled = if sigma > cutoff {
            host_to_f64(t1_host[i]) / sigma
        } else {
            0.0 // drop the near-zero singular direction (no 1/0 blow-up).
        };
        t2_host[i] = f64_to_host::<F>(scaled);
    }
    let t2_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &t2_host);

    // coef = V · t2  (n×1). Vᵀ is (k×n) row-major; transa reads it as V
    // (n×k) — no transpose buffer (D-06).
    let coef = gemm::<F>(
        pool,
        &vt,
        (n_features, k), // logical V is (n × k)
        &t2_dev,
        (k, 1),
        true, // vt buffer is Vᵀ (k×n) row-major; transa reads it as V.
        false,
        None,
    )?;

    // --- 5. intercept_ = ȳ − x̄·coef_ when fit_intercept, else 0 (D-05). ---
    let coef_host = coef.to_host(pool);
    let intercept = if fit_intercept {
        let mut dot = 0.0f64;
        for c in 0..n_features {
            dot += x_mean[c] * host_to_f64(coef_host[c]);
        }
        y_mean - dot
    } else {
        0.0
    };
    let intercept_dev: DeviceArray<ActiveRuntime, F> =
        DeviceArray::from_host(pool, &[f64_to_host::<F>(intercept)]);

    // --- 6. Release scratch; store device-resident fitted state (D-03). ---
    u.release_into(pool);
    s.release_into(pool);
    vt.release_into(pool);
    t1.release_into(pool);
    t2_dev.release_into(pool);
    x_c_dev.release_into(pool);
    y_c_dev.release_into(pool);

    Ok((coef, intercept_dev))
}

/// Gram+eig pseudo-inverse solve (`n_samples.max(n_features) >
/// DIRECT_SVD_MAX_ROWS`, `n_features <= GRAM_EIG_MAX_FEATURES`) — the
/// large-`n_samples` LINEAR-01 path. Mirrors cuML's default
/// `algorithm='eig'`: forms the RAW (unscaled) Gram `G = XᵀX` (`d×d`) and
/// `c = Xᵀy` (`d×1`) via the Phase-2 [`gemm`] (embarrassingly parallel in
/// `n_samples`, no row cap — unlike the direct-SVD path's single-cube Jacobi
/// kernel), eigendecomposes the symmetric PSD `G` via the Phase-3 [`eig`]
/// primitive (`G = V·diag(w)·Vᵀ`, `w` DESCENDING), and recovers
/// `coef = V · diag(w⁺) · Vᵀ · c`.
///
/// ## σ⁺ cutoff in eigenvalue space (numerically mirrors `fit_direct_svd`)
/// `w_i` are EIGENVALUES of `G = XᵀX`, i.e. `σ_i²` of `X` (PSD, so `w_i >= 0`
/// up to rounding noise — clamped at 0 before the `sqrt` below). The cutoff
/// is evaluated on `σ_i = sqrt(w_i)` against `RCOND · σ_max` — the SAME
/// physical criterion as `fit_direct_svd` (drop singular directions below
/// `rcond · σ_max`) — and the pseudo-inverse scaling divides by `w_i` (not
/// `σ_i`), since `G⁺ = V·diag(1/w_i)·Vᵀ` for a symmetric PSD matrix. Forming
/// the Gram squares `X`'s condition number (the reason D-02 forbids this for
/// the small/exact path — see the module docs and `ridge.rs`'s "MUST NOT
/// unify" note) — that tradeoff is accepted ONLY here, above
/// `DIRECT_SVD_MAX_ROWS`, where the direct-SVD path cannot run at all.
///
/// ## Known limit: f32 near-singular `X` (measured, accepted — see the test)
/// Squaring the condition number pushes a near-null direction's eigenVALUE
/// down to `σ_ratio²`; the cutoff still correctly detects and drops it (no
/// explosion — `coef` stays finite), but eigenVECTOR accuracy degrades as
/// `~eps / gap` near-degenerate eigenvalues, so at f32 (`eps ≈ 1.2e-7`) a
/// `σ_ratio` in `(RCOND, ~3.5e-4)` cannot be reliably resolved by ANY
/// Gram-based algorithm — `σ_ratio² < eps` is indistinguishable from rounding
/// noise, and no choice of test fixture avoids this (the window
/// `σ_ratio < RCOND` [cutoff must fire] AND `σ_ratio > sqrt(eps_f32)`
/// [eigenvector resolvable] is empty for f32). The coefficient MASS can then
/// legitimately split differently between near-collinear features than
/// `fit_direct_svd`'s exact SVD answer while still being an equally good
/// least-squares fit (RSS-equivalent, not coefficient-identical) — this is
/// the same `algorithm='eig'` vs `algorithm='svd'` tradeoff cuML documents.
/// `linear_regression_large_collinear_cutoff_f32`
/// (`linear_regression_test.rs`) checks fit QUALITY (RSS vs sklearn) for
/// exactly this reason, not coefficient bit-parity; f64 (`eps ≈ 2.2e-16`,
/// window `(RCOND, ~1.5e-8)` non-empty) keeps the strict bit-parity check.
#[allow(clippy::type_complexity)]
fn fit_gram_eig<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_features: usize,
    fit_intercept: bool,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    let d = n_features;

    // LR_PROFILE=1: per-phase wall-clock attribution (the KM_PROFILE/RF_PROFILE
    // precedent — attribution only, since kernel launches are async and the
    // lap only completes at the next readback that drains the queue; a tiny
    // forced readback after `gram_xty`/`eig` pins each phase's lap to ITS OWN
    // kernels rather than bleeding into the next phase's).
    let profile = std::env::var("LR_PROFILE").is_ok();
    let lap0 = std::time::Instant::now();

    // --- 1. Centering, DEVICE-resident (D-05 / perf): the direct-SVD path's
    //        O(n·d) host round-trip is fine at n <= 256 but would dominate at
    //        the scale this path targets, so centering here composes
    //        `column_reduce` + `center_columns` on-device (`prims::center`).
    //        When !fit_intercept there is nothing to remove — `x`/`y` are
    //        read directly by the Gram/Xty GEMMs below with NO copy. ---
    let (x_mean, y_mean, x_owned, y_owned) = if fit_intercept {
        let (x_c, x_mean_dev) = center_columns::<F>(pool, x, (n_samples, d))?;
        let (y_c, y_mean_dev) = center_columns::<F>(pool, y, (n_samples, 1))?;
        let x_mean: Vec<f64> = x_mean_dev
            .to_host(pool)
            .iter()
            .map(|&v| host_to_f64(v))
            .collect();
        let y_mean = host_to_f64(y_mean_dev.to_host(pool)[0]);
        x_mean_dev.release_into(pool);
        y_mean_dev.release_into(pool);
        (x_mean, y_mean, Some(x_c), Some(y_c))
    } else {
        (vec![0.0f64; d], 0.0f64, None, None)
    };
    let x_ref = x_owned.as_ref().unwrap_or(x);
    let y_ref = y_owned.as_ref().unwrap_or(y);
    let t_center = if profile { lap0.elapsed().as_secs_f64() } else { 0.0 };

    // --- 2. Raw Gram G = XᵀX (d×d) and c = Xᵀy (d×1) via the row-blocked
    //        gram_xty prim (LINEAR-01 perf lever, D-02): a shared-memory
    //        accumulation over row BLOCKS, replacing the skinny-output/huge-K
    //        `gemm` pair that starved the GPU of parallel work regardless of
    //        `n_samples` (the KMeans "GEMM sums" pathology — see
    //        `mlrs_kernels::gram` module docs). UNSCALED (unlike
    //        covariance.rs, D-09): the eig pseudo-inverse below is invariant
    //        to any shared X/y scale, so there is no 1/(n-ddof) normalisation
    //        to apply or undo. No row cap (D-02). ---
    let lap1 = std::time::Instant::now();
    let (gram, xty) = gram_xty::<F>(pool, x_ref, y_ref, n_samples, d)?;
    if profile {
        // Force a drain so this lap attributes ONLY gram_xty's kernels, not
        // whatever runs next (the KM_PROFILE precedent's readback-boundary
        // caveat) — a tiny d-element readback, not the n-heavy data.
        let _ = xty.to_host(pool);
    }
    let t_gram = if profile { lap1.elapsed().as_secs_f64() } else { 0.0 };
    if let Some(xc) = x_owned {
        xc.release_into(pool);
    }
    if let Some(yc) = y_owned {
        yc.release_into(pool);
    }

    // --- 3. Eigendecomposition of the symmetric PSD Gram (D-02 large-N path,
    //        mirrors cuML's default `algorithm='eig'`): G = V·diag(w)·Vᵀ, w
    //        DESCENDING (`n_features <= GRAM_EIG_MAX_FEATURES` validated by
    //        the caller BEFORE this function is entered — T-04-03-02). ---
    let lap2 = std::time::Instant::now();
    let (w, v) = eig::<F>(pool, &gram, d, None)?;
    if profile {
        // Force a drain so this lap attributes ONLY eig's kernel (a tiny
        // length-d readback, ahead of step 5's real one).
        let _ = w.to_host(pool);
    }
    let t_eig = if profile { lap2.elapsed().as_secs_f64() } else { 0.0 };
    gram.release_into(pool);

    // --- 4. t1 = Vᵀ · c  (d×1) via GEMM (small, device). `v` is (d×d)
    //        COLUMN-major (eigenvectors as columns, per `eig`'s docs); a
    //        ROW-MAJOR read of that SAME buffer is therefore already Vᵀ
    //        (mirrors `svd.rs`'s identical column-major-V/row-major-Vᵀ
    //        convention) — transa=false reads it directly, no transpose
    //        buffer (D-06). ---
    let t1 = gemm::<F>(pool, &v, (d, d), &xty, (d, 1), false, false, None)?;
    xty.release_into(pool);

    // --- 5. σ⁺ cutoff in eigenvalue space (see the function docs) + t2 =
    //        diag(w⁺) · t1. w/t1 are length-d (<= GRAM_EIG_MAX_FEATURES) —
    //        this host pass is over a TINY vector, not the n_samples data. ---
    let w_host: Vec<f64> = w.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let t1_host: Vec<f64> = t1.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    w.release_into(pool);
    t1.release_into(pool);

    let sigma_max = w_host.first().copied().unwrap_or(0.0).max(0.0).sqrt();
    let cutoff = (RCOND * sigma_max).max(NEAR_ZERO_FLOOR);
    let mut t2_host: Vec<F> = vec![F::from_int(0i64); d];
    for i in 0..d {
        let wi = w_host[i].max(0.0);
        let sigma = wi.sqrt();
        let scaled = if sigma > cutoff { t1_host[i] / wi } else { 0.0 };
        t2_host[i] = f64_to_host::<F>(scaled);
    }
    let t2_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &t2_host);

    // --- 6. coef = V · t2  (d×1) via GEMM. `v`'s buffer read row-major is Vᵀ
    //        (see step 4); transa=true reads its transpose, i.e. V, with no
    //        transpose buffer (D-06) — mirrors `fit_direct_svd`'s identical
    //        `vt`-buffer `transa=true` "read it as V" step. ---
    let coef = gemm::<F>(pool, &v, (d, d), &t2_dev, (d, 1), true, false, None)?;
    v.release_into(pool);
    t2_dev.release_into(pool);

    // --- 7. intercept_ = ȳ − x̄·coef_ when fit_intercept, else 0 (D-05). ---
    let coef_host = coef.to_host(pool);
    let intercept = if fit_intercept {
        let mut dot = 0.0f64;
        for c in 0..d {
            dot += x_mean[c] * host_to_f64(coef_host[c]);
        }
        y_mean - dot
    } else {
        0.0
    };
    let intercept_dev: DeviceArray<ActiveRuntime, F> =
        DeviceArray::from_host(pool, &[f64_to_host::<F>(intercept)]);

    if profile {
        eprintln!(
            "LR_PROFILE n={n_samples} d={d}: center={t_center:.4}s gram_xty={t_gram:.4}s eig={t_eig:.4}s"
        );
    }

    Ok((coef, intercept_dev))
}

impl<F> Predict<F> for LinearRegression<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_samples, n_features) = shape;

        // `coef_`/`intercept_` are `Some` by construction on
        // `LinearRegression<F, Fitted>` (the compile-time typestate replaces the
        // old runtime `NotFitted` guard, D-03).
        let coef = self
            .coef_
            .as_ref()
            .expect("coef_ is Some by construction on LinearRegression<F, Fitted>");
        let intercept = self
            .intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on LinearRegression<F, Fitted>");

        // --- T-04-03-02 / ASVS V5: geometry + fitted-n_features consistency. ---
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        if coef.len() != n_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: coef.len(),
                rhs: n_features,
            }));
        }

        // y_pred = X_test · coef + intercept via ONE fused device launch
        // (LINEAR-01 predict perf lever): the `linear_predict` prim's GATHER
        // matvec+bias kernel replaces the prior gemm→`intercept.to_host()`→
        // `raw.to_host()`→host bias-loop→`from_host` round-trips (the
        // `center`/`gram` host-sync pathology, same class of fix). The result
        // stays device-resident; the PyO3 boundary's terminal readback is the
        // only host↔device crossing.
        Ok(linear_predict::<F>(
            pool,
            x,
            coef,
            intercept,
            (n_samples, n_features),
        )?)
    }
}
