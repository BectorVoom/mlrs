//! `gmm_host` — the parallel host EM engine behind `GaussianMixture` (MIX-01).
//!
//! ## Why the mixture EM loop is a HOST algorithm on every backend
//! Every other estimator in this crate that ended up host-resident got there by
//! measurement; this one gets there by three structural facts that hold on all
//! four backends at once:
//!
//! 1. **The loop is tiny and long.** One `fit` is `max_iter` iterations of two
//!    passes each — 200 passes at sklearn's defaults, times `n_init`. A cubecl
//!    launch costs ~50 µs of pure dispatch ([[mlrs-rf-fit-optimization]]), and
//!    on `cubecl-cpu` one launch is one OS thread per unit
//!    ([[mlrs-cubecl-cpu-execution-model]]), so the launch overhead alone
//!    dominates every rung a user actually runs.
//! 2. **The reduction must be `f64`.** The E-step exponentiates a Mahalanobis
//!    distance and the M-step forms a covariance as a weighted second moment;
//!    both amplify summation error into `precisions_cholesky_`, which is then
//!    inverted. cubecl's cuda backend does not advertise `f64` at all
//!    ([[mlrs-cubecl-cuda-f64-not-advertised]]), so a device arm would silently
//!    be an `f32` arm on the fastest `f64` hardware in the fleet.
//! 3. **The per-iteration tail is `O(k·d³)`**, a Cholesky and a triangular
//!    inverse per component. That is a serial, tiny, branchy factorization —
//!    the shape `cubecl` is worst at and a host `-O3` loop is best at.
//!
//! So the engine here is the WHOLE algorithm, and the estimator's device
//! ingress (`Fit::fit`) reads the design back and
//! calls into it. What it is NOT is a naive transcription of sklearn: three
//! structural wins over `sklearn.mixture._gaussian_mixture` are baked in, and
//! they are why a single-threaded run of this file already competes.
//!
//! ## A device arm now exists too — for large `n`, not instead of this file
//! [`crate::prims::gmm_device`] adds a genuine on-device EM engine despite the
//! three facts above, by being surgical about what actually moves: it keeps
//! reasons #2 and #3 pinned exactly where they are (the `O(k·d³)` Cholesky /
//! triangular-inverse tail stays HOST arithmetic, called from the estimator
//! every iteration same as here, and the reduction is gated off any backend
//! whose `f64` — or `f64` TRANSCENDENTALS, the sharper landmine
//! [`crate::capability::f64_transcendental_supported`] documents — are not
//! genuinely available, via [`crate::capability::f64_device_kernels_available`]
//! rather than the under-reporting `supports_type(F64)` probe). What moves is
//! ONLY the two passes whose cost scales with `n`: the E-step's weighted-log-
//! prob/responsibility/`nk`+`means` sweep and the M-step's covariance sweep,
//! kept device-resident (`X` and `resp` never leave the device) for the WHOLE
//! `max_iter` loop of a restart — the same shape `KMeans`'s Lloyd loop uses.
//! That answers reason #1 too: the per-iteration host traffic shrinks from two
//! `O(n·k·d)`-ish passes to a few KB of `O(k·d)`-ish scalars, so the fixed
//! launch overhead is paid a constant number of times per iteration rather than
//! scaling with what crosses the bus. Below a conservative `n·k·d` size floor
//! the launch overhead still dominates (reason #1 unmodified), so
//! [`crate::prims::gmm_device::gmm_device_applicable`] keeps the estimator on
//! this file at small scale — the device arm is an ADDITIONAL fast path for
//! large `n` on cuda/rocm hardware, not a replacement for anything here.
//!
//! ## The three algorithmic wins over sklearn
//!
//! ### 1. The precision Cholesky is TRIANGULAR and sklearn ignores it
//! `_compute_precision_cholesky` produces `precisions_chol[k] = inv(L_k)ᵀ`,
//! which is UPPER triangular by construction. sklearn then evaluates
//! `np.dot(X, prec_chol)` — a DENSE `n×d · d×d` GEMM that multiplies the
//! `d(d−1)/2` structural zeros anyway. `maha_full` walks only the stored
//! triangle, so the dominant `O(n·k·d²)` E-step term is **halved** with zero
//! numerical difference (it is the same sum, minus terms that are exactly `0`).
//! The same applies to the M-step: a covariance is symmetric, so
//! `GmmHost::cov_full` fills only the upper triangle and mirrors it, while
//! sklearn's `np.dot(resp * diff.T, diff)` computes both halves.
//!
//! ### 2. `tied` recomputes `X @ P` per component in sklearn — `k` times
//! sklearn's `tied` branch is literally
//! ```text
//! for k, mu in enumerate(means):
//!     y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
//! ```
//! with `precisions_chol` LOOP-INVARIANT, making its `tied` E-step `O(n·k·d²)`
//! — the same cost as `full`, for a model with one shared covariance. Hoisting
//! it (`maha_tied_row`) makes it `O(n·d²/2 + n·k·d)`, i.e. asymptotically
//! `k/2`× cheaper. And its M-step `avg_X2 = X.T @ X` is loop-invariant too:
//! `GmmHost::ensure_xtx` computes it ONCE per fit instead of once per
//! iteration, so the `tied` M-step drops from `O(n·d²)` per iteration to
//! `O(k·d²)`.
//!
//! ### 3. No `n×d` temporaries, and one pass where sklearn makes four
//! sklearn's E-step materializes an `n×d` `y` per component and then makes a
//! second pass over it for `np.sum(np.square(y), axis=1)`; its M-step makes
//! separate passes for `nk`, `means`, and the covariance. Here the Mahalanobis
//! distance is accumulated in registers, and pass A fuses
//! `log_resp` + `nk` + `means` into ONE sweep of the design. Two sweeps of `X`
//! per iteration total, against sklearn's five-plus, on a loop whose working
//! set is what limits it.
//!
//! ## Parallelism
//! Both passes are row-blocked over a persistent [`WorkerPool`] (spawned once
//! per fit, not once per pass — [[mlrs-svm-fit-worker-pool]]), each unit owning
//! a disjoint row range and a private accumulator that the driver reduces at
//! the barrier. There are no atomics and no shared writes within a phase, which
//! is the same discipline the device kernels use.
//!
//! Everything internal is `f64` regardless of the estimator's `F` (see #2
//! above); the estimator narrows at its accessors.
//!
//! Tests live in `crates/mlrs-algos/tests/gaussian_mixture_test.rs` and
//! `crates/mlrs-algos/tests/gaussian_mixture_perf_test.rs` (AGENTS.md §2 —
//! never an in-source `#[cfg(test)] mod tests`).

use crate::abflag;
use crate::capability;
use crate::prims::host_pool::{Shared, WorkerPool};
use crate::prims::rng::SplitMix64;

/// `ln(2π)`, the Gaussian log-normalizer constant sklearn spells
/// `n_features * np.log(2 * np.pi)`.
///
/// `pub(crate)` so [`crate::prims::gmm_device`]'s host-side bias computation
/// (the E-step's `bias[c] = ln(weight_c) + log_det_c − ½·d·LOG_2PI`) shares the
/// exact same constant rather than a second copy that could drift.
pub(crate) const LOG_2PI: f64 = 1.837_877_066_409_345_5;

/// Rows processed together by the blocked passes.
///
/// Chosen so a block's slice of the design (`ROW_BLOCK · d` doubles — 64 KB at
/// `d = 128`) stays in L2 alongside the ONE component parameter block the
/// component-outer nest is working on. Larger blocks stop helping once both no
/// longer fit; smaller ones stop amortizing the parameter reload.
const ROW_BLOCK: usize = 64;

/// sklearn adds `10 * np.finfo(resp.dtype).eps` to every `nk` so an empty
/// component cannot divide by zero. The reference computes `resp` in the
/// design's dtype; we always compute in `f64`, and matching sklearn's `float64`
/// path is what the oracle compares against.
///
/// `pub(crate)` so [`crate::prims::gmm_device`]'s device-arm `nk` finish uses
/// the identical floor rather than a second copy that could drift.
pub(crate) const NK_EPS: f64 = 10.0 * f64::EPSILON;

/// sklearn's four `covariance_type` values (`StrOptions({'full', 'tied',
/// 'diag', 'spherical'})`), which select BOTH the parameterization of
/// `covariances_` and the whole shape of the E-step / M-step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceType {
    /// Each component has its own full `d × d` covariance. `covariances_` is
    /// `k × d × d`; the E-step is `O(n·k·d²/2)` here (`O(n·k·d²)` in sklearn).
    Full,
    /// All components SHARE one `d × d` covariance. `covariances_` is `d × d`;
    /// the E-step is `O(n·d²/2 + n·k·d)` here — see win #2 in the module docs.
    Tied,
    /// Each component has its own diagonal covariance. `covariances_` is
    /// `k × d`; everything is `O(n·k·d)`.
    Diag,
    /// Each component has one shared variance across features. `covariances_`
    /// is length `k`; everything is `O(n·k·d)`.
    Spherical,
}

impl CovarianceType {
    /// The sklearn string spelling, for diagnostics and the Python accessor.
    pub fn name(self) -> &'static str {
        match self {
            CovarianceType::Full => "full",
            CovarianceType::Tied => "tied",
            CovarianceType::Diag => "diag",
            CovarianceType::Spherical => "spherical",
        }
    }

    /// Number of `f64` elements in `covariances_` / `precisions_cholesky_` for
    /// this parameterization — the flat length every buffer in this module is
    /// sized by.
    pub fn param_len(self, k: usize, d: usize) -> usize {
        match self {
            CovarianceType::Full => k * d * d,
            CovarianceType::Tied => d * d,
            CovarianceType::Diag => k * d,
            CovarianceType::Spherical => k,
        }
    }

    /// The sklearn shape of `covariances_`, as a slice of dimensions. Used by
    /// the Python layer to reshape the flat buffer.
    pub fn param_shape(self, k: usize, d: usize) -> Vec<usize> {
        match self {
            CovarianceType::Full => vec![k, d, d],
            CovarianceType::Tied => vec![d, d],
            CovarianceType::Diag => vec![k, d],
            CovarianceType::Spherical => vec![k],
        }
    }
}

impl TryFrom<&str> for CovarianceType {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, ()> {
        match value {
            "full" => Ok(CovarianceType::Full),
            "tied" => Ok(CovarianceType::Tied),
            "diag" => Ok(CovarianceType::Diag),
            "spherical" => Ok(CovarianceType::Spherical),
            _ => Err(()),
        }
    }
}

/// A component's covariance was not positive definite, so its Cholesky factor
/// does not exist — sklearn's
/// `"Fitting the mixture model failed because some components have ill-defined
/// empirical covariance"` failure, surfaced as a typed value.
#[derive(Debug, Clone, Copy)]
pub struct IllConditioned {
    /// Component index whose covariance failed (`usize::MAX` for the single
    /// shared `tied` covariance, which has no component index).
    pub component: usize,
    /// Diagonal index where the running pivot went non-positive.
    pub pivot_index: usize,
    /// The non-positive pivot value.
    pub pivot_value: f64,
}

// ---------------------------------------------------------------------------
// Packed symmetric (upper-triangle) indexing
// ---------------------------------------------------------------------------

/// Elements in the packed upper triangle of a `d × d` symmetric matrix.
#[inline(always)]
fn tri_len(d: usize) -> usize {
    d * (d + 1) / 2
}

/// Flat offset of the first stored element of row `a` in the packed upper
/// triangle (`(a, a)`), so `(a, b)` with `b >= a` lives at `row_off(a, d) + b`.
///
/// Derived rather than looked up: row `a` starts after `Σ_{r<a} (d − r)`
/// elements, which is `a·d − a(a−1)/2`; subtracting the `a` skipped columns of
/// its own row folds the `+ (b − a)` into a plain `+ b`.
#[inline(always)]
fn row_off(a: usize, d: usize) -> usize {
    a * d - a * (a.saturating_sub(1)) / 2 - a
}

// ---------------------------------------------------------------------------
// Cholesky / precision-Cholesky
// ---------------------------------------------------------------------------

/// In-place lower Cholesky of a row-major `d × d` SPD matrix, writing `L` into
/// the lower triangle of `out` and zeros above it.
///
/// Returns the failing `(pivot_index, pivot_value)` when a pivot is
/// non-positive, which is what turns sklearn's `ValueError` into a typed
/// [`IllConditioned`] one level up.
fn cholesky_lower(a: &[f64], d: usize, out: &mut [f64]) -> Result<(), (usize, f64)> {
    out.fill(0.0);
    for i in 0..d {
        for j in 0..=i {
            let mut sum = a[i * d + j];
            for p in 0..j {
                sum -= out[i * d + p] * out[j * d + p];
            }
            if i == j {
                if !(sum > 0.0) {
                    return Err((i, sum));
                }
                out[i * d + i] = sum.sqrt();
            } else {
                out[i * d + j] = sum / out[j * d + j];
            }
        }
    }
    Ok(())
}

/// `inv(L)ᵀ` for a lower-triangular `L`, written row-major into `out` as an
/// UPPER-triangular `d × d` matrix (zeros below the diagonal).
///
/// This is exactly sklearn's `solve_triangular(cov_chol, eye, lower=True).T`,
/// computed by forward substitution column-by-column rather than by a general
/// triangular solve against a dense identity.
fn inv_lower_transposed(l: &[f64], d: usize, out: &mut [f64]) {
    out.fill(0.0);
    // inv(L) is lower triangular: solve L · Z = I one column at a time.
    // Z[i][c] for i >= c; the transpose stores it at out[c*d + i].
    for c in 0..d {
        // Z[c][c] = 1 / L[c][c]
        let mut zcol = vec![0.0f64; d];
        zcol[c] = 1.0 / l[c * d + c];
        for i in (c + 1)..d {
            let mut sum = 0.0;
            for p in c..i {
                sum += l[i * d + p] * zcol[p];
            }
            zcol[i] = -sum / l[i * d + i];
        }
        for i in c..d {
            out[c * d + i] = zcol[i];
        }
    }
}

/// sklearn's `_compute_precision_cholesky`: the Cholesky of the PRECISION for
/// every component, in the layout [`CovarianceType::param_len`] describes.
///
/// `full` / `tied` store an UPPER-triangular `d × d` per matrix (`inv(L)ᵀ`);
/// `diag` / `spherical` store `1/√σ` elementwise, exactly as sklearn does.
pub fn precisions_cholesky(
    covariances: &[f64],
    k: usize,
    d: usize,
    ct: CovarianceType,
) -> Result<Vec<f64>, IllConditioned> {
    let mut out = vec![0.0f64; ct.param_len(k, d)];
    match ct {
        CovarianceType::Full => {
            let mut l = vec![0.0f64; d * d];
            for c in 0..k {
                let cov = &covariances[c * d * d..(c + 1) * d * d];
                cholesky_lower(cov, d, &mut l).map_err(|(pivot_index, pivot_value)| {
                    IllConditioned {
                        component: c,
                        pivot_index,
                        pivot_value,
                    }
                })?;
                inv_lower_transposed(&l, d, &mut out[c * d * d..(c + 1) * d * d]);
            }
        }
        CovarianceType::Tied => {
            let mut l = vec![0.0f64; d * d];
            cholesky_lower(covariances, d, &mut l).map_err(|(pivot_index, pivot_value)| {
                IllConditioned {
                    component: usize::MAX,
                    pivot_index,
                    pivot_value,
                }
            })?;
            inv_lower_transposed(&l, d, &mut out);
        }
        CovarianceType::Diag | CovarianceType::Spherical => {
            for (i, (o, &c)) in out.iter_mut().zip(covariances.iter()).enumerate() {
                if !(c > 0.0) {
                    return Err(IllConditioned {
                        component: if ct == CovarianceType::Diag { i / d.max(1) } else { i },
                        pivot_index: if ct == CovarianceType::Diag { i % d.max(1) } else { 0 },
                        pivot_value: c,
                    });
                }
                *o = 1.0 / c.sqrt();
            }
        }
    }
    Ok(out)
}

/// sklearn's `_compute_log_det_cholesky`: `log|precision_chol|` per component,
/// length `k` for every parameterization (`tied` broadcasts its single value).
pub fn log_det_cholesky(
    prec_chol: &[f64],
    k: usize,
    d: usize,
    ct: CovarianceType,
) -> Vec<f64> {
    match ct {
        CovarianceType::Full => (0..k)
            .map(|c| {
                let m = &prec_chol[c * d * d..(c + 1) * d * d];
                (0..d).map(|j| m[j * d + j].ln()).sum()
            })
            .collect(),
        CovarianceType::Tied => {
            let v: f64 = (0..d).map(|j| prec_chol[j * d + j].ln()).sum();
            vec![v; k]
        }
        CovarianceType::Diag => (0..k)
            .map(|c| prec_chol[c * d..(c + 1) * d].iter().map(|v| v.ln()).sum())
            .collect(),
        CovarianceType::Spherical => {
            (0..k).map(|c| d as f64 * prec_chol[c].ln()).collect()
        }
    }
}

/// Recover `covariances_` from `precisions_cholesky_` — the inverse of
/// [`precisions_cholesky`], used when the caller injects `precisions_init`
/// (sklearn's `_set_parameters` keeps both, and `covariances_` has to agree).
pub fn covariances_from_precisions_cholesky(
    prec_chol: &[f64],
    k: usize,
    d: usize,
    ct: CovarianceType,
) -> Vec<f64> {
    let mut out = vec![0.0f64; ct.param_len(k, d)];
    match ct {
        CovarianceType::Full | CovarianceType::Tied => {
            let mats = if ct == CovarianceType::Full { k } else { 1 };
            for c in 0..mats {
                let p = &prec_chol[c * d * d..(c + 1) * d * d];
                // p is inv(L)ᵀ (upper). Λ = p·pᵀ, and Σ = Λ⁻¹ = Lᵀ⁻¹... but the
                // numerically direct route is Σ = (p·pᵀ)⁻¹ = L·Lᵀ with
                // L = inv(pᵀ), so invert the upper triangle back and multiply.
                let mut l = vec![0.0f64; d * d];
                // pᵀ is lower triangular with pᵀ[i][j] = p[j][i]; inv of it is L.
                for i in 0..d {
                    l[i * d + i] = 1.0 / p[i * d + i];
                }
                for c0 in 0..d {
                    for i in (c0 + 1)..d {
                        let mut sum = 0.0;
                        for q in c0..i {
                            sum += p[q * d + i] * l[q * d + c0];
                        }
                        l[i * d + c0] = -sum / p[i * d + i];
                    }
                }
                // Σ = L · Lᵀ (L lower).
                let out_m = &mut out[c * d * d..(c + 1) * d * d];
                for i in 0..d {
                    for j in 0..=i {
                        let mut sum = 0.0;
                        for q in 0..=j {
                            sum += l[i * d + q] * l[j * d + q];
                        }
                        out_m[i * d + j] = sum;
                        out_m[j * d + i] = sum;
                    }
                }
            }
        }
        CovarianceType::Diag | CovarianceType::Spherical => {
            for (o, &p) in out.iter_mut().zip(prec_chol.iter()) {
                *o = 1.0 / (p * p);
            }
        }
    }
    out
}

/// The LOWER Cholesky factor `L` (with `L·Lᵀ = Σ`) of every block of a
/// covariance buffer, in the same layout.
///
/// Distinct from [`precisions_cholesky`], which factors the PRECISION and
/// returns the upper `inv(L)ᵀ`. Sampling from the fitted mixture needs `L`
/// itself (`x = μ + L·z`), so it is exposed rather than re-derived by two
/// inversions.
pub fn cholesky_lower_blocks(
    covariances: &[f64],
    k: usize,
    d: usize,
    ct: CovarianceType,
) -> Result<Vec<f64>, IllConditioned> {
    match ct {
        CovarianceType::Full | CovarianceType::Tied => {
            let mats = if ct == CovarianceType::Full { k } else { 1 };
            let mut out = vec![0.0f64; ct.param_len(k, d)];
            for c in 0..mats {
                cholesky_lower(
                    &covariances[c * d * d..(c + 1) * d * d],
                    d,
                    &mut out[c * d * d..(c + 1) * d * d],
                )
                .map_err(|(pivot_index, pivot_value)| IllConditioned {
                    component: if ct == CovarianceType::Full { c } else { usize::MAX },
                    pivot_index,
                    pivot_value,
                })?;
            }
            Ok(out)
        }
        CovarianceType::Diag | CovarianceType::Spherical => {
            let mut out = vec![0.0f64; ct.param_len(k, d)];
            for (i, (o, &v)) in out.iter_mut().zip(covariances.iter()).enumerate() {
                if !(v > 0.0) {
                    return Err(IllConditioned {
                        component: if ct == CovarianceType::Diag { i / d.max(1) } else { i },
                        pivot_index: if ct == CovarianceType::Diag { i % d.max(1) } else { 0 },
                        pivot_value: v,
                    });
                }
                *o = v.sqrt();
            }
            Ok(out)
        }
    }
}

/// Invert every SPD block of a parameter buffer in place of a matrix inverse —
/// `Σ = Λ⁻¹` (or `Λ = Σ⁻¹`), in the [`CovarianceType::param_len`] layout.
///
/// This is what makes sklearn's `precisions_init` usable without changing the
/// module's triangularity invariant. sklearn stores
/// `precisions_cholesky_ = cholesky(precisions_init, lower=True)` — a LOWER
/// triangular factor — whereas everything here assumes the UPPER `inv(L)ᵀ` form
/// [`precisions_cholesky`] produces. Both satisfy `P·Pᵀ = Λ`, so they give
/// identical Mahalanobis distances and identical `log|P|`; rather than teach the
/// kernels two layouts, the estimator inverts `precisions_init` to a covariance
/// here and re-derives the canonical upper factor from it.
pub fn invert_spd(
    a: &[f64],
    k: usize,
    d: usize,
    ct: CovarianceType,
) -> Result<Vec<f64>, IllConditioned> {
    match ct {
        CovarianceType::Full | CovarianceType::Tied => {
            let chol = precisions_cholesky(a, k, d, ct)?;
            Ok(precisions_from_cholesky(&chol, k, d, ct))
        }
        CovarianceType::Diag | CovarianceType::Spherical => {
            let mut out = vec![0.0f64; ct.param_len(k, d)];
            for (i, (o, &v)) in out.iter_mut().zip(a.iter()).enumerate() {
                if !(v > 0.0) {
                    return Err(IllConditioned {
                        component: if ct == CovarianceType::Diag { i / d.max(1) } else { i },
                        pivot_index: if ct == CovarianceType::Diag { i % d.max(1) } else { 0 },
                        pivot_value: v,
                    });
                }
                *o = 1.0 / v;
            }
            Ok(out)
        }
    }
}

/// `precisions_` from `precisions_cholesky_` — sklearn's `precisions_`
/// attribute (`prec_chol · prec_cholᵀ` for the matrix forms, `prec_chol²`
/// otherwise).
pub fn precisions_from_cholesky(
    prec_chol: &[f64],
    k: usize,
    d: usize,
    ct: CovarianceType,
) -> Vec<f64> {
    let mut out = vec![0.0f64; ct.param_len(k, d)];
    match ct {
        CovarianceType::Full | CovarianceType::Tied => {
            let mats = if ct == CovarianceType::Full { k } else { 1 };
            for c in 0..mats {
                let p = &prec_chol[c * d * d..(c + 1) * d * d];
                let o = &mut out[c * d * d..(c + 1) * d * d];
                for i in 0..d {
                    for j in 0..=i {
                        // p is upper triangular: (p·pᵀ)[i][j] = Σ_q p[i][q]·p[j][q]
                        // and p[i][q] = 0 for q < i.
                        let start = i.max(j);
                        let mut sum = 0.0;
                        for q in start..d {
                            sum += p[i * d + q] * p[j * d + q];
                        }
                        o[i * d + j] = sum;
                        o[j * d + i] = sum;
                    }
                }
            }
        }
        CovarianceType::Diag | CovarianceType::Spherical => {
            for (o, &p) in out.iter_mut().zip(prec_chol.iter()) {
                *o = p * p;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Mahalanobis kernels (the E-step inner loops)
// ---------------------------------------------------------------------------

/// `‖(x − μ)ᵀ P‖²` for an UPPER-triangular `d × d` precision Cholesky `P`.
///
/// Win #1 from the module docs lives here: the `j` loop starts at `a`, so the
/// `d(d−1)/2` structurally-zero entries sklearn's dense GEMM multiplies are
/// never touched. `scratch` is a caller-owned length-`d` register file so the
/// pass allocates nothing.
#[inline(always)]
fn maha_full(x: &[f64], mu: &[f64], p: &[f64], d: usize, scratch: &mut [f64]) -> f64 {
    scratch[..d].fill(0.0);
    for a in 0..d {
        let da = x[a] - mu[a];
        if da == 0.0 {
            continue;
        }
        let prow = &p[a * d..a * d + d];
        let out = &mut scratch[..d];
        for j in a..d {
            out[j] += da * prow[j];
        }
    }
    scratch[..d].iter().map(|v| v * v).sum()
}

/// `xᵀ P` for the shared `tied` precision Cholesky, hoisted OUT of the
/// component loop — win #2 from the module docs. Written into `out[..d]`.
#[inline(always)]
fn maha_tied_row(x: &[f64], p: &[f64], d: usize, out: &mut [f64]) {
    out[..d].fill(0.0);
    for a in 0..d {
        let xa = x[a];
        if xa == 0.0 {
            continue;
        }
        let prow = &p[a * d..a * d + d];
        let o = &mut out[..d];
        for j in a..d {
            o[j] += xa * prow[j];
        }
    }
}

/// Squared euclidean distance between two length-`d` rows.
#[inline(always)]
fn sq_dist(a: &[f64], b: &[f64], d: usize) -> f64 {
    let mut acc = 0.0;
    for j in 0..d {
        let t = a[j] - b[j];
        acc += t * t;
    }
    acc
}

/// The contiguous, pairwise-DISJOINT row block `[lo, hi)` each of `units` units
/// owns. Every pass in this module shares this one decomposition, which is what
/// lets all of them write their partials with no atomics and no locks: a unit
/// only ever touches rows it exclusively owns (the [`Shared`] contract).
fn row_ranges(n: usize, units: usize) -> Vec<(usize, usize)> {
    let per = n.div_ceil(units.max(1));
    (0..units)
        .map(|u| {
            let lo = (u * per).min(n);
            (lo, (lo + per).min(n))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

/// Per-fit workspace: the design, the geometry, the worker pool, and every
/// reduction buffer the two passes need — all allocated ONCE per fit and reused
/// across `max_iter · n_init` iterations.
pub struct GmmHost<'a> {
    x: &'a [f64],
    n: usize,
    d: usize,
    k: usize,
    ct: CovarianceType,
    reg_covar: f64,
    units: usize,
    pool: WorkerPool,
    /// `n × k` posterior responsibilities, the only `O(n·k)` buffer.
    resp: Vec<f64>,
    /// Per-unit `Σ log p(xᵢ)` partial (length `units`).
    part_lpn: Vec<f64>,
    /// Per-unit `nk` partial (`units × k`).
    part_nk: Vec<f64>,
    /// Per-unit unnormalized mean partial (`units × k × d`).
    part_means: Vec<f64>,
    /// Per-unit covariance partial. `full` packs the upper triangle
    /// (`units × k × tri_len(d)`); `diag`/`spherical` use `units × k × d`;
    /// `tied` needs none (win #2 — its second moment is loop-invariant).
    part_cov: Vec<f64>,
    /// `Xᵀ X` upper triangle, computed once per fit for the `tied` M-step.
    xtx: Vec<f64>,
}

impl<'a> GmmHost<'a> {
    /// Build the workspace for one fit over the row-major `n × d` design `x`.
    ///
    /// The worker width is [`capability::cpu_launch_units`] clamped by the work
    /// available: a pass with fewer than [`MIN_ROWS_PER_UNIT`] rows per unit
    /// spends more on the two barrier crossings than on the arithmetic, and the
    /// pool degrades to the inline, thread-free form at `units == 1`.
    pub fn new(
        x: &'a [f64],
        n: usize,
        d: usize,
        k: usize,
        ct: CovarianceType,
        reg_covar: f64,
    ) -> Self {
        let units = Self::plan_units(n, d, k, ct);
        let cov_part_stride = match ct {
            CovarianceType::Full => k * tri_len(d),
            CovarianceType::Diag | CovarianceType::Spherical => k * d,
            CovarianceType::Tied => 0,
        };
        Self {
            x,
            n,
            d,
            k,
            ct,
            reg_covar,
            units,
            pool: WorkerPool::new(units),
            resp: vec![0.0; n * k],
            part_lpn: vec![0.0; units],
            part_nk: vec![0.0; units * k],
            part_means: vec![0.0; units * k * d],
            part_cov: vec![0.0; units * cov_part_stride],
            xtx: Vec::new(),
        }
    }

    /// Multiply-adds below which an extra worker cannot pay for its two barrier
    /// crossings.
    ///
    /// The crossings cost single-digit microseconds ([`WorkerPool`] docs), so
    /// the floor is stated in WORK, not in rows: a `full` row at `d = 128` costs
    /// ~500× what a `spherical` row at `d = 4` does, and a rows-only threshold
    /// would over-split the first and under-split the second.
    const MIN_WORK_PER_UNIT: usize = 1 << 16;

    /// Worker count for a pass over `n` rows, from the per-row arithmetic the
    /// `covariance_type` implies.
    ///
    /// `MLRS_GMM_UNITS` overrides it for on-target A/B (read through
    /// [`abflag`] so a test can scope the override to its own thread rather
    /// than racing `environ` — [[mlrs-abflag-test-knobs]]).
    fn plan_units(n: usize, d: usize, k: usize, ct: CovarianceType) -> usize {
        if let Some(v) = abflag::var("MLRS_GMM_UNITS").and_then(|v| v.parse::<usize>().ok()) {
            return v.max(1);
        }
        // `full`/`tied` walk the `d × d` precision triangle per component;
        // `diag`/`spherical` walk `d` scalars.
        let per_component = match ct {
            CovarianceType::Full | CovarianceType::Tied => d.saturating_mul(d + 1) / 2,
            CovarianceType::Diag | CovarianceType::Spherical => d,
        };
        let per_row = k.saturating_mul(per_component).max(1);
        let total = n.saturating_mul(per_row);
        let by_work = (total / Self::MIN_WORK_PER_UNIT).max(1);
        let hw = capability::cpu_launch_units() as usize;
        by_work.min(hw).min(n).max(1)
    }

    /// Read-only view of the current responsibilities (`n × k`, row-major).
    pub fn resp(&self) -> &[f64] {
        &self.resp
    }

    /// Overwrite the responsibilities — the init path writes `resp` directly and
    /// then runs an M-step, exactly like sklearn's `_initialize_parameters`.
    pub fn set_resp(&mut self, resp: &[f64]) {
        self.resp.copy_from_slice(resp);
    }

    /// Number of pool participants (driver included). Reported by the perf probe.
    pub fn units(&self) -> usize {
        self.units
    }

    // -- Initialization: the two k-means routes ------------------------------

    /// sklearn's `_kmeans_plusplus`: `k` D²-weighted seed INDICES, each chosen as
    /// the best of `2 + ⌊ln k⌋` local trials.
    ///
    /// Reproduces the reference's structure exactly — greedy sampling
    /// proportional to the squared distance to the nearest chosen seed, with the
    /// candidate that minimises the resulting potential winning each round — but
    /// not its bit stream: the numpy `Generator` draw is not reproducible from
    /// Rust, which is the same D-09 concession `KMeans` makes. `rng` is the
    /// caller's seeded stream.
    ///
    /// Serves BOTH `init_params`: `"k-means++"` uses the indices directly as
    /// one-hot responsibilities, and `"kmeans"` uses them to seed
    /// [`GmmHost::kmeans_labels`].
    pub fn kmeans_plusplus(&self, k: usize, rng: &mut SplitMix64) -> Vec<usize> {
        let (n, d) = (self.n, self.d);
        let x = self.x;
        let n_trials = 2 + (k as f64).ln() as usize;

        let mut indices = Vec::with_capacity(k);
        let first = (rng.next_below(n as u64)) as usize;
        indices.push(first);

        // `closest[i]` = squared distance from row `i` to its nearest seed.
        let mut closest: Vec<f64> = (0..n).map(|i| sq_dist(&x[i * d..], &x[first * d..], d)).collect();
        let mut potential: f64 = closest.iter().sum();

        let mut cand = vec![0usize; n_trials];
        let mut cand_closest = vec![0.0f64; n_trials * n];
        for _ in 1..k {
            // Inverse-CDF sample of `n_trials` candidates ∝ `closest`.
            for t in 0..n_trials {
                let target = rng.next_f64() * potential;
                let mut acc = 0.0;
                let mut pick = n - 1;
                for (i, &c) in closest.iter().enumerate() {
                    acc += c;
                    if acc >= target {
                        pick = i;
                        break;
                    }
                }
                cand[t] = pick;
            }
            // Each candidate's resulting potential, row-blocked over the pool.
            let sh = Shared::new(&mut cand_closest);
            let ranges = row_ranges(n, self.units);
            let cand_ref = &cand[..];
            let closest_ref = &closest[..];
            let pass = |u: usize| {
                let (lo, hi) = ranges[u];
                if lo >= hi {
                    return;
                }
                // SAFETY: unit `u` writes only column block `[lo, hi)` of every
                // candidate row, and the blocks are disjoint.
                let cc = unsafe { sh.get_mut() };
                for (t, &c) in cand_ref.iter().enumerate() {
                    let cx = &x[c * d..];
                    for i in lo..hi {
                        let dsq = sq_dist(&x[i * d..], cx, d);
                        cc[t * n + i] = dsq.min(closest_ref[i]);
                    }
                }
            };
            self.pool.run(&pass);

            let mut best = 0usize;
            let mut best_pot = f64::INFINITY;
            for t in 0..n_trials {
                let p: f64 = cand_closest[t * n..(t + 1) * n].iter().sum();
                if p < best_pot {
                    best_pot = p;
                    best = t;
                }
            }
            closest.copy_from_slice(&cand_closest[best * n..(best + 1) * n]);
            potential = best_pot;
            indices.push(cand[best]);
        }
        indices
    }

    /// Lloyd's k-means over the design, returning the hard label of every row —
    /// the `init_params="kmeans"` route, which sklearn implements as a full
    /// `KMeans(n_clusters=k, n_init=1).fit(X).labels_`.
    ///
    /// Mirrors sklearn's stopping rule (`center_shift_tot <= mean(var(X, 0)) *
    /// tol`, `max_iter = 300`) and its k-means++ seeding, parallelized over the
    /// same pool the EM loop uses. An emptied cluster keeps its previous center
    /// rather than being relocated: this is an INITIALIZER for EM, and the first
    /// M-step immediately overwrites whatever it produces.
    pub fn kmeans_labels(&self, rng: &mut SplitMix64) -> Vec<u32> {
        const KM_MAX_ITER: usize = 300;
        const KM_TOL: f64 = 1e-4;
        let (n, d, k, units) = (self.n, self.d, self.k, self.units);
        let x = self.x;

        let seeds = self.kmeans_plusplus(k, rng);
        let mut centers = vec![0.0f64; k * d];
        for (c, &s) in seeds.iter().enumerate() {
            centers[c * d..(c + 1) * d].copy_from_slice(&x[s * d..s * d + d]);
        }

        // sklearn's `_tolerance`: the tolerance is RELATIVE to the data scale.
        let mut mean_var = 0.0f64;
        for j in 0..d {
            let mut s = 0.0;
            let mut s2 = 0.0;
            for i in 0..n {
                let v = x[i * d + j];
                s += v;
                s2 += v * v;
            }
            let m = s / n as f64;
            mean_var += s2 / n as f64 - m * m;
        }
        let tol_abs = (mean_var / d as f64) * KM_TOL;

        let mut labels = vec![0u32; n];
        let mut part_sums = vec![0.0f64; units * k * d];
        let mut part_counts = vec![0.0f64; units * k];
        let ranges = row_ranges(n, units);

        for _ in 0..KM_MAX_ITER {
            part_sums.fill(0.0);
            part_counts.fill(0.0);
            let sh_lab = Shared::new(&mut labels);
            let sh_sum = Shared::new(&mut part_sums);
            let sh_cnt = Shared::new(&mut part_counts);
            let centers_ref = &centers[..];
            let pass = |u: usize| {
                let (lo, hi) = ranges[u];
                if lo >= hi {
                    return;
                }
                // SAFETY: rows `[lo, hi)` and unit-`u` accumulator slots only.
                let lab = unsafe { sh_lab.get_mut() };
                let sums = unsafe { sh_sum.get_mut() };
                let cnts = unsafe { sh_cnt.get_mut() };
                let su = &mut sums[u * k * d..(u + 1) * k * d];
                let cu = &mut cnts[u * k..(u + 1) * k];
                for i in lo..hi {
                    let xi = &x[i * d..(i + 1) * d];
                    let mut best = 0usize;
                    let mut bd = f64::INFINITY;
                    for c in 0..k {
                        let dsq = sq_dist(xi, &centers_ref[c * d..], d);
                        if dsq < bd {
                            bd = dsq;
                            best = c;
                        }
                    }
                    lab[i] = best as u32;
                    cu[best] += 1.0;
                    let acc = &mut su[best * d..(best + 1) * d];
                    for j in 0..d {
                        acc[j] += xi[j];
                    }
                }
            };
            self.pool.run(&pass);

            let mut shift = 0.0f64;
            for c in 0..k {
                let mut cnt = 0.0;
                for u in 0..units {
                    cnt += part_counts[u * k + c];
                }
                if cnt == 0.0 {
                    continue;
                }
                for j in 0..d {
                    let mut s = 0.0;
                    for u in 0..units {
                        s += part_sums[u * k * d + c * d + j];
                    }
                    let nv = s / cnt;
                    let dv = nv - centers[c * d + j];
                    shift += dv * dv;
                    centers[c * d + j] = nv;
                }
            }
            if shift <= tol_abs {
                break;
            }
        }
        labels
    }

    // -- Pass A: E-step fused with the `nk` / `means` reduction ---------------

    /// One fused E-step: fills `resp` with the posterior responsibilities and
    /// returns `(mean_log_prob_norm, nk, means)` — the M-step's first two
    /// outputs computed in the SAME sweep of the design.
    ///
    /// `mean_log_prob_norm` is sklearn's `lower_bound_`: the average over
    /// samples of `logsumexp_k(log π_k + log N(xᵢ | μ_k, Σ_k))`.
    pub fn e_step(
        &mut self,
        weights: &[f64],
        means: &[f64],
        prec_chol: &[f64],
    ) -> (f64, Vec<f64>, Vec<f64>) {
        let (n, d, k, ct) = (self.n, self.d, self.k, self.ct);
        let log_det = log_det_cholesky(prec_chol, k, d, ct);
        let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        // The `−0.5·d·ln(2π)` normalizer and `log π_k` and `log|P_k|` are all
        // per-component constants; fold them into ONE bias so the inner loop
        // adds a single number instead of three.
        let bias: Vec<f64> = (0..k)
            .map(|c| log_w[c] + log_det[c] - 0.5 * d as f64 * LOG_2PI)
            .collect();
        // `tied` shares one precision Cholesky, so `μ_kᵀP` is loop-invariant per
        // component and hoists out of the ROW loop entirely (win #2).
        let mu_p: Vec<f64> = if ct == CovarianceType::Tied {
            let mut out = vec![0.0f64; k * d];
            let mut scratch = vec![0.0f64; d];
            for c in 0..k {
                maha_tied_row(&means[c * d..(c + 1) * d], prec_chol, d, &mut scratch);
                out[c * d..(c + 1) * d].copy_from_slice(&scratch[..d]);
            }
            out
        } else {
            Vec::new()
        };

        self.part_lpn.fill(0.0);
        self.part_nk.fill(0.0);
        self.part_means.fill(0.0);

        let x = self.x;
        let resp = Shared::new(&mut self.resp);
        let p_lpn = Shared::new(&mut self.part_lpn);
        let p_nk = Shared::new(&mut self.part_nk);
        let p_means = Shared::new(&mut self.part_means);
        let units = self.units;
        let ranges = row_ranges(n, units);

        let pass = |u: usize| {
            let (lo, hi) = ranges[u];
            if lo >= hi {
                return;
            }
            // SAFETY: rows `[lo, hi)` and the unit-`u` slots of every partial
            // are owned exclusively by this unit for the whole phase.
            let resp = unsafe { resp.get_mut() };
            let p_lpn = unsafe { p_lpn.get_mut() };
            let p_nk = unsafe { p_nk.get_mut() };
            let p_means = unsafe { p_means.get_mut() };

            let mut wlp = vec![0.0f64; ROW_BLOCK * k];
            let mut scratch = vec![0.0f64; d];
            let mut acc_lpn = 0.0f64;
            let nk_u = &mut p_nk[u * k..(u + 1) * k];
            let means_u = &mut p_means[u * k * d..(u + 1) * k * d];

            let mut i0 = lo;
            while i0 < hi {
                let i1 = (i0 + ROW_BLOCK).min(hi);
                let nb = i1 - i0;

                // --- phase 1: the weighted log-probabilities for this block ---
                match ct {
                    // COMPONENT-outer, row-inner. The loop nest is the whole
                    // point: `prec_chol` is `k · d²` doubles (1 MB at k=8,
                    // d=128), so a row-outer nest streams every precision
                    // matrix through L2 once PER ROW and the E-step becomes
                    // bandwidth-bound instead of compute-bound. Hoisting the
                    // component out and blocking the rows keeps ONE `d × d`
                    // matrix (128 KB) resident across `ROW_BLOCK` rows.
                    CovarianceType::Full => {
                        for c in 0..k {
                            let mu = &means[c * d..(c + 1) * d];
                            let p = &prec_chol[c * d * d..(c + 1) * d * d];
                            let b = bias[c];
                            for t in 0..nb {
                                let xi = &x[(i0 + t) * d..(i0 + t + 1) * d];
                                wlp[t * k + c] =
                                    b - 0.5 * maha_full(xi, mu, p, d, &mut scratch);
                            }
                        }
                    }
                    // `tied` has ONE shared `d × d` factor, which is resident by
                    // construction; the per-row `xᵀP` is the expensive part and
                    // is computed once for all components (win #2).
                    CovarianceType::Tied => {
                        for t in 0..nb {
                            let xi = &x[(i0 + t) * d..(i0 + t + 1) * d];
                            maha_tied_row(xi, prec_chol, d, &mut scratch);
                            for c in 0..k {
                                let mc = &mu_p[c * d..(c + 1) * d];
                                let mut acc = 0.0;
                                for j in 0..d {
                                    let e = scratch[j] - mc[j];
                                    acc += e * e;
                                }
                                wlp[t * k + c] = bias[c] - 0.5 * acc;
                            }
                        }
                    }
                    // `k · d` doubles of precision — cache-resident at any
                    // realistic geometry, so row-outer is fine and keeps `xi`
                    // hot instead.
                    CovarianceType::Diag => {
                        for t in 0..nb {
                            let xi = &x[(i0 + t) * d..(i0 + t + 1) * d];
                            for c in 0..k {
                                let mu = &means[c * d..(c + 1) * d];
                                let pc = &prec_chol[c * d..(c + 1) * d];
                                let mut acc = 0.0;
                                for j in 0..d {
                                    let e = (xi[j] - mu[j]) * pc[j];
                                    acc += e * e;
                                }
                                wlp[t * k + c] = bias[c] - 0.5 * acc;
                            }
                        }
                    }
                    CovarianceType::Spherical => {
                        for t in 0..nb {
                            let xi = &x[(i0 + t) * d..(i0 + t + 1) * d];
                            for c in 0..k {
                                let mu = &means[c * d..(c + 1) * d];
                                let pc = prec_chol[c];
                                let mut acc = 0.0;
                                for j in 0..d {
                                    let e = xi[j] - mu[j];
                                    acc += e * e;
                                }
                                wlp[t * k + c] = bias[c] - 0.5 * acc * pc * pc;
                            }
                        }
                    }
                }

                // --- phase 2: normalize, and fold `nk` / `means` in ---------
                for t in 0..nb {
                    let i = i0 + t;
                    let xi = &x[i * d..(i + 1) * d];
                    let w = &wlp[t * k..(t + 1) * k];

                    let mut mx = f64::NEG_INFINITY;
                    for &v in w {
                        if v > mx {
                            mx = v;
                        }
                    }
                    let mut se = 0.0;
                    for &v in w {
                        se += (v - mx).exp();
                    }
                    let lse = mx + se.ln();
                    acc_lpn += lse;

                    let row = &mut resp[i * k..(i + 1) * k];
                    for c in 0..k {
                        let r = (w[c] - lse).exp();
                        row[c] = r;
                        nk_u[c] += r;
                        if r != 0.0 {
                            let mu_acc = &mut means_u[c * d..(c + 1) * d];
                            for j in 0..d {
                                mu_acc[j] += r * xi[j];
                            }
                        }
                    }
                }
                i0 = i1;
            }
            p_lpn[u] = acc_lpn;
        };
        self.pool.run(&pass);

        let mean_lpn: f64 = self.part_lpn.iter().sum::<f64>() / n as f64;
        let mut nk = vec![0.0f64; k];
        let mut mu = vec![0.0f64; k * d];
        for u in 0..units {
            for c in 0..k {
                nk[c] += self.part_nk[u * k + c];
            }
            for j in 0..k * d {
                mu[j] += self.part_means[u * k * d + j];
            }
        }
        for c in 0..k {
            nk[c] += NK_EPS;
            let inv = 1.0 / nk[c];
            for j in 0..d {
                mu[c * d + j] *= inv;
            }
        }
        (mean_lpn, nk, mu)
    }

    /// `nk` and `means` from an EXTERNALLY supplied `resp` — the init path,
    /// where responsibilities come from k-means / a random draw rather than from
    /// an E-step. Mirrors the reduction half of [`GmmHost::e_step`].
    pub fn nk_and_means_from_resp(&mut self) -> (Vec<f64>, Vec<f64>) {
        let (n, d, k) = (self.n, self.d, self.k);
        let mut nk = vec![0.0f64; k];
        let mut mu = vec![0.0f64; k * d];
        for i in 0..n {
            let row = &self.resp[i * k..(i + 1) * k];
            let xi = &self.x[i * d..(i + 1) * d];
            for c in 0..k {
                let r = row[c];
                if r == 0.0 {
                    continue;
                }
                nk[c] += r;
                let acc = &mut mu[c * d..(c + 1) * d];
                for j in 0..d {
                    acc[j] += r * xi[j];
                }
            }
        }
        for c in 0..k {
            nk[c] += NK_EPS;
            let inv = 1.0 / nk[c];
            for j in 0..d {
                mu[c * d + j] *= inv;
            }
        }
        (nk, mu)
    }

    // -- Pass B: the covariance -------------------------------------------

    /// The M-step covariance for the current `resp` / `nk` / `means`, in the
    /// [`CovarianceType::param_len`] layout, with `reg_covar` already added to
    /// the diagonal exactly where sklearn adds it.
    pub fn covariances(&mut self, nk: &[f64], means: &[f64]) -> Vec<f64> {
        match self.ct {
            CovarianceType::Full => self.cov_full(nk, means),
            CovarianceType::Tied => self.cov_tied(nk, means),
            CovarianceType::Diag => self.cov_diag(nk, means),
            CovarianceType::Spherical => {
                let diag = self.cov_diag(nk, means);
                let d = self.d;
                (0..self.k)
                    .map(|c| diag[c * d..(c + 1) * d].iter().sum::<f64>() / d as f64)
                    .collect()
            }
        }
    }

    /// `full`: a packed-upper-triangle weighted second moment per component,
    /// mirrored into the full `d × d` at the end — half the multiplies of
    /// sklearn's `np.dot(resp * diff.T, diff)` (win #1).
    fn cov_full(&mut self, nk: &[f64], means: &[f64]) -> Vec<f64> {
        let (n, d, k, units) = (self.n, self.d, self.k, self.units);
        let tl = tri_len(d);
        self.part_cov.fill(0.0);

        let x = self.x;
        let resp = &self.resp[..];
        let p_cov = Shared::new(&mut self.part_cov);
        let ranges = row_ranges(n, units);

        let pass = |u: usize| {
            let (lo, hi) = ranges[u];
            if lo >= hi {
                return;
            }
            // SAFETY: unit `u` writes only its own `[u*k*tl, (u+1)*k*tl)` slots.
            let p_cov = unsafe { p_cov.get_mut() };
            let acc = &mut p_cov[u * k * tl..(u + 1) * k * tl];
            let mut diff = vec![0.0f64; d];
            // COMPONENT-outer over a row block, for the same cache reason as
            // the E-step: this unit's accumulator is `k · d(d+1)/2` doubles
            // (528 KB at k=8, d=128), so a row-outer nest evicts and reloads
            // every component's triangle on every row. One triangle (66 KB)
            // stays resident across `ROW_BLOCK` rows instead.
            let mut i0 = lo;
            while i0 < hi {
                let i1 = (i0 + ROW_BLOCK).min(hi);
                for c in 0..k {
                    let mu = &means[c * d..(c + 1) * d];
                    let ac = &mut acc[c * tl..(c + 1) * tl];
                    for i in i0..i1 {
                        let r = resp[i * k + c];
                        if r == 0.0 {
                            continue;
                        }
                        let xi = &x[i * d..(i + 1) * d];
                        for j in 0..d {
                            diff[j] = xi[j] - mu[j];
                        }
                        for a in 0..d {
                            let da = r * diff[a];
                            if da == 0.0 {
                                continue;
                            }
                            let off = row_off(a, d);
                            for b in a..d {
                                ac[off + b] += da * diff[b];
                            }
                        }
                    }
                }
                i0 = i1;
            }
        };
        self.pool.run(&pass);

        let mut out = vec![0.0f64; k * d * d];
        for c in 0..k {
            let inv = 1.0 / nk[c];
            let m = &mut out[c * d * d..(c + 1) * d * d];
            for a in 0..d {
                let off = row_off(a, d);
                for b in a..d {
                    let mut s = 0.0;
                    for u in 0..units {
                        s += self.part_cov[u * k * tl + c * tl + off + b];
                    }
                    let v = s * inv;
                    m[a * d + b] = v;
                    m[b * d + a] = v;
                }
            }
            for a in 0..d {
                m[a * d + a] += self.reg_covar;
            }
        }
        out
    }

    /// `tied`: `(XᵀX − Σ_k nk_k μ_k μ_kᵀ) / Σ nk`, with `XᵀX` computed ONCE per
    /// fit (win #2 — sklearn recomputes it every iteration).
    fn cov_tied(&mut self, nk: &[f64], means: &[f64]) -> Vec<f64> {
        let (d, k) = (self.d, self.k);
        self.ensure_xtx();
        let tl = tri_len(d);
        let nk_sum: f64 = nk.iter().sum();
        let mut out = vec![0.0f64; d * d];
        for a in 0..d {
            let off = row_off(a, d);
            for b in a..d {
                let mut s = self.xtx[off + b];
                for c in 0..k {
                    s -= nk[c] * means[c * d + a] * means[c * d + b];
                }
                let v = s / nk_sum;
                out[a * d + b] = v;
                out[b * d + a] = v;
            }
        }
        debug_assert_eq!(self.xtx.len(), tl);
        for a in 0..d {
            out[a * d + a] += self.reg_covar;
        }
        out
    }

    /// The loop-invariant `XᵀX` upper triangle for the `tied` M-step, materialized
    /// on first use and then reused for every iteration of every `n_init` restart.
    fn ensure_xtx(&mut self) {
        if !self.xtx.is_empty() {
            return;
        }
        let (n, d, units) = (self.n, self.d, self.units);
        let tl = tri_len(d);
        let mut parts = vec![0.0f64; units * tl];
        let x = self.x;
        let sh = Shared::new(&mut parts);
        let ranges = row_ranges(n, units);
        let pass = |u: usize| {
            let (lo, hi) = ranges[u];
            if lo >= hi {
                return;
            }
            // SAFETY: unit `u` owns `[u*tl, (u+1)*tl)` exclusively.
            let acc = &mut unsafe { sh.get_mut() }[u * tl..(u + 1) * tl];
            for i in lo..hi {
                let xi = &x[i * d..(i + 1) * d];
                for a in 0..d {
                    let xa = xi[a];
                    if xa == 0.0 {
                        continue;
                    }
                    let off = row_off(a, d);
                    for b in a..d {
                        acc[off + b] += xa * xi[b];
                    }
                }
            }
        };
        self.pool.run(&pass);
        let mut xtx = vec![0.0f64; tl];
        for u in 0..units {
            for j in 0..tl {
                xtx[j] += parts[u * tl + j];
            }
        }
        self.xtx = xtx;
    }

    /// `diag`: `Σ_i r_ik x_ij² / nk_k − μ_kj² + reg_covar`, the algebraic
    /// simplification of sklearn's
    /// `avg_X2 − 2·avg_X_means + avg_means2` (its `avg_X_means` IS `avg_means2`,
    /// because `means` is by construction `respᵀX / nk`).
    fn cov_diag(&mut self, nk: &[f64], means: &[f64]) -> Vec<f64> {
        let (n, d, k, units) = (self.n, self.d, self.k, self.units);
        self.part_cov.fill(0.0);
        let x = self.x;
        let resp = &self.resp[..];
        let p_cov = Shared::new(&mut self.part_cov);
        let ranges = row_ranges(n, units);
        let pass = |u: usize| {
            let (lo, hi) = ranges[u];
            if lo >= hi {
                return;
            }
            // SAFETY: unit `u` owns `[u*k*d, (u+1)*k*d)` exclusively.
            let p_cov = unsafe { p_cov.get_mut() };
            let acc = &mut p_cov[u * k * d..(u + 1) * k * d];
            for i in lo..hi {
                let xi = &x[i * d..(i + 1) * d];
                let row = &resp[i * k..(i + 1) * k];
                for c in 0..k {
                    let r = row[c];
                    if r == 0.0 {
                        continue;
                    }
                    let ac = &mut acc[c * d..(c + 1) * d];
                    for j in 0..d {
                        ac[j] += r * xi[j] * xi[j];
                    }
                }
            }
        };
        self.pool.run(&pass);

        let mut out = vec![0.0f64; k * d];
        for c in 0..k {
            let inv = 1.0 / nk[c];
            for j in 0..d {
                let mut s = 0.0;
                for u in 0..units {
                    s += self.part_cov[u * k * d + c * d + j];
                }
                let m = means[c * d + j];
                out[c * d + j] = s * inv - m * m + self.reg_covar;
            }
        }
        out
    }
}

/// Standalone (no workspace) E-step for SCORING: the `n × k` weighted log
/// probabilities `log π_k + log N(xᵢ | μ_k, Σ_k)` for an arbitrary design.
///
/// This is what `score_samples` / `predict_proba` / `predict` all reduce to; it
/// shares the same triangular Mahalanobis kernels as the fit loop but keeps no
/// state, so a fitted estimator can score without rebuilding a [`GmmHost`].
pub fn weighted_log_prob(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    ct: CovarianceType,
    weights: &[f64],
    means: &[f64],
    prec_chol: &[f64],
) -> Vec<f64> {
    let log_det = log_det_cholesky(prec_chol, k, d, ct);
    let bias: Vec<f64> = (0..k)
        .map(|c| weights[c].ln() + log_det[c] - 0.5 * d as f64 * LOG_2PI)
        .collect();
    let mu_p: Vec<f64> = if ct == CovarianceType::Tied {
        let mut out = vec![0.0f64; k * d];
        let mut scratch = vec![0.0f64; d];
        for c in 0..k {
            maha_tied_row(&means[c * d..(c + 1) * d], prec_chol, d, &mut scratch);
            out[c * d..(c + 1) * d].copy_from_slice(&scratch[..d]);
        }
        out
    } else {
        Vec::new()
    };

    let units = GmmHost::plan_units(n, d, k, ct);
    let pool = WorkerPool::new(units);
    let mut out = vec![0.0f64; n * k];
    let sh = Shared::new(&mut out);
    let ranges = row_ranges(n, units);
    let pass = |u: usize| {
        let (lo, hi) = ranges[u];
        if lo >= hi {
            return;
        }
        // SAFETY: unit `u` writes only rows `[lo, hi)`.
        let out = unsafe { sh.get_mut() };
        let mut scratch = vec![0.0f64; d];
        // Same blocked, component-outer nest as `GmmHost::e_step` phase 1, and
        // for the same reason — see the comments there.
        let mut i0 = lo;
        while i0 < hi {
            let i1 = (i0 + ROW_BLOCK).min(hi);
            match ct {
                CovarianceType::Full => {
                    for c in 0..k {
                        let mu = &means[c * d..(c + 1) * d];
                        let p = &prec_chol[c * d * d..(c + 1) * d * d];
                        let b = bias[c];
                        for i in i0..i1 {
                            let xi = &x[i * d..(i + 1) * d];
                            out[i * k + c] = b - 0.5 * maha_full(xi, mu, p, d, &mut scratch);
                        }
                    }
                }
                CovarianceType::Tied => {
                    for i in i0..i1 {
                        let xi = &x[i * d..(i + 1) * d];
                        maha_tied_row(xi, prec_chol, d, &mut scratch);
                        for c in 0..k {
                            let mc = &mu_p[c * d..(c + 1) * d];
                            let mut acc = 0.0;
                            for j in 0..d {
                                let t = scratch[j] - mc[j];
                                acc += t * t;
                            }
                            out[i * k + c] = bias[c] - 0.5 * acc;
                        }
                    }
                }
                CovarianceType::Diag => {
                    for i in i0..i1 {
                        let xi = &x[i * d..(i + 1) * d];
                        for c in 0..k {
                            let mu = &means[c * d..(c + 1) * d];
                            let pc = &prec_chol[c * d..(c + 1) * d];
                            let mut acc = 0.0;
                            for j in 0..d {
                                let t = (xi[j] - mu[j]) * pc[j];
                                acc += t * t;
                            }
                            out[i * k + c] = bias[c] - 0.5 * acc;
                        }
                    }
                }
                CovarianceType::Spherical => {
                    for i in i0..i1 {
                        let xi = &x[i * d..(i + 1) * d];
                        for c in 0..k {
                            let mu = &means[c * d..(c + 1) * d];
                            let pc = prec_chol[c];
                            let mut acc = 0.0;
                            for j in 0..d {
                                let t = xi[j] - mu[j];
                                acc += t * t;
                            }
                            out[i * k + c] = bias[c] - 0.5 * acc * pc * pc;
                        }
                    }
                }
            }
            i0 = i1;
        }
    };
    pool.run(&pass);
    out
}

/// Row-wise `logsumexp` of an `n × k` matrix — sklearn's `log_prob_norm`.
pub fn logsumexp_rows(m: &[f64], n: usize, k: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let row = &m[i * k..(i + 1) * k];
            let mut mx = f64::NEG_INFINITY;
            for &v in row {
                if v > mx {
                    mx = v;
                }
            }
            if !mx.is_finite() {
                return mx;
            }
            let se: f64 = row.iter().map(|&v| (v - mx).exp()).sum();
            mx + se.ln()
        })
        .collect()
}
