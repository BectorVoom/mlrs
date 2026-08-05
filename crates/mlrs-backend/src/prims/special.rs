//! `special` — the scalar special functions the variational mixture models need.
//!
//! `BayesianGaussianMixture` (MIX-02) is the first estimator in the crate whose
//! CLOSED FORM contains `Γ`: its variational E-step needs `ψ` (digamma) for the
//! expected log-precision and the expected log-weights, and its evidence lower
//! bound needs `lnΓ` and `lnB` for the Wishart and Dirichlet log-normalizers.
//! scipy supplies all three to sklearn; Rust's `std` exposes none of them
//! (`f64::ln_gamma` is permanently unstable), so they live here.
//!
//! ## Why a prim and not a private helper
//! Three reasons, in order of weight:
//!
//! 1. **Accuracy is a correctness gate, not a nicety.** The oracle contract is
//!    1e-5 against scipy. `ψ` is evaluated on `½(ν − j)` for every feature `j`
//!    of every component of every iteration, and the results are EXPONENTIATED
//!    in the E-step — a 1e-6 error in `ψ` moves a responsibility, which moves
//!    `nk`, which moves everything. Each function here is accurate to a few
//!    ULP over the whole domain the mixture models reach, which is why the
//!    implementations are asymptotic-series-with-recurrence rather than the
//!    short rational fits that would be "good enough" for a plot.
//! 2. **`density::kernel_density` already carries a private `lgamma`.** A
//!    second private copy would be the third `lnΓ` in the workspace. This
//!    module is the shared home; `kernel_density`'s copy is left alone only
//!    because moving it would be an unrelated change to a shipped estimator.
//! 3. **They are pure host scalars by construction.** `d` digamma calls per
//!    component per iteration is `O(k·d)` work against the E-step's
//!    `O(n·k·d²)` — there is nothing here for a device kernel to do, and
//!    `cubecl`'s cuda backend has no `f64` transcendentals to do it with
//!    anyway ([[mlrs-cubecl-cuda-f64-not-advertised]]).
//!
//! Tests live in `crates/mlrs-backend/tests/special_test.rs` (AGENTS.md §2).

/// `½·ln(2π)` — the constant term of the Stirling series.
const HALF_LOG_2PI: f64 = 0.918_938_533_204_672_74;

/// Argument above which the Stirling series for `lnΓ` is used directly.
///
/// At `x = 16` the first DROPPED term (`691/(360360·x¹¹)`) is `~4e-17`
/// relative — below one ULP of the `~35` the function returns there — so the
/// series is exact to rounding from this point up, and the recurrence below
/// only ever runs a bounded number of times.
const LGAMMA_STIRLING_MIN: f64 = 16.0;

/// Argument above which the asymptotic series for `ψ` is used directly.
///
/// Same reasoning as [`LGAMMA_STIRLING_MIN`]: with terms through `B₁₂` the
/// first dropped term is `1/(12·x¹⁴)`, i.e. `~1e-18` at `x = 10`.
const DIGAMMA_ASYMPTOTIC_MIN: f64 = 10.0;

/// `lnΓ(x)` — scipy's `gammaln`, for `x > 0`.
///
/// Stirling's series above [`LGAMMA_STIRLING_MIN`], reached from below by the
/// upward recurrence `lnΓ(x) = lnΓ(x + 1) − ln x`. The recurrence direction
/// matters: downward (`lnΓ(x+1) = lnΓ(x) + ln x`) would accumulate the same
/// number of roundings but around a SMALLER value, so the relative error would
/// grow; going up, every correction is subtracted from a result that already
/// dominates it.
///
/// Non-positive integers are poles and return `+∞`; other negative arguments
/// use the reflection formula. Neither branch is reachable from the mixture
/// models (every argument there is `½(ν − j)` with `ν > d − 1`, or a Dirichlet
/// concentration, both strictly positive), but a special function that silently
/// returns garbage off its documented domain is a trap for the next caller.
pub fn lgamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::INFINITY;
        }
        // Reflection: lnΓ(x) = ln(π / |sin(πx)|) − lnΓ(1 − x).
        let s = (std::f64::consts::PI * x).sin().abs();
        return std::f64::consts::PI.ln() - s.ln() - lgamma(1.0 - x);
    }
    // Upward recurrence into the Stirling regime.
    let mut acc = 0.0;
    let mut z = x;
    while z < LGAMMA_STIRLING_MIN {
        acc -= z.ln();
        z += 1.0;
    }
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    // Σ B₂ₙ / (2n(2n−1) z^(2n−1)) for n = 1..6, by Horner in 1/z².
    let series = inv
        * (1.0 / 12.0
            + inv2
                * (-1.0 / 360.0
                    + inv2
                        * (1.0 / 1260.0
                            + inv2
                                * (-1.0 / 1680.0
                                    + inv2 * (1.0 / 1188.0 + inv2 * (-691.0 / 360_360.0))))));
    acc + (z - 0.5) * z.ln() - z + HALF_LOG_2PI + series
}

/// `ψ(x) = d/dx lnΓ(x)` — scipy's `digamma`, for `x > 0`.
///
/// The asymptotic series above [`DIGAMMA_ASYMPTOTIC_MIN`], reached by the
/// upward recurrence `ψ(x) = ψ(x + 1) − 1/x`.
///
/// The mixture models call this on `½(ν − j)`, and `ν > d − 1` is exactly the
/// constraint sklearn's `degrees_of_freedom_prior` check enforces — so the
/// argument is positive but can be arbitrarily CLOSE to zero, where `ψ` has a
/// `−1/x` pole. The recurrence handles that correctly (the `−1/x` term IS the
/// pole), which is why there is no small-argument rational fit here.
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        // Reflection: ψ(1 − x) − ψ(x) = π·cot(πx).
        return digamma(1.0 - x) - std::f64::consts::PI / (std::f64::consts::PI * x).tan();
    }
    let mut acc = 0.0;
    let mut z = x;
    while z < DIGAMMA_ASYMPTOTIC_MIN {
        acc -= 1.0 / z;
        z += 1.0;
    }
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    // ln z − 1/(2z) − Σ B₂ₙ/(2n z^2n) for n = 1..6, by Horner in 1/z².
    let series = inv2
        * (1.0 / 12.0
            + inv2
                * (-1.0 / 120.0
                    + inv2
                        * (1.0 / 252.0
                            + inv2
                                * (-1.0 / 240.0
                                    + inv2 * (1.0 / 132.0 + inv2 * (-691.0 / 32_760.0))))));
    acc + z.ln() - 0.5 * inv - series
}

/// `ln B(a, b) = lnΓ(a) + lnΓ(b) − lnΓ(a + b)` — scipy's `betaln`, for
/// `a, b > 0`.
///
/// The naive three-term form is used deliberately. The cancellation it is
/// famous for only bites when `a + b` is enormous relative to the result, and
/// the caller here is the Dirichlet-process log-normalizer, where
/// `a + b ≈ n_samples + prior`: at `n = 10⁶` the three terms are `~10⁷` and the
/// result `~10⁶`, so the lost digits are `log₁₀(10⁷·2⁻⁵³ / 10⁶) ≈ −15` — still
/// below the 1e-5 oracle band by nine orders of magnitude. A continued-fraction
/// form would buy precision this caller cannot spend.
pub fn betaln(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}
