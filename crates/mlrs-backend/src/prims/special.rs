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

// ===========================================================================
// Distribution tails — the univariate feature-selection p-values (FSEL-01)
// ===========================================================================
//
// `sklearn.feature_selection` turns every univariate statistic into a p-value
// through a scipy distribution tail:
//
//   * `f_oneway` / `f_classif` → `scipy.special.fdtrc(dfn, dfd, f)`
//   * `f_regression`           → `scipy.stats.f.sf(f, 1, dfd)` (the same tail)
//   * `chi2`                   → `scipy.special.chdtrc(k − 1, chisq)`
//
// Both reduce to ONE of two incomplete special functions:
//
//   fdtrc(a, b, x) = I_{b/(b + a·x)}(b/2, a/2)   — regularized incomplete beta
//   chdtrc(v, x)   = Q(v/2, x/2)                 — regularized upper incomplete
//                                                  gamma
//
// so [`betainc`] and [`gammaincc`] below are the whole implementation, and
// [`f_sf`] / [`chi2_sf`] are thin wrappers naming the statistical intent.
//
// ## Why these are held to machine precision, like `lgamma`/`digamma` above
// The oracle contract is abs-AND-rel `1e-5` (D-09), and a p-value's RELATIVE
// error is what that contract measures. `f_classif` on a genuinely informative
// feature produces p-values around `1e-27` (sklearn's own docstring example
// prints `7.14e-27`); a tail approximation good to `1e-8` ABSOLUTE is useless
// there — it would report `0.0` and miss the relative band by 19 orders of
// magnitude. The continued-fraction forms below hold a small MULTIPLE of the
// result across the whole domain, which is what makes the `1e-5` estimator band
// a statement about the selection algorithm rather than about these
// approximations.
//
// ## The accuracy that is actually achieved, and what bounds it
// The continued fractions themselves converge to `CF_EPS` (~1.35 ULP). The
// binding error is the LOG-DOMAIN PREFACTOR both functions share: `betainc`
// needs `ln B(a, b) = lnΓ(a) + lnΓ(b) − lnΓ(a + b)` and `gammainc` needs
// `−lnΓ(a)`, and an absolute error `δ` in a log becomes a RELATIVE error `δ` in
// its exponential. `betaln`'s three-term form loses
// `|lnΓ(a)| + |lnΓ(b)| ≈ a·ln a + b·ln b` worth of absolute precision to
// cancellation, so the achieved relative accuracy is
//
//     ~ (a·ln a + b·ln b) · 2⁻⁵³
//
// — 4e-14 at `a + b = 300`, ~7e-10 at `a + b = 10⁶` (i.e. `f_classif` on a
// million samples). `special_tails_test.rs` pins both regimes against scipy at
// tolerances following that law. Tightening it would mean a Boost-style stable
// `lbeta` (a log-gamma-RATIO evaluation rather than a difference of three large
// logs); at seven orders of magnitude inside the estimator contract, that is
// precision this caller cannot spend — the same trade `betaln`'s own docs
// record for the Dirichlet-process log-normalizer.
//
// ## Why host scalars, not device kernels
// The same reasoning `lgamma` carries: this is `O(n_features)` scalar work
// against the score's own `O(n_samples · n_features)`, it is branch-heavy
// (continued fractions with data-dependent iteration counts), and it must run
// in `f64` regardless of the estimator's `F` — which cubecl's cuda backend does
// not even advertise (`supports_type(F64)` is false there).

/// Relative-accuracy target for the continued fractions below. `3e-16` is
/// ~1.35 ULP of an `f64` mantissa, i.e. about the tightest bound that still
/// terminates for every argument.
const CF_EPS: f64 = 3.0e-16;

/// Smallest safe positive denominator, used to keep a continued-fraction
/// denominator away from an exact zero (modified-Lentz's one failure mode).
const CF_TINY: f64 = 1.0e-300;

/// Iteration cap for the incomplete-BETA continued fraction.
///
/// [`beta_cf`] converges geometrically once [`betainc`]'s symmetry swap has put
/// it on the right side, but the rate degrades as `min(a, b)` grows: the
/// `f_sf(_, 1, n − 2)` call `f_regression` makes on a million samples evaluates
/// it at `a = 5·10⁵`, which needs a few thousand iterations. `100_000` clears
/// every `(a, b)` this crate can reach by two orders of magnitude while still
/// bounding a pathological argument.
const BETA_CF_MAX_ITER: usize = 100_000;

/// Iteration cap for the incomplete-GAMMA series and continued fraction.
///
/// The series ([`gamma_series`], used for `x < a + 1`) has term ratio
/// `x/(a + n)`, so at the worst case `x ≈ a` its terms decay like
/// `exp(−n²/2a)` and it needs `~sqrt(2a·ln(1/CF_EPS))` of them — `~6·10³` at
/// `a = 5·10⁵`. `250_000` therefore covers `a` past `10⁸`, which is many orders
/// beyond the only argument this crate's own caller produces (`chi2_sf`'s
/// `a = df/2 = (n_classes − 1)/2`).
///
/// The textbook Numerical Recipes cap of 300 does NOT: it silently returns
/// `0.835` for `Q(5·10⁵, 5·10⁵)` where the answer is `0.4998`, which is why
/// [`CF_DIVERGED`] exists and why this constant is three orders larger.
const GAMMA_MAX_ITER: usize = 250_000;

/// Sentinel written by the two kernels when they exhaust their iteration cap
/// without meeting [`CF_EPS`]. Propagated as `NaN` by the public wrappers — a
/// non-converged continued fraction returns a number that LOOKS plausible (the
/// `a = x = 5·10⁵` gamma series truncated at 300 terms returns `0.835` where the
/// answer is `0.4998`), and a silently-wrong p-value is strictly worse than a
/// missing one, because every downstream selector compares it to `alpha`.
const CF_DIVERGED: f64 = f64::NAN;

/// `I_x(a, b)` — the REGULARIZED incomplete beta function, scipy's `betainc`,
/// for `a, b > 0` and `x ∈ [0, 1]`.
///
/// Continued fraction (modified Lentz), with the standard
/// `x > (a + 1)/(a + b + 2)` symmetry swap `I_x(a, b) = 1 − I_{1−x}(b, a)` so
/// the fraction is only ever evaluated on its rapidly-converging side. Returns
/// `NaN` for an out-of-domain argument rather than a plausible-looking number.
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    // The prefactor `x^a (1−x)^b / B(a, b)` is formed in the LOG domain:
    // `f_classif` reaches `a = n/2` with `n` in the thousands, where
    // `x^a` underflows f64 long before the ratio it appears in does.
    let log_prefix = a * x.ln() + b * (-x).ln_1p() - betaln(a, b);
    if x < (a + 1.0) / (a + b + 2.0) {
        log_prefix.exp() * beta_cf(a, b, x) / a
    } else {
        1.0 - log_prefix.exp() * beta_cf(b, a, 1.0 - x) / b
    }
}

/// The modified-Lentz continued fraction for `I_x(a, b)`'s core (Numerical
/// Recipes §6.4 `betacf`). Only ever called on the converging side, by
/// [`betainc`]. Returns [`CF_DIVERGED`] if [`BETA_CF_MAX_ITER`] is exhausted
/// without reaching [`CF_EPS`].
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < CF_TINY {
        d = CF_TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    let mut converged = false;
    for m in 1..=BETA_CF_MAX_ITER {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        // Even step.
        let mut aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < CF_TINY {
            d = CF_TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < CF_TINY {
            c = CF_TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < CF_TINY {
            d = CF_TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < CF_TINY {
            c = CF_TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < CF_EPS {
            converged = true;
            break;
        }
    }
    if converged {
        h
    } else {
        CF_DIVERGED
    }
}

/// `P(a, x)` — the REGULARIZED LOWER incomplete gamma function, scipy's
/// `gammainc`, for `a > 0` and `x >= 0`.
///
/// Converges for `a` past `10⁸` (see [`GAMMA_MAX_ITER`]); should an argument
/// beyond that ever be reached, this returns `NaN` rather than a
/// plausible-looking wrong answer. The only caller in this crate is
/// [`chi2_sf`], whose `a` is `(n_classes − 1)/2`.
pub fn gammainc(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// `Q(a, x) = 1 − P(a, x)` — the REGULARIZED UPPER incomplete gamma function,
/// scipy's `gammaincc`, for `a > 0` and `x >= 0`.
///
/// The branch is chosen so the SMALL quantity is always the one computed
/// directly: for `x >= a + 1` the continued fraction gives `Q` itself, and the
/// `1 − P` subtraction — which would cancel away every significant digit of a
/// `1e-30` tail — never happens on that side. That is the whole reason `chi2`'s
/// small p-values survive the RELATIVE half of the oracle band.
///
/// Same convergence domain as [`gammainc`], and the same
/// `NaN`-rather-than-wrong behaviour beyond it.
pub fn gammaincc(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cf(a, x)
    }
}

/// Series representation of `P(a, x)`, converging for `x < a + 1` (Numerical
/// Recipes §6.2 `gser`). Returns [`CF_DIVERGED`] if [`GAMMA_MAX_ITER`] is
/// exhausted — the term ratio is `x/(a + n)`, so the term count grows like `x`
/// itself when `x ≈ a`, and this is the branch that sets the documented `a`
/// domain.
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    let mut converged = false;
    for _ in 0..GAMMA_MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * CF_EPS {
            converged = true;
            break;
        }
    }
    if !converged {
        return CF_DIVERGED;
    }
    // `exp(−x + a·ln x − lnΓ(a))` in the log domain, for the same underflow
    // reason `betainc` forms its prefactor there.
    sum * (-x + a * x.ln() - lgamma(a)).exp()
}

/// Modified-Lentz continued fraction for `Q(a, x)`, converging for
/// `x >= a + 1` (Numerical Recipes §6.2 `gcf`). Returns [`CF_DIVERGED`] on
/// cap exhaustion.
fn gamma_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / CF_TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut converged = false;
    for i in 1..=GAMMA_MAX_ITER {
        let i_f = i as f64;
        let an = -i_f * (i_f - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < CF_TINY {
            d = CF_TINY;
        }
        c = b + an / c;
        if c.abs() < CF_TINY {
            c = CF_TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < CF_EPS {
            converged = true;
            break;
        }
    }
    if !converged {
        return CF_DIVERGED;
    }
    h * (-x + a * x.ln() - lgamma(a)).exp()
}

/// The F-distribution SURVIVAL function `P(F > f)` with `(dfn, dfd)` degrees of
/// freedom — scipy's `special.fdtrc(dfn, dfd, f)` / `stats.f.sf(f, dfn, dfd)`,
/// which is what `f_oneway` / `f_classif` / `f_regression` call.
///
/// `fdtrc(a, b, x) = I_{b/(b + a·x)}(b/2, a/2)`. A non-positive statistic
/// returns `1.0` and `+∞` returns `0.0` (both the exact tails, and `0.0` is
/// also what `f_regression`'s `force_finite` branch writes for a perfectly
/// correlated feature); `NaN` propagates, because a `NaN` F-statistic is how
/// `f_oneway` reports a constant column and the selectors' `_clean_nans` gate
/// is what handles it downstream.
pub fn f_sf(f: f64, dfn: f64, dfd: f64) -> f64 {
    if f.is_nan() || dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    if f.is_infinite() {
        return 0.0;
    }
    betainc(dfd / 2.0, dfn / 2.0, dfd / (dfd + dfn * f))
}

/// The chi-squared SURVIVAL function `P(χ² > x)` with `df` degrees of freedom —
/// scipy's `special.chdtrc(df, x)`, which is what `chi2` calls.
///
/// `chdtrc(v, x) = Q(v/2, x/2)`. `df <= 0` returns `NaN` (scipy's own
/// out-of-domain answer), and a `NaN` statistic propagates for the same
/// `_clean_nans` reason [`f_sf`] documents — `chi2` on an all-zero feature
/// column divides `0/0`.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if x.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    gammaincc(df / 2.0, x / 2.0)
}
