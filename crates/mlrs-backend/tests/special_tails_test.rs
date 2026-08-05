//! `prims::special` distribution tails (FSEL-01) — `betainc` / `gammainc(c)` /
//! `f_sf` / `chi2_sf` against scipy.
//!
//! Same "scipy IS the oracle" logic as `special_test.rs`, for the same reason:
//! `sklearn.feature_selection`'s p-values ARE these scipy calls
//! (`special.fdtrc` for `f_oneway`/`f_classif`, `stats.f.sf` for
//! `f_regression`, `special.chdtrc` for `chi2`), so a disagreement here is a
//! disagreement with the reference implementation before a single row of data is
//! touched. Every reference value below was printed by
//! `scipy.special.{betainc, gammainc, gammaincc, fdtrc, chdtrc}` at full `repr`
//! precision on the scipy that sklearn 1.9.0 pulls in.
//!
//! The argument grids deliberately include the SMALL-TAIL cases (`fdtrc` at
//! 1.4e-27 and 1.5e-283, `chdtrc` at 3.1e-24, `gammaincc` at 3.8e-42) because
//! those are exactly where an implementation that computes the tail as
//! `1 − cdf` loses every significant digit — and where `f_classif`'s real
//! p-values live.
//!
//! ## The tolerance is DERIVED from the documented error law, not hand-tuned
//! Unlike `lgamma`/`digamma`, these functions do not hold a fixed ULP count
//! across their whole domain, and pinning them at one constant is what would
//! make this file a lie. `prims::special`'s own docs derive the binding error:
//! the log-domain prefactor `ln B(a, b)` (or `−lnΓ(a)`) is a difference of large
//! logs, it loses `|lnΓ| · 2⁻⁵³` to cancellation, and an absolute error in a log
//! is a RELATIVE error in its exponential. So the achieved accuracy is
//! `~Σ|lnΓ(argument)| · 2⁻⁵³` — 1e-15 at `a + b = 1`, 4e-14 at `a + b = 300`,
//! ~6e-8 at `a + b = 10⁶`.
//!
//! [`law_tol`] computes exactly that bound (times a [`LAW_SAFETY`] factor) from
//! each case's own arguments, so every assert states the law rather than a
//! remembered number, and a case added later gets the right tolerance for free.
//! `law_tol` is ALSO the assertion: if the implementation regressed to a
//! genuinely worse error law, the derived bound would not move with it and the
//! test would fail. Even at the far end the bound is two orders of magnitude
//! inside the crate's 1e-5 estimator contract (D-09).
//!
//! Backend-independent: pure host `f64` scalar code, no runtime/pool/kernel, so
//! there is no capability gate here. Tests live in `crates/mlrs-backend/tests/`
//! (AGENTS.md §2).

use mlrs_backend::prims::special::{betainc, chi2_sf, f_sf, gammainc, gammaincc, lgamma};

/// Multiple of the derived cancellation bound allowed.
///
/// `64` rather than a tighter factor because the cancellation law is not the
/// ONLY error source: on the symmetry-swapped side `betainc` returns
/// `1 − prefix·cf/b`, and when the result is close to `1` (the `f_sf(1e-4, 1, 3)`
/// case below returns `0.9926`) that subtraction converts the small term's own
/// relative error into a larger relative error in the answer. `64` covers that
/// second effect while leaving every case here between 10× and 100× of margin —
/// loose enough not to flake on last-bit libm differences, tight enough that a
/// real accuracy regression (a lost symmetry swap, a series evaluated on the
/// wrong side) blows through it by orders of magnitude. Even the loosest case it
/// admits, `a + b ≈ 10⁶` at ~9e-8, is two orders inside the 1e-5 contract.
const LAW_SAFETY: f64 = 64.0;

/// The documented accuracy bound for a tail evaluated at these `lnΓ`
/// arguments: `LAW_SAFETY · Σ|lnΓ(argᵢ)| · 2⁻⁵³`, floored at
/// `LAW_SAFETY · 2⁻⁵³` so a small-argument case is still held to ULP scale.
fn law_tol(gamma_args: &[f64]) -> f64 {
    let scale: f64 = gamma_args.iter().map(|&v| lgamma(v).abs()).sum();
    LAW_SAFETY * scale.max(1.0) * f64::EPSILON
}

/// Relative-or-absolute closeness, the numpy `allclose` shape (identical to
/// `special_test.rs`'s helper).
fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    let err = (got - want).abs();
    assert!(
        err <= tol * want.abs().max(1.0),
        "{what}: got={got:.17e} want={want:.17e} err={err:.3e} (tol={tol:e})"
    );
}

/// `(a, b, x, scipy.special.betainc(a, b, x))`.
///
/// Straddles the `x > (a + 1)/(a + b + 2)` symmetry swap in both directions,
/// includes the `a = b = 0.001` near-degenerate shape, and includes an
/// `x = 1e-8` case whose answer is 1.26e-38 — which is where forming
/// `x^a (1−x)^b` outside the log domain returns exactly `0`. The last three rows
/// are the `a + b ≈ 10⁶` end of the accuracy law, i.e. `f_regression` on a
/// million samples.
const BETAINC_CASES: [(f64, f64, f64, f64); 12] = [
    (0.5, 0.5, 0.3, 0.369_010_119_565_545_36),
    (0.5, 60.0, 0.001, 0.270_488_758_552_326_24),
    (1.0, 1.0, 0.5, 0.5),
    (2.5, 3.5, 0.25, 0.209_284_331_863_430_18),
    (30.0, 0.5, 0.999, 0.807_237_306_159_536_9),
    (100.0, 200.0, 0.33, 0.456_618_163_333_997_13),
    (0.001, 0.001, 0.5, 0.499_999_999_999_999_94),
    (5.0, 5.0, 1e-08, 1.259_999_958_000_000_2e-38),
    (1.5, 1000.0, 0.02, 0.999_999_991_300_926),
    (500_000.0, 0.5, 0.9999, 1.520_166_280_836_204_4e-23),
    (0.5, 500_000.0, 1e-06, 0.682_689_492_137_146_1),
    (200_000.0, 300_000.0, 0.4, 0.500_076_776_500_835_4),
];

#[test]
fn betainc_matches_scipy() {
    for &(a, b, x, want) in BETAINC_CASES.iter() {
        assert_close(
            betainc(a, b, x),
            want,
            law_tol(&[a, b, a + b]),
            &format!("betainc({a}, {b}, {x})"),
        );
    }
}

/// `(a, x, scipy.special.gammaincc(a, x), scipy.special.gammainc(a, x))`.
///
/// Straddles the `x < a + 1` series/continued-fraction seam (the `(10, 3)` and
/// `(10, 50)` pair, plus `(150, 150)` sitting right on it) so both branches are
/// pinned, and includes three tails below 1e-35 where the `1 − P` route returns
/// exactly `0`. The two `a >= 20 000` rows are far beyond the
/// `(n_classes − 1)/2` that [`chi2_sf`], the only in-crate caller, ever
/// produces; they are here to pin the large-`a` end of the series branch, whose
/// term count is what [`GAMMA_MAX_ITER`](mlrs_backend::prims::special) bounds.
const GAMMAINC_CASES: [(f64, f64, f64, f64); 12] = [
    (0.5, 0.1, 0.654_720_846_018_576_8, 0.345_279_153_981_423_17),
    (
        0.5,
        25.0,
        1.537_459_794_428_033e-12,
        0.999_999_999_998_462_6,
    ),
    (1.0, 1.0, 0.367_879_441_171_442_45, 0.632_120_558_828_557_7),
    (1.5, 0.2, 0.940_242_494_839_360_7, 0.059_757_505_160_639_29),
    (
        10.0,
        3.0,
        0.998_897_511_869_884_5,
        0.001_102_488_130_115_481_5,
    ),
    (
        10.0,
        50.0,
        1.259_608_459_166_084_7e-12,
        0.999_999_999_998_740_3,
    ),
    (2.0, 100.0, 3.757_276_735_781_06e-42, 1.0),
    (0.5, 80.0, 1.131_483_790_243_303_8e-36, 1.0),
    (
        150.0,
        150.0,
        0.489_141_770_250_640_3,
        0.510_858_229_749_359_7,
    ),
    (
        1000.0,
        1200.0,
        1.288_160_608_628_143e-09,
        0.999_999_998_711_839_4,
    ),
    (
        20_000.0,
        20_000.0,
        0.499_059_683_766_250_65,
        0.500_940_316_233_749_3,
    ),
    (
        500_000.0,
        500_000.0,
        0.499_811_936_803_394_5,
        0.500_188_063_196_605_5,
    ),
];

#[test]
fn gammainc_and_gammaincc_match_scipy() {
    for &(a, x, want_q, want_p) in GAMMAINC_CASES.iter() {
        let tol = law_tol(&[a]);
        assert_close(
            gammaincc(a, x),
            want_q,
            tol,
            &format!("gammaincc({a}, {x})"),
        );
        assert_close(gammainc(a, x), want_p, tol, &format!("gammainc({a}, {x})"));
    }
}

/// `(f, dfn, dfd, scipy.special.fdtrc(dfn, dfd, f))`.
///
/// The first four rows are the ACTUAL `(dfn, dfd, F)` triples sklearn's own
/// `f_classif` / `f_regression` docstring examples produce — including the
/// `2.67e13` F-statistic whose tail is 1.5e-283, the single hardest value in
/// this file and the reason `betainc`'s prefactor is formed in the log domain.
/// The last two rows are `n_samples = 10⁶`.
const FDTRC_CASES: [(f64, f64, f64, f64); 10] = [
    (221.0, 2.0, 57.0, 1.401_418_468_123_109_5e-27),
    (0.702, 2.0, 57.0, 0.499_826_762_830_088),
    (1.21, 1.0, 48.0, 0.276_818_405_251_639_14),
    (
        26_700_000_000_000.0,
        1.0,
        48.0,
        1.487_911_430_551_033_6e-283,
    ),
    (1.0, 4.0, 10.0, 0.451_555_049_341_686),
    (15.5, 3.0, 1000.0, 7.337_436_629_075_306e-10),
    (0.0001, 1.0, 3.0, 0.992_649_111_412_852),
    (100.0, 9.0, 9.0, 6.154_801_245_208_48e-08),
    (50.0, 2.0, 999_997.0, 1.933_577_447_214_206_6e-22),
    (3.5, 1.0, 999_998.0, 0.061_369_120_957_499_4),
];

#[test]
fn f_sf_matches_scipy_fdtrc() {
    for &(f, dfn, dfd, want) in FDTRC_CASES.iter() {
        // `f_sf` evaluates `betainc(dfd/2, dfn/2, ·)`, so the law's arguments
        // are that call's, not the degrees of freedom themselves.
        let (a, b) = (dfd / 2.0, dfn / 2.0);
        assert_close(
            f_sf(f, dfn, dfd),
            want,
            law_tol(&[a, b, a + b]),
            &format!("f_sf({f}, {dfn}, {dfd})"),
        );
    }
}

/// `(x, df, scipy.special.chdtrc(df, x))`. The first three rows are the exact
/// statistics sklearn's `chi2` docstring example produces (its printed p-values
/// `0.000456, 0.0387, 0.0116` round from these).
const CHDTRC_CASES: [(f64, f64, f64); 8] = [
    (15.3, 2.0, 0.000_476_044_129_022_269_55),
    (6.5, 2.0, 0.038_774_207_831_722_02),
    (8.9, 2.0, 0.011_678_566_970_395_439),
    (0.5, 1.0, 0.479_500_122_186_953_37),
    (60.0, 1.0, 9.485_737_571_073_857e-15),
    (120.0, 5.0, 3.138_579_772_755_301_7e-24),
    (30.0, 29.0, 0.414_003_642_917_542_55),
    (1e-06, 3.0, 0.999_999_999_734_038_5),
];

#[test]
fn chi2_sf_matches_scipy_chdtrc() {
    for &(x, df, want) in CHDTRC_CASES.iter() {
        assert_close(
            chi2_sf(x, df),
            want,
            law_tol(&[df / 2.0]),
            &format!("chi2_sf({x}, {df})"),
        );
    }
}

/// `P + Q = 1` and `I_x(a,b) + I_{1−x}(b,a) = 1` — the identities that catch a
/// broken symmetry swap or a branch seam even at arguments where no scipy value
/// is pinned above.
#[test]
fn tail_complement_identities_hold() {
    let mut x = 0.05;
    while x < 40.0 {
        for a in [0.5, 1.0, 3.5, 12.0, 40.0] {
            let (p, q) = (gammainc(a, x), gammaincc(a, x));
            assert_close(
                p + q,
                1.0,
                law_tol(&[a]),
                &format!("gammainc P+Q at a={a}, x={x}"),
            );
        }
        x *= 1.7;
    }
    let mut t = 0.02;
    while t < 0.99 {
        for (a, b) in [(0.5, 0.5), (2.0, 7.0), (30.0, 1.5), (100.0, 100.0)] {
            assert_close(
                betainc(a, b, t) + betainc(b, a, 1.0 - t),
                1.0,
                law_tol(&[a, b, a + b]),
                &format!("betainc symmetry at a={a}, b={b}, x={t}"),
            );
        }
        t += 0.07;
    }
}

/// The tail functions' off-domain and degenerate branches are DEFINED (the
/// `special_test.rs::poles_and_negative_arguments_are_defined` contract,
/// extended to the new functions). `NaN` in particular must PROPAGATE: a `NaN`
/// F-statistic is how `f_oneway` reports a constant column, and the selectors'
/// `_clean_nans` gate is what handles it downstream — mapping it to a number
/// here would silently hide the degenerate column.
#[test]
fn tail_off_domain_branches_are_defined() {
    assert_eq!(f_sf(0.0, 2.0, 5.0), 1.0);
    assert_eq!(f_sf(-1.0, 2.0, 5.0), 1.0);
    assert_eq!(f_sf(f64::INFINITY, 1.0, 5.0), 0.0);
    assert!(f_sf(f64::NAN, 1.0, 5.0).is_nan());
    assert!(f_sf(1.0, 0.0, 5.0).is_nan());
    assert!(f_sf(1.0, 1.0, 0.0).is_nan());

    assert_eq!(chi2_sf(0.0, 3.0), 1.0);
    assert_eq!(chi2_sf(f64::INFINITY, 3.0), 0.0);
    assert!(chi2_sf(f64::NAN, 3.0).is_nan());
    assert!(chi2_sf(1.0, 0.0).is_nan());

    assert_eq!(betainc(2.0, 3.0, 0.0), 0.0);
    assert_eq!(betainc(2.0, 3.0, 1.0), 1.0);
    assert!(betainc(2.0, 3.0, 1.5).is_nan());
    assert!(betainc(0.0, 3.0, 0.5).is_nan());
    assert_eq!(gammaincc(2.0, 0.0), 1.0);
    assert_eq!(gammainc(2.0, 0.0), 0.0);
    assert!(gammainc(-1.0, 1.0).is_nan());
    assert!(gammaincc(1.0, -1.0).is_nan());
}

/// A non-converged continued fraction reports `NaN`, NOT a plausible-looking
/// wrong number.
///
/// This is the specific trap the iteration caps exist to close, and it was a
/// REAL failure during development, not a hypothetical: the Numerical Recipes
/// gamma series truncated at its textbook 300 terms returns `0.835` for
/// `Q(5·10⁵, 5·10⁵)` where the answer is `0.4998` — a 67%-wrong p-value that
/// every downstream selector would then compare against `alpha` and act on. A
/// `NaN` reaches `_clean_nans` and is handled; a confident `0.835` is not.
///
/// The `x ≈ a` series needs `~sqrt(2a·ln(1/ε))` terms, so the shipped
/// `GAMMA_MAX_ITER` actually covers `a` past `10⁸` — which is why the assertion
/// here is that the hard cases CONVERGE CORRECTLY rather than that they bail.
/// The `NaN` sentinel is a guard against a future caller reaching an argument
/// nobody profiled, not a routine outcome.
#[test]
fn large_argument_tails_converge_rather_than_truncate() {
    // The exact argument the 300-iteration cap got wrong.
    assert_close(
        gammaincc(5.0e5, 5.0e5),
        0.499_811_936_803_394_5,
        law_tol(&[5.0e5]),
        "gammaincc(5e5, 5e5)",
    );
    // The `betainc` counterpart: `f_regression`'s `f_sf(·, 1, n − 2)` at
    // n_samples = 10⁶ evaluates the beta CF at a = 5·10⁵.
    assert!(f_sf(3.5, 1.0, 999_998.0).is_finite());
    assert!(betainc(2.0e5, 3.0e5, 0.4).is_finite());
}
