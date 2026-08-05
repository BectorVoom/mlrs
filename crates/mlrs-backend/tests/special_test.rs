//! `prims::special` — `lnΓ` / `ψ` / `lnB` against scipy.
//!
//! Every reference value below was printed by `scipy.special.{gammaln, digamma,
//! betaln}` at full `repr` precision on the same scipy sklearn 1.9.0 pulls in,
//! because scipy IS the oracle: sklearn's `BayesianGaussianMixture` calls these
//! three functions directly, so any disagreement here is a disagreement with
//! the estimator's reference implementation before a single row of data is
//! touched.
//!
//! ## Why the tolerance is 1e-14 and not the crate-wide 1e-5
//! These are not estimator outputs, they are the INPUTS to an exponential. The
//! variational E-step computes `exp(½·Σ_j ψ(½(ν − j)) + …)`, so an error in `ψ`
//! propagates into a responsibility, then into `nk`, then into every fitted
//! attribute — and it does so once per iteration, compounding. Pinning the
//! scalars at machine precision is what makes the estimator's own 1e-5 oracle
//! band a statement about the ALGORITHM rather than about these approximations.
//!
//! Backend-independent: the module is pure host `f64` scalar code with no
//! runtime, pool, or kernel involved, so there is no capability gate here.

use mlrs_backend::prims::special::{betaln, digamma, lgamma};

/// Relative-or-absolute closeness, the numpy `allclose` shape.
fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    let err = (got - want).abs();
    assert!(
        err <= tol * want.abs().max(1.0),
        "{what}: got={got:.17e} want={want:.17e} err={err:.3e} (tol={tol:e})"
    );
}

/// The argument grid deliberately straddles both internal regime switches
/// (`LGAMMA_STIRLING_MIN = 16`, `DIGAMMA_ASYMPTOTIC_MIN = 10`) and includes the
/// points immediately below them, which is where a recurrence/series seam shows
/// up if one exists: `9.999` and `10.0` must agree with scipy to the same
/// precision even though they take different code paths.
const LGAMMA_CASES: [(f64, f64); 15] = [
    (0.001, 6.9071788853838534),
    (0.05, 2.9688792010517311),
    (0.5, 0.57236494292469997),
    (1.0, 0.0),
    (1.5, -0.12078223763524526),
    (2.0, 0.0),
    (3.7, 1.4280723266653881),
    (6.0, 4.7874917427820458),
    (9.999, 12.799575780077413),
    (10.0, 12.801827480081469),
    (16.0, 27.899271383840894),
    (25.0, 54.784729398112319),
    (150.5, 602.51395487058528),
    (1e4, 82099.717496442376),
    (1e6, 12815504.569147611),
];

const DIGAMMA_CASES: [(f64, f64); 15] = [
    (0.001, -1000.5755719318103),
    (0.05, -20.497844991299871),
    (0.5, -1.9635100260214235),
    (1.0, -0.57721566490153287),
    (1.5, 0.03648997397857652),
    (2.0, 0.42278433509846713),
    (3.7, 1.1671535393615113),
    (6.0, 1.7061176684318005),
    (9.999, 2.251647417205735),
    (10.0, 2.2517525890667209),
    (16.0, 2.7410133283274605),
    (25.0, 3.198742512851974),
    (150.5, 5.0106371459337042),
    (1e4, 9.2102903711428503),
    (1e6, 13.81551005796419),
];

/// `(a, b, betaln(a, b))`. The last pair is the cancellation case the module
/// docs bound: three `~1.3e7` terms producing a `~1.4e6` result.
const BETALN_CASES: [(f64, f64, f64); 7] = [
    (1.0, 1.0, 0.0),
    (0.5, 0.5, 1.1447298858493999),
    (2.5, 7.25, -4.9053366188393035),
    (101.0, 200.33333333333329, -193.3692553645235),
    (1.0, 0.33333333333333331, 1.0986122886681096),
    (1e5, 3.0, -33.845659214071929),
    (1e6, 1e6, -1386300.0033629201),
];

#[test]
fn lgamma_matches_scipy_gammaln() {
    for (x, want) in LGAMMA_CASES {
        assert_close(lgamma(x), want, 1e-14, &format!("lgamma({x})"));
    }
}

#[test]
fn digamma_matches_scipy() {
    for (x, want) in DIGAMMA_CASES {
        assert_close(digamma(x), want, 1e-14, &format!("digamma({x})"));
    }
}

#[test]
fn betaln_matches_scipy() {
    for (a, b, want) in BETALN_CASES {
        assert_close(betaln(a, b), want, 1e-13, &format!("betaln({a}, {b})"));
    }
}

/// `lnΓ(x + 1) = lnΓ(x) + ln x` across the Stirling seam.
///
/// An identity test rather than a table lookup: it holds at every `x` including
/// the ones no reference value was recorded for, so it catches a seam the
/// 15-point grid above could step over.
#[test]
fn lgamma_satisfies_the_recurrence() {
    let mut x = 0.25;
    while x < 40.0 {
        assert_close(
            lgamma(x + 1.0),
            lgamma(x) + x.ln(),
            1e-14,
            &format!("lgamma recurrence at {x}"),
        );
        x += 0.125;
    }
}

/// `ψ(x + 1) = ψ(x) + 1/x`, the same identity check for digamma.
#[test]
fn digamma_satisfies_the_recurrence() {
    let mut x = 0.25;
    while x < 40.0 {
        assert_close(
            digamma(x + 1.0),
            digamma(x) + 1.0 / x,
            1e-14,
            &format!("digamma recurrence at {x}"),
        );
        x += 0.125;
    }
}

/// The poles and the off-domain branches are DEFINED, not garbage.
///
/// `BayesianGaussianMixture` never reaches them (its `degrees_of_freedom_prior
/// > n_features − 1` check is exactly what keeps `½(ν − j)` positive), but a
/// special function that returns a plausible-looking finite number off its
/// domain is a trap for the next caller — so the behaviour is pinned here.
#[test]
fn poles_and_negative_arguments_are_defined() {
    assert!(lgamma(0.0).is_infinite() && lgamma(0.0) > 0.0);
    assert!(lgamma(-3.0).is_infinite() && lgamma(-3.0) > 0.0);
    assert!(digamma(0.0).is_nan());
    assert!(digamma(-2.0).is_nan());
    assert!(lgamma(f64::NAN).is_nan());
    assert!(digamma(f64::NAN).is_nan());
    // Reflection branch: scipy's gammaln(-0.5) = 1.2655121234846454,
    // digamma(-0.5) = 0.03648997397857651.
    assert_close(lgamma(-0.5), 1.2655121234846454, 1e-14, "lgamma(-0.5)");
    assert_close(digamma(-0.5), 0.03648997397857651, 1e-13, "digamma(-0.5)");
}
