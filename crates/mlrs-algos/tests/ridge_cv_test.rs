//! `RidgeCV` (RIDGECV-01) — correctness gates for the two engines in
//! `mlrs_algos::linear::ridge_cv`.
//!
//! These do NOT compare against a stored sklearn fixture, and that is
//! deliberate. The whole claim of the GCV engine is an IDENTITY — that
//! `c / diag(G⁻¹)` is the leave-one-out residual of a ridge fit with an
//! unpenalized intercept — and the way to test an identity is to compute the
//! other side of it. So the gates here refit `n` times, leaving one row out
//! each time, through a COMPLETELY DIFFERENT code path (weighted-centered
//! normal equations + Cholesky, `ridge_solvers::cholesky_ridge`) and demand the
//! closed form reproduce it. A fixture would only ever prove that mlrs still
//! agrees with whatever sklearn did on the day it was generated; this proves
//! the math.
//!
//! The live sklearn comparison — including every string-valued parameter — is
//! `crates/mlrs-py/python/tests/test_oracle_ridge_cv_params.py`.
//!
//! ```text
//! cargo test -p mlrs-algos --features cpu --test ridge_cv_test
//! ```
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use mlrs_algos::linear::ridge_cv::{
    resolve_route, ridge_cv_grid, ridge_gcv, GcvMode, GcvRoute,
};
use mlrs_algos::linear::ridge_solvers::cholesky_ridge;
use mlrs_backend::prims::gram_host::centered_gram_multi_xty;

/// Counter-based splitmix64 — the workspace's deterministic design source.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform_pm1(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A well-conditioned `n × d` design and an `n × n_y` target.
fn design(n: usize, d: usize, n_y: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut sx = seed;
    let x: Vec<f64> = (0..n * d).map(|_| uniform_pm1(&mut sx)).collect();
    let mut sc = seed + 1;
    let coef: Vec<f64> = (0..d * n_y).map(|_| uniform_pm1(&mut sc)).collect();
    let mut sn = seed + 2;
    let mut y = vec![0.0f64; n * n_y];
    for r in 0..n {
        for t in 0..n_y {
            let mut acc = 1.5 + 0.05 * uniform_pm1(&mut sn);
            for c in 0..d {
                acc += x[r * d + c] * coef[c * n_y + t];
            }
            y[r * n_y + t] = acc;
        }
    }
    (x, y)
}

/// A single-target ridge fit through the OTHER path: weighted-centered normal
/// equations, Cholesky, then the center-then-solve intercept. Returns
/// `(coef, intercept)`.
fn ridge_normal_equations(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    sw: Option<&[f64]>,
    alpha: f64,
    fit_intercept: bool,
) -> (Vec<f64>, f64) {
    let (x_mean, y_mean, gram, xty) =
        centered_gram_multi_xty::<f64>(x, y, n, d, 1, sw, fit_intercept);
    let coef = cholesky_ridge(&gram, &xty, d, alpha).expect("well-conditioned gram");
    let intercept = if fit_intercept {
        y_mean[0] - x_mean.iter().zip(coef.iter()).map(|(m, c)| m * c).sum::<f64>()
    } else {
        0.0
    };
    (coef, intercept)
}

/// Brute-force leave-one-out predictions: `n` independent refits.
fn brute_force_loo(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    sw: Option<&[f64]>,
    alpha: f64,
    fit_intercept: bool,
) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    for hold in 0..n {
        let mut xk = Vec::with_capacity((n - 1) * d);
        let mut yk = Vec::with_capacity(n - 1);
        let mut wk = Vec::with_capacity(n - 1);
        for i in 0..n {
            if i == hold {
                continue;
            }
            xk.extend_from_slice(&x[i * d..i * d + d]);
            yk.push(y[i]);
            if let Some(w) = sw {
                wk.push(w[i]);
            }
        }
        let swk = if sw.is_some() { Some(&wk[..]) } else { None };
        let (coef, intercept) =
            ridge_normal_equations(&xk, &yk, n - 1, d, swk, alpha, fit_intercept);
        let mut p = intercept;
        for c in 0..d {
            p += x[hold * d + c] * coef[c];
        }
        out[hold] = p;
    }
    out
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

// ---------------------------------------------------------------------------
// The identity: the closed-form LOO equals `n` real refits
// ---------------------------------------------------------------------------

/// The `"cov"` route (`n > d`), unweighted, with the intercept fitted.
#[test]
fn gcv_loo_equals_brute_force_cov_route() {
    let (n, d) = (40usize, 6usize);
    let (x, y) = design(n, d, 1, 7);
    let alphas = [0.05f64, 1.0, 25.0];

    let fit = ridge_gcv::<f64>(
        &x,
        &y,
        n,
        d,
        1,
        None,
        &alphas,
        true,
        GcvMode::Auto,
        /* want_predictions */ true,
        false,
    )
    .expect("gcv fit");
    assert_eq!(fit.route, GcvRoute::Cov);

    for (a, &alpha) in alphas.iter().enumerate() {
        let got: Vec<f64> = (0..n).map(|i| fit.cv_values[i * alphas.len() + a]).collect();
        let want = brute_force_loo(&x, &y, n, d, None, alpha, true);
        let err = max_abs_diff(&got, &want);
        assert!(
            err < 1e-9,
            "alpha={alpha}: closed-form LOO differs from {n} refits by {err:.3e}"
        );
    }
}

/// The `"gram"` route (`n ≤ d`), where the `n × n` Gram is the smaller one.
#[test]
fn gcv_loo_equals_brute_force_gram_route() {
    let (n, d) = (18usize, 30usize);
    let (x, y) = design(n, d, 1, 11);
    let alphas = [0.3f64, 4.0];

    let fit = ridge_gcv::<f64>(
        &x, &y, n, d, 1, None, &alphas, true, GcvMode::Auto, true, false,
    )
    .expect("gcv fit");
    assert_eq!(fit.route, GcvRoute::Gram);

    for (a, &alpha) in alphas.iter().enumerate() {
        let got: Vec<f64> = (0..n).map(|i| fit.cv_values[i * alphas.len() + a]).collect();
        let want = brute_force_loo(&x, &y, n, d, None, alpha, true);
        let err = max_abs_diff(&got, &want);
        assert!(
            err < 1e-8,
            "alpha={alpha}: gram-route LOO differs from {n} refits by {err:.3e}"
        );
    }
}

/// `fit_intercept = false` drops the rank-1 correction from the denominator —
/// the branch most likely to be silently wrong, because the fitted `coef_` is
/// unaffected by it and only the LOO residual moves.
#[test]
fn gcv_loo_equals_brute_force_without_intercept() {
    let (n, d) = (36usize, 5usize);
    let (x, y) = design(n, d, 1, 13);
    let alphas = [0.5f64, 8.0];

    let fit = ridge_gcv::<f64>(
        &x, &y, n, d, 1, None, &alphas, false, GcvMode::Auto, true, false,
    )
    .expect("gcv fit");

    for (a, &alpha) in alphas.iter().enumerate() {
        let got: Vec<f64> = (0..n).map(|i| fit.cv_values[i * alphas.len() + a]).collect();
        let want = brute_force_loo(&x, &y, n, d, None, alpha, false);
        let err = max_abs_diff(&got, &want);
        assert!(
            err < 1e-9,
            "alpha={alpha}: no-intercept LOO differs by {err:.3e}"
        );
    }
}

/// `sample_weight`: the weighted LOO, where the held-out row's weight leaves
/// both the fit AND the weighted centering.
#[test]
fn gcv_loo_equals_brute_force_weighted() {
    let (n, d) = (34usize, 5usize);
    let (x, y) = design(n, d, 1, 17);
    let mut sr = 99u64;
    let sw: Vec<f64> = (0..n).map(|_| 0.4 + 1.6 * (uniform_pm1(&mut sr) + 1.0)).collect();
    let alphas = [0.2f64, 6.0];

    let fit = ridge_gcv::<f64>(
        &x,
        &y,
        n,
        d,
        1,
        Some(&sw),
        &alphas,
        true,
        GcvMode::Auto,
        true,
        false,
    )
    .expect("gcv fit");

    for (a, &alpha) in alphas.iter().enumerate() {
        let got: Vec<f64> = (0..n).map(|i| fit.cv_values[i * alphas.len() + a]).collect();
        let want = brute_force_loo(&x, &y, n, d, Some(&sw), alpha, true);
        let err = max_abs_diff(&got, &want);
        assert!(
            err < 1e-8,
            "alpha={alpha}: weighted LOO differs by {err:.3e}"
        );
    }
}

// ---------------------------------------------------------------------------
// The fitted coefficients
// ---------------------------------------------------------------------------

/// Every alpha's `coefs` block must be the ridge solution at that alpha —
/// checked against the normal-equations path, not against itself.
#[test]
fn gcv_coefs_match_the_normal_equations() {
    for (n, d, route) in [(60usize, 8usize, GcvRoute::Cov), (14, 20, GcvRoute::Gram)] {
        let (x, y) = design(n, d, 1, 23);
        let alphas = [0.1f64, 2.0, 50.0];
        let fit = ridge_gcv::<f64>(
            &x, &y, n, d, 1, None, &alphas, true, GcvMode::Auto, false, false,
        )
        .expect("gcv fit");
        assert_eq!(fit.route, route);

        for (a, &alpha) in alphas.iter().enumerate() {
            let (want, want_b) = ridge_normal_equations(&x, &y, n, d, None, alpha, true);
            let got = &fit.coefs[a * d..a * d + d];
            let err = max_abs_diff(got, &want);
            assert!(
                err < 1e-9,
                "n={n} d={d} alpha={alpha}: coefs differ by {err:.3e}"
            );
            let b: f64 = fit.y_offset[0]
                - fit
                    .x_offset
                    .iter()
                    .zip(got.iter())
                    .map(|(m, c)| m * c)
                    .sum::<f64>();
            assert!(
                (b - want_b).abs() < 1e-9,
                "n={n} d={d} alpha={alpha}: intercept {b} != {want_b}"
            );
        }
    }
}

/// The engine's own `scores` must be `−mean(looe²)` over the values it also
/// reports — a shim reading the wrong axis of either buffer shows up here.
#[test]
fn gcv_scores_are_the_mean_squared_loo_error() {
    let (n, d, n_y) = (50usize, 7usize, 3usize);
    let (x, y) = design(n, d, n_y, 29);
    let alphas = [0.4f64, 3.0, 9.0];
    let fit = ridge_gcv::<f64>(
        &x, &y, n, d, n_y, None, &alphas, true, GcvMode::Auto, false, true,
    )
    .expect("gcv fit");

    for a in 0..alphas.len() {
        for t in 0..n_y {
            let mut acc = 0.0f64;
            for i in 0..n {
                acc += fit.cv_values[i * alphas.len() * n_y + a * n_y + t];
            }
            let want = -acc / n as f64;
            let got = fit.scores[a * n_y + t];
            assert!(
                (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
                "alpha #{a} target #{t}: score {got} != {want}"
            );
        }
    }
}

/// Multi-target `coefs` are per-column independent fits — the property
/// `alpha_per_target` relies on.
#[test]
fn gcv_multi_target_columns_are_independent_fits() {
    let (n, d, n_y) = (55usize, 6usize, 3usize);
    let (x, y) = design(n, d, n_y, 31);
    let alphas = [0.7f64, 11.0];
    let fit = ridge_gcv::<f64>(
        &x, &y, n, d, n_y, None, &alphas, true, GcvMode::Auto, false, false,
    )
    .expect("gcv fit");

    for (a, &alpha) in alphas.iter().enumerate() {
        for t in 0..n_y {
            let yt: Vec<f64> = (0..n).map(|i| y[i * n_y + t]).collect();
            let (want, _) = ridge_normal_equations(&x, &yt, n, d, None, alpha, true);
            let got: Vec<f64> = (0..d)
                .map(|j| fit.coefs[(a * d + j) * n_y + t])
                .collect();
            let err = max_abs_diff(&got, &want);
            assert!(err < 1e-9, "alpha={alpha} target={t}: coefs differ by {err:.3e}");
        }
    }
}

// ---------------------------------------------------------------------------
// gcv_mode / route selection
// ---------------------------------------------------------------------------

/// The three `gcv_mode` values are ONE code path here (module docs), so they
/// must agree BIT-for-bit. If this ever fails, the doc claim and the perf
/// story both need revisiting — which is the point of asserting equality
/// rather than closeness.
#[test]
fn gcv_modes_are_bit_identical() {
    let (n, d) = (70usize, 9usize);
    let (x, y) = design(n, d, 1, 37);
    let alphas = [0.2f64, 1.0, 5.0];
    let mut prev: Option<Vec<f64>> = None;
    for mode in [GcvMode::Auto, GcvMode::Svd, GcvMode::Eigen] {
        let fit = ridge_gcv::<f64>(
            &x, &y, n, d, 1, None, &alphas, true, mode, false, false,
        )
        .expect("gcv fit");
        if let Some(p) = &prev {
            assert_eq!(&fit.coefs, p, "gcv_mode={} diverged", mode.name());
        }
        prev = Some(fit.coefs);
    }
}

#[test]
fn route_follows_the_shape_not_the_mode() {
    for mode in [GcvMode::Auto, GcvMode::Svd, GcvMode::Eigen] {
        assert_eq!(resolve_route(mode, 100, 10), GcvRoute::Cov);
        assert_eq!(resolve_route(mode, 10, 100), GcvRoute::Gram);
        // sklearn's boundary: `n <= p` is the gram side.
        assert_eq!(resolve_route(mode, 10, 10), GcvRoute::Gram);
    }
}

#[test]
fn gcv_mode_parses_and_rejects() {
    assert_eq!(GcvMode::try_from("auto").unwrap(), GcvMode::Auto);
    assert_eq!(GcvMode::try_from("svd").unwrap(), GcvMode::Svd);
    assert_eq!(GcvMode::try_from("eigen").unwrap(), GcvMode::Eigen);
    let err = GcvMode::try_from("lanczos").unwrap_err();
    assert!(
        format!("{err}").contains("lanczos"),
        "the rejection must name the offending value, got: {err}"
    );
}

/// The two engines have DIFFERENT `alpha` boundaries, and that is sklearn's
/// rule: the LOO identity divides by `alpha`, an explicit `cv` does not.
#[test]
fn alpha_boundaries_differ_by_engine() {
    let (n, d) = (24usize, 4usize);
    let (x, y) = design(n, d, 1, 41);
    assert!(ridge_gcv::<f64>(
        &x, &y, n, d, 1, None, &[0.0, 1.0], true, GcvMode::Auto, false, false,
    )
    .is_err());
    let splits = vec![((0..12).collect::<Vec<_>>(), (12..24).collect::<Vec<_>>())];
    assert!(
        ridge_cv_grid::<f64>(
            &x, &y, n, d, 1, None, &[0.0, 1.0], true, &splits, false, false,
        )
            .is_ok()
    );
}

// ---------------------------------------------------------------------------
// The grid engine
// ---------------------------------------------------------------------------

/// The hoisted-Gram grid must score exactly what a per-`(split, alpha)` refit
/// scores. This is the whole optimization, stated as a test.
#[test]
fn grid_scores_match_per_split_refits() {
    let (n, d) = (60usize, 5usize);
    let (x, y) = design(n, d, 1, 43);
    let alphas = [0.5f64, 4.0, 20.0];
    let folds = 3usize;
    let splits: Vec<(Vec<usize>, Vec<usize>)> = (0..folds)
        .map(|f| {
            let test: Vec<usize> = (0..n).filter(|i| i % folds == f).collect();
            let train: Vec<usize> = (0..n).filter(|i| i % folds != f).collect();
            (train, test)
        })
        .collect();

    let fit = ridge_cv_grid::<f64>(&x, &y, n, d, 1, None, &alphas, true, &splits, true, false)
        .expect("grid fit");

    let mut base = 0usize;
    for (s, (train, test)) in splits.iter().enumerate() {
        let mut xtr = Vec::new();
        let mut ytr = Vec::new();
        for &i in train {
            xtr.extend_from_slice(&x[i * d..i * d + d]);
            ytr.push(y[i]);
        }
        let ybar: f64 = test.iter().map(|&i| y[i]).sum::<f64>() / test.len() as f64;
        for (a, &alpha) in alphas.iter().enumerate() {
            let (coef, intercept) =
                ridge_normal_equations(&xtr, &ytr, train.len(), d, None, alpha, true);
            let (mut ss_res, mut ss_tot) = (0.0f64, 0.0f64);
            for (j, &i) in test.iter().enumerate() {
                let mut p = intercept;
                for c in 0..d {
                    p += x[i * d + c] * coef[c];
                }
                let got = fit.predictions[base * alphas.len() + a * test.len() + j];
                assert!(
                    (got - p).abs() < 1e-9,
                    "split {s} alpha={alpha} row {j}: prediction {got} != {p}"
                );
                ss_res += (y[i] - p) * (y[i] - p);
                ss_tot += (y[i] - ybar) * (y[i] - ybar);
            }
            let want = 1.0 - ss_res / ss_tot;
            let got = fit.scores[s * alphas.len() + a];
            assert!(
                (got - want).abs() < 1e-9,
                "split {s} alpha={alpha}: R² {got} != {want}"
            );
        }
        base += test.len();
    }
}

#[test]
fn grid_rejects_out_of_range_indices() {
    let (n, d) = (20usize, 3usize);
    let (x, y) = design(n, d, 1, 47);
    let splits = vec![(vec![0usize, 1, 2], vec![3usize, 999])];
    assert!(
        ridge_cv_grid::<f64>(&x, &y, n, d, 1, None, &[1.0], true, &splits, false, false)
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Geometry / f32
// ---------------------------------------------------------------------------

#[test]
fn geometry_is_validated_before_any_work() {
    let x = vec![0.0f64; 12];
    let y = vec![0.0f64; 4];
    // x is 4x3 but declared 4x4.
    assert!(ridge_gcv::<f64>(
        &x, &y, 4, 4, 1, None, &[1.0], true, GcvMode::Auto, false, false,
    )
    .is_err());
    // y is length 4 but 2 targets are declared.
    assert!(ridge_gcv::<f64>(
        &x, &y, 4, 3, 2, None, &[1.0], true, GcvMode::Auto, false, false,
    )
    .is_err());
}

/// The `f32` ingress accumulates in `f64`, so it must land on the same answer
/// as feeding the same VALUES in as `f64` — the ingress width is not supposed
/// to be a second source of error on top of the input's own rounding.
#[test]
fn f32_ingress_agrees_with_f64_on_the_same_values() {
    let (n, d) = (48usize, 6usize);
    let (x64, y64) = design(n, d, 1, 53);
    let x32: Vec<f32> = x64.iter().map(|v| *v as f32).collect();
    let y32: Vec<f32> = y64.iter().map(|v| *v as f32).collect();
    let x_back: Vec<f64> = x32.iter().map(|v| *v as f64).collect();
    let y_back: Vec<f64> = y32.iter().map(|v| *v as f64).collect();
    let alphas = [0.3f64, 7.0];

    let a = ridge_gcv::<f32>(
        &x32, &y32, n, d, 1, None, &alphas, true, GcvMode::Auto, false, false,
    )
    .expect("f32 gcv");
    let b = ridge_gcv::<f64>(
        &x_back, &y_back, n, d, 1, None, &alphas, true, GcvMode::Auto, false, false,
    )
    .expect("f64 gcv");
    let err = max_abs_diff(&a.coefs, &b.coefs);
    assert!(err < 1e-10, "f32 ingress diverged from f64 by {err:.3e}");
}
