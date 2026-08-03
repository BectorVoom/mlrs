//! `BayesianRidge` (LINEAR-06) FULL sklearn parameter-surface oracle tests.
//!
//! Gates every `sklearn.linear_model.BayesianRidge` parameter that changes the
//! fit against the committed `bayesian_ridge_{f32,f64}_seed42` fixture
//! (`scripts/gen_oracle.py`::`gen_bayesian_ridge`):
//!
//! | parameter | how it is gated |
//! |---|---|
//! | `max_iter` | `1` (the non-converged path, which cannot reach the `iter != 0` test at all) and `5`, each asserting `n_iter_` |
//! | `tol` | a tight/loose pair, asserting `n_iter_` from both sides |
//! | `alpha_1` / `alpha_2` / `lambda_1` / `lambda_2` | one case far off the defaults (the fixture asserts it MOVES the fit) plus the all-zero boundary |
//! | `alpha_init` / `lambda_init` | both set, and `alpha_init` alone |
//! | `compute_score` | `scores_` compared element-wise, including its `n_iter_ + 1` length |
//! | `fit_intercept` | a `False` case in each family |
//! | `sample_weight` | unweighted/weighted pairs, incl. a scored one (`sw_sum` enters BOTH the `alpha_` update and the log marginal likelihood) |
//! | `copy_X` | asserted observationally inert — the documented mlrs contract |
//! | `n_samples < n_features` | a rank-deficient wide design, exercising sklearn's `U`-branch posterior mean, the zero-padded `logdet_sigma`, and the full-basis `sigma_` |
//!
//! Every case compares SIX fitted attributes, not just `coef_`: `coef_`,
//! `intercept_`, `alpha_`, `lambda_`, `sigma_` and `n_iter_`. That is the point
//! of the file — a wrong evidence update that happens to land on a similar
//! penalty would still reproduce `coef_` to a few digits while missing the
//! precisions and the iteration count, so gating `coef_` alone would test a
//! ridge solve rather than the iteration.
//!
//! BOTH fit ingress paths are driven (`Fit::fit` over a `DeviceArray` and
//! `fit_from_host_slice` over host slices), and a dedicated test asserts they
//! agree — they share `finish_fit`, and this is what keeps the split above that
//! line honest.
//!
//! The builder's data-INDEPENDENT rejections (`max_iter = 0`, `tol <= 0`, a
//! negative hyperprior, a negative init) are gated here too, since sklearn's
//! `_parameter_constraints` raises on exactly those.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). Per AGENTS.md §2 tests
//! live in `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod
//! tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::bayesian_ridge::BayesianRidge;
use mlrs_algos::typestate::{Fit, Fitted};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// Fixture geometry — `gen_oracle.py`'s `BAYES_TALL_*` / `BAYES_WIDE_*`.
const TALL: (usize, usize) = (60, 8);
const WIDE: (usize, usize) = (6, 10);
/// Held-out rows — `gen_oracle.py`'s `BAYES_N_TEST`.
const N_TEST: usize = 7;

/// Tolerance for the scalars accumulated in `f64` on BOTH sides regardless of
/// the design's storage dtype — `alpha_`, `lambda_`, `sigma_`, `scores_`.
///
/// An f32 FIXTURE still stores these as `f64`, because they are derived from an
/// `f64` accumulation in sklearn and in mlrs alike; what an f32 design changes
/// is the input bytes, not the working precision. They are therefore compared at
/// the f32 design's tolerance rather than at `F64_TOL` — the inputs differ in
/// the 7th digit, so the outputs cannot agree past it.
fn scalar_tol(design: &Tolerance) -> Tolerance {
    Tolerance {
        abs: design.abs,
        rel: design.rel,
    }
}

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("bayesian_ridge fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("bayesian_ridge fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict `1e-5` ABSOLUTE arm never loosened (the D-10 floored
/// precedent from `svd_test.rs`/`gemm_test.rs`).
fn assert_close(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= tol.abs + tol.rel * e.abs(),
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs,
            tol.rel
        );
    }
}

/// One fixture case: the ctor configuration and which design it was fitted on.
struct Case {
    /// Fixture key suffix (`coef_<name>`, `alpha_<name>`, …).
    name: &'static str,
    max_iter: usize,
    tol: f64,
    alpha_1: f64,
    alpha_2: f64,
    lambda_1: f64,
    lambda_2: f64,
    alpha_init: Option<f64>,
    lambda_init: Option<f64>,
    compute_score: bool,
    fit_intercept: bool,
    sample_weight: bool,
    /// Fit the rank-deficient `n_samples < n_features` design instead.
    wide: bool,
}

/// sklearn's ctor defaults, so each case below states only what it CHANGES —
/// which keeps the table readable and makes a drifted default a one-line fix.
const DEF: Case = Case {
    name: "",
    max_iter: 300,
    tol: 1e-3,
    alpha_1: 1e-6,
    alpha_2: 1e-6,
    lambda_1: 1e-6,
    lambda_2: 1e-6,
    alpha_init: None,
    lambda_init: None,
    compute_score: false,
    fit_intercept: true,
    sample_weight: false,
    wide: false,
};

/// The full case table, mirroring `gen_bayesian_ridge`'s `cases` list
/// one-for-one.
const CASES: &[Case] = &[
    Case {
        name: "default",
        ..DEF
    },
    Case {
        name: "noint",
        fit_intercept: false,
        ..DEF
    },
    // `max_iter = 1` never reaches the `iter != 0` convergence test.
    Case {
        name: "maxiter1",
        max_iter: 1,
        ..DEF
    },
    Case {
        name: "maxiter5",
        max_iter: 5,
        ..DEF
    },
    Case {
        name: "tol_tight",
        tol: 1e-8,
        max_iter: 1000,
        ..DEF
    },
    Case {
        name: "tol_loose",
        tol: 1e-1,
        ..DEF
    },
    Case {
        name: "priors",
        alpha_1: 1.0,
        alpha_2: 5.0,
        lambda_1: 50.0,
        lambda_2: 1.0,
        ..DEF
    },
    Case {
        name: "priors_zero",
        alpha_1: 0.0,
        alpha_2: 0.0,
        lambda_1: 0.0,
        lambda_2: 0.0,
        ..DEF
    },
    Case {
        name: "init",
        alpha_init: Some(2.5),
        lambda_init: Some(0.1),
        ..DEF
    },
    Case {
        name: "init_alpha_only",
        alpha_init: Some(10.0),
        ..DEF
    },
    Case {
        name: "score",
        compute_score: true,
        ..DEF
    },
    Case {
        name: "score_maxiter3",
        compute_score: true,
        max_iter: 3,
        ..DEF
    },
    Case {
        name: "score_noint",
        compute_score: true,
        fit_intercept: false,
        ..DEF
    },
    Case {
        name: "sw",
        sample_weight: true,
        ..DEF
    },
    Case {
        name: "sw_noint",
        fit_intercept: false,
        sample_weight: true,
        ..DEF
    },
    Case {
        name: "sw_score",
        compute_score: true,
        sample_weight: true,
        ..DEF
    },
    Case {
        name: "wide",
        wide: true,
        ..DEF
    },
    Case {
        name: "wide_noint",
        fit_intercept: false,
        wide: true,
        ..DEF
    },
    Case {
        name: "wide_score",
        compute_score: true,
        wide: true,
        ..DEF
    },
];

/// The design, target and weights a case is fitted on, in the fixture's dtype.
fn case_data<F>(case: &OracleCase, spec: &Case) -> (Vec<F>, Vec<F>, Vec<F>, (usize, usize))
where
    F: Float + CubeElement + Pod,
{
    let (xk, yk, shape) = if spec.wide {
        ("X_wide", "y_wide", WIDE)
    } else {
        ("X", "y", TALL)
    };
    let x = case
        .expect_f64(xk)
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let y = case
        .expect_f64(yk)
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let sw = case
        .expect_f64("sample_weight")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    (x, y, sw, shape)
}

/// Build the estimator a case describes.
fn build_case<F>(spec: &Case) -> BayesianRidge<F>
where
    F: Float + CubeElement + Pod,
{
    BayesianRidge::<F>::builder()
        .max_iter(spec.max_iter)
        .tol(spec.tol)
        .alpha_1(spec.alpha_1)
        .alpha_2(spec.alpha_2)
        .lambda_1(spec.lambda_1)
        .lambda_2(spec.lambda_2)
        .alpha_init(spec.alpha_init)
        .lambda_init(spec.lambda_init)
        .compute_score(spec.compute_score)
        .fit_intercept(spec.fit_intercept)
        .build::<F>()
        .unwrap_or_else(|e| panic!("case '{}' must build: {e}", spec.name))
}

/// Fit one case through the requested ingress and assert every fitted attribute
/// against sklearn.
///
/// `host_ingress` picks [`BayesianRidge::fit_from_host_slice`] over
/// [`Fit::fit`]. The host arm is only APPLICABLE on the cpu backend or below the
/// dispatch-cost floor, so the caller forces it through
/// `MLRS_RIDGE_GRAM_HOST` — asking for it where it does not apply would get a
/// typed error rather than a silently different route, which is the contract
/// under test.
fn check_case<F>(case: &OracleCase, spec: &Case, tol: &Tolerance, label: &str, host_ingress: bool)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x_host, y_host, sw_host, shape) = case_data::<F>(case, spec);
    let sw = spec.sample_weight.then_some(sw_host.as_slice());
    let est = build_case::<F>(spec);

    let fitted: BayesianRidge<F, Fitted> = if host_ingress {
        est.fit_from_host_slice(&mut pool, &x_host, &y_host, shape, sw)
            .unwrap_or_else(|e| panic!("case '{}' must fit (host): {e}", spec.name))
    } else {
        let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
        let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);
        est.fit_with_sample_weight(&mut pool, &x_dev, Some(&y_dev), shape, sw)
            .unwrap_or_else(|e| panic!("case '{}' must fit (device): {e}", spec.name))
    };

    let n = spec.name;
    let stol = scalar_tol(tol);

    let coef: Vec<f64> = fitted.coef(&pool).iter().map(|&v| host_to_f64(v)).collect();
    assert_close(
        &coef,
        case.expect_f64(&format!("coef_{n}")),
        tol,
        &format!("{label} coef_ [{n}]"),
    );
    assert_close(
        &[host_to_f64(fitted.intercept(&pool))],
        case.expect_f64(&format!("intercept_{n}")),
        tol,
        &format!("{label} intercept_ [{n}]"),
    );
    assert_close(
        &[fitted.alpha()],
        case.expect_f64(&format!("alpha_{n}")),
        &stol,
        &format!("{label} alpha_ [{n}]"),
    );
    assert_close(
        &[fitted.lambda()],
        case.expect_f64(&format!("lambda_{n}")),
        &stol,
        &format!("{label} lambda_ [{n}]"),
    );
    assert_close(
        fitted.sigma(),
        case.expect_f64(&format!("sigma_{n}")),
        &stol,
        &format!("{label} sigma_ [{n}]"),
    );

    let want_iter = case.expect_f64(&format!("n_iter_{n}"))[0] as usize;
    assert_eq!(
        fitted.n_iter(),
        want_iter,
        "{label} n_iter_ [{n}]: got {} expected {want_iter}",
        fitted.n_iter()
    );

    if spec.compute_score {
        let scores = case.expect_f64(&format!("scores_{n}"));
        // sklearn appends one score per iteration PLUS a final post-loop one.
        assert_eq!(
            fitted.scores().len(),
            want_iter + 1,
            "{label} scores_ [{n}]: length {} is not n_iter_ + 1 = {}",
            fitted.scores().len(),
            want_iter + 1
        );
        assert_close(
            fitted.scores(),
            scores,
            &stol,
            &format!("{label} scores_ [{n}]"),
        );
    } else {
        assert!(
            fitted.scores().is_empty(),
            "{label} scores_ [{n}]: must be empty without compute_score"
        );
    }

    // `X_offset_` / `X_scale_`: zeros / ones respectively when `!fit_intercept`,
    // and `X_scale_` is ALWAYS ones (the attribute outlived `normalize`).
    assert_eq!(
        fitted.x_scale().len(),
        shape.1,
        "{label} X_scale_ [{n}] length"
    );
    assert!(
        fitted.x_scale().iter().all(|&v| v == 1.0),
        "{label} X_scale_ [{n}]: must be all ones"
    );
    if !spec.fit_intercept {
        assert!(
            fitted.x_offset().iter().all(|&v| v == 0.0),
            "{label} X_offset_ [{n}]: must be zeros without fit_intercept"
        );
    }
}

/// Drive every case in [`CASES`] through the DEVICE ingress.
fn run_all_cases<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    for spec in CASES {
        check_case::<F>(case, spec, tol, label, false);
    }
}

/// Every parameter case vs sklearn, f32, device ingress.
#[test]
fn bayesian_ridge_params_all_cases_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("bayesian_ridge_f32_seed42.npz")).expect("load bayes f32");
    run_all_cases::<f32>(&case, &F32_TOL, "bayes f32");
}

/// Every parameter case vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn bayesian_ridge_params_all_cases_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("bayes f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("bayesian_ridge_f64_seed42.npz")).expect("load bayes f64");
    run_all_cases::<f64>(&case, &F64_TOL, "bayes f64");
}

/// The HOST ingress (`fit_from_host_slice`) against the SAME sklearn references.
///
/// This is the cpu fit path — the one the Python boundary takes and the one the
/// perf campaign targets — so it is gated against sklearn directly rather than
/// only against the device arm. `MLRS_RIDGE_GRAM_HOST=1` forces it applicable on
/// every backend (via `abflag`, so no environment data race — the
/// `mlrs-abflag-test-knobs` contract).
#[test]
fn bayesian_ridge_host_ingress_all_cases_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("bayes host f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let _guard = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
    let case = load_npz(fixture("bayesian_ridge_f64_seed42.npz")).expect("load bayes f64");
    for spec in CASES {
        check_case::<f64>(&case, spec, &F64_TOL, "bayes host f64", true);
    }
}

/// The two ingress paths must agree BIT-for-bit on the fitted precisions and to
/// the oracle tolerance on `coef_`.
///
/// They share `finish_fit`, so any disagreement is in the normal-equations
/// formation above it — the device `center_columns`+`gram_xty` composition vs
/// the host `centered_gram_xty` sweep. Comparing them to EACH OTHER (not just
/// each to sklearn) is what catches a drift that happens to stay inside the
/// oracle tolerance on both sides.
#[test]
fn bayesian_ridge_host_and_device_agree_f64() {
    let backend = capability::active_backend_name();
    if capability::skip_f64_with_log() {
        println!("bayes agree f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("bayesian_ridge_f64_seed42.npz")).expect("load bayes f64");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    for spec in CASES {
        let (x, y, sw_all, shape) = case_data::<f64>(&case, spec);
        let sw = spec.sample_weight.then_some(sw_all.as_slice());

        let host = {
            let _guard = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
            build_case::<f64>(spec)
                .fit_from_host_slice(&mut pool, &x, &y, shape, sw)
                .expect("host fit")
        };
        let device = {
            let _guard = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "0");
            let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
            let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);
            build_case::<f64>(spec)
                .fit_with_sample_weight(&mut pool, &xd, Some(&yd), shape, sw)
                .expect("device fit")
        };

        assert_eq!(
            host.n_iter(),
            device.n_iter(),
            "[{}] n_iter_ disagrees between ingress paths: host={} device={}",
            spec.name,
            host.n_iter(),
            device.n_iter()
        );
        assert_close(
            &host.coef(&pool),
            &device.coef(&pool),
            &F64_TOL,
            &format!("host-vs-device coef_ [{}]", spec.name),
        );
        assert_close(
            &[host.alpha(), host.lambda()],
            &[device.alpha(), device.lambda()],
            &F64_TOL,
            &format!("host-vs-device precisions [{}]", spec.name),
        );
    }
}

/// `predict` and `predict(return_std=True)` on HELD-OUT rows.
///
/// The mean gates the device-resident `coef_`/`intercept_` through the shared
/// fused `linear_predict` kernel; the std is the only place `sigma_` becomes an
/// observable rather than a stored attribute, which is why it is gated here as
/// well as compared element-wise above.
#[test]
fn bayesian_ridge_predict_and_std_f64() {
    let backend = capability::active_backend_name();
    if capability::skip_f64_with_log() {
        println!("bayes predict f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("bayesian_ridge_f64_seed42.npz")).expect("load bayes f64");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x: Vec<f64> = case.expect_f64("X").to_vec();
    let y: Vec<f64> = case.expect_f64("y").to_vec();
    let xt: Vec<f64> = case.expect_f64("X_test").to_vec();

    let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);
    let fitted = BayesianRidge::<f64>::new()
        .fit(&mut pool, &xd, Some(&yd), TALL)
        .expect("default fit");

    // Device-ingress predict.
    let xtd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &xt);
    let pred = mlrs_algos::typestate::Predict::predict(&fitted, &mut pool, &xtd, (N_TEST, TALL.1))
        .expect("predict");
    assert_close(
        &pred.to_host(&pool),
        case.expect_f64("pred_default"),
        &F64_TOL,
        "bayes predict mean (device ingress)",
    );

    // Host-ingress predict — the same numbers by a different route.
    let host_pred = fitted
        .predict_from_host(&mut pool, &xt, (N_TEST, TALL.1))
        .expect("predict_from_host");
    assert_close(
        &host_pred.values,
        case.expect_f64("pred_default"),
        &F64_TOL,
        "bayes predict mean (host ingress)",
    );

    let std = fitted
        .predict_std_from_host(&mut pool, &xt, (N_TEST, TALL.1))
        .expect("predict_std_from_host");
    assert_close(
        &std,
        case.expect_f64("predstd_default"),
        &F64_TOL,
        "bayes predict std",
    );
}

/// `copy_X` is observationally inert (the documented mlrs contract): the same
/// data fitted with `copy_X = true` and `copy_X = false` gives identical
/// coefficients, because mlrs never writes into the caller's buffer.
#[test]
fn bayesian_ridge_copy_x_is_inert_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("bayesian_ridge_f64_seed42.npz")).expect("load bayes f64");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x: Vec<f64> = case.expect_f64("X").to_vec();
    let y: Vec<f64> = case.expect_f64("y").to_vec();

    let mut fits = Vec::new();
    for copy_x in [true, false] {
        let est = BayesianRidge::<f64>::builder()
            .copy_x(copy_x)
            .build::<f64>()
            .expect("build");
        let f = est
            .fit_from_host_slice(&mut pool, &x, &y, TALL, None)
            .expect("fit");
        fits.push(f.coef(&pool));
    }
    assert_eq!(fits[0], fits[1], "copy_X changed the fit; it must be inert");
    // The caller's buffer is untouched — which is WHY the parameter can be a
    // no-op here.
    assert_eq!(x, case.expect_f64("X"), "fit mutated the caller's design");
}

/// `BayesianRidge::new()` and a default-built builder must agree (BLDR-01
/// default-drift gate — D-08's single source of truth is `new()`).
#[test]
fn bayesian_ridge_builder_defaults_match_new() {
    let from_new = BayesianRidge::<f64>::new();
    let from_builder = BayesianRidge::<f64>::builder()
        .build::<f64>()
        .expect("default builder must build");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "BayesianRidge::new() and the default builder disagree — a default has drifted"
    );
}

/// Every data-INDEPENDENT rejection sklearn's `_parameter_constraints` makes.
#[test]
fn bayesian_ridge_builder_rejects_invalid_hyperparameters() {
    // `max_iter >= 1` — sklearn's `Interval(Integral, 1, None, closed="left")`.
    assert!(matches!(
        BayesianRidge::<f64>::builder().max_iter(0).build::<f64>(),
        Err(BuildError::InvalidMaxIter { .. })
    ));

    // `tol > 0` — sklearn's `closed="neither"`, so unlike Ridge, 0 is REJECTED.
    for bad in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                BayesianRidge::<f64>::builder().tol(bad).build::<f64>(),
                Err(BuildError::InvalidHyperprior { param: "tol", .. })
            ),
            "tol = {bad} must be rejected"
        );
    }

    // The four Gamma hyperpriors: `>= 0` (`closed="left"` — 0 is ACCEPTED).
    let priors: [(&str, fn(f64) -> Result<BayesianRidge<f64>, BuildError>); 4] = [
        ("alpha_1", |v| {
            BayesianRidge::<f64>::builder().alpha_1(v).build::<f64>()
        }),
        ("alpha_2", |v| {
            BayesianRidge::<f64>::builder().alpha_2(v).build::<f64>()
        }),
        ("lambda_1", |v| {
            BayesianRidge::<f64>::builder().lambda_1(v).build::<f64>()
        }),
        ("lambda_2", |v| {
            BayesianRidge::<f64>::builder().lambda_2(v).build::<f64>()
        }),
    ];
    for (name, make) in priors {
        assert!(
            matches!(make(-1.0), Err(BuildError::InvalidHyperprior { param, .. }) if param == name),
            "{name} = -1 must be rejected"
        );
        assert!(
            matches!(make(f64::NAN), Err(BuildError::InvalidHyperprior { .. })),
            "{name} = NaN must be rejected"
        );
        assert!(
            make(0.0).is_ok(),
            "{name} = 0 must be ACCEPTED (closed='left')"
        );
    }

    // The two initial precisions: `>= 0` when given, and `None` is fine.
    assert!(matches!(
        BayesianRidge::<f64>::builder()
            .alpha_init(Some(-1.0))
            .build::<f64>(),
        Err(BuildError::InvalidHyperprior {
            param: "alpha_init",
            ..
        })
    ));
    assert!(matches!(
        BayesianRidge::<f64>::builder()
            .lambda_init(Some(f64::NAN))
            .build::<f64>(),
        Err(BuildError::InvalidHyperprior {
            param: "lambda_init",
            ..
        })
    ));
    assert!(BayesianRidge::<f64>::builder()
        .alpha_init(None)
        .lambda_init(None)
        .build::<f64>()
        .is_ok());
}

/// A mis-shaped operand is a typed geometry error, never a panic or a silent
/// wrong answer (ASVS V5 — the data-DEPENDENT half of the D-08 split).
#[test]
fn bayesian_ridge_rejects_bad_geometry() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = vec![1.0f64; 12];
    let y = vec![1.0f64; 4];
    let _guard = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "1");

    // y shorter than n_samples.
    assert!(BayesianRidge::<f64>::new()
        .fit_from_host_slice(&mut pool, &x, &y[..3], (4, 3), None)
        .is_err());
    // x length disagrees with n_samples * n_features.
    assert!(BayesianRidge::<f64>::new()
        .fit_from_host_slice(&mut pool, &x, &y, (4, 4), None)
        .is_err());
    // A negative sample_weight would make `√w` NaN and poison every reduction.
    let bad_sw = vec![1.0f64, -1.0, 1.0, 1.0];
    assert!(BayesianRidge::<f64>::new()
        .fit_from_host_slice(&mut pool, &x, &y, (4, 3), Some(&bad_sw))
        .is_err());
}

// ==================== BAYES-GPU — the device arms ====================

/// The device Gram arm must actually be REACHABLE on this backend, or every
/// agreement test below silently compares the host arm against itself.
///
/// This is the vacuity guard the `mlrs-ridge-positive-cuda` campaign learned to
/// write: forcing an A/B knob past a gate that has already refused the
/// configuration produces a green test that exercises one arm twice. So this
/// asserts the gate's verdict directly, against what the backend can do:
///
/// - **cpu** → `false`, unconditionally. The arm is a GPU-shaped reduction and
///   the host sweep is the right code there.
/// - **any backend WITH f64** → `true` at the fixture's feature counts, which
///   are far below the fused-Gram cap.
/// - **a backend WITHOUT f64** → `false`. `BayesianRidge`'s Gram must be
///   accumulated in `f64`, so this is a correctness refusal, not a perf one, and
///   the estimator falls back to the host sweep rather than losing precision.
#[test]
fn bayesian_ridge_device_gram_arm_is_reachable() {
    use mlrs_backend::prims::normal_eq::device_gram_applicable;

    let backend = capability::active_backend_name();
    let advertised = capability::feature_enabled(capability::FloatKind::F64);
    let runnable = capability::f64_device_kernels_available();
    let got = device_gram_applicable::<f64>(TALL.1);
    let want = backend != "cpu" && runnable;
    assert_eq!(
        got, want,
        "device_gram_applicable(d={}) on backend={backend} \
         (f64 advertised={advertised}, runnable={runnable}) must be {want}, got \
         {got} — if this fails the device/host agreement tests are vacuous",
        TALL.1
    );
    // Printed rather than asserted: the two flags DISAGREEING is the expected
    // state on cuda (`capability::f64_device_kernels_available` documents why),
    // and the log line is how a reader of CI output sees which one drove the
    // verdict on the machine that ran.
    println!(
        "bayes device gram arm backend={backend} f64_advertised={advertised} \
         f64_runnable={runnable} applicable={got}"
    );
}

/// The device Gram arm and the host sweep must agree on EVERY fitted attribute,
/// for EVERY case in [`CASES`] — the full sklearn parameter surface.
///
/// The two arms are switched with `MLRS_BAYES_GRAM_DEVICE` through the SAME
/// entry point ([`BayesianRidge::fit_with_sample_weight`]), so nothing but the
/// normal-equations formation differs: the eigendecomposition, the evidence
/// loop, `sigma_` and the intercept are literally the same code on both sides.
/// Any disagreement is therefore in the reduction, which is exactly what this
/// gates.
///
/// Six attributes are compared, not just `coef_`, for the reason the module docs
/// give: a Gram that is wrong in a way the shrinkage partly absorbs still
/// reproduces `coef_` to a few digits while missing `alpha_`, `sigma_` and the
/// iteration count.
///
/// `n_iter_` is compared for EQUALITY rather than for closeness. It is the
/// integer output of a `Σ|Δcoef| < tol` test, so an arm that drifts enough to
/// stop one iteration early is solving a different problem — and would still
/// pass a tolerance-based comparison of everything else.
fn device_gram_agrees<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    for spec in CASES {
        let (x, y, sw_all, shape) = case_data::<F>(case, spec);
        let sw = spec.sample_weight.then_some(sw_all.as_slice());
        let xd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
        let yd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

        let mut fits = Vec::new();
        for arm in ["1", "0"] {
            let _guard = mlrs_backend::abflag::force("MLRS_BAYES_GRAM_DEVICE", arm);
            fits.push(
                build_case::<F>(spec)
                    .fit_with_sample_weight(&mut pool, &xd, Some(&yd), shape, sw)
                    .unwrap_or_else(|e| {
                        panic!("case '{}' must fit (gram_device={arm}): {e}", spec.name)
                    }),
            );
        }
        let (dev, host) = (&fits[0], &fits[1]);
        let n = spec.name;
        let stol = scalar_tol(tol);

        assert_eq!(
            dev.n_iter(),
            host.n_iter(),
            "{label} [{n}] n_iter_: device={} host={}",
            dev.n_iter(),
            host.n_iter()
        );
        let dcoef: Vec<f64> = dev.coef(&pool).iter().map(|&v| host_to_f64(v)).collect();
        let hcoef: Vec<f64> = host.coef(&pool).iter().map(|&v| host_to_f64(v)).collect();
        assert_close(&dcoef, &hcoef, tol, &format!("{label} [{n}] coef_"));
        assert_close(
            &[host_to_f64(dev.intercept(&pool))],
            &[host_to_f64(host.intercept(&pool))],
            tol,
            &format!("{label} [{n}] intercept_"),
        );
        assert_close(
            &[dev.alpha(), dev.lambda()],
            &[host.alpha(), host.lambda()],
            &stol,
            &format!("{label} [{n}] precisions"),
        );
        assert_close(
            dev.sigma(),
            host.sigma(),
            &stol,
            &format!("{label} [{n}] sigma_"),
        );
        assert_close(
            dev.x_offset(),
            host.x_offset(),
            &stol,
            &format!("{label} [{n}] X_offset_"),
        );
        assert_close(
            dev.scores(),
            host.scores(),
            &stol,
            &format!("{label} [{n}] scores_"),
        );
    }
}

/// The device Gram arm vs the host sweep, `f64` — the arm's native width, where
/// the Gram kernels already accumulate in `f64` and no widening happens.
#[test]
fn bayesian_ridge_device_gram_agrees_f64() {
    let backend = capability::active_backend_name();
    if !capability::f64_device_kernels_available() {
        println!("bayes device gram f64 backend={backend}: SKIPPED (no f64 on this adapter)");
        return;
    }
    let case = load_npz(fixture("bayesian_ridge_f64_seed42.npz")).expect("load bayes f64");
    device_gram_agrees::<f64>(&case, &F64_TOL, "device-vs-host gram f64");
}

/// The device Gram arm vs the host sweep, `f32` — which is where the WIDENING
/// kernel is under test.
///
/// This is the case the arm exists to get right. An `f32` design whose Gram were
/// accumulated at `f32` would reproduce `coef_` and fail on `alpha_`; here both
/// arms widen to `f64` first (the device one on-device via
/// `elementwise::widen_elem`, the host one in its element read), so they must
/// agree to the same tolerance the `f64` pair does. A regression that dropped
/// the widening would surface as an `alpha_` mismatch on the interpolating wide
/// fixture, which `CASES` includes.
#[test]
fn bayesian_ridge_device_gram_agrees_f32() {
    let backend = capability::active_backend_name();
    if !capability::f64_device_kernels_available() {
        println!("bayes device gram f32 backend={backend}: SKIPPED (arm needs f64 on the adapter)");
        return;
    }
    let case = load_npz(fixture("bayesian_ridge_f32_seed42.npz")).expect("load bayes f32");
    device_gram_agrees::<f32>(&case, &F32_TOL, "device-vs-host gram f32");
}

/// `predict(X, return_std=True)`: the fused device kernel, the host sweep, and
/// sklearn must all agree.
///
/// Three-way rather than two-way on purpose. The two mlrs arms share the
/// covariance FACTOR (`posterior_sigma_sqrt_t`), so comparing them to each other
/// alone would not catch a wrong factor — it would be wrong identically on both
/// sides. The sklearn leg pins the factor itself; the arm-vs-arm leg pins the
/// kernel against the host code.
///
/// The device leg is reached on the cpu backend too, by forcing
/// `MLRS_BAYES_STD_HOST=0` — so this kernel is gated on every backend the suite
/// runs, not only where a GPU happens to be the default.
#[test]
fn bayesian_ridge_predict_std_arms_agree_f64() {
    let backend = capability::active_backend_name();
    if !capability::f64_device_kernels_available() {
        println!("bayes predict_std arms f64 backend={backend}: SKIPPED (no f64 on this adapter)");
        return;
    }
    let case = load_npz(fixture("bayesian_ridge_f64_seed42.npz")).expect("load bayes f64");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x: Vec<f64> = case.expect_f64("X").to_vec();
    let y: Vec<f64> = case.expect_f64("y").to_vec();
    let xt: Vec<f64> = case.expect_f64("X_test").to_vec();
    let shape = (N_TEST, TALL.1);

    let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);
    let fitted = BayesianRidge::<f64>::new()
        .fit(&mut pool, &xd, Some(&yd), TALL)
        .expect("default fit");

    let want = case.expect_f64("predstd_default");

    for (arm, label) in [("1", "host sweep"), ("0", "device kernel")] {
        let _guard = mlrs_backend::abflag::force("MLRS_BAYES_STD_HOST", arm);
        let got = fitted
            .predict_std_from_host(&mut pool, &xt, shape)
            .unwrap_or_else(|e| panic!("predict_std ({label}): {e}"));
        assert_close(&got, want, &F64_TOL, &format!("predict_std {label}"));
    }

    // The device-RESIDENT entry point (`predict_std`), whose result never
    // reaches the host until the caller asks — the `return_std` twin of
    // `Predict::predict`, and the one a device-side consumer would use.
    let xtd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &xt);
    let dev = fitted
        .predict_std(&mut pool, &xtd, shape)
        .expect("predict_std (device ingress)");
    assert_close(
        &dev.to_host(&pool),
        want,
        &F64_TOL,
        "predict_std device ingress",
    );
}

/// `predict_std`'s two arms agree at `f32` too, against the `f32` fixture.
///
/// The kernel accumulates in `F`, so this is the leg that pins the claim that an
/// `f32` sum of `d²` non-negative terms stays inside the oracle contract — the
/// quantity has no cancellation, unlike the `Σ`-matvec form it replaced.
#[test]
fn bayesian_ridge_predict_std_arms_agree_f32() {
    let case = load_npz(fixture("bayesian_ridge_f32_seed42.npz")).expect("load bayes f32");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x: Vec<f32> = case.expect_f64("X").iter().map(|&v| v as f32).collect();
    let y: Vec<f32> = case.expect_f64("y").iter().map(|&v| v as f32).collect();
    let xt: Vec<f32> = case.expect_f64("X_test").iter().map(|&v| v as f32).collect();
    let shape = (N_TEST, TALL.1);

    let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let yd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);
    let fitted = BayesianRidge::<f32>::new()
        .fit(&mut pool, &xd, Some(&yd), TALL)
        .expect("default fit");

    let want = case.expect_f64("predstd_default");
    for (arm, label) in [("1", "host sweep"), ("0", "device kernel")] {
        let _guard = mlrs_backend::abflag::force("MLRS_BAYES_STD_HOST", arm);
        let got = fitted
            .predict_std_from_host(&mut pool, &xt, shape)
            .unwrap_or_else(|e| panic!("predict_std f32 ({label}): {e}"));
        assert_close(&got, want, &scalar_tol(&F32_TOL), &format!("predict_std f32 {label}"));
    }
}
