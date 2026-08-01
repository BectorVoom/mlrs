//! Ridge (LINEAR-02) FULL sklearn parameter-surface oracle tests.
//!
//! `ridge_test.rs` gates the default `Ridge(alpha, fit_intercept)` path across
//! the alpha sweep. THIS file gates every REMAINING
//! `sklearn.linear_model.Ridge` parameter against the committed
//! `ridge_params_{f32,f64}_seed42` fixture (`scripts/gen_oracle.py`
//! ::`gen_ridge_params`):
//!
//! | parameter | how it is gated |
//! |---|---|
//! | `solver` | all EIGHT values, each vs sklearn's `coef_`/`intercept_` for the SAME solver, plus the `solver_` resolution `auto` lands on |
//! | `fit_intercept` | a `False` case per solver family |
//! | `positive` | `lbfgs` + `auto`, with a coefficient-sign assert so the bound is proven to BIND |
//! | `sample_weight` | every solver, covering BOTH of sklearn's regimes (the `√w` row rescale, and `sag`/`saga`'s direct weighting) |
//! | `max_iter` / `tol` | the convergence knobs the iterative cases are run at (`1e-10` / `100000`), plus `n_iter_` presence and a `max_iter = 1` non-convergence case |
//! | `copy_X` | asserted to be observationally inert — the same fit either way — which is the documented mlrs contract |
//! | `random_state` | asserted reproducible for `sag`/`saga`, and irrelevant to the converged optimum |
//!
//! The builder's data-INDEPENDENT rejections (`alpha < 0`, `tol < 0`,
//! `max_iter = 0`, the `positive`/`solver` incompatibility pair, an unknown
//! solver name) are gated here too, since sklearn raises on exactly those.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). Per AGENTS.md §2 tests
//! live in `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod
//! tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::ridge::{Ridge, RidgeSolver};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// Fixture geometry — `gen_oracle.py`'s `RIDGE_PARAMS_N_SAMPLES/N_FEATURES`.
const N_SAMPLES: usize = 40;
const N_FEATURES: usize = 5;

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
        _ => unreachable!("ridge fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ridge fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened (the D-10 floored
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

/// One fixture case: the ctor configuration and what sklearn produced for it.
struct Case {
    /// Fixture key suffix (`coef_<name>` / `intercept_<name>`).
    name: &'static str,
    solver: RidgeSolver,
    fit_intercept: bool,
    positive: bool,
    sample_weight: bool,
    /// The `solver_` sklearn resolved to (asserted in `gen_ridge_params`).
    expect_solver: RidgeSolver,
    /// Does sklearn populate `n_iter_` for this solver? Only `lsqr` and the SAG
    /// family — `_ridge_regression` leaves it `None` everywhere else.
    expect_n_iter: bool,
}

/// The full case table, mirroring `gen_ridge_params`'s `cases` list one-for-one.
const CASES: &[Case] = &[
    // --- the solver set, fit_intercept=True, unweighted ---------------------
    Case { name: "auto", solver: RidgeSolver::Auto, fit_intercept: true, positive: false, sample_weight: false, expect_solver: RidgeSolver::Cholesky, expect_n_iter: false },
    Case { name: "cholesky", solver: RidgeSolver::Cholesky, fit_intercept: true, positive: false, sample_weight: false, expect_solver: RidgeSolver::Cholesky, expect_n_iter: false },
    Case { name: "svd", solver: RidgeSolver::Svd, fit_intercept: true, positive: false, sample_weight: false, expect_solver: RidgeSolver::Svd, expect_n_iter: false },
    Case { name: "lsqr", solver: RidgeSolver::Lsqr, fit_intercept: true, positive: false, sample_weight: false, expect_solver: RidgeSolver::Lsqr, expect_n_iter: true },
    Case { name: "sparse_cg", solver: RidgeSolver::SparseCg, fit_intercept: true, positive: false, sample_weight: false, expect_solver: RidgeSolver::SparseCg, expect_n_iter: false },
    Case { name: "sag", solver: RidgeSolver::Sag, fit_intercept: true, positive: false, sample_weight: false, expect_solver: RidgeSolver::Sag, expect_n_iter: true },
    Case { name: "saga", solver: RidgeSolver::Saga, fit_intercept: true, positive: false, sample_weight: false, expect_solver: RidgeSolver::Saga, expect_n_iter: true },
    // --- positive=True ------------------------------------------------------
    Case { name: "lbfgs_pos", solver: RidgeSolver::Lbfgs, fit_intercept: true, positive: true, sample_weight: false, expect_solver: RidgeSolver::Lbfgs, expect_n_iter: false },
    Case { name: "auto_pos", solver: RidgeSolver::Auto, fit_intercept: true, positive: true, sample_weight: false, expect_solver: RidgeSolver::Lbfgs, expect_n_iter: false },
    // --- fit_intercept=False ------------------------------------------------
    Case { name: "cholesky_noint", solver: RidgeSolver::Cholesky, fit_intercept: false, positive: false, sample_weight: false, expect_solver: RidgeSolver::Cholesky, expect_n_iter: false },
    Case { name: "svd_noint", solver: RidgeSolver::Svd, fit_intercept: false, positive: false, sample_weight: false, expect_solver: RidgeSolver::Svd, expect_n_iter: false },
    Case { name: "lsqr_noint", solver: RidgeSolver::Lsqr, fit_intercept: false, positive: false, sample_weight: false, expect_solver: RidgeSolver::Lsqr, expect_n_iter: true },
    Case { name: "sag_noint", solver: RidgeSolver::Sag, fit_intercept: false, positive: false, sample_weight: false, expect_solver: RidgeSolver::Sag, expect_n_iter: true },
    Case { name: "lbfgs_pos_noint", solver: RidgeSolver::Lbfgs, fit_intercept: false, positive: true, sample_weight: false, expect_solver: RidgeSolver::Lbfgs, expect_n_iter: false },
    // --- sample_weight ------------------------------------------------------
    Case { name: "cholesky_sw", solver: RidgeSolver::Cholesky, fit_intercept: true, positive: false, sample_weight: true, expect_solver: RidgeSolver::Cholesky, expect_n_iter: false },
    Case { name: "svd_sw", solver: RidgeSolver::Svd, fit_intercept: true, positive: false, sample_weight: true, expect_solver: RidgeSolver::Svd, expect_n_iter: false },
    Case { name: "lsqr_sw", solver: RidgeSolver::Lsqr, fit_intercept: true, positive: false, sample_weight: true, expect_solver: RidgeSolver::Lsqr, expect_n_iter: true },
    Case { name: "sparse_cg_sw", solver: RidgeSolver::SparseCg, fit_intercept: true, positive: false, sample_weight: true, expect_solver: RidgeSolver::SparseCg, expect_n_iter: false },
    Case { name: "sag_sw", solver: RidgeSolver::Sag, fit_intercept: true, positive: false, sample_weight: true, expect_solver: RidgeSolver::Sag, expect_n_iter: true },
    Case { name: "saga_sw", solver: RidgeSolver::Saga, fit_intercept: true, positive: false, sample_weight: true, expect_solver: RidgeSolver::Saga, expect_n_iter: true },
    Case { name: "lbfgs_pos_sw", solver: RidgeSolver::Lbfgs, fit_intercept: true, positive: true, sample_weight: true, expect_solver: RidgeSolver::Lbfgs, expect_n_iter: false },
    Case { name: "cholesky_noint_sw", solver: RidgeSolver::Cholesky, fit_intercept: false, positive: false, sample_weight: true, expect_solver: RidgeSolver::Cholesky, expect_n_iter: false },
];

/// The `tol` / `max_iter` the fixture's references were produced at
/// (`RIDGE_PARAMS_TOL` / `RIDGE_PARAMS_MAX_ITER`). Both sides run TIGHT so the
/// comparison is against the converged optimum, not against a particular
/// early-stop point — see the generator's docstring.
fn fixture_tol_and_max_iter(case: &OracleCase) -> (f64, usize) {
    let tol = case.expect_f64("tol")[0];
    let max_iter = case.expect_f64("max_iter")[0] as usize;
    (tol, max_iter)
}

/// Host copies of the fixture's `X` / `y` / `sample_weight` at width `F`.
fn fixture_data<F>(case: &OracleCase) -> (Vec<F>, Vec<F>, Vec<F>)
where
    F: Float + CubeElement + Pod,
{
    let x = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let y = case.expect_f64("y").iter().map(|&v| f64_to::<F>(v)).collect();
    let sw = case
        .expect_f64("sample_weight")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    (x, y, sw)
}

/// Fit one case and return `(coef_, intercept_, n_iter_, solver_)`.
fn fit_case<F>(case: &OracleCase, spec: &Case) -> (Vec<f64>, f64, Option<usize>, RidgeSolver)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x_host, y_host, sw_host) = fixture_data::<F>(case);
    let (tol, max_iter) = fixture_tol_and_max_iter(case);

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);

    let est = Ridge::<F>::builder()
        .alpha(case.expect_f64("alpha")[0])
        .fit_intercept(spec.fit_intercept)
        .solver(spec.solver)
        .positive(spec.positive)
        .tol(tol)
        .max_iter(Some(max_iter))
        .random_state(Some(0))
        .build::<F>()
        .unwrap_or_else(|e| panic!("case '{}' must build: {e}", spec.name));

    let sw = if spec.sample_weight {
        Some(sw_host.as_slice())
    } else {
        None
    };
    let fitted = est
        .fit_with_sample_weight(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES), sw)
        .unwrap_or_else(|e| panic!("case '{}' must fit: {e}", spec.name));

    let coef = fitted.coef(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let intercept = host_to_f64(fitted.intercept(&pool));
    (coef, intercept, fitted.n_iter(), fitted.solver())
}

/// Drive every case in [`CASES`] against its sklearn reference.
fn run_all_cases<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    for spec in CASES {
        let (coef, intercept, n_iter, solver_used) = fit_case::<F>(case, spec);

        assert_close(
            &coef,
            case.expect_f64(&format!("coef_{}", spec.name)),
            tol,
            &format!("{label} coef_ [{}]", spec.name),
        );
        assert_close(
            &[intercept],
            case.expect_f64(&format!("intercept_{}", spec.name)),
            tol,
            &format!("{label} intercept_ [{}]", spec.name),
        );

        // `solver_`: sklearn's resolved-solver attribute (the `auto` dispatch
        // and, for `auto_pos`, the positive⇒lbfgs rule).
        assert_eq!(
            solver_used,
            spec.expect_solver,
            "{label} solver_ [{}]: got '{}' expected '{}'",
            spec.name,
            solver_used.name(),
            spec.expect_solver.name()
        );

        // `n_iter_`: Some exactly for the solvers sklearn populates it for.
        assert_eq!(
            n_iter.is_some(),
            spec.expect_n_iter,
            "{label} n_iter_ [{}]: got {n_iter:?}, sklearn populates it = {}",
            spec.name,
            spec.expect_n_iter
        );

        // `positive=True` must actually hold on the OUTPUT (and the fixture
        // asserts the unconstrained answer has a negative entry, so this is a
        // real constraint, not a coincidence).
        if spec.positive {
            assert!(
                coef.iter().all(|&c| c >= -tol.abs),
                "{label} [{}]: positive=true produced a negative coef_: {coef:?}",
                spec.name
            );
        }
    }
}

/// Every parameter case vs sklearn, f32.
#[test]
fn ridge_params_all_cases_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_params_f32_seed42.npz")).expect("load ridge_params_f32");
    run_all_cases::<f32>(&case, &F32_TOL, "ridge params f32");
}

/// Every parameter case vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn ridge_params_all_cases_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge params f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("ridge_params_f64_seed42.npz")).expect("load ridge_params_f64");
    run_all_cases::<f64>(&case, &F64_TOL, "ridge params f64");
}

/// `copy_X` is observationally inert (the documented mlrs contract): the same
/// data fitted with `copy_X = true` and `copy_X = false` must give bit-identical
/// coefficients, AND the caller's device buffer must be unchanged by the fit —
/// which is WHY the parameter can be a no-op here.
#[test]
fn ridge_copy_x_is_inert_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_params_f32_seed42.npz")).expect("load ridge_params_f32");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x_host, y_host, _) = fixture_data::<f32>(&case);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_host);

    let mut coefs = Vec::new();
    for copy_x in [true, false] {
        let fitted = Ridge::<f32>::builder()
            .copy_x(copy_x)
            .build::<f32>()
            .expect("builds")
            .fit_with_sample_weight(
                &mut pool,
                &x_dev,
                Some(&y_dev),
                (N_SAMPLES, N_FEATURES),
                None,
            )
            .expect("fits");
        coefs.push(fitted.coef(&pool));
    }
    assert_eq!(coefs[0], coefs[1], "copy_X changed the fitted coef_");

    // The caller's X is untouched by fit — the reason copy_X has nothing to do.
    assert_eq!(
        x_dev.to_host(&pool),
        x_host,
        "fit mutated the caller's X buffer (copy_X would then NOT be a no-op)"
    );
}

/// `random_state` is reproducible for the stochastic solvers, and does not move
/// the CONVERGED optimum: two different seeds must agree with each other (and
/// with sklearn) to the oracle tolerance, while the same seed is bit-identical.
#[test]
fn ridge_random_state_reproducible_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge random_state f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("ridge_params_f64_seed42.npz")).expect("load ridge_params_f64");
    let (tol, max_iter) = fixture_tol_and_max_iter(&case);

    let fit_seed = |seed: u64| -> Vec<f64> {
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let (x_host, y_host, _) = fixture_data::<f64>(&case);
        let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
        let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);
        let fitted = Ridge::<f64>::builder()
            .solver(RidgeSolver::Sag)
            .tol(tol)
            .max_iter(Some(max_iter))
            .random_state(Some(seed))
            .build::<f64>()
            .expect("builds")
            .fit_with_sample_weight(
                &mut pool,
                &x_dev,
                Some(&y_dev),
                (N_SAMPLES, N_FEATURES),
                None,
            )
            .expect("fits");
        fitted.coef(&pool)
    };

    assert_eq!(fit_seed(7), fit_seed(7), "same random_state must be bit-identical");
    let a: Vec<f64> = fit_seed(7);
    let b: Vec<f64> = fit_seed(12345);
    assert_close(
        &a,
        &b,
        &F64_TOL,
        "sag coef_ across random_state (converged optimum is seed-independent)",
    );
    assert_close(
        &a,
        case.expect_f64("coef_sag"),
        &F64_TOL,
        "sag coef_ vs sklearn (seed 7)",
    );
}

/// `max_iter` is load-bearing: capping an iterative solver at ONE iteration must
/// stop it short of the optimum. Without this, a `max_iter` that was silently
/// ignored would pass every other test in this file.
#[test]
fn ridge_max_iter_caps_the_iteration_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge max_iter f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("ridge_params_f64_seed42.npz")).expect("load ridge_params_f64");
    let (tol, _) = fixture_tol_and_max_iter(&case);
    let converged = case.expect_f64("coef_sag");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x_host, y_host, _) = fixture_data::<f64>(&case);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);

    let fitted = Ridge::<f64>::builder()
        .solver(RidgeSolver::Sag)
        .tol(tol)
        .max_iter(Some(1))
        .random_state(Some(0))
        .build::<f64>()
        .expect("builds")
        .fit_with_sample_weight(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES), None)
        .expect("fits");

    assert_eq!(fitted.n_iter(), Some(1), "max_iter=1 must report n_iter_=1");
    let coef = fitted.coef(&pool);
    let far: f64 = coef
        .iter()
        .zip(converged.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0, f64::max);
    assert!(
        far > F64_TOL.abs,
        "max_iter=1 landed within the oracle tolerance of the converged optimum \
         ({far:e}) — max_iter is not actually capping the iteration"
    );
}

/// A `sample_weight` of all ones must reproduce the UNWEIGHTED fit exactly (the
/// weighted preprocessing path is a strict generalization of the device one).
#[test]
fn ridge_unit_sample_weight_matches_unweighted_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge unit-sw f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("ridge_params_f64_seed42.npz")).expect("load ridge_params_f64");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x_host, y_host, _) = fixture_data::<f64>(&case);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);
    let ones = vec![1.0f64; N_SAMPLES];

    let mut coefs = Vec::new();
    for sw in [None, Some(ones.as_slice())] {
        let fitted = Ridge::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_with_sample_weight(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES), sw)
            .expect("fits");
        coefs.push(fitted.coef(&pool));
    }
    assert_close(
        &coefs[1],
        &coefs[0],
        &F64_TOL,
        "unit sample_weight vs unweighted coef_",
    );
}

/// A negative / non-finite `sample_weight` is rejected at `fit` as a typed
/// error rather than propagating a NaN through `√w` into every reduction.
#[test]
fn ridge_rejects_invalid_sample_weight_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge bad-sw f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("ridge_params_f64_seed42.npz")).expect("load ridge_params_f64");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x_host, y_host, _) = fixture_data::<f64>(&case);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);

    for bad in [-1.0f64, f64::NAN] {
        let mut sw = vec![1.0f64; N_SAMPLES];
        sw[3] = bad;
        let err = Ridge::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_with_sample_weight(
                &mut pool,
                &x_dev,
                Some(&y_dev),
                (N_SAMPLES, N_FEATURES),
                Some(&sw),
            )
            // `.err().expect(..)` rather than `.expect_err(..)`: the Ok arm is a
            // fitted estimator holding device buffers, which is deliberately not
            // `Debug` (D-03), and `expect_err` requires `T: Debug`.
            .err()
            .expect("an invalid sample_weight must be rejected");
        assert!(
            format!("{err}").contains("sample_weight"),
            "unexpected error for sample_weight = {bad}: {err}"
        );
    }

    // An ALL-ZERO weight vector leaves nothing to fit — sklearn's
    // `check_all_zero_sample_weights_error` requires a hard error here, not the
    // all-zero coefficient vector the penalized solve would otherwise return.
    let zeros = vec![0.0f64; N_SAMPLES];
    let err = Ridge::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_with_sample_weight(
            &mut pool,
            &x_dev,
            Some(&y_dev),
            (N_SAMPLES, N_FEATURES),
            Some(&zeros),
        )
        .err()
        .expect("an all-zero sample_weight must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("weight") && msg.contains("zero"),
        "the all-zero sample_weight error must name both 'weight' and 'zero' \
         (sklearn's check_all_zero_sample_weights_error pattern): {msg}"
    );

    // Wrong LENGTH is a geometry error, not a validity one.
    let short = vec![1.0f64; N_SAMPLES - 1];
    Ridge::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_with_sample_weight(
            &mut pool,
            &x_dev,
            Some(&y_dev),
            (N_SAMPLES, N_FEATURES),
            Some(&short),
        )
        .err()
        .expect("a short sample_weight must be rejected");
}

/// The data-INDEPENDENT builder rejections, matching the `ValueError`s
/// `sklearn.linear_model.Ridge` raises for the same inputs. Pure host — no
/// device, so no f64 gate.
#[test]
fn ridge_builder_rejects_invalid_hyperparameters() {
    // alpha < 0
    assert!(matches!(
        Ridge::<f64>::builder().alpha(-1.0).build::<f64>(),
        Err(BuildError::InvalidAlpha { .. })
    ));
    // tol < 0 / non-finite (sklearn: Interval(Real, 0, None, closed="left")).
    assert!(matches!(
        Ridge::<f64>::builder().tol(-1e-4).build::<f64>(),
        Err(BuildError::InvalidTol { .. })
    ));
    assert!(matches!(
        Ridge::<f64>::builder().tol(f64::NAN).build::<f64>(),
        Err(BuildError::InvalidTol { .. })
    ));
    // tol == 0 is ACCEPTED (sklearn's interval is closed on the left).
    assert!(Ridge::<f64>::builder().tol(0.0).build::<f64>().is_ok());
    // max_iter == 0 (sklearn: Interval(Integral, 1, None, closed="left")).
    assert!(matches!(
        Ridge::<f64>::builder().max_iter(Some(0)).build::<f64>(),
        Err(BuildError::InvalidMaxIter { .. })
    ));
    // solver='lbfgs' without positive=True.
    assert!(matches!(
        Ridge::<f64>::builder()
            .solver(RidgeSolver::Lbfgs)
            .build::<f64>(),
        Err(BuildError::LbfgsRequiresPositive { .. })
    ));
    // positive=True with a solver that cannot carry the bound.
    for solver in [
        RidgeSolver::Cholesky,
        RidgeSolver::Svd,
        RidgeSolver::Lsqr,
        RidgeSolver::SparseCg,
        RidgeSolver::Sag,
        RidgeSolver::Saga,
    ] {
        assert!(
            matches!(
                Ridge::<f64>::builder()
                    .solver(solver)
                    .positive(true)
                    .build::<f64>(),
                Err(BuildError::PositiveUnsupportedSolver { .. })
            ),
            "positive=true with solver='{}' must be rejected",
            solver.name()
        );
    }
    // positive=True with 'auto' / 'lbfgs' is the accepted pair.
    assert!(Ridge::<f64>::builder().positive(true).build::<f64>().is_ok());
    assert!(Ridge::<f64>::builder()
        .solver(RidgeSolver::Lbfgs)
        .positive(true)
        .build::<f64>()
        .is_ok());
}

/// The sklearn solver STRINGS parse to the enum, and an unknown one is a typed
/// `UnknownSolver` (the PyO3 boundary's `ValueError`).
#[test]
fn ridge_solver_string_parse() {
    let expected = [
        ("auto", RidgeSolver::Auto),
        ("svd", RidgeSolver::Svd),
        ("cholesky", RidgeSolver::Cholesky),
        ("lsqr", RidgeSolver::Lsqr),
        ("sparse_cg", RidgeSolver::SparseCg),
        ("sag", RidgeSolver::Sag),
        ("saga", RidgeSolver::Saga),
        ("lbfgs", RidgeSolver::Lbfgs),
    ];
    for (name, want) in expected {
        let got = RidgeSolver::try_from(name).expect("sklearn solver name must parse");
        assert_eq!(got, want, "solver '{name}' parsed to '{}'", got.name());
        assert_eq!(got.name(), name, "name() must round-trip");
    }
    assert!(matches!(
        RidgeSolver::try_from("newton-cholesky"),
        Err(BuildError::UnknownSolver { .. })
    ));
}

// ---------------------------------------------------------------------------
// solver='svd' ABOVE the Jacobi caps — the Gram+eig arm (`solve_svd_gram_eig`)
// ---------------------------------------------------------------------------
//
// The committed fixture is 40×5, which sits inside the one-sided Jacobi SVD
// kernel's shape caps, so it exercises the `svd`-prim arm ONLY. The
// `n_samples > 256` fallback is a genuinely different code path (Gram + `eig`
// instead of `U/σ/Vᵀ`) and would otherwise ship untested. These two tests cover
// it WITHOUT a new oracle fixture, by asserting it agrees with the `cholesky`
// solve of the same data — the two are algebraically identical for `α > 0`
// (module docs), so any disagreement is a real defect in one of them.

/// Rows above `SVD_JACOBI_MAX_ROWS` (256) so `solver='svd'` takes the Gram+eig
/// route; `d` well inside the eig order cap.
const LARGE_N: usize = 300;
const LARGE_D: usize = 5;

/// splitmix64 — the deterministic host data source the perf probes and
/// `scripts/bench_linear.py` already share, so this test needs no fixture.
fn splitmix64(seed: u64, i: u64) -> u64 {
    let mut z = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform_pm1(seed: u64, count: usize) -> Vec<f64> {
    (1..=count as u64)
        .map(|i| ((splitmix64(seed, i) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0)
        .collect()
}

/// A well-conditioned `LARGE_N × LARGE_D` regression problem.
fn large_problem<F>() -> (Vec<F>, Vec<F>)
where
    F: Float + CubeElement + Pod,
{
    let x = uniform_pm1(42, LARGE_N * LARGE_D);
    let coef = uniform_pm1(43, LARGE_D);
    let noise = uniform_pm1(44, LARGE_N);
    let y: Vec<f64> = (0..LARGE_N)
        .map(|r| {
            let dot: f64 = (0..LARGE_D).map(|c| x[r * LARGE_D + c] * coef[c]).sum();
            dot + 0.5 + 0.01 * noise[r]
        })
        .collect();
    (
        x.iter().map(|&v| f64_to::<F>(v)).collect(),
        y.iter().map(|&v| f64_to::<F>(v)).collect(),
    )
}

/// Fit the large problem with `solver` and return `(coef_, intercept_)`.
fn fit_large<F>(solver: RidgeSolver) -> (Vec<f64>, f64)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x_host, y_host) = large_problem::<F>();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);

    let fitted = Ridge::<F>::builder()
        .alpha(1.0)
        .solver(solver)
        .build::<F>()
        .expect("builds")
        .fit_with_sample_weight(&mut pool, &x_dev, Some(&y_dev), (LARGE_N, LARGE_D), None)
        .unwrap_or_else(|e| panic!("solver '{}' must fit the large problem: {e}", solver.name()));

    let coef = fitted.coef(&pool).iter().map(|&v| host_to_f64(v)).collect();
    (coef, host_to_f64(fitted.intercept(&pool)))
}

/// f64 (cpu): above the Jacobi caps, `solver='svd'` must still agree with
/// `solver='cholesky'` to the strict oracle tolerance.
///
/// This case is what surfaced the wgpu f64 `eig` defect: the kernel's shared
/// tiles need 66 048 B at f64 against that adapter's 65 536 B budget, so
/// pipeline creation failed SILENTLY and the all-zero output read back as
/// `NotConverged`. Fixed in `prims::eig` by routing to the existing host arm
/// when the tiles do not fit (`eig_shared_memory_fits`); it also reddened
/// `eig_symmetric_f64_fixture` and `linear_regression_large_*_f64`, which are
/// green again too. Kept un-gated on purpose — this is the test that would
/// catch a regression of that dispatch.
#[test]
fn ridge_svd_gram_eig_matches_cholesky_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge svd-large f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (svd_coef, svd_b) = fit_large::<f64>(RidgeSolver::Svd);
    let (chol_coef, chol_b) = fit_large::<f64>(RidgeSolver::Cholesky);
    assert_close(
        &svd_coef,
        &chol_coef,
        &F64_TOL,
        "svd (Gram+eig arm, n>256) coef_ vs cholesky",
    );
    assert_close(
        &[svd_b],
        &[chol_b],
        &F64_TOL,
        "svd (Gram+eig arm, n>256) intercept_ vs cholesky",
    );
}

/// f32 (the wgpu/rocm gate): the same agreement, at the LOOSER tolerance the
/// Gram+eig route earns.
///
/// This is deliberately not the 1e-5 gate. Forming the Gram squares `X`'s
/// condition number, so at f32 (`eps ≈ 1.2e-7`) the eigenvector accuracy of the
/// `d×d` decomposition degrades exactly as `linear_regression.rs::fit_gram_eig`
/// documents for its own large-`n` path — the same tradeoff cuML's
/// `algorithm='eig'` carries. `1e-3` still catches a WRONG arm (a mis-scaled
/// `1/(λ+α)`, a missed `Vᵀ` transpose, a dropped direction) by orders of
/// magnitude, while not encoding f32 Gram noise as a correctness requirement.
#[test]
fn ridge_svd_gram_eig_matches_cholesky_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let (svd_coef, svd_b) = fit_large::<f32>(RidgeSolver::Svd);
    let (chol_coef, chol_b) = fit_large::<f32>(RidgeSolver::Cholesky);
    let gram_eig_f32_tol = Tolerance { abs: 1e-3, rel: 1e-3 };
    assert_close(
        &svd_coef,
        &chol_coef,
        &gram_eig_f32_tol,
        "svd (Gram+eig arm, n>256) coef_ vs cholesky",
    );
    assert_close(
        &[svd_b],
        &[chol_b],
        &gram_eig_f32_tol,
        "svd (Gram+eig arm, n>256) intercept_ vs cholesky",
    );
}

/// BLDR-01: `Ridge::new()` equals `Ridge::builder().build()?` across the FULL
/// hyperparameter set (sklearn's defaults). Pure host — no device, so no f64
/// gate. `ridge_test.rs` carries the same assert for the original two-parameter
/// surface; this one would catch a new parameter added to `new()` but not to
/// `into_builder`/`build` (the D-08 single-source contract).
#[test]
fn defaults_equal_full_surface() {
    let from_new = Ridge::<f64>::new();
    let from_builder = Ridge::<f64>::builder()
        .build::<f64>()
        .expect("default RidgeBuilder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "Ridge::new() and builder().build()? must agree on ALL hyperparameters"
    );
}
