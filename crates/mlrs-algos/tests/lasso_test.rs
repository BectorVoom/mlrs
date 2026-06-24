//! Plan 05-09 — Lasso (LINEAR-03) sklearn oracle tests.
//!
//! Activated from the Wave-0 `#[ignore]` scaffold: each function loads the
//! committed `Lasso(fit_intercept=True)` fixture, fits the device estimator with
//! the fixture `alpha`, materializes `coef_`/`intercept_`, and asserts against the
//! sklearn reference within the 1e-5 abs+rel contract INCLUDING the exact
//! sparsity (zero) pattern (Pitfall 1). Lasso is the `l1_ratio = 1` case of
//! `ElasticNet` (D-03): it delegates to the SAME shared coordinate-descent helper
//! (`l2_reg = 0`, pure L1), mapping `α` → `l1_reg = α·n` and recovering the
//! unpenalized `intercept_ = ȳ − x̄·coef_` (D-13).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::lasso::Lasso;
use mlrs_algos::typestate::Fit;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// CD fixture geometry (gen_oracle.py CD_N_SAMPLES × CD_N_FEATURES).
const CD_N_SAMPLES: usize = 50;
const CD_N_FEATURES: usize = 8;

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
        _ => unreachable!("lasso fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("lasso fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened. For an exact-zero
/// expected entry the absolute arm forces `|got| ≤ atol`, so the sparsity pattern
/// (Pitfall 1) is checked literally, not merely "small".
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
        let allclose = abs_err <= tol.abs + tol.rel * e.abs();
        assert!(
            allclose,
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs, tol.rel
        );
    }
}

/// Fit `Lasso(alpha, fit_intercept=true)` on the fixture `(X, y)` and return host
/// `(coef_, intercept_)`.
fn fit_coef_intercept<F>(case: &OracleCase, alpha: f64) -> (Vec<f64>, f64)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case
        .expect_f64("X")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let y_host: Vec<F> = case
        .expect_f64("y")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);

    let reg = Lasso::<F>::builder()
        .alpha(alpha)
        .fit_intercept(true)
        .build::<F>()
        .expect("Lasso build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (CD_N_SAMPLES, CD_N_FEATURES))
        .expect("Lasso::fit on a valid shape");

    let coef = reg
        .coef(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let intercept = host_to_f64(reg.intercept(&pool));
    (coef, intercept)
}

/// Drive the Lasso oracle: fit with the fixture `alpha`, assert the SPARSE
/// `coef_` (incl. exact zeros) and `intercept_` vs sklearn within `tol`.
fn run_oracle<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let alpha = case.expect_f64("alpha")[0];
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    assert_eq!(coef_ref.len(), CD_N_FEATURES, "fixture coef length");
    assert_eq!(intercept_ref.len(), 1, "fixture intercept length");

    let (coef, intercept) = fit_coef_intercept::<F>(case, alpha);
    assert_close(
        &coef,
        coef_ref,
        tol,
        &format!("{label} sparse coef_ (incl. exact zeros) alpha={alpha}"),
    );
    assert_close(
        &[intercept],
        &[intercept_ref[0]],
        tol,
        &format!("{label} intercept_ alpha={alpha}"),
    );
}

/// LOAD-NOT-JUST-PRESENT: the `lasso` fixture loads with well-formed
/// X/y/alpha/coef/intercept.
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("lasso_f64_seed42.npz")).expect("load lasso_f64");
    assert_eq!(case.expect_f64("X").len(), CD_N_SAMPLES * CD_N_FEATURES);
    assert_eq!(case.expect_f64("y").len(), CD_N_SAMPLES);
    assert_eq!(case.expect_f64("alpha").len(), 1);
    assert_eq!(case.expect_f64("coef").len(), CD_N_FEATURES);
    assert_eq!(case.expect_f64("intercept").len(), 1);
}

/// sparse coef_ (exact zeros) + intercept_ match sklearn, f32.
#[test]
fn lasso_sparse_coef_match_sklearn_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("lasso_f32_seed42.npz")).expect("load lasso_f32");
    run_oracle::<f32>(&case, &F32_TOL, "lasso f32");
}

/// coef_/intercept_ match sklearn, f64 (cpu runs; rocm skips).
#[test]
fn lasso_coef_intercept_match_sklearn_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("lasso f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("lasso_f64_seed42.npz")).expect("load lasso_f64");
    run_oracle::<f64>(&case, &F64_TOL, "lasso f64");
}

/// BLDR-01 defaults equality: the zero-arg `new()` (sklearn defaults
/// alpha=1.0/fit_intercept=true/max_iter=1000/tol=1e-4) reproduces every
/// hyperparameter of `builder().build()` — the single-source-of-defaults
/// invariant (D-08). Exercises that the builder subsumes the former `new` /
/// `with_opts` constructors with matching defaults.
#[test]
fn defaults_equal() {
    let from_new = Lasso::<f64>::new();
    let from_builder = Lasso::<f64>::builder()
        .build::<f64>()
        .expect("default Lasso builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "Lasso::new() must equal Lasso::builder().build()"
    );
}
