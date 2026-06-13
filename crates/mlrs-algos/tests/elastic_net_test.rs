//! Plan 05-09 — ElasticNet (LINEAR-04) sklearn oracle tests.
//!
//! Activated from the Wave-0 `#[ignore]` scaffold: each function loads the
//! committed `ElasticNet(fit_intercept=True)` fixture, fits the device estimator
//! with the fixture `alpha`/`l1_ratio`, materializes `coef_`/`intercept_`, and
//! asserts against the sklearn reference within the 1e-5 abs+rel contract
//! INCLUDING the exact sparsity (zero) pattern (Pitfall 1). ElasticNet fits via
//! the shared coordinate-descent helper on the centered design, mapping
//! `(α, l1_ratio)` → `(l1_reg = α·l1_ratio·n, l2_reg = α·(1−l1_ratio)·n)` and
//! recovering the unpenalized `intercept_ = ȳ − x̄·coef_` (D-13).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::elastic_net::ElasticNet;
use mlrs_algos::traits::Fit;
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
        _ => unreachable!("elastic_net fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("elastic_net fixtures are f32/f64 only"),
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

/// Fit `ElasticNet(alpha, l1_ratio, fit_intercept=true)` on the fixture `(X, y)`
/// and return host `(coef_, intercept_)`.
fn fit_coef_intercept<F>(case: &OracleCase, alpha: f64, l1_ratio: f64) -> (Vec<f64>, f64)
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

    let mut reg = ElasticNet::<F>::new(f64_to::<F>(alpha), f64_to::<F>(l1_ratio), true);
    reg.fit(&mut pool, &x_dev, Some(&y_dev), (CD_N_SAMPLES, CD_N_FEATURES))
        .expect("ElasticNet::fit on a valid shape");

    let coef = reg
        .coef(&pool)
        .expect("coef_ after fit")
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let intercept = host_to_f64(reg.intercept(&pool).expect("intercept_ after fit"));
    (coef, intercept)
}

/// Drive the ElasticNet oracle: fit with the fixture `alpha`/`l1_ratio`, assert
/// `coef_` (incl. exact zeros) and `intercept_` vs sklearn within `tol`.
fn run_oracle<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let alpha = case.expect_f64("alpha")[0];
    let l1_ratio = case.expect_f64("l1_ratio")[0];
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    assert_eq!(coef_ref.len(), CD_N_FEATURES, "fixture coef length");
    assert_eq!(intercept_ref.len(), 1, "fixture intercept length");

    let (coef, intercept) = fit_coef_intercept::<F>(case, alpha, l1_ratio);
    assert_close(
        &coef,
        coef_ref,
        tol,
        &format!("{label} coef_ (incl. sparsity) alpha={alpha} l1_ratio={l1_ratio}"),
    );
    assert_close(
        &[intercept],
        &[intercept_ref[0]],
        tol,
        &format!("{label} intercept_ alpha={alpha} l1_ratio={l1_ratio}"),
    );
}

/// LOAD-NOT-JUST-PRESENT: the `elastic_net` fixture loads with well-formed
/// X/y/alpha/l1_ratio/coef/intercept.
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("elastic_net_f64_seed42.npz")).expect("load elastic_net_f64");
    assert_eq!(case.expect_f64("X").len(), CD_N_SAMPLES * CD_N_FEATURES);
    assert_eq!(case.expect_f64("y").len(), CD_N_SAMPLES);
    assert_eq!(case.expect_f64("alpha").len(), 1);
    assert_eq!(case.expect_f64("l1_ratio").len(), 1);
    assert_eq!(case.expect_f64("coef").len(), CD_N_FEATURES);
    assert_eq!(case.expect_f64("intercept").len(), 1);
}

/// coef_ (incl. exact zeros) + intercept_ match sklearn, f32.
#[test]
fn elastic_net_coef_match_sklearn_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("elastic_net_f32_seed42.npz")).expect("load elastic_net_f32");
    run_oracle::<f32>(&case, &F32_TOL, "elastic_net f32");
}

/// coef_/intercept_ match sklearn, f64 (cpu runs; rocm skips).
#[test]
fn elastic_net_coef_intercept_match_sklearn_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("elastic_net f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("elastic_net_f64_seed42.npz")).expect("load elastic_net_f64");
    run_oracle::<f64>(&case, &F64_TOL, "elastic_net f64");
}
