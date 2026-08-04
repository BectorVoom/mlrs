//! `preprocessing` (PREP-01, Phase 24) sklearn oracle tests.
//!
//! Each function loads its committed fixture (`scripts/gen_oracle.py`'s
//! `gen_standard_scaler`/`gen_min_max_scaler`/`gen_max_abs_scaler`/
//! `gen_robust_scaler`/`gen_normalizer`/`gen_binarizer`, each pinning one
//! CONSTANT or all-zero column/row so the degenerate zero-scale gate is
//! exercised), fits the device estimator, and asserts the fitted attributes +
//! `transform` (+ `inverse_transform` where sklearn supports it) against the
//! sklearn reference within the 1e-5 abs+rel contract (D-09). No sign
//! ambiguity here (unlike PCA/SVD) — every fitted attribute compares directly,
//! no `align_rows`.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::preprocessing::normalizer::Norm;
use mlrs_algos::preprocessing::{Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler};
use mlrs_algos::typestate::{Fit, Transform};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

const N_SAMPLES: usize = 60;
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
        _ => unreachable!("preprocessing fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("preprocessing fixtures are f32/f64 only"),
    }
}

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
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs, tol.rel
        );
    }
}

fn load_x<F: Float + CubeElement + Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    case: &OracleCase,
) -> DeviceArray<ActiveRuntime, F> {
    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    DeviceArray::from_host(pool, &x_host)
}

fn promote<F: Pod>(v: Vec<F>) -> Vec<f64> {
    v.iter().map(|&x| host_to_f64(x)).collect()
}

// ===========================================================================
// StandardScaler
// ===========================================================================

fn run_standard_scaler<F: Float + CubeElement + Pod>(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = load_x::<F>(&mut pool, case);

    let fitted = StandardScaler::<F>::new()
        .fit(&mut pool, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("StandardScaler::fit");
    assert_close(&promote(fitted.mean(&pool)), case.expect_f64("mean_"), tol, &format!("mean_ {tag}"));
    assert_close(&promote(fitted.var(&pool)), case.expect_f64("var_"), tol, &format!("var_ {tag}"));
    assert_close(&promote(fitted.scale(&pool)), case.expect_f64("scale_"), tol, &format!("scale_ {tag}"));

    let z = fitted.transform(&mut pool, &x, (N_SAMPLES, N_FEATURES)).expect("transform");
    assert_close(&promote(z.to_host(&pool)), case.expect_f64("transform"), tol, &format!("transform {tag}"));

    let inv = fitted.inverse_transform(&mut pool, &z, (N_SAMPLES, N_FEATURES)).expect("inverse_transform");
    assert_close(&promote(inv.to_host(&pool)), case.expect_f64("inverse"), tol, &format!("inverse_transform {tag}"));
}

#[test]
fn standard_scaler_f32() {
    let case = load_npz(fixture("standard_scaler_f32_seed42.npz")).expect("load standard_scaler_f32");
    run_standard_scaler::<f32>(&case, &F32_TOL, "f32");
}

#[test]
fn standard_scaler_f64() {
    if capability::skip_f64_with_log() {
        println!("standard_scaler f64: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("standard_scaler_f64_seed42.npz")).expect("load standard_scaler_f64");
    run_standard_scaler::<f64>(&case, &F64_TOL, "f64");
}

// ===========================================================================
// MinMaxScaler
// ===========================================================================

fn run_min_max_scaler<F: Float + CubeElement + Pod>(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = load_x::<F>(&mut pool, case);
    let range = case.expect_f64("feature_range");

    let fitted = MinMaxScaler::<F>::builder()
        .feature_range(range[0], range[1])
        .build::<F>()
        .expect("MinMaxScaler::builder().build()")
        .fit(&mut pool, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("MinMaxScaler::fit");
    assert_close(&promote(fitted.data_min(&pool)), case.expect_f64("data_min_"), tol, &format!("data_min_ {tag}"));
    assert_close(&promote(fitted.data_max(&pool)), case.expect_f64("data_max_"), tol, &format!("data_max_ {tag}"));
    assert_close(&promote(fitted.scale(&pool)), case.expect_f64("scale_"), tol, &format!("scale_ {tag}"));
    assert_close(&promote(fitted.min(&pool)), case.expect_f64("min_"), tol, &format!("min_ {tag}"));

    let z = fitted.transform(&mut pool, &x, (N_SAMPLES, N_FEATURES)).expect("transform");
    assert_close(&promote(z.to_host(&pool)), case.expect_f64("transform"), tol, &format!("transform {tag}"));

    let inv = fitted.inverse_transform(&mut pool, &z, (N_SAMPLES, N_FEATURES)).expect("inverse_transform");
    assert_close(&promote(inv.to_host(&pool)), case.expect_f64("inverse"), tol, &format!("inverse_transform {tag}"));
}

#[test]
fn min_max_scaler_f32() {
    let case = load_npz(fixture("min_max_scaler_f32_seed42.npz")).expect("load min_max_scaler_f32");
    run_min_max_scaler::<f32>(&case, &F32_TOL, "f32");
}

#[test]
fn min_max_scaler_f64() {
    if capability::skip_f64_with_log() {
        println!("min_max_scaler f64: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("min_max_scaler_f64_seed42.npz")).expect("load min_max_scaler_f64");
    run_min_max_scaler::<f64>(&case, &F64_TOL, "f64");
}

// ===========================================================================
// MaxAbsScaler
// ===========================================================================

fn run_max_abs_scaler<F: Float + CubeElement + Pod>(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = load_x::<F>(&mut pool, case);

    let fitted = MaxAbsScaler::<F>::new()
        .fit(&mut pool, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("MaxAbsScaler::fit");
    assert_close(&promote(fitted.max_abs(&pool)), case.expect_f64("max_abs_"), tol, &format!("max_abs_ {tag}"));
    assert_close(&promote(fitted.scale(&pool)), case.expect_f64("scale_"), tol, &format!("scale_ {tag}"));

    let z = fitted.transform(&mut pool, &x, (N_SAMPLES, N_FEATURES)).expect("transform");
    assert_close(&promote(z.to_host(&pool)), case.expect_f64("transform"), tol, &format!("transform {tag}"));

    let inv = fitted.inverse_transform(&mut pool, &z, (N_SAMPLES, N_FEATURES)).expect("inverse_transform");
    assert_close(&promote(inv.to_host(&pool)), case.expect_f64("inverse"), tol, &format!("inverse_transform {tag}"));
}

#[test]
fn max_abs_scaler_f32() {
    let case = load_npz(fixture("max_abs_scaler_f32_seed42.npz")).expect("load max_abs_scaler_f32");
    run_max_abs_scaler::<f32>(&case, &F32_TOL, "f32");
}

#[test]
fn max_abs_scaler_f64() {
    if capability::skip_f64_with_log() {
        println!("max_abs_scaler f64: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("max_abs_scaler_f64_seed42.npz")).expect("load max_abs_scaler_f64");
    run_max_abs_scaler::<f64>(&case, &F64_TOL, "f64");
}

// ===========================================================================
// RobustScaler
// ===========================================================================

fn run_robust_scaler<F: Float + CubeElement + Pod>(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = load_x::<F>(&mut pool, case);

    let fitted = RobustScaler::<F>::new()
        .fit(&mut pool, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("RobustScaler::fit");
    assert_close(&promote(fitted.center(&pool)), case.expect_f64("center_"), tol, &format!("center_ {tag}"));
    assert_close(&promote(fitted.scale(&pool)), case.expect_f64("scale_"), tol, &format!("scale_ {tag}"));

    let z = fitted.transform(&mut pool, &x, (N_SAMPLES, N_FEATURES)).expect("transform");
    assert_close(&promote(z.to_host(&pool)), case.expect_f64("transform"), tol, &format!("transform {tag}"));

    let inv = fitted.inverse_transform(&mut pool, &z, (N_SAMPLES, N_FEATURES)).expect("inverse_transform");
    assert_close(&promote(inv.to_host(&pool)), case.expect_f64("inverse"), tol, &format!("inverse_transform {tag}"));
}

#[test]
fn robust_scaler_f32() {
    let case = load_npz(fixture("robust_scaler_f32_seed42.npz")).expect("load robust_scaler_f32");
    run_robust_scaler::<f32>(&case, &F32_TOL, "f32");
}

#[test]
fn robust_scaler_f64() {
    if capability::skip_f64_with_log() {
        println!("robust_scaler f64: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("robust_scaler_f64_seed42.npz")).expect("load robust_scaler_f64");
    run_robust_scaler::<f64>(&case, &F64_TOL, "f64");
}

// ===========================================================================
// Normalizer (l1 / l2 / max)
// ===========================================================================

fn run_normalizer<F: Float + CubeElement + Pod>(case: &OracleCase, tol: &Tolerance, norm: Norm, tag: &str) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = load_x::<F>(&mut pool, case);

    let fitted = Normalizer::<F>::with_norm(norm)
        .fit(&mut pool, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("Normalizer::fit");
    let z = fitted.transform(&mut pool, &x, (N_SAMPLES, N_FEATURES)).expect("transform");
    assert_close(&promote(z.to_host(&pool)), case.expect_f64("transform"), tol, &format!("transform {tag}"));
}

#[test]
fn normalizer_l1_f32() {
    let case = load_npz(fixture("normalizer_l1_f32_seed42.npz")).expect("load normalizer_l1_f32");
    run_normalizer::<f32>(&case, &F32_TOL, Norm::L1, "l1 f32");
}
#[test]
fn normalizer_l2_f32() {
    let case = load_npz(fixture("normalizer_l2_f32_seed42.npz")).expect("load normalizer_l2_f32");
    run_normalizer::<f32>(&case, &F32_TOL, Norm::L2, "l2 f32");
}
#[test]
fn normalizer_max_f32() {
    let case = load_npz(fixture("normalizer_max_f32_seed42.npz")).expect("load normalizer_max_f32");
    run_normalizer::<f32>(&case, &F32_TOL, Norm::Max, "max f32");
}
#[test]
fn normalizer_l2_f64() {
    if capability::skip_f64_with_log() {
        println!("normalizer f64: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("normalizer_l2_f64_seed42.npz")).expect("load normalizer_l2_f64");
    run_normalizer::<f64>(&case, &F64_TOL, Norm::L2, "l2 f64");
}

// ===========================================================================
// Binarizer
// ===========================================================================

fn run_binarizer<F: Float + CubeElement + Pod>(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = load_x::<F>(&mut pool, case);
    let threshold = case.expect_f64("threshold")[0];

    let fitted = Binarizer::<F>::with_threshold(threshold)
        .fit(&mut pool, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("Binarizer::fit");
    let z = fitted.transform(&mut pool, &x, (N_SAMPLES, N_FEATURES)).expect("transform");
    assert_close(&promote(z.to_host(&pool)), case.expect_f64("transform"), tol, &format!("transform {tag}"));
}

#[test]
fn binarizer_f32() {
    let case = load_npz(fixture("binarizer_f32_seed42.npz")).expect("load binarizer_f32");
    run_binarizer::<f32>(&case, &F32_TOL, "f32");
}

#[test]
fn binarizer_f64() {
    if capability::skip_f64_with_log() {
        println!("binarizer f64: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("binarizer_f64_seed42.npz")).expect("load binarizer_f64");
    run_binarizer::<f64>(&case, &F64_TOL, "f64");
}
