//! Plan 05-05 — coordinate-descent primitive standalone oracle.
//!
//! Drives `prims::coordinate_descent::cd_solve` against the committed sklearn
//! `lasso_{f32,f64}_seed42.npz` and `elastic_net_{f32,f64}_seed42.npz` fixtures.
//! The shared CD kernel serves BOTH Lasso (`l1_ratio=1` → `l2_reg=0`) and
//! ElasticNet (D-03), so this prim oracle exercises both families.
//!
//! ## sklearn `fit_intercept=True` ⇒ center before CD (Pitfall 1)
//! sklearn's `Lasso`/`ElasticNet` with `fit_intercept=True` CENTER `X` (per-column
//! mean) and `y` (mean) before running `enet_coordinate_descent`, fit `coef_` on
//! the centered design, then recover the intercept from the means. `cd_solve`
//! solves the centered problem, so the oracle centers `(X, y)` the same way and
//! compares `coef` to the fixture `coef_` within 1e-5 INCLUDING the exact-zero
//! sparsity pattern (the soft-threshold zeroing — Pitfall 1).
//!
//! ## Penalty mapping (pinned, `_coordinate_descent.py:781-782`)
//! `(l1_reg, l2_reg) = (α·l1_ratio·n, α·(1−l1_ratio)·n)`; Lasso has `l1_ratio=1`.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log, D-07). Per AGENTS.md §2 tests live here, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::coordinate_descent::cd_solve;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// CD fixture geometry (gen_oracle.py CD_N_SAMPLES × CD_N_FEATURES).
const CD_N_SAMPLES: usize = 50;
const CD_N_FEATURES: usize = 8;

/// sklearn CD hyperparameters pinned in gen_oracle.py (LASSO_ALPHA = 0.5,
/// EN_ALPHA = 0.5, EN_L1_RATIO = 0.5). Lasso is `l1_ratio = 1`.
const LASSO_ALPHA: f64 = 0.5;
const EN_ALPHA: f64 = 0.5;
const EN_L1_RATIO: f64 = 0.5;

/// sklearn CD stopping constants (`tol`, `max_iter`).
const CD_TOL: f64 = 1e-4;
const CD_MAX_ITER: usize = 1000;

/// `coef_` 1e-5 contract; exact-zero entries must be EXACTLY zero (sparsity).
const COEF_TOL: f64 = 1e-5;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn assert_len(case: &OracleCase, name: &str, len: usize) {
    let got = case.expect_f64(name).len();
    assert_eq!(
        got, len,
        "fixture array '{name}' should have {len} elements, got {got}"
    );
}

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("cd tests are f32/f64 only"),
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("cd tests are f32/f64 only"),
    }
}

/// Shared oracle body: load the fixture, CENTER `(X, y)` (sklearn
/// `fit_intercept=True`), map `(α, l1_ratio)` → `(l1_reg, l2_reg)`, run
/// `cd_solve`, and assert every `coef[j]` matches the fixture `coef_[j]` within
/// 1e-5 INCLUDING the exact-zero sparsity pattern.
fn check_cd<F>(fixture_name: &str, alpha: f64, l1_ratio: f64)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load cd fixture");
    let x_raw: Vec<f64> = case.expect_f64("X").to_vec(); // n × d row-major
    let y_raw: Vec<f64> = case.expect_f64("y").to_vec(); // length n
    let ref_coef: Vec<f64> = case.expect_f64("coef").to_vec(); // length d
    let n = CD_N_SAMPLES;
    let d = CD_N_FEATURES;
    assert_eq!(x_raw.len(), n * d, "X geometry");
    assert_eq!(y_raw.len(), n, "y geometry");
    assert_eq!(ref_coef.len(), d, "coef geometry");

    // --- Center X (per-column mean) and y (mean), exactly as sklearn does for
    //     fit_intercept=True before enet_coordinate_descent. ---
    let mut col_mean = vec![0.0f64; d];
    for j in 0..d {
        let mut s = 0.0;
        for i in 0..n {
            s += x_raw[i * d + j];
        }
        col_mean[j] = s / n as f64;
    }
    let y_mean: f64 = y_raw.iter().sum::<f64>() / n as f64;
    let x_centered: Vec<F> = (0..n * d)
        .map(|idx| {
            let j = idx % d;
            from_f64::<F>(x_raw[idx] - col_mean[j])
        })
        .collect();
    let y_centered: Vec<F> = y_raw.iter().map(|&v| from_f64::<F>(v - y_mean)).collect();

    // Penalty mapping (_coordinate_descent.py:781-782): l1_reg/l2_reg un-normalized.
    let l1_reg = alpha * l1_ratio * n as f64;
    let l2_reg = alpha * (1.0 - l1_ratio) * n as f64;

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_centered);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_centered);

    let coef_dev = cd_solve::<F>(
        &mut pool,
        &x_dev,
        &y_dev,
        n,
        d,
        l1_reg,
        l2_reg,
        CD_TOL,
        CD_MAX_ITER,
    )
    .expect("cd_solve");
    let got_coef: Vec<f64> = coef_dev
        .to_host(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    assert_eq!(got_coef.len(), d, "cd_solve coef length");

    for j in 0..d {
        let got = got_coef[j];
        let want = ref_coef[j];
        if want == 0.0 {
            // Pitfall 1: an exact-zero sklearn coefficient must be EXACTLY zero
            // (the soft-threshold `max(|t|−l1_reg,0)` zeroing), not merely small.
            assert_eq!(
                got, 0.0,
                "{fixture_name} coef[{j}] must be EXACTLY zero (sparsity, Pitfall 1), got {got}"
            );
        } else {
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1e-12);
            assert!(
                abs <= COEF_TOL || rel <= COEF_TOL,
                "{fixture_name} coef[{j}] mismatch: got {got}, want {want} (abs {abs:e}, rel {rel:e})"
            );
        }
    }
}

/// LOAD-NOT-JUST-PRESENT: BOTH the `lasso` and `elastic_net` fixtures load with
/// well-formed X/y/coef/intercept arrays (the shared CD kernel serves both, D-03).
#[test]
fn fixture_loads() {
    let lasso = load_npz(fixture("lasso_f64_seed42.npz")).expect("load lasso_f64");
    assert_len(&lasso, "X", CD_N_SAMPLES * CD_N_FEATURES);
    assert_len(&lasso, "y", CD_N_SAMPLES);
    assert_len(&lasso, "coef", CD_N_FEATURES);
    assert_len(&lasso, "intercept", 1);

    let en = load_npz(fixture("elastic_net_f64_seed42.npz")).expect("load elastic_net_f64");
    assert_len(&en, "X", CD_N_SAMPLES * CD_N_FEATURES);
    assert_len(&en, "coef", CD_N_FEATURES);
    assert_len(&en, "l1_ratio", 1);
}

/// CD soft-threshold reproduces the Lasso sparse `coef_` (exact zeros), f32.
#[test]
fn cd_lasso_coef_match_sklearn_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_cd::<f32>("lasso_f32_seed42.npz", LASSO_ALPHA, 1.0);
}

/// CD soft-threshold reproduces the Lasso sparse `coef_`, f64 (cpu runs; rocm
/// skips, D-07).
#[test]
fn cd_lasso_coef_match_sklearn_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    check_cd::<f64>("lasso_f64_seed42.npz", LASSO_ALPHA, 1.0);
}

/// CD residual update reproduces the ElasticNet `coef_`, f32.
#[test]
fn cd_elastic_net_coef_match_sklearn_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_cd::<f32>("elastic_net_f32_seed42.npz", EN_ALPHA, EN_L1_RATIO);
}

/// CD residual update reproduces the ElasticNet `coef_`, f64 (cpu runs; rocm
/// skips, D-07).
#[test]
fn cd_elastic_net_coef_match_sklearn_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    check_cd::<f64>("elastic_net_f64_seed42.npz", EN_ALPHA, EN_L1_RATIO);
}
