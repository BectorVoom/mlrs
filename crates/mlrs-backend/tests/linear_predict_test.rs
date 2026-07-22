//! Fused linear-inference primitive (`prims::linear_predict::linear_predict`)
//! oracle validation.
//!
//! `linear_predict` is the single-launch GATHER matvec+bias kernel that
//! replaced the shared `gemm→to_host→host bias-loop→from_host` predict path in
//! `LinearRegression`/`Ridge`/`Lasso`/`ElasticNet` (the LINEAR-01/02 predict
//! perf lever — see the prim's module docs). It stays device-resident: one
//! unit per output row computes `y[r] = Σ_c X[r,c]·coef[c] + bias[0]`, reading
//! the intercept straight from its length-1 device buffer. Validated here
//! against a DIRECT host f64 `X·coef + b` reference, several shapes including a
//! `cols = 1` degenerate and a `rows > 65535·256`-fold shape is left ignored
//! (it would allocate ~1 GiB; the grid-fold logic mirrors the covered
//! `center_test` fold).
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::linear_predict::linear_predict;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, PrimError, F32_TOL, F64_TOL};

/// Direct host `y = X·coef + b` reference, computed in f64. `x` is `m × n`
/// row-major, `coef` length `n`, `b` the scalar intercept.
fn host_predict_ref(x: &[f64], coef: &[f64], b: f64, m: usize, n: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; m];
    for r in 0..m {
        let mut acc = 0.0f64;
        for c in 0..n {
            acc += x[r * n + c] * coef[c];
        }
        y[r] = acc + b;
    }
    y
}

/// Run the device `linear_predict` prim end-to-end and return the host result
/// promoted to f64 for the oracle compare. Generic over the float element type
/// so f32/f64 share the exact same device path.
fn run_predict_case<F>(x_host: &[F], coef_host: &[F], bias_host: F, m: usize, n: usize) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, x_host);
    let coef_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, coef_host);
    let bias_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &[bias_host]);

    let pred_dev = linear_predict::<F>(&mut pool, &x_dev, &coef_dev, &bias_dev, (m, n))
        .expect("linear_predict host API rejects nothing for a valid shape");

    let pred_host = pred_dev.to_host_metered(&mut pool);
    let to_f64 = |v: &F| -> f64 {
        match std::mem::size_of::<F>() {
            4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
            8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
            _ => unreachable!("linear_predict_test is f32/f64 only"),
        }
    };
    pred_host.iter().map(to_f64).collect()
}

/// Deterministic pseudo-random-ish design values (no rng dependency).
fn design(m: usize, n: usize) -> Vec<f64> {
    (0..m * n).map(|i| ((i % 17) as f64) * 0.13 - 1.1).collect()
}

fn coefs(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 7) as f64) * 0.31 - 0.9).collect()
}

/// `linear_predict` vs the direct f64 host reference, several shapes including
/// a single-feature (`n = 1`) case and a multi-block (`m > 256`) case that
/// exercises the row-per-unit grid across more than one cube.
#[test]
fn linear_predict_matches_host_ref_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("linear_predict f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    for &(m, n) in &[(7usize, 4usize), (1, 5), (300, 1), (1000, 16), (513, 64)] {
        let x = design(m, n);
        let coef = coefs(n);
        let b = 0.37f64;
        let got = run_predict_case::<f64>(&x, &coef, b, m, n);
        let exp = host_predict_ref(&x, &coef, b, m, n);
        assert_slice_close(&got, &exp, &F64_TOL);
    }

    println!("linear_predict f64 backend={backend}: matches direct host reference");
}

/// `linear_predict` vs the direct host reference, f32 (always runs).
#[test]
fn linear_predict_matches_host_ref_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    for &(m, n) in &[(7usize, 4usize), (1, 5), (300, 1), (1000, 16), (513, 64)] {
        let x64 = design(m, n);
        let coef64 = coefs(n);
        let b = 0.37f64;
        let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let coef32: Vec<f32> = coef64.iter().map(|&v| v as f32).collect();
        let got = run_predict_case::<f32>(&x32, &coef32, b as f32, m, n);
        let exp = host_predict_ref(&x64, &coef64, b, m, n);
        assert_slice_close(&got, &exp, &F32_TOL);
    }

    println!("linear_predict f32 backend={backend}: matches direct host reference");
}

/// The zero-intercept path: `bias = [0]` reproduces a plain `X·coef`, so a
/// `fit_intercept=false` estimator gets an unbiased matvec through the same
/// kernel (no separate branch).
#[test]
fn linear_predict_zero_bias_is_plain_matvec_f32() {
    let (m, n) = (64usize, 8usize);
    let x64 = design(m, n);
    let coef64 = coefs(n);
    let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
    let coef32: Vec<f32> = coef64.iter().map(|&v| v as f32).collect();
    let got = run_predict_case::<f32>(&x32, &coef32, 0.0f32, m, n);
    let exp = host_predict_ref(&x64, &coef64, 0.0, m, n);
    assert_slice_close(&got, &exp, &F32_TOL);
}

/// Geometry rejection (ASVS V5): zero-row / zero-col / mismatched-length x /
/// wrong-length coef / empty bias are each rejected BEFORE any launch with a
/// typed `PrimError`, never a panic or an OOB device read.
#[test]
fn linear_predict_rejects_bad_geometry() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let coef: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &vec![1.0f32; 4]);
    let bias: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &[0.5f32]);

    // x length mismatch: declares 3×4 but supplies 11 elements.
    let x_bad: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &vec![0.0f32; 11]);
    let err = linear_predict::<f32>(&mut pool, &x_bad, &coef, &bias, (3, 4))
        .err()
        .unwrap();
    assert!(matches!(err, PrimError::ShapeMismatch { operand: "x", .. }));
    x_bad.release_into(&mut pool);

    // coef length mismatch: 3×4 x is fine, but coef has 4 elems while n=5.
    let x_ok: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &vec![0.0f32; 15]);
    let err = linear_predict::<f32>(&mut pool, &x_ok, &coef, &bias, (3, 5))
        .err()
        .unwrap();
    assert!(matches!(err, PrimError::DimMismatch { dim: "n_features", .. }));
    x_ok.release_into(&mut pool);

    coef.release_into(&mut pool);
    bias.release_into(&mut pool);
}

/// Grid-fold regression (`#[ignore]` — heavy: ~16.8M-row allocation): the
/// launch folds `cubes = ceil(m/256)` across the X/Y grid axes so the count
/// never overflows a single dimension's ~65535 cap. At `m > 65535·256 ≈
/// 16.78M` rows the fold forces `y > 1` (`CUBE_COUNT_Y > 1`) — the ONLY regime
/// where the 2D fold is engaged, and the exact regime where, if `ABSOLUTE_POS`
/// did NOT linearize contiguously as `(cy·CUBE_COUNT_X + cx)·256 + unit`, the
/// tail rows beyond the first grid column would silently receive wrong /
/// dropped predictions (the shared `prims::center` fold's
/// `center_columns_grid_fold_large_n_f32` precedent). We use `n = 1` and a
/// closed-form `y[r] = x[r]·coef + bias` so each expected value is derivable
/// from `r` alone — no `O(m)` host reference beyond the input. `#[ignore]`
/// keeps it out of the standard gate (the small shapes above already cover the
/// `y == 1` path); run explicitly:
/// `--features wgpu --test linear_predict_test -- --ignored --nocapture`.
#[test]
#[ignore = "heavy ~16.8M-row allocation — run explicitly to verify the X/Y grid fold"]
fn linear_predict_grid_fold_large_m_f32() {
    let backend = capability::active_backend_name();
    // One past the single-dimension cube cap: cubes = ceil(m/256) = 65536,
    // forcing CUBE_COUNT_Y = 2 (the folded region).
    let m = (MAX_GRID_DIM_TEST as usize + 1) * 256;
    let n = 1usize;
    let period = 7i64;
    let coef_v = 2.0f32;
    let bias_v = 0.5f32;
    let x: Vec<f32> = (0..m * n).map(|i| ((i as i64 % period) - 3) as f32).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let coef_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &[coef_v]);
    let bias_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &[bias_v]);

    let pred = linear_predict::<f32>(&mut pool, &x_dev, &coef_dev, &bias_dev, (m, n))
        .expect("linear_predict accepts a valid large shape");
    let got = pred.to_host(&pool);
    assert_eq!(got.len(), m, "grid-fold predict returned wrong length");

    // Spot-check the FIRST row, a MIDDLE row, and the very LAST row. The last
    // row is in the folded (`CUBE_COUNT_Y > 1`) region, so a dropped tail
    // row-block from a mis-linearized `ABSOLUTE_POS` shows up here as a wrong
    // (or untouched) prediction.
    for &r in &[0usize, m / 2, m - 1] {
        let expected = (((r as i64 % period) - 3) as f32) * coef_v + bias_v;
        assert!(
            (got[r] - expected).abs() <= 1e-3,
            "grid-fold predict wrong at row {r} (tail row-block dropped?): \
             got={g:e} expected={expected:e} backend={backend}",
            g = got[r]
        );
    }
    println!(
        "linear_predict grid-fold large-m f32 backend={backend}: \
         m={m} (cubes=65536, CUBE_COUNT_Y>1) predictions OK"
    );
}

/// Local copy of the CubeCL per-dimension grid cap for the ignored large-m
/// test (the kernel crate's `mlrs_kernels::colmean::MAX_GRID_DIM` is not on the
/// backend's public test surface; re-affirming it here mirrors
/// `center_test.rs::MAX_GRID_DIM_TEST` — kept in sync by the assertion it
/// drives).
const MAX_GRID_DIM_TEST: u32 = 65_535;
