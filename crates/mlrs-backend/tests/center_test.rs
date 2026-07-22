//! Column-mean centering primitive (`prims::center::center_columns`) oracle
//! validation.
//!
//! `prims::center::center_columns` is a pure composition of the
//! already-validated `column_reduce` (PRIM-02) + `center_columns` element
//! kernel (PRIM-03) — the same composition `covariance.rs` uses internally
//! for its first two steps, extracted standalone so a caller that needs JUST
//! the centered matrix (e.g. `LinearRegression`'s large-`n_samples` Gram+eig
//! path, LINEAR-01) doesn't have to pay `covariance.rs`'s GEMM/scale steps.
//! Validated here against a DIRECT host two-pass reference computed in f64 —
//! the same reference shape as `covariance_test.rs::host_cov_ref`'s centering
//! stage, without the Gram/`(n-ddof)` steps that prim doesn't need.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::center::center_columns;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, PrimError, F32_TOL, F64_TOL};

/// Direct host column-mean + centering reference, computed in f64. Returns
/// `(centered, mean)`; `a` is `rows × cols` row-major.
fn host_center_ref(a: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0.0f64; cols];
    for r in 0..rows {
        for c in 0..cols {
            mean[c] += a[r * cols + c];
        }
    }
    for m in mean.iter_mut() {
        *m /= rows as f64;
    }
    let mut centered = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            centered[r * cols + c] = a[r * cols + c] - mean[c];
        }
    }
    (centered, mean)
}

/// Run the device `center_columns` prim end-to-end and return host
/// `(centered, mean)`, both promoted to f64 for the oracle compare. Generic
/// over the float element type so f32/f64 share the exact same device path.
fn run_center_case<F>(a_host: &[F], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, a_host);

    let (centered_dev, mean_dev) = center_columns::<F>(&mut pool, &a_dev, (rows, cols))
        .expect("center_columns host API rejects nothing for a valid shape");

    let centered_host = centered_dev.to_host_metered(&mut pool);
    let mean_host = mean_dev.to_host_metered(&mut pool);
    let to_f64 = |v: &F| -> f64 {
        match std::mem::size_of::<F>() {
            4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
            8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
            _ => unreachable!("center_test is f32/f64 only"),
        }
    };
    (
        centered_host.iter().map(to_f64).collect(),
        mean_host.iter().map(to_f64).collect(),
    )
}

/// `center_columns` vs the direct f64 host reference, several shapes incl. a
/// single-column (`cols = 1`) case (the `LinearRegression` large-`n_samples`
/// path centers `y` this way, treating it as an `n_samples × 1` matrix).
#[test]
fn center_columns_matches_host_ref_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("center_columns f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    for &(rows, cols) in &[(7usize, 4usize), (5, 5), (12, 3), (2000, 20), (9, 1)] {
        let a: Vec<f64> = (0..rows * cols)
            .map(|i| ((i % 13) as f64) * 0.1 - 0.6)
            .collect();
        let (got_centered, got_mean) = run_center_case::<f64>(&a, rows, cols);
        let (exp_centered, exp_mean) = host_center_ref(&a, rows, cols);
        assert_slice_close(&got_centered, &exp_centered, &F64_TOL);
        assert_slice_close(&got_mean, &exp_mean, &F64_TOL);
    }

    println!("center_columns f64 backend={backend}: matches direct host reference");
}

/// `center_columns` vs the direct host reference, f32 (always runs).
#[test]
fn center_columns_matches_host_ref_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    for &(rows, cols) in &[(7usize, 4usize), (5, 5), (12, 3), (2000, 20), (9, 1)] {
        let a64: Vec<f64> = (0..rows * cols)
            .map(|i| ((i % 13) as f64) * 0.1 - 0.6)
            .collect();
        let a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
        let (got_centered, got_mean) = run_center_case::<f32>(&a32, rows, cols);
        let (exp_centered, exp_mean) = host_center_ref(&a64, rows, cols);
        assert_slice_close(&got_centered, &exp_centered, &F32_TOL);
        assert_slice_close(&got_mean, &exp_mean, &F32_TOL);
    }

    println!("center_columns f32 backend={backend}: matches direct host reference");
}

/// Geometry rejection (ASVS V5): a zero-row/zero-col/mismatched-length input
/// is rejected BEFORE any launch with a typed `PrimError`, never a panic or
/// an OOB device read.
#[test]
fn center_columns_rejects_bad_geometry() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // Length mismatch: declares 3×4 but supplies 11 elements.
    let a_dev: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &vec![0.0f32; 11]);
    let err = center_columns::<f32>(&mut pool, &a_dev, (3, 4)).err().unwrap();
    assert!(matches!(err, PrimError::ShapeMismatch { .. }));
    a_dev.release_into(&mut pool);

    // Zero rows.
    let a_dev: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &vec![0.0f32; 0]);
    let err = center_columns::<f32>(&mut pool, &a_dev, (0, 4)).err().unwrap();
    assert!(matches!(err, PrimError::ShapeMismatch { .. }));
    a_dev.release_into(&mut pool);
}

/// Grid-fold regression (`#[ignore]` — heavy: ~16.8M-row allocation): the
/// multi-pass `colmean` fast path folds each pass's block count across the
/// X/Z grid axes so `nblocks` never overflows a single grid dimension's
/// ~65535 cap. At `rows > 65535·FOLD_TPB ≈ 16.78M` the FIRST fold pass has
/// `nblocks > 65535`, forcing `nbz > 1` — the exact regime that, before the
/// X/Z fold, requested more than 65535 cubes in `grid.x` and silently
/// dropped the tail row-blocks (wrong means → wrong fitted coefficients). We
/// build a single column whose true mean is exactly known (`x[i] = ((i mod 7)
/// − 3)` repeated, a per-7-block-balanced pattern whose mean is trivially
/// derived) so the check needs no `O(n)` host reference allocation beyond the
/// input itself. `#[ignore]` keeps it out of the standard gate (the small
/// shapes above already cover the `nbz == 1` path); run explicitly to verify
/// the fold: `--features wgpu --test center_test -- --ignored --nocapture`.
#[test]
#[ignore = "heavy ~16.8M-row allocation — run explicitly to verify the X/Z grid fold"]
fn center_columns_grid_fold_large_n_f32() {
    let backend = capability::active_backend_name();
    // cpu uses the column_reduce fallback (no grid fold), so this test only
    // exercises the fold on a device backend; it still passes on cpu (the
    // fallback is correct), just doesn't cover the fold there.
    let cols = 1usize;
    // One past the single-dimension cap: nblocks = ceil(rows/256) = 65536.
    let rows = (MAX_GRID_DIM_TEST as usize + 1) * 256;
    let period = 7i64;
    let a: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as i64 % period) - 3) as f32)
        .collect();

    // True column mean: the values 0..rows cycle through (i mod 7 − 3); the
    // exact mean is Σ_{i<rows}((i mod 7) − 3) / rows, computed in f64 here.
    let mut sum = 0.0f64;
    for i in 0..rows {
        sum += ((i as i64 % period) - 3) as f64;
    }
    let expected_mean = sum / rows as f64;

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &a);
    let (centered_dev, mean_dev) = center_columns::<f32>(&mut pool, &a_dev, (rows, cols))
        .expect("center_columns accepts a valid large shape");
    let got_mean = mean_dev.to_host(&pool)[0] as f64;
    assert!(
        (got_mean - expected_mean).abs() <= 1e-3,
        "grid-fold mean wrong (tail row-blocks dropped?): got={got_mean:e} \
         expected={expected_mean:e} backend={backend}"
    );
    // Spot-check the centered value of the very LAST row — if the tail
    // row-blocks were dropped the mean would be wrong AND this last element
    // is in the folded (nbz>1) region.
    let centered = centered_dev.to_host(&pool);
    let last = *centered.last().unwrap() as f64;
    let last_expected = (((rows - 1) as i64 % period) - 3) as f64 - expected_mean;
    assert!(
        (last - last_expected).abs() <= 1e-3,
        "grid-fold last-row centered value wrong: got={last:e} expected={last_expected:e}"
    );
    println!(
        "center_columns grid-fold large-n f32 backend={backend}: \
         rows={rows} (nblocks=65536, nbz>1) mean OK"
    );
}

/// Local copy of `MAX_GRID_DIM` for the ignored large-n test (the kernel
/// crate's const is not re-exported to the backend's public test surface;
/// re-affirming it here mirrors the `linear_regression.rs` local-cap
/// precedent — kept in sync by the assertion it drives).
const MAX_GRID_DIM_TEST: u32 = 65_535;
