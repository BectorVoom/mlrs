//! Gram/Xty primitive (`prims::gram::gram_xty`) oracle validation
//! (LINEAR-01 perf lever, D-02).
//!
//! `gram_xty` dispatches to a row-blocked shared-memory kernel pair on every
//! backend except cpu (which falls back to the original `gemm`-based
//! formation — `use_shared_gram`'s `#[cfg(feature = "cpu")]` gate). Running
//! this suite under BOTH `--features cpu` (exercises the `gemm` fallback) and
//! `--features wgpu` (exercises the shared-memory kernels) validates both
//! dispatch arms against the SAME direct host f64 reference.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gram::gram_xty;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, is_close, PrimError, Tolerance, F32_TOL, F64_TOL};

/// Off-diagonal Gram/Xty entries can legitimately cancel near zero across a
/// small sample (sign-mixed products), where `F32_TOL`'s strict rel check
/// (`1e-5`) is unstable purely from f32 rounding — the SAME category of issue
/// `covariance_test.rs`'s `F32_COV_NEAR_ZERO_FLOOR` documents, just triggered
/// by cancellation instead of the covariance normalisation. Raised well above
/// `NEAR_ZERO_FLOOR` (`1e-8`) to cover that band; never loosens `tol.abs`.
const F32_GRAM_NEAR_ZERO_FLOOR: f64 = 1e-2;

/// Element-wise f32 Gram/Xty oracle compare: strict abs-AND-rel per
/// `F32_TOL`, except abs-only (still bounded by `tol.abs`) when
/// `|expected| < F32_GRAM_NEAR_ZERO_FLOOR` (the `assert_slice_close_f32_cov`
/// precedent in `covariance_test.rs`).
fn assert_slice_close_f32_gram(got: &[f64], expected: &[f64], tol: &Tolerance) {
    assert_eq!(
        got.len(),
        expected.len(),
        "f32 gram oracle length mismatch: got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if e.abs() < F32_GRAM_NEAR_ZERO_FLOOR {
            let abs_err = (g - e).abs();
            assert!(
                abs_err <= tol.abs,
                "f32 gram near-zero abs check failed at index {i}: got={g:e}, expected={e:e}, \
                 abs_err={abs_err:e} (tol.abs={:e})",
                tol.abs
            );
        } else {
            assert!(
                is_close(g, e, tol),
                "f32 gram assert_close failed at index {i}: got={g:e}, expected={e:e}, \
                 abs_err={:e} (tol.abs={:e}, tol.rel={:e})",
                (g - e).abs(),
                tol.abs,
                tol.rel
            );
        }
    }
}

/// Direct host `gram = XᵀX` (`d×d`) + `xty = Xᵀy` (`d×1`) reference, computed
/// in f64. `x` is `n × d` row-major, `y` is length `n`.
fn host_gram_xty_ref(x: &[f64], y: &[f64], n: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut gram = vec![0.0f64; d * d];
    let mut xty = vec![0.0f64; d];
    for i in 0..n {
        for a in 0..d {
            xty[a] += x[i * d + a] * y[i];
            for b in 0..d {
                gram[a * d + b] += x[i * d + a] * x[i * d + b];
            }
        }
    }
    (gram, xty)
}

/// Run the device `gram_xty` prim end-to-end and return host `(gram, xty)`,
/// both promoted to f64 for the oracle compare.
fn run_gram_case<F>(x_host: &[F], y_host: &[F], n: usize, d: usize) -> (Vec<f64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, y_host);

    let (gram_dev, xty_dev) = gram_xty::<F>(&mut pool, &x_dev, &y_dev, n, d)
        .expect("gram_xty host API rejects nothing for a valid shape");

    let gram_host = gram_dev.to_host_metered(&mut pool);
    let xty_host = xty_dev.to_host_metered(&mut pool);
    let to_f64 = |v: &F| -> f64 {
        match std::mem::size_of::<F>() {
            4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
            8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
            _ => unreachable!("gram_test is f32/f64 only"),
        }
    };
    (
        gram_host.iter().map(to_f64).collect(),
        xty_host.iter().map(to_f64).collect(),
    )
}

/// Shapes exercised: small single-block cases, a `cols = 1` degenerate Gram,
/// a multi-row-block case (`n = 2000`), and `d = 64` (`GRAM_EIG_MAX_FEATURES`
/// — the shared-kernel's SharedMemory budget ceiling, `d*d = 4096`).
const SHAPES: &[(usize, usize)] = &[
    (7, 4),
    (5, 5),
    (12, 3),
    (9, 1),
    (2000, 20),
    (600, 64),
];

/// `gram_xty` vs the direct f64 host reference.
#[test]
fn gram_xty_matches_host_ref_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("gram_xty f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    for &(n, d) in SHAPES {
        let x: Vec<f64> = (0..n * d).map(|i| ((i % 17) as f64) * 0.1 - 0.8).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i % 11) as f64) * 0.2 - 1.0).collect();
        let (got_gram, got_xty) = run_gram_case::<f64>(&x, &y, n, d);
        let (exp_gram, exp_xty) = host_gram_xty_ref(&x, &y, n, d);
        assert_slice_close(&got_gram, &exp_gram, &F64_TOL);
        assert_slice_close(&got_xty, &exp_xty, &F64_TOL);
    }

    println!("gram_xty f64 backend={backend}: matches direct host reference");
}

/// `gram_xty` vs the direct host reference, f32 (always runs).
///
/// `gram`/`xty` are RAW (unscaled) sums over `n` rows (D-09 — no
/// `1/(n-ddof)` normalisation, unlike `covariance.rs`), so their magnitude
/// grows with `n`; `F32_TOL`'s `abs = 1e-5` is unrealistic for an f32
/// accumulation of hundreds of O(1)-magnitude terms regardless of HOW
/// correctly it's summed (the global `1e-5` policy assumes O(1)-magnitude
/// outputs — see `docs/tolerance-policy.md`). The input magnitude here is
/// scaled down so the raw sums stay small enough for the strict abs+rel
/// check to be meaningful (this only shrinks the OUTPUT magnitude, not the
/// relative rounding error the check actually probes for a bug).
#[test]
fn gram_xty_matches_host_ref_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    for &(n, d) in SHAPES {
        let x64: Vec<f64> = (0..n * d).map(|i| ((i % 17) as f64) * 0.002 - 0.016).collect();
        let y64: Vec<f64> = (0..n).map(|i| ((i % 11) as f64) * 0.004 - 0.02).collect();
        let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let y32: Vec<f32> = y64.iter().map(|&v| v as f32).collect();
        let (got_gram, got_xty) = run_gram_case::<f32>(&x32, &y32, n, d);
        let (exp_gram, exp_xty) = host_gram_xty_ref(&x64, &y64, n, d);
        assert_slice_close_f32_gram(&got_gram, &exp_gram, &F32_TOL);
        assert_slice_close_f32_gram(&got_xty, &exp_xty, &F32_TOL);
    }

    println!("gram_xty f32 backend={backend}: matches direct host reference");
}

/// Geometry rejection (ASVS V5): a zero-row/zero-col/mismatched-length input
/// is rejected BEFORE any launch with a typed `PrimError`, never a panic or
/// an OOB device read.
#[test]
fn gram_xty_rejects_bad_geometry() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // x length mismatch: declares 3×4 but supplies 11 elements.
    let x_dev: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &vec![0.0f32; 11]);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &vec![0.0f32; 3]);
    let err = gram_xty::<f32>(&mut pool, &x_dev, &y_dev, 3, 4).err().unwrap();
    assert!(matches!(err, PrimError::ShapeMismatch { .. }));
    x_dev.release_into(&mut pool);
    y_dev.release_into(&mut pool);

    // y length mismatch: n=5 but y has 4 elements.
    let x_dev: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &vec![0.0f32; 20]);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &vec![0.0f32; 4]);
    let err = gram_xty::<f32>(&mut pool, &x_dev, &y_dev, 5, 4).err().unwrap();
    assert!(matches!(err, PrimError::ShapeMismatch { .. }));
    x_dev.release_into(&mut pool);
    y_dev.release_into(&mut pool);

    // Zero rows.
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &vec![0.0f32; 0]);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &vec![0.0f32; 0]);
    let err = gram_xty::<f32>(&mut pool, &x_dev, &y_dev, 0, 4).err().unwrap();
    assert!(matches!(err, PrimError::ShapeMismatch { .. }));
    x_dev.release_into(&mut pool);
    y_dev.release_into(&mut pool);
}
