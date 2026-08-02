//! Gram/Xty primitive (`prims::gram::gram_xty`) oracle validation
//! (LINEAR-01 perf lever, D-02).
//!
//! `gram_xty` dispatches to the register-blocked kernel pair on every backend
//! except cpu (which falls back to the original `gemm`-based formation —
//! `gram_path`'s `#[cfg(feature = "cpu")]` gate). Running this suite under BOTH
//! `--features cpu` (exercises the `gemm` fallback) and `--features wgpu`
//! (exercises the blocked kernels) validates both dispatch arms against the
//! SAME direct host f64 reference. `LR_GRAM_SHARED=1` runs it against the
//! previous shared-memory kernels, which are kept as the A/B arm.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gram::{column_means, gram_xty, gram_xty_centered};
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
/// a multi-row-block case (`n = 2000`), `d = 64` (the shared-kernel's
/// SharedMemory budget ceiling, `d*d = 4096`), and `d = 100` — past that
/// ceiling, which the register-blocked kernel reaches and the shared one never
/// could (it used to fall into the starved-GEMM formation).
///
/// The odd widths are load-bearing for the register-blocked path: it walks the
/// Gram in groups of `GRAM_REG_TILE = 8` columns, so `d ∈ {1, 3, 4, 5, 20,
/// 100}` all leave a RAGGED final group and exercise the tail branch, while
/// `d = 64` exercises the full-tile branch exclusively.
const SHAPES: &[(usize, usize)] = &[
    (7, 4),
    (5, 5),
    (12, 3),
    (9, 1),
    (2000, 20),
    (600, 64),
    (500, 100),
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

/// Direct host reference for the CENTERED Gram: column means, target mean, and
/// the Gram/Xty of the centered design, all in f64.
#[allow(clippy::type_complexity)]
fn host_centered_ref(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    let mut xm = vec![0.0f64; d];
    let mut ym = 0.0f64;
    for i in 0..n {
        for (a, m) in xm.iter_mut().enumerate() {
            *m += x[i * d + a];
        }
        ym += y[i];
    }
    for m in xm.iter_mut() {
        *m /= n as f64;
    }
    ym /= n as f64;

    let xc: Vec<f64> = (0..n * d).map(|k| x[k] - xm[k % d]).collect();
    let yc: Vec<f64> = y.iter().map(|v| v - ym).collect();
    let (gram, xty) = host_gram_xty_ref(&xc, &yc, n, d);
    (xm, ym, gram, xty)
}

/// Run `column_means` + `gram_xty_centered` end-to-end, returning everything
/// promoted to f64.
#[allow(clippy::type_complexity)]
fn run_centered_case<F>(
    x_host: &[F],
    y_host: &[F],
    n: usize,
    d: usize,
) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, y_host);

    let (xm_dev, ym_dev) = column_means::<F>(&mut pool, &x_dev, &y_dev, n, d)
        .expect("column_means rejects nothing for a valid shape");
    let (gram_dev, xty_dev) =
        gram_xty_centered::<F>(&mut pool, &x_dev, &y_dev, (&xm_dev, &ym_dev), n, d)
            .expect("gram_xty_centered rejects nothing for a valid shape");

    let to_f64 = |v: &F| -> f64 {
        match std::mem::size_of::<F>() {
            4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
            8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
            _ => unreachable!("gram_test is f32/f64 only"),
        }
    };
    (
        xm_dev.to_host_metered(&mut pool).iter().map(to_f64).collect(),
        to_f64(&ym_dev.to_host_metered(&mut pool)[0]),
        gram_dev.to_host_metered(&mut pool).iter().map(to_f64).collect(),
        xty_dev.to_host_metered(&mut pool).iter().map(to_f64).collect(),
    )
}

/// Shapes for the fused-centering oracles.
///
/// Same widths as [`SHAPES`] — the widths are what exercise the `8`-column
/// group split and its ragged tail — but `n` is capped at 400 rather than 2000.
/// That is a TOLERANCE bound, not a speed one: these fixtures deliberately sit
/// on a large column mean (see the tests), so centering is a cancellation, and
/// the `gemm` fallback arm then accumulates the products in ONE `n`-long f32
/// chain. At `n = 2000` that arm lands at rel `1.16e-5` against the strict
/// abs-AND-rel `1e-5` gate — the same marginal band the `colmean` campaign hit
/// (`prims::center` history) and the reason `assert_slice_close_f32_gram`
/// exists at all. 400 rows still crosses a row-block boundary (blocks are 256),
/// so the multi-block fold is still covered.
const CENTERED_SHAPES: &[(usize, usize)] =
    &[(7, 4), (5, 5), (12, 3), (9, 1), (400, 20), (300, 64), (300, 100)];

/// Is the fused-centering pair worth running on this backend?
///
/// `false` on cpu, and NOT because the answer would be wrong there: `gram_path`
/// sends cpu to the `gemm` arm, where `column_means`/`gram_xty_centered` are
/// literally `center_columns` + `gram_xty` — two prims with their own oracle
/// suites (`center_test.rs`, the tests above), and a composition `Ridge` never
/// reaches on cpu (its `positive` arm takes `prims::gram_host` instead). What
/// running it there DOES cost is minutes per call: cpu `center_columns` falls
/// back to `column_reduce`, which does an upload + launch + blocking readback
/// PER COLUMN. Paying that for a path with no production caller is what makes a
/// suite too slow to run.
fn skip_fused_centering() -> bool {
    if capability::active_backend_name() == "cpu" {
        println!(
            "gram_xty_centered backend=cpu: SKIPPED (no fused kernel there — the arm is \
             center_columns + gram_xty, each already gated by its own suite)"
        );
        return true;
    }
    false
}

/// `column_means` + `gram_xty_centered` (the FUSED centering `Ridge`'s
/// `positive` arm takes) vs the explicit centre-then-Gram host reference.
///
/// This is the property that makes the fusion safe to substitute for the
/// `center_columns` → `gram_xty` composition it replaced: the means must match
/// the plain column means, and the Gram must match the Gram OF THE CENTERED
/// DESIGN — not an `XᵀX − n·x̄x̄ᵀ` correction, which is a different (and far
/// less stable) computation.
#[test]
fn gram_xty_centered_matches_host_ref_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("gram_xty_centered f64 backend={backend}: SKIPPED (no f64 on this adapter)");
        return;
    }
    if skip_fused_centering() {
        return;
    }

    for &(n, d) in CENTERED_SHAPES {
        // Column means deliberately far from zero (the `+ 3.0` offset), so a
        // dropped or mis-indexed mean is a large, visible error rather than a
        // rounding-scale one.
        let x: Vec<f64> = (0..n * d)
            .map(|i| ((i % 17) as f64) * 0.1 - 0.8 + 3.0)
            .collect();
        let y: Vec<f64> = (0..n).map(|i| ((i % 11) as f64) * 0.2 - 1.0 + 2.0).collect();
        let (got_xm, got_ym, got_gram, got_xty) = run_centered_case::<f64>(&x, &y, n, d);
        let (exp_xm, exp_ym, exp_gram, exp_xty) = host_centered_ref(&x, &y, n, d);
        assert_slice_close(&got_xm, &exp_xm, &F64_TOL);
        assert_slice_close(&[got_ym], &[exp_ym], &F64_TOL);
        assert_slice_close(&got_gram, &exp_gram, &F64_TOL);
        assert_slice_close(&got_xty, &exp_xty, &F64_TOL);
    }

    println!("gram_xty_centered f64 backend={backend}: matches centre-then-Gram reference");
}

/// f32 twin of [`gram_xty_centered_matches_host_ref_f64`] (always runs).
/// Magnitudes are scaled down for the same reason
/// `gram_xty_matches_host_ref_f32` scales them: the sums are RAW over `n` rows.
#[test]
fn gram_xty_centered_matches_host_ref_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");
    if skip_fused_centering() {
        return;
    }

    for &(n, d) in CENTERED_SHAPES {
        let x64: Vec<f64> = (0..n * d)
            .map(|i| ((i % 17) as f64) * 0.002 - 0.016 + 0.05)
            .collect();
        let y64: Vec<f64> = (0..n)
            .map(|i| ((i % 11) as f64) * 0.004 - 0.02 + 0.03)
            .collect();
        let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let y32: Vec<f32> = y64.iter().map(|&v| v as f32).collect();
        let (got_xm, got_ym, got_gram, got_xty) = run_centered_case::<f32>(&x32, &y32, n, d);
        let (exp_xm, exp_ym, exp_gram, exp_xty) = host_centered_ref(&x64, &y64, n, d);
        assert_slice_close_f32_gram(&got_xm, &exp_xm, &F32_TOL);
        assert_slice_close_f32_gram(&[got_ym], &[exp_ym], &F32_TOL);
        assert_slice_close_f32_gram(&got_gram, &exp_gram, &F32_TOL);
        assert_slice_close_f32_gram(&got_xty, &exp_xty, &F32_TOL);
    }

    println!("gram_xty_centered f32 backend={backend}: matches centre-then-Gram reference");
}

/// Shapes for the symmetry assert. Same widths as [`SHAPES`] (the mirroring is
/// an index formula, so the WIDTH is what matters — including one past the old
/// `d ≤ 64` ceiling) but a small `n` throughout: this re-runs the whole prim,
/// and on the cpu backend that means the `gemm` fallback, where a `n = 2000`
/// case is minutes rather than milliseconds.
const SYMMETRY_SHAPES: &[(usize, usize)] = &[(7, 4), (5, 5), (12, 3), (9, 1), (40, 20), (30, 100)];

/// The returned Gram must be SYMMETRIC.
///
/// The register-blocked kernel accumulates only the lower triangle and lets
/// `gram_xty_reduce_partials` mirror it, so a wrong `lower_only` wiring — the
/// mirrored slot read, or the flag passed to the wrong stage-1 kernel — shows
/// up here as an asymmetric (or uninitialized-garbage) upper triangle, which
/// the value oracles above could in principle miss if the reference happened to
/// agree on one triangle.
#[test]
fn gram_xty_output_is_symmetric_f32() {
    for &(n, d) in SYMMETRY_SHAPES {
        let x: Vec<f32> = (0..n * d).map(|i| ((i % 17) as f32) * 0.002 - 0.016).collect();
        let y: Vec<f32> = (0..n).map(|i| ((i % 11) as f32) * 0.004 - 0.02).collect();
        let (gram, _) = run_gram_case::<f32>(&x, &y, n, d);
        for a in 0..d {
            for b in 0..a {
                assert_eq!(
                    gram[a * d + b],
                    gram[b * d + a],
                    "gram not symmetric at ({a},{b}) for n={n} d={d}"
                );
            }
        }
    }
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

/// The 2-D register-tiled kernel must agree with the 1×8 register-blocked one
/// it replaced, BIT FOR BIT.
///
/// This is a stronger claim than an oracle tolerance and it is deliberate: the
/// two kernels differ only in which unit owns which output slot, not in how any
/// single slot is summed. Every `gram[a][b]` is still one register chain walked
/// over the block's rows in ascending order, the row blocking
/// (`row_blocking`) is shared, and `gram_xty_reduce_partials` folds the same
/// partials in the same order. So the answer must be identical, and an
/// `assert_eq` here catches a re-associated accumulation — the one refactor
/// that would silently move results inside the oracle band while breaking the
/// bitwise-reproducibility property the rest of the suite relies on.
///
/// Also the only test that pins the TILE geometry: a wrong triangular decode,
/// a missed ragged edge (`d % 4 != 0`), or an `Xᵀy` accumulated by the wrong
/// tiles shows up as a mismatch rather than as a tolerance drift.
#[test]
fn gram_xty_tiled_matches_blocked_bitwise_f32() {
    if capability::active_backend_name() == "cpu" {
        println!("gram_xty tiled/blocked A/B backend=cpu: SKIPPED (gram_path is the gemm arm)");
        return;
    }
    for &(n, d) in SHAPES {
        let x: Vec<f32> = (0..n * d).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
        let y: Vec<f32> = (0..n).map(|i| ((i % 11) as f32) * 0.2 - 1.0).collect();

        let (tiled_gram, tiled_xty) = {
            // FORCED, not defaulted: `gram_path` sends `d < 128` to the blocked
            // arm, so relying on the default here would compare the blocked
            // kernel against itself and pass vacuously at every shape in
            // `SHAPES` (all have `d <= 100`).
            let _g = mlrs_backend::abflag::force("LR_GRAM_TILED", "1");
            run_gram_case::<f32>(&x, &y, n, d)
        };
        let (blocked_gram, blocked_xty) = {
            let _g = mlrs_backend::abflag::force("LR_GRAM_BLOCKED", "1");
            run_gram_case::<f32>(&x, &y, n, d)
        };

        let (shared_gram, shared_xty) = {
            let _g = mlrs_backend::abflag::force("LR_GRAM_SHARED_TILED", "1");
            run_gram_case::<f32>(&x, &y, n, d)
        };

        assert_eq!(
            tiled_gram, blocked_gram,
            "tiled/blocked gram differ at n={n} d={d}"
        );
        assert_eq!(
            shared_gram, blocked_gram,
            "shared-tiled/blocked gram differ at n={n} d={d}"
        );
        assert_eq!(
            shared_xty, blocked_xty,
            "shared-tiled/blocked xty differ at n={n} d={d}"
        );
        assert_eq!(
            tiled_xty, blocked_xty,
            "tiled/blocked xty differ at n={n} d={d}"
        );

        // f64 is a shipped element width for this kernel, and it is the one the
        // oracle suites above exercise most strictly — cover it here too rather
        // than assuming the f32 agreement generalises across the element type.
        if !capability::skip_f64_with_log() {
            let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
            let (t_gram, t_xty) = {
                let _g = mlrs_backend::abflag::force("LR_GRAM_TILED", "1");
                run_gram_case::<f64>(&x64, &y64, n, d)
            };
            let (b_gram, b_xty) = {
                let _g = mlrs_backend::abflag::force("LR_GRAM_BLOCKED", "1");
                run_gram_case::<f64>(&x64, &y64, n, d)
            };
            assert_eq!(t_gram, b_gram, "f64 tiled/blocked gram differ at n={n} d={d}");
            assert_eq!(t_xty, b_xty, "f64 tiled/blocked xty differ at n={n} d={d}");
        }
    }
}

/// The same bitwise A/B for the FUSED-CENTERING pair, which is the entry point
/// `Ridge(positive=True)` actually takes. Centering is folded into the tile
/// build, so a mean applied to the wrong axis of a `4 × 4` tile (the `ma`/`mb`
/// swap) is invisible on a symmetric fixture — hence the asymmetric column
/// offsets below.
#[test]
fn gram_xty_centered_tiled_matches_blocked_bitwise_f32() {
    if skip_fused_centering() {
        return;
    }
    for &(n, d) in CENTERED_SHAPES {
        // Per-COLUMN offsets: a `ma`/`mb` swap inside the tile cancels exactly
        // when every column shares one mean, so the offsets must differ by
        // column for this test to have any power.
        let x: Vec<f32> = (0..n * d)
            .map(|i| ((i % 17) as f32) * 0.1 - 0.8 + 3.0 + (i % d) as f32)
            .collect();
        let y: Vec<f32> = (0..n).map(|i| ((i % 11) as f32) * 0.2 - 1.0 + 2.0).collect();

        let tiled = {
            // Forced for the same reason as above — `CENTERED_SHAPES` is also
            // entirely below the `TILED_MIN_DD` dispatch threshold.
            let _g = mlrs_backend::abflag::force("LR_GRAM_TILED", "1");
            run_centered_case::<f32>(&x, &y, n, d)
        };
        let blocked = {
            let _g = mlrs_backend::abflag::force("LR_GRAM_BLOCKED", "1");
            run_centered_case::<f32>(&x, &y, n, d)
        };

        let shared = {
            let _g = mlrs_backend::abflag::force("LR_GRAM_SHARED_TILED", "1");
            run_centered_case::<f32>(&x, &y, n, d)
        };
        assert_eq!(shared.2, blocked.2, "shared-tiled centered gram differs at n={n} d={d}");
        assert_eq!(shared.3, blocked.3, "shared-tiled centered xty differs at n={n} d={d}");

        assert_eq!(tiled.0, blocked.0, "x_mean differs at n={n} d={d}");
        assert_eq!(tiled.1, blocked.1, "y_mean differs at n={n} d={d}");
        assert_eq!(tiled.2, blocked.2, "centered gram differs at n={n} d={d}");
        assert_eq!(tiled.3, blocked.3, "centered xty differs at n={n} d={d}");
    }
}
