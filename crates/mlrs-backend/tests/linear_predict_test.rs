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
use mlrs_backend::prims::linear_predict::{
    linear_predict, linear_predict_host, linear_predict_host_units,
};
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

/// The shapes that exercise the shared-tile kernel BELOW its `n = 64` ceiling
/// (`n = 32`, `n = 48`): the padded-tile stride math (`row·65 + c`, read only
/// up to `c < n`) and the partial-tail block guard, at `n < PREDICT_MAX_FEATURES`
/// — the regime the existing `(513, 64)` case (n exactly at the ceiling) does
/// not cover. Row counts cross the 64-row block boundary with a non-zero
/// remainder (`200 = 3·64 + 8`, `1000 = 15·64 + 40`, `130 = 2·64 + 2`) to hit
/// the tail guard. A large bias (`7.5`) keeps every prediction clear of the
/// zero-crossing, so the compare is not tripped by a near-cancellation
/// relative-error blow-up (the prediction magnitude, not the kernel, would
/// otherwise decide the tolerance).
///
/// Since the perf kernel is now dispatched ONLY on wgpu (`use_shared_predict`),
/// the cpu primary gate can no longer witness it — so this wgpu-run coverage is
/// the shared kernel's PRIMARY correctness oracle, and it must exercise BOTH
/// float widths (the padded tile is `n·size_of::<F>()`-strided in bytes, and
/// the f64 sub-ceiling path is otherwise unchecked). The two closures share the
/// deterministic design/coef/bias so f32 and f64 validate the identical shapes.
fn shared_band_case<F>(m: usize, n: usize) -> (Vec<f64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let x64 = design(m, n);
    let coef64 = coefs(n);
    let b = 7.5f64;
    let cast = |v: &[f64]| -> Vec<F> {
        v.iter()
            .map(|&x| match std::mem::size_of::<F>() {
                4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
                8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
                _ => unreachable!("shared_band_case is f32/f64 only"),
            })
            .collect()
    };
    let bias_f = cast(&[b])[0];
    let got = run_predict_case::<F>(&cast(&x64), &cast(&coef64), bias_f, m, n);
    let exp = host_predict_ref(&x64, &coef64, b, m, n);
    (got, exp)
}

const SHARED_BAND_SHAPES: &[(usize, usize)] = &[(200, 32), (1000, 32), (130, 48), (513, 32)];

#[test]
fn linear_predict_shared_band_multiblock_f32() {
    for &(m, n) in SHARED_BAND_SHAPES {
        let (got, exp) = shared_band_case::<f32>(m, n);
        assert_slice_close(&got, &exp, &F32_TOL);
    }
}

/// f64 twin of [`linear_predict_shared_band_multiblock_f32`] — the sub-ceiling
/// f64 shared-tile path (`33 KiB` tile), skipped on adapters without f64.
#[test]
fn linear_predict_shared_band_multiblock_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    for &(m, n) in SHARED_BAND_SHAPES {
        let (got, exp) = shared_band_case::<f64>(m, n);
        assert_slice_close(&got, &exp, &F64_TOL);
    }
}

// ---------------------------------------------------------------------------
// linear_predict_host — the cpu backend's zero-copy path (LINEAR-PRED-CPU)
// ---------------------------------------------------------------------------

/// Shapes for the host path, chosen to straddle every internal boundary:
/// `n = 1` and `n = 3` fall entirely in `host_dot`'s scalar remainder (below
/// `HOST_DOT_LANES = 8`), `n = 16`/`64` are exact lane multiples, `n = 11`/`37`
/// leave a non-empty remainder after full lane groups, and the row counts run
/// from a single row up past `HOST_ELEMS_PER_UNIT` so both the single-threaded
/// and the multi-threaded chunk split are exercised (including a row count that
/// does NOT divide evenly by the unit count, so the last chunk is short).
const HOST_SHAPES: &[(usize, usize)] = &[
    (1, 5),
    (7, 4),
    (300, 1),
    (999, 3),
    (1000, 16),
    (513, 64),
    (4097, 11),
    (20_003, 37),
];

/// `linear_predict_host` vs the same direct f64 host reference the kernel paths
/// are gated against.
///
/// This is the primary oracle for the cpu predict path: the estimator routes
/// cpu through this function instead of a kernel, so without this test the
/// backend that CI gates on would have no coverage of its own predict
/// arithmetic. The reassociated lane accumulation (`HOST_DOT_LANES` independent
/// chains) is deliberately compared against the strictly-sequential reference at
/// the project tolerance — that is the claim being checked, not an accident of
/// the tolerance.
#[test]
fn linear_predict_host_matches_host_ref_f32() {
    for &(m, n) in HOST_SHAPES {
        let x64 = design(m, n);
        let coef64 = coefs(n);
        let b = 0.37f64;
        let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let coef32: Vec<f32> = coef64.iter().map(|&v| v as f32).collect();

        let pred = linear_predict_host::<f32>(&x32, &coef32, b as f32, (m, n)).expect("host");
        assert!(pred.operand_finite, "the `design` operand is finite");
        let got: Vec<f64> = pred.values.iter().map(|&v| v as f64).collect();
        assert_slice_close(&got, &host_predict_ref(&x64, &coef64, b, m, n), &F32_TOL);
    }
}

/// f64 twin of [`linear_predict_host_matches_host_ref_f32`]. Runs on every
/// backend — the host path never touches the device, so an f64-incapable
/// adapter is irrelevant to it.
#[test]
fn linear_predict_host_matches_host_ref_f64() {
    for &(m, n) in HOST_SHAPES {
        let x = design(m, n);
        let coef = coefs(n);
        let b = 0.37f64;
        let pred = linear_predict_host::<f64>(&x, &coef, b, (m, n)).expect("host");
        assert!(pred.operand_finite, "the `design` operand is finite");
        assert_slice_close(
            &pred.values,
            &host_predict_ref(&x, &coef, b, m, n),
            &F64_TOL,
        );
    }
}

/// The host path is the SAME function whatever the thread split, so forcing the
/// unit count must not move a single value. Pins the contiguous-chunk row
/// arithmetic (`i·rows .. i·rows + chunk.len()`), which is the one place a
/// split-dependent bug could hide: a wrong slab offset would shift whole blocks
/// of predictions while leaving the shape and the first chunk correct. Unit
/// counts that do not divide the row count (`3`, `7` into `20_003`) are included
/// so the short final chunk is exercised.
///
/// The split is pinned through the `linear_predict_host_units` ARGUMENT, never
/// by `set_var`ing `MLRS_CPU_UNITS`: that variable is read per call, and libtest
/// runs this binary's tests on parallel threads, so mutating it here would race
/// glibc's `environ` against every sibling test's `getenv` AND silently change
/// the launch width they run under (see that function's docs).
#[test]
fn linear_predict_host_is_thread_split_invariant_f32() {
    let (m, n) = (20_003usize, 37usize);
    let x: Vec<f32> = design(m, n).iter().map(|&v| v as f32).collect();
    let coef: Vec<f32> = coefs(n).iter().map(|&v| v as f32).collect();

    let mut baseline: Option<Vec<f32>> = None;
    for units in [1usize, 2, 3, 7, 16] {
        let got = linear_predict_host_units::<f32>(&x, &coef, 0.37, (m, n), Some(units))
            .expect("host")
            .values;
        match &baseline {
            None => baseline = Some(got),
            // Bit-exact, not `close`: each row's lane accumulation is
            // independent of which thread runs it, so the split cannot change
            // a single bit. Anything else is a chunking bug.
            Some(want) => assert_eq!(&got, want, "units={units} changed the result"),
        }
    }

    // The production entry point (unit count chosen by operand size) must land
    // on the same values as every pinned split.
    assert_eq!(
        &linear_predict_host::<f32>(&x, &coef, 0.37, (m, n))
            .expect("host")
            .values,
        baseline.as_ref().expect("swept at least one unit count"),
    );
}

/// The fused NaN/inf verdict (`HostPrediction::operand_finite`) — the sklearn
/// `check_array(ensure_all_finite=True)` rejection that
/// `LinearRegression.predict` now sources from this pass instead of a second
/// scan of its own.
///
/// Checked at BOTH ends of the operand and on both sides of the
/// single-vs-multi-threaded split, since a chunked scan that folded its
/// per-thread verdicts wrongly (e.g. taking only the first chunk's) would still
/// pass a small single-threaded case.
#[test]
fn linear_predict_host_flags_nonfinite_operand_f32() {
    for &(m, n) in &[(7usize, 4usize), (20_003, 37)] {
        let coef: Vec<f32> = coefs(n).iter().map(|&v| v as f32).collect();
        let clean: Vec<f32> = design(m, n).iter().map(|&v| v as f32).collect();

        assert!(
            linear_predict_host::<f32>(&clean, &coef, 0.5, (m, n))
                .expect("host")
                .operand_finite,
            "({m}, {n}): a finite operand must not be flagged"
        );

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for at in [0usize, m * n / 2, m * n - 1] {
                let mut x = clean.clone();
                x[at] = bad;
                assert!(
                    !linear_predict_host::<f32>(&x, &coef, 0.5, (m, n))
                        .expect("host")
                        .operand_finite,
                    "({m}, {n}): {bad} at index {at} went undetected"
                );
            }
        }
    }
}

/// A finite operand whose PREDICTION overflows to infinity must NOT be flagged.
///
/// The verdict is about the operand, not the result — the cheap-looking
/// shortcut of testing `y[r].is_finite()` instead of scanning `x` would reject
/// this input, which sklearn accepts. Coefficients and values near the `f32`
/// ceiling make every product finite and their sum overflow.
#[test]
fn linear_predict_host_overflowing_prediction_is_not_rejected_f32() {
    let (m, n) = (4usize, 8usize);
    let x = vec![1.0e38f32; m * n];
    let coef = vec![1.0f32; n];
    let pred = linear_predict_host::<f32>(&x, &coef, 0.0, (m, n)).expect("host");
    assert!(
        pred.operand_finite,
        "every element of x is finite — the operand verdict must say so"
    );
    assert!(
        pred.values.iter().all(|v| v.is_infinite()),
        "the case is only meaningful if the sum actually overflows"
    );
}

/// Geometry rejection on the host path, mirroring
/// [`linear_predict_rejects_bad_geometry`] for the kernel one: the same typed
/// `PrimError`s, raised before any work.
#[test]
fn linear_predict_host_rejects_bad_geometry() {
    let coef = vec![1.0f32; 4];

    let err = linear_predict_host::<f32>(&vec![0.0f32; 11], &coef, 0.5, (3, 4))
        .err()
        .unwrap();
    assert!(matches!(err, PrimError::ShapeMismatch { operand: "x", .. }));

    let err = linear_predict_host::<f32>(&vec![0.0f32; 15], &coef, 0.5, (3, 5))
        .err()
        .unwrap();
    assert!(matches!(err, PrimError::DimMismatch { dim: "n_features", .. }));

    let err = linear_predict_host::<f32>(&[], &coef, 0.5, (0, 4)).err().unwrap();
    assert!(matches!(err, PrimError::ShapeMismatch { operand: "x", .. }));
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
