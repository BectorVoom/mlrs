//! Plan 04-05 — Ridge (LINEAR-02) sklearn oracle tests.
//!
//! Activated from the 04-01 Nyquist `#[ignore]` scaffold: each function now
//! loads its committed `Ridge(solver='cholesky', fit_intercept=True)` fixture
//! across a 3-alpha sweep {0.1, 1.0, 10.0}, fits the device estimator per alpha,
//! materializes `coef_`/`intercept_`, and asserts against the sklearn reference
//! within the 1e-5 abs+rel contract. Ridge solves `(XᵀX + αI)·coef = Xᵀy` via
//! the Phase-4 Cholesky primitive (D-02), with α on the Gram diagonal only and
//! the intercept recovered by centering (NEVER penalized, D-05).
//!
//! Two case families per dtype:
//!   - **alpha sweep** (`coef_`/`intercept_` across {0.1, 1.0, 10.0} vs sklearn).
//!   - **intercept-not-penalized** (D-05): the recovered intercept matches
//!     sklearn's (`ȳ − x̄·coef_`) — α applies only to `coef_`, never the bias —
//!     verified by reproducing sklearn's intercept analytically from the fitted
//!     `coef_` and the column means, and confirming it equals the fixture value.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::ridge::Ridge;
use mlrs_algos::typestate::{Fit, Predict};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// Ridge fixture geometry (gen_oracle.py `LIN_N_SAMPLES` × `LIN_N_FEATURES`)
/// with a 3-alpha sweep {0.1, 1.0, 10.0}: coef is (n_alphas × n_features),
/// intercept is length n_alphas.
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;
const N_ALPHAS: usize = 3;

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
        let allclose = abs_err <= tol.abs + tol.rel * e.abs();
        assert!(
            allclose,
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs, tol.rel
        );
    }
}

/// Fit `Ridge(alpha, fit_intercept=true)` on the fixture `(X, y)` and return host
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

    let reg = Ridge::<F>::builder()
        .alpha(alpha)
        .fit_intercept(true)
        .build::<F>()
        .expect("Ridge builds with valid hyperparameters")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("Ridge::fit on a valid shape");

    let coef = reg.coef(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let intercept = host_to_f64(reg.intercept(&pool));
    (coef, intercept)
}

/// Drive the full {0.1, 1.0, 10.0} alpha sweep, asserting `coef_`/`intercept_`
/// against the fixture's `(N_ALPHAS × N_FEATURES)` coef and length-`N_ALPHAS`
/// intercept.
fn run_alpha_sweep<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let alphas = case.expect_f64("alpha");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    assert_eq!(alphas.len(), N_ALPHAS, "fixture alpha sweep length");
    assert_eq!(coef_ref.len(), N_ALPHAS * N_FEATURES, "fixture coef length");
    assert_eq!(intercept_ref.len(), N_ALPHAS, "fixture intercept length");

    for (a_idx, &alpha) in alphas.iter().enumerate() {
        let (coef, intercept) = fit_coef_intercept::<F>(case, alpha);
        let expected_coef = &coef_ref[a_idx * N_FEATURES..(a_idx + 1) * N_FEATURES];
        assert_close(
            &coef,
            expected_coef,
            tol,
            &format!("{label} coef_ alpha={alpha}"),
        );
        assert_close(
            &[intercept],
            &[intercept_ref[a_idx]],
            tol,
            &format!("{label} intercept_ alpha={alpha}"),
        );
    }
}

/// `coef_`/`intercept_` across the alpha sweep vs sklearn, f32.
#[test]
fn ridge_coef_intercept_alpha_sweep_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_f32_seed42.npz")).expect("load ridge_f32");
    run_alpha_sweep::<f32>(&case, &F32_TOL, "ridge f32");
}

/// `coef_`/`intercept_` across the alpha sweep vs sklearn, f64 (cpu runs; rocm skips).
#[test]
fn ridge_coef_intercept_alpha_sweep_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("ridge_f64_seed42.npz")).expect("load ridge_f64");
    run_alpha_sweep::<f64>(&case, &F64_TOL, "ridge f64");
}

/// Verify the intercept is NOT penalized by α (D-05): for every alpha, the
/// recovered `intercept_` must equal the analytic center-then-solve form
/// `ȳ − x̄·coef_` computed from the fitted `coef_` and the (unpenalized) column
/// means — and that, in turn, equals the sklearn fixture value. If α leaked into
/// the intercept, the recovered bias would diverge from this analytic form.
fn run_intercept_not_penalized<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let alphas = case.expect_f64("alpha");
    let x = case.expect_f64("X");
    let y = case.expect_f64("y");
    let intercept_ref = case.expect_f64("intercept");

    // Unpenalized column means (the intercept is recovered from these, never the
    // penalized system).
    let mut x_mean = [0.0f64; N_FEATURES];
    let mut y_mean = 0.0f64;
    for r in 0..N_SAMPLES {
        for c in 0..N_FEATURES {
            x_mean[c] += x[r * N_FEATURES + c];
        }
        y_mean += y[r];
    }
    for m in x_mean.iter_mut() {
        *m /= N_SAMPLES as f64;
    }
    y_mean /= N_SAMPLES as f64;

    for (a_idx, &alpha) in alphas.iter().enumerate() {
        let (coef, intercept) = fit_coef_intercept::<F>(case, alpha);
        // Analytic unpenalized intercept from the fitted coef_ and the means.
        let analytic = y_mean
            - x_mean
                .iter()
                .zip(coef.iter())
                .map(|(m, c)| m * c)
                .sum::<f64>();
        assert_close(
            &[intercept],
            &[analytic],
            tol,
            &format!("{label} intercept==analytic(ȳ−x̄·coef) alpha={alpha}"),
        );
        // And the analytic (=recovered) intercept matches sklearn's fixture value.
        assert_close(
            &[intercept],
            &[intercept_ref[a_idx]],
            tol,
            &format!("{label} intercept==sklearn alpha={alpha}"),
        );
    }
}

/// Intercept-not-penalized check, f32 (D-05).
#[test]
fn ridge_intercept_not_penalized_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_f32_seed42.npz")).expect("load ridge_f32");
    run_intercept_not_penalized::<f32>(&case, &F32_TOL, "ridge f32");
}

/// Intercept-not-penalized check, f64 (cpu runs; rocm skips-with-log).
#[test]
fn ridge_intercept_not_penalized_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge intercept f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("ridge_f64_seed42.npz")).expect("load ridge_f64");
    run_intercept_not_penalized::<f64>(&case, &F64_TOL, "ridge f64");
}

/// Sanity: a fitted Ridge can `predict`, exercising the device-resident
/// `coef_`/`intercept_` GEMM path (the `Predict` import is load-bearing). Asserts
/// predictions on the training X reproduce `X·coef_ + intercept_` (consistency,
/// not a separate oracle — the coef/intercept oracle above is the strict gate).
#[test]
fn ridge_predict_consistency_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_f32_seed42.npz")).expect("load ridge_f32");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f32> = case.expect_f64("X").iter().map(|&v| v as f32).collect();
    let y_host: Vec<f32> = case.expect_f64("y").iter().map(|&v| v as f32).collect();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_host);

    let reg = Ridge::<f32>::builder()
        .alpha(1.0)
        .fit_intercept(true)
        .build::<f32>()
        .expect("Ridge builds")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("Ridge::fit");
    let pred = reg
        .predict(&mut pool, &x_dev, (N_SAMPLES, N_FEATURES))
        .expect("Ridge::predict on training X");
    let pred_host: Vec<f64> = pred.to_host(&pool).iter().map(|&v| v as f64).collect();

    // Reference: X·coef_ + intercept_ from the materialized fitted state.
    let coef: Vec<f64> = reg.coef(&pool).iter().map(|&v| v as f64).collect();
    let intercept = reg.intercept(&pool) as f64;
    let x64 = case.expect_f64("X");
    let mut reference = vec![0.0f64; N_SAMPLES];
    for r in 0..N_SAMPLES {
        let mut acc = intercept;
        for c in 0..N_FEATURES {
            acc += x64[r * N_FEATURES + c] * coef[c];
        }
        reference[r] = acc;
    }
    assert_close(
        &pred_host,
        &reference,
        &F32_TOL,
        "ridge predict==X·coef+b f32",
    );
}

/// BLDR-01: `Ridge::new()` equals `Ridge::builder().build()?` on the
/// hyperparameter subset (sklearn defaults: `alpha = 1.0`, `fit_intercept =
/// true`). Pure host comparison — no device, so no f64 gate.
#[test]
fn defaults_equal() {
    let from_new = Ridge::<f64>::new();
    let from_builder = Ridge::<f64>::builder()
        .build::<f64>()
        .expect("default RidgeBuilder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "Ridge::new() and builder().build()? must agree on hyperparameters (BLDR-01)"
    );
}

/// The ON-DEVICE intercept must agree with the host twin it replaced.
///
/// `Ridge(positive=True)`'s fused arm now finishes entirely on the device:
/// `ridge_intercept_device` computes `ȳ − x̄·coef` where the means and `coef`
/// already live, instead of reading all three back to do the dot in `f64` and
/// uploading one scalar. The two arms differ in exactly one respect — the
/// kernel accumulates in `F`, the host in `f64` — so this pins that difference
/// to rounding rather than letting it become a silent behaviour change.
///
/// The bound is relative to the operand magnitude, not to the intercept: the
/// intercept is a DIFFERENCE (`ȳ` minus a `d`-term dot), so it can sit near
/// zero while its inputs do not, and a bound relative to the result alone would
/// be unsatisfiable by any correct implementation. `f32` carries ~7 decimal
/// digits, so `1e-5 · max(|ȳ|, |x̄·coef|)` is a rounding-scale bound that a real
/// defect (a dropped term, a wrong index, a missing `ȳ`) blows through.
#[test]
fn ridge_device_intercept_matches_host_f32() {
    let widths: &[usize] = &[4, 17, 64, 200];
    for &d in widths {
        let n = 500usize;
        // Deliberately large, per-column-varying means: centering is then a real
        // cancellation, which is the regime where an f32 dot would drift if the
        // implementation were doing something other than the host's summation.
        let x: Vec<f32> = (0..n * d)
            .map(|i| ((i % 23) as f32) * 0.05 - 0.5 + 4.0 + (i % d) as f32 * 0.25)
            .collect();
        let y: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.1 + 7.0).collect();

        let fit_one = |host_arm: bool| -> f64 {
            let _g = if host_arm {
                mlrs_backend::abflag::force("MLRS_RIDGE_HOST_INTERCEPT", "1")
            } else {
                mlrs_backend::abflag::clear("MLRS_RIDGE_HOST_INTERCEPT")
            };
            let client = runtime::active_client();
            let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
            let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
            let yd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);
            let fitted = Ridge::<f32>::builder()
                .alpha(1.0)
                .fit_intercept(true)
                .positive(true)
                .build::<f32>()
                .expect("build")
                .fit(&mut pool, &xd, Some(&yd), (n, d))
                .expect("fit");
            fitted.intercept(&pool) as f64
        };

        let dev = fit_one(false);
        let host = fit_one(true);
        assert!(
            dev.is_finite() && host.is_finite(),
            "non-finite intercept at d={d}: device={dev} host={host}"
        );

        // Scale: the dot's magnitude, reconstructed from the operands rather
        // than from the (possibly cancelling) result.
        let y_bar: f64 = y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let scale = y_bar.abs().max(host.abs()).max(1.0);
        let diff = (dev - host).abs();
        assert!(
            diff <= 1e-5 * scale,
            "device intercept drifted from host at d={d}: device={dev} host={host} \
             diff={diff} > {}",
            1e-5 * scale
        );
    }
}

/// RIDGE-DEFAULT-CUDA: the default (`positive = false`) fit works — and is
/// right — ABOVE the shared-memory Cholesky kernel's `MAX_DIM = 64` order.
///
/// Before the wide factorization arm this was not a slow path, it was an error:
/// `cholesky_solve` rejected `n > MAX_DIM` with `PrimError::NotSquare`, so
/// `Ridge()` at `d = 128` returned `Err` on every GPU backend, and the shipped
/// perf ladder stopped at `d = 64` for that reason. `d ≥ 128` is also exactly
/// the regime where a GPU fit can beat a CPU one (the arithmetic is `n·d²/2`
/// over an `n·d` transfer), so the cap was capping the only shapes worth
/// running on a device.
///
/// The reference is the HOST arm — `centered_gram_xty` + `cholesky_ridge`, all
/// in `f64` — forced on with `MLRS_RIDGE_GRAM_HOST=1`. It shares no code with
/// the device composition being checked: different means pass, different Gram,
/// different factorization schedule, different accumulator width.
#[test]
fn ridge_default_fit_above_cholesky_max_dim_f32() {
    if capability::active_backend_name() == "cpu" {
        // This test's whole subject is the DEVICE factorization arm, and the cpu
        // backend cannot be made to run it in bounded time: `gram_path` there is
        // the `gemm` fallback, so the fused route defers to `center_columns`,
        // whose cpu arm walks the `d` columns one at a time with an upload +
        // launch + blocking readback each (59.6 s for `d = 8` at `n = 1 000`).
        // At `d = 256` that is half an hour to re-check something the host arm —
        // the arm the cpu backend actually takes — already covers in
        // `ridge_host_fit_test.rs`.
        println!("ridge default d>MAX_DIM: SKIPPED on cpu (device arm is the \
                  center_columns per-column round-trip; the host arm is the cpu path)");
        return;
    }
    let n = 2_000usize;
    for &d in &[100usize, 128, 256] {
        // A well-conditioned design with per-column offsets, so centering does
        // real work and the Gram is not trivially diagonal.
        let x: Vec<f32> = (0..n * d)
            .map(|i| {
                let r = (i / d) as f32;
                let c = (i % d) as f32;
                ((i % 37) as f32) * 0.031 - 0.5 + 0.01 * c + 0.001 * (r % 11.0)
            })
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|r| ((r % 17) as f32) * 0.07 + 1.5 + 0.002 * r as f32)
            .collect();

        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

        let (dev_coef, dev_int) = {
            let _g = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "0");
            let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
            let yd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);
            let fitted = Ridge::<f32>::builder()
                .alpha(1.0)
                .build::<f32>()
                .expect("build")
                .fit(&mut pool, &xd, Some(&yd), (n, d))
                .unwrap_or_else(|e| panic!("device fit at d={d} must succeed: {e}"));
            (
                fitted.coef(&pool).iter().map(|&v| v as f64).collect::<Vec<_>>(),
                fitted.intercept(&pool) as f64,
            )
        };

        let (host_coef, host_int) = {
            let _g = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
            let est = Ridge::<f32>::builder().alpha(1.0).build::<f32>().expect("build");
            assert!(
                est.host_fit_applicable((n, d)),
                "the knob must force the host reference arm at d={d}"
            );
            let fitted = est
                .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
                .unwrap_or_else(|e| panic!("host fit at d={d} must succeed: {e}"));
            (
                fitted.coef(&pool).iter().map(|&v| v as f64).collect::<Vec<_>>(),
                fitted.intercept(&pool) as f64,
            )
        };

        assert_eq!(dev_coef.len(), d);
        // Compare on the RESIDUAL scale rather than entry-by-entry: an f32 Gram
        // of a 2 000-row design is the accuracy limit here, not the solve, and
        // the two arms differ in accumulator width by construction.
        let num: f64 = dev_coef
            .iter()
            .zip(&host_coef)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let den = host_coef.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1.0);
        let rel = num / den;
        // 2e-4 leaves ~8x headroom over the measured 1.3-2.6e-5 (wgpu, f32) —
        // enough for adapter-to-adapter rounding, tight enough that a wrong
        // factorization cannot slip through.
        assert!(
            rel <= 2e-4,
            "d={d}: device coef_ diverged from the f64 host arm, rel={rel:e}"
        );
        assert!(
            (dev_int - host_int).abs() <= 2e-4 * host_int.abs().max(1.0),
            "d={d}: device intercept_={dev_int} vs host {host_int}"
        );
        println!("ridge default d={d}: coef rel={rel:e}, intercept {dev_int} vs {host_int}");
    }
}

/// RIDGE-DEFAULT-CUDA: the singular-Gram retry still fires ON THE FUSED ROUTE.
///
/// sklearn's `_ridge_regression` wraps `_solve_cholesky` in
/// `except LinAlgError` and re-solves with `svd`, reporting the fallback through
/// `solver_`. Fusing the centering into the Gram changed what that retry has to
/// work with: the SVD arm consumes the centered DESIGN, and the fused route
/// deliberately never materializes one. The retry therefore builds it on demand,
/// and this is the only test that reaches that branch.
///
/// A duplicated column with `alpha = 0` makes `XᵀX` exactly rank-deficient, so
/// the factorization hits a non-positive pivot rather than merely a small one.
#[test]
fn ridge_singular_gram_falls_back_to_svd_on_the_fused_route() {
    if capability::active_backend_name() == "cpu" {
        println!("singular-Gram retry: SKIPPED on cpu (device arm is the \
                  center_columns per-column round-trip)");
        return;
    }
    let n = 200usize;
    let d = 4usize;
    let mut x = vec![0.0f32; n * d];
    for r in 0..n {
        let a = ((r % 7) as f32) * 0.3 - 1.0;
        let b = ((r % 11) as f32) * 0.17 + 0.4;
        x[r * d] = a;
        x[r * d + 1] = b;
        x[r * d + 2] = a; // exact duplicate of column 0 -> rank-deficient Gram
        x[r * d + 3] = 1.5 * b; // and an exact multiple of column 1
    }
    let y: Vec<f32> = (0..n).map(|r| ((r % 5) as f32) * 0.6 + 2.0).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let yd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    let fitted = Ridge::<f32>::builder()
        .alpha(0.0)
        .fit_intercept(true)
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &xd, Some(&yd), (n, d))
        .expect("a singular Gram must fall back, not error");

    assert_eq!(
        fitted.solver().name(),
        "svd",
        "a rank-deficient Gram must report the sklearn-faithful svd fallback"
    );
    let coef = fitted.coef(&pool);
    assert!(
        coef.iter().all(|v| v.is_finite()) && fitted.intercept(&pool).is_finite(),
        "the fallback must never emit NaN coefficients: {coef:?}"
    );
}
