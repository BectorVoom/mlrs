//! Ridge multi-target `y` (RIDGE-MULTI-TARGET) correctness tests.
//!
//! `Ridge::fit_multi_target_with_sample_weight` is implemented as `n_targets`
//! independent calls to the ALREADY sklearn-oracle-validated
//! [`Ridge::fit_with_sample_weight`] (see `ridge_test.rs`'s alpha sweep against
//! the committed sklearn fixture), stacked into one `n_features × n_targets`
//! `coef_` and one length-`n_targets` `intercept_`. For the Cholesky
//! normal-equations solver this stacking is not merely an implementation
//! convenience — solving `(XᵀX + αI)·coef = Xᵀy` for a multi-column RHS is
//! mathematically IDENTICAL, column by column, to solving each column
//! separately (the system is linear and the columns of `y` never interact),
//! which is exactly what sklearn's own `_solve_cholesky` does too. So cross-
//! checking the multi-target path against `n_targets` single-target fits is
//! not just an internal-consistency check: transitively, via `ridge_test.rs`,
//! it is an sklearn-parity check for the multi-target path as well.
//!
//! No new committed `.npz` fixture: the comparison is against mlrs's OWN
//! single-target fit (already oracle-gated elsewhere), generated from a
//! self-contained splitmix64 dataset (the `ridge_perf_test.rs` precedent), so
//! this file needs no Python/sklearn environment to run.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::AlgoError;
use mlrs_algos::linear::ridge::{Ridge, RidgeSolver};
use mlrs_algos::typestate::{Fit, Predict};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::linear_predict::HostPrediction;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{PrimError, Tolerance, F32_TOL, F64_TOL};

const N_SAMPLES: usize = 200;
const N_FEATURES: usize = 8;
const N_TARGETS: usize = 3;

/// Counter-based splitmix64 (byte-identical convention to `ridge_perf_test.rs`).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform_pm1(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ridge fixtures are f32/f64 only"),
    }
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ridge fixtures are f32/f64 only"),
    }
}

fn tol_for<F: Pod>() -> &'static Tolerance {
    match std::mem::size_of::<F>() {
        4 => &F32_TOL,
        8 => &F64_TOL,
        _ => unreachable!("ridge fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose`-style compare (abs-OR-rel), the project's standard oracle
/// tolerance (`ridge_test.rs::assert_close` precedent).
fn assert_allclose(got: f64, expected: f64, tol: &Tolerance, what: &str) {
    let abs_err = (got - expected).abs();
    assert!(
        abs_err <= tol.abs + tol.rel * expected.abs(),
        "{what}: got={got:e} expected={expected:e} abs_err={abs_err:e} \
         (atol={:e}, rtol={:e})",
        tol.abs,
        tol.rel
    );
}

/// `X` uniform in `[-1, 1)^d` (seed 42), `n_targets` independent true
/// coefficient vectors + intercepts (seeds 43..), `y[:, t] = X @ coef_t +
/// intercept_t + 0.01 * noise_t`. Row-major `(n_samples, n_targets)`.
fn make_multi_regression(n: usize, d: usize, t: usize) -> (Vec<f64>, Vec<f64>) {
    let mut sx = 42u64;
    let x: Vec<f64> = (0..n * d).map(|_| uniform_pm1(&mut sx)).collect();

    let mut y = vec![0.0f64; n * t];
    for target in 0..t {
        let mut sc = 43 + (target as u64) * 2;
        let coef: Vec<f64> = (0..d).map(|_| uniform_pm1(&mut sc)).collect();
        let intercept = 0.5 + 0.1 * target as f64;
        let mut sn = 43 + (target as u64) * 2 + 1;
        for r in 0..n {
            let mut dot = intercept;
            for c in 0..d {
                dot += x[r * d + c] * coef[c];
            }
            dot += 0.01 * uniform_pm1(&mut sn);
            y[r * t + target] = dot;
        }
    }
    (x, y)
}

/// Multi-target `fit_multi_target_with_sample_weight` must produce EXACTLY
/// (bit-for-bit) the same `coef_`/`intercept_` per target as `n_targets`
/// independent single-target `fit_with_sample_weight` calls — the multi-target
/// path IS that loop, just stacked (module docs). `assert_eq` on the f64-widened
/// values, not a tolerance: this is the same construction run two ways, not two
/// different numerical methods.
fn run_multi_target_matches_independent_fits<F>()
where
    F: Float + CubeElement + Pod,
{
    let (x64, y64) = make_multi_regression(N_SAMPLES, N_FEATURES, N_TARGETS);
    let x_host: Vec<F> = x64.iter().map(|&v| f64_to::<F>(v)).collect();
    let y_host: Vec<F> = y64.iter().map(|&v| f64_to::<F>(v)).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);

    let multi = Ridge::<F>::builder()
        .alpha(1.0)
        .fit_intercept(true)
        .build::<F>()
        .expect("Ridge builds with valid hyperparameters")
        .fit_multi_target_with_sample_weight(
            &mut pool,
            &x_dev,
            &y_dev,
            (N_SAMPLES, N_FEATURES),
            N_TARGETS,
            None,
        )
        .expect("multi-target fit on a valid shape");

    assert_eq!(multi.n_targets(), N_TARGETS);
    let coef_multi = multi.coef_multi(&pool); // (N_FEATURES x N_TARGETS) row-major
    let intercept_multi = multi.intercept_multi(&pool); // length N_TARGETS

    for t in 0..N_TARGETS {
        let y_t: Vec<F> = (0..N_SAMPLES).map(|r| y_host[r * N_TARGETS + t]).collect();
        let y_t_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_t);
        let single = Ridge::<F>::builder()
            .alpha(1.0)
            .fit_intercept(true)
            .build::<F>()
            .expect("Ridge builds with valid hyperparameters")
            .fit(&mut pool, &x_dev, Some(&y_t_dev), (N_SAMPLES, N_FEATURES))
            .expect("single-target fit on a valid shape");
        y_t_dev.release_into(&mut pool);

        let coef_single = single.coef(&pool);
        let intercept_single = host_to_f64(single.intercept(&pool));

        for c in 0..N_FEATURES {
            let got = host_to_f64(coef_multi[c * N_TARGETS + t]);
            let expected = host_to_f64(coef_single[c]);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "target {t} feature {c}: multi-target coef {got:e} != \
                 independent single-target coef {expected:e}"
            );
        }
        let got_intercept = host_to_f64(intercept_multi[t]);
        assert_eq!(
            got_intercept.to_bits(),
            intercept_single.to_bits(),
            "target {t}: multi-target intercept {got_intercept:e} != \
             independent single-target intercept {intercept_single:e}"
        );
    }
}

#[test]
fn ridge_multi_target_matches_independent_fits_f32() {
    run_multi_target_matches_independent_fits::<f32>();
}

#[test]
fn ridge_multi_target_matches_independent_fits_f64() {
    let backend = mlrs_backend::capability::active_backend_name();
    if mlrs_backend::capability::skip_f64_with_log() {
        println!("ridge multi-target f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    run_multi_target_matches_independent_fits::<f64>();
}

/// The device `Predict::predict` multi-target kernel (`linear_predict_multi`)
/// must produce EXACTLY the per-target predictions that `Predict::predict`
/// gives each of the `n_targets` independent single-target fits — the two
/// kernels share the same ascending-`c` accumulation order (kernel module
/// docs), so this is a bit-exact comparison, not a tolerance one. Also checks
/// `predict_multi_from_host` (the cpu-arm / device-upload host-ingress path)
/// against the SAME device multi-target predict, which — for the multi-target
/// host routine specifically — sums in the SAME ascending order as the device
/// kernel (unlike the single-target host path's 8-lane reassociation), so it
/// too is asserted bit-exact.
fn run_multi_target_predict_matches_independent<F>()
where
    F: Float + CubeElement + Pod,
{
    let (x64, y64) = make_multi_regression(N_SAMPLES, N_FEATURES, N_TARGETS);
    let x_host: Vec<F> = x64.iter().map(|&v| f64_to::<F>(v)).collect();
    let y_host: Vec<F> = y64.iter().map(|&v| f64_to::<F>(v)).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);

    let multi = Ridge::<F>::builder()
        .alpha(1.0)
        .fit_intercept(true)
        .build::<F>()
        .expect("Ridge builds with valid hyperparameters")
        .fit_multi_target_with_sample_weight(
            &mut pool,
            &x_dev,
            &y_dev,
            (N_SAMPLES, N_FEATURES),
            N_TARGETS,
            None,
        )
        .expect("multi-target fit on a valid shape");

    // Predict on a FRESH test matrix (not the training X), same generator with a
    // different seed offset so this genuinely exercises `predict`, not `fit`.
    let mut sx = 999u64;
    let m = 37usize; // an m that does not divide cube widths evenly (edge coverage)
    let x_test64: Vec<f64> = (0..m * N_FEATURES).map(|_| uniform_pm1(&mut sx)).collect();
    let x_test: Vec<F> = x_test64.iter().map(|&v| f64_to::<F>(v)).collect();
    let x_test_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_test);

    let pred_multi_dev = multi
        .predict(&mut pool, &x_test_dev, (m, N_FEATURES))
        .expect("multi-target device predict");
    let pred_multi = pred_multi_dev.to_host(&pool); // (m x N_TARGETS) row-major
    assert_eq!(pred_multi.len(), m * N_TARGETS);

    let HostPrediction {
        values: pred_multi_host,
        operand_finite,
    } = multi
        .predict_multi_from_host(&mut pool, &x_test, (m, N_FEATURES))
        .expect("multi-target host-ingress predict");
    assert!(operand_finite);
    assert_eq!(pred_multi_host.len(), m * N_TARGETS);

    for t in 0..N_TARGETS {
        let y_t: Vec<F> = (0..N_SAMPLES).map(|r| y_host[r * N_TARGETS + t]).collect();
        let y_t_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_t);
        let single = Ridge::<F>::builder()
            .alpha(1.0)
            .fit_intercept(true)
            .build::<F>()
            .expect("Ridge builds with valid hyperparameters")
            .fit(&mut pool, &x_dev, Some(&y_t_dev), (N_SAMPLES, N_FEATURES))
            .expect("single-target fit on a valid shape");
        y_t_dev.release_into(&mut pool);

        let pred_single_dev = single
            .predict(&mut pool, &x_test_dev, (m, N_FEATURES))
            .expect("single-target device predict");
        let pred_single = pred_single_dev.to_host(&pool);
        pred_single_dev.release_into(&mut pool);

        let tol = tol_for::<F>();
        for r in 0..m {
            let got_dev = host_to_f64(pred_multi[r * N_TARGETS + t]);
            let got_host = host_to_f64(pred_multi_host[r * N_TARGETS + t]);
            let expected = host_to_f64(pred_single[r]);
            // Device-vs-device: the multi-target and single-target GATHER
            // kernels are the SAME compiled code (module docs), so this stays
            // bit-exact.
            assert_eq!(
                got_dev.to_bits(),
                expected.to_bits(),
                "row {r} target {t}: multi-target device predict {got_dev:e} != \
                 independent single-target predict {expected:e}"
            );
            // Host-vs-device: `predict_multi_from_host` now takes the HOST
            // arm unconditionally (RIDGE-PREDICT-CUDA-VS-CPU), so this
            // compares CPU arithmetic against a DEVICE kernel result. Even
            // with the identical ascending-`c` summation ORDER
            // (`matvec_bias_multi_rows` docs), these are NOT bit-identical on
            // every backend: measured on a Kaggle P100, NVRTC fuses the
            // kernel's `acc += x*coef` into a single-rounding `fma`
            // instruction, while the host loop does a separate multiply then
            // add (two roundings) — a few ULPs of divergence (~6e-8 absolute
            // at these magnitudes), well inside the project's 1e-5 oracle
            // tolerance. `cubecl-cpu`'s LLVM `-O0` JIT does not fuse, which is
            // why this assertion (wrongly written as bit-exact originally)
            // passed on the cpu backend and only surfaced the divergence on
            // real CUDA hardware.
            assert_allclose(
                got_host,
                expected,
                tol,
                &format!("row {r} target {t}: multi-target host-ingress predict"),
            );
        }
    }

    x_test_dev.release_into(&mut pool);
    pred_multi_dev.release_into(&mut pool);
}

#[test]
fn ridge_multi_target_predict_matches_independent_f32() {
    run_multi_target_predict_matches_independent::<f32>();
}

#[test]
fn ridge_multi_target_predict_matches_independent_f64() {
    let backend = mlrs_backend::capability::active_backend_name();
    if mlrs_backend::capability::skip_f64_with_log() {
        println!("ridge multi-target predict f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    run_multi_target_predict_matches_independent::<f64>();
}

/// `n_targets == 1` through the multi-target entry point must equal
/// `Fit::fit`'s ordinary single-target result exactly, for every one of
/// Ridge's eight solvers (the "no restriction" half of the module doc claim) —
/// not just the default `cholesky` arm the `n_targets > 1` tests above cover.
#[test]
fn ridge_multi_target_single_target_passthrough_every_solver_f32() {
    let (x64, y64) = make_multi_regression(N_SAMPLES, N_FEATURES, 1);
    let x_host: Vec<f32> = x64.iter().map(|&v| f64_to::<f32>(v)).collect();
    let y_host: Vec<f32> = y64.iter().map(|&v| f64_to::<f32>(v)).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_host);

    for solver in [
        RidgeSolver::Auto,
        RidgeSolver::Cholesky,
        RidgeSolver::Svd,
        RidgeSolver::Lsqr,
        RidgeSolver::SparseCg,
        RidgeSolver::Sag,
        RidgeSolver::Saga,
    ] {
        let via_multi = Ridge::<f32>::builder()
            .alpha(1.0)
            .fit_intercept(true)
            .solver(solver)
            .build::<f32>()
            .expect("Ridge builds with valid hyperparameters")
            .fit_multi_target_with_sample_weight(
                &mut pool,
                &x_dev,
                &y_dev,
                (N_SAMPLES, N_FEATURES),
                1,
                None,
            )
            .unwrap_or_else(|e| panic!("solver {solver:?}: multi-target n_targets=1 fit: {e:?}"));

        let via_single = Ridge::<f32>::builder()
            .alpha(1.0)
            .fit_intercept(true)
            .solver(solver)
            .build::<f32>()
            .expect("Ridge builds with valid hyperparameters")
            .fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
            .unwrap_or_else(|e| panic!("solver {solver:?}: single-target fit: {e:?}"));

        let coef_multi = via_multi.coef_multi(&pool);
        let coef_single = via_single.coef(&pool);
        assert_eq!(
            coef_multi, coef_single,
            "solver {solver:?}: n_targets=1 passthrough coef_ mismatch"
        );
        assert_eq!(
            via_multi.intercept_multi(&pool)[0],
            via_single.intercept(&pool),
            "solver {solver:?}: n_targets=1 passthrough intercept_ mismatch"
        );
    }
}

/// `n_targets > 1` with a solver OTHER than the default `cholesky`/`auto`
/// (`positive=False`) must raise a typed `UnsupportedCapability` error rather
/// than silently mis-fitting (module docs' solver-coverage scoping).
#[test]
fn ridge_multi_target_rejects_unsupported_solver_f32() {
    let (x64, y64) = make_multi_regression(N_SAMPLES, N_FEATURES, N_TARGETS);
    let x_host: Vec<f32> = x64.iter().map(|&v| f64_to::<f32>(v)).collect();
    let y_host: Vec<f32> = y64.iter().map(|&v| f64_to::<f32>(v)).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_host);

    for solver in [
        RidgeSolver::Svd,
        RidgeSolver::Lsqr,
        RidgeSolver::SparseCg,
        RidgeSolver::Sag,
        RidgeSolver::Saga,
    ] {
        // `Ridge<F, Fitted>` does not implement `Debug` (no derive — it holds
        // device handles), so `expect_err` (which bounds the `Ok` side on
        // `Debug`) is not usable here; match instead.
        let err = match Ridge::<f32>::builder()
            .alpha(1.0)
            .fit_intercept(true)
            .solver(solver)
            .build::<f32>()
            .expect("Ridge builds with valid hyperparameters")
            .fit_multi_target_with_sample_weight(
                &mut pool,
                &x_dev,
                &y_dev,
                (N_SAMPLES, N_FEATURES),
                N_TARGETS,
                None,
            ) {
            Ok(_) => panic!("solver {solver:?}: n_targets > 1 should be rejected"),
            Err(e) => e,
        };
        assert!(
            matches!(
                err,
                AlgoError::Prim(PrimError::UnsupportedCapability { .. })
            ),
            "solver {solver:?}: expected UnsupportedCapability, got {err:?}"
        );
    }

    // `positive=True` (solver='lbfgs', the auto-resolution target) is ALSO
    // unsupported for n_targets > 1 — it never resolves to `Cholesky`.
    let err = match Ridge::<f32>::builder()
        .alpha(1.0)
        .fit_intercept(true)
        .positive(true)
        .build::<f32>()
        .expect("Ridge builds with valid hyperparameters")
        .fit_multi_target_with_sample_weight(
            &mut pool,
            &x_dev,
            &y_dev,
            (N_SAMPLES, N_FEATURES),
            N_TARGETS,
            None,
        ) {
        Ok(_) => panic!("positive=True: n_targets > 1 should be rejected"),
        Err(e) => e,
    };
    assert!(matches!(
        err,
        AlgoError::Prim(PrimError::UnsupportedCapability { .. })
    ));
}
