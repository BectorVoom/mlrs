//! `Ridge::fit_from_host_slice` — the no-upload host ingress for the
//! `positive=True` / `solver='lbfgs'` arm (RIDGE-POS-PERF) — against the SAME
//! committed sklearn fixture `ridge_params_test.rs` gates the device arm with.
//!
//! Two things have to hold, and they are different claims:
//!
//! 1. **sklearn agreement.** The host arm is a genuinely separate route to
//!    `coef_`/`intercept_` — its own column means, its own Gram, its own
//!    centering — so it is checked against sklearn's reference directly, not
//!    against the device arm's output. A shared bug in the two mlrs arms could
//!    otherwise pass an arm-vs-arm comparison.
//! 2. **Arm equivalence.** Whichever arm `fit` picks, a caller must get the same
//!    answer; the `MLRS_RIDGE_GRAM_HOST` A/B knob is only meaningful if that
//!    holds, and so is the size-based dispatch.
//!
//! The gated cases are the fixture's three `positive=True` configurations
//! (`lbfgs_pos`, `auto_pos`, `lbfgs_pos_noint` — covering the resolved-solver
//! path, the explicit one, and `fit_intercept=False`) plus a weighted case,
//! since the host arm folds the `√w` rescale into the same pass.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::ridge::{Ridge, RidgeSolver};
use mlrs_backend::abflag;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// Fixture geometry — `gen_oracle.py`'s `RIDGE_PARAMS_N_SAMPLES/N_FEATURES`.
const N_SAMPLES: usize = 40;
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

/// numpy-`allclose` element compare (the `ridge_params_test.rs` shape).
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
        assert!(
            abs_err <= tol.abs + tol.rel * e.abs(),
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs,
            tol.rel
        );
    }
}

/// One fixture case: the ctor configuration, and the fixture key its sklearn
/// reference is stored under.
struct Case {
    name: &'static str,
    solver: RidgeSolver,
    fit_intercept: bool,
    sample_weight: bool,
}

/// The fixture's `positive=True` configurations — every one it has a sklearn
/// reference for. The WEIGHTED `positive` fit has no fixture reference, so it
/// is gated arm-against-arm instead (see [`host_and_device_arms_agree`]).
const ORACLE_CASES: &[Case] = &[
    Case {
        name: "lbfgs_pos",
        solver: RidgeSolver::Lbfgs,
        fit_intercept: true,
        sample_weight: false,
    },
    Case {
        name: "auto_pos",
        solver: RidgeSolver::Auto,
        fit_intercept: true,
        sample_weight: false,
    },
    Case {
        name: "lbfgs_pos_noint",
        solver: RidgeSolver::Lbfgs,
        fit_intercept: false,
        sample_weight: false,
    },
];

/// Host copies of the fixture's `X` / `y` / `sample_weight` at width `F`.
fn fixture_data<F>(case: &OracleCase) -> (Vec<F>, Vec<F>, Vec<F>)
where
    F: Float + CubeElement + Pod,
{
    let x = case
        .expect_f64("X")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let y = case
        .expect_f64("y")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let sw = case
        .expect_f64("sample_weight")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    (x, y, sw)
}

fn build<F>(case: &OracleCase, spec: &Case) -> Ridge<F, mlrs_algos::typestate::Unfit>
where
    F: Float + CubeElement + Pod,
{
    Ridge::<F>::builder()
        .alpha(case.expect_f64("alpha")[0])
        .fit_intercept(spec.fit_intercept)
        .solver(spec.solver)
        .positive(true)
        .tol(case.expect_f64("tol")[0])
        .max_iter(Some(case.expect_f64("max_iter")[0] as usize))
        .random_state(Some(0))
        .build::<F>()
        .unwrap_or_else(|e| panic!("case '{}' must build: {e}", spec.name))
}

/// Fit one case through the HOST arm and return `(coef_, intercept_)`.
fn fit_host<F>(case: &OracleCase, spec: &Case) -> (Vec<f64>, f64)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, y, sw) = fixture_data::<F>(case);
    // Force the host arm regardless of backend and of the size floor — this
    // fixture is 40×5, so the floor would pick it anyway, but the test must
    // gate the ARM, not the dispatch.
    let _g = abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
    let est = build::<F>(case, spec);
    assert!(
        est.host_fit_applicable((N_SAMPLES, N_FEATURES)),
        "case '{}' must route to the host arm under MLRS_RIDGE_GRAM_HOST=1",
        spec.name
    );
    let fitted = est
        .fit_from_host_slice(
            &mut pool,
            &x,
            &y,
            (N_SAMPLES, N_FEATURES),
            spec.sample_weight.then_some(sw.as_slice()),
        )
        .unwrap_or_else(|e| panic!("case '{}' must fit on the host arm: {e}", spec.name));
    (
        fitted.coef(&pool).iter().map(|&v| host_to_f64(v)).collect(),
        host_to_f64(fitted.intercept(&pool)),
    )
}

/// Fit one case through the DEVICE arm (`Fit::fit`'s body) and return
/// `(coef_, intercept_)`.
fn fit_device<F>(case: &OracleCase, spec: &Case) -> (Vec<f64>, f64)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, y, sw) = fixture_data::<F>(case);
    let _g = abflag::force("MLRS_RIDGE_GRAM_HOST", "0");
    let est = build::<F>(case, spec);
    assert!(
        !est.host_fit_applicable((N_SAMPLES, N_FEATURES)),
        "MLRS_RIDGE_GRAM_HOST=0 must send case '{}' to the device arm",
        spec.name
    );
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);
    let fitted = est
        .fit_with_sample_weight(
            &mut pool,
            &x_dev,
            Some(&y_dev),
            (N_SAMPLES, N_FEATURES),
            spec.sample_weight.then_some(sw.as_slice()),
        )
        .unwrap_or_else(|e| panic!("case '{}' must fit on the device arm: {e}", spec.name));
    (
        fitted.coef(&pool).iter().map(|&v| host_to_f64(v)).collect(),
        host_to_f64(fitted.intercept(&pool)),
    )
}

fn run_oracle<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    for spec in ORACLE_CASES {
        let (coef, intercept) = fit_host::<F>(case, spec);
        assert_close(
            &coef,
            case.expect_f64(&format!("coef_{}", spec.name)),
            tol,
            &format!("{label} host coef_ [{}]", spec.name),
        );
        assert_close(
            &[intercept],
            case.expect_f64(&format!("intercept_{}", spec.name)),
            tol,
            &format!("{label} host intercept_ [{}]", spec.name),
        );
        // The bound must actually BIND (the fixture's unconstrained answer has
        // a negative entry — see `ridge_params_test.rs`).
        assert!(
            coef.iter().all(|&c| c >= -tol.abs),
            "{label} [{}]: the host arm produced a negative coef_: {coef:?}",
            spec.name
        );
    }
}

/// The host arm reproduces sklearn's `positive=True` references, f32.
#[test]
fn ridge_host_fit_matches_sklearn_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_params_f32_seed42.npz")).expect("load ridge_params_f32");
    run_oracle::<f32>(&case, &F32_TOL, "ridge_host_fit f32");
}

/// The host arm reproduces sklearn's `positive=True` references, f64.
#[test]
fn ridge_host_fit_matches_sklearn_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("ridge_params_f64_seed42.npz")).expect("load ridge_params_f64");
    run_oracle::<f64>(&case, &F64_TOL, "ridge_host_fit f64");
}

/// The two arms agree — including on a WEIGHTED fit, which the fixture has no
/// `positive` reference for, and on `fit_intercept=False`, where the host arm
/// must return the RAW Gram's solution rather than a centered one.
#[test]
fn host_and_device_arms_agree() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("ridge_params_f64_seed42.npz")).expect("load ridge_params_f64");
    let specs = [
        Case {
            name: "lbfgs_pos",
            solver: RidgeSolver::Lbfgs,
            fit_intercept: true,
            sample_weight: false,
        },
        Case {
            name: "lbfgs_pos_noint",
            solver: RidgeSolver::Lbfgs,
            fit_intercept: false,
            sample_weight: false,
        },
        Case {
            name: "lbfgs_pos_sw",
            solver: RidgeSolver::Lbfgs,
            fit_intercept: true,
            sample_weight: true,
        },
        Case {
            name: "lbfgs_pos_noint_sw",
            solver: RidgeSolver::Lbfgs,
            fit_intercept: false,
            sample_weight: true,
        },
    ];
    for spec in &specs {
        let (hc, hi) = fit_host::<f64>(&case, spec);
        let (dc, di) = fit_device::<f64>(&case, spec);
        assert_close(&hc, &dc, &F64_TOL, &format!("arm coef_ [{}]", spec.name));
        assert_close(
            &[hi],
            &[di],
            &F64_TOL,
            &format!("arm intercept_ [{}]", spec.name),
        );
    }
}

/// The host entry point REFUSES a configuration it does not apply to, rather
/// than silently returning a differently-computed answer: a non-`positive`
/// solver never reaches it, and neither does anything while the A/B knob points
/// at the device arm.
#[test]
fn fit_from_host_slice_rejects_an_inapplicable_configuration() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = vec![0.5f32; N_SAMPLES * N_FEATURES];
    let y = vec![1.0f32; N_SAMPLES];

    {
        let _g = abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
        let est = Ridge::<f32>::builder()
            .solver(RidgeSolver::Cholesky)
            .build::<f32>()
            .expect("cholesky builds");
        assert!(!est.host_fit_applicable((N_SAMPLES, N_FEATURES)));
        assert!(est
            .fit_from_host_slice(&mut pool, &x, &y, (N_SAMPLES, N_FEATURES), None)
            .is_err());
    }
    {
        let _g = abflag::force("MLRS_RIDGE_GRAM_HOST", "0");
        let est = Ridge::<f32>::builder()
            .positive(true)
            .build::<f32>()
            .expect("positive builds");
        assert!(!est.host_fit_applicable((N_SAMPLES, N_FEATURES)));
        assert!(est
            .fit_from_host_slice(&mut pool, &x, &y, (N_SAMPLES, N_FEATURES), None)
            .is_err());
    }
}

/// Geometry rejection (ASVS V5): the slice entry point validates its own shapes
/// — it cannot lean on `validate_geometry`, which reads a `DeviceArray` length.
#[test]
fn fit_from_host_slice_rejects_bad_geometry() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let _g = abflag::force("MLRS_RIDGE_GRAM_HOST", "1");

    let mk = || {
        Ridge::<f32>::builder()
            .positive(true)
            .build::<f32>()
            .expect("positive builds")
    };
    // x shorter than n*d.
    let x = vec![0.5f32; 11];
    let y = vec![1.0f32; 3];
    assert!(mk()
        .fit_from_host_slice(&mut pool, &x, &y, (3, 4), None)
        .is_err());
    // y length != n.
    let x = vec![0.5f32; 20];
    let y = vec![1.0f32; 4];
    assert!(mk()
        .fit_from_host_slice(&mut pool, &x, &y, (5, 4), None)
        .is_err());
    // Zero rows.
    assert!(mk()
        .fit_from_host_slice(&mut pool, &[], &[], (0, 4), None)
        .is_err());
    // sample_weight length != n.
    let x = vec![0.5f32; 20];
    let y = vec![1.0f32; 5];
    let sw = vec![1.0f32; 4];
    assert!(mk()
        .fit_from_host_slice(&mut pool, &x, &y, (5, 4), Some(&sw))
        .is_err());
    // Negative sample_weight.
    let sw = vec![1.0f32, 1.0, -1.0, 1.0, 1.0];
    assert!(mk()
        .fit_from_host_slice(&mut pool, &x, &y, (5, 4), Some(&sw))
        .is_err());
    // All-zero sample_weight.
    let sw = vec![0.0f32; 5];
    assert!(mk()
        .fit_from_host_slice(&mut pool, &x, &y, (5, 4), Some(&sw))
        .is_err());
}
