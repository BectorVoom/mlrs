//! `RidgeClassifier` (LINEAR-07) FULL sklearn parameter-surface oracle tests.
//!
//! Gates every `sklearn.linear_model.RidgeClassifier` ctor parameter against
//! the committed `ridge_classifier_{binary,multi}_{f32,f64}_seed42` fixtures
//! (`scripts/gen_oracle.py`::`gen_ridge_classifier`):
//!
//! | parameter | how it is gated |
//! |---|---|
//! | `solver` | all EIGHT values (via the DEVICE delegation arm), each vs sklearn's `coef_`/`intercept_`/`solver_` for the SAME solver |
//! | `fit_intercept` | a `False` case |
//! | `positive` | `lbfgs`, with a coefficient-sign assert so the bound is proven to BIND |
//! | `class_weight` | `None` / `'balanced'` / a PARTIAL dict (the "class absent from the dict keeps weight 1.0" fill rule) |
//! | `sample_weight` | combined with `class_weight='balanced'` too (the multiplicative combination) |
//! | `classes_` / `predict` / `decision_function` | every case, both binary (sign) and multiclass (argmax) |
//!
//! The dedicated cpu shared-Gram HOST arm
//! ([`RidgeClassifier::fit_from_host_slice`]) is gated SEPARATELY against the
//! `cholesky`/`lbfgs_pos` cases, confirming it agrees with both the DEVICE
//! delegation arm and sklearn.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::ridge::RidgeSolver;
use mlrs_algos::linear::ridge_classifier::{ClassWeight, RidgeClassifier};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

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
        _ => unreachable!("ridge_classifier fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ridge_classifier fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare (abs-OR-rel, D-10 precedent).
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

/// Which `class_weight` a [`Case`] exercises — the concrete [`ClassWeight`]
/// for the `Partial` variant is only knowable from the fixture (the dict's
/// label/weight are stored as `cw_partial_label`/`cw_partial_weight`).
#[derive(Clone, Copy)]
enum CaseClassWeight {
    Uniform,
    Balanced,
    Partial,
}

impl CaseClassWeight {
    fn resolve(self, case: &OracleCase) -> ClassWeight {
        match self {
            CaseClassWeight::Uniform => ClassWeight::Uniform,
            CaseClassWeight::Balanced => ClassWeight::Balanced,
            CaseClassWeight::Partial => {
                let label = case.expect_f64("cw_partial_label")[0].round() as i64;
                let weight = case.expect_f64("cw_partial_weight")[0];
                ClassWeight::Map(vec![(label, weight)])
            }
        }
    }
}

/// One fixture case: the ctor configuration and what sklearn produced for it.
struct Case {
    name: &'static str,
    solver: RidgeSolver,
    fit_intercept: bool,
    positive: bool,
    sample_weight: bool,
    class_weight: CaseClassWeight,
    /// The `solver_` sklearn resolved to (asserted in `gen_ridge_classifier`).
    expect_solver: RidgeSolver,
}

/// The full case table, mirroring `gen_ridge_classifier`'s `cases` list.
const CASES: &[Case] = &[
    Case { name: "auto", solver: RidgeSolver::Auto, fit_intercept: true, positive: false, sample_weight: false, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::Cholesky },
    Case { name: "cholesky", solver: RidgeSolver::Cholesky, fit_intercept: true, positive: false, sample_weight: false, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::Cholesky },
    Case { name: "svd", solver: RidgeSolver::Svd, fit_intercept: true, positive: false, sample_weight: false, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::Svd },
    Case { name: "lsqr", solver: RidgeSolver::Lsqr, fit_intercept: true, positive: false, sample_weight: false, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::Lsqr },
    Case { name: "sparse_cg", solver: RidgeSolver::SparseCg, fit_intercept: true, positive: false, sample_weight: false, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::SparseCg },
    Case { name: "sag", solver: RidgeSolver::Sag, fit_intercept: true, positive: false, sample_weight: false, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::Sag },
    Case { name: "saga", solver: RidgeSolver::Saga, fit_intercept: true, positive: false, sample_weight: false, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::Saga },
    Case { name: "lbfgs_pos", solver: RidgeSolver::Lbfgs, fit_intercept: true, positive: true, sample_weight: false, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::Lbfgs },
    Case { name: "cholesky_noint", solver: RidgeSolver::Cholesky, fit_intercept: false, positive: false, sample_weight: false, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::Cholesky },
    Case { name: "cholesky_balanced", solver: RidgeSolver::Cholesky, fit_intercept: true, positive: false, sample_weight: false, class_weight: CaseClassWeight::Balanced, expect_solver: RidgeSolver::Cholesky },
    Case { name: "cholesky_dict_partial", solver: RidgeSolver::Cholesky, fit_intercept: true, positive: false, sample_weight: false, class_weight: CaseClassWeight::Partial, expect_solver: RidgeSolver::Cholesky },
    Case { name: "cholesky_sw", solver: RidgeSolver::Cholesky, fit_intercept: true, positive: false, sample_weight: true, class_weight: CaseClassWeight::Uniform, expect_solver: RidgeSolver::Cholesky },
    Case { name: "cholesky_sw_balanced", solver: RidgeSolver::Cholesky, fit_intercept: true, positive: false, sample_weight: true, class_weight: CaseClassWeight::Balanced, expect_solver: RidgeSolver::Cholesky },
];

/// Host copies of the fixture's `X` / `Xq` / `y` / `sample_weight` at width `F`.
struct FixtureData<F> {
    x: Vec<F>,
    xq: Vec<F>,
    y: Vec<F>,
    sw: Vec<F>,
    n: usize,
    d: usize,
    n_test: usize,
    n_classes: usize,
}

fn fixture_data<F>(case: &OracleCase) -> FixtureData<F>
where
    F: Float + CubeElement + Pod,
{
    let x64 = case.expect_f64("X");
    let xq64 = case.expect_f64("Xq");
    let y64 = case.expect_f64("y");
    let n_classes = case.expect_f64("n_classes")[0].round() as usize;
    let n = y64.len();
    let d = x64.len() / n;
    let n_test = xq64.len() / d;
    FixtureData {
        x: x64.iter().map(|&v| f64_to::<F>(v)).collect(),
        xq: xq64.iter().map(|&v| f64_to::<F>(v)).collect(),
        y: y64.iter().map(|&v| f64_to::<F>(v)).collect(),
        sw: case.expect_f64("sample_weight").iter().map(|&v| f64_to::<F>(v)).collect(),
        n,
        d,
        n_test,
        n_classes,
    }
}

/// Fit one case via the DEVICE delegation arm ([`RidgeClassifier::fit_with_sample_weight`])
/// and return `(coef_, intercept_, solver_, classes_, predict, decision)`.
#[allow(clippy::type_complexity)]
fn fit_case_device<F>(
    case: &OracleCase,
    spec: &Case,
    data: &FixtureData<F>,
) -> (Vec<f64>, Vec<f64>, RidgeSolver, Vec<i64>, Vec<i64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &data.x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &data.y);

    let tol = case.expect_f64("tol")[0];
    let max_iter = case.expect_f64("max_iter")[0] as usize;
    let alpha = case.expect_f64("alpha")[0];

    let est = RidgeClassifier::<F>::builder()
        .alpha(alpha)
        .fit_intercept(spec.fit_intercept)
        .solver(spec.solver)
        .positive(spec.positive)
        .class_weight(spec.class_weight.resolve(case))
        .tol(tol)
        .max_iter(Some(max_iter))
        .random_state(Some(0))
        .build::<F>()
        .unwrap_or_else(|e| panic!("case '{}' must build: {e}", spec.name));

    let sw = if spec.sample_weight { Some(data.sw.as_slice()) } else { None };
    let fitted = est
        .fit_with_sample_weight(&mut pool, &x_dev, Some(&y_dev), (data.n, data.d), sw)
        .unwrap_or_else(|e| panic!("case '{}' must fit: {e}", spec.name));

    let coef = fitted.coef(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let intercept = fitted.intercept(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let solver_used = fitted.solver();
    let classes = fitted.classes().to_vec();

    let pred = fitted
        .predict_labels_from_host(&pool, &data.xq, (data.n_test, data.d))
        .expect("predict_labels_from_host must succeed");
    assert!(pred.operand_finite, "case '{}': Xq must be finite", spec.name);
    let predict: Vec<i64> = pred.labels.iter().map(|&l| l as i64).collect();

    let decision = fitted
        .decision_function_from_host(&pool, &data.xq, (data.n_test, data.d))
        .expect("decision_function_from_host must succeed");

    (coef, intercept, solver_used, classes, predict, decision.values)
}

fn run_all_cases<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let data = fixture_data::<F>(case);
    let expect_classes: Vec<i64> = (0..data.n_classes as i64).collect();

    for spec in CASES {
        let (coef, intercept, solver_used, classes, predict, decision) =
            fit_case_device::<F>(case, spec, &data);

        assert_eq!(classes, expect_classes, "{label} classes_ [{}]", spec.name);
        assert_close(
            &coef,
            case.expect_f64(&format!("coef_{}", spec.name)),
            tol,
            &format!("{label} coef_ [{}]", spec.name),
        );
        // sklearn's `_set_intercept` stores a bare scalar `0.0` (not a
        // length-`n_targets` array) whenever `fit_intercept=False`, REGARDLESS
        // of `n_targets` — a shape quirk, not a value difference (every target
        // gets 0 either way), so a length-1 fixture broadcasts before the
        // element-wise compare.
        let expect_intercept = case.expect_f64(&format!("intercept_{}", spec.name));
        let expect_intercept: Vec<f64> = if expect_intercept.len() == 1 && intercept.len() > 1 {
            vec![expect_intercept[0]; intercept.len()]
        } else {
            expect_intercept.to_vec()
        };
        assert_close(
            &intercept,
            &expect_intercept,
            tol,
            &format!("{label} intercept_ [{}]", spec.name),
        );
        assert_eq!(
            solver_used, spec.expect_solver,
            "{label} solver_ [{}]: got '{}' expected '{}'",
            spec.name, solver_used.name(), spec.expect_solver.name()
        );

        let expect_predict: Vec<i64> = case
            .expect_f64(&format!("predict_{}", spec.name))
            .iter()
            .map(|&v| v.round() as i64)
            .collect();
        assert_eq!(predict, expect_predict, "{label} predict [{}]", spec.name);

        assert_close(
            &decision,
            case.expect_f64(&format!("decision_{}", spec.name)),
            tol,
            &format!("{label} decision_function [{}]", spec.name),
        );

        if spec.positive {
            assert!(
                coef.iter().all(|&c| c >= -tol.abs),
                "{label} [{}]: positive=true produced a negative coef_: {coef:?}",
                spec.name
            );
        }
    }
}

#[test]
fn ridge_classifier_binary_all_cases_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_classifier_binary_f32_seed42.npz")).expect("load fixture");
    run_all_cases::<f32>(&case, &F32_TOL, "ridge_classifier binary f32");
}

#[test]
fn ridge_classifier_multiclass_all_cases_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_classifier_multi_f32_seed42.npz")).expect("load fixture");
    run_all_cases::<f32>(&case, &F32_TOL, "ridge_classifier multiclass f32");
}

#[test]
fn ridge_classifier_binary_all_cases_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge_classifier binary f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("ridge_classifier_binary_f64_seed42.npz")).expect("load fixture");
    run_all_cases::<f64>(&case, &F64_TOL, "ridge_classifier binary f64");
}

#[test]
fn ridge_classifier_multiclass_all_cases_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge_classifier multiclass f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("ridge_classifier_multi_f64_seed42.npz")).expect("load fixture");
    run_all_cases::<f64>(&case, &F64_TOL, "ridge_classifier multiclass f64");
}

// ---------------------------------------------------------------------------
// The dedicated cpu shared-Gram HOST arm — `fit_from_host_slice`
// ---------------------------------------------------------------------------

/// [`RidgeClassifier::fit_from_host_slice`] (the no-upload cpu fast path) must
/// agree with the DEVICE delegation arm AND sklearn, for both `cholesky`
/// (default) and `lbfgs_pos` (the `positive=True` arm) — the only two cases
/// `host_fit_applicable` actually covers.
fn run_host_arm_cases<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let data = fixture_data::<F>(case);
    for spec in CASES.iter().filter(|c| c.name == "cholesky" || c.name == "lbfgs_pos") {
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let tol_hp = case.expect_f64("tol")[0];
        let max_iter = case.expect_f64("max_iter")[0] as usize;
        let alpha = case.expect_f64("alpha")[0];

        let est = RidgeClassifier::<F>::builder()
            .alpha(alpha)
            .fit_intercept(spec.fit_intercept)
            .solver(spec.solver)
            .positive(spec.positive)
            .tol(tol_hp)
            .max_iter(Some(max_iter))
            .build::<F>()
            .expect("builds");

        assert!(
            est.host_fit_applicable((data.n, data.d)),
            "{label} [{}]: expected host_fit_applicable on the cpu backend",
            spec.name
        );

        let fitted = est
            .fit_from_host_slice(&mut pool, &data.x, &data.y, (data.n, data.d), None)
            .unwrap_or_else(|e| panic!("{label} [{}] host-arm fit: {e}", spec.name));

        let coef: Vec<f64> = fitted.coef(&pool).iter().map(|&v| host_to_f64(v)).collect();
        let intercept: Vec<f64> = fitted.intercept(&pool).iter().map(|&v| host_to_f64(v)).collect();

        assert_close(
            &coef,
            case.expect_f64(&format!("coef_{}", spec.name)),
            tol,
            &format!("{label} host-arm coef_ [{}]", spec.name),
        );
        assert_close(
            &intercept,
            case.expect_f64(&format!("intercept_{}", spec.name)),
            tol,
            &format!("{label} host-arm intercept_ [{}]", spec.name),
        );
        assert_eq!(fitted.n_iter(), None, "{label} [{}]: host arm never reports n_iter_", spec.name);
    }
}

#[test]
fn ridge_classifier_host_arm_binary_f32() {
    let case = load_npz(fixture("ridge_classifier_binary_f32_seed42.npz")).expect("load fixture");
    run_host_arm_cases::<f32>(&case, &F32_TOL, "ridge_classifier host-arm binary f32");
}

#[test]
fn ridge_classifier_host_arm_multiclass_f32() {
    let case = load_npz(fixture("ridge_classifier_multi_f32_seed42.npz")).expect("load fixture");
    run_host_arm_cases::<f32>(&case, &F32_TOL, "ridge_classifier host-arm multiclass f32");
}

// ---------------------------------------------------------------------------
// Builder validation — the data-INDEPENDENT rejections
// ---------------------------------------------------------------------------

#[test]
fn ridge_classifier_builder_rejects_invalid_hyperparameters() {
    assert!(matches!(
        RidgeClassifier::<f64>::builder().alpha(-1.0).build::<f64>(),
        Err(BuildError::InvalidAlpha { .. })
    ));
    assert!(matches!(
        RidgeClassifier::<f64>::builder().tol(-1e-4).build::<f64>(),
        Err(BuildError::InvalidTol { .. })
    ));
    assert!(matches!(
        RidgeClassifier::<f64>::builder().max_iter(Some(0)).build::<f64>(),
        Err(BuildError::InvalidMaxIter { .. })
    ));
    assert!(matches!(
        RidgeClassifier::<f64>::builder()
            .solver(RidgeSolver::Lbfgs)
            .build::<f64>(),
        Err(BuildError::LbfgsRequiresPositive { .. })
    ));
    assert!(matches!(
        RidgeClassifier::<f64>::builder()
            .solver(RidgeSolver::Cholesky)
            .positive(true)
            .build::<f64>(),
        Err(BuildError::PositiveUnsupportedSolver { .. })
    ));
    assert!(RidgeClassifier::<f64>::builder().positive(true).build::<f64>().is_ok());
}

/// BLDR-01: `RidgeClassifier::new()` equals `RidgeClassifier::builder().build()?`
/// across the full hyperparameter set.
#[test]
fn ridge_classifier_defaults_equal_full_surface() {
    let from_new = RidgeClassifier::<f64>::new();
    let from_builder = RidgeClassifier::<f64>::builder()
        .build::<f64>()
        .expect("default builder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "RidgeClassifier::new() and builder().build()? must agree on ALL hyperparameters"
    );
}
