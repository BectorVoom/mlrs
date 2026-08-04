//! `GaussianMixture` (MIX-01) sklearn oracle tests.
//!
//! Reads the committed `gaussian_mixture_{f32,f64}_seed42.npz` fixture, whose
//! generator (`scripts/gen_oracle.py::gen_gaussian_mixture`) packs three
//! families of cases, each pinning a different half of the estimator:
//!
//! | family | cases | what it pins | comparison |
//! |---|---|---|---|
//! | `{cov}_{init}` | 4 × 4 | every `covariance_type` × every `init_params` reaches sklearn's optimum | up to a component PERMUTATION |
//! | `inj_{cov}` / `iter1_{cov}` | 4 + 4 | the EM arithmetic itself, with `weights_init`/`means_init`/`precisions_init` all injected so NO RNG is involved | exact, in order |
//! | `reg{i}_{cov}` | 3 × 4 | the `reg_covar` sweep | exact, in order |
//!
//! ## Why the two families need different comparisons
//! `init_params` is the one hyperparameter whose result depends on an RNG, and
//! numpy's `Generator` stream is not reproducible from Rust (D-09, the same
//! concession `KMeans` makes). So family 1 cannot compare the PATH — it compares
//! the DESTINATION: the fixture design is three blobs separated just widely
//! enough that all four initializations provably converge to one optimum (the
//! generator asserts this before writing), and the fit is run to `tol = 1e-12`
//! so both engines sit on the same stationary point rather than stopping
//! anywhere inside a `1e-3` band. The component ORDER still differs, so
//! `match_components` aligns by nearest mean first.
//!
//! Family 2 removes the RNG entirely, which is what makes an EXACT comparison —
//! including `n_iter_`, `converged_` and `lower_bound_` — meaningful. The
//! `iter1_{cov}` cases run `max_iter=1`, leaving no room at all for two engines
//! to reach the same answer by different routes.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate. Per AGENTS.md §2
//! tests live here, never in-source.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::mixture::gaussian_mixture::GaussianMixture;
use mlrs_algos::typestate::{Fit, Fitted};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{best_match_accuracy, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// Fixture geometry (gen_oracle.py `GMM_N_SAMPLES` × `GMM_N_FEATURES`, K = `GMM_K`).
const N: usize = 300;
const D: usize = 4;
const K: usize = 3;
/// Query-block rows (`GMM_N_QUERY`).
const NQ: usize = 40;

const COV_TYPES: [&str; 4] = ["full", "tied", "diag", "spherical"];
/// The four `init_params` values, paired with the fixture's case-name spelling
/// (the generator strips `-` so `k-means++` becomes `kmeans++`).
const INITS: [(&str, &str); 4] = [
    ("kmeans", "kmeans"),
    ("k-means++", "kmeans++"),
    ("random", "random"),
    ("random_from_data", "random_from_data"),
];

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("gaussian_mixture fixtures are f32/f64 only"),
    }
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("gaussian_mixture fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: `|got − exp| ≤ atol + rtol·|exp|`.
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

fn load(dtype_tag: &str) -> OracleCase {
    load_npz(fixture(&format!("gaussian_mixture_{dtype_tag}_seed42.npz")))
        .unwrap_or_else(|e| panic!("load gaussian_mixture_{dtype_tag} fixture: {e}"))
}

fn design<F: Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name).iter().map(|&v| f64_to::<F>(v)).collect()
}

/// Build + fit one `GaussianMixture` from a case spec, over the HOST ingress.
#[allow(clippy::too_many_arguments)]
fn fit_case<F>(
    x: &[F],
    cov: &str,
    init: &str,
    tol: f64,
    max_iter: usize,
    reg_covar: f64,
    injected: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
) -> GaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let (w0, m0, p0) = match injected {
        Some((w, m, p)) => (Some(w), Some(m), Some(p)),
        None => (None, None, None),
    };
    GaussianMixture::<F>::builder()
        .n_components(K)
        .covariance_type(cov)
        .init_params(init)
        .tol(tol)
        .max_iter(max_iter)
        .reg_covar(reg_covar)
        .random_state(Some(0))
        .weights_init(w0)
        .means_init(m0)
        .precisions_init(p0)
        .build::<F>()
        .expect("valid GaussianMixture hyperparameters")
        .fit_from_host_slice(x, (N, D))
        .expect("gaussian mixture fit")
}

/// Align our components with the reference's by nearest mean, returning
/// `perm[ref_component] = our_component`.
///
/// Greedy nearest-neighbour rather than a full assignment solve: the fixture's
/// blobs are separated by ~5σ, so the mean-to-mean distance matrix is
/// diagonally dominant and greedy IS optimal. The function asserts it consumed a
/// genuine permutation, which is what would fail loudly if that ever stopped
/// holding.
fn match_components(ours: &[f64], reference: &[f64]) -> Vec<usize> {
    let mut perm = vec![usize::MAX; K];
    let mut taken = [false; K];
    for r in 0..K {
        let mut best = usize::MAX;
        let mut bd = f64::INFINITY;
        for (o, t) in taken.iter().enumerate() {
            if *t {
                continue;
            }
            let dist: f64 = (0..D)
                .map(|j| {
                    let v = ours[o * D + j] - reference[r * D + j];
                    v * v
                })
                .sum();
            if dist < bd {
                bd = dist;
                best = o;
            }
        }
        taken[best] = true;
        perm[r] = best;
    }
    assert!(
        perm.iter().all(|&p| p < K),
        "match_components: failed to build a permutation ({perm:?})"
    );
    perm
}

/// Reorder a per-component parameter buffer (stride `stride` per component) into
/// the reference's component order.
fn permute(v: &[f64], perm: &[usize], stride: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len());
    for &p in perm {
        out.extend_from_slice(&v[p * stride..(p + 1) * stride]);
    }
    out
}

/// Per-component stride of `covariances_` for a `covariance_type`. `tied` has
/// ONE shared block, so it is not permuted at all (stride 0 signals that).
fn cov_stride(cov: &str) -> usize {
    match cov {
        "full" => D * D,
        "tied" => 0,
        "diag" => D,
        "spherical" => 1,
        other => unreachable!("unknown covariance_type '{other}'"),
    }
}

// ---------------------------------------------------------------------------
// Family 1 — covariance_type × init_params, compared up to a permutation
// ---------------------------------------------------------------------------

fn covariance_x_init_body<F>(dtype_tag: &str, tol: &Tolerance)
where
    F: Float + CubeElement + Pod,
{
    let case = load(dtype_tag);
    let x: Vec<F> = design(&case, "X");

    for cov in COV_TYPES {
        for (init, tag) in INITS {
            let name = format!("{cov}_{tag}");
            let fitted = fit_case::<F>(&x, cov, init, 1e-12, 2000, 1e-6, None);
            let p = fitted.params_f64();

            let ref_means = case.expect_f64(&format!("means_{name}"));
            let perm = match_components(&p.means, ref_means);

            assert_close(
                &permute(&p.means, &perm, D),
                ref_means,
                tol,
                &format!("{name}: means_"),
            );
            assert_close(
                &permute(&p.weights, &perm, 1),
                case.expect_f64(&format!("weights_{name}")),
                tol,
                &format!("{name}: weights_"),
            );
            let stride = cov_stride(cov);
            let ours_cov = if stride == 0 {
                p.covariances.clone()
            } else {
                permute(&p.covariances, &perm, stride)
            };
            assert_close(
                &ours_cov,
                case.expect_f64(&format!("cov_{name}")),
                tol,
                &format!("{name}: covariances_"),
            );
            assert_close(
                &[fitted.lower_bound()],
                case.expect_f64(&format!("lower_bound_{name}")),
                tol,
                &format!("{name}: lower_bound_"),
            );

            // The training labels from `fit`'s terminal E-step, compared up to
            // the same permutation via the shared label-matching helper.
            let ours: Vec<i64> = fitted.labels().iter().map(|&v| v as i64).collect();
            let want: Vec<i64> = case
                .expect_f64(&format!("labels_{name}"))
                .iter()
                .map(|&v| v.round() as i64)
                .collect();
            assert_eq!(
                best_match_accuracy(&ours, &want),
                1.0,
                "{name}: labels_ disagree with sklearn beyond a permutation"
            );
        }
    }
}

/// Every `covariance_type` × every `init_params` reaches sklearn's optimum, f32.
#[test]
fn covariance_type_x_init_params_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    covariance_x_init_body::<f32>("f32", &F32_TOL);
}

/// Every `covariance_type` × every `init_params` reaches sklearn's optimum, f64.
#[test]
fn covariance_type_x_init_params_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    covariance_x_init_body::<f64>("f64", &F64_TOL);
}

// ---------------------------------------------------------------------------
// Family 2 — injected init, compared EXACTLY and in order
// ---------------------------------------------------------------------------

fn injected(case: &OracleCase, cov: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    (
        case.expect_f64(&format!("winit_{cov}")).to_vec(),
        case.expect_f64(&format!("minit_{cov}")).to_vec(),
        case.expect_f64(&format!("pinit_{cov}")).to_vec(),
    )
}

fn injected_body<F>(dtype_tag: &str, tol: &Tolerance)
where
    F: Float + CubeElement + Pod,
{
    let case = load(dtype_tag);
    let x: Vec<F> = design(&case, "X");
    let xq: Vec<F> = design(&case, "Xq");

    for cov in COV_TYPES {
        let name = format!("inj_{cov}");
        let fitted = fit_case::<F>(&x, cov, "kmeans", 1e-8, 200, 1e-6, Some(injected(&case, cov)));
        let p = fitted.params_f64();

        // No permutation: the injected means fix the component order.
        assert_close(&p.weights, case.expect_f64(&format!("weights_{name}")), tol, &format!("{name}: weights_"));
        assert_close(&p.means, case.expect_f64(&format!("means_{name}")), tol, &format!("{name}: means_"));
        assert_close(&p.covariances, case.expect_f64(&format!("cov_{name}")), tol, &format!("{name}: covariances_"));
        assert_close(
            &p.precisions_cholesky,
            case.expect_f64(&format!("prec_chol_{name}")),
            tol,
            &format!("{name}: precisions_cholesky_"),
        );
        assert_close(
            &[fitted.lower_bound()],
            case.expect_f64(&format!("lower_bound_{name}")),
            tol,
            &format!("{name}: lower_bound_"),
        );
        // `n_iter_` and `converged_` are INTEGER/boolean agreements, not
        // tolerances: with no RNG the two engines must take the same number of
        // steps, or something about the convergence rule differs.
        assert_eq!(
            fitted.n_iter() as f64,
            case.expect_f64(&format!("n_iter_{name}"))[0],
            "{name}: n_iter_"
        );
        assert_eq!(
            fitted.converged() as u8 as f64,
            case.expect_f64(&format!("converged_{name}"))[0],
            "{name}: converged_"
        );
        // `lower_bounds_` pins the SHAPE of the ascent, not just its endpoint:
        // a convergence rule that reaches the same optimum by a different route
        // matches every other assertion here and fails this one.
        assert_close(
            fitted.lower_bounds(),
            case.expect_f64(&format!("lower_bounds_{name}")),
            tol,
            &format!("{name}: lower_bounds_"),
        );
        assert_eq!(
            fitted.lower_bounds().len(),
            fitted.n_iter(),
            "{name}: lower_bounds_ must have length n_iter_"
        );

        // --- the scoring surface on the disjoint query block --------------- #
        let got_pred: Vec<i64> = fitted
            .predict_labels_host(&xq, (NQ, D))
            .expect("predict_labels_host")
            .iter()
            .map(|&v| v as i64)
            .collect();
        let want_pred: Vec<i64> = case
            .expect_f64(&format!("predict_{name}"))
            .iter()
            .map(|&v| v.round() as i64)
            .collect();
        assert_eq!(got_pred, want_pred, "{name}: predict(Xq)");

        let proba: Vec<f64> = fitted
            .predict_proba_host(&xq, (NQ, D))
            .expect("predict_proba_host")
            .iter()
            .map(|&v| host_to_f64(v))
            .collect();
        assert_close(&proba, case.expect_f64(&format!("proba_{name}")), tol, &format!("{name}: predict_proba(Xq)"));

        let ss = fitted
            .score_samples_host(&xq, (NQ, D))
            .expect("score_samples_host");
        assert_close(&ss, case.expect_f64(&format!("score_samples_{name}")), tol, &format!("{name}: score_samples(Xq)"));

        assert_eq!(
            fitted.n_parameters() as f64,
            case.expect_f64(&format!("n_parameters_{name}"))[0],
            "{name}: _n_parameters()"
        );
        assert_close(
            &[fitted.bic(&xq, (NQ, D)).expect("bic")],
            case.expect_f64(&format!("bic_{name}")),
            tol,
            &format!("{name}: bic(Xq)"),
        );
        assert_close(
            &[fitted.aic(&xq, (NQ, D)).expect("aic")],
            case.expect_f64(&format!("aic_{name}")),
            tol,
            &format!("{name}: aic(Xq)"),
        );
    }
}

/// Injected-init parity, f32 — the case family with no RNG anywhere.
#[test]
fn injected_init_matches_sklearn_exactly_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    injected_body::<f32>("f32", &F32_TOL);
}

/// Injected-init parity, f64.
#[test]
fn injected_init_matches_sklearn_exactly_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    injected_body::<f64>("f64", &F64_TOL);
}

/// `max_iter = 1` from an injected init: ONE E-step and ONE M-step, compared
/// element-for-element. Nothing here can converge to the right answer by a
/// different route, so this is the strictest arithmetic gate in the file.
#[test]
fn single_em_iteration_matches_sklearn() {
    let case = load("f64");
    if capability::skip_f64_with_log() {
        return;
    }
    let x: Vec<f64> = design(&case, "X");
    for cov in COV_TYPES {
        let name = format!("iter1_{cov}");
        let fitted = fit_case::<f64>(&x, cov, "kmeans", 0.0, 1, 1e-6, Some(injected(&case, cov)));
        let p = fitted.params_f64();
        assert_close(&p.weights, case.expect_f64(&format!("weights_{name}")), &F64_TOL, &format!("{name}: weights_"));
        assert_close(&p.means, case.expect_f64(&format!("means_{name}")), &F64_TOL, &format!("{name}: means_"));
        assert_close(&p.covariances, case.expect_f64(&format!("cov_{name}")), &F64_TOL, &format!("{name}: covariances_"));
        assert_close(
            &p.precisions_cholesky,
            case.expect_f64(&format!("prec_chol_{name}")),
            &F64_TOL,
            &format!("{name}: precisions_cholesky_"),
        );
        assert_eq!(fitted.n_iter(), 1, "{name}: n_iter_");
        assert_close(
            fitted.lower_bounds(),
            case.expect_f64(&format!("lower_bounds_{name}")),
            &F64_TOL,
            &format!("{name}: lower_bounds_"),
        );
        assert!(!fitted.converged(), "{name}: tol=0 must never report converged");
    }
}

/// The `reg_covar` sweep: the one numeric hyperparameter that changes the fitted
/// covariance DIRECTLY (it is added to the diagonal) rather than through the
/// convergence test.
#[test]
fn reg_covar_sweep_matches_sklearn() {
    let case = load("f64");
    if capability::skip_f64_with_log() {
        return;
    }
    let x: Vec<f64> = design(&case, "X");
    for (i, reg) in [1e-6, 1e-2, 1.0].into_iter().enumerate() {
        for cov in COV_TYPES {
            let name = format!("reg{i}_{cov}");
            let fitted =
                fit_case::<f64>(&x, cov, "kmeans", 1e-8, 200, reg, Some(injected(&case, cov)));
            let p = fitted.params_f64();
            assert_close(&p.covariances, case.expect_f64(&format!("cov_{name}")), &F64_TOL, &format!("{name}: covariances_"));
            assert_close(&p.means, case.expect_f64(&format!("means_{name}")), &F64_TOL, &format!("{name}: means_"));
            assert_close(
                &[fitted.lower_bound()],
                case.expect_f64(&format!("lower_bound_{name}")),
                &F64_TOL,
                &format!("{name}: lower_bound_"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Behavioural gates that need no oracle
// ---------------------------------------------------------------------------

/// `predict_proba` rows are a probability distribution, and `predict` is its
/// argmax — the internal consistency sklearn's own tests assert.
#[test]
fn predict_proba_rows_sum_to_one_and_agree_with_predict() {
    let case = load("f64");
    let x: Vec<f64> = design(&case, "X");
    let xq: Vec<f64> = design(&case, "Xq");
    for cov in COV_TYPES {
        let fitted = fit_case::<f64>(&x, cov, "kmeans", 1e-8, 200, 1e-6, None);
        let proba = fitted.predict_proba_host(&xq, (NQ, D)).expect("predict_proba");
        let labels = fitted.predict_labels_host(&xq, (NQ, D)).expect("predict");
        for i in 0..NQ {
            let row = &proba[i * K..(i + 1) * K];
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "{cov}: predict_proba row {i} sums to {sum}, not 1"
            );
            let am = row
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (j, &v)| {
                    if v > bv {
                        (j, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0;
            assert_eq!(am as i32, labels[i], "{cov}: predict != argmax predict_proba at row {i}");
        }
    }
}

/// `score_samples` is `logsumexp` of the weighted log-probabilities, so
/// `exp(predict_log_proba + score_samples)` must recover the joint density —
/// i.e. `predict_log_proba` really is a NORMALIZED posterior.
#[test]
fn log_proba_and_score_samples_are_consistent() {
    let case = load("f64");
    let x: Vec<f64> = design(&case, "X");
    let xq: Vec<f64> = design(&case, "Xq");
    let fitted = fit_case::<f64>(&x, "full", "kmeans", 1e-8, 200, 1e-6, None);
    let lp = fitted.predict_log_proba_host(&xq, (NQ, D)).expect("log_proba");
    for i in 0..NQ {
        let lse = lp[i * K..(i + 1) * K]
            .iter()
            .map(|v| v.exp())
            .sum::<f64>();
        assert!((lse - 1.0).abs() < 1e-10, "row {i}: exp(log_proba) sums to {lse}");
    }
    let score = fitted.score_host(&xq, (NQ, D)).expect("score");
    let ss = fitted.score_samples_host(&xq, (NQ, D)).expect("score_samples");
    let mean = ss.iter().sum::<f64>() / ss.len() as f64;
    assert!((score - mean).abs() < 1e-12, "score != mean(score_samples)");
}

/// `n_init > 1` never returns a WORSE optimum than `n_init = 1` from the same
/// seed — the property the restart loop exists for.
#[test]
fn n_init_never_lowers_the_best_bound() {
    let case = load("f64");
    let x: Vec<f64> = design(&case, "X");
    let one = GaussianMixture::<f64>::builder()
        .n_components(K)
        .init_params("random")
        .n_init(1)
        .max_iter(30)
        .random_state(Some(7))
        .build::<f64>()
        .expect("valid")
        .fit_from_host_slice(&x, (N, D))
        .expect("fit");
    let many = GaussianMixture::<f64>::builder()
        .n_components(K)
        .init_params("random")
        .n_init(5)
        .max_iter(30)
        .random_state(Some(7))
        .build::<f64>()
        .expect("valid")
        .fit_from_host_slice(&x, (N, D))
        .expect("fit");
    assert!(
        many.lower_bound() >= one.lower_bound() - 1e-12,
        "n_init=5 lower_bound {} is below n_init=1's {}",
        many.lower_bound(),
        one.lower_bound()
    );
}

/// `warm_start` RESUMES: a second `fit` from the carried parameters continues
/// the ascent instead of re-initializing, so ten iterations split five-and-five
/// end at least as high as five alone.
#[test]
fn warm_start_resumes_the_ascent() {
    let case = load("f64");
    let x: Vec<f64> = design(&case, "X");
    let build = |warm: bool| {
        GaussianMixture::<f64>::builder()
            .n_components(K)
            .init_params("random")
            .max_iter(5)
            .tol(0.0)
            .warm_start(warm)
            .random_state(Some(3))
            .build::<f64>()
            .expect("valid")
    };
    let first = build(true).fit_from_host_slice(&x, (N, D)).expect("fit 1");
    let cold = first.lower_bound();
    let second = first
        .into_warm_start()
        .fit_from_host_slice(&x, (N, D))
        .expect("fit 2");
    assert!(
        second.lower_bound() >= cold,
        "warm_start second fit dropped the bound: {} < {cold}",
        second.lower_bound()
    );

    // Without warm_start the second fit must RE-INITIALIZE, i.e. land exactly
    // where the first one did (same seed, same data) rather than continuing.
    let a = build(false).fit_from_host_slice(&x, (N, D)).expect("cold 1");
    let b = build(false).fit_from_host_slice(&x, (N, D)).expect("cold 2");
    assert_eq!(a.lower_bound(), b.lower_bound(), "a cold fit is not deterministic");
}

/// `max_iter = 0` is LEGAL in sklearn and reports the initialization itself:
/// `n_iter_ == 0`, `converged_ == false`, and the parameters are the ones the
/// injected init supplied.
#[test]
fn max_iter_zero_reports_the_initialization() {
    let case = load("f64");
    let x: Vec<f64> = design(&case, "X");
    let (w0, m0, p0) = injected(&case, "full");
    let fitted = fit_case::<f64>(&x, "full", "kmeans", 1e-3, 0, 1e-6, Some((w0.clone(), m0.clone(), p0)));
    assert_eq!(fitted.n_iter(), 0);
    assert!(!fitted.converged());
    assert_close(&fitted.params_f64().weights, &w0, &F64_TOL, "max_iter=0: weights_");
    assert_close(&fitted.params_f64().means, &m0, &F64_TOL, "max_iter=0: means_");
}

/// `precisions_` is the inverse of `covariances_` — the identity that makes
/// `precisions_cholesky_` the factor it claims to be.
#[test]
fn precisions_invert_the_covariances() {
    let case = load("f64");
    let x: Vec<f64> = design(&case, "X");
    let fitted = fit_case::<f64>(&x, "full", "kmeans", 1e-8, 200, 1e-6, None);
    let cov = fitted.params_f64().covariances.clone();
    let prec: Vec<f64> = fitted.precisions();
    for c in 0..K {
        let s = &cov[c * D * D..(c + 1) * D * D];
        let p = &prec[c * D * D..(c + 1) * D * D];
        for a in 0..D {
            for b in 0..D {
                let v: f64 = (0..D).map(|q| s[a * D + q] * p[q * D + b]).sum();
                let want = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (v - want).abs() < 1e-8,
                    "component {c}: (Σ·Λ)[{a}][{b}] = {v}, expected {want}"
                );
            }
        }
    }
}

/// `sample` draws from the FITTED model: the empirical mean of a large draw
/// tracks `means_`, and the component tally tracks `weights_`.
#[test]
fn sample_reproduces_the_fitted_moments() {
    let case = load("f64");
    let x: Vec<f64> = design(&case, "X");
    let fitted = fit_case::<f64>(&x, "full", "kmeans", 1e-8, 200, 1e-6, None);
    let n_draw = 60_000;
    let (draws, y) = fitted.sample(n_draw, 12345).expect("sample");
    assert_eq!(draws.len(), n_draw * D);
    assert_eq!(y.len(), n_draw);

    let means = fitted.params_f64().means.clone();
    let weights = fitted.params_f64().weights.clone();
    for c in 0..K {
        let rows: Vec<usize> = (0..n_draw).filter(|&i| y[i] == c as i32).collect();
        let share = rows.len() as f64 / n_draw as f64;
        assert!(
            (share - weights[c]).abs() < 0.02,
            "component {c}: sampled share {share} vs weight {}",
            weights[c]
        );
        for j in 0..D {
            let m: f64 = rows.iter().map(|&i| draws[i * D + j]).sum::<f64>() / rows.len() as f64;
            assert!(
                (m - means[c * D + j]).abs() < 0.15,
                "component {c} feature {j}: sampled mean {m} vs {}",
                means[c * D + j]
            );
        }
    }
}

/// The DEVICE ingress (`Fit::fit`, which uploads and reads back) and the host
/// ingress produce the same fit — the invariant the two-entry-point design rests
/// on.
#[test]
fn device_and_host_ingress_agree() {
    let case = load("f64");
    if capability::skip_f64_with_log() {
        return;
    }
    let x: Vec<f64> = design(&case, "X");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    for cov in COV_TYPES {
        let host = fit_case::<f64>(&x, cov, "kmeans", 1e-8, 200, 1e-6, None);
        let dev = GaussianMixture::<f64>::builder()
            .n_components(K)
            .covariance_type(cov)
            .tol(1e-8)
            .max_iter(200)
            .random_state(Some(0))
            .build::<f64>()
            .expect("valid")
            .fit(&mut pool, &x_dev, None, (N, D))
            .expect("device fit");
        assert_close(
            &dev.params_f64().means,
            &host.params_f64().means,
            &F64_TOL,
            &format!("{cov}: device vs host means_"),
        );
        assert_eq!(dev.n_iter(), host.n_iter(), "{cov}: device vs host n_iter_");
    }
}

// ---------------------------------------------------------------------------
// Builder validation (data-INDEPENDENT, D-08)
// ---------------------------------------------------------------------------

/// Every string-valued hyperparameter rejects an unknown value at `build()`
/// with a TYPED error rather than silently falling back to a default — the
/// failure mode that would otherwise fit a different model than the caller
/// asked for.
#[test]
fn unknown_string_hyperparameters_are_rejected() {
    let e = GaussianMixture::<f64>::builder()
        .covariance_type("diagonal")
        .build::<f64>()
        .expect_err("'diagonal' is not a sklearn covariance_type");
    assert!(
        matches!(e, BuildError::UnknownCovarianceType { ref value } if value == "diagonal"),
        "expected UnknownCovarianceType, got {e:?}"
    );

    let e = GaussianMixture::<f64>::builder()
        .init_params("kmeans++")
        .build::<f64>()
        .expect_err("sklearn spells it 'k-means++'");
    assert!(
        matches!(e, BuildError::UnknownInit { ref value } if value == "kmeans++"),
        "expected UnknownInit, got {e:?}"
    );

    // Every LEGAL value must build.
    for cov in COV_TYPES {
        GaussianMixture::<f64>::builder()
            .covariance_type(cov)
            .build::<f64>()
            .unwrap_or_else(|e| panic!("covariance_type='{cov}' must build: {e}"));
    }
    for (init, _) in INITS {
        GaussianMixture::<f64>::builder()
            .init_params(init)
            .build::<f64>()
            .unwrap_or_else(|e| panic!("init_params='{init}' must build: {e}"));
    }
}

/// The numeric hyperparameters reject their out-of-range values at `build()`.
#[test]
fn invalid_numeric_hyperparameters_are_rejected() {
    assert!(matches!(
        GaussianMixture::<f64>::builder().n_components(0).build::<f64>(),
        Err(BuildError::InvalidNComponents { param: "n_components", .. })
    ));
    assert!(matches!(
        GaussianMixture::<f64>::builder().n_init(0).build::<f64>(),
        Err(BuildError::InvalidNComponents { param: "n_init", .. })
    ));
    assert!(matches!(
        GaussianMixture::<f64>::builder().tol(-1.0).build::<f64>(),
        Err(BuildError::InvalidTol { .. })
    ));
    assert!(matches!(
        GaussianMixture::<f64>::builder().reg_covar(-1e-6).build::<f64>(),
        Err(BuildError::InvalidRegCovar { .. })
    ));
    // `tol = 0` and `max_iter = 0` are LEGAL in sklearn.
    assert!(GaussianMixture::<f64>::builder().tol(0.0).max_iter(0).build::<f64>().is_ok());
}

/// The builder's defaults ARE `GaussianMixture::new`'s (BLDR-01, single source).
#[test]
fn builder_defaults_match_new() {
    let a = GaussianMixture::<f64>::new();
    let b = GaussianMixture::<f64>::builder().build::<f64>().expect("defaults build");
    assert!(a.hyperparams_eq(&b), "builder defaults diverged from new()");
}

/// A `n_components > n_samples` fit is a data-DEPENDENT rejection at `fit`, not
/// a panic and not a silent truncation.
#[test]
fn more_components_than_samples_is_rejected_at_fit() {
    let x = vec![0.0f64; 3 * D];
    let err = GaussianMixture::<f64>::builder()
        .n_components(5)
        .build::<f64>()
        .expect("5 components is a valid hyperparameter")
        .fit_from_host_slice(&x, (3, D))
        .expect_err("5 components cannot be fitted to 3 samples");
    assert!(
        format!("{err}").contains("out of range"),
        "unexpected error: {err}"
    );
}
