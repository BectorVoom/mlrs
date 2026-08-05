//! `BayesianGaussianMixture` (MIX-02) sklearn oracle tests.
//!
//! Reads the committed `bayesian_mixture_{f32,f64}_seed42.npz` fixture, whose
//! generator (`scripts/gen_oracle.py::gen_bayesian_mixture`) packs four
//! families:
//!
//! | family | cases | what it pins | comparison |
//! |---|---|---|---|
//! | `{cov}_{init}_{ptype}` | 4 × 4 × 2 | every `covariance_type` × every `init_params` × every `weight_concentration_prior_type` reaches sklearn's optimum | up to a component PERMUTATION |
//! | `k1{cov}_{ptype}` / `k1i{cov}_{ptype}` | 4 × 2 × 2 | the variational arithmetic, RNG-free | exact, in order, incl. the scoring surface |
//! | `pr{i}{cov}_{ptype}` | 5 × 4 × 2 | the five prior hyperparameters | exact, in order |
//! | `stick_{ptype}` | 2 | the weight-posterior recursion at `k = 5` on a fixed `nk` | exact |
//!
//! ## Why four families and not two
//! The estimator has parts that a converged end-to-end fit provably cannot pin,
//! so each got its own family rather than a looser tolerance on one:
//!
//! - **`init_params` needs a permutation-tolerant comparison** because numpy's
//!   `Generator` stream is not reproducible from Rust (D-09), so family 1
//!   compares the DESTINATION, not the path — and even the destination's
//!   component ORDER differs.
//! - **The Bayesian-only attributes need an RNG-FREE case.** `n_components = 1`
//!   with `init_params='random'` is exactly that: a one-column responsibility
//!   matrix row-normalizes to `1.0` in both engines whatever was drawn, so
//!   family 2 compares every posterior, every resolved prior, `lower_bound_`,
//!   `lower_bounds_`, `n_iter_` and `converged_` bit-for-bit. Family 3 is the
//!   same construction with each prior moved off its default, which is the only
//!   thing that would notice a prior being silently ignored.
//! - **The `dirichlet_process` weight recursion is ORDER-DEPENDENT**, so it
//!   cannot ride family 1 at all: component `c`'s second Beta parameter sums
//!   the `nk` of every component AFTER it, and two engines that find the same
//!   clustering in a different order legitimately disagree on
//!   `weight_concentration_` and `weights_` by `O(γ/n)`. Family 4 evaluates the
//!   recursion on a FIXED, unequal `nk` at `k = 5`, where ordering is not a
//!   question.
//!
//! ## The two cases the fixture marks unstable
//! Family 1 carries a `stable_{name}` flag per case, measured by the generator
//! rather than assumed. It is `0` for exactly two of the thirty-two:
//! `covariance_type='tied'` + `weight_concentration_prior_type=
//! 'dirichlet_process'` + a sparse initialization (`k-means++` /
//! `random_from_data`). There the variational objective has several attracting
//! basins and the initialization picks one, so no tolerance can bridge two
//! different RNGs — see the generator's comment for the mechanism. Those two
//! cases are still FITTED here (a panic, a non-finite parameter or a failed
//! factorization would still fail the test) and their qualitative outcome is
//! asserted, but their values are not compared. No parameter VALUE loses
//! coverage: `tied` appears in six other compared cases, `dirichlet_process` in
//! fourteen, and each sparse init in seven.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate. Per AGENTS.md
//! §2 tests live here, never in-source.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::mixture::bayesian_gaussian_mixture::{
    mixing_weights, weight_concentration, BayesianGaussianMixture, WeightConcentrationPriorType,
};
use mlrs_algos::typestate::Fitted;
use mlrs_backend::capability;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{best_match_accuracy, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// Fixture geometry — the generator shares `GaussianMixture`'s design, so these
/// are `GMM_N_SAMPLES` × `GMM_N_FEATURES` with `K = GMM_K`.
const N: usize = 300;
const D: usize = 4;
const K: usize = 3;
/// Query-block rows (`GMM_N_QUERY`).
const NQ: usize = 40;

const COV_TYPES: [&str; 4] = ["full", "tied", "diag", "spherical"];
/// The four `init_params` values paired with the fixture's case-name spelling
/// (the generator strips `-`, so `k-means++` becomes `kmeans++`).
const INITS: [(&str, &str); 4] = [
    ("kmeans", "kmeans"),
    ("k-means++", "kmeans++"),
    ("random", "random"),
    ("random_from_data", "random_from_data"),
];
/// The two `weight_concentration_prior_type` values with their fixture tags.
const PRIOR_TYPES: [(&str, &str); 2] = [
    ("dirichlet_process", "dp"),
    ("dirichlet_distribution", "dd"),
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
        _ => unreachable!("bayesian_mixture fixtures are f32/f64 only"),
    }
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("bayesian_mixture fixtures are f32/f64 only"),
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
    load_npz(fixture(&format!("bayesian_mixture_{dtype_tag}_seed42.npz")))
        .unwrap_or_else(|e| panic!("load bayesian_mixture_{dtype_tag} fixture: {e}"))
}

fn design<F: Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect()
}

fn prior_type_of(name: &str) -> WeightConcentrationPriorType {
    match name {
        "dirichlet_process" => WeightConcentrationPriorType::DirichletProcess,
        "dirichlet_distribution" => WeightConcentrationPriorType::DirichletDistribution,
        other => unreachable!("unknown weight_concentration_prior_type '{other}'"),
    }
}

/// Every knob one fixture case sets. Grouped into a struct rather than passed as
/// nine positional arguments, because the prior sweep varies exactly one of
/// them per case and a positional call would make which one invisible.
#[derive(Clone)]
struct Case {
    n_components: usize,
    /// Restart count. Family 1 uses `BGM_N_INIT = 5` (see below); the RNG-free
    /// families use 1, where a restart could only repeat itself.
    n_init: usize,
    cov: &'static str,
    init: &'static str,
    prior_type: &'static str,
    tol: f64,
    max_iter: usize,
    weight_concentration_prior: Option<f64>,
    mean_precision_prior: Option<f64>,
    mean_prior: Option<Vec<f64>>,
    degrees_of_freedom_prior: Option<f64>,
    covariance_prior: Option<Vec<f64>>,
}

impl Case {
    /// The `k = 1`, `init_params='random'`, RNG-free base of families 2 and 3.
    fn k1(cov: &'static str, prior_type: &'static str, tol: f64, max_iter: usize) -> Self {
        Self {
            n_components: 1,
            n_init: 1,
            cov,
            init: "random",
            prior_type,
            tol,
            max_iter,
            weight_concentration_prior: None,
            mean_precision_prior: None,
            mean_prior: None,
            degrees_of_freedom_prior: None,
            covariance_prior: None,
        }
    }
}

fn fit_case<F>(x: &[F], n: usize, c: &Case) -> BayesianGaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    BayesianGaussianMixture::<F>::builder()
        .n_components(c.n_components)
        .covariance_type(c.cov)
        .init_params(c.init)
        .weight_concentration_prior_type(c.prior_type)
        .tol(c.tol)
        .max_iter(c.max_iter)
        .n_init(c.n_init)
        .random_state(Some(0))
        .weight_concentration_prior(c.weight_concentration_prior)
        .mean_precision_prior(c.mean_precision_prior)
        .mean_prior(c.mean_prior.clone())
        .degrees_of_freedom_prior(c.degrees_of_freedom_prior)
        .covariance_prior(c.covariance_prior.clone())
        .build::<F>()
        .expect("valid BayesianGaussianMixture hyperparameters")
        .fit_from_host_slice(&mut pool, x, (n, D))
        .expect("bayesian gaussian mixture fit")
}

/// Align our components with the reference's by nearest mean, returning
/// `perm[ref_component] = our_component`.
///
/// Greedy nearest-neighbour, which IS optimal here: the fixture's blobs are
/// ~5σ apart, so the mean-to-mean distance matrix is diagonally dominant. The
/// function asserts it consumed a genuine permutation, which is what would fail
/// loudly if that stopped holding.
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

/// Reorder a per-component buffer (stride `stride`) into the reference's order.
fn permute(v: &[f64], perm: &[usize], stride: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len());
    for &p in perm {
        out.extend_from_slice(&v[p * stride..(p + 1) * stride]);
    }
    out
}

/// Reorder a reference buffer into OUR component order — the inverse of
/// [`permute`], needed by the `dirichlet_process` weight check below.
fn permute_inverse(v: &[f64], perm: &[usize], stride: usize) -> Vec<f64> {
    let mut inv = vec![usize::MAX; perm.len()];
    for (r, &o) in perm.iter().enumerate() {
        inv[o] = r;
    }
    let mut out = Vec::with_capacity(v.len());
    for &r in &inv {
        out.extend_from_slice(&v[r * stride..(r + 1) * stride]);
    }
    out
}

/// Per-component stride of `covariances_`. `tied` has ONE shared block, so it
/// is not permuted at all (stride 0 signals that).
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
// Family 1 — the string-parameter cross, compared up to a permutation
// ---------------------------------------------------------------------------

fn string_param_cross_body<F>(dtype_tag: &str, tol: &Tolerance)
where
    F: Float + CubeElement + Pod,
{
    let case = load(dtype_tag);
    let x: Vec<F> = design(&case, "X");

    for cov in COV_TYPES {
        for (init, init_tag) in INITS {
            for (ptype, ptag) in PRIOR_TYPES {
                let name = format!("{cov}_{init_tag}_{ptag}");
                let spec = Case {
                    n_components: K,
                    // Five restarts, matching the fixture. The two SPARSE
                    // initializations are a lottery over which `k` rows seed
                    // the first M-step, and mlrs draws from a different stream
                    // than numpy (D-09) — so with ONE restart the two engines
                    // can land in different basins even where sklearn's own
                    // four routes agree. Defeating exactly that is what
                    // `n_init` is for, and it exercises the restart loop.
                    n_init: 5,
                    cov,
                    init,
                    prior_type: ptype,
                    tol: 1e-12,
                    max_iter: 2000,
                    weight_concentration_prior: None,
                    mean_precision_prior: None,
                    mean_prior: None,
                    degrees_of_freedom_prior: None,
                    covariance_prior: None,
                };
                let fitted = fit_case::<F>(&x, N, &spec);
                let p = fitted.params_f64();

                let stable = case.expect_f64(&format!("stable_{name}"))[0] > 0.5;
                if !stable {
                    // The documented multi-basin case (module docs). What is
                    // still asserted is that the fit RAN and produced a usable
                    // model — and that the Dirichlet process did the thing that
                    // makes the case multi-basin in the first place: prune a
                    // component to a negligible weight.
                    let w = fitted.weights_f64();
                    assert!(
                        w.iter().all(|v| v.is_finite())
                            && p.means.iter().all(|v| v.is_finite())
                            && p.covariances.iter().all(|v| v.is_finite()),
                        "{name}: unstable case produced a non-finite parameter"
                    );
                    assert!(
                        (w.iter().sum::<f64>() - 1.0).abs() < 1e-9,
                        "{name}: weights_ do not sum to 1"
                    );
                    assert!(
                        w.iter().cloned().fold(f64::INFINITY, f64::min) < 1e-2,
                        "{name}: the fixture marks this case as a \
                         dirichlet_process collapse, but no component was \
                         pruned (weights_ = {w:?})"
                    );
                    continue;
                }

                let ref_means = case.expect_f64(&format!("means_{name}"));
                let perm = match_components(&p.means, ref_means);

                assert_close(
                    &permute(&p.means, &perm, D),
                    ref_means,
                    tol,
                    &format!("{name}: means_"),
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
                let ours_chol = if stride == 0 {
                    p.precisions_cholesky.clone()
                } else {
                    permute(&p.precisions_cholesky, &perm, stride)
                };
                assert_close(
                    &ours_chol,
                    case.expect_f64(&format!("prec_chol_{name}")),
                    tol,
                    &format!("{name}: precisions_cholesky_"),
                );
                // `mean_precision_` is `β₀ + nk`, so comparing it pins the
                // component COUNTS exactly — the input every Bayesian block
                // downstream is a function of.
                assert_close(
                    &permute(&p.mean_precision, &perm, 1),
                    case.expect_f64(&format!("beta_{name}")),
                    tol,
                    &format!("{name}: mean_precision_"),
                );
                let ours_dof = if cov == "tied" {
                    p.degrees_of_freedom.clone()
                } else {
                    permute(&p.degrees_of_freedom, &perm, 1)
                };
                assert_close(
                    &ours_dof,
                    case.expect_f64(&format!("dof_{name}")),
                    tol,
                    &format!("{name}: degrees_of_freedom_"),
                );

                let pt = prior_type_of(ptype);
                let ours_weights = fitted.weights_f64();
                match pt {
                    WeightConcentrationPriorType::DirichletDistribution => {
                        // Exchangeable: `α = γ + nk` and `weights_ = α/Σα` are
                        // order-EQUIVARIANT, so permuting is enough and the
                        // bound (a sum over components) is directly comparable.
                        assert_close(
                            &permute(&p.weight_concentration_a, &perm, 1),
                            case.expect_f64(&format!("wca_{name}")),
                            tol,
                            &format!("{name}: weight_concentration_"),
                        );
                        assert_close(
                            &permute(&ours_weights, &perm, 1),
                            case.expect_f64(&format!("weights_{name}")),
                            tol,
                            &format!("{name}: weights_"),
                        );
                        assert_close(
                            &[fitted.lower_bound()],
                            case.expect_f64(&format!("lower_bound_{name}")),
                            tol,
                            &format!("{name}: lower_bound_"),
                        );
                    }
                    WeightConcentrationPriorType::DirichletProcess => {
                        // Stick-breaking is NOT exchangeable, so sklearn's
                        // `weight_concentration_` / `weights_` / `lower_bound_`
                        // are values for ITS component order and cannot be
                        // permuted into ours. What is checkable — and is
                        // checked — is that our weight posterior is the correct
                        // FUNCTION of the component counts sklearn agrees with:
                        // take sklearn's `nk` (from the `mean_precision_` just
                        // compared above), put it in OUR order, and run the
                        // recursion. The recursion ITSELF is pinned against
                        // sklearn by family 4, at a `k` and an `nk` where
                        // ordering is not a question.
                        let beta0 = case.expect_f64(&format!("pbeta_{name}"))[0];
                        let gamma = case.expect_f64(&format!("pwc_{name}"))[0];
                        let ref_beta = case.expect_f64(&format!("beta_{name}"));
                        let nk_ours: Vec<f64> = permute_inverse(ref_beta, &perm, 1)
                            .into_iter()
                            .map(|b| b - beta0)
                            .collect();
                        let (a, b) = weight_concentration(pt, gamma, &nk_ours);
                        assert_close(
                            &p.weight_concentration_a,
                            &a,
                            tol,
                            &format!("{name}: weight_concentration_[0]"),
                        );
                        assert_close(
                            &p.weight_concentration_b,
                            &b,
                            tol,
                            &format!("{name}: weight_concentration_[1]"),
                        );
                        assert_close(
                            &ours_weights,
                            &mixing_weights(pt, &a, &b),
                            tol,
                            &format!("{name}: weights_"),
                        );
                    }
                }

                // The training labels from `fit`'s terminal E-step, compared up
                // to the same permutation via the shared label-matching helper.
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
}

/// Every `covariance_type` × `init_params` × `weight_concentration_prior_type`
/// reaches sklearn's optimum, f32.
#[test]
fn string_parameter_cross_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    string_param_cross_body::<f32>("f32", &F32_TOL);
}

/// Every `covariance_type` × `init_params` × `weight_concentration_prior_type`
/// reaches sklearn's optimum, f64.
#[test]
fn string_parameter_cross_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    string_param_cross_body::<f64>("f64", &F64_TOL);
}

// ---------------------------------------------------------------------------
// Families 2 & 3 — the RNG-free exact comparison
// ---------------------------------------------------------------------------

/// Compare EVERY fitted attribute of one `k = 1` case, in order.
///
/// `k = 1` is what makes this exact: `init_params='random'` draws an `n × 1`
/// responsibility matrix and row-normalizes it, which is `1.0` everywhere in
/// both engines regardless of the stream. So there is no RNG left, no component
/// permutation to resolve, and `n_iter_` / `converged_` / the per-iteration
/// bound trace are all meaningful comparisons.
fn assert_exact_case<F>(
    case: &OracleCase,
    x: &[F],
    xq: Option<&[F]>,
    name: &str,
    spec: &Case,
    tol: &Tolerance,
) where
    F: Float + CubeElement + Pod,
{
    let fitted = fit_case::<F>(x, N, spec);
    let p = fitted.params_f64();
    let pr = fitted.priors();

    // -- the resolved priors (sklearn's `*_prior_` attributes) -------------- //
    assert_close(
        &[pr.weight_concentration],
        case.expect_f64(&format!("pwc_{name}")),
        tol,
        &format!("{name}: weight_concentration_prior_"),
    );
    assert_close(
        &[pr.mean_precision],
        case.expect_f64(&format!("pbeta_{name}")),
        tol,
        &format!("{name}: mean_precision_prior_"),
    );
    assert_close(
        &pr.mean,
        case.expect_f64(&format!("pmean_{name}")),
        tol,
        &format!("{name}: mean_prior_"),
    );
    assert_close(
        &[pr.degrees_of_freedom],
        case.expect_f64(&format!("pdof_{name}")),
        tol,
        &format!("{name}: degrees_of_freedom_prior_"),
    );
    assert_close(
        &pr.covariance,
        case.expect_f64(&format!("pcov_{name}")),
        tol,
        &format!("{name}: covariance_prior_"),
    );

    // -- the variational posteriors ----------------------------------------- //
    assert_close(
        &p.weight_concentration_a,
        case.expect_f64(&format!("wca_{name}")),
        tol,
        &format!("{name}: weight_concentration_[0]"),
    );
    if prior_type_of(spec.prior_type) == WeightConcentrationPriorType::DirichletProcess {
        assert_close(
            &p.weight_concentration_b,
            case.expect_f64(&format!("wcb_{name}")),
            tol,
            &format!("{name}: weight_concentration_[1]"),
        );
    } else {
        assert!(
            p.weight_concentration_b.is_empty(),
            "{name}: dirichlet_distribution must carry no second Beta parameter"
        );
    }
    assert_close(
        &fitted.weights_f64(),
        case.expect_f64(&format!("weights_{name}")),
        tol,
        &format!("{name}: weights_"),
    );
    assert_close(
        &p.mean_precision,
        case.expect_f64(&format!("beta_{name}")),
        tol,
        &format!("{name}: mean_precision_"),
    );
    assert_close(
        &p.degrees_of_freedom,
        case.expect_f64(&format!("dof_{name}")),
        tol,
        &format!("{name}: degrees_of_freedom_"),
    );
    assert_close(
        &p.means,
        case.expect_f64(&format!("means_{name}")),
        tol,
        &format!("{name}: means_"),
    );
    assert_close(
        &p.covariances,
        case.expect_f64(&format!("cov_{name}")),
        tol,
        &format!("{name}: covariances_"),
    );
    assert_close(
        &p.precisions_cholesky,
        case.expect_f64(&format!("prec_chol_{name}")),
        tol,
        &format!("{name}: precisions_cholesky_"),
    );

    // -- the convergence record --------------------------------------------- //
    assert_close(
        &[fitted.lower_bound()],
        case.expect_f64(&format!("lower_bound_{name}")),
        tol,
        &format!("{name}: lower_bound_"),
    );
    assert_close(
        fitted.lower_bounds(),
        case.expect_f64(&format!("lower_bounds_{name}")),
        tol,
        &format!("{name}: lower_bounds_"),
    );
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

    // -- the scoring surface ------------------------------------------------ //
    if let Some(xq) = xq {
        let labels: Vec<f64> = fitted
            .predict_labels_host(xq, (NQ, D))
            .expect("predict")
            .into_iter()
            .map(|v| v as f64)
            .collect();
        assert_close(
            &labels,
            case.expect_f64(&format!("predict_{name}")),
            tol,
            &format!("{name}: predict"),
        );
        let proba: Vec<f64> = fitted
            .predict_proba_host(xq, (NQ, D))
            .expect("predict_proba")
            .into_iter()
            .map(host_to_f64)
            .collect();
        assert_close(
            &proba,
            case.expect_f64(&format!("proba_{name}")),
            tol,
            &format!("{name}: predict_proba"),
        );
        // sklearn's mixtures have no `predict_log_proba`, so mlrs's is pinned
        // against `ln(predict_proba)` — which is the definition, and catches a
        // narrowing or a missing normalization all the same.
        let logp: Vec<f64> = fitted
            .predict_log_proba_host(xq, (NQ, D))
            .expect("predict_log_proba")
            .into_iter()
            .map(|v| host_to_f64::<F>(v).exp())
            .collect();
        assert_close(
            &logp,
            case.expect_f64(&format!("proba_{name}")),
            tol,
            &format!("{name}: exp(predict_log_proba)"),
        );
        assert_close(
            &fitted
                .score_samples_host(xq, (NQ, D))
                .expect("score_samples"),
            case.expect_f64(&format!("score_samples_{name}")),
            tol,
            &format!("{name}: score_samples"),
        );
        assert_close(
            &[fitted.score_host(xq, (NQ, D)).expect("score")],
            case.expect_f64(&format!("score_{name}")),
            tol,
            &format!("{name}: score"),
        );
    }
}

fn rng_free_body<F>(dtype_tag: &str, tol: &Tolerance)
where
    F: Float + CubeElement + Pod,
{
    let case = load(dtype_tag);
    let x: Vec<F> = design(&case, "X");
    let xq: Vec<F> = design(&case, "Xq");

    for cov in COV_TYPES {
        for (ptype, ptag) in PRIOR_TYPES {
            // Converged, with the full scoring surface.
            assert_exact_case::<F>(
                &case,
                &x,
                Some(&xq),
                &format!("k1{cov}_{ptag}"),
                &Case::k1(cov, ptype, 1e-8, 200),
                tol,
            );
            // ONE iteration at `tol = 0`: no room at all for two engines to
            // reach the same place by different routes.
            assert_exact_case::<F>(
                &case,
                &x,
                None,
                &format!("k1i{cov}_{ptag}"),
                &Case::k1(cov, ptype, 0.0, 1),
                tol,
            );
        }
    }
}

/// The RNG-free `k = 1` cases, compared exactly, f32.
#[test]
fn rng_free_exact_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    rng_free_body::<f32>("f32", &F32_TOL);
}

/// The RNG-free `k = 1` cases, compared exactly, f64.
#[test]
fn rng_free_exact_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    rng_free_body::<f64>("f64", &F64_TOL);
}

// ---------------------------------------------------------------------------
// Family 3 — the five priors, swept off their defaults
// ---------------------------------------------------------------------------

fn prior_sweep_body<F>(dtype_tag: &str, tol: &Tolerance)
where
    F: Float + CubeElement + Pod,
{
    let case = load(dtype_tag);
    let x: Vec<F> = design(&case, "X");
    // The same values `gen_bayesian_mixture`'s `prior_sweep` sets, each far
    // enough from the default that ignoring the parameter cannot pass.
    let mean_prior: Vec<f64> = vec![1.0, -2.0, 0.5, 3.0];

    for cov in COV_TYPES {
        for (ptype, ptag) in PRIOR_TYPES {
            for i in 0..5 {
                let mut spec = Case::k1(cov, ptype, 0.0, 1);
                match i {
                    0 => spec.weight_concentration_prior = Some(0.01),
                    1 => spec.mean_precision_prior = Some(5.0),
                    2 => spec.degrees_of_freedom_prior = Some(D as f64 + 3.5),
                    3 => spec.mean_prior = Some(mean_prior.clone()),
                    // `covariance_prior`'s shape depends on `covariance_type`,
                    // so its value is read back from the fixture rather than
                    // duplicated here.
                    _ => {
                        spec.covariance_prior =
                            Some(case.expect_f64(&format!("cpin_{cov}")).to_vec())
                    }
                }
                assert_exact_case::<F>(&case, &x, None, &format!("pr{i}{cov}_{ptag}"), &spec, tol);
            }
        }
    }
}

/// Each of the five prior hyperparameters changes the fit exactly as sklearn's
/// does, f32.
#[test]
fn prior_sweep_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    prior_sweep_body::<f32>("f32", &F32_TOL);
}

/// Each of the five prior hyperparameters changes the fit exactly as sklearn's
/// does, f64.
#[test]
fn prior_sweep_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    prior_sweep_body::<f64>("f64", &F64_TOL);
}

// ---------------------------------------------------------------------------
// Family 4 — the weight-posterior recursion on a fixed `nk`
// ---------------------------------------------------------------------------

/// The stick-breaking / Dirichlet weight update, evaluated at `k = 5` on the
/// fixture's fixed, deliberately UNEQUAL `nk`.
///
/// This is the one piece of the estimator family 1 cannot pin, and the reason
/// is structural rather than incidental: under `dirichlet_process` component
/// `c`'s second Beta parameter is `γ + Σ_{j>c} nk_j`, so the update is
/// order-dependent and two engines whose initializations discover the same
/// clustering in a different order legitimately disagree. Fixing `nk` removes
/// the ordering question, and the unequal counts mean a transposed or reversed
/// cumulative sum cannot pass by symmetry.
///
/// Not dtype-parameterized: these are `f64` scalars on both sides of the
/// boundary, with no design and no narrowing involved.
#[test]
fn weight_posterior_recursion_matches_sklearn() {
    let case = load("f64");
    let nk = case.expect_f64("stick_nk").to_vec();
    let gamma = case.expect_f64("stick_prior")[0];

    for (ptype, ptag) in PRIOR_TYPES {
        let pt = prior_type_of(ptype);
        let (a, b) = weight_concentration(pt, gamma, &nk);
        assert_close(
            &a,
            case.expect_f64(&format!("stick_wca_{ptag}")),
            &F64_TOL,
            &format!("{ptag}: weight_concentration_[0]"),
        );
        if pt == WeightConcentrationPriorType::DirichletProcess {
            assert_close(
                &b,
                case.expect_f64(&format!("stick_wcb_{ptag}")),
                &F64_TOL,
                &format!("{ptag}: weight_concentration_[1]"),
            );
        } else {
            assert!(b.is_empty(), "{ptag}: expected no second Beta parameter");
        }
        assert_close(
            &mixing_weights(pt, &a, &b),
            case.expect_f64(&format!("stick_weights_{ptag}")),
            &F64_TOL,
            &format!("{ptag}: weights_"),
        );
        assert_close(
            &mlrs_algos::mixture::bayesian_gaussian_mixture::expected_log_weights(pt, &a, &b),
            case.expect_f64(&format!("stick_logw_{ptag}")),
            &F64_TOL,
            &format!("{ptag}: _estimate_log_weights()"),
        );
    }
}

// ---------------------------------------------------------------------------
// Parameter validation (D-08: build-time vs fit-time)
// ---------------------------------------------------------------------------

/// The data-INDEPENDENT rejections happen at `build()`, before any data exists.
#[test]
fn invalid_hyperparameters_are_rejected_at_build() {
    let b = || BayesianGaussianMixture::<f64>::builder().n_components(2);
    assert!(matches!(
        b().weight_concentration_prior_type("dirichlet")
            .build::<f64>(),
        Err(BuildError::UnknownWeightConcentrationPriorType { .. })
    ));
    assert!(matches!(
        b().covariance_type("blockdiag").build::<f64>(),
        Err(BuildError::UnknownCovarianceType { .. })
    ));
    assert!(matches!(
        b().init_params("kmeans++").build::<f64>(),
        Err(BuildError::UnknownInit { .. })
    ));
    assert!(matches!(
        b().weight_concentration_prior(Some(0.0)).build::<f64>(),
        Err(BuildError::InvalidPrior { .. })
    ));
    assert!(matches!(
        b().mean_precision_prior(Some(-1.0)).build::<f64>(),
        Err(BuildError::InvalidPrior { .. })
    ));
    assert!(matches!(
        b().tol(-1.0).build::<f64>(),
        Err(BuildError::InvalidTol { .. })
    ));
    assert!(matches!(
        b().reg_covar(-1e-6).build::<f64>(),
        Err(BuildError::InvalidRegCovar { .. })
    ));
    // sklearn's `max_iter` constraint is `closed="left"` at 0 — `0` is LEGAL
    // and means "report the initialization".
    assert!(b().max_iter(0).build::<f64>().is_ok());
}

/// The data-DEPENDENT rejections wait for `fit`, where `n_features` exists.
#[test]
fn invalid_priors_are_rejected_at_fit() {
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let x: Vec<f64> = (0..40).map(|i| (i % 7) as f64 * 0.5 - 1.0).collect();
    let shape = (10usize, 4usize);
    let build = |f: &dyn Fn(
        mlrs_algos::mixture::BayesianGaussianMixtureBuilder,
    ) -> mlrs_algos::mixture::BayesianGaussianMixtureBuilder| {
        f(BayesianGaussianMixture::<f64>::builder().n_components(2))
            .build::<f64>()
            .expect("hyperparameters are build-legal; the prior is data-invalid")
    };

    // ν₀ must exceed n_features − 1 = 3.
    assert!(build(&|b| b.degrees_of_freedom_prior(Some(3.0)))
        .fit_from_host_slice(&mut pool, &x, shape)
        .is_err());
    assert!(build(&|b| b.degrees_of_freedom_prior(Some(3.5)))
        .fit_from_host_slice(&mut pool, &x, shape)
        .is_ok());
    // m₀ must have length n_features.
    assert!(build(&|b| b.mean_prior(Some(vec![0.0; 3])))
        .fit_from_host_slice(&mut pool, &x, shape)
        .is_err());
    // W₀ must have the covariance_type's shape...
    assert!(build(&|b| b.covariance_prior(Some(vec![1.0; 9])))
        .fit_from_host_slice(&mut pool, &x, shape)
        .is_err());
    // ...be symmetric...
    let mut asym = vec![0.0f64; 16];
    for i in 0..4 {
        asym[i * 4 + i] = 1.0;
    }
    asym[1] = 0.5;
    assert!(build(&|b| b.covariance_prior(Some(asym.clone())))
        .fit_from_host_slice(&mut pool, &x, shape)
        .is_err());
    // ...and be positive definite.
    let mut indef = vec![0.0f64; 16];
    for i in 0..4 {
        indef[i * 4 + i] = if i == 2 { -1.0 } else { 1.0 };
    }
    assert!(build(&|b| b.covariance_prior(Some(indef.clone())))
        .fit_from_host_slice(&mut pool, &x, shape)
        .is_err());
    // n_components > n_samples is the other fit-time rejection.
    assert!(BayesianGaussianMixture::<f64>::builder()
        .n_components(11)
        .build::<f64>()
        .expect("11 components is build-legal")
        .fit_from_host_slice(&mut pool, &x, shape)
        .is_err());
}

/// The builder's defaults ARE `BayesianGaussianMixture::new`'s (BLDR-01).
#[test]
fn builder_defaults_match_new() {
    let a = BayesianGaussianMixture::<f64>::new();
    let b = BayesianGaussianMixture::<f64>::builder()
        .build::<f64>()
        .expect("the default builder is valid");
    assert!(a.hyperparams_eq(&b), "builder defaults drifted from new()");
}
