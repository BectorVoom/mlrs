//! KERNEL-PARAMS — the `KernelRidge` full-parameter surface at the RUST layer.
//!
//! The live sklearn comparison for this surface is
//! `crates/mlrs-py/python/tests/test_oracle_kernel_ridge_params.py`, and it is
//! the authority on numerical agreement. What that suite cannot reach is the
//! engine's own invariants, which is what this file is for: the properties that
//! must hold whether or not sklearn is installed, and that would still be true
//! if sklearn changed.
//!
//! Each test pins one such invariant:
//!
//! * a per-target `alpha` vector produces EXACTLY the per-target independent
//!   fits (and a uniform vector produces exactly the scalar fit, so the
//!   `one_alpha` fast path is an optimisation and not a second computation);
//! * an integer `sample_weight` produces the same fit as physically duplicating
//!   the rows, which is what a weight MEANS and is not checkable against
//!   sklearn without inheriting sklearn's own rounding;
//! * `precomputed` fed a kernel matrix agrees with the kernel that produced it,
//!   which is what makes the callable-kernel route in the shim sound;
//! * every `KernelKind` name round-trips through `from_name`/`name`, so a new
//!   family cannot be added without a model-file spelling;
//! * the validation gates reject what sklearn's parameter constraints reject.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::AlgoError;
use mlrs_algos::kernel_ridge::{KernelKind, KernelRidge};
use mlrs_algos::typestate::{Fit as TypestateFit, Predict as TypestatePredict};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::kernel_matrix::{kernel_matrix, Kernel};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{f64_to_host, host_to_f64};

const N: usize = 16;
const D: usize = 3;

/// `Result::expect_err`, without its `T: Debug` bound.
///
/// Every fallible call here returns a fitted `KernelRidge` or a `PrimError` on
/// success, and neither is `Debug` — a fitted estimator holds device handles,
/// which have nothing meaningful to print. Deriving `Debug` on the estimator to
/// satisfy a test assertion would be the tail wagging the dog.
trait ErrOrPanic<E> {
    fn err_or_panic(self, msg: &str) -> E;
}

impl<T, E> ErrOrPanic<E> for Result<T, E> {
    fn err_or_panic(self, msg: &str) -> E {
        match self {
            Ok(_) => panic!("{msg}"),
            Err(e) => e,
        }
    }
}

fn pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

/// A small NON-NEGATIVE deterministic design. Non-negative so the same matrix
/// serves the chi² families as well as the rest — the point of a shared design
/// is that a cross-kernel disagreement is about the kernel.
fn x_host<F: Pod>() -> Vec<F> {
    (0..N * D)
        .map(|i| f64_to_host::<F>(0.2 + ((i * 7) % 11) as f64 * 0.13))
        .collect()
}

/// `t` target columns, row-major `N × t`, each a different smooth function of
/// the row index so a per-target mix-up is visible.
fn y_host<F: Pod>(t: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(N * t);
    for i in 0..N {
        for j in 0..t {
            v.push(f64_to_host::<F>(
                (i as f64) * 0.31 + (j as f64) * 2.0 - 1.5,
            ));
        }
    }
    v
}

fn upload<F: Float + CubeElement + Pod>(
    p: &mut BufferPool<ActiveRuntime>,
    v: &[F],
) -> DeviceArray<ActiveRuntime, F> {
    DeviceArray::from_host(p, v)
}

/// Fit and return the host duals, for an arbitrary builder configuration.
fn duals<F>(
    p: &mut BufferPool<ActiveRuntime>,
    kind: KernelKind,
    alphas: Vec<f64>,
    x: &[F],
    (n, d): (usize, usize),
    y: &[F],
    sample_weight: Option<&[F]>,
) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let xd = upload(p, x);
    let yd = upload(p, y);
    let est = KernelRidge::<F>::builder()
        .kernel(kind)
        .alphas(alphas)
        .gamma(Some(0.4))
        .build::<F>()
        .expect("builder accepts the configuration");
    let fitted = est
        .fit_weighted(p, &xd, Some(&yd), (n, d), sample_weight)
        .expect("fit succeeds");
    fitted.dual_coef(p)
}

fn max_abs_diff<F: Pod>(a: &[F], b: &[F]) -> f64 {
    assert_eq!(a.len(), b.len(), "compared vectors must have equal length");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (host_to_f64(x) - host_to_f64(y)).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// alpha — scalar, uniform vector, per-target vector
// ---------------------------------------------------------------------------

#[test]
fn a_uniform_alpha_vector_is_bit_identical_to_the_scalar() {
    // The `one_alpha` test in `fit` routes a uniform vector to the SHARED
    // factorisation. If that were a different computation rather than a
    // shortcut, this would drift — and a caller who wrote `alpha=[2.0, 2.0]`
    // instead of `alpha=2.0` would silently get a different model.
    let mut p = pool();
    let (x, y) = (x_host::<f32>(), y_host::<f32>(2));
    let scalar = duals::<f32>(&mut p, KernelKind::Rbf, vec![2.0], &x, (N, D), &y, None);
    let vector = duals::<f32>(
        &mut p,
        KernelKind::Rbf,
        vec![2.0, 2.0],
        &x,
        (N, D),
        &y,
        None,
    );
    assert_eq!(scalar, vector, "a uniform alpha vector must take the scalar path");
}

#[test]
fn a_per_target_alpha_equals_independent_single_target_fits() {
    // The DEFINING property of the per-target penalty: column `j` of the
    // multi-target fit must equal the single-target fit under `alpha[j]`. This
    // is what the `zip` in sklearn's `_solve_cholesky_kernel` means, and it is
    // checkable here without sklearn.
    let mut p = pool();
    let x = x_host::<f32>();
    let y2 = y_host::<f32>(2);
    let alphas = vec![0.05, 9.0];

    let joint = duals::<f32>(&mut p, KernelKind::Rbf, alphas.clone(), &x, (N, D), &y2, None);

    for (j, &alpha) in alphas.iter().enumerate() {
        let col: Vec<f32> = (0..N).map(|i| y2[i * 2 + j]).collect();
        let solo = duals::<f32>(&mut p, KernelKind::Rbf, vec![alpha], &x, (N, D), &col, None);
        let joint_col: Vec<f32> = (0..N).map(|i| joint[i * 2 + j]).collect();
        assert!(
            max_abs_diff(&joint_col, &solo) <= 1e-5,
            "target {j} under alpha={alpha} must match its independent fit \
             (max abs diff {})",
            max_abs_diff(&joint_col, &solo)
        );
    }
}

#[test]
fn distinct_per_target_alphas_actually_produce_distinct_columns() {
    // The guard on the test above. If the per-target alphas were collapsed to
    // the first entry, the equality check would pass for target 0 and the whole
    // suite would still be green for the wrong reason.
    let mut p = pool();
    let (x, y) = (x_host::<f32>(), y_host::<f32>(2));
    let varied = duals::<f32>(
        &mut p,
        KernelKind::Rbf,
        vec![0.01, 100.0],
        &x,
        (N, D),
        &y,
        None,
    );
    let flat = duals::<f32>(&mut p, KernelKind::Rbf, vec![0.01], &x, (N, D), &y, None);
    let varied_col1: Vec<f32> = (0..N).map(|i| varied[i * 2 + 1]).collect();
    let flat_col1: Vec<f32> = (0..N).map(|i| flat[i * 2 + 1]).collect();
    assert!(
        max_abs_diff(&varied_col1, &flat_col1) > 1e-3,
        "a 10000x larger penalty on target 1 must change its duals"
    );
}

#[test]
fn a_mismatched_alpha_length_is_rejected() {
    let mut p = pool();
    let (x, y) = (x_host::<f32>(), y_host::<f32>(2));
    let xd = upload(&mut p, &x);
    let yd = upload(&mut p, &y);
    let est = KernelRidge::<f32>::builder()
        .alphas(vec![1.0, 2.0, 3.0])
        .build::<f32>()
        .expect("three alphas build fine — the target count is unknown here");
    let err = TypestateFit::fit(est, &mut p, &xd, Some(&yd), (N, D))
        .err_or_panic("three alphas against two targets must be rejected");
    assert!(
        matches!(
            err,
            AlgoError::AlphaTargetMismatch {
                n_alphas: 3,
                n_targets: 2,
                ..
            }
        ),
        "expected AlphaTargetMismatch, got {err}"
    );
}

#[test]
fn a_negative_alpha_anywhere_in_the_vector_is_rejected() {
    // Not just the first entry: a `find` over the whole vector, so a bad
    // penalty on the last target is caught as loudly as one on the first.
    let err = KernelRidge::<f32>::builder()
        .alphas(vec![1.0, 2.0, -0.5])
        .build::<f32>()
        .err_or_panic("a negative penalty must be rejected at build");
    assert!(err.to_string().contains("alpha"), "message names alpha: {err}");
}

#[test]
fn an_empty_alpha_vector_is_rejected() {
    let err = KernelRidge::<f32>::builder()
        .alphas(vec![])
        .build::<f32>()
        .err_or_panic("no penalty at all is not a configuration");
    assert!(err.to_string().contains("alpha"), "message names alpha: {err}");
}

// ---------------------------------------------------------------------------
// sample_weight
// ---------------------------------------------------------------------------

#[test]
fn an_integer_sample_weight_matches_duplicating_the_rows() {
    // What a sample weight MEANS: weight 2 on a row is that row appearing
    // twice. Checking the meaning rather than sklearn's arithmetic is what makes
    // this test independent of sklearn's rounding — and the dual problem gives
    // no reason to expect it a priori, since the duplicated design has a LARGER
    // kernel matrix and more dual coefficients. What must agree is the
    // PREDICTION, which lives in the same space either way.
    let mut p = pool();

    // Row i gets weight (i % 3) + 1, i.e. 1, 2 or 3 copies.
    let base_x = x_host::<f64>();
    let base_y = y_host::<f64>(1);
    let weights: Vec<usize> = (0..N).map(|i| (i % 3) + 1).collect();

    let mut dup_x: Vec<f64> = Vec::new();
    let mut dup_y: Vec<f64> = Vec::new();
    for (i, &w) in weights.iter().enumerate() {
        for _ in 0..w {
            dup_x.extend_from_slice(&base_x[i * D..(i + 1) * D]);
            dup_y.push(base_y[i]);
        }
    }
    let n_dup = dup_y.len();

    let sw: Vec<f64> = weights.iter().map(|&w| w as f64).collect();
    let query = x_host::<f64>();

    let predict = |p: &mut BufferPool<ActiveRuntime>,
                   x: &[f64],
                   y: &[f64],
                   n: usize,
                   w: Option<&[f64]>| {
        let xd = upload(p, x);
        let yd = upload(p, y);
        let fitted = KernelRidge::<f64>::builder()
            .kernel(KernelKind::Rbf)
            .alpha(0.5)
            .gamma(Some(0.4))
            .build::<f64>()
            .expect("builds")
            .fit_weighted(p, &xd, Some(&yd), (n, D), w)
            .expect("fits");
        let q = upload(p, &query);
        TypestatePredict::predict(&fitted, p, &q, (N, D))
            .expect("predicts")
            .to_host(p)
    };

    let weighted = predict(&mut p, &base_x, &base_y, N, Some(&sw));
    let duplicated = predict(&mut p, &dup_x, &dup_y, n_dup, None);
    let err = max_abs_diff(&weighted, &duplicated);
    assert!(
        err <= 1e-8,
        "weight-w rows must fit like w duplicated rows (max abs diff {err:.3e})"
    );
}

#[test]
fn a_uniform_sample_weight_rescales_alpha() {
    // A constant weight is NOT a no-op, and the exact way it is not is the
    // point. `α` lands on the diagonal AFTER the `S·K·S` scaling, so with
    // `S = sI` the system is `(s²K + αI)c̃ = s·y`, which unwinds to
    // `(K + (α/s²)I)c = y`: a constant weight `w = s²` divides the EFFECTIVE
    // penalty by `w`. The obvious guess — that a constant weight cancels — is
    // wrong, and asserting it would have looked like a defect in the weighting.
    let mut p = pool();
    let (x, y) = (x_host::<f64>(), y_host::<f64>(1));
    let w = 4.0;
    let sw = vec![w; N];

    let weighted = duals::<f64>(
        &mut p,
        KernelKind::Rbf,
        vec![0.5],
        &x,
        (N, D),
        &y,
        Some(&sw),
    );
    let rescaled = duals::<f64>(&mut p, KernelKind::Rbf, vec![0.5 / w], &x, (N, D), &y, None);
    assert!(
        max_abs_diff(&weighted, &rescaled) <= 1e-9,
        "weight w must be alpha/w (max abs diff {})",
        max_abs_diff(&weighted, &rescaled)
    );

    let plain = duals::<f64>(&mut p, KernelKind::Rbf, vec![0.5], &x, (N, D), &y, None);
    assert!(
        max_abs_diff(&weighted, &plain) > 1e-6,
        "a constant weight must move the fit — it rescales the penalty"
    );
}

#[test]
fn an_all_zero_sample_weight_is_rejected() {
    let mut p = pool();
    let (x, y) = (x_host::<f32>(), y_host::<f32>(1));
    let xd = upload(&mut p, &x);
    let yd = upload(&mut p, &y);
    let sw = vec![0.0f32; N];
    let err = KernelRidge::<f32>::builder()
        .build::<f32>()
        .expect("builds")
        .fit_weighted(&mut p, &xd, Some(&yd), (N, D), Some(&sw))
        .err_or_panic("weighting every sample out leaves nothing to fit");
    assert!(
        matches!(err, AlgoError::ZeroSampleWeightSum { .. }),
        "expected ZeroSampleWeightSum, got {err}"
    );
}

#[test]
fn a_negative_sample_weight_is_rejected() {
    let mut p = pool();
    let (x, y) = (x_host::<f32>(), y_host::<f32>(1));
    let xd = upload(&mut p, &x);
    let yd = upload(&mut p, &y);
    let mut sw = vec![1.0f32; N];
    sw[4] = -2.0;
    let err = KernelRidge::<f32>::builder()
        .build::<f32>()
        .expect("builds")
        .fit_weighted(&mut p, &xd, Some(&yd), (N, D), Some(&sw))
        .err_or_panic("sqrt of a negative weight would poison the whole solve");
    assert!(
        matches!(err, AlgoError::InvalidSampleWeight { index: 4, .. }),
        "expected InvalidSampleWeight at index 4, got {err}"
    );
}

// ---------------------------------------------------------------------------
// precomputed
// ---------------------------------------------------------------------------

#[test]
fn precomputed_agrees_with_the_kernel_that_produced_it() {
    // The soundness argument for the shim's callable-kernel route, checked at
    // the layer that implements it: feeding `precomputed` the matrix a named
    // kernel computes must reproduce the named kernel's model exactly.
    let mut p = pool();
    let x = x_host::<f64>();
    let y = y_host::<f64>(1);
    let query = x_host::<f64>();

    let xd = upload(&mut p, &x);
    let named = KernelRidge::<f64>::builder()
        .kernel(KernelKind::Rbf)
        .alpha(0.5)
        .gamma(Some(0.4))
        .build::<f64>()
        .expect("builds");
    let yd = upload(&mut p, &y);
    let named = TypestateFit::fit(named, &mut p, &xd, Some(&yd), (N, D)).expect("fits");
    let qd = upload(&mut p, &query);
    let named_pred = TypestatePredict::predict(&named, &mut p, &qd, (N, D))
        .expect("predicts")
        .to_host(&mut p);

    // The same Gram, computed through the prim and handed over as `precomputed`.
    let k = kernel_matrix::<f64>(
        &mut p,
        &xd,
        (N, D),
        &xd,
        (N, D),
        Kernel::Rbf { gamma: 0.4 },
        None,
    )
    .expect("kernel matrix");
    let k_host = k.to_host(&mut p);
    k.release_into(&mut p);
    let k_test = kernel_matrix::<f64>(
        &mut p,
        &qd,
        (N, D),
        &xd,
        (N, D),
        Kernel::Rbf { gamma: 0.4 },
        None,
    )
    .expect("cross kernel");
    let k_test_host = k_test.to_host(&mut p);
    k_test.release_into(&mut p);

    let kd = upload(&mut p, &k_host);
    let yd2 = upload(&mut p, &y);
    let pre = KernelRidge::<f64>::builder()
        .kernel(KernelKind::Precomputed)
        .alpha(0.5)
        .build::<f64>()
        .expect("builds");
    let pre = TypestateFit::fit(pre, &mut p, &kd, Some(&yd2), (N, N)).expect("fits");
    let ktd = upload(&mut p, &k_test_host);
    let pre_pred = TypestatePredict::predict(&pre, &mut p, &ktd, (N, N))
        .expect("predicts")
        .to_host(&mut p);

    assert!(
        max_abs_diff(&named_pred, &pre_pred) <= 1e-9,
        "precomputed must reproduce the kernel it was given (max abs diff {})",
        max_abs_diff(&named_pred, &pre_pred)
    );
}

#[test]
fn precomputed_rejects_a_non_square_fit_matrix() {
    let mut p = pool();
    let (x, y) = (x_host::<f32>(), y_host::<f32>(1));
    let xd = upload(&mut p, &x);
    let yd = upload(&mut p, &y);
    let err = KernelRidge::<f32>::builder()
        .kernel(KernelKind::Precomputed)
        .build::<f32>()
        .expect("builds")
        .fit(&mut p, &xd, Some(&yd), (N, D))
        .err_or_panic("a design matrix is not a kernel matrix");
    assert!(
        matches!(
            err,
            AlgoError::PrecomputedNotSquare {
                rows: N,
                cols: D,
                ..
            }
        ),
        "expected PrecomputedNotSquare, got {err}"
    );
}

// ---------------------------------------------------------------------------
// kernel names, gamma, degree
// ---------------------------------------------------------------------------

#[test]
fn every_kernel_kind_round_trips_through_its_name() {
    // The model file stores the family as this string. A variant added without
    // a `name`/`from_name` arm would round-trip to the wrong kernel — or, with
    // an integer tag, would silently renumber an existing file's.
    let all = [
        KernelKind::Linear,
        KernelKind::Rbf,
        KernelKind::Poly,
        KernelKind::Sigmoid,
        KernelKind::Laplacian,
        KernelKind::Cosine,
        KernelKind::Chi2,
        KernelKind::AdditiveChi2,
        KernelKind::Precomputed,
    ];
    for kind in all {
        assert_eq!(
            KernelKind::from_name(kind.name()),
            Some(kind),
            "{} must parse back to itself",
            kind.name()
        );
    }
    // The alias, which normalises rather than round-tripping.
    assert_eq!(KernelKind::from_name("polynomial"), Some(KernelKind::Poly));
    assert_eq!(KernelKind::from_name("gaussian"), None);
}

#[test]
fn chi2_has_no_gamma_default() {
    let mut p = pool();
    let (x, y) = (x_host::<f32>(), y_host::<f32>(1));
    let xd = upload(&mut p, &x);
    let yd = upload(&mut p, &y);
    let err = KernelRidge::<f32>::builder()
        .kernel(KernelKind::Chi2)
        .gamma(None)
        .build::<f32>()
        .expect("builds")
        .fit(&mut p, &xd, Some(&yd), (N, D))
        .err_or_panic("chi2 has no 1/n_features fallback, in sklearn or here");
    assert!(
        matches!(err, AlgoError::KernelRequiresGamma { kernel: "chi2", .. }),
        "expected KernelRequiresGamma, got {err}"
    );
}

#[test]
fn the_other_gamma_kernels_do_have_a_default() {
    // The other half of the rule above: without it, "chi2 raises" could be
    // satisfied by every kernel raising.
    let mut p = pool();
    let (x, y) = (x_host::<f32>(), y_host::<f32>(1));
    for kind in [
        KernelKind::Rbf,
        KernelKind::Poly,
        KernelKind::Sigmoid,
        KernelKind::Laplacian,
    ] {
        let xd = upload(&mut p, &x);
        let yd = upload(&mut p, &y);
        KernelRidge::<f32>::builder()
            .kernel(kind)
            .gamma(None)
            .build::<f32>()
            .expect("builds")
            .fit(&mut p, &xd, Some(&yd), (N, D))
            .unwrap_or_else(|e| panic!("{} must resolve gamma=None: {e}", kind.name()));
    }
}

#[test]
fn a_negative_gamma_is_rejected_at_build() {
    // Data-INDEPENDENT, so it belongs at `build()` beside `alpha` — sklearn's
    // interval is `[0, inf)`.
    let err = KernelRidge::<f32>::builder()
        .gamma(Some(-1.0))
        .build::<f32>()
        .err_or_panic("a negative kernel coefficient is out of domain");
    assert!(err.to_string().contains("gamma"), "message names gamma: {err}");
}

#[test]
fn a_fractional_degree_is_accepted_and_a_negative_one_is_not() {
    // sklearn's degree interval is `[0, inf)` over the REALS. The tighter
    // `>= 1` guard this once carried rejected `degree=0.5`, which sklearn fits.
    let mut p = pool();
    let (x, y) = (x_host::<f32>(), y_host::<f32>(1));

    let xd = upload(&mut p, &x);
    let yd = upload(&mut p, &y);
    KernelRidge::<f32>::builder()
        .kernel(KernelKind::Poly)
        .degree(0.5)
        .gamma(Some(0.4))
        .build::<f32>()
        .expect("builds")
        .fit(&mut p, &xd, Some(&yd), (N, D))
        .expect("a fractional polynomial degree is legal");

    let xd = upload(&mut p, &x);
    let yd = upload(&mut p, &y);
    let err = KernelRidge::<f32>::builder()
        .kernel(KernelKind::Poly)
        .degree(-1.0)
        .gamma(Some(0.4))
        .build::<f32>()
        .expect("builds")
        .fit(&mut p, &xd, Some(&yd), (N, D))
        .err_or_panic("a negative polynomial order is not");
    assert!(
        matches!(err, AlgoError::InvalidDegree { .. }),
        "expected InvalidDegree, got {err}"
    );
}

#[test]
fn the_chi2_kernels_reject_a_negative_input() {
    let mut p = pool();
    let mut x = x_host::<f32>();
    x[5] = -0.25;
    let y = y_host::<f32>(1);
    for kind in [KernelKind::Chi2, KernelKind::AdditiveChi2] {
        let xd = upload(&mut p, &x);
        let yd = upload(&mut p, &y);
        let err = KernelRidge::<f32>::builder()
            .kernel(kind)
            .gamma(Some(0.5))
            .build::<f32>()
            .expect("builds")
            .fit(&mut p, &xd, Some(&yd), (N, D))
            .err_or_panic("chi2 is defined on histogram-like data");
        assert!(
            matches!(err, AlgoError::NegativeKernelInput { index: 5, .. }),
            "{}: expected NegativeKernelInput at index 5, got {err}",
            kind.name()
        );
    }
}

#[test]
fn an_indefinite_gram_falls_back_to_least_squares_and_says_so() {
    // `additive_chi2` has a zero diagonal and non-positive entries elsewhere, so
    // `(K + I)` is indefinite and the Cholesky cannot factor it. sklearn
    // re-solves in the least-squares sense rather than failing; without the same
    // fallback the kernel would be unusable at every alpha.
    let mut p = pool();
    let (x, y) = (x_host::<f64>(), y_host::<f64>(1));
    let xd = upload(&mut p, &x);
    let yd = upload(&mut p, &y);
    let fitted = KernelRidge::<f64>::builder()
        .kernel(KernelKind::AdditiveChi2)
        .alpha(1.0)
        .build::<f64>()
        .expect("builds")
        .fit(&mut p, &xd, Some(&yd), (N, D))
        .expect("the least-squares fallback carries the fit");
    assert!(
        fitted.used_lstsq_fallback(),
        "additive_chi2 must report that it took the fallback"
    );
    // The fallback's answer is still an ANSWER: it must solve the system it was
    // given, which a Cholesky-shaped bug in the fallback would not.
    let duals = fitted.dual_coef(&p);
    assert!(
        duals.iter().all(|v| host_to_f64(*v).is_finite()),
        "the fallback must not produce non-finite duals"
    );
}

#[test]
fn a_definite_gram_does_not_take_the_fallback() {
    // The guard on the test above.
    let mut p = pool();
    let (x, y) = (x_host::<f64>(), y_host::<f64>(1));
    let xd = upload(&mut p, &x);
    let yd = upload(&mut p, &y);
    let fitted = KernelRidge::<f64>::builder()
        .kernel(KernelKind::Rbf)
        .gamma(Some(0.4))
        .build::<f64>()
        .expect("builds")
        .fit(&mut p, &xd, Some(&yd), (N, D))
        .expect("fits");
    assert!(
        !fitted.used_lstsq_fallback(),
        "an SPD rbf Gram must take the Cholesky"
    );
}
