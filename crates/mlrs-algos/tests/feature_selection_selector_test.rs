//! `feature_selection` SELECTOR oracle tests (FSEL-01) — `VarianceThreshold`
//! and the six univariate filters, against scikit-learn 1.9.0.
//!
//! ## The support mask is compared for EXACT equality, not closeness
//! A selector's output is a discrete decision: a feature is kept or it is not.
//! Comparing masks with a tolerance would let an off-by-one selection pass, which
//! is the only failure mode that matters here — so every mask assert is
//! `assert_eq!` on the boolean vector. `scores_` / `pvalues_` / `variances_` keep
//! the usual 1e-5 abs+rel band (D-09), since those are continuous quantities.
//!
//! That split is what makes the fixture's deliberately degenerate design pay off:
//! the design has a constant column (`f_classif` → `NaN` score), an all-zero
//! column (`chi2` → `NaN`), and a DUPLICATE column (two features with bit-identical
//! scores). An implementation that ranks `NaN` as large, or that breaks a score
//! tie toward the lower index instead of the higher one, produces the same
//! `scores_` and a DIFFERENT mask — so only the exact mask comparison catches it.
//!
//! Every configuration below is a real sklearn call recorded by
//! `scripts/gen_feature_selection_oracle.py`; the parameter values sit on branch
//! boundaries (`k = 0`, `k = "all"`, `k > n_features`, `percentile ∈ {0, 100}`,
//! an `alpha` that selects nothing) rather than in their middles.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::feature_selection::{
    GenericParam, KBest, ScoreFunc, Selector, UnivariateFilter, VarianceThreshold,
};
use mlrs_algos::typestate::{Fit, Transform};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{f64_to_host, host_to_f64, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

const N_SAMPLES: usize = 90;
const N_FEATURES: usize = 8;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Upload a fixture array as the estimator's float type.
fn upload<F: Float + CubeElement + Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    values: &[f64],
) -> DeviceArray<ActiveRuntime, F> {
    let host: Vec<F> = values.iter().map(|&v| f64_to_host::<F>(v)).collect();
    DeviceArray::from_host(pool, &host)
}

/// The fixture's `float64` 0.0/1.0 mask as a `Vec<bool>` (the generator records
/// masks as floats because `load_npz` decodes only float dtypes).
fn expect_mask(case: &OracleCase, name: &str) -> Vec<bool> {
    case.expect_f64(name).iter().map(|&v| v == 1.0).collect()
}

fn assert_close(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() || g.is_nan() {
            assert!(
                e.is_nan() && g.is_nan(),
                "{what}: NaN mismatch at {i}: got={g:e} expected={e:e}"
            );
            continue;
        }
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= tol.abs + tol.rel * e.abs(),
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} abs_err={abs_err:e}"
        );
    }
}

// ===========================================================================
// VarianceThreshold
// ===========================================================================

fn run_variance_threshold<F: Float + CubeElement + Pod>(
    case: &OracleCase,
    tol: &Tolerance,
    tag: &str,
) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    for (t, key) in [(0.0, "0p0"), (0.25, "0p25"), (1.0, "1p0")] {
        let x = upload::<F>(&mut pool, case.expect_f64("X"));
        let fitted = VarianceThreshold::<F>::with_threshold(t)
            .fit(&mut pool, &x, None, (N_SAMPLES, N_FEATURES))
            .expect("VarianceThreshold::fit");
        assert_close(
            fitted.variances(),
            case.expect_f64(&format!("vt_{key}_variances")),
            tol,
            &format!("variances_ {tag} threshold={t}"),
        );
        assert_eq!(
            fitted.get_support(),
            expect_mask(case, &format!("vt_{key}_support")).as_slice(),
            "support {tag} threshold={t}"
        );
    }

    // The NaN design — the one selector that ACCEPTS non-finite input
    // (`ensure_all_finite="allow-nan"`) and computes `np.nanvar`. Two columns
    // carry NaNs with DIFFERENT non-NaN counts, so a per-column count is what
    // the variance must be normalised by.
    let x = upload::<F>(&mut pool, case.expect_f64("X_nan"));
    let fitted = VarianceThreshold::<F>::with_threshold(0.0)
        .fit(&mut pool, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("VarianceThreshold::fit on NaN input");
    assert_close(
        fitted.variances(),
        case.expect_f64("vt_nan_variances"),
        tol,
        &format!("variances_ {tag} nan design"),
    );
    assert_eq!(
        fitted.get_support(),
        expect_mask(case, "vt_nan_support").as_slice(),
        "support {tag} nan design"
    );
}

#[test]
fn variance_threshold_matches_sklearn_f32() {
    let case = load_npz(fixture("fsel_variance_f32_seed42.npz")).expect("load fsel_variance_f32");
    run_variance_threshold::<f32>(&case, &F32_TOL, "f32");
}

#[test]
fn variance_threshold_matches_sklearn_f64() {
    let case = load_npz(fixture("fsel_variance_f64_seed42.npz")).expect("load fsel_variance_f64");
    run_variance_threshold::<f64>(&case, &F64_TOL, "f64");
}

/// An all-dropped fit RAISES, carrying sklearn's message verbatim.
///
/// sklearn's `VarianceThreshold` is alone in refusing to produce an empty
/// selector: it raises `ValueError("No feature in X meets the variance threshold
/// {:.5f}")`, with a `(X contains only one sample)` suffix when `n == 1`. Both
/// halves are asserted because a caller matching on the message sees both.
#[test]
fn variance_threshold_rejects_an_all_dropped_fit() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    // Two constant columns: every variance is 0, nothing exceeds threshold 0.
    let x = upload::<f64>(&mut pool, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    let err = VarianceThreshold::<f64>::new()
        .fit(&mut pool, &x, None, (3, 2))
        .expect_err("an all-constant design must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("No feature in X meets the variance threshold"),
        "message should be sklearn's, got: {msg}"
    );

    // The single-sample suffix.
    let x1 = upload::<f64>(&mut pool, &[1.0, 2.0]);
    let err = VarianceThreshold::<f64>::new()
        .fit(&mut pool, &x1, None, (1, 2))
        .expect_err("a one-row design must be rejected");
    assert!(
        err.to_string().contains("(X contains only one sample)"),
        "one-sample suffix missing from: {err}"
    );
}

// ===========================================================================
// The six univariate filters
// ===========================================================================

/// Fit one filter and assert its mask, `scores_` and `pvalues_`.
fn check_filter<F: Float + CubeElement + Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    case: &OracleCase,
    tol: &Tolerance,
    name: &str,
    filter: UnivariateFilter<F>,
) {
    let x = upload::<F>(pool, case.expect_f64("X"));
    let y = upload::<F>(pool, case.expect_f64("y_class"));
    let fitted = filter
        .fit(pool, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .unwrap_or_else(|e| panic!("{name}: fit failed: {e}"));

    assert_eq!(
        fitted.get_support(),
        expect_mask(case, &format!("{name}_support")).as_slice(),
        "{name}: support mask"
    );
    assert_close(
        fitted.scores(),
        case.expect_f64(&format!("{name}_scores")),
        tol,
        &format!("{name}: scores_"),
    );
    if let Some(want) = case.f64(&format!("{name}_pvalues")) {
        assert_close(
            fitted.pvalues().expect("fixture recorded pvalues_"),
            want,
            tol,
            &format!("{name}: pvalues_"),
        );
    }
}

fn run_univariate<F: Float + CubeElement + Pod>(case: &OracleCase, tol: &Tolerance) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // SelectKBest: 0 (nothing), 1, 3, 12 (> n_features → keep all), "all".
    for (k, name) in [
        (KBest::Count(0), "kbest_0"),
        (KBest::Count(1), "kbest_1"),
        (KBest::Count(3), "kbest_3"),
        (KBest::Count(12), "kbest_12"),
        (KBest::All, "kbest_all"),
    ] {
        let f = UnivariateFilter::<F>::k_best(ScoreFunc::FClassif, k).expect("build");
        check_filter(&mut pool, case, tol, name, f);
    }
    // SelectKBest with chi2 rather than the default score function.
    let f = UnivariateFilter::<F>::k_best(ScoreFunc::Chi2, KBest::Count(3)).expect("build");
    check_filter(&mut pool, case, tol, "kbest_chi2_3", f);

    // SelectPercentile, including both short-circuit endpoints and a value that
    // lands the threshold BETWEEN two scores (37.5) so the tie-refill runs.
    for (p, name) in [
        (0.0, "percentile_0"),
        (25.0, "percentile_25"),
        (37.5, "percentile_37.5"),
        (50.0, "percentile_50"),
        (100.0, "percentile_100"),
    ] {
        let f = UnivariateFilter::<F>::percentile(ScoreFunc::FClassif, p).expect("build");
        check_filter(&mut pool, case, tol, name, f);
    }

    // The three p-value filters, at alphas that select most / some / none.
    for (a, name) in [(1e-8, "1e-08"), (0.05, "0.05"), (0.5, "0.5")] {
        let f = UnivariateFilter::<F>::fpr(ScoreFunc::FClassif, a).expect("build");
        check_filter(&mut pool, case, tol, &format!("fpr_{name}"), f);
        let f = UnivariateFilter::<F>::fdr(ScoreFunc::FClassif, a).expect("build");
        check_filter(&mut pool, case, tol, &format!("fdr_{name}"), f);
        let f = UnivariateFilter::<F>::fwe(ScoreFunc::FClassif, a).expect("build");
        check_filter(&mut pool, case, tol, &format!("fwe_{name}"), f);
    }

    // GenericUnivariateSelect over every mode, which must reproduce the
    // corresponding specific class exactly (that is its definition).
    for (mode, param, name) in [
        (
            "percentile",
            GenericParam::Value(30.0),
            "generic_percentile_30",
        ),
        ("k_best", GenericParam::Value(4.0), "generic_k_best_4"),
        ("k_best", GenericParam::All, "generic_k_best_all"),
        ("fpr", GenericParam::Value(0.05), "generic_fpr_0.05"),
        ("fdr", GenericParam::Value(0.05), "generic_fdr_0.05"),
        ("fwe", GenericParam::Value(0.05), "generic_fwe_0.05"),
    ] {
        let f = UnivariateFilter::<F>::generic(ScoreFunc::FClassif, mode, param).expect("build");
        check_filter(&mut pool, case, tol, name, f);
    }
}

#[test]
fn univariate_filters_match_sklearn_f32() {
    let case =
        load_npz(fixture("fsel_univariate_f32_seed42.npz")).expect("load fsel_univariate_f32");
    run_univariate::<f32>(&case, &F32_TOL);
}

#[test]
fn univariate_filters_match_sklearn_f64() {
    let case =
        load_npz(fixture("fsel_univariate_f64_seed42.npz")).expect("load fsel_univariate_f64");
    run_univariate::<f64>(&case, &F64_TOL);
}

/// `transform` / `inverse_transform` — the DEVICE column gather and scatter.
///
/// The only test here that exercises `mlrs_kernels::feature_select`; everything
/// above compares masks and host statistics. `inverse_transform` must restore the
/// original GEOMETRY with ZEROS in the dropped columns, which is what sklearn
/// defines it as (a selector discards information, so the inverse cannot restore
/// values).
fn run_transform<F: Float + CubeElement + Pod>(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = upload::<F>(&mut pool, case.expect_f64("X"));
    let y = upload::<F>(&mut pool, case.expect_f64("y_class"));
    let fitted = UnivariateFilter::<F>::k_best(ScoreFunc::FClassif, KBest::Count(3))
        .expect("build")
        .fit(&mut pool, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("fit");
    let k = fitted.n_features_out();
    assert_eq!(k, 3, "SelectKBest(k=3) must keep 3 features");

    let z = fitted
        .transform(&mut pool, &x, (N_SAMPLES, N_FEATURES))
        .expect("transform");
    let got: Vec<f64> = z.to_host(&pool).into_iter().map(host_to_f64).collect();
    assert_close(
        &got,
        case.expect_f64("kbest_3_transform"),
        tol,
        &format!("transform {tag}"),
    );

    let inv = fitted
        .inverse_transform(&mut pool, &z, (N_SAMPLES, k))
        .expect("inverse_transform");
    let got: Vec<f64> = inv.to_host(&pool).into_iter().map(host_to_f64).collect();
    assert_close(
        &got,
        case.expect_f64("kbest_3_inverse"),
        tol,
        &format!("inverse_transform {tag}"),
    );
}

#[test]
fn selector_transform_matches_sklearn_f32() {
    let case =
        load_npz(fixture("fsel_univariate_f32_seed42.npz")).expect("load fsel_univariate_f32");
    run_transform::<f32>(&case, &F32_TOL, "f32");
}

#[test]
fn selector_transform_matches_sklearn_f64() {
    let case =
        load_npz(fixture("fsel_univariate_f64_seed42.npz")).expect("load fsel_univariate_f64");
    run_transform::<f64>(&case, &F64_TOL, "f64");
}

/// The hyperparameter domains sklearn's `_parameter_constraints` enforce, and the
/// mode/score-function mismatch sklearn fails on deep inside `_get_support_mask`.
#[test]
fn univariate_filters_reject_out_of_domain_params() {
    // percentile ∉ [0, 100], alpha ∉ [0, 1].
    assert!(UnivariateFilter::<f64>::percentile(ScoreFunc::FClassif, 101.0).is_err());
    assert!(UnivariateFilter::<f64>::percentile(ScoreFunc::FClassif, -1.0).is_err());
    assert!(UnivariateFilter::<f64>::fpr(ScoreFunc::FClassif, 1.5).is_err());
    assert!(UnivariateFilter::<f64>::fdr(ScoreFunc::FClassif, -0.1).is_err());
    assert!(UnivariateFilter::<f64>::fwe(ScoreFunc::FClassif, 2.0).is_err());
    // An unknown mode names the accepted set.
    let err =
        UnivariateFilter::<f64>::generic(ScoreFunc::FClassif, "kbest", GenericParam::Value(3.0))
            .expect_err("'kbest' is not a mode; the sklearn spelling is 'k_best'");
    assert!(
        err.to_string()
            .contains("percentile, k_best, fpr, fdr, fwe"),
        "unknown-mode error should list the accepted modes, got: {err}"
    );

    // A p-value mode with a scores-only score function: sklearn fails with a
    // TypeError inside `_get_support_mask`; mlrs reports it as a typed error at
    // `fit`, naming both sides.
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = upload::<f64>(&mut pool, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = upload::<f64>(&mut pool, &[1.0, 2.0, 3.0]);
    let err = UnivariateFilter::<f64>::fdr(
        ScoreFunc::RRegression {
            center: true,
            force_finite: true,
        },
        0.05,
    )
    .expect("build")
    .fit(&mut pool, &x, Some(&y), (3, 2))
    .expect_err("SelectFdr cannot use a scores-only score function");
    assert!(
        err.to_string().contains("requires p-values"),
        "mode/score-func mismatch should say so, got: {err}"
    );
}
