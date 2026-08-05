//! `feature_selection` META-SELECTOR oracle tests (FSEL-01) — `SelectFromModel`,
//! `RFE`, `RFECV`, `SequentialFeatureSelector` against scikit-learn 1.9.0.
//!
//! ## Why these can be compared for EXACT equality
//! A meta-selector's answer is a function of its inner estimator, so comparing
//! against sklearn is only meaningful if BOTH sides run the same inner model. The
//! fixture is generated with `Ridge(alpha=1, fit_intercept=False)`, chosen because
//! its coefficients are a CLOSED FORM with no centering and no iteration:
//!
//! ```text
//! coef = (XᵀX + αI)⁻¹ Xᵀy
//! ```
//!
//! [`RidgeImportance`] below implements exactly that as an
//! [`ImportanceEstimator`], so the two sides agree on the coefficients to
//! floating-point rounding and the SELECTION — which is a discrete decision — is
//! then identical rather than merely close. A tree-based inner model would have
//! turned this file into a comparison of two RNG streams; a solver-based one into
//! a comparison of two convergence paths.
//!
//! `ridge_coef_full` is asserted FIRST for that reason: if the inner model
//! disagrees, every mask below disagrees too, and diagnosing that from a wrong
//! mask is much harder than from a wrong coefficient.
//!
//! `RFECV` and `SequentialFeatureSelector` additionally need a CV split and a
//! score. The fixture uses `cv=3` (unshuffled `KFold`) and sklearn's default `r2`
//! scorer, both of which are deterministic — [`R2FoldScorer`] and
//! [`Cv::Folds`]`{stratified: false}` reproduce them, which is why `Cv` documents
//! that it deliberately does not offer `shuffle=True`.
//!
//! Masks and rankings compare with `assert_eq!`; `cv_results_`'s scores are
//! continuous and take the 1e-5 band (D-09).
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::path::PathBuf;

use mlrs_algos::feature_selection::{
    Cv, FoldScorer, ImportanceEstimator, Importances, NFeatures, Rfe, RfeStep, Rfecv,
    SelectFromModel, Selector, SequentialFeatureSelector, SfsTarget, Threshold,
};
use mlrs_algos::typestate::Fit;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F64_TOL};

use mlrs_algos::error::AlgoError;

const N_SAMPLES: usize = 90;
const N_FEATURES: usize = 8;
/// The fixture's `Ridge(alpha=..)`; see `gen_feature_selection_oracle.py`.
const ALPHA: f64 = 1.0;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    root.join("tests").join("fixtures").join(name)
}

fn expect_mask(case: &OracleCase, name: &str) -> Vec<bool> {
    case.expect_f64(name).iter().map(|&v| v == 1.0).collect()
}

fn expect_usize(case: &OracleCase, name: &str) -> Vec<usize> {
    case.expect_f64(name).iter().map(|&v| v as usize).collect()
}

fn assert_close(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        assert!(
            err <= tol.abs + tol.rel * e.abs(),
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} err={err:e}"
        );
    }
}

// ===========================================================================
// The inner model: Ridge(alpha, fit_intercept=False) in closed form
// ===========================================================================

/// `sklearn.linear_model.Ridge(alpha, fit_intercept=False)`'s coefficients,
/// solved directly.
///
/// `coef = (XᵀX + αI)⁻¹ Xᵀy` via Gaussian elimination with partial pivoting. The
/// system is `d × d` with `d <= 8` here, so a dedicated solver would buy nothing;
/// what matters is that it is EXACT in the sense of having no convergence
/// tolerance for the two sides to stop at different points inside.
#[derive(Clone)]
struct RidgeImportance {
    alpha: f64,
}

impl ImportanceEstimator for RidgeImportance {
    fn fit_importances(
        &self,
        x: &[f64],
        y: &[f64],
        n: usize,
        d: usize,
    ) -> Result<Importances, AlgoError> {
        // Normal equations: A = XᵀX + αI, b = Xᵀy.
        let mut a = vec![0.0f64; d * d];
        let mut b = vec![0.0f64; d];
        for r in 0..n {
            let row = &x[r * d..r * d + d];
            for i in 0..d {
                b[i] += row[i] * y[r];
                for j in 0..d {
                    a[i * d + j] += row[i] * row[j];
                }
            }
        }
        for i in 0..d {
            a[i * d + i] += self.alpha;
        }
        Ok(Importances::Flat(solve(&mut a, &mut b, d)))
    }
}

/// Gaussian elimination with partial pivoting, in place.
fn solve(a: &mut [f64], b: &mut [f64], d: usize) -> Vec<f64> {
    for col in 0..d {
        // Partial pivot: the largest remaining |a[r][col]|.
        let mut pivot = col;
        for r in (col + 1)..d {
            if a[r * d + col].abs() > a[pivot * d + col].abs() {
                pivot = r;
            }
        }
        if pivot != col {
            for j in 0..d {
                a.swap(col * d + j, pivot * d + j);
            }
            b.swap(col, pivot);
        }
        let p = a[col * d + col];
        for r in (col + 1)..d {
            let f = a[r * d + col] / p;
            if f == 0.0 {
                continue;
            }
            for j in col..d {
                a[r * d + j] -= f * a[col * d + j];
            }
            b[r] -= f * b[col];
        }
    }
    let mut out = vec![0.0f64; d];
    for col in (0..d).rev() {
        let mut acc = b[col];
        for j in (col + 1)..d {
            acc -= a[col * d + j] * out[j];
        }
        out[col] = acc / a[col * d + col];
    }
    out
}

/// sklearn's default regression scorer, `r2` — `1 − SS_res/SS_tot` on the test
/// split, from a `Ridge` fitted on the train split.
///
/// `SS_tot` uses the TEST split's own mean, which is what `sklearn.metrics.r2_score`
/// does (not the train mean); getting that wrong shifts every fold score and
/// therefore `RFECV`'s chosen subset size.
struct R2FoldScorer {
    alpha: f64,
}

impl FoldScorer for R2FoldScorer {
    fn fit_score(
        &self,
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        x_test: &[f64],
        y_test: &[f64],
        n_test: usize,
        d: usize,
    ) -> Result<f64, AlgoError> {
        let model = RidgeImportance { alpha: self.alpha };
        let coef = match model.fit_importances(x_train, y_train, n_train, d)? {
            Importances::Flat(c) => c,
            Importances::Rows { values, .. } => values,
        };
        let mean = y_test.iter().sum::<f64>() / n_test as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for r in 0..n_test {
            let pred: f64 = (0..d).map(|j| coef[j] * x_test[r * d + j]).sum();
            ss_res += (y_test[r] - pred) * (y_test[r] - pred);
            ss_tot += (y_test[r] - mean) * (y_test[r] - mean);
        }
        Ok(1.0 - ss_res / ss_tot)
    }
}

/// The design and target, uploaded for the typestate `fit` calls.
fn upload(
    pool: &mut BufferPool<ActiveRuntime>,
    case: &OracleCase,
) -> (
    DeviceArray<ActiveRuntime, f64>,
    DeviceArray<ActiveRuntime, f64>,
) {
    (
        DeviceArray::from_host(pool, case.expect_f64("X")),
        DeviceArray::from_host(pool, case.expect_f64("y_reg")),
    )
}

fn case() -> OracleCase {
    load_npz(fixture("fsel_meta_f64_seed42.npz")).expect("load fsel_meta_f64")
}

/// The INNER MODEL first: if the ridge coefficients disagree with sklearn's, every
/// mask below disagrees too, and this is where that is diagnosable.
#[test]
fn the_inner_ridge_matches_sklearn() {
    let case = case();
    let model = RidgeImportance { alpha: ALPHA };
    let coef = match model
        .fit_importances(
            case.expect_f64("X"),
            case.expect_f64("y_reg"),
            N_SAMPLES,
            N_FEATURES,
        )
        .expect("ridge")
    {
        Importances::Flat(c) => c,
        Importances::Rows { values, .. } => values,
    };
    assert_close(
        &coef,
        case.expect_f64("ridge_coef_full"),
        &F64_TOL,
        "Ridge(alpha=1, fit_intercept=False).coef_",
    );
}

// ===========================================================================
// SelectFromModel
// ===========================================================================

#[test]
fn select_from_model_matches_sklearn() {
    let case = case();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // `(fixture tag, threshold, max_features)` — sklearn's threshold forms, plus
    // the `max_features` cap whose interaction with the threshold is ORDER
    // dependent (the cap picks the top-N first, then the threshold removes any of
    // those below it, so the result can be smaller than N but never larger).
    let cases: Vec<(&str, Threshold, Option<usize>)> = vec![
        ("none", Threshold::Default, None),
        (
            "mean",
            Threshold::Scaled {
                scale: 1.0,
                median: false,
            },
            None,
        ),
        (
            "median",
            Threshold::Scaled {
                scale: 1.0,
                median: true,
            },
            None,
        ),
        (
            "scaled",
            Threshold::parse("1.25*mean").expect("parse"),
            None,
        ),
        ("num", Threshold::Value(0.05), None),
        ("maxf3", Threshold::Default, Some(3)),
        ("maxf3_num", Threshold::Value(0.5), Some(3)),
    ];

    for (tag, threshold, max_features) in cases {
        let (x, y) = upload(&mut pool, &case);
        let fitted = SelectFromModel::<f64, _>::new(RidgeImportance { alpha: ALPHA })
            .with_threshold(threshold.clone())
            .with_max_features(max_features)
            .fit(&mut pool, &x, Some(&y), (N_SAMPLES, N_FEATURES))
            .unwrap_or_else(|e| panic!("SelectFromModel({tag}): {e}"));

        assert_eq!(
            fitted.get_support(),
            expect_mask(&case, &format!("sfm_{tag}_support")).as_slice(),
            "SelectFromModel({tag}): support mask"
        );
        assert_close(
            &[fitted.threshold_value()],
            case.expect_f64(&format!("sfm_{tag}_threshold")),
            &F64_TOL,
            &format!("SelectFromModel({tag}): threshold_"),
        );
    }
}

/// `threshold=None`'s documented meaning: `"mean"` for a non-L1 estimator.
///
/// sklearn decides this by INSPECTING the estimator's class name and `penalty`
/// attribute; a Rust trait object has neither, so `Threshold::Default` means
/// `"mean"` and an L1 caller passes `Value(1e-5)` explicitly. Asserted here
/// because it is the one place mlrs's Rust API deliberately differs in MECHANISM
/// from sklearn's while agreeing in RESULT — and the fixture's `sfm_none_*` and
/// `sfm_mean_*` arrays are what prove the result agrees.
#[test]
fn select_from_model_default_threshold_is_mean() {
    let case = case();
    assert_eq!(
        expect_mask(&case, "sfm_none_support"),
        expect_mask(&case, "sfm_mean_support"),
        "sklearn's threshold=None must equal threshold='mean' for a plain Ridge"
    );
}

// ===========================================================================
// RFE
// ===========================================================================

#[test]
fn rfe_matches_sklearn() {
    let case = case();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let cases: Vec<(&str, NFeatures, RfeStep)> = vec![
        ("default", NFeatures::Half, RfeStep::Count(1)),
        ("n3", NFeatures::Count(3), RfeStep::Count(1)),
        ("n3_step2", NFeatures::Count(3), RfeStep::Count(2)),
        ("frac", NFeatures::Fraction(0.5), RfeStep::Count(1)),
        // `step=0.3` is a FRACTION (sklearn's `0 < step < 1` branch), so it
        // removes `max(1, int(0.3 * 8)) = 2` features per iteration.
        ("stepfrac", NFeatures::Count(2), RfeStep::Fraction(0.3)),
    ];

    for (tag, n_features, step) in cases {
        let (x, y) = upload(&mut pool, &case);
        let fitted = Rfe::<f64, _>::new(RidgeImportance { alpha: ALPHA })
            .with_n_features_to_select(n_features)
            .with_step(step)
            .fit(&mut pool, &x, Some(&y), (N_SAMPLES, N_FEATURES))
            .unwrap_or_else(|e| panic!("RFE({tag}): {e}"));

        assert_eq!(
            fitted.get_support(),
            expect_mask(&case, &format!("rfe_{tag}_support")).as_slice(),
            "RFE({tag}): support mask"
        );
        assert_eq!(
            fitted.ranking(),
            expect_usize(&case, &format!("rfe_{tag}_ranking")).as_slice(),
            "RFE({tag}): ranking_"
        );
    }
}

/// `RFE` needs at least 2 features, as sklearn's `ensure_min_features=2` requires.
#[test]
fn rfe_rejects_a_single_feature() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = DeviceArray::from_host(&mut pool, &[1.0f64, 2.0, 3.0]);
    let y = DeviceArray::from_host(&mut pool, &[1.0f64, 2.0, 3.0]);
    assert!(Rfe::<f64, _>::new(RidgeImportance { alpha: ALPHA })
        .fit(&mut pool, &x, Some(&y), (3, 1))
        .is_err());
}

// ===========================================================================
// RFECV
// ===========================================================================

#[test]
fn rfecv_matches_sklearn() {
    let case = case();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let cases: Vec<(&str, usize, RfeStep)> = vec![
        ("cv3", 1, RfeStep::Count(1)),
        ("cv3_min3", 3, RfeStep::Count(1)),
        ("cv3_step2", 1, RfeStep::Count(2)),
    ];

    for (tag, min_features, step) in cases {
        let (x, y) = upload(&mut pool, &case);
        let fitted = Rfecv::<f64, _, _>::new(
            RidgeImportance { alpha: ALPHA },
            R2FoldScorer { alpha: ALPHA },
        )
        .with_cv(Cv::Folds {
            n_splits: 3,
            stratified: false,
        })
        .with_min_features_to_select(min_features)
        .with_step(step)
        .fit(&mut pool, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .unwrap_or_else(|e| panic!("RFECV({tag}): {e}"));

        assert_eq!(
            fitted.get_support(),
            expect_mask(&case, &format!("rfecv_{tag}_support")).as_slice(),
            "RFECV({tag}): support mask"
        );
        assert_eq!(
            fitted.ranking(),
            expect_usize(&case, &format!("rfecv_{tag}_ranking")).as_slice(),
            "RFECV({tag}): ranking_"
        );

        // `cv_results_` — ordered by ASCENDING feature count, which is the reverse
        // of the elimination order and part of the public attribute.
        let cv = fitted.cv_results();
        assert_eq!(
            cv.n_features,
            expect_usize(&case, &format!("rfecv_{tag}_n_features")),
            "RFECV({tag}): cv_results_['n_features']"
        );
        assert_close(
            &cv.mean_test_score,
            case.expect_f64(&format!("rfecv_{tag}_mean_test_score")),
            &F64_TOL,
            &format!("RFECV({tag}): mean_test_score"),
        );
        assert_close(
            &cv.std_test_score,
            case.expect_f64(&format!("rfecv_{tag}_std_test_score")),
            &F64_TOL,
            &format!("RFECV({tag}): std_test_score"),
        );
        assert_eq!(
            cv.split_test_score.len(),
            3,
            "RFECV({tag}): one split_test_score row per fold"
        );
    }
}

// ===========================================================================
// SequentialFeatureSelector
// ===========================================================================

#[test]
fn sequential_feature_selector_matches_sklearn() {
    use mlrs_algos::feature_selection::Direction;

    let case = case();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let cases: Vec<(&str, SfsTarget, Direction, Option<f64>)> = vec![
        ("fwd3", SfsTarget::Count(3), Direction::Forward, None),
        ("bwd3", SfsTarget::Count(3), Direction::Backward, None),
        // `'auto'` with no `tol` is `n_features // 2`.
        ("auto", SfsTarget::Auto, Direction::Forward, None),
        // `'auto'` WITH `tol` runs until the score stops improving by `tol`, so
        // the selected count is whatever the early stop landed on — which is why
        // `n_features_to_select_` is a fitted attribute rather than a parameter.
        ("tol", SfsTarget::Auto, Direction::Forward, Some(0.01)),
    ];

    for (tag, target, direction, tol) in cases {
        let (x, y) = upload(&mut pool, &case);
        let fitted = SequentialFeatureSelector::<f64, _>::new(R2FoldScorer { alpha: ALPHA })
            .with_n_features_to_select(target)
            .with_direction(direction)
            .with_tol(tol)
            .with_cv(Cv::Folds {
                n_splits: 3,
                stratified: false,
            })
            .fit(&mut pool, &x, Some(&y), (N_SAMPLES, N_FEATURES))
            .unwrap_or_else(|e| panic!("SFS({tag}): {e}"));

        assert_eq!(
            fitted.get_support(),
            expect_mask(&case, &format!("sfs_{tag}_support")).as_slice(),
            "SFS({tag}): support mask"
        );
        assert_eq!(
            fitted.n_features_to_select(),
            expect_usize(&case, &format!("sfs_{tag}_n_selected"))[0],
            "SFS({tag}): n_features_to_select_"
        );
    }
}

/// `n_features_to_select >= n_features` is rejected, and a NEGATIVE `tol` is
/// rejected for forward selection — both sklearn's own validations.
#[test]
fn sequential_feature_selector_rejects_out_of_domain_params() {
    use mlrs_algos::feature_selection::Direction;

    let case = case();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let (x, y) = upload(&mut pool, &case);
    assert!(
        SequentialFeatureSelector::<f64, _>::new(R2FoldScorer { alpha: ALPHA })
            .with_n_features_to_select(SfsTarget::Count(N_FEATURES))
            .fit(&mut pool, &x, Some(&y), (N_SAMPLES, N_FEATURES))
            .is_err(),
        "n_features_to_select must be < n_features"
    );

    let (x, y) = upload(&mut pool, &case);
    assert!(
        SequentialFeatureSelector::<f64, _>::new(R2FoldScorer { alpha: ALPHA })
            .with_tol(Some(-0.1))
            .with_direction(Direction::Forward)
            .fit(&mut pool, &x, Some(&y), (N_SAMPLES, N_FEATURES))
            .is_err(),
        "tol must be strictly positive for forward selection"
    );
}
