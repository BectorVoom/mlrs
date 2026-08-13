//! FSEL-PERSIST (prototype) — safetensors save/load round-trips for the six
//! `feature_selection` estimators.
//!
//! Every selector here reduces to one boolean mask: `transform` is a column
//! gather driven by `support_`, and everything else in these files is the
//! evidence behind it. So the round-trip gates are about that mask and the
//! evidence agreeing.
//!
//! The interesting half is what CANNOT round-trip. `SelectFromModel`, `Rfe`,
//! `Rfecv` and `SequentialFeatureSelector` are parameterized over a
//! caller-supplied estimator or fold scorer — a trait object with no on-disk
//! representation. mlrs does not paper over that: those four implement
//! `SaveModel` but not `LoadModel`, and expose `load_with(pool, path, estimator)`
//! instead. `the_meta_selectors_require_their_estimator` is the gate that shows
//! the reloaded selector transforms correctly from the file alone, and
//! `a_custom_importance_getter_refuses_to_load` covers the second closure —
//! silently substituting `Auto` would change what a re-fit selects.
//!
//! `a_custom_score_func_cannot_be_saved` is the mirror image on the write side:
//! `UnivariateFilter` with a closure score function fails to SAVE rather than
//! writing a file that would load as `f_classif` and rank every column
//! differently.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::AlgoError;
use mlrs_algos::feature_selection::fsel_persist::{
    AlignedBytes, FselFile, LoadModel, PersistError, SaveModel,
};
use mlrs_algos::feature_selection::{
    Cv, FoldScorer, ImportanceEstimator, Importances, KBest, NFeatures, Rfe, RfeStep, Rfecv,
    ScoreFunc, SelectFromModel, Selector, SequentialFeatureSelector, SfsTarget, Threshold,
    UnivariateFilter, VarianceThreshold,
};
use mlrs_algos::typestate::{Fit, Fitted, Transform};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

const N_SAMPLES: usize = 24;
const N_FEATURES: usize = 5;

/// A fixture whose five columns have clearly different variances and clearly
/// different relevance to `y`, so every selector below produces a NON-TRIVIAL
/// mask — one that keeps some columns and drops others. A mask of all-true would
/// make most of these gates pass without testing anything.
fn fixture<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES)
        .flat_map(|i| {
            let t = i as f64;
            [
                t * 0.5,         // high variance, strongly related to y
                t * 0.05,        // low variance
                (t * 0.7).sin(), // moderate, unrelated
                1.0,             // CONSTANT — zero variance, always dropped
                (t % 4.0) - 1.5, // moderate, weakly related
            ]
        })
        .map(mlrs_core::f64_to_host::<F>)
        .collect()
}

fn targets() -> Vec<f64> {
    let x = fixture::<f64>();
    (0..N_SAMPLES)
        .map(|i| 3.0 * x[i * N_FEATURES] + 0.5 * x[i * N_FEATURES + 4])
        .collect()
}

fn pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

fn upload<F: Float + CubeElement + Pod>(
    p: &mut BufferPool<ActiveRuntime>,
) -> DeviceArray<ActiveRuntime, F> {
    DeviceArray::from_host(p, &fixture::<F>())
}

/// A closed-form ridge, so the meta-selectors have a deterministic inner model.
/// Mirrors the one in `feature_selection_meta_test.rs`.
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
        // Gaussian elimination with partial pivoting, in place.
        for col in 0..d {
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
            if p.abs() < 1e-300 {
                continue;
            }
            for r in (col + 1)..d {
                let f = a[r * d + col] / p;
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
            let p = a[col * d + col];
            out[col] = if p.abs() < 1e-300 { 0.0 } else { acc / p };
        }
        Ok(Importances::Flat(out))
    }
}

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
        let (mut ss_res, mut ss_tot) = (0.0, 0.0);
        for r in 0..n_test {
            let pred: f64 = (0..d).map(|j| coef[j] * x_test[r * d + j]).sum();
            ss_res += (y_test[r] - pred) * (y_test[r] - pred);
            ss_tot += (y_test[r] - mean) * (y_test[r] - mean);
        }
        Ok(if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        })
    }
}

// ---------------------------------------------------------------------------
// The two selectors that round-trip completely
// ---------------------------------------------------------------------------

#[test]
fn variance_threshold_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("vt.safetensors");
    let mut p = pool();

    let x = upload::<f32>(&mut p);
    let fitted = VarianceThreshold::<f32>::with_threshold(0.05)
        .fit(&mut p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("VarianceThreshold fits the fixture");

    // The mask must be non-trivial, or every gate below passes for free.
    assert!(
        fitted.get_support().iter().any(|&b| b) && fitted.get_support().iter().any(|&b| !b),
        "the fixture must produce a mixed mask, got {:?}",
        fitted.get_support()
    );

    fitted.save(&p, &path).expect("save succeeds");
    let loaded: VarianceThreshold<f32, Fitted> =
        VarianceThreshold::load(&mut p, &path).expect("load succeeds");

    // `==` rather than a tolerance: the file stores the exact IEEE bits.
    assert_eq!(loaded.get_support(), fitted.get_support(), "support_");
    assert_eq!(loaded.variances(), fitted.variances(), "variances_");
    assert_eq!(loaded.threshold(), fitted.threshold(), "threshold");

    // And the observable: the reloaded selector gathers the same columns.
    let x = upload::<f32>(&mut p);
    let before = fitted
        .transform(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("transform succeeds")
        .to_host(&p);
    let x = upload::<f32>(&mut p);
    assert_eq!(
        loaded
            .transform(&mut p, &x, (N_SAMPLES, N_FEATURES))
            .expect("transform succeeds")
            .to_host(&p),
        before,
        "the reloaded selector must gather the same columns"
    );
}

#[test]
fn univariate_filter_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("uf.safetensors");
    let mut p = pool();

    let x = upload::<f32>(&mut p);
    let y: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(
        &mut p,
        &targets().iter().map(|&v| v as f32).collect::<Vec<_>>(),
    );
    let fitted = UnivariateFilter::<f32>::k_best(
        ScoreFunc::FRegression {
            center: true,
            force_finite: true,
        },
        KBest::Count(2),
    )
    .expect("UnivariateFilter builds")
    .fit(&mut p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
    .expect("UnivariateFilter fits the fixture");

    fitted.save(&p, &path).expect("save succeeds");
    let loaded: UnivariateFilter<f32, Fitted> =
        UnivariateFilter::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.get_support(), fitted.get_support(), "support_");
    assert_eq!(loaded.scores(), fitted.scores(), "scores_");
    assert_eq!(loaded.pvalues(), fitted.pvalues(), "pvalues_");
    assert!(
        fitted.pvalues().is_some(),
        "f_regression must produce p-values, or the gate above is vacuous"
    );
}

#[test]
fn a_score_function_without_pvalues_stores_none() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("uf-nopv.safetensors");
    let mut p = pool();

    // `r_regression` produces scores only. The absent tensor is MEANINGFUL —
    // it round-trips as the `None` the estimator holds — where a zero-filled
    // array would claim p-values the fit never computed.
    let x = upload::<f32>(&mut p);
    let y: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(
        &mut p,
        &targets().iter().map(|&v| v as f32).collect::<Vec<_>>(),
    );
    let fitted = UnivariateFilter::<f32>::k_best(
        ScoreFunc::RRegression {
            center: true,
            force_finite: true,
        },
        KBest::Count(2),
    )
    .expect("builds")
    .fit(&mut p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
    .expect("fits");
    assert!(
        fitted.pvalues().is_none(),
        "r_regression yields no p-values"
    );

    fitted.save(&p, &path).expect("save succeeds");
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = FselFile::parse(&raw, "univariate_filter").expect("parse succeeds");
    assert!(
        file.tensor_opt("pvalues_").is_none(),
        "an absent pvalues_ must write no tensor at all"
    );

    let loaded: UnivariateFilter<f32, Fitted> =
        UnivariateFilter::load(&mut p, &path).expect("load succeeds");
    assert!(
        loaded.pvalues().is_none(),
        "an absent pvalues_ must stay absent"
    );
}

#[test]
fn a_custom_score_func_cannot_be_saved() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("custom.safetensors");
    let mut p = pool();

    // The write-side mirror of the meta-selectors' load-side refusal. A closure
    // has no on-disk representation, so `save` FAILS rather than writing a file
    // that would load as `f_classif` and rank every column differently.
    let x = upload::<f32>(&mut p);
    let y: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(
        &mut p,
        &targets().iter().map(|&v| v as f32).collect::<Vec<_>>(),
    );
    let custom = ScoreFunc::Custom(std::sync::Arc::new(
        |_x: &[f64], _y: &[f64], _n: usize, d: usize| {
            Ok(mlrs_algos::feature_selection::ScoreResult {
                scores: (0..d).map(|i| i as f64).collect(),
                pvalues: None,
            })
        },
    ));
    let fitted = UnivariateFilter::<f32>::k_best(custom, KBest::Count(2))
        .expect("builds")
        .fit(&mut p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("fits");

    let err = match fitted.save(&p, &path) {
        Ok(()) => panic!("a custom score function must not be saved"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::MissingState { field, .. } if field.contains("score_func")),
        "expected a MissingState naming score_func, got {err:?}"
    );
    assert!(
        !path.exists(),
        "a refused save must leave no file behind at all"
    );
}

// ---------------------------------------------------------------------------
// The four that need their estimator back
// ---------------------------------------------------------------------------

#[test]
fn the_meta_selectors_require_their_estimator() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();
    let x_host = fixture::<f64>();
    let y_host = targets();

    // `SelectFromModel` — the fitted state is the mask plus the RESOLVED
    // threshold, and `load_with` takes the inner estimator the file cannot hold.
    let path = dir.path().join("sfm.safetensors");
    let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &x_host);
    let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &y_host);
    let sfm = SelectFromModel::<f64, _>::new(RidgeImportance { alpha: 1.0 })
        .with_threshold(Threshold::Scaled {
            scale: 1.0,
            median: false,
        })
        .fit(&mut p, &xd, Some(&yd), (N_SAMPLES, N_FEATURES))
        .expect("SelectFromModel fits");
    let sfm_support = sfm.get_support().to_vec();
    sfm.save(&p, &path).expect("save succeeds");
    let loaded =
        SelectFromModel::<f64, _, Fitted>::load_with(&mut p, &path, RidgeImportance { alpha: 1.0 })
            .expect("load_with succeeds");
    assert_eq!(
        loaded.get_support(),
        sfm_support,
        "SelectFromModel support_"
    );
    assert_eq!(
        loaded.threshold_value(),
        sfm.threshold_value(),
        "the RESOLVED threshold is not derivable from the mask"
    );

    // `Rfe` — adds the elimination ranking, which carries strictly more than the
    // mask and is cross-checked against it on load.
    let path = dir.path().join("rfe.safetensors");
    let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &x_host);
    let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &y_host);
    let rfe = Rfe::<f64, _>::new(RidgeImportance { alpha: 1.0 })
        .with_n_features_to_select(NFeatures::Count(2))
        .with_step(RfeStep::Count(1))
        .fit(&mut p, &xd, Some(&yd), (N_SAMPLES, N_FEATURES))
        .expect("Rfe fits");
    let rfe_support = rfe.get_support().to_vec();
    let rfe_ranking = rfe.ranking().to_vec();
    rfe.save(&p, &path).expect("save succeeds");
    let loaded = Rfe::<f64, _, Fitted>::load_with(&mut p, &path, RidgeImportance { alpha: 1.0 })
        .expect("load_with succeeds");
    assert_eq!(loaded.get_support(), rfe_support, "Rfe support_");
    assert_eq!(loaded.ranking(), rfe_ranking, "Rfe ranking_");

    // `Rfecv` — adds the whole `cv_results_` table, which is what a caller plots
    // and is not recoverable from the mask.
    let path = dir.path().join("rfecv.safetensors");
    let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &x_host);
    let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &y_host);
    let rfecv =
        Rfecv::<f64, _, _>::new(RidgeImportance { alpha: 1.0 }, R2FoldScorer { alpha: 1.0 })
            .with_cv(Cv::Folds {
                n_splits: 3,
                stratified: false,
            })
            .with_min_features_to_select(1)
            .fit(&mut p, &xd, Some(&yd), (N_SAMPLES, N_FEATURES))
            .expect("Rfecv fits");
    let rfecv_support = rfecv.get_support().to_vec();
    let mean_before = rfecv.cv_results().mean_test_score.clone();
    let ranking_before = rfecv.cv_results().split_ranking.clone();
    rfecv.save(&p, &path).expect("save succeeds");
    let loaded = Rfecv::<f64, _, _, Fitted>::load_with(
        &mut p,
        &path,
        RidgeImportance { alpha: 1.0 },
        R2FoldScorer { alpha: 1.0 },
    )
    .expect("load_with succeeds");
    assert_eq!(loaded.get_support(), rfecv_support, "Rfecv support_");
    assert_eq!(
        loaded.cv_results().mean_test_score,
        mean_before,
        "cv_results_.mean_test_score"
    );
    assert_eq!(
        loaded.cv_results().split_ranking,
        ranking_before,
        "cv_results_.split_ranking — the nested per-fold, per-subset table"
    );

    // `SequentialFeatureSelector` — the simplest, and the one whose resolved
    // count is cross-checked against the mask.
    let path = dir.path().join("sfs.safetensors");
    let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &x_host);
    let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &y_host);
    let sfs = SequentialFeatureSelector::<f64, _>::new(R2FoldScorer { alpha: 1.0 })
        .with_n_features_to_select(SfsTarget::Count(2))
        .with_cv(Cv::Folds {
            n_splits: 3,
            stratified: false,
        })
        .fit(&mut p, &xd, Some(&yd), (N_SAMPLES, N_FEATURES))
        .expect("SequentialFeatureSelector fits");
    let sfs_support = sfs.get_support().to_vec();
    sfs.save(&p, &path).expect("save succeeds");
    let loaded = SequentialFeatureSelector::<f64, _, Fitted>::load_with(
        &mut p,
        &path,
        R2FoldScorer { alpha: 1.0 },
    )
    .expect("load_with succeeds");
    assert_eq!(loaded.get_support(), sfs_support, "SFS support_");
}

// ---------------------------------------------------------------------------
// Rejection — the file is untrusted input (T-04-01-01)
// ---------------------------------------------------------------------------

#[test]
fn sibling_selectors_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("vt.safetensors");
    let mut p = pool();

    // Every selector in this family writes a `support_` of the same shape and
    // dtype, so the `estimator` tag is what keeps six different models apart.
    let x = upload::<f32>(&mut p);
    VarianceThreshold::<f32>::with_threshold(0.05)
        .fit(&mut p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("fits")
        .save(&p, &path)
        .expect("save succeeds");

    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (several hold device handles), so the Ok arm is rejected by hand.
    let err = match UnivariateFilter::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a variance_threshold file must not load as a univariate_filter"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "univariate_filter" && found == "variance_threshold"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_ranking_disagreeing_with_the_mask_is_rejected() {
    use mlrs_algos::feature_selection::fsel_persist::{pack_bools, FselWriter, TensorRef};

    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad-rank.safetensors");
    let mut p = pool();

    // sklearn's contract is that rank 1 means selected. A file whose mask says
    // "kept" where its ranking says 3 describes two different selections, and
    // each half is individually well-formed — only the cross-check catches it.
    let support = [true, false, true, false, false];
    let packed = pack_bools(&support);
    let ranking = [1u64, 2, 3, 4, 5]; // feature 2 is kept but ranked 3
    let mut w = FselWriter::new("rfe");
    w.scalar_str("param:n_features_to_select", "2");
    w.scalar_str("param:step", "1");
    w.scalar_usize("param:verbose", 0);
    w.scalar_bool("importance_getter_is_custom", false);
    w.tensor(
        "support_",
        TensorRef::bools(&packed, vec![5]).expect("well-formed"),
    );
    w.tensor(
        "ranking_",
        TensorRef::u64s(&ranking, vec![5]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match Rfe::<f64, _, Fitted>::load_with(&mut p, &path, RidgeImportance { alpha: 1.0 })
    {
        Ok(_) => panic!("a ranking disagreeing with the mask must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_custom_importance_getter_refuses_to_load() {
    use mlrs_algos::feature_selection::fsel_persist::{pack_bools, FselWriter, TensorRef};

    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("custom-getter.safetensors");
    let mut p = pool();

    // The custom getter is a closure, so it is not in the file. Loading would
    // hand back a selector whose `support_` came from one post-processor and
    // whose getter is `Auto` — a difference invisible until someone re-fits.
    let support = [true, false, true, false, false];
    let packed = pack_bools(&support);
    let mut w = FselWriter::new("select_from_model");
    w.scalar_str("param:threshold", "default");
    w.scalar_bool("param:prefit", false);
    w.scalar_f64("param:norm_order", 1.0);
    w.scalar_bool("importance_getter_is_custom", true);
    w.scalar_f64("threshold_", 0.5);
    w.tensor(
        "support_",
        TensorRef::bools(&packed, vec![5]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match SelectFromModel::<f64, _, Fitted>::load_with(
        &mut p,
        &path,
        RidgeImportance { alpha: 1.0 },
    ) {
        Ok(_) => panic!("a custom importance_getter must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { reason } if reason.contains("closure")),
        "expected a refusal naming the closure, got {err:?}"
    );
}

#[test]
fn saving_twice_produces_an_identical_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    // RAW BYTES: a model file must be a deterministic function of the model.
    // This is also the gate on the `third_party/safetensors` `BTreeMap` patch.
    let x = upload::<f32>(&mut p);
    let y: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(
        &mut p,
        &targets().iter().map(|&v| v as f32).collect::<Vec<_>>(),
    );
    let fitted = UnivariateFilter::<f32>::k_best(
        ScoreFunc::FRegression {
            center: true,
            force_finite: true,
        },
        KBest::Count(2),
    )
    .expect("builds")
    .fit(&mut p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
    .expect("fits");
    fitted.save(&p, &first).expect("save succeeds");
    fitted.save(&p, &second).expect("save succeeds");
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("vt.safetensors");
    let mut p = pool();

    let x = upload::<f32>(&mut p);
    VarianceThreshold::<f32>::with_threshold(0.05)
        .fit(&mut p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("fits")
        .save(&p, &path)
        .expect("save succeeds");

    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .expect("the scratch directory is readable")
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .filter(|n| n.to_string_lossy().contains("mlrs-tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a successful save must leave no temporary file, found {leftovers:?}"
    );
    assert!(path.exists(), "the model file must exist");
}
