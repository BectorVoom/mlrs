//! Rust-side oracle for the full metrics surface beyond the original eleven
//! (`prfs`, the rest of `classification`/`regression`, `ranking`, `cluster`,
//! `pairwise`). Every expected value below was produced by
//! `scikit-learn==1.9.1` on the same small inputs; the exhaustive randomized
//! comparison lives in `crates/mlrs-py/python/tests/test_oracle_metrics_surface.py`.
//! Per AGENTS.md §2 tests live here, never in-source.

use mlrs_algos::metrics::classification as cls;
use mlrs_algos::metrics::cluster as clu;
use mlrs_algos::metrics::pairwise::{self as pw, Distance, Kernel};
use mlrs_algos::metrics::prfs::{self, SetAverage};
use mlrs_algos::metrics::ranking as rk;
use mlrs_algos::metrics::regression as reg;
use mlrs_algos::metrics::{MetricError, MetricOut, MultiOutput, PrfOut, ZeroDivision};

const TOL: f64 = 1e-9;

fn close(got: f64, want: f64) {
    if want.is_nan() {
        assert!(got.is_nan(), "got {got}, want NaN");
        return;
    }
    if want.is_infinite() {
        assert_eq!(got, want);
        return;
    }
    assert!((got - want).abs() <= TOL * (1.0 + want.abs()), "got {got}, want {want}");
}

fn close_all(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "length: got {got:?}, want {want:?}");
    for (g, w) in got.iter().zip(want) {
        close(*g, *w);
    }
}

fn scalar(o: PrfOut) -> f64 {
    match o {
        PrfOut::Scalar(v) => v,
        PrfOut::PerClass(v) => panic!("expected scalar, got {v:?}"),
    }
}

fn mscalar(o: MetricOut) -> f64 {
    match o {
        MetricOut::Scalar(v) => v,
        MetricOut::Raw(v) => panic!("expected scalar, got {v:?}"),
    }
}

const YT: [i32; 8] = [0, 1, 2, 2, 1, 0, 1, 2];
const YP: [i32; 8] = [0, 2, 1, 2, 1, 0, 0, 2];
const SW: [f64; 8] = [1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 2.0, 1.0];

// ==================== prfs ====================

#[test]
fn prfs_macro_fbeta_matches_sklearn() {
    let r = prfs::prfs_multiclass(&YT, &YP, 2.0, None, 1, SetAverage::Macro, None, ZeroDivision::Zero)
        .unwrap();
    close(scalar(r.precision), 0.611111111111111);
    close(scalar(r.recall), 0.6666666666666666);
    close(scalar(r.fscore), 0.6443001443001443);
    assert!(r.support.is_none());
}

#[test]
fn prfs_weighted_with_sample_weight_matches_sklearn() {
    let r = prfs::prfs_multiclass(&YT, &YP, 1.0, None, 1, SetAverage::Weighted, Some(&SW), ZeroDivision::Zero)
        .unwrap();
    close(scalar(r.precision), 0.42105263157894735);
    close(scalar(r.recall), 0.47368421052631576);
    close(scalar(r.fscore), 0.39974937343358397);
}

#[test]
fn nan_zero_division_is_skipped_by_the_macro_average() {
    // sklearn's _nanaverage: [1, 1, NaN(no predicted 2... )] -> mean of the defined.
    let r = prfs::prfs_multiclass(
        &[0, 0, 1, 1, 2],
        &[0, 0, 1, 1, 1],
        1.0,
        None,
        1,
        SetAverage::Macro,
        None,
        ZeroDivision::Nan,
    )
    .unwrap();
    close(scalar(r.precision), 0.8333333333333333);
    assert!(r.precision_zd);
}

const MT: [i32; 12] = [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1];
const MP: [i32; 12] = [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1];

#[test]
fn prfs_multilabel_samples_average_matches_sklearn() {
    let r = prfs::prfs_indicator(&MT, &MP, 3, 1.0, None, SetAverage::Samples, None, ZeroDivision::Zero)
        .unwrap();
    close(scalar(r.precision), 0.875);
    close(scalar(r.recall), 0.75);
    close(scalar(r.fscore), 0.75);
    let (j, _) = prfs::jaccard_indicator(&MT, &MP, 3, None, SetAverage::Macro, None, ZeroDivision::Zero).unwrap();
    close(scalar(j), 0.6666666666666666);
}

#[test]
fn multilabel_confusion_matrix_matches_sklearn() {
    let m = prfs::multilabel_confusion_matrix(&YT, &YP, None, None).unwrap();
    close_all(&m, &[5.0, 1.0, 0.0, 2.0, 4.0, 1.0, 2.0, 1.0, 4.0, 1.0, 1.0, 2.0]);
}

#[test]
fn prfs_rejects_negative_beta_and_samples_on_1d() {
    let e = prfs::prfs_multiclass(&YT, &YP, -1.0, None, 1, SetAverage::Macro, None, ZeroDivision::Zero);
    assert_eq!(e.unwrap_err(), MetricError::InvalidParameter);
    let e = prfs::prfs_multiclass(&YT, &YP, 1.0, None, 1, SetAverage::Samples, None, ZeroDivision::Zero);
    assert_eq!(e.unwrap_err(), MetricError::UnsupportedTarget);
}

// ==================== classification ====================

#[test]
fn label_scores_match_sklearn() {
    let (b, dropped) = cls::balanced_accuracy_score(&YT, &YP, None, false).unwrap();
    close(b, 0.6666666666666666);
    assert!(!dropped);
    close(cls::balanced_accuracy_score(&YT, &YP, None, true).unwrap().0, 0.5);
    close(cls::matthews_corrcoef(&YT, &YP, Some(&SW)).unwrap(), 0.28319693016191544);
    let k = |w| match cls::cohen_kappa_score(&YT, &YP, None, w, None).unwrap() {
        cls::KappaOut::Value(v) => v,
        o => panic!("{o:?}"),
    };
    close(k(cls::KappaWeights::None_), 0.4418604651162791);
    close(k(cls::KappaWeights::Quadratic), 0.7272727272727273);
    close(cls::hamming_loss_indicator(&MT, &MP, 3, None).unwrap(), 0.25);
}

#[test]
fn hinge_and_probabilistic_scores_match_sklearn() {
    let dec = [
        1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.2, 0.3, 0.1, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0,
    ];
    close(cls::hinge_loss_multiclass(&YT, &dec, &[0, 1, 2], None).unwrap(), 0.3375);
    let y = [0, 1, 2, 1];
    let p = [0.8, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.1, 0.8, 0.3, 0.3, 0.4];
    close(cls::brier_score_loss(&y, &p, &[0, 1, 2], None, false).unwrap(), 0.25);
    close(cls::d2_brier_score(&y, &p, &[0, 1, 2], None).unwrap(), 0.6);
    close(cls::d2_log_loss_score(&y, &p, &[0, 1, 2], None, f64::EPSILON).unwrap(), 0.5174341738715518);
    let s = [0.5, 0.5, 0.0, 0.2, 0.7, 0.1, 0.4, 0.4, 0.2, 0.3, 0.3, 0.4];
    // Ties go to the HIGHER column index (sklearn's reversed stable sort).
    close(cls::top_k_accuracy_score(&y, &s, &[0, 1, 2], 1, true, None).unwrap(), 0.25);
    let lr = cls::class_likelihood_ratios(&[0, 1, 1, 0, 1, 0, 1], &[0, 1, 0, 1, 1, 0, 1], None, None).unwrap();
    close(lr.lr_pos, 2.25);
    close(lr.lr_neg, 0.375);
}

// ==================== regression ====================

const RT: [f64; 5] = [3.0, -0.5, 2.0, 7.0, 4.2];
const RP: [f64; 5] = [2.5, 0.0, 2.0, 8.0, 3.9];
const RW: [f64; 5] = [1.0, 2.0, 1.0, 1.0, 3.0];

#[test]
fn regression_scores_match_sklearn() {
    let u = MultiOutput::UniformAverage;
    close(mscalar(reg::explained_variance_score(&RT, &RP, 1, None, u, true).unwrap()), 0.9512290794979079);
    close(reg::max_error(&RT, &RP, 1).unwrap(), 1.0);
    close(mscalar(reg::mean_absolute_percentage_error(&RT, &RP, 1, Some(&RW), u).unwrap()), 0.31547619047619047);
    close(mscalar(reg::median_absolute_error(&RT, &RP, 1, None, u).unwrap()), 0.5);
    close(mscalar(reg::median_absolute_error(&RT, &RP, 1, Some(&RW), u).unwrap()), 0.4);
    close(mscalar(reg::mean_pinball_loss(&RT, &RP, 1, None, 0.9, u).unwrap()), 0.174);
    close(mscalar(reg::d2_pinball_score(&RT, &RP, 1, None, 0.9, u).unwrap()), 0.5492227979274609);
    let at: Vec<f64> = RT.iter().map(|v| v.abs() + 0.1).collect();
    let ap: Vec<f64> = RP.iter().map(|v| v.abs() + 0.1).collect();
    close(reg::mean_tweedie_deviance(&at, &ap, 1, None, 1.5).unwrap(), 0.5538479059841859);
    close(reg::d2_tweedie_score(&at, &ap, 1, None, 1.5).unwrap(), 0.371742036261135);
    let bt: Vec<f64> = RT.iter().map(|v| v.abs()).collect();
    let bp: Vec<f64> = RP.iter().map(|v| v.abs()).collect();
    close(mscalar(reg::root_mean_squared_log_error(&bt, &bp, 1, None, u).unwrap()), 0.1998182039246136);
}

#[test]
fn regression_domain_errors_are_typed() {
    let u = MultiOutput::UniformAverage;
    assert_eq!(
        reg::mean_squared_log_error(&[-2.0, 1.0], &[1.0, 1.0], 1, None, u).unwrap_err(),
        MetricError::DomainError
    );
    assert_eq!(reg::mean_tweedie_deviance(&[1.0, 1.0], &[1.0, 1.0], 1, None, 0.5).unwrap_err(), MetricError::InvalidParameter);
    assert_eq!(reg::max_error(&[1.0, 2.0], &[1.0, 2.0], 2).unwrap_err(), MetricError::UnsupportedTarget);
}

// ==================== ranking ====================

const YB: [i32; 6] = [0, 0, 1, 1, 0, 1];
const YS: [f64; 6] = [0.1, 0.4, 0.35, 0.8, 0.8, 0.2];

#[test]
fn curves_match_sklearn() {
    let r = rk::roc_curve(&YB, &YS, 1, None, true).unwrap();
    close_all(&r.fpr, &[0.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0]);
    close_all(&r.tpr, &[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0]);
    close_all(&r.thresholds, &[f64::INFINITY, 0.8, 0.4, 0.2, 0.1]);
    let (fpr, fnr, thr) = rk::det_curve(&YB, &YS, 1, None, false).unwrap();
    close_all(&fpr, &[2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]);
    close_all(&fnr, &[0.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0]);
    close_all(&thr, &[0.2, 0.35, 0.4, 0.8, f64::INFINITY]);
    close(rk::average_precision_binary(&YB, &YS, 1, None).unwrap(), 0.5333333333333333);
    close(rk::auc(&[0.0, 0.5, 1.0], &[0.0, 0.8, 1.0]).unwrap(), 0.65);
    assert_eq!(rk::auc(&[0.0, 2.0, 1.0], &[1.0, 1.0, 1.0]).unwrap_err(), MetricError::NonMonotonic);
}

#[test]
fn zero_weight_samples_are_not_thresholds() {
    let c = rk::binary_clf_curve(&[0, 1, 1], &[0.1, 0.5, 0.9], 1, Some(&[1.0, 0.0, 1.0])).unwrap();
    close_all(&c.thresholds, &[0.9, 0.1]);
}

#[test]
fn label_ranking_metrics_match_sklearn() {
    let t = [1, 0, 0, 0, 0, 1];
    let s = [0.75, 0.5, 1.0, 1.0, 0.2, 0.1];
    close(rk::coverage_error(&t, &s, 3, None).unwrap(), 2.5);
    close(rk::label_ranking_average_precision_score(&t, &s, 3, None).unwrap(), 0.41666666666666663);
    close(rk::label_ranking_loss(&t, &s, 3, None).unwrap(), 0.75);
    let g = [10.0, 0.0, 0.0, 1.0, 5.0];
    close(rk::dcg_score(&g, &[0.1, 0.2, 0.3, 4.0, 70.0], 5, None, 2.0, None, false).unwrap(), 9.499457825916874);
    close(rk::ndcg_score(&g, &[0.1, 0.2, 0.3, 4.0, 70.0], 5, None, None, false).unwrap(), 0.6956940443813076);
    // Tie-averaged gains at k = 2.
    close(rk::ndcg_score(&g, &[1.0, 0.0, 0.0, 0.0, 1.0], 5, Some(2), None, false).unwrap(), 0.9298593499260985);
}

// ==================== cluster ====================

const CA: [i64; 8] = [0, 0, 0, 1, 1, 1, 2, 2];
const CB: [i64; 8] = [0, 0, 1, 1, 2, 2, 2, 2];

#[test]
fn supervised_cluster_scores_match_sklearn() {
    close(clu::adjusted_rand_score(&CA, &CB).unwrap(), 0.18181818181818182);
    close(clu::adjusted_mutual_info_score(&CA, &CB, clu::AverageMethod::Arithmetic).unwrap(), 0.2745416497368333);
    close(clu::normalized_mutual_info_score(&CA, &CB, clu::AverageMethod::Geometric).unwrap(), 0.530131974074931);
    let (h, c, v) = clu::homogeneity_completeness_v_measure(&CA, &CB, 1.0).unwrap();
    close(h, 0.5196243460723442);
    close(c, 0.5408520829727553);
    close(v, 0.5300257549140327);
    close(clu::fowlkes_mallows_score(&CA, &CB).unwrap(), 0.4008918628686365);
    close(clu::rand_score(&CA, &CB).unwrap(), 0.6785714285714286);
    assert_eq!(clu::pair_confusion_matrix(&CA, &CB).unwrap(), [32, 10, 8, 6]);
}

#[test]
fn unsupervised_cluster_scores_match_sklearn() {
    let x = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 5.0, 5.0, 5.0, 6.0, 6.0, 5.0, 9.0, 0.0, 9.0, 1.0];
    let s = clu::silhouette_samples(&x, 2, &CA, &Distance::Euclidean).unwrap();
    close_all(
        &s,
        &[
            0.8677922957740025,
            0.824932619353653,
            0.824932619353653,
            0.8341622249021181,
            0.8226697997687924,
            0.7771005184205394,
            0.8457201114541079,
            0.8241498368653161,
        ],
    );
    close(clu::calinski_harabasz_score(&x, &CA).unwrap(), 110.69078947368418);
    close(clu::davies_bouldin_score(&x, &CA).unwrap(), 0.18847891292763608);
    assert_eq!(clu::calinski_harabasz_score(&x, &[0; 8]).unwrap_err(), MetricError::TooFewPoints);
}

#[test]
fn hungarian_finds_the_optimum_on_a_rectangular_matrix() {
    // 2 x 3: best is (0 -> 2, 1 -> 0) with cost 1 + 2 = 3.
    let cost = [4.0, 5.0, 1.0, 2.0, 6.0, 3.0];
    assert_eq!(clu::linear_sum_assignment(&cost, 2, 3).unwrap(), vec![(0, 2), (1, 0)]);
    assert_eq!(clu::linear_sum_assignment(&[4.0, 2.0, 5.0, 6.0, 1.0, 3.0], 3, 2).unwrap(), vec![(0, 1), (2, 0)]);
}

// ==================== pairwise ====================

const A: [f64; 6] = [0.0, 1.0, 2.0, 1.0, 1.0, 0.0];
const B: [f64; 9] = [1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 1.0];

#[test]
fn real_valued_distances_match_sklearn() {
    let d = |m: Distance| pw::pairwise_distances(&A, Some(&B), 3, &m).unwrap();
    close_all(&d(Distance::Euclidean), &[3f64.sqrt(), 5f64.sqrt(), 2f64.sqrt(), 2f64.sqrt(), 6f64.sqrt(), 3f64.sqrt()]);
    close_all(
        &d(Distance::Cosine),
        &[0.36754446796632423, 0.22540333075851648, 0.10557280900008414, 0.5000000000000001, 0.18350341907227397, 1.0],
    );
    close_all(&d(Distance::BrayCurtis), &[0.6, 1.0 / 3.0, 0.5, 0.5, 0.5, 1.0]);
    close_all(&d(Distance::Canberra), &[7.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0, 2.0, 5.0 / 3.0, 3.0]);
    // A constant row has no correlation: NaN, as scipy.
    close_all(&d(Distance::Correlation), &[1.0, f64::NAN, 0.13397459621556151, 1.5, f64::NAN, 2.0]);
    close_all(&d(Distance::Chebyshev), &[1.0, 2.0, 1.0, 1.0, 2.0, 1.0]);
    close_all(
        &d(Distance::Minkowski { p: 3.0, w: None }),
        &[1.4422495703074083, 2.080083823051904, 1.2599210498948732, 1.2599210498948732, 2.154434690031884, 1.4422495703074083],
    );
}

#[test]
fn boolean_distances_match_sklearn() {
    let d = |m: Distance| pw::pairwise_distances(&A, Some(&B), 3, &m).unwrap();
    close_all(&d(Distance::Dice), &[0.5, 0.2, 1.0 / 3.0, 0.5, 0.2, 1.0]);
    close_all(&d(Distance::Jaccard), &[2.0 / 3.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0 / 3.0, 1.0]);
    close_all(&d(Distance::RogersTanimoto), &[0.8, 0.5, 0.5, 0.8, 0.5, 1.0]);
    close_all(&d(Distance::RussellRao), &[2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0]);
    close_all(&d(Distance::SokalSneath), &[0.8, 0.5, 2.0 / 3.0, 0.8, 0.5, 1.0]);
    close_all(&d(Distance::Yule), &[2.0, 0.0, 0.0, 2.0, 0.0, 2.0]);
}

#[test]
fn kernels_match_sklearn() {
    close_all(
        &pw::pairwise_kernels(&A, Some(&B), 3, Kernel::Rbf { gamma: 0.5 }).unwrap(),
        &[0.22313016014842982, 0.0820849986238988, 0.36787944117144233, 0.36787944117144233, 0.049787068367863944, 0.22313016014842982],
    );
    close_all(
        &pw::pairwise_kernels(&A, Some(&B), 3, Kernel::AdditiveChi2).unwrap(),
        &[-7.0 / 3.0, -7.0 / 3.0, -4.0 / 3.0, -2.0, -8.0 / 3.0, -3.0],
    );
    assert_eq!(
        pw::pairwise_kernels(&[-1.0, 0.0], None, 2, Kernel::AdditiveChi2).unwrap_err(),
        MetricError::DomainError
    );
}

#[test]
fn self_distance_diagonal_is_exactly_zero_except_all_missing_nan_euclidean() {
    let x = [1.0, 2.0, f64::NAN, f64::NAN, 3.0, 1.0];
    let d = pw::pairwise_distances(&x, None, 2, &Distance::NanEuclidean { missing: f64::NAN }).unwrap();
    assert_eq!(d[0], 0.0);
    assert!(d[4].is_nan());
    assert_eq!(d[8], 0.0);
    let (idx, val) = pw::argmin_min(&A, &B, 3, &Distance::Euclidean).unwrap();
    assert_eq!(idx, vec![2, 0]);
    close_all(&val, &[2f64.sqrt(), 2f64.sqrt()]);
}
