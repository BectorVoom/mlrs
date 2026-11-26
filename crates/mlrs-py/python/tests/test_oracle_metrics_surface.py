"""Live sklearn oracle for the full ``mlrs.metrics`` surface beyond the
original eleven metrics (classification, regression, ranking, clustering,
pairwise, scorers).

Every case calls the sklearn function and the mlrs function with the same
arguments and compares at the project gate (abs/rel 1e-5), over random data at
both input dtypes, with and without ``sample_weight``, across the
string-valued parameters. Warnings sklearn emits are compared by category.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("mlrs._mlrs")
sk = pytest.importorskip("sklearn.metrics")
skc = pytest.importorskip("sklearn.metrics.cluster")
skp = pytest.importorskip("sklearn.metrics.pairwise")

import mlrs.metrics as mm  # noqa: E402
import mlrs.metrics.cluster as mmc  # noqa: E402
import mlrs.metrics.pairwise as mmp  # noqa: E402

TOL = dict(rtol=1e-5, atol=1e-5)
DTYPES = [np.float32, np.float64]


def close(a, b):
    if isinstance(b, tuple):
        assert isinstance(a, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y)
        return
    if b is None:
        assert a is None
        return
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in b:
            close(a[k], b[k])
        return
    if isinstance(b, str):
        assert a == b
        return
    np.testing.assert_allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True, **TOL)
    if np.ndim(b) == 0:
        assert np.ndim(a) == 0


def both(fn_sk, fn_mm, *args, **kwargs):
    """Call both with warnings captured; return the results and the sets of
    warning categories."""
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        want = fn_sk(*args, **kwargs)
    with warnings.catch_warnings(record=True) as wm:
        warnings.simplefilter("always")
        got = fn_mm(*args, **kwargs)
    return got, want, {w.category for w in wm}, {w.category for w in ws}


def check(fn_sk, fn_mm, *args, **kwargs):
    got, want, wcat_m, wcat_s = both(fn_sk, fn_mm, *args, **kwargs)
    close(got, want)
    from sklearn.exceptions import UndefinedMetricWarning

    assert (UndefinedMetricWarning in wcat_m) == (UndefinedMetricWarning in wcat_s)
    return got


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _cls(rng, n=200, k=4):
    yt = rng.integers(0, k, n)
    yp = np.where(rng.random(n) < 0.6, yt, rng.integers(0, k, n))
    sw = rng.random(n) * 2
    return yt, yp, sw


def _ml(rng, n=120, L=5):
    yt = (rng.random((n, L)) < 0.4).astype(int)
    yp = np.where(rng.random((n, L)) < 0.7, yt, 1 - yt)
    sw = rng.random(n) * 2
    return yt, yp, sw


# ==================== precision / recall / F-beta / jaccard ====================


@pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_prfs_multiclass(rng, average, weighted, beta):
    yt, yp, sw = _cls(rng)
    kw = dict(beta=beta, average=average, sample_weight=sw if weighted else None)
    check(sk.precision_recall_fscore_support, mm.precision_recall_fscore_support, yt, yp, **kw)
    check(sk.fbeta_score, mm.fbeta_score, yt, yp, **kw)


@pytest.mark.parametrize("average", [None, "micro", "macro", "weighted", "samples"])
@pytest.mark.parametrize("weighted", [False, True])
def test_prfs_multilabel(rng, average, weighted):
    yt, yp, sw = _ml(rng)
    kw = dict(average=average, sample_weight=sw if weighted else None)
    check(sk.precision_recall_fscore_support, mm.precision_recall_fscore_support, yt, yp, **kw)
    for f in ("precision_score", "recall_score", "f1_score", "jaccard_score"):
        check(getattr(sk, f), getattr(mm, f), yt, yp, **kw)
    check(sk.fbeta_score, mm.fbeta_score, yt, yp, beta=0.7, **kw)


def test_prfs_multilabel_labels_subset(rng):
    yt, yp, _ = _ml(rng)
    for average in (None, "macro", "micro"):
        check(sk.precision_recall_fscore_support, mm.precision_recall_fscore_support, yt, yp, labels=[3, 0], average=average)


@pytest.mark.parametrize("average", ["binary", None, "micro", "macro", "weighted"])
@pytest.mark.parametrize("zero_division", ["warn", 0, 1, np.nan])
def test_prf_zero_division(average, zero_division):
    # class 2 is never predicted, class 3 never true
    yt = np.array([0, 0, 1, 1, 2, 2])
    yp = np.array([0, 1, 1, 1, 3, 0])
    if average == "binary":
        yt, yp = yt % 2, np.array([0, 0, 0, 0, 0, 0])
    kw = dict(average=average, zero_division=zero_division)
    for f in ("precision_score", "recall_score", "f1_score"):
        check(getattr(sk, f), getattr(mm, f), yt, yp, **kw)
    if isinstance(zero_division, float) and np.isnan(zero_division):
        with pytest.raises(ValueError, match="zero_division"):
            mm.jaccard_score(yt, yp, **kw)
    else:
        check(sk.jaccard_score, mm.jaccard_score, yt, yp, **kw)
    check(sk.precision_recall_fscore_support, mm.precision_recall_fscore_support, yt, yp, **kw)


def test_prf_nan_zero_division_skips_undefined_classes():
    # The bug the shared core fixes: sklearn's _nanaverage drops NaN entries.
    yt, yp = [0, 0, 1, 1, 2], [0, 0, 1, 1, 1]
    for avg in ("macro", "weighted"):
        check(sk.precision_score, mm.precision_score, yt, yp, average=avg, zero_division=np.nan)


@pytest.mark.parametrize("beta", [0.0, np.inf])
def test_fbeta_limits(rng, beta):
    yt, yp, _ = _cls(rng)
    check(sk.fbeta_score, mm.fbeta_score, yt, yp, beta=beta, average="macro")


@pytest.mark.parametrize("average", ["binary", None, "micro", "macro", "weighted"])
def test_jaccard_multiclass(rng, average):
    yt, yp, sw = _cls(rng, k=2 if average == "binary" else 4)
    check(sk.jaccard_score, mm.jaccard_score, yt, yp, average=average)
    check(sk.jaccard_score, mm.jaccard_score, yt, yp, average=average, sample_weight=sw)


def test_binary_guards(rng):
    yt, yp, _ = _cls(rng)
    with pytest.raises(ValueError, match="Target is multiclass"):
        mm.fbeta_score(yt, yp, beta=1)
    with pytest.raises(ValueError, match="pos_label=5 is not a valid label"):
        mm.precision_recall_fscore_support(yt % 2, yp % 2, average="binary", pos_label=5)
    yt, yp, _ = _ml(rng)
    with pytest.raises(ValueError, match="multilabel-indicator but average='binary'"):
        mm.f1_score(yt, yp)


@pytest.mark.parametrize("samplewise", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_multilabel_confusion_matrix(rng, samplewise, weighted):
    yt, yp, sw = _ml(rng)
    kw = dict(samplewise=samplewise, sample_weight=sw if weighted else None)
    got = check(sk.multilabel_confusion_matrix, mm.multilabel_confusion_matrix, yt, yp, **kw)
    assert got.dtype == sk.multilabel_confusion_matrix(yt, yp, **kw).dtype
    if not samplewise:
        yt, yp, sw = _cls(rng)
        check(sk.multilabel_confusion_matrix, mm.multilabel_confusion_matrix, yt, yp, sample_weight=sw if weighted else None)
        check(sk.multilabel_confusion_matrix, mm.multilabel_confusion_matrix, yt, yp, labels=[3, 1, 7])


@pytest.mark.parametrize("output_dict", [False, True])
def test_classification_report(rng, output_dict):
    yt, yp, sw = _cls(rng)
    check(sk.classification_report, mm.classification_report, yt, yp, output_dict=output_dict)
    check(
        sk.classification_report,
        mm.classification_report,
        yt,
        yp,
        digits=4,
        target_names=["a", "b", "c", "d"],
        output_dict=output_dict,
    )
    if output_dict:
        # Weighted: the text form prints the float support total at full
        # repr precision, so only the dict is compared at the tolerance.
        check(sk.classification_report, mm.classification_report, yt, yp, sample_weight=sw, output_dict=True)
    yt, yp, _ = _ml(rng)
    check(sk.classification_report, mm.classification_report, yt, yp, output_dict=output_dict)


# ==================== label-based scores ====================


@pytest.mark.parametrize("weighted", [False, True])
def test_label_scores(rng, weighted):
    yt, yp, sw = _cls(rng)
    w = sw if weighted else None
    check(sk.balanced_accuracy_score, mm.balanced_accuracy_score, yt, yp, sample_weight=w)
    check(sk.balanced_accuracy_score, mm.balanced_accuracy_score, yt, yp, sample_weight=w, adjusted=True)
    check(sk.matthews_corrcoef, mm.matthews_corrcoef, yt, yp, sample_weight=w)
    check(sk.matthews_corrcoef, mm.matthews_corrcoef, yt % 2, yp % 2, sample_weight=w)
    for weights in (None, "linear", "quadratic"):
        check(sk.cohen_kappa_score, mm.cohen_kappa_score, yt, yp, weights=weights, sample_weight=w)
    check(sk.cohen_kappa_score, mm.cohen_kappa_score, yt, yp, labels=[0, 2, 3])
    check(sk.hamming_loss, mm.hamming_loss, yt, yp, sample_weight=w)
    for normalize in (True, False):
        check(sk.zero_one_loss, mm.zero_one_loss, yt, yp, normalize=normalize, sample_weight=w)
    yt, yp, sw = _ml(rng)
    w = sw if weighted else None
    check(sk.hamming_loss, mm.hamming_loss, yt, yp, sample_weight=w)
    for normalize in (True, False):
        check(sk.zero_one_loss, mm.zero_one_loss, yt, yp, normalize=normalize, sample_weight=w)
        check(sk.accuracy_score, mm.accuracy_score, yt, yp, normalize=normalize, sample_weight=w)


def test_label_scores_degenerate():
    check(sk.balanced_accuracy_score, mm.balanced_accuracy_score, [0, 0, 1], [0, 2, 1])
    check(sk.matthews_corrcoef, mm.matthews_corrcoef, [1, 1, 1], [1, 1, 1])
    check(sk.cohen_kappa_score, mm.cohen_kappa_score, [0, 0, 0], [0, 0, 0])
    check(sk.cohen_kappa_score, mm.cohen_kappa_score, [0, 0, 1], [1, 1, 1], labels=[0], replace_undefined_by=-1.0)


@pytest.mark.parametrize("weighted", [False, True])
def test_hinge_loss(rng, weighted):
    n = 150
    sw = rng.random(n) if weighted else None
    yb = rng.choice([-1, 1], n)
    check(sk.hinge_loss, mm.hinge_loss, yb, rng.normal(size=n), sample_weight=sw)
    check(sk.hinge_loss, mm.hinge_loss, yb * 2 + 5, rng.normal(size=n), sample_weight=sw)
    yt = rng.integers(0, 4, n)
    dec = rng.normal(size=(n, 4))
    check(sk.hinge_loss, mm.hinge_loss, yt, dec, sample_weight=sw)
    check(sk.hinge_loss, mm.hinge_loss, yt[yt < 3], dec[yt < 3], labels=[0, 1, 2, 3])


def _proba(rng, n, k):
    p = rng.random((n, k)) + 0.05
    return p / p.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("weighted", [False, True])
def test_probabilistic_scores(rng, dtype, weighted):
    n = 160
    sw = rng.random(n) if weighted else None
    yb = rng.integers(0, 2, n)
    pb = rng.random(n).astype(dtype)
    check(sk.brier_score_loss, mm.brier_score_loss, yb, pb, sample_weight=sw)
    check(sk.brier_score_loss, mm.brier_score_loss, yb + 3, pb, sample_weight=sw)
    check(sk.brier_score_loss, mm.brier_score_loss, yb, pb, pos_label=0, scale_by_half=False)
    check(sk.d2_brier_score, mm.d2_brier_score, yb, pb, sample_weight=sw)
    yt = rng.integers(0, 3, n)
    P = _proba(rng, n, 3).astype(dtype)
    check(sk.brier_score_loss, mm.brier_score_loss, yt, P, sample_weight=sw)
    check(sk.brier_score_loss, mm.brier_score_loss, yt, P, scale_by_half=True)
    check(sk.d2_brier_score, mm.d2_brier_score, yt, P, sample_weight=sw)
    check(sk.d2_log_loss_score, mm.d2_log_loss_score, yt, P, sample_weight=sw)
    check(sk.d2_log_loss_score, mm.d2_log_loss_score, yb, pb, sample_weight=sw)
    check(sk.d2_log_loss_score, mm.d2_log_loss_score, yt[yt < 2], P[yt < 2], labels=[0, 1, 2])


def test_class_likelihood_ratios(rng):
    yt, yp, sw = _cls(rng, k=2)
    check(sk.class_likelihood_ratios, mm.class_likelihood_ratios, yt, yp)
    check(sk.class_likelihood_ratios, mm.class_likelihood_ratios, yt, yp, sample_weight=sw)
    check(sk.class_likelihood_ratios, mm.class_likelihood_ratios, yt, yp, labels=[1, 0])
    for rub in (np.nan, 1.0, {"LR+": 2.0, "LR-": 0.5}):
        check(sk.class_likelihood_ratios, mm.class_likelihood_ratios, [0, 0, 1, 1], [0, 0, 1, 1], replace_undefined_by=rub)
        check(sk.class_likelihood_ratios, mm.class_likelihood_ratios, [0, 0, 1, 1], [0, 0, 0, 0], replace_undefined_by=rub)
        check(sk.class_likelihood_ratios, mm.class_likelihood_ratios, [0, 0, 0, 0], [0, 1, 0, 1], replace_undefined_by=rub)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("normalize", [True, False])
def test_top_k_accuracy(rng, k, normalize):
    n = 150
    sw = rng.random(n)
    yt = rng.integers(0, 4, n)
    S = rng.random((n, 4))
    S[:10, 1] = S[:10, 2]  # ties
    check(sk.top_k_accuracy_score, mm.top_k_accuracy_score, yt, S, k=k, normalize=normalize)
    check(sk.top_k_accuracy_score, mm.top_k_accuracy_score, yt, S, k=k, normalize=normalize, sample_weight=sw)
    S5 = np.column_stack([S, rng.random(n)])
    check(sk.top_k_accuracy_score, mm.top_k_accuracy_score, yt, S5, k=k, labels=[0, 1, 2, 3, 4])
    yb = rng.integers(0, 2, n)
    check(sk.top_k_accuracy_score, mm.top_k_accuracy_score, yb, rng.random(n), k=k, normalize=normalize)
    check(sk.top_k_accuracy_score, mm.top_k_accuracy_score, yb, rng.normal(size=n), k=k)


# ==================== regression ====================


def _reg(rng, n, k, dtype, positive=False):
    yt = rng.normal(size=(n, k)) if k > 1 else rng.normal(size=n)
    yp = yt + rng.normal(scale=0.5, size=yt.shape)
    if positive:
        yt, yp = np.abs(yt) + 0.1, np.abs(yp) + 0.1
    return yt.astype(dtype), yp.astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_regression_multioutput(rng, dtype, k, weighted):
    n = 120
    sw = rng.random(n) if weighted else None
    yt, yp = _reg(rng, n, k, dtype)
    mos = ["raw_values", "uniform_average"] + ([[0.2, 0.3, 0.5]] if k == 3 else [])
    for mo in mos:
        for f in (
            "mean_absolute_percentage_error",
            "root_mean_squared_error",
            "median_absolute_error",
            "d2_absolute_error_score",
        ):
            check(getattr(sk, f), getattr(mm, f), yt, yp, sample_weight=sw, multioutput=mo)
        for alpha in (0.1, 0.5, 0.9):
            check(sk.mean_pinball_loss, mm.mean_pinball_loss, yt, yp, sample_weight=sw, alpha=alpha, multioutput=mo)
            check(sk.d2_pinball_score, mm.d2_pinball_score, yt, yp, sample_weight=sw, alpha=alpha, multioutput=mo)
        for ff in (True, False):
            check(sk.explained_variance_score, mm.explained_variance_score, yt, yp, sample_weight=sw, multioutput=mo, force_finite=ff)
        pt, pp = _reg(rng, n, k, dtype, positive=True)
        check(sk.mean_squared_log_error, mm.mean_squared_log_error, pt, pp, sample_weight=sw, multioutput=mo)
        check(sk.root_mean_squared_log_error, mm.root_mean_squared_log_error, pt, pp, sample_weight=sw, multioutput=mo)
    check(sk.explained_variance_score, mm.explained_variance_score, yt, yp, sample_weight=sw, multioutput="variance_weighted")


def test_weighted_median_ties(rng):
    # Integer weights put the cdf exactly on 50% — the averaged branch.
    yt = np.arange(10.0)
    yp = np.zeros(10)
    for sw in (np.ones(10), np.r_[np.ones(5), np.zeros(5)], rng.integers(0, 3, 10).astype(float) + (np.arange(10) == 0)):
        check(sk.median_absolute_error, mm.median_absolute_error, yt, yp, sample_weight=sw)
        check(sk.d2_absolute_error_score, mm.d2_absolute_error_score, yt, yp + 1, sample_weight=sw)


def test_regression_constant_target():
    for ff in (True, False):
        check(sk.explained_variance_score, mm.explained_variance_score, [1.0, 1.0, 1.0], [1.0, 2.0, 1.0], force_finite=ff)
        check(sk.explained_variance_score, mm.explained_variance_score, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], force_finite=ff)
    check(sk.d2_pinball_score, mm.d2_pinball_score, [1.0], [1.0])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("power", [-1.0, 0.0, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_tweedie(rng, dtype, power, weighted):
    n = 100
    sw = rng.random(n) if weighted else None
    yt, yp = _reg(rng, n, 1, dtype, positive=True)
    if power in (0.0, 1.0, 1.5):
        yt[:5] = 0  # zero targets are in-domain here
    check(sk.mean_tweedie_deviance, mm.mean_tweedie_deviance, yt, yp, sample_weight=sw, power=power)
    check(sk.d2_tweedie_score, mm.d2_tweedie_score, yt, yp, sample_weight=sw, power=power)
    if power == 1.0:
        check(sk.mean_poisson_deviance, mm.mean_poisson_deviance, yt, yp, sample_weight=sw)
    if power == 2.0:
        check(sk.mean_gamma_deviance, mm.mean_gamma_deviance, yt, yp, sample_weight=sw)


def test_regression_errors(rng):
    yt, yp = _reg(rng, 20, 1, np.float64)
    check(sk.max_error, mm.max_error, yt, yp)
    with pytest.raises(ValueError, match="less than or equal to -1"):
        mm.mean_squared_log_error([-2.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="strictly positive y_pred"):
        mm.mean_tweedie_deviance([1.0, 2.0], [0.0, 1.0], power=1)
    with pytest.raises(ValueError, match="Multioutput not supported"):
        mm.max_error(np.ones((3, 2)), np.ones((3, 2)))


# ==================== ranking ====================


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("weighted", [False, True, "zeros"])
@pytest.mark.parametrize("drop", [False, True])
def test_curves(rng, dtype, weighted, drop):
    n = 300
    yt = rng.integers(0, 2, n)
    ys = np.round(rng.random(n), 2).astype(dtype)  # ties
    sw = None
    if weighted:
        sw = rng.random(n)
        if weighted == "zeros":
            sw[::7] = 0
    check(sk.roc_curve, mm.roc_curve, yt, ys, sample_weight=sw, drop_intermediate=drop)
    check(sk.det_curve, mm.det_curve, yt, ys, sample_weight=sw, drop_intermediate=drop)
    check(sk.confusion_matrix_at_thresholds, mm.confusion_matrix_at_thresholds, yt, ys, sample_weight=sw)
    check(sk.precision_recall_curve, mm.precision_recall_curve, yt, ys, sample_weight=sw, drop_intermediate=drop)
    check(sk.roc_curve, mm.roc_curve, yt * 3 + 2, ys, pos_label=2, drop_intermediate=drop)
    check(sk.average_precision_score, mm.average_precision_score, yt, ys, sample_weight=sw)


def test_roc_curve_single_class_warns():
    check(sk.roc_curve, mm.roc_curve, [1, 1, 1], [0.2, 0.5, 0.9])
    check(sk.roc_curve, mm.roc_curve, [0, 0, 0], [0.2, 0.5, 0.9])


def test_auc(rng):
    x = np.sort(rng.random(50))
    y = rng.random(50)
    check(sk.auc, mm.auc, x, y)
    check(sk.auc, mm.auc, x[::-1], y)
    with pytest.raises(ValueError, match="neither increasing nor decreasing"):
        mm.auc([0, 2, 1], [1, 1, 1])


@pytest.mark.parametrize("average", [None, "micro", "macro", "weighted", "samples"])
@pytest.mark.parametrize("weighted", [False, True])
def test_average_precision_multilabel(rng, average, weighted):
    yt, _, sw = _ml(rng)
    S = rng.random(yt.shape)
    check(sk.average_precision_score, mm.average_precision_score, yt, S, average=average, sample_weight=sw if weighted else None)
    if average != "samples":
        ymc = rng.integers(0, 3, 100)
        P = _proba(rng, 100, 3)
        check(sk.average_precision_score, mm.average_precision_score, ymc, P, average=average)


@pytest.mark.parametrize("weighted", [False, True])
def test_label_ranking(rng, weighted):
    yt, _, sw = _ml(rng)
    yt[0] = 0
    yt[1] = 1
    S = np.round(rng.random(yt.shape), 1)  # ties
    w = sw if weighted else None
    for f in ("coverage_error", "label_ranking_average_precision_score", "label_ranking_loss"):
        check(getattr(sk, f), getattr(mm, f), yt, S, sample_weight=w)


@pytest.mark.parametrize("k", [None, 1, 3])
@pytest.mark.parametrize("ignore_ties", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_dcg_ndcg(rng, k, ignore_ties, weighted):
    n, L = 60, 6
    yt = rng.integers(0, 4, (n, L)).astype(float)
    S = rng.random((n, L)) if ignore_ties else np.round(rng.random((n, L)), 1)
    w = rng.random(n) if weighted else None
    check(sk.dcg_score, mm.dcg_score, yt, S, k=k, sample_weight=w, ignore_ties=ignore_ties)
    check(sk.dcg_score, mm.dcg_score, yt, S, k=k, log_base=10, ignore_ties=ignore_ties)
    check(sk.ndcg_score, mm.ndcg_score, yt, S, k=k, sample_weight=w, ignore_ties=ignore_ties)


def test_metric_at_thresholds(rng):
    yt = rng.integers(0, 2, 60)
    ys = np.round(rng.random(60), 1)
    sw = rng.random(60)
    sw[::5] = 0
    check(sk.metric_at_thresholds, mm.metric_at_thresholds, yt, ys, sk.accuracy_score)
    check(sk.metric_at_thresholds, mm.metric_at_thresholds, yt, ys, sk.f1_score, sample_weight=sw, metric_params={"zero_division": 0})


# ==================== clustering ====================


def _clusters(rng, n=300, k1=5, k2=7):
    a = rng.integers(0, k1, n)
    b = np.where(rng.random(n) < 0.5, a, rng.integers(0, k2, n)) * 3 - 4
    return a, b


@pytest.mark.parametrize("method", ["min", "geometric", "arithmetic", "max"])
def test_cluster_supervised(rng, method):
    a, b = _clusters(rng)
    check(skc.adjusted_mutual_info_score, mmc.adjusted_mutual_info_score, a, b, average_method=method)
    check(skc.normalized_mutual_info_score, mmc.normalized_mutual_info_score, a, b, average_method=method)


def test_cluster_scores(rng):
    for a, b in (_clusters(rng), _clusters(rng, n=50, k1=2, k2=2), (np.zeros(10, int), np.arange(10))):
        for f in (
            "adjusted_rand_score",
            "rand_score",
            "mutual_info_score",
            "fowlkes_mallows_score",
            "homogeneity_score",
            "completeness_score",
            "v_measure_score",
            "homogeneity_completeness_v_measure",
            "adjusted_mutual_info_score",
            "normalized_mutual_info_score",
        ):
            check(getattr(skc, f), getattr(mmc, f), a, b)
        check(skc.v_measure_score, mmc.v_measure_score, a, b, beta=2.5)
        np.testing.assert_array_equal(mmc.pair_confusion_matrix(a, b), skc.pair_confusion_matrix(a, b))
        np.testing.assert_array_equal(mmc.contingency_matrix(a, b), skc.contingency_matrix(a, b))
        check(skc.entropy, mmc.entropy, a)
        C = skc.contingency_matrix(a, b)
        check(skc.mutual_info_score, mmc.mutual_info_score, None, None, contingency=C)
        from sklearn.metrics.cluster import expected_mutual_information as sk_emi

        close(mmc.expected_mutual_information(C, len(a)), sk_emi(C, len(a)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine", "sqeuclidean", "chebyshev"])
def test_silhouette(rng, dtype, metric):
    X = rng.normal(size=(120, 4)).astype(dtype)
    lab = rng.integers(0, 4, 120)
    lab[0] = 9  # a singleton cluster -> silhouette 0
    check(sk.silhouette_samples, mm.silhouette_samples, X, lab, metric=metric)
    check(sk.silhouette_score, mm.silhouette_score, X, lab, metric=metric)
    check(sk.silhouette_score, mm.silhouette_score, X, lab, metric=metric, sample_size=60, random_state=0)
    D = skp.pairwise_distances(X, metric=metric)
    check(sk.silhouette_samples, mm.silhouette_samples, D, lab, metric="precomputed")


@pytest.mark.parametrize("dtype", DTYPES)
def test_dispersion_scores(rng, dtype):
    X = rng.normal(size=(150, 5)).astype(dtype)
    lab = rng.integers(0, 4, 150) * 2 + 1
    check(sk.calinski_harabasz_score, mm.calinski_harabasz_score, X, lab)
    check(sk.davies_bouldin_score, mm.davies_bouldin_score, X, lab)


def test_consensus_score(rng):
    def bic(n, R=20, C=15):
        return (rng.random((n, R)) < 0.3, rng.random((n, C)) < 0.3)

    a, b = bic(4), bic(6)
    check(sk.consensus_score, mm.consensus_score, a, b)
    check(sk.consensus_score, mm.consensus_score, b, a)
    sim = lambda ar, ac, br, bc: float((ar == br).mean() * (ac == bc).mean())  # noqa: E731
    check(sk.consensus_score, mm.consensus_score, a, b, similarity=sim)


# ==================== pairwise ====================

SK_METRICS = [
    "euclidean",
    "l1",
    "cityblock",
    "cosine",
    "sqeuclidean",
    "chebyshev",
    "braycurtis",
    "canberra",
    "correlation",
    "hamming",
    "matching",
    "minkowski",
    "dice",
    "jaccard",
    "rogerstanimoto",
    "russellrao",
    "sokalsneath",
    "yule",
    "seuclidean",
    "mahalanobis",
]


@pytest.mark.parametrize("metric", SK_METRICS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_y", [False, True])
def test_pairwise_distances(rng, metric, dtype, with_y):
    boolean = metric in mmp.PAIRWISE_BOOLEAN_FUNCTIONS
    X = rng.random((30, 6))
    Y = rng.random((20, 6))
    if boolean:
        X, Y = X > 0.5, Y > 0.5
    else:
        X, Y = X.astype(dtype), Y.astype(dtype)
    kw = {}
    if metric == "minkowski":
        kw = dict(p=3)
    if with_y and metric == "seuclidean":
        kw = dict(V=np.var(np.vstack([X, Y]), axis=0, ddof=1))
    if with_y and metric == "mahalanobis":
        kw = dict(VI=np.linalg.inv(np.cov(np.vstack([X, Y]).T)))
    args = (X, Y) if with_y else (X,)
    got = check(skp.pairwise_distances, mmp.pairwise_distances, *args, metric=metric, **kw)
    assert got.dtype == skp.pairwise_distances(*args, metric=metric, **kw).dtype


def test_pairwise_named(rng):
    X = rng.random((25, 4))
    Y = rng.random((15, 4))
    for f in ("euclidean_distances", "manhattan_distances", "cosine_distances", "cosine_similarity"):
        check(getattr(skp, f), getattr(mmp, f), X)
        check(getattr(skp, f), getattr(mmp, f), X, Y)
    check(skp.euclidean_distances, mmp.euclidean_distances, X, Y, squared=True)
    check(skp.haversine_distances, mmp.haversine_distances, X[:, :2], Y[:, :2])
    Xn = X.copy()
    Xn[::3, 1] = np.nan
    Xn[4] = np.nan
    check(skp.nan_euclidean_distances, mmp.nan_euclidean_distances, Xn)
    check(skp.nan_euclidean_distances, mmp.nan_euclidean_distances, Xn, Y, squared=True)
    check(skp.pairwise_distances, mmp.pairwise_distances, X, Y, metric="minkowski", p=1.5, w=rng.random(4))
    check(skp.pairwise_distances, mmp.pairwise_distances, X, Y, metric=lambda a, b: np.abs(a - b).max())
    check(skp.pairwise_distances, mmp.pairwise_distances, X, metric=lambda a, b: np.abs(a - b).sum() + 1)
    for f in ("paired_euclidean_distances", "paired_manhattan_distances", "paired_cosine_distances"):
        check(getattr(skp, f), getattr(mmp, f), X, X[::-1])
    check(skp.paired_distances, mmp.paired_distances, X, X[::-1], metric="l1")


@pytest.mark.parametrize("dtype", DTYPES)
def test_pairwise_kernels(rng, dtype):
    X = rng.random((25, 4)).astype(dtype)
    Y = rng.random((15, 4)).astype(dtype)
    for f, kw in (
        ("linear_kernel", {}),
        ("polynomial_kernel", dict(degree=2, gamma=0.3, coef0=0.5)),
        ("polynomial_kernel", {}),
        ("rbf_kernel", {}),
        ("rbf_kernel", dict(gamma=2.0)),
        ("sigmoid_kernel", {}),
        ("laplacian_kernel", dict(gamma=0.7)),
        ("cosine_similarity", {}),
        ("chi2_kernel", dict(gamma=0.5)),
        ("additive_chi2_kernel", {}),
    ):
        got = check(getattr(skp, f), getattr(mmp, f), X, Y, **kw)
        assert got.dtype == getattr(skp, f)(X, Y, **kw).dtype
        check(getattr(skp, f), getattr(mmp, f), X, **kw)
    for metric in mmp.PAIRWISE_KERNEL_FUNCTIONS:
        check(skp.pairwise_kernels, mmp.pairwise_kernels, X, Y, metric=metric)
    check(skp.pairwise_kernels, mmp.pairwise_kernels, X, Y, metric="rbf", gamma=0.1, degree=9, filter_params=True)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine", "chebyshev"])
@pytest.mark.parametrize("axis", [0, 1])
def test_argmin_and_chunked(rng, metric, axis):
    X = rng.random((40, 3))
    Y = rng.random((30, 3))
    gi, gv = mmp.pairwise_distances_argmin_min(X, Y, axis=axis, metric=metric)
    wi, wv = skp.pairwise_distances_argmin_min(X, Y, axis=axis, metric=metric)
    np.testing.assert_array_equal(gi, wi)
    close(gv, wv)
    np.testing.assert_array_equal(mmp.pairwise_distances_argmin(X, Y, axis=axis, metric=metric), wi)
    got = np.vstack(list(mmp.pairwise_distances_chunked(X, metric=metric, working_memory=0.005)))
    close(got, np.vstack(list(skp.pairwise_distances_chunked(X, metric=metric, working_memory=0.005))))
    red = lambda D, start: D.min(axis=1)  # noqa: E731
    close(
        np.concatenate(list(mmp.pairwise_distances_chunked(X, Y, metric=metric, reduce_func=red, working_memory=0.001))),
        np.concatenate(list(skp.pairwise_distances_chunked(X, Y, metric=metric, reduce_func=red, working_memory=0.001))),
    )


def test_distance_metric(rng):
    X = rng.random((20, 3))
    Y = rng.random((10, 3))
    for name, kw in (("euclidean", {}), ("manhattan", {}), ("chebyshev", {}), ("minkowski", dict(p=3)), ("canberra", {})):
        dm, ds = mm.DistanceMetric.get_metric(name, **kw), sk.DistanceMetric.get_metric(name, **kw)
        close(dm.pairwise(X, Y), ds.pairwise(X, Y))
        close(dm.pairwise(X), ds.pairwise(X))
        d = ds.pairwise(X, Y)
        close(dm.dist_to_rdist(d), ds.dist_to_rdist(d))
        close(dm.rdist_to_dist(ds.dist_to_rdist(d)), d)


# ==================== scorers ====================


def test_scorer_registry_matches_sklearn():
    assert mm.get_scorer_names() == sk.get_scorer_names()


def test_scorers_score_like_sklearn(rng):
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression, LogisticRegression

    X = rng.normal(size=(120, 4))
    yb = (X[:, 0] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    y3 = np.digitize(X[:, 1], [-0.5, 0.5])
    yr = np.abs(X @ [1.0, 2.0, -1.0, 0.5]) + 0.5
    clf_b = LogisticRegression().fit(X, yb)
    clf_3 = LogisticRegression().fit(X, y3)
    reg = LinearRegression().fit(X, yr)
    km = KMeans(3, n_init=2, random_state=0).fit(X)
    for name in mm.get_scorer_names():
        sk_scorer = sk.get_scorer(name)
        fn = sk_scorer._score_func.__name__
        if fn in ("roc_auc_score",) and "ov" in name or fn in ("top_k_accuracy_score",) or name in ("neg_log_loss",):
            est, y = clf_3, y3
        elif any(t in name for t in ("_samples",)):
            continue  # multilabel-only scorers need a multilabel estimator
        elif sk_scorer._response_method == "predict" and fn in dir(skc):
            est, y = km, y3
        elif "deviance" in name or "log_error" in name or "percentage" in name or fn in (
            "r2_score",
            "explained_variance_score",
            "max_error",
            "mean_absolute_error",
            "mean_squared_error",
            "root_mean_squared_error",
            "median_absolute_error",
            "d2_absolute_error_score",
        ):
            est, y = reg, yr
        else:
            est, y = clf_b, yb
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = sk_scorer(est, X, y)
            got = mm.get_scorer(name)(est, X, y)
        close(got, want)


def test_check_scoring(rng):
    from sklearn.linear_model import LogisticRegression

    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    est = LogisticRegression().fit(X, y)
    close(mm.check_scoring(est, "f1")(est, X, y), sk.check_scoring(est, "f1")(est, X, y))
    multi = mm.check_scoring(est, ["accuracy", "roc_auc"])(est, X, y)
    close(multi, sk.check_scoring(est, ["accuracy", "roc_auc"])(est, X, y))
    scorer = mm.make_scorer(mm.fbeta_score, beta=2)
    close(scorer(est, X, y), sk.make_scorer(sk.fbeta_score, beta=2)(est, X, y))


def test_surface_is_complete():
    """Every public name of ``sklearn.metrics`` exists in ``mlrs.metrics``,
    except the plotting Display classes."""
    missing = [
        n for n in sk.__all__ if not n.endswith("Display") and not hasattr(mm, n)
    ]
    assert missing == []
    missing_pw = [n for n in dir(skp) if not n.startswith("_") and callable(getattr(skp, n))
                  and getattr(getattr(skp, n), "__module__", "") == "sklearn.metrics.pairwise"
                  and not hasattr(mmp, n)]
    assert missing_pw == []
    missing_cl = [n for n in skc.__all__ if not hasattr(mmc, n)]
    assert missing_cl == []
