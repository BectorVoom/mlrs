"""Family-module shim structure tests (Task 2 — D-01 / PY-01 / PY-02).

These cover the *pure-Python* contract of the 12 estimator shims WITHOUT the
compiled ``_mlrs`` extension: every class subclasses ``MlrsBase`` + the right
sklearn family mixin, ``__init__`` stores its sklearn-named ctor args verbatim
(so ``get_params`` / ``set_params`` / ``clone`` round-trip), fitted-attr access
before ``fit`` raises ``NotFittedError``, and the predict-less estimators
(DBSCAN, NearestNeighbors) expose no ``predict``. The actual fit/predict
delegation to ``_mlrs`` is exercised by the live-extension oracle gate.

Req: PY-01 (fit returns self / NotFitted), PY-02 (sklearn names + get/set_params).
"""

import pytest
from sklearn.base import (
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.exceptions import NotFittedError

import mlrs
from mlrs.base import MlrsBase


def _exported_shim_names():
    """Every exported ``mlrs`` symbol that is an ``MlrsBase`` estimator shim.

    Derived from ``mlrs.__all__`` (excludes the ``backend_supports_f64`` flag and
    the ``johnson_lindenstrauss_min_dim`` helper) so the matrix grows
    automatically as new shim classes are registered — no hard-coded list.
    """
    names = []
    for name in mlrs.__all__:
        obj = getattr(mlrs, name)
        if isinstance(obj, type) and issubclass(obj, MlrsBase):
            names.append(name)
    return names


# The full estimator-shim matrix (32 shims as of Plan 16-11: 17 pre-existing +
# 15 new), derived from the exported set so it cannot drift.
ALL_SHIMS = _exported_shim_names()


def _ctor(name):
    """Construct an estimator with the v1 required ctor args.

    PCA and IncrementalPCA require an explicit ``n_components`` (no zero-arg
    default); every other shim is zero-arg constructible (Pitfall 6).
    """
    cls = getattr(mlrs, name)
    if name in ("PCA", "IncrementalPCA"):
        return cls(n_components=2)
    return cls()


@pytest.mark.parametrize("name", ALL_SHIMS)
def test_all_shims_importable(name):
    assert hasattr(mlrs, name)
    _ctor(name)  # constructs pure-Python (no _mlrs needed)


def test_family_mixins_composed():
    assert isinstance(mlrs.LinearRegression(), RegressorMixin)
    assert isinstance(mlrs.Ridge(), RegressorMixin)
    assert isinstance(mlrs.Lasso(), RegressorMixin)
    assert isinstance(mlrs.ElasticNet(), RegressorMixin)
    assert isinstance(mlrs.KNeighborsRegressor(), RegressorMixin)
    assert isinstance(mlrs.LogisticRegression(), ClassifierMixin)
    assert isinstance(mlrs.KNeighborsClassifier(), ClassifierMixin)
    assert isinstance(mlrs.KMeans(), ClusterMixin)
    assert isinstance(mlrs.DBSCAN(), ClusterMixin)
    assert isinstance(mlrs.PCA(n_components=2), TransformerMixin)
    assert isinstance(mlrs.TruncatedSVD(), TransformerMixin)
    # NearestNeighbors has no scoring/transformer mixin.
    nn = mlrs.NearestNeighbors()
    assert not isinstance(nn, (RegressorMixin, ClassifierMixin, ClusterMixin))
    assert not isinstance(nn, TransformerMixin)
    # --- Plan 16-11 new shims: family-mixin composition. ------------------ #
    assert isinstance(mlrs.LinearSVR(), RegressorMixin)
    assert isinstance(mlrs.MBSGDRegressor(), RegressorMixin)
    assert isinstance(mlrs.KernelRidge(), RegressorMixin)
    assert isinstance(mlrs.LinearSVC(), ClassifierMixin)
    assert isinstance(mlrs.MBSGDClassifier(), ClassifierMixin)
    for nb in (
        mlrs.GaussianNB(),
        mlrs.MultinomialNB(),
        mlrs.BernoulliNB(),
        mlrs.ComplementNB(),
        mlrs.CategoricalNB(),
    ):
        assert isinstance(nb, ClassifierMixin)
    assert isinstance(mlrs.SpectralClustering(), ClusterMixin)
    assert isinstance(mlrs.HDBSCAN(), ClusterMixin)
    assert isinstance(mlrs.SpectralEmbedding(), TransformerMixin)
    assert isinstance(mlrs.UMAP(), TransformerMixin)
    # KernelDensity has no scoring/transformer/cluster mixin (fit + score_samples).
    kd = mlrs.KernelDensity()
    assert not isinstance(
        kd, (RegressorMixin, ClassifierMixin, ClusterMixin, TransformerMixin)
    )


def test_new_shim_family_surfaces():
    """Family-specific method surface for the Plan 16-11 shims.

    Transformers expose ``transform`` (UMAP) or ``fit_transform`` (SpectralEmbedding /
    UMAP); cluster shims are labels-only (no standalone ``predict``); the
    classifiers expose ``predict``; KernelDensity exposes ``score_samples``.
    """
    # UMAP: out-of-sample transform + fit_transform.
    u = mlrs.UMAP()
    assert hasattr(u, "transform")
    assert hasattr(u, "fit_transform")
    # SpectralEmbedding: fit_transform only (no out-of-sample transform).
    se = mlrs.SpectralEmbedding()
    assert hasattr(se, "fit_transform")
    # Cluster shims: labels-only, no standalone predict.
    assert not hasattr(mlrs.SpectralClustering(), "predict")
    assert not hasattr(mlrs.HDBSCAN(), "predict")
    # Classifiers expose predict; regressors expose predict.
    assert hasattr(mlrs.LinearSVC(), "predict")
    assert hasattr(mlrs.MBSGDClassifier(), "predict")
    assert hasattr(mlrs.LinearSVR(), "predict")
    assert hasattr(mlrs.KernelRidge(), "predict")
    assert hasattr(mlrs.GaussianNB(), "predict")
    # KernelDensity: score_samples, no predict.
    kd = mlrs.KernelDensity()
    assert hasattr(kd, "score_samples")
    assert not hasattr(kd, "predict")


def test_logreg_exposes_capital_C_not_c():
    m = mlrs.LogisticRegression(C=2.0)
    assert m.C == 2.0
    assert "C" in m.get_params()
    assert "c" not in m.get_params()


def test_kmeans_exposes_random_state():
    assert "random_state" in mlrs.KMeans().get_params()


def test_get_set_params_roundtrip():
    m = mlrs.Ridge(alpha=2.0)
    assert m.get_params()["alpha"] == 2.0
    m.set_params(alpha=3.0)
    assert m.get_params()["alpha"] == 3.0


def test_clone_preserves_unfitted_params():
    c = clone(mlrs.KMeans(n_clusters=5))
    assert c.n_clusters == 5
    assert (
        not c.__sklearn_is_fitted__()
        if hasattr(c, "__sklearn_is_fitted__")
        else True
    )


@pytest.mark.parametrize("name", ALL_SHIMS)
def test_output_type_param_present(name):
    assert "output_type" in _ctor(name).get_params()


def test_dbscan_has_no_predict():
    assert not hasattr(mlrs.DBSCAN(), "predict")


def test_nearest_neighbors_has_no_predict_but_has_kneighbors():
    nn = mlrs.NearestNeighbors()
    assert not hasattr(nn, "predict")
    assert hasattr(nn, "kneighbors")


@pytest.mark.parametrize(
    "name,attr",
    [
        ("LinearRegression", "coef_"),
        ("Ridge", "coef_"),
        ("Lasso", "coef_"),
        ("ElasticNet", "coef_"),
        ("LogisticRegression", "coef_"),
        ("KMeans", "cluster_centers_"),
        ("KMeans", "labels_"),
        ("DBSCAN", "labels_"),
        ("PCA", "components_"),
        ("TruncatedSVD", "components_"),
        # --- Plan 16-11 new shims. ----------------------------------------- #
        ("LinearSVC", "coef_"),
        ("LinearSVR", "coef_"),
        ("MBSGDClassifier", "coef_"),
        ("MBSGDRegressor", "coef_"),
        ("KernelRidge", "dual_coef_"),
        ("SpectralClustering", "labels_"),
        ("SpectralEmbedding", "embedding_"),
        ("HDBSCAN", "labels_"),
        ("HDBSCAN", "probabilities_"),
        ("UMAP", "embedding_"),
        # --- TASK-16 (PY-ENS-05, RF): always-present fitted attribute. ----- #
        ("RandomForestClassifier", "feature_importances_"),
        ("RandomForestRegressor", "feature_importances_"),
    ],
)
def test_fitted_attr_raises_before_fit(name, attr):
    with pytest.raises(NotFittedError):
        getattr(_ctor(name), attr)


# --- TASK-16 (PY-ENS-05, RF-OOB-02): `oob_score_` conditional attribute. --- #
#
# `oob_score_` is the FIRST mlrs fitted attribute whose presence depends on a
# constructor arg (RF-OOB-02's own unresolved-question resolution, PLAN.md
# "Q-conditional-attribute test machinery"): `test_fitted_attr_raises_before_fit`
# only proves "raises NotFittedError before fit" for an ALWAYS-eventually-present
# attribute — it has no support for the THIRD state this attribute has
# ("AttributeError even AFTER fit, when a constructor flag is False"). This gets
# its own dedicated test rather than an entry in that generic parametrize list.
def test_random_forest_oob_score_conditional_attribute():
    """`oob_score_` (RF-OOB-02): NotFittedError before fit; AttributeError
    after fit when `oob_score=False` (sklearn parity — NOT a silent `None`)."""
    # (a) oob_score=True, before fit -> standard NotFittedError.
    with pytest.raises(NotFittedError):
        mlrs.RandomForestClassifier(oob_score=True).oob_score_

    # (b) oob_score=False (the default), AFTER fit -> AttributeError. This half
    # needs a real `.fit()` through the compiled `_mlrs` extension, so it is
    # skipped (not failed) on a not-yet-built tree, keeping this file's
    # pure-Python collection contract intact for part (a) above.
    pytest.importorskip("mlrs._mlrs")
    X = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [0.5, 2.0], [1.5, 1.5], [3.0, 3.0]]
    y = [0.0, 1.0, 2.0, 0.5, 1.5, 3.0]
    est = mlrs.RandomForestRegressor(
        n_estimators=2, max_depth=2, oob_score=False
    ).fit(X, y)
    with pytest.raises(AttributeError):
        est.oob_score_


@pytest.mark.parametrize("name", ALL_SHIMS)
def test_fit_returns_self_signature(name):
    # PY-01: every shim's fit must `return self`. We can't run the device path
    # here, but we can assert the source contract: fit is defined on the class
    # (not inherited as a stub raising NotImplementedError).
    est = _ctor(name)
    assert callable(est.fit)
    assert "fit" in type(est).__dict__
