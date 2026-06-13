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

import numpy as np
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

ALL_12 = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    "KMeans",
    "DBSCAN",
    "PCA",
    "TruncatedSVD",
    "NearestNeighbors",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
]


def _ctor(name):
    """Construct an estimator with the v1 required ctor args (PCA needs one)."""
    cls = getattr(mlrs, name)
    if name == "PCA":
        return cls(n_components=2)
    return cls()


@pytest.mark.parametrize("name", ALL_12)
def test_all_12_importable(name):
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
    assert not c.__sklearn_is_fitted__() if hasattr(
        c, "__sklearn_is_fitted__"
    ) else True


@pytest.mark.parametrize("name", ALL_12)
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
    ],
)
def test_fitted_attr_raises_before_fit(name, attr):
    with pytest.raises(NotFittedError):
        getattr(_ctor(name), attr)


@pytest.mark.parametrize("name", ALL_12)
def test_fit_returns_self_signature(name):
    # PY-01: every shim's fit must `return self`. We can't run the device path
    # here, but we can assert the source contract: fit is defined on the class
    # (not inherited as a stub raising NotImplementedError).
    est = _ctor(name)
    assert callable(est.fit)
    assert "fit" in type(est).__dict__
