"""Hyperparameter round-trip (PY-02: sklearn-named ctor params + get/set_params).

Wave-0 COLLECTING stub: ``importorskip("mlrs")`` so it collects green
pre-wrapper. Plan 04-06 wire real assertions that every estimator's ``__init__``
stores its sklearn-named args verbatim (so ``get_params`` returns them and
``set_params``/``clone`` round-trip) — e.g. LogisticRegression exposes ``C``
(not the Rust field ``c``), KMeans exposes ``random_state``. See RESEARCH 06
§Hyperparameter Mapping + the ``__init__`` purity rule.
"""

import pytest

# Req: PY-02 (sklearn-named constructor hyperparameters + get_params/set_params).
PARAM_EXPECTATIONS = [
    ("LinearRegression", {"fit_intercept": True}),
    ("Ridge", {"alpha": 1.0, "fit_intercept": True}),
    ("Lasso", {"alpha": 1.0, "max_iter": 1000, "tol": 1e-4}),
    ("ElasticNet", {"alpha": 1.0, "l1_ratio": 0.5}),
    ("LogisticRegression", {"C": 1.0, "max_iter": 100}),
    ("KMeans", {"n_clusters": 8, "random_state": None}),
    ("DBSCAN", {"eps": 0.5, "min_samples": 5}),
    ("TruncatedSVD", {"n_components": 2}),
    ("NearestNeighbors", {"n_neighbors": 5}),
    ("KNeighborsClassifier", {"n_neighbors": 5}),
    ("KNeighborsRegressor", {"n_neighbors": 5}),
]


@pytest.mark.parametrize("estimator_name,expected", PARAM_EXPECTATIONS)
def test_param_roundtrip(estimator_name, expected):
    """PY-02: ctor args are stored verbatim and round-trip via get/set_params."""
    mlrs = pytest.importorskip("mlrs")  # noqa: F841  (wired in Plan 04)
    pytest.xfail("get_params/set_params round-trip lands in Plan 04")
