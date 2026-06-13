"""Hyperparameter round-trip (PY-02: sklearn-named ctor params + get/set_params).

Real assertions over all 12 estimators (the Wave-0 xfail/importorskip guard is
removed now that the shims exist). For each estimator this proves:

  (a) ``get_params()`` contains exactly the sklearn-named keys from the RESEARCH
      06 §Hyperparameter Mapping table (plus the base ``output_type``), with the
      documented sklearn defaults.
  (b) ``set_params(**{param: new})`` round-trips through ``get_params()``.
  (c) ``__init__`` purity — constructing with explicit kwargs then ``get_params``
      returns those exact values verbatim (no transformation; e.g. ``self.C = C``).
  (d) LogisticRegression exposes ``C`` (not ``c``); KMeans exposes ``random_state``.

These are pure-Python (no compiled ``_mlrs`` needed): they exercise only the
sklearn ``BaseEstimator`` machinery over the faithful ``__init__`` (PY-02).
"""

import pytest

import mlrs

# Req: PY-02 — the per-estimator sklearn-named ctor params + defaults
# (RESEARCH 06 §Hyperparameter Mapping). `output_type` is the base param every
# mlrs estimator adds. PCA has no default n_components (v1 requires explicit int).
EXPECTED_PARAMS = {
    "LinearRegression": {"fit_intercept": True, "output_type": "input"},
    "Ridge": {"alpha": 1.0, "fit_intercept": True, "output_type": "input"},
    "Lasso": {
        "alpha": 1.0,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-4,
        "output_type": "input",
    },
    "ElasticNet": {
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-4,
        "output_type": "input",
    },
    "LogisticRegression": {
        "C": 1.0,
        "fit_intercept": True,
        "max_iter": 100,
        "tol": 1e-4,
        "output_type": "input",
    },
    "KMeans": {
        "n_clusters": 8,
        "init": "k-means++",
        "max_iter": 300,
        "tol": 1e-4,
        "random_state": None,
        "output_type": "input",
    },
    "DBSCAN": {"eps": 0.5, "min_samples": 5, "output_type": "input"},
    "TruncatedSVD": {"n_components": 2, "output_type": "input"},
    "NearestNeighbors": {"n_neighbors": 5, "output_type": "input"},
    "KNeighborsClassifier": {"n_neighbors": 5, "output_type": "input"},
    "KNeighborsRegressor": {"n_neighbors": 5, "output_type": "input"},
    # PCA requires an explicit n_components — constructed with n_components=2.
    "PCA": {"n_components": 2, "output_type": "input"},
}

# The first non-output_type param to round-trip via set_params, with a new value.
SET_PARAM = {
    "LinearRegression": ("fit_intercept", False),
    "Ridge": ("alpha", 2.0),
    "Lasso": ("alpha", 2.0),
    "ElasticNet": ("l1_ratio", 0.25),
    "LogisticRegression": ("C", 2.0),
    "KMeans": ("n_clusters", 5),
    "DBSCAN": ("eps", 1.5),
    "TruncatedSVD": ("n_components", 3),
    "NearestNeighbors": ("n_neighbors", 7),
    "KNeighborsClassifier": ("n_neighbors", 7),
    "KNeighborsRegressor": ("n_neighbors", 7),
    "PCA": ("n_components", 3),
}

ALL_12 = list(EXPECTED_PARAMS)


def _construct(name):
    """Construct with the v1-required ctor args (PCA needs n_components)."""
    cls = getattr(mlrs, name)
    if name == "PCA":
        return cls(n_components=2)
    return cls()


@pytest.mark.parametrize("name", ALL_12)
def test_default_params_match_sklearn_names(name):
    """(a) get_params has exactly the sklearn-named keys + documented defaults."""
    params = _construct(name).get_params()
    assert set(params) == set(EXPECTED_PARAMS[name]), (
        f"{name}: unexpected param keys {set(params)} "
        f"!= {set(EXPECTED_PARAMS[name])}"
    )
    for key, expected in EXPECTED_PARAMS[name].items():
        assert params[key] == expected, (
            f"{name}.{key} default {params[key]!r} != {expected!r}"
        )


@pytest.mark.parametrize("name", ALL_12)
def test_set_params_roundtrip(name):
    """(b) set_params(**{param: new}) round-trips through get_params."""
    est = _construct(name)
    param, new_value = SET_PARAM[name]
    est.set_params(**{param: new_value})
    assert est.get_params()[param] == new_value


@pytest.mark.parametrize("name", ALL_12)
def test_init_purity_stores_kwargs_verbatim(name):
    """(c) __init__ stores explicit kwargs verbatim (no transformation)."""
    param, value = SET_PARAM[name]
    cls = getattr(mlrs, name)
    kwargs = {param: value}
    if name == "PCA" and param != "n_components":
        kwargs["n_components"] = 2
    est = cls(**kwargs)
    assert getattr(est, param) == value  # stored under the SAME name
    assert est.get_params()[param] == value


def test_logreg_exposes_capital_C():
    """(d) LogisticRegression exposes sklearn ``C``, not the Rust field ``c``."""
    params = mlrs.LogisticRegression().get_params()
    assert "C" in params
    assert "c" not in params


def test_kmeans_exposes_random_state():
    """(d) KMeans exposes ``random_state`` (mapped to Rust seed at the boundary)."""
    assert "random_state" in mlrs.KMeans().get_params()
