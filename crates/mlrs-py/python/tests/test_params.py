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

import ast
import inspect

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
    # --- pre-existing shims that were not in the original ALL_12 matrix (now
    # covered so the matrix spans the full exported set, Plan 16-11). -------- #
    # IncrementalPCA requires an explicit n_components (like PCA).
    "IncrementalPCA": {
        "n_components": 2,
        "whiten": False,
        "batch_size": None,
        "output_type": "input",
    },
    "EmpiricalCovariance": {
        "store_precision": True,
        "assume_centered": False,
        "output_type": "input",
    },
    "LedoitWolf": {"assume_centered": False, "output_type": "input"},
    "GaussianRandomProjection": {
        "n_components": "auto",
        "eps": 0.1,
        "random_state": None,
        "output_type": "input",
    },
    "SparseRandomProjection": {
        "n_components": "auto",
        "density": "auto",
        "eps": 0.1,
        "random_state": None,
        "output_type": "input",
    },
    # --- Plan 16-11: the 15 newly-added shim classes (sklearn-named defaults
    # matching each Py* #[new] signature). ----------------------------------- #
    "LinearSVC": {
        "loss": "squared_hinge",
        "penalty": "l2",
        "C": 1.0,
        "intercept_scaling": 1.0,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-4,
        "output_type": "input",
    },
    "LinearSVR": {
        "loss": "squared_epsilon_insensitive",
        "penalty": "l2",
        "C": 1.0,
        "epsilon": 0.0,
        "intercept_scaling": 1.0,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-4,
        "output_type": "input",
    },
    "MBSGDClassifier": {
        "loss": "hinge",
        "penalty": "l2",
        "alpha": 1e-4,
        "l1_ratio": 0.15,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-3,
        "learning_rate": "optimal",
        "eta0": 0.01,
        "power_t": 0.5,
        "batch_size": 1,
        "shuffle": True,
        "seed": 0,
        "output_type": "input",
    },
    "MBSGDRegressor": {
        "loss": "squared_error",
        "penalty": "l2",
        "alpha": 1e-4,
        "l1_ratio": 0.15,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-3,
        "learning_rate": "invscaling",
        "eta0": 0.01,
        "power_t": 0.25,
        "epsilon": 0.1,
        "batch_size": 1,
        "shuffle": True,
        "seed": 0,
        "output_type": "input",
    },
    "GaussianNB": {
        "var_smoothing": 1e-9,
        "priors": None,
        "output_type": "input",
    },
    "MultinomialNB": {
        "alpha": 1.0,
        "force_alpha": True,
        "fit_prior": True,
        "class_prior": None,
        "output_type": "input",
    },
    "BernoulliNB": {
        "alpha": 1.0,
        "force_alpha": True,
        "binarize": 0.0,
        "fit_prior": True,
        "class_prior": None,
        "output_type": "input",
    },
    "ComplementNB": {
        "alpha": 1.0,
        "force_alpha": True,
        "fit_prior": True,
        "class_prior": None,
        "norm": False,
        "output_type": "input",
    },
    "CategoricalNB": {
        "alpha": 1.0,
        "force_alpha": True,
        "fit_prior": True,
        "class_prior": None,
        "min_categories": None,
        "output_type": "input",
    },
    "KernelRidge": {
        "kernel": "linear",
        "alpha": 1.0,
        "gamma": None,
        "degree": 3.0,
        "coef0": 1.0,
        "output_type": "input",
    },
    "KernelDensity": {
        "kernel": "gaussian",
        "bandwidth": 1.0,
        "bandwidth_rule": "numeric",
        "output_type": "input",
    },
    "SpectralClustering": {
        "n_clusters": 8,
        "n_components": None,
        "affinity": "rbf",
        "gamma": 1.0,
        "n_neighbors": 10,
        "random_state": None,
        "output_type": "input",
    },
    "SpectralEmbedding": {
        "n_components": 2,
        "affinity": "nearest_neighbors",
        "gamma": None,
        "n_neighbors": 10,
        "output_type": "input",
    },
    "UMAP": {
        "n_neighbors": 15,
        "n_components": 2,
        "min_dist": 0.1,
        "spread": 1.0,
        "metric": "euclidean",
        "n_epochs": None,
        "init": "spectral",
        "random_state": None,
        "learning_rate": 1.0,
        "set_op_mix_ratio": 1.0,
        "local_connectivity": 1.0,
        "repulsion_strength": 1.0,
        "negative_sample_rate": 5,
        "a": None,
        "b": None,
        "output_type": "input",
    },
    "HDBSCAN": {
        "min_cluster_size": 5,
        "min_samples": None,
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_method": "eom",
        "metric": "euclidean",
        "alpha": 1.0,
        "max_cluster_size": 0,
        "output_type": "input",
    },
    # --- TASK-16 (PY-ENS-05, RF): RandomForestClassifier/Regressor. ------- #
    "RandomForestClassifier": {
        "n_estimators": 100,
        "max_depth": 10,
        "n_bins": 32,
        "max_features": "sqrt",
        "min_samples_split": 2.0,
        "min_samples_leaf": 1.0,
        "bootstrap": True,
        "oob_score": False,
        "seed": 42,
        "output_type": "input",
    },
    "RandomForestRegressor": {
        "n_estimators": 100,
        "max_depth": 10,
        "n_bins": 32,
        "max_features": 1.0,
        "min_samples_split": 2.0,
        "min_samples_leaf": 1.0,
        "bootstrap": True,
        "oob_score": False,
        "seed": 42,
        "output_type": "input",
    },
    # --- TASK-25 (PY-ENS-05, HGB): HistGradientBoostingClassifier/Regressor.
    "HistGradientBoostingClassifier": {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_bins": 64,
        "l2_regularization": 0.0,
        "min_samples_leaf": 20,
        "output_type": "input",
    },
    "HistGradientBoostingRegressor": {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_bins": 64,
        "l2_regularization": 0.0,
        "min_samples_leaf": 20,
        "output_type": "input",
    },
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
    # --- pre-existing shims newly added to the matrix (Plan 16-11). -------- #
    "IncrementalPCA": ("whiten", True),
    "EmpiricalCovariance": ("assume_centered", True),
    "LedoitWolf": ("assume_centered", True),
    "GaussianRandomProjection": ("eps", 0.2),
    "SparseRandomProjection": ("eps", 0.2),
    # --- Plan 16-11: the 15 newly-added shim classes. --------------------- #
    "LinearSVC": ("C", 2.0),
    "LinearSVR": ("C", 2.0),
    "MBSGDClassifier": ("alpha", 1e-3),
    "MBSGDRegressor": ("alpha", 1e-3),
    "GaussianNB": ("var_smoothing", 1e-8),
    "MultinomialNB": ("alpha", 2.0),
    "BernoulliNB": ("alpha", 2.0),
    "ComplementNB": ("alpha", 2.0),
    "CategoricalNB": ("alpha", 2.0),
    "KernelRidge": ("alpha", 2.0),
    "KernelDensity": ("bandwidth", 2.0),
    "SpectralClustering": ("n_clusters", 4),
    "SpectralEmbedding": ("n_components", 3),
    "UMAP": ("n_neighbors", 10),
    "HDBSCAN": ("min_cluster_size", 10),
    # --- TASK-16 (PY-ENS-05, RF): RandomForestClassifier/Regressor. ------- #
    "RandomForestClassifier": ("n_estimators", 10),
    "RandomForestRegressor": ("n_estimators", 10),
    # --- TASK-25 (PY-ENS-05, HGB): HistGradientBoostingClassifier/Regressor.
    "HistGradientBoostingClassifier": ("max_iter", 10),
    "HistGradientBoostingRegressor": ("max_iter", 10),
}

# The full estimator-shim matrix, derived from EXPECTED_PARAMS so it cannot drift
# from the per-class default tables (and grows automatically with them).
ALL_SHIMS = list(EXPECTED_PARAMS)


def _exported_shim_names():
    """Every exported ``mlrs`` symbol that is a pure-Python estimator shim.

    The estimator shims are the exported names whose object is an
    ``MlrsBase`` subclass (excludes the surfaced ``backend_supports_f64`` flag
    and the ``johnson_lindenstrauss_min_dim`` helper function). Deriving the
    expected matrix membership from this set keeps EXPECTED_PARAMS honest: a
    newly-added shim that is not in the table fails ``test_matrix_covers_exports``.
    """
    from mlrs.base import MlrsBase

    names = []
    for name in mlrs.__all__:
        obj = getattr(mlrs, name)
        if isinstance(obj, type) and issubclass(obj, MlrsBase):
            names.append(name)
    return names


# Shims that require an explicit positional ctor arg (no zero-arg default).
_REQUIRES_N_COMPONENTS = ("PCA", "IncrementalPCA")


def _construct(name):
    """Construct with the v1-required ctor args (PCA/IncrementalPCA need one)."""
    cls = getattr(mlrs, name)
    if name in _REQUIRES_N_COMPONENTS:
        return cls(n_components=2)
    return cls()


def test_matrix_covers_exports():
    """The static matrix covers EXACTLY the exported estimator-shim set.

    Proves the EXPECTED_PARAMS / SET_PARAM tables track the real exported
    surface (no shim left untested, no stale entry) — so the parametrized tests
    below exercise every estimator the package ships.
    """
    exported = set(_exported_shim_names())
    assert set(EXPECTED_PARAMS) == exported, (
        f"EXPECTED_PARAMS keys {set(EXPECTED_PARAMS) ^ exported} "
        f"differ from the exported estimator shims"
    )
    assert set(SET_PARAM) == exported, (
        f"SET_PARAM keys {set(SET_PARAM) ^ exported} differ from the "
        f"exported estimator shims"
    )


@pytest.mark.parametrize("name", ALL_SHIMS)
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


@pytest.mark.parametrize("name", ALL_SHIMS)
def test_set_params_roundtrip(name):
    """(b) set_params(**{param: new}) round-trips through get_params."""
    est = _construct(name)
    param, new_value = SET_PARAM[name]
    est.set_params(**{param: new_value})
    assert est.get_params()[param] == new_value


@pytest.mark.parametrize("name", ALL_SHIMS)
def test_init_purity_stores_kwargs_verbatim(name):
    """(c) __init__ stores explicit kwargs verbatim (no transformation)."""
    param, value = SET_PARAM[name]
    cls = getattr(mlrs, name)
    kwargs = {param: value}
    if name in _REQUIRES_N_COMPONENTS and param != "n_components":
        kwargs["n_components"] = 2
    est = cls(**kwargs)
    assert getattr(est, param) == value  # stored under the SAME name
    assert est.get_params()[param] == value


@pytest.mark.parametrize("name", ALL_SHIMS)
def test_init_purity_ast(name):
    """(c') STATIC __init__ purity — the strongest SHIM-01 guarantee without FFI.

    Parses ``cls.__init__`` with the ``ast`` module (no instance constructed, no
    compiled ``_mlrs`` extension imported) and asserts every statement in the
    body is a bare ``self.<name> = <name>`` assignment: each ctor arg stored
    verbatim under the SAME name, with NO computation/validation node
    (``ast.Call`` / ``ast.BinOp`` / ``ast.Compare`` / etc.). This makes any
    impure ``self.x = validate(x)`` body a hard test FAILURE rather than a
    runtime surprise (SHIM-01 invariant, D-07 step 3). The parametrization draws
    from the shared ``ALL_SHIMS`` list (derived from EXPECTED_PARAMS) so it
    the shim matrix.
    """
    cls = getattr(mlrs, name)
    src = inspect.getsource(cls.__init__).strip()
    tree = ast.parse(src)
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef), (
        f"{name}.__init__ did not parse as a function def"
    )
    assert fn.body, f"{name}.__init__ has an empty body"

    for stmt in fn.body:
        # Only assignments — no `if`/`for`/`raise`/`assert`/expression calls.
        assert isinstance(stmt, ast.Assign), (
            f"{name}.__init__ has a non-assignment statement "
            f"{type(stmt).__name__} — __init__ must be pure (store-only)"
        )
        # Exactly one target, of the shape `self.<attr>`.
        assert len(stmt.targets) == 1, (
            f"{name}.__init__ has a multi-target assignment — only "
            f"`self.<name> = <name>` is allowed"
        )
        tgt = stmt.targets[0]
        assert (
            isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"
        ), (
            f"{name}.__init__ assigns to {ast.dump(tgt)} — only attributes of "
            f"`self` may be set in __init__"
        )
        # Value must be a BARE Name (no Call/BinOp/Compare/etc.).
        assert isinstance(stmt.value, ast.Name), (
            f"{name}.__init__ stores a computed value "
            f"({type(stmt.value).__name__}) into self.{tgt.attr} — __init__ "
            f"must store each ctor arg verbatim with no computation/validation"
        )
        # Stored under the SAME identifier (`self.x = x`, never `self.x = y`).
        assert tgt.attr == stmt.value.id, (
            f"{name}.__init__ stores `{stmt.value.id}` into self.{tgt.attr} — "
            f"each arg must be stored under its own name"
        )


def test_logreg_exposes_capital_C():
    """(d) LogisticRegression exposes sklearn ``C``, not the Rust field ``c``."""
    params = mlrs.LogisticRegression().get_params()
    assert "C" in params
    assert "c" not in params


def test_kmeans_exposes_random_state():
    """(d) KMeans exposes ``random_state`` (mapped to Rust seed at the boundary)."""
    assert "random_state" in mlrs.KMeans().get_params()
