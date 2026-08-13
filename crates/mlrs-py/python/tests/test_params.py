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
    # Ridge carries sklearn's FULL Ridge signature (alpha .. random_state).
    # `device` (DEVICE-PARAM-01) is mlrs-only surface, not sklearn's: it pins
    # the host/device execution arm that was previously reachable only through
    # an `MLRS_*` environment flag. It sits BEFORE `output_type` (the other
    # mlrs-only param) and after the whole sklearn signature, so the sklearn
    # prefix of the signature still matches upstream positionally.
    "Ridge": {
        "alpha": 1.0,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": None,
        "tol": 1e-4,
        "solver": "auto",
        "positive": False,
        "random_state": None,
        "device": "auto",
        "output_type": "input",
    },
    # RidgeCV carries sklearn's FULL RidgeCV signature. `alphas` is a TUPLE in
    # sklearn's default and stays one here: `get_params` must round-trip the
    # ctor argument verbatim (the __init__ purity rule), so normalizing it to an
    # ndarray at construction would break `clone`.
    "RidgeCV": {
        "alphas": (0.1, 1.0, 10.0),
        "fit_intercept": True,
        "scoring": None,
        "cv": None,
        "gcv_mode": None,
        "store_cv_results": False,
        "alpha_per_target": False,
        "device": "auto",
        "output_type": "input",
    },
    # RidgeClassifier carries sklearn's FULL RidgeClassifier signature
    # (alpha .. random_state, minus `tol`'s docstring alias) — the
    # cpu/cuda fit work (mlrs-ridge-classifier-cpu / mlrs-ridge-classifier-cuda).
    "RidgeClassifier": {
        "alpha": 1.0,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": None,
        "tol": 1e-4,
        "class_weight": None,
        "solver": "auto",
        "positive": False,
        "random_state": None,
        "device": "auto",
        "output_type": "input",
    },
    # BayesianRidge carries sklearn's FULL signature (max_iter .. verbose).
    "BayesianRidge": {
        "max_iter": 300,
        "tol": 1e-3,
        "alpha_1": 1e-6,
        "alpha_2": 1e-6,
        "lambda_1": 1e-6,
        "lambda_2": 1e-6,
        "alpha_init": None,
        "lambda_init": None,
        "compute_score": False,
        "fit_intercept": True,
        "copy_X": True,
        "verbose": False,
        "device": "auto",
        "output_type": "input",
    },
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
    # --- HUBER-01: HuberRegressor's full sklearn ctor surface. ------------- #
    "HuberRegressor": {
        "epsilon": 1.35,
        "max_iter": 100,
        "alpha": 1e-4,
        "warm_start": False,
        "fit_intercept": True,
        "tol": 1e-5,
        "device": "auto",
        "output_type": "input",
    },
    # --- RANSAC-01: RANSACRegressor's full sklearn ctor surface. ----------- #
    "RANSACRegressor": {
        "estimator": None,
        "min_samples": None,
        "residual_threshold": None,
        "is_data_valid": None,
        "is_model_valid": None,
        "max_trials": 100,
        "max_skips": float("inf"),
        "stop_n_inliers": float("inf"),
        "stop_score": float("inf"),
        "stop_probability": 0.99,
        "loss": "absolute_error",
        "random_state": None,
        "output_type": "input",
        "device": "auto",
    },
    "LogisticRegression": {
        "C": 1.0,
        "fit_intercept": True,
        "max_iter": 100,
        "tol": 1e-4,
        "output_type": "input",
    },
    # sklearn's FULL KMeans ctor surface (verified against a live
    # ``SkKMeans().get_params()`` in test_oracle_kmeans_params.py).
    "KMeans": {
        "n_clusters": 8,
        "init": "k-means++",
        "n_init": "auto",
        "max_iter": 300,
        "tol": 1e-4,
        "verbose": 0,
        "random_state": None,
        "copy_x": True,
        "algorithm": "lloyd",
        "output_type": "input",
    },
    "DBSCAN": {"eps": 0.5, "min_samples": 5, "output_type": "input"},
    "TruncatedSVD": {"n_components": 2, "output_type": "input"},
    # NEIGH-PARAMS: NearestNeighbors carries sklearn's FULL parameter surface
    # too, `radius` in place of the two k-neighbours estimators' `weights`
    # (neither direction has a vote/mean to weight the other lacks).
    "NearestNeighbors": {
        "n_neighbors": 5,
        "output_type": "input",
        "radius": 1.0,
        "algorithm": "auto",
        "leaf_size": 30,
        "metric": "minkowski",
        "p": 2,
        "metric_params": None,
        "n_jobs": None,
        "device": "auto",
    },
    # KNN-CLF-PARAMS / KNN-REG-PARAMS: both k-neighbours estimators carry
    # sklearn's FULL parameter surface too.
    "KNeighborsClassifier": {
        "n_neighbors": 5,
        "output_type": "input",
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": None,
        "device": "auto",
    },
    "KNeighborsRegressor": {
        "n_neighbors": 5,
        "output_type": "input",
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": None,
        "device": "auto",
    },
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
        # sklearn's SGDClassifier default; the OvR multiclass fit's
        # loss-plateau early stop (mlrs-mbsgd-wgpu-and-convergence).
        "n_iter_no_change": 5,
        "device": "auto",
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
        # sklearn's SGDRegressor default (see MBSGDClassifier's note).
        "n_iter_no_change": 5,
        "device": "auto",
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
    # --- PREP-01: the six preprocessing scalers. Every default matches
    # sklearn's own, EXCEPT MaxAbsScaler: sklearn's has a `clip` param mlrs
    # does not implement, so its entry is 2 keys, not 3. --------------------
    "StandardScaler": {
        "copy": True,
        "with_mean": True,
        "with_std": True,
        "output_type": "input",
    },
    "MinMaxScaler": {
        "feature_range": (0, 1),
        "copy": True,
        "clip": False,
        "output_type": "input",
    },
    "MaxAbsScaler": {
        "copy": True,
        "output_type": "input",
    },
    "RobustScaler": {
        "with_centering": True,
        "with_scaling": True,
        "quantile_range": (25.0, 75.0),
        "copy": True,
        "unit_variance": False,
        "output_type": "input",
    },
    "Normalizer": {
        "norm": "l2",
        "copy": True,
        "output_type": "input",
    },
    "Binarizer": {
        "threshold": 0.0,
        "copy": True,
        "output_type": "input",
    },
    # GaussianMixture (MIX-01) — sklearn's full ctor surface, including BOTH
    # string-valued hyperparameters (`covariance_type`, `init_params`).
    "GaussianMixture": {
        "n_components": 1,
        "covariance_type": "full",
        "tol": 1e-3,
        "reg_covar": 1e-6,
        "max_iter": 100,
        "n_init": 1,
        "init_params": "kmeans",
        "weights_init": None,
        "means_init": None,
        "precisions_init": None,
        "random_state": None,
        "warm_start": False,
        "verbose": 0,
        "verbose_interval": 10,
        "device": "auto",
        "output_type": "input",
    },
    # BayesianGaussianMixture (MIX-02) — sklearn's full ctor surface, including
    # all THREE string-valued hyperparameters (`covariance_type`,
    # `init_params`, `weight_concentration_prior_type`) and the five priors.
    # Note `n_components` is keyword-ONLY on this estimator (sklearn's own
    # signature), unlike GaussianMixture's.
    "BayesianGaussianMixture": {
        "n_components": 1,
        "covariance_type": "full",
        "tol": 1e-3,
        "reg_covar": 1e-6,
        "max_iter": 100,
        "n_init": 1,
        "init_params": "kmeans",
        "weight_concentration_prior_type": "dirichlet_process",
        "weight_concentration_prior": None,
        "mean_precision_prior": None,
        "mean_prior": None,
        "degrees_of_freedom_prior": None,
        "covariance_prior": None,
        "random_state": None,
        "warm_start": False,
        "verbose": 0,
        "verbose_interval": 10,
        "device": "auto",
        "output_type": "input",
    },
    "SpectralClustering": {
        "n_clusters": 8,
        "n_components": None,
        "affinity": "rbf",
        "gamma": 1.0,
        "n_neighbors": 10,
        "random_state": None,
        # sklearn's SpectralClustering defaults; the n_samples<=64 cap lift
        # (mlrs-spectral-cpu-optimization) widened the shim to sklearn's full
        # param surface, not just the previously-implemented subset.
        "eigen_solver": None,
        "n_init": 10,
        "eigen_tol": "auto",
        "assign_labels": "kmeans",
        "degree": 3,
        "coef0": 1,
        "n_jobs": None,
        "verbose": False,
        "output_type": "input",
    },
    "SpectralEmbedding": {
        "n_components": 2,
        "affinity": "nearest_neighbors",
        "gamma": None,
        # sklearn's SpectralEmbedding default is None (resolved at fit), not
        # 10 — the full param-surface widening also fixed this pre-existing
        # divergence (see SpectralClustering's note).
        "n_neighbors": None,
        "random_state": None,
        "eigen_solver": None,
        "eigen_tol": "auto",
        "n_jobs": None,
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
        "device": "auto",
        "output_type": "input",
    },
    # HDBS-PARAMS: sklearn's COMPLETE 14-parameter surface, in sklearn's own
    # declaration order. `max_cluster_size` is `None` (sklearn's spelling of
    # "unbounded"; the shim maps it to the core's 0 sentinel) and `copy` is
    # `False` rather than sklearn 1.9's transitional `'warn'` — mlrs never
    # mutates the caller's array, so it is already in the post-1.10 state and
    # has nothing to warn about.
    "HDBSCAN": {
        "min_cluster_size": 5,
        "min_samples": None,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,
        "metric": "euclidean",
        "metric_params": None,
        "alpha": 1.0,
        "algorithm": "auto",
        "leaf_size": 40,
        "n_jobs": None,
        "cluster_selection_method": "eom",
        "allow_single_cluster": False,
        "store_centers": None,
        "copy": False,
        "output_type": "input",
    },
    # --- AGGLO-01: AgglomerativeClustering (single-linkage). -------------- #
    "AgglomerativeClustering": {
        "n_clusters": 2,
        "metric": "euclidean",
        "linkage": "single",
        "output_type": "input",
    },
    # --- TSNE-01 / TSNE-PARAMS: sklearn 1.9.0's FULL TSNE signature. ------ #
    "TSNE": {
        "n_components": 2,
        "perplexity": 30.0,
        "early_exaggeration": 12.0,
        "learning_rate": "auto",
        "max_iter": 1000,
        "n_iter_without_progress": 300,
        "min_grad_norm": 1e-7,
        "metric": "euclidean",
        "metric_params": None,
        "init": "pca",
        "verbose": 0,
        "random_state": None,
        "method": "barnes_hut",
        "angle": 0.5,
        "n_jobs": None,
        "device": "auto",
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
        "device": "auto",
        "output_type": "input",
    },
    "HistGradientBoostingRegressor": {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_bins": 64,
        "l2_regularization": 0.0,
        "min_samples_leaf": 20,
        "device": "auto",
        "output_type": "input",
    },
}

# The first non-output_type param to round-trip via set_params, with a new value.
SET_PARAM = {
    "LinearRegression": ("fit_intercept", False),
    "Ridge": ("device", "gpu"),
    "RidgeCV": ("gcv_mode", "svd"),
    "RidgeClassifier": ("alpha", 2.0),
    "BayesianRidge": ("max_iter", 50),
    "Lasso": ("alpha", 2.0),
    "ElasticNet": ("l1_ratio", 0.25),
    "HuberRegressor": ("epsilon", 1.5),
    "RANSACRegressor": ("max_trials", 42),
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
    # --- PREP-01: the six preprocessing scalers. --------------------------- #
    "StandardScaler": ("with_mean", False),
    "MinMaxScaler": ("feature_range", (-1, 1)),
    "MaxAbsScaler": ("copy", False),
    "RobustScaler": ("with_centering", False),
    "Normalizer": ("norm", "l1"),
    "Binarizer": ("threshold", 1.0),
    "SpectralClustering": ("n_clusters", 4),
    "GaussianMixture": ("covariance_type", "diag"),
    "BayesianGaussianMixture": ("weight_concentration_prior_type", "dirichlet_distribution"),
    "SpectralEmbedding": ("n_components", 3),
    "UMAP": ("n_neighbors", 10),
    "HDBSCAN": ("min_cluster_size", 10),
    # --- AGGLO-01 / TSNE-01. ---------------------------------------------- #
    "AgglomerativeClustering": ("n_clusters", 4),
    "TSNE": ("perplexity", 15.0),
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

    The ``getattr`` is guarded for the same reason as its twin in
    ``test_shims.py``: ``backend_supports_f64`` /
    ``backend_supports_f64_transcendental`` are exported names that the package
    ``__getattr__`` serves by importing the compiled extension, so a bare
    ``getattr`` raises ``ImportError`` on a not-yet-built tree. This module is
    called out as pure-Python (module docstring), and this function is its only
    line that was not — it is reached from ``test_matrix_covers_exports``, which
    would have been the one red test pre-build. Skipping an unresolvable name is
    safe: it cannot be an ``MlrsBase`` shim.
    """
    from mlrs.base import MlrsBase

    names = []
    for name in mlrs.__all__:
        try:
            obj = getattr(mlrs, name)
        except ImportError:
            continue
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
