"""Observe public mlrs and scikit-learn regression outputs for the Lean runner."""

import json

import mlrs
import numpy as np
from mlrs.linear import enet_path, lasso_path
from scipy import sparse
from sklearn import linear_model as oracle
from sklearn.base import clone


def flat(value):
    return np.asarray(value).ravel().tolist()


basis = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [1, 1, 1]])
X = np.tile(basis, (3, 1)) + [2, 0, 1]
Y = (X - [2, 0, 1]) @ [[3, 4], [-1.2, 0], [0, 0]] + [7, -3]
lasso = mlrs.MultiTaskLasso(alpha=1.5, tol=1e-11).fit(X, Y)
elastic = mlrs.MultiTaskElasticNet(alpha=1.5, tol=1e-11).fit(X, Y)
alphas, coefficients, gaps, iterations = lasso_path(
    basis, Y[:4] - [7, -3], alphas=[1.5, 5.0, 0.5], tol=1e-11, return_n_iter=True
)

rng = np.random.default_rng(1)
design = rng.normal(size=(24, 9))
design[:, 2:] += 3 * design[:, :1]
target = rng.normal(size=len(design))
multi_target = np.column_stack([target, rng.normal(size=len(design))]) + [2, -3]
weights = np.linspace(0.2, 2.0, len(design))
examples = [
    (
        "LassoCV",
        sparse.csr_matrix(design),
        target,
        {"alphas": [0.4, 0.2, 0.1], "cv": 3, "positive": True, "tol": 1e-9},
        weights,
    ),
    (
        "ElasticNetCV",
        design,
        target,
        {
            "alphas": [0.4, 0.2, 0.1],
            "cv": 3,
            "l1_ratio": [0.2, 0.8],
            "n_jobs": 2,
            "tol": 1e-9,
        },
        weights,
    ),
    ("MultiTaskLasso", design, multi_target, {"alpha": 0.25, "tol": 1e-9}, None),
    (
        "MultiTaskElasticNet",
        design.astype(np.float32),
        multi_target,
        {"alpha": 0.25, "l1_ratio": 0.7, "tol": 1e-7},
        None,
    ),
    (
        "MultiTaskLassoCV",
        design,
        multi_target,
        {"alphas": 7, "cv": 3, "tol": 1e-9},
        None,
    ),
    (
        "MultiTaskElasticNetCV",
        design,
        multi_target,
        {"alphas": [0.4, 0.2, 0.1], "cv": 3, "l1_ratio": [0.2, 0.8], "tol": 1e-9},
        None,
    ),
    ("LassoLars", design, target, {"alpha": 0.0, "precompute": False}, None),
    ("LassoLarsCV", design, target, {"cv": 3}, None),
    ("LassoLarsIC", design, target, {"criterion": "bic"}, None),
]
observations = []
for name, x, y, options, weight in examples:
    params = {} if weight is None else {"sample_weight": weight}
    actual = clone(getattr(mlrs, name)(**options)).fit(x, y, **params)
    expected = getattr(oracle, name)(**options).fit(x, y, **params)
    fields = {}
    for attr in (
        "coef_",
        "intercept_",
        "alpha_",
        "l1_ratio_",
        "alphas_",
        "mse_path_",
        "coef_path_",
        "cv_alphas_",
        "criterion_",
        "noise_variance_",
    ):
        if hasattr(expected, attr):
            fields[attr] = [flat(getattr(actual, attr)), flat(getattr(expected, attr))]
    fields["predict"] = [flat(actual.predict(x)), flat(expected.predict(x))]
    observations.append({"name": name, "fields": fields})

enet_options = {
    "l1_ratio": 0.6,
    "alphas": [0.03, 0.1, 0.3],
    "positive": True,
    "tol": 1e-9,
}
path_x = sparse.csc_matrix(design)
actual_path = enet_path(path_x, target, **enet_options)
expected_path = oracle.enet_path(path_x, target, **enet_options)
observations.append(
    {
        "name": "enet_path",
        "fields": {
            name: [flat(actual), flat(expected)]
            for name, actual, expected in zip(
                ("alphas", "coefficients", "gaps"),
                actual_path,
                expected_path,
                strict=True,
            )
        },
    }
)

print(
    json.dumps(
        {
            "lasso": flat(lasso.coef_),
            "elastic": flat(elastic.coef_),
            "intercept": flat(lasso.intercept_),
            "alphas": flat(alphas),
            "pathAtHalf": flat(coefficients[..., -1]),
            "gaps": flat(gaps),
            "iterations": iterations,
            "larsDropIterations": mlrs.LassoLars(alpha=0.0, precompute=False)
            .fit(design, target)
            .n_iter_,
            "observations": observations,
        },
        allow_nan=False,
    )
)
