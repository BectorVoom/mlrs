"""Observe public mlrs and scikit-learn regression outputs for the Lean runner."""

import json
import os

import mlrs
import numpy as np
from mlrs.linear import enet_path, lasso_path
from scipy import sparse
from sklearn import linear_model as oracle
from sklearn.base import clone


def flat(value):
    return np.asarray(value).ravel().tolist()


expected_backend = os.environ.get("MLRS_LEAN_GPU_BACKEND")
device = "gpu" if expected_backend else "cpu"
backend = "cpu"
if expected_backend:
    backend = mlrs._mlrs.regularized_gram_backend(9, True)

basis = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [1, 1, 1]])
X = np.tile(basis, (3, 1)) + [2, 0, 1]
Y = (X - [2, 0, 1]) @ [[3, 4], [-1.2, 0], [0, 0]] + [7, -3]
lasso = mlrs.MultiTaskLasso(alpha=1.5, tol=1e-11, device=device).fit(X, Y)
elastic = mlrs.MultiTaskElasticNet(alpha=1.5, tol=1e-11, device=device).fit(X, Y)
alphas, coefficients, gaps, iterations = lasso_path(
    basis,
    Y[:4] - [7, -3],
    alphas=[1.5, 5.0, 0.5],
    tol=1e-11,
    return_n_iter=True,
    device=device,
)

rng = np.random.default_rng(1)
design = rng.normal(size=(24, 9))
design[:, 2:] += 3 * design[:, :1]
target = rng.normal(size=len(design))
multi_target = np.column_stack([target, rng.normal(size=len(design))]) + [2, -3]
# Use enough rows to dispatch the dense CPU regressors through the native
# Gram path. The original shorter fixture still exercises LARS feature drops.
cd_design = np.tile(design, (4, 1))
cd_target = np.tile(target, 4)
cd_multi_target = np.tile(multi_target, (4, 1))
weights = np.linspace(0.2, 2.0, len(cd_design))
examples = [
    (
        "LassoCV",
        sparse.csr_matrix(cd_design),
        cd_target,
        {"alphas": [0.4, 0.2, 0.1], "cv": 3, "positive": True, "tol": 1e-9},
        weights,
    ),
    (
        "ElasticNetCV",
        cd_design,
        cd_target,
        {
            "alphas": [0.4, 0.2, 0.1],
            "cv": 3,
            "l1_ratio": [0.2, 0.8],
            "n_jobs": 2,
            "tol": 1e-9,
        },
        weights,
    ),
    ("MultiTaskLasso", cd_design, cd_multi_target, {"alpha": 0.25, "tol": 1e-9}, None),
    (
        "MultiTaskElasticNet",
        cd_design.astype(np.float32),
        cd_multi_target,
        {"alpha": 0.25, "l1_ratio": 0.7, "tol": 1e-7},
        None,
    ),
    (
        "MultiTaskLassoCV",
        cd_design,
        cd_multi_target,
        {"alphas": 7, "cv": 3, "tol": 1e-9},
        None,
    ),
    (
        "MultiTaskElasticNetCV",
        cd_design,
        cd_multi_target,
        {"alphas": [0.4, 0.2, 0.1], "cv": 3, "l1_ratio": [0.2, 0.8], "tol": 1e-9},
        None,
    ),
    ("LassoLars", design, target, {"alpha": 0.0, "precompute": True}, None),
    ("LassoLarsCV", design, target, {"cv": 3}, None),
    ("LassoLarsIC", design, target, {"criterion": "bic"}, None),
]
observations = []
for name, x, y, options, weight in examples:
    if device == "gpu" and sparse.issparse(x):
        x = x.toarray()
    params = {} if weight is None else {"sample_weight": weight}
    actual = clone(getattr(mlrs, name)(**options, device=device)).fit(x, y, **params)
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
    observations.append({"name": name, "fields": fields, "device": actual.device_})

enet_options = {
    "l1_ratio": 0.6,
    "alphas": [0.03, 0.1, 0.3],
    "positive": True,
    "tol": 1e-9,
}
path_x = design if device == "gpu" else sparse.csc_matrix(design)
actual_path = enet_path(path_x, target, **enet_options, device=device)
expected_path = oracle.enet_path(path_x, target, **enet_options)
observations.append(
    {
        "name": "enet_path",
        "device": device,
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
            "device": device,
            "backend": backend,
            "expectedBackend": expected_backend or "cpu",
            "lasso": flat(lasso.coef_),
            "elastic": flat(elastic.coef_),
            "intercept": flat(lasso.intercept_),
            "alphas": flat(alphas),
            "pathAtHalf": flat(coefficients[..., -1]),
            "gaps": flat(gaps),
            "iterations": iterations,
            "larsDropIterations": mlrs.LassoLars(
                alpha=0.0, precompute=False, device=device
            )
            .fit(design, target)
            .n_iter_,
            "observations": observations,
        },
        allow_nan=False,
    )
)
