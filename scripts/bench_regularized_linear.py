"""Compare fit speed and prediction parity with scikit-learn."""

import argparse
from statistics import median
from time import perf_counter

import mlrs
import numpy as np
import sklearn
from sklearn import linear_model
from threadpoolctl import threadpool_limits


def timed_fit(estimator, options, X, y, repeats):
    model = estimator(**options).fit(X, y)
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        model = estimator(**options).fit(X, y)
        samples.append(perf_counter() - start)
    return model, median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--shapes", nargs="+", default=["4096x24", "16384x64", "512x256"]
    )
    parser.add_argument("--estimators", nargs="+")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    rng = np.random.default_rng(42)
    cv = {"alphas": 24, "cv": 3, "n_jobs": 1, "tol": 1e-8}
    multitask = {"alpha": 0.06, "tol": 1e-8}
    cases = [
        ("LassoCV", False, cv),
        ("ElasticNetCV", False, {**cv, "l1_ratio": [0.5, 0.9]}),
        ("MultiTaskLasso", True, multitask),
        ("MultiTaskElasticNet", True, multitask),
        ("MultiTaskLassoCV", True, cv),
        ("MultiTaskElasticNetCV", True, {**cv, "l1_ratio": [0.5, 0.9]}),
        ("LassoLars", False, {"alpha": 0.06}),
        ("LassoLarsCV", False, {"cv": 3, "n_jobs": 1}),
        ("LassoLarsIC", False, {"criterion": "bic"}),
    ]
    print(
        f"scikit-learn {sklearn.__version__}; float64; one BLAS thread; median of {args.repeats}"
    )
    print(
        "| Estimator | Shape | mlrs ms | sklearn ms | sklearn/mlrs | Max prediction error |"
    )
    print("|---|---|---:|---:|---:|---:|")
    with threadpool_limits(limits=1):
        for shape in args.shapes:
            rows, features = map(int, shape.split("x"))
            X = rng.standard_normal((rows, features))
            weights = rng.standard_normal((features, 4))
            weights[12:] = 0
            Y = X @ weights + rng.standard_normal((rows, 4)) * 0.2
            for name, multi, options in cases:
                if args.estimators and name not in args.estimators:
                    continue
                y = Y if multi else Y[:, 0]
                actual, mlrs_time = timed_fit(
                    getattr(mlrs, name), options, X, y, args.repeats
                )
                expected, sklearn_time = timed_fit(
                    getattr(linear_model, name), options, X, y, args.repeats
                )
                target = expected.predict(X)
                prediction = actual.predict(X)
                np.testing.assert_allclose(prediction, target, atol=1e-5, rtol=1e-5)
                if hasattr(expected, "alpha_"):
                    np.testing.assert_allclose(
                        actual.alpha_, expected.alpha_, atol=1e-8, rtol=1e-5
                    )
                if hasattr(expected, "l1_ratio_"):
                    np.testing.assert_equal(actual.l1_ratio_, expected.l1_ratio_)
                error = np.max(np.abs(prediction - target))
                print(
                    f"| {name} | {shape} | {mlrs_time * 1000:.2f} | {sklearn_time * 1000:.2f} | "
                    f"{sklearn_time / mlrs_time:.2f}× | {error:.3g} |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
