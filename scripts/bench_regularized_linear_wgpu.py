"""Benchmark regularized regression fits on the host and the wgpu Gram path.

Run with an mlrs wgpu wheel importable by Python:
    OPENBLAS_NUM_THREADS=1 python scripts/bench_regularized_linear_wgpu.py
"""

import argparse
from statistics import median
from time import perf_counter

import mlrs
import numpy as np
from threadpoolctl import threadpool_limits


def timed_fit(estimator, options, X, y, device, repeats):
    def fit():
        return estimator(**options, device=device).fit(X, y)

    model = fit()  # Warm kernels and allocator outside the timing loop.
    if model.device_ != device:
        raise RuntimeError(
            f"{estimator.__name__}: requested {device}, ran {model.device_}"
        )
    times = []
    for _ in range(repeats):
        start = perf_counter()
        model = fit()
        times.append(perf_counter() - start)
    return model, median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rows", type=int, nargs="+", default=[2048, 32768])
    args = parser.parse_args()
    if args.repeats < 1 or min(args.rows) < 129:
        parser.error("Requires at least one repeat and 129 rows for each case.")
    backend = mlrs._mlrs.regularized_gram_backend(128, True)
    if backend != "wgpu":
        raise RuntimeError(
            f"Expected wgpu with float64 Gram support, found {backend!r}."
        )

    rng = np.random.default_rng(42)
    print(
        "| Estimator | Samples | Features | CPU ms | wgpu ms | CPU/wgpu | Max prediction error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    with threadpool_limits(limits=1):
        for n in args.rows:
            for d in (32, 128):
                X = rng.standard_normal((n, d))
                weights = rng.standard_normal((d, 4))
                weights[10:] = 0
                Y = X @ weights + rng.standard_normal((n, 4)) * 0.2
                cases = (
                    (mlrs.MultiTaskElasticNet, {"alpha": 0.07, "tol": 1e-8}, Y),
                    (
                        mlrs.ElasticNetCV,
                        {"alphas": 8, "cv": 3, "l1_ratio": [0.5, 0.9], "tol": 1e-8},
                        Y[:, 0],
                    ),
                    (mlrs.LassoLars, {"alpha": 0.04}, Y[:, 0]),
                )
                for estimator, options, y in cases:
                    cpu, host = timed_fit(estimator, options, X, y, "cpu", args.repeats)
                    gpu, wgpu = timed_fit(estimator, options, X, y, "gpu", args.repeats)
                    expected = cpu.predict(X)
                    predicted = gpu.predict(X)
                    np.testing.assert_allclose(
                        predicted, expected, atol=1e-8, rtol=1e-8
                    )
                    error = np.max(np.abs(predicted - expected))
                    print(
                        f"| {estimator.__name__} | {n} | {d} | {host * 1000:.2f} | "
                        f"{wgpu * 1000:.2f} | {host / wgpu:.2f}× | {error:.3g} |",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
