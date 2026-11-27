#!/usr/bin/env python3
"""Benchmark decomposition against scikit-learn using one BLAS thread.

Build and expose mlrs to Python, then run python scripts/bench_decomposition.py.
Each timing is the median of alternating calls after warm-up and parity checks.
"""

import argparse
import platform
from statistics import median
from time import perf_counter

import numpy as np
import scipy
import sklearn
from mlrs import decomposition as mlrs
from scipy import sparse
from sklearn import decomposition as sklearn_decomp
from threadpoolctl import threadpool_limits


def workloads():
    rng = np.random.RandomState(42)
    signals = rng.normal(size=(400, 96))
    dictionary = rng.normal(size=(128, 96))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    tall = rng.normal(size=(1200, 48))
    nmf = np.abs(rng.normal(size=(700, 160)))
    counts = sparse.random(
        600, 1500, density=0.015, random_state=rng,
        data_rvs=lambda n: rng.randint(1, 6, n), format="csr",
    )
    for algorithm, size in (("omp", 8), ("omp", 16), ("lars", 8)):
        def encode(module, algorithm=algorithm, size=size):
            return module.sparse_encode(
                signals, dictionary, algorithm=algorithm, n_nonzero_coefs=size
            )

        yield f"sparse_encode {algorithm.upper()} ({size} atoms)", encode

    def kernel_pca(module, n_components):
        return module.KernelPCA(
            n_components, kernel="linear", random_state=0
        ).fit_transform(tall)

    def nmf_cd(module):
        return module.NMF(
            12, init="random", random_state=0, max_iter=25, tol=0
        ).fit_transform(nmf)

    def lda(module):
        return module.LatentDirichletAllocation(
            8, random_state=0, max_iter=3
        ).fit_transform(counts)

    yield "KernelPCA linear (3 components)", lambda module: kernel_pca(module, 3)
    yield "KernelPCA linear (12 components)", lambda module: kernel_pca(module, 12)
    yield "NMF coordinate descent", nmf_cd
    yield "LDA batch", lda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    print(
        f"Python {platform.python_version()} | NumPy {np.__version__} | "
        f"SciPy {scipy.__version__} | scikit-learn {sklearn.__version__}"
    )
    print(f"{'Workload':<36} {'sklearn ms':>12} {'mlrs ms':>12} {'Speedup':>10}")
    modules = (sklearn_decomp, mlrs)
    with threadpool_limits(limits=1):
        for label, run in workloads():
            expected, actual = (run(module) for module in modules)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            timings = ([], [])
            for repetition in range(args.repeats):
                order = (0, 1) if repetition % 2 == 0 else (1, 0)
                for index in order:
                    start = perf_counter()
                    run(modules[index])
                    timings[index].append(perf_counter() - start)
            reference, optimized = (median(times) * 1000 for times in timings)
            print(
                f"{label:<36} {reference:>12.2f} {optimized:>12.2f} "
                f"{reference / optimized:>9.2f}×",
                flush=True,
            )


if __name__ == "__main__":
    main()
