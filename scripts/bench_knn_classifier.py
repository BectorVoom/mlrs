#!/usr/bin/env python3
"""KNeighborsClassifier ``fit`` wall-clock comparison harness.

Times ``sklearn.neighbors.KNeighborsClassifier(algorithm='brute')`` on the SAME
splitmix64 design matrix as the mlrs probe
(``crates/mlrs-algos/tests/knn_classifier_fit_perf_test.rs``), so the numbers
are directly comparable:

    # mlrs (pick the backend feature for your machine: cpu / wgpu / cuda)
    cargo test -p mlrs-algos --release --features cpu \
        --test knn_classifier_fit_perf_test -- --ignored --nocapture

    # sklearn
    python3 scripts/bench_knn_classifier.py

For k-NN, ``fit`` only stores the training set (there's no model to solve
for) — for sklearn that's near-free; for mlrs it additionally validates the
integer labels host-side and takes device-resident ownership of the training
matrix. This harness isolates the ``fit`` wall-clock so that ingestion path is
the thing being measured, not `predict` (already covered by
``scripts/bench_knn.py``).

Requires numpy + scikit-learn.
"""

from __future__ import annotations

import os
import time

import numpy as np

CONFIGS = [
    (10_000, 16, 5),
    (50_000, 16, 5),
    (100_000, 32, 10),
    (200_000, 32, 10),
]

_MAX_N = int(os.environ.get("KNN_PERF_MAX_N", 0)) or None
if _MAX_N:
    CONFIGS = [c for c in CONFIGS if c[0] <= _MAX_N]


def _splitmix64_block(seed: int, count: int) -> np.ndarray:
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        state = (np.uint64(seed) + idx * np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        z = state
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
        return (z ^ (z >> np.uint64(31))).astype(np.uint64)


def _uniform_pm1(seed: int, count: int) -> np.ndarray:
    u = (_splitmix64_block(seed, count) >> np.uint64(11)) / float(1 << 53)
    return u * 2.0 - 1.0


def make_classification(n: int, d: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Byte-identical to knn_classifier_fit_perf_test.rs::make_classification
    (f32 X, 3-class integer y bucketed by the linear-target sign/margin)."""
    x = _uniform_pm1(seed, n * d).reshape(n, d)
    coef = _uniform_pm1(seed + 1, d)
    dot = x @ coef
    y = np.where(dot < -0.15, 0, np.where(dot > 0.15, 2, 1))
    return x.astype(np.float32), y.astype(np.float32)


def bench_fit(fit_fn, repeats: int = 3) -> float:
    """BEST-OF-N (matches the mlrs probe's methodology)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fit_fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    header = f"{'n_train':>9} {'d':>4} {'k':>4} | {'engine':>8} {'fit (s)':>10}"
    print(header)
    print("-" * len(header))

    for n, d, k in CONFIGS:
        x, y = make_classification(n, d, 42)

        fit_s = bench_fit(
            lambda: SkKNN(n_neighbors=k, algorithm="brute", metric="euclidean").fit(x, y)
        )
        print(f"{n:>9} {d:>4} {k:>4} | {'sklearn':>8} {fit_s:>10.6f}")


if __name__ == "__main__":
    main()
