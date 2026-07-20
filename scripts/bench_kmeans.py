#!/usr/bin/env python3
"""KMeans wall-clock comparison harness (CLUSTER-01).

Times ``sklearn.cluster.KMeans`` — and ``cuml.cluster.KMeans`` when importable
(CUDA hosts) — on the SAME splitmix64 blob ladder and the SAME injected init
centers as the mlrs probe (`crates/mlrs-algos/tests/kmeans_perf_test.rs`), so
the three numbers are directly comparable:

    # mlrs (pick the backend feature for your machine: wgpu / cuda / cpu)
    cargo test -p mlrs-algos --release --features wgpu \
        --test kmeans_perf_test -- --ignored --nocapture

    # sklearn (+ cuML when installed)
    python3 scripts/bench_kmeans.py

Every engine runs Lloyd from IDENTICAL starting centers (init array injected,
n_init=1) on the byte-identical f32 dataset with matched max_iter=300 and
tol=1e-4, so per-iteration kernel speed — not init strategy — is what the
comparison measures. Inertia is printed as the quality cross-check: all
engines should land on (nearly) the same local optimum.

Requires numpy + scikit-learn; cuML optional.
"""

from __future__ import annotations

import time

import numpy as np

CONFIGS = [
    (100_000, 16, 8),
    (100_000, 64, 32),
    (500_000, 16, 8),
    (500_000, 32, 32),
    (1_000_000, 16, 8),
]

MASK = (1 << 64) - 1


def _splitmix64_block(seed: int, count: int) -> np.ndarray:
    """splitmix64 is counter-based, so the whole stream vectorizes exactly."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        state = (np.uint64(seed) + idx * np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        z = state
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
        return (z ^ (z >> np.uint64(31))).astype(np.uint64)


def _uniform01(seed: int, count: int) -> np.ndarray:
    return (_splitmix64_block(seed, count) >> np.uint64(11)) / float(1 << 53)


def make_blobs(n: int, d: int, k: int, seed: int = 42) -> np.ndarray:
    """Byte-identical to kmeans_perf_test.rs::make_blobs (f32 features)."""
    centers = (_uniform01(seed + 1, k * d) * 10.0).reshape(k, d)
    noise = (_uniform01(seed, n * d) - 0.5) * 2.0
    labels = np.arange(n, dtype=np.int64) % k
    x = centers[labels] + noise.reshape(n, d)
    return x.astype(np.float32)


def init_indices(n: int, k: int, seed: int = 42) -> list[int]:
    """Byte-identical to kmeans_perf_test.rs::init_indices (seed+2 stream,
    sequential rejection on duplicates)."""
    idx: list[int] = []
    state = (seed + 2) & MASK
    # Draw one value at a time from the counter-based stream (cheap: k is tiny).
    counter = 0
    while len(idx) < k:
        counter += 1
        v = int(_splitmix64_block(state, counter)[-1]) % n
        if v not in idx:
            idx.append(v)
    return idx


def bench(fit_fn, predict_fn):
    t0 = time.perf_counter()
    model = fit_fn()
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    predict_fn(model)
    pred_s = time.perf_counter() - t1
    return model, fit_s, pred_s


def main() -> None:
    from sklearn.cluster import KMeans as SkKMeans

    try:
        from cuml.cluster import KMeans as CuKMeans  # type: ignore

        have_cuml = True
    except Exception:
        have_cuml = False

    print(f"cuML available: {have_cuml}")
    header = (
        f"{'n':>9} {'d':>4} {'k':>4} | {'engine':>8} {'fit (s)':>10} "
        f"{'pred (s)':>10} {'inertia':>14} {'iters':>6}"
    )
    print(header)
    print("-" * len(header))

    warmed = False
    for n, d, k in CONFIGS:
        x = make_blobs(n, d, k)
        init = x[init_indices(n, k)].copy()

        model, fit_s, pred_s = bench(
            lambda: SkKMeans(
                n_clusters=k, init=init, n_init=1, max_iter=300, tol=1e-4,
                algorithm="lloyd",
            ).fit(x),
            lambda m: m.predict(x),
        )
        print(
            f"{n:>9} {d:>4} {k:>4} | {'sklearn':>8} {fit_s:>10.4f} "
            f"{pred_s:>10.4f} {model.inertia_:>14.6e} {model.n_iter_:>6}"
        )

        if have_cuml:
            if not warmed:
                # JIT/context warmup so the first timed config is steady-state.
                CuKMeans(n_clusters=k, init=init, max_iter=5).fit(x[:10_000])
                warmed = True
            model, fit_s, pred_s = bench(
                lambda: CuKMeans(
                    n_clusters=k, init=init, n_init=1, max_iter=300, tol=1e-4
                ).fit(x),
                lambda m: m.predict(x),
            )
            print(
                f"{n:>9} {d:>4} {k:>4} | {'cuml':>8} {fit_s:>10.4f} "
                f"{pred_s:>10.4f} {model.inertia_:>14.6e} {model.n_iter_:>6}"
            )


if __name__ == "__main__":
    main()
