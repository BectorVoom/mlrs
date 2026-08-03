#!/usr/bin/env python3
"""KNeighborsRegressor wall-clock comparison harness (KNN-01).

Times ``sklearn.neighbors.KNeighborsRegressor(algorithm='brute')`` — and
``cuml.neighbors.KNeighborsRegressor`` when importable (CUDA hosts) — on the
SAME splitmix64 design matrix as the mlrs probe
(``crates/mlrs-algos/tests/knn_regressor_perf_test.rs``), so the numbers are
directly comparable:

    # mlrs (pick the backend feature for your machine: wgpu / cuda / cpu)
    cargo test -p mlrs-algos --release --features cuda \
        --test knn_regressor_perf_test -- --ignored --nocapture

    # sklearn (+ cuML when installed)
    python3 scripts/bench_knn.py

``algorithm='brute'`` is the primary sklearn row because that is the method
mlrs implements (a full pairwise distance matrix + partial selection); comparing
against sklearn's default kd-tree/ball-tree auto-selection would compare two
different algorithms rather than two implementations of one. ``algorithm='auto'``
is reported ALONGSIDE it (``KNN_BENCH_AUTO=0`` drops the row) because that is
what a user actually gets from ``KNeighborsRegressor()``, and on a k-NN
estimator the index build lands in ``fit`` — so it is the only row where the
``fit`` column measures anything but data ingestion.

Note on what ``fit`` means for k-NN: for all three engines it only stores the
training set (there is no model to solve for), so ``fit`` times the data
ingestion path — for mlrs specifically that is where a device→host→device
round-trip of the whole training matrix used to live. The interesting column is
``predict``, which is where the brute-force search actually happens.

Since KNN-REG-FIT, ``mlrs.KNeighborsRegressor.fit`` is a validation pass only:
it checks the training set for NaN/inf and parks the host buffers, and the
device upload happens on the FIRST query. That is what makes its ``fit`` column
comparable to sklearn's (which likewise only validates a reference it keeps)
rather than a copy sklearn never makes. The cost did not vanish — it moved into
the first ``predict``, where it measured ~10% of that call and under 1% of the
warm search. When reading the two columns together, attribute the first
``predict`` accordingly.

Both columns are unreliable on a busy machine: a competing job inflated
wall-clock here by 10-30x at random. Prefer ``time.process_time`` and interleave
the engines within each repetition (min-of-N) when the load average is not close
to zero.

Requires numpy + scikit-learn; cuML optional.
"""

from __future__ import annotations

import time

import numpy as np

# (n_train, d, k, n_query) — the ladder the Rust probe walks.
import os

CONFIGS = [
    (10_000, 16, 5, 2_000),
    (50_000, 16, 5, 10_000),
    (100_000, 32, 10, 10_000),
    (200_000, 32, 10, 20_000),
]

# Mirror the Rust probe's ladder cap so both sides bench the SAME configs when a
# run is capped (``KNN_PERF_MAX_N=50000 python3 scripts/bench_knn.py``).
_MAX_N = int(os.environ.get("KNN_PERF_MAX_N", 0)) or None
if _MAX_N:
    CONFIGS = [c for c in CONFIGS if c[0] <= _MAX_N]

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


def _uniform_pm1(seed: int, count: int) -> np.ndarray:
    u = (_splitmix64_block(seed, count) >> np.uint64(11)) / float(1 << 53)
    return u * 2.0 - 1.0


def make_regression(n: int, d: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Byte-identical to knn_regressor_perf_test.rs::make_regression (f32 X/y;
    the seed/seed+1/seed+2 stream split)."""
    x = _uniform_pm1(seed, n * d).reshape(n, d)
    coef = _uniform_pm1(seed + 1, d)
    noise = _uniform_pm1(seed + 2, n)
    y = x @ coef + 0.5 + 0.01 * noise
    return x.astype(np.float32), y.astype(np.float32)


def bench(fit_fn, predict_fn):
    """Fit once, predict BEST-OF-3 (matching the mlrs probe's methodology —
    a single cold call measures the GPU's idle-clock ramp / library warmup,
    not throughput; the min of three is the standard steady-state figure)."""
    t0 = time.perf_counter()
    model = fit_fn()
    fit_s = time.perf_counter() - t0
    out = None
    pred_s = float("inf")
    for _ in range(3):
        t1 = time.perf_counter()
        out = predict_fn(model)
        pred_s = min(pred_s, time.perf_counter() - t1)
    return out, fit_s, pred_s


def main() -> None:
    from sklearn.neighbors import KNeighborsRegressor as SkKNN

    try:
        from cuml.neighbors import KNeighborsRegressor as CuKNN  # type: ignore

        have_cuml = True
    except Exception:
        have_cuml = False

    want_auto = os.environ.get("KNN_BENCH_AUTO", "1") != "0"

    print(f"cuML available: {have_cuml}")
    header = (
        f"{'n_train':>9} {'d':>4} {'k':>4} {'n_query':>8} | "
        f"{'engine':>8} {'fit (s)':>10} {'pred (s)':>10}"
    )
    print(header)
    print("-" * len(header))

    warmed = False
    for n, d, k, nq in CONFIGS:
        x, y = make_regression(n, d, 42)
        xq, _ = make_regression(nq, d, 7)

        _, fit_s, pred_s = bench(
            lambda: SkKNN(n_neighbors=k, algorithm="brute", metric="euclidean").fit(x, y),
            lambda m: m.predict(xq),
        )
        print(
            f"{n:>9} {d:>4} {k:>4} {nq:>8} | {'sklearn':>8} {fit_s:>10.4f} {pred_s:>10.4f}"
        )

        if want_auto:
            # What `KNeighborsRegressor()` does out of the box: sklearn picks
            # brute / kd_tree / ball_tree itself, and any tree build is charged
            # to `fit`.
            _, fit_s, pred_s = bench(
                lambda: SkKNN(n_neighbors=k, algorithm="auto", metric="euclidean").fit(x, y),
                lambda m: m.predict(xq),
            )
            print(
                f"{n:>9} {d:>4} {k:>4} {nq:>8} | {'sk-auto':>8} {fit_s:>10.4f} {pred_s:>10.4f}"
            )

        if have_cuml:
            if not warmed:
                # JIT/context warmup so the first timed config is steady-state.
                CuKNN(n_neighbors=k).fit(x[:2_000], y[:2_000]).predict(xq[:200])
                warmed = True
            _, fit_s, pred_s = bench(
                lambda: CuKNN(n_neighbors=k).fit(x, y),
                lambda m: m.predict(xq),
            )
            print(
                f"{n:>9} {d:>4} {k:>4} {nq:>8} | {'cuml':>8} {fit_s:>10.4f} {pred_s:>10.4f}"
            )


if __name__ == "__main__":
    main()
