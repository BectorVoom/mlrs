#!/usr/bin/env python3
"""Ridge wall-clock comparison harness (LINEAR-02).

Times ``sklearn.linear_model.Ridge(solver='cholesky')`` — and
``cuml.linear_model.Ridge`` when importable (CUDA hosts) — on the SAME
splitmix64 design matrix as the mlrs probe
(``crates/mlrs-algos/tests/ridge_perf_test.rs``), so the numbers are directly
comparable:

    # mlrs (pick the backend feature for your machine: wgpu / cuda / cpu)
    cargo test -p mlrs-algos --release --features cuda \
        --test ridge_perf_test -- --ignored --nocapture

    # sklearn (+ cuML when installed)
    python3 scripts/bench_ridge.py

Ridge has no dual-path dispatch (D-02: always the Cholesky normal-equations
solver, no direct-SVD/Gram+eig split like LinearRegression), so this ladder
reuses the same shapes as ``bench_linear.py`` for a direct comparison.

Requires numpy + scikit-learn; cuML optional.
"""

from __future__ import annotations

import time

import numpy as np

CONFIGS = [
    (200, 16),
    (10_000, 16),
    (10_000, 64),
    (100_000, 16),
    (100_000, 64),
    (500_000, 16),
    (1_000_000, 16),
]

ALPHA = 1.0

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
    """Byte-identical to ridge_perf_test.rs::make_regression (f32 X/y; the
    seed/seed+1/seed+2 stream split)."""
    x = _uniform_pm1(seed, n * d).reshape(n, d)
    coef = _uniform_pm1(seed + 1, d)
    noise = _uniform_pm1(seed + 2, n)
    y = x @ coef + 0.5 + 0.01 * noise
    return x.astype(np.float32), y.astype(np.float32)


def bench(fit_fn, predict_fn):
    t0 = time.perf_counter()
    model = fit_fn()
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    predict_fn(model)
    pred_s = time.perf_counter() - t1
    return model, fit_s, pred_s


def main() -> None:
    from sklearn.linear_model import Ridge as SkRidge

    try:
        from cuml.linear_model import Ridge as CuRidge  # type: ignore

        have_cuml = True
    except Exception:
        have_cuml = False

    print(f"cuML available: {have_cuml}")
    header = (
        f"{'n':>9} {'d':>4} | {'engine':>8} {'fit (s)':>10} {'pred (s)':>10}"
    )
    print(header)
    print("-" * len(header))

    warmed = False
    for n, d in CONFIGS:
        x, y = make_regression(n, d)

        model, fit_s, pred_s = bench(
            lambda: SkRidge(alpha=ALPHA, solver="cholesky", fit_intercept=True).fit(x, y),
            lambda m: m.predict(x),
        )
        print(f"{n:>9} {d:>4} | {'sklearn':>8} {fit_s:>10.4f} {pred_s:>10.4f}")

        if have_cuml:
            if not warmed:
                # JIT/context warmup so the first timed config is steady-state.
                CuRidge(alpha=ALPHA, solver="eig").fit(x[:10_000], y[:10_000])
                warmed = True
            model, fit_s, pred_s = bench(
                lambda: CuRidge(alpha=ALPHA, solver="eig").fit(x, y),
                lambda m: m.predict(x),
            )
            print(f"{n:>9} {d:>4} | {'cuml':>8} {fit_s:>10.4f} {pred_s:>10.4f}")


if __name__ == "__main__":
    main()
