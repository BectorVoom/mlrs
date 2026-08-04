#!/usr/bin/env python3
"""RidgeClassifier wall-clock comparison harness (RIDGECLF-CUDA).

Times ``sklearn.linear_model.RidgeClassifier`` — plus ``mlrs.RidgeClassifier``
when the extension is importable, and ``cuml.linear_model.RidgeClassifier`` on
a CUDA host — on the SAME class-shifted design the Rust probe uses
(``crates/mlrs-algos/tests/ridge_classifier_cuda_perf_test.rs``), so every
engine fits byte-identical data:

    # mlrs Rust probe (pick the backend feature for your machine)
    cargo test -p mlrs-algos --release --features cuda \
        --test ridge_classifier_cuda_perf_test -- --ignored --nocapture

    # sklearn (+ mlrs / cuML when installed)
    python3 scripts/bench_ridge_classifier.py

The ladder varies ``n_classes`` as well as ``(n, d)``, which the regression
ladders do not, because ``n_classes`` is the axis that decides whether a device
`predict` can win at all: the compute is ``O(m·d·K)`` over an ``O(m·d)``
transfer, so the arithmetic intensity is linear in ``K`` (and the fused classify
kernel's egress is ``K``× smaller than the raw scores). ``fit`` gains from ``K``
too, but less — its Gram is already ``O(n·d²)`` and is formed once regardless.

Query rows are deliberately LARGE (100 000). A sub-millisecond ``predict``
measures fixed overhead rather than the kernel: the LINEAR-07 cpu campaign
concluded a regression from an ``n_query = 1000`` ladder that reversed at a
realistic batch size.

Requires numpy + scikit-learn; mlrs and cuML optional.
"""

from __future__ import annotations

import time

import numpy as np

# (n_samples, n_features, n_classes) — the Rust probe's CONFIGS, verbatim.
CONFIGS = [
    (10_000, 16, 2),
    (10_000, 64, 3),
    (100_000, 16, 3),
    (100_000, 64, 3),
    (100_000, 64, 10),
    (100_000, 64, 26),
    (100_000, 128, 10),
    (100_000, 256, 10),
]

N_QUERY = 100_000
ALPHA = 1.0


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


def make_classification(
    n: int, d: int, k: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Byte-identical to ridge_classifier_cuda_perf_test.rs::make_classification.

    The Rust generator draws ONE splitmix64 stream row-major over ``n * d``, so
    element ``r*d + c`` is draw ``r*d + c + 1`` — reshaping the same flat stream
    reproduces it exactly. Class ``r % k`` shifts feature ``(r % k) % d`` by 1.5.
    """
    x = _uniform_pm1(seed, n * d).reshape(n, d)
    cls = np.arange(n) % k
    shift_col = cls % d
    x[np.arange(n), shift_col] += 1.5
    return x.astype(np.float32), cls.astype(np.float32)


def bench(fit_fn, predict_fn, reps: int = 3):
    """min-of-N on both phases — a shared machine's noise is one-sided."""
    best_fit = float("inf")
    best_pred = float("inf")
    model = None
    for _ in range(reps):
        t0 = time.perf_counter()
        model = fit_fn()
        best_fit = min(best_fit, time.perf_counter() - t0)
        t1 = time.perf_counter()
        predict_fn(model)
        best_pred = min(best_pred, time.perf_counter() - t1)
    return model, best_fit, best_pred


def main() -> None:
    from sklearn.linear_model import RidgeClassifier as SkRC

    try:
        import mlrs  # type: ignore

        have_mlrs = True
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"(mlrs unavailable: {e})")
        have_mlrs = False

    try:
        from cuml.linear_model import RidgeClassifier as CuRC  # type: ignore

        have_cuml = True
    except Exception:
        have_cuml = False

    print(f"mlrs available: {have_mlrs}   cuML available: {have_cuml}")
    header = (
        f"{'n':>9} {'d':>4} {'k':>3} | {'engine':>8} "
        f"{'fit (s)':>10} {'pred (s)':>10} {'agree':>7}"
    )
    print(header)
    print("-" * len(header))

    for n, d, k in CONFIGS:
        x, y = make_classification(n, d, k, seed=42)
        xq, _ = make_classification(N_QUERY, d, k, seed=4242)

        _, fit_s, pred_s = bench(
            lambda: SkRC(alpha=ALPHA, fit_intercept=True).fit(x, y),
            lambda m: m.predict(xq),
        )
        ref = SkRC(alpha=ALPHA, fit_intercept=True).fit(x, y).predict(xq)
        print(f"{n:>9} {d:>4} {k:>3} | {'sklearn':>8} {fit_s:>10.4f} {pred_s:>10.4f} {'-':>7}")

        if have_mlrs:
            model, fit_s, pred_s = bench(
                lambda: mlrs.RidgeClassifier(alpha=ALPHA, fit_intercept=True).fit(x, y),
                lambda m: m.predict(xq),
            )
            # A faster arm that disagrees is not a faster arm. Report the
            # fraction of query rows whose label matches sklearn's rather than
            # asserting: the two engines solve the same convex problem, so a
            # handful of rows can straddle a decision boundary in f32 without
            # either being wrong.
            agree = float(np.mean(np.asarray(model.predict(xq)) == ref))
            print(
                f"{n:>9} {d:>4} {k:>3} | {'mlrs':>8} {fit_s:>10.4f} "
                f"{pred_s:>10.4f} {agree:>7.4f}"
            )

        if have_cuml:
            model, fit_s, pred_s = bench(
                lambda: CuRC(alpha=ALPHA, fit_intercept=True).fit(x, y),
                lambda m: m.predict(xq),
            )
            agree = float(np.mean(np.asarray(model.predict(xq)) == ref))
            print(
                f"{n:>9} {d:>4} {k:>3} | {'cuml':>8} {fit_s:>10.4f} "
                f"{pred_s:>10.4f} {agree:>7.4f}"
            )


if __name__ == "__main__":
    main()
