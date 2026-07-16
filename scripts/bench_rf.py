#!/usr/bin/env python3
"""Random Forest wall-clock comparison harness (ENSEMBLE-01).

Times ``sklearn.ensemble.RandomForestClassifier`` — and ``cuml.ensemble
.RandomForestClassifier`` when importable (CUDA hosts) — on the SAME synthetic
geometry ladder as the mlrs probe
(`crates/mlrs-algos/tests/random_forest_perf_test.rs`), so the three numbers
are directly comparable:

    # mlrs (pick the backend feature for your machine: wgpu / cuda / cpu)
    cargo test -p mlrs-algos --release --features wgpu \
        --test random_forest_perf_test -- --ignored --nocapture

    # sklearn (+ cuML when installed)
    python3 scripts/bench_rf.py

The data uses the same splitmix64 stream as the Rust probe, so all engines fit
the byte-identical dataset. sklearn/cuML hyperparameters are matched to the
mlrs run (trees, depth, sqrt features, bootstrap); remaining differences
(exact vs binned splitter for sklearn; n_bins for cuML) are inherent to each
engine and are what the comparison is about.

Requires numpy + scikit-learn (a /tmp venv is fine, PEP 668); cuML optional.
"""

from __future__ import annotations

import time

import numpy as np

CONFIGS = [
    (10_000, 16, 32, 8),
    (50_000, 16, 32, 8),
    (100_000, 16, 32, 8),
    (50_000, 16, 100, 8),
    (50_000, 16, 32, 12),
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


def make_data(n: int, d: int, seed: int = 42):
    """Byte-identical to random_forest_perf_test.rs::make_data (f32 features).

    The Rust probe draws, per row, `d` feature values then one noise draw —
    a strided view over one flat splitmix64 stream.
    """
    u = (_splitmix64_block(seed, n * (d + 1)) >> np.uint64(11)) / float(1 << 53)
    u = u.reshape(n, d + 1)
    x = u[:, :d].astype(np.float32)
    noise = u[:, d] < 0.05
    a, b = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    label = np.where(a < 0.5, 0, np.where(b < 0.5, 1, 2))
    y = np.where(noise, (label + 1) % 3, label).astype(np.int32)
    return x, y


def bench_engine(name: str, fit_fn, predict_fn) -> tuple[float, float]:
    t0 = time.perf_counter()
    model = fit_fn()
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    predict_fn(model)
    pred_s = time.perf_counter() - t1
    return fit_s, pred_s


def main() -> None:
    from sklearn.ensemble import RandomForestClassifier as SkRF

    try:
        from cuml.ensemble import RandomForestClassifier as CuRF  # type: ignore

        have_cuml = True
    except Exception:
        have_cuml = False

    print(f"cuML available: {have_cuml}")
    header = f"{'n':>8} {'d':>4} {'trees':>6} {'depth':>6} | {'engine':>8} {'fit (s)':>10} {'pred (s)':>10}"
    print(header)
    print("-" * len(header))

    for n, d, trees, depth in CONFIGS:
        x, y = make_data(n, d)

        fit_s, pred_s = bench_engine(
            "sklearn",
            lambda: SkRF(
                n_estimators=trees, max_depth=depth, n_jobs=-1, random_state=0
            ).fit(x, y),
            lambda m: m.predict(x),
        )
        print(f"{n:>8} {d:>4} {trees:>6} {depth:>6} | {'sklearn':>8} {fit_s:>10.3f} {pred_s:>10.3f}")

        if have_cuml:
            fit_s, pred_s = bench_engine(
                "cuml",
                lambda: CuRF(
                    n_estimators=trees, max_depth=depth, n_bins=32, random_state=0
                ).fit(x, y.astype(np.int32)),
                lambda m: m.predict(x),
            )
            print(f"{n:>8} {d:>4} {trees:>6} {depth:>6} | {'cuml':>8} {fit_s:>10.3f} {pred_s:>10.3f}")


if __name__ == "__main__":
    main()
