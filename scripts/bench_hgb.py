#!/usr/bin/env python3
"""HistGradientBoosting wall-clock comparison harness (GBT-01).

Times ``sklearn.ensemble.HistGradientBoostingClassifier`` on the SAME
synthetic geometry ladder as the mlrs probe
(`crates/mlrs-algos/tests/hist_gradient_boosting_perf_test.rs`), so the two
numbers are directly comparable:

    # mlrs (pick the backend feature for your machine: wgpu / cuda / cpu)
    cargo test -p mlrs-algos --release --features wgpu \
        --test hist_gradient_boosting_perf_test -- --ignored --nocapture

    # sklearn
    python3 scripts/bench_hgb.py

The data uses the same splitmix64 stream as the Rust probe, so both engines
fit the byte-identical dataset. Hyperparameters are matched to the mlrs run
(max_iter, learning_rate 0.1, min_samples_leaf 20, early_stopping=False);
the leaf-wise-vs-level-wise difference is neutralized by passing
``max_leaf_nodes=None`` + the same ``max_depth``, so both build
identically-shaped trees. Remaining engine-inherent difference: sklearn bins
to its fixed 255 while the mlrs default is ``n_bins=64`` (the documented
histogram-lattice deviation) — both learn this rule equally well. This is
sklearn's OpenMP home turf.

Requires numpy + scikit-learn (a /tmp venv is fine, PEP 668).
"""

from __future__ import annotations

import time

import numpy as np

CONFIGS = [
    (10_000, 16, 100, 6),
    (50_000, 16, 100, 6),
    (100_000, 16, 100, 6),
    (50_000, 16, 200, 6),
    (50_000, 16, 100, 8),
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
    """Byte-identical to hist_gradient_boosting_perf_test.rs::make_data."""
    u = (_splitmix64_block(seed, n * (d + 1)) >> np.uint64(11)) / float(1 << 53)
    u = u.reshape(n, d + 1)
    x = u[:, :d].astype(np.float32)
    noise = u[:, d] < 0.05
    a, b = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    label = np.where(a < 0.5, 0, np.where(b < 0.5, 1, 2))
    y = np.where(noise, (label + 1) % 3, label).astype(np.int32)
    return x, y


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    print(f"{'n':>8} {'d':>4} {'iters':>6} {'depth':>6} | {'engine':>8} "
          f"{'fit (s)':>10} {'pred (s)':>10}")
    for n, d, max_iter, depth in CONFIGS:
        x, y = make_data(n, d)
        clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=0.1,
            max_depth=depth,
            max_leaf_nodes=None,
            min_samples_leaf=20,
            max_bins=255,
            early_stopping=False,
            random_state=0,
        )
        t0 = time.perf_counter()
        clf.fit(x, y)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = clf.predict(x)
        pred_s = time.perf_counter() - t1
        acc = float((pred == y).mean())
        assert acc > 0.9, f"sklearn train accuracy {acc} too low — bench is broken"
        print(f"{n:>8} {d:>4} {max_iter:>6} {depth:>6} | {'sklearn':>8} "
              f"{fit_s:>10.3f} {pred_s:>10.3f}")


if __name__ == "__main__":
    main()
