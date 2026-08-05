#!/usr/bin/env python
"""mlrs vs scikit-learn `TSNE` — wall-clock sweeps over the parameters that
actually move the clock (TSNE-PARAMS).

Which parameters are perf-significant, and why each one is here:

* ``method``   — the whole asymptotic class. ``'barnes_hut'`` is ``O(n log n)``
                 over a sparse k-NN ``P``; ``'exact'`` is ``O(n²)`` over a dense
                 one. Nothing else in the surface changes the complexity.
* ``angle``    — how much of the quadtree the negative force walks. θ is the
                 single dial on the dominant per-iteration term, and the one
                 that exposed the traversal as this engine's critical path.
* ``perplexity`` — sets ``n_neighbors = int(3·perplexity + 1)``, so it sizes
                 BOTH the k-NN graph built once and the positive-force edge loop
                 run every iteration.
* ``n_components`` — 2 vs 3 switches the quadtree to an octree: 8 children per
                 cell instead of 4, and a deeper walk.
* ``max_iter`` — the iteration count, and the lever that separates one-time
                 setup cost from per-iteration cost.
* ``n_jobs``   — the worker count. Value-neutral by construction here (every
                 reduction runs in point order), so it is pure wall clock.
* ``metric``   — decides whether the neighbour graph can be built by KD-tree
                 pruning (axis-separable metrics) or must be scanned.

Deliberately absent: ``init``, ``verbose``, ``min_grad_norm``,
``n_iter_without_progress``, ``early_exaggeration``, ``learning_rate``. The
first two cost nothing measurable; the rest change only WHEN the descent stops,
so timing them measures the stopping rule rather than the implementation, and
they are gated for correctness instead.

Run each sweep in its OWN process on a quiet machine — in-process interleaving
and a loaded box have each inverted an mlrs-vs-sklearn verdict before
([[mlrs-cpu-bench-separate-processes]]).

    python scripts/bench_tsne_params.py --sweep method
    python scripts/bench_tsne_params.py --sweep all --reps 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def make_blobs(n: int, d: int, k: int, seed: int) -> np.ndarray:
    """Well-separated Gaussian blobs — the shape t-SNE is actually used on, and
    the same generator the Rust perf probe uses so the two are comparable."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-15.0, 15.0, size=(k, d))
    x = np.empty((n, d))
    for i in range(n):
        x[i] = centers[i % k] + 0.7 * rng.standard_normal(d)
    return x


def best_of(fn, reps: int) -> float:
    """Min-of-N wall clock. Min, not mean: the distribution's lower tail is the
    machine doing our work and nothing else, which is what we want to compare."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _row(label: str, tm: float, ts: float) -> None:
    ratio = ts / tm if tm > 0 else float("inf")
    verdict = "mlrs" if ratio >= 1.0 else "SKLEARN"
    print(f"  {label:<26} mlrs {tm:8.3f}s   sklearn {ts:8.3f}s   {ratio:6.2f}x  {verdict}")


def run_pair(mlrs_kw: dict, sk_kw: dict, x: np.ndarray, reps: int) -> tuple[float, float]:
    import mlrs
    from sklearn.manifold import TSNE as SkTSNE

    tm = best_of(lambda: mlrs.TSNE(**mlrs_kw).fit(x), reps)
    ts = best_of(lambda: SkTSNE(**sk_kw).fit(x), reps)
    return tm, ts


def sweep_method(reps: int) -> None:
    print("\n== method (the asymptotic class) ==")
    for n in (500, 1000, 2000):
        x = make_blobs(n, 8, 5, 42)
        for method in ("barnes_hut", "exact"):
            kw = dict(method=method, perplexity=30.0, init="pca", random_state=0)
            tm, ts = run_pair(kw, kw, x, reps)
            _row(f"n={n} method={method}", tm, ts)


def sweep_angle(reps: int) -> None:
    print("\n== angle (quadtree summary threshold; lower = more traversal) ==")
    x = make_blobs(3000, 8, 5, 42)
    for angle in (0.2, 0.5, 0.8, 1.0):
        kw = dict(angle=angle, perplexity=30.0, init="pca", random_state=0)
        tm, ts = run_pair(kw, kw, x, reps)
        _row(f"n=3000 angle={angle}", tm, ts)


def sweep_perplexity(reps: int) -> None:
    print("\n== perplexity (sizes the k-NN graph AND the edge loop) ==")
    x = make_blobs(3000, 8, 5, 42)
    for perp in (5.0, 15.0, 30.0, 50.0):
        kw = dict(perplexity=perp, init="pca", random_state=0)
        tm, ts = run_pair(kw, kw, x, reps)
        _row(f"n=3000 perplexity={perp:g} (k={int(3 * perp + 1)})", tm, ts)


def sweep_n_components(reps: int) -> None:
    print("\n== n_components (quadtree -> octree at 3) ==")
    x = make_blobs(2000, 8, 5, 42)
    for nc in (2, 3):
        kw = dict(n_components=nc, perplexity=30.0, init="pca", random_state=0)
        tm, ts = run_pair(kw, kw, x, reps)
        _row(f"n=2000 n_components={nc}", tm, ts)


def sweep_max_iter(reps: int) -> None:
    print("\n== max_iter (separates one-time setup from per-iteration cost) ==")
    x = make_blobs(2000, 8, 5, 42)
    times = {}
    for it in (250, 500, 1000):
        kw = dict(max_iter=it, perplexity=30.0, init="pca", random_state=0)
        tm, ts = run_pair(kw, kw, x, reps)
        times[it] = (tm, ts)
        _row(f"n=2000 max_iter={it}", tm, ts)
    # Two points on a straight line give the slope (per-iteration) and the
    # intercept (setup), which is what says WHERE a win comes from.
    (m250, s250), (m1000, s1000) = times[250], times[1000]
    per_m = (m1000 - m250) / 750.0
    per_s = (s1000 - s250) / 750.0
    _row("  -> per-iteration (ms)", per_m * 1e3, per_s * 1e3)
    _row("  -> setup (s)", m250 - per_m * 250, s250 - per_s * 250)


def sweep_n_jobs(reps: int) -> None:
    print("\n== n_jobs (pure wall clock — value-neutral by construction) ==")
    import mlrs

    x = make_blobs(3000, 8, 5, 42)
    for n_jobs in (1, 2, 4, 8, None):
        kw = dict(n_jobs=n_jobs, perplexity=30.0, init="pca", random_state=0)
        tm = best_of(lambda: mlrs.TSNE(**kw).fit(x), reps)
        print(f"  n=3000 n_jobs={str(n_jobs):<5} mlrs {tm:8.3f}s")


def sweep_metric(reps: int) -> None:
    print("\n== metric (KD-tree-prunable vs scanned neighbour graph) ==")
    x = make_blobs(2000, 8, 5, 42)
    for metric in ("euclidean", "manhattan", "chebyshev", "cosine", "correlation"):
        kw = dict(metric=metric, perplexity=30.0, init="pca", random_state=0)
        tm, ts = run_pair(kw, kw, x, reps)
        _row(f"n=2000 metric={metric}", tm, ts)


SWEEPS = {
    "method": sweep_method,
    "angle": sweep_angle,
    "perplexity": sweep_perplexity,
    "n_components": sweep_n_components,
    "max_iter": sweep_max_iter,
    "n_jobs": sweep_n_jobs,
    "metric": sweep_metric,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="all", choices=[*SWEEPS, "all"])
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    import sklearn

    import mlrs

    print(f"mlrs f64={mlrs.backend_supports_f64()}  sklearn={sklearn.__version__}")
    names = list(SWEEPS) if args.sweep == "all" else [args.sweep]
    for name in names:
        SWEEPS[name](args.reps)


if __name__ == "__main__":
    main()
