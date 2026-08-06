#!/usr/bin/env python3
"""Per-PARAMETER `NearestNeighbors` comparison against scikit-learn (NEIGH-PARAMS).

Mirrors ``bench_knn_classifier_params.py``'s methodology, with the ``weights``/
``n_classes``/``proba`` sweeps dropped (``NearestNeighbors`` has no vote to
weight or classes to predict) and a ``radius`` sweep added for
``radius_neighbors``.

Only the parameters that can move the wall clock get a sweep:

``metric``      five different inner loops (``minkowski`` additionally splits on
                whether ``p`` is an integer).
``n_neighbors`` the selection width for ``kneighbors``. Past the device
                kernel's slot cap the search falls into a slower composition
                (see KNN-HOST); this sweep is the regression gate for that.
``p``           only for ``metric='minkowski'`` — see ``metric``.
``algorithm``   mlrs runs brute force for every value, so its own row is flat by
                construction. Swept anyway because sklearn's is not: its
                ``'auto'`` picks a k-d tree on low-dimensional data, and that —
                not its brute-force path — is what a user actually gets from
                ``NearestNeighbors()``.
``n_jobs``      sklearn's joblib fan-out over query chunks. mlrs always uses the
                whole machine, so ``n_jobs=-1`` is the sklearn row that makes the
                comparison core-for-core rather than 16-against-1.
``radius``      ``radius_neighbors``' selection threshold. Unlike ``kneighbors``,
                the match COUNT is data-dependent, so this also prices the
                host-side ragged-compaction pass (`radius_neighbor_indices_metric`
                in `crates/mlrs-algos/src/neighbors/nearest.rs`) at different
                output densities.

``leaf_size`` gets no sweep: it is read only on a tree route, which mlrs never
takes, and sklearn's own sensitivity to it is a property of its trees.

    python3 scripts/bench_nearest_neighbors_params.py [sweep ...]

with ``sweep`` one of ``metric`` / ``k`` / ``p`` / ``algorithm`` / ``n_jobs`` /
``radius`` / ``fit`` (default: all of them).
``NN_PARAM_N`` / ``NN_PARAM_D`` / ``NN_PARAM_NQ`` override the design size.

## Methodology
Each cell is the MINIMUM of ``NN_PARAM_REPS`` (default 5) alternating
mlrs/sklearn calls, taken after one untimed warm-up call per engine (mlrs defers
its device upload to the first query, and sklearn's first call pays its own
import-time lazy work). Interleaving is what makes the pair comparable on a
machine that is not idle — see `bench_knn_classifier_params.py`'s identical
warning: a co-tenant job can INVERT a verdict on a busy box, so check
``/proc/loadavg`` and re-run marginal (<~1.3x) rungs in separate processes.

Timing uses ``time.process_time()`` (this PROCESS's own CPU time), not
wall-clock, by default — see ``mlrs-hgb-cpu-bench-caveat`` /
``mlrs-cpu-bench-separate-processes``: on a co-tenant-loaded box, wall-clock
``perf_counter`` measures time spent WAITING for a CPU slot, not work done, and
has inverted a verdict here before. CPU time is immune to that as long as the
scheduler gives this process any time slice at all. Set ``NN_PARAM_WALLCLOCK=1``
to compare against wall-clock instead (only meaningful on an otherwise-idle
box).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

N = int(os.environ.get("NN_PARAM_N", 50_000))
D = int(os.environ.get("NN_PARAM_D", 16))
NQ = int(os.environ.get("NN_PARAM_NQ", 5_000))
REPS = int(os.environ.get("NN_PARAM_REPS", 5))
_CLOCK = time.perf_counter if os.environ.get("NN_PARAM_WALLCLOCK") else time.process_time

# Every metric the device serves, as the kwargs BOTH libraries take.
METRICS = [
    ("euclidean", {"metric": "euclidean"}),
    ("manhattan", {"metric": "manhattan"}),
    ("chebyshev", {"metric": "chebyshev"}),
    ("cosine", {"metric": "cosine"}),
    ("minkowski p=3", {"metric": "minkowski", "p": 3.0}),
]

K_SWEEP = [1, 5, 15, 20, 50, 100]
P_SWEEP = [1.5, 2.5, 3.0, 4.0, 6.0]
ALGORITHMS = ["auto", "brute", "kd_tree", "ball_tree"]
N_JOBS = [None, 1, -1]
# Fractions of the pairwise-distance range (see `_radius_for_density`) so the
# sweep covers sparse -> dense match sets rather than one arbitrary threshold.
RADIUS_DENSITIES = [0.01, 0.05, 0.15, 0.35, 0.6]


def make_data(seed: int = 42):
    """A shifted design (so the cosine rows are well conditioned)."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((N, D)) + 3.0).astype(np.float32)
    xq = (rng.standard_normal((NQ, D)) + 3.0).astype(np.float32)
    return x, xq


def _radius_for_density(x, density: float, sample: int = 400, seed: int = 0) -> float:
    """A euclidean radius landing near the `density` quantile of pairwise
    distances, estimated from a SUBSAMPLE (an n=50_000 all-pairs distance
    matrix is not worth computing just to pick a threshold)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=min(sample, x.shape[0]), replace=False)
    sub = x[idx]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    return float(np.quantile(d, density))


def interleaved(call_a, call_b, reps: int = REPS) -> tuple[float, float]:
    """Best-of-`reps` for two calls, ALTERNATING so both see the same machine."""
    ta = tb = float("inf")
    for _ in range(reps):
        t0 = _CLOCK()
        call_a()
        ta = min(ta, _CLOCK() - t0)
        t0 = _CLOCK()
        call_b()
        tb = min(tb, _CLOCK() - t0)
    return ta, tb


def row(label: str, tm: float, ts: float) -> None:
    verdict = "WIN " if ts > tm else "LOSS"
    print(f"  {label:<22} mlrs {tm:>9.4f}s  sklearn {ts:>9.4f}s  {ts / tm:>7.2f}x  {verdict}")


def bench_call(label, mlrs_kwargs, sk_kwargs, data, method="kneighbors", call_kwargs=None) -> None:
    """One cell: fit both engines, then time `method` interleaved."""
    import mlrs
    from sklearn.neighbors import NearestNeighbors as Sk

    x, xq = data
    call_kwargs = call_kwargs or {}
    m = mlrs.NearestNeighbors(**mlrs_kwargs).fit(x)
    s = Sk(**sk_kwargs).fit(x)
    mf, sf = getattr(m, method), getattr(s, method)
    mf(xq[:32], **call_kwargs)  # materialize the deferred upload
    sf(xq[:32], **call_kwargs)
    tm, ts = interleaved(
        lambda: mf(xq, **call_kwargs), lambda: sf(xq, **call_kwargs)
    )
    row(label, tm, ts)


def sweep_metric(data) -> None:
    print(f"metric sweep (n={N} d={D} nq={NQ}, k=5)")
    for name, kw in METRICS:
        bench_call(
            name,
            {"n_neighbors": 5, **kw},
            {"n_neighbors": 5, "algorithm": "brute", **kw},
            data,
        )


def sweep_k(data) -> None:
    print(f"n_neighbors sweep (n={N} d={D} nq={NQ}, metric='euclidean')")
    for k in K_SWEEP:
        bench_call(
            f"k={k}",
            {"n_neighbors": k},
            {"n_neighbors": k, "algorithm": "brute"},
            data,
        )


def sweep_p(data) -> None:
    print(f"p sweep (n={N} d={D} nq={NQ}, k=5, metric='minkowski')")
    for p in P_SWEEP:
        bench_call(
            f"p={p}",
            {"n_neighbors": 5, "metric": "minkowski", "p": p},
            {"n_neighbors": 5, "metric": "minkowski", "p": p, "algorithm": "brute"},
            data,
        )


def sweep_algorithm(data) -> None:
    print(f"algorithm sweep (n={N} d={D} nq={NQ}, k=5) — mlrs is brute for every value")
    for a in ALGORITHMS:
        bench_call(
            a,
            {"n_neighbors": 5, "algorithm": a},
            {"n_neighbors": 5, "algorithm": a},
            data,
        )


def sweep_n_jobs(data) -> None:
    print(f"n_jobs sweep (n={N} d={D} nq={NQ}, k=5) — sklearn's fan-out, mlrs always full")
    for j in N_JOBS:
        bench_call(
            f"n_jobs={j}",
            {"n_neighbors": 5},
            {"n_neighbors": 5, "algorithm": "brute", "n_jobs": j},
            data,
        )


def sweep_radius(data) -> None:
    """`radius_neighbors` at increasing match DENSITY.

    Unlike every other sweep here, the per-query OUTPUT size is not fixed by a
    constructor argument — it is a property of the data and the threshold — so
    this is the one cell that prices the ragged host-compaction pass at
    different loads, from "almost nobody matches" to "most of the set matches".
    """
    x, _ = data
    print(f"radius sweep (n={N} d={D} nq={NQ}, metric='euclidean') — match density varies")
    for density in RADIUS_DENSITIES:
        r = _radius_for_density(x, density)
        bench_call(
            f"density~{density:.0%}",
            {},
            {"algorithm": "brute"},
            data,
            method="radius_neighbors",
            call_kwargs={"radius": r},
        )


def sweep_fit(data) -> None:
    """`fit` is a validation pass for both engines — see `bench_knn.py`."""
    import mlrs
    from sklearn.neighbors import NearestNeighbors as Sk

    x, _ = data
    print(f"fit (n={N} d={D})")
    for name, kw in METRICS:
        tm, ts = interleaved(
            lambda kw=kw: mlrs.NearestNeighbors(n_neighbors=5, **kw).fit(x),
            lambda kw=kw: Sk(n_neighbors=5, algorithm="brute", **kw).fit(x),
        )
        row(name, tm, ts)


SWEEPS = {
    "metric": sweep_metric,
    "k": sweep_k,
    "p": sweep_p,
    "algorithm": sweep_algorithm,
    "n_jobs": sweep_n_jobs,
    "radius": sweep_radius,
    "fit": sweep_fit,
}


def main() -> None:
    wanted = sys.argv[1:] or list(SWEEPS)
    unknown = [w for w in wanted if w not in SWEEPS]
    if unknown:
        raise SystemExit(f"unknown sweep(s) {unknown}; pick from {list(SWEEPS)}")
    with open("/proc/loadavg") as f:
        print(f"loadavg: {f.read().strip()}")
    print(f"reps: {REPS} (best of, interleaved)\n")
    data = make_data()
    for w in wanted:
        SWEEPS[w](data)
        print()


if __name__ == "__main__":
    main()
