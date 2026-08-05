#!/usr/bin/env python3
"""Per-PARAMETER `KNeighborsRegressor` comparison against scikit-learn (KNN-HOST).

``scripts/bench_knn.py`` benches the DEFAULT configuration
(``metric='euclidean'``, ``weights='uniform'``, ``k`` from a fixed ladder). This
harness benches the PARAMETER SURFACE: it walks the values of one parameter at a
time with everything else at its sklearn default, so a regression is attributable
to the parameter that caused it rather than to "k-NN got slower".

Only the parameters that can move the wall clock get a sweep:

``metric``      five different inner loops (``minkowski`` additionally splits on
                whether ``p`` is an integer — the repeated-multiplication lane
                loop against ``powf``).
``n_neighbors`` the selection width. It used to be the sharpest cliff in the
                whole estimator: past the device kernel's 16-slot list the search
                fell into the GPU-shaped composition. That is what KNN-HOST
                removed, and this sweep is the regression gate for it.
``p``           only for ``metric='minkowski'`` — see ``metric``.
``algorithm``   mlrs runs brute force for every value, so its own row is flat by
                construction. It is swept anyway because SKLEARN's is not: its
                ``'auto'`` picks a k-d tree on low-dimensional data, and that —
                not its brute-force path — is what a user actually gets from
                ``KNeighborsRegressor()``. Beating ``algorithm='brute'`` while
                losing to the default would not be beating sklearn.
``n_jobs``      sklearn's joblib fan-out over query chunks. mlrs always uses the
                whole machine, so ``n_jobs=-1`` is the sklearn row that makes the
                comparison core-for-core rather than 16-against-1.

``leaf_size`` gets no sweep: it is read only on a tree route, which mlrs never
takes, and sklearn's own sensitivity to it is a property of its trees.

    python3 scripts/bench_knn_regressor_params.py [sweep ...]

with ``sweep`` one of ``metric`` / ``k`` / ``p`` / ``algorithm`` / ``n_jobs`` /
``fit`` (default: all of them). ``KNN_PARAM_N`` / ``KNN_PARAM_D`` /
``KNN_PARAM_NQ`` override the design size.

## Methodology
Each cell is the MINIMUM of ``KNN_PARAM_REPS`` (default 5) alternating
mlrs/sklearn calls, taken after one untimed warm-up call per engine (mlrs defers
its device upload to the first query, and sklearn's first call pays its own
import-time lazy work). Interleaving is what makes the pair comparable on a
machine that is not idle: a competing job that inflates one engine's median
inflates the other's too, where two separately-timed blocks would attribute the
whole difference to whichever ran during the spike.

Read the load average before trusting anything here. Past sessions in this repo
have had a single co-tenant job INVERT an mlrs-vs-sklearn verdict; if
``/proc/loadavg`` is not close to zero, the numbers are noise.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

N = int(os.environ.get("KNN_PARAM_N", 50_000))
D = int(os.environ.get("KNN_PARAM_D", 16))
NQ = int(os.environ.get("KNN_PARAM_NQ", 5_000))
REPS = int(os.environ.get("KNN_PARAM_REPS", 5))

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


def make_data(seed: int = 42):
    """A shifted design (so the cosine rows are well conditioned) + linear target."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((N, D)) + 3.0).astype(np.float32)
    y = (x @ rng.standard_normal(D).astype(np.float32)).astype(np.float32)
    xq = (rng.standard_normal((NQ, D)) + 3.0).astype(np.float32)
    return x, y, xq


def interleaved(call_a, call_b, reps: int = REPS) -> tuple[float, float]:
    """Best-of-`reps` for two calls, ALTERNATING so both see the same machine."""
    ta = tb = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        call_a()
        ta = min(ta, time.perf_counter() - t0)
        t0 = time.perf_counter()
        call_b()
        tb = min(tb, time.perf_counter() - t0)
    return ta, tb


def row(label: str, tm: float, ts: float) -> None:
    verdict = "WIN " if ts > tm else "LOSS"
    print(f"  {label:<22} mlrs {tm:>9.4f}s  sklearn {ts:>9.4f}s  {ts / tm:>7.2f}x  {verdict}")


def bench_predict(label, mlrs_kwargs, sk_kwargs, data) -> None:
    """One cell: fit both engines, then time `predict` interleaved."""
    import mlrs
    from sklearn.neighbors import KNeighborsRegressor as Sk

    x, y, xq = data
    m = mlrs.KNeighborsRegressor(**mlrs_kwargs).fit(x, y)
    s = Sk(**sk_kwargs).fit(x, y)
    m.predict(xq[:32])  # materialize the deferred upload
    s.predict(xq[:32])
    tm, ts = interleaved(lambda: m.predict(xq), lambda: s.predict(xq))
    row(label, tm, ts)


def sweep_metric(data) -> None:
    print(f"metric sweep (n={N} d={D} nq={NQ}, k=5, weights='uniform')")
    for name, kw in METRICS:
        bench_predict(
            name,
            {"n_neighbors": 5, **kw},
            {"n_neighbors": 5, "algorithm": "brute", **kw},
            data,
        )
    print(f"\nweights sweep (n={N} d={D} nq={NQ}, k=5, metric='euclidean')")
    for w in ("uniform", "distance"):
        bench_predict(
            w,
            {"n_neighbors": 5, "weights": w, "metric": "euclidean"},
            {"n_neighbors": 5, "weights": w, "algorithm": "brute", "metric": "euclidean"},
            data,
        )


def sweep_k(data) -> None:
    print(f"n_neighbors sweep (n={N} d={D} nq={NQ}, metric='euclidean')")
    for k in K_SWEEP:
        bench_predict(
            f"k={k}",
            {"n_neighbors": k},
            {"n_neighbors": k, "algorithm": "brute"},
            data,
        )


def sweep_p(data) -> None:
    print(f"p sweep (n={N} d={D} nq={NQ}, k=5, metric='minkowski')")
    for p in P_SWEEP:
        bench_predict(
            f"p={p}",
            {"n_neighbors": 5, "metric": "minkowski", "p": p},
            {"n_neighbors": 5, "metric": "minkowski", "p": p, "algorithm": "brute"},
            data,
        )


def sweep_algorithm(data) -> None:
    print(f"algorithm sweep (n={N} d={D} nq={NQ}, k=5) — mlrs is brute for every value")
    for a in ALGORITHMS:
        bench_predict(
            a,
            {"n_neighbors": 5, "algorithm": a},
            {"n_neighbors": 5, "algorithm": a},
            data,
        )


def sweep_n_jobs(data) -> None:
    print(f"n_jobs sweep (n={N} d={D} nq={NQ}, k=5) — sklearn's fan-out, mlrs always full")
    for j in N_JOBS:
        bench_predict(
            f"n_jobs={j}",
            {"n_neighbors": 5},
            {"n_neighbors": 5, "algorithm": "brute", "n_jobs": j},
            data,
        )


def sweep_fit(data) -> None:
    """`fit` is a validation pass for both engines — see `bench_knn.py`."""
    import mlrs
    from sklearn.neighbors import KNeighborsRegressor as Sk

    x, y, _ = data
    print(f"fit (n={N} d={D})")
    for name, kw in METRICS:
        tm, ts = interleaved(
            lambda kw=kw: mlrs.KNeighborsRegressor(n_neighbors=5, **kw).fit(x, y),
            lambda kw=kw: Sk(n_neighbors=5, algorithm="brute", **kw).fit(x, y),
        )
        row(name, tm, ts)


SWEEPS = {
    "metric": sweep_metric,
    "k": sweep_k,
    "p": sweep_p,
    "algorithm": sweep_algorithm,
    "n_jobs": sweep_n_jobs,
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
