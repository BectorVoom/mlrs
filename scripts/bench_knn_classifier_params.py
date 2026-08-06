#!/usr/bin/env python3
"""Per-PARAMETER `KNeighborsClassifier` comparison against scikit-learn (KNN-CLF-PARAMS).

``scripts/bench_knn_classifier.py`` benches ``fit`` alone. This harness benches
the PARAMETER SURFACE of the query path: it walks the values of one parameter at
a time with everything else at its sklearn default, so a regression is
attributable to the parameter that caused it rather than to "k-NN got slower".

Only the parameters that can move the wall clock get a sweep:

``metric``      five different inner loops (``minkowski`` additionally splits on
                whether ``p`` is an integer — the repeated-multiplication lane
                loop against ``powf``).
``n_neighbors`` the selection width, and the vote's inner loop length. It used to
                be the sharpest cliff in the whole estimator: past the device
                kernel's 16-slot list the search fell into the GPU-shaped
                composition. That is what KNN-HOST removed, and this sweep is the
                regression gate for it.
``weights``     ``'distance'`` is NOT free here even though the search is
                identical: the ``1/d`` weighting needs the top-k DISTANCE matrix
                on the host, and the uniform vote does not read it back at all.
                This sweep is what prices that read-back.
``p``           only for ``metric='minkowski'`` — see ``metric``.
``n_classes``   not a constructor argument, but it is the width of the vote's
                accumulator and of every ``predict_proba`` row, so it is the one
                data property the vote's cost depends on.
``algorithm``   mlrs runs brute force for every value, so its own row is flat by
                construction. It is swept anyway because SKLEARN's is not: its
                ``'auto'`` picks a k-d tree on low-dimensional data, and that —
                not its brute-force path — is what a user actually gets from
                ``KNeighborsClassifier()``. Beating ``algorithm='brute'`` while
                losing to the default would not be beating sklearn.
``n_jobs``      sklearn's joblib fan-out over query chunks. mlrs always uses the
                whole machine, so ``n_jobs=-1`` is the sklearn row that makes the
                comparison core-for-core rather than 16-against-1.

``leaf_size`` gets no sweep: it is read only on a tree route, which mlrs never
takes, and sklearn's own sensitivity to it is a property of its trees.

    python3 scripts/bench_knn_classifier_params.py [sweep ...]

with ``sweep`` one of ``metric`` / ``k`` / ``weights`` / ``p`` / ``classes`` /
``algorithm`` / ``n_jobs`` / ``proba`` / ``fit`` (default: all of them).
``KNN_PARAM_N`` / ``KNN_PARAM_D`` / ``KNN_PARAM_NQ`` / ``KNN_PARAM_C`` override
the design size.

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
``/proc/loadavg`` is not close to zero, the numbers are noise. For a MARGINAL
rung (anything inside ~1.3x either way) re-run the two engines in SEPARATE
processes — in-process interleaving has inverted a verdict here before.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

N = int(os.environ.get("KNN_PARAM_N", 50_000))
D = int(os.environ.get("KNN_PARAM_D", 16))
NQ = int(os.environ.get("KNN_PARAM_NQ", 5_000))
C = int(os.environ.get("KNN_PARAM_C", 3))
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
CLASS_SWEEP = [2, 3, 10, 50, 200]
ALGORITHMS = ["auto", "brute", "kd_tree", "ball_tree"]
N_JOBS = [None, 1, -1]


def make_data(n_classes: int = C, seed: int = 42):
    """A shifted design (so the cosine rows are well conditioned) + class labels.

    Labels are NON-CONTIGUOUS (``* 2 + 1``) and near-balanced (quantile cuts on a
    linear response), matching the oracle fixtures: a dominant class would make
    the vote's accumulator sparse in a way real data is not.
    """
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((N, D)) + 3.0).astype(np.float32)
    lin = x @ rng.standard_normal(D).astype(np.float32)
    cuts = np.quantile(lin, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = (np.searchsorted(cuts, lin) * 2 + 1).astype(np.int64)
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


def bench_call(label, mlrs_kwargs, sk_kwargs, data, method="predict") -> None:
    """One cell: fit both engines, then time `method` interleaved."""
    import mlrs
    from sklearn.neighbors import KNeighborsClassifier as Sk

    x, y, xq = data
    m = mlrs.KNeighborsClassifier(**mlrs_kwargs).fit(x, y)
    s = Sk(**sk_kwargs).fit(x, y)
    mf, sf = getattr(m, method), getattr(s, method)
    mf(xq[:32])  # materialize the deferred upload
    sf(xq[:32])
    tm, ts = interleaved(lambda: mf(xq), lambda: sf(xq))
    row(label, tm, ts)


def sweep_metric(data) -> None:
    print(f"metric sweep (n={N} d={D} nq={NQ} c={C}, k=5, weights='uniform')")
    for name, kw in METRICS:
        bench_call(
            name,
            {"n_neighbors": 5, **kw},
            {"n_neighbors": 5, "algorithm": "brute", **kw},
            data,
        )


def sweep_weights(data) -> None:
    print(f"weights sweep (n={N} d={D} nq={NQ} c={C}, k=5, metric='euclidean')")
    for w in ("uniform", "distance"):
        bench_call(
            w,
            {"n_neighbors": 5, "weights": w},
            {"n_neighbors": 5, "weights": w, "algorithm": "brute"},
            data,
        )


def sweep_k(data) -> None:
    print(f"n_neighbors sweep (n={N} d={D} nq={NQ} c={C}, metric='euclidean')")
    for k in K_SWEEP:
        bench_call(
            f"k={k}",
            {"n_neighbors": k},
            {"n_neighbors": k, "algorithm": "brute"},
            data,
        )


def sweep_p(data) -> None:
    print(f"p sweep (n={N} d={D} nq={NQ} c={C}, k=5, metric='minkowski')")
    for p in P_SWEEP:
        bench_call(
            f"p={p}",
            {"n_neighbors": 5, "metric": "minkowski", "p": p},
            {"n_neighbors": 5, "metric": "minkowski", "p": p, "algorithm": "brute"},
            data,
        )


def sweep_classes(_data) -> None:
    """The vote's accumulator width. Rebuilds the data per cell — `n_classes` is a
    property of `y`, not a constructor argument."""
    print(f"n_classes sweep (n={N} d={D} nq={NQ}, k=5, metric='euclidean')")
    for c in CLASS_SWEEP:
        bench_call(
            f"n_classes={c}",
            {"n_neighbors": 5},
            {"n_neighbors": 5, "algorithm": "brute"},
            make_data(n_classes=c),
        )


def sweep_algorithm(data) -> None:
    print(f"algorithm sweep (n={N} d={D} nq={NQ} c={C}, k=5) — mlrs is brute for every value")
    for a in ALGORITHMS:
        bench_call(
            a,
            {"n_neighbors": 5, "algorithm": a},
            {"n_neighbors": 5, "algorithm": a},
            data,
        )


def sweep_n_jobs(data) -> None:
    print(f"n_jobs sweep (n={N} d={D} nq={NQ} c={C}, k=5) — sklearn's fan-out, mlrs always full")
    for j in N_JOBS:
        bench_call(
            f"n_jobs={j}",
            {"n_neighbors": 5},
            {"n_neighbors": 5, "algorithm": "brute", "n_jobs": j},
            data,
        )


def sweep_proba(data) -> None:
    """`predict_proba` egresses an `n_query x n_classes` matrix rather than a
    label vector, so it prices the vote and the egress without the argmax."""
    print(f"predict_proba (n={N} d={D} nq={NQ} c={C}, k=5)")
    for w in ("uniform", "distance"):
        bench_call(
            w,
            {"n_neighbors": 5, "weights": w},
            {"n_neighbors": 5, "weights": w, "algorithm": "brute"},
            data,
            method="predict_proba",
        )


def sweep_fit(data) -> None:
    """`fit` is a validation pass for both engines — see `bench_knn.py`."""
    import mlrs
    from sklearn.neighbors import KNeighborsClassifier as Sk

    x, y, _ = data
    print(f"fit (n={N} d={D} c={C})")
    for name, kw in METRICS:
        tm, ts = interleaved(
            lambda kw=kw: mlrs.KNeighborsClassifier(n_neighbors=5, **kw).fit(x, y),
            lambda kw=kw: Sk(n_neighbors=5, algorithm="brute", **kw).fit(x, y),
        )
        row(name, tm, ts)


SWEEPS = {
    "metric": sweep_metric,
    "weights": sweep_weights,
    "k": sweep_k,
    "p": sweep_p,
    "classes": sweep_classes,
    "algorithm": sweep_algorithm,
    "n_jobs": sweep_n_jobs,
    "proba": sweep_proba,
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
