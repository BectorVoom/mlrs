#!/usr/bin/env python3
"""VotingRegressor performance (VOTE-01) — the parameters that move the clock.

``mlrs.VotingRegressor(estimators, *, weights=None, n_jobs=None,
verbose=False)``. Three of those four can change how long a call takes, and this
harness measures each at the level where it acts:

    ==============  ============  ============================================
    parameter       ``--level``   what it decides
    ==============  ============  ============================================
    (the arm)       ``agg``       ``MLRS_VOTING_ENGINE`` — which of numpy /
                                  Rust-host / CubeCL performs the aggregation
    ``weights``     ``agg``       ``np.average(weights=...)`` versus the
                                  uniform ``mean`` fast path
    ``n_jobs``      ``fit``       the joblib fan-out over the member fits.
                                  Run it BOTH ways (see ``--balanced``): a
                                  voting ensemble fits each member once, so the
                                  ceiling is Amdahl's `total / slowest` and an
                                  imbalanced pool reports "n_jobs does nothing"
                                  when nothing is in fact being left on the table
    ``verbose``     —             one ``print`` per member; not measured, it
                                  cannot be a measurable share of a fit
    ==============  ============  ============================================

``estimators`` is not a "parameter" so much as the whole workload: a voting
ensemble's cost is the sum of its members' costs plus one aggregation, and the
member costs belong to the members. So the ladders vary ``k`` (member COUNT) and
``n`` and hold the member TYPE fixed, which is what isolates the part this
estimator owns.

## Why the arm comparison is not obvious enough to skip measuring

``docs/stacking.md`` settled the neighbouring question — a pure `n x width`
copy, where `np.hstack` wins on every backend because the Rust arms start an
Arrow round-trip in debt. ``predict`` here is a different shape: it consumes
``n * k`` and emits ``n``, so the device arm's DOWNLOAD is `k` times smaller than
its upload and there is real arithmetic in between. Whether that repays the
crossing is exactly the kind of thing this project has guessed wrong about
before, so it gets a number.

## Two levels, because they answer different questions

* ``--level agg`` (default) — the aggregation ALONE, on synthetic prediction
  columns. This is the arm comparison, undiluted, and where the ``weights``
  question lives.
* ``--level fit`` — a whole ``VotingRegressor.fit`` per ``n_jobs``. Says whether
  the fan-out over member fits pays, and (with ``--arm``) whether the arm choice
  is visible at all in an end-to-end call.

## Measurement discipline

Every cell runs in a FRESH subprocess (``--cell``) and the harness reports the
MINIMUM of ``--repeat`` runs; cells are interleaved across arms rather than run
in blocks, so a drifting machine penalizes all of them equally. Both rules are
load-bearing on this project's cpu backend, where in-process interleaving and a
busy box have each inverted a verdict before.

The one-time ``_mlrs`` extension load is warmed OUTSIDE every timed region and
reported separately: it is ~35-95 ms, which is larger than the entire
aggregation at most sizes, and charging it to the first cell once made mlrs look
6x slower than it is (see ``bench_stacking.py``).

Usage
-----
    python3 scripts/bench_voting.py                      # arm ladder
    python3 scripts/bench_voting.py --weights            # ...weighted
    python3 scripts/bench_voting.py --level fit           # n_jobs, mixed pool
    python3 scripts/bench_voting.py --level fit --balanced   # ...ceiling = k
    python3 scripts/bench_voting.py --repeat 7 --cpu-time

Requires numpy + sklearn + mlrs (built for the backend under test).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

SEED = 42
ARMS = ["numpy", "host", "device"]

#: ``(n_rows, k_members)`` for the aggregation level. The ladder walks from
#: "smaller than the FFI crossing" to "big enough for bandwidth to matter", and
#: separately walks ``k`` at fixed ``n`` — because ``k`` is what sets the
#: upload/download ratio the device arm's whole case rests on.
AGG_LADDER = [
    (1_000, 3),
    (10_000, 3),
    (100_000, 3),
    (1_000_000, 3),
    (100_000, 2),
    (100_000, 8),
    (1_000_000, 8),
]

#: ``(n_rows, n_features, k_members)`` for the end-to-end level.
FIT_LADDER = [
    (10_000, 32, 4),
    (100_000, 32, 4),
    (100_000, 128, 4),
]

#: ``n_jobs`` values swept at ``--level fit``. Members are sklearn estimators,
#: because a joblib fan-out over an mlrs member is reduced to serial BY DESIGN
#: (`_effective_n_jobs`) and would measure the warning rather than the fan-out.
N_JOBS = [None, 2, 4]


def columns_for(n, k, dtype):
    """``k`` prediction columns of ``n`` rows — what the members hand over."""
    rng = np.random.default_rng(SEED)
    return [rng.standard_normal(n).astype(dtype) for _ in range(k)]


def time_agg(arm, n, k, dtype, reps, weighted, mode):
    """Seconds for one aggregation on `arm`, min over `reps` in-cell.

    The in-cell minimum is on top of the across-process minimum, not instead of
    it: the aggregation is short enough at small ``n`` that a single sample is
    mostly scheduler noise.
    """
    import mlrs

    cols = columns_for(n, k, dtype)
    weights = [1.0 + j * 0.5 for j in range(k)] if weighted else None

    if arm == "numpy":
        if mode == "predict":
            def once():
                return np.average(np.asarray(cols).T, axis=1, weights=weights)
        else:
            def once():
                return np.asarray(cols).T
    else:
        def once():
            return mlrs.ensemble._vote_via_rust(cols, mode, weights, arm)

    warm = once()
    if warm is None:
        raise RuntimeError(f"arm {arm!r} declined these columns")

    best, best_cpu = float("inf"), float("inf")
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time()
        out = once()
        best = min(best, time.perf_counter() - w0)
        best_cpu = min(best_cpu, time.process_time() - c0)
    return best, best_cpu, float(np.asarray(out, dtype=np.float64).sum())


def time_fit(n, d, k, n_jobs, dtype, reps, balanced):
    """Seconds for a whole ``VotingRegressor.fit`` at one ``n_jobs``.

    Two member sets, because they answer different questions:

    * the MIXED pool (default) is what an ensemble usually looks like — members
      of wildly different cost. Its speedup ceiling is Amdahl's, `total /
      slowest`, and for this pool that is only ~1.05x: the depth-8 tree is ~94%
      of the total. A ladder run only on this pool would report "n_jobs does
      nothing" and hide WHY.
    * the BALANCED pool (``--balanced``) is `k` copies of the same tree with
      different seeds, so the ceiling is `k`. This is the cell that measures the
      fan-out rather than the imbalance.
    """
    import mlrs
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, d)).astype(dtype)
    w = rng.standard_normal(d).astype(dtype)
    y = (X @ w + 0.05 * rng.standard_normal(n)).astype(dtype)

    if balanced:
        pool = [
            (f"tree{i}", DecisionTreeRegressor(max_depth=8, random_state=SEED + i))
            for i in range(k)
        ]
    else:
        pool = [
            ("lr", LinearRegression()),
            ("ridge", Ridge(alpha=1.0)),
            ("tree", DecisionTreeRegressor(max_depth=8, random_state=SEED)),
            ("lasso", Lasso(alpha=0.01, max_iter=200)),
        ]

    def build():
        return mlrs.VotingRegressor(pool[:k], n_jobs=n_jobs)

    build().fit(X[:64], y[:64])

    best, best_cpu = float("inf"), float("inf")
    for _ in range(reps):
        est = build()
        w0, c0 = time.perf_counter(), time.process_time()
        est.fit(X, y)
        best = min(best, time.perf_counter() - w0)
        best_cpu = min(best_cpu, time.process_time() - c0)
    return best, best_cpu, float(np.asarray(est.predict(X[:64]), dtype=np.float64).mean())


def cell(args):
    """One measurement, in this fresh interpreter, as a JSON line."""
    os.environ["MLRS_VOTING_ENGINE"] = args.arm

    t0 = time.perf_counter()
    import mlrs

    mlrs._load_ext()
    startup_s = time.perf_counter() - t0

    resolved = mlrs._load_ext().voting_engine()
    # A knob that did not take would make the whole sweep a comparison of numpy
    # against numpy and report it as "no difference". Fail loudly instead
    # (`mlrs-bench-verify-knob-is-live`).
    if resolved != args.arm:
        print(json.dumps({"error": f"knob resolved to {resolved!r}, wanted {args.arm!r}"}))
        return 1

    dtype = np.float64 if mlrs.backend_supports_f64() else np.float32
    if args.level == "agg":
        seconds, cpu_seconds, checksum = time_agg(
            args.arm, args.n, args.k, dtype, args.inner, args.weights, args.mode
        )
    else:
        n_jobs = None if args.n_jobs == 0 else args.n_jobs
        seconds, cpu_seconds, checksum = time_fit(
            args.n, args.d, args.k, n_jobs, dtype, args.inner, args.balanced
        )

    print(
        json.dumps(
            {
                "arm": args.arm,
                "seconds": seconds,
                "cpu_seconds": cpu_seconds,
                "checksum": checksum,
                "startup_s": startup_s,
                "dtype": np.dtype(dtype).name,
                "loadavg": os.getloadavg()[0],
            }
        )
    )
    return 0


def spawn(args, *, arm, n, k, d=0, n_jobs=0):
    """Run one cell in a FRESH interpreter and parse its JSON line."""
    argv = [
        sys.executable, __file__, "--cell",
        "--arm", arm, "--level", args.level, "--mode", args.mode,
        "--n", str(n), "--k", str(k), "--d", str(d),
        "--n-jobs", str(n_jobs), "--inner", str(args.inner),
    ]
    if args.weights:
        argv.append("--weights")
    if args.balanced:
        argv.append("--balanced")
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            payload = json.loads(line)
            if "seconds" not in payload:
                raise RuntimeError(payload.get("error", "cell produced no timing"))
            return payload
    return None


def warn_if_contended(args):
    if not args.cpu_time and os.getloadavg()[0] > 4:
        # A co-tenanted box has INVERTED a verdict on this project before
        # (`mlrs-cpu-bench-separate-processes`). Say so on the output itself, so
        # a ladder read months later carries its own caveat.
        print("# WARNING: loadavg > 4 — wall-clock cells are contended; re-run with --cpu-time")


def sweep_agg(args):
    clock = "cpu_seconds" if args.cpu_time else "seconds"
    print(
        f"# level=agg mode={args.mode} weights={'yes' if args.weights else 'no'} "
        f"repeat={args.repeat} inner={args.inner} "
        f"clock={'CPU time' if args.cpu_time else 'wall'} loadavg={os.getloadavg()[0]:.1f}"
    )
    warn_if_contended(args)
    header = (
        f"{'config':<24}{'numpy':>12}{'host':>12}{'device':>12}"
        f"{'host/np':>10}{'dev/np':>10}"
    )
    print(header)
    print("-" * len(header))

    startups, dtypes = [], set()
    for n, k in AGG_LADDER:
        best = {arm: float("inf") for arm in ARMS}
        checks = {}
        for _ in range(args.repeat):
            for arm in ARMS:  # interleaved, not blocked
                got = spawn(args, arm=arm, n=n, k=k)
                if got is None:
                    continue
                best[arm] = min(best[arm], got[clock])
                checks[arm] = got["checksum"]
                startups.append(got["startup_s"])
                dtypes.add(got["dtype"])

        print(
            f"{f'n={n:,} k={k}':<24}"
            f"{fmt(best['numpy'])}{fmt(best['host'])}{fmt(best['device'])}"
            f"{ratio(best['numpy'], best['host'])}{ratio(best['numpy'], best['device'])}"
        )
        # A checksum split means an arm computed something else; the ladder is
        # meaningless then, so say so on the line itself.
        if len({round(v, 6) for v in checks.values()}) > 1:
            print(f"    !! checksums disagree: {checks}")

    report_startup(startups)
    print(f"# dtype: {', '.join(sorted(dtypes)) or 'n/a'}")
    print("# ratios are numpy/arm — above 1.00x means the arm BEAT numpy")


def sweep_fit(args):
    clock = "cpu_seconds" if args.cpu_time else "seconds"
    print(
        f"# level=fit arm={args.arm} members={'balanced' if args.balanced else 'mixed'} "
        f"repeat={args.repeat} inner={args.inner} "
        f"clock={'CPU time' if args.cpu_time else 'wall'} loadavg={os.getloadavg()[0]:.1f}"
    )
    warn_if_contended(args)
    cols = "".join(f"{('n_jobs=' + str(j)):>14}" for j in N_JOBS)
    header = f"{'config':<28}{cols}{'best/serial':>13}"
    print(header)
    print("-" * len(header))

    startups = []
    for n, d, k in FIT_LADDER:
        best = {}
        for j in N_JOBS:
            b = float("inf")
            for _ in range(args.repeat):
                got = spawn(args, arm=args.arm, n=n, k=k, d=d, n_jobs=0 if j is None else j)
                if got is None:
                    continue
                b = min(b, got[clock])
                startups.append(got["startup_s"])
            best[j] = b
        serial = best[None]
        fastest = min(v for v in best.values() if v != float("inf"))
        row = "".join(
            f"{(f'{best[j] * 1000:.1f}m' if best[j] != float('inf') else 'n/a'):>14}"
            for j in N_JOBS
        )
        speedup = f"{serial / fastest:.2f}x" if fastest else "n/a"
        print(f"{f'n={n:,} d={d} k={k}':<28}{row}{speedup:>13}")

    report_startup(startups)
    print("# best/serial above 1.00x means SOME n_jobs beat the serial fit")


def fmt(v):
    return f"{v * 1000:>11.3f}m" if v != float("inf") else f"{'n/a':>12}"


def ratio(base, v):
    if v == float("inf") or base == float("inf") or v == 0:
        return f"{'n/a':>10}"
    return f"{base / v:>9.2f}x"


def report_startup(startups):
    if startups:
        print(
            f"\n# _mlrs load (excluded from every cell): "
            f"min {min(startups) * 1000:.1f} ms, max {max(startups) * 1000:.1f} ms"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--arm", choices=ARMS, default="numpy")
    p.add_argument("--level", choices=["agg", "fit"], default="agg")
    p.add_argument(
        "--mode",
        choices=["predict", "transform"],
        default="predict",
        help="which aggregation to time; `predict` reduces, `transform` does not",
    )
    p.add_argument(
        "--weights",
        action="store_true",
        help="use a non-uniform weight vector (the `np.average` path)",
    )
    p.add_argument(
        "--balanced",
        action="store_true",
        help=(
            "fit level: use k equal-cost members instead of the mixed pool, so "
            "the n_jobs ceiling is k rather than Amdahl's ~1.05x"
        ),
    )
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--k", type=int, default=3, help="member count")
    p.add_argument("--d", type=int, default=32, help="feature count (fit level)")
    p.add_argument("--n-jobs", type=int, default=0, help="0 means None")
    p.add_argument("--repeat", type=int, default=3, help="fresh processes per cell")
    p.add_argument("--inner", type=int, default=5, help="in-process reps per cell")
    p.add_argument(
        "--cpu-time",
        action="store_true",
        help="report process CPU time instead of wall clock (use on a loaded box)",
    )
    args = p.parse_args()

    if args.cell:
        return cell(args)
    if args.level == "agg":
        sweep_agg(args)
    else:
        sweep_fit(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
