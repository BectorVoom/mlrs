#!/usr/bin/env python3
"""StackingRegressor parameter-sweep harness (STACK-01).

Measures the three ``StackingRegressor`` parameters that can actually move the
clock, against ``sklearn.ensemble.StackingRegressor`` on byte-identical data:

    ==============  ==================================================
    parameter       why it is on the perf ladder
    ==============  ==================================================
    ``cv``          THE cost driver. An int ``k`` costs ``k + 1`` base
                    fits per member (``k`` out-of-fold + one on the
                    full data); ``cv="prefit"`` costs ZERO base fits.
                    Fit time should be very close to linear in ``k``.
    ``passthrough`` widens the meta matrix from ``n x m`` to
                    ``n x (m + d)`` — one extra ``n x d`` copy per
                    fit/transform plus a wider final fit. Cheap in
                    theory; the ladder is what says so.
    ``n_jobs``      joblib fan-out over the members and over the folds.
                    Only a scheduling knob, so a WIN here and an
                    identical answer are both required (the oracle
                    suite gates the answer). Expect a win on the
                    ``host`` arm and a FLAT line on ``device``: an mlrs
                    member holds an unpicklable device handle, so
                    ``mlrs.StackingRegressor`` warns and fits serially
                    rather than crashing (see ``_effective_n_jobs`` in
                    ``mlrs/ensemble.py`` for the two failure modes that
                    ruled the alternatives out).
    ==============  ==================================================

``final_estimator`` and ``verbose`` are excluded deliberately:
``final_estimator``'s cost is that estimator's own (already benchmarked
wherever it lives — one fit on an ``n x m`` matrix with ``m`` in single digits
is noise next to the base fits), and ``verbose`` only prints.

## Two arms, because they answer different questions

* ``host`` — sklearn base members on BOTH sides. Isolates the meta-estimator
  layer itself: mlrs's Rust ``KFold`` + Rust meta-layout + numpy hstack against
  sklearn's Python equivalents, with every base fit identical. A regression here
  is mlrs's own orchestration overhead and nothing else.
* ``device`` — mlrs base members in the mlrs stack vs sklearn base members in
  the sklearn stack. The end-to-end deployment comparison; the number is
  dominated by the base estimators, which is the point.

## Measurement discipline

Every cell runs in a FRESH subprocess (``--cell``), and the harness reports the
MINIMUM of ``--repeat`` runs. Both are load-bearing on this project's cpu
backend: in-process interleaving and a busy box have each inverted a verdict
before. Cells are also interleaved across implementations rather than run in
blocks, so a drifting machine penalizes both sides equally.

Usage
-----
    python3 scripts/bench_stacking.py                 # full sweep
    python3 scripts/bench_stacking.py --arm host      # orchestration only
    python3 scripts/bench_stacking.py --n 20000 --repeat 5

Requires numpy + scikit-learn; ``mlrs`` optional (its rows are skipped when the
extension is not importable).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np

N_DEFAULT = 20_000
D_DEFAULT = 32
SEED = 42

# `cv` values swept. "prefit" is on the ladder because it is the one setting
# that removes the base fits entirely — the floor every int `k` is measured
# against.
CV_LADDER = [2, 3, 5, 10, "prefit"]
PASSTHROUGH_LADDER = [False, True]
N_JOBS_LADDER = [None, 2, 4]


def design(n, d, dtype):
    """A well-conditioned linear regression problem, deterministic in SEED."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, d)).astype(dtype)
    w = rng.standard_normal(d).astype(dtype)
    y = (X @ w + 0.05 * rng.standard_normal(n)).astype(dtype)
    return X, y


def build(impl, arm, cv, passthrough, n_jobs, X, y):
    """The estimator under test, plus the members it will be given.

    ``cv="prefit"`` needs its members ALREADY fitted, so that construction
    happens here — and its cost is deliberately NOT timed, because "prefit"
    exists precisely for the case where those fits already happened elsewhere.
    """
    if impl == "sklearn" or arm == "host":
        from sklearn.linear_model import LinearRegression, Ridge

        members = [("lr", LinearRegression()), ("ridge", Ridge(alpha=1.0))]
        final = Ridge(alpha=1.0)
    else:
        import mlrs

        members = [("lr", mlrs.LinearRegression()), ("ridge", mlrs.Ridge(alpha=1.0))]
        final = mlrs.Ridge(alpha=1.0)

    if cv == "prefit":
        members = [(name, est.fit(X, y)) for name, est in members]

    if impl == "sklearn":
        from sklearn.ensemble import StackingRegressor
    else:
        from mlrs import StackingRegressor

    return StackingRegressor(
        members,
        final_estimator=final,
        cv=cv,
        passthrough=passthrough,
        n_jobs=n_jobs,
    )


def run_cell(impl, arm, cv, passthrough, n_jobs, n, d):
    """One timed cell: ``(fit_seconds, predict_seconds, checksum)``.

    The checksum is the prediction mean; the harness prints it so a
    configuration that got fast by computing something else is visible in the
    output rather than only in the oracle suite.
    """
    dtype = np.float64
    if impl == "mlrs":
        import mlrs

        # WARM-UP, and why it is not cheating. Loading `_mlrs.abi3.so` runs the
        # driver probe and brings up the CubeCL runtime — measured at ~90 ms on
        # this machine, which is 15x a whole `cv="prefit"` fit. Charging it to
        # the first timed cell would have said mlrs's orchestration was 6x
        # slower than sklearn's when warm it is FASTER. It is a real cost, but a
        # once-per-process one; it belongs in a startup line, not in a
        # per-parameter sweep.
        t0 = time.perf_counter()
        mlrs._load_ext()
        startup_s = time.perf_counter() - t0
        if arm == "device":
            dtype = np.float64 if mlrs.backend_supports_f64() else np.float32
    else:
        startup_s = 0.0
    X, y = design(n, d, dtype)

    # A tiny same-shape fit, so device-kernel JIT (device arm) and any
    # first-call import inside the composed estimators land here rather than in
    # the timed fit below. 64 rows is far too few to warm a data cache that
    # matters at n=20000.
    warm_X, warm_y = design(64, d, dtype)
    build(impl, arm, cv, passthrough, n_jobs, warm_X, warm_y).fit(warm_X, warm_y)

    est = build(impl, arm, cv, passthrough, n_jobs, X, y)

    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = est.predict(X)
    predict_s = time.perf_counter() - t0

    return fit_s, predict_s, float(np.asarray(pred, dtype=np.float64).mean()), startup_s


def spawn(impl, arm, cv, passthrough, n_jobs, n, d):
    """Run one cell in a FRESH interpreter and parse its JSON line."""
    argv = [
        sys.executable,
        __file__,
        "--cell",
        "--impl",
        impl,
        "--arm",
        arm,
        "--cv",
        str(cv),
        "--n",
        str(n),
        "--d",
        str(d),
        "--n-jobs",
        "none" if n_jobs is None else str(n_jobs),
    ]
    if passthrough:
        argv.append("--passthrough")
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return None


def measure(impl, arm, cv, passthrough, n_jobs, n, d, repeat):
    """MIN over ``repeat`` fresh-process runs — the noise-resistant statistic."""
    best = None
    checksum = None
    startup = 0.0
    for _ in range(repeat):
        got = spawn(impl, arm, cv, passthrough, n_jobs, n, d)
        if got is None:
            return None, None, None, None
        checksum = got["checksum"]
        startup = got["startup_s"]
        cand = (got["fit_s"], got["predict_s"])
        if best is None or cand[0] < best[0]:
            best = cand
    return best[0], best[1], checksum, startup


_STARTUP_REPORTED = []


def _report_startup(startup_s):
    """Print the one-time `_mlrs` load cost ONCE, so it is visible but not
    silently folded into every cell (see `run_cell`'s warm-up note)."""
    if startup_s and not _STARTUP_REPORTED:
        _STARTUP_REPORTED.append(startup_s)
        print(
            f"  [mlrs one-time `_mlrs` extension load: {startup_s * 1e3:.0f} ms "
            "per process — excluded from the timings below]"
        )


def sweep(title, cells, arm, n, d, repeat):
    print(f"\n=== {title} (arm={arm}, n={n}, d={d}, min of {repeat}) ===")
    print(
        f"{'config':<28}{'sklearn fit':>13}{'mlrs fit':>12}{'speedup':>9}"
        f"{'sklearn pred':>14}{'mlrs pred':>12}{'speedup':>9}"
    )
    for label, cv, passthrough, n_jobs in cells:
        sk_fit, sk_pred, sk_sum, _ = measure(
            "sklearn", arm, cv, passthrough, n_jobs, n, d, repeat
        )
        ml_fit, ml_pred, ml_sum, ml_startup = measure(
            "mlrs", arm, cv, passthrough, n_jobs, n, d, repeat
        )
        _report_startup(ml_startup)
        if sk_fit is None:
            print(f"{label:<28}{'sklearn cell failed':>13}")
            continue
        if ml_fit is None:
            print(
                f"{label:<28}{sk_fit:>12.4f}s{'n/a':>12}{'':>9}"
                f"{sk_pred:>13.4f}s{'n/a':>12}"
            )
            continue
        drift = "" if abs(sk_sum - ml_sum) < 1e-3 * (1 + abs(sk_sum)) else "  !checksum"
        print(
            f"{label:<28}{sk_fit:>12.4f}s{ml_fit:>11.4f}s{sk_fit / ml_fit:>8.2f}x"
            f"{sk_pred:>13.4f}s{ml_pred:>11.4f}s{sk_pred / ml_pred:>8.2f}x{drift}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--d", type=int, default=D_DEFAULT)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--arm", choices=["host", "device", "both"], default="both")
    # --cell and its arguments are the subprocess protocol, not a user knob.
    ap.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--impl", choices=["sklearn", "mlrs"], help=argparse.SUPPRESS)
    ap.add_argument("--cv", help=argparse.SUPPRESS)
    ap.add_argument("--passthrough", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--n-jobs", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.cell:
        cv = args.cv if args.cv == "prefit" else int(args.cv)
        n_jobs = None if args.n_jobs == "none" else int(args.n_jobs)
        fit_s, predict_s, checksum, startup_s = run_cell(
            args.impl, args.arm, cv, args.passthrough, n_jobs, args.n, args.d
        )
        print(
            json.dumps(
                {
                    "fit_s": fit_s,
                    "predict_s": predict_s,
                    "checksum": checksum,
                    "startup_s": startup_s,
                }
            )
        )
        return

    arms = ["host", "device"] if args.arm == "both" else [args.arm]
    for arm in arms:
        sweep(
            "cv ladder — the base-fit multiplier",
            [(f"cv={cv!r}", cv, False, None) for cv in CV_LADDER],
            arm,
            args.n,
            args.d,
            args.repeat,
        )
        sweep(
            "passthrough — the wider meta matrix",
            [
                (f"passthrough={p} (cv=5)", 5, p, None)
                for p in PASSTHROUGH_LADDER
            ],
            arm,
            args.n,
            args.d,
            args.repeat,
        )
        sweep(
            "n_jobs — joblib fan-out",
            [(f"n_jobs={j!r} (cv=5)", 5, False, j) for j in N_JOBS_LADDER],
            arm,
            args.n,
            args.d,
            args.repeat,
        )


if __name__ == "__main__":
    main()
